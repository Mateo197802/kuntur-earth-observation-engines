from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import psutil

from scripts.statistical_correction import run_calibration as calibration
from scripts.statistical_correction.freeze_interval_comparison_v1_3_2 import seed_for, seed_inventory, verify_payload
from scripts.statistical_correction.interval_comparison_v1_3_2 import METHODS, fit_candidate, residual_ljung_box
from scripts.statistical_correction.provenance_interval_comparison_v1_3_2 import FREEZE_PATH, RESULTS, ROOT as PROJECT_ROOT, sha256_file, source_hashes


CONFIG_PATH = ROOT / "configs/statistical_interval_comparison_v1_3_2.json"
FREEZE_PATH = ROOT / "results/statistical_correction/interval_comparison_v1_3_2/protocol_freeze.json"
PROCESS = psutil.Process()
PEAK_RSS = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if np.isfinite(result) else None


def current_environment() -> dict[str, Any]:
    import scipy
    import statsmodels

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
        "thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        "process_count": 1,
    }


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("a+b")
    try:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another interval-comparison process holds the run lock") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another interval-comparison process holds the run lock") from exc
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()).encode("ascii"))
        stream.flush()
        yield
    finally:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        stream.close()


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = load_config()
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    verify_payload(payload)
    if payload.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("Configuration differs from the immutable preflight freeze")
    if payload.get("seed_inventory") != seed_inventory(config):
        raise RuntimeError("Seed inventory differs from the immutable preflight freeze")
    return config, payload, sha256_file(FREEZE_PATH)


def expected_cell(config: dict[str, Any], phase: str, slope: float) -> dict[str, Any]:
    design = config["design"]
    reps = int(design["preflight_replicates_per_slope"] if phase == "preflight" else design["development_replicates_per_slope"])
    namespace = design["preflight_seed_namespace"] if phase == "preflight" else design["series_seed_namespace"]
    return {
        "phase": phase,
        "regime": config["regime"],
        "slope_per_decade": float(slope),
        "replicates": reps,
        "seed_namespace": namespace,
        "candidate_methods": list(METHODS),
    }


def checkpoint_path(run_dir: Path, slope: float) -> Path:
    label = f"{slope:.2f}".replace("-", "m").replace(".", "p")
    return run_dir / "checkpoints" / f"cell_iid_gaussian_n276_s{label}.json"


def read_checkpoint(path: Path, *, expected: dict[str, Any], sources: dict[str, str], freeze_hash: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cell") != expected or payload.get("source_hashes") != sources or payload.get("protocol_freeze_sha256") != freeze_hash:
        raise RuntimeError(f"Checkpoint provenance or design mismatch: {path.name}")
    records = payload.get("replicates")
    if not isinstance(records, list):
        raise RuntimeError(f"Malformed checkpoint rows: {path.name}")
    expected_reps = int(expected["replicates"])
    if len(records) % len(METHODS) != 0 or len(records) > expected_reps * len(METHODS):
        raise RuntimeError(f"Checkpoint does not contain complete paired replicate groups: {path.name}")
    completed = len(records) // len(METHODS)
    for rep in range(completed):
        group = records[rep * len(METHODS):(rep + 1) * len(METHODS)]
        if [row.get("method") for row in group] != list(METHODS) or any(int(row.get("replicate", -1)) != rep for row in group):
            raise RuntimeError(f"Checkpoint replicate sequence or candidate ordering is invalid: {path.name}")
        if len({row.get("series_seed") for row in group}) != 1:
            raise RuntimeError(f"Paired methods do not share a series seed: {path.name}")
    declared = payload.get("completed_paired_replicates")
    if int(declared if declared is not None else -1) != completed:
        raise RuntimeError(f"Checkpoint progress count does not match rows: {path.name}")
    return records


def write_run_manifest(
    path: Path,
    *,
    config: dict[str, Any],
    phase: str,
    freeze_hash: str,
    sources: dict[str, str],
    status: str,
    started_utc: str,
    prior_cpu: float,
    prior_wall: float,
    session_cpu: float,
    session_wall: float,
    checkpoint_files: list[Path],
    stop_reason: str | None,
) -> dict[str, Any]:
    budget = config["resource_budget"]
    target_reps = int(config["design"]["preflight_replicates_per_slope"] if phase == "preflight" else config["design"]["development_replicates_per_slope"])
    cells: list[dict[str, Any]] = []
    paired_total = candidate_total = 0
    for slope in config["design"]["slopes_per_decade"]:
        path_for_slope = checkpoint_path(RESULTS / phase, float(slope))
        if not path_for_slope.exists():
            cells.append({"slope_per_decade": float(slope), "completed_paired_replicates": 0, "status": "NOT_STARTED"})
            continue
        cell = json.loads(path_for_slope.read_text(encoding="utf-8"))
        completed = int(cell.get("completed_paired_replicates", 0))
        paired_total += completed
        candidate_total += len(cell.get("replicates", []))
        cells.append({
            "slope_per_decade": float(slope),
            "completed_paired_replicates": completed,
            "target_paired_replicates": target_reps,
            "status": cell.get("status"),
            "checkpoint": path_for_slope.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(path_for_slope),
        })
    nonnull_paths = [path for path in checkpoint_files if path.exists()]
    max_rss = int(PROCESS.memory_info().rss)
    peak_rss = max(max_rss, PEAK_RSS)
    manifest = {
        "schema_version": "kuntur-v1.3.2-interval-comparison-run-manifest-v1",
        "protocol_id": config["protocol_id"],
        "phase": phase,
        "status": status,
        "started_utc": started_utc,
        "updated_utc": utc_now(),
        "protocol_freeze_sha256": freeze_hash,
        "source_hashes": sources,
        "environment": current_environment(),
        "seed_namespace": config["design"]["preflight_seed_namespace"] if phase == "preflight" else config["design"]["series_seed_namespace"],
        "seed_inventory": seed_inventory(config),
        "completed_paired_replicates_total": paired_total,
        "completed_candidate_replicates_total": candidate_total,
        "target_paired_replicates_total": target_reps * len(config["design"]["slopes_per_decade"]),
        "target_candidate_replicates_total": target_reps * len(config["design"]["slopes_per_decade"]) * len(METHODS),
        "cells": cells,
        "resource_usage": {
            "cumulative_process_cpu_seconds": round(prior_cpu + session_cpu, 3),
            "cumulative_active_wall_seconds": round(prior_wall + session_wall, 3),
            "session_process_cpu_seconds": round(session_cpu, 3),
            "session_active_wall_seconds": round(session_wall, 3),
            "peak_rss_bytes_observed": peak_rss,
            "hard_caps": {
                "cpu_seconds": int(budget["preflight_hard_max_cpu_seconds"] if phase == "preflight" else budget["development_hard_max_cpu_seconds"]),
                "active_wall_seconds": int(budget["preflight_hard_max_wall_seconds"] if phase == "preflight" else budget["development_hard_max_wall_seconds"]),
                "rss_bytes": int(budget["hard_max_rss_bytes"]),
                "processes": int(budget["max_processes"]),
                "blas_threads": int(budget["blas_threads"]),
                "slurm_jobs": int(budget["slurm_jobs"]),
            },
        },
        "stop_reason": stop_reason,
    }
    atomic_json(path, manifest)
    return manifest


def row_for_fit(
    *,
    phase: str,
    slope: float,
    replicate: int,
    series_seed: int,
    method: str,
    fit: Any,
    elapsed: dict[str, float],
    oracle_width: float | None,
) -> dict[str, Any]:
    slope_estimate = finite(getattr(fit, "slope", None))
    p_value = finite(getattr(fit, "p_value", None))
    ci_low = finite(getattr(fit, "ci_low", None))
    ci_high = finite(getattr(fit, "ci_high", None))
    converged = bool(getattr(fit, "converged", False)) and slope_estimate is not None
    ci_width = ci_high - ci_low if ci_low is not None and ci_high is not None else None
    residual = residual_ljung_box(fit.residuals) if fit.residuals is not None else {"kind": "NO_RESIDUALS", "p_values": None, "failure": None}
    return {
        "phase": phase,
        "regime": "iid_gaussian",
        "slope_per_decade": float(slope),
        "replicate": int(replicate),
        "series_seed": str(series_seed),
        "method": method,
        "supported": True,
        "converged": converged,
        "inferential_output": bool(converged and p_value is not None and ci_low is not None and ci_high is not None),
        "slope_estimate_per_decade": slope_estimate,
        "p_value": p_value,
        "reject_at_005": bool(p_value is not None and p_value < 0.05),
        "ci_low_per_decade": ci_low,
        "ci_high_per_decade": ci_high,
        "ci_covers_true_slope": bool(ci_low <= slope <= ci_high) if converged and ci_low is not None and ci_high is not None else None,
        "ci_width": finite(ci_width),
        "oracle_ci_width": finite(oracle_width),
        "ci_width_over_oracle": finite(ci_width / oracle_width) if ci_width is not None and oracle_width and oracle_width > 0 else None,
        "residual_diagnostic": {
            "kind": residual.get("kind"),
            "ljung_box_failure": residual.get("failure"),
            "p_values": residual.get("p_values"),
            "lags": residual.get("lags"),
            "alpha": residual.get("alpha"),
        },
        "residual_ljung_box_failure": residual.get("failure"),
        "residual_df": getattr(fit, "residual_df", None),
        "profile_likelihood_evaluations": int(getattr(fit, "profile_evaluations", 0)),
        "profile_log_likelihood_max": finite(getattr(fit, "profile_log_likelihood_max", None)),
        "log_likelihood": finite(getattr(fit, "log_likelihood", None)),
        "cpu_seconds": finite(elapsed.get("cpu_seconds")),
        "wall_seconds": finite(elapsed.get("wall_seconds")),
        "note": str(getattr(fit, "note", "") or "")[:300],
        "warnings": [str(item)[:220] for item in getattr(fit, "warnings", ())],
    }


def prior_resources(manifest_path: Path) -> tuple[float, float, int]:
    if not manifest_path.exists():
        return 0.0, 0.0, 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    usage = manifest.get("resource_usage", {})
    return (
        float(usage.get("cumulative_process_cpu_seconds", 0)),
        float(usage.get("cumulative_active_wall_seconds", 0)),
        int(usage.get("peak_rss_bytes_observed", 0)),
    )


def run(phase: str, plan_only: bool = False) -> dict[str, Any]:
    global PEAK_RSS
    config, freeze, freeze_hash = verify_frozen_inputs()
    sources = source_hashes()
    if phase not in {"preflight", "development"}:
        raise ValueError("phase must be preflight or development")
    run_dir = RESULTS / phase
    manifest_path = run_dir / "run_manifest.json"
    if phase == "development":
        preflight_summary = RESULTS / "preflight" / "preflight_summary.json"
        if not preflight_summary.exists():
            raise RuntimeError("Development is not permitted before a completed, summarized preflight")
        preflight = json.loads(preflight_summary.read_text(encoding="utf-8"))
        if preflight.get("protocol_freeze_sha256") != freeze_hash or preflight.get("decision") != "PROCEED":
            raise RuntimeError("Measured preflight did not authorize the frozen development budget")

    design = config["design"]
    reps_target = int(design["preflight_replicates_per_slope"] if phase == "preflight" else design["development_replicates_per_slope"])
    plan = []
    for slope in design["slopes_per_decade"]:
        slope_value = float(slope)
        cell = expected_cell(config, phase, slope_value)
        cell_hash = hashlib.sha256(json.dumps(cell, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        seeds = [seed_for(cell["seed_namespace"], slope_value, rep) for rep in range(reps_target)]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError("Duplicate series seeds within a slope cell")
        plan.append({"cell": cell, "cell_sha256": cell_hash, "series_seed_count": len(seeds), "first_series_seed": str(seeds[0]), "last_series_seed": str(seeds[-1])})
    inventory = seed_inventory(config)
    if plan_only:
        result = {
            "status": "PLAN_VERIFIED",
            "phase": phase,
            "protocol_id": config["protocol_id"],
            "protocol_freeze_sha256": freeze_hash,
            "source_count": len(sources),
            "cells": plan,
            "candidate_count": len(METHODS),
            "candidate_replicates_planned": reps_target * len(plan) * len(METHODS),
            "seed_inventory": inventory,
            "resource_budget": config["resource_budget"],
            "preflight_projection": json.loads((RESULTS / "preflight" / "preflight_summary.json").read_text(encoding="utf-8")) if phase == "development" else None,
        }
        print(json.dumps(result, indent=2))
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".run.lock"
    with exclusive_lock(lock_path):
        prior_cpu, prior_wall, prior_peak_rss = prior_resources(manifest_path)
        PEAK_RSS = prior_peak_rss
        started_utc = utc_now()
        session_cpu_start, session_wall_start = time.process_time(), time.perf_counter()
        status = "RUNNING"
        stop_reason = None
        checkpoint_files: list[Path] = []
        manifest = write_run_manifest(
            manifest_path, config=config, phase=phase, freeze_hash=freeze_hash, sources=sources,
            status=status, started_utc=started_utc, prior_cpu=prior_cpu, prior_wall=prior_wall,
            session_cpu=0.0, session_wall=0.0, checkpoint_files=checkpoint_files, stop_reason=None,
        )
        cap_cpu = int(config["resource_budget"]["preflight_hard_max_cpu_seconds"] if phase == "preflight" else config["resource_budget"]["development_hard_max_cpu_seconds"])
        cap_wall = int(config["resource_budget"]["preflight_hard_max_wall_seconds"] if phase == "preflight" else config["resource_budget"]["development_hard_max_wall_seconds"])
        rss_cap = int(config["resource_budget"]["hard_max_rss_bytes"])
        intervals = int(config["resource_budget"]["checkpoint_every_paired_replicates"])
        for slope in design["slopes_per_decade"]:
            slope_value = float(slope)
            path = checkpoint_path(run_dir, slope_value)
            checkpoint_files.append(path)
            cell = expected_cell(config, phase, slope_value)
            records = read_checkpoint(path, expected=cell, sources=sources, freeze_hash=freeze_hash)
            start_rep = len(records) // len(METHODS)
            if start_rep == reps_target:
                continue
            periods = pd.period_range(design["calendar_start"], periods=int(design["n_calendar_months"]), freq="M")
            x = calibration._design(periods)
            oracle = calibration.oracle_ci_width("iid_gaussian", periods, np.ones(len(periods), dtype=bool))
            for rep in range(start_rep, reps_target):
                if prior_cpu + (time.process_time() - session_cpu_start) >= cap_cpu:
                    stop_reason = "CPU_HARD_CAP"
                    break
                if prior_wall + (time.perf_counter() - session_wall_start) >= cap_wall:
                    stop_reason = "ACTIVE_WALL_HARD_CAP"
                    break
                if PEAK_RSS >= rss_cap:
                    stop_reason = "RSS_HARD_CAP"
                    break
                series_seed = seed_for(cell["seed_namespace"], slope_value, rep)
                y, generated_periods, observed = calibration.make_series("iid_gaussian", len(periods), slope_value, np.random.default_rng(series_seed))
                if not generated_periods.equals(periods) or not bool(observed.all()):
                    raise RuntimeError("Frozen IID generator returned unexpected support or calendar")
                for method in METHODS:
                    try:
                        fit, elapsed = fit_candidate(y, x, method, alpha=float(config["acceptance_criteria"]["type_i_alpha"]), maxiter=int(config["candidates"][0].get("optimizer_maxiter", 200)))
                    except Exception as exc:
                        fit = type("FailedFit", (), {
                            "method": method, "slope": None, "standard_error": None, "ci_low": None,
                            "ci_high": None, "p_value": None, "converged": False, "residuals": None,
                            "log_likelihood": None, "residual_df": None, "profile_evaluations": 0,
                            "note": f"UNHANDLED_FIT_EXCEPTION_{type(exc).__name__}", "warnings": (),
                        })()
                        elapsed = {"cpu_seconds": 0.0, "wall_seconds": 0.0}
                    records.append(row_for_fit(phase=phase, slope=slope_value, replicate=rep, series_seed=series_seed, method=method, fit=fit, elapsed=elapsed, oracle_width=oracle))

                completed = rep + 1
                session_cpu = time.process_time() - session_cpu_start
                session_wall = time.perf_counter() - session_wall_start
                PEAK_RSS = max(PEAK_RSS, int(PROCESS.memory_info().rss))
                cumulative_cpu = prior_cpu + session_cpu
                cumulative_wall = prior_wall + session_wall
                resource_stop = cumulative_cpu >= cap_cpu or cumulative_wall >= cap_wall or PEAK_RSS >= rss_cap
                if completed % intervals == 0 or completed == reps_target or resource_stop:
                    payload = {
                        "schema_version": "kuntur-v1.3.2-interval-comparison-cell-v1",
                        "phase": phase,
                        "status": "COMPLETE" if completed == reps_target else "BOUNDED_PARTIAL" if resource_stop else "IN_PROGRESS",
                        "cell": cell,
                        "cell_sha256": hashlib.sha256(json.dumps(cell, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                        "source_hashes": sources,
                        "protocol_freeze_sha256": freeze_hash,
                        "completed_paired_replicates": completed,
                        "candidate_count": len(METHODS),
                        "replicates": records,
                        "updated_utc": utc_now(),
                    }
                    atomic_json(path, payload)
                    manifest = write_run_manifest(
                        manifest_path, config=config, phase=phase, freeze_hash=freeze_hash, sources=sources,
                        status="BOUNDED_PARTIAL" if resource_stop and completed < reps_target else "RUNNING",
                        started_utc=started_utc, prior_cpu=prior_cpu, prior_wall=prior_wall,
                        session_cpu=session_cpu, session_wall=session_wall, checkpoint_files=checkpoint_files,
                        stop_reason="CPU_HARD_CAP" if cumulative_cpu >= cap_cpu else "ACTIVE_WALL_HARD_CAP" if cumulative_wall >= cap_wall else "RSS_HARD_CAP" if PEAK_RSS >= rss_cap else None,
                    )
                    print(f"CHECKPOINT phase={phase} slope={slope_value:.2f} paired={completed}/{reps_target} candidate_rows={len(records)} cpu={cumulative_cpu:.1f}s wall={cumulative_wall:.1f}s rss={PEAK_RSS}", flush=True)
                if resource_stop:
                    stop_reason = "CPU_HARD_CAP" if cumulative_cpu >= cap_cpu else "ACTIVE_WALL_HARD_CAP" if cumulative_wall >= cap_wall else "RSS_HARD_CAP"
                    break
            if stop_reason:
                break

        session_cpu = time.process_time() - session_cpu_start
        session_wall = time.perf_counter() - session_wall_start
        complete = True
        for slope in design["slopes_per_decade"]:
            path = checkpoint_path(run_dir, float(slope))
            if not path.exists() or int(json.loads(path.read_text(encoding="utf-8")).get("completed_paired_replicates", -1)) != reps_target:
                complete = False
        final_status = "COMPLETE" if complete else "BOUNDED_PARTIAL" if stop_reason else "INCOMPLETE"
        manifest = write_run_manifest(
            manifest_path, config=config, phase=phase, freeze_hash=freeze_hash, sources=sources,
            status=final_status, started_utc=started_utc, prior_cpu=prior_cpu, prior_wall=prior_wall,
            session_cpu=session_cpu, session_wall=session_wall, checkpoint_files=checkpoint_files,
            stop_reason=stop_reason,
        )
    print(json.dumps({"status": manifest["status"], "phase": phase, "completed_paired_replicates": manifest["completed_paired_replicates_total"], "completed_candidate_replicates": manifest["completed_candidate_replicates_total"], "resource_usage": manifest["resource_usage"], "stop_reason": manifest["stop_reason"], "manifest": manifest_path.relative_to(ROOT).as_posix()}, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "development"))
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(args.phase, args.plan_only)


if __name__ == "__main__":
    main()
