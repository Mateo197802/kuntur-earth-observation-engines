from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from scripts.statistical_correction.freeze_interval_confirmation_v1_3_2 import (
    CONFIG_PATH,
    FREEZE_PATH,
    RESULTS,
    ROOT as PROJECT_ROOT,
    seed_for,
    seed_inventory,
    sha256_file,
    source_hashes,
    verify_payload,
)
from scripts.statistical_correction.interval_comparison_v1_3_2 import fit_candidate
from scripts.statistical_correction.run_interval_comparison_v1_3_2 import row_for_fit

RUN_DIR = RESULTS / "confirmation"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
LOCK_PATH = RUN_DIR / ".run.lock"
PROCESS = psutil.Process()
PEAK_RSS = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


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
                raise RuntimeError("Another confirmation process holds the run lock") from exc
        else:
            import fcntl

            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Another confirmation process holds the run lock") from exc
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


def frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], str, dict[str, str]]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    verify_payload(payload)
    freeze_hash = sha256_file(FREEZE_PATH)
    sources = source_hashes()
    if payload.get("source_hashes") != sources:
        raise RuntimeError("Confirmation source hashes no longer match the frozen protocol")
    return config, payload, freeze_hash, sources


def expected_cell(config: dict[str, Any], slope: float) -> dict[str, Any]:
    design = config["design"]
    return {
        "phase": "confirmation",
        "regime": config["scope"]["regime"],
        "slope_per_decade": float(slope),
        "replicates": int(design["replicates_per_slope"]),
        "seed_namespace": design["series_seed_namespace"],
        "candidate_method": config["scope"]["candidate_method"],
    }


def checkpoint_path(slope: float) -> Path:
    label = f"{slope:.2f}".replace("-", "m").replace(".", "p")
    return RUN_DIR / "checkpoints" / f"cell_iid_gaussian_n276_s{label}_wald_student_t.json"


def read_checkpoint(path: Path, expected: dict[str, Any], sources: dict[str, str], freeze_hash: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("cell") != expected or payload.get("source_hashes") != sources or payload.get("protocol_freeze_sha256") != freeze_hash:
        raise RuntimeError(f"Checkpoint provenance or frozen design mismatch: {path.name}")
    rows = payload.get("replicates")
    if not isinstance(rows, list):
        raise RuntimeError(f"Malformed checkpoint: {path.name}")
    if len(rows) > int(expected["replicates"]):
        raise RuntimeError(f"Checkpoint has more rows than the frozen design: {path.name}")
    if [int(row.get("replicate", -1)) for row in rows] != list(range(len(rows))):
        raise RuntimeError(f"Checkpoint replicate IDs are missing or duplicated: {path.name}")
    if any(row.get("method") != expected["candidate_method"] or row.get("phase") != "confirmation" for row in rows):
        raise RuntimeError(f"Checkpoint method or phase differs from the frozen design: {path.name}")
    if len({row.get("series_seed") for row in rows}) != len(rows):
        raise RuntimeError(f"Checkpoint series seeds are duplicated: {path.name}")
    if int(payload.get("completed_replicates", -1)) != len(rows):
        raise RuntimeError(f"Checkpoint progress count does not match its rows: {path.name}")
    return rows


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


def prior_resources() -> tuple[float, float, int]:
    if not MANIFEST_PATH.is_file():
        return 0.0, 0.0, 0
    usage = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")).get("resource_usage", {})
    return (
        float(usage.get("cumulative_process_cpu_seconds", 0.0)),
        float(usage.get("cumulative_active_wall_seconds", 0.0)),
        int(usage.get("peak_rss_bytes_observed", 0)),
    )


def build_manifest(
    config: dict[str, Any], freeze_hash: str, sources: dict[str, str], status: str,
    start_utc: str, prior_cpu: float, prior_wall: float, session_cpu: float,
    session_wall: float, stop_reason: str | None,
) -> dict[str, Any]:
    cells = []
    completed_total = 0
    for slope in config["design"]["slopes_per_decade"]:
        path = checkpoint_path(float(slope))
        if not path.is_file():
            cells.append({"slope_per_decade": float(slope), "completed_replicates": 0, "status": "NOT_STARTED"})
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        completed = int(payload.get("completed_replicates", 0))
        completed_total += completed
        cells.append({
            "slope_per_decade": float(slope),
            "completed_replicates": completed,
            "target_replicates": int(config["design"]["replicates_per_slope"]),
            "status": payload.get("status"),
            "checkpoint": path.relative_to(ROOT).as_posix(),
            "checkpoint_sha256": sha256_file(path),
        })
    rss = max(int(PROCESS.memory_info().rss), PEAK_RSS, prior_resources()[2])
    return {
        "schema_version": "kuntur-v1.3.2-independent-confirmation-run-manifest-v1",
        "protocol_id": config["protocol_id"],
        "phase": "confirmation",
        "status": status,
        "started_utc": start_utc,
        "updated_utc": utc_now(),
        "protocol_freeze_sha256": freeze_hash,
        "source_hashes": sources,
        "environment": current_environment(),
        "seed_namespace": config["design"]["series_seed_namespace"],
        "seed_inventory": seed_inventory(config),
        "candidate_method": config["scope"]["candidate_method"],
        "completed_replicates_total": completed_total,
        "target_replicates_total": int(config["design"]["replicates_per_slope"]) * len(config["design"]["slopes_per_decade"]),
        "cells": cells,
        "resource_usage": {
            "cumulative_process_cpu_seconds": round(prior_cpu + session_cpu, 3),
            "cumulative_active_wall_seconds": round(prior_wall + session_wall, 3),
            "session_process_cpu_seconds": round(session_cpu, 3),
            "session_active_wall_seconds": round(session_wall, 3),
            "peak_rss_bytes_observed": rss,
            "hard_caps": {
                "cpu_seconds": int(config["resource_budget"]["hard_max_cpu_seconds"]),
                "active_wall_seconds": int(config["resource_budget"]["hard_max_active_wall_seconds"]),
                "rss_bytes": int(config["resource_budget"]["hard_max_rss_bytes"]),
                "processes": int(config["resource_budget"]["max_processes"]),
                "blas_threads": int(config["resource_budget"]["blas_threads"]),
                "slurm_jobs": int(config["resource_budget"]["slurm_jobs"]),
            },
        },
        "stop_reason": stop_reason,
    }


def run(plan_only: bool = False) -> dict[str, Any]:
    global PEAK_RSS
    config, frozen, freeze_hash, sources = frozen_inputs()
    design = config["design"]
    slopes = [float(value) for value in design["slopes_per_decade"]]
    reps = int(design["replicates_per_slope"])
    method = str(config["scope"]["candidate_method"])
    seeds_by_cell = {}
    for slope in slopes:
        seeds = [seed_for(design["series_seed_namespace"], slope, rep) for rep in range(reps)]
        if len(set(seeds)) != reps:
            raise RuntimeError(f"Duplicate confirmation seeds at slope {slope}")
        seeds_by_cell[f"{slope:.2f}"] = {
            "count": len(seeds),
            "first_seed": str(seeds[0]),
            "last_seed": str(seeds[-1]),
        }
    inventory = seed_inventory(config)
    if plan_only:
        plan = {
            "status": "PLAN_VERIFIED",
            "protocol_id": config["protocol_id"],
            "protocol_freeze_sha256": freeze_hash,
            "candidate_method": method,
            "candidate_fits_planned": reps * len(slopes),
            "seed_inventory": inventory,
            "cells": seeds_by_cell,
            "measured_resource_projection": frozen["measured_resource_projection"],
            "resource_caps": config["resource_budget"],
            "registry_mutation": False,
        }
        print(json.dumps(plan, indent=2))
        return plan

    global_start = time.perf_counter()
    with exclusive_lock(LOCK_PATH):
        prior_cpu, prior_wall, prior_rss = prior_resources()
        PEAK_RSS = prior_rss
        start_utc = utc_now()
        session_cpu_start = time.process_time()
        session_wall_start = time.perf_counter()
        cpu_cap = int(config["resource_budget"]["hard_max_cpu_seconds"])
        wall_cap = int(config["resource_budget"]["hard_max_active_wall_seconds"])
        rss_cap = int(config["resource_budget"]["hard_max_rss_bytes"])
        checkpoint_every = int(config["resource_budget"]["checkpoint_every_replicates"])
        stop_reason = None
        atomic_json(MANIFEST_PATH, build_manifest(config, freeze_hash, sources, "RUNNING", start_utc, prior_cpu, prior_wall, 0.0, 0.0, None))

        for slope in slopes:
            path = checkpoint_path(slope)
            expected = expected_cell(config, slope)
            records = read_checkpoint(path, expected, sources, freeze_hash)
            if len(records) == reps:
                continue
            periods = pd.period_range(config["scope"]["calendar_start"], periods=int(config["scope"]["calendar_months"]), freq="M")
            exog = calibration._design(periods)
            oracle_width = calibration.oracle_ci_width("iid_gaussian", periods, np.ones(len(periods), dtype=bool))
            for rep in range(len(records), reps):
                session_cpu = time.process_time() - session_cpu_start
                session_wall = time.perf_counter() - session_wall_start
                peak = max(PEAK_RSS, int(PROCESS.memory_info().rss))
                if prior_cpu + session_cpu >= cpu_cap:
                    stop_reason = "CPU_HARD_CAP"
                    break
                if prior_wall + session_wall >= wall_cap:
                    stop_reason = "ACTIVE_WALL_HARD_CAP"
                    break
                if peak >= rss_cap:
                    stop_reason = "RSS_HARD_CAP"
                    break

                seed = seed_for(design["series_seed_namespace"], slope, rep)
                y, generated_periods, observed = calibration.make_series(
                    "iid_gaussian", len(periods), slope, np.random.default_rng(seed)
                )
                if not generated_periods.equals(periods) or not bool(observed.all()):
                    raise RuntimeError("Frozen IID generator returned an unexpected calendar or support mask")
                try:
                    fit, elapsed = fit_candidate(
                        y,
                        exog,
                        method,
                        alpha=float(config["acceptance_criteria"]["type_i_alpha"]),
                        maxiter=int(config["scope"]["optimizer_maxiter"]),
                    )
                except Exception as exc:
                    fit = type("FailedFit", (), {
                        "method": method, "slope": None, "standard_error": None, "ci_low": None,
                        "ci_high": None, "p_value": None, "converged": False, "residuals": None,
                        "log_likelihood": None, "residual_df": None, "profile_evaluations": 0,
                        "profile_log_likelihood_max": None,
                        "note": f"UNHANDLED_FIT_EXCEPTION_{type(exc).__name__}", "warnings": (),
                    })()
                    elapsed = {"cpu_seconds": 0.0, "wall_seconds": 0.0}
                row = row_for_fit(
                    phase="confirmation", slope=slope, replicate=rep, series_seed=seed,
                    method=method, fit=fit, elapsed=elapsed, oracle_width=oracle_width,
                )
                records.append(row)
                session_cpu = time.process_time() - session_cpu_start
                session_wall = time.perf_counter() - session_wall_start
                PEAK_RSS = max(PEAK_RSS, int(PROCESS.memory_info().rss))
                resource_stop = prior_cpu + session_cpu >= cpu_cap or prior_wall + session_wall >= wall_cap or PEAK_RSS >= rss_cap
                complete_cell = len(records) == reps
                if len(records) % checkpoint_every == 0 or complete_cell or resource_stop:
                    cell_payload = {
                        "schema_version": "kuntur-v1.3.2-independent-confirmation-cell-v1",
                        "phase": "confirmation",
                        "status": "COMPLETE" if complete_cell else "BOUNDED_PARTIAL" if resource_stop else "IN_PROGRESS",
                        "cell": expected,
                        "source_hashes": sources,
                        "protocol_freeze_sha256": freeze_hash,
                        "completed_replicates": len(records),
                        "replicates": records,
                        "updated_utc": utc_now(),
                    }
                    atomic_json(path, cell_payload)
                    manifest_status = "BOUNDED_PARTIAL" if resource_stop and not complete_cell else "RUNNING"
                    manifest = build_manifest(config, freeze_hash, sources, manifest_status, start_utc, prior_cpu, prior_wall, session_cpu, session_wall, "RESOURCE_LIMIT_REACHED" if resource_stop else None)
                    atomic_json(MANIFEST_PATH, manifest)
                    print(f"CHECKPOINT phase=confirmation slope={slope:.2f} completed={len(records)}/{reps} fits={sum(int(c.get('completed_replicates', 0)) for c in manifest['cells'])} cpu={prior_cpu + session_cpu:.1f}s wall={prior_wall + session_wall:.1f}s rss={PEAK_RSS}", flush=True)
                if resource_stop:
                    stop_reason = "CPU_HARD_CAP" if prior_cpu + session_cpu >= cpu_cap else "ACTIVE_WALL_HARD_CAP" if prior_wall + session_wall >= wall_cap else "RSS_HARD_CAP"
                    break
            if stop_reason:
                break

        session_cpu = time.process_time() - session_cpu_start
        session_wall = time.perf_counter() - session_wall_start
        all_complete = all(
            checkpoint_path(slope).is_file()
            and int(json.loads(checkpoint_path(slope).read_text(encoding="utf-8")).get("completed_replicates", -1)) == reps
            for slope in slopes
        )
        status = "COMPLETE" if all_complete else "BOUNDED_PARTIAL" if stop_reason else "INCOMPLETE"
        manifest = build_manifest(config, freeze_hash, sources, status, start_utc, prior_cpu, prior_wall, session_cpu, session_wall, stop_reason)
        atomic_json(MANIFEST_PATH, manifest)

    manifest["elapsed_wall_seconds"] = round(time.perf_counter() - global_start, 3)
    print(json.dumps({
        "status": manifest["status"],
        "completed_replicates": manifest["completed_replicates_total"],
        "target_replicates": manifest["target_replicates_total"],
        "resource_usage": manifest["resource_usage"],
        "stop_reason": manifest["stop_reason"],
        "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
    }, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    run(plan_only=args.plan_only)


if __name__ == "__main__":
    main()
