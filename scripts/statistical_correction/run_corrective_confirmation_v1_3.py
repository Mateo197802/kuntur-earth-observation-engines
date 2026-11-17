"""Run the frozen independent IID confirmation for the v1.3 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/statistical_correction"))

import numpy as np
import pandas as pd
import psutil
import scipy
import statsmodels

import run_calibration as calibration
import run_corrective_development_v1_3_expansion as development
from kuntur_data.phase2.statistical_v12 import _Fit, _design, _fit_method

CONFIG_PATH = ROOT / "configs/statistical_corrective_confirmation_v1_3.json"
PROTOCOL_PATH = ROOT / "docs/protocols/statistical_corrective_confirmation_v1_3.md"
FREEZE_PATH = ROOT / "results/statistical_correction/corrective_development_v1_3/selection_freeze_iid_v1_3_r1.json"
FREEZE_RECEIPT_PATH = ROOT / "results/statistical_correction/corrective_development_v1_3/selection_freeze_receipt_iid_v1_3_r1.json"
FREEZE_SCRIPT_PATH = ROOT / "scripts/statistical_correction/freeze_corrective_selection_v1_3_r1.py"
SUMMARIZER_PATH = ROOT / "scripts/statistical_correction/summarize_corrective_confirmation_v1_3.py"
TEST_PATH = ROOT / "tests/test_statistical_confirmation_v13.py"
RUNNER_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/statistical_correction/independent_confirmation_v1_3_iid"
CHECKPOINTS = OUTPUT / "checkpoints"
STATE_PATH = OUTPUT / "run_state.json"
MANIFEST_PATH = OUTPUT / "confirmation_manifest.json"
LOCK_PATH = OUTPUT / ".runner.lock"
PEAK_RSS_BYTES = 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def observe_rss() -> int:
    global PEAK_RSS_BYTES
    value = int(psutil.Process().memory_info().rss)
    PEAK_RSS_BYTES = max(PEAK_RSS_BYTES, value)
    return value


def current_environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "statsmodels": statsmodels.__version__,
    }


def source_paths() -> dict[str, Path]:
    return {
        "confirmation_config": CONFIG_PATH,
        "confirmation_protocol": PROTOCOL_PATH,
        "confirmation_runner": RUNNER_PATH,
        "selection_freeze_script": FREEZE_SCRIPT_PATH,
        "confirmation_summarizer": SUMMARIZER_PATH,
        "confirmation_unit_tests": TEST_PATH,
        "statistical_module": ROOT / "src/kuntur_data/phase2/statistical_v12.py",
        "calibration_generator": Path(calibration.__file__).resolve(),
        "development_helpers": Path(development.__file__).resolve(),
    }


def source_hashes() -> dict[str, str]:
    return {name: sha256(path) for name, path in source_paths().items()}


def verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    for path in (CONFIG_PATH, PROTOCOL_PATH, FREEZE_PATH, FREEZE_RECEIPT_PATH, FREEZE_SCRIPT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {path.relative_to(ROOT)}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    receipt = json.loads(FREEZE_RECEIPT_PATH.read_text(encoding="utf-8"))
    hashes = source_hashes()
    if config.get("selection_freeze") != str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"):
        raise RuntimeError("Confirmation config does not point to the current freeze revision")
    if frozen.get("status") != "FROZEN_BEFORE_INDEPENDENT_CONFIRMATION" or frozen.get("confirmation_authorized") is not True:
        raise RuntimeError("No authorized, frozen v1.3 candidate selection exists")
    if receipt.get("selection_manifest_sha256") != sha256(FREEZE_PATH):
        raise RuntimeError("Selection-freeze receipt hash mismatch")
    if frozen.get("frozen_candidate", {}).get("method_name") != "SARIMAX_SEASONAL_AR12":
        raise RuntimeError("Frozen candidate differs from this confirmation runner")
    if frozen.get("frozen_candidate", {}).get("regime_id") != "iid_gaussian":
        raise RuntimeError("Only the frozen IID regime is in scope")
    if frozen.get("frozen_design", {}).get("series_seed_namespace") != config.get("design", {}).get("series_seed_namespace"):
        raise RuntimeError("Confirmation seed namespace differs from the frozen candidate")
    if frozen.get("frozen_acceptance_criteria") != config.get("acceptance_criteria"):
        raise RuntimeError("Acceptance criteria changed after candidate freeze")
    if frozen.get("frozen_resource_budget") != config.get("resource_budget"):
        raise RuntimeError("Resource budget changed after candidate freeze")
    for name, digest in hashes.items():
        if frozen.get("source_hashes", {}).get(name) != digest:
            raise RuntimeError(f"Frozen source SHA-256 changed: {name}")
    if current_environment() != frozen.get("runtime_environment"):
        raise RuntimeError("Python or numerical-library versions differ from frozen development environment")
    if config.get("protected_inputs", {}).get("v1_2_final_validation_seeds_used") is not False:
        raise RuntimeError("Confirmation config allows v1.2 final-validation seeds")
    if config.get("protected_inputs", {}).get("v1_3_development_expansion_seeds_used") is not False:
        raise RuntimeError("Confirmation config allows development seed reuse")
    return config, frozen, hashes


def checkpoint_filename(slope: float, method_name: str) -> str:
    label = f"{slope:.2f}".replace("-", "m").replace(".", "p")
    return f"cell_iid_gaussian_n276_s{label}_{method_name.lower()}.json"


def acquire_lock() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    stale_lock: dict[str, Any] | None = None
    if LOCK_PATH.exists():
        try:
            prior = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            pid = int(prior["pid"])
        except Exception as exc:
            raise RuntimeError("Confirmation lock is malformed; inspect the confirmation process before changing it") from exc
        if psutil.pid_exists(pid):
            try:
                command = " ".join(psutil.Process(pid).cmdline())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                command = ""
            if "run_corrective_confirmation_v1_3.py" in command:
                raise RuntimeError(f"Independent-confirmation lock belongs to active PID {pid}")
        stale_lock = {"prior_lock": prior, "prior_lock_sha256": sha256(LOCK_PATH), "prior_pid_reused_or_stopped": True}
        LOCK_PATH.unlink()
    payload = {"pid": os.getpid(), "started_utc": utc_now(), "command": "run_corrective_confirmation_v1_3.py"}
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return stale_lock or {}


def release_lock() -> None:
    if not LOCK_PATH.exists():
        return
    try:
        prior = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        if int(prior.get("pid", -1)) == os.getpid():
            LOCK_PATH.unlink()
    except (OSError, ValueError, json.JSONDecodeError):
        return


def seed_for(namespace: str, *parts: object) -> int:
    payload = "|".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def planned_seed_inventory(config: dict[str, Any]) -> dict[str, Any]:
    namespace = config["design"]["series_seed_namespace"]
    regime = config["design"]["regimes"][0]
    method_name = config["candidate"]["method_name"]
    seeds: list[int] = []
    for slope in config["design"]["slopes_per_decade"]:
        for rep in range(int(config["design"]["outer_replicates_per_cell"])):
            seeds.append(seed_for(namespace, "series", regime, f"{float(slope):.8f}", rep))
            seeds.append(seed_for(namespace, "fit", regime, f"{float(slope):.8f}", rep, method_name))
    if len(seeds) != len(set(seeds)) or any(seed < 2**64 for seed in seeds):
        raise RuntimeError("Confirmation seeds are not unique unsigned 128-bit values")
    return {
        "seed_count": len(seeds),
        "unique_seed_count": len(set(seeds)),
        "all_seeds_at_least_64_bit": all(seed >= 2**64 for seed in seeds),
        "seed_inventory_sha256": hashlib.sha256("\n".join(str(value) for value in seeds).encode("ascii")).hexdigest(),
    }


def resource_snapshot(cumulative_cpu: float, cumulative_wall: float, session_cpu: float, session_wall: float) -> dict[str, Any]:
    return {
        "cumulative_cpu_seconds": round(cumulative_cpu + session_cpu, 3),
        "cumulative_active_wall_seconds": round(cumulative_wall + session_wall, 3),
        "peak_rss_bytes_observed": max(PEAK_RSS_BYTES, int(psutil.Process().memory_info().rss)),
    }


def run() -> dict[str, Any]:
    config, frozen, hashes = verify_frozen_inputs()
    lock_history = acquire_lock()
    try:
        CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        budget = config["resource_budget"]
        namespace = config["design"]["series_seed_namespace"]
        slopes = [float(value) for value in config["design"]["slopes_per_decade"]]
        reps_target = int(config["design"]["outer_replicates_per_cell"])
        method_name = str(config["candidate"]["method_name"])
        method_key = str(config["candidate"]["method_key"])
        frozen_source_hashes = {
            **hashes,
            "selection_freeze_manifest": sha256(FREEZE_PATH),
            "selection_freeze_receipt": sha256(FREEZE_RECEIPT_PATH),
        }
        plan = {
            "regime_count": 1,
            "method_count": 1,
            "slope_cells": len(slopes),
            "outer_replicates_per_cell": reps_target,
            "planned_fit_count": len(slopes) * reps_target,
            "estimated_cpu_seconds": budget["estimated_confirmation_cpu_seconds"],
            "reserved_cpu_seconds": budget["estimated_reserved_cpu_seconds"],
            "hard_max_cpu_seconds": budget["hard_max_cpu_seconds"],
            "hard_max_wall_seconds": budget["hard_max_wall_seconds"],
            "hard_max_rss_bytes": budget["hard_max_rss_bytes"],
            "max_processes": budget["max_processes"],
            "blas_threads": budget["blas_threads"],
            "slurm_jobs": budget["slurm_jobs"],
        }
        old_manifest: dict[str, Any] = {}
        old_state: dict[str, Any] = {}
        if MANIFEST_PATH.exists():
            old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            if old_manifest.get("source_hashes") != frozen_source_hashes or old_manifest.get("config_sha256") != sha256(CONFIG_PATH):
                raise RuntimeError("Existing confirmation run used different frozen inputs; refusing to resume")
            if old_manifest.get("status") == "INDEPENDENT_CONFIRMATION_COMPLETE":
                return old_manifest
        if STATE_PATH.exists():
            old_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        cumulative_cpu = float(old_state.get("cumulative_cpu_seconds", 0.0))
        cumulative_wall = float(old_state.get("cumulative_active_wall_seconds", 0.0))
        start_cpu = time.process_time()
        start_wall = time.perf_counter()
        seed_inventory = planned_seed_inventory(config)
        manifest: dict[str, Any] = {
            "schema_version": "kuntur-v1.3-independent-confirmation-run-manifest-v1",
            "protocol_id": config["protocol_id"],
            "status": "RUNNING",
            "classification": "INDEPENDENT_CONFIRMATION_IN_PROGRESS",
            "started_or_resumed_utc": utc_now(),
            "candidate": config["candidate"],
            "design": config["design"],
            "acceptance_criteria": config["acceptance_criteria"],
            "plan": plan,
            "config_sha256": sha256(CONFIG_PATH),
            "protocol_sha256": sha256(PROTOCOL_PATH),
            "selection_freeze_sha256": sha256(FREEZE_PATH),
            "selection_freeze_payload_sha256": frozen["freeze_payload_sha256"],
            "source_hashes": frozen_source_hashes,
            "runtime_environment": current_environment(),
            "seed_inventory": seed_inventory,
            "resource_usage": {**resource_snapshot(cumulative_cpu, cumulative_wall, 0.0, 0.0), "hard_caps": budget},
            "stale_lock_recovered": lock_history,
            "registry_updated": False,
            "v1_2_registry_modified": False,
            "sealed_2025_score_touched": False,
        }
        atomic_json(MANIFEST_PATH, manifest)
        periods = pd.period_range(config["design"]["calendar_start"], periods=int(config["design"]["n_calendar_months"]), freq="M")
        x = _design(periods)
        resource_stop = False
        stop_reason: str | None = None
        for slope in slopes:
            checkpoint = CHECKPOINTS / checkpoint_filename(slope, method_name)
            descriptor = {
                "regime": "iid_gaussian",
                "slope_per_decade": slope,
                "method_name": method_name,
                "method_key": method_key,
                "n_calendar_months": len(periods),
                "outer_replicates": reps_target,
                "seed_namespace": namespace,
                "config_sha256": sha256(CONFIG_PATH),
                "source_hashes": frozen_source_hashes,
                "selection_freeze_sha256": sha256(FREEZE_PATH),
            }
            cell_hash = canonical_hash(descriptor)
            records: list[dict[str, Any]] = []
            if checkpoint.exists():
                prior = json.loads(checkpoint.read_text(encoding="utf-8"))
                if prior.get("cell_config_sha256") != cell_hash or prior.get("source_hashes") != frozen_source_hashes:
                    raise RuntimeError(f"Checkpoint does not match frozen inputs: {checkpoint.name}")
                records = prior.get("replicates", [])
                if len(records) != int(prior.get("completed_replicates", -1)) or [r.get("replicate") for r in records] != list(range(len(records))):
                    raise RuntimeError(f"Checkpoint replicate prefix is invalid: {checkpoint.name}")
                if prior.get("status") == "COMPLETE" and len(records) == reps_target:
                    print(f"CELL_COMPLETE iid_gaussian|slope={slope:.2f}|{method_name} reps={len(records)}")
                    continue
            resource_stop = False
            for rep in range(len(records), reps_target):
                elapsed_cpu = time.process_time() - start_cpu
                elapsed_wall = time.perf_counter() - start_wall
                rss = observe_rss()
                if cumulative_cpu + elapsed_cpu >= float(budget["hard_max_cpu_seconds"]):
                    resource_stop, stop_reason = True, "CPU_HARD_CAP"
                    break
                if cumulative_wall + elapsed_wall >= float(budget["hard_max_wall_seconds"]):
                    resource_stop, stop_reason = True, "ACTIVE_WALL_HARD_CAP"
                    break
                if rss >= int(budget["hard_max_rss_bytes"]):
                    resource_stop, stop_reason = True, "RSS_HARD_CAP"
                    break
                series_seed = seed_for(namespace, "series", "iid_gaussian", f"{slope:.8f}", rep)
                estimator_seed = seed_for(namespace, "fit", "iid_gaussian", f"{slope:.8f}", rep, method_name)
                rng = np.random.default_rng(series_seed)
                y, generated_periods, observed = calibration.make_series("iid_gaussian", len(periods), slope, rng)
                if not generated_periods.equals(periods) or not bool(observed.all()):
                    raise RuntimeError("IID confirmation generator returned unexpected calendar or missing support")
                try:
                    fit = _fit_method(
                        method_key, y, x, periods,
                        hac_lag_months=12,
                        n_bootstrap=int(config.get("bootstrap_inner_draws", 399)),
                        random_seed=estimator_seed,
                        block_length=12,
                    )
                except Exception as exc:
                    fit = _Fit(None, converged=False, note=f"FIT_EXCEPTION_{type(exc).__name__}")
                slope_estimate = development.finite_float(getattr(fit, "slope", None))
                p_value = development.finite_float(getattr(fit, "p_value", None))
                ci_low = development.finite_float(getattr(fit, "ci_low", None))
                ci_high = development.finite_float(getattr(fit, "ci_high", None))
                converged = bool(getattr(fit, "converged", False)) and slope_estimate is not None
                ci_width = ci_high - ci_low if ci_low is not None and ci_high is not None else None
                oracle_width = calibration.oracle_ci_width("iid_gaussian", periods, observed)
                residual = development.residual_diagnostic(fit, periods, observed)
                records.append({
                    "replicate": rep,
                    "series_seed": str(series_seed),
                    "estimator_seed": str(estimator_seed),
                    "supported": True,
                    "converged": converged,
                    "inferential_output": bool(converged and p_value is not None),
                    "slope_estimate_per_decade": slope_estimate,
                    "p_value": p_value,
                    "reject_at_005": bool(converged and p_value is not None and p_value < 0.05),
                    "ci_low_per_decade": ci_low,
                    "ci_high_per_decade": ci_high,
                    "ci_covers_true_slope": bool(converged and ci_low is not None and ci_high is not None and ci_low <= slope <= ci_high),
                    "ci_width": development.finite_float(ci_width),
                    "oracle_ci_width": development.finite_float(oracle_width),
                    "ci_width_over_oracle": development.finite_float(ci_width / oracle_width) if ci_width is not None and oracle_width is not None and oracle_width > 0 else None,
                    "residual_diagnostic": residual,
                    "note": str(getattr(fit, "note", "") or "")[:500],
                    "warnings": [str(item)[:240] for item in getattr(fit, "warnings", ())],
                })
                elapsed_cpu = time.process_time() - start_cpu
                elapsed_wall = time.perf_counter() - start_wall
                rss = observe_rss()
                cap_after_fit = (
                    cumulative_cpu + elapsed_cpu >= float(budget["hard_max_cpu_seconds"])
                    or cumulative_wall + elapsed_wall >= float(budget["hard_max_wall_seconds"])
                    or rss >= int(budget["hard_max_rss_bytes"])
                )
                checkpoint_due = len(records) % int(budget["checkpoint_every_replicates"]) == 0 or len(records) == reps_target
                if checkpoint_due or cap_after_fit:
                    payload = {
                        "schema_version": "kuntur-v1.3-independent-confirmation-cell-v1",
                        "status": "COMPLETE" if len(records) == reps_target else "IN_PROGRESS",
                        "classification": "INDEPENDENT_CONFIRMATION_RAW_EVIDENCE",
                        "cell": descriptor,
                        "cell_config_sha256": cell_hash,
                        "source_hashes": frozen_source_hashes,
                        "selection_freeze_sha256": sha256(FREEZE_PATH),
                        "completed_replicates": len(records),
                        "replicates": records,
                        "updated_utc": utc_now(),
                    }
                    atomic_json(checkpoint, payload)
                    state = {
                        "schema_version": "kuntur-v1.3-independent-confirmation-run-state-v1",
                        "status": "BOUNDED_PARTIAL" if cap_after_fit and len(records) < reps_target else "RUNNING",
                        "updated_utc": utc_now(),
                        "completed_cells": sum(1 for p in CHECKPOINTS.glob("cell_*.json") if json.loads(p.read_text(encoding="utf-8")).get("status") == "COMPLETE"),
                        "planned_cells": len(slopes),
                        "cumulative_cpu_seconds": round(cumulative_cpu + elapsed_cpu, 3),
                        "cumulative_active_wall_seconds": round(cumulative_wall + elapsed_wall, 3),
                        "peak_rss_bytes_observed": PEAK_RSS_BYTES,
                        "last_checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                        "last_checkpoint_replicates": len(records),
                        "last_checkpoint_sha256": sha256(checkpoint),
                    }
                    atomic_json(STATE_PATH, state)
                    manifest["resource_usage"] = {**resource_snapshot(cumulative_cpu, cumulative_wall, elapsed_cpu, elapsed_wall), "hard_caps": budget}
                    manifest["last_checkpoint"] = state["last_checkpoint"]
                    manifest["stop_reason"] = stop_reason if cap_after_fit else None
                    atomic_json(MANIFEST_PATH, manifest)
                    print(f"CHECKPOINT {checkpoint.name} reps={len(records)}/{reps_target} cpu={state['cumulative_cpu_seconds']:.1f}s wall={state['cumulative_active_wall_seconds']:.1f}s")
                    if cap_after_fit:
                        resource_stop = True
                        stop_reason = "CPU_HARD_CAP" if cumulative_cpu + elapsed_cpu >= float(budget["hard_max_cpu_seconds"]) else "ACTIVE_WALL_HARD_CAP" if cumulative_wall + elapsed_wall >= float(budget["hard_max_wall_seconds"]) else "RSS_HARD_CAP"
                        break
            if resource_stop:
                break
            if len(records) == reps_target:
                print(f"CELL_COMPLETE iid_gaussian|slope={slope:.2f}|{method_name} reps={len(records)}")
            else:
                break

        completed_cells = 0
        completed_fits = 0
        for path in CHECKPOINTS.glob("cell_*.json"):
            cell = json.loads(path.read_text(encoding="utf-8"))
            if cell.get("status") == "COMPLETE":
                completed_cells += 1
                completed_fits += int(cell.get("completed_replicates", 0))
        elapsed_cpu = time.process_time() - start_cpu
        elapsed_wall = time.perf_counter() - start_wall
        status = "INDEPENDENT_CONFIRMATION_COMPLETE" if completed_cells == len(slopes) and completed_fits == len(slopes) * reps_target else "BOUNDED_PARTIAL"
        if status == "BOUNDED_PARTIAL" and stop_reason is None:
            stop_reason = "NOT_ALL_FROZEN_CELLS_COMPLETE"
        manifest.update({
            "status": status,
            "classification": "INDEPENDENT_CONFIRMATION_RAW_EVIDENCE" if status == "INDEPENDENT_CONFIRMATION_COMPLETE" else "INDEPENDENT_CONFIRMATION_PARTIAL",
            "completed_utc": utc_now() if status == "INDEPENDENT_CONFIRMATION_COMPLETE" else None,
            "completed_cells": completed_cells,
            "completed_outer_fits": completed_fits,
            "stop_reason": stop_reason,
            "resource_usage": {**resource_snapshot(cumulative_cpu, cumulative_wall, elapsed_cpu, elapsed_wall), "hard_caps": budget},
            "seed_inventory": seed_inventory,
        })
        atomic_json(MANIFEST_PATH, manifest)
        atomic_json(STATE_PATH, {
            "schema_version": "kuntur-v1.3-independent-confirmation-run-state-v1",
            "status": status,
            "classification": manifest["classification"],
            "updated_utc": utc_now(),
            "completed_cells": completed_cells,
            "planned_cells": len(slopes),
            "completed_outer_fits": completed_fits,
            "cumulative_cpu_seconds": manifest["resource_usage"]["cumulative_cpu_seconds"],
            "cumulative_active_wall_seconds": manifest["resource_usage"]["cumulative_active_wall_seconds"],
            "peak_rss_bytes_observed": manifest["resource_usage"]["peak_rss_bytes_observed"],
            "stop_reason": stop_reason,
        })
        return manifest
    finally:
        release_lock()


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan-only", action="store_true", help="verify frozen inputs and report the bounded plan without simulation")
    group.add_argument("--execute", action="store_true", help="run/resume the already frozen independent confirmation")
    args = parser.parse_args()
    config, frozen, hashes = verify_frozen_inputs()
    budget = config["resource_budget"]
    plan = {
        "status": "READY_TO_EXECUTE" if args.plan_only else "EXECUTION_REQUESTED",
        "selection_freeze_status": frozen["status"],
        "candidate": config["candidate"]["method_name"],
        "regimes": config["design"]["regimes"],
        "slopes_per_decade": config["design"]["slopes_per_decade"],
        "fits": len(config["design"]["slopes_per_decade"]) * int(config["design"]["outer_replicates_per_cell"]),
        "seed_namespace": config["design"]["series_seed_namespace"],
        "source_hashes_verified": hashes,
        "resource_budget": budget,
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    result = run()
    print(json.dumps({"status": result["status"], "manifest": str(MANIFEST_PATH.relative_to(ROOT)), "completed_cells": result.get("completed_cells"), "completed_outer_fits": result.get("completed_outer_fits"), "resource_usage": result.get("resource_usage"), "stop_reason": result.get("stop_reason")}, indent=2))


if __name__ == "__main__":
    main()
