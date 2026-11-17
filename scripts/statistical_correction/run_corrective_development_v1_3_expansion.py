"""Run the frozen, resumable KUNTUR v1.3 corrective-development expansion.

This is development-only. It does not modify v1.2 evidence, the inference
registry, environmental scores, or the consumed hindcast.
"""
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
from scipy.stats import beta
from statsmodels.stats.diagnostic import acorr_ljungbox

import run_calibration as calibration
from kuntur_data.phase2.statistical_v12 import _Fit, _design, _fit_method

CONFIG_PATH = ROOT / "configs/statistical_corrective_development_expansion_v1_3.json"
PROTOCOL_PATH = ROOT / "docs/protocols/statistical_corrective_development_expansion_v1_3.md"
MODULE_PATH = ROOT / "src/kuntur_data/phase2/statistical_v12.py"
GENERATOR_PATH = Path(calibration.__file__).resolve()
SCRIPT_PATH = Path(__file__).resolve()
OUTPUT = ROOT / "results/statistical_correction/corrective_development_v1_3/expanded_20261117"
CHECKPOINTS = OUTPUT / "checkpoints"
STATE_PATH = OUTPUT / "run_state.json"
MANIFEST_PATH = OUTPUT / "development_manifest.json"
PROGRESS_PATH = OUTPUT / "progress.json"
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
    current = int(psutil.Process().memory_info().rss)
    PEAK_RSS_BYTES = max(PEAK_RSS_BYTES, current)
    return current


def derive_seed(namespace: str, *parts: object) -> int:
    payload = "|".join((namespace, *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def cp_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def calendar_acf(values: np.ndarray, periods: pd.PeriodIndex, max_lag: int = 24) -> dict[str, dict[str, float | int | None]]:
    ordinals = periods.year.to_numpy(dtype=int) * 12 + periods.month.to_numpy(dtype=int) - 1
    lookup = {int(month): float(value) for month, value in zip(ordinals, values) if np.isfinite(value)}
    output: dict[str, dict[str, float | int | None]] = {}
    for lag in range(1, max_lag + 1):
        left: list[float] = []
        right: list[float] = []
        for month, value in lookup.items():
            other = lookup.get(month + lag)
            if other is not None:
                left.append(value)
                right.append(other)
        acf: float | None = None
        if len(left) >= 3 and np.std(left) > 0 and np.std(right) > 0:
            acf = float(np.corrcoef(left, right)[0, 1])
        output[str(lag)] = {"acf": acf, "pair_count": len(left)}
    return output


def residual_diagnostic(fit: Any, periods: pd.PeriodIndex, observed: np.ndarray) -> dict[str, Any]:
    residual = getattr(fit, "residuals", None)
    if residual is None:
        return {"kind": "NO_METHOD_RESIDUALS", "ljung_box_failure": None, "acf": None}
    residual = np.asarray(residual, dtype=float).copy()
    if len(residual) != len(periods):
        return {
            "kind": "RESIDUAL_LENGTH_MISMATCH",
            "n_residual": len(residual),
            "n_calendar": len(periods),
            "ljung_box_failure": None,
            "acf": None,
        }
    residual[~observed] = np.nan
    acf = calendar_acf(residual, periods)
    selected_acf = {lag: acf[lag] for lag in ("1", "12", "24")}
    if observed.all() and np.isfinite(residual).all() and len(residual) > 37:
        try:
            result = acorr_ljungbox(residual, lags=[12, 24], return_df=True)
            pvalues = {str(int(lag)): finite_float(result.loc[lag, "lb_pvalue"]) for lag in result.index}
            failed = any(value is not None and value < 0.01 for value in pvalues.values())
            return {
                "kind": "LJUNG_BOX_COMPLETE_SUPPORT",
                "p_values": pvalues,
                "ljung_box_failure": failed,
                "acf_lags_1_12_24": selected_acf,
            }
        except (ValueError, np.linalg.LinAlgError):
            return {
                "kind": "LJUNG_BOX_UNAVAILABLE",
                "ljung_box_failure": None,
                "acf_lags_1_12_24": selected_acf,
            }
    return {
        "kind": "EXACT_CALENDAR_ACF_DESCRIPTIVE_ONLY",
        "ljung_box_failure": None,
        "acf_lags_1_12_24": selected_acf,
        "interpretation": "pairwise exact-calendar offsets; no calibrated p-value or acceptance gate",
    }


def dgp_error_acf(y: np.ndarray, periods: pd.PeriodIndex, slope: float) -> dict[str, dict[str, float | int | None]]:
    t = np.arange(len(periods), dtype=float)
    month = periods.month.to_numpy(dtype=float)
    seasonal = 0.55 * np.sin(2 * np.pi * (month - 1) / 12) + 0.18 * np.cos(4 * np.pi * (month - 1) / 12)
    error = np.asarray(y, dtype=float) - (12.0 + seasonal + slope * t / 120.0)
    return calendar_acf(error, periods)


def method_has_complete_requirement(method: str) -> bool:
    return method in {"ols_hac", "gls_ar1", "moving_block_bootstrap", "seasonal_block_bootstrap"}


def cell_filename(regime: str, slope: float, method_name: str) -> str:
    slope_label = f"{slope:.2f}".replace("-", "m").replace(".", "p")
    return f"cell_{regime}_n276_s{slope_label}_{method_name.lower()}.json"


def input_hashes() -> dict[str, str]:
    return {
        "configuration": sha256(CONFIG_PATH),
        "protocol": sha256(PROTOCOL_PATH),
        "script": sha256(SCRIPT_PATH),
        "statistical_module": sha256(MODULE_PATH),
        "calibration_generator": sha256(GENERATOR_PATH),
    }


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    method_regime = sum(len(methods) for methods in config["regimes"].values())
    cells = method_regime * len(config["slopes_per_decade"])
    fits = cells * int(config["outer_replicates_per_cell"])
    budget = config["resource_budget"]
    return {
        "regime_count": len(config["regimes"]),
        "method_regime_combinations": method_regime,
        "slope_count": len(config["slopes_per_decade"]),
        "cell_count": cells,
        "outer_replicates_per_cell": int(config["outer_replicates_per_cell"]),
        "planned_fit_count": fits,
        "bootstrap_inner_draws": int(config["bootstrap_inner_draws"]),
        "expected_cpu_hours": budget["conservative_estimated_expansion_cpu_hours"],
        "max_cpu_hours": budget["max_cpu_hours"],
        "max_wall_hours": budget["max_wall_hours"],
        "max_rss_bytes": budget["max_rss_bytes"],
        "max_processes": budget["max_processes"],
        "blas_threads": budget["blas_threads"],
        "slurm_jobs": budget["slurm_jobs"],
    }


def verify_seed_namespace(config: dict[str, Any]) -> dict[str, Any]:
    namespace = config["development_seed_namespace"]
    if namespace != "KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-EXPANSION-V1.3-2026-11-17":
        raise RuntimeError("Unexpected expansion development namespace")
    if config["reserved_confirmation_seed_namespace_used"]:
        raise RuntimeError("The reserved independent-confirmation namespace must remain unused")
    if namespace == config["reserved_confirmation_seed_namespace"] or "CONFIRMATION" in namespace:
        raise RuntimeError("Development and confirmation namespaces are not separate")
    sample: list[int] = []
    for regime, methods in config["regimes"].items():
        for slope in config["slopes_per_decade"]:
            for rep in (0, 1, config["outer_replicates_per_cell"] - 1):
                sample.append(derive_seed(namespace, "series", regime, f"{float(slope):.8f}", rep))
                for method in methods:
                    sample.append(derive_seed(namespace, "fit", regime, f"{float(slope):.8f}", rep, method))
    if any(seed < 2**64 for seed in sample) or len(sample) != len(set(sample)):
        raise RuntimeError("Fresh development seed sample is not unique 128-bit output")
    return {
        "namespace": namespace,
        "sampled_seed_count": len(sample),
        "all_sampled_seeds_128_bit": True,
        "sampled_seed_collision_count": len(sample) - len(set(sample)),
        "confirmation_namespace_used": False,
        "v1_2_seed_bases_used": False,
        "prior_608_fit_pilot_namespace_reused": False,
    }


def acquire_lock() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            prior = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
            pid = int(prior["pid"])
        except Exception:
            raise RuntimeError(f"Malformed lock at {LOCK_PATH}; verify no expansion runner is active before removing it")
        if psutil.pid_exists(pid):
            raise RuntimeError(f"Expansion runner lock belongs to live process {pid}")
        LOCK_PATH.unlink()
    fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({"pid": os.getpid(), "started_utc": utc_now()}).encode("utf-8"))
    os.close(fd)
    return os.getpid()


def checkpoint_payload(
    *, status: str, config: dict[str, Any], cell: dict[str, Any], config_sha: str,
    records: list[dict[str, Any]], sources: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "kuntur-v1.3-expansion-cell-v1",
        "status": status,
        "classification": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        "cell_key": cell["cell_key"],
        "config": cell,
        "config_sha256": config_sha,
        "source_hashes": sources,
        "seed_namespace": config["development_seed_namespace"],
        "completed_replicates": len(records),
        "replicates": records,
        "updated_utc": utc_now(),
    }


def run_cell(
    config: dict[str, Any], cell: dict[str, Any], sources: dict[str, str],
    cumulative_cpu: float, cumulative_wall: float, session_cpu_start: float,
    session_wall_start: float,
) -> tuple[dict[str, Any], bool, float, float]:
    budget = config["resource_budget"]
    reps_target = int(config["outer_replicates_per_cell"])
    namespace = config["development_seed_namespace"]
    checkpoint_path = CHECKPOINTS / cell_filename(cell["regime"], cell["slope_per_decade"], cell["method_name"])
    descriptor = {
        **cell,
        "n_months": int(config["n_months"]),
        "outer_replicates": reps_target,
        "bootstrap_inner_draws": int(config["bootstrap_inner_draws"]),
        "seed_namespace": namespace,
        "source_hashes": sources,
    }
    cell_config_sha = canonical_hash(descriptor)
    records: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        previous = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if previous.get("config_sha256") != cell_config_sha or previous.get("source_hashes") != sources:
            raise RuntimeError(f"Existing cell checkpoint has changed frozen inputs: {checkpoint_path.name}")
        records = previous.get("replicates", [])
        if len(records) != int(previous.get("completed_replicates", -1)):
            raise RuntimeError(f"Checkpoint replicate count mismatch: {checkpoint_path.name}")
        if [row.get("replicate") for row in records] != list(range(len(records))):
            raise RuntimeError(f"Checkpoint replicate sequence is not contiguous: {checkpoint_path.name}")
        if previous.get("status") == "COMPLETE":
            if len(records) != reps_target:
                raise RuntimeError(f"Complete checkpoint has an unexpected length: {checkpoint_path.name}")
            return previous, True, cumulative_cpu, cumulative_wall

    periods = pd.period_range("2000-01", periods=int(config["n_months"]), freq="M")
    x = _design(periods)
    method_name = cell["method_name"]
    spec = config["candidate_method_specs"][method_name]
    method = str(spec["method"])
    slope = float(cell["slope_per_decade"])
    last_checkpoint_count = len(records)
    for rep in range(len(records), reps_target):
        session_cpu = time.process_time() - session_cpu_start
        session_wall = time.perf_counter() - session_wall_start
        if cumulative_cpu + session_cpu >= float(budget["max_cpu_seconds"]):
            return checkpoint_payload(status="IN_PROGRESS", config=config, cell=descriptor, config_sha=cell_config_sha, records=records, sources=sources), False, cumulative_cpu, cumulative_wall
        if cumulative_wall + session_wall >= float(budget["max_wall_seconds"]):
            return checkpoint_payload(status="IN_PROGRESS", config=config, cell=descriptor, config_sha=cell_config_sha, records=records, sources=sources), False, cumulative_cpu, cumulative_wall
        if observe_rss() >= int(budget["max_rss_bytes"]):
            return checkpoint_payload(status="IN_PROGRESS", config=config, cell=descriptor, config_sha=cell_config_sha, records=records, sources=sources), False, cumulative_cpu, cumulative_wall

        series_seed = derive_seed(namespace, "series", cell["regime"], f"{slope:.8f}", rep)
        estimator_seed = derive_seed(namespace, "fit", cell["regime"], f"{slope:.8f}", rep, method_name)
        rng = np.random.default_rng(series_seed)
        y, generated_periods, observed = calibration.make_series(cell["regime"], int(config["n_months"]), slope, rng)
        if not generated_periods.equals(periods):
            raise RuntimeError("Generator calendar differs from the frozen calendar")
        common_dgp_acf = dgp_error_acf(y, periods, slope)
        support_eligible = not (method_has_complete_requirement(method) and not observed.all())
        if not support_eligible:
            fit = _Fit(None, converged=False, note="IRREGULAR_TIME_SUPPORT; time gaps were preserved and not compressed.")
        else:
            try:
                fit = _fit_method(
                    method,
                    y,
                    x,
                    periods,
                    hac_lag_months=int(spec["lag_months"]),
                    n_bootstrap=int(config["bootstrap_inner_draws"]),
                    random_seed=estimator_seed,
                    block_length=int(spec["block_length_months"]),
                )
            except Exception as exc:
                fit = _Fit(None, converged=False, note=f"FIT_EXCEPTION_{type(exc).__name__}")

        slope_estimate = finite_float(getattr(fit, "slope", None))
        p_value = finite_float(getattr(fit, "p_value", None))
        ci_low = finite_float(getattr(fit, "ci_low", None))
        ci_high = finite_float(getattr(fit, "ci_high", None))
        converged = bool(getattr(fit, "converged", False)) and slope_estimate is not None
        ci_width = ci_high - ci_low if ci_low is not None and ci_high is not None else None
        oracle_width = finite_float(calibration.oracle_ci_width(cell["regime"], periods, observed))
        residual = residual_diagnostic(fit, periods, observed)
        row = {
            "replicate": rep,
            "series_seed": str(series_seed),
            "estimator_seed": str(estimator_seed),
            "observed_fraction": float(np.mean(observed)),
            "support_eligible": bool(support_eligible),
            "converged": converged,
            "inferential_output": bool(converged and p_value is not None),
            "estimate_per_decade": slope_estimate,
            "p_value": p_value,
            "reject_at_005": bool(converged and p_value is not None and p_value < 0.05),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_width": finite_float(ci_width),
            "ci_covers_true_slope": bool(converged and ci_low is not None and ci_high is not None and ci_low <= slope <= ci_high),
            "oracle_ci_width": oracle_width,
            "ci_width_over_oracle": finite_float(ci_width / oracle_width) if ci_width is not None and oracle_width is not None and oracle_width > 0 else None,
            "residual_diagnostic": residual,
            "known_dgp_error_acf_lags_1_12_24": {lag: common_dgp_acf[lag] for lag in ("1", "12", "24")},
            "note": str(getattr(fit, "note", "") or "")[:500],
            "warnings": [str(item)[:240] for item in getattr(fit, "warnings", ())],
        }
        records.append(row)
        observe_rss()
        checkpoint_due = len(records) % int(config["checkpoint_every_replicates"]) == 0 or len(records) == reps_target
        resource_stop = (
            cumulative_cpu + (time.process_time() - session_cpu_start) >= float(budget["max_cpu_seconds"])
            or cumulative_wall + (time.perf_counter() - session_wall_start) >= float(budget["max_wall_seconds"])
            or observe_rss() >= int(budget["max_rss_bytes"])
        )
        if checkpoint_due or resource_stop:
            checkpoint_status = "COMPLETE" if len(records) == reps_target else "IN_PROGRESS"
            payload = checkpoint_payload(
                status=checkpoint_status,
                config=config,
                cell=descriptor,
                config_sha=cell_config_sha,
                records=records,
                sources=sources,
            )
            atomic_json(checkpoint_path, payload)
            elapsed_cpu = time.process_time() - session_cpu_start
            elapsed_wall = time.perf_counter() - session_wall_start
            state = {
                "schema_version": "kuntur-v1.3-expansion-run-state-v1",
                "status": "RUNNING" if not resource_stop else "BOUNDED_PARTIAL",
                "classification": "DEVELOPMENT_ONLY_NOT_VALIDATED",
                "updated_utc": utc_now(),
                "cumulative_cpu_seconds": round(cumulative_cpu + elapsed_cpu, 3),
                "cumulative_active_wall_seconds": round(cumulative_wall + elapsed_wall, 3),
                "peak_rss_bytes_observed": max(PEAK_RSS_BYTES, int(psutil.Process().memory_info().rss)),
                "last_checkpoint": checkpoint_path.relative_to(ROOT).as_posix(),
                "last_checkpoint_replicates": len(records),
                "last_checkpoint_sha256": sha256(checkpoint_path),
            }
            atomic_json(STATE_PATH, state)
            atomic_json(PROGRESS_PATH, state)
            last_checkpoint_count = len(records)
        if resource_stop:
            return checkpoint_payload(status="IN_PROGRESS", config=config, cell=descriptor, config_sha=cell_config_sha, records=records, sources=sources), False, cumulative_cpu, cumulative_wall

    payload = checkpoint_payload(
        status="COMPLETE",
        config=config,
        cell=descriptor,
        config_sha=cell_config_sha,
        records=records,
        sources=sources,
    )
    atomic_json(checkpoint_path, payload)
    return payload, True, cumulative_cpu, cumulative_wall


def inventory(config: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for regime, method_names in config["regimes"].items():
        for slope in config["slopes_per_decade"]:
            for method in method_names:
                path = CHECKPOINTS / cell_filename(regime, float(slope), method)
                if not path.exists():
                    result.append({
                        "path": path.relative_to(ROOT).as_posix(),
                        "status": "NOT_STARTED",
                        "completed_replicates": 0,
                        "bytes": 0,
                        "sha256": None,
                    })
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                result.append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "status": data.get("status", "UNKNOWN"),
                    "completed_replicates": int(data.get("completed_replicates", 0)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "cell_key": data.get("cell_key"),
                })
    return result


def write_manifest(config: dict[str, Any], plan: dict[str, Any], sources: dict[str, str], *, status: str, stop_reason: str | None, started_utc: str, cumulative_cpu: float, cumulative_wall: float, seed_check: dict[str, Any]) -> dict[str, Any]:
    items = inventory(config)
    completed = sum(item["status"] == "COMPLETE" for item in items)
    fits = sum(item["completed_replicates"] for item in items)
    payload = {
        "schema_version": "kuntur-statistical-corrective-development-expansion-manifest-v1.3",
        "status": status,
        "classification": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        "started_utc": started_utc,
        "updated_utc": utc_now(),
        "stop_reason": stop_reason,
        "cumulative_active_wall_seconds": round(cumulative_wall, 3),
        "cumulative_process_cpu_seconds": round(cumulative_cpu, 3),
        "peak_rss_bytes_observed": max(PEAK_RSS_BYTES, int(psutil.Process().memory_info().rss)),
        "resource_budget": config["resource_budget"],
        "plan": plan,
        "seed_separation_check": seed_check,
        "source_hashes": sources,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "completed_cells": completed,
        "total_cells": plan["cell_count"],
        "completed_outer_fits": fits,
        "planned_outer_fits": plan["planned_fit_count"],
        "checkpoint_inventory": items,
        "v1_2_decisions_or_gates_changed": False,
        "v1_2_registry_modified": False,
        "v1_2_final_validation_seeds_used": False,
        "prior_608_fit_pilot_repeated": False,
        "real_series_or_2025_hindcast_used": False,
        "confirmation_namespace_used": False,
        "confirmation_started": False,
        "scientific_acceptance": "DEVELOPMENT METRICS ONLY; independent confirmation has not been run",
    }
    atomic_json(MANIFEST_PATH, payload)
    return payload


def main() -> int:
    global PEAK_RSS_BYTES
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true", help="validate and print the frozen execution plan without writing or simulating")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    expected_namespace = "KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-EXPANSION-V1.3-2026-11-17"
    if config["development_seed_namespace"] != expected_namespace:
        raise RuntimeError("Frozen expansion seed namespace changed")
    if config["reserved_confirmation_seed_namespace_used"]:
        raise RuntimeError("Confirmation namespace must remain unused")
    seed_check = verify_seed_namespace(config)
    plan = build_plan(config)
    if plan["method_regime_combinations"] != 60 or plan["cell_count"] != 180 or plan["planned_fit_count"] != 180000:
        raise RuntimeError(f"Frozen plan shape mismatch: {plan}")
    if args.plan_only:
        print(json.dumps({"status": "PLAN_VALIDATED_NO_SIMULATION", "plan": plan, "seed_separation_check": seed_check}, indent=2, sort_keys=True))
        return 0

    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    acquire_lock()
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    sources = input_hashes()
    if MANIFEST_PATH.exists():
        old = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if old.get("source_hashes") != sources:
            raise RuntimeError("Frozen source hashes differ from the existing expansion manifest")
    if STATE_PATH.exists():
        old_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        cumulative_cpu = float(old_state.get("cumulative_cpu_seconds", 0.0))
        cumulative_wall = float(old_state.get("cumulative_active_wall_seconds", 0.0))
        PEAK_RSS_BYTES = int(old_state.get("peak_rss_bytes_observed", 0))
    elif MANIFEST_PATH.exists():
        old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cumulative_cpu = float(old_manifest.get("cumulative_process_cpu_seconds", 0.0))
        cumulative_wall = float(old_manifest.get("cumulative_active_wall_seconds", 0.0))
    else:
        cumulative_cpu = 0.0
        cumulative_wall = 0.0
    started_utc = utc_now()
    reason: str | None = None
    run_status = "RUNNING"
    try:
        if cumulative_cpu >= float(config["resource_budget"]["max_cpu_seconds"]):
            reason, run_status = "CUMULATIVE_CPU_CAP_ALREADY_REACHED", "BOUNDED_PARTIAL"
        elif cumulative_wall >= float(config["resource_budget"]["max_wall_seconds"]):
            reason, run_status = "CUMULATIVE_ACTIVE_WALL_CAP_ALREADY_REACHED", "BOUNDED_PARTIAL"
        else:
            for regime, methods in config["regimes"].items():
                for slope in config["slopes_per_decade"]:
                    for method_name in methods:
                        cell = {
                            "cell_key": f"{regime}|slope={float(slope):.2f}|{method_name}",
                            "regime": regime,
                            "slope_per_decade": float(slope),
                            "method_name": method_name,
                            "method_spec": config["candidate_method_specs"][method_name],
                        }
                        payload, complete, cumulative_cpu, cumulative_wall = run_cell(
                            config,
                            cell,
                            sources,
                            cumulative_cpu,
                            cumulative_wall,
                            started_cpu,
                            started_wall,
                        )
                        if not complete:
                            reason = "CPU_WALL_OR_RSS_BOUNDARY"
                            run_status = "BOUNDED_PARTIAL"
                            break
                        item_count = len(payload.get("replicates", []))
                        print(f"CELL_COMPLETE {cell['cell_key']} reps={item_count}", flush=True)
                        current_cpu = cumulative_cpu + (time.process_time() - started_cpu)
                        current_wall = cumulative_wall + (time.perf_counter() - started_wall)
                        cell_state = {
                            "schema_version": "kuntur-v1.3-expansion-run-state-v1",
                            "status": "RUNNING",
                            "classification": "DEVELOPMENT_ONLY_NOT_VALIDATED",
                            "updated_utc": utc_now(),
                            "cumulative_cpu_seconds": round(current_cpu, 3),
                            "cumulative_active_wall_seconds": round(current_wall, 3),
                            "peak_rss_bytes_observed": max(PEAK_RSS_BYTES, int(psutil.Process().memory_info().rss)),
                            "last_checkpoint": (CHECKPOINTS / cell_filename(regime, float(slope), method_name)).relative_to(ROOT).as_posix(),
                            "last_checkpoint_replicates": item_count,
                            "last_checkpoint_sha256": sha256(CHECKPOINTS / cell_filename(regime, float(slope), method_name)),
                        }
                        atomic_json(STATE_PATH, cell_state)
                        atomic_json(PROGRESS_PATH, cell_state)
                        write_manifest(
                            config,
                            plan,
                            sources,
                            status="RUNNING",
                            stop_reason=None,
                            started_utc=started_utc,
                            cumulative_cpu=current_cpu,
                            cumulative_wall=current_wall,
                            seed_check=seed_check,
                        )
                        if (
                            current_cpu >= float(config["resource_budget"]["max_cpu_seconds"])
                            or current_wall >= float(config["resource_budget"]["max_wall_seconds"])
                            or observe_rss() >= int(config["resource_budget"]["max_rss_bytes"])
                        ):
                            reason = "CPU_WALL_OR_RSS_BOUNDARY"
                            run_status = "BOUNDED_PARTIAL"
                            break
                    if reason:
                        break
                if reason:
                    break
        elapsed_cpu = time.process_time() - started_cpu
        elapsed_wall = time.perf_counter() - started_wall
        cumulative_cpu += elapsed_cpu
        cumulative_wall += elapsed_wall
        all_items = inventory(config)
        if not reason and all(item["status"] == "COMPLETE" for item in all_items):
            run_status = "DEVELOPMENT_EXPANSION_COMPLETE"
        elif not reason:
            run_status, reason = "BOUNDED_PARTIAL", "CELL_COMPLETION_MISMATCH"
        state = {
            "schema_version": "kuntur-v1.3-expansion-run-state-v1",
            "status": run_status,
            "classification": "DEVELOPMENT_ONLY_NOT_VALIDATED",
            "updated_utc": utc_now(),
            "cumulative_cpu_seconds": round(cumulative_cpu, 3),
            "cumulative_active_wall_seconds": round(cumulative_wall, 3),
            "peak_rss_bytes_observed": max(PEAK_RSS_BYTES, int(psutil.Process().memory_info().rss)),
            "stop_reason": reason,
        }
        atomic_json(STATE_PATH, state)
        atomic_json(PROGRESS_PATH, state)
        final = write_manifest(
            config,
            plan,
            sources,
            status=run_status,
            stop_reason=reason,
            started_utc=started_utc,
            cumulative_cpu=cumulative_cpu,
            cumulative_wall=cumulative_wall,
            seed_check=seed_check,
        )
        print(json.dumps({
            "status": final["status"],
            "plan": final["plan"],
            "completed_cells": final["completed_cells"],
            "completed_outer_fits": final["completed_outer_fits"],
            "cumulative_process_cpu_seconds": final["cumulative_process_cpu_seconds"],
            "cumulative_active_wall_seconds": final["cumulative_active_wall_seconds"],
            "peak_rss_bytes_observed": final["peak_rss_bytes_observed"],
            "stop_reason": final["stop_reason"],
            "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        }, indent=2, sort_keys=True), flush=True)
        return 0 if final["status"] == "DEVELOPMENT_EXPANSION_COMPLETE" else 2
    finally:
        if LOCK_PATH.exists():
            try:
                lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
                if int(lock.get("pid", -1)) == os.getpid():
                    LOCK_PATH.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
