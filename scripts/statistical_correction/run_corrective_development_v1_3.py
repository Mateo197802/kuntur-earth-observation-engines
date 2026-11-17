"""Run the fixed, resource-bounded v1.3 corrective-development pilot.

This script never runs the v1.2 selection/validation plan, never writes the
validated registry, and labels every output DEVELOPMENT_ONLY_NOT_VALIDATED.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Set numerical-library worker limits before importing NumPy/Statsmodels.
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_variable] = "1"

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

import run_calibration as frozen_calibration
from kuntur_data.phase2.statistical_v12 import _Fit, _design, _fit_method, calendar_lag_acf


CONFIG_PATH = ROOT / "configs/statistical_corrective_development_v1_3.json"
PROTOCOL_PATH = ROOT / "docs/protocols/statistical_corrective_development_v1_3.md"
MODULE_PATH = ROOT / "src/kuntur_data/phase2/statistical_v12.py"
OUTPUT = ROOT / "results/statistical_correction/corrective_development_v1_3"
CHECKPOINTS = OUTPUT / "checkpoints"
SEED_NAMESPACE = "KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-V1.3-2026-11-16"
MAX_WALL_SECONDS = 240
MAX_RSS_BYTES = 1_073_741_824
OUTER_REPS = 8
INNER_DRAWS = 199
PEAK_RSS_BYTES = 0

SPECS: dict[str, dict[str, Any]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def observe_rss() -> int:
    global PEAK_RSS_BYTES
    current = int(psutil.Process().memory_info().rss)
    PEAK_RSS_BYTES = max(PEAK_RSS_BYTES, current)
    return current


def derive_seed(*parts: object) -> int:
    payload = "|".join((SEED_NAMESPACE, *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def verify_seed_separation() -> None:
    if SEED_NAMESPACE != "KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-V1.3-2026-11-16":
        raise RuntimeError("Unexpected development seed namespace")
    if "CONFIRMATION" in SEED_NAMESPACE:
        raise RuntimeError("The reserved confirmation namespace must never be used by development")
    # v1.2 derives base seeds from 20261126/20261127 + n*10000 + small
    # scenario offsets, then adds at most 999*1009+15 for method streams.
    # Every derived v1.3 seed is 128-bit and therefore disjoint from that
    # historical integer seed range (which is below 2**32).
    sample = [derive_seed("series", regime, slope, rep) for regime in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["scenarios"] for slope in (0.0, 0.25) for rep in range(OUTER_REPS)]
    sample += [derive_seed("fit", regime, slope, rep, name) for regime, names in json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["scenarios"].items() for slope in (0.0, 0.25) for rep in range(OUTER_REPS) for name in names]
    if any(seed < 2**64 for seed in sample) or len(sample) != len(set(sample)):
        raise RuntimeError("Development seed collision or invalid seed width")


def cp_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials == 0:
        return None, None
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def numerical_gate_passes(failures: int, supported: int) -> bool:
    if supported <= 0 or failures < 0 or failures > supported:
        raise ValueError("invalid failure/support counts")
    return failures * 100 <= supported


def verify_integer_gate_boundary() -> None:
    # This is only a deterministic arithmetic regression check; it does not
    # rewrite the v1.2 numerical-failure classifications.
    assert numerical_gate_passes(9, 1000)
    assert numerical_gate_passes(10, 1000)
    assert not numerical_gate_passes(11, 1000)
    assert numerical_gate_passes(1, 100)


def residual_diagnostic(fit: Any, periods: pd.PeriodIndex, observed: np.ndarray, method: str) -> dict[str, Any]:
    if method not in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"}:
        return {"kind": "NOT_APPLICABLE", "flag": None}
    residual = getattr(fit, "residuals", None)
    if residual is None:
        return {"kind": "UNAVAILABLE", "flag": None}
    residual = np.asarray(residual, dtype=float)
    if observed.all() and np.isfinite(residual).all() and len(residual) > 37:
        try:
            result = acorr_ljungbox(residual, lags=[12, 24], return_df=True)
            pvalues = {str(int(lag)): float(result.loc[lag, "lb_pvalue"]) for lag in result.index}
            return {"kind": "LJUNG_BOX_COMPLETE_SUPPORT", "p_values": pvalues, "flag": any(value < 0.01 for value in pvalues.values())}
        except (ValueError, np.linalg.LinAlgError) as exc:
            return {"kind": "LJUNG_BOX_UNAVAILABLE", "error_type": type(exc).__name__, "flag": None}
    if len(residual) != len(periods):
        return {"kind": "UNAVAILABLE_RESIDUAL_LENGTH", "n_residual": len(residual), "n_calendar": len(periods), "flag": None}
    # For gapped fits, report exact-calendar-offset ACFs. This conservative
    # screening proxy is descriptive only and is never treated as a calibrated
    # hypothesis test or an acceptance gate.
    acf = calendar_lag_acf(pd.Series(residual, index=periods), max_lag_months=24)
    return {
        "kind": "EXACT_CALENDAR_ACF_SCREENING_PROXY",
        "lags_12_24": {lag: acf[lag] for lag in ("12", "24")},
        "flag": None,
        "interpretation": "descriptive exact-offset ACF; no p-value or gate",
    }


def evaluate_cell(regime: str, slope: float, method_name: str, spec: dict[str, Any], *, start_time: float, deadline: float, input_hashes: dict[str, str]) -> tuple[dict[str, Any], bool]:
    checkpoint = CHECKPOINTS / f"dev_{regime}_n276_s{str(slope).replace('-', 'm').replace('.', 'p')}_{method_name.lower()}.json"
    config = {
        "protocol_id": "kuntur-statistical-corrective-development-v1.3",
        "regime": regime,
        "n_months": 276,
        "slope_per_decade": slope,
        "method_name": method_name,
        "method_spec": spec,
        "outer_replicates": OUTER_REPS,
        "bootstrap_inner_draws": INNER_DRAWS,
        "development_seed_namespace": SEED_NAMESPACE,
        "source_hashes": input_hashes,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
    prior: dict[str, Any] | None = None
    if checkpoint.exists():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        if prior.get("config_sha256") != config_hash:
            raise RuntimeError(f"Existing v1.3 cell has a different frozen config: {checkpoint.name}")
        if prior.get("status") == "COMPLETE":
            return prior, True
    records = prior.get("replicates", []) if prior else []
    if len(records) > OUTER_REPS:
        raise RuntimeError(f"Cell has more replicates than configured: {checkpoint.name}")
    process = psutil.Process()
    periods = pd.period_range("2000-01", periods=276, freq="M")
    x = _design(periods)
    spec_method = str(spec["method"])
    for rep in range(len(records), OUTER_REPS):
        if time.perf_counter() >= deadline or observe_rss() > MAX_RSS_BYTES:
            status = "IN_PROGRESS"
            payload = {
                "status": status,
                "development_label": "DEVELOPMENT_ONLY_NOT_VALIDATED",
                "config": config,
                "config_sha256": config_hash,
                "completed_reps": len(records),
                "replicates": records,
            }
            atomic_json(checkpoint, payload)
            return payload, False
        series_seed = derive_seed("series", regime, slope, rep)
        fit_seed = derive_seed("fit", regime, slope, rep, method_name)
        if series_seed < 2**64 or fit_seed < 2**64:
            raise RuntimeError("Development seed is not separated from legacy seed range")
        rng = np.random.default_rng(series_seed)
        y, generated_periods, observed = frozen_calibration.make_series(regime, 276, slope, rng)
        if not generated_periods.equals(periods):
            raise RuntimeError("DGP calendar differs from the declared complete calendar")
        requires_complete = spec_method in {"gls_ar1", "moving_block_bootstrap", "seasonal_block_bootstrap"} or (spec_method == "ols_hac" and not observed.all())
        if requires_complete and not observed.all():
            fit = _Fit(None, converged=False, note="IRREGULAR_TIME_SUPPORT; candidate requires complete monthly support.")
            support_eligible = False
        else:
            try:
                fit = _fit_method(
                    spec_method,
                    y,
                    x,
                    periods,
                    hac_lag_months=int(spec["lag_months"]),
                    n_bootstrap=INNER_DRAWS,
                    random_seed=fit_seed,
                    block_length=int(spec["block_length_months"]),
                )
            except Exception as exc:
                fit = _Fit(None, converged=False, note=f"FIT_EXCEPTION_{type(exc).__name__}")
            support_eligible = True
        p_value = getattr(fit, "p_value", None)
        ci_low = getattr(fit, "ci_low", None)
        ci_high = getattr(fit, "ci_high", None)
        estimate = getattr(fit, "slope", None)
        numeric = lambda value: float(value) if value is not None and np.isfinite(value) else None
        record = {
            "replicate": rep,
            "series_seed": str(series_seed),
            "estimator_seed": str(fit_seed),
            "observed_fraction": float(np.mean(observed)),
            "support_eligible": support_eligible,
            "converged": bool(getattr(fit, "converged", False)),
            "estimate_per_decade": numeric(estimate),
            "p_value": numeric(p_value),
            "reject_at_005": bool(p_value is not None and np.isfinite(p_value) and p_value < 0.05),
            "ci_low": numeric(ci_low),
            "ci_high": numeric(ci_high),
            "ci_covers_true_slope": bool(ci_low is not None and ci_high is not None and np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= slope <= ci_high),
            "residual_diagnostic": residual_diagnostic(fit, periods, observed, spec_method),
            "note": str(getattr(fit, "note", "") or "")[:500],
            "warnings": [str(item)[:240] for item in getattr(fit, "warnings", ())],
        }
        records.append(record)
        observe_rss()
        if len(records) % 4 == 0 or len(records) == OUTER_REPS:
            atomic_json(checkpoint, {
                "status": "IN_PROGRESS" if len(records) < OUTER_REPS else "COMPLETE",
                "development_label": "DEVELOPMENT_ONLY_NOT_VALIDATED",
                "config": config,
                "config_sha256": config_hash,
                "completed_reps": len(records),
                "replicates": records,
            })
        if time.perf_counter() >= deadline or observe_rss() > MAX_RSS_BYTES:
            payload = {
                "status": "IN_PROGRESS",
                "development_label": "DEVELOPMENT_ONLY_NOT_VALIDATED",
                "config": config,
                "config_sha256": config_hash,
                "completed_reps": len(records),
                "replicates": records,
            }
            atomic_json(checkpoint, payload)
            return payload, False
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    return payload, True


def summarize(cell: dict[str, Any]) -> dict[str, Any]:
    records = cell.get("replicates", [])
    eligible = [row for row in records if row.get("support_eligible")]
    converged = [row for row in eligible if row.get("converged") and row.get("estimate_per_decade") is not None]
    inferential = [row for row in converged if row.get("p_value") is not None]
    intervals = [row for row in converged if row.get("ci_low") is not None and row.get("ci_high") is not None]
    failures = len(eligible) - len(converged)
    slope = float(cell["config"]["slope_per_decade"])
    type1_count = sum(bool(row["reject_at_005"]) for row in inferential) if slope == 0 else None
    coverage_count = sum(bool(row["ci_covers_true_slope"]) for row in intervals)
    residual = [row["residual_diagnostic"].get("flag") for row in records if row["residual_diagnostic"].get("flag") is not None]
    lo, hi = cp_interval(type1_count, len(inferential)) if type1_count is not None else (None, None)
    clo, chi = cp_interval(coverage_count, len(intervals))
    estimates = [float(row["estimate_per_decade"]) for row in converged]
    return {
        "regime": cell["config"]["regime"],
        "slope_per_decade": slope,
        "method": cell["config"]["method_name"],
        "status": cell["status"],
        "completed_reps": len(records),
        "support_eligible_reps": len(eligible),
        "converged_reps": len(converged),
        "numerical_failures": failures,
        "numerical_failure_rate": failures / len(eligible) if eligible else None,
        "development_integer_gate_1pct": numerical_gate_passes(failures, len(eligible)) if eligible else None,
        "inferential_output_fraction": len(inferential) / len(eligible) if eligible else None,
        "type1_rejections": type1_count,
        "type1_rate": type1_count / len(inferential) if type1_count is not None and inferential else None,
        "type1_cp95_diagnostic": [lo, hi] if lo is not None else None,
        "interval_replicates": len(intervals),
        "coverage_count": coverage_count,
        "coverage_rate": coverage_count / len(intervals) if intervals else None,
        "coverage_cp95_diagnostic": [clo, chi] if clo is not None else None,
        "slope_bias_per_decade": float(np.mean(estimates) - slope) if estimates else None,
        "residual_diagnostic_rejections": sum(bool(flag) for flag in residual) if residual else None,
        "residual_diagnostic_applicable_reps": len(residual),
        "acceptance_gate_evaluated": False,
        "classification": "DEVELOPMENT_ONLY_NOT_VALIDATED",
    }


def write_manifest(cells: list[dict[str, Any]], *, status: str, started_wall: float, started_cpu: float, started_utc: str, reason: str | None = None) -> dict[str, Any]:
    inventory = []
    for cell in sorted(cells, key=lambda item: item["config"]["regime"] + item["config"]["method_name"] + str(item["config"]["slope_per_decade"])):
        checkpoint = CHECKPOINTS / f"dev_{cell['config']['regime']}_n276_s{str(cell['config']['slope_per_decade']).replace('-', 'm').replace('.', 'p')}_{cell['config']['method_name'].lower()}.json"
        inventory.append({
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "status": cell["status"],
            "completed_reps": cell.get("completed_reps", 0),
            "bytes": checkpoint.stat().st_size if checkpoint.exists() else 0,
            "sha256": sha256(checkpoint) if checkpoint.exists() else None,
            "summary": summarize(cell),
        })
    process = psutil.Process()
    observe_rss()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload = {
        "schema_version": "kuntur-statistical-corrective-development-manifest-v1.3",
        "status": status,
        "development_label": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        "started_utc": started_utc,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "stop_reason": reason,
        "elapsed_wall_seconds": round(time.perf_counter() - started_wall, 3),
        "process_cpu_seconds": round(time.process_time() - started_cpu, 3),
        "peak_rss_bytes_observed": PEAK_RSS_BYTES,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_rss_bytes": MAX_RSS_BYTES,
        "processes": 1,
        "blas_threads": 1,
        "slurm_jobs_submitted": 0,
        "protocol": {"path": PROTOCOL_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(PROTOCOL_PATH)},
        "configuration": {"path": CONFIG_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(CONFIG_PATH)},
        "script": {"path": Path(__file__).resolve().relative_to(ROOT).as_posix(), "sha256": sha256(Path(__file__).resolve())},
        "v1_2_statistical_module": {"path": MODULE_PATH.relative_to(ROOT).as_posix(), "sha256": sha256(MODULE_PATH)},
        "v1_2_selection_validation_criterion_changed": False,
        "v1_2_registry_modified": False,
        "confirmation_seed_namespace_used": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
        "scenario_count": len(config["scenarios"]),
        "method_regime_combinations": sum(len(methods) for methods in config["scenarios"].values()),
        "expected_fit_count": sum(len(methods) for methods in config["scenarios"].values()) * len(config["slopes_per_decade"]) * config["outer_replicates_per_cell"],
        "completed_cells": sum(cell["status"] == "COMPLETE" for cell in cells),
        "in_progress_cells": sum(cell["status"] == "IN_PROGRESS" for cell in cells),
        "checkpoint_inventory": inventory,
        "scientific_acceptance": "NOT_EVALUATED; pilot size is intentionally inadequate for calibration or selection",
        "hindcast_rescored_or_retuned": False,
    }
    atomic_json(OUTPUT / "development_manifest.json", payload)
    return payload


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["development_seed_namespace"] != SEED_NAMESPACE:
        raise RuntimeError("Config and script development seed namespaces differ")
    if config["outer_replicates_per_cell"] != OUTER_REPS or config["bootstrap_inner_draws"] != INNER_DRAWS:
        raise RuntimeError("Frozen development pilot dimensions differ from script constants")
    if config["resource_budget"]["max_wall_seconds"] != MAX_WALL_SECONDS or config["resource_budget"]["max_rss_bytes"] != MAX_RSS_BYTES:
        raise RuntimeError("Resource caps differ from frozen config")
    if config["reserved_confirmation_seed_namespace_used"]:
        raise RuntimeError("Reserved independent-confirmation namespace must remain unused")
    verify_seed_separation()
    verify_integer_gate_boundary()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    if (OUTPUT / "development_manifest.json").exists():
        prior_manifest = json.loads((OUTPUT / "development_manifest.json").read_text(encoding="utf-8"))
        if prior_manifest.get("configuration", {}).get("sha256") != sha256(CONFIG_PATH) or prior_manifest.get("protocol", {}).get("sha256") != sha256(PROTOCOL_PATH):
            raise RuntimeError("Existing development pilot manifest has different frozen inputs")

    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    started_utc = datetime.now(timezone.utc).isoformat()
    deadline = start_wall + MAX_WALL_SECONDS
    input_hashes = {
        "protocol": sha256(PROTOCOL_PATH),
        "configuration": sha256(CONFIG_PATH),
        "v1_2_module": sha256(MODULE_PATH),
        "calibration_generator": sha256(Path(frozen_calibration.__file__).resolve()),
    }
    cells: list[dict[str, Any]] = []
    specs_by_name = config["candidate_method_specs"]
    stop_reason = None
    for regime, names in config["scenarios"].items():
        for slope in config["slopes_per_decade"]:
            for method_name in names:
                if time.perf_counter() >= deadline:
                    stop_reason = "WALL_TIME_CAP"
                    break
                if observe_rss() > MAX_RSS_BYTES:
                    stop_reason = "RSS_CAP"
                    break
                cell, complete = evaluate_cell(regime, float(slope), method_name, specs_by_name[method_name], start_time=start_wall, deadline=deadline, input_hashes=input_hashes)
                cells.append(cell)
                if not complete:
                    stop_reason = "WALL_TIME_CAP_OR_RSS_CAP"
                    break
                write_manifest(cells, status="RUNNING", started_wall=start_wall, started_cpu=start_cpu, started_utc=started_utc)
            if stop_reason:
                break
        if stop_reason:
            break
    expected_cells = sum(len(names) for names in config["scenarios"].values()) * len(config["slopes_per_decade"])
    status = "BOUNDED_PILOT_COMPLETE" if not stop_reason and len(cells) == expected_cells and all(cell["status"] == "COMPLETE" for cell in cells) else "BOUNDED_PARTIAL"
    if stop_reason is None and status != "BOUNDED_PILOT_COMPLETE":
        stop_reason = "CELL_COUNT_OR_COMPLETION_MISMATCH"
    final = write_manifest(cells, status=status, started_wall=start_wall, started_cpu=start_cpu, started_utc=started_utc, reason=stop_reason)
    summaries = [item["summary"] for item in final["checkpoint_inventory"]]
    by_status: dict[str, int] = {}
    for item in final["checkpoint_inventory"]:
        by_status[item["status"]] = by_status.get(item["status"], 0) + 1
    print(json.dumps({
        "status": final["status"],
        "development_label": final["development_label"],
        "method_regime_combinations": final["method_regime_combinations"],
        "expected_fit_count": final["expected_fit_count"],
        "completed_cells": final["completed_cells"],
        "in_progress_cells": final["in_progress_cells"],
        "checkpoint_status_counts": by_status,
        "elapsed_wall_seconds": final["elapsed_wall_seconds"],
        "process_cpu_seconds": final["process_cpu_seconds"],
        "peak_rss_bytes_observed": final["peak_rss_bytes_observed"],
        "slurm_jobs_submitted": 0,
        "stop_reason": final["stop_reason"],
        "manifest": (OUTPUT / "development_manifest.json").relative_to(ROOT).as_posix(),
        "pilot_acceptance_gates_evaluated": False,
        "complete_cell_summaries": sum(item["status"] == "COMPLETE" for item in final["checkpoint_inventory"]),
        "available_summaries": len(summaries),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
