from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import platform
import subprocess
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import scipy
import statsmodels
from scipy.stats import beta
from scipy.stats import t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox

from kuntur_data.phase2.statistical_v12 import _design, _fit_method


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results/statistical_correction"
SIGMA = 0.25
MONTH_EFFECT = 0.55 * np.sin(2 * np.pi * np.arange(12) / 12) + 0.18 * np.cos(4 * np.pi * np.arange(12) / 12)
PROCESS_ORDER = (
    "iid_gaussian", "ar1_m030", "ar1_p030", "ar1_060", "ar1_0844", "arma11",
    "seasonal_ar1_12", "heteroskedastic_iid", "student_t3", "ar1_060_mcar20",
    "ar1_060_seasonal_missing", "level_break", "random_walk",
)
EFFECT_ORDER = (0.0, 0.1, 0.25, 0.5, -0.5)
PRIMARY_PROCESSES = ("iid_gaussian", "ar1_0844", "arma11", "seasonal_ar1_12", "ar1_060_mcar20")
SECONDARY_NULL_PROCESSES = (
    "ar1_m030", "ar1_p030", "ar1_060", "heteroskedastic_iid", "student_t3",
    "ar1_060_seasonal_missing", "level_break", "random_walk",
)
LENGTH_SENSITIVITY_PROCESSES = ("ar1_060", "ar1_0844", "seasonal_ar1_12")

METHODS: tuple[dict[str, Any], ...] = (
    {"name": "OLS_HAC_LAG6", "method": "ols_hac", "lag": 6, "block": 12},
    {"name": "OLS_HAC_LAG12", "method": "ols_hac", "lag": 12, "block": 12},
    {"name": "OLS_HAC_LAG18", "method": "ols_hac", "lag": 18, "block": 12},
    {"name": "OLS_HAC_LAG24", "method": "ols_hac", "lag": 24, "block": 12},
    {"name": "OLS_HAC_CALENDAR12", "method": "ols_hac_calendar", "lag": 12, "block": 12},
    {"name": "FEGLS_AR1", "method": "gls_ar1", "lag": 12, "block": 12},
    {"name": "SARIMAX_AR1", "method": "sarimax_ar1", "lag": 12, "block": 12},
    {"name": "SARIMAX_ARMA11", "method": "arma11", "lag": 12, "block": 12},
    {"name": "SARIMAX_SEASONAL_AR12", "method": "seasonal_ar12", "lag": 12, "block": 12},
    {"name": "MOVING_BLOCK6", "method": "moving_block_bootstrap", "lag": 12, "block": 6},
    {"name": "MOVING_BLOCK12", "method": "moving_block_bootstrap", "lag": 12, "block": 12},
    {"name": "MOVING_BLOCK24", "method": "moving_block_bootstrap", "lag": 12, "block": 24},
    {"name": "SEASONAL_BLOCK_YEAR2", "method": "seasonal_block_bootstrap", "lag": 12, "block": 24},
    {"name": "SEASONAL_BLOCK_YEAR3", "method": "seasonal_block_bootstrap", "lag": 12, "block": 36},
    {"name": "SEASONAL_MK_YEAR2", "method": "seasonal_mk_block", "lag": 12, "block": 24},
    {"name": "SEASONAL_MK_YEAR3", "method": "seasonal_mk_block", "lag": 12, "block": 36},
)
CHECKPOINT_EVERY_REPS = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def clopper_pearson(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials and trials > 0")
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def scenario_seed(phase: str, process: str, slope: float, n_months: int) -> int:
    bases = {"selection": 20261126, "validation": 20261127}
    if phase not in bases or process not in PROCESS_ORDER or slope not in EFFECT_ORDER:
        raise ValueError("unknown phase, process, or effect size")
    return bases[phase] + n_months * 10000 + PROCESS_ORDER.index(process) * 100 + EFFECT_ORDER.index(slope)


def _ar1(n: int, rho: float, rng: np.random.Generator, sigma: float = SIGMA) -> np.ndarray:
    errors = np.empty(n, dtype=float)
    errors[0] = rng.normal(0.0, sigma)
    innovation_sd = sigma * np.sqrt(1.0 - rho * rho)
    innovations = rng.normal(0.0, innovation_sd, n)
    for i in range(1, n):
        errors[i] = rho * errors[i - 1] + innovations[i]
    return errors


def _arma11(n: int, rng: np.random.Generator, sigma: float = SIGMA) -> np.ndarray:
    phi, theta, burn = 0.45, 0.40, 600
    variance_factor = (1.0 + theta**2 + 2.0 * phi * theta) / (1.0 - phi**2)
    innovation = rng.normal(0.0, sigma / np.sqrt(variance_factor), n + burn)
    error = np.zeros(n + burn, dtype=float)
    for t in range(1, len(error)):
        error[t] = phi * error[t - 1] + innovation[t] + theta * innovation[t - 1]
    return error[-n:]


def make_series(process: str, n_months: int, slope_per_decade: float, rng: np.random.Generator) -> tuple[np.ndarray, pd.PeriodIndex, np.ndarray]:
    if process not in PROCESS_ORDER or n_months < 24:
        raise ValueError("unknown process or fewer than 24 calendar months")
    periods = pd.period_range("2000-01", periods=n_months, freq="M")
    t = np.arange(n_months, dtype=float)
    month_effect = np.resize(MONTH_EFFECT, n_months)
    if process == "iid_gaussian":
        error = rng.normal(0.0, SIGMA, n_months)
    elif process.startswith("ar1_"):
        rho = {"ar1_m030": -0.3, "ar1_p030": 0.3, "ar1_060": 0.6, "ar1_0844": 0.844, "ar1_060_mcar20": 0.6, "ar1_060_seasonal_missing": 0.6}[process]
        error = _ar1(n_months, rho, rng)
    elif process == "arma11":
        error = _arma11(n_months, rng)
    elif process == "seasonal_ar1_12":
        rho = 0.6
        error = np.empty(n_months, dtype=float)
        innovations = rng.normal(0.0, SIGMA * np.sqrt(1.0 - rho * rho), n_months)
        for t_idx in range(n_months):
            if t_idx < 12:
                error[t_idx] = rng.normal(0.0, SIGMA)
            else:
                error[t_idx] = rho * error[t_idx - 12] + innovations[t_idx]
    elif process == "heteroskedastic_iid":
        variance = 1.0 + 0.6 * np.sin(2 * np.pi * t / 60.0)
        error = rng.normal(0.0, SIGMA, n_months) * np.sqrt(variance)
    elif process == "student_t3":
        error = rng.standard_t(3, size=n_months) * SIGMA / np.sqrt(3.0)
    elif process == "level_break":
        error = rng.normal(0.0, SIGMA, n_months)
        error[n_months // 2:] += 0.5
    else:
        error = np.cumsum(rng.normal(0.0, SIGMA, n_months))
    values = 12.0 + month_effect + slope_per_decade * t / 120.0 + error
    observed = np.ones(n_months, dtype=bool)
    if process == "ar1_060_mcar20":
        count = int(round(0.20 * n_months))
        observed[rng.choice(n_months, size=count, replace=False)] = False
    elif process == "ar1_060_seasonal_missing":
        month = periods.month.to_numpy()
        p_missing = np.where(np.isin(month, [11, 12, 1, 2, 3, 4]), 0.35, 0.05)
        observed = rng.random(n_months) >= p_missing
    values = values.astype(float)
    values[~observed] = np.nan
    return values, periods, observed


def _normal_covariance(process: str, n: int) -> np.ndarray | None:
    i = np.arange(n)
    distance = np.abs(i[:, None] - i[None, :])
    if process in {"iid_gaussian", "student_t3"}:
        acov = np.zeros(n, dtype=float)
        acov[0] = SIGMA**2
    elif process in {"ar1_m030", "ar1_p030", "ar1_060", "ar1_0844", "ar1_060_mcar20", "ar1_060_seasonal_missing"}:
        rho = {"ar1_m030": -0.3, "ar1_p030": 0.3, "ar1_060": 0.6, "ar1_0844": 0.844, "ar1_060_mcar20": 0.6, "ar1_060_seasonal_missing": 0.6}[process]
        acov = SIGMA**2 * rho ** np.arange(n)
    elif process == "seasonal_ar1_12":
        acov = np.zeros(n, dtype=float)
        acov[np.arange(n) % 12 == 0] = SIGMA**2 * 0.6 ** (np.arange(n)[np.arange(n) % 12 == 0] // 12)
    elif process == "arma11":
        from statsmodels.tsa.arima_process import arma_acovf
        phi, theta = 0.45, 0.40
        raw = arma_acovf(ar=np.array([1.0, -phi]), ma=np.array([1.0, theta]), nobs=n)
        acov = SIGMA**2 * raw / raw[0]
    elif process == "heteroskedastic_iid":
        variance = 1.0 + 0.6 * np.sin(2 * np.pi * i / 60.0)
        return np.diag(SIGMA**2 * variance)
    else:
        return None
    return acov[distance]


def oracle_ci_width(process: str, periods: pd.PeriodIndex, observed: np.ndarray) -> float | None:
    covariance = _normal_covariance(process, len(periods))
    if covariance is None:
        return None
    x = _design(periods)
    xo = x[observed]
    if len(xo) <= xo.shape[1]:
        return None
    slope_weights = np.linalg.pinv(xo.T @ xo) @ xo.T
    weights = slope_weights[1]
    obs_cov = covariance[np.ix_(observed, observed)]
    var = float(weights @ obs_cov @ weights)
    return float(2.0 * 1.959963984540054 * np.sqrt(max(0.0, var)))


def _method_rows() -> list[dict[str, Any]]:
    return [dict(spec) for spec in METHODS]


def _residual_dependence_flag(fit: Any, periods: pd.PeriodIndex, method: str) -> bool | None:
    """Flag residual serial dependence only for fitted covariance models."""
    if method not in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"}:
        return None
    residual = getattr(fit, "residuals", None)
    if residual is None:
        return None
    values = np.asarray(residual, dtype=float)
    valid_length = len(values) == len(periods) or (method == "gls_ar1" and len(values) == len(periods) - 1)
    if not valid_length or not np.isfinite(values).all() or len(values) <= 37:
        return None
    try:
        result = acorr_ljungbox(values, lags=[12, 24], return_df=True)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return bool((result["lb_pvalue"] < 0.01).any())


def _max_missing_run(observed: np.ndarray) -> int:
    longest = current = 0
    for is_observed in observed:
        current = current + 1 if not bool(is_observed) else 0
        longest = max(longest, current)
    return longest


def _paired_mean_difference_ci(left: np.ndarray, right: np.ndarray, *, seed: int = 0) -> tuple[int, float | None, float | None, float | None]:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    differences = left[valid] - right[valid]
    if len(differences) == 0:
        return 0, None, None, None
    mean = float(differences.mean())
    if len(differences) < 2:
        return len(differences), mean, None, None
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
    margin = float(student_t.ppf(0.975, len(differences) - 1) * standard_error)
    return len(differences), mean, mean - margin, mean + margin


def _summarize_method(
    *, phase: str, process: str, n_months: int, slope: float, method_spec: dict[str, Any],
    reps_requested: int, seed: int, fits: list[Any], method_wall: float, method_cpu: float,
    rss_peak: int, oracle_widths: list[float], stress: bool, bootstrap_inner: int,
    diagnostic_flags: list[bool | None] | None = None,
) -> dict[str, Any]:
    supported = [fit for fit in fits if getattr(fit, "support_eligible", True)]
    valid = [fit for fit in supported if fit.converged and fit.slope is not None and np.isfinite(fit.slope)]
    slopes = np.asarray([fit.slope for fit in valid], dtype=float)
    inferential = [fit for fit in valid if fit.p_value is not None and np.isfinite(fit.p_value)]
    pvalues = np.asarray([fit.p_value for fit in inferential], dtype=float)
    inference_fraction = float(len(pvalues) / len(supported)) if supported else 0.0
    intervals = [fit for fit in valid if fit.ci_low is not None and fit.ci_high is not None and np.isfinite(fit.ci_low) and np.isfinite(fit.ci_high)]
    covered = sum(int(fit.ci_low <= slope <= fit.ci_high) for fit in intervals)
    rejects = sum(int(p < 0.05) for p in pvalues)
    coverage_low, coverage_high = clopper_pearson(covered, len(intervals)) if intervals else (None, None)
    rate_low, rate_high = clopper_pearson(rejects, len(pvalues)) if len(pvalues) else (None, None)
    widths = np.asarray([fit.ci_high - fit.ci_low for fit in intervals], dtype=float)
    failure_rate = float(1.0 - len(valid) / len(supported)) if supported else None
    support_fraction = float(len(supported) / reps_requested) if reps_requested else None
    mean_slope = float(slopes.mean()) if len(slopes) else None
    bias = float(mean_slope - slope) if mean_slope is not None else None
    rmse = float(np.sqrt(np.mean((slopes - slope) ** 2))) if len(slopes) else None
    type1 = rejects / len(pvalues) if slope == 0.0 and not stress and pvalues.size else None
    power = rejects / len(pvalues) if slope != 0.0 and pvalues.size else None
    cp_target = 0.05 if slope == 0.0 else 0.95
    rate_pass = bool(type1 is not None and 0.035 <= type1 <= 0.065 and rate_low <= 0.05 <= rate_high) if slope == 0.0 and not stress and pvalues.size else None
    coverage = covered / len(intervals) if intervals else None
    coverage_pass = bool(0.935 <= coverage <= 0.965 and coverage_low <= 0.95 <= coverage_high) if coverage is not None else None
    bias_pass = bool(abs(bias) <= 0.025) if bias is not None and slope != 0.0 else None
    oracle_mean = float(np.mean(oracle_widths)) if oracle_widths else None
    mean_width = float(widths.mean()) if len(widths) else None
    width_ratio = mean_width / oracle_mean if mean_width is not None and oracle_mean and oracle_mean > 0 else None
    width_pass = bool(width_ratio <= 2.0) if width_ratio is not None else None
    failure_pass = bool(failure_rate <= 0.01) if failure_rate is not None else None
    inference_pass = inference_fraction >= 0.99
    method_needs_whiteness = method_spec["method"] in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"}
    applicable_flags = [flag for fit, flag in zip(fits, diagnostic_flags or [None] * len(fits)) if getattr(fit, "support_eligible", True) and flag is not None]
    residual_dependence_rate = (sum(applicable_flags) / len(applicable_flags)) if applicable_flags else None
    residual_dependence_pass = (residual_dependence_rate <= 0.05) if method_needs_whiteness and residual_dependence_rate is not None else (None if not method_needs_whiteness else False)
    gates = [gate for gate in (rate_pass, coverage_pass, bias_pass, width_pass, failure_pass, inference_pass, residual_dependence_pass) if gate is not None]
    validation = "VALIDATED_FOR_DECLARED_REGIME" if phase == "validation" and gates and all(gates) else "SELECTION_ONLY_NOT_VALIDATED" if phase == "selection" else "NOT_VALIDATED"
    if not supported or (support_fraction is not None and support_fraction < 1.0):
        validation = "UNSUPPORTED_REGIME"
    elif failure_rate is not None and failure_rate > 0.01:
        validation = "NUMERICAL_FAILURE"
    elif stress:
        validation = "NOT_VALIDATED_STRESS_ONLY"
    warning_counts: Counter[str] = Counter()
    for fit in supported:
        for warning_message in getattr(fit, "warnings", ()):
            warning_counts[str(warning_message).split(":", 1)[0]] += 1
    return {
        "phase": phase, "process": process, "n_months": n_months, "true_slope_per_decade": slope,
        "method": method_spec["name"], "method_id": method_spec["method"],
        "replications_requested": reps_requested, "replications_valid": len(valid),
        "replications_supported": len(supported), "support_fraction": support_fraction,
        "replications_with_p_value": len(pvalues), "replications_with_interval": len(intervals),
        "inferential_output_fraction": inference_fraction,
        "seed": seed, "bootstrap_inner_draws": bootstrap_inner if "BLOCK" in method_spec["name"] else None,
        "coverage_95": coverage, "coverage_mc95_low": coverage_low, "coverage_mc95_high": coverage_high,
        "type1_fpr_005": type1, "type1_mc95_low": rate_low if slope == 0.0 else None, "type1_mc95_high": rate_high if slope == 0.0 else None,
        "power_005": power, "false_negative_rate": (1.0 - power) if power is not None else None,
        "power_mc95_low": rate_low if slope != 0.0 else None, "power_mc95_high": rate_high if slope != 0.0 else None,
        "mean_slope_per_decade": mean_slope, "slope_bias_per_decade": bias,
        "slope_mae_per_decade": float(np.mean(np.abs(slopes - slope))) if len(slopes) else None,
        "slope_rmse_per_decade": rmse, "mean_ci_width_per_decade": mean_width,
        "median_ci_width_per_decade": float(np.median(widths)) if len(widths) else None,
        "oracle_ci_width_mean": oracle_mean, "mean_ci_width_over_oracle": width_ratio,
        "convergence_failure_rate": failure_rate, "type1_gate_pass": rate_pass,
        "coverage_gate_pass": coverage_pass, "bias_gate_pass": bias_pass,
        "width_gate_pass": width_pass, "failure_gate_pass": failure_pass,
        "inference_availability_gate_pass": inference_pass,
        "residual_dependence_failure_rate": residual_dependence_rate,
        "residual_dependence_gate_pass": residual_dependence_pass,
        "optimizer_warning_counts": dict(sorted(warning_counts.items())),
        "validity_status": validation if not stress else "NOT_VALIDATED_STRESS_ONLY",
        "dgp_null_interpretation": "STATIONARY_NULL" if slope == 0.0 and not stress else "STRESS_OR_ALTERNATIVE",
        "runtime_seconds": method_wall, "process_cpu_seconds": method_cpu,
        "runtime_seconds_per_requested_series": method_wall / max(1, reps_requested),
        "peak_rss_bytes": rss_peak,
    }


def _run_scenario(phase: str, process: str, n_months: int, slope: float, reps: int, bootstrap_inner: int, output_dir: Path) -> list[dict[str, Any]]:
    stress = process in {"level_break", "random_walk"}
    seed = scenario_seed(phase, process, slope, n_months)
    checkpoint = output_dir / "checkpoints" / f"{phase}_{process}_n{n_months}_s{str(slope).replace('-', 'm').replace('.', 'p')}.json"
    protocol_hash = sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md")
    config = {
        "configuration_version": 1, "phase": phase, "process": process,
        "n_months": n_months, "slope": slope, "reps": reps,
        "bootstrap_inner": bootstrap_inner, "seed": seed,
        "protocol_sha256": protocol_hash,
        "script_sha256": sha256(Path(__file__).resolve()),
        "statistical_module_sha256": sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py"),
        "methods": [dict(spec) for spec in METHODS],
        "sigma": SIGMA,
        "checkpoint_every_reps": CHECKPOINT_EVERY_REPS,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    cached: dict[str, Any] | None = None
    if checkpoint.exists():
        cached = json.loads(checkpoint.read_text(encoding="utf-8"))
        if cached.get("config_sha256") != config_hash:
            raise RuntimeError(f"Existing scenario checkpoint has a different frozen configuration: {checkpoint.name}")
        if cached.get("status") == "COMPLETE":
            return cached["rows"]
    rng = np.random.default_rng(seed)
    fits_by_method: dict[str, list] = {spec["name"]: [] for spec in METHODS}
    oracle_by_method: dict[str, list[float]] = {spec["name"]: [] for spec in METHODS}
    diagnostics_by_method: dict[str, list[bool | None]] = {spec["name"]: [] for spec in METHODS}
    times = {spec["name"]: [0.0, 0.0] for spec in METHODS}
    process_handle = psutil.Process()
    peak_rss = int(process_handle.memory_info().rss)
    observation_fractions: list[float] = []
    maximum_missing_runs: list[int] = []
    periods_template = pd.period_range("2000-01", periods=n_months, freq="M")
    x_template = _design(periods_template)
    start_rep = 0
    if cached is not None and cached.get("status") == "IN_PROGRESS":
        progress = cached["progress"]
        start_rep = int(progress["completed_reps"])
        rng.bit_generator.state = progress["rng_state"]
        for spec in METHODS:
            name = spec["name"]
            fits_by_method[name] = [SimpleNamespace(**item) for item in progress["fits"][name]]
            oracle_by_method[name] = [float(value) for value in progress["oracle_widths"][name]]
            diagnostics_by_method[name] = [value if value is None else bool(value) for value in progress["diagnostic_flags"][name]]
            times[name] = [float(value) for value in progress["times"][name]]
        peak_rss = max(peak_rss, int(progress["peak_rss_bytes"]))
        observation_fractions = [float(value) for value in progress["observation_fractions"]]
        maximum_missing_runs = [int(value) for value in progress["maximum_missing_runs"]]

    def checkpoint_progress(completed_reps: int) -> None:
        serialized_fits: dict[str, list[dict[str, Any]]] = {}
        for name, items in fits_by_method.items():
            serialized_fits[name] = []
            for fit in items:
                row = {}
                for field in ("slope", "standard_error", "ci_low", "ci_high", "p_value"):
                    value = getattr(fit, field, None)
                    row[field] = float(value) if value is not None and np.isfinite(value) else None
                row["converged"] = bool(getattr(fit, "converged", False))
                row["support_eligible"] = bool(getattr(fit, "support_eligible", True))
                row["warnings"] = [str(value) for value in getattr(fit, "warnings", ())]
                serialized_fits[name].append(row)
        atomic_json(checkpoint, {
            "status": "IN_PROGRESS", "config": config, "config_sha256": config_hash,
            "progress": {
                "completed_reps": completed_reps,
                "fits": serialized_fits,
                "oracle_widths": oracle_by_method,
                "diagnostic_flags": diagnostics_by_method,
                "times": times,
                "peak_rss_bytes": peak_rss,
                "observation_fractions": observation_fractions,
                "maximum_missing_runs": maximum_missing_runs,
                "rng_state": rng.bit_generator.state,
            },
        })

    for rep in range(start_rep, reps):
        y, periods, observed = make_series(process, n_months, slope, rng)
        if not periods.equals(periods_template):
            raise RuntimeError("DGP calendar index changed unexpectedly")
        observation_fractions.append(float(observed.mean()))
        maximum_missing_runs.append(_max_missing_run(observed))
        oracle_width = oracle_ci_width(process, periods, observed)
        for method_index, spec in enumerate(METHODS):
            needs_complete = spec["method"] in {"gls_ar1", "moving_block_bootstrap", "seasonal_block_bootstrap"} or (spec["method"] == "ols_hac" and not observed.all())
            if needs_complete and not observed.all():
                fits_by_method[spec["name"]].append(_FitPlaceholder(False, support_eligible=False))
                diagnostics_by_method[spec["name"]].append(None)
                continue
            before_wall, before_cpu = time.perf_counter(), time.process_time()
            fit = _fit_method(
                spec["method"], y, x_template, periods,
                hac_lag_months=int(spec["lag"]), n_bootstrap=bootstrap_inner,
                random_seed=seed + rep * 1009 + method_index, block_length=int(spec["block"]),
            )
            times[spec["name"]][0] += time.perf_counter() - before_wall
            times[spec["name"]][1] += time.process_time() - before_cpu
            fits_by_method[spec["name"]].append(fit)
            diagnostics_by_method[spec["name"]].append(_residual_dependence_flag(fit, periods, spec["method"]))
            if oracle_width is not None:
                oracle_by_method[spec["name"]].append(oracle_width)
        peak_rss = max(peak_rss, int(process_handle.memory_info().rss))
        completed = rep + 1
        if completed % CHECKPOINT_EVERY_REPS == 0 and completed < reps:
            checkpoint_progress(completed)
            print(f"SCENARIO_PROGRESS phase={phase} process={process} n={n_months} slope={slope:g} reps={completed}/{reps}", flush=True)
    rows = []
    for spec in METHODS:
        name = spec["name"]
        fits = fits_by_method[name]
        # Invalid-support sentinels count as method failures for the scenario.
        rows.append(_summarize_method(
            phase=phase, process=process, n_months=n_months, slope=slope,
            method_spec=spec, reps_requested=reps, seed=seed, fits=fits,
            method_wall=times[name][0], method_cpu=times[name][1], rss_peak=peak_rss,
            oracle_widths=oracle_by_method[name], stress=stress, bootstrap_inner=bootstrap_inner,
            diagnostic_flags=diagnostics_by_method[name],
        ))
        rows[-1]["mean_observation_fraction"] = float(np.mean(observation_fractions))
        rows[-1]["min_observation_fraction"] = float(np.min(observation_fractions))
        rows[-1]["max_observation_fraction"] = float(np.max(observation_fractions))
        rows[-1]["mean_max_calendar_missing_run_months"] = float(np.mean(maximum_missing_runs))
        rows[-1]["max_calendar_missing_run_months"] = int(np.max(maximum_missing_runs))
    replicate_records: list[dict[str, Any]] = []
    for method_index, spec in enumerate(METHODS):
        name = spec["name"]
        for rep, fit in enumerate(fits_by_method[name]):
            slope_value = getattr(fit, "slope", None)
            p_value = getattr(fit, "p_value", None)
            ci_low = getattr(fit, "ci_low", None)
            ci_high = getattr(fit, "ci_high", None)
            slope_value = float(slope_value) if slope_value is not None and np.isfinite(slope_value) else None
            p_value = float(p_value) if p_value is not None and np.isfinite(p_value) else None
            ci_low = float(ci_low) if ci_low is not None and np.isfinite(ci_low) else None
            ci_high = float(ci_high) if ci_high is not None and np.isfinite(ci_high) else None
            replicate_records.append({
                "phase": phase, "process": process, "n_months": n_months,
                "true_slope_per_decade": slope, "replication": rep,
                "seed": seed, "method": name, "method_id": spec["method"],
                "slope_per_decade": slope_value,
                "slope_error_per_decade": slope_value - slope if slope_value is not None else None,
                "absolute_slope_error_per_decade": abs(slope_value - slope) if slope_value is not None else None,
                "p_value": p_value, "reject_at_005": bool(p_value < 0.05) if p_value is not None else None,
                "ci_low": ci_low, "ci_high": ci_high,
                "coverage_95": bool(ci_low <= slope <= ci_high) if ci_low is not None and ci_high is not None else None,
                "ci_width": ci_high - ci_low if ci_low is not None and ci_high is not None else None,
                "converged": bool(getattr(fit, "converged", False)),
                "support_eligible": bool(getattr(fit, "support_eligible", True)),
                "residual_dependence_misspecified": diagnostics_by_method[name][rep],
                "warning_count": len(getattr(fit, "warnings", ())),
            })
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    replicate_path = output_dir / "replicates" / f"{checkpoint.stem}.parquet"
    replicate_path.parent.mkdir(parents=True, exist_ok=True)
    replicate_tmp = replicate_path.with_suffix(".parquet.tmp")
    pd.DataFrame(replicate_records).to_parquet(replicate_tmp, index=False)
    replicate_tmp.replace(replicate_path)
    replicate_digest = sha256(replicate_path)
    atomic_json(checkpoint, {"status": "COMPLETE", "config": config, "config_sha256": config_hash, "rows": rows, "replicate_records": replicate_records, "replicate_metrics": {"path": replicate_path.relative_to(output_dir).as_posix(), "bytes": replicate_path.stat().st_size, "sha256": replicate_digest}})
    return rows


class _FitPlaceholder:
    def __init__(self, converged: bool, *, support_eligible: bool = True):
        self.converged = converged
        self.support_eligible = support_eligible
        self.slope = None
        self.standard_error = None
        self.ci_low = None
        self.ci_high = None
        self.p_value = None


def _scenario_plan(phase: str) -> list[tuple[str, int, float, int, int]]:
    if phase == "selection":
        return [(process, 276, slope, 500, 199) for process in PRIMARY_PROCESSES for slope in (0.0, 0.25)]
    cases = [(process, 276, 0.0, 1000, 399) for process in PRIMARY_PROCESSES + SECONDARY_NULL_PROCESSES]
    for process in PRIMARY_PROCESSES:
        for slope in (0.1, 0.25, 0.5, -0.5):
            cases.append((process, 276, slope, 1000, 399))
    for process in LENGTH_SENSITIVITY_PROCESSES:
        for n_months in (60, 120, 180):
            for slope in (0.0, 0.25, 0.5):
                cases.append((process, n_months, slope, 1000, 399))
    return cases


def _scenario_checkpoint_path(output_dir: Path, phase: str, process: str, n_months: int, slope: float) -> Path:
    slope_tag = str(slope).replace("-", "m").replace(".", "p")
    return output_dir / "checkpoints" / f"{phase}_{process}_n{n_months}_s{slope_tag}.json"


def _write_selection_decisions(selection_dir: Path) -> dict[str, Any]:
    """Freeze deterministic per-DGP selection using only selection-seed rows."""
    complete_rows: list[dict[str, Any]] = []
    missing_scenarios = []
    for process, n_months, slope, _reps, _inner in _scenario_plan("selection"):
        checkpoint = _scenario_checkpoint_path(selection_dir, "selection", process, n_months, slope)
        if not checkpoint.exists():
            missing_scenarios.append({"process": process, "n_months": n_months, "slope": slope, "reason": "NO_CHECKPOINT"})
            continue
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            missing_scenarios.append({"process": process, "n_months": n_months, "slope": slope, "reason": "INCOMPLETE_CHECKPOINT"})
            continue
        complete_rows.extend(payload["rows"])
    if missing_scenarios:
        raise RuntimeError(f"Selection is incomplete; refusing to freeze decisions: {len(missing_scenarios)} scenarios missing.")

    lookup = {(row["process"], float(row["true_slope_per_decade"]), row["method"]): row for row in complete_rows}
    decisions: dict[str, Any] = {}
    for process in PRIMARY_PROCESSES:
        eligible: list[dict[str, Any]] = []
        for spec in METHODS:
            null = lookup[(process, 0.0, spec["name"])]
            alternative = lookup[(process, 0.25, spec["name"])]
            required = [null["type1_gate_pass"], null["failure_gate_pass"], null["inference_availability_gate_pass"], alternative["failure_gate_pass"], alternative["inference_availability_gate_pass"]]
            has_ci = null["replications_with_interval"] > 0 or alternative["replications_with_interval"] > 0
            if has_ci:
                required.extend([null["coverage_gate_pass"], alternative["coverage_gate_pass"], null["width_gate_pass"], alternative["width_gate_pass"]])
            if spec["method"] in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"}:
                required.extend([null["residual_dependence_gate_pass"], alternative["residual_dependence_gate_pass"]])
            required.extend([null["support_fraction"] == 1.0, alternative["support_fraction"] == 1.0])
            if not all(value is True for value in required):
                continue
            score = abs(float(null["type1_fpr_005"]) - 0.05) / 0.05
            coverage_values = []
            if has_ci:
                coverage_values.extend([float(null["coverage_95"]), float(alternative["coverage_95"])])
                score += sum(abs(value - 0.95) / 0.05 for value in coverage_values)
            score += abs(float(alternative["slope_bias_per_decade"])) / 0.025
            ratios = [row["mean_ci_width_over_oracle"] for row in (null, alternative) if row["mean_ci_width_over_oracle"] is not None]
            eligible.append({
                "method": spec["name"], "method_id": spec["method"],
                "calibration_score": score,
                "mean_ci_width_over_oracle": float(np.mean(ratios)) if ratios else None,
                "null_type1": null["type1_fpr_005"],
                "null_coverage": null["coverage_95"],
                "alternative_coverage": alternative["coverage_95"],
                "alternative_bias_per_decade": alternative["slope_bias_per_decade"],
            })
        eligible.sort(key=lambda item: (
            item["calibration_score"],
            item["mean_ci_width_over_oracle"] if item["mean_ci_width_over_oracle"] is not None else float("inf"),
            item["method"],
        ))
        decisions[process] = {
            "selection_status": "PROVISIONAL_SELECTED_FOR_INDEPENDENT_VALIDATION" if eligible else "NO_SELECTION_CANDIDATE_PASSED_SELECTION_GATES",
            "selected": eligible[0] if eligible else None,
            "passing_candidate_count": len(eligible),
            "eligible_candidates_ranked": eligible,
        }
    result = {
        "selection_seed_base": 20261126,
        "validation_seed_base": 20261127,
        "protocol_sha256": sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
        "selection_matrix_sha256": sha256(selection_dir / "method_validation_matrix.csv"),
        "status": "FROZEN_BEFORE_INDEPENDENT_VALIDATION",
        "selection_rule": "Per primary DGP, require all predeclared selection gates; rank by standardized nominal calibration error, then mean CI/oracle width, then method ID.",
        "decisions": decisions,
    }
    atomic_json(selection_dir / "selection_decisions.json", result)
    atomic_json(selection_dir / "selection_manifest.json", {
        "status": "FROZEN_BEFORE_INDEPENDENT_VALIDATION",
        "selection_decisions_sha256": sha256(selection_dir / "selection_decisions.json"),
        "selection_matrix_sha256": result["selection_matrix_sha256"],
        "protocol_sha256": result["protocol_sha256"],
        "selection_seed_base": 20261126,
        "validation_seed_base": 20261127,
    })
    return result


def _verify_frozen_selection(selection_dir: Path) -> dict[str, Any]:
    decisions_path = selection_dir / "selection_decisions.json"
    manifest_path = selection_dir / "selection_manifest.json"
    if not decisions_path.exists() or not manifest_path.exists():
        raise RuntimeError("Independent validation requires a frozen selection decision and manifest.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256(decisions_path) != manifest.get("selection_decisions_sha256"):
        raise RuntimeError("Frozen selection decision SHA-256 does not match its manifest.")
    matrix_path = selection_dir / "method_validation_matrix.csv"
    if not matrix_path.exists() or sha256(matrix_path) != manifest.get("selection_matrix_sha256"):
        raise RuntimeError("Frozen selection matrix SHA-256 does not match its manifest.")
    decision = json.loads(decisions_path.read_text(encoding="utf-8"))
    if decision.get("protocol_sha256") != sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"):
        raise RuntimeError("Frozen selection was created under a different protocol hash.")
    return decision


def _write_phase_results(
    output_dir: Path, phase: str, *, expected_cases: list[tuple[str, int, float, int, int]],
    smoke: bool = False,
) -> dict[str, Any]:
    rows = []
    replicate_frames: list[pd.DataFrame] = []
    complete_paths: set[str] = set()
    for path in sorted((output_dir / "checkpoints").glob(f"{phase}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") == "COMPLETE":
            rows.extend(payload["rows"])
            replicate_records = payload.get("replicate_records", [])
            if replicate_records:
                replicate_frames.append(pd.DataFrame(replicate_records))
            complete_paths.add(path.name)
    if not rows:
        raise RuntimeError("No scenario checkpoints were completed")
    frame = pd.DataFrame(rows).sort_values(["process", "n_months", "true_slope_per_decade", "method"]).reset_index(drop=True)
    csv_path = output_dir / "method_validation_matrix.csv"
    parquet_path = output_dir / "monte_carlo_results.parquet"
    csv_tmp, parquet_tmp = csv_path.with_suffix(".csv.tmp"), parquet_path.with_suffix(".parquet.tmp")
    frame.to_csv(csv_tmp, index=False)
    frame.to_parquet(parquet_tmp, index=False)
    csv_tmp.replace(csv_path)
    parquet_tmp.replace(parquet_path)
    replicate_path = output_dir / "replicate_metrics.parquet"
    pairwise_path = output_dir / "paired_method_differences.csv"
    output_paths = [csv_path, parquet_path]
    if replicate_frames:
        replicate_frame = pd.concat(replicate_frames, ignore_index=True).sort_values(
            ["process", "n_months", "true_slope_per_decade", "replication", "method"]
        ).reset_index(drop=True)
        replicate_tmp = replicate_path.with_suffix(".parquet.tmp")
        replicate_frame.to_parquet(replicate_tmp, index=False)
        replicate_tmp.replace(replicate_path)
        metrics = ("slope_error_per_decade", "absolute_slope_error_per_decade", "reject_at_005", "coverage_95", "ci_width")
        paired_rows: list[dict[str, Any]] = []
        scenario_columns = ["process", "n_months", "true_slope_per_decade", "seed"]
        for scenario_key, group in replicate_frame.groupby(scenario_columns, sort=True):
            scenario = dict(zip(scenario_columns, scenario_key))
            methods = sorted(group.method.unique())
            by_method = {name: group.loc[group.method == name].set_index("replication") for name in methods}
            for left_index, left_name in enumerate(methods):
                for right_name in methods[left_index + 1:]:
                    joined = by_method[left_name].add_suffix("_left").join(by_method[right_name].add_suffix("_right"), how="inner")
                    supported = joined["support_eligible_left"].astype(bool) & joined["support_eligible_right"].astype(bool)
                    converged = joined["converged_left"].astype(bool) & joined["converged_right"].astype(bool)
                    for metric in metrics:
                        left_col, right_col = f"{metric}_left", f"{metric}_right"
                        if left_col not in joined or right_col not in joined:
                            continue
                        a = pd.to_numeric(joined.loc[supported & converged, left_col], errors="coerce").to_numpy(dtype=float)
                        b = pd.to_numeric(joined.loc[supported & converged, right_col], errors="coerce").to_numpy(dtype=float)
                        n_pair, difference, low, high = _paired_mean_difference_ci(a, b)
                        if n_pair < 2:
                            continue
                        paired_rows.append({
                            **scenario, "method_left": left_name, "method_right": right_name,
                            "metric": metric, "mean_paired_difference_left_minus_right": difference,
                            "paired_t_ci95_low": low, "paired_t_ci95_high": high,
                            "paired_replications": n_pair,
                            "interval_method": "paired Student t, df=n-1; unadjusted across comparisons",
                        })
        pd.DataFrame(paired_rows).to_csv(pairwise_path, index=False)
        output_paths.extend([replicate_path, pairwise_path])
    expected_paths = {
        _scenario_checkpoint_path(output_dir, phase, process, n_months, slope).name
        for process, n_months, slope, _reps, _inner in expected_cases
    }
    missing_scenarios = sorted(expected_paths - complete_paths)
    complete = not missing_scenarios and not smoke
    manifest = {
        "phase": phase,
        "status": "SMOKE_ONLY_NOT_VALIDATION" if smoke else "COMPLETED_PHASE" if complete else "PARTIAL_PHASE_RUN",
        "execution_scope": "SMOKE" if smoke else "FULL_PHASE" if complete else "SUBSET_OR_INCOMPLETE",
        "protocol_sha256": sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "statistical_module_sha256": sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py"),
        "source_commit": subprocess.check_output(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "resources": {"logical_cpus": psutil.cpu_count(logical=True), "physical_cpus": psutil.cpu_count(logical=False)},
        "scenario_count": int(frame[["process", "n_months", "true_slope_per_decade"]].drop_duplicates().shape[0]),
        "planned_scenario_count": len(expected_cases),
        "missing_scenarios": missing_scenarios,
        "rows": int(len(frame)),
        "completed_checkpoints": len(complete_paths),
        "replicate_metrics_rows": int(sum(len(frame) for frame in replicate_frames)),
        "paired_comparison_rows": int(len(paired_rows)) if replicate_frames else 0,
        "outputs": {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_paths},
        "limitations": ["Validation is regime-specific; any failed or unrun regime stays NOT_VALIDATED.", "Structural-break and random-walk rows are stress diagnostics, not stationary null calibration."],
    }
    atomic_json(output_dir / "calibration_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("selection", "validation"), required=True)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny code-path smoke check; never use for method selection or validation.")
    parser.add_argument("--only-process", choices=PROCESS_ORDER)
    parser.add_argument("--limit-scenarios", type=int)
    args = parser.parse_args()
    if args.limit_scenarios is not None and args.limit_scenarios < 1:
        parser.error("--limit-scenarios must be positive")
    if args.smoke:
        smoke_identity = (
            sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md")
            + sha256(Path(__file__).resolve())
            + sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py")
        )
        smoke_tag = hashlib.sha256(smoke_identity.encode()).hexdigest()[:8]
        output_dir = OUT_ROOT / f"smoke_{smoke_tag}"
        full_plan = [("iid_gaussian", 60, 0.0, 8, 19)]
    else:
        output_dir = OUT_ROOT / args.phase
        full_plan = _scenario_plan(args.phase)
    plan = list(full_plan)
    if not args.smoke and args.phase == "validation":
        _verify_frozen_selection(OUT_ROOT / "selection")
    if args.only_process:
        plan = [case for case in plan if case[0] == args.only_process]
    if args.limit_scenarios:
        plan = plan[:args.limit_scenarios]
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, (process, n_months, slope, reps, inner) in enumerate(plan, start=1):
        scenario_rows = _run_scenario(args.phase, process, n_months, slope, reps, inner, output_dir)
        print(f"SCENARIO_COMPLETE phase={args.phase} process={process} n={n_months} slope={slope:g} methods={len(scenario_rows)} {index}/{len(plan)}", flush=True)
    manifest = _write_phase_results(output_dir, args.phase, expected_cases=full_plan, smoke=args.smoke)
    if args.phase == "selection" and not args.smoke and manifest["status"] == "COMPLETED_PHASE":
        _write_selection_decisions(output_dir)
        manifest["selection_decisions_sha256"] = sha256(output_dir / "selection_decisions.json")
        atomic_json(output_dir / "calibration_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
