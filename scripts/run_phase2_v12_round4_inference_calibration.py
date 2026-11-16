#!/usr/bin/env python3
"""Reproducible synthetic calibration of monthly trend inference methods.

No KUNTUR observations or sealed evaluation targets are opened by this script.
The previous v1.1 calibration artifacts are retained as separate historical
evidence; this run adds seeded, fully reproducible method comparisons.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import psutil
import scipy
import statsmodels
from scipy.stats import beta, norm, t as student_t, theilslopes
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/optimization/round_04_inference_calibration"
N = 276
REPS = 1000
BOOTSTRAP_REPS = 500
BOOTSTRAP_DRAWS = 299
ARMA_REPS = 200
SEED = 20261123
LAGS = (6, 12, 18, 24)
BOOTSTRAP_PROCESSES = {"iid_gaussian", "ar1_060", "ar1_0844", "seasonal_ar1_12"}
SEN_CI_PROCESSES = {"iid_gaussian", "ar1_060", "ar1_0844", "seasonal_ar1_12"}

MONTH = np.arange(N) % 12
YEAR_DEC = np.arange(N, dtype=float) / 120.0
DUMMIES = np.eye(12, dtype=float)[MONTH, 1:]
X = np.column_stack([np.ones(N), YEAR_DEC, DUMMIES])
P = X.shape[1]
XT_X_INV = np.linalg.inv(X.T @ X)
OLS_OPERATOR = XT_X_INV @ X.T
SLOPE_WEIGHTS = OLS_OPERATOR[1]
Z975 = float(norm.ppf(0.975))
T975 = float(student_t.ppf(0.975, N - P))
MONTH_EFFECT = 0.55 * np.sin(2.0 * np.pi * MONTH / 12.0) + 0.18 * np.cos(4.0 * np.pi * MONTH / 12.0)
BASELINE = 12.0 + MONTH_EFFECT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _make_errors(name: str, rng: np.random.Generator) -> np.ndarray:
    sigma = 0.25
    if name == "iid_gaussian":
        return rng.normal(0.0, sigma, N)
    if name.startswith("ar1_"):
        rho = {"ar1_m030": -0.3, "ar1_p030": 0.3, "ar1_060": 0.6, "ar1_0844": 0.844}[name]
        innov = rng.normal(0.0, sigma * np.sqrt(1.0 - rho * rho), N)
        e = np.empty(N, dtype=float)
        e[0] = rng.normal(0.0, sigma)
        for i in range(1, N):
            e[i] = rho * e[i - 1] + innov[i]
        return e
    if name == "arma11":
        phi, theta = 0.45, 0.40
        variance_factor = (1.0 + theta**2 + 2.0 * phi * theta) / (1.0 - phi**2)
        eps = rng.normal(0.0, sigma / np.sqrt(variance_factor), N + 600)
        e = np.zeros(N + 600, dtype=float)
        for i in range(1, len(e)):
            e[i] = phi * e[i - 1] + eps[i] + theta * eps[i - 1]
        return e[-N:]
    if name == "seasonal_ar1_12":
        rho = 0.60
        e = np.empty(N, dtype=float)
        e[:12] = rng.normal(0.0, sigma, 12)
        innov = rng.normal(0.0, sigma * np.sqrt(1.0 - rho * rho), N)
        for i in range(12, N):
            e[i] = rho * e[i - 12] + innov[i]
        return e
    if name == "heteroskedastic_iid":
        phase = 1.0 + 0.65 * np.sin(2.0 * np.pi * np.arange(N) / 60.0)
        return rng.normal(0.0, sigma * np.sqrt(phase), N)
    if name == "student_t3":
        return rng.standard_t(df=3, size=N) * sigma / np.sqrt(3.0)
    raise ValueError(f"Unknown generating process: {name}")


def _fit_ols(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beta_hat = OLS_OPERATOR @ y
    residual = y - X @ beta_hat
    return beta_hat, residual


def _ols_hac(y: np.ndarray, maxlags: int) -> tuple[float, float, float, float, float]:
    beta_hat, residual = _fit_ols(y)
    score = X * residual[:, None]
    meat = score.T @ score
    for lag in range(1, maxlags + 1):
        weight = 1.0 - lag / (maxlags + 1.0)
        cross = score[lag:].T @ score[:-lag]
        meat += weight * (cross + cross.T)
    covariance = (N / (N - P)) * (XT_X_INV @ meat @ XT_X_INV)
    se = float(np.sqrt(max(0.0, covariance[1, 1])))
    slope = float(beta_hat[1])
    low, high = slope - Z975 * se, slope + Z975 * se
    p = float(2.0 * norm.sf(abs(slope / se))) if se > 0 else 1.0
    return slope, float(low), float(high), p, se


def _fgls_ar1(y: np.ndarray, iterations: int) -> tuple[float, float, float, float, float]:
    beta_hat, residual = _fit_ols(y)
    rho = 0.0
    xtxi = XT_X_INV
    transformed_y = y
    transformed_x = X
    for _ in range(iterations):
        denominator = float(residual[:-1] @ residual[:-1])
        rho = float(np.clip((residual[:-1] @ residual[1:]) / denominator if denominator else 0.0, -0.98, 0.98))
        first_scale = np.sqrt(max(1e-5, 1.0 - rho * rho))
        transformed_x = np.empty_like(X)
        transformed_y = np.empty_like(y)
        transformed_x[0] = first_scale * X[0]
        transformed_y[0] = first_scale * y[0]
        transformed_x[1:] = X[1:] - rho * X[:-1]
        transformed_y[1:] = y[1:] - rho * y[:-1]
        xtxi = np.linalg.inv(transformed_x.T @ transformed_x)
        beta_hat = xtxi @ transformed_x.T @ transformed_y
        residual = y - X @ beta_hat
    transformed_residual = transformed_y - transformed_x @ beta_hat
    dof = N - P
    sigma2 = float((transformed_residual @ transformed_residual) / dof)
    se = float(np.sqrt(max(0.0, sigma2 * xtxi[1, 1])))
    slope = float(beta_hat[1])
    low, high = slope - T975 * se, slope + T975 * se
    p = float(2.0 * student_t.sf(abs(slope / se), dof)) if se > 0 else 1.0
    return slope, float(low), float(high), p, se


def _seasonal_theil_sen(y: np.ndarray) -> float:
    """Median within-calendar-month pairwise slope, in units per decade."""
    annual_x = np.arange(N // 12, dtype=float) / 10.0
    slopes = []
    i, j = np.triu_indices(len(annual_x), k=1)
    denominator = annual_x[j] - annual_x[i]
    for month in range(12):
        values = y[month::12]
        slopes.append(float(np.median((values[j] - values[i]) / denominator)))
    return float(np.median(slopes))


def _theil_sen_iid_ci(y: np.ndarray) -> tuple[float, float, float, float]:
    # Synthetic seasonal means are known by design; the Sen interval is the
    # classic independence-based interval on seasonally centered observations.
    anomaly = y - MONTH_EFFECT
    fit = theilslopes(anomaly, YEAR_DEC, alpha=0.95)
    return float(fit.slope), float(fit.low_slope), float(fit.high_slope), float(fit.high_slope - fit.low_slope)


def _block_ci(y: np.ndarray, rng: np.random.Generator, *, seasonal: bool, draws: int) -> tuple[float, float, float]:
    beta_hat, residual = _fit_ols(y)
    n_blocks = N // 12
    if seasonal:
        year_residuals = residual.reshape(n_blocks, 12)
        sampled_years = rng.integers(0, n_blocks, size=(draws, n_blocks))
        boot_resid = year_residuals[sampled_years].reshape(draws, N)
    else:
        n_sampled_blocks = int(np.ceil(N / 12))
        starts = rng.integers(0, N, size=(draws, n_sampled_blocks))
        offsets = np.arange(12)
        indexes = (starts[:, :, None] + offsets[None, None, :]) % N
        indexes = indexes.reshape(draws, -1)[:, :N]
        boot_resid = residual[indexes]
    boot_slopes = beta_hat[1] + boot_resid @ SLOPE_WEIGHTS
    low, high = np.quantile(boot_slopes, [0.025, 0.975])
    return float(low), float(high), float(high - low)


def _cp_interval(successes: int, trials: int) -> tuple[float, float]:
    low = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return low, high


def _summary(process: str, true_slope: float, method: str, slopes: list[float], lows: list[float], highs: list[float], pvalues: list[float], replications: int, seed: int, *, coverage_enabled: bool = True, note: str = "") -> dict:
    slope_values = np.asarray(slopes, dtype=float)
    low_values, high_values = np.asarray(lows, dtype=float), np.asarray(highs, dtype=float)
    p_values = np.asarray(pvalues, dtype=float)
    valid = np.isfinite(slope_values) & np.isfinite(low_values) & np.isfinite(high_values) & np.isfinite(p_values)
    slope_values, low_values, high_values, p_values = slope_values[valid], low_values[valid], high_values[valid], p_values[valid]
    n_valid = len(slope_values)
    if n_valid == 0:
        raise RuntimeError(f"No valid simulations for {process} {method}")
    inside = (low_values <= true_slope) & (true_slope <= high_values)
    rejects = p_values < 0.05
    coverage = int(inside.sum()) if coverage_enabled else None
    rejection_count = int(rejects.sum())
    coverage_ci = _cp_interval(coverage, n_valid) if coverage is not None else (None, None)
    rejection_ci = _cp_interval(rejection_count, n_valid) if true_slope == 0 else (None, None)
    power_ci = _cp_interval(rejection_count, n_valid) if true_slope != 0 else (None, None)
    return {
        "process": process, "n_months": N, "true_slope_per_decade": true_slope,
        "method": method, "replications_requested": replications, "replications_valid": n_valid,
        "seed": seed, "coverage_95": (coverage / n_valid) if coverage is not None else None,
        "coverage_mc95_low": coverage_ci[0], "coverage_mc95_high": coverage_ci[1],
        "type1_fpr_005": (rejection_count / n_valid) if true_slope == 0 else None,
        "type1_mc95_low": rejection_ci[0], "type1_mc95_high": rejection_ci[1],
        "power_005": (rejection_count / n_valid) if true_slope != 0 else None,
        "power_mc95_low": power_ci[0], "power_mc95_high": power_ci[1],
        "mean_slope_per_decade": float(np.mean(slope_values)),
        "slope_bias_per_decade": float(np.mean(slope_values) - true_slope),
        "slope_rmse_per_decade": float(np.sqrt(np.mean((slope_values - true_slope) ** 2))),
        "mean_ci_width_per_decade": float(np.mean(high_values - low_values)),
        "note": note,
    }


def _run_case(process: str, true_slope: float, reps: int, seed: int, *, bootstrap_reps: int = 0, bootstrap_draws: int = BOOTSTRAP_DRAWS) -> list[dict]:
    rng = np.random.default_rng(seed)
    methods: dict[str, dict[str, list]] = {}

    def add(method: str, slope: float, low: float, high: float, pvalue: float) -> None:
        item = methods.setdefault(method, {"slopes": [], "lows": [], "highs": [], "pvalues": []})
        item["slopes"].append(slope)
        item["lows"].append(low)
        item["highs"].append(high)
        item["pvalues"].append(pvalue)

    arma_fits: list[tuple[float, float, float, float]] = []
    for i in range(reps):
        noise = _make_errors(process, rng)
        y = BASELINE + true_slope * YEAR_DEC + noise
        selected_lags = LAGS if process == "ar1_0844" else (12,)
        for lag in selected_lags:
            add(f"OLS_HAC_lag{lag}", *_ols_hac(y, lag)[:4])
        add("FEGLS_AR1_3iter", *_fgls_ar1(y, iterations=3)[:4])
        add("PREWHITENED_AR1_1pass", *_fgls_ar1(y, iterations=1)[:4])
        sen_slope = _seasonal_theil_sen(y)
        # The seasonal Sen implementation is a point-estimator diagnostic; its
        # inferential interval is evaluated separately with iid Sen limits.
        add("SEASONAL_THEIL_SEN_point_only", sen_slope, np.nan, np.nan, np.nan)
        if process in SEN_CI_PROCESSES and i < reps:
            sen, lo, hi, _width = _theil_sen_iid_ci(y)
            # The binomial rejection summary mirrors the interval decision:
            # reject a zero trend only when the 95% interval excludes zero.
            add("THEIL_SEN_iid_interval", sen, lo, hi, 1.0 if lo <= 0.0 <= hi else 0.0)
        if bootstrap_reps and i < bootstrap_reps:
            boot_rng = np.random.default_rng(seed + 100_000 + i)
            for seasonal, label in ((False, "MOVING_BLOCK12"), (True, "SEASONAL_YEAR_BLOCK12")):
                lo, hi, _width = _block_ci(y, boot_rng, seasonal=seasonal, draws=bootstrap_draws)
                add(label, float(OLS_OPERATOR[1] @ y), lo, hi, 0.01 if (lo > 0.0 or hi < 0.0) else 0.5)
        if process == "arma11" and i < ARMA_REPS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                try:
                    fit = sm.tsa.SARIMAX(
                        y, exog=X, order=(1, 0, 1), trend="n",
                        enforce_stationarity=True, enforce_invertibility=True,
                    ).fit(disp=False, maxiter=80)
                    params = np.asarray(fit.params, dtype=float)
                    ses = np.asarray(fit.bse, dtype=float)
                    slope = float(params[1])
                    se = float(ses[1])
                    lo, hi = slope - Z975 * se, slope + Z975 * se
                    p = float(2.0 * norm.sf(abs(slope / se))) if se > 0 else 1.0
                    arma_fits.append((slope, lo, hi, p))
                except Exception:
                    # Preserve progress and count failures without logging any
                    # exception detail that could contain a data path.
                    continue

    note = "Seeded Monte Carlo; fixed 12-month seasonal effects and known linear trend; synthetic data only."
    rows = []
    for method, metrics in methods.items():
        if method == "SEASONAL_THEIL_SEN_point_only":
            s = np.asarray(metrics["slopes"], dtype=float)
            rows.append({
                "process": process, "n_months": N, "true_slope_per_decade": true_slope,
                "method": method, "replications_requested": reps, "replications_valid": len(s), "seed": seed,
                "coverage_95": None, "coverage_mc95_low": None, "coverage_mc95_high": None,
                "type1_fpr_005": None, "type1_mc95_low": None, "type1_mc95_high": None,
                "power_005": None, "power_mc95_low": None, "power_mc95_high": None,
                "mean_slope_per_decade": float(s.mean()), "slope_bias_per_decade": float(s.mean() - true_slope),
                "slope_rmse_per_decade": float(np.sqrt(np.mean((s - true_slope) ** 2)),),
                "mean_ci_width_per_decade": None, "note": "Robust point estimator only; no calibrated interval is implied.",
            })
            continue
        requested = reps if method != "ARMA11_MLE" else min(reps, ARMA_REPS)
        rows.append(_summary(process, true_slope, method, metrics["slopes"], metrics["lows"], metrics["highs"], metrics["pvalues"], requested, seed, note=note))
    if process == "arma11":
        arma_metric = {"slopes": [r[0] for r in arma_fits], "lows": [r[1] for r in arma_fits], "highs": [r[2] for r in arma_fits], "pvalues": [r[3] for r in arma_fits]}
        arma_row = _summary(process, true_slope, "ARMA11_MLE", arma_metric["slopes"], arma_metric["lows"], arma_metric["highs"], arma_metric["pvalues"], ARMA_REPS, seed, note="SARIMAX ARMA(1,1) error model; only this process family, bounded to 200 fits per effect due MLE cost.")
        arma_row["convergence_failures"] = ARMA_REPS - len(arma_fits)
        rows.append(arma_row)
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    proc = psutil.Process()
    cpu_start = proc.cpu_times()
    processes = ["iid_gaussian", "ar1_m030", "ar1_p030", "ar1_060", "ar1_0844", "arma11", "seasonal_ar1_12", "heteroskedastic_iid", "student_t3"]
    rows: list[dict] = []
    for process_index, process in enumerate(processes):
        for effect_index, true_slope in enumerate((0.0, 0.5)):
            seed = SEED + process_index * 100 + effect_index
            do_bootstrap = BOOTSTRAP_REPS if process in BOOTSTRAP_PROCESSES else 0
            case_rows = _run_case(process, true_slope, REPS, seed, bootstrap_reps=do_bootstrap)
            rows.extend(case_rows)
            partial = pd.DataFrame(rows)
            temp_csv = OUT / "inference_calibration.csv.tmp"
            partial.to_csv(temp_csv, index=False)
            temp_csv.replace(OUT / "inference_calibration.csv")
            print(f"CASE_COMPLETE process={process} true_slope={true_slope:g} rows={len(case_rows)}", flush=True)
    output = OUT / "inference_calibration.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    elapsed = time.perf_counter() - start
    cpu_end = proc.cpu_times()
    manifest = {
        "manifest_version": "kuntur-phase2-v1.2-round-4-inference-calibration-0.1.0",
        "status": "COMPLETED_SYNTHETIC_CALIBRATION",
        "scope": "Synthetic monthly-trend inference calibration; no observations or sealed target values read.",
        "protocol_sha256": _sha256(ROOT / "docs/protocols/continuous_optimization_v1_2.md"),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "processes": processes, "n_months": N, "primary_replications_per_effect": REPS,
        "bootstrap_processes": sorted(BOOTSTRAP_PROCESSES), "bootstrap_outer_replications": BOOTSTRAP_REPS,
        "bootstrap_draws_per_outer_replication": BOOTSTRAP_DRAWS,
        "arma_mle_replications_per_effect": ARMA_REPS,
        "inference_methods": ["OLS_HAC lags 6/12/18/24 (all four for rho=.844; lag 12 otherwise)", "iterated feasible GLS AR(1)", "one-pass AR(1) prewhitening", "ARMA(1,1) SARIMAX on ARMA DGP", "moving residual block bootstrap (length 12)", "seasonally aligned year-block bootstrap (length 12)", "Theil-Sen iid interval", "within-calendar-month Theil-Sen point estimator"],
        "coverage_interval": "Exact two-sided Clopper-Pearson 95% interval for Monte Carlo binomial rates.",
        "primary_seed": SEED,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "resources": {"elapsed_wall_seconds": elapsed, "process_cpu_seconds": (cpu_end.user - cpu_start.user) + (cpu_end.system - cpu_start.system), "peak_rss_bytes": int(proc.memory_info().rss), "logical_cpus_available": psutil.cpu_count(logical=True), "physical_cpus_available": psutil.cpu_count(logical=False)},
        "limitations": ["ARMA maximum-likelihood calibration is intentionally bounded to 200 replications per effect; its Monte Carlo intervals are correspondingly wide.", "Bootstrap methods use 500 outer simulations and 299 inner draws on four representative processes.", "DGP-specific method comparison does not establish a universally calibrated method; seasonal AR dependence is intentionally adverse to simple Newey-West lag choices.", "The Theil-Sen iid interval is an independence-based reference, while the within-season Theil-Sen row is a point-estimator diagnostic only.", "Finite-sample results at n=276 should not be treated as evidence that a specific model is correct for environmental observations."],
        "rows": int(len(frame)), "output_path": output.relative_to(ROOT).as_posix(), "output_sha256": _sha256(output), "output_bytes": output.stat().st_size,
    }
    _atomic_json(OUT / "round_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
