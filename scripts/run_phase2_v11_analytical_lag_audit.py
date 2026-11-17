#!/usr/bin/env python3
"""Lag sensitivity and controlled AR(1) validation for v1.1 trend inference."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
from scipy.stats import beta

from kuntur_data.phase2.analytical import analyze_variable, benjamini_hochberg, monthly_anomalies


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
OUT = ROOT / "results/phase2/v1_1/analytical"
HAC_LAGS = (6, 12, 18, 24)
VARIABLES = {
    "merra2_t2m_c": 0.80,
    "imerg_precip_monthly_total": 0.80,
    "gldas_soil_moisture_0_10cm": 0.80,
    "gldas_soil_moisture_10_40cm": 0.80,
    "gldas_soil_moisture_40_100cm": 0.80,
    "gldas_soil_moisture_100_200cm": 0.80,
    "modis_ndvi": 0.10,
    "modis_evi": 0.10,
}
SIM_RHOS = (0.0, 0.4, 0.7, 0.844)
SIM_REPS = 500
SEED = 20261116


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def binomial_interval(successes: int, trials: int) -> list[float]:
    alpha = 0.05
    low = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    high = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [low, high]


def simulation_design(periods: pd.PeriodIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    elapsed = np.arange(len(periods), dtype=float) / 12.0
    months = pd.Series(periods.month, dtype="category")
    month_dummies = pd.get_dummies(months, drop_first=True, dtype=float).to_numpy()
    x = np.column_stack([np.ones(len(periods)), elapsed, month_dummies])
    seasonal = np.tile(np.array([0.0, 0.5, 1.0, 0.7, 0.3, -0.1, -0.4, -0.3, -0.1, 0.1, 0.3, 0.2]), len(periods) // 12)
    return x, elapsed, seasonal


def ar1_series(rng: np.random.Generator, n: int, rho: float, sigma: float = 0.25) -> np.ndarray:
    eps = rng.normal(0.0, sigma, n)
    values = np.empty(n, dtype=float)
    values[0] = eps[0]
    innovation_scale = np.sqrt(max(0.0, 1.0 - rho**2))
    for t in range(1, n):
        values[t] = rho * values[t - 1] + innovation_scale * eps[t]
    return values


def fit_glsar(y: np.ndarray, x: np.ndarray):
    """Fit a seasonal-regression AR(1) GLS model with iterated rho estimation."""
    return sm.GLSAR(y, x, rho=1).iterative_fit(maxiter=100)


def simulate_hac(reps: int = SIM_REPS) -> list[dict]:
    simulation_start = time.perf_counter()
    rng = np.random.default_rng(SEED)
    periods = pd.period_range("2002-01", periods=276, freq="M")
    x, elapsed, seasonal = simulation_design(periods)
    output = []
    for rho in SIM_RHOS:
        for true_slope_per_decade in (0.0, 0.5):
            covered = {lag: 0 for lag in HAC_LAGS}
            rejected = {lag: 0 for lag in HAC_LAGS}
            slopes = {lag: [] for lag in HAC_LAGS}
            glsar_covered = 0
            glsar_rejected = 0
            glsar_slopes = []
            for _ in range(reps):
                y = seasonal + (true_slope_per_decade / 10.0) * elapsed + ar1_series(rng, len(periods), rho)
                for lag in HAC_LAGS:
                    result = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "use_correction": True})
                    slope = float(result.params[1] * 10.0)
                    ci = np.asarray(result.conf_int(alpha=0.05)[1], dtype=float) * 10.0
                    p = float(result.pvalues[1])
                    slopes[lag].append(slope)
                    covered[lag] += int(ci[0] <= true_slope_per_decade <= ci[1])
                    rejected[lag] += int(p < 0.05)
                ar_fit = fit_glsar(y, x)
                ar_slope = float(ar_fit.params[1] * 10.0)
                ar_ci = np.asarray(ar_fit.conf_int(alpha=0.05)[1], dtype=float) * 10.0
                ar_p = float(ar_fit.pvalues[1])
                glsar_slopes.append(ar_slope)
                glsar_covered += int(ar_ci[0] <= true_slope_per_decade <= ar_ci[1])
                glsar_rejected += int(ar_p < 0.05)
            for lag in HAC_LAGS:
                success = covered[lag] if true_slope_per_decade else reps - rejected[lag]
                output.append({
                    "rho": rho,
                    "true_slope_per_decade": true_slope_per_decade,
                    "hac_maxlags_months": lag,
                    "repetitions": reps,
                    "ci95_coverage": covered[lag] / reps,
                    "ci95_coverage_clopper_pearson": binomial_interval(covered[lag], reps),
                    "false_positive_rate_at_0_05": rejected[lag] / reps if true_slope_per_decade == 0 else None,
                    "false_positive_rate_clopper_pearson": binomial_interval(rejected[lag], reps) if true_slope_per_decade == 0 else None,
                    "power_at_0_05": rejected[lag] / reps if true_slope_per_decade != 0 else None,
                    "power_clopper_pearson": binomial_interval(rejected[lag], reps) if true_slope_per_decade != 0 else None,
                    "mean_estimated_slope_per_decade": float(np.mean(slopes[lag])),
                    "mean_abs_slope_error_per_decade": float(np.mean(np.abs(np.asarray(slopes[lag]) - true_slope_per_decade))),
                    "process": "monthly fixed seasonal effects plus stationary Gaussian AR(1), innovation scaled to stationary SD 0.25",
                    "method": "ols_hac",
                })
            output.append({
                "rho": rho,
                "true_slope_per_decade": true_slope_per_decade,
                "hac_maxlags_months": None,
                "method": "glsar_ar1_iterative",
                "repetitions": reps,
                "ci95_coverage": glsar_covered / reps,
                "ci95_coverage_clopper_pearson": binomial_interval(glsar_covered, reps),
                "false_positive_rate_at_0_05": glsar_rejected / reps if true_slope_per_decade == 0 else None,
                "false_positive_rate_clopper_pearson": binomial_interval(glsar_rejected, reps) if true_slope_per_decade == 0 else None,
                "power_at_0_05": glsar_rejected / reps if true_slope_per_decade != 0 else None,
                "power_clopper_pearson": binomial_interval(glsar_rejected, reps) if true_slope_per_decade != 0 else None,
                "mean_estimated_slope_per_decade": float(np.mean(glsar_slopes)),
                "mean_abs_slope_error_per_decade": float(np.mean(np.abs(np.asarray(glsar_slopes) - true_slope_per_decade))),
                "process": "monthly fixed seasonal effects plus stationary Gaussian AR(1), innovation scaled to stationary SD 0.25",
            })
            print(json.dumps({"event": "AR1_SIMULATION_BLOCK_COMPLETE", "rho": rho, "true_slope_per_decade": true_slope_per_decade, "repetitions": reps, "elapsed_seconds": round(time.perf_counter() - simulation_start, 2)}, allow_nan=False), flush=True)
    return output


def main() -> None:
    start = time.perf_counter()
    if not GOLD.is_file():
        raise SystemExit(f"Frozen Gold not found: {GOLD}")
    gold = pd.read_parquet(GOLD)
    if len(gold) != 5520 or gold.month.nunique() != 276:
        raise SystemExit("Frozen Gold v1 dimensions changed; refusing the v1.1 audit")
    source_hash = sha256(GOLD)
    trend_rows = []
    for variable, threshold in VARIABLES.items():
        for lag in HAC_LAGS:
            trend = analyze_variable(gold, variable, coverage_threshold=threshold, hac_maxlags=lag)["trend"]
            trend_rows.append({"variable_id": variable, "method": "ols_hac", "hac_maxlags_months": lag, **trend,
                               "slope_per_decade": trend["ols_hac_slope_per_decade"],
                               "ci95_per_decade": trend["ols_hac_ci95_per_decade"],
                               "p_value": trend["ols_hac_p_value"]})
        ts = monthly_anomalies(gold, variable, coverage_threshold=threshold).dropna(subset=["anomaly"]).copy()
        date_index = ts.month_start.dt.year * 12 + ts.month_start.dt.month - 1
        ts["elapsed_years"] = (date_index - date_index.iloc[0]) / 12.0
        month_effects = pd.get_dummies(ts.calendar_month.astype("category"), prefix="month", drop_first=True, dtype=float)
        x = np.column_stack([np.ones(len(ts)), ts.elapsed_years.to_numpy(dtype=float), month_effects.to_numpy(dtype=float)])
        fit = fit_glsar(ts.value.to_numpy(dtype=float), x)
        ci = np.asarray(fit.conf_int(alpha=0.05)[1], dtype=float) * 10.0
        trend_rows.append({
            "variable_id": variable,
            "method": "glsar_ar1_iterative",
            "hac_maxlags_months": None,
            "units": str(ts.units.iloc[0]),
            "n_months": len(ts),
            "first_month": str(ts.month.iloc[0]),
            "last_month": str(ts.month.iloc[-1]),
            "coverage_threshold": threshold,
            "median_valid_area_fraction": float(ts.valid_area_fraction_bbox.median()),
            "slope_per_decade": float(fit.params[1] * 10.0),
            "ci95_per_decade": [float(ci[0]), float(ci[1])],
            "p_value": float(fit.pvalues[1]),
            "estimated_ar1_rho": float(np.asarray(fit.model.rho).reshape(-1)[0]),
            "trend_interpretation": "AR(1) GLS sensitivity estimate; model assumes a stationary AR(1) error process",
        })
    # Freeze one eight-hypothesis confirmatory family for each sensitivity lag;
    # this remains descriptive because the lag comparison is exploratory.
    for method, lag in [("ols_hac", x) for x in HAC_LAGS] + [("glsar_ar1_iterative", None)]:
        rows = [r for r in trend_rows if r["method"] == method and r["hac_maxlags_months"] == lag]
        adjusted = benjamini_hochberg([r["p_value"] for r in rows])
        for row, q in zip(rows, adjusted):
            row["p_value_fdr_bh"] = q if np.isfinite(q) else None
            row["fdr_family_size"] = len(rows)

    simulation_start = time.perf_counter()
    simulation = simulate_hac()
    OUT.mkdir(parents=True, exist_ok=True)
    trend_path = OUT / "trend_hac_lag_sensitivity_2002_2024.csv"
    sim_path = OUT / "synthetic_ar1_hac_validation.csv"
    manifest_path = OUT / "analytical_lag_audit_manifest.json"
    pd.DataFrame(trend_rows).to_csv(trend_path, index=False)
    pd.DataFrame(simulation).to_csv(sim_path, index=False)
    manifest = {
        "version": "kuntur-analytical-v1.1-lag-audit-0.1.0",
        "status": "COMPLETED_EXPLORATORY_SENSITIVITY",
        "source_gold": "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
        "source_gold_sha256": source_hash,
        "source_rows": len(gold),
        "coverage_thresholds": VARIABLES,
        "fixed_hypothesis_family": list(VARIABLES),
        "fdr_method": "Benjamini-Hochberg separately at each HAC lag across the predeclared eight-variable family",
        "methods": {"ols_hac": {"maxlags_months": list(HAC_LAGS), "finite_sample_correction": True}, "glsar_ar1_iterative": {"rho_estimation": "iterated GLSAR, up to 100 iterations", "assumption": "stationary AR(1) errors"}},
        "synthetic_validation": {"repetitions_per_rho_and_slope": SIM_REPS, "rhos": list(SIM_RHOS), "true_slopes_per_decade": [0.0, 0.5], "seed": SEED, "design": "276 monthly observations with 12 calendar-month fixed effects, linear trend, and stationary Gaussian AR(1) residuals; tests OLS-HAC lags and iterated AR(1) GLS."},
        "artifacts": {str(trend_path.relative_to(ROOT)): {"sha256": sha256(trend_path), "bytes": trend_path.stat().st_size}, str(sim_path.relative_to(ROOT)): {"sha256": sha256(sim_path), "bytes": sim_path.stat().st_size}},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "elapsed_seconds": time.perf_counter() - start,
        "limitations": ["Lag comparison is a sensitivity analysis, not a post-hoc method-selection license.", "The synthetic AR(1) process is a controlled reference, not a complete climate-generating model.", "A single regional bounding-box series cannot resolve subregional heterogeneity.", "Classical seasonal Mann-Kendall inference retains an independence caveat and is not used to claim discoveries here."],
    }
    atomic_json(manifest_path, manifest)
    # Compact terminal report for the immediate checkpoint.
    print(json.dumps({"status": manifest["status"], "gold_sha256": source_hash, "trend_rows": len(trend_rows), "simulation_rows": len(simulation), "elapsed_seconds": manifest["elapsed_seconds"], "trend_path": str(trend_path), "simulation_path": str(sim_path), "manifest_path": str(manifest_path)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
