#!/usr/bin/env python3
"""Run frozen coverage-aware trend and analytical sensitivity analyses on Gold v1."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import beta

from kuntur_data.phase2.analytical import (
    ANALYTICAL_COMBINED,
    ANALYTICAL_VERSION,
    analyze_variable,
    benjamini_hochberg,
    perturb_analytical_series,
)


ROOT = Path(__file__).resolve().parents[1]
GOLD_REL = "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
SEED = 20261115
LEVELS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.50)
FAMILIES = ("gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness")
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
ROBUST_VARIABLES = ("merra2_t2m_c", "imerg_precip_monthly_total", "gldas_soil_moisture_0_10cm", "modis_ndvi")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def source_seed(variable_id: str, family: str, intensity: float) -> int:
    value = f"{SEED}|{variable_id}|{family}|{intensity:.3f}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def binomial_ci(successes: int, trials: int, alpha: float = 0.05) -> list[float]:
    if trials <= 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lo, hi]


def synthetic_reference_checks(reps: int = 200) -> dict:
    rng = np.random.default_rng(SEED)
    periods = pd.period_range("2002-01", periods=276, freq="M")
    seasonal = np.tile(np.array([0.0, 0.5, 1.0, 0.7, 0.3, -0.1, -0.4, -0.3, -0.1, 0.1, 0.3, 0.2]), 23)
    null_p = []
    null_slopes = []
    injected_slopes = []
    injected_detected = []
    injected_ci_contains = []
    for rep in range(reps):
        noise = rng.normal(0.0, 0.25, len(periods))
        null = pd.DataFrame({"month": periods.astype(str), "variable_id": "fixture", "value": seasonal + noise, "units": "u", "valid_area_fraction_bbox": 1.0})
        null_trend = analyze_variable(null, "fixture", coverage_threshold=0.0)["trend"]
        null_p.append(null_trend["ols_hac_p_value"])
        null_slopes.append(null_trend["ols_hac_slope_per_decade"])
        t_year = np.arange(len(periods), dtype=float) / 12.0
        injected_noise = rng.normal(0.0, 0.25, len(periods))
        for i in range(1, len(periods)):
            injected_noise[i] += 0.35 * injected_noise[i - 1]
        true_slope_per_decade = 0.5
        injected = pd.DataFrame({"month": periods.astype(str), "variable_id": "fixture", "value": seasonal + 0.05 * t_year + injected_noise, "units": "u", "valid_area_fraction_bbox": 1.0})
        trend = analyze_variable(injected, "fixture", coverage_threshold=0.0)["trend"]
        injected_slopes.append(trend["ols_hac_slope_per_decade"])
        injected_detected.append(trend["ols_hac_p_value"] < 0.05)
        ci = trend["ols_hac_ci95_per_decade"]
        injected_ci_contains.append(ci[0] <= true_slope_per_decade <= ci[1])
    fpr_n = sum(p < 0.05 for p in null_p)
    return {
        "repetitions": reps,
        "null_trend": {"false_positive_rate_at_0_05": fpr_n / reps, "clopper_pearson_95": binomial_ci(fpr_n, reps), "median_abs_slope_per_decade": float(np.median(np.abs(null_slopes))), "p_value_median": float(np.median(null_p)), "synthetic_fixture_only": True},
        "injected_trend": {"true_slope_per_decade": 0.5, "mean_estimated_slope_per_decade": float(np.mean(injected_slopes)), "median_estimated_slope_per_decade": float(np.median(injected_slopes)), "mean_absolute_error_slope_per_decade": float(np.mean(np.abs(np.asarray(injected_slopes) - 0.5))), "detection_rate_at_0_05": float(np.mean(injected_detected)), "clopper_pearson_95_detection_rate": binomial_ci(sum(injected_detected), reps), "hac_ci_contains_true_rate": float(np.mean(injected_ci_contains)), "synthetic_fixture_only": True},
    }


def main() -> None:
    start_time = time.perf_counter()
    gold_path = ROOT / GOLD_REL
    if not gold_path.is_file():
        raise SystemExit(f"Missing frozen Gold input: {gold_path}")
    gold_hash = sha256(gold_path)
    gold = pd.read_parquet(gold_path)
    if len(gold) != 5520 or gold.month.nunique() != 276:
        raise SystemExit("Frozen Gold v1 shape/coverage gate failed")

    trend_rows = []
    anomaly_series = []
    coverage_sensitivity = []
    for variable_id, threshold in VARIABLES.items():
        try:
            result = analyze_variable(gold, variable_id, coverage_threshold=threshold)
            trend_rows.append(result["trend"])
            anomaly_series.append(result["time_series"])
            if variable_id.startswith("modis_"):
                for sensitivity_threshold in (0.10, 0.20):
                    try:
                        sensitivity = analyze_variable(gold, variable_id, coverage_threshold=sensitivity_threshold)
                        coverage_sensitivity.append({"variable_id": variable_id, "coverage_threshold": sensitivity_threshold, "status": "CALCULATED", "n_months": sensitivity["trend"]["n_months"], "slope_per_decade": sensitivity["trend"]["ols_hac_slope_per_decade"], "ci95": sensitivity["trend"]["ols_hac_ci95_per_decade"], "median_valid_area_fraction": sensitivity["trend"]["median_valid_area_fraction"]})
                    except ValueError as exc:
                        coverage_sensitivity.append({"variable_id": variable_id, "coverage_threshold": sensitivity_threshold, "status": "INSUFFICIENT_COVERAGE", "reason": str(exc)})
        except ValueError as exc:
            trend_rows.append({"variable_id": variable_id, "coverage_threshold": threshold, "status": "INSUFFICIENT_COVERAGE", "reason": str(exc)})
    calculated = [r for r in trend_rows if "ols_hac_p_value" in r]
    adjusted = benjamini_hochberg([r["ols_hac_p_value"] for r in calculated])
    for row, p_adj in zip(calculated, adjusted):
        row["ols_hac_p_value_fdr_bh"] = p_adj
        row["fdr_family_size"] = len(calculated)

    robust_rows = []
    derivative_frames = []
    for variable_id in ROBUST_VARIABLES:
        threshold = VARIABLES[variable_id]
        try:
            baseline = analyze_variable(gold, variable_id, coverage_threshold=threshold)["trend"]
        except ValueError as exc:
            robust_rows.append({"variable_id": variable_id, "status": "BASELINE_INSUFFICIENT", "reason": str(exc)})
            continue
        scenarios = [(family, level) for family in FAMILIES for level in LEVELS]
        scenarios += [(family, level) for family in ANALYTICAL_COMBINED for level in (0.25, 0.50)]
        for family, intensity in scenarios:
            seed = source_seed(variable_id, family, intensity)
            derivative = perturb_analytical_series(gold, variable_id, family=family, intensity=intensity, seed=seed, coverage_threshold=threshold)
            derivative["source_gold_sha256"] = gold_hash
            derivative["coverage_threshold_for_inference"] = threshold
            derivative_frames.append(derivative)
            try:
                stress = analyze_variable(derivative, variable_id, coverage_threshold=threshold)["trend"]
                base_slope = float(baseline["ols_hac_slope_per_decade"])
                stress_slope = float(stress["ols_hac_slope_per_decade"])
                base_ci_width = baseline["ols_hac_ci95_per_decade"][1] - baseline["ols_hac_ci95_per_decade"][0]
                stress_ci_width = stress["ols_hac_ci95_per_decade"][1] - stress["ols_hac_ci95_per_decade"][0]
                robust_rows.append({
                    "variable_id": variable_id, "noise_type": family, "noise_intensity": intensity, "seed": seed,
                    "status": "CALCULATED", "baseline_slope_per_decade": base_slope, "stress_slope_per_decade": stress_slope,
                    "absolute_slope_deviation": abs(stress_slope - base_slope),
                    "relative_slope_deviation": abs(stress_slope - base_slope) / abs(base_slope) if abs(base_slope) > 1e-12 else None,
                    "trend_sign_agreement": bool(np.sign(stress_slope) == np.sign(base_slope)),
                    "baseline_ols_ci95": baseline["ols_hac_ci95_per_decade"], "stress_ols_ci95": stress["ols_hac_ci95_per_decade"],
                    "ci_width_change": stress_ci_width - base_ci_width,
                    "significance_decision_stable_p05": bool((stress["ols_hac_p_value"] < 0.05) == (baseline["ols_hac_p_value"] < 0.05)),
                    "ols_theil_sen_sign_concordance": bool(np.sign(stress_slope) == np.sign(stress["theil_sen_slope_per_decade"])),
                    "baseline_valid_months": baseline["n_months"], "stress_valid_months": stress["n_months"],
                    "valid_month_change": stress["n_months"] - baseline["n_months"],
                    "ols_hac_p_value": stress["ols_hac_p_value"], "theil_sen_slope_per_decade": stress["theil_sen_slope_per_decade"],
                    "residual_lag1": stress["residual_autocorrelation_lag1"], "residual_lag12": stress["residual_autocorrelation_lag12"],
                    "independence_claim": False,
                })
            except ValueError as exc:
                robust_rows.append({"variable_id": variable_id, "noise_type": family, "noise_intensity": intensity, "seed": seed, "status": "INSUFFICIENT_COVERAGE_AFTER_PERTURBATION", "reason": str(exc), "baseline_valid_months": baseline["n_months"]})

    metrics_dir = ROOT / "results/phase2/metrics"
    derivative_dir = ROOT / "data/derived/phase2"
    manifest_dir = ROOT / "data/manifests"
    for p in (metrics_dir, derivative_dir, manifest_dir):
        p.mkdir(parents=True, exist_ok=True)
    trend_path = metrics_dir / "analytical_trends_v1.csv"
    robustness_path = metrics_dir / "analytical_robustness_v1.csv"
    series_path = metrics_dir / "analytical_monthly_series_v1.parquet"
    derivative_path = derivative_dir / "analytical_robustness_series_v1.parquet"
    pd.DataFrame(trend_rows).to_csv(trend_path, index=False)
    pd.DataFrame(robust_rows).to_csv(robustness_path, index=False)
    pd.concat(anomaly_series, ignore_index=True).to_parquet(series_path, index=False, compression="zstd")
    pd.concat(derivative_frames, ignore_index=True).to_parquet(derivative_path, index=False, compression="zstd")

    # Regression diagnostics against synthetic reference processes, not observations.
    reference = synthetic_reference_checks(reps=200)
    reference_path = metrics_dir / "analytical_synthetic_reference_checks_v1.json"
    atomic_json(reference_path, reference)

    plot_path = metrics_dir / "analytical_robustness_v1.png"
    robustness_df = pd.DataFrame(robust_rows)
    plt.figure(figsize=(10, 6))
    for variable_id, group in robustness_df[robustness_df.status == "CALCULATED"].groupby("variable_id"):
        singles = group[group.noise_type.isin(FAMILIES)].groupby("noise_intensity").absolute_slope_deviation.median()
        plt.plot(singles.index * 100, singles.values, marker="o", label=variable_id)
    plt.xlabel("Perturbation intensity (%)")
    plt.ylabel("Median absolute trend-slope deviation (units/decade)")
    plt.title("Analytical slope sensitivity on versioned derivatives")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    files = [trend_path, robustness_path, series_path, derivative_path, reference_path, plot_path]
    artifacts = {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in files}
    manifest = {
        "manifest_version": "1.0.0", "status": "VALIDATED_WITH_COVERAGE_LIMITS", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": "kuntur-phase2-analytical-v1-20261115", "analytical_version": ANALYTICAL_VERSION,
        "source_gold_path": GOLD_REL, "source_gold_sha256": gold_hash, "source_rows": len(gold),
        "source_lineage_sha256_values": sorted(gold.source_sha256.dropna().astype(str).unique().tolist()),
        "region_id": "ecuador_pilot_bbox", "region_status": "PRELIMINARY_EXTRACTION_AREA",
        "coverage_thresholds": VARIABLES, "modis_coverage_sensitivity": coverage_sensitivity,
        "methods": ["calendar-month climatology", "seasonal anomaly", "OLS with calendar-month fixed effects and HAC/Newey-West uncertainty (12-month maxlag)", "seasonal within-month Theil-Sen", "classic seasonal Mann-Kendall with within-month independence assumption and autocorrelation diagnostics", "Benjamini-Hochberg FDR across calculated variables"],
        "primary_trend_rows": trend_rows, "robustness_scenarios_per_variable": 46,
        "robustness_families": list(FAMILIES) + list(ANALYTICAL_COMBINED), "robustness_levels": LEVELS,
        "synthetic_reference_checks": reference,
        "limitations": ["Synthetic perturbations are not independent observations or domain validation.", "Mann-Kendall p-values use classic seasonal independence; residual lag-1/lag-12 diagnostics are reported and serial dependence can affect inference.", "MODIS strict-QA coverage is sparse; thresholds 10% and 20% are reported separately.", "A preliminary Ecuador bbox is not Coast/Andes/Amazon administrative coverage.", "Trend associations do not establish causal attribution."],
        "artifacts": artifacts, "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "resource_usage": {"elapsed_seconds": time.perf_counter() - start_time},
    }
    manifest_path = manifest_dir / "analytical_benchmark_manifest_v1.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "source_gold_sha256": gold_hash, "trend_rows": len(trend_rows), "robustness_rows": len(robust_rows), "robustness_derivative_rows": sum(len(x) for x in derivative_frames), "synthetic_reference": reference, "artifact_paths": {"manifest": str(manifest_path), "trends": str(trend_path), "robustness": str(robustness_path), "series": str(series_path), "derivatives": str(derivative_path), "plot": str(plot_path)}, "elapsed_seconds": manifest["resource_usage"]["elapsed_seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
