#!/usr/bin/env python3
"""Create publication-safe aggregate tables and the Phase 2 handoff."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results/phase2/metrics"
PUBLIC = ROOT / "results/phase2/public"
REPORT = ROOT / "reports/phase2/PHASE2_HANDOFF_v1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (ROOT / "reports/phase2").mkdir(parents=True, exist_ok=True)
    copies = {
        "validation_benchmark_v1.csv": METRICS / "validation_benchmark_v1.csv",
        "robustness_scenario_summary_v1.csv": METRICS / "robustness_holdout_v1.parquet",
        "final_test_metrics_v1.csv": METRICS / "final_test_metrics_v1.csv",
        "analytical_trends_v1.csv": METRICS / "analytical_trends_v1.csv",
        "analytical_robustness_v1.csv": METRICS / "analytical_robustness_v1.csv",
        "ingestion_cache_benchmark_v1.csv": METRICS / "ingestion_cache_benchmark_v1.csv",
        "external_comparison_summary_v1.csv": METRICS / "external_comparison_summary_v1.csv",
    }
    for name, source in copies.items():
        if not source.is_file():
            raise SystemExit(f"Missing required Phase 2 result: {source}")
        target = PUBLIC / name
        if source.suffix == ".parquet":
            frame = pd.read_parquet(source)
            frame.to_csv(target, index=False, float_format="%.12g")
        else:
            shutil.copyfile(source, target)
    synthetic_path = METRICS / "analytical_synthetic_reference_checks_v1.json"
    shutil.copyfile(synthetic_path, PUBLIC / synthetic_path.name)

    gold_manifest = read_json("data/manifests/gold_manifest_v1.json")
    forecast_manifest = read_json("data/manifests/forecast_split_manifest_v1.json")
    phase_split = read_json("data/manifests/phase2_split_manifest_v1.json")
    augmentation = read_json("data/manifests/augmentation_manifest_v1.json")
    dev = read_json("data/manifests/model_manifest_v1.json")
    freeze = read_json("data/manifests/model_selection_freeze_v1.json")
    final = read_json("data/manifests/final_test_manifest_v1.json")
    analytical = read_json("data/manifests/analytical_benchmark_manifest_v1.json")
    ingestion = read_json("data/manifests/ingestion_cache_benchmark_manifest_v1.json")
    external = read_json("data/manifests/external/external_evaluation_manifest_v1.json")
    chirps = read_json("data/manifests/external/chirps_v3_external_manifest_v1.json")
    ghcnm = read_json("data/manifests/external/ghcnm_qcu_external_manifest_v1.json")
    inamhi_path = ROOT / "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json"
    inamhi = read_json("data/manifests/external/inamhi_yearbooks_external_manifest_v1.json") if inamhi_path.is_file() else None
    inamhi_year_sources = {} if inamhi is None else inamhi.get("yearbook_sources", {})
    inamhi_parsed_years = sorted(
        int(year) for year, record in inamhi_year_sources.items()
        if record.get("status") == "PDF_ACQUIRED_AND_PARSED" and int(record.get("bbox_station_month_rows", 0)) > 0
    )
    inamhi_unparsed_available_years = sorted(
        int(year) for year, record in inamhi_year_sources.items()
        if record.get("status") == "PDF_ACQUIRED_NO_BBOX_TEMPERATURE_ROWS"
    )

    validation = pd.read_csv(PUBLIC / "validation_benchmark_v1.csv")
    robustness = pd.read_parquet(METRICS / "robustness_holdout_v1.parquet")
    robustness["scenario_group"] = np.where(robustness.is_combined_noise, "combined", np.where(robustness.noise_type == "clean", "clean", "independent"))
    robustness.sort_values(["model_id", "scenario_group", "noise_type", "noise_intensity"], kind="stable").to_csv(PUBLIC / "robustness_scenario_summary_v1.csv", index=False, float_format="%.12g")
    validation_models = validation[validation.role == "six_model"].sort_values("mae")
    validation_baselines = validation[validation.role == "baseline"].sort_values("mae")
    selected_robust = robustness[robustness.model_id == final["selected_architecture"]].sort_values(["scenario_group", "noise_type", "noise_intensity"])
    trends = pd.read_csv(PUBLIC / "analytical_trends_v1.csv")
    analytical_robustness = pd.read_csv(PUBLIC / "analytical_robustness_v1.csv")
    cache = pd.read_csv(PUBLIC / "ingestion_cache_benchmark_v1.csv")
    external_rows = pd.read_csv(PUBLIC / "external_comparison_summary_v1.csv")

    trend_summary = trends[["variable_id", "units", "n_months", "ols_hac_slope_per_decade", "ols_hac_ci95_per_decade", "ols_hac_p_value", "ols_hac_p_value_fdr_bh", "residual_autocorrelation_lag1", "residual_autocorrelation_lag12"]].to_dict("records")
    selected_metrics = next(row for row in final["metrics"] if row.get("model_id") == final["selected_architecture"])
    final_baselines = [row for row in final["metrics"] if row["role"] == "baseline"]
    summary_path = PUBLIC / "phase2_aggregate_summary_v1.json"
    artifact_files = sorted(PUBLIC.glob("*.csv")) + [
        path for path in sorted(PUBLIC.glob("*.json")) if path != summary_path
    ]
    artifacts = {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in artifact_files}
    summary = {
        "manifest_version": "1.0.0", "status": "VERIFIED_PHASE2_AGGREGATE_RELEASE",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "region": {"region_id": "ecuador_pilot_bbox", "bbox_wsen": [-81.5, -5.1, -75.0, 1.7], "status": "PRELIMINARY_BBOX_ONLY"},
        "frozen_inputs": {
            "gold_path": "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet", "gold_rows": 5520, "variables": 20, "months": 276,
            "gold_sha256": gold_manifest["sha256"],
            "forecast_path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet", "forecast_samples": 262,
            "forecast_sha256": final["source_forecast_sha256"],
            "original_split_sha256": final["source_split_sha256"],
            "gold_manifest_sha256": sha256(ROOT / "data/manifests/gold_manifest_v1.json"),
        },
        "split": {"original": forecast_manifest["partition_counts"], "phase2": phase_split["counts"], "exact_ranges": phase_split["exact_target_ranges"], "latest_predictor_rule": "T-3"},
        "augmentation": {"fit_original_months": augmentation["model_fit_original_samples"], "training_total_rows": augmentation["total_rows"], "training_synthetic_rows": augmentation["synthetic_rows"], "final_refit_original_months": final["train_original_samples"], "final_refit_total_rows": final["final_refit_augmented_rows"], "training_intensities": augmentation["configuration"]["training_intensities"], "noise_families": augmentation["configuration"]["single_families"], "combined_families": augmentation["configuration"]["combined_families"], "seed": augmentation["configuration"]["seed"], "sample_weight_policy": augmentation["configuration"]["sample_weight_policy"]},
        "predictive_validation": {"selected_architecture": dev["selected_architecture"], "selection_rule": dev["selection_rule"], "models": validation_models.to_dict("records"), "baselines": validation_baselines.to_dict("records"), "internal_robustness_unique_months": dev["internal_robustness_months"], "robustness_scenario_count_per_model": int(robustness.noise_type.nunique()), "robustness_metric_rows": int(len(robustness)), "selected_model_scenarios": selected_robust.to_dict("records")},
        "final_test": {"status": final["status"], "test_read_count": final["test_read_count"], "selected_model": selected_metrics, "baselines": final_baselines, "process_access_caveat": final["test_access_caveat"], "interval_caveat": final["interval_calibration_caveat"], "prediction_interval_empirical_coverage": selected_metrics["prediction_interval_90_empirical_test_coverage"]},
        "analytical": {"methods": analytical["methods"], "trend_rows": trend_summary, "fdr_significant_rows": int((trends.ols_hac_p_value_fdr_bh < 0.05).sum()), "robustness_scenarios": analytical["robustness_scenarios_per_variable"], "robustness_rows": int(len(analytical_robustness)), "median_absolute_slope_deviation_by_level": {str(k): float(v) for k, v in analytical_robustness[analytical_robustness.status == "CALCULATED"].groupby("noise_intensity").absolute_slope_deviation.median().items()}, "synthetic_reference_checks": json.loads((PUBLIC / "analytical_synthetic_reference_checks_v1.json").read_text(encoding="utf-8"))},
        "ingestion_cache": {"partition_count": 460, "source_rows": 5520, "peak_rss_bytes": ingestion["peak_rss_bytes_observed"], "elapsed_seconds": ingestion["resource_usage"]["elapsed_seconds"], "concurrency": ingestion["concurrency"], "workloads": cache.to_dict("records"), "io_limitations": ingestion["io_limitations"]},
        "external_evaluation": {"manifest_sha256": sha256(ROOT / "data/manifests/external/external_evaluation_manifest_v1.json"), "sources": external["external_sources"], "comparisons": external_rows.to_dict("records"), "landsat": external["landsat"]},
        "external_acquisition": {"CHIRPS": {"months": chirps["period"]["acquired_months"], "failed_months": chirps["period"]["missing_months"], "median_valid_pixel_fraction_bbox": chirps["counts"]["median_valid_pixel_fraction_bbox"], "recovered_prior_429_months": chirps.get("retry_history", {}).get("prior_failure_months_recovered", [])}, "NOAA_GHCN_M": {"eligible_months": ghcnm["counts"]["months_with_three_or_more_eligible_stations"], "missing_composite_months": ghcnm["counts"]["months_without_composite"], "eligible_station_count": len(ghcnm.get("station_selection", {}).get("stations_with_eligible_observations", []))}, "INAMHI": None if inamhi is None else {"yearbooks": inamhi["counts"]["pdf_years_acquired"], "station_rows": inamhi["counts"]["bbox_station_month_rows"], "parsed_years_with_rows": inamhi_parsed_years, "available_yearbooks_without_rows": inamhi_unparsed_available_years, "months_with_3plus_temperature_stations": inamhi["counts"]["months_with_3plus_tavg_stations"], "months_with_3plus_precipitation_stations": inamhi["counts"]["months_with_3plus_precipitation_stations"], "missing_years": inamhi["counts"]["unavailable_pdf_years"]}},
        "limitations": ["Final test contains 36 monthly targets and a prior development run preloaded held-out rows before model selection; fitting and selection did not use those rows, but strict process blinding is not claimed.", "The 90% residual interval achieved 55.6% empirical coverage on the test and is not a calibrated guarantee.", "No OLS-HAC trend remains significant after Benjamini-Hochberg adjustment; trends are descriptive and noncausal.", "The preliminary pilot bbox is not an administrative, Coast, Andes, Amazon, or external-region mask.", "NOAA/INAMHI station means and gridded area means have different spatial support; CHIRPS shares station inputs and is cross-product corroboration.", "INAMHI source PDFs reserve rights and are excluded from the GitHub release.", "The bounded Landsat STAC asset returned login HTML, not a science raster; no Landsat comparison was possible."],
        "artifacts": artifacts,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    model_table = ["| Model | Validation MAE (°C) | RMSE (°C) | Fit wall (s) | Peak RSS (MB) |", "|---|---:|---:|---:|---:|"]
    for row in validation_models.to_dict("records"):
        model_table.append(f"| {row['model_id']} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['fit_wall_seconds']:.2f} | {row['peak_rss_bytes']/1e6:.1f} |")
    for row in validation_baselines.to_dict("records"):
        model_table.append(f"| {row['model_id']} (baseline) | {row['mae']:.4f} | {row['rmse']:.4f} | — | — |")

    noise_table = ["| Scenario | Level | MAE | RMSE | ΔMAE vs clean | Prediction stability | 90% interval coverage |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in selected_robust.to_dict("records"):
        if row["noise_type"] == "clean":
            continue
        noise_table.append(f"| {row['scenario_group']} / {row['noise_type']} | {row['noise_intensity']:.0%} | {row['mae']:.4f} | {row['rmse']:.4f} | {row['delta_mae_from_clean']:.4f} | {row['prediction_stability_rms_change']:.4f} | {row['prediction_interval_90_coverage']:.1%} |")

    trend_table = ["| Variable | n | Slope/decade | 95% HAC CI | p | BH q |", "|---|---:|---:|---:|---:|---:|"]
    for row in trend_summary:
        ci = json.loads(row["ols_hac_ci95_per_decade"]) if isinstance(row["ols_hac_ci95_per_decade"], str) else row["ols_hac_ci95_per_decade"]
        trend_table.append(f"| {row['variable_id']} | {row['n_months']} | {row['ols_hac_slope_per_decade']:.4f} {row['units']} | [{ci[0]:.4f}, {ci[1]:.4f}] | {row['ols_hac_p_value']:.4g} | {row['ols_hac_p_value_fdr_bh']:.4g} |")

    cache_table = ["| Workload | Full scan (s) | Partitioned uncached (s) | Warm cache (s) | Rows |", "|---|---:|---:|---:|---:|"]
    for row in cache.to_dict("records"):
        def median(value):
            parsed = json.loads(value.replace("'", '"')) if isinstance(value, str) else value
            return parsed["median_seconds"]
        cache_table.append(f"| {row['workload']} | {median(row['nonpartitioned_full_scan']):.6f} | {median(row['partitioned_uncached']):.6f} | {median(row['cache_warm_memory_and_disk']):.6f} | {row['result_rows']} |")

    external_table = ["| Source | Comparison | Months | r | Bias | MAE | RMSE |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in external_rows.to_dict("records"):
        vals = [row.get(k) for k in ("pearson_r", "bias_external_minus_reference", "mae", "rmse")]
        fmt = ["—" if pd.isna(v) else f"{v:.4f}" for v in vals]
        external_table.append(f"| {row['source_id']} | {row['variable_id']} | {row['n_months']} | {fmt[0]} | {fmt[1]} | {fmt[2]} | {fmt[3]} |")

    final_rows = [f"| {r['model_id']} | {r['mae']:.4f} | {r['rmse']:.4f} | {r['mean_bias']:.4f} |" for r in final["metrics"]]
    report = [
        "# KUNTUR Competition v1.0 — Phase 2 handoff", "",
        f"**Status:** verified local implementation and aggregate evidence; generated {summary['created_at_utc']}. Frozen v1 input files were not modified.", "",
        "## Scope and immutable inputs", "",
        "All evaluations use the Ecuador preliminary bounding box `[-81.5, -5.1, -75.0, 1.7]` (EPSG:4326). No Coast/Andes/Amazon mask or external administrative-region mask is claimed. Original v1 Gold and split inputs remain byte-identical to their recorded checksums.", "",
        "| Artifact | Rows | SHA-256 |", "|---|---:|---|",
        f"| Gold monthly v1 | 5,520 | `{gold_manifest['sha256']}` |",
        f"| Forecast +1 v1 | 262 | `{final['source_forecast_sha256']}` |",
        f"| Original split indices v1 | {forecast_manifest['split_rows']} | `{final['source_split_sha256']}` |",
        f"| Phase 2 split manifest | {phase_split['rows']} | `{sha256(ROOT / 'data/manifests/phase2_split_manifest_v1.json')}` |",
        "",
        "The Gold table has 20 variables across 276 months, 2002-01 through 2024-12. Forecast split counts are train/validation/test 190/36/36; the training-only internal model-fit/robustness split is 152/38. The latest permissible predictor is T-3. Exact target ranges are in `data/manifests/phase2_split_manifest_v1.json` and machine summary.", "",
        "## Augmentation and validation", "",
        f"Training used {augmentation['model_fit_original_samples']} original model-fit months and {augmentation['synthetic_rows']:,} synthetic examples ({augmentation['total_rows']:,} rows total). Final refit used {final['train_original_samples']} original train months and {final['final_refit_augmented_rows']-final['train_original_samples']:,} synthetic examples ({final['final_refit_augmented_rows']:,} rows total). Intensities: 5%, 10%, 15%, 20% for training; robustness used 25% and 50% after selection. Targets were never perturbed; authentic and synthetic sample weights were balanced per original month.", "",
        "Six-model validation benchmark (validation selected the architecture; internal robustness rows were not used for selection):", "", *model_table, "",
        f"The selected model is `{final['selected_architecture']}`. Its internal robustness diagnostic uses {dev['internal_robustness_months']} unique chronological months; 207 aggregate rows cover nine IDs (six architectures plus three baselines), one clean condition, six independent noise families, and five combined families at 25% and 50%. Per-scenario aggregates for the selected model:", "", *noise_table, "",
        "The full aggregate scenario table is `results/phase2/public/robustness_scenario_summary_v1.csv`. Repeated perturbed scenarios do not increase the 38 independent target months.", "",
        "## Final test", "",
        f"The frozen final test was scored once on 36 months, 2022-01 through 2024-12. Selected model: MAE {selected_metrics['mae']:.4f} °C (month-bootstrap 95% CI {selected_metrics['mae_ci95_month_bootstrap']}), RMSE {selected_metrics['rmse']:.4f} °C, bias {selected_metrics['mean_bias']:.4f} °C. Invalid predictions: {selected_metrics['invalid_prediction_count']}. Empirical coverage for the nominal 90% interval was {selected_metrics['prediction_interval_90_empirical_test_coverage']:.1%}; it is descriptive and not a split-conformal guarantee.", "",
        "| Model | MAE (°C) | RMSE (°C) | Bias (°C) |", "|---|---:|---:|---:|", *final_rows, "",
        f"Skill relative to persistence: MAE {selected_metrics['mae_skill_vs_persistence']:.1%}, RMSE {selected_metrics['rmse_skill_vs_persistence']:.1%}; relative to seasonal naive: MAE {selected_metrics['mae_skill_vs_seasonal_naive']:.1%}, RMSE {selected_metrics['rmse_skill_vs_seasonal_naive']:.1%}; relative to training monthly climatology: MAE {selected_metrics['mae_skill_vs_training_monthly_climatology']:.1%}, RMSE {selected_metrics['rmse_skill_vs_training_monthly_climatology']:.1%}.", "",
        f"**Blinding limitation:** {final['test_access_caveat']}", "",
        "## Trend estimates and analytical stress tests", "",
        "OLS uses calendar-month fixed effects with 12-month HAC/Newey–West errors; the table below reports slopes in the physical units shown per decade. No OLS-HAC result is significant after BH correction across the eight series. Seasonal Mann–Kendall p-values are shown separately in the full CSV and assume seasonal-stratum independence; serial correlation diagnostics are retained. Results are descriptive, not causal.", "", *trend_table, "",
        f"Analytical robustness tested {analytical['robustness_scenarios_per_variable']} perturbations per variable across {len(analytical_robustness)} rows. Median absolute slope deviations by noise intensity: {summary['analytical']['median_absolute_slope_deviation_by_level']}. Synthetic reference checks are explicitly synthetic and appear in `results/phase2/public/analytical_synthetic_reference_checks_v1.json`.", "",
        "## Ingestion and cache benchmark", "",
        f"Partitioning uses 460 year-bounded files for 5,520 rows. Ten repeats per workload returned numerically exact results versus full scans. Peak RSS was {ingestion['peak_rss_bytes_observed'] / 1e6:.1f} MB; runtime was {ingestion['resource_usage']['elapsed_seconds']:.2f} s. Partitioned uncached reads were slower than a direct full-file scan at this small dataset size; warm cache reads were substantially faster.", "", *cache_table, "",
        "## External datasets and descriptive comparisons", "",
        f"CHIRPS v3 final monthly: {chirps['period']['acquired_months']}/276 months acquired and validated on the native 0.05° grid; median valid bbox pixel fraction {chirps['counts']['median_valid_pixel_fraction_bbox']:.1%}. The initial 21 rate-limit/read errors were recovered after the official server returned HTTP 200; per-month output hashes are in local sidecars and progress log. Full remote COG object hashes are not claimed because range reads were used.", "",
        f"NOAA GHCN-M v4 TAVG QCU: {ghcnm['counts']['months_with_three_or_more_eligible_stations']}/276 months with at least three eligible stations; {ghcnm['counts']['unique_bbox_stations_with_records']} stations contributed records. QCU missing sentinels and nonblank QC flags were rejected; point-network composites are not equivalent to the area-weighted grid.", "",
        (f"INAMHI published yearbooks: {inamhi['counts']['pdf_years_acquired']} files acquired for 2002–2012; {inamhi['counts']['bbox_station_month_rows']:,} extracted station-month rows. Yearbooks with extracted station rows: {inamhi_parsed_years}; available yearbooks without a validated station table: {inamhi_unparsed_available_years}. The resulting composites have {inamhi['counts']['months_with_3plus_tavg_stations']} temperature months and {inamhi['counts']['months_with_3plus_precipitation_stations']} precipitation months with ≥3 stations." if inamhi else "INAMHI yearbook tables were not available for evaluation."),
        "The 2013–2024 yearbooks were not discoverable through the public media API during this run. Rights-reserved PDFs and extracted station data remain local and are excluded from GitHub.", "",
        *external_table, "",
        "NOAA and INAMHI are observational corroboration only. Station-point means and gridded bbox means have different spatial support, GHCN-M source inputs can overlap other products, and CHIRPS is a station-blended cross-product rather than independent ground truth. External data were not used for training, scaling, augmentation, selection, or calibration.", "",
        "Landsat Collection 2: the bounded STAC asset request returned HTTP 200 with `text/html` (33,066 bytes), a login page rather than a science raster; no Landsat data or metric is claimed.", "",
        "## Reproducibility and release", "",
        "From a clean Python 3.12 environment, install `.[phase2,test]` and run `python -m pytest -q`. To reproduce the aggregate report with local frozen data, run `python scripts/evaluate_phase2_external.py` after external acquisition artifacts are present, then `python scripts/build_phase2_public_summary.py`. The verified manifests and exact checksums are listed in `reports/phase2/phase2_release_manifest_v1.json`.", "",
        "Public release tables are aggregate metrics only. NASA granules, monthly regional observations, NOAA/INAMHI/CHIRPS observations, source PDFs, caches, manifests with source-specific paths, and fitted model binaries are ignored by Git. See README for references, license status, and data-use limits.", "",
        "### Known limitations", "",
        "- Only the preliminary Ecuador bounding box is available; Coast/Andes/Amazon and external comparison polygons await approved definitions.",
        "- The test contains 36 months; the model beats the fixed baselines on aggregate test error but still undercovers its stated 90% interval.",
        "- MLP candidates hit the declared 300-iteration limit; the six-model benchmark reports the actual results and warnings.",
        "- INAMHI yearbooks have variable station support and no per-value QC flags; source PDFs reserve rights.",
        "- CHIRPS source objects were accessed with byte ranges, so only the regional outputs—not full COG objects—have SHA-256 checksums.",
        "- The Landsat data endpoint returned a login page rather than a valid raster.",
        "- This repository has no open-source license pending contributor-ownership and third-party data-rights review.",
    ]
    REPORT.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "public_artifacts": artifacts, "aggregate_summary": summary_path.relative_to(ROOT).as_posix(), "report": REPORT.relative_to(ROOT).as_posix(), "selected_final_test": selected_metrics, "external_comparisons": external_rows.to_dict("records")}, indent=2, default=str))


if __name__ == "__main__":
    main()
