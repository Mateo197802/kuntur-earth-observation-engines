#!/usr/bin/env python3
"""Aggregate completed v1.2 development rounds into local reproducibility artifacts."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kuntur_data.phase2.predictive import sha256_file


ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "results/optimization"
REPORT = ROOT / "reports/optimization"


def _paired_v11_reference(round1_dir: Path, rows: pd.DataFrame) -> dict:
    manifest = json.loads((round1_dir / "round_manifest.json").read_text(encoding="utf-8"))
    run_id = manifest["best_candidate"]["run_id"]
    registry = [json.loads(line) for line in (round1_dir / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    current = [r for r in registry if r.get("run_id") == run_id and r.get("status") == "COMPLETED"]
    previous = pd.read_parquet(ROOT / "results/phase2/v1_1/rolling_predictions_development_2003_2018.parquet")
    previous = previous[previous.model_id == "svr_rbf"].groupby(["fold_id", "sample_id", "target_month", "y_true"], as_index=False).y_pred.mean()
    cur = {(r["fold_id"], p["target_month"]): p for r in current for p in r["predictions"]}
    old = {(r.fold_id, r.target_month): r for r in previous.itertuples()}
    keys = sorted(set(cur) & set(old))
    if len(keys) != 108:
        raise ValueError(f"Round 1 and v1.1 SVR do not share exactly 108 targets ({len(keys)})")
    fold_diffs = {}
    for fold in sorted({k[0] for k in keys}):
        fold_keys = [k for k in keys if k[0] == fold]
        fold_diffs[fold] = np.asarray([
            abs(cur[k]["y_pred"] - cur[k]["y_true"]) - abs(old[k].y_pred - old[k].y_true)
            for k in fold_keys
        ], dtype=float)
    rng = np.random.default_rng(20261115)
    draws = np.empty(2000, dtype=float)
    for b in range(len(draws)):
        parts = []
        for values in fold_diffs.values():
            indexes = []
            while len(indexes) < len(values):
                start = int(rng.integers(0, len(values)))
                indexes.extend(((start + np.arange(12)) % len(values)).tolist())
            parts.extend(values[np.asarray(indexes[:len(values)])])
        draws[b] = float(np.mean(parts))
    difference = float(np.mean(np.concatenate(list(fold_diffs.values()))))
    return {
        "comparison": "round_1_selected_svr_minus_v1_1_svr",
        "new_mean_fold_mae": float(manifest["best_candidate"]["mean_fold_mae"]),
        "v1_1_mean_fold_mae": float(rows.loc[rows.model_id == "svr_rbf", "mean_fold_mae"].iloc[0]),
        "paired_delta_mae_new_minus_v1_1": difference,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "paired_target_months": len(keys),
        "months_with_lower_absolute_error": int(sum(abs(cur[k]["y_pred"] - cur[k]["y_true"]) < abs(old[k].y_pred - old[k].y_true) for k in keys)),
        "folds_with_lower_mean_absolute_error": int(sum(float(np.mean(x)) < 0 for x in fold_diffs.values())),
        "ci_interpretation": "uncertain; interval includes zero" if float(np.quantile(draws, 0.025)) <= 0 <= float(np.quantile(draws, 0.975)) else "interval excludes zero; still post-selection development evidence",
    }


def _paired_run_delta(new_dir: Path, new_run_id: str, ref_dir: Path, ref_run_id: str, seed: int) -> dict:
    def load(path: Path, run_id: str) -> list[dict]:
        return [json.loads(line) for line in (path / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip() and json.loads(line).get("run_id") == run_id and json.loads(line).get("status") == "COMPLETED"]
    new_records = load(new_dir, new_run_id)
    ref_records = load(ref_dir, ref_run_id)
    new_map = {(r["fold_id"], p["target_month"]): p for r in new_records for p in r["predictions"]}
    ref_map = {(r["fold_id"], p["target_month"]): p for r in ref_records for p in r["predictions"]}
    keys = sorted(set(new_map) & set(ref_map))
    if len(keys) != 108:
        raise ValueError(f"Expected 108 paired development targets, found {len(keys)}")
    by_fold = {}
    for fold in sorted({key[0] for key in keys}):
        by_fold[fold] = np.asarray([
            abs(new_map[key]["y_pred"] - new_map[key]["y_true"]) - abs(ref_map[key]["y_pred"] - ref_map[key]["y_true"])
            for key in keys if key[0] == fold
        ], dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(2000, dtype=float)
    for i in range(len(draws)):
        sample = []
        for values in by_fold.values():
            indexes = []
            while len(indexes) < len(values):
                start = int(rng.integers(0, len(values)))
                indexes.extend(((start + np.arange(12)) % len(values)).tolist())
            sample.extend(values[np.asarray(indexes[:len(values)])])
        draws[i] = float(np.mean(sample))
    observed = float(np.mean(np.concatenate(list(by_fold.values()))))
    return {
        "new_run_id": new_run_id, "reference_run_id": ref_run_id,
        "delta_mae_new_minus_reference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
        "paired_targets": len(keys), "months_with_lower_absolute_error": int(sum(abs(new_map[k]["y_pred"] - new_map[k]["y_true"]) < abs(ref_map[k]["y_pred"] - ref_map[k]["y_true"]) for k in keys)),
        "folds_with_lower_mean_absolute_error": int(sum(float(np.mean(x)) < 0 for x in by_fold.values())),
        "ci_interpretation": "uncertain; interval includes zero" if float(np.quantile(draws, 0.025)) <= 0 <= float(np.quantile(draws, 0.975)) else "interval excludes zero; post-selection development evidence",
    }


def _write_dashboard(history: pd.DataFrame, comparison: pd.DataFrame, paired: dict, round3_noise: pd.DataFrame | None = None, round4_calibration: pd.DataFrame | None = None, round5_intervals: pd.DataFrame | None = None, round6_score: dict | None = None, external_validation: dict | None = None) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    latest = comparison.sort_values(["round", "mean_fold_mae", "model_id"], ascending=[False, True, True])
    by_model = latest.groupby(["round", "model_id"], as_index=False, sort=False).first()
    latest_round = int(by_model["round"].max())
    plot_rows = by_model[by_model["round"] == latest_round].sort_values("mean_fold_mae")
    max_mae = max(0.1, float(plot_rows["mean_fold_mae"].max()))
    bar_svg = ["<svg viewBox='0 0 920 420' role='img' aria-label='Mean rolling fold MAE by model'>"]
    bar_svg.append("<text x='15' y='24' font-size='15'>Development rolling validation · lower is better · °C</text>")
    y = 50
    for _, row in plot_rows.iterrows():
        width = max(2, 620 * float(row["mean_fold_mae"]) / max_mae)
        bar_svg.append(f"<text x='15' y='{y+15}' font-size='12'>{html.escape(str(row['model_id']))}</text>")
        bar_svg.append(f"<rect x='235' y='{y}' width='{width:.2f}' height='20' fill='#3976a8'/>")
        bar_svg.append(f"<text x='{245+width:.2f}' y='{y+15}' font-size='12'>{float(row['mean_fold_mae']):.3f}</text>")
        y += 35
    bar_svg.append("</svg>")
    chart = "".join(bar_svg)
    rows_html = []
    for _, row in plot_rows.iterrows():
        skill = row.get("skill_vs_best_baseline_mae", np.nan)
        skill_text = "—" if pd.isna(skill) else f"{float(skill):.1%}"
        rows_html.append(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                row["model_id"], f"{row['mean_fold_mae']:.4f}", f"{row['mean_fold_rmse']:.4f}",
                f"{row['mean_fold_bias']:.4f}", f"{row['max_fold_mae']:.4f}",
                skill_text, row["status"],
            )) + "</tr>"
        )
    stress_html = "<p>No augmentation stress results have been aggregated.</p>"
    if round3_noise is not None and not round3_noise.empty:
        best_record = json.loads((OPT / "round_03_augmentation_robustness" / "round_manifest.json").read_text(encoding="utf-8"))["best_candidate"]
        selected = round3_noise[
            (round3_noise.model_id == best_record["model_id"])
            & (round3_noise.training_scheme == best_record["training_scheme"])
        ]
        summaries = selected.groupby("intensity").agg(
            mean_scenario_mae=("mean_fold_mae", "mean"), worst_scenario_mae=("mean_fold_mae", "max"),
            scenario_count=("scenario", "nunique"),
        ).reset_index().sort_values("intensity")
        stress_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                f"{row.intensity:.0%}", int(row.scenario_count), f"{row.mean_scenario_mae:.4f}",
                f"{row.worst_scenario_mae:.4f}",
            )) + "</tr>"
            for row in summaries.itertuples(index=False)
        )
        stress_html = (
            f"<p>Selected development candidate: {html.escape(best_record['model_id'])} / "
            f"{html.escape(best_record['training_scheme'])}. Results average 11 corruption scenarios "
            "across the three rolling folds; each fold scored 36 valid months.</p>"
            "<table><thead><tr><th>Input corruption</th><th>Scenarios</th><th>Mean scenario MAE (°C)</th>"
            f"<th>Worst scenario MAE (°C)</th></tr></thead><tbody>{stress_rows}</tbody></table>"
            "<p>These are synthetic input sensitivity checks, not estimated real-world error rates. "
            "Candidate selection and stress assessment share development folds.</p>"
        )
    calibration_html = "<p>Round 4 simulation evidence is pending.</p>"
    if round4_calibration is not None and not round4_calibration.empty:
        show = round4_calibration[
            (round4_calibration.true_slope_per_decade == 0.0)
            & round4_calibration.process.isin(["iid_gaussian", "ar1_0844", "seasonal_ar1_12"])
            & round4_calibration.method.isin(["OLS_HAC_lag12", "OLS_HAC_lag24", "FEGLS_AR1_3iter", "MOVING_BLOCK12", "SEASONAL_YEAR_BLOCK12", "ARMA11_MLE"])
            & round4_calibration.coverage_95.notna()
        ].copy()
        cal_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                row.process, row.method, int(row.replications_valid), f"{row.coverage_95:.1%}",
                f"{row.coverage_mc95_low:.1%}–{row.coverage_mc95_high:.1%}",
                f"{row.type1_fpr_005:.1%}",
            )) + "</tr>"
            for row in show.itertuples(index=False)
        )
        calibration_html = (
            "<p>Synthetic null scenarios; coverage and rejection-rate columns include exact 95% Monte Carlo intervals. "
            "No inference method is universally calibrated across these process families.</p>"
            "<table><thead><tr><th>Process</th><th>Method</th><th>Replicates</th><th>95% CI coverage</th>"
            f"<th>Coverage Monte Carlo interval</th><th>Type-I rate</th></tr></thead><tbody>{cal_rows}</tbody></table>"
        )
    interval_html = "<p>Development interval calibration is pending.</p>"
    if round5_intervals is not None and not round5_intervals.empty:
        overall = round5_intervals[round5_intervals.group_type == "overall"].sort_values("mean_winkler_score_c")
        interval_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in (
                row.method, int(row.target_count), f"{row.coverage_90:.1%}",
                f"{row.mean_interval_width_c:.3f}", f"{row.mean_winkler_score_c:.3f}",
            )) + "</tr>"
            for row in overall.itertuples(index=False)
        )
        interval_html = (
            "<p>Chronological development-only scoring on 72 targets from 2013–2018; earlier out-of-fold residuals calibrated each later fold. "
            "The selected model used the same development folds, so these intervals are post-selection evidence.</p>"
            "<table><thead><tr><th>Calibration method</th><th>Months</th><th>Coverage</th><th>Mean width (°C)</th>"
            f"<th>Winkler (°C)</th></tr></thead><tbody>{interval_rows}</tbody></table>"
        )
    sealed_test_html = "<p>The frozen 2025 temporal evaluation has not been scored.</p>"
    if round6_score is not None:
        model = round6_score["model_metrics"]
        baseline = round6_score["baseline_metrics"]
        paired_test = round6_score["paired_baseline_comparison"]
        interval = round6_score["interval_calibration"]
        sealed_test_html = (
            f"<p>Frozen Jan–Dec 2025 retrospective hindcast, n={round6_score['test_target_count']}. "
            f"Model MAE {model['mae_c']:.3f} °C and monthly-climatology MAE {baseline['mae_c']:.3f} °C; "
            f"paired MAE difference {paired_test['paired_delta_mae_model_minus_training_monthly_climatology_c']:+.3f} °C "
            f"({paired_test['months_with_lower_absolute_error_than_baseline']}/12 months lower absolute error for the model). "
            f"Observed interval coverage was {interval['observed_coverage']:.1%} ({int(round(interval['observed_coverage'] * 12))}/12), "
            f"with mean width {interval['mean_width_c']:.3f} °C. These 12 months provide a low-precision descriptive result.</p>"
            "<p>Inputs use rolling T-14 through T-3 histories, strictly before each target month; realized 2025 observations may serve as past inputs for later 2025 targets. "
            "Contemporaneous NASA release timestamps were not established, so this is labeled a retrospective hindcast. "
            "The model and interval choice were frozen before the target data were opened; no model selection used this test.</p>"
            "<p><code>results/optimization/round_06_2025_test_score/sealed_2025_score.json</code></p>"
        )
    external_html = "<p>No independently checked point-to-grid comparison is recorded.</p>"
    if external_validation is not None and external_validation.get("status") == "PASSED":
        external_rows = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
                source, metrics["matched_station_months"], metrics["matched_months"],
                f"{metrics['mae_c']:.3f}", f"{metrics['rmse_c']:.3f}",
                f"{metrics['bias_c']:+.3f}", f"{metrics['pearson_r']:.3f}",
                f"{metrics['mean_cell_distance_km']:.1f}",
            )) + "</tr>"
            for source, metrics in sorted(external_validation.get("sources", {}).items())
        )
        external_html = (
            "<p>Validated monthly station comparisons to nearest MERRA-2 native-grid cell centers (2002–2024); no regridding or elevation adjustment.</p>"
            "<table><thead><tr><th>Source</th><th>Pairs</th><th>Months</th><th>MAE °C</th><th>RMSE °C</th><th>Bias °C</th><th>Pearson r</th><th>Mean distance km</th></tr></thead><tbody>"
            f"{external_rows}</tbody></table>"
            "<p>Descriptive agreement only: MERRA-2 may assimilate stations, spatial support differs, and INAMHI covers 2002–2012. Station-level tables remain in CEDIA.</p>"
            "<p><code>results/optimization/native_grid_collocation_summary.json</code> · <code>results/optimization/native_grid_collocation_validation.json</code></p>"
        )
    history_scope = "completed candidate, corruption-stress, synthetic-calibration and development interval results"
    if round6_score is not None:
        history_scope += " plus the frozen 2025 temporal test"
    if external_validation is not None:
        history_scope += " plus validated descriptive external point-to-grid comparisons"
    content = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KUNTUR v1.2 optimization dashboard</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#152536}}h1,h2{{color:#173d5a}}.card{{border:1px solid #ccd6df;border-radius:8px;padding:1rem;margin:1rem 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:.5rem;border-bottom:1px solid #d9e1e8;text-align:left}}th{{background:#edf3f7}}.note{{color:#485968}}svg{{width:100%;height:auto;max-height:450px}}code{{word-break:break-all}}</style>
</head><body>
<h1>KUNTUR v1.2 continuous optimization</h1>
<p class="note">Local, self-contained dashboard. Development selection and the frozen 2025 temporal test are reported separately.</p>
<section class="card"><h2>Latest development candidate round</h2>{chart}</section>
<section class="card"><h2>Model comparison</h2><table><thead><tr><th>Model</th><th>Mean fold MAE</th><th>Mean fold RMSE</th><th>Bias</th><th>Worst fold MAE</th><th>Skill vs strongest baseline</th><th>Status</th></tr></thead><tbody>{''.join(rows_html)}</tbody></table></section>
<section class="card"><h2>Paired reference comparison</h2>
<p>Selected round 1 SVR minus the v1.1 SVR on 108 matched development targets: <strong>{paired['paired_delta_mae_new_minus_v1_1']:.4f} °C MAE</strong>; 95% 12-month block-bootstrap interval [{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}] °C. {html.escape(paired['ci_interpretation'])}.</p>
<p>Individual target months with lower absolute error: {paired['months_with_lower_absolute_error']}/108. Folds improved: {paired['folds_with_lower_mean_absolute_error']}/3.</p>
</section>
<section class="card"><h2>Round 3: augmentation and input corruption</h2>{stress_html}</section>
<section class="card"><h2>Round 4: trend inference calibration</h2>{calibration_html}<p><code>results/optimization/round_04_inference_calibration/inference_calibration.csv</code></p></section>
<section class="card"><h2>Round 5: forecast-interval calibration</h2>{interval_html}<p><code>results/optimization/round_05_interval_calibration/round_manifest.json</code></p></section>
<section class="card"><h2>Frozen 2025 temporal test</h2>{sealed_test_html}</section>
<!-- KUNTUR_V12_EXTERNAL_COLLOCATION_START --><section class="card"><h2>External point-to-grid comparison</h2>{external_html}</section><!-- KUNTUR_V12_EXTERNAL_COLLOCATION_END -->
<section class="card"><h2>Metric registry</h2><p>{len(history)} metric rows across {html.escape(history_scope)}. Each row carries its partition, run ID, status and supporting artifact SHA-256.</p><p><code>results/optimization/metric_history.parquet</code></p></section>
</body></html>"""
    (REPORT / "metric_dashboard.html").write_text(content, encoding="utf-8")


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    round1 = OPT / "round_01_architecture_search"
    round_manifest = json.loads((round1 / "round_manifest.json").read_text(encoding="utf-8"))
    all_candidates = []
    for round_dir in sorted(OPT.glob("round_*")):
        for comparison_name in ("candidate_comparison.csv", "feature_comparison.csv"):
            comparison_path = round_dir / comparison_name
            if comparison_path.is_file():
                all_candidates.append(pd.read_csv(comparison_path))
    round3_dir = OPT / "round_03_augmentation_robustness"
    round3_candidates = pd.DataFrame()
    round3_noise = pd.DataFrame()
    round3_manifest: dict = {}
    if (round3_dir / "augmentation_comparison.csv").is_file():
        round3_candidates = pd.read_csv(round3_dir / "augmentation_comparison.csv")
        round3_noise = pd.read_csv(round3_dir / "noise_robustness_by_scheme.csv")
        round3_manifest = json.loads((round3_dir / "round_manifest.json").read_text(encoding="utf-8"))
        registry = [json.loads(line) for line in (round3_dir / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        run_lookup = {}
        for row in registry:
            if row.get("status") == "COMPLETED":
                key = (row["model_id"], row["training_scheme"])
                previous = run_lookup.setdefault(key, row["run_id"])
                if previous != row["run_id"]:
                    raise ValueError(f"Round 3 scheme maps to multiple run IDs: {key}")
        strongest_baseline_mae = float(round_manifest["strongest_baseline"]["mean_fold_mae"])
        normalized = []
        for row in round3_candidates.to_dict(orient="records"):
            model, scheme = row["model_id"], row["training_scheme"]
            mean_mae = float(row["mean_fold_mae"])
            normalized.append({
                **row,
                "run_id": run_lookup[(model, scheme)],
                "feature_group": row["feature_group"],
                "config": json.dumps(next(r["config"] for r in registry if r.get("status") == "COMPLETED" and r["model_id"] == model and r["training_scheme"] == scheme), sort_keys=True),
                "skill_vs_best_baseline_mae": ((strongest_baseline_mae - mean_mae) / strongest_baseline_mae) if strongest_baseline_mae else 0.0,
                "mean_fit_wall_seconds_per_fold": float(np.mean([r["fit_wall_seconds_sum_seeds"] for r in registry if r.get("status") == "COMPLETED" and r["model_id"] == model and r["training_scheme"] == scheme])),
                "mean_fit_cpu_seconds_per_fold": float(np.mean([r["fit_cpu_seconds_sum_seeds"] for r in registry if r.get("status") == "COMPLETED" and r["model_id"] == model and r["training_scheme"] == scheme])),
                "status": "COMPLETED_DEVELOPMENT_ONLY",
            })
        round3_candidates = pd.DataFrame(normalized)
        all_candidates.append(round3_candidates)
    round4_dir = OPT / "round_04_inference_calibration"
    round4_calibration = pd.DataFrame()
    round4_manifest: dict = {}
    if (round4_dir / "inference_calibration.csv").is_file():
        round4_calibration = pd.read_csv(round4_dir / "inference_calibration.csv")
        round4_manifest = json.loads((round4_dir / "round_manifest.json").read_text(encoding="utf-8"))
    round5_dir = OPT / "round_05_interval_calibration"
    round5_intervals = pd.DataFrame()
    round5_manifest: dict = {}
    if (round5_dir / "interval_calibration_summary.csv").is_file():
        round5_intervals = pd.read_csv(round5_dir / "interval_calibration_summary.csv")
        round5_manifest = json.loads((round5_dir / "round_manifest.json").read_text(encoding="utf-8"))
    round6_score_path = OPT / "round_06_2025_test_score" / "sealed_2025_score.json"
    round6_score: dict | None = None
    if round6_score_path.is_file():
        round6_score = json.loads(round6_score_path.read_text(encoding="utf-8"))
        if round6_score.get("status") != "SCORED_FROZEN_2025_TEST":
            raise ValueError("Round 6 score artifact has an unexpected status")
    collocation_validation_path = OPT / "native_grid_collocation_validation.json"
    collocation_summary_path = OPT / "native_grid_collocation_summary.json"
    external_validation: dict | None = None
    if collocation_validation_path.is_file() and collocation_summary_path.is_file():
        external_validation = json.loads(collocation_validation_path.read_text(encoding="utf-8"))
        if external_validation.get("status") != "PASSED" or external_validation.get("summary_sha256") != sha256_file(collocation_summary_path):
            raise ValueError("Native-grid collocation validation does not match the aggregate summary")
    if not all_candidates:
        raise SystemExit("No completed optimization rounds are available")
    comparison = pd.concat(all_candidates, ignore_index=True)
    best_arch = pd.concat([pd.read_csv(p) for p in sorted(OPT.glob("round_*/best_by_architecture.csv"))], ignore_index=True)
    old_summary = json.loads((ROOT / "results/phase2/v1_1/rolling_model_summary_development_2003_2018.json").read_text(encoding="utf-8"))
    old_rows = pd.DataFrame(old_summary["summary"])
    paired = _paired_v11_reference(round1, old_rows)
    paired_path = round1 / "paired_v11_reference.json"
    paired_path.write_text(json.dumps(paired, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    comparison_path = OPT / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    best_path = OPT / "best_by_architecture.csv"
    best_arch.to_csv(best_path, index=False)

    artifact_hash = sha256_file(comparison_path)
    metric_rows = []
    for row in comparison.itertuples():
        base = {
            "round": int(row.round), "run_id": row.run_id, "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M",
            "partition": "rolling_validation_2010_2018_development", "model_id": row.model_id,
            "model_version": f"v1.2-round-{row.round}", "sample_count": int(row.target_count),
            "status": row.status, "candidate_spec_sha256": row.candidate_spec_sha256,
            "supporting_artifact_path": comparison_path.relative_to(ROOT).as_posix(),
            "supporting_artifact_sha256": artifact_hash,
        }
        for metric in ("mean_fold_mae", "mean_fold_rmse", "mean_fold_bias", "max_fold_mae",
                       "skill_vs_best_baseline_mae", "mean_fit_wall_seconds_per_fold",
                       "mean_fit_cpu_seconds_per_fold", "peak_rss_bytes"):
            metric_rows.append({**base, "metric": metric, "value": float(getattr(row, metric))})
    metric_rows.append({
        "round": 1, "run_id": str(round_manifest["best_candidate"]["run_id"]),
        "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M", "partition": "rolling_validation_2010_2018_development",
        "model_id": str(round_manifest["best_candidate"]["model_id"]), "model_version": "v1.2-round-1",
        "sample_count": 108, "status": "COMPLETED_DEVELOPMENT_ONLY",
        "candidate_spec_sha256": round_manifest["best_candidate"]["candidate_spec_sha256"],
        "supporting_artifact_path": paired_path.relative_to(ROOT).as_posix(),
        "supporting_artifact_sha256": sha256_file(paired_path),
        "metric": "paired_delta_mae_vs_v1_1_svr", "value": paired["paired_delta_mae_new_minus_v1_1"],
        "ci95_low": paired["ci95_low"], "ci95_high": paired["ci95_high"],
    })
    round2_manifest_path = OPT / "round_02_feature_ablation" / "round_manifest.json"
    if round2_manifest_path.is_file():
        round2_manifest = json.loads(round2_manifest_path.read_text(encoding="utf-8"))
        paired2 = round2_manifest.get("paired_vs_round1_best")
        if paired2:
            metric_rows.append({
                "round": 2, "run_id": paired2["new_run_id"],
                "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M", "partition": "rolling_validation_2010_2018_development",
                "model_id": str(round2_manifest["best_candidate"]["model_id"]), "model_version": "v1.2-round-2",
                "sample_count": int(paired2["paired_targets"]), "status": "POST_SELECTION_DEVELOPMENT_COMPARISON",
                "candidate_spec_sha256": str(round2_manifest["best_candidate"]["candidate_spec_sha256"]),
                "supporting_artifact_path": (OPT / "round_02_feature_ablation/feature_comparison.csv").relative_to(ROOT).as_posix(),
                "supporting_artifact_sha256": sha256_file(OPT / "round_02_feature_ablation/feature_comparison.csv"),
                "metric": "paired_delta_mae_vs_round1_best", "value": paired2["delta_mae_new_minus_reference"],
                "ci95_low": paired2["ci95_low"], "ci95_high": paired2["ci95_high"],
            })
    if not round3_candidates.empty:
        round3_manifest_path = round3_dir / "round_manifest.json"
        round3_artifact_hash = sha256_file(round3_dir / "augmentation_comparison.csv")
        noise_artifact_hash = sha256_file(round3_dir / "noise_robustness_by_scheme.csv")
        for row in round3_candidates.itertuples():
            base = {
                "round": 3, "run_id": row.run_id, "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M",
                "partition": "rolling_validation_2010_2018_development", "model_id": row.model_id,
                "model_version": "v1.2-round-3", "sample_count": int(row.target_count),
                "status": row.status, "candidate_spec_sha256": row.candidate_spec_sha256,
                "supporting_artifact_path": (round3_dir / "augmentation_comparison.csv").relative_to(ROOT).as_posix(),
                "supporting_artifact_sha256": round3_artifact_hash,
            }
            for metric in ("mean_fold_mae", "mean_fold_rmse", "mean_fold_bias", "max_fold_mae",
                           "worst_noise_mean_fold_mae", "training_synthetic_rows_mean",
                           "mean_fit_wall_seconds_per_fold", "mean_fit_cpu_seconds_per_fold", "peak_rss_bytes"):
                metric_rows.append({**base, "metric": metric, "value": float(getattr(row, metric))})
        for row in round3_noise.itertuples():
            metric_rows.append({
                "round": 3, "run_id": run_lookup[(row.model_id, row.training_scheme)],
                "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M", "partition": "rolling_validation_2010_2018_development_corruption_stress",
                "model_id": row.model_id, "model_version": "v1.2-round-3",
                "sample_count": int((row.valid_targets_min + row.valid_targets_max) / 2),
                "status": row.status, "candidate_spec_sha256": str(round3_candidates[
                    (round3_candidates.model_id == row.model_id) & (round3_candidates.training_scheme == row.training_scheme)
                ].candidate_spec_sha256.iloc[0]),
                "supporting_artifact_path": (round3_dir / "noise_robustness_by_scheme.csv").relative_to(ROOT).as_posix(),
                "supporting_artifact_sha256": noise_artifact_hash,
                "metric": "noise_mae", "value": float(row.mean_fold_mae),
                "scenario": row.scenario, "intensity": float(row.intensity),
            })
        best3 = round3_manifest["best_candidate"]
        metric_rows.append({
            "round": 3, "run_id": str(round3_candidates.loc[
                (round3_candidates.model_id == best3["model_id"]) & (round3_candidates.training_scheme == best3["training_scheme"]), "run_id"
            ].iloc[0]),
            "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M", "partition": "rolling_validation_2010_2018_development",
            "model_id": best3["model_id"], "model_version": "v1.2-round-3", "sample_count": int(round3_manifest["dataset"]["validation_targets"]),
            "status": "POST_SELECTION_DEVELOPMENT_COMPARISON", "candidate_spec_sha256": best3["candidate_spec_sha256"],
            "supporting_artifact_path": round3_manifest_path.relative_to(ROOT).as_posix(),
            "supporting_artifact_sha256": sha256_file(round3_manifest_path),
            "metric": "paired_delta_mae_vs_same_model_clean_only", "value": float(round3_manifest["paired_vs_same_model_clean_only"]["delta_mae_model_minus_baseline"]),
            "ci95_low": float(round3_manifest["paired_vs_same_model_clean_only"]["ci95_low"]),
            "ci95_high": float(round3_manifest["paired_vs_same_model_clean_only"]["ci95_high"]),
        })
    if not round4_calibration.empty:
        calibration_artifact_hash = sha256_file(round4_dir / "inference_calibration.csv")
        for row in round4_calibration.itertuples():
            spec = hashlib.sha256(json.dumps(
                {"method": row.method, "process": row.process, "slope": row.true_slope_per_decade, "n": row.n_months},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            base = {
                "round": 4, "run_id": f"r4:{row.process}:slope={row.true_slope_per_decade:g}:{row.method}",
                "dataset_id": "SYNTHETIC_MONTHLY_TREND_DGP",
                "partition": "monte_carlo_null" if row.true_slope_per_decade == 0 else "monte_carlo_injected_trend",
                "model_id": row.method, "model_version": "v1.2-round-4",
                "sample_count": int(row.replications_valid), "n_months": int(row.n_months),
                "scenario": row.process, "true_slope_per_decade": float(row.true_slope_per_decade),
                "status": "COMPLETED_SYNTHETIC_CALIBRATION", "candidate_spec_sha256": spec,
                "supporting_artifact_path": (round4_dir / "inference_calibration.csv").relative_to(ROOT).as_posix(),
                "supporting_artifact_sha256": calibration_artifact_hash,
            }
            for metric in ("coverage_95", "coverage_mc95_low", "coverage_mc95_high", "type1_fpr_005",
                           "type1_mc95_low", "type1_mc95_high", "power_005", "power_mc95_low", "power_mc95_high",
                           "mean_slope_per_decade", "slope_bias_per_decade", "slope_rmse_per_decade", "mean_ci_width_per_decade"):
                value = getattr(row, metric)
                if pd.notna(value):
                    metric_rows.append({**base, "metric": metric, "value": float(value)})
    if not round5_intervals.empty:
        interval_artifact = round5_dir / "interval_calibration_summary.csv"
        interval_hash = sha256_file(interval_artifact)
        selected_spec = round5_manifest["selected_candidate"]["candidate_spec_sha256"]
        for row in round5_intervals.itertuples():
            if row.group_type != "overall":
                continue
            spec = hashlib.sha256(json.dumps(
                {"selected_model_spec": selected_spec, "method": row.method, "nominal_coverage": 0.9},
                sort_keys=True, separators=(",", ":"), allow_nan=False,
            ).encode("utf-8")).hexdigest()
            base = {
                "round": 5, "run_id": f"r5:{row.method}", "dataset_id": "KUNTUR_Ecuador_MERRA2_T2M",
                "partition": "chronological_development_interval_calibration_2013_2018",
                "model_id": f"interval::{row.method}", "model_version": "v1.2-round-5",
                "sample_count": int(row.target_count), "status": "POST_SELECTION_DEVELOPMENT_ONLY",
                "candidate_spec_sha256": spec,
                "supporting_artifact_path": interval_artifact.relative_to(ROOT).as_posix(),
                "supporting_artifact_sha256": interval_hash,
            }
            for metric in ("coverage_90", "mean_interval_width_c", "mean_winkler_score_c"):
                metric_rows.append({**base, "metric": metric, "value": float(getattr(row, metric))})
    if round6_score is not None:
        score_artifact_hash = sha256_file(round6_score_path)
        score_base = {
            "round": 6, "run_id": "r6:sealed_2025", "dataset_id": round6_score["dataset_id"],
            "partition": "frozen_temporal_test_2025_retrospective_hindcast",
            "model_id": round6_score["model"]["id"], "model_version": "v1.2-round-6-frozen",
            "sample_count": int(round6_score["test_target_count"]), "status": round6_score["status"],
            "candidate_spec_sha256": round6_score["model"]["candidate_spec_sha256"],
            "supporting_artifact_path": round6_score_path.relative_to(ROOT).as_posix(),
            "supporting_artifact_sha256": score_artifact_hash,
        }
        metric_values = {
            "model_mae_c": round6_score["model_metrics"]["mae_c"],
            "model_rmse_c": round6_score["model_metrics"]["rmse_c"],
            "model_bias_c": round6_score["model_metrics"]["mean_prediction_minus_observation_bias_c"],
            "baseline_mae_c": round6_score["baseline_metrics"]["mae_c"],
            "baseline_rmse_c": round6_score["baseline_metrics"]["rmse_c"],
            "baseline_bias_c": round6_score["baseline_metrics"]["mean_prediction_minus_observation_bias_c"],
            "paired_delta_mae_vs_training_climatology_c": round6_score["paired_baseline_comparison"]["paired_delta_mae_model_minus_training_monthly_climatology_c"],
            "paired_months_with_lower_absolute_error_than_baseline": round6_score["paired_baseline_comparison"]["months_with_lower_absolute_error_than_baseline"],
            "paired_delta_mae_ci95_low_c": round6_score["paired_baseline_comparison"]["delta_mae_ci95"][0],
            "paired_delta_mae_ci95_high_c": round6_score["paired_baseline_comparison"]["delta_mae_ci95"][1],
            "interval_observed_coverage": round6_score["interval_calibration"]["observed_coverage"],
            "interval_mean_width_c": round6_score["interval_calibration"]["mean_width_c"],
            "interval_mean_winkler_score_c": round6_score["interval_calibration"]["mean_winkler_score_c"],
            "target_bbox_valid_area_fraction_min": round6_score["target_coverage_fraction_bbox_min_max"][0],
            "target_bbox_valid_area_fraction_max": round6_score["target_coverage_fraction_bbox_min_max"][1],
        }
        metric_rows.extend({**score_base, "metric": metric, "value": float(value)} for metric, value in metric_values.items())
        metric_rows.append({
            **score_base, "metric": "test_gold_sha256_present", "value": 1.0,
            "source_test_gold_sha256": round6_score["source_test_gold_sha256"],
        })
    if external_validation is not None:
        collocation_base = {
            "round": 7, "run_id": "external:merra2_native_grid_collocation_2002_2024",
            "dataset_id": "M2TMNXSLV.5.12.4", "partition": "external_point_to_grid_2002_2024",
            "model_id": "MERRA-2-native-grid-cell-center", "model_version": "M2TMNXSLV.5.12.4",
            "status": "VALIDATED_DESCRIPTIVE_POINT_TO_GRID_AGREEMENT",
            "supporting_artifact_path": collocation_validation_path.relative_to(ROOT).as_posix(),
            "supporting_artifact_sha256": sha256_file(collocation_validation_path),
            "collocation_summary_sha256": sha256_file(collocation_summary_path),
        }
        for source, metrics in sorted(external_validation.get("sources", {}).items()):
            base = {
                **collocation_base, "external_source": source,
                "sample_count": int(metrics["matched_station_months"]),
            }
            for metric in (
                "mae_c", "rmse_c", "bias_c", "pearson_r", "mean_cell_distance_km", "p95_cell_distance_km",
                "matched_station_months", "matched_months", "matched_stations", "unique_grid_cells",
            ):
                metric_rows.append({**base, "metric": metric, "value": float(metrics[metric])})
    history = pd.DataFrame(metric_rows)
    history_path = OPT / "metric_history.parquet"
    temp = history_path.with_suffix(".parquet.tmp")
    history.to_parquet(temp, index=False, compression="zstd")
    temp.replace(history_path)

    best = round_manifest["best_candidate"]
    round1_candidates = pd.read_csv(round1 / "candidate_comparison.csv")
    round1_selected_row = round1_candidates.loc[round1_candidates.run_id == best["run_id"]].iloc[0]
    best["config"] = json.loads(round1_selected_row.config)
    log = f"""# Optimization checkpoint

## Round 1 — architecture and regularization

**Root-cause hypothesis:** v1.1's strongest learned model had only a modest advantage over training-only climatology. A bounded regularization search may improve forecasting, while high-capacity candidates may overfit the small sample.

**Experiments executed:** 27 predeclared model configurations across Ridge, RBF-SVR, Random Forest, histogram gradient boosting, MLP and temporal CNN; all three expanding-window folds; three fixed seeds for stochastic models. The 108 monthly targets were identical for each complete candidate. No augmentation in this round.

**Dataset and split:** forecast Gold SHA-256 `{round_manifest['dataset']['sha256']}`; development targets 2003-03–2018-12; validation folds 2010-01–2012-12, 2013-01–2015-12, 2016-01–2018-12. Historical 2019–2024 partitions and sealed 2025 target values were not read.

**Previous metrics:** v1.1 SVR mean-fold MAE 0.3228 °C; strongest baseline training monthly climatology 0.3658 °C.

**New best development candidate:** {best['model_id']} `{best['run_id']}`, config C=1, epsilon=0.05, gamma=scale; mean-fold MAE {best['mean_fold_mae']:.4f} °C, mean-fold RMSE {best['mean_fold_rmse']:.4f} °C, mean-fold bias {best['mean_fold_bias']:.4f} °C, worst-fold MAE {pd.read_csv(round1 / 'candidate_comparison.csv').query('run_id == @best["run_id"]')['max_fold_mae'].iloc[0]:.4f} °C.

**Paired differences and uncertainty:** new SVR minus v1.1 SVR MAE = {paired['paired_delta_mae_new_minus_v1_1']:.4f} °C (95% 12-month circular block-bootstrap interval [{paired['ci95_low']:.4f}, {paired['ci95_high']:.4f}]); lower absolute error in {paired['months_with_lower_absolute_error']}/108 months and all {paired['folds_with_lower_mean_absolute_error']}/3 folds. The interval includes zero; improvement remains uncertain. Versus the strongest baseline, mean-fold delta = {round_manifest['paired_comparison']['delta_mae_model_minus_baseline']:.4f} °C, 95% interval [{round_manifest['paired_comparison']['ci95_low']:.4f}, {round_manifest['paired_comparison']['ci95_high']:.4f}] °C, also including zero.

**Noise robustness:** not newly evaluated in Round 1. Existing v1.1 robustness results remain separate and are not attributed to these tuned candidates.

**Statistical validation:** no new inference calibration in Round 1. Existing v1.1 analytical audit is not counted as a v1.2 result.

**Computational cost:** {round_manifest['resources']['elapsed_wall_seconds']:.2f} s wall time, {round_manifest['resources']['cpu_count_logical']} logical CPUs / {round_manifest['resources']['cpu_count_physical']} physical CPUs available; peak RSS across selected folds {best['peak_rss_bytes'] / (1024**2):.1f} MiB. No CEDIA or SLURM training was used.

**Model artifact hash:** candidate specification SHA-256 `{best['candidate_spec_sha256']}`. This is a content-addressed specification; no final model binary is frozen in this round.

**Decision:** INCONCLUSIVE for scientific superiority. Retain SVR as the next development candidate because it has the lowest mean rolling MAE; do not claim proven improvement over climatology or v1.1 SVR.

**Next highest-value experiment:** controlled feature-group ablations with fold-local anomaly/climatology fitting and past-only rolling features, followed by training augmentation and 25%/50% robustness.
"""
    round2_dir = OPT / "round_02_feature_ablation"
    round2_manifest_path = round2_dir / "round_manifest.json"
    if round2_manifest_path.is_file():
        round2 = json.loads(round2_manifest_path.read_text(encoding="utf-8"))
        previous = json.loads((round1 / "round_manifest.json").read_text(encoding="utf-8"))["best_candidate"]
        current = round2["best_candidate"]
        round2_candidates = pd.read_csv(round2_dir / "feature_comparison.csv")
        round2_selected_row = round2_candidates.loc[round2_candidates.run_id == current["run_id"]].iloc[0]
        current["config"] = json.loads(round2_selected_row.config)
        paired_round1 = _paired_run_delta(round2_dir, current["run_id"], round1, previous["run_id"], 20261117)
        round2["paired_vs_round1_best"] = paired_round1
        current["config"] = json.loads(round2_selected_row.config)
        round2.setdefault("feature_module_sha256", sha256_file(ROOT / "src/kuntur_data/phase2/optimization_features.py"))
        round2.setdefault("dependency_hashes", {
            "predictive_module_sha256": sha256_file(ROOT / "src/kuntur_data/phase2/predictive.py"),
            "augmentation_module_sha256": sha256_file(ROOT / "src/kuntur_data/phase2/augmentation.py"),
        })
        round2_manifest_path.write_text(json.dumps(round2, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        log += f"""

## Round 2 — fold-local feature ablation

**Root-cause hypothesis:** the full lagged feature vector may dilute the short-sample temperature signal; training-derived anomalies, seasonality, history-only summaries, or stable channel reduction may improve development error.

**Experiments executed:** eight predeclared feature groups × all six model families across three expanding-window folds (48 candidates, 144 completed candidate-folds; 108 identical targets each). Imputation, scaling, climatology, and the stability channel subset were fitted using only each fold's training window. The CNN used the same T-3 histories with variable sequence channels and static features joined after temporal pooling. No augmentation in this round.

**Dataset and split:** same frozen forecast Gold SHA-256 `{round_manifest['dataset']['sha256']}` and 108 targets across 2010-01–2018-12; 2025 target data remain sealed and unread.

**Previous metric:** Round 1 selected SVR mean-fold MAE {previous['mean_fold_mae']:.4f} °C; strongest baseline {round2['strongest_baseline_mean_fold_mae']:.4f} °C.

**New best development candidate:** {current['model_id']} with `{current['feature_group']}` and configuration `{json.dumps(current['config'], sort_keys=True)}`; mean-fold MAE {current['mean_fold_mae']:.4f} °C, mean-fold RMSE {current['mean_fold_rmse']:.4f} °C, bias {current['mean_fold_bias']:.4f} °C, worst-fold MAE {current['max_fold_mae']:.4f} °C.

**Paired differences and uncertainty:** versus Round 1 SVR, MAE delta = {paired_round1['delta_mae_new_minus_reference']:.4f} °C, 95% 12-month block-bootstrap interval [{paired_round1['ci95_low']:.4f}, {paired_round1['ci95_high']:.4f}] °C, {paired_round1['months_with_lower_absolute_error']}/108 months and {paired_round1['folds_with_lower_mean_absolute_error']}/3 folds improved. {paired_round1['ci_interpretation']}. Versus the strongest baseline the selected-candidate interval is [{round2['paired_vs_baseline']['ci95_low']:.4f}, {round2['paired_vs_baseline']['ci95_high']:.4f}] °C, but this winner was selected among 48 candidates on those same targets; that interval is optimistic and not confirmatory.

**Noise robustness:** not newly evaluated in Round 2. No augmentation or 25%/50% stress results are attributed to this candidate yet.

**Statistical validation:** no new inference calibration in Round 2.

**Computational cost:** {round2['resources']['elapsed_wall_seconds']:.2f} s wall time on local CPU; six model families × eight groups; all folds checkpointed. The 2025 test remained sealed.

**Model artifact hash:** candidate specification SHA-256 `{current['candidate_spec_sha256']}`; feature implementation SHA-256 `{round2['feature_module_sha256']}`. The feature source hash supplements the run's script hash for reproducibility; no final model binary is frozen.

**Decision:** retain the anomaly-feature Random Forest as a provisional challenger and the Round 1 SVR as a reference. The apparent gain is selected from shared development folds; do not call it independently validated.

**Next highest-value experiment:** controlled training augmentation and 25%/50% corruption on the SVR and Random Forest challengers, using identical noise realizations and base-month-balanced weights.
"""
    if not round3_candidates.empty:
        best3 = round3_manifest["best_candidate"]
        paired3 = round3_manifest["paired_vs_same_model_clean_only"]
        best_row = round3_candidates[
            (round3_candidates.model_id == best3["model_id"])
            & (round3_candidates.training_scheme == best3["training_scheme"])
        ].iloc[0]
        stress_best = round3_noise[
            (round3_noise.model_id == best3["model_id"])
            & (round3_noise.training_scheme == best3["training_scheme"])
        ]
        stress_summary = stress_best.groupby("intensity").mean_fold_mae.agg(["mean", "max"]).to_dict(orient="index")
        log += f"""

## Round 3 — training augmentation and 25%/50% corruption robustness

**Experiments executed:** 14 predeclared training schemes × two Round 2 challenger families × three rolling folds (28 candidates, 84 completed candidate-folds); the same 108 targets were used. Every candidate also received 11 corruption scenarios at 25% and 50% intensity. Each scenario had 36 eligible months per fold. Target values were never augmented.

**Best selected development candidate:** {best3['model_id']} / `{best3['training_scheme']}` using `{best_row['feature_group']}`; mean-fold MAE {best3['mean_fold_mae']:.4f} °C, RMSE {best3['mean_fold_rmse']:.4f} °C, bias {best3['mean_fold_bias']:.4f} °C, worst-fold MAE {best3['worst_fold_mae']:.4f} °C. The best is selected among 28 schemes on these same development folds.

**Paired comparison to same-model clean-only:** MAE delta {paired3['delta_mae_model_minus_baseline']:.4f} °C; 95% 12-month block-bootstrap interval [{paired3['ci95_low']:.4f}, {paired3['ci95_high']:.4f}] °C; {paired3['months_lower_absolute_error']}/108 target-months and {paired3['folds_improved']}/3 folds improved. This is post-selection development evidence, not a confirmatory interval.

**Corruption robustness for selected candidate:** at 25%, mean across 11 scenario MAEs {stress_summary[0.25]['mean']:.4f} °C and worst scenario {stress_summary[0.25]['max']:.4f} °C; at 50%, mean {stress_summary[0.50]['mean']:.4f} °C and worst scenario {stress_summary[0.50]['max']:.4f} °C. These synthetic perturbations measure sensitivity, not real sensor failure prevalence.

**Reproducibility:** source dataset SHA-256 `{round3_manifest['dataset']['sha256']}`; round script SHA-256 `{round3_manifest['script_sha256']}`; runtime {round3_manifest['resources']['elapsed_wall_seconds']:.2f} s. The smoother's all-missing behavior was subsequently tested as warning-free and numerically unchanged for finite windows; the execution manifest retains the exact source hash used by the completed run.

**Decision:** retain the heteroscedastic-only Random Forest as a development candidate, without making a superiority claim before independent calibration and the sealed 2025 score.
"""
    if not round4_calibration.empty:
        def get_cal(process: str, method: str, slope: float = 0.0) -> pd.Series:
            found = round4_calibration[
                (round4_calibration.process == process)
                & (round4_calibration.method == method)
                & (round4_calibration.true_slope_per_decade == slope)
            ]
            if len(found) != 1:
                raise ValueError(f"Expected one Round 4 row for {process}/{method}/{slope}; found {len(found)}")
            return found.iloc[0]

        hac_persistent = get_cal("ar1_0844", "OLS_HAC_lag12")
        fgls_persistent = get_cal("ar1_0844", "FEGLS_AR1_3iter")
        seasonal_hac = get_cal("seasonal_ar1_12", "OLS_HAC_lag12")
        arma_mle = get_cal("arma11", "ARMA11_MLE")
        log += f"""

## Round 4 — synthetic trend-inference calibration

**Experiments executed:** seeded null and +0.5 °C/decade scenarios across nine monthly error processes; 1,000 replicates per slope condition for HAC, feasible GLS, prewhitening and Sen estimators. All four HAC lags were compared under rho=0.844 and lag 12 elsewhere. Moving and seasonally aligned 12-month block bootstraps used 500 outer replicates × 299 inner draws for four representative processes. ARMA(1,1) likelihood intervals used 200 fits per slope condition; no convergence failures were recorded.

**Monte Carlo calibration:** with rho=0.844, OLS-HAC lag 12 covered {hac_persistent['coverage_95']:.1%} of null slopes (exact 95% Monte Carlo interval [{hac_persistent['coverage_mc95_low']:.1%}, {hac_persistent['coverage_mc95_high']:.1%}]); its type-I rejection rate was {hac_persistent['type1_fpr_005']:.1%} [{hac_persistent['type1_mc95_low']:.1%}, {hac_persistent['type1_mc95_high']:.1%}]. Iterated AR(1) feasible GLS covered {fgls_persistent['coverage_95']:.1%} and rejected {fgls_persistent['type1_fpr_005']:.1%}. Under seasonal AR(1) at lag 12, OLS-HAC coverage was {seasonal_hac['coverage_95']:.1%} and type-I error {seasonal_hac['type1_fpr_005']:.1%}. On its matching ARMA(1,1) DGP, the bounded ARMA MLE null run covered {arma_mle['coverage_95']:.1%} with {int(arma_mle['replications_valid'])} valid replicates.

**Interpretation:** no method calibrated at 95% across all dependence structures. Strong persistence and seasonal dependence produced substantial undercoverage for HAC, AR(1) GLS and block-bootstrap intervals. Theil–Sen iid intervals were calibrated for iid noise but failed under dependence. The process-specific ARMA fit performed well in the limited matching case, but 200 replicates give wider Monte Carlo uncertainty. The selector remains scenario-aware; none of these synthetic results proves the residual process of an observed environmental series.

**Reproducibility and scope:** seed {round4_manifest['primary_seed']}; results SHA-256 `{round4_manifest['output_sha256']}`; script SHA-256 `{round4_manifest['script_sha256']}`; {round4_manifest['resources']['elapsed_wall_seconds']:.1f} s elapsed, peak resident memory {round4_manifest['resources']['peak_rss_bytes'] / (1024**2):.1f} MiB. The simulation reads no KUNTUR observations or sealed 2025 targets.
"""
    if not round5_intervals.empty:
        overall = round5_intervals[round5_intervals.group_type == "overall"].set_index("method")
        selected_interval = overall.loc["expanding_symmetric_abs_residual"]
        log += f"""

## Round 5 — chronological forecast-interval calibration

**Scoring design:** the first 2010–2012 out-of-fold window supplied calibration residuals only; the 2013–2015 evaluation used 36 prior months, and the 2016–2018 evaluation used the earlier 72-month history for expanding methods or the latest 36 months for rolling methods. The audit confirms each calibration end date precedes its scored window. No 2019–2024 or 2025 targets were read.

**Methods compared:** expanding and rolling-36-month symmetric finite-sample absolute-residual quantiles, signed empirical residual quantiles, and bias-corrected Gaussian intervals. Overall post-selection development results across 72 targets:

| Method | 90% empirical coverage | Mean width (°C) | Winkler score (°C) |
|---|---:|---:|---:|
"""
        for row in round5_intervals[round5_intervals.group_type == "overall"].sort_values("mean_winkler_score_c").itertuples():
            log += f"| {row.method} | {row.coverage_90:.1%} | {row.mean_interval_width_c:.3f} | {row.mean_winkler_score_c:.3f} |\n"
        log += f"""

**Provisional calibration choice:** expanding symmetric absolute-residual intervals, with {selected_interval['coverage_90']:.1%} observed coverage, {selected_interval['mean_interval_width_c']:.3f} °C mean width and {selected_interval['mean_winkler_score_c']:.3f} °C Winkler score. This choice minimizes the displayed development Winkler score among the compared methods, but it is not independent validation: the candidate was selected on the same folds and monthly residuals are dependent. Freeze it for the sealed evaluation without claiming nominal coverage.

**Artifacts:** interval output SHA-256 `{round5_manifest['artifacts']['development_prediction_intervals']['sha256']}`; summary SHA-256 `{round5_manifest['artifacts']['interval_calibration_summary']['sha256']}`; chronological audit `{round5_manifest['artifacts']['chronological_leakage_audit']['path']}`.
"""
    (REPORT / "ITERATION_LOG.md").write_text(log, encoding="utf-8")
    if round6_score is not None:
        test = round6_score
        log += f"""

## Round 6 — frozen final model and 2025 temporal test

**Pre-test freeze:** the three-seed Random Forest, training cutoff 2018-12, T-14 through T-3 feature policy, augmentation scheme, and 90% expanding symmetric residual interval were frozen before 2025 Gold values were prepared. Model bundle SHA-256 `{test['model']['model_bundle_sha256']}`; candidate spec SHA-256 `{test['model']['candidate_spec_sha256']}`. Training used {test['training']['months']} months, 2003-03–2018-12, from forecast Gold SHA-256 `{test['training']['forecast_source_sha256']}`.

**Test data:** MERRA-2 `M2TMNXSLV.5.12.4`, Ecuador pilot bounding box, January–December 2025; 12 target months with {test['test_gold_rows']} Gold rows across six variables. The Gold artifact SHA-256 is `{test['source_test_gold_sha256']}` and bbox valid-area coverage ranged {test['target_coverage_fraction_bbox_min_max'][0]:.1%}–{test['target_coverage_fraction_bbox_min_max'][1]:.1%}. All monthly feature windows were T-14 through T-3 and precede their target month. This is a rolling-origin retrospective hindcast: realized 2025 observations are eligible as lagged inputs for later 2025 targets. Contemporaneous source-release dates were not established.

**Frozen test results:** model MAE {test['model_metrics']['mae_c']:.4f} °C, RMSE {test['model_metrics']['rmse_c']:.4f} °C, and bias {test['model_metrics']['mean_prediction_minus_observation_bias_c']:+.4f} °C. The training-only monthly-climatology baseline had MAE {test['baseline_metrics']['mae_c']:.4f} °C and RMSE {test['baseline_metrics']['rmse_c']:.4f} °C. Model-minus-baseline MAE was {test['paired_baseline_comparison']['paired_delta_mae_model_minus_training_monthly_climatology_c']:+.4f} °C; the model had lower absolute error in {test['paired_baseline_comparison']['months_with_lower_absolute_error_than_baseline']}/12 months. The 12-month circular block interval is degenerate and non-informative.

**Prediction intervals:** the frozen nominal 90% interval covered {test['interval_calibration']['observed_coverage']:.1%} of the 12 targets, with {test['interval_calibration']['mean_width_c']:.4f} °C mean width and {test['interval_calibration']['mean_winkler_score_c']:.4f} °C mean Winkler score. This observed coverage is descriptive only: calibration residuals were post-selection and serially dependent, and 12 months do not establish future coverage.

**Artifacts:** score `{round6_score_path.relative_to(ROOT).as_posix()}` (SHA-256 `{sha256_file(round6_score_path)}`); monthly predictions `{test['artifacts']['monthly_predictions']['path']}` (SHA-256 `{test['artifacts']['monthly_predictions']['sha256']}`); frozen model/scorer manifest `results/optimization/round_06_frozen_test/frozen_test_manifest.json`. The test was scored once; no model selection or tuning followed.
"""
    _write_dashboard(history, comparison, paired, round3_noise, round4_calibration, round5_intervals, round6_score, external_validation)
    print(json.dumps({
        "status": "AGGREGATED",
        "model_comparison": comparison_path.relative_to(ROOT).as_posix(),
        "metric_history": history_path.relative_to(ROOT).as_posix(),
        "dashboard": (REPORT / "metric_dashboard.html").relative_to(ROOT).as_posix(),
        "iteration_log": (REPORT / "ITERATION_LOG.md").relative_to(ROOT).as_posix(),
        "metric_rows": len(history), "candidate_rows": len(comparison), "paired_v11_reference": paired,
        "sealed_2025_test_status": round6_score.get("status") if round6_score else "NOT_SCORED",
        "sealed_2025_score_sha256": sha256_file(round6_score_path) if round6_score else None,
        "external_collocation_validation_sha256": sha256_file(collocation_validation_path) if external_validation is not None else None,
        "supporting_artifact_sha256": artifact_hash,
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
