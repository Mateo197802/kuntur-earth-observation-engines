#!/usr/bin/env python3
"""Build aggregate KUNTUR v1.2 experiment and calibration release artifacts.

The registry contains candidate-level aggregates and provenance only. It does
not copy monthly targets, prediction rows, NASA files, or fitted model weights.
Incomplete statistical calibration scenarios remain explicitly incomplete.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "results/optimization"
STAT = ROOT / "results/statistical_correction"
REPORTS = ROOT / "reports/optimization"
REGISTRY_PATH = OPT / "experiment_registry.jsonl"
CALIBRATION_PATH = OPT / "statistical_calibration.csv"
ARTIFACT_MANIFEST_PATH = OPT / "phase2_v12_release_artifacts_manifest.json"
COLLOCATION_SUMMARY_PATH = OPT / "native_grid_collocation_summary.json"
COLLOCATION_VALIDATION_PATH = OPT / "native_grid_collocation_validation.json"
TREND_ESTIMATES_PATH = ROOT / "results/statistical_correction/real_series_v12/descriptive_trend_estimates.csv"
TREND_MANIFEST_PATH = ROOT / "results/statistical_correction/real_series_v12/descriptive_trend_manifest.json"
NOISE_ROBUSTNESS_PATH = OPT / "round_03_augmentation_robustness/noise_robustness_by_scheme.csv"
SEALED_2025_SCORE_PATH = OPT / "round_06_2025_test_score/sealed_2025_score.json"
FROZEN_TEST_MANIFEST_PATH = OPT / "round_06_frozen_test/frozen_test_manifest.json"

PREDICTIVE_ROUNDS = {
    1: ("round_01_architecture_search", "candidate_comparison.csv", "scripts/run_phase2_v12_round1.py"),
    2: ("round_02_feature_ablation", "feature_comparison.csv", "scripts/run_phase2_v12_round2_features.py"),
    3: ("round_03_augmentation_robustness", "augmentation_comparison.csv", "scripts/run_phase2_v12_round3_augmentation.py"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return value
        return _json_value(parsed)
    if pd.isna(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path.relative_to(ROOT)}")
    return payload


def _validation_checkpoints() -> list[tuple[Path, dict[str, Any]]]:
    directory = STAT / "validation" / "checkpoints"
    if not directory.is_dir():
        return []
    return [(path, _read_json(path)) for path in sorted(directory.glob("validation_*.json"))]


def _validation_phase_status() -> str:
    checkpoints = _validation_checkpoints()
    if any(payload.get("status") == "IN_PROGRESS" for _, payload in checkpoints):
        return "RUNNING_INDEPENDENT_VALIDATION"
    phase_manifest_path = STAT / "validation" / "calibration_manifest.json"
    if phase_manifest_path.is_file():
        phase_manifest = _read_json(phase_manifest_path)
        if phase_manifest.get("status") == "COMPLETED_PHASE":
            return "COMPLETED_PHASE_GATE_REVIEW_PENDING"
        if phase_manifest.get("status") == "PARTIAL_PHASE_RUN":
            return "PARTIAL_VALIDATION_RUN_INCOMPLETE"
        if phase_manifest.get("status") == "SMOKE_ONLY_NOT_VALIDATION":
            return "SMOKE_ONLY_NOT_VALIDATION"
    selection_manifest_path = STAT / "selection" / "selection_manifest.json"
    if selection_manifest_path.is_file():
        selection_manifest = _read_json(selection_manifest_path)
        if selection_manifest.get("status") == "FROZEN_BEFORE_INDEPENDENT_VALIDATION":
            return "NOT_STARTED_SELECTION_FROZEN"
    return "NOT_STARTED_SELECTION_INCOMPLETE"


def _csv_records(path: Path) -> tuple[pd.DataFrame, str]:
    frame = pd.read_csv(path)
    return frame, _sha256(path)


def _candidate_rows() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    split_manifest = ROOT / "data/manifests/phase2_split_manifest_v1.json"
    split_manifest_sha = _sha256(split_manifest) if split_manifest.is_file() else None

    for iteration, (directory, filename, entrypoint) in PREDICTIVE_ROUNDS.items():
        round_dir = OPT / directory
        manifest_path = round_dir / "round_manifest.json"
        comparison_path = round_dir / filename
        manifest = _read_json(manifest_path)
        frame, comparison_sha = _csv_records(comparison_path)
        best = manifest.get("best_candidate") or {}
        best_run = best.get("run_id")
        best_spec = best.get("candidate_spec_sha256")
        dataset = manifest.get("dataset") or {}
        experiment_meta = manifest.get("experiments") or {}
        folds = experiment_meta.get("folds") or manifest.get("folds")
        hypothesis = manifest.get("root_cause_hypothesis")
        for source_row in frame.to_dict(orient="records"):
            row = {key: _json_value(value) for key, value in source_row.items()}
            run_id = row.get("run_id")
            candidate_sha = row.get("candidate_spec_sha256")
            chosen = bool(
                (best_run and run_id == best_run)
                or (best_spec and candidate_sha == best_spec)
            )
            if iteration == 1 and chosen:
                decision = "INCONCLUSIVE"
                decision_reason = "Lowest development MAE, but the paired 95% block interval versus the v1.1 SVR includes zero."
            elif chosen:
                decision = "ACCEPTED"
                decision_reason = "Selected as the next development candidate by the predeclared round comparison; this is post-selection development evidence."
            else:
                decision = "REJECTED"
                decision_reason = "Not selected to advance from this shared development comparison."
            is_baseline = str(row.get("model_id", "")).lower() in {
                "persistence", "seasonal_naive", "training_monthly_climatology"
            }
            provenance = {
                "round_manifest_sha256": _sha256(manifest_path),
                "candidate_table_sha256": comparison_sha,
                "round_script_sha256": manifest.get("script_sha256"),
                "protocol_sha256": manifest.get("protocol_sha256"),
                "feature_module_sha256": manifest.get("feature_module_sha256"),
                "split_manifest_sha256": split_manifest_sha,
                "candidate_spec_sha256": candidate_sha,
                "model_artifact_sha256": None,
                "git_commit_sha": manifest.get("git_commit_sha"),
            }
            missing_evidence = []
            if provenance["git_commit_sha"] is None:
                missing_evidence.append("The completed run manifest does not record a Git commit SHA.")
            if provenance["model_artifact_sha256"] is None:
                missing_evidence.append("No per-candidate fitted model binary was retained; the candidate specification hash is recorded instead.")
            validation_ranges = dataset.get("used_validation_ranges")
            if validation_ranges is None and isinstance(folds, list):
                validation_ranges = [
                    {key: fold.get(key) for key in ("fold_id", "train_end", "validation_start", "validation_end")}
                    for fold in folds
                ]
            records.append({
                "record_type": "predictive_candidate",
                "experiment_id": f"round-{iteration}:{run_id or candidate_sha or row.get('model_id')}:{candidate_sha or 'no-spec-hash'}",
                "iteration_number": iteration,
                "hypothesis": hypothesis,
                "candidate_id": run_id,
                "parent_model": None,
                "parent_model_evidence": "No parent-model field was recorded for this candidate in the completed round manifest.",
                "model_id": row.get("model_id"),
                "candidate_type": "baseline" if is_baseline else "learned_model",
                "dataset": {
                    "path": dataset.get("path"),
                    "sha256": dataset.get("sha256"),
                    "split_manifest_sha256": split_manifest_sha,
                    "training_target_range": dataset.get("target_range_loaded"),
                    "validation_targets": dataset.get("validation_targets") or dataset.get("validation_targets_count"),
                    "validation_ranges": validation_ranges,
                },
                "feature_definition": {
                    "feature_group": row.get("feature_group"),
                    "selected_channels_by_fold": row.get("selected_channels_by_fold"),
                    "feature_count": row.get("feature_count"),
                },
                "training_configuration": {
                    "hyperparameters": row.get("config"),
                    "augmentation_scheme": row.get("training_scheme"),
                    "synthetic_rows_mean": row.get("training_synthetic_rows_mean"),
                    "random_seeds": experiment_meta.get("seeds") or manifest.get("seeds"),
                    "targets_perturbed": False if iteration == 3 else None,
                },
                "metrics": {
                    key: row.get(key) for key in (
                        "fold_count", "target_count", "mean_fold_mae", "mean_fold_rmse", "mean_fold_bias",
                        "max_fold_mae", "pooled_mae", "worst_noise_mean_fold_mae", "skill_vs_best_baseline_mae",
                    ) if key in row
                },
                "run_status": row.get("status"),
                "decision": decision,
                "decision_reason": decision_reason,
                "runtime_resources": {
                    "round_elapsed_wall_seconds": (manifest.get("resources") or {}).get("elapsed_wall_seconds"),
                    "mean_fit_wall_seconds_per_fold": row.get("mean_fit_wall_seconds_per_fold"),
                    "mean_fit_cpu_seconds_per_fold": row.get("mean_fit_cpu_seconds_per_fold"),
                    "peak_rss_bytes": row.get("peak_rss_bytes"),
                    "round_cpu_count_logical": (manifest.get("resources") or {}).get("cpu_count_logical") or (manifest.get("resources") or {}).get("logical_cpus"),
                    "round_cpu_count_physical": (manifest.get("resources") or {}).get("cpu_count_physical") or (manifest.get("resources") or {}).get("physical_cpus"),
                },
                "artifacts": provenance,
                "missing_evidence": missing_evidence,
                "scope_note": "Development-fold results only; candidates share validation targets and are post-selection.",
            })
    return records


def _calibration_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    round4_dir = OPT / "round_04_inference_calibration"
    round4_manifest_path = round4_dir / "round_manifest.json"
    round4_csv_path = round4_dir / "inference_calibration.csv"
    if round4_manifest_path.is_file() and round4_csv_path.is_file():
        manifest = _read_json(round4_manifest_path)
        frame, csv_sha = _csv_records(round4_csv_path)
        for index, row in enumerate(frame.to_dict(orient="records")):
            normalized = {key: _json_value(value) for key, value in row.items()}
            scenario = f"{normalized.get('process')}|n={normalized.get('n_months')}|slope={normalized.get('true_slope_per_decade')}|method={normalized.get('method')}"
            records.append({
                "record_type": "statistical_calibration_scenario",
                "experiment_id": f"round-4:{hashlib.sha256(scenario.encode()).hexdigest()[:16]}:{index:03d}",
                "iteration_number": 4,
                "hypothesis": "Estimate inference calibration under the listed synthetic error processes; results are exploratory and not universal method validation.",
                "candidate_id": normalized.get("method"),
                "model_id": normalized.get("method"),
                "dataset": {"kind": "synthetic_monte_carlo", "n_months": normalized.get("n_months"), "process": normalized.get("process"), "true_slope_per_decade": normalized.get("true_slope_per_decade")},
                "training_configuration": {"seed": normalized.get("seed"), "replications_requested": normalized.get("replications_requested"), "replications_valid": normalized.get("replications_valid")},
                "metrics": {key: normalized.get(key) for key in (
                    "coverage_95", "coverage_mc95_low", "coverage_mc95_high", "type1_fpr_005", "type1_mc95_low", "type1_mc95_high",
                    "power_005", "power_mc95_low", "power_mc95_high", "slope_bias_per_decade", "slope_rmse_per_decade",
                    "mean_ci_width_per_decade", "convergence_failures",
                )},
                "run_status": "COMPLETED_EXPLORATORY_NOT_FINAL_VALIDATION",
                "decision": "INCONCLUSIVE",
                "decision_reason": "Historical R4 calibration pilot is not the frozen v1.2 independent validation and did not establish a universally calibrated method.",
                "artifacts": {
                    "round_manifest_sha256": _sha256(round4_manifest_path),
                    "calibration_table_sha256": csv_sha,
                    "round_script_sha256": manifest.get("script_sha256"),
                    "protocol_sha256": manifest.get("protocol_sha256"),
                },
                "missing_evidence": [],
            })

    round5_dir = OPT / "round_05_interval_calibration"
    round5_manifest_path = round5_dir / "round_manifest.json"
    round5_csv_path = round5_dir / "interval_calibration_summary.csv"
    if round5_manifest_path.is_file() and round5_csv_path.is_file():
        manifest = _read_json(round5_manifest_path)
        frame, csv_sha = _csv_records(round5_csv_path)
        overall = frame[frame.group_type.astype(str) == "overall"]
        selected_info = manifest.get("frozen_interval_calibration") or manifest.get("selected_candidate") or {}
        selected_method = selected_info.get("method") if isinstance(selected_info, dict) else None
        if not selected_method:
            selected_method = "expanding_symmetric_abs_residual"
        for row in overall.to_dict(orient="records"):
            normalized = {key: _json_value(value) for key, value in row.items()}
            method = normalized.get("method")
            chosen = method == selected_method
            records.append({
                "record_type": "forecast_interval_candidate",
                "experiment_id": f"round-5:interval:{method}",
                "iteration_number": 5,
                "hypothesis": "Chronological development-only residual calibration can improve forecast interval coverage and score.",
                "candidate_id": method,
                "model_id": method,
                "dataset": {"kind": "forecast_development_oof", "evaluation_targets": normalized.get("target_count"), "target_years": manifest.get("target_years")},
                "metrics": {key: normalized.get(key) for key in ("target_count", "coverage_90", "mean_interval_width_c", "mean_winkler_score_c", "calibration_count_min", "calibration_count_max")},
                "run_status": "COMPLETED_DEVELOPMENT_ONLY",
                "decision": "ACCEPTED" if chosen else "REJECTED",
                "decision_reason": "Selected on development Winkler score and frozen for the separate 2025 evaluation." if chosen else "Not selected as the frozen interval method on development folds.",
                "artifacts": {"round_manifest_sha256": _sha256(round5_manifest_path), "summary_sha256": csv_sha, "protocol_sha256": manifest.get("protocol_sha256")},
                "missing_evidence": [],
                "scope_note": "Post-selection development calibration; no exchangeability guarantee under serial dependence.",
            })

    score_path = OPT / "round_06_2025_test_score" / "sealed_2025_score.json"
    freeze_path = OPT / "round_06_frozen_test" / "frozen_test_manifest.json"
    if score_path.is_file() and freeze_path.is_file():
        score = _read_json(score_path)
        freeze = _read_json(freeze_path)
        model = score.get("model") or {}
        model_metrics = score.get("model_metrics") or {}
        baseline_metrics = score.get("baseline_metrics") or {}
        paired = score.get("paired_baseline_comparison") or {}
        decision = "REJECTED"
        reason = "Frozen model MAE exceeded training-only climatology; model had lower absolute error in 0/12 test months. The test was not used for retuning."
        records.append({
            "record_type": "sealed_temporal_evaluation",
            "experiment_id": "round-6:frozen-2025-merra2-bbox",
            "iteration_number": 6,
            "hypothesis": "A frozen development-selected model may generalize to a full post-training calendar-year MERRA-2 extension.",
            "candidate_id": model.get("candidate_spec_sha256"),
            "model_id": model.get("id"),
            "dataset": {"dataset_id": score.get("dataset_id"), "region_id": score.get("region_id"), "target_variable": score.get("target_variable"), "test_months": score.get("test_months"), "source_test_gold_sha256": score.get("source_test_gold_sha256"), "evaluation_label": score.get("evaluation_label")},
            "training_configuration": {"training_cutoff_month": score.get("training", {}).get("training_cutoff_month"), "training_scheme": model.get("training_scheme"), "candidate_spec_sha256": model.get("candidate_spec_sha256")},
            "metrics": {"model": model_metrics, "training_monthly_climatology": baseline_metrics, "paired_model_minus_baseline": paired, "interval_calibration": score.get("interval_calibration")},
            "run_status": score.get("status"),
            "decision": decision,
            "decision_reason": reason,
            "runtime_resources": freeze.get("resources"),
            "artifacts": {
                "score_sha256": _sha256(score_path),
                "freeze_manifest_sha256": _sha256(freeze_path),
                "model_bundle_sha256": model.get("model_bundle_sha256"),
                "candidate_spec_sha256": model.get("candidate_spec_sha256"),
                "monthly_predictions_sha256": (score.get("artifacts", {}).get("monthly_predictions") or {}).get("sha256"),
                "scorer_sha256": freeze.get("scorer_sha256"),
                "interval_manifest_sha256": freeze.get("interval_manifest_sha256"),
            },
            "missing_evidence": ["Contemporaneous NASA product release timestamps were not established; this is a retrospective hindcast."],
            "scope_note": "One frozen 12-month test; low precision; no test-driven retuning.",
        })

    selection_dir = STAT / "selection" / "checkpoints"
    if selection_dir.is_dir():
        for path in sorted(selection_dir.glob("selection_*.json")):
            checkpoint = _read_json(path)
            config = checkpoint.get("config") or {}
            progress = checkpoint.get("progress") or {}
            rows = checkpoint.get("rows")
            if checkpoint.get("status") == "COMPLETE" and isinstance(rows, list):
                method_specs = {item.get("name"): item for item in config.get("methods", [])}
                for row in rows:
                    method_name = row.get("method")
                    scenario = f"{config.get('process')}|n={config.get('n_months')}|slope={config.get('slope')}|method={method_name}"
                    records.append({
                        "record_type": "v1_2_calibration_selection_candidate",
                        "experiment_id": f"v1.2-selection:{hashlib.sha256(scenario.encode()).hexdigest()[:20]}",
                        "iteration_number": 4,
                        "hypothesis": "Compare the frozen candidate inference procedures under a predeclared synthetic DGP; selection results are not final validation.",
                        "candidate_id": method_name,
                        "model_id": method_name,
                        "dataset": {"kind": "synthetic_monte_carlo", "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope")},
                        "training_configuration": {"method": method_specs.get(method_name), "replications_requested": row.get("replications_requested"), "seed": row.get("seed"), "bootstrap_inner_draws": row.get("bootstrap_inner_draws")},
                        "metrics": {key: row.get(key) for key in ("type1_fpr_005", "type1_mc95_low", "type1_mc95_high", "coverage_95", "coverage_mc95_low", "coverage_mc95_high", "slope_bias_per_decade", "slope_rmse_per_decade", "mean_ci_width_per_decade", "convergence_failure_rate", "residual_dependence_failure_rate")},
                        "run_status": row.get("validity_status") or "SELECTION_ONLY_NOT_VALIDATED",
                        "decision": "INCONCLUSIVE",
                        "decision_reason": "Selection-stage evidence only; the independent final-validation phase has not started.",
                        "runtime_resources": {"runtime_seconds": row.get("runtime_seconds"), "process_cpu_seconds": row.get("process_cpu_seconds"), "peak_rss_bytes": row.get("peak_rss_bytes")},
                        "artifacts": {"checkpoint_sha256": _sha256(path), "config_sha256": checkpoint.get("config_sha256"), "protocol_sha256": config.get("protocol_sha256"), "statistical_module_sha256": config.get("statistical_module_sha256"), "script_sha256": config.get("script_sha256")},
                        "missing_evidence": ["Independent final-validation result is not available."],
                    })
            elif checkpoint.get("status") == "IN_PROGRESS":
                scenario = f"{config.get('process')}|n={config.get('n_months')}|slope={config.get('slope')}"
                records.append({
                    "record_type": "v1_2_calibration_selection_progress",
                    "experiment_id": f"v1.2-selection-progress:{hashlib.sha256(scenario.encode()).hexdigest()[:20]}",
                    "iteration_number": 4,
                    "hypothesis": "Run the locked selection scenario to its declared replication count with resumable checkpoints.",
                    "candidate_id": None,
                    "model_id": None,
                    "dataset": {"kind": "synthetic_monte_carlo", "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope")},
                    "training_configuration": {"methods": config.get("methods"), "replications_requested": config.get("reps"), "seed": config.get("seed")},
                    "metrics": {"completed_reps": progress.get("completed_reps")},
                    "run_status": "RUNNING_INCOMPLETE",
                    "decision": "INCONCLUSIVE",
                    "decision_reason": "Scenario checkpoint is incomplete; no aggregate method result is emitted until the scenario completes.",
                    "artifacts": {"checkpoint_sha256": _sha256(path), "config_sha256": checkpoint.get("config_sha256")},
                    "missing_evidence": ["Scenario completion and independent final validation."],
                })

    validation_entries = _validation_checkpoints()
    validation_status = _validation_phase_status()
    for path, checkpoint in validation_entries:
        config = checkpoint.get("config") or {}
        progress = checkpoint.get("progress") or {}
        scenario = f"{config.get('process')}|n={config.get('n_months')}|slope={config.get('slope')}"
        if checkpoint.get("status") == "COMPLETE" and isinstance(checkpoint.get("rows"), list):
            for row in checkpoint["rows"]:
                method_name = row.get("method")
                row_scenario = f"{scenario}|method={method_name}"
                records.append({
                    "record_type": "v1_2_independent_validation_result",
                    "experiment_id": f"v1.2-validation:{hashlib.sha256(row_scenario.encode()).hexdigest()[:20]}",
                    "iteration_number": 4,
                    "hypothesis": "Evaluate the preselected method under its locked independent-validation DGP and seed.",
                    "candidate_id": method_name,
                    "model_id": method_name,
                    "dataset": {"kind": "synthetic_monte_carlo", "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope")},
                    "training_configuration": {"replications_requested": row.get("replications_requested"), "seed": row.get("seed"), "bootstrap_inner_draws": row.get("bootstrap_inner_draws")},
                    "metrics": {key: row.get(key) for key in ("type1_fpr_005", "type1_mc95_low", "type1_mc95_high", "coverage_95", "coverage_mc95_low", "coverage_mc95_high", "slope_bias_per_decade", "slope_rmse_per_decade", "mean_ci_width_per_decade", "convergence_failure_rate", "residual_dependence_failure_rate")},
                    "run_status": row.get("validity_status") or "VALIDATION_SCENARIO_COMPLETE",
                    "decision": "INCONCLUSIVE",
                    "decision_reason": "Scenario output is recorded; final release gates are evaluated only after the complete independent-validation phase.",
                    "runtime_resources": {"runtime_seconds": row.get("runtime_seconds"), "process_cpu_seconds": row.get("process_cpu_seconds"), "peak_rss_bytes": row.get("peak_rss_bytes")},
                    "artifacts": {"checkpoint_sha256": _sha256(path), "protocol_sha256": config.get("protocol_sha256"), "statistical_module_sha256": config.get("statistical_module_sha256"), "script_sha256": config.get("script_sha256")},
                    "missing_evidence": ["Complete independent-validation phase and acceptance-gate review."],
                })
        elif checkpoint.get("status") == "IN_PROGRESS":
            records.append({
                "record_type": "v1_2_independent_validation_progress",
                "experiment_id": f"v1.2-validation-progress:{hashlib.sha256(scenario.encode()).hexdigest()[:20]}",
                "iteration_number": 4,
                "hypothesis": "Run the locked independent-validation scenario with the frozen validation seed and resumable checkpoints.",
                "candidate_id": None,
                "model_id": None,
                "dataset": {"kind": "synthetic_monte_carlo", "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope")},
                "training_configuration": {"replications_requested": config.get("reps"), "bootstrap_inner_draws": config.get("bootstrap_inner"), "seed": config.get("seed"), "selection_manifest_sha256": _sha256(STAT / "selection" / "selection_manifest.json")},
                "metrics": {"completed_reps": progress.get("completed_reps")},
                "run_status": "RUNNING_INDEPENDENT_VALIDATION",
                "decision": "INCONCLUSIVE",
                "decision_reason": "Partial validation replications are progress only; aggregate method metrics and release gates are withheld until scenario completion.",
                "artifacts": {"checkpoint_sha256": _sha256(path), "config_sha256": checkpoint.get("config_sha256")},
                "missing_evidence": ["Scenario completion and complete independent-validation phase."],
            })

    selection_manifest_path = STAT / "selection" / "selection_manifest.json"
    selection_decisions_path = STAT / "selection" / "selection_decisions.json"
    phase_manifest_path = STAT / "validation" / "calibration_manifest.json"
    records.append({
        "record_type": "v1_2_independent_validation_phase",
        "experiment_id": "v1.2-statistical-independent-validation-phase",
        "iteration_number": 4,
        "hypothesis": "Independently validate preselected methods using the frozen validation seed and acceptance gates.",
        "candidate_id": None,
        "model_id": None,
        "run_status": validation_status,
        "decision": "INCONCLUSIVE",
        "decision_reason": "Validation status is based on the phase checkpoints and manifest; scenario progress is not a final method-regime decision.",
        "artifacts": {
            "protocol_sha256": _sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
            "selection_manifest_sha256": _sha256(selection_manifest_path) if selection_manifest_path.is_file() else None,
            "selection_decisions_sha256": _sha256(selection_decisions_path) if selection_decisions_path.is_file() else None,
            "validation_phase_manifest_sha256": _sha256(phase_manifest_path) if phase_manifest_path.is_file() else None,
        },
        "missing_evidence": ["Complete independent-validation phase and acceptance-gate review."] if validation_status != "COMPLETED_PHASE_GATE_REVIEW_PENDING" else ["Acceptance-gate review and real-series inference release decision."],
    })
    return records


def _native_grid_collocation_record() -> dict[str, Any] | None:
    if not COLLOCATION_SUMMARY_PATH.is_file() or not COLLOCATION_VALIDATION_PATH.is_file():
        return None
    summary = _read_json(COLLOCATION_SUMMARY_PATH)
    validation = _read_json(COLLOCATION_VALIDATION_PATH)
    if summary.get("status") != "VALIDATED" or validation.get("status") != "PASSED":
        raise ValueError("External native-grid collocation lacks a passing validation artifact")
    summary_sha = _sha256(COLLOCATION_SUMMARY_PATH)
    validation_sha = _sha256(COLLOCATION_VALIDATION_PATH)
    if validation.get("summary_sha256") != summary_sha:
        raise ValueError("External collocation validation does not match its summary checksum")
    return {
        "record_type": "external_point_to_grid_assessment",
        "experiment_id": "v1.2-external-merra2-native-grid-collocation-2002-2024",
        "iteration_number": 7,
        "hypothesis": "Describe agreement between eligible station-month observations and the nearest native MERRA-2 grid-cell center without claiming independent ground truth.",
        "candidate_id": None,
        "model_id": "MERRA-2-native-grid-cell-center",
        "dataset": {
            "dataset_id": summary.get("dataset_id"),
            "region_id": summary.get("region_id"),
            "period_start": summary.get("period_start"),
            "period_end": summary.get("period_end"),
            "native_grid_shape_lat_lon": summary.get("grid_shape_lat_lon"),
            "station_sources": sorted((summary.get("inputs", {}).get("station_tables") or {}).keys()),
        },
        "metrics_by_source": {
            source: {key: values.get(key) for key in (
                "matched_station_months", "matched_months", "matched_stations", "unique_grid_cells",
                "mae_c", "rmse_c", "bias_c", "pearson_r", "mean_cell_distance_km", "p95_cell_distance_km",
            )}
            for source, values in (summary.get("inputs", {}).get("station_tables") or {}).items()
        },
        "run_status": "VALIDATED_DESCRIPTIVE_POINT_TO_GRID_AGREEMENT",
        "decision": "INCONCLUSIVE",
        "decision_reason": "Nearest-grid-cell comparisons are descriptive; reanalysis may assimilate stations, spatial support and elevation differ, and INAMHI only covers 2002–2012.",
        "runtime_resources": None,
        "artifacts": {
            "summary_path": COLLOCATION_SUMMARY_PATH.relative_to(ROOT).as_posix(),
            "summary_sha256": summary_sha,
            "validation_path": COLLOCATION_VALIDATION_PATH.relative_to(ROOT).as_posix(),
            "validation_sha256": validation_sha,
            "station_monthly_detail_sha256": summary["outputs"]["station_monthly_collocation"]["sha256"],
            "monthly_summary_sha256": summary["outputs"]["monthly_summary"]["sha256"],
        },
        "missing_evidence": [
            "Independent reference status is not established for these station records.",
            "No elevation correction or grid regridding was applied.",
        ],
        "scope_note": "CEDIA-only station details remain outside Git; this public record contains aggregates and checksums only.",
    }


def _descriptive_trend_record() -> dict[str, Any] | None:
    if not TREND_ESTIMATES_PATH.is_file() or not TREND_MANIFEST_PATH.is_file():
        return None
    manifest = _read_json(TREND_MANIFEST_PATH)
    if manifest.get("status") != "DESCRIPTIVE_ESTIMATES_COMPLETE_INFERENCE_WITHHELD":
        raise ValueError("Real-series trend artifact has an unexpected inference status")
    if manifest.get("output", {}).get("sha256") != _sha256(TREND_ESTIMATES_PATH):
        raise ValueError("Real-series trend CSV does not match its manifest checksum")
    frame = pd.read_csv(TREND_ESTIMATES_PATH)
    if len(frame) != int(manifest.get("rows", -1)) or frame["p_value"].notna().any() or frame["confidence_interval_95"].notna().any():
        raise ValueError("Real-series trend output count or withheld-inference contract failed")
    return {
        "record_type": "real_series_descriptive_trend_summary",
        "experiment_id": "v1.2-ecuador-gold-descriptive-trends-2002-2024",
        "iteration_number": 0,
        "hypothesis": "Summarize predeclared descriptive slopes on the frozen Ecuador Gold series without making calibrated inference claims.",
        "candidate_id": None,
        "model_id": manifest.get("method"),
        "dataset": {
            "dataset_id": "KUNTUR_Ecuador_Gold_v1",
            "region_id": manifest.get("region_id"),
            "period_start": manifest.get("temporal_coverage", {}).get("start_month"),
            "period_end": manifest.get("temporal_coverage", {}).get("end_month"),
            "calendar_months": manifest.get("temporal_coverage", {}).get("calendar_months"),
            "gold_sha256": manifest.get("input", {}).get("sha256"),
        },
        "metrics_by_variable": {
            str(row["variable_id"]): {
                "units": str(row["units"]),
                "slope_per_decade": float(row["slope_per_decade"]),
                "slope_units": str(row["slope_units"]),
                "n_eligible_observations": int(row["n_eligible_observations"]),
                "calendar_coverage_fraction": float(row["calendar_coverage_fraction"]),
                "validity_status": str(row["validity_status"]),
                "inference_status": str(row["inference_status"]),
            }
            for row in frame.to_dict(orient="records")
        },
        "run_status": "DESCRIPTIVE_ESTIMATES_COMPLETE_INFERENCE_WITHHELD",
        "decision": "INCONCLUSIVE",
        "decision_reason": "The independent synthetic-method validation gate is incomplete; p-values, confidence intervals, and multiplicity-adjusted inference remain withheld.",
        "artifacts": {
            "estimates_path": TREND_ESTIMATES_PATH.relative_to(ROOT).as_posix(),
            "estimates_sha256": _sha256(TREND_ESTIMATES_PATH),
            "manifest_path": TREND_MANIFEST_PATH.relative_to(ROOT).as_posix(),
            "manifest_sha256": _sha256(TREND_MANIFEST_PATH),
        },
    }


def _statistical_csv_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    r4_dir = OPT / "round_04_inference_calibration"
    r4_manifest = r4_dir / "round_manifest.json"
    r4_csv = r4_dir / "inference_calibration.csv"
    if r4_manifest.is_file() and r4_csv.is_file():
        manifest = _read_json(r4_manifest)
        frame = pd.read_csv(r4_csv)
        csv_sha = _sha256(r4_csv)
        for source_row in frame.to_dict(orient="records"):
            row = {key: _json_value(value) for key, value in source_row.items()}
            rows.append({
                "phase": "round_04_exploratory_pilot",
                "source": "results/optimization/round_04_inference_calibration/inference_calibration.csv",
                "scenario_id": f"{row.get('process')}|n={row.get('n_months')}|slope={row.get('true_slope_per_decade')}|{row.get('method')}",
                "process": row.get("process"), "n_months": row.get("n_months"), "true_slope_per_decade": row.get("true_slope_per_decade"),
                "method": row.get("method"), "method_configuration_json": None,
                "replications_requested": row.get("replications_requested"), "replications_completed": row.get("replications_valid"), "replications_valid": row.get("replications_valid"),
                "seed": row.get("seed"), "coverage_95": row.get("coverage_95"), "coverage_mc95_low": row.get("coverage_mc95_low"), "coverage_mc95_high": row.get("coverage_mc95_high"),
                "type1_fpr_005": row.get("type1_fpr_005"), "type1_mc95_low": row.get("type1_mc95_low"), "type1_mc95_high": row.get("type1_mc95_high"),
                "power_005": row.get("power_005"), "power_mc95_low": row.get("power_mc95_low"), "power_mc95_high": row.get("power_mc95_high"),
                "slope_bias_per_decade": row.get("slope_bias_per_decade"), "slope_rmse_per_decade": row.get("slope_rmse_per_decade"),
                "mean_ci_width_per_decade": row.get("mean_ci_width_per_decade"), "convergence_failures": row.get("convergence_failures"),
                "status": "COMPLETED_EXPLORATORY_NOT_FINAL_VALIDATION", "decision": "INCONCLUSIVE",
                "round_manifest_sha256": _sha256(r4_manifest), "source_table_sha256": csv_sha, "checkpoint_sha256": None,
                "note": row.get("note"),
            })

    selection_dir = STAT / "selection" / "checkpoints"
    active_statuses = []
    if selection_dir.is_dir():
        for path in sorted(selection_dir.glob("selection_*.json")):
            checkpoint = _read_json(path)
            config = checkpoint.get("config") or {}
            progress = checkpoint.get("progress") or {}
            if checkpoint.get("status") == "COMPLETE" and isinstance(checkpoint.get("rows"), list):
                method_specs = {item.get("name"): item for item in config.get("methods", [])}
                for row in checkpoint["rows"]:
                    method_name = row.get("method")
                    rows.append({
                        "phase": "v1_2_selection",
                        "source": "results/statistical_correction/selection/checkpoints",
                        "scenario_id": f"{config.get('process')}|n={config.get('n_months')}|slope={config.get('slope')}|{method_name}",
                        "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope"),
                        "method": method_name, "method_configuration_json": json.dumps(method_specs.get(method_name), sort_keys=True, separators=(",", ":")),
                        "replications_requested": row.get("replications_requested"), "replications_completed": row.get("replications_valid"), "replications_valid": row.get("replications_valid"),
                        "seed": row.get("seed"), "coverage_95": row.get("coverage_95"), "coverage_mc95_low": row.get("coverage_mc95_low"), "coverage_mc95_high": row.get("coverage_mc95_high"),
                        "type1_fpr_005": row.get("type1_fpr_005"), "type1_mc95_low": row.get("type1_mc95_low"), "type1_mc95_high": row.get("type1_mc95_high"),
                        "power_005": row.get("power_005"), "power_mc95_low": row.get("power_mc95_low"), "power_mc95_high": row.get("power_mc95_high"),
                        "slope_bias_per_decade": row.get("slope_bias_per_decade"), "slope_rmse_per_decade": row.get("slope_rmse_per_decade"),
                        "mean_ci_width_per_decade": row.get("mean_ci_width_per_decade"), "convergence_failures": row.get("convergence_failure_rate"),
                        "status": row.get("validity_status") or "SELECTION_ONLY_NOT_VALIDATED", "decision": "INCONCLUSIVE",
                        "round_manifest_sha256": None, "source_table_sha256": None, "checkpoint_sha256": _sha256(path),
                        "note": "Selection-stage simulation only; not independent validation.",
                    })
            elif checkpoint.get("status") == "IN_PROGRESS":
                requested = config.get("reps")
                completed = progress.get("completed_reps")
                active_statuses.append({"process": config.get("process"), "n_months": config.get("n_months"), "slope": config.get("slope"), "completed": completed, "requested": requested})
                rows.append({
                    "phase": "v1_2_selection", "source": "results/statistical_correction/selection/checkpoints",
                    "scenario_id": f"{config.get('process')}|n={config.get('n_months')}|slope={config.get('slope')}",
                    "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope"),
                    "method": None, "method_configuration_json": json.dumps(config.get("methods"), sort_keys=True, separators=(",", ":")),
                    "replications_requested": requested, "replications_completed": completed, "replications_valid": None,
                    "seed": config.get("seed"), "coverage_95": None, "coverage_mc95_low": None, "coverage_mc95_high": None,
                    "type1_fpr_005": None, "type1_mc95_low": None, "type1_mc95_high": None,
                    "power_005": None, "power_mc95_low": None, "power_mc95_high": None,
                    "slope_bias_per_decade": None, "slope_rmse_per_decade": None, "mean_ci_width_per_decade": None, "convergence_failures": None,
                    "status": "RUNNING_INCOMPLETE", "decision": "INCONCLUSIVE",
                    "round_manifest_sha256": None, "source_table_sha256": None, "checkpoint_sha256": _sha256(path),
                    "note": "Aggregate method metrics are withheld until this scenario reaches its declared replication count.",
                })
    if not any(row["phase"] == "v1_2_selection" for row in rows):
        rows.append({
            "phase": "v1_2_selection", "source": "results/statistical_correction/selection/checkpoints", "scenario_id": "selection_checkpoints_not_found",
            "status": "NOT_STARTED_OR_NO_CHECKPOINT", "decision": "INCONCLUSIVE", "note": "No selection checkpoint is available.",
        })
    validation_status = _validation_phase_status()
    for path, checkpoint in _validation_checkpoints():
        config = checkpoint.get("config") or {}
        progress = checkpoint.get("progress") or {}
        scenario = f"{config.get('process')}|n={config.get('n_months')}|slope={config.get('slope')}"
        if checkpoint.get("status") == "COMPLETE" and isinstance(checkpoint.get("rows"), list):
            method_specs = {item.get("name"): item for item in config.get("methods", [])}
            for result in checkpoint["rows"]:
                rows.append({
                    "phase": "v1_2_independent_validation",
                    "source": "results/statistical_correction/validation/checkpoints",
                    "scenario_id": f"{scenario}|{result.get('method')}",
                    "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope"),
                    "method": result.get("method"), "method_configuration_json": json.dumps(method_specs.get(result.get("method")), sort_keys=True, separators=(",", ":")),
                    "replications_requested": result.get("replications_requested"), "replications_completed": result.get("replications_valid"), "replications_valid": result.get("replications_valid"),
                    "seed": result.get("seed"), "coverage_95": result.get("coverage_95"), "coverage_mc95_low": result.get("coverage_mc95_low"), "coverage_mc95_high": result.get("coverage_mc95_high"),
                    "type1_fpr_005": result.get("type1_fpr_005"), "type1_mc95_low": result.get("type1_mc95_low"), "type1_mc95_high": result.get("type1_mc95_high"),
                    "power_005": result.get("power_005"), "power_mc95_low": result.get("power_mc95_low"), "power_mc95_high": result.get("power_mc95_high"),
                    "slope_bias_per_decade": result.get("slope_bias_per_decade"), "slope_rmse_per_decade": result.get("slope_rmse_per_decade"),
                    "mean_ci_width_per_decade": result.get("mean_ci_width_per_decade"), "convergence_failures": result.get("convergence_failure_rate"),
                    "status": result.get("validity_status") or "VALIDATION_SCENARIO_COMPLETE", "decision": "INCONCLUSIVE",
                    "round_manifest_sha256": None, "source_table_sha256": None, "checkpoint_sha256": _sha256(path),
                    "note": "Validation scenario is complete; final release gates await completion of the full independent-validation phase.",
                })
        elif checkpoint.get("status") == "IN_PROGRESS":
            rows.append({
                "phase": "v1_2_independent_validation", "source": "results/statistical_correction/validation/checkpoints",
                "scenario_id": scenario, "process": config.get("process"), "n_months": config.get("n_months"), "true_slope_per_decade": config.get("slope"),
                "method": None, "method_configuration_json": json.dumps(config.get("methods"), sort_keys=True, separators=(",", ":")),
                "replications_requested": config.get("reps"), "replications_completed": progress.get("completed_reps"), "replications_valid": None,
                "seed": config.get("seed"), "coverage_95": None, "coverage_mc95_low": None, "coverage_mc95_high": None,
                "type1_fpr_005": None, "type1_mc95_low": None, "type1_mc95_high": None,
                "power_005": None, "power_mc95_low": None, "power_mc95_high": None,
                "slope_bias_per_decade": None, "slope_rmse_per_decade": None, "mean_ci_width_per_decade": None, "convergence_failures": None,
                "status": "RUNNING_INCOMPLETE", "decision": "INCONCLUSIVE",
                "round_manifest_sha256": None, "source_table_sha256": None, "checkpoint_sha256": _sha256(path),
                "note": "Partial independent-validation replications are progress only; aggregate metrics and release gates are withheld.",
            })
    rows.append({
        "phase": "v1_2_independent_validation", "source": "docs/protocols/statistical_correction_v1_2.md", "scenario_id": "independent_validation_phase",
        "process": None, "n_months": None, "true_slope_per_decade": None, "method": None, "method_configuration_json": None,
        "replications_requested": None, "replications_completed": None, "replications_valid": None, "seed": 20261127,
        "coverage_95": None, "coverage_mc95_low": None, "coverage_mc95_high": None, "type1_fpr_005": None, "type1_mc95_low": None, "type1_mc95_high": None,
        "power_005": None, "power_mc95_low": None, "power_mc95_high": None, "slope_bias_per_decade": None, "slope_rmse_per_decade": None,
        "mean_ci_width_per_decade": None, "convergence_failures": None,
        "status": "RUNNING_INCOMPLETE" if validation_status == "RUNNING_INDEPENDENT_VALIDATION" else validation_status,
        "decision": "INCONCLUSIVE",
        "round_manifest_sha256": None, "source_table_sha256": None, "checkpoint_sha256": None,
        "note": "Current validation progress is not a final method-regime decision; inferential real-series output remains withheld.",
    })
    return rows


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def _update_dashboard(calibration_rows: list[dict[str, Any]]) -> None:
    dashboard_path = REPORTS / "metric_dashboard.html"
    if not dashboard_path.is_file():
        return
    text = dashboard_path.read_text(encoding="utf-8")
    section_start = "<!-- KUNTUR_V12_CALIBRATION_STATUS_START -->"
    section_end = "<!-- KUNTUR_V12_CALIBRATION_STATUS_END -->"
    if section_start in text and section_end in text:
        prefix = text.split(section_start, 1)[0]
        suffix = text.split(section_end, 1)[1]
        text = prefix + suffix
    latest_progress = [row for row in calibration_rows if row.get("phase") == "v1_2_selection" and row.get("status") == "RUNNING_INCOMPLETE"]
    validation_progress = [row for row in calibration_rows if row.get("phase") == "v1_2_independent_validation" and row.get("status") == "RUNNING_INCOMPLETE" and row.get("replications_requested")]
    complete_scenarios = [row for row in calibration_rows if row.get("phase") == "v1_2_selection" and row.get("status") == "SELECTION_ONLY_NOT_VALIDATED"]
    if validation_progress:
        row = validation_progress[-1]
        progress_text = f"{row.get('replications_completed')} / {row.get('replications_requested')} outer replicates for {row.get('scenario_id')}."
        state = "Independent validation running under the frozen selection; partial replicates are not final results."
    elif latest_progress:
        row = latest_progress[-1]
        progress_text = f"{row.get('replications_completed')} / {row.get('replications_requested')} outer replicates for {row.get('scenario_id')}."
        state = "Selection running; scenario-level metrics withheld until completion."
    else:
        progress_text = f"{len(complete_scenarios)} method-scenario aggregates are selection-only."
        state = "No independent final-validation output is currently recorded."
    section = (
        section_start
        + "<section class='card' id='v12-calibration-status'><h2>v1.2 statistical-calibration gate</h2>"
        + f"<p>{state} {progress_text}</p>"
        + "<p>Selection results are not independent validation. Environmental p-values, confidence intervals, and q-values remain withheld until a frozen method-regime entry passes the independent calibration gates.</p>"
        + "<p><code>results/optimization/statistical_calibration.csv</code></p></section>"
        + section_end
    )
    insertion = text.lower().rfind("</body>")
    if insertion < 0:
        raise ValueError("The local metric dashboard is missing its closing body tag")
    text = text[:insertion] + section + text[insertion:]
    _atomic_write(dashboard_path, text.encode("utf-8"))


def _update_external_collocation_dashboard() -> None:
    dashboard_path = REPORTS / "metric_dashboard.html"
    if not dashboard_path.is_file() or not COLLOCATION_SUMMARY_PATH.is_file() or not COLLOCATION_VALIDATION_PATH.is_file():
        return
    summary = _read_json(COLLOCATION_SUMMARY_PATH)
    validation = _read_json(COLLOCATION_VALIDATION_PATH)
    if summary.get("status") != "VALIDATED" or validation.get("status") != "PASSED":
        raise ValueError("External collocation dashboard requires passing validation artifacts")
    text = dashboard_path.read_text(encoding="utf-8")
    start = "<!-- KUNTUR_V12_EXTERNAL_COLLOCATION_START -->"
    end = "<!-- KUNTUR_V12_EXTERNAL_COLLOCATION_END -->"
    if start in text and end in text:
        text = text.split(start, 1)[0] + text.split(end, 1)[1]
    table_rows = []
    for source, metrics in (validation.get("sources") or {}).items():
        table_rows.append(
            "<tr>" + "".join(f"<td>{value}</td>" for value in (
                html.escape(source),
                str(metrics["matched_station_months"]),
                str(metrics["matched_months"]),
                f"{metrics['mae_c']:.3f}",
                f"{metrics['rmse_c']:.3f}",
                f"{metrics['bias_c']:+.3f}",
                f"{metrics['pearson_r']:.3f}",
                f"{metrics['mean_cell_distance_km']:.1f}",
            )) + "</tr>"
        )
    section = (
        start
        + "<section class='card' id='v12-external-collocation'><h2>External point-to-grid comparison</h2>"
        + "<p>Validated 2002–2024 monthly station comparisons to nearest MERRA-2 native-grid cell centers; no regridding or elevation adjustment.</p>"
        + "<table><thead><tr><th>Source</th><th>Pairs</th><th>Months</th><th>MAE °C</th><th>RMSE °C</th><th>Bias °C</th><th>Pearson r</th><th>Mean distance km</th></tr></thead><tbody>"
        + "".join(table_rows)
        + "</tbody></table><p>Descriptive agreement only: MERRA-2 may assimilate station observations; spatial support differs; INAMHI ends in 2012. Station-level tables remain in the private CEDIA workspace.</p>"
        + "<p><code>results/optimization/native_grid_collocation_summary.json</code> · <code>results/optimization/native_grid_collocation_validation.json</code></p></section>"
        + end
    )
    insertion = text.lower().rfind("</body>")
    if insertion < 0:
        raise ValueError("The local metric dashboard is missing its closing body tag")
    _atomic_write(dashboard_path, (text[:insertion] + section + text[insertion:]).encode("utf-8"))


def main() -> None:
    OPT.mkdir(parents=True, exist_ok=True)
    registry = _candidate_rows() + _calibration_records()
    external_record = _native_grid_collocation_record()
    if external_record is not None:
        registry.append(external_record)
    trend_record = _descriptive_trend_record()
    if trend_record is not None:
        registry.append(trend_record)
    calibration_rows = _statistical_csv_rows()
    registry_bytes = ("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in registry)).encode("utf-8")
    calibration_frame = pd.DataFrame(calibration_rows)
    calibration_bytes = calibration_frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    _atomic_write(REGISTRY_PATH, registry_bytes)
    _atomic_write(CALIBRATION_PATH, calibration_bytes)
    _update_dashboard(calibration_rows)
    _update_external_collocation_dashboard()

    model_comparison_path = OPT / "model_comparison.csv"
    history_path = OPT / "metric_history.parquet"
    selection_progress = [row for row in calibration_rows if row.get("phase") == "v1_2_selection" and row.get("status") == "RUNNING_INCOMPLETE"]
    validation_progress = [row for row in calibration_rows if row.get("phase") == "v1_2_independent_validation" and row.get("status") == "RUNNING_INCOMPLETE" and row.get("replications_requested")]
    validation_status = _validation_phase_status()
    if selection_progress:
        release_status = "AGGREGATES_BUILT_STATISTICAL_SELECTION_INCOMPLETE"
    elif validation_status == "RUNNING_INDEPENDENT_VALIDATION":
        release_status = "AGGREGATES_BUILT_INDEPENDENT_VALIDATION_INCOMPLETE"
    elif validation_status == "COMPLETED_PHASE_GATE_REVIEW_PENDING":
        release_status = "AGGREGATES_BUILT_VALIDATION_PHASE_COMPLETE_GATE_REVIEW_INCOMPLETE"
    elif validation_status == "NOT_STARTED_SELECTION_FROZEN":
        release_status = "AGGREGATES_BUILT_INDEPENDENT_VALIDATION_NOT_STARTED"
    else:
        release_status = "AGGREGATES_BUILT_STATISTICAL_SELECTION_INCOMPLETE"
    selection_manifest_path = STAT / "selection" / "selection_manifest.json"
    selection_decisions_path = STAT / "selection" / "selection_decisions.json"
    validation_manifest_path = STAT / "validation" / "calibration_manifest.json"
    manifest = {
        "manifest_version": "kuntur-phase2-v12-release-artifacts-0.1.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": release_status,
        "scope": "Aggregate candidate metrics, experiment provenance, and calibration scenario summaries only; raw environmental observations, prediction rows, and model binaries are excluded.",
        "candidate_registry_rows": len([row for row in registry if row.get("record_type") == "predictive_candidate"]),
        "experiment_registry_rows": len(registry),
        "statistical_calibration_rows": len(calibration_rows),
        "current_selection_progress": selection_progress,
        "selection_status": "RUNNING_SELECTION_INCOMPLETE" if selection_progress else "SELECTION_COMPLETE_FROZEN" if selection_manifest_path.is_file() and _read_json(selection_manifest_path).get("status") == "FROZEN_BEFORE_INDEPENDENT_VALIDATION" else "SELECTION_NOT_FROZEN",
        "selection_manifest_sha256": _sha256(selection_manifest_path) if selection_manifest_path.is_file() else None,
        "selection_decisions_sha256": _sha256(selection_decisions_path) if selection_decisions_path.is_file() else None,
        "current_validation_progress": validation_progress,
        "independent_validation_status": validation_status,
        "validation_phase_manifest_sha256": _sha256(validation_manifest_path) if validation_manifest_path.is_file() else None,
        "inputs": {
            "round_01_manifest_sha256": _sha256(OPT / "round_01_architecture_search" / "round_manifest.json"),
            "round_02_manifest_sha256": _sha256(OPT / "round_02_feature_ablation" / "round_manifest.json"),
            "round_03_manifest_sha256": _sha256(OPT / "round_03_augmentation_robustness" / "round_manifest.json"),
            "round_04_manifest_sha256": _sha256(OPT / "round_04_inference_calibration" / "round_manifest.json"),
            "round_05_manifest_sha256": _sha256(OPT / "round_05_interval_calibration" / "round_manifest.json"),
            "round_06_score_sha256": _sha256(OPT / "round_06_2025_test_score" / "sealed_2025_score.json"),
            "statistical_protocol_sha256": _sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
            "native_grid_collocation_summary_sha256": _sha256(COLLOCATION_SUMMARY_PATH) if COLLOCATION_SUMMARY_PATH.is_file() else None,
            "native_grid_collocation_validation_sha256": _sha256(COLLOCATION_VALIDATION_PATH) if COLLOCATION_VALIDATION_PATH.is_file() else None,
            "descriptive_trend_estimates_sha256": _sha256(TREND_ESTIMATES_PATH) if TREND_ESTIMATES_PATH.is_file() else None,
            "descriptive_trend_manifest_sha256": _sha256(TREND_MANIFEST_PATH) if TREND_MANIFEST_PATH.is_file() else None,
        },
        "outputs": {
            "experiment_registry": {"path": REGISTRY_PATH.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(registry_bytes).hexdigest(), "bytes": len(registry_bytes)},
            "statistical_calibration": {"path": CALIBRATION_PATH.relative_to(ROOT).as_posix(), "sha256": hashlib.sha256(calibration_bytes).hexdigest(), "bytes": len(calibration_bytes)},
            "model_comparison": {"path": model_comparison_path.relative_to(ROOT).as_posix(), "sha256": _sha256(model_comparison_path) if model_comparison_path.is_file() else None},
            "metric_history": {"path": history_path.relative_to(ROOT).as_posix(), "sha256": _sha256(history_path) if history_path.is_file() else None},
            "metric_dashboard": {"path": (REPORTS / "metric_dashboard.html").relative_to(ROOT).as_posix(), "sha256": _sha256(REPORTS / "metric_dashboard.html") if (REPORTS / "metric_dashboard.html").is_file() else None},
            "native_grid_collocation_summary": {"path": COLLOCATION_SUMMARY_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(COLLOCATION_SUMMARY_PATH), "bytes": COLLOCATION_SUMMARY_PATH.stat().st_size} if COLLOCATION_SUMMARY_PATH.is_file() else None,
            "native_grid_collocation_validation": {"path": COLLOCATION_VALIDATION_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(COLLOCATION_VALIDATION_PATH), "bytes": COLLOCATION_VALIDATION_PATH.stat().st_size} if COLLOCATION_VALIDATION_PATH.is_file() else None,
            "descriptive_trend_estimates": {"path": TREND_ESTIMATES_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(TREND_ESTIMATES_PATH), "bytes": TREND_ESTIMATES_PATH.stat().st_size} if TREND_ESTIMATES_PATH.is_file() else None,
            "descriptive_trend_manifest": {"path": TREND_MANIFEST_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(TREND_MANIFEST_PATH), "bytes": TREND_MANIFEST_PATH.stat().st_size} if TREND_MANIFEST_PATH.is_file() else None,
            "noise_robustness": {"path": NOISE_ROBUSTNESS_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(NOISE_ROBUSTNESS_PATH), "bytes": NOISE_ROBUSTNESS_PATH.stat().st_size} if NOISE_ROBUSTNESS_PATH.is_file() else None,
            "sealed_2025_score": {"path": SEALED_2025_SCORE_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(SEALED_2025_SCORE_PATH), "bytes": SEALED_2025_SCORE_PATH.stat().st_size} if SEALED_2025_SCORE_PATH.is_file() else None,
            "frozen_test_manifest": {"path": FROZEN_TEST_MANIFEST_PATH.relative_to(ROOT).as_posix(), "sha256": _sha256(FROZEN_TEST_MANIFEST_PATH), "bytes": FROZEN_TEST_MANIFEST_PATH.stat().st_size} if FROZEN_TEST_MANIFEST_PATH.is_file() else None,
        },
        "software_versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "limitations": [
            "Completed synthetic R4 is a historical exploratory pilot and is not the frozen independent-validation phase.",
            "Selection-stage results are not independent validation.",
            "Independent-validation progress is not final method-regime evidence; the real-series engine withholds environmental p-values and intervals until acceptance gates are evaluated.",
            "Station comparisons are nearest-grid-cell descriptive checks, not independent ground truth or station-adjusted validation.",
        ],
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(ARTIFACT_MANIFEST_PATH, manifest_bytes)
    print(json.dumps({
        "status": manifest["status"],
        "predictive_candidate_rows": len([r for r in registry if r.get("record_type") == "predictive_candidate"]),
        "registry_rows": len(registry),
        "statistical_calibration_rows": len(calibration_rows),
        "registry_sha256": manifest["outputs"]["experiment_registry"]["sha256"],
        "statistical_calibration_sha256": manifest["outputs"]["statistical_calibration"]["sha256"],
        "manifest": ARTIFACT_MANIFEST_PATH.relative_to(ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
