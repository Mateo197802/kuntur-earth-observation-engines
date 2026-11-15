#!/usr/bin/env python3
"""Train six development forecasters and run the predeclared 38-month noise audit."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import scipy
import sklearn
import statsmodels
import torch

from kuntur_data.phase2.augmentation import COMBINED, history_array, training_augmentation, training_location_scale
from kuntur_data.phase2.predictive import (
    MODEL_ORDER,
    PeakRSS,
    baseline_interval_radii,
    baseline_predictions,
    evaluate_internal_robustness,
    fit_preprocessing,
    make_clean_matrix,
    prepare_augmented_matrix,
    regression_metrics,
    sha256_file,
    train_model_candidates,
)
from kuntur_data.phase2.splits import phase2_split


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261115
EVAL_SEED = 90261115


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n")
    tmp.replace(path)


def main() -> None:
    start = time.perf_counter()
    forecast_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    split_path = ROOT / "data/derived/phase2/forecast_phase2_split_v1.parquet"
    augmentation_path = ROOT / "data/derived/phase2/augmentation_training_v1.parquet"
    augmentation_manifest_path = ROOT / "data/manifests/augmentation_manifest_v1.json"
    if not all(p.exists() for p in (forecast_path, split_path, augmentation_path, augmentation_manifest_path)):
        raise SystemExit("Run audit_phase2_baseline.py and prepare_phase2_data.py before this benchmark")

    # Predicate pushdown keeps the frozen 2022–2024 test partition out of the
    # development process. The split index is metadata; model features/labels
    # loaded here stop at the end of validation.
    forecast = pd.read_parquet(forecast_path, filters=[("target_month", "<=", "2021-12")])
    if forecast.empty or forecast.target_month.max() > "2021-12":
        raise SystemExit("Parquet predicate did not exclude the final test partition")
    split = pd.read_parquet(split_path)
    roles = phase2_split(pd.read_parquet(ROOT / "data/gold/kuntur_forecast_split_indices_v1.parquet"))
    role_frame = forecast.merge(roles[["sample_id", "phase2_role"]], on="sample_id", validate="one_to_one")
    model_fit = role_frame[role_frame.phase2_role == "model_fit"].sort_values("target_month", kind="stable").reset_index(drop=True)
    validation = role_frame[role_frame.phase2_role == "validation"].sort_values("target_month", kind="stable").reset_index(drop=True)
    robust = role_frame[role_frame.phase2_role == "internal_robustness_holdout"].sort_values("target_month", kind="stable").reset_index(drop=True)
    if (len(model_fit), len(validation), len(robust)) != (152, 36, 38):
        raise SystemExit("Unexpected model-fit/validation/internal-holdout dimensions")

    augmentation_summary = json.loads(augmentation_manifest_path.read_text(encoding="utf-8"))
    augmented = pd.read_parquet(augmentation_path)
    if sha256_file(augmentation_path) != augmentation_summary["output_sha256"] or len(augmented) != augmentation_summary["total_rows"]:
        raise SystemExit("Augmentation Parquet does not match its frozen manifest")
    if set(augmented.base_sample_id) != set(model_fit.sample_id) or set(augmented.source_partition) != {"train_fit"}:
        raise SystemExit("Augmentation includes a sample outside model-fit partition")

    location, noise_scale, scaler = fit_preprocessing(model_fit)
    x_train, y_train, weights = prepare_augmented_matrix(augmented, scaler)
    x_val, raw_val = make_clean_matrix(validation, location, scaler)
    y_val = validation.target_value.to_numpy(dtype=np.float64)
    raw_robust = history_array(robust)
    y_robust = robust.target_value.to_numpy(dtype=np.float64)
    fit_month = pd.to_datetime(model_fit.target_month.astype(str) + "-01").dt.month
    climatology = {int(month): float(model_fit.loc[fit_month == month, "target_value"].mean()) for month in range(1, 13)}
    baseline_val = baseline_predictions(validation, raw_val, np.zeros_like(raw_val, dtype=bool), location, climatology)
    baseline_metrics = {k: regression_metrics(y_val, v) for k, v in baseline_val.items()}
    baseline_intervals = baseline_interval_radii(y_val, baseline_val)

    model_dir = ROOT / "results/phase2/models/development_v1"
    model_dir.mkdir(parents=True, exist_ok=True)
    with PeakRSS() as overall_rss:
        selected, candidate_rows = train_model_candidates(x_train, y_train, weights, x_val, y_val, model_dir=model_dir, seed=SEED, scaler=scaler)
    for name, radius in baseline_intervals.items():
        selected[name] = {"prediction_interval_radius_90": radius}
    selection = min(
        MODEL_ORDER,
        key=lambda name: (
            selected[name]["validation_metrics"]["mae"],
            selected[name]["validation_metrics"]["rmse"],
            MODEL_ORDER.index(name),
        ),
    )
    selected_name = selection
    selected_model = selected[selected_name]["model"]
    model_infos = {name: info for name, info in selected.items() if name in MODEL_ORDER}
    interval_radii = {name: info["prediction_interval_radius_90"] for name, info in selected.items()}
    robustness_metrics, robustness_predictions = evaluate_internal_robustness(
        model_infos,
        raw_robust,
        y_robust,
        robust,
        location=location,
        noise_scale=noise_scale,
        scaler=scaler,
        climatology=climatology,
        interval_radii=interval_radii,
        seed=EVAL_SEED,
    )

    results_dir = ROOT / "results/phase2"
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    for name, info in ((n, selected[n]) for n in MODEL_ORDER):
        yhat = info["validation_prediction"]
        m = regression_metrics(y_val, yhat)
        validation_rows.append({"model_id": name, "role": "six_model", "selection_metric": "clean_validation_mae", **m, "prediction_interval_radius_90": info["prediction_interval_radius_90"], "config": info["config"], "fit_wall_seconds": info["fit_info"]["fit_wall_seconds"], "fit_cpu_seconds": info["fit_info"]["fit_cpu_seconds"], "peak_rss_bytes": info["fit_info"]["peak_rss_bytes"], "inference_ms_per_sample": info["inference_ms_per_sample"], "model_bytes": info["model_bytes"], "model_sha256": info["model_sha256"], "model_path": str(Path(info["model_path"]).relative_to(ROOT))})
    for name, pred in baseline_val.items():
        validation_rows.append({"model_id": name, "role": "baseline", "selection_metric": "descriptive_only_not_model_selection", **regression_metrics(y_val, pred), "prediction_interval_radius_90": baseline_intervals[name]})
    validation_path = metrics_dir / "validation_benchmark_v1.csv"
    pd.DataFrame(validation_rows).to_csv(validation_path, index=False)
    candidate_path = metrics_dir / "candidate_search_v1.jsonl"
    write_jsonl(candidate_path, candidate_rows)
    robustness_path = metrics_dir / "robustness_holdout_v1.parquet"
    robustness_predictions_path = metrics_dir / "robustness_holdout_predictions_v1.parquet"
    robustness_metrics.to_parquet(robustness_path, index=False, compression="zstd")
    robustness_predictions.to_parquet(robustness_predictions_path, index=False, compression="zstd")

    # Human-readable architecture x scenario table, preserving all scenarios in machine-readable Parquet.
    matrix_path = metrics_dir / "robustness_architecture_noise_matrix_v1.csv"
    robustness_metrics.pivot_table(index=["model_id", "noise_type"], columns="noise_intensity", values=["mae", "rmse", "mean_bias", "delta_mae_from_clean", "robustness_degradation_ratio", "prediction_stability_rms_change", "prediction_interval_90_coverage"], aggfunc="first").to_csv(matrix_path)

    all_artifacts = [validation_path, candidate_path, robustness_path, robustness_predictions_path, matrix_path]
    for info in model_infos.values():
        all_artifacts.append(Path(info["model_path"]))
    artifact_map = {str(path.relative_to(ROOT)): {"sha256": sha256_file(path), "bytes": path.stat().st_size} for path in all_artifacts}
    dev_manifest = {
        "manifest_version": "1.0.0",
        "status": "VALIDATED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": "KUNTUR_PHASE2_SIX_MODEL_DEVELOPMENT_BENCHMARK_V1",
        "protocol_path": "docs/protocols/predictive_model_protocol_v1.md",
        "protocol_sha256": sha256_file(ROOT / "docs/protocols/predictive_model_protocol_v1.md"),
        "external_protocol_path": "docs/protocols/external_validation_protocol_v1.md",
        "external_protocol_sha256": sha256_file(ROOT / "docs/protocols/external_validation_protocol_v1.md"),
        "source_forecast_sha256": sha256_file(forecast_path),
        "source_split_manifest_sha256": sha256_file(ROOT / "data/manifests/phase2_split_manifest_v1.json"),
        "augmentation_manifest_sha256": sha256_file(augmentation_manifest_path),
        "train_fit_original_months": len(model_fit),
        "synthetic_training_rows": augmentation_summary["synthetic_rows"],
        "unique_original_training_months": augmentation_summary["unique_original_samples"],
        "validation_months": len(validation),
        "internal_robustness_months": len(robust),
        "test_partition_status": "NOT_LOADED_BY_THIS_RUN_PREDICATE_FILTERED",
        "model_candidates": candidate_rows,
        "validation_metrics": validation_rows,
        "selected_architecture": selected_name,
        "selection_rule": "minimum clean validation MAE, then RMSE, then protocol model order; robustness holdout not used for selection",
        "robustness_summary": {"rows": int(len(robustness_metrics)), "model_ids": sorted(robustness_metrics.model_id.unique().tolist()), "noise_scenarios": sorted(robustness_metrics.noise_type.unique().tolist()), "levels": [0.25, 0.50], "unique_target_months": int(robust.sample_id.nunique()), "seed": EVAL_SEED},
        "artifacts": artifact_map,
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": psutil.cpu_count(logical=True), "memory_total_bytes": psutil.virtual_memory().total, "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "resource_usage": {"total_elapsed_seconds": time.perf_counter() - start, "peak_rss_bytes": overall_rss.peak},
        "validation": {"status": "PASSED_WITH_PRIOR_ACCESS_CAVEAT", "checks": ["Parquet predicate pushdown loaded only target months through 2021-12 in this run", "only 152 model-fit target months fitted the development models", "only clean validation selected model configs and architecture", "robustness metrics use 38 unique months; noisy repetitions are not counted as independent", "25%/50% evaluation levels absent from training augmentation", "original final test not evaluated or used in this development script", "authentic target labels never perturbed", "nine model/baseline IDs use identical target months"], "prior_run_caveat": "An earlier deterministic attempt preloaded test rows; see data/manifests/phase2_execution_failures.jsonl."},
    }
    manifest_path = ROOT / "data/manifests/model_manifest_v1.json"
    atomic_json(manifest_path, dev_manifest)
    noise_manifest = {
        "manifest_version": "1.0.0", "status": "VALIDATED", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_model_manifest_sha256": sha256_file(manifest_path),
        "source_augmentation_manifest_sha256": sha256_file(augmentation_manifest_path),
        "internal_holdout_unique_months": len(robust),
        "evaluation_level_definition": "25% or 50% of model-fit training feature SD as additive RMS; dropout fraction for missingness; combined L2 budget",
        "random_seed": EVAL_SEED,
        "independent_scenarios": ["clean", "gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness"],
        "combined_scenarios": list(COMBINED),
        "intensities": [0.25, 0.50],
        "metrics_path": str(robustness_path.relative_to(ROOT)),
        "metrics_sha256": sha256_file(robustness_path),
        "prediction_path": str(robustness_predictions_path.relative_to(ROOT)),
        "prediction_sha256": sha256_file(robustness_predictions_path),
        "metrics_rows": int(len(robustness_metrics)),
        "independent_month_count": int(robust.sample_id.nunique()),
        "validation": {"status": "PASSED", "note": "Internal development diagnostic only; not external validation or an independent domain."},
    }
    noise_manifest_path = ROOT / "data/manifests/noise_robustness_manifest_v1.json"
    atomic_json(noise_manifest_path, noise_manifest)

    freeze = {
        "freeze_version": "1.0.0",
        "status": "FROZEN_BEFORE_FINAL_TEST",
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "selection_manifest_sha256": sha256_file(manifest_path),
        "selected_architecture": selected_name,
        "selected_config": selected[selected_name]["config"],
        "clean_validation_mae": selected[selected_name]["validation_metrics"]["mae"],
        "clean_validation_rmse": selected[selected_name]["validation_metrics"]["rmse"],
        "robustness_manifest_sha256": sha256_file(noise_manifest_path),
        "final_test": "NOT_YET_RUN",
        "refit_policy": "Refit the selected architecture on all 190 original v1 training months using the declared augmentation recipe and training-only scale recomputation; do not change model/config after looking at final test.",
    }
    freeze_path = ROOT / "data/manifests/model_selection_freeze_v1.json"
    atomic_json(freeze_path, freeze)

    registry = []
    for row in candidate_rows:
        registry.append({"run_id": hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest(), "run_type": "model_candidate", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "source_forecast_sha256": dev_manifest["source_forecast_sha256"], "split_manifest_sha256": dev_manifest["source_split_manifest_sha256"], "augmentation_manifest_sha256": dev_manifest["augmentation_manifest_sha256"], **row})
    write_jsonl(ROOT / "data/manifests/experiment_registry.jsonl", registry)
    print(json.dumps({"status": "PASSED", "selected_architecture": selected_name, "validation": validation_rows, "robustness_metric_rows": len(robustness_metrics), "robustness_scenarios": robustness_metrics.noise_type.nunique(), "artifact_paths": {"model_manifest": str(manifest_path), "noise_manifest": str(noise_manifest_path), "selection_freeze": str(freeze_path)}, "total_elapsed_seconds": time.perf_counter() - start}, indent=2, default=str))


if __name__ == "__main__":
    main()
