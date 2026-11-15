#!/usr/bin/env python3
"""Refit the frozen selected model on v1 train and evaluate the 2022–2024 set once."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.preprocessing import StandardScaler

from kuntur_data.phase2.augmentation import training_augmentation
from kuntur_data.phase2.predictive import (
    PeakRSS,
    TemporalCNN,
    baseline_predictions,
    fit_candidate,
    fit_preprocessing,
    make_clean_matrix,
    prepare_augmented_matrix,
    predict,
    regression_metrics,
    sha256_file,
)
from kuntur_data.phase2.splits import phase2_split


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261116


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_development_model(architecture: str, path: Path):
    if path.suffix == ".joblib":
        payload = joblib.load(path)
        return payload["estimator"], payload["scaler"], payload["config"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = TemporalCNN(channels=payload["config"]["channels"], seed=payload["seed"])
    model.model.load_state_dict(payload["state_dict"])
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(payload["scaler_mean"], dtype=float)
    scaler.scale_ = np.asarray(payload["scaler_scale"], dtype=float)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(scaler.mean_)
    scaler.n_samples_seen_ = 1
    return model, scaler, payload["config"]


def main() -> None:
    started = time.perf_counter()
    forecast_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    split_path = ROOT / "data/gold/kuntur_forecast_split_indices_v1.parquet"
    freeze_path = ROOT / "data/manifests/model_selection_freeze_v1.json"
    model_manifest_path = ROOT / "data/manifests/model_manifest_v1.json"
    access_path = ROOT / "data/manifests/final_test_access_v1.json"
    output_path = ROOT / "results/phase2/metrics/final_test_selected_model_v1.parquet"
    manifest_path = ROOT / "data/manifests/final_test_manifest_v1.json"
    if manifest_path.exists() or access_path.exists():
        raise SystemExit("Final test access has already been started; refusing a second test read/evaluation")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_FINAL_TEST" or freeze.get("final_test") != "NOT_YET_RUN":
        raise SystemExit("Model selection is not frozen for a first final test evaluation")
    if freeze.get("selection_manifest_sha256") != sha256_file(model_manifest_path):
        raise SystemExit("Frozen selection no longer matches the model manifest")
    architecture = freeze["selected_architecture"]
    if architecture != model_manifest.get("selected_architecture"):
        raise SystemExit("Selected architecture differs between the frozen and model manifests")
    split_roles = phase2_split(pd.read_parquet(split_path))
    role_map = split_roles.set_index("sample_id")["phase2_role"].to_dict()

    # The development rows are loaded with predicate pushdown and exclude every
    # 2022–2024 forecast record.
    development = pd.read_parquet(forecast_path, filters=[("target_month", "<=", "2021-12")])
    if development.target_month.max() != "2021-12" or (development.target_month >= "2022-01").any():
        raise SystemExit("Development read unexpectedly contains the final test period")
    development["phase2_role"] = development.sample_id.map(role_map)
    train = development[development.phase2_role.isin(["model_fit", "internal_robustness_holdout"])].sort_values("target_month", kind="stable").reset_index(drop=True)
    model_fit = train.iloc[:152].reset_index(drop=True)
    validation = development[development.phase2_role == "validation"].sort_values("target_month", kind="stable").reset_index(drop=True)
    if (len(train), len(model_fit), len(validation)) != (190, 152, 36):
        raise SystemExit("Unexpected train/model-fit/validation partition sizes")

    model_path_dev = ROOT / f"results/phase2/models/development_v1/{architecture}_development_v1.{ 'pt' if architecture == 'temporal_cnn' else 'joblib' }"
    dev_model, dev_scaler, _ = load_development_model(architecture, model_path_dev)
    dev_location, _, _ = fit_preprocessing(model_fit)
    x_validation, _ = make_clean_matrix(validation, dev_location, dev_scaler)
    val_predictions = predict(dev_model, architecture, x_validation)
    val_errors = np.abs(validation.target_value.to_numpy(dtype=float) - val_predictions)
    interval_radius = float(np.quantile(val_errors, 0.90, method="higher"))

    # Final refit uses all 190 authentic train samples, with augmentation and
    # scale estimation derived exclusively from those train histories.
    augmented, aug_state = training_augmentation(train, seed=20261115, source_partition="train_final_refit")
    synthetic_counts = augmented.loc[augmented.noise_type != "clean"].groupby("base_sample_id").size().to_dict()
    augmented["augmentation_weight"] = [1.0 if kind == "clean" else 1.0 / synthetic_counts[sid] for kind, sid in zip(augmented.noise_type, augmented.base_sample_id)]
    weight_sums = augmented.groupby("base_sample_id").augmentation_weight.sum()
    if not np.allclose(weight_sums.to_numpy(), 2.0, atol=1e-12):
        raise SystemExit("Final refit augmentation changed authentic/synthetic sample balance")
    location, noise_scale, scaler = fit_preprocessing(train)
    x_train, y_train, weights = prepare_augmented_matrix(augmented, scaler)
    x_val_final, _ = make_clean_matrix(validation, location, scaler)
    y_val_final = validation.target_value.to_numpy(dtype=float)
    with PeakRSS() as peak_rss:
        final_model, fit_info = fit_candidate(architecture, freeze["selected_config"], x_train, y_train, weights, x_val_final, y_val_final, seed=SEED)

    final_model_dir = ROOT / "results/phase2/models/final_v1"
    final_model_dir.mkdir(parents=True, exist_ok=True)
    final_model_path = final_model_dir / f"{architecture}_final_refit_v1.{ 'pt' if architecture == 'temporal_cnn' else 'joblib' }"
    if architecture == "temporal_cnn":
        torch.save({"architecture": architecture, "config": freeze["selected_config"], "seed": SEED, "state_dict": final_model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "feature_order": list(train.columns), "preprocessing_location": location.tolist(), "noise_scale": noise_scale.tolist()}, final_model_path)
    else:
        joblib.dump({"architecture": architecture, "config": freeze["selected_config"], "seed": SEED, "estimator": final_model, "scaler": scaler, "imputation_location": location, "noise_scale": noise_scale}, final_model_path, compress=3)
    augmented_path = ROOT / "data/derived/phase2/augmentation_final_refit_v1.parquet"
    augmented.to_parquet(augmented_path, index=False, compression="zstd")
    augmentation_jsonl = ROOT / "data/manifests/augmentation_final_refit_v1.jsonl"
    with augmentation_jsonl.open("w", encoding="utf-8", newline="\n") as stream:
        for row in augmented.to_dict("records"):
            record = {
                "base_sample_id": row["base_sample_id"], "augmentation_id": row["augmentation_id"],
                "noise_type": row["noise_type"], "noise_intensity": row["noise_intensity"],
                "noise_components": json.loads(row["noise_components"]), "is_combined_noise": bool(row["is_combined_noise"]),
                "random_seed": int(row["random_seed"]), "noise_parameters": json.loads(row["noise_parameters"]),
                "source_partition": row["source_partition"], "target_variable": row["target_variable"],
                "augmentation_version": row["augmentation_version"], "augmentation_weight": float(row["augmentation_weight"]),
                "target_value_sha256": hashlib.sha256(f"{float(row['target_value']):.12g}".encode()).hexdigest(),
                "status": "ACCEPTED",
            }
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    # Durable one-shot guard is written only after all non-test fitting and
    # artifact creation has succeeded.
    atomic_json(access_path, {"status": "TEST_READ_STARTED", "evaluation_count": 0, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "selection_freeze_sha256": sha256_file(freeze_path), "development_process_prior_access_caveat": "Earlier development runner preloaded test rows; see phase2_execution_failures.jsonl. No test metrics were used for selection."})

    # This is the only model-metric read of the 2022–2024 rows. Predicate
    # pushdown and explicit exact dates are asserted before scoring.
    test = pd.read_parquet(forecast_path, filters=[("target_month", ">=", "2022-01")])
    test = test.sort_values("target_month", kind="stable").reset_index(drop=True)
    if len(test) != 36 or test.target_month.iloc[0] != "2022-01" or test.target_month.iloc[-1] != "2024-12":
        atomic_json(access_path, {"status": "READ_BUT_VALIDATION_FAILED", "evaluation_count": 0, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "row_count": len(test), "min_target_month": test.target_month.min(), "max_target_month": test.target_month.max()})
        raise SystemExit("The final test rows failed the frozen date/count gate; no metrics calculated")
    x_test, raw_test = make_clean_matrix(test, location, scaler)
    y_test = test.target_value.to_numpy(dtype=float)
    predictions = {architecture: predict(final_model, architecture, x_test)}
    train_month = pd.to_datetime(train.target_month.astype(str) + "-01").dt.month
    climatology = {int(month): float(train.loc[train_month == month, "target_value"].mean()) for month in range(1, 13)}
    baselines = baseline_predictions(test, raw_test, np.zeros_like(raw_test, dtype=bool), location, climatology)
    predictions.update(baselines)
    metric_rows = []
    prediction_rows = []
    for model_id, pred in predictions.items():
        pred = np.asarray(pred, dtype=float)
        metrics = regression_metrics(y_test, pred)
        errors = pred - y_test
        abs_error = np.abs(errors)
        rng = np.random.default_rng(20261116 + len(metric_rows))
        bootstrap_indices = rng.integers(0, len(y_test), size=(10000, len(y_test)))
        mae_ci = np.quantile(abs_error[bootstrap_indices].mean(axis=1), [0.025, 0.975])
        rmse_ci = np.sqrt(np.quantile((errors[bootstrap_indices] ** 2).mean(axis=1), [0.025, 0.975]))
        radius = interval_radius
        row = {"model_id": model_id, "role": "frozen_final_selected_model" if model_id == architecture else "baseline", **metrics,
               "mae_ci95_month_bootstrap": mae_ci.tolist(), "rmse_ci95_month_bootstrap": rmse_ci.tolist(),
               "prediction_interval_90_radius_calibrated_validation": radius,
               "prediction_interval_90_empirical_test_coverage": float(np.mean(abs_error <= radius)),
               "independent_target_months": len(test), "final_test_once": True}
        if model_id == architecture:
            for baseline_id, baseline_pred in baselines.items():
                bmetrics = regression_metrics(y_test, baseline_pred)
                row[f"mae_skill_vs_{baseline_id}"] = 1.0 - metrics["mae"] / bmetrics["mae"] if bmetrics["mae"] else None
                row[f"rmse_skill_vs_{baseline_id}"] = 1.0 - metrics["rmse"] / bmetrics["rmse"] if bmetrics["rmse"] else None
        metric_rows.append(row)
        for i, target_month in enumerate(test.target_month.astype(str)):
            prediction_rows.append({"target_month": target_month, "model_id": model_id, "y_true_degC": float(y_test[i]), "y_pred_degC": float(pred[i]), "absolute_error_degC": float(abs_error[i]), "prediction_interval_low_degC": float(pred[i] - radius), "prediction_interval_high_degC": float(pred[i] + radius), "coverage_valid_fraction_bbox": float(test.iloc[i].target_valid_area_fraction_bbox)})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = ROOT / "results/phase2/metrics/final_test_metrics_v1.csv"
    pd.DataFrame(prediction_rows).to_parquet(output_path, index=False, compression="zstd")
    pd.DataFrame(metric_rows).to_csv(metrics_path, index=False)
    if len(test) != test.sample_id.nunique() or test.target_month.duplicated().any():
        raise SystemExit("Final test target-month uniqueness gate failed")
    augmented_path = ROOT / "data/derived/phase2/augmentation_final_refit_v1.parquet"
    artifacts = [final_model_path, augmented_path, augmentation_jsonl, output_path, metrics_path]
    artifact_map = {str(p.relative_to(ROOT)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in artifacts}
    manifest = {
        "manifest_version": "1.0.0", "status": "FINAL_TEST_EVALUATED_ONCE", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": "kuntur-phase2-final-test-20261116", "source_forecast_sha256": sha256_file(forecast_path),
        "source_split_sha256": sha256_file(split_path), "source_model_manifest_sha256": sha256_file(model_manifest_path),
        "selection_freeze_sha256": sha256_file(freeze_path), "selected_architecture": architecture,
        "selected_config": freeze["selected_config"], "final_refit_seed": SEED,
        "train_original_samples": len(train), "final_refit_augmented_rows": len(augmented), "unique_original_train_samples": int(augmented.base_sample_id.nunique()),
        "final_refit_rejected_synthetic_rows": len(aug_state["rejected"]),
        "final_refit_augmentation_weight_sum_per_original_min_max": [float(weight_sums.min()), float(weight_sums.max())],
        "validation_calibration_samples": len(validation), "validation_interval_radius_90_degC": interval_radius,
        "interval_calibration_caveat": "Validation also selected the model, so test interval coverage is descriptive and not a split-conformal guarantee.",
        "test_target_months": test.target_month.astype(str).tolist(), "test_samples": len(test), "test_read_count": 1,
        "test_access_caveat": "Earlier development code preloaded held-out rows before selection but did not fit, tune, select, predict, or report test metrics with them. This is recorded in phase2_execution_failures.jsonl; strict process blinding is not claimed.",
        "metrics": metric_rows, "fit_info": fit_info, "peak_rss_bytes_fit": int(peak_rss.peak),
        "artifacts": artifact_map, "validation": {"status": "PASSED_WITH_DISCLOSED_ACCESS_CAVEAT", "checks": ["frozen architecture/config used", "only original 190 train rows used for final refit and monthly climatology", "augmentation and feature scales fitted on training only", "validation used for 90% residual interval calibration only", "test predicate loaded exactly 36 months from 2022-01 to 2024-12", "test scored once after model selection and refit frozen", "only selected model plus fixed baselines scored on final test", "model selection not changed after test evaluation"]},
        "resource_usage": {"elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": int(peak_rss.peak)},
    }
    atomic_json(manifest_path, manifest)
    atomic_json(access_path, {"status": "EVALUATED_ONCE", "evaluation_count": 1, "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "manifest_sha256": sha256_file(manifest_path), "test_target_months": len(test)})
    print(json.dumps({"status": manifest["status"], "selected_architecture": architecture, "test_metrics": metric_rows, "test_months": len(test), "final_refit_augmented_rows": len(augmented), "model_sha256": artifact_map[str(final_model_path.relative_to(ROOT))]["sha256"], "prediction_path": str(output_path), "manifest_path": str(manifest_path), "test_access_count": 1, "elapsed_seconds": manifest["resource_usage"]["elapsed_seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
