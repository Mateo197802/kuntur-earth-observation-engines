#!/usr/bin/env python3
"""Development-fold 25%/50% perturbation audit for all nine forecast policies."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kuntur_data.phase2.augmentation import COMBINED, SINGLE_FAMILIES, history_array, perturb_sample
from kuntur_data.phase2.predictive import deterministic_seed, fit_preprocessing_v1_1, make_clean_matrix_v1_1, model_inputs, predict, regression_metrics
from scripts.run_phase2_v11_rolling_benchmark import (
    BASELINE_IDS,
    FOLDS,
    MODEL_CONFIGS,
    SEEDS,
    _assert_t3_policy,
    _baseline_predictions,
    _fit_model,
)


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
PROTOCOL = ROOT / "docs/protocols/predictive_model_protocol_v1_1.md"
OUT = ROOT / "results/phase2/v1_1/robustness"
SCENARIOS = tuple(SINGLE_FAMILIES) + tuple(COMBINED)
LEVELS = (0.25, 0.50)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    started = time.perf_counter()
    frame = pd.read_parquet(FORECAST, filters=[("target_month", "<=", "2018-12")])
    if frame.empty or frame.target_month.max() != "2018-12":
        raise SystemExit("Development-only data filter failed")
    if frame.sample_id.duplicated().any():
        raise SystemExit("Duplicate development sample IDs")
    _assert_t3_policy(frame)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in FOLDS:
        train = frame[frame.target_month <= fold["train_end"]].reset_index(drop=True)
        val = frame[(frame.target_month >= fold["validation_start"]) & (frame.target_month <= fold["validation_end"])].reset_index(drop=True)
        if len(val) != 36 or train.target_month.max() >= val.target_month.min():
            raise SystemExit(f"Invalid fold {fold['fold_id']}")
        location, noise_scale, scaler = fit_preprocessing_v1_1(train)
        raw_train = history_array(train)
        x_train = model_inputs(raw_train, ~np.isfinite(raw_train), location, scaler)
        raw_val = history_array(val)
        _, _, missing_val = make_clean_matrix_v1_1(val, location, scaler)
        y_train = train.target_value.to_numpy(float)
        y_val = val.target_value.to_numpy(float)
        model_predictions: dict[str, list[np.ndarray]] = {}

        base = _baseline_predictions(train, val, raw_val, missing_val, location)
        for name, pred in base.items():
            model_predictions[name] = [np.asarray(pred, dtype=float)]
        fitted_models = {}
        for name, config in MODEL_CONFIGS.items():
            seeds = SEEDS if name in {"random_forest", "hist_gradient_boosting", "mlp", "temporal_cnn"} else (SEEDS[0],)
            fitted_models[name] = []
            for seed in seeds:
                model, _ = _fit_model(name, config, x_train, y_train, seed)
                fitted_models[name].append((seed, model))
            model_predictions[name] = [predict(model, name, make_clean_matrix_v1_1(val, location, scaler)[0]) for _, model in fitted_models[name]]

        clean_mean_preds = {name: np.mean(preds, axis=0) for name, preds in model_predictions.items()}
        for model_id, pred in clean_mean_preds.items():
            m = regression_metrics(y_val, pred)
            metric_rows.append({"fold_id": fold["fold_id"], "model_id": model_id, "noise_type": "clean", "noise_intensity": 0.0, "role": "baseline" if model_id in BASELINE_IDS else "trained_architecture", "valid_input_count": len(val), "invalid_input_count": 0, **m})
            for i, row in val.iterrows():
                prediction_rows.append({"fold_id": fold["fold_id"], "model_id": model_id, "noise_type": "clean", "noise_intensity": 0.0, "sample_id": row.sample_id, "target_month": row.target_month, "y_true": float(y_val[i]), "y_pred": float(pred[i])})

        for scenario in SCENARIOS:
            for intensity in LEVELS:
                items = [perturb_sample(raw_val[i], scenario=scenario, intensity=intensity, location=location, scale=noise_scale,
                                        seed=deterministic_seed(90261115, str(val.iloc[i].sample_id), scenario, intensity)) for i in range(len(val))]
                corrupted = np.stack([item["values"] for item in items])
                missing = np.stack([item["missing"] for item in items])
                valid = np.asarray([not item["parameters"].get("physical_range_violations") for item in items], dtype=bool)
                realized = np.asarray([item["effective_intensity"] for item in items], dtype=float)
                x_eval = model_inputs(corrupted, missing, location, scaler)
                noisy_predictions = {}
                noisy_baseline = _baseline_predictions(train, val, corrupted, missing, location)
                noisy_predictions.update({name: [np.asarray(pred, dtype=float)] for name, pred in noisy_baseline.items()})
                for model_id, fitted in fitted_models.items():
                    noisy_predictions[model_id] = [predict(model, model_id, x_eval) for _, model in fitted]
                for model_id, outputs in noisy_predictions.items():
                    pred = np.mean(outputs, axis=0)
                    pred[~valid] = np.nan
                    metrics = regression_metrics(y_val, pred)
                    metric_rows.append({"fold_id": fold["fold_id"], "model_id": model_id, "noise_type": scenario, "noise_intensity": intensity, "role": "baseline" if model_id in BASELINE_IDS else "trained_architecture", "valid_input_count": int(valid.sum()), "invalid_input_count": int((~valid).sum()), "mean_realized_noise_intensity": float(realized.mean()), **metrics})
                    for i, row in val.iterrows():
                        prediction_rows.append({"fold_id": fold["fold_id"], "model_id": model_id, "noise_type": scenario, "noise_intensity": intensity, "sample_id": row.sample_id, "target_month": row.target_month, "y_true": float(y_val[i]), "y_pred": float(pred[i]) if np.isfinite(pred[i]) else None, "input_valid": bool(valid[i]), "realized_noise_intensity": float(realized[i])})
        print(json.dumps({"event": "V11_ROBUSTNESS_FOLD_COMPLETE", "fold_id": fold["fold_id"], "train_rows": len(train), "validation_rows": len(val), "scenario_runs": len(SCENARIOS) * len(LEVELS), "elapsed_seconds": round(time.perf_counter() - started, 2)}, allow_nan=False), flush=True)

    metrics = pd.DataFrame(metric_rows)
    preds = pd.DataFrame(prediction_rows)
    # Fold-balanced scores. If any inputs fail the shared physical screen, the
    # input-valid fraction is reported explicitly; errors use common valid rows.
    fold_agg = metrics.groupby(["fold_id", "model_id", "noise_type", "noise_intensity"], as_index=False).agg(
        mae=("mae", "mean"), rmse=("rmse", "mean"), mean_bias=("mean_bias", "mean"),
        valid_input_count=("valid_input_count", "min"), invalid_input_count=("invalid_input_count", "max"),
        mean_realized_noise_intensity=("mean_realized_noise_intensity", "mean"),
    )
    overall = fold_agg.groupby(["model_id", "noise_type", "noise_intensity"], as_index=False).agg(
        mean_fold_mae=("mae", "mean"), median_fold_mae=("mae", "median"), sd_fold_mae=("mae", "std"),
        mean_fold_rmse=("rmse", "mean"), mean_fold_bias=("mean_bias", "mean"),
        min_valid_input_months=("valid_input_count", "min"), max_invalid_input_months=("invalid_input_count", "max"),
    )
    clean = overall[overall.noise_type.eq("clean")].set_index("model_id").mean_fold_mae
    stress50 = overall[overall.noise_intensity.eq(0.50)].groupby("model_id").mean_fold_mae.mean()
    decision = pd.DataFrame({"clean_mean_fold_mae": clean, "mean_50pct_scenario_mae": stress50}).dropna()
    decision["composite_equal_weight"] = 0.5 * decision.clean_mean_fold_mae + 0.5 * decision.mean_50pct_scenario_mae
    decision["selected_by_development_composite"] = False
    selected = str(decision.composite_equal_weight.sort_values(kind="stable").index[0])
    decision.loc[selected, "selected_by_development_composite"] = True
    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "robustness_metrics_by_fold_v1_1.csv"
    overall_path = OUT / "robustness_model_noise_matrix_v1_1.csv"
    decision_path = OUT / "robustness_selection_composite_v1_1.csv"
    predictions_path = OUT / "robustness_predictions_v1_1.parquet"
    manifest_path = OUT / "robustness_manifest_v1_1.json"
    metrics.to_csv(metrics_path, index=False)
    overall.to_csv(overall_path, index=False)
    decision.reset_index(names="model_id").to_csv(decision_path, index=False)
    preds.to_parquet(predictions_path, index=False, compression="zstd")
    manifest = {
        "version": "kuntur-phase2-v1.1-robustness-0.1.0",
        "status": "COMPLETED_DEVELOPMENT_STRESS_AUDIT",
        "source_forecast_sha256": sha256(FORECAST),
        "protocol_sha256": sha256(PROTOCOL),
        "development_target_range": [str(frame.target_month.min()), str(frame.target_month.max())],
        "unique_validation_target_months": int(preds.loc[preds.noise_type.eq("clean"), "sample_id"].nunique()),
        "folds": list(FOLDS),
        "candidate_ids": list(BASELINE_IDS) + list(MODEL_CONFIGS),
        "scenarios": list(SCENARIOS),
        "levels": list(LEVELS),
        "selection_objective": "0.5 times mean fold clean MAE plus 0.5 times the unweighted mean fold MAE across six individual and five combined 50% corruption scenarios; all nine policies are eligible.",
        "selected_development_policy": selected,
        "decision_table": decision.reset_index(names="model_id").to_dict("records"),
        "limitations": ["All scores are development evidence; 2015-2018 was previously explored in v1 robustness work.", "No independent future stress-test set is available.", "Corruption-invalid inputs are excluded from loss and counted; valid input counts are reported by scenario.", "Physical-range violation screening is a perturbation validity filter, not an input detector trained for deployment."],
        "artifacts": {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (metrics_path, overall_path, decision_path, predictions_path)},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "torch": torch.__version__},
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(safe({"status": manifest["status"], "selected": selected, "decision_table": manifest["decision_table"], "manifest": str(manifest_path), "elapsed_seconds": manifest["elapsed_seconds"]}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
