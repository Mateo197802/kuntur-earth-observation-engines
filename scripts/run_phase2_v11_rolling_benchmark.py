#!/usr/bin/env python3
"""Baseline-inclusive rolling-origin forecast benchmark for KUNTUR v1.1.

Development-only protocol. It reads forecast rows through 2018-12, uses three
predeclared future blocks, and never opens the historical validation or test
periods. Hyperparameters are fixed in this file before the run; no fold is used
to tune another fold's model.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

from kuntur_data.phase2.augmentation import CHANNELS, FEATURE_COLUMNS, history_array
from kuntur_data.phase2.predictive import (
    TemporalCNN,
    fit_preprocessing_v1_1,
    make_clean_matrix_v1_1,
    model_inputs,
    predict,
    regression_metrics,
    sha256_file,
)
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
OUTPUT = ROOT / "results/phase2/v1_1/target_scaled_candidates"
SEEDS = (20261115, 20261215, 20261315)
FOLDS = (
    {"fold_id": "origin_2009", "train_end": "2009-12", "validation_start": "2010-01", "validation_end": "2012-12"},
    {"fold_id": "origin_2012", "train_end": "2012-12", "validation_start": "2013-01", "validation_end": "2015-12"},
    {"fold_id": "origin_2015", "train_end": "2015-12", "validation_start": "2016-01", "validation_end": "2018-12"},
)
MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 120, "max_depth": 3, "min_samples_leaf": 3},
    "hist_gradient_boosting": {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 7, "l2_regularization": 1.0},
    "svr_rbf": {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
    "mlp": {"hidden_layer_sizes": (16,), "alpha": 0.01, "max_iter": 1000},
    "temporal_cnn": {"channels": 8, "learning_rate": 0.001, "max_epochs": 75, "patience": 10},
}
BASELINE_IDS = ("persistence", "seasonal_naive", "training_monthly_climatology")


class TargetScaledTemporalCNN:
    """CNN wrapper whose target transform is learned from gradient-fit rows only."""

    def __init__(self, channels: int, seed: int, center: float, scale: float):
        self.core = TemporalCNN(channels=channels, seed=seed)
        self.center = float(center)
        self.scale = float(scale)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.center + self.scale * self.core.predict(x)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _assert_t3_policy(frame: pd.DataFrame) -> None:
    target = pd.PeriodIndex(frame["target_month"].astype(str), freq="M")
    history_end = pd.PeriodIndex(frame["history_end_month"].astype(str), freq="M")
    origin = pd.PeriodIndex(frame["forecast_origin_month"].astype(str), freq="M")
    if not np.all(history_end == target - 3):
        raise ValueError("Forecast feature history must end exactly at target T-3")
    if not np.all(origin == target - 1):
        raise ValueError("Forecast origin must be exactly one month before the target")


def _fit_model(name: str, config: dict, x: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, int | None]:
    if name == "ridge":
        return Ridge(**config).fit(x, y), None
    if name == "random_forest":
        return RandomForestRegressor(**config, random_state=seed, n_jobs=1).fit(x, y), None
    if name == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**config, random_state=seed, early_stopping=False).fit(x, y), None
    if name == "svr_rbf":
        return SVR(kernel="rbf", **config).fit(x, y), None
    if name == "mlp":
        estimator = MLPRegressor(**config, solver="adam", batch_size=32, learning_rate_init=0.001,
                                 tol=1e-4, n_iter_no_change=50, random_state=seed, shuffle=True)
        model = TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
        model.fit(x, y)
        return model, int(model.regressor_.n_iter_)
    if name == "temporal_cnn":
        # Reserve the most recent 12 training targets for early stopping. These
        # labels are excluded from gradient updates and outer validation.
        if len(y) < 48:
            raise ValueError("Temporal CNN requires at least 48 training rows")
        stop_n = 12
        train_y = y[:-stop_n]
        center = float(np.mean(train_y))
        scale = float(np.std(train_y, ddof=0))
        if not np.isfinite(scale) or scale <= 1e-8:
            scale = 1.0
        model = TargetScaledTemporalCNN(config["channels"], seed, center, scale)
        epochs = model.core.fit(x[:-stop_n], (train_y - center) / scale, np.ones(len(train_y)),
                           x[-stop_n:], (y[-stop_n:] - center) / scale, learning_rate=config["learning_rate"],
                           max_epochs=config["max_epochs"], patience=config["patience"], seed=seed)
        return model, int(epochs)
    raise ValueError(f"Unknown model: {name}")


def _baseline_predictions(train: pd.DataFrame, val: pd.DataFrame, raw: np.ndarray, missing: np.ndarray, location: np.ndarray) -> dict[str, np.ndarray]:
    vals = raw.copy()
    vals[missing] = np.broadcast_to(location.reshape(1, 1, -1), vals.shape)[missing]
    train_month = pd.to_datetime(train.target_month.astype(str) + "-01").dt.month
    climatology = train.assign(_month=train_month).groupby("_month").target_value.mean().to_dict()
    val_month = pd.to_datetime(val.target_month.astype(str) + "-01").dt.month
    if set(val_month.unique()) - set(climatology):
        raise ValueError("Training fold lacks one or more target calendar months")
    return {
        "persistence": vals[:, -1, 0],
        "seasonal_naive": vals[:, 2, 0],
        "training_monthly_climatology": np.asarray([climatology[int(m)] for m in val_month], dtype=float),
    }


def main() -> None:
    started = time.perf_counter()
    if not FORECAST.is_file():
        raise SystemExit(f"Missing frozen forecast Gold file: {FORECAST}")
    # Strict predicate limits data materialization to development observations.
    frame = pd.read_parquet(FORECAST, filters=[("target_month", "<=", "2018-12")])
    if frame.empty or frame.target_month.max() != "2018-12" or frame.target_month.max() > "2018-12":
        raise SystemExit("Unexpected development forecast coverage; later rows must not be loaded")
    if frame.sample_id.duplicated().any():
        raise SystemExit("Duplicate forecast sample IDs")
    _assert_t3_policy(frame)
    expected_months = pd.period_range(frame.target_month.min(), frame.target_month.max(), freq="M").astype(str).tolist()
    if frame.target_month.tolist() != expected_months:
        raise SystemExit("Development targets are not a continuous monthly sequence")

    predictions: list[dict] = []
    fit_rows: list[dict] = []
    fold_counts: list[dict] = []
    for fold in FOLDS:
        train = frame[frame.target_month <= fold["train_end"]].reset_index(drop=True)
        val = frame[(frame.target_month >= fold["validation_start"]) & (frame.target_month <= fold["validation_end"])].reset_index(drop=True)
        if len(val) != 36 or not len(train):
            raise SystemExit(f"Unexpected fold dimensions for {fold['fold_id']}: {len(train)} train, {len(val)} val")
        if train.target_month.max() >= val.target_month.min():
            raise SystemExit("Chronology violation in rolling fold")
        fold_counts.append({**fold, "train_count": len(train), "validation_count": len(val), "train_sample_ids_sha256": _sha(sorted(train.sample_id.astype(str).tolist())), "validation_sample_ids_sha256": _sha(sorted(val.sample_id.astype(str).tolist()))})
        location, _, scaler = fit_preprocessing_v1_1(train)
        raw_train = history_array(train)
        train_missing = ~np.isfinite(raw_train)
        x_train = model_inputs(raw_train, train_missing, location, scaler)
        raw_val = history_array(val)
        x_val, _, val_missing = make_clean_matrix_v1_1(val, location, scaler)
        y_train = train.target_value.to_numpy(dtype=float)
        y_val = val.target_value.to_numpy(dtype=float)
        if not np.isfinite(y_train).all() or not np.isfinite(y_val).all():
            raise SystemExit("Missing target values in a rolling-origin fold")

        baselines = _baseline_predictions(train, val, raw_val, val_missing, location)
        for name, pred in baselines.items():
            metrics = regression_metrics(y_val, pred)
            for i, row in val.iterrows():
                predictions.append({"fold_id": fold["fold_id"], "model_id": name, "seed": None, "sample_id": row.sample_id, "target_month": row.target_month, "y_true": float(y_val[i]), "y_pred": float(pred[i])})
            fit_rows.append({"fold_id": fold["fold_id"], "model_id": name, "seed": None, "role": "baseline", **metrics, "fit_seconds": 0.0, "epochs": None})

        for model_name, config in MODEL_CONFIGS.items():
            model_seeds = SEEDS if model_name in {"random_forest", "hist_gradient_boosting", "mlp", "temporal_cnn"} else (SEEDS[0],)
            for seed in model_seeds:
                fit_start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model, epochs = _fit_model(model_name, config, x_train, y_train, seed)
                fit_seconds = time.perf_counter() - fit_start
                pred = predict(model, model_name, x_val)
                metrics = regression_metrics(y_val, pred)
                fit_rows.append({"fold_id": fold["fold_id"], "model_id": model_name, "seed": seed, "role": "trained_architecture", "config": config, **metrics, "fit_seconds": fit_seconds, "epochs": epochs, "warnings": [f"{type(item.message).__name__}: {str(item.message)[:200]}" for item in caught]})
                for i, row in val.iterrows():
                    predictions.append({"fold_id": fold["fold_id"], "model_id": model_name, "seed": seed, "sample_id": row.sample_id, "target_month": row.target_month, "y_true": float(y_val[i]), "y_pred": float(pred[i])})

    pred_frame = pd.DataFrame(predictions)
    fit_frame = pd.DataFrame(fit_rows)
    # First average random-seed predictions per month, then calculate the pooled
    # development loss. Fold metrics remain separately available below.
    per_month = pred_frame.groupby(["fold_id", "model_id", "sample_id", "target_month", "y_true"], as_index=False).y_pred.mean()
    summary_rows = []
    for model_id, group in per_month.groupby("model_id", sort=False):
        fold_metrics = []
        for fold_id, fold_group in group.groupby("fold_id", sort=False):
            fold_metrics.append({"fold_id": fold_id, **regression_metrics(fold_group.y_true.to_numpy(), fold_group.y_pred.to_numpy())})
        pooled = regression_metrics(group.y_true.to_numpy(), group.y_pred.to_numpy())
        summary_rows.append({"model_id": model_id, "mean_fold_mae": float(np.mean([r["mae"] for r in fold_metrics])), "median_fold_mae": float(np.median([r["mae"] for r in fold_metrics])), "sd_fold_mae": float(np.std([r["mae"] for r in fold_metrics], ddof=1)), "max_fold_mae": float(np.max([r["mae"] for r in fold_metrics])), "mean_fold_rmse": float(np.mean([r["rmse"] for r in fold_metrics])), "mean_fold_bias": float(np.mean([r["mean_bias"] for r in fold_metrics])), "pooled_mae": pooled["mae"], "pooled_rmse": pooled["rmse"], "pooled_bias": pooled["mean_bias"], "fold_metrics": fold_metrics})
    summary = pd.DataFrame(summary_rows)
    # All nine alternatives compete. This is a development selection score,
    # not an unbiased final generalization estimate.
    selected = summary.sort_values(["mean_fold_mae", "mean_fold_rmse", "max_fold_mae", "model_id"], kind="stable").iloc[0].model_id
    baseline_mae = float(summary.loc[summary.model_id == "training_monthly_climatology", "mean_fold_mae"].iloc[0])
    summary["skill_vs_training_climatology_mae"] = 1.0 - summary.mean_fold_mae / baseline_mae
    summary["selected_by_predeclared_rule"] = summary.model_id.eq(selected)
    SUMMARY_VERSION = "kuntur-phase2-v1.1-rolling-development-target-scaled-0.2.0"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pred_path = OUTPUT / "rolling_predictions_development_2003_2018.parquet"
    fit_path = OUTPUT / "rolling_fold_metrics_development_2003_2018.csv"
    summary_path = OUTPUT / "rolling_model_summary_development_2003_2018.json"
    pred_frame.to_parquet(pred_path, index=False, compression="zstd")
    fit_frame.to_csv(fit_path, index=False)
    payload = {
        "manifest_version": SUMMARY_VERSION,
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "source_forecast_path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet",
        "source_forecast_sha256": sha256_file(FORECAST),
        "development_target_range_loaded": [str(frame.target_month.min()), str(frame.target_month.max())],
        "development_rows_loaded": int(len(frame)),
        "test_policy": "Rows after 2018-12 are excluded by Parquet predicate; frozen 2022-2024 test is not read or scored.",
        "forecast_policy": "Input histories retain frozen 12-month T-14 through T-3 policy; each target strictly follows its issue origin.",
        "preprocessing": "Training-fold channel means for missing-value imputation; StandardScaler fitted after imputation using training rows only; explicit missing indicators. MLP target scaler and temporal CNN target centering/scaling are fit on each fold's training targets only.",
        "protocol": {"folds": fold_counts, "model_configs": MODEL_CONFIGS, "baseline_ids": BASELINE_IDS, "seeds": list(SEEDS), "cnn_early_stopping": "Last 12 rows of each training fold are reserved from gradient updates and used only for early stopping.", "selection_rule": "Lowest mean fold MAE across six trained architectures and three baselines; tie-break by mean fold RMSE, then maximum fold MAE, then model ID. Development selection evidence only."},
        "selected_development_candidate": str(selected),
        "summary": summary.to_dict("records"),
        "fit_rows_sha256": sha256_file(fit_path),
        "predictions_sha256": sha256_file(pred_path),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "caveats": ["Rolling development metrics were used for candidate comparison and are not an unbiased post-selection performance estimate.", "The 2019-2021 v1 validation and 2022-2024 v1 test were not loaded by this script.", "No new independent future test is available in this protocol.", "No augmentation was used in this base corrected benchmark."],
    }
    _atomic_json(summary_path, payload)
    print(json.dumps({"status": payload["status"], "elapsed_seconds": payload["elapsed_seconds"], "rows_loaded": len(frame), "target_range": payload["development_target_range_loaded"], "selected": selected, "summary": [{k: r[k] for k in ("model_id", "mean_fold_mae", "mean_fold_rmse", "mean_fold_bias", "max_fold_mae", "skill_vs_training_climatology_mae")} for r in summary.to_dict("records")], "manifest": str(summary_path)}, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()
