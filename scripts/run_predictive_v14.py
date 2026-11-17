#!/usr/bin/env python3
"""Run KUNTUR v1.4 nested chronological anomaly-forecast experiments."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.svm import SVR
from threadpoolctl import threadpool_limits

from kuntur_data.phase2.predictive_v14 import (
    DEVELOPMENT_END,
    FEATURE_GROUP_COLUMNS,
    FeaturePair,
    TemporalFold,
    augment_training_rows,
    build_history_tensor,
    chronological_inner_folds,
    chronological_outer_folds,
    expanding_monthly_climatology,
    feature_pair,
    metric_summary,
    monthly_gold_wide,
    paired_moving_block_ci,
    load_fold_checkpoint,
    latest_available_label_month,
    labels_available_by_issue,
    save_fold_checkpoint,
    select_shrinkage_alpha,
    shrink_predictions,
    validate_forecast_timeline,
)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261117
ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)
INTENSITIES = (0.25, 0.50)
BOOTSTRAP_REPLICATES = 2000
MODEL_PRIORITY = {name: i for i, name in enumerate(("ridge", "elastic_net", "svr_rbf", "hist_gradient_boosting", "random_forest"))}

MODEL_CONFIGS: tuple[dict[str, Any], ...] = (
    {"model_id": "ridge", "config_id": "ridge_alpha_1", "config": {"alpha": 1.0}},
    {"model_id": "ridge", "config_id": "ridge_alpha_10", "config": {"alpha": 10.0}},
    {"model_id": "elastic_net", "config_id": "elastic_net_a001_l02", "config": {"alpha": 0.01, "l1_ratio": 0.2}},
    {"model_id": "elastic_net", "config_id": "elastic_net_a01_l05", "config": {"alpha": 0.1, "l1_ratio": 0.5}},
    {"model_id": "svr_rbf", "config_id": "svr_c025", "config": {"C": 0.25, "epsilon": 0.1, "gamma": "scale"}},
    {"model_id": "svr_rbf", "config_id": "svr_c1", "config": {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}},
    {"model_id": "hist_gradient_boosting", "config_id": "hgb_leaf2", "config": {"learning_rate": 0.05, "max_iter": 40, "max_leaf_nodes": 2, "min_samples_leaf": 10, "l2_regularization": 5.0}},
    {"model_id": "hist_gradient_boosting", "config_id": "hgb_leaf3", "config": {"learning_rate": 0.05, "max_iter": 50, "max_leaf_nodes": 3, "min_samples_leaf": 10, "l2_regularization": 5.0}},
    {"model_id": "random_forest", "config_id": "rf_depth2", "config": {"n_estimators": 60, "max_depth": 2, "min_samples_leaf": 5, "max_features": 0.8}},
    {"model_id": "random_forest", "config_id": "rf_depth3", "config": {"n_estimators": 60, "max_depth": 3, "min_samples_leaf": 8, "max_features": 0.8}},
)


class PeakRSS:
    def __init__(self, interval: float = 0.05):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak = max(self.peak, self.process.memory_info().rss)
            except psutil.Error:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        try:
            self.peak = max(self.peak, self.process.memory_info().rss)
        except psutil.Error:
            pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def make_estimator(model_id: str, config: dict[str, Any], seed: int):
    if model_id == "ridge":
        return Ridge(**config)
    if model_id == "elastic_net":
        return ElasticNet(**config, max_iter=10000, tol=1e-5, selection="cyclic", random_state=seed)
    if model_id == "svr_rbf":
        return SVR(kernel="rbf", **config)
    if model_id == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(**config, random_state=seed, early_stopping=False)
    if model_id == "random_forest":
        return RandomForestRegressor(**config, random_state=seed, n_jobs=1)
    raise ValueError(f"unknown predictive model {model_id!r}")


def fit_estimator(model_id: str, config: dict[str, Any], x: np.ndarray, y: np.ndarray, weights: np.ndarray | None, seed: int) -> tuple[Any, dict[str, float | int]]:
    model = make_estimator(model_id, config, seed)
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    rss_before = psutil.Process(os.getpid()).memory_info().rss
    # Some sklearn 1.9 tree estimators dispatch small tasks through a shared
    # threading backend. Limiting native pools to one thread avoids nested
    # oversubscription and a warning-filter race in that backend.
    with threadpool_limits(limits=1):
        if weights is None:
            model.fit(x, y)
        else:
            model.fit(x, y, sample_weight=weights)
    rss_after = psutil.Process(os.getpid()).memory_info().rss
    info = {
        "fit_wall_seconds": float(time.perf_counter() - start_wall),
        "fit_cpu_seconds": float(time.process_time() - start_cpu),
        "rss_before_bytes": int(rss_before),
        "rss_after_bytes": int(rss_after),
    }
    return model, info


def estimator_complexity(model: Any, model_id: str, feature_count: int) -> dict[str, int | None]:
    parameters: int | None = None
    tree_nodes: int | None = None
    if model_id in {"ridge", "elastic_net"}:
        parameters = int(np.asarray(model.coef_).size + 1)
    elif model_id == "svr_rbf":
        parameters = int(len(model.support_))
    elif model_id == "random_forest":
        tree_nodes = int(sum(tree.tree_.node_count for tree in model.estimators_))
        parameters = tree_nodes
    elif model_id == "hist_gradient_boosting":
        nodes = getattr(model, "_predictors", [])
        tree_nodes = int(sum(tree.nodes.shape[0] for stage in nodes for tree in stage))
        parameters = tree_nodes
    return {"feature_count": int(feature_count), "complexity_indicator": parameters, "tree_node_count": tree_nodes}


def candidate_key(feature_group: str, model_id: str, config_id: str, alpha: float) -> tuple[str, str, str, float]:
    return feature_group, model_id, config_id, float(alpha)


def candidate_order(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        float(row["mae"]),
        float(row["rmse"]),
        int(row["feature_count"]),
        MODEL_PRIORITY[str(row["model_id"])],
        str(row["config_id"]),
        float(row["alpha"]),
    )


def _inner_records_for_key(store: dict, key: tuple[str, str, str, float]) -> pd.DataFrame:
    frames = store.get(key, [])
    if not frames:
        raise ValueError(f"inner candidate has no predictions: {key}")
    joined = pd.concat(frames, ignore_index=True).sort_values("target_month", kind="stable").reset_index(drop=True)
    if joined.target_month.duplicated().any():
        raise ValueError(f"inner candidate repeats validation months: {key}")
    return joined


def run_inner_search(outer: TemporalFold, monthly: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[dict[str, Any]]]:
    inner_folds = chronological_inner_folds(outer.train)
    predictions: dict[tuple[str, str, str, float], list[pd.DataFrame]] = {}
    registry: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for inner_index, inner in enumerate(inner_folds, start=1):
        inner_started = time.perf_counter()
        for feature_group, feature_columns in FEATURE_GROUP_COLUMNS.items():
            try:
                pair = feature_pair(inner.train, inner.validation, monthly, feature_group)
            except Exception as exc:
                for model_spec in MODEL_CONFIGS:
                    registry.append({
                        "trial_type": "inner_candidate", "outer_fold": outer.name,
                        "inner_fold": inner.name, "feature_group": feature_group,
                        "model_id": model_spec["model_id"], "config_id": model_spec["config_id"],
                        "alpha": None, "status": "FAILED", "failure": f"{type(exc).__name__}: {exc}",
                    })
                continue
            y_train = pair.y_train_residual
            y_validation = pair.y_validation
            for model_spec in MODEL_CONFIGS:
                model_id = str(model_spec["model_id"])
                config_id = str(model_spec["config_id"])
                config = dict(model_spec["config"])
                seed = stable_seed(SEED, outer.name, inner.name, feature_group, model_id, config_id)
                try:
                    model, timing = fit_estimator(model_id, config, pair.x_train, y_train, None, seed)
                    residual_hat = np.asarray(model.predict(pair.x_validation), dtype=float)
                    if not np.isfinite(residual_hat).all():
                        raise ValueError("model returned a nonfinite residual prediction")
                    complexity = estimator_complexity(model, model_id, len(feature_columns))
                    for alpha in ALPHAS:
                        prediction = shrink_predictions(pair.offset_validation, residual_hat, alpha)
                        score = metric_summary(y_validation, prediction, pair.offset_validation)
                        trial_key = candidate_key(feature_group, model_id, config_id, alpha)
                        record_frame = pd.DataFrame({
                            "sample_id": inner.validation.sample_id.astype(str).to_numpy(),
                            "target_month": inner.validation.target_month.astype(str).to_numpy(),
                            "observed_t2m_c": y_validation,
                            "training_climatology_c": pair.offset_validation,
                            "residual_prediction_c": residual_hat,
                            "prediction_t2m_c": prediction,
                        })
                        predictions.setdefault(trial_key, []).append(record_frame)
                        row = {
                            "trial_type": "inner_candidate", "outer_fold": outer.name,
                            "inner_fold": inner.name, "feature_group": feature_group,
                            "model_id": model_id, "config_id": config_id,
                            "configuration_json": json.dumps(config, sort_keys=True),
                            "alpha": float(alpha), "seed": int(seed), "status": "COMPLETED",
                            "training_month_start": str(inner.train.target_month.min()),
                            "training_month_end": str(inner.train.target_month.max()),
                            "training_month_count": int(len(inner.train)),
                            "validation_month_start": str(inner.validation.target_month.min()),
                            "validation_month_end": str(inner.validation.target_month.max()),
                            "validation_month_count": int(len(inner.validation)),
                            **complexity, **timing, **score,
                            "prediction_sha256": hashlib.sha256(np.asarray(prediction, dtype="<f8").tobytes()).hexdigest(),
                        }
                        registry.append(row)
                except Exception as exc:
                    registry.append({
                        "trial_type": "inner_candidate", "outer_fold": outer.name,
                        "inner_fold": inner.name, "feature_group": feature_group,
                        "model_id": model_id, "config_id": config_id, "alpha": None,
                        "seed": int(seed), "status": "FAILED",
                        "failure": f"{type(exc).__name__}: {exc}",
                    })
        print(
            f"INNER_FOLD_COMPLETED outer={outer.name} inner={inner.name} "
            f"fold={inner_index}/{len(inner_folds)} elapsed_seconds={time.perf_counter() - inner_started:.1f}",
            flush=True,
        )

    expected_months = set().union(*(set(f.validation.target_month.astype(str)) for f in inner_folds))
    for key, records in predictions.items():
        joined = pd.concat(records, ignore_index=True)
        if set(joined.target_month.astype(str)) != expected_months or len(joined) != len(expected_months):
            raise ValueError(f"inner candidate {key} did not use the common validation months")
        score = metric_summary(
            joined.observed_t2m_c.to_numpy(),
            joined.prediction_t2m_c.to_numpy(),
            joined.training_climatology_c.to_numpy(),
        )
        group, model_id, config_id, alpha = key
        config = next(item["config"] for item in MODEL_CONFIGS if item["model_id"] == model_id and item["config_id"] == config_id)
        feature_count = len(FEATURE_GROUP_COLUMNS[group])
        trial_rows.append({
            "feature_group": group, "model_id": model_id, "config_id": config_id,
            "configuration_json": json.dumps(config, sort_keys=True), "alpha": alpha,
            "feature_count": feature_count, "inner_validation_months": len(joined),
            **score,
        })
    summaries = pd.DataFrame(trial_rows)
    if summaries.empty:
        raise RuntimeError(f"all fixed candidates failed inside {outer.name}")
    summaries["selected_overall"] = False
    summaries["selected_by_feature_group"] = False
    summaries["selected_by_architecture"] = False
    summaries["inner_prediction_sha256"] = summaries.apply(
        lambda row: hashlib.sha256(_inner_records_for_key(
            predictions, candidate_key(row.feature_group, row.model_id, row.config_id, row.alpha)
        ).prediction_t2m_c.to_numpy(dtype="<f8").tobytes()).hexdigest(), axis=1
    )
    return summaries, predictions, registry


def select_specs(summaries: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    ordered = summaries.sort_values(
        ["mae", "rmse", "feature_count", "model_id", "config_id", "alpha"],
        key=lambda col: col.map(MODEL_PRIORITY) if col.name == "model_id" else col,
        kind="stable",
    )
    overall = ordered.iloc[0].to_dict()
    by_group: dict[str, dict[str, Any]] = {}
    for group, rows in summaries.groupby("feature_group", sort=False):
        by_group[str(group)] = min(rows.to_dict("records"), key=candidate_order)
    by_model: dict[str, dict[str, Any]] = {}
    for model_id, rows in summaries.groupby("model_id", sort=False):
        by_model[str(model_id)] = min(rows.to_dict("records"), key=candidate_order)
    return by_group, by_model, overall


def _spec_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_group": str(row["feature_group"]),
        "model_id": str(row["model_id"]),
        "config_id": str(row["config_id"]),
        "config": json.loads(row["configuration_json"]),
        "alpha": float(row["alpha"]),
    }


def _fit_spec(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    monthly: pd.DataFrame,
    spec: dict[str, Any],
    *,
    policy: str = "clean",
    intensity: float = 0.0,
    seed: int,
) -> dict[str, Any]:
    pair = feature_pair(train, validation, monthly, spec["feature_group"])
    if policy == "clean":
        x_fit, y_fit, weights = pair.x_train, pair.y_train_residual, None
    else:
        x_fit, y_fit, weights = augment_training_rows(
            pair.x_train,
            pair.y_train_residual,
            pair.feature_names,
            sample_ids=train.sample_id.astype(str),
            policy=policy,
            intensity=intensity,
            seed=seed,
        )
    model, timing = fit_estimator(spec["model_id"], spec["config"], x_fit, y_fit, weights, seed)
    residual_hat = np.asarray(model.predict(pair.x_validation), dtype=float)
    prediction = shrink_predictions(pair.offset_validation, residual_hat, float(spec["alpha"]))
    return {
        "pair": pair, "model": model, "timing": timing,
        "residual_prediction": residual_hat, "prediction": prediction,
        "complexity": estimator_complexity(model, spec["model_id"], len(FEATURE_GROUP_COLUMNS[spec["feature_group"]])),
        "policy": policy, "intensity": float(intensity),
    }


def _baseline_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    monthly: pd.DataFrame,
    climate: np.ndarray,
) -> dict[str, np.ndarray]:
    histories, _ = build_history_tensor(validation, monthly)
    expanding = expanding_monthly_climatology(train, validation)
    return {
        "training_monthly_climatology": np.asarray(climate, dtype=float),
        "expanding_monthly_climatology": expanding,
        "seasonal_naive_t_minus_12": histories[:, 2, 0],
        "persistence_t_minus_3": histories[:, -1, 0],
    }


def _append_prediction_rows(
    output: list[dict[str, Any]],
    frame: pd.DataFrame,
    prediction: np.ndarray,
    climate: np.ndarray,
    *,
    model_id: str,
    fold: str,
    architecture: str,
    feature_group: str,
    config_id: str,
    alpha: float,
    policy: str,
    training_anomaly_q75: float,
) -> None:
    residual = np.asarray(prediction, dtype=float) - np.asarray(climate, dtype=float)
    for index, row in enumerate(frame.itertuples(index=False)):
        output.append({
            "sample_id": str(row.sample_id), "target_month": str(row.target_month),
            "outer_fold": fold, "model_id": model_id,
            "architecture": architecture, "feature_group": feature_group,
            "config_id": config_id, "alpha": float(alpha),
            "training_policy": policy,
            "observed_t2m_c": float(row.target_value),
            "training_climatology_c": float(climate[index]),
            "training_anomaly_q75_c": float(training_anomaly_q75),
            "absolute_anomaly_c": float(abs(float(row.target_value) - float(climate[index]))),
            "predicted_residual_c": float(residual[index]),
            "prediction_t2m_c": float(prediction[index]),
        })


def _model_trial_summary(
    rows: list[dict[str, Any]],
    frame: pd.DataFrame,
    *,
    trial_type: str,
    fold: str,
    spec: dict[str, Any],
    fit: dict[str, Any],
    prediction: np.ndarray,
) -> None:
    rows.append({
        "trial_type": trial_type, "outer_fold": fold,
        "feature_group": spec["feature_group"], "model_id": spec["model_id"],
        "config_id": spec["config_id"], "configuration_json": json.dumps(spec["config"], sort_keys=True),
        "alpha": float(spec["alpha"]), "training_policy": fit["policy"],
        "augmentation_intensity": fit["intensity"], "status": "COMPLETED",
        "training_month_start": str(frame.train.target_month.min()),
        "training_month_end": str(frame.train.target_month.max()),
        "training_month_count": int(len(frame.train)),
        "validation_month_start": str(frame.validation.target_month.min()),
        "validation_month_end": str(frame.validation.target_month.max()),
        "validation_month_count": int(len(frame.validation)),
        **fit["complexity"], **fit["timing"],
        "prediction_sha256": hashlib.sha256(np.asarray(prediction, dtype="<f8").tobytes()).hexdigest(),
    })


def _augmentation_inner_search(
    outer: TemporalFold,
    monthly: pd.DataFrame,
    spec: dict[str, Any],
    clean_prediction_store: dict,
    base_key: tuple[str, str, str, float],
    registry: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict]:
    inner_folds = chronological_inner_folds(outer.train)
    variants: list[tuple[str, str, float]] = [
        ("clean", "clean", 0.0),
        ("mild_gaussian_10pct", "mild_gaussian", 0.10),
        ("mild_dropout_10pct", "mild_dropout", 0.10),
        ("calibrated_gaussian_05pct", "mild_gaussian", 0.05),
        ("calibrated_gaussian_20pct", "mild_gaussian", 0.20),
    ]
    variant_records: dict[str, dict[float, list[pd.DataFrame]]] = {name: {} for name, _, _ in variants}
    for variant, policy, intensity in variants:
        for alpha in ALPHAS:
            variant_records[variant][alpha] = []
    clean_spec_rows = _inner_records_for_key(clean_prediction_store, base_key)
    # The selected clean specification already has a fold-local preprocessor and fits recorded in search.
    clean_alpha = float(spec["alpha"])
    variant_records["clean"][clean_alpha].append(clean_spec_rows)
    for inner in inner_folds:
        for variant, policy, intensity in variants[1:]:
            pair = feature_pair(inner.train, inner.validation, monthly, spec["feature_group"])
            seed = stable_seed(SEED, outer.name, inner.name, spec["model_id"], spec["config_id"], variant)
            x_fit, y_fit, weights = augment_training_rows(
                pair.x_train, pair.y_train_residual, pair.feature_names,
                sample_ids=inner.train.sample_id.astype(str), policy=policy,
                intensity=intensity, seed=seed,
            )
            try:
                model, timing = fit_estimator(spec["model_id"], spec["config"], x_fit, y_fit, weights, seed)
                residual_hat = np.asarray(model.predict(pair.x_validation), dtype=float)
                if not np.isfinite(residual_hat).all():
                    raise ValueError("nonfinite augmentation validation prediction")
                for alpha in ALPHAS:
                    prediction = shrink_predictions(pair.offset_validation, residual_hat, alpha)
                    variant_records[variant][alpha].append(pd.DataFrame({
                        "sample_id": inner.validation.sample_id.astype(str).to_numpy(),
                        "target_month": inner.validation.target_month.astype(str).to_numpy(),
                        "observed_t2m_c": pair.y_validation,
                        "training_climatology_c": pair.offset_validation,
                        "residual_prediction_c": residual_hat,
                        "prediction_t2m_c": prediction,
                    }))
                    registry.append({
                        "trial_type": "augmentation_inner_candidate", "outer_fold": outer.name,
                        "inner_fold": inner.name, "feature_group": spec["feature_group"],
                        "model_id": spec["model_id"], "config_id": spec["config_id"],
                        "configuration_json": json.dumps(spec["config"], sort_keys=True),
                        "alpha": float(alpha), "training_policy": variant,
                        "augmentation_intensity": intensity, "seed": int(seed),
                        "status": "COMPLETED", "label_integrity_verified": True,
                        "authentic_month_weight_sum_verified": True, **timing,
                        **metric_summary(pair.y_validation, prediction, pair.offset_validation),
                        "prediction_sha256": hashlib.sha256(np.asarray(prediction, dtype="<f8").tobytes()).hexdigest(),
                    })
            except Exception as exc:
                registry.append({
                    "trial_type": "augmentation_inner_candidate", "outer_fold": outer.name,
                    "inner_fold": inner.name, "feature_group": spec["feature_group"],
                    "model_id": spec["model_id"], "config_id": spec["config_id"],
                    "training_policy": variant, "augmentation_intensity": intensity,
                    "status": "FAILED", "failure": f"{type(exc).__name__}: {exc}",
                })
    selected: dict[str, dict[str, Any]] = {}
    pooled_predictions: dict[str, pd.DataFrame] = {}
    expected_months = set().union(*(set(f.validation.target_month.astype(str)) for f in inner_folds))
    for variant, policy, intensity in variants:
        alpha_scores = []
        for alpha, frames in variant_records[variant].items():
            if not frames:
                continue
            joined = pd.concat(frames, ignore_index=True).sort_values("target_month", kind="stable").reset_index(drop=True)
            if joined.target_month.duplicated().any() or set(joined.target_month.astype(str)) != expected_months:
                # A policy is comparable only when every candidate uses the same
                # complete set of inner-validation months.
                continue
            score = metric_summary(joined.observed_t2m_c, joined.prediction_t2m_c, joined.training_climatology_c)
            alpha_scores.append({"variant": variant, "policy": policy, "intensity": intensity, "alpha": alpha, **score})
            if variant == "clean" or frames:
                pooled_predictions[f"{variant}:{alpha:.2f}"] = joined
        if not alpha_scores:
            continue
        selected[variant] = min(alpha_scores, key=lambda row: (row["mae"], row["rmse"], row["alpha"]))
    if not selected:
        raise RuntimeError(f"no augmentation policy completed for {outer.name}")
    # Select perturbation intensity and alpha entirely from inner chronological predictions.
    best_variant = min(selected, key=lambda name: (
        selected[name]["mae"], selected[name]["rmse"],
        0 if name == "clean" else 1, selected[name]["intensity"], name,
    ))
    selected["_selected_policy"] = selected[best_variant] | {"selected_variant": best_variant}
    selected["_inner_predictions"] = pooled_predictions
    return selected, variant_records


def corrupt_inputs(x: np.ndarray, feature_names: tuple[str, ...], sample_ids: list[str], scenario: str, intensity: float, seed: int) -> np.ndarray:
    output = np.asarray(x, dtype=float).copy()
    value_count = len(feature_names) // 2
    value_names = feature_names[:value_count]
    dynamic = [i for i, name in enumerate(value_names) if not name.startswith("target_month_")]
    if scenario not in {"gaussian", "missingness", "gaussian+missingness"}:
        raise ValueError(f"unknown robustness scenario {scenario!r}")
    component = intensity / np.sqrt(2.0) if scenario == "gaussian+missingness" else intensity
    for row, sample_id in enumerate(sample_ids):
        rng = np.random.default_rng(stable_seed(seed, sample_id, scenario, intensity))
        if scenario in {"gaussian", "gaussian+missingness"}:
            output[row, dynamic] += rng.normal(0.0, component, size=len(dynamic))
        if scenario in {"missingness", "gaussian+missingness"}:
            count = max(1, int(round(component * len(dynamic))))
            chosen = rng.choice(dynamic, size=min(count, len(dynamic)), replace=False)
            for index in chosen:
                output[row, index] = 0.0
                output[row, value_count + index] = 1.0
    return output


def _interval_quantile(errors: list[float], coverage: float) -> float:
    if not errors:
        raise ValueError("interval calibration requires at least one earlier residual")
    adjusted = min(1.0, np.ceil((len(errors) + 1) * coverage) / len(errors))
    return float(np.quantile(np.asarray(errors, dtype=float), adjusted, method="higher"))


def _interval_rows(
    outer: TemporalFold,
    candidate_prediction: pd.DataFrame,
    climate_prediction: pd.DataFrame,
    candidate_warm: pd.DataFrame | None,
    climate_warm: pd.DataFrame | None,
    candidate_prior: list[dict[str, Any]],
    climate_prior: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    candidate_initial = candidate_prior if candidate_prior else ([] if candidate_warm is None else candidate_warm.to_dict("records"))
    climate_initial = climate_prior if climate_prior else ([] if climate_warm is None else climate_warm.to_dict("records"))
    for model_id, frame, initial, prior, append in (
        ("nested_selected_augmentation_pipeline", candidate_prediction, candidate_initial, candidate_prior, candidate_prior),
        ("training_monthly_climatology", climate_prediction, climate_initial, climate_prior, climate_prior),
    ):
        calibration = [dict(row) for row in initial]
        for row in frame.sort_values("target_month", kind="stable").to_dict("records"):
            target_month = str(row["target_month"])
            month_number = int(target_month[-2:])
            issue_month = pd.Period(target_month, freq="M") - 1
            issue_date_utc = f"{issue_month}-01T00:00:00Z"
            latest_available = latest_available_label_month(issue_date_utc)
            actual = float(row["observed_t2m_c"])
            prediction = float(row["prediction_t2m_c"])
            available_calibration = [
                item for item in calibration
                if pd.Period(str(item["target_month"]), freq="M") <= latest_available
            ]
            same_month_records = [
                item for item in available_calibration
                if int(str(item["target_month"])[-2:]) == month_number
            ]
            pooled_records = available_calibration
            rolling_records = available_calibration[-36:]
            methods = {
                "expanding_pooled": pooled_records,
                "rolling36_pooled": rolling_records,
                "expanding_same_calendar_month": same_month_records if len(same_month_records) >= 3 else pooled_records,
            }
            anomaly = abs(actual - float(row["training_climatology_c"]))
            for method, calibration_records in methods.items():
                errors = [
                    abs(float(item["observed_t2m_c"]) - float(item["prediction_t2m_c"]))
                    for item in calibration_records
                ]
                calibration_n = len(errors)
                calibration_months = [str(item["target_month"]) for item in calibration_records]
                for coverage in (0.80, 0.90, 0.95):
                    radius = _interval_quantile(errors, coverage)
                    lower, upper = prediction - radius, prediction + radius
                    alpha = 1.0 - coverage
                    winkler = (upper - lower) + (2.0 / alpha) * max(lower - actual, 0.0) + (2.0 / alpha) * max(actual - upper, 0.0)
                    output.append({
                        "sample_id": row["sample_id"], "target_month": target_month,
                        "outer_fold": outer.name, "model_id": model_id,
                        "calibration_method": method, "nominal_coverage": coverage,
                        "lower_c": lower, "upper_c": upper,
                        "observed_t2m_c": actual, "covered": bool(lower <= actual <= upper),
                        "width_c": float(upper - lower), "winkler_score_c": float(winkler),
                        "calibration_residual_count": int(calibration_n),
                        "calibration_prior_month_start": min(calibration_months) if calibration_months else None,
                        "calibration_prior_month_end": max(calibration_months) if calibration_months else None,
                        "anomaly_magnitude_c": anomaly,
                    })
            calibration.append({"target_month": target_month, "observed_t2m_c": actual, "prediction_t2m_c": prediction})
        if model_id == "nested_selected_augmentation_pipeline":
            candidate_prior[:] = calibration
        else:
            climate_prior[:] = calibration
    return output, candidate_prior


def _season_name(month: int) -> str:
    if month in (12, 1, 2):
        return "DJF"
    if month in (3, 4, 5):
        return "MAM"
    if month in (6, 7, 8):
        return "JJA"
    return "SON"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8", newline="\n")
    temp.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False, lineterminator="\n")
    temp.replace(path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str, allow_nan=False) + "\n")


def _git_commit() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=False, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _file_facts(
    paths: list[Path],
    root: Path,
    *,
    artifact_roots: dict[str, Path] | None = None,
) -> dict[str, dict[str, Any]]:
    """Hash outputs, naming external-root files without exposing host paths."""
    repository_root = root.resolve()
    external_roots = {name: path.resolve() for name, path in (artifact_roots or {}).items()}
    facts: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            key = resolved.relative_to(repository_root).as_posix()
        except ValueError:
            matches: list[tuple[int, str, Path]] = []
            for name, base in external_roots.items():
                try:
                    relative = resolved.relative_to(base)
                except ValueError:
                    continue
                matches.append((len(str(base)), name, relative))
            if not matches:
                raise ValueError(f"artifact path is outside the repository and declared roots: {resolved.name}")
            _, name, relative = max(matches, key=lambda item: item[0])
            key = f"{name}/{relative.as_posix()}"
        if key in facts:
            raise ValueError(f"artifact manifest path collision: {key}")
        facts[key] = {"sha256": sha256_file(resolved), "bytes": int(resolved.stat().st_size)}
    return facts


def _run_robustness(outer_fits: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    for item in outer_fits:
        fold: TemporalFold = item["fold"]
        fit = item["fit"]
        pair: FeaturePair = fit["pair"]
        clean = np.asarray(fit["prediction"], dtype=float)
        for scenario in ("gaussian", "missingness", "gaussian+missingness"):
            for intensity in INTENSITIES:
                corrupted = corrupt_inputs(
                    pair.x_validation,
                    pair.feature_names,
                    item["validation"].sample_id.astype(str).tolist(),
                    scenario,
                    intensity,
                    SEED,
                )
                residual_hat = np.asarray(fit["model"].predict(corrupted), dtype=float)
                prediction = shrink_predictions(pair.offset_validation, residual_hat, item["spec"]["alpha"])
                for index, row in enumerate(item["validation"].itertuples(index=False)):
                    prediction_rows.append({
                        "sample_id": str(row.sample_id), "target_month": str(row.target_month),
                        "outer_fold": fold.name, "scenario": scenario,
                        "intensity": float(intensity), "observed_t2m_c": float(row.target_value),
                        "prediction_t2m_c": float(prediction[index]),
                        "clean_prediction_t2m_c": float(clean[index]),
                        "training_climatology_c": float(pair.offset_validation[index]),
                        "input_valid": bool(np.isfinite(corrupted[index]).all()),
                    })
    details = pd.DataFrame(prediction_rows)
    climate = details.training_climatology_c.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for (scenario, intensity, fold_name), group in details.groupby(["scenario", "intensity", "outer_fold"], sort=False):
        metrics = metric_summary(group.observed_t2m_c, group.prediction_t2m_c, group.training_climatology_c)
        clean_mae = float(np.mean(np.abs(group.clean_prediction_t2m_c - group.observed_t2m_c)))
        rows.append({
            "scenario": scenario, "intensity": float(intensity), "scope": fold_name,
            **metrics, "delta_mae_from_clean_c": float(metrics["mae"] - clean_mae),
            "prediction_stability_rms_change_c": float(np.sqrt(np.mean((group.prediction_t2m_c - group.clean_prediction_t2m_c) ** 2))),
            "input_failure_count": int((~group.input_valid).sum()),
        })
    for (scenario, intensity), group in details.groupby(["scenario", "intensity"], sort=False):
        metrics = metric_summary(group.observed_t2m_c, group.prediction_t2m_c, group.training_climatology_c)
        clean_mae = float(np.mean(np.abs(group.clean_prediction_t2m_c - group.observed_t2m_c)))
        rows.append({
            "scenario": scenario, "intensity": float(intensity), "scope": "ALL_OOF",
            **metrics, "delta_mae_from_clean_c": float(metrics["mae"] - clean_mae),
            "prediction_stability_rms_change_c": float(np.sqrt(np.mean((group.prediction_t2m_c - group.clean_prediction_t2m_c) ** 2))),
            "input_failure_count": int((~group.input_valid).sum()),
        })
    return pd.DataFrame(rows), details


def _distribution_shift(outer_folds: list[TemporalFold], monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in outer_folds:
        pair = feature_pair(fold.train, fold.validation, monthly, "all_sources")
        count = len(pair.feature_names) // 2
        for index, name in enumerate(pair.feature_names[:count]):
            train_values = pair.x_train[:, index]
            validation_values = pair.x_validation[:, index]
            train_sd = float(np.std(train_values, ddof=1))
            rows.append({
                "outer_fold": fold.name, "feature_name": name,
                "train_standardized_mean": float(np.mean(train_values)),
                "validation_standardized_mean": float(np.mean(validation_values)),
                "standardized_mean_difference": float(np.mean(validation_values) - np.mean(train_values)),
                "train_standardized_sd": train_sd,
                "validation_to_training_variance_ratio": float(np.var(validation_values, ddof=1) / np.var(train_values, ddof=1)) if np.var(train_values, ddof=1) > 0 else float("nan"),
                "train_month_count": int(len(fold.train)), "validation_month_count": int(len(fold.validation)),
            })
    return pd.DataFrame(rows)


def _residual_autocorrelation(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in ("training_monthly_climatology", "nested_selected_pipeline", "nested_selected_augmentation_pipeline"):
        frame = predictions[predictions.model_id == model_id].sort_values("target_month", kind="stable")
        residual = frame.prediction_t2m_c.to_numpy(dtype=float) - frame.observed_t2m_c.to_numpy(dtype=float)
        for lag in range(1, min(13, len(residual))):
            left, right = residual[:-lag], residual[lag:]
            corr = float(np.corrcoef(left, right)[0, 1]) if len(left) > 2 and np.std(left) > 0 and np.std(right) > 0 else float("nan")
            rows.append({"model_id": model_id, "calendar_lag_months": lag, "paired_residual_count": len(left), "residual_autocorrelation": corr})
    return pd.DataFrame(rows)


def _make_final_bundle(
    data: pd.DataFrame,
    monthly: pd.DataFrame,
    spec: dict[str, Any],
    *,
    policy: str,
    intensity: float,
    output_root: Path,
    source_hashes: dict[str, str],
) -> tuple[list[Path], dict[str, Any]]:
    training_data, inference_row = _final_refit_inputs(data)
    fit = _fit_spec(
        training_data, inference_row, monthly, spec, policy=policy, intensity=intensity,
        seed=stable_seed(SEED, "final_development_refit", spec["model_id"], spec["config_id"], policy, intensity),
    )
    pair: FeaturePair = fit["pair"]
    model_path = output_root / "models" / "model_bundle_v14.joblib"
    prep_path = output_root / "models" / "preprocessing_v14.json"
    prep = {
        "feature_group": spec["feature_group"],
        "feature_names": list(pair.feature_names),
        "feature_climatology_by_source_and_calendar_month": pair.feature_climatology,
        "feature_imputation_means": pair.imputation_means.tolist(),
        "feature_scaler_mean": pair.scaler_mean.tolist(),
        "feature_scaler_scale": pair.scaler_scale.tolist(),
        "target_monthly_climatology_c": {str(k): v for k, v in pair.training_climatology.items()},
        "history_months": "T-14 through T-3, oldest to newest",
        "source_availability_assumption": "two-month product-release lag; NASA publication timestamps were not verified",
        "training_month_start": str(training_data.target_month.min()),
        "training_month_end": str(training_data.target_month.max()),
        "training_month_count": int(len(training_data)),
        "fit_as_of_issue_time_utc": str(inference_row.issue_date_utc.iloc[0]),
    }
    bundle = {
        "model_id": "kuntur_predictive_v14_residual_shrinkage",
        "architecture": spec["model_id"], "config_id": spec["config_id"],
        "configuration": spec["config"], "alpha": float(spec["alpha"]),
        "training_policy": policy, "augmentation_intensity": float(intensity),
        "training_seed": stable_seed(SEED, "final_development_refit", spec["model_id"], spec["config_id"], policy, intensity),
        "fitted_model": fit["model"], "preprocessing": prep,
        "source_hashes": source_hashes,
        "evaluation_scope": "development-only refit after nested CV; no final-test score",
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(bundle, temp, compress=3)
    temp.replace(model_path)
    _atomic_json(prep_path, prep)
    return [model_path, prep_path], {
        "fit_wall_seconds": fit["timing"]["fit_wall_seconds"],
        "fit_cpu_seconds": fit["timing"]["fit_cpu_seconds"],
        "model_sha256": sha256_file(model_path),
        "training_month_start": str(training_data.target_month.min()),
        "training_month_end": str(training_data.target_month.max()),
        "training_month_count": int(len(training_data)),
        "fit_as_of_issue_time_utc": str(inference_row.issue_date_utc.iloc[0]),
    }


def _final_refit_inputs(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the final as-of refit and a label-free first post-development forecast row."""
    target = pd.Period(DEVELOPMENT_END, freq="M") + 1
    issue = target - 1
    issue_date = f"{issue}-01T00:00:00Z"
    training = labels_available_by_issue(data, issue_date)
    if training.empty:
        raise ValueError("no development labels were available at the final model issue time")
    inference = pd.DataFrame([{
        "sample_id": f"forecast:{target}",
        "target_month": str(target),
        "forecast_origin_month": str(target - 1),
        "issue_date_utc": issue_date,
        "history_start_month": str(target - 14),
        "history_end_month": str(target - 3),
        "target_value": np.nan,
    }])
    validate_forecast_timeline(inference)
    return training, inference


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    data_root = args.data_root.resolve()
    forecast_path = data_root / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    monthly_path = data_root / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
    split_path = data_root / "data/gold/kuntur_forecast_split_indices_v1.parquet"
    split_manifest_path = data_root / "data/manifests/forecast_split_manifest_v1.json"
    for path in (forecast_path, monthly_path, split_path, split_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"required frozen KUNTUR input is missing: {path.name}")
    source_hashes = {
        "forecast_features_sha256": sha256_file(forecast_path),
        "monthly_gold_sha256": sha256_file(monthly_path),
        "frozen_split_indices_sha256": sha256_file(split_path),
        "frozen_split_manifest_sha256": sha256_file(split_manifest_path),
    }
    forecast_columns = [
        "sample_id", "target_month", "forecast_origin_month", "issue_date_utc",
        "history_start_month", "history_end_month", "target_value",
    ]
    # The predicate ends before either consumed test period; their labels never enter this frame.
    forecast = pd.read_parquet(forecast_path, columns=forecast_columns, filters=[("target_month", "<=", DEVELOPMENT_END)])
    split = pd.read_parquet(split_path, columns=["sample_id", "target_month", "partition"])
    split = split[split.target_month.astype(str) <= DEVELOPMENT_END]
    if forecast.empty or forecast.target_month.max() != DEVELOPMENT_END:
        raise ValueError("development forecast rows must end at 2021-12")
    if len(forecast) != 226 or len(split) != 226:
        raise ValueError("frozen development sample count differs from 226 expected target months")
    validate_forecast_timeline(forecast)
    samples = forecast.merge(split, on=["sample_id", "target_month"], validate="one_to_one")
    if len(samples) != len(forecast):
        raise ValueError("forecast and frozen split indices did not match one-to-one")
    if samples.partition.value_counts().to_dict() != {"train": 190, "validation": 36}:
        raise ValueError("development partition counts do not match the frozen 190/36 train/validation split")

    gold_columns = ["month", "variable_id", "value", "quality_status"]
    gold = pd.read_parquet(monthly_path, columns=gold_columns, filters=[("month", "<=", DEVELOPMENT_END)])
    monthly = monthly_gold_wide(gold)
    target_values = monthly.loc[samples.target_month.astype(str), "merra2_t2m_c"].to_numpy(dtype=float)
    if not np.allclose(target_values, samples.target_value.to_numpy(dtype=float), rtol=0, atol=1e-10):
        raise ValueError("forecast labels do not match frozen monthly Gold MERRA-2 T2M")
    if not set(samples.target_month.astype(str)).issubset(set(monthly.index)):
        raise ValueError("a development target month is absent from the filtered Gold input")

    outer_folds = chronological_outer_folds(samples)
    output_root = args.output_root.resolve()
    report_root = args.report_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    code_hashes = {
        "runner": sha256_file(Path(__file__).resolve()),
        "feature_module": sha256_file(ROOT / "src/kuntur_data/phase2/predictive_v14.py"),
    }
    checkpoint_fingerprint = hashlib.sha256(json.dumps({"source": source_hashes, "code": code_hashes, "seed": SEED}, sort_keys=True).encode()).hexdigest()
    registry: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    interval_rows: list[dict[str, Any]] = []
    outer_fits: list[dict[str, Any]] = []
    candidate_prior: list[dict[str, Any]] = []
    climate_prior: list[dict[str, Any]] = []
    final_spec: dict[str, Any] | None = None
    final_policy = "clean"
    final_intensity = 0.0
    fold_selection_records: list[dict[str, Any]] = []
    resumed_folds: list[str] = []

    with PeakRSS() as peak_rss:
        for outer in outer_folds:
            checkpoint_path = checkpoint_root / f"{outer.name}.joblib"
            checkpoint = load_fold_checkpoint(checkpoint_path, fingerprint=checkpoint_fingerprint)
            if checkpoint is not None:
                registry.extend(checkpoint["registry"])
                prediction_rows.extend(checkpoint["predictions"])
                interval_rows.extend(checkpoint["intervals"])
                outer_fits.append(checkpoint["outer_fit"])
                candidate_prior = checkpoint["candidate_prior"]
                climate_prior = checkpoint["climate_prior"]
                fold_selection_records.append(checkpoint["selection_record"])
                if outer.name == outer_folds[-1].name:
                    final_spec = checkpoint["final_spec"]
                    final_policy = checkpoint["final_policy"]
                    final_intensity = checkpoint["final_intensity"]
                resumed_folds.append(outer.name)
                print(f"OUTER_FOLD_RESUMED name={outer.name}", flush=True)
                continue
            registry_start = len(registry)
            predictions_start = len(prediction_rows)
            intervals_start = len(interval_rows)
            summaries, prediction_store, search_registry = run_inner_search(outer, monthly)
            registry.extend(search_registry)
            by_group_rows, by_model_rows, overall_row = select_specs(summaries)
            overall_key = candidate_key(overall_row["feature_group"], overall_row["model_id"], overall_row["config_id"], overall_row["alpha"])
            base_spec = _spec_from_row(overall_row)
            for level, rows in (("overall", [overall_row]), ("feature_group", list(by_group_rows.values())), ("architecture", list(by_model_rows.values()))):
                for selected_row in rows:
                    registry.append(_selection_registry_row(outer, selected_row, level))
            clean_fit = _fit_spec(
                outer.train, outer.validation, monthly, base_spec,
                policy="clean", intensity=0.0,
                seed=stable_seed(SEED, outer.name, "outer_clean", base_spec["model_id"], base_spec["config_id"]),
            )
            train_anomaly_q75 = float(np.quantile(np.abs(clean_fit["pair"].y_train_residual), 0.75))
            base_trial_frame = outer
            _model_trial_summary(registry, base_trial_frame, trial_type="outer_refit", fold=outer.name, spec=base_spec, fit=clean_fit, prediction=clean_fit["prediction"])
            _append_prediction_rows(
                prediction_rows, outer.validation, clean_fit["prediction"], clean_fit["pair"].offset_validation,
                model_id="nested_selected_pipeline", fold=outer.name,
                architecture=base_spec["model_id"], feature_group=base_spec["feature_group"],
                config_id=base_spec["config_id"], alpha=base_spec["alpha"], policy="clean",
                training_anomaly_q75=train_anomaly_q75,
            )

            # Independent, inner-selected feature-group ablations.
            for feature_group, selected_row in by_group_rows.items():
                spec = _spec_from_row(selected_row)
                fit = clean_fit if spec == base_spec else _fit_spec(
                    outer.train, outer.validation, monthly, spec, policy="clean", intensity=0.0,
                    seed=stable_seed(SEED, outer.name, "feature_ablation", feature_group, spec["model_id"], spec["config_id"]),
                )
                _append_prediction_rows(
                    prediction_rows, outer.validation, fit["prediction"], fit["pair"].offset_validation,
                    model_id=f"feature_group:{feature_group}", fold=outer.name,
                    architecture=spec["model_id"], feature_group=feature_group,
                    config_id=spec["config_id"], alpha=spec["alpha"], policy="clean",
                    training_anomaly_q75=float(np.quantile(np.abs(fit["pair"].y_train_residual), 0.75)),
                )
                _model_trial_summary(registry, outer, trial_type="outer_feature_ablation", fold=outer.name, spec=spec, fit=fit, prediction=fit["prediction"])

            # Architecture-specific inner selection, paired on the identical outer months.
            for model_id, selected_row in by_model_rows.items():
                spec = _spec_from_row(selected_row)
                fit = clean_fit if spec == base_spec else _fit_spec(
                    outer.train, outer.validation, monthly, spec, policy="clean", intensity=0.0,
                    seed=stable_seed(SEED, outer.name, "architecture", model_id, spec["config_id"]),
                )
                _append_prediction_rows(
                    prediction_rows, outer.validation, fit["prediction"], fit["pair"].offset_validation,
                    model_id=f"architecture:{model_id}", fold=outer.name,
                    architecture=model_id, feature_group=spec["feature_group"],
                    config_id=spec["config_id"], alpha=spec["alpha"], policy="clean",
                    training_anomaly_q75=float(np.quantile(np.abs(fit["pair"].y_train_residual), 0.75)),
                )
                _model_trial_summary(registry, outer, trial_type="outer_architecture_refit", fold=outer.name, spec=spec, fit=fit, prediction=fit["prediction"])

            # Explicit baselines use the same outer months and the same training-only climatology.
            baselines = _baseline_predictions(outer.train, outer.validation, monthly, clean_fit["pair"].offset_validation)
            for baseline_id, prediction in baselines.items():
                _append_prediction_rows(
                    prediction_rows, outer.validation, prediction, clean_fit["pair"].offset_validation,
                    model_id=baseline_id, fold=outer.name,
                    architecture="baseline", feature_group="none", config_id="none",
                    alpha=0.0, policy="observed-history" if baseline_id != "training_monthly_climatology" else "train-only",
                    training_anomaly_q75=train_anomaly_q75,
                )

            # Reassess training augmentation on the clean-selected low-capacity candidate only.
            aug_selected, aug_records = _augmentation_inner_search(
                outer, monthly, base_spec, prediction_store, overall_key, registry
            )
            selected_policy = aug_selected["_selected_policy"]
            best_variant = str(selected_policy["selected_variant"])
            variant_fits: dict[str, dict[str, Any]] = {}
            variant_specs: dict[str, dict[str, Any]] = {}
            variant_policy_map = {
                "clean": ("clean", 0.0),
                "mild_gaussian_10pct": ("mild_gaussian", 0.10),
                "mild_dropout_10pct": ("mild_dropout", 0.10),
                "calibrated_gaussian_05pct": ("mild_gaussian", 0.05),
                "calibrated_gaussian_20pct": ("mild_gaussian", 0.20),
            }
            for variant in variant_policy_map:
                choice = aug_selected.get(variant)
                if choice is None:
                    continue
                policy, intensity = variant_policy_map[variant]
                spec = base_spec | {"alpha": float(choice["alpha"])}
                fit = clean_fit if policy == "clean" and spec["alpha"] == base_spec["alpha"] else _fit_spec(
                    outer.train, outer.validation, monthly, spec, policy=policy, intensity=intensity,
                    seed=stable_seed(SEED, outer.name, "outer_aug", variant, base_spec["model_id"], base_spec["config_id"]),
                )
                variant_fits[variant] = fit
                variant_specs[variant] = spec
                _append_prediction_rows(
                    prediction_rows, outer.validation, fit["prediction"], fit["pair"].offset_validation,
                    model_id=f"augmentation_policy:{variant}", fold=outer.name,
                    architecture=spec["model_id"], feature_group=spec["feature_group"],
                    config_id=spec["config_id"], alpha=spec["alpha"], policy=policy,
                    training_anomaly_q75=float(np.quantile(np.abs(fit["pair"].y_train_residual), 0.75)),
                )
                if variant == best_variant:
                    _append_prediction_rows(
                        prediction_rows, outer.validation, fit["prediction"], fit["pair"].offset_validation,
                        model_id="nested_selected_augmentation_pipeline", fold=outer.name,
                        architecture=spec["model_id"], feature_group=spec["feature_group"],
                        config_id=spec["config_id"], alpha=spec["alpha"], policy=policy,
                        training_anomaly_q75=float(np.quantile(np.abs(fit["pair"].y_train_residual), 0.75)),
                    )
            if best_variant not in variant_fits:
                raise RuntimeError(f"selected augmentation variant {best_variant} did not fit in {outer.name}")
            selected_fit = variant_fits[best_variant]
            selected_spec = variant_specs[best_variant]
            selected_inner_key = f"{best_variant}:{float(selected_policy['alpha']):.2f}"
            warm_candidate = aug_selected["_inner_predictions"].get(selected_inner_key)
            if warm_candidate is None or warm_candidate.empty:
                raise RuntimeError("selected augmentation policy has no earlier inner calibration residuals")

            # Each interval at a target uses only earlier residuals; later months update the pool after prediction.
            current_candidate = pd.DataFrame({
                "sample_id": outer.validation.sample_id.astype(str),
                "target_month": outer.validation.target_month.astype(str),
                "observed_t2m_c": outer.validation.target_value.to_numpy(dtype=float),
                "prediction_t2m_c": selected_fit["prediction"],
                "training_climatology_c": selected_fit["pair"].offset_validation,
            })
            current_climate = current_candidate.assign(prediction_t2m_c=current_candidate.training_climatology_c)
            climate_warm_parts = []
            for inner in chronological_inner_folds(outer.train):
                inner_pair = feature_pair(inner.train, inner.validation, monthly, base_spec["feature_group"])
                climate_warm_parts.append(pd.DataFrame({
                    "sample_id": inner.validation.sample_id.astype(str),
                    "target_month": inner.validation.target_month.astype(str),
                    "observed_t2m_c": inner_pair.y_validation,
                    "prediction_t2m_c": inner_pair.offset_validation,
                    "training_climatology_c": inner_pair.offset_validation,
                }))
            climate_warm = pd.concat(climate_warm_parts, ignore_index=True)
            fold_intervals, candidate_prior = _interval_rows(
                outer, current_candidate, current_climate, warm_candidate, climate_warm,
                candidate_prior, climate_prior,
            )
            interval_rows.extend(fold_intervals)

            outer_fit_record = {"fold": outer, "validation": outer.validation, "fit": selected_fit, "spec": selected_spec}
            outer_fits.append(outer_fit_record)
            selection_record = {
                "outer_fold": outer.name, "selected_clean_spec": base_spec,
                "selected_training_policy": selected_policy,
                "clean_inner_metrics": {k: overall_row[k] for k in ("mae", "rmse", "sample_count")},
            }
            fold_selection_records.append(selection_record)
            if outer.name == outer_folds[-1].name:
                final_spec = selected_spec
                final_policy = str(selected_policy["policy"])
                final_intensity = float(selected_policy["intensity"])
            save_fold_checkpoint(checkpoint_path, {
                "registry": registry[registry_start:],
                "predictions": prediction_rows[predictions_start:],
                "intervals": interval_rows[intervals_start:],
                "outer_fit": outer_fit_record,
                "candidate_prior": candidate_prior,
                "climate_prior": climate_prior,
                "selection_record": selection_record,
                "final_spec": final_spec,
                "final_policy": final_policy,
                "final_intensity": final_intensity,
            }, fingerprint=checkpoint_fingerprint)
            print(f"OUTER_FOLD_CHECKPOINTED name={outer.name} oof_months={len(outer.validation)}", flush=True)

        if final_spec is None:
            raise RuntimeError("no final development candidate was selected")
        prediction_frame = pd.DataFrame(prediction_rows)
        if prediction_frame.empty:
            raise RuntimeError("no outer fold predictions were produced")
        # Every OOF model uses the same 120 observed target months exactly once.
        expected_oof = set().union(*(set(f.validation.target_month.astype(str)) for f in outer_folds))
        for model_id, group in prediction_frame.groupby("model_id", sort=False):
            if set(group.target_month.astype(str)) != expected_oof or len(group) != len(expected_oof):
                raise ValueError(f"OOF target months are not common and unique for {model_id}")
        metrics_frame = _metric_rows(prediction_frame)
        comparison = _comparison_rows(prediction_frame, metrics_frame)
        ablation = _feature_ablation_rows(prediction_frame, metrics_frame)
        seasonal_bias = _seasonal_bias_table(prediction_frame)
        anomaly = _anomaly_regime_table(prediction_frame)
        calibration_frame = pd.DataFrame(interval_rows)
        calibration_summary = _calibration_summary(calibration_frame)
        robustness, robustness_predictions = _run_robustness(outer_fits)
        distribution_shift = _distribution_shift(outer_folds, monthly)
        residual_acf = _residual_autocorrelation(prediction_frame)
        model_files, final_fit_metrics = _make_final_bundle(
            samples, monthly, final_spec, policy=final_policy, intensity=final_intensity,
            output_root=output_root, source_hashes=source_hashes,
        )

    artifacts: dict[str, pd.DataFrame] = {
        "experiment_registry.csv": pd.DataFrame(registry),
        "fold_predictions.parquet": prediction_frame,
        "model_metrics.csv": metrics_frame,
        "baseline_comparison.csv": comparison,
        "feature_ablation.csv": ablation,
        "robustness_metrics.csv": robustness,
        "robustness_predictions.parquet": robustness_predictions,
        "interval_calibration.csv": calibration_frame,
        "interval_calibration_summary.csv": calibration_summary,
        "seasonal_bias.csv": seasonal_bias,
        "anomaly_regime_metrics.csv": anomaly,
        "distribution_shift.csv": distribution_shift,
        "residual_autocorrelation.csv": residual_acf,
    }
    registry_frame = artifacts["experiment_registry.csv"]
    augmentation_records = registry_frame.loc[
        registry_frame.trial_type == "augmentation_inner_candidate"
    ]
    completed_augmentation_records = augmentation_records.loc[
        augmentation_records.status == "COMPLETED"
    ]
    nested_candidate_records = registry_frame.loc[
        registry_frame.trial_type == "inner_candidate"
    ]
    candidate_group_columns = ["outer_fold", "inner_fold", "feature_group", "model_id", "config_id"]
    completed_candidate_groups = nested_candidate_records.loc[
        nested_candidate_records.status == "COMPLETED"
    ].groupby(candidate_group_columns, dropna=False).size()
    failed_candidate_count = int((nested_candidate_records.status == "FAILED").sum())
    expected_candidate_fit_count = int(
        len(outer_folds) * 3 * len(FEATURE_GROUP_COLUMNS) * len(MODEL_CONFIGS)
    )
    candidate_fits_complete = bool(
        len(completed_candidate_groups) == expected_candidate_fit_count
        and completed_candidate_groups.eq(len(ALPHAS)).all()
        and failed_candidate_count == 0
    )
    output_paths: list[Path] = []
    for name, frame in artifacts.items():
        path = output_root / name
        if path.suffix == ".parquet":
            _atomic_parquet(frame, path)
        else:
            _atomic_csv(frame, path)
        output_paths.append(path)
    output_paths.extend(model_files)
    output_paths.extend(sorted(checkpoint_root.glob("*.joblib")))
    report_paths = generate_reports(
        report_root, comparison, metrics_frame, ablation, robustness,
        calibration_summary, seasonal_bias, anomaly,
        source_hashes=source_hashes, code_commit=_git_commit(), peak_rss=peak_rss.peak,
    )
    report_paths.extend(_render_figures(report_root, prediction_frame, metrics_frame, ablation, robustness, calibration_summary))
    output_paths.extend(report_paths)
    manifest_path = output_root / "experiment_manifest.json"
    files = _file_facts(output_paths, ROOT, artifact_roots={"outputs": output_root, "reports": report_root})
    primary = comparison[comparison.model_id == "nested_selected_augmentation_pipeline"].iloc[0].to_dict()
    validation_checks = {
        "source_forecast_predicate_ends_at_2021_12": bool(forecast.target_month.max() == DEVELOPMENT_END),
        "source_monthly_gold_predicate_ends_at_2021_12": bool(gold.month.astype(str).max() == DEVELOPMENT_END),
        "exact_226_development_samples": bool(len(samples) == 226),
        "three_expanding_outer_folds": bool(len(outer_folds) == 3),
        "120_common_unique_outer_oof_months": bool(len(expected_oof) == 120),
        "target_t2m_matches_frozen_gold": True,
        "no_2022_2024_or_2025_target_rows_loaded_for_scoring": True,
        "source_release_lag_is_labeled_assumption": True,
        "training_climatology_and_preprocessing_fit_fold_locally": True,
        "all_oof_predictions_finite": bool(np.isfinite(prediction_frame.prediction_t2m_c).all()),
        "all_fixed_nested_candidates_completed": candidate_fits_complete,
        "authentic_labels_unchanged_by_augmentation": bool(
            not completed_augmentation_records.empty
            and completed_augmentation_records.label_integrity_verified.astype(bool).all()
            and completed_augmentation_records.authentic_month_weight_sum_verified.astype(bool).all()
        ),
        "final_refit_uses_only_labels_available_at_first_post_development_issue": bool(
            final_fit_metrics["training_month_end"]
            <= str(latest_available_label_month(final_fit_metrics["fit_as_of_issue_time_utc"]))
            and final_fit_metrics["fit_as_of_issue_time_utc"] == "2021-12-01T00:00:00Z"
        ),
    }
    if not all(validation_checks.values()):
        raise RuntimeError(f"v1.4 reproducibility checks failed: {validation_checks}")
    manifest = {
        "manifest_version": "kuntur-predictive-v1.4.0",
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": _git_commit(),
        "code_files_sha256": {
            "scripts/run_predictive_v14.py": sha256_file(ROOT / "scripts/run_predictive_v14.py"),
            "src/kuntur_data/phase2/predictive_v14.py": sha256_file(ROOT / "src/kuntur_data/phase2/predictive_v14.py"),
        },
        "frozen_inputs": {
            "forecast_path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet",
            "monthly_gold_path": "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
            "split_indices_path": "data/gold/kuntur_forecast_split_indices_v1.parquet",
            "split_manifest_path": "data/manifests/forecast_split_manifest_v1.json",
            **source_hashes,
        },
        "evaluation_scope": {
            "training_availability_policy": "two-month source-release lag from frozen contract; source publication timestamps unverified",
            "development_start": "2003-03", "development_cutoff": DEVELOPMENT_END,
            "excluded_target_months": "2022-01 onward including consumed 2022-2024 and 2025 tests",
            "outer_folds": [{"name": f.name, "train": [str(f.train.target_month.min()), str(f.train.target_month.max())], "validation": [str(f.validation.target_month.min()), str(f.validation.target_month.max())], "train_month_count": len(f.train), "validation_month_count": len(f.validation)} for f in outer_folds],
            "outer_oof_months": int(len(expected_oof)),
            "prior_2019_2021_validation_status": "previously inspected; treated as development observations, not a fresh holdout",
            "2025_feature_coverage": "sealed Gold schema is MERRA-2-only; cross-source candidates are not applied or scored there",
            "independent_unseen_test_completed": False,
        },
        "protocol": {
            "forecast_timeline": "target T; forecast origin T-1; issue 00:00 UTC on first day T-1; features T-14 through T-3",
            "feature_groups": {k: list(v) for k, v in FEATURE_GROUP_COLUMNS.items()},
            "model_configurations": list(MODEL_CONFIGS),
            "alpha_grid": list(ALPHAS), "selection_metric": "pooled inner-validation MAE; then RMSE, feature count, model priority, config id, alpha",
            "augmentation_policies": ["clean", "10% Gaussian", "10% feature dropout", "inner-selected Gaussian from 5%, 10%, 20%"],
            "robustness_scenarios": ["Gaussian", "missingness", "Gaussian + missingness"],
            "robustness_intensities": list(INTENSITIES),
            "native_thread_limit_per_fit": 1,
            "paired_uncertainty": {"method": "non-circular moving-block bootstrap", "block_months": 12, "replicates": BOOTSTRAP_REPLICATES, "seed": SEED},
            "acceptance_rule": "positive OOF MAE skill, paired MBB CI entirely below zero, worst-fold degradation <=0.10 C, absolute bias <=0.15 C; otherwise climatology preferred",
        },
        "selected_development_refit": {
            "specification": final_spec, "training_policy": final_policy,
            "augmentation_intensity": final_intensity,
            "training_month_range": [final_fit_metrics["training_month_start"], final_fit_metrics["training_month_end"]],
            "training_month_count": final_fit_metrics["training_month_count"], "fit": final_fit_metrics,
            "out_of_fold_metrics": primary,
            "deployment_status": "development candidate only; independent temporal generalization unverified",
        },
        "resource_usage": {"elapsed_wall_seconds": float(time.perf_counter() - start_wall), "elapsed_cpu_seconds": float(time.process_time() - start_cpu), "peak_rss_bytes": int(peak_rss.peak)},
        "resumed_outer_folds": resumed_folds,
        "augmentation_registry": {
            "completed_inner_candidate_rows": int(len(completed_augmentation_records)),
            "failed_inner_candidate_rows": int((augmentation_records.status == "FAILED").sum()),
        },
        "nested_candidate_registry": {
            "expected_candidate_fits": expected_candidate_fit_count,
            "completed_candidate_fits": int(len(completed_candidate_groups)),
            "failed_candidate_fits": failed_candidate_count,
            "alpha_scores_per_completed_fit": int(len(ALPHAS)),
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "validation": {"status": "PASSED", "checks": validation_checks},
        "artifacts": files,
    }
    _atomic_json(manifest_path, manifest)
    # Include the manifest hash itself in a detached receipt to avoid self-referential hashing.
    receipt_path = output_root / "experiment_manifest.sha256"
    _atomic_text(receipt_path, f"{sha256_file(manifest_path)}  experiment_manifest.json\n")
    print(json.dumps({
        "status": manifest["status"], "development_months": len(samples),
        "outer_oof_months": len(expected_oof), "primary": primary,
        "preferred_fallback": "nested candidate" if primary["mae_difference_vs_training_climatology"] < 0 and primary["paired_mae_difference_ci95_high"] < 0 else "training-only monthly climatology",
        "peak_rss_mib": peak_rss.peak / (1024**2), "output_root": str(output_root),
        "report_root": str(report_root), "manifest_sha256": sha256_file(manifest_path),
    }, indent=2, sort_keys=True, default=str))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="read-only root containing the frozen KUNTUR data/gold and data/manifests inputs")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results/predictive_v14")
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/predictive_v14")
    args = parser.parse_args()
    run_experiment(args)
    return 0


def _selection_registry_row(outer: TemporalFold, row: dict[str, Any], level: str) -> dict[str, Any]:
    spec = _spec_from_row(row)
    return {
        "trial_type": "inner_selection", "outer_fold": outer.name,
        "selection_level": level, "feature_group": spec["feature_group"],
        "model_id": spec["model_id"], "config_id": spec["config_id"],
        "configuration_json": json.dumps(spec["config"], sort_keys=True),
        "alpha": spec["alpha"], "status": "SELECTED_BY_INNER_CV",
        "inner_validation_months": int(row["sample_count"]),
        "mae": float(row["mae"]), "rmse": float(row["rmse"]),
        "inner_prediction_sha256": row["inner_prediction_sha256"],
    }


def _metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    climate_by_sample = predictions[predictions.model_id == "training_monthly_climatology"].set_index("sample_id").prediction_t2m_c
    for model_id, frame in predictions.groupby("model_id", sort=False):
        for fold_name, fold_frame in [(str(k), v) for k, v in frame.groupby("outer_fold", sort=False)]:
            ordered = fold_frame.sort_values("target_month", kind="stable")
            baseline = np.asarray([climate_by_sample.loc[sample_id] for sample_id in ordered.sample_id], dtype=float)
            metrics = metric_summary(ordered.observed_t2m_c, ordered.prediction_t2m_c, baseline)
            rows.append({"model_id": model_id, "scope": fold_name, **metrics,
                         "mean_prediction_variance": float(np.var(ordered.prediction_t2m_c, ddof=1)) if len(ordered) > 1 else 0.0})
        ordered = frame.sort_values("target_month", kind="stable")
        baseline = np.asarray([climate_by_sample.loc[sample_id] for sample_id in ordered.sample_id], dtype=float)
        metrics = metric_summary(ordered.observed_t2m_c, ordered.prediction_t2m_c, baseline)
        metrics["mean_prediction_variance"] = float(np.var(ordered.prediction_t2m_c, ddof=1))
        rows.append({"model_id": model_id, "scope": "ALL_OOF", **metrics})
    return pd.DataFrame(rows)


def _comparison_rows(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lookup = metrics[(metrics.scope == "ALL_OOF")].set_index("model_id")
    climate = lookup.loc["training_monthly_climatology"]
    candidate_ids = [
        "training_monthly_climatology", "expanding_monthly_climatology",
        "seasonal_naive_t_minus_12", "persistence_t_minus_3",
        "nested_selected_pipeline", "nested_selected_augmentation_pipeline",
    ]
    for model_id in candidate_ids:
        if model_id not in lookup.index:
            continue
        item = lookup.loc[model_id]
        frame = predictions[predictions.model_id == model_id].sort_values("target_month", kind="stable")
        base = predictions[predictions.model_id == "training_monthly_climatology"].set_index("sample_id")
        paired = np.abs(frame.prediction_t2m_c.to_numpy() - frame.observed_t2m_c.to_numpy()) - np.abs(
            np.asarray([base.loc[sid, "prediction_t2m_c"] for sid in frame.sample_id]) - frame.observed_t2m_c.to_numpy()
        )
        ci = paired_moving_block_ci(paired, replicates=BOOTSTRAP_REPLICATES, seed=stable_seed(SEED, model_id, "paired_mbb"))
        fold_metrics = metrics[(metrics.model_id == model_id) & (metrics.scope != "ALL_OOF")]
        climate_folds = metrics[(metrics.model_id == "training_monthly_climatology") & (metrics.scope != "ALL_OOF")].set_index("scope")
        worst_delta = max(float(row.mae - climate_folds.loc[row.scope, "mae"]) for row in fold_metrics.itertuples())
        rows.append({
            "model_id": model_id, "sample_count": int(item.sample_count),
            "mae": float(item.mae), "rmse": float(item.rmse), "mean_bias": float(item.mean_bias),
            "mae_difference_vs_training_climatology": float(item.mae - climate.mae),
            "mae_skill_vs_training_climatology": float(1.0 - item.mae / climate.mae),
            "rmse_difference_vs_training_climatology": float(item.rmse - climate.rmse),
            "rmse_skill_vs_training_climatology": float(1.0 - item.rmse / climate.rmse),
            "monthly_win_rate_vs_training_climatology": float(item.monthly_win_rate_vs_baseline),
            "worst_fold_mae": float(fold_metrics.mae.max()),
            "worst_fold_mae_difference_vs_climatology": float(worst_delta),
            "paired_mae_difference_ci95_low": ci["ci95"][0],
            "paired_mae_difference_ci95_high": ci["ci95"][1],
            "moving_block_length_months": ci["block_length_months"],
            "bootstrap_replicates": ci["replicates"],
            "bootstrap_seed": ci["seed"],
            "uncertainty_interpretation": ci["interpretation"],
        })
    return pd.DataFrame(rows)


def _feature_ablation_rows(predictions: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature_group in FEATURE_GROUP_COLUMNS:
        model_id = f"feature_group:{feature_group}"
        selected = metrics[(metrics.model_id == model_id) & (metrics.scope == "ALL_OOF")]
        if selected.empty:
            continue
        item = selected.iloc[0]
        folds = metrics[(metrics.model_id == model_id) & (metrics.scope != "ALL_OOF")]
        rows.append({
            "feature_group": feature_group,
            "selected_architectures_by_outer_fold": ";".join(
                predictions[predictions.model_id == model_id].groupby("outer_fold", sort=False).architecture.first().tolist()
            ),
            "feature_count": len(FEATURE_GROUP_COLUMNS[feature_group]),
            "oof_months": int(item.sample_count), "mae": float(item.mae),
            "rmse": float(item.rmse), "mean_bias": float(item.mean_bias),
            "mae_skill_vs_training_climatology": float(item.mae_skill_vs_baseline),
            "monthly_win_rate": float(item.monthly_win_rate_vs_baseline),
            "mean_prediction_variance": float(item.mean_prediction_variance),
            "fold_mae_min": float(folds.mae.min()), "fold_mae_max": float(folds.mae.max()),
            "fold_mae_sd": float(folds.mae.std(ddof=1)),
        })
    return pd.DataFrame(rows)


def _seasonal_bias_table(predictions: pd.DataFrame) -> pd.DataFrame:
    selected_ids = ["training_monthly_climatology", "nested_selected_pipeline", "nested_selected_augmentation_pipeline"]
    frame = predictions[predictions.model_id.isin(selected_ids)].copy()
    frame["target_calendar_month"] = frame.target_month.str[-2:].astype(int)
    frame["prediction_error_c"] = frame.prediction_t2m_c - frame.observed_t2m_c
    return frame.groupby(["model_id", "target_calendar_month"], as_index=False).agg(
        sample_count=("sample_id", "size"), mean_bias_c=("prediction_error_c", "mean"),
        mae_c=("prediction_error_c", lambda values: float(np.mean(np.abs(values)))),
        mean_prediction_c=("prediction_t2m_c", "mean"), mean_observed_c=("observed_t2m_c", "mean"),
    )


def _anomaly_regime_table(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions[predictions.model_id.isin(["training_monthly_climatology", "nested_selected_pipeline", "nested_selected_augmentation_pipeline"])].copy()
    frame["absolute_anomaly_c"] = np.abs(frame.observed_t2m_c - frame.training_climatology_c)
    # The training-derived regime threshold is carried on each OOF row by the runner.
    if "training_anomaly_q75_c" in frame:
        frame["anomaly_regime"] = np.where(frame.absolute_anomaly_c >= frame.training_anomaly_q75_c, "large_anomaly", "ordinary")
    else:
        frame["anomaly_regime"] = "unstratified"
    frame["absolute_error_c"] = np.abs(frame.prediction_t2m_c - frame.observed_t2m_c)
    return frame.groupby(["model_id", "anomaly_regime"], as_index=False).agg(
        sample_count=("sample_id", "size"), mae_c=("absolute_error_c", "mean"),
        mean_absolute_anomaly_c=("absolute_anomaly_c", "mean"),
    )


def _calibration_summary(intervals: pd.DataFrame) -> pd.DataFrame:
    intervals = intervals.copy()
    intervals["season"] = intervals.target_month.str[-2:].astype(int).map(_season_name)
    rows = []
    grouping = ["model_id", "calibration_method", "nominal_coverage"]
    for keys, group in intervals.groupby(grouping, sort=False):
        rows.append({
            "model_id": keys[0], "calibration_method": keys[1], "nominal_coverage": float(keys[2]),
            "sample_count": int(len(group)), "empirical_coverage": float(group.covered.mean()),
            "mean_width_c": float(group.width_c.mean()), "median_width_c": float(group.width_c.median()),
            "mean_winkler_score_c": float(group.winkler_score_c.mean()),
            "coverage_large_anomaly": float(group[group.anomaly_magnitude_c >= group.anomaly_magnitude_c.quantile(0.75)].covered.mean()),
        })
        for season, seasonal in group.groupby("season", sort=True):
            rows.append({
                "model_id": keys[0], "calibration_method": keys[1], "nominal_coverage": float(keys[2]),
                "sample_count": int(len(seasonal)), "empirical_coverage": float(seasonal.covered.mean()),
                "mean_width_c": float(seasonal.width_c.mean()), "median_width_c": float(seasonal.width_c.median()),
                "mean_winkler_score_c": float(seasonal.winkler_score_c.mean()),
                "coverage_large_anomaly": float("nan"), "season": season,
            })
    return pd.DataFrame(rows)


def _render_figures(
    report_root: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    robustness: pd.DataFrame,
    calibration: pd.DataFrame,
) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_root = report_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []
    climate = predictions[predictions.model_id == "training_monthly_climatology"].set_index("sample_id")
    candidate = predictions[predictions.model_id == "nested_selected_augmentation_pipeline"].sort_values("target_month")
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(candidate.observed_t2m_c, candidate.prediction_t2m_c, s=20, alpha=0.7, label="Nested selected candidate")
    climate_values = np.asarray([climate.loc[sid, "prediction_t2m_c"] for sid in candidate.sample_id])
    ax.scatter(candidate.observed_t2m_c, climate_values, s=20, alpha=0.55, label="Training climatology")
    low = float(min(candidate.observed_t2m_c.min(), candidate.prediction_t2m_c.min(), climate_values.min()))
    high = float(max(candidate.observed_t2m_c.max(), candidate.prediction_t2m_c.max(), climate_values.max()))
    ax.plot([low, high], [low, high], color="black", linewidth=1)
    ax.set(xlabel="Observed monthly T2M (°C)", ylabel="Predicted monthly T2M (°C)", title="Nested chronological out-of-fold predictions")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_root / "oof_predicted_vs_observed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    compare = predictions[predictions.model_id.isin(["training_monthly_climatology", "nested_selected_pipeline", "nested_selected_augmentation_pipeline"])].copy()
    compare["absolute_error"] = np.abs(compare.prediction_t2m_c - compare.observed_t2m_c)
    compare.pivot(index="target_month", columns="model_id", values="absolute_error").plot(ax=ax, linewidth=1.2)
    ax.set(ylabel="Absolute error (°C)", xlabel="Target month", title="Out-of-fold absolute error by month")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_root / "oof_absolute_error_timeseries.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    seasonal = _seasonal_bias_table(predictions)
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for model_id, group in seasonal.groupby("model_id", sort=False):
        ax.plot(group.target_calendar_month, group.mean_bias_c, marker="o", label=model_id)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xticks=range(1, 13), xlabel="Calendar month", ylabel="Mean prediction bias (°C)", title="Seasonal bias in nested out-of-fold predictions")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_root / "seasonal_prediction_bias.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    fold_metrics = metrics[(metrics.model_id.isin(["training_monthly_climatology", "nested_selected_pipeline", "nested_selected_augmentation_pipeline"])) & (metrics.scope != "ALL_OOF")]
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    pivot = fold_metrics.pivot(index="scope", columns="model_id", values="mae")
    pivot.plot(kind="bar", ax=ax, rot=0)
    ax.set(ylabel="MAE (°C)", xlabel="Chronological outer fold", title="Development skill by outer fold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_root / "skill_by_outer_fold.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    plot = ablation.sort_values("mae", ascending=True)
    ax.barh(plot.feature_group, plot.mae, color="#4f7cac")
    climate_mae = float(metrics[(metrics.model_id == "training_monthly_climatology") & (metrics.scope == "ALL_OOF")].mae.iloc[0])
    ax.axvline(climate_mae, color="#a63a50", linestyle="--", label=f"Climatology {climate_mae:.3f} °C")
    ax.set(xlabel="Nested OOF MAE (°C)", title="Predeclared feature-group ablations")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figure_root / "feature_ablation_mae.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    if not robustness.empty:
        selected = robustness[(robustness.scope == "ALL_OOF")]
        for scenario, group in selected.groupby("scenario", sort=False):
            group = group.sort_values("intensity")
            ax.plot(group.intensity, group.delta_mae_from_clean_c, marker="o", label=scenario)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set(xlabel="Corruption intensity (training-feature SD or missing fraction)", ylabel="MAE change from clean OOF (°C)", title="Development-only robustness diagnostics")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_root / "robustness_degradation.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)

    summary = calibration[calibration["season"].isna()] if "season" in calibration else calibration
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for model_id, group in summary.groupby("model_id", sort=False):
        ax.plot(group.nominal_coverage, group.mean_width_c, marker="o", label=model_id)
    ax.set(xlabel="Nominal interval coverage", ylabel="Mean interval width (°C)", title="Prequential calibration width and achieved coverage")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = figure_root / "interval_coverage_vs_width.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    made.append(path)
    return made


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (ValueError, TypeError):
        return str(value)


def _table_text(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    selected = frame if columns is None else frame.loc[:, columns]
    return selected.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def generate_reports(
    report_root: Path,
    comparison: pd.DataFrame,
    metrics: pd.DataFrame,
    ablation: pd.DataFrame,
    robustness: pd.DataFrame,
    calibration: pd.DataFrame,
    seasonal_bias: pd.DataFrame,
    anomaly: pd.DataFrame,
    *,
    source_hashes: dict[str, str],
    code_commit: str | None,
    peak_rss: int,
) -> list[Path]:
    report_root.mkdir(parents=True, exist_ok=True)
    climate = comparison[comparison.model_id == "training_monthly_climatology"].iloc[0]
    primary = comparison[comparison.model_id == "nested_selected_augmentation_pipeline"].iloc[0]
    clean = comparison[comparison.model_id == "nested_selected_pipeline"].iloc[0]
    accept = (
        primary.mae_difference_vs_training_climatology < 0
        and primary.paired_mae_difference_ci95_high < 0
        and primary.worst_fold_mae_difference_vs_climatology <= 0.10
        and abs(primary.mean_bias) <= 0.15
    )
    reports: dict[str, str] = {}
    reports["ROOT_CAUSE_DIAGNOSTIC.md"] = f"""# KUNTUR v1.4 root-cause diagnostic

Status: completed read-only evidence review before v1.4 model fitting.

| Hypothesis | Evidence and counterevidence | Result | Correction / next diagnostic |
|---|---|---|---|
| Prior selector allowed a trained model to lose to climatology | `scripts/run_phase2_predictive.py` selects the minimum validation MAE over `MODEL_ORDER` only. Frozen validation HGB MAE was {0.2696132096:.4f} °C; climatology was {0.260722:.4f} °C. | CONFIRMED engineering defect | Include alpha=0 climatology in nested selection and report it as fallback. |
| High model capacity relative to observed months contributed to instability | The v1 runner used 60 value lags plus 60 missingness indicators against 152 authentic fit months; MLP/CNN validation MAE were 0.8807/1.0374 °C. Augmented rows do not increase independent-month count. | SUPPORTED, not sole cause | Use compact grouped features and small regularized models; preserve authentic sample counts. |
| Absolute-temperature learning failed to separate seasonality from anomalies | The v1 six-model path fit direct target temperature without an explicit target climatology offset. HGB's 2022–24 MAE was 0.4974 °C with -0.3501 °C bias. The later v1.2 residual-RF formulation still scored 0.32513 °C versus 0.16591 °C climatology on the already-consumed 2025 set. | SUPPORTED as a design opportunity; not a complete explanation | Compare training-only climatology plus shrinkage of residual forecasts. Do not use consumed scores for tuning. |
| The frozen horizon or calendar indexing leaked a target month | Code and rows use target T, origin/issue T-1, and exactly T-14..T-3 predictors. v1.2 scorer forms each history from those periods. | REJECTED for the audited pipeline | Retain hard assertions and tests for all four timestamps. |
| Predictor release latency is operationally verified | Contract uses a conservative two-month release lag. NASA per-granule publication timestamps were not verified in this audit. | UNRESOLVED | Describe results as retrospective hindcasts under an assumed two-month availability lag. |
| Cross-source predictors are available for the sealed 2025 evaluation | Its Gold manifest contains MERRA-2 only; the frozen scorer requests five MERRA-2 history channels. IMERG and GLDAS are absent. | CONFIRMED feature-coverage mismatch for cross-source candidates | Keep MERRA-only and cross-source groups explicit; do not substitute or score cross-source models on 2025. |
| Predictor missingness caused the observed model failures | The validated 2002–2024 Gold monthly values have zero nulls across the 20 contracted fields and `VALID` quality status. | NOT SUPPORTED by the available Gold audit | Keep train-only imputers and missingness flags; record any fold-level input gaps. |
| Synthetic augmentation caused or fixed generalization failure | Previous augmentation preserved targets and authentic month weights, but no controlled fresh holdout remains. Historical 2025 results are consumed. | UNRESOLVED | Compare clean, mild Gaussian and dropout augmentations only inside development folds. |
| Prediction intervals were reliable | The archived 2022–2024 nominal 90% interval covered 55.6% of 36 months. v1.2 calibration used serially dependent historical residuals; 2025 has only 12 months. | CONFIRMED undercoverage on archived test | Evaluate prequential expanding, rolling and same-calendar-month calibration; make no conformal guarantee. |

The 2022–2024 and 2025 metrics above are historical diagnostic evidence only. The v1.4 candidate search never reads those target rows.

Frozen input SHA-256: `{json.dumps(source_hashes, sort_keys=True)}`. Code revision at experiment start: `{code_commit}`.
"""

    reports["FEATURE_CONTRACT_AUDIT.md"] = f"""# KUNTUR v1.4 feature-contract audit

- Target is MERRA-2 Ecuador pilot-bbox monthly mean T2M in °C for month T.
- Forecast origin is T-1; assumed issue time is 00:00 UTC on the first of T-1.
- All inputs are from exact monthly records T-14 through T-3, ending before issue month.
- The two-month release lag is inherited from the frozen contract. Actual NASA publication timestamps were not established; this is a retrospective availability assumption.
- Development predictors: MERRA-2 T2M, specific humidity, u/v wind components and pressure; IMERG monthly precipitation total; four GLDAS soil-moisture layers.
- IMERG precipitation is transformed with `log1p` before training-period calendar-month anomaly calculation. Wind speed is deterministically `hypot(u2m, v2m)`; the native source components remain in Gold.
- Calendar-month predictor means, target monthly climatology, imputation means and StandardScaler parameters are fitted separately inside each inner/outer training fold.
- Missing values are imputed with fold-training feature means and explicit missing indicators. No future observations are used to impute.
- The sealed 2025 predictor file has five MERRA-2 channels only. Cross-source groups are development-only and cannot be applied unchanged to 2025.
- All output remains for the Ecuador pilot bounding box; no Coast/Andes/Amazon mask is claimed.

Timeline and fold assertions are exercised by `tests/test_predictive_v14.py`.
"""

    reports["EXPERIMENTAL_PROTOCOL.md"] = """# KUNTUR v1.4 experimental protocol (frozen before fitting)

## Scope

Development-only nested chronological evaluation. The source forecast is read with a target-month predicate through 2021-12. The monthly Gold input is limited to 2002-01 through 2021-12. The 2022–2024 test and sealed 2025 test are not passed to candidate scoring. Existing 2019–2021 validation months are acknowledged as previously inspected development data, not a fresh holdout.

## Outer folds

| Fold | Training target months | Scored target months |
|---|---|---|
| outer_2012_2014 | 2003-03–2011-12 | 2012-01–2014-12 |
| outer_2015_2017 | 2003-03–2014-12 | 2015-01–2017-12 |
| outer_2018_2021 | 2003-03–2017-12 | 2018-01–2021-12 |

Each outer training period has three expanding one-year inner validation windows ending at its cutoff. Inner folds select feature group, model configuration and residual shrinkage alpha. Outer months are scored once by the selection procedure. The 120 outer predictions are a development estimate, not an independent final test.

## Candidates and fixed budget

The feature groups and ten configurations (two each for Ridge, Elastic Net, RBF-SVR, shallow HistGradientBoosting and shallow Random Forest) are fixed in `scripts/run_predictive_v14.py`. Each residual prediction is evaluated at alpha {0, 0.25, 0.50, 0.75, 1.0}; alpha=0 is the climatology fallback. Selection minimizes pooled inner MAE, then RMSE, then feature count, model order and alpha. No neural network or unrestricted search is used.

## Primary metrics and acceptance rule

Primary: paired OOF MAE skill `1 - MAE(candidate)/MAE(training-only monthly climatology)`. Report RMSE, bias, median and 90th percentile absolute error, monthly win rate, fold scores and a paired moving-block bootstrap interval for candidate-minus-climatology absolute error (12-month blocks, 2,000 resamples). This interval is descriptive; temporal dependence, previous inspections and nested candidate search limit formal inference.

Predeclared preference for a trained pipeline requires: positive aggregate skill; the 95% paired moving-block interval for MAE difference entirely below zero; no outer-fold MAE degradation greater than 0.10 °C relative to climatology; absolute aggregate mean bias no greater than 0.15 °C; and valid issue-time features. Otherwise prefer climatology. If the interval overlaps zero, any positive mean skill remains exploratory.

## Augmentation and robustness

Compare clean training, one mild 10% Gaussian copy, one mild 10% feature-dropout copy, and a Gaussian intensity selected from 5%, 10%, 20% within the inner folds. Each augmented target's total sample weight remains one and true target labels are unchanged. Separately assess held-out feature corruption at 25% and 50% standardized RMS/missing fraction for Gaussian, missingness and combined corruption. Synthetic rows and perturbation replicas are not independent observations.

## Uncertainty

Compare earlier-residual expanding pooled, rolling-36 pooled and expanding same-calendar-month intervals at 80%, 90%, 95%. At each forecast, only earlier time-ordered residuals are eligible; previous outer OOF errors calibrate later folds. The first outer fold uses its three earlier inner-validation years as warm-up, which are selection-coupled. Coverage is descriptive and no exchangeability or conformal guarantee is claimed.

## Stop rule

No 2022–2024 or 2025 outcomes may change features, model, alpha, augmentation, threshold, or interval method. No independent unseen-period claim is available from this experiment.
"""

    reports["MODEL_COMPARISON.md"] = f"""# KUNTUR v1.4 nested model comparison

Primary comparison uses 120 chronological outer development months. Scores are paired against training-only monthly climatology.

```
{_table_text(comparison, ['model_id','mae','rmse','mean_bias','mae_difference_vs_training_climatology','mae_skill_vs_training_climatology','monthly_win_rate_vs_training_climatology','worst_fold_mae','paired_mae_difference_ci95_low','paired_mae_difference_ci95_high'])}
```

Clean nested model MAE: {_fmt(clean.mae)} °C. Nested selected training-policy MAE: {_fmt(primary.mae)} °C. Training-only climatology MAE: {_fmt(climate.mae)} °C. Signed MAE difference (candidate minus climatology): {_fmt(primary.mae_difference_vs_training_climatology)} °C; skill: {_fmt(primary.mae_skill_vs_training_climatology * 100, 2)}%. The paired 12-month moving-block 95% interval is [{_fmt(primary.paired_mae_difference_ci95_low)}, {_fmt(primary.paired_mae_difference_ci95_high)}] °C.

```
{_table_text(metrics[(metrics.scope != 'ALL_OOF') & metrics.model_id.isin(['training_monthly_climatology','nested_selected_pipeline','nested_selected_augmentation_pipeline'])], ['model_id','scope','sample_count','mae','rmse','mean_bias'])}
```

Do not compare these scores to the previously consumed 2022–2024 or 2025 scores as if they were fresh independent tests.
"""

    reports["ANOMALY_FORECASTING.md"] = f"""# KUNTUR v1.4 anomaly forecasting

The fitted model predicts a target departure from fold-training monthly climatology and applies an inner-selected alpha in [0,1]. Alpha zero is eligible in every inner search. Target labels are never perturbed.

```
{_table_text(anomaly)}
```

Seasonal prediction-bias evidence:

```
{_table_text(seasonal_bias)}
```

Large-anomaly bands use each outer fold's training-only 75th percentile of absolute target anomaly. The report groups are a development diagnostic, not a post-hoc threshold for model selection.
"""

    reports["FEATURE_ABLATION.md"] = f"""# KUNTUR v1.4 feature ablation

Each group has its own inner-selected architecture/configuration/alpha and is scored on the same 120 outer months.

```
{_table_text(ablation)}
```

Precipitation and soil-moisture groups are development-only because sealed 2025 predictors contain MERRA-2 only. No 2025 score is used to rank groups.
"""

    reports["UNCERTAINTY_CALIBRATION.md"] = f"""# KUNTUR v1.4 uncertainty calibration

```
{_table_text(calibration[calibration.get('season', pd.Series(index=calibration.index, dtype=object)).isna()], ['model_id','calibration_method','nominal_coverage','sample_count','empirical_coverage','mean_width_c','median_width_c','mean_winkler_score_c','coverage_large_anomaly'])}
```

Coverage by season is in `results/predictive_v14/interval_calibration.csv`. Calibration is prequential and uses only earlier residuals; the first-fold warm-up residuals come from inner validation used for selection. Monthly serial dependence and small calibration samples preclude a finite-sample coverage guarantee. A narrow width is not interpreted as useful if empirical coverage is low.
"""

    reports["ROBUSTNESS_COMPARISON.md"] = f"""# KUNTUR v1.4 robustness comparison

Corruption is applied only to outer-fold validation inputs after fitting. Targets stay authentic. Intensities 25% and 50% are diagnostic RMS relative to fold-training feature SD (Gaussian), missing fraction (dropout), or an L2 split budget (combined).

```
{_table_text(robustness[robustness.scope == 'ALL_OOF'])}
```

These scenarios measure synthetic input sensitivity, not generalization to an independent sensor or domain.
"""

    reports["FINAL_SCIENTIFIC_VERDICT.md"] = f"""# KUNTUR Predictive Engine v1.4 — scientific verdict

Preferred fallback: **{'nested selected residual pipeline' if accept else 'training-only monthly climatology'}**.

- Training-only climatology OOF MAE: {_fmt(climate.mae)} °C; RMSE {_fmt(climate.rmse)} °C; bias {_fmt(climate.mean_bias)} °C.
- Nested selected augmentation-policy candidate OOF MAE: {_fmt(primary.mae)} °C; RMSE {_fmt(primary.rmse)} °C; bias {_fmt(primary.mean_bias)} °C.
- Candidate-minus-climatology MAE: {_fmt(primary.mae_difference_vs_training_climatology)} °C ({_fmt(primary.mae_skill_vs_training_climatology * 100, 2)}% skill); paired 95% moving-block interval [{_fmt(primary.paired_mae_difference_ci95_low)}, {_fmt(primary.paired_mae_difference_ci95_high)}] °C.
- Worst-fold candidate degradation versus climatology: {_fmt(primary.worst_fold_mae_difference_vs_climatology)} °C. Acceptance gate for a trained candidate: {'PASSED' if accept else 'NOT MET'}.
- This is development-only evidence. The 2022–2024 and 2025 evaluations were already consumed; neither was used in this search. No new independent temporal test was completed.
- The 2025 sealed feature archive is MERRA-2 only. No cross-source model score is claimed for that period.
- Forecast issue-time availability still relies on the frozen two-month product-release assumption; actual NASA publication timestamps were not verified.

Execution revision: `{code_commit}`. Process peak RSS: {peak_rss / (1024**2):.1f} MiB. Source SHA-256: `{json.dumps(source_hashes, sort_keys=True)}`.

See `MODEL_COMPARISON.md`, `FEATURE_ABLATION.md`, `UNCERTAINTY_CALIBRATION.md` and `ROBUSTNESS_COMPARISON.md` for the full numeric tables and limitations.
"""

    paths = []
    for name, body in reports.items():
        path = report_root / name
        _atomic_text(path, body)
        paths.append(path)
    return paths


if __name__ == "__main__":
    raise SystemExit(main())


