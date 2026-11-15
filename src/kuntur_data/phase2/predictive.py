"""Small-sample, leakage-safe forecasting benchmark helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import psutil
import sklearn
from scipy.stats import bootstrap
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .augmentation import (
    CHANNELS,
    COMBINED,
    FEATURE_COLUMNS,
    MISSING_COLUMNS,
    PHYSICAL_RANGES,
    SINGLE_FAMILIES,
    history_array,
    perturb_sample,
    training_augmentation,
    training_location_scale,
)


ROBUSTNESS_LEVELS = (0.25, 0.50)
MODEL_ORDER = ("ridge", "svr_rbf", "hist_gradient_boosting", "random_forest", "mlp", "temporal_cnn")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regression_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(y) & np.isfinite(pred)
    invalid = int((~np.isfinite(pred)).sum())
    if not valid.any():
        return {"sample_count": 0, "invalid_prediction_count": invalid, "mae": float("nan"), "rmse": float("nan"), "mean_bias": float("nan")}
    error = pred[valid] - y[valid]
    return {
        "sample_count": int(valid.sum()),
        "invalid_prediction_count": invalid,
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_bias": float(np.mean(error)),
    }


def bootstrap_interval(values: np.ndarray, *, seed: int = 0, n_resamples: int = 2000) -> tuple[float, float]:
    """Percentile bootstrap CI over independent target months, not noisy replicas."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    stats = np.mean(values[indices], axis=1)
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def _candidate_grid() -> dict[str, list[dict[str, Any]]]:
    return {
        "ridge": [{"alpha": x} for x in (0.1, 1.0, 10.0)],
        "random_forest": [{"n_estimators": n, "max_depth": d, "min_samples_leaf": 3} for n in (100, 200) for d in (2, 4)],
        "hist_gradient_boosting": [{"learning_rate": lr, "max_iter": it, "max_leaf_nodes": 7, "l2_regularization": 1.0} for lr in (0.05, 0.1) for it in (100, 150)],
        "svr_rbf": [{"C": c, "epsilon": 0.1, "gamma": "scale"} for c in (0.1, 1.0, 10.0)],
        "mlp": [{"hidden_layer_sizes": (n,), "alpha": a, "max_iter": 300, "early_stopping": False} for n in (16, 32) for a in (0.001, 0.01)],
        "temporal_cnn": [{"channels": c, "learning_rate": lr, "max_epochs": 100, "patience": 12} for c in (8, 16) for lr in (0.001, 0.003)],
    }


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


class TemporalCNN:
    """Tiny CPU CNN over 12 x (5 values + 5 missing indicators)."""

    def __init__(self, channels: int, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        class Net(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.conv1 = nn.Conv1d(10, width, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(width, width, kernel_size=3, padding=1)
                self.dropout = nn.Dropout(p=0.1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.output = nn.Linear(width, 1)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.dropout(torch.relu(self.conv2(x)))
                return self.output(self.pool(x).squeeze(-1)).squeeze(-1)

        self.channels = channels
        self.seed = seed
        self.torch = torch
        self.model = Net(channels)
        self.model.cpu()

    @staticmethod
    def to_tensor(x: np.ndarray):
        import torch
        arr = np.asarray(x, dtype=np.float32)
        values = arr[:, :60].reshape(-1, 12, 5)
        missing = arr[:, 60:].reshape(-1, 12, 5)
        return torch.from_numpy(np.concatenate([values, missing], axis=2).transpose(0, 2, 1))

    def fit(self, x: np.ndarray, y: np.ndarray, weight: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, *, learning_rate: float, max_epochs: int, patience: int, seed: int) -> int:
        torch = self.torch
        rng = np.random.default_rng(seed)
        xt = self.to_tensor(x)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
        wt = torch.as_tensor(np.asarray(weight, dtype=np.float32))
        xvt = self.to_tensor(x_val)
        yvt = torch.as_tensor(np.asarray(y_val, dtype=np.float32))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-3)
        best_loss = float("inf")
        best_state = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        stale = 0
        batch_size = 256
        for epoch in range(max_epochs):
            self.model.train()
            order = rng.permutation(len(yt))
            for start in range(0, len(order), batch_size):
                ids = order[start : start + batch_size]
                optimizer.zero_grad(set_to_none=True)
                pred = self.model(xt[ids])
                per_sample = (pred - yt[ids]) ** 2
                loss = torch.sum(per_sample * wt[ids]) / torch.clamp(wt[ids].sum(), min=1e-12)
                loss.backward()
                optimizer.step()
            self.model.eval()
            with torch.no_grad():
                val_loss = float(torch.mean((self.model(xvt) - yvt) ** 2).item())
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        self.model.load_state_dict(best_state)
        return best_epoch

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            return self.model(self.to_tensor(x)).cpu().numpy().astype(np.float64)

    def state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}


def fit_candidate(
    architecture: str,
    config: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, dict]:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    with PeakRSS() as monitor, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if architecture == "ridge":
            model = Ridge(alpha=config["alpha"]).fit(x_train, y_train, sample_weight=weight)
            epochs = None
        elif architecture == "random_forest":
            model = RandomForestRegressor(**config, random_state=seed, n_jobs=1).fit(x_train, y_train, sample_weight=weight)
            epochs = None
        elif architecture == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(**config, random_state=seed, early_stopping=False).fit(x_train, y_train, sample_weight=weight)
            epochs = None
        elif architecture == "svr_rbf":
            model = SVR(kernel="rbf", **config).fit(x_train, y_train, sample_weight=weight)
            epochs = None
        elif architecture == "mlp":
            model = MLPRegressor(**config, solver="adam", batch_size=128, learning_rate_init=0.001, tol=1e-4, n_iter_no_change=30, random_state=seed, shuffle=True).fit(x_train, y_train, sample_weight=weight)
            epochs = int(model.n_iter_)
        elif architecture == "temporal_cnn":
            model = TemporalCNN(channels=config["channels"], seed=seed)
            epochs = model.fit(x_train, y_train, weight, x_val, y_val, learning_rate=config["learning_rate"], max_epochs=config["max_epochs"], patience=config["patience"], seed=seed)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    info = {
        "fit_wall_seconds": time.perf_counter() - start_wall,
        "fit_cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_bytes": int(monitor.peak),
        "epochs_or_iterations": epochs,
        "converged_within_declared_budget": (bool(epochs < int(config.get("max_iter", epochs + 1))) if architecture == "mlp" else None),
        "warnings": [f"{type(item.message).__name__}: {str(item.message)[:240]}" for item in caught],
    }
    return model, info


def predict(model: Any, architecture: str, x: np.ndarray) -> np.ndarray:
    if architecture == "temporal_cnn":
        return model.predict(x)
    return np.asarray(model.predict(x), dtype=np.float64)


def model_inputs(raw: np.ndarray, missing: np.ndarray, location: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).copy()
    missing = np.asarray(missing, dtype=bool)
    impute = np.broadcast_to(location.reshape(1, 1, -1), values.shape)
    values[missing] = impute[missing]
    scaled = scaler.transform(values.reshape(len(values), -1))
    return np.concatenate([scaled, missing.reshape(len(values), -1).astype(np.float64)], axis=1)


def make_clean_matrix(frame: pd.DataFrame, location: np.ndarray, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray]:
    raw = history_array(frame)
    missing = np.zeros_like(raw, dtype=bool)
    return model_inputs(raw, missing, location, scaler), raw


def fit_preprocessing(train_frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    raw = history_array(train_frame)
    location, noise_scale = training_location_scale(raw)
    clean = np.nan_to_num(raw, nan=0.0).reshape(len(raw), -1)
    scaler = StandardScaler().fit(clean)
    return location, noise_scale, scaler


def prepare_augmented_matrix(augmented: pd.DataFrame, scaler: StandardScaler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = augmented.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mask = augmented.loc[:, MISSING_COLUMNS].to_numpy(dtype=bool)
    x = np.concatenate([scaler.transform(raw), mask.astype(np.float64)], axis=1)
    return x, augmented["target_value"].to_numpy(dtype=np.float64), augmented["augmentation_weight"].to_numpy(dtype=np.float64)


def baseline_predictions(frame: pd.DataFrame, raw: np.ndarray, missing: np.ndarray, location: np.ndarray, climatology: dict[int, float]) -> dict[str, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64).copy()
    miss = np.asarray(missing, dtype=bool)
    impute = np.broadcast_to(location.reshape(1, 1, -1), values.shape)
    values[miss] = impute[miss]
    pers = values[:, -1, 0]
    # `lag_10` is T-12: index 2 in oldest-to-newest T-14..T-3 history.
    seasonal = values[:, 2, 0]
    month_index = pd.to_datetime(frame["target_month"].astype(str) + "-01").dt.month.to_numpy()
    clim = np.asarray([climatology[int(m)] for m in month_index], dtype=np.float64)
    return {"persistence": pers, "seasonal_naive": seasonal, "training_monthly_climatology": clim}


def train_model_candidates(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weight: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    model_dir: Path,
    seed: int = 20261115,
    scaler: StandardScaler | None = None,
) -> tuple[dict[str, dict], list[dict]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    selected: dict[str, dict] = {}
    candidate_rows: list[dict] = []
    for model_index, architecture in enumerate(MODEL_ORDER):
        candidates = []
        for candidate_index, config in enumerate(_candidate_grid()[architecture]):
            candidate_seed = seed + model_index * 1009 + candidate_index * 37
            model, fit_info = fit_candidate(architecture, config, x_train, y_train, weight, x_val, y_val, seed=candidate_seed)
            pred = predict(model, architecture, x_val)
            metrics = regression_metrics(y_val, pred)
            inference_start = time.perf_counter()
            repeats = 20
            for _ in range(repeats):
                predict(model, architecture, x_val)
            inference_ms = (time.perf_counter() - inference_start) * 1000 / (repeats * len(x_val))
            row = {"architecture": architecture, "candidate_index": candidate_index, "config": config, **metrics, **fit_info, "inference_ms_per_sample": inference_ms, "seed": candidate_seed}
            candidate_rows.append(row)
            candidates.append((metrics["mae"], metrics["rmse"], json.dumps(config, sort_keys=True), model, config, fit_info, candidate_seed, inference_ms, pred))
        best = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        mae, rmse, _, model, config, fit_info, model_seed, inference_ms, val_pred = best
        model_path = model_dir / f"{architecture}_development_v1.{ 'pt' if architecture == 'temporal_cnn' else 'joblib' }"
        if architecture == "temporal_cnn":
            import torch
            torch.save({"architecture": architecture, "config": config, "seed": model_seed, "state_dict": model.state_dict(), "scaler_mean": None if scaler is None else scaler.mean_, "scaler_scale": None if scaler is None else scaler.scale_, "feature_order": list(FEATURE_COLUMNS), "channel_order": list(CHANNELS)}, model_path)
        else:
            joblib.dump({"architecture": architecture, "config": config, "seed": model_seed, "estimator": model, "scaler": scaler}, model_path, compress=3)
        abs_resid = np.abs(np.asarray(y_val) - np.asarray(val_pred))
        pi_radius = float(np.quantile(abs_resid, 0.90, method="higher"))
        selected[architecture] = {
            "model": model,
            "config": config,
            "seed": model_seed,
            "validation_prediction": np.asarray(val_pred, dtype=np.float64),
            "validation_metrics": regression_metrics(y_val, val_pred),
            "prediction_interval_radius_90": pi_radius,
            "fit_info": fit_info,
            "inference_ms_per_sample": inference_ms,
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "model_bytes": model_path.stat().st_size,
        }
    return selected, candidate_rows


def corruption_scenarios() -> list[tuple[str, float]]:
    return [(name, level) for name in (*SINGLE_FAMILIES, *COMBINED) for level in ROBUSTNESS_LEVELS]


def deterministic_seed(base_seed: int, sample_id: str, scenario: str, intensity: float) -> int:
    payload = f"{base_seed}|eval|{sample_id}|{scenario}|{intensity:.4f}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def evaluate_internal_robustness(
    models: dict[str, dict],
    raw: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    *,
    location: np.ndarray,
    noise_scale: np.ndarray,
    scaler: StandardScaler,
    climatology: dict[int, float],
    interval_radii: dict[str, float],
    seed: int = 90261115,
    bootstrap_reps: int = 2000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(frame) != 38 or frame["sample_id"].duplicated().any():
        raise ValueError("Robustness evaluation requires the 38 unique internal holdout months")
    scenarios = [("clean", 0.0)] + corruption_scenarios()
    pred_records: list[dict] = []
    metric_rows: list[dict] = []
    clean_predictions: dict[str, np.ndarray] = {}
    sample_ids = frame["sample_id"].astype(str).tolist()
    for scenario, intensity in scenarios:
        if scenario == "clean":
            corrupted = raw.copy()
            missing = np.zeros_like(raw, dtype=bool)
            valid_input = np.ones(len(raw), dtype=bool)
            effective_values = np.zeros(len(raw))
        else:
            items = [perturb_sample(raw[i], scenario=scenario, intensity=intensity, location=location, scale=noise_scale, seed=deterministic_seed(seed, sample_ids[i], scenario, intensity)) for i in range(len(raw))]
            corrupted = np.stack([item["values"] for item in items])
            missing = np.stack([item["missing"] for item in items])
            valid_input = np.asarray([not item["parameters"].get("physical_range_violations") for item in items], dtype=bool)
            effective_values = np.asarray([item["effective_intensity"] for item in items], dtype=np.float64)
        x_eval = model_inputs(corrupted, missing, location, scaler)
        baseline = baseline_predictions(frame, corrupted, missing, location, climatology)
        current_preds = {name: values.copy() for name, values in baseline.items()}
        for architecture, info in models.items():
            current_preds[architecture] = predict(info["model"], architecture, x_eval)
        for model_name, pred in current_preds.items():
            pred = np.asarray(pred, dtype=np.float64).copy()
            pred[~valid_input] = np.nan
            if scenario == "clean":
                clean_predictions[model_name] = pred.copy()
            metrics = regression_metrics(y, pred)
            errors = pred - y
            valid = np.isfinite(errors)
            mae_values = np.abs(errors[valid])
            ci_mae = bootstrap_interval(mae_values, seed=seed + len(metric_rows) + 1, n_resamples=bootstrap_reps)
            ci_rmse_sq = bootstrap_interval(errors[valid] ** 2, seed=seed + len(metric_rows) + 101, n_resamples=bootstrap_reps)
            ci_rmse = (math.sqrt(max(0.0, ci_rmse_sq[0])), math.sqrt(max(0.0, ci_rmse_sq[1])))
            ci_bias = bootstrap_interval(errors[valid], seed=seed + len(metric_rows) + 201, n_resamples=bootstrap_reps)
            clean_mae = regression_metrics(y, clean_predictions[model_name])["mae"] if scenario != "clean" else metrics["mae"]
            clean_rmse = regression_metrics(y, clean_predictions[model_name])["rmse"] if scenario != "clean" else metrics["rmse"]
            stability = float(np.sqrt(np.nanmean((pred - clean_predictions[model_name]) ** 2))) if scenario != "clean" and valid.any() else 0.0
            radius = interval_radii.get(model_name)
            interval_coverage = float(np.mean(np.abs(errors[valid]) <= radius)) if radius is not None and valid.any() else float("nan")
            clean_error = clean_predictions[model_name] - y
            delta_ci = bootstrap_interval(np.abs(errors[valid]) - np.abs(clean_error[valid]), seed=seed + len(metric_rows) + 301, n_resamples=bootstrap_reps) if scenario != "clean" else (0.0, 0.0)
            metric_rows.append({
                "model_id": model_name,
                "noise_type": scenario,
                "noise_intensity": float(intensity),
                "is_combined_noise": scenario in COMBINED,
                "independent_target_months": int(frame["sample_id"].nunique()),
                **metrics,
                "mae_ci95_low": ci_mae[0],
                "mae_ci95_high": ci_mae[1],
                "rmse_ci95_low": ci_rmse[0],
                "rmse_ci95_high": ci_rmse[1],
                "mean_bias_ci95_low": ci_bias[0],
                "mean_bias_ci95_high": ci_bias[1],
                "delta_mae_from_clean": float(metrics["mae"] - clean_mae) if np.isfinite(clean_mae) else float("nan"),
                "delta_mae_ci95_low": delta_ci[0],
                "delta_mae_ci95_high": delta_ci[1],
                "delta_rmse_from_clean": float(metrics["rmse"] - clean_rmse) if np.isfinite(clean_rmse) else float("nan"),
                "robustness_degradation_ratio": float(metrics["mae"] / clean_mae) if clean_mae and np.isfinite(clean_mae) else float("nan"),
                "prediction_stability_rms_change": stability,
                "rejected_physical_input_count": int((~valid_input).sum()),
                "mean_realized_noise_intensity": float(np.nanmean(effective_values)),
                "prediction_interval_90_coverage": interval_coverage,
            })
            for i, sid in enumerate(sample_ids):
                pred_records.append({"sample_id": sid, "target_month": str(frame.iloc[i]["target_month"]), "model_id": model_name, "noise_type": scenario, "noise_intensity": float(intensity), "y_true": float(y[i]), "y_pred": float(pred[i]) if np.isfinite(pred[i]) else None, "input_valid": bool(valid_input[i]), "realized_noise_intensity": float(effective_values[i])})
    return pd.DataFrame(metric_rows), pd.DataFrame(pred_records)


def baseline_interval_radii(y_val: np.ndarray, baseline_val: dict[str, np.ndarray]) -> dict[str, float]:
    return {name: float(np.quantile(np.abs(np.asarray(y_val) - np.asarray(pred)), 0.90, method="higher")) for name, pred in baseline_val.items()}

