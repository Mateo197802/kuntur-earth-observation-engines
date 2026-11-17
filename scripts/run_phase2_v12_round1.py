#!/usr/bin/env python3
"""Run finite, resumable v1.2 architecture/regularization search on rolling development folds.

This command reads forecast targets only through 2018-12. It never reads or
scores the historical 2019-2024 partitions or the separately sealed 2025 set.
Each completed candidate/fold is appended to a checkpoint registry and skipped
on resume.
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from kuntur_data.phase2.augmentation import history_array
from kuntur_data.phase2.predictive import (
    PeakRSS,
    TemporalCNN,
    baseline_predictions,
    fit_preprocessing_v1_1,
    make_clean_matrix_v1_1,
    model_inputs,
    regression_metrics,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
OUT = ROOT / "results/optimization/round_01_architecture_search"
REGISTRY = OUT / "fold_registry.jsonl"
PROTOCOL = ROOT / "docs/protocols/continuous_optimization_v1_2.md"
FOLDS = (
    {"fold_id": "origin_2009", "train_end": "2009-12", "validation_start": "2010-01", "validation_end": "2012-12"},
    {"fold_id": "origin_2012", "train_end": "2012-12", "validation_start": "2013-01", "validation_end": "2015-12"},
    {"fold_id": "origin_2015", "train_end": "2015-12", "validation_start": "2016-01", "validation_end": "2018-12"},
)
SEEDS = (20261115, 20261116, 20261117)
STOCHASTIC = {"random_forest", "hist_gradient_boosting", "mlp", "temporal_cnn"}
N_BOOT = 2000
BLOCK_LENGTH = 12


def config_grid() -> list[tuple[str, dict[str, Any]]]:
    """Small predeclared grids; this is one bounded search round, not an open optimizer."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for alpha in (0.1, 1.0, 10.0):
        rows.append(("ridge", {"alpha": alpha}))
    for c in (0.3, 1.0, 3.0):
        for epsilon in (0.05, 0.1):
            rows.append(("svr_rbf", {"C": c, "epsilon": epsilon, "gamma": "scale"}))
    for depth, leaf in ((2, 3), (3, 3), (4, 3), (3, 5)):
        rows.append(("random_forest", {"n_estimators": 120, "max_depth": depth, "min_samples_leaf": leaf, "max_features": 1.0}))
    for lr, leaves, l2 in ((0.03, 3, 1.0), (0.05, 3, 1.0), (0.05, 7, 1.0), (0.05, 7, 5.0), (0.1, 3, 1.0), (0.1, 7, 5.0)):
        rows.append(("hist_gradient_boosting", {"learning_rate": lr, "max_iter": 100, "max_leaf_nodes": leaves, "l2_regularization": l2}))
    for width in (8, 16):
        for alpha in (0.01, 0.1):
            rows.append(("mlp", {"hidden_layer_sizes": [width], "alpha": alpha, "max_iter": 800}))
    for channels in (4, 8):
        for lr in (0.001, 0.003):
            rows.append(("temporal_cnn", {"channels": channels, "learning_rate": lr, "max_epochs": 50, "patience": 8}))
    return rows


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, allow_nan=False).encode("utf-8")


def _sha_obj(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _registry_keys() -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    if not REGISTRY.is_file():
        return keys
    with REGISTRY.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "COMPLETED":
                keys.add((str(row["run_id"]), str(row["fold_id"])))
    return keys


def _append_registry(row: dict[str, Any]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    # A single encoded record is appended only after an entire config/fold finishes.
    with REGISTRY.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


class _TargetScaledCNN:
    def __init__(self, core: TemporalCNN, center: float, scale: float):
        self.core = core
        self.center = center
        self.scale = scale

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.center + self.scale * self.core.predict(x)


def _fit_model(name: str, config: dict[str, Any], x: np.ndarray, y: np.ndarray, seed: int) -> tuple[Any, int | None, list[str]]:
    captured: list[str] = []
    if name == "ridge":
        return Ridge(alpha=float(config["alpha"])).fit(x, y), None, captured
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if name == "svr_rbf":
            model = SVR(kernel="rbf", **config).fit(x, y)
            epochs = None
        elif name == "random_forest":
            model = RandomForestRegressor(**config, random_state=seed, n_jobs=1).fit(x, y)
            epochs = None
        elif name == "hist_gradient_boosting":
            model = HistGradientBoostingRegressor(**config, random_state=seed, early_stopping=False).fit(x, y)
            epochs = None
        elif name == "mlp":
            estimator = MLPRegressor(**config, solver="adam", batch_size=32, learning_rate_init=0.001,
                                     tol=1e-4, n_iter_no_change=40, random_state=seed, shuffle=True)
            model = TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler()).fit(x, y)
            epochs = int(model.regressor_.n_iter_)
        elif name == "temporal_cnn":
            if len(y) < 48:
                raise ValueError("Temporal CNN requires 48 or more training targets")
            stop_n = 12
            gradient_y = y[:-stop_n]
            center = float(np.mean(gradient_y))
            scale = float(np.std(gradient_y, ddof=0))
            if not np.isfinite(scale) or scale <= 1e-8:
                scale = 1.0
            core = TemporalCNN(channels=int(config["channels"]), seed=seed)
            epochs = core.fit(
                x[:-stop_n], (gradient_y - center) / scale, np.ones(len(gradient_y)),
                x[-stop_n:], (y[-stop_n:] - center) / scale,
                learning_rate=float(config["learning_rate"]), max_epochs=int(config["max_epochs"]),
                patience=int(config["patience"]), seed=seed,
            )
            model = _TargetScaledCNN(core, center, scale)
        else:
            raise ValueError(f"Unknown model: {name}")
        captured = [f"{type(item.message).__name__}: {str(item.message)[:180]}" for item in caught]
    return model, epochs, captured


def _fold_baselines(train: pd.DataFrame, val: pd.DataFrame, raw: np.ndarray, missing: np.ndarray, location: np.ndarray) -> dict[str, np.ndarray]:
    train_month = pd.to_datetime(train.target_month.astype(str) + "-01").dt.month.to_numpy()
    monthly = train.assign(_month=train_month).groupby("_month").target_value.mean().to_dict()
    return baseline_predictions(val, raw, missing, location, monthly)


def _run_id(source_hash: str, protocol_hash: str, script_hash: str, name: str, config: dict[str, Any]) -> tuple[str, str]:
    spec = {"round": 1, "source_sha256": source_hash, "protocol_sha256": protocol_hash,
            "script_sha256": script_hash, "architecture": name, "config": config,
            "seeds": list(SEEDS) if name in STOCHASTIC else [SEEDS[0]],
            "folds": FOLDS, "feature_contract": "frozen-12-month-5-channel-T14-through-T3-v1.0"}
    spec_hash = _sha_obj(spec)
    return f"r1-{name}-{spec_hash[:16]}", spec_hash


def _run_candidate_fold(name: str, config: dict[str, Any], run_id: str, spec_hash: str,
                        fold: dict[str, str], train: pd.DataFrame, val: pd.DataFrame,
                        x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray,
                        source_hash: str) -> dict[str, Any]:
    seeds = SEEDS if name in STOCHASTIC else (SEEDS[0],)
    val_predictions: list[np.ndarray] = []
    train_predictions: list[np.ndarray] = []
    seed_metrics: list[dict[str, Any]] = []
    warnings_seen: list[str] = []
    wall_total = 0.0
    cpu_total = 0.0
    peak_rss = 0
    for seed in seeds:
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        with PeakRSS() as monitor:
            model, epochs, model_warnings = _fit_model(name, config, x_train, y_train, seed)
            pred_val = np.asarray(model.predict(x_val), dtype=np.float64)
            pred_train = np.asarray(model.predict(x_train), dtype=np.float64)
        if not np.isfinite(pred_val).all() or not np.isfinite(pred_train).all():
            raise ValueError(f"{name} produced a non-finite prediction")
        wall_total += time.perf_counter() - start_wall
        cpu_total += time.process_time() - start_cpu
        peak_rss = max(peak_rss, int(monitor.peak))
        val_predictions.append(pred_val)
        train_predictions.append(pred_train)
        warnings_seen.extend(model_warnings)
        seed_metrics.append({"seed": seed, "epochs_or_iterations": epochs,
                             "train": regression_metrics(y_train, pred_train),
                             "validation": regression_metrics(y_val, pred_val)})
    avg_val = np.mean(np.stack(val_predictions), axis=0)
    avg_train = np.mean(np.stack(train_predictions), axis=0)
    predictions = [{"sample_id": str(row.sample_id), "target_month": str(row.target_month),
                    "y_true": float(y_val[i]), "y_pred": float(avg_val[i])}
                   for i, row in val.reset_index(drop=True).iterrows()]
    return {
        "run_id": run_id, "candidate_spec_sha256": spec_hash, "round": 1, "model_id": name,
        "config": config, "fold_id": fold["fold_id"], "status": "COMPLETED",
        "train_count": len(train), "validation_count": len(val),
        "train_metrics_seed_averaged": regression_metrics(y_train, avg_train),
        "validation_metrics_seed_averaged": regression_metrics(y_val, avg_val),
        "seed_metrics": seed_metrics, "fit_wall_seconds_sum_seeds": wall_total,
        "fit_cpu_seconds_sum_seeds": cpu_total, "peak_rss_bytes": peak_rss,
        "warnings": sorted(set(warnings_seen)), "predictions": predictions,
        "source_sha256": source_hash,
    }


def _block_bootstrap_delta(model_records: list[dict[str, Any]], baseline_records: list[dict[str, Any]], seed: int = 20261115) -> dict[str, Any]:
    model_rows = {(r["fold_id"], p["target_month"]): p for r in model_records for p in r["predictions"]}
    base_rows = {(r["fold_id"], p["target_month"]): p for r in baseline_records for p in r["predictions"]}
    keys = sorted(set(model_rows) & set(base_rows))
    if len(keys) != 108:
        raise ValueError(f"Expected 108 paired rolling targets, found {len(keys)}")
    diff_by_fold: dict[str, np.ndarray] = {}
    for fold_id in (f["fold_id"] for f in FOLDS):
        months = sorted(k[1] for k in keys if k[0] == fold_id)
        values = []
        for month in months:
            m, b = model_rows[(fold_id, month)], base_rows[(fold_id, month)]
            values.append(abs(m["y_pred"] - m["y_true"]) - abs(b["y_pred"] - b["y_true"]))
        diff_by_fold[fold_id] = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = np.empty(N_BOOT, dtype=np.float64)
    for rep in range(N_BOOT):
        resampled = []
        for values in diff_by_fold.values():
            n = len(values)
            chunks = []
            while sum(len(x) for x in chunks) < n:
                start = int(rng.integers(0, n))
                chunks.append(values[(start + np.arange(BLOCK_LENGTH)) % n])
            resampled.extend(np.concatenate(chunks)[:n])
        draws[rep] = float(np.mean(resampled))
    observed = float(np.mean(np.concatenate(list(diff_by_fold.values()))))
    return {"paired_targets": len(keys), "block_length_months": BLOCK_LENGTH, "replicates": N_BOOT,
            "delta_mae_model_minus_baseline": observed,
            "ci95_low": float(np.quantile(draws, 0.025)), "ci95_high": float(np.quantile(draws, 0.975)),
            "months_lower_absolute_error": int(sum(abs(model_rows[k]["y_pred"] - model_rows[k]["y_true"]) < abs(base_rows[k]["y_pred"] - base_rows[k]["y_true"]) for k in keys)),
            "months_lower_squared_error": int(sum(float(model_rows[k]["y_pred"] - model_rows[k]["y_true"]) ** 2 < float(base_rows[k]["y_pred"] - base_rows[k]["y_true"]) ** 2 for k in keys)),
            "folds_improved": int(sum(float(np.mean(diff)) < 0 for diff in diff_by_fold.values()))}


def _load_registry_rows() -> list[dict[str, Any]]:
    if not REGISTRY.is_file():
        return []
    rows = []
    with REGISTRY.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    # New run IDs are content-addressed. Latest record wins if a failed attempt was logged.
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row.get("run_id")), str(row.get("fold_id")))] = row
    return list(latest.values())


def _summarize(records: list[dict[str, Any]], source_hash: str, protocol_hash: str, script_hash: str, elapsed: float) -> dict[str, Any]:
    completed = [r for r in records if r.get("status") == "COMPLETED"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in completed:
        groups.setdefault((row["model_id"], row["run_id"]), []).append(row)
    candidate_rows: list[dict[str, Any]] = []
    for (model_id, run_id), rows in groups.items():
        if len(rows) != len(FOLDS):
            continue
        folds = sorted(rows, key=lambda r: r["fold_id"])
        metrics = [r["validation_metrics_seed_averaged"] for r in folds]
        candidate_rows.append({
            "round": 1, "run_id": run_id, "model_id": model_id,
            "config": json.dumps(folds[0]["config"], sort_keys=True, separators=(",", ":")),
            "fold_count": len(folds), "target_count": sum(int(r["validation_count"]) for r in folds),
            "mean_fold_mae": float(np.mean([m["mae"] for m in metrics])),
            "mean_fold_rmse": float(np.mean([m["rmse"] for m in metrics])),
            "mean_fold_bias": float(np.mean([m["mean_bias"] for m in metrics])),
            "max_fold_mae": float(max(m["mae"] for m in metrics)),
            "pooled_mae": float(np.mean([m["mae"] * m["sample_count"] for m in metrics]) / sum(m["sample_count"] for m in metrics)),
            "mean_fit_wall_seconds_per_fold": float(np.mean([r["fit_wall_seconds_sum_seeds"] for r in folds])),
            "mean_fit_cpu_seconds_per_fold": float(np.mean([r["fit_cpu_seconds_sum_seeds"] for r in folds])),
            "peak_rss_bytes": int(max(r["peak_rss_bytes"] for r in folds)),
            "candidate_spec_sha256": folds[0]["candidate_spec_sha256"],
            "status": "COMPLETED_DEVELOPMENT_ONLY",
        })
    comparison = pd.DataFrame(candidate_rows)
    if comparison.empty:
        raise RuntimeError("No complete model candidates are available")
    # Competitors use the exact same 108 targets; selection among all configs is development-only.
    best_by_model = comparison.sort_values(["mean_fold_mae", "mean_fold_rmse", "max_fold_mae", "run_id"], kind="stable").groupby("model_id", as_index=False).first()
    best_candidates = best_by_model.sort_values(["mean_fold_mae", "mean_fold_rmse", "max_fold_mae", "model_id"], kind="stable")
    selected = best_candidates.iloc[0]
    base_ids = {"persistence", "seasonal_naive", "training_monthly_climatology"}
    baseline = comparison[comparison.model_id.isin(base_ids)].sort_values(["mean_fold_mae", "mean_fold_rmse", "model_id"], kind="stable").iloc[0]
    by_run = {(r["run_id"], r["fold_id"]): r for r in completed}
    selected_records = [by_run[(selected.run_id, fold["fold_id"])] for fold in FOLDS]
    baseline_records = [by_run[(baseline.run_id, fold["fold_id"])] for fold in FOLDS]
    paired = _block_bootstrap_delta(selected_records, baseline_records)
    baseline_mae = float(baseline.mean_fold_mae)
    comparison["skill_vs_best_baseline_mae"] = 1.0 - comparison.mean_fold_mae / baseline_mae
    comparison["selected_candidate"] = comparison.run_id.eq(selected.run_id)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_path = OUT / "candidate_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    best_path = OUT / "best_by_architecture.csv"
    best_by_model = best_by_model.copy()
    best_by_model["skill_vs_best_baseline_mae"] = 1.0 - best_by_model.mean_fold_mae / baseline_mae
    best_by_model.to_csv(best_path, index=False)
    payload = {
        "manifest_version": "kuntur-phase2-v1.2-round-1-architecture-search-0.1.0",
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "round": 1,
        "root_cause_hypothesis": "The v1.1 strongest learned model is only modestly better than the training-only climatology; bounded regularization changes may improve error or stability, while higher-capacity models may overfit the short sequence.",
        "experiments": {"config_count": int(len(config_grid())), "completed_candidate_folds": len(completed), "expected_candidate_folds": int(len(config_grid()) * len(FOLDS)), "folds": list(FOLDS), "seeds": list(SEEDS), "primary_metric": "mean_fold_mae", "development_targets": 108},
        "dataset": {"path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet", "sha256": source_hash, "target_range_loaded": ["2003-03", "2018-12"], "reader_filter": "target_month <= 2018-12", "used_validation_ranges": ["2010-01..2012-12", "2013-01..2015-12", "2016-01..2018-12"], "sealed_test_read": False},
        "protocol_sha256": protocol_hash,
        "script_sha256": script_hash,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "resources": {"elapsed_wall_seconds": elapsed, "cpu_count_logical": psutil.cpu_count(logical=True), "cpu_count_physical": psutil.cpu_count(logical=False), "memory_available_bytes_after_run": psutil.virtual_memory().available},
        "best_candidate": {"model_id": str(selected.model_id), "run_id": str(selected.run_id), "mean_fold_mae": float(selected.mean_fold_mae), "mean_fold_rmse": float(selected.mean_fold_rmse), "mean_fold_bias": float(selected.mean_fold_bias), "mean_fit_wall_seconds_per_fold": float(selected.mean_fit_wall_seconds_per_fold), "peak_rss_bytes": int(selected.peak_rss_bytes), "candidate_spec_sha256": str(selected.candidate_spec_sha256)},
        "strongest_baseline": {"model_id": str(baseline.model_id), "mean_fold_mae": baseline_mae},
        "paired_comparison": paired,
        "limitations": ["All 108 validation targets were used to compare configurations; these are development selection estimates, not unbiased performance estimates.", "The original 2019-2024 partitions and the new sealed 2025 period were not read by this script.", "A 12-month block bootstrap over three 36-month folds is descriptive and has limited precision."],
        "artifacts": {"candidate_comparison": {"path": str(comparison_path.relative_to(ROOT).as_posix()), "sha256": sha256_file(comparison_path)}, "best_by_architecture": {"path": str(best_path.relative_to(ROOT).as_posix()), "sha256": sha256_file(best_path)}, "fold_registry": {"path": str(REGISTRY.relative_to(ROOT).as_posix()), "sha256": sha256_file(REGISTRY)}},
    }
    manifest_path = OUT / "round_manifest.json"
    _atomic_json(manifest_path, payload)
    return payload


def main() -> None:
    started = time.perf_counter()
    if not FORECAST.is_file() or not PROTOCOL.is_file():
        raise SystemExit("Frozen forecast input or v1.2 protocol is missing")
    source_hash, protocol_hash, script_hash = sha256_file(FORECAST), sha256_file(PROTOCOL), sha256_file(Path(__file__))
    frame = pd.read_parquet(FORECAST, filters=[("target_month", "<=", "2018-12")])
    if frame.empty or frame.target_month.max() != "2018-12" or frame.target_month.min() != "2003-03":
        raise SystemExit("Unexpected development-only forecast coverage")
    target = pd.PeriodIndex(frame.target_month.astype(str), freq="M")
    if not np.all(pd.PeriodIndex(frame.history_end_month.astype(str), freq="M") == target - 3):
        raise SystemExit("T-3 input-history policy failed")
    if not np.all(pd.PeriodIndex(frame.forecast_origin_month.astype(str), freq="M") == target - 1):
        raise SystemExit("Issue-time chronology failed")
    if not frame.target_month.is_monotonic_increasing or frame.sample_id.duplicated().any():
        raise SystemExit("Development forecast rows are not sorted and unique")

    completed = _registry_keys()
    OUT.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        train = frame[frame.target_month <= fold["train_end"]].reset_index(drop=True)
        val = frame[(frame.target_month >= fold["validation_start"]) & (frame.target_month <= fold["validation_end"])].reset_index(drop=True)
        if len(val) != 36 or train.target_month.max() >= val.target_month.min():
            raise SystemExit(f"Unexpected temporal fold {fold['fold_id']}")
        location, _, scaler = fit_preprocessing_v1_1(train)
        raw_train = history_array(train)
        missing_train = ~np.isfinite(raw_train)
        x_train = model_inputs(raw_train, missing_train, location, scaler)
        x_val, raw_val, missing_val = make_clean_matrix_v1_1(val, location, scaler)
        y_train, y_val = train.target_value.to_numpy(dtype=np.float64), val.target_value.to_numpy(dtype=np.float64)
        baseline_rows = _fold_baselines(train, val, raw_val, missing_val, location)
        base_spec_hash = _sha_obj({"round": 1, "source_sha256": source_hash, "protocol_sha256": protocol_hash,
                                   "script_sha256": script_hash, "folds": FOLDS, "model": "baseline-v1.2"})
        for base_id, pred in baseline_rows.items():
            run_id = f"r1-{base_id}-{base_spec_hash[:16]}"
            if (run_id, fold["fold_id"]) in completed:
                continue
            predictions = [{"sample_id": str(row.sample_id), "target_month": str(row.target_month),
                            "y_true": float(y_val[i]), "y_pred": float(pred[i])}
                           for i, row in val.iterrows()]
            _append_registry({"run_id": run_id, "candidate_spec_sha256": base_spec_hash, "round": 1,
                              "model_id": base_id, "config": {}, "fold_id": fold["fold_id"], "status": "COMPLETED",
                              "train_count": len(train), "validation_count": len(val),
                              "train_metrics_seed_averaged": None, "validation_metrics_seed_averaged": regression_metrics(y_val, pred),
                              "seed_metrics": [], "fit_wall_seconds_sum_seeds": 0.0, "fit_cpu_seconds_sum_seeds": 0.0,
                              "peak_rss_bytes": 0, "warnings": [], "predictions": predictions, "source_sha256": source_hash})
            completed.add((run_id, fold["fold_id"]))
        for name, config in config_grid():
            run_id, spec_hash = _run_id(source_hash, protocol_hash, script_hash, name, config)
            if (run_id, fold["fold_id"]) in completed:
                continue
            try:
                row = _run_candidate_fold(name, config, run_id, spec_hash, fold, train, val,
                                          x_train, y_train, x_val, y_val, source_hash)
            except Exception as exc:
                row = {"run_id": run_id, "candidate_spec_sha256": spec_hash, "round": 1, "model_id": name,
                       "config": config, "fold_id": fold["fold_id"], "status": "FAILED",
                       "error_type": type(exc).__name__, "error": str(exc)[:500], "source_sha256": source_hash}
            _append_registry(row)
            if row["status"] == "COMPLETED":
                completed.add((run_id, fold["fold_id"]))
            print(json.dumps({"event": "ROUND1_FOLD_CANDIDATE_COMPLETE", "fold_id": fold["fold_id"],
                              "model_id": name, "run_id": run_id, "status": row["status"],
                              "validation_mae": row.get("validation_metrics_seed_averaged", {}).get("mae"),
                              "elapsed_seconds": round(time.perf_counter() - started, 2)}, allow_nan=False), flush=True)
    records = _load_registry_rows()
    payload = _summarize(records, source_hash, protocol_hash, script_hash, time.perf_counter() - started)
    print(json.dumps({"status": payload["status"], "round": 1, "best_candidate": payload["best_candidate"],
                      "strongest_baseline": payload["strongest_baseline"], "paired_comparison": payload["paired_comparison"],
                      "manifest": str((OUT / "round_manifest.json").relative_to(ROOT).as_posix())}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
