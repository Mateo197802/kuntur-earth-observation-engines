#!/usr/bin/env python3
"""Fold-local feature ablation across all six v1.2 model families."""

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

from kuntur_data.phase2.optimization_features import FEATURE_GROUPS, build_feature_pair
from kuntur_data.phase2.predictive import PeakRSS, TemporalCNN, regression_metrics, sha256_file
from run_phase2_v12_round1 import FOLDS, SEEDS, STOCHASTIC, _block_bootstrap_delta, _fit_model, _sha_obj, _atomic_json


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
OUT = ROOT / "results/optimization/round_02_feature_ablation"
REGISTRY = OUT / "fold_registry.jsonl"
R1_DIR = ROOT / "results/optimization/round_01_architecture_search"


def _completed_keys() -> set[tuple[str, str]]:
    if not REGISTRY.is_file():
        return set()
    return {(r["run_id"], r["fold_id"]) for r in _load_r2_rows() if r.get("status") == "COMPLETED"}


def _append_r2_registry(row: dict[str, Any]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


def _load_r2_rows() -> list[dict[str, Any]]:
    if not REGISTRY.is_file():
        return []
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for line in REGISTRY.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(str(row.get("run_id")), str(row.get("fold_id")))] = row
    return list(latest.values())


class _FeatureTemporalCNN:
    """Small temporal CNN supporting fold-specific channel and static-feature counts."""

    def __init__(self, sequence_channels: int, static_features: int, width: int, seed: int):
        import torch
        from torch import nn

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(1)

        class Net(nn.Module):
            def __init__(self, seq_channels: int, statics: int, hidden: int):
                super().__init__()
                self.conv1 = nn.Conv1d(2 * seq_channels, hidden, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
                self.dropout = nn.Dropout(p=0.1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.output = nn.Linear(hidden + statics, 1)

            def forward(self, sequence, static):
                z = torch.relu(self.conv1(sequence))
                z = self.dropout(torch.relu(self.conv2(z)))
                z = self.pool(z).squeeze(-1)
                if static.shape[1]:
                    z = torch.cat([z, static], dim=1)
                return self.output(z).squeeze(-1)

        self.torch = torch
        self.sequence_channels = sequence_channels
        self.static_features = static_features
        self.net = Net(sequence_channels, static_features, width).cpu()

    def _tensors(self, x: np.ndarray):
        torch = self.torch
        sequence_size = 12 * self.sequence_channels
        values = np.asarray(x[:, :sequence_size], dtype=np.float32).reshape(-1, 12, self.sequence_channels).transpose(0, 2, 1)
        missing = np.asarray(x[:, sequence_size:2 * sequence_size], dtype=np.float32).reshape(-1, 12, self.sequence_channels).transpose(0, 2, 1)
        static = np.asarray(x[:, 2 * sequence_size:], dtype=np.float32)
        if static.shape[1] != self.static_features:
            raise ValueError("Temporal CNN static-feature width does not match fold transform")
        seq = torch.from_numpy(np.concatenate([values, missing], axis=1))
        stat = torch.from_numpy(static)
        return seq, stat

    def fit(self, x: np.ndarray, y: np.ndarray, x_stop: np.ndarray, y_stop: np.ndarray,
            *, learning_rate: float, max_epochs: int, patience: int, seed: int) -> int:
        torch = self.torch
        rng = np.random.default_rng(seed)
        xt, st = self._tensors(x)
        xv, sv = self._tensors(x_stop)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float32))
        yv = torch.as_tensor(np.asarray(y_stop, dtype=np.float32))
        optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-3)
        best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
        for epoch in range(max_epochs):
            self.net.train()
            order = rng.permutation(len(yt))
            for start in range(0, len(order), 64):
                idx = order[start:start + 64]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((self.net(xt[idx], st[idx]) - yt[idx]) ** 2)
                loss.backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(torch.mean((self.net(xv, sv) - yv) ** 2).item())
            if val_loss < best_loss - 1e-8:
                best_loss, best_state, best_epoch, stale = val_loss, {k: v.detach().clone() for k, v in self.net.state_dict().items()}, epoch + 1, 0
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state is None:
            raise RuntimeError("Temporal CNN did not produce a finite early-stopping checkpoint")
        self.net.load_state_dict(best_state)
        self.best_scaled_validation_mse = best_loss
        return best_epoch

    def predict_scaled(self, x: np.ndarray) -> np.ndarray:
        self.net.eval()
        sequence, static = self._tensors(x)
        with self.torch.no_grad():
            return self.net(sequence, static).cpu().numpy().astype(float)


class _ScaledFeatureCNN:
    def __init__(self, core: _FeatureTemporalCNN, center: float, scale: float):
        self.core, self.center, self.scale = core, center, scale

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.center + self.scale * self.core.predict_scaled(x)


def _fit_feature_cnn(config: dict[str, Any], pair, x_train: np.ndarray, y_train: np.ndarray, seed: int):
    if len(y_train) < 48:
        raise ValueError("Temporal CNN requires 48 or more training targets")
    stop_n = 12
    grad_y = y_train[:-stop_n]
    center = float(np.mean(grad_y))
    scale = float(np.std(grad_y)) or 1.0
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
    core = _FeatureTemporalCNN(pair.sequence_channels, pair.static_features, int(config["channels"]), seed)
    epochs = core.fit(x_train[:-stop_n], (grad_y - center) / scale,
                      x_train[-stop_n:], (y_train[-stop_n:] - center) / scale,
                      learning_rate=float(config["learning_rate"]), max_epochs=int(config["max_epochs"]),
                      patience=int(config["patience"]), seed=seed)
    return _ScaledFeatureCNN(core, center, scale), epochs


def _run_fold(model_id: str, group: str, config: dict[str, Any], run_id: str, spec_hash: str,
              train: pd.DataFrame, validation: pd.DataFrame, pair, fold: dict[str, str], source_hash: str) -> dict[str, Any]:
    seeds = SEEDS if model_id in STOCHASTIC else (SEEDS[0],)
    ytr_model, yv_model = pair.y_train, pair.y_validation
    ytr_actual, yv_actual = train.target_value.to_numpy(dtype=float), validation.target_value.to_numpy(dtype=float)
    pred_train, pred_val, seeds_data, warning_items = [], [], [], []
    wall, cpu, peak_rss = 0.0, 0.0, 0
    for seed in seeds:
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        with PeakRSS() as monitor, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if model_id == "temporal_cnn":
                model, epochs = _fit_feature_cnn(config, pair, pair.x_train, ytr_model, seed)
            else:
                model, epochs, model_warnings = _fit_model(model_id, config, pair.x_train, ytr_model, seed)
                warning_items.extend(model_warnings)
            train_prediction = np.asarray(model.predict(pair.x_train), dtype=float) + pair.offset_train
            val_prediction = np.asarray(model.predict(pair.x_validation), dtype=float) + pair.offset_validation
        if not np.isfinite(train_prediction).all() or not np.isfinite(val_prediction).all():
            raise ValueError(f"{model_id} generated non-finite predictions for {group}")
        wall += time.perf_counter() - started_wall
        cpu += time.process_time() - started_cpu
        peak_rss = max(peak_rss, int(monitor.peak))
        pred_train.append(train_prediction)
        pred_val.append(val_prediction)
        warning_items.extend(f"{type(item.message).__name__}: {str(item.message)[:180]}" for item in caught)
        seeds_data.append({"seed": seed, "epochs_or_iterations": epochs,
                           "train_metrics": regression_metrics(ytr_actual, train_prediction),
                           "validation_metrics": regression_metrics(yv_actual, val_prediction)})
    avg_train, avg_val = np.mean(pred_train, axis=0), np.mean(pred_val, axis=0)
    records = [{"sample_id": str(row.sample_id), "target_month": str(row.target_month),
                "y_true": float(yv_actual[i]), "y_pred": float(avg_val[i])}
               for i, row in validation.reset_index(drop=True).iterrows()]
    return {
        "run_id": run_id, "candidate_spec_sha256": spec_hash, "round": 2, "model_id": model_id,
        "feature_group": group, "config": config, "fold_id": fold["fold_id"], "status": "COMPLETED",
        "train_count": len(train), "validation_count": len(validation),
        "train_metrics_seed_averaged": regression_metrics(ytr_actual, avg_train),
        "validation_metrics_seed_averaged": regression_metrics(yv_actual, avg_val),
        "seed_metrics": seeds_data, "fit_wall_seconds_sum_seeds": wall,
        "fit_cpu_seconds_sum_seeds": cpu, "peak_rss_bytes": peak_rss,
        "warnings": sorted(set(warning_items)), "predictions": records,
        "selected_training_channels": list(pair.selected_channels),
        "feature_columns": list(pair.feature_columns), "source_sha256": source_hash,
    }


def _summary(source_hash: str, protocol_hash: str, script_hash: str, elapsed: float) -> dict[str, Any]:
    records = _load_r2_rows()
    completed = [r for r in records if r.get("status") == "COMPLETED" and int(r.get("round", 0)) == 2]
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in completed:
        groups.setdefault((row["feature_group"], row["model_id"], row["run_id"]), []).append(row)
    comparison_rows = []
    for (feature_group, model_id, run_id), fold_rows in groups.items():
        if len(fold_rows) != 3:
            continue
        fold_rows = sorted(fold_rows, key=lambda r: r["fold_id"])
        metrics = [r["validation_metrics_seed_averaged"] for r in fold_rows]
        comparison_rows.append({
            "round": 2, "feature_group": feature_group, "model_id": model_id, "run_id": run_id,
            "config": json.dumps(fold_rows[0]["config"], sort_keys=True, separators=(",", ":")),
            "fold_count": 3, "target_count": sum(m["sample_count"] for m in metrics),
            "mean_fold_mae": float(np.mean([m["mae"] for m in metrics])),
            "mean_fold_rmse": float(np.mean([m["rmse"] for m in metrics])),
            "mean_fold_bias": float(np.mean([m["mean_bias"] for m in metrics])),
            "max_fold_mae": float(max(m["mae"] for m in metrics)),
            "mean_fit_wall_seconds_per_fold": float(np.mean([r["fit_wall_seconds_sum_seeds"] for r in fold_rows])),
            "mean_fit_cpu_seconds_per_fold": float(np.mean([r["fit_cpu_seconds_sum_seeds"] for r in fold_rows])),
            "peak_rss_bytes": int(max(r["peak_rss_bytes"] for r in fold_rows)),
            "candidate_spec_sha256": fold_rows[0]["candidate_spec_sha256"],
            "selected_channels_by_fold": json.dumps({r["fold_id"]: r["selected_training_channels"] for r in fold_rows}, sort_keys=True),
            "status": "COMPLETED_DEVELOPMENT_ONLY",
        })
    comparison = pd.DataFrame(comparison_rows)
    if comparison.empty:
        raise RuntimeError("No complete Round 2 candidates are available")
    baseline_path = R1_DIR / "fold_registry.jsonl"
    baseline_all = [json.loads(line) for line in baseline_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline_rows = [r for r in baseline_all if r.get("round") == 1 and r.get("model_id") == "training_monthly_climatology" and r.get("status") == "COMPLETED"]
    base_mae = float(np.mean([r["validation_metrics_seed_averaged"]["mae"] for r in baseline_rows]))
    comparison["skill_vs_best_baseline_mae"] = 1.0 - comparison.mean_fold_mae / base_mae
    comparison["selected_candidate"] = False
    winner = comparison.sort_values(["mean_fold_mae", "mean_fold_rmse", "max_fold_mae", "model_id", "feature_group"], kind="stable").iloc[0]
    comparison.loc[comparison.run_id == winner.run_id, "selected_candidate"] = True
    comparison_path = OUT / "feature_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    # Best feature group per architecture is itself a development selection.
    best_arch = comparison.sort_values(["mean_fold_mae", "mean_fold_rmse", "max_fold_mae", "feature_group"], kind="stable").groupby("model_id", as_index=False).first()
    best_arch_path = OUT / "best_by_architecture.csv"
    best_arch.to_csv(best_arch_path, index=False)
    baseline_registry = {(r["run_id"], r["fold_id"]): r for r in baseline_rows}
    winner_records = [r for r in completed if r["run_id"] == winner.run_id]
    if len(winner_records) != 3:
        raise RuntimeError("Selected Round 2 candidate is missing a completed fold")
    baseline_records = []
    for fold in FOLDS:
        match = [r for (run_id, fold_id), r in baseline_registry.items() if fold_id == fold["fold_id"]]
        if len(match) != 1:
            raise RuntimeError(f"Missing frozen baseline for {fold['fold_id']}")
        baseline_records.extend(match)
    paired = _block_bootstrap_delta(winner_records, baseline_records, seed=20261116)
    best_payload = {"model_id": str(winner.model_id), "feature_group": str(winner.feature_group),
                    "run_id": str(winner.run_id), "mean_fold_mae": float(winner.mean_fold_mae),
                    "mean_fold_rmse": float(winner.mean_fold_rmse), "mean_fold_bias": float(winner.mean_fold_bias),
                    "max_fold_mae": float(winner.max_fold_mae), "candidate_spec_sha256": str(winner.candidate_spec_sha256)}
    payload = {
        "manifest_version": "kuntur-phase2-v1.2-round-2-feature-ablation-0.1.0",
        "status": "COMPLETED_DEVELOPMENT_ONLY", "round": 2,
        "root_cause_hypothesis": "The 60-value full-history representation may dilute the short-sample temperature signal; known seasonality, training-only anomaly transforms, past-only summaries, or channel reduction may improve performance without breaking T-3 availability.",
        "dataset": {"path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet", "sha256": source_hash,
                    "target_range_loaded": ["2003-03", "2018-12"], "validation_targets": 108, "sealed_2025_read": False},
        "feature_groups": FEATURE_GROUPS,
        "learned_transforms": "Sequence and static imputation/scaling and target climatology are fitted independently within each outer training fold. Stable channel selection uses expanding subwindows that are subsets of the outer training window only. All MERRA-2 predictor values terminate at T-3.",
        "candidate_count": len(comparison), "completed_candidate_folds": len(completed), "expected_candidate_folds": len(FEATURE_GROUPS) * 6 * 3,
        "common_targets_per_candidate": 108, "seeds": list(SEEDS), "best_candidate": best_payload,
        "strongest_baseline_mean_fold_mae": base_mae, "paired_vs_baseline": paired,
        "protocol_sha256": protocol_hash, "script_sha256": script_hash,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "resources": {"elapsed_wall_seconds": elapsed, "logical_cpus": psutil.cpu_count(logical=True), "physical_cpus": psutil.cpu_count(logical=False)},
        "artifacts": {"feature_comparison": {"path": comparison_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(comparison_path)},
                      "best_by_architecture": {"path": best_arch_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(best_arch_path)},
                      "fold_registry": {"path": REGISTRY.relative_to(ROOT).as_posix(), "sha256": sha256_file(REGISTRY)}},
        "limitations": ["All candidates are searched on the same 108 development targets; the best reported feature/model combination is post-selection evidence.",
                        "The original 2019-2024 partitions and the sealed 2025 targets were not loaded.",
                        "The 2025 twelve-month temporal test has not yet been accessed or scored."],
    }
    manifest_path = OUT / "round_manifest.json"
    _atomic_json(manifest_path, payload)
    print(json.dumps({"status": payload["status"], "candidate_count": len(comparison), "completed_candidate_folds": len(completed),
                      "best_candidate": best_payload, "paired_vs_baseline": paired, "manifest": manifest_path.relative_to(ROOT).as_posix()},
                     indent=2, allow_nan=False))


def main() -> None:
    started = time.perf_counter()
    if not FORECAST.is_file() or not (ROOT / "docs/protocols/continuous_optimization_v1_2.md").is_file():
        raise SystemExit("Frozen forecast input or protocol missing")
    source_hash = sha256_file(FORECAST)
    protocol_hash = sha256_file(ROOT / "docs/protocols/continuous_optimization_v1_2.md")
    script_hash = sha256_file(Path(__file__))
    frame = pd.read_parquet(FORECAST, filters=[("target_month", "<=", "2018-12")])
    if frame.empty or frame.target_month.max() != "2018-12":
        raise SystemExit("Unexpected Round 2 data availability")
    target_period = pd.PeriodIndex(frame.target_month.astype(str), freq="M")
    if not np.all(pd.PeriodIndex(frame.history_end_month.astype(str), freq="M") == target_period - 3):
        raise SystemExit("T-3 input history policy failed")
    if frame.sample_id.duplicated().any():
        raise SystemExit("Duplicate forecast sample IDs")
    r1_best = pd.read_csv(R1_DIR / "best_by_architecture.csv")
    model_ids = ["ridge", "svr_rbf", "random_forest", "hist_gradient_boosting", "mlp", "temporal_cnn"]
    configs = {r.model_id: json.loads(r.config) for r in r1_best.itertuples() if r.model_id in model_ids}
    if set(configs) != set(model_ids):
        raise SystemExit("Round 1 must supply one selected configuration for all six architectures")
    OUT.mkdir(parents=True, exist_ok=True)
    completed = _completed_keys()
    for fold in FOLDS:
        train = frame[frame.target_month <= fold["train_end"]].reset_index(drop=True)
        validation = frame[(frame.target_month >= fold["validation_start"]) & (frame.target_month <= fold["validation_end"])].reset_index(drop=True)
        if len(validation) != 36 or train.target_month.max() >= validation.target_month.min():
            raise SystemExit(f"Invalid chronological fold {fold['fold_id']}")
        for group in FEATURE_GROUPS:
            pair = build_feature_pair(train, validation, group)
            for model_id in model_ids:
                config = configs[model_id]
                spec = {"round": 2, "source_sha256": source_hash, "protocol_sha256": protocol_hash,
                        "script_sha256": script_hash, "feature_group": group, "model_id": model_id,
                        "config": config, "folds": FOLDS, "seeds": list(SEEDS) if model_id in STOCHASTIC else [SEEDS[0]]}
                spec_hash = _sha_obj(spec)
                run_id = f"r2-{group}-{model_id}-{spec_hash[:12]}"
                if (run_id, fold["fold_id"]) in completed:
                    continue
                try:
                    row = _run_fold(model_id, group, config, run_id, spec_hash, train, validation, pair, fold, source_hash)
                except Exception as exc:
                    row = {"run_id": run_id, "candidate_spec_sha256": spec_hash, "round": 2,
                           "model_id": model_id, "feature_group": group, "config": config,
                           "fold_id": fold["fold_id"], "status": "FAILED",
                           "error_type": type(exc).__name__, "error": str(exc)[:500], "source_sha256": source_hash}
                _append_r2_registry(row)
                if row["status"] == "COMPLETED":
                    completed.add((run_id, fold["fold_id"]))
                print(json.dumps({"event": "ROUND2_FEATURE_FOLD_COMPLETE", "fold_id": fold["fold_id"],
                                  "feature_group": group, "model_id": model_id, "status": row["status"],
                                  "mae": row.get("validation_metrics_seed_averaged", {}).get("mae"),
                                  "selected_channels": row.get("selected_training_channels"),
                                  "elapsed_seconds": round(time.perf_counter() - started, 2)}, allow_nan=False), flush=True)
    _summary(source_hash, protocol_hash, script_hash, time.perf_counter() - started)


if __name__ == "__main__":
    main()
