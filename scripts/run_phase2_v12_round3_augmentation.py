#!/usr/bin/env python3
"""Controlled, balanced training-augmentation and 25%/50% noise evaluation."""

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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from kuntur_data.phase2.augmentation import (
    COMBINED, FEATURE_COLUMNS, PHYSICAL_RANGES, SINGLE_FAMILIES,
    _seed, history_array, perturb_sample,
)
from kuntur_data.phase2.optimization_features import build_feature_pair
from kuntur_data.phase2.predictive import PeakRSS, fit_preprocessing_v1_1, regression_metrics, sha256_file
try:  # Support both `python scripts/...py` and package imports in pytest.
    from .run_phase2_v12_round1 import FOLDS, SEEDS, _block_bootstrap_delta, _sha_obj, _atomic_json
except ImportError:  # pragma: no cover - direct-script execution path
    from run_phase2_v12_round1 import FOLDS, SEEDS, _block_bootstrap_delta, _sha_obj, _atomic_json


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
ROUND1_DIR = ROOT / "results/optimization/round_01_architecture_search"
ROUND2_DIR = ROOT / "results/optimization/round_02_feature_ablation"
OUT = ROOT / "results/optimization/round_03_augmentation_robustness"
REGISTRY = OUT / "fold_registry.jsonl"
ROBUST_SCENARIOS = tuple(SINGLE_FAMILIES) + tuple(COMBINED)
ROBUST_LEVELS = (0.25, 0.50)
TRAIN_LEVELS = (0.05, 0.10, 0.15, 0.20)
MODEL_IDS = ("random_forest", "svr_rbf")
FEATURE_GROUP = "C_training_anomalies"


def scheme_spec() -> dict[str, dict[str, Any]]:
    all_single_mixed = [(family, level) for family in SINGLE_FAMILIES for level in TRAIN_LEVELS]
    return {
        "clean_only": {"scenarios": [], "weighting": "authentic_month_total_1", "smoothed": False, "mask_indicators": True},
        "gaussian_only": {"scenarios": [("gaussian", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "laplace_only": {"scenarios": [("laplace", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "drift_only": {"scenarios": [("drift", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "impulsive_only": {"scenarios": [("impulsive", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "heteroscedastic_only": {"scenarios": [("heteroscedastic", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "missingness_only": {"scenarios": [("missingness", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "single_family_low_05": {"scenarios": [(family, 0.05) for family in SINGLE_FAMILIES], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "single_family_balanced_mixed": {"scenarios": all_single_mixed, "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "mixed_multi_family": {"scenarios": [("mixed_multi_family", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": True},
        "mixed_multi_family_naive_10_20": {"scenarios": [("mixed_multi_family", level) for level in (0.10, 0.20)], "weighting": "naive_each_variant_weight_1", "smoothed": False, "mask_indicators": True},
        "clean_only_smoothed": {"scenarios": [], "weighting": "authentic_month_total_1", "smoothed": True, "mask_indicators": True},
        "single_family_balanced_mixed_smoothed": {"scenarios": all_single_mixed, "weighting": "balanced_50_50", "smoothed": True, "mask_indicators": True},
        "missingness_only_simple_imputation": {"scenarios": [("missingness", level) for level in TRAIN_LEVELS], "weighting": "balanced_50_50", "smoothed": False, "mask_indicators": False},
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[(row.get("run_id"), row.get("fold_id"))] = row
    return list(latest.values())


def _append(row: dict[str, Any]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        stream.flush()


def _smooth(values: np.ndarray, window: int = 3) -> np.ndarray:
    """Trailing causal filter along the T-14..T-3 input history."""
    x = np.asarray(values, dtype=float).copy()
    result = x.copy()
    for t in range(x.shape[1]):
        start = max(0, t - window + 1)
        current = x[:, start:t + 1, :]
        valid = np.isfinite(current)
        counts = valid.sum(axis=1)
        totals = np.nansum(current, axis=1)
        result[:, t, :] = np.divide(totals, counts, out=np.full_like(totals, np.nan), where=counts > 0)
    return result


def _frame_with_histories(template: pd.DataFrame, histories: np.ndarray, suffix: str, *, smooth: bool = False) -> pd.DataFrame:
    values = _smooth(histories) if smooth else np.asarray(histories, dtype=float)
    output = template.copy().reset_index(drop=True)
    output.loc[:, FEATURE_COLUMNS] = values.reshape(len(values), -1)
    if "sample_id" in output:
        output["sample_id"] = [f"{value}{suffix}" for value in output.sample_id.astype(str)]
    return output


def _generate_training_scheme(train: pd.DataFrame, scheme: str, specification: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = history_array(train)
    location, scale, _ = fit_preprocessing_v1_1(train)
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    synthetic_total = 0
    per_base_counts = []
    for i, base in train.reset_index(drop=True).iterrows():
        base_values = raw[i].copy()
        synthetic_rows = []
        for scenario, intensity in specification["scenarios"]:
            noise_seed = _seed(seed, str(base.sample_id), str(scenario), float(intensity))
            item = perturb_sample(base_values, scenario=str(scenario), intensity=float(intensity),
                                  location=location, scale=scale, seed=noise_seed)
            violations = item["parameters"].get("physical_range_violations", [])
            if violations:
                rejected.append({"base_sample_id": str(base.sample_id), "scenario": scenario,
                                 "intensity": float(intensity), "reason": "broad physical range violation",
                                 "channels": violations})
                continue
            record = base.to_dict()
            record.update({column: float(item["values"].reshape(-1)[j]) for j, column in enumerate(FEATURE_COLUMNS)})
            record["sample_id"] = f"{base.sample_id}__{scheme}__{scenario}__{int(round(float(intensity)*100)):02d}"
            record["_augmentation_type"] = str(scenario)
            record["_augmentation_intensity"] = float(intensity)
            record["_noise_seed"] = int(noise_seed)
            synthetic_rows.append(record)
        clean_record = base.to_dict()
        clean_record.update({column: float(base_values.reshape(-1)[j]) for j, column in enumerate(FEATURE_COLUMNS)})
        clean_record["sample_id"] = f"{base.sample_id}__{scheme}__clean"
        clean_record["_augmentation_type"] = "clean"
        clean_record["_augmentation_intensity"] = 0.0
        clean_record["_noise_seed"] = None
        if specification["weighting"] == "naive_each_variant_weight_1":
            clean_record["_sample_weight"] = 1.0
            for record in synthetic_rows:
                record["_sample_weight"] = 1.0
        elif not synthetic_rows:
            clean_record["_sample_weight"] = 1.0
        else:
            clean_record["_sample_weight"] = 0.5
            for record in synthetic_rows:
                record["_sample_weight"] = 0.5 / len(synthetic_rows)
        rows.append(clean_record)
        rows.extend(synthetic_rows)
        synthetic_total += len(synthetic_rows)
        per_base_counts.append(len(synthetic_rows))
    augmented = pd.DataFrame.from_records(rows).sort_values(["target_month", "sample_id"], kind="stable").reset_index(drop=True)
    weights = augmented.pop("_sample_weight").to_numpy(dtype=float)
    metadata = {
        "scheme": scheme, "source_train_months": int(train.target_month.nunique()),
        "augmented_rows": int(len(augmented)), "synthetic_rows": int(synthetic_total),
        "rejected_variants": len(rejected), "rejection_examples": rejected[:20],
        "min_variants_per_original": int(min(per_base_counts, default=0)),
        "max_variants_per_original": int(max(per_base_counts, default=0)),
        "sample_weight_sum_by_original_sample_min": float(pd.DataFrame({"sample_id": augmented.sample_id.str.split("__").str[0], "w": weights}).groupby("sample_id").w.sum().min()),
        "weighting": specification["weighting"], "noise_scales_source": "clean outer-fold training inputs only",
        "targets_perturbed": False, "smoothed_inputs": bool(specification["smoothed"]),
        "mask_indicators": bool(specification["mask_indicators"]),
    }
    if specification["smoothed"]:
        augmented = _frame_with_histories(augmented, history_array(augmented), "", smooth=True)
    return augmented, metadata | {"sample_weights": weights}


def _corrupted_validation(validation: pd.DataFrame, scenario: str, intensity: float,
                          location: np.ndarray, scale: np.ndarray, *, smooth: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    raw = history_array(validation)
    output = raw.copy()
    valid = np.ones(len(validation), dtype=bool)
    effective, rejected = [], []
    for i, row in validation.reset_index(drop=True).iterrows():
        noise_seed = _seed(20261115, str(row.sample_id), scenario, float(intensity))
        item = perturb_sample(raw[i], scenario=scenario, intensity=intensity,
                              location=location, scale=scale, seed=noise_seed)
        violations = item["parameters"].get("physical_range_violations", [])
        if violations:
            valid[i] = False
            rejected.append(str(row.sample_id))
            output[i] = raw[i]
        else:
            output[i] = item["values"]
            effective.append(float(item["effective_intensity"]))
    corrupted = _frame_with_histories(validation, output, f"__{scenario}_{int(intensity*100)}", smooth=smooth)
    return corrupted, valid, rejected


def _fit(model_id: str, config: dict[str, Any], x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int):
    if model_id == "random_forest":
        return RandomForestRegressor(**config, random_state=seed, n_jobs=1).fit(x, y, sample_weight=weights)
    if model_id == "svr_rbf":
        return SVR(kernel="rbf", **config).fit(x, y, sample_weight=weights)
    raise ValueError(model_id)


def _scheme_best_configs() -> dict[str, dict[str, Any]]:
    comparison = pd.read_csv(ROUND2_DIR / "feature_comparison.csv")
    result = {}
    for model in MODEL_IDS:
        row = comparison[(comparison.model_id == model) & (comparison.feature_group == FEATURE_GROUP)].iloc[0]
        result[model] = json.loads(row.config)
    return result


def _run_cell(model_id: str, config: dict[str, Any], scheme: str, spec: dict[str, Any], fold: dict[str, str],
              train: pd.DataFrame, validation: pd.DataFrame, source_hash: str, run_id: str, spec_hash: str,
              feature_module_hash: str) -> dict[str, Any]:
    augmentation_start = time.perf_counter()
    augmented, augmentation_info = _generate_training_scheme(train, scheme, spec, 20261115)
    augmentation_seconds = time.perf_counter() - augmentation_start
    weights = augmentation_info.pop("sample_weights")
    climate_source = train.copy()
    pair = build_feature_pair(augmented, validation, FEATURE_GROUP,
                              climate_source=climate_source, selection_source=train,
                              sample_weight=weights, include_missing_indicators=spec["mask_indicators"])
    y_train, y_val = pair.y_train, pair.y_validation
    fit_seeds = SEEDS if model_id == "random_forest" else (SEEDS[0],)
    models, seed_results = [], []
    wall, cpu, peak_rss = 0.0, 0.0, 0
    for seed in fit_seeds:
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        with PeakRSS() as monitor, warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = _fit(model_id, config, pair.x_train, y_train, weights, seed)
            train_prediction = np.asarray(model.predict(pair.x_train), dtype=float) + pair.offset_train
            val_prediction = np.asarray(model.predict(pair.x_validation), dtype=float) + pair.offset_validation
        if not np.isfinite(train_prediction).all() or not np.isfinite(val_prediction).all():
            raise ValueError(f"{model_id} produced non-finite prediction under {scheme}")
        wall += time.perf_counter() - started_wall
        cpu += time.process_time() - started_cpu
        peak_rss = max(peak_rss, int(monitor.peak))
        models.append(model)
        seed_results.append({"seed": seed,
                             "train_metrics": regression_metrics(augmented.target_value.to_numpy(dtype=float), train_prediction),
                             "validation_metrics": regression_metrics(validation.target_value.to_numpy(dtype=float), val_prediction),
                             "warnings": [f"{type(item.message).__name__}: {str(item.message)[:200]}" for item in caught]})
    pred_clean = np.mean([m.predict(pair.x_validation) + pair.offset_validation for m in models], axis=0)
    predictions = [{"sample_id": str(row.sample_id), "target_month": str(row.target_month),
                    "y_true": float(validation.target_value.iloc[i]), "y_pred": float(pred_clean[i])}
                   for i, row in validation.reset_index(drop=True).iterrows()]

    loc, noise_scale, _ = fit_preprocessing_v1_1(train)
    robust_rows = []
    robust_start = time.perf_counter()
    for scenario in ROBUST_SCENARIOS:
        for intensity in ROBUST_LEVELS:
            noisy_frame, eligible, rejected = _corrupted_validation(validation, scenario, intensity, loc, noise_scale,
                                                                     smooth=bool(spec["smoothed"]))
            noisy_pair = build_feature_pair(augmented, noisy_frame, FEATURE_GROUP,
                                            climate_source=climate_source, selection_source=train,
                                            sample_weight=weights, include_missing_indicators=spec["mask_indicators"])
            noisy_predictions = np.mean([m.predict(noisy_pair.x_validation) + noisy_pair.offset_validation for m in models], axis=0)
            metric = regression_metrics(validation.target_value.to_numpy(dtype=float)[eligible], noisy_predictions[eligible])
            robust_rows.append({"scenario": scenario, "intensity": float(intensity),
                                "eligible_targets": int(eligible.sum()), "excluded_physical_range": len(rejected),
                                **metric, "rejected_sample_ids_sha256": hashlib.sha256("|".join(rejected).encode()).hexdigest()})
    robust_seconds = time.perf_counter() - robust_start
    return {
        "run_id": run_id, "candidate_spec_sha256": spec_hash, "round": 3, "model_id": model_id,
        "feature_group": FEATURE_GROUP, "training_scheme": scheme, "config": config, "fold_id": fold["fold_id"],
        "status": "COMPLETED", "train_count_original_months": int(train.target_month.nunique()),
        "train_count_augmented_rows": int(len(augmented)), "validation_count": int(len(validation)),
        "clean_validation_metrics": regression_metrics(validation.target_value.to_numpy(dtype=float), pred_clean),
        "seed_metrics": seed_results, "predictions": predictions, "noise_metrics": robust_rows,
        "augmentation": augmentation_info, "augmentation_generation_seconds": augmentation_seconds,
        "fit_wall_seconds_sum_seeds": wall, "fit_cpu_seconds_sum_seeds": cpu,
        "robustness_wall_seconds": robust_seconds, "peak_rss_bytes": peak_rss,
        "source_sha256": source_hash, "dependency_hashes": {"optimization_features": feature_module_hash,
                                                               "augmentation": sha256_file(ROOT / "src/kuntur_data/phase2/augmentation.py"),
                                                               "predictive": sha256_file(ROOT / "src/kuntur_data/phase2/predictive.py")},
    }


def _summarize(source_hash: str, protocol_hash: str, script_hash: str, feature_hash: str, elapsed: float) -> dict[str, Any]:
    records = [r for r in _load_jsonl(REGISTRY) if r.get("round") == 3 and r.get("status") == "COMPLETED"]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in records:
        groups.setdefault((row["model_id"], row["training_scheme"]), []).append(row)
    rows = []
    robust_agg = []
    for (model_id, scheme), fold_rows in groups.items():
        if len(fold_rows) != 3:
            continue
        fold_rows = sorted(fold_rows, key=lambda r: r["fold_id"])
        metrics = [r["clean_validation_metrics"] for r in fold_rows]
        noise_by_key: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for fold_row in fold_rows:
            for metric in fold_row["noise_metrics"]:
                noise_by_key.setdefault((metric["scenario"], float(metric["intensity"])), []).append(metric)
        noise_means = {f"{name}@{level:.2f}": float(np.mean([r["mae"] for r in vals]))
                       for (name, level), vals in noise_by_key.items()}
        worst_noise = max(noise_means.values())
        rows.append({
            "round": 3, "model_id": model_id, "training_scheme": scheme,
            "feature_group": FEATURE_GROUP, "fold_count": 3, "target_count": sum(m["sample_count"] for m in metrics),
            "mean_fold_mae": float(np.mean([m["mae"] for m in metrics])),
            "mean_fold_rmse": float(np.mean([m["rmse"] for m in metrics])),
            "mean_fold_bias": float(np.mean([m["mean_bias"] for m in metrics])),
            "max_fold_mae": float(max(m["mae"] for m in metrics)),
            "worst_noise_mean_fold_mae": worst_noise,
            "mean_fit_wall_seconds_per_fold": float(np.mean([r["fit_wall_seconds_sum_seeds"] for r in fold_rows])),
            "peak_rss_bytes": max(int(r["peak_rss_bytes"]) for r in fold_rows),
            "training_synthetic_rows_mean": float(np.mean([r["augmentation"]["synthetic_rows"] for r in fold_rows])),
            "candidate_spec_sha256": fold_rows[0]["candidate_spec_sha256"], "status": "COMPLETED_DEVELOPMENT_ONLY",
        })
        for (name, level), vals in sorted(noise_by_key.items()):
            robust_agg.append({"round": 3, "model_id": model_id, "training_scheme": scheme,
                               "scenario": name, "intensity": level,
                               "mean_fold_mae": float(np.mean([r["mae"] for r in vals])),
                               "mean_fold_rmse": float(np.mean([r["rmse"] for r in vals])),
                               "mean_fold_bias": float(np.mean([r["mean_bias"] for r in vals])),
                               "valid_targets_min": int(min(r["eligible_targets"] for r in vals)),
                               "valid_targets_max": int(max(r["eligible_targets"] for r in vals)),
                               "status": "COMPLETED_DEVELOPMENT_ROBUSTNESS"})
    comparison = pd.DataFrame(rows)
    robustness = pd.DataFrame(robust_agg)
    if comparison.empty or robustness.empty:
        raise RuntimeError("No complete augmentation/robustness candidates available")
    clean_best = comparison.sort_values(["mean_fold_mae", "mean_fold_rmse", "max_fold_mae", "model_id", "training_scheme"], kind="stable").iloc[0]
    clean_only = {(r["model_id"], r["fold_id"]): r for r in records if r["training_scheme"] == "clean_only"}
    candidate_records = [r for r in records if r["model_id"] == clean_best.model_id and r["training_scheme"] == clean_best.training_scheme]
    baseline_records = [clean_only[(str(clean_best.model_id), fold["fold_id"])] for fold in FOLDS]
    paired = _block_bootstrap_delta(candidate_records, baseline_records, seed=20261118)
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_path = OUT / "augmentation_comparison.csv"
    robustness_path = OUT / "noise_robustness_by_scheme.csv"
    comparison.to_csv(comparison_path, index=False, lineterminator="\n")
    robustness.to_csv(robustness_path, index=False, lineterminator="\n")
    payload = {
        "manifest_version": "kuntur-phase2-v1.2-round-3-augmentation-robustness-0.1.0",
        "status": "COMPLETED_DEVELOPMENT_ONLY", "round": 3,
        "root_cause_hypothesis": "Fold-local training augmentation may help models tolerate input corruption, but repeated synthetic samples can overweight authentic months and can damage clean forecasts.",
        "dataset": {"path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet", "sha256": source_hash,
                    "target_range_loaded": ["2003-03", "2018-12"], "validation_targets": 108, "sealed_2025_read": False},
        "training_schemes": scheme_spec(), "model_ids": list(MODEL_IDS), "feature_group": FEATURE_GROUP,
        "test_noise_scenarios": list(ROBUST_SCENARIOS), "test_noise_levels": list(ROBUST_LEVELS),
        "candidate_count": len(comparison), "completed_candidate_folds": len(records),
        "expected_candidate_folds": len(scheme_spec()) * len(MODEL_IDS) * len(FOLDS),
        "weights": "balanced schemes assign 0.5 total weight to authentic clean rows and 0.5 across each original month's accepted synthetic variants; clean-only has weight 1.0; explicit naive scheme assigns 1 to every row. Targets are never perturbed.",
        "augmentation_scales": "Estimated from only clean outer-fold training histories; identical per-sample noise seeds are reused across models and schemes. Broad physical-range violations are rejected without clipping.",
        "best_candidate": {"model_id": str(clean_best.model_id), "training_scheme": str(clean_best.training_scheme),
                           "mean_fold_mae": float(clean_best.mean_fold_mae), "mean_fold_rmse": float(clean_best.mean_fold_rmse),
                           "mean_fold_bias": float(clean_best.mean_fold_bias), "worst_fold_mae": float(clean_best.max_fold_mae),
                           "worst_noise_mean_fold_mae": float(clean_best.worst_noise_mean_fold_mae),
                           "candidate_spec_sha256": str(clean_best.candidate_spec_sha256)},
        "paired_vs_same_model_clean_only": paired,
        "resources": {"elapsed_wall_seconds": elapsed, "logical_cpus": psutil.cpu_count(logical=True), "physical_cpus": psutil.cpu_count(logical=False)},
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "protocol_sha256": protocol_hash, "script_sha256": script_hash, "feature_module_sha256": feature_hash,
        "artifacts": {"augmentation_comparison": {"path": comparison_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(comparison_path)},
                      "noise_robustness_by_scheme": {"path": robustness_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(robustness_path)},
                      "fold_registry": {"path": REGISTRY.relative_to(ROOT).as_posix(), "sha256": sha256_file(REGISTRY)}},
        "limitations": ["Development rolling folds also selected the Round 2 challenger and all Round 3 training schemes; cross-scheme intervals are descriptive post-selection evidence.",
                        "Only the two Round 2 leading model families were evaluated for augmentation.",
                        "Corruption robustness is an input sensitivity experiment, not evidence of real-world sensor error rates.",
                        "The old 2019-2024 partitions and sealed 2025 targets were not read."],
    }
    manifest_path = OUT / "round_manifest.json"
    _atomic_json(manifest_path, payload)
    print(json.dumps({"status": payload["status"], "candidate_count": len(comparison),
                      "completed_candidate_folds": len(records), "best_candidate": payload["best_candidate"],
                      "paired_vs_clean": paired, "manifest": manifest_path.relative_to(ROOT).as_posix()}, indent=2, allow_nan=False))


def main() -> None:
    started = time.perf_counter()
    if not FORECAST.is_file():
        raise SystemExit("Frozen development forecast data are missing")
    source_hash = sha256_file(FORECAST)
    protocol_hash = sha256_file(ROOT / "docs/protocols/continuous_optimization_v1_2.md")
    script_hash = sha256_file(Path(__file__))
    feature_hash = sha256_file(ROOT / "src/kuntur_data/phase2/optimization_features.py")
    frame = pd.read_parquet(FORECAST, filters=[("target_month", "<=", "2018-12")])
    if frame.empty or frame.target_month.max() != "2018-12":
        raise SystemExit("Development-only filter did not enforce the 2018 cutoff")
    if not np.all(pd.PeriodIndex(frame.history_end_month.astype(str), freq="M") == pd.PeriodIndex(frame.target_month.astype(str), freq="M") - 3):
        raise SystemExit("Frozen T-3 history policy failed")
    selected_configs = _scheme_best_configs()
    schemes = scheme_spec()
    existing = {(r["run_id"], r["fold_id"]) for r in _load_jsonl(REGISTRY) if r.get("status") == "COMPLETED"}
    OUT.mkdir(parents=True, exist_ok=True)
    for fold in FOLDS:
        train = frame[frame.target_month <= fold["train_end"]].reset_index(drop=True)
        validation = frame[(frame.target_month >= fold["validation_start"]) & (frame.target_month <= fold["validation_end"])].reset_index(drop=True)
        if len(validation) != 36 or train.target_month.max() >= validation.target_month.min():
            raise SystemExit(f"Invalid fold chronology: {fold['fold_id']}")
        for scheme, spec in schemes.items():
            for model_id in MODEL_IDS:
                config = selected_configs[model_id]
                descriptor = {"round": 3, "source_sha256": source_hash, "protocol_sha256": protocol_hash,
                              "script_sha256": script_hash, "feature_module_sha256": feature_hash,
                              "augmentation_module_sha256": sha256_file(ROOT / "src/kuntur_data/phase2/augmentation.py"),
                              "model_id": model_id, "feature_group": FEATURE_GROUP, "training_scheme": scheme,
                              "scheme_spec": spec, "config": config, "folds": FOLDS}
                spec_hash = _sha_obj(descriptor)
                run_id = f"r3-{scheme}-{model_id}-{spec_hash[:12]}"
                if (run_id, fold["fold_id"]) in existing:
                    continue
                try:
                    row = _run_cell(model_id, config, scheme, spec, fold, train, validation,
                                    source_hash, run_id, spec_hash, feature_hash)
                except Exception as exc:
                    row = {"run_id": run_id, "candidate_spec_sha256": spec_hash, "round": 3,
                           "model_id": model_id, "feature_group": FEATURE_GROUP, "training_scheme": scheme,
                           "config": config, "fold_id": fold["fold_id"], "status": "FAILED",
                           "error_type": type(exc).__name__, "error": str(exc)[:500], "source_sha256": source_hash}
                _append(row)
                if row["status"] == "COMPLETED":
                    existing.add((run_id, fold["fold_id"]))
                print(json.dumps({"event": "ROUND3_AUGMENTATION_FOLD_COMPLETE", "fold_id": fold["fold_id"],
                                  "training_scheme": scheme, "model_id": model_id, "status": row["status"],
                                  "clean_mae": row.get("clean_validation_metrics", {}).get("mae"),
                                  "noise_scenarios": len(row.get("noise_metrics", [])),
                                  "elapsed_seconds": round(time.perf_counter() - started, 2)}, allow_nan=False), flush=True)
    _summarize(source_hash, protocol_hash, script_hash, feature_hash, time.perf_counter() - started)


if __name__ == "__main__":
    main()
