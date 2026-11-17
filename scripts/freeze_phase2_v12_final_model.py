#!/usr/bin/env python3
"""Fit and freeze the development-selected KUNTUR v1.2 model through 2018 only."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kuntur_data.phase2.optimization_features import build_feature_pair
from kuntur_data.phase2.predictive import fit_preprocessing_v1_1, sha256_file
import run_phase2_v12_round1 as round1
import run_phase2_v12_round3_augmentation as round3


FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
ROUND3_DIR = ROOT / "results/optimization/round_03_augmentation_robustness"
ROUND5_DIR = ROOT / "results/optimization/round_05_interval_calibration"
OUT = ROOT / "results/optimization/round_06_frozen_test"
BEST_MODEL_ID = "random_forest"
BEST_SCHEME = "heteroscedastic_only"
FEATURE_GROUP = "C_training_anomalies"
CUTOFF = "2018-12"
SEEDS = (20261115, 20261116, 20261117)
AUGMENTATION_SEED = 20261115


def _json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    round3_manifest = json.loads((ROUND3_DIR / "round_manifest.json").read_text(encoding="utf-8"))
    if round3_manifest["best_candidate"]["model_id"] != BEST_MODEL_ID or round3_manifest["best_candidate"]["training_scheme"] != BEST_SCHEME:
        raise ValueError("The final model does not match the selected Round 3 candidate")
    if round3_manifest["best_candidate"]["candidate_spec_sha256"] != "98876e3caf905353b8eadb3afc77f29fae766a2b1cf75fe8915b4c32791b54e8":
        raise ValueError("The selected candidate specification hash differs from the frozen choice")
    source_sha = sha256_file(FORECAST)
    if source_sha != round3_manifest["dataset"]["sha256"]:
        raise ValueError("Frozen forecast Gold hash differs from Round 3")

    # The parquet reader predicate is the only label-reading path in this fit.
    train = pd.read_parquet(FORECAST, filters=[("target_month", "<=", CUTOFF)])
    if train.empty or train.target_month.max() != CUTOFF or train.target_month.min() != "2003-03":
        raise ValueError("The final training frame failed its frozen date boundary")
    if not train.target_month.is_monotonic_increasing or train.sample_id.duplicated().any():
        raise ValueError("The final training frame is not unique and chronological")
    history_end = pd.PeriodIndex(train.history_end_month.astype(str), freq="M")
    target = pd.PeriodIndex(train.target_month.astype(str), freq="M")
    if not np.all(history_end == target - 3):
        raise ValueError("The final model training histories violate the T-3 boundary")

    scheme = round3.scheme_spec()[BEST_SCHEME]
    augmented, augmentation = round3._generate_training_scheme(train, BEST_SCHEME, scheme, AUGMENTATION_SEED)
    weights = augmentation.pop("sample_weights")
    if int(augmentation["synthetic_rows"]) != 4 * len(train) or int(augmentation["rejected_variants"]) != 0:
        raise ValueError("Final training augmentation differs from the accepted Round 3 scheme")
    if not np.allclose(
        pd.DataFrame({"sample_id": augmented.sample_id.str.split("__").str[0], "weight": weights}).groupby("sample_id").weight.sum().to_numpy(),
        1.0, atol=1e-12,
    ):
        raise ValueError("Balanced augmentation weights do not sum to one per authentic month")
    pair = build_feature_pair(
        augmented, augmented, FEATURE_GROUP,
        climate_source=train, selection_source=train,
        sample_weight=weights, include_missing_indicators=True,
    )
    if not np.isfinite(pair.x_train).all() or not np.isfinite(pair.y_train).all():
        raise ValueError("Final training features or labels contain nonfinite values after the frozen transforms")

    selected_records = [
        json.loads(line) for line in (ROUND3_DIR / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    configs = {
        json.dumps(row["config"], sort_keys=True)
        for row in selected_records
        if row.get("status") == "COMPLETED" and row.get("model_id") == BEST_MODEL_ID and row.get("training_scheme") == BEST_SCHEME
    }
    if len(configs) != 1:
        raise ValueError(f"The selected folds do not agree on one model configuration: {len(configs)}")
    config = json.loads(next(iter(configs)))
    if config != {"max_depth": 4, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 120}:
        raise ValueError("The final Random Forest config differs from the selected Round 3 candidate")
    models = [
        RandomForestRegressor(**config, random_state=seed, n_jobs=1).fit(pair.x_train, pair.y_train, sample_weight=weights)
        for seed in SEEDS
    ]
    climatology = {
        str(month): float(values.mean())
        for month, values in train.assign(_month=pd.PeriodIndex(train.target_month.astype(str), freq="M").month).groupby("_month").target_value
    }
    interval_manifest = json.loads((ROUND5_DIR / "round_manifest.json").read_text(encoding="utf-8"))
    interval_calibration = interval_manifest["frozen_interval_calibration"]
    if interval_calibration["source_candidate_spec_sha256"] != round3_manifest["best_candidate"]["candidate_spec_sha256"]:
        raise ValueError("Interval calibration is linked to a different model candidate")
    bundle = {
        "models": models,
        "model_id": BEST_MODEL_ID,
        "training_scheme": BEST_SCHEME,
        "feature_group": FEATURE_GROUP,
        "config": config,
        "seeds": list(SEEDS),
        "candidate_spec_sha256": round3_manifest["best_candidate"]["candidate_spec_sha256"],
        "training_cutoff_month": CUTOFF,
        "train_month_count": int(train.target_month.nunique()),
        "augmented_row_count": int(len(augmented)),
        "feature_column_count": int(pair.x_train.shape[1]),
        "monthly_target_climatology_c": climatology,
        "interval_calibration": interval_calibration,
        "source_forecast_sha256": source_sha,
    }
    model_path = OUT / "model_bundle.joblib"
    temp_model = model_path.with_suffix(".joblib.tmp")
    joblib.dump(bundle, temp_model, compress=3)
    temp_model.replace(model_path)
    module_paths = [
        ROOT / "scripts/run_phase2_v12_round1.py",
        ROOT / "scripts/run_phase2_v12_round3_augmentation.py",
        ROOT / "src/kuntur_data/phase2/augmentation.py",
        ROOT / "src/kuntur_data/phase2/optimization_features.py",
        ROOT / "src/kuntur_data/phase2/predictive.py",
    ]
    manifest = {
        "manifest_version": "kuntur-phase2-v1.2-round-6-model-freeze-0.1.0",
        "status": "MODEL_AND_SCORER_FROZEN_TEST_NOT_ACCESSED",
        "test_target_values_accessed": False,
        "model": {
            "model_id": BEST_MODEL_ID, "training_scheme": BEST_SCHEME,
            "feature_group": FEATURE_GROUP, "config": config, "seeds": list(SEEDS),
            "candidate_spec_sha256": round3_manifest["best_candidate"]["candidate_spec_sha256"],
            "training_cutoff_month": CUTOFF,
            "training_target_range": [str(train.target_month.min()), str(train.target_month.max())],
            "training_rows": int(len(train)), "authentic_training_months": int(train.target_month.nunique()),
            "augmented_rows": int(len(augmented)), "synthetic_rows": int(augmentation["synthetic_rows"]),
            "feature_count": int(pair.x_train.shape[1]), "source_forecast_sha256": source_sha,
            "augmentation": {k: v for k, v in augmentation.items() if k != "sample_weights"},
        },
        "interval_calibration": interval_calibration,
        "test_contract": {
            "dataset_id": "M2TMNXSLV.5.12.4", "target_variable": "merra2_t2m_c",
            "region": "ecuador_pilot_bbox", "target_months": ["2025-01", "2025-12"],
            "features": "MERRA-2 T-14 through T-3; latest predictor month is target-3; 12 monthly records × five contracted channels.",
            "target_outcome": "area-weighted Ecuador pilot-bbox monthly mean T2M converted to Celsius, minimum valid area fraction 0.8.",
            "scores": ["MAE", "RMSE", "mean prediction-minus-observation bias", "paired MAE difference vs training-only monthly climatology", "90% interval coverage, width and Winkler score"],
            "paired_block_bootstrap": {"block_length_months": 12, "replicates": 2000, "seed": 20261127, "interpretation": "With only 12 test months (one full block), the interval is degenerate and is reported as non-informative; no precision claim is made."},
            "evaluation_label": "retrospective hindcast; contemporaneous MERRA-2 release timestamps are not established",
        },
        "artifacts": {"model_bundle": {"path": model_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(model_path), "bytes": model_path.stat().st_size}},
        "code_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in module_paths},
        "scorer_sha256": sha256_file(ROOT / "scripts/score_phase2_v12_sealed_2025.py"),
        "training_script_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_sha256": sha256_file(ROOT / "docs/protocols/continuous_optimization_v1_2.md"),
        "interval_manifest_sha256": sha256_file(ROUND5_DIR / "round_manifest.json"),
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "resources": {"elapsed_wall_seconds": time.perf_counter() - started},
    }
    _atomic_json(OUT / "frozen_test_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "training_cutoff": CUTOFF, "training_months": len(train), "synthetic_rows": augmentation["synthetic_rows"], "model_sha256": manifest["artifacts"]["model_bundle"]["sha256"], "scorer_sha256": manifest["scorer_sha256"], "test_target_values_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
