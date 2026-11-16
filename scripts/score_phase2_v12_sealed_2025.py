#!/usr/bin/env python3
"""Open the frozen Jan–Dec 2025 MERRA-2 test values once and score v1.2."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kuntur_data.phase2.augmentation import FEATURE_COLUMNS, LAGS
from kuntur_data.phase2.optimization_features import build_feature_pair
from kuntur_data.phase2.predictive import sha256_file
import run_phase2_v12_round3_augmentation as round3


FORECAST = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
HISTORICAL_GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
TEST_GOLD = ROOT / "data/gold/phase2_v12_sealed_2025_merra2_bbox.parquet"
ROUND3_DIR = ROOT / "results/optimization/round_03_augmentation_robustness"
ROUND5_DIR = ROOT / "results/optimization/round_05_interval_calibration"
FREEZE_DIR = ROOT / "results/optimization/round_06_frozen_test"
MODEL_PATH = FREEZE_DIR / "model_bundle.joblib"
FREEZE_MANIFEST_PATH = FREEZE_DIR / "frozen_test_manifest.json"
OUTPUT_DIR = ROOT / "results/optimization/round_06_2025_test_score"
TRAIN_CUTOFF = "2018-12"
PREDICTOR_VARIABLES = ("merra2_t2m_c", "merra2_qv2m", "merra2_u2m", "merra2_v2m", "merra2_ps")
TEST_MONTHS = tuple(str(x) for x in pd.period_range("2025-01", "2025-12", freq="M"))
MIN_VALID_AREA = 0.80


def make_test_frame(historical_gold: pd.DataFrame, test_gold: pd.DataFrame) -> pd.DataFrame:
    """Build twelve T-14..T-3 frames without reading labels outside 2025."""
    required = {"month", "variable_id", "value", "valid_area_fraction_bbox"}
    for name, frame in (("historical Gold predictors", historical_gold), ("2025 Gold test", test_gold)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    hist = historical_gold[
        historical_gold.variable_id.isin(PREDICTOR_VARIABLES)
        & historical_gold.month.astype(str).between("2023-11", "2024-12")
    ].copy()
    test = test_gold[test_gold.variable_id.isin(PREDICTOR_VARIABLES)].copy()
    test["month"] = test.month.astype(str)
    if set(test.month.unique()) != set(TEST_MONTHS) or test.duplicated(["month", "variable_id"]).any():
        raise ValueError("The sealed target file must have exactly one row per contracted variable for all 12 months of 2025")
    test_variables = test.groupby("month").variable_id.apply(set)
    if not all(set(PREDICTOR_VARIABLES) == variables for variables in test_variables):
        raise ValueError("One or more 2025 months lack a contracted MERRA-2 variable")
    hist["month"] = hist.month.astype(str)
    if hist.duplicated(["month", "variable_id"]).any():
        raise ValueError("Historical predictor Gold has duplicate month/variable rows")
    all_rows = pd.concat([hist, test], ignore_index=True)
    values = all_rows.set_index([all_rows.month.astype(str), all_rows.variable_id], drop=False)
    target_rows = test[(test.variable_id == "merra2_t2m_c") & test.month.isin(TEST_MONTHS)].set_index("month")
    records = []
    for target_month in TEST_MONTHS:
        target_period = pd.Period(target_month, freq="M")
        target_row = target_rows.loc[target_month]
        if pd.isna(target_row.value) or float(target_row.valid_area_fraction_bbox) < MIN_VALID_AREA:
            raise ValueError(f"2025 target failed the frozen area coverage gate: {target_month}")
        record = {
            "sample_id": f"KUNTUR_ECUADOR_BBOX_T2M_PLUS1_V1:{target_month}",
            "target_month": target_month,
            "forecast_origin_month": str(target_period - 1),
            "issue_date_utc": f"{target_period - 1}-01T00:00:00Z",
            "history_start_month": str(target_period - 14),
            "history_end_month": str(target_period - 3),
            "target_value": float(target_row.value),
            "target_valid_area_fraction_bbox": float(target_row.valid_area_fraction_bbox),
        }
        if record["history_end_month"] >= record["forecast_origin_month"]:
            raise AssertionError(f"T-3 history does not precede the issue month for {target_month}")
        for lag in LAGS:
            feature_month = str(target_period - (lag + 2))
            for variable_id in PREDICTOR_VARIABLES:
                key = (feature_month, variable_id)
                if key not in values.index:
                    raise ValueError(f"Missing 2025 predictor month/variable: {feature_month} {variable_id}")
                feature = values.loc[key]
                if isinstance(feature, pd.DataFrame):
                    raise ValueError(f"Ambiguous predictor month/variable: {feature_month} {variable_id}")
                if pd.isna(feature.value) or float(feature.valid_area_fraction_bbox) < MIN_VALID_AREA:
                    raise ValueError(f"Predictor failed the frozen area coverage gate: {feature_month} {variable_id}")
                record[f"{variable_id}_lag_{lag:02d}"] = float(feature.value)
        records.append(record)
    frame = pd.DataFrame(records)
    if len(frame) != 12 or frame.target_month.duplicated().any() or set(FEATURE_COLUMNS) - set(frame.columns):
        raise ValueError("The 2025 forecast frame failed the twelve-row frozen feature contract")
    return frame


def _paired_circular_block_ci(delta: np.ndarray, *, replicates: int = 2000, seed: int = 20261127, block: int = 12) -> dict:
    values = np.asarray(delta, dtype=float)
    if values.shape != (12,):
        raise ValueError("Frozen test contains exactly one 12-month year")
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for i in range(replicates):
        indexes: list[int] = []
        while len(indexes) < len(values):
            start = int(rng.integers(0, len(values)))
            indexes.extend(((start + np.arange(block)) % len(values)).tolist())
        draws[i] = float(np.mean(values[np.asarray(indexes[:len(values)])]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "delta_mae_ci95": [float(low), float(high)],
        "replicates": replicates, "seed": seed, "block_length_months": block,
        "degenerate_single_block_interval": bool(np.isclose(low, high, rtol=0.0, atol=1e-12)),
        "interpretation": "non-informative: a 12-month resampling block equals the entire one-year test set" if block == len(values) else "low precision with only 12 monthly outcomes",
    }


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    return {"target_count": int(len(error)), "mae_c": float(np.mean(np.abs(error))), "rmse_c": float(np.sqrt(np.mean(error**2))), "mean_prediction_minus_observation_bias_c": float(np.mean(error))}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    freeze = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "MODEL_AND_SCORER_FROZEN_TEST_NOT_ACCESSED" or freeze.get("test_target_values_accessed") is not False:
        raise ValueError("The frozen test gate is not satisfied")
    if freeze.get("scorer_sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("Scorer source changed after the evaluation protocol was frozen")
    if freeze["artifacts"]["model_bundle"]["sha256"] != sha256_file(MODEL_PATH):
        raise ValueError("Frozen model bundle hash mismatch")
    if freeze["interval_manifest_sha256"] != sha256_file(ROUND5_DIR / "round_manifest.json"):
        raise ValueError("Frozen interval calibration manifest hash mismatch")
    if not TEST_GOLD.is_file():
        raise FileNotFoundError("The 2025 regional MERRA-2 test file is not available")

    bundle = joblib.load(MODEL_PATH)
    if bundle["training_cutoff_month"] != TRAIN_CUTOFF or bundle["candidate_spec_sha256"] != freeze["model"]["candidate_spec_sha256"]:
        raise ValueError("Loaded model bundle does not match the frozen candidate")
    training = pd.read_parquet(FORECAST, filters=[("target_month", "<=", TRAIN_CUTOFF)])
    if training.empty or training.target_month.max() != TRAIN_CUTOFF or sha256_file(FORECAST) != bundle["source_forecast_sha256"]:
        raise ValueError("Training source does not match the frozen data and cutoff")
    hist_months = pd.read_parquet(
        HISTORICAL_GOLD,
        filters=[("month", ">=", "2023-11"), ("month", "<=", "2024-12")],
        columns=["month", "variable_id", "value", "valid_area_fraction_bbox"],
    )
    test_gold = pd.read_parquet(TEST_GOLD)
    test_frame = make_test_frame(hist_months, test_gold)

    scheme = round3.scheme_spec()[bundle["training_scheme"]]
    augmented, augmentation = round3._generate_training_scheme(training, bundle["training_scheme"], scheme, 20261115)
    weights = augmentation.pop("sample_weights")
    pair = build_feature_pair(
        augmented, test_frame, bundle["feature_group"], climate_source=training,
        selection_source=training, sample_weight=weights, include_missing_indicators=True,
    )
    per_seed = [np.asarray(model.predict(pair.x_validation), dtype=float) + pair.offset_validation for model in bundle["models"]]
    prediction = np.mean(per_seed, axis=0)
    observation = test_frame.target_value.to_numpy(dtype=float)
    if not np.isfinite(prediction).all() or not np.isfinite(observation).all():
        raise ValueError("The frozen test scorer produced nonfinite predictions or observations")
    baseline = np.asarray([bundle["monthly_target_climatology_c"][str(pd.Period(m, freq="M").month)] for m in test_frame.target_month], dtype=float)
    interval_q = float(bundle["interval_calibration"]["absolute_residual_quantile_c"])
    lower, upper = prediction - interval_q, prediction + interval_q
    model_metrics = _metrics(observation, prediction)
    baseline_metrics = _metrics(observation, baseline)
    delta = np.abs(prediction - observation) - np.abs(baseline - observation)
    paired = {
        "paired_delta_mae_model_minus_training_monthly_climatology_c": float(np.mean(delta)),
        "months_with_lower_absolute_error_than_baseline": int(np.sum(delta < 0)),
        "months_with_equal_absolute_error": int(np.sum(np.isclose(delta, 0.0, atol=1e-12))),
        **_paired_circular_block_ci(delta),
    }
    alpha = 1.0 - float(bundle["interval_calibration"]["nominal_coverage"])
    interval_rows = []
    for row, pred, obs, low, high in zip(test_frame.to_dict("records"), prediction, observation, lower, upper):
        interval_rows.append({
            "target_month": row["target_month"], "forecast_origin_month": row["forecast_origin_month"],
            "history_start_month": row["history_start_month"], "history_end_month": row["history_end_month"],
            "observed_t2m_c": float(obs), "predicted_t2m_c": float(pred),
            "baseline_monthly_climatology_c": float(baseline[len(interval_rows)]),
            "interval_lower_c": float(low), "interval_upper_c": float(high),
            "interval_covered": bool(low <= obs <= high),
            "interval_width_c": float(high - low),
            "winkler_score_c": float((high - low) + 2.0 / alpha * max(low - obs, 0.0) + 2.0 / alpha * max(obs - high, 0.0)),
            "target_valid_area_fraction_bbox": float(row["target_valid_area_fraction_bbox"]),
        })
    interval_frame = pd.DataFrame(interval_rows)
    summary = {
        "status": "SCORED_FROZEN_2025_TEST",
        "dataset_id": "M2TMNXSLV.5.12.4", "target_variable": "MERRA-2 T2M (°C)",
        "region_id": "ecuador_pilot_bbox", "test_months": list(test_frame.target_month),
        "test_target_count": int(len(test_frame)),
        "model": {"id": bundle["model_id"], "training_scheme": bundle["training_scheme"], "candidate_spec_sha256": bundle["candidate_spec_sha256"], "model_bundle_sha256": sha256_file(MODEL_PATH)},
        "training": {"cutoff_month": TRAIN_CUTOFF, "months": bundle["train_month_count"], "forecast_source_sha256": sha256_file(FORECAST)},
        "model_metrics": model_metrics, "baseline_metrics": baseline_metrics, "paired_baseline_comparison": paired,
        "interval_calibration": {
            "method": bundle["interval_calibration"]["method"], "nominal_coverage": float(bundle["interval_calibration"]["nominal_coverage"]),
            "absolute_residual_quantile_c": interval_q, "calibration_residual_count": int(bundle["interval_calibration"]["residual_count"]),
            "observed_coverage": float(interval_frame.interval_covered.mean()),
            "mean_width_c": float(interval_frame.interval_width_c.mean()), "mean_winkler_score_c": float(interval_frame.winkler_score_c.mean()),
            "status": "one-year observed coverage is descriptive; calibration residuals are post-selection and serially dependent",
        },
        "target_coverage_fraction_bbox_min_max": [float(test_frame.target_valid_area_fraction_bbox.min()), float(test_frame.target_valid_area_fraction_bbox.max())],
        "predictor_coverage_min": float(test_gold[test_gold.variable_id.isin(PREDICTOR_VARIABLES)].valid_area_fraction_bbox.min()),
        "source_test_gold_sha256": sha256_file(TEST_GOLD),
        "test_gold_rows": int(len(test_gold)),
        "evaluation_label": "retrospective hindcast; contemporaneous product-release timestamps were not established",
        "limitations": ["Only 12 months are in this independent temporal test; metrics have low precision.", "The 12-month circular block-bootstrap CI resamples one complete annual block and is degenerate, so it is reported only as a non-informative diagnostic.", "The model and interval method were selected using 2003-2018 development data; no 2019-2024 validation or test labels were read by this scorer.", "The finite-sample interval does not guarantee exchangeable coverage for dependent monthly errors."],
    }
    predictions_path = OUTPUT_DIR / "sealed_2025_monthly_predictions.csv"
    metrics_path = OUTPUT_DIR / "sealed_2025_score.json"
    interval_frame.to_csv(predictions_path, index=False)
    summary["artifacts"] = {
        "monthly_predictions": {"path": predictions_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(predictions_path), "rows": int(len(interval_frame))},
    }
    temp = metrics_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(metrics_path)
    print(json.dumps({"status": summary["status"], "target_count": len(test_frame), "model_metrics": model_metrics, "baseline_metrics": baseline_metrics, "paired_baseline_comparison": paired, "interval_calibration": summary["interval_calibration"], "source_test_gold_sha256": summary["source_test_gold_sha256"], "monthly_predictions_sha256": summary["artifacts"]["monthly_predictions"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
