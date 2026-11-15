"""Stable, data-contract-aware interfaces for the KUNTUR downstream app."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .analytical import analyze_variable, cached_analyze_variable, monthly_anomalies
from .augmentation import CHANNELS, PHYSICAL_RANGES
from .ingestion import GlobalAnalyticalCache, PartitionedGoldStore
from .predictive import model_inputs


REGION_ID = "ecuador_pilot_bbox"
TREND_METHODS = {
    "ols_hac": ("ols_hac_slope_per_decade", "ols_hac_ci95_per_decade", "ols_hac_p_value"),
    "seasonal_theil_sen": ("theil_sen_slope_per_decade", "theil_sen_seasonal_stratum_slope_range_per_decade", None),
    "seasonal_mann_kendall": ("seasonal_mk_s", None, "seasonal_mk_p_value"),
}


def _month(value: str) -> pd.Period:
    try:
        period = pd.Period(str(value), freq="M")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Month must use YYYY-MM: {value!r}") from exc
    if str(period) != str(value):
        raise ValueError(f"Month must use YYYY-MM: {value!r}")
    return period


def query_analytical(
    store: PartitionedGoldStore,
    *,
    region_id: str,
    variable_id: str,
    start_month: str,
    end_month: str,
    trend_method: str = "ols_hac",
    qa_threshold: float = 0.8,
) -> dict:
    """Query one region/variable and return a trend, diagnostics and provenance."""
    start, end = _month(start_month), _month(end_month)
    if start > end:
        raise ValueError("start_month must be on or before end_month")
    if trend_method not in TREND_METHODS:
        raise ValueError(f"Unsupported trend_method: {trend_method}")
    if not np.isfinite(qa_threshold) or not 0.0 <= qa_threshold <= 1.0:
        raise ValueError("qa_threshold must be between 0 and 1")
    if region_id != REGION_ID:
        raise ValueError(f"Region {region_id!r} has no approved Gold mask; only {REGION_ID!r} is available")
    manifest_path = store.partition_root / "partition_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Partitioned Gold is not materialized; call store.materialize() first")
    partition_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = store.query(
        variable_ids=[variable_id], region_id=region_id,
        start_month=str(start), end_month=str(end),
        coverage_threshold=qa_threshold,
    )
    if selected.empty:
        raise ValueError("No Gold observations meet the requested date and QA filters")
    context = {
        "gold_sha256": partition_manifest["source_gold_sha256"],
        "contract_version": partition_manifest.get("contract_version", "unknown"),
    }
    cached = cached_analyze_variable(
        selected, variable_id, cache=store.cache, coverage_threshold=qa_threshold,
        key_context=context,
    ) if store.cache is not None else None
    analysis = cached or {"trend": analyze_variable(selected, variable_id, coverage_threshold=qa_threshold)["trend"], "cache_hit": False, "cache_key": None}
    trend = analysis["trend"]
    estimate_key, interval_key, p_key = TREND_METHODS[trend_method]
    series = monthly_anomalies(selected, variable_id, coverage_threshold=qa_threshold)
    expected = pd.period_range(start, end, freq="M")
    available = set(series.loc[series["value"].notna(), "month"].astype(str))
    source_hashes = sorted(selected.source_sha256.dropna().astype(str).unique().tolist()) if "source_sha256" in selected else []
    selected_series = series[[c for c in ("month", "value", "anomaly", "valid_area_fraction_bbox", "calendar_month") if c in series]].copy()
    if "month" in selected_series:
        selected_series["month"] = selected_series["month"].astype(str)
    series_records = [
        {key: (None if pd.isna(value) else value.item() if isinstance(value, np.generic) else value)
         for key, value in row.items()}
        for row in selected_series.to_dict("records")
    ]
    cache_stats = store.cache.stats() if store.cache is not None else None
    if cache_stats is not None:
        cache_stats.pop("root", None)
    return {
        "region_id": region_id,
        "region_status": str(selected.region_status.iloc[0]) if "region_status" in selected else "UNKNOWN",
        "variable_id": variable_id,
        "units": trend["units"],
        "requested_period": [str(start), str(end)],
        "series": series_records,
        "trend": {
            "method": trend_method,
            "estimate": trend[estimate_key],
            "estimate_units": trend["trend_units"] if trend_method != "seasonal_mann_kendall" else "unitless S statistic",
            "uncertainty": trend[interval_key] if interval_key else None,
            "uncertainty_kind": "95% HAC interval" if trend_method == "ols_hac" else ("seasonal-stratum slope range; not a confidence interval" if trend_method == "seasonal_theil_sen" else None),
            "p_value": trend[p_key] if p_key else None,
            "diagnostics": {key: trend.get(key) for key in (
                "n_months", "first_month", "last_month", "median_valid_area_fraction",
                "residual_autocorrelation_lag1", "residual_autocorrelation_lag12",
                "durbin_watson_residual", "seasonal_mk_status",
            )},
            "full_methods": trend,
            "interpretation": "descriptive temporal association; not causal attribution",
        },
        "coverage": {
            "expected_months": len(expected), "eligible_months": len(available),
            "missing_months": [str(month) for month in expected if str(month) not in available],
            "qa_threshold_valid_area_fraction": qa_threshold,
            "median_valid_area_fraction": float(series.valid_area_fraction_bbox.median()),
            "mean_qa_pass_fraction": float(selected.qa_pass_fraction_of_candidates.mean()) if "qa_pass_fraction_of_candidates" in selected else None,
        },
        "cache": {"hit": bool(analysis["cache_hit"]), "key": analysis["cache_key"], "stats": cache_stats},
        "provenance": {
            "source_gold_sha256": partition_manifest["source_gold_sha256"],
            "source_sha256_values": source_hashes,
            "contract_version": context["contract_version"],
            "partition_manifest": str(manifest_path),
            "aggregation": "precomputed Gold spherical-cell-area-weighted mean; native collection grids retained",
        },
    }


def build_forecast_history(history: pd.DataFrame, *, target_month: str, coverage_threshold: float = 0.8) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Validate 12 monthly records ending at target T-3 and return a (1,12,5) array."""
    target = _month(target_month)
    expected = pd.period_range(target - 14, target - 3, freq="M")
    required = {"month", "variable_id", "value", "valid_area_fraction_bbox"}
    missing_cols = required - set(history.columns)
    if missing_cols:
        raise ValueError(f"Forecast history missing fields: {sorted(missing_cols)}")
    frame = history[history.variable_id.isin(CHANNELS)].copy()
    frame["month"] = frame.month.astype(str)
    if frame.duplicated(["month", "variable_id"]).any():
        raise ValueError("Forecast history has duplicate month-variable records")
    expected_keys = {(str(month), channel) for month in expected for channel in CHANNELS}
    actual_keys = set(zip(frame.month, frame.variable_id))
    if not expected_keys.issubset(actual_keys):
        absent = sorted(expected_keys - actual_keys)
        raise ValueError(f"Forecast history does not cover the required T-14..T-3 window: {absent[:5]}")
    frame = frame[frame.apply(lambda row: (row.month, row.variable_id) in expected_keys, axis=1)]
    frame["valid"] = (
        pd.to_numeric(frame.value, errors="coerce").notna()
        & np.isfinite(pd.to_numeric(frame.value, errors="coerce"))
        & (pd.to_numeric(frame.valid_area_fraction_bbox, errors="coerce") >= coverage_threshold)
    )
    indexed = frame.set_index(["month", "variable_id"])
    values = np.full((12, len(CHANNELS)), np.nan, dtype=np.float64)
    for i, month in enumerate(expected):
        for j, channel in enumerate(CHANNELS):
            row = indexed.loc[(str(month), channel)]
            if bool(row.valid):
                value = float(row.value)
                low, high = PHYSICAL_RANGES[channel]
                if not low <= value <= high:
                    raise ValueError(f"Out-of-range predictor {channel} at {month}: {value}")
                values[i, j] = value
    mask = ~np.isfinite(values)
    if mask.mean() > 0.20 or mask.all(axis=0).any():
        raise ValueError("Forecast history exceeds the frozen missingness allowance (20%) or lacks a complete channel")
    lineage = sorted(frame.loc[frame.source_sha256.notna(), "source_sha256"].astype(str).unique().tolist()) if "source_sha256" in frame else []
    return values.reshape(1, 12, len(CHANNELS)), mask.reshape(1, 12, len(CHANNELS)), lineage


def forecast_temperature(
    history: pd.DataFrame,
    *,
    region_id: str,
    initialization_month: str,
    model_id: str,
    horizon_months: int = 1,
    model_path: Path,
    gold_sha256: str,
    training_climatology: dict[int, float],
    interval_radius_deg_c: float | None = None,
) -> dict:
    """Forecast T from a caller-provided monthly history under the v1 release lag."""
    init = _month(initialization_month)
    target = init + int(horizon_months)
    if region_id != REGION_ID:
        raise ValueError(f"Region {region_id!r} has no approved model; only {REGION_ID!r} is available")
    if int(horizon_months) != 1:
        raise ValueError("The frozen model is validated for a +1-month horizon only")
    if model_id != "hist_gradient_boosting":
        raise ValueError("Only the frozen, final-refit hist_gradient_boosting model is packaged for inference")
    if len(gold_sha256) != 64 or any(char not in "0123456789abcdef" for char in gold_sha256.lower()):
        raise ValueError("gold_sha256 must be a SHA-256 hex digest")
    raw, missing, source_hashes = build_forecast_history(history, target_month=str(target))
    artifact = joblib.load(model_path)
    if artifact.get("architecture") != model_id:
        raise ValueError("Model artifact architecture does not match model_id")
    matrix = model_inputs(raw, missing, artifact["imputation_location"], artifact["scaler"])
    prediction = float(artifact["estimator"].predict(matrix)[0])
    filled = raw.copy()
    location = np.asarray(artifact["imputation_location"], dtype=float).reshape(1, 1, -1)
    filled[missing] = np.broadcast_to(location, filled.shape)[missing]
    month_number = target.month
    baseline_clim = training_climatology.get(month_number)
    if baseline_clim is None or not np.isfinite(float(baseline_clim)):
        raise ValueError(f"Missing training-only climatology for calendar month {month_number}")
    low = prediction - interval_radius_deg_c if interval_radius_deg_c is not None else None
    high = prediction + interval_radius_deg_c if interval_radius_deg_c is not None else None
    return {
        "region_id": region_id, "initialization_month": str(init),
        "issue_date_utc": f"{init}-01T00:00:00Z", "target_month": str(target),
        "target_date_utc": f"{target}-01T00:00:00Z", "forecast_horizon_months": 1,
        "feature_window_months": [str(target - 14), str(target - 3)],
        "forecast_origin_max_feature_month": str(target - 3),
        "model_id": model_id, "model_version": "KUNTUR_PHASE2_FINAL_REFIT_V1",
        "prediction_degC": prediction, "units": "degC",
        "prediction_interval": {"low_degC": low, "high_degC": high, "radius_degC": interval_radius_deg_c, "calibration_note": "Validation-residual interval is descriptive; empirical test coverage was below nominal and no split-conformal guarantee is claimed." if interval_radius_deg_c is not None else None},
        "baselines": {"persistence_degC": float(filled[0, -1, 0]), "seasonal_naive_degC": float(filled[0, 2, 0]), "training_monthly_climatology_degC": float(baseline_clim)},
        "evaluation_metrics": {"final_test_metrics_path": "results/phase2/metrics/final_test_metrics_v1.csv", "selection_or_external_data_used": False},
        "coverage": {"history_observed_fraction": float(1.0 - missing.mean()), "coverage_threshold_valid_area_fraction": 0.8},
        "provenance": {"model_artifact": Path(model_path).name, "model_sha256": _sha256(Path(model_path)), "source_gold_sha256": gold_sha256, "history_source_sha256_values": source_hashes},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

