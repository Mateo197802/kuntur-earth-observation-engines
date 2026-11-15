import numpy as np
import pandas as pd
import pytest
from joblib import dump
from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler

from kuntur_data.phase2.api import build_forecast_history, forecast_temperature, query_analytical
from kuntur_data.phase2.augmentation import CHANNELS
from kuntur_data.phase2.ingestion import GlobalAnalyticalCache, PartitionedGoldStore


def _history(target="2025-01"):
    periods = pd.period_range(pd.Period(target, freq="M") - 14, pd.Period(target, freq="M") - 3, freq="M")
    rows = []
    for month in periods:
        for j, channel in enumerate(CHANNELS):
            value = [22.0, 0.015, 0.5, 0.4, 93000.0][j]
            rows.append({"month": str(month), "variable_id": channel, "value": value,
                         "valid_area_fraction_bbox": 1.0, "source_sha256": f"sha-{month}-{channel}"})
    return pd.DataFrame(rows)


def _artifact(path):
    scaler = StandardScaler().fit(np.zeros((2, 60)))
    estimator = DummyRegressor(strategy="constant", constant=23.5).fit(np.zeros((2, 120)), [0.0, 1.0])
    dump({"architecture": "hist_gradient_boosting", "scaler": scaler,
          "imputation_location": np.array([22.0, 0.015, 0.5, 0.4, 93000.0]),
          "estimator": estimator}, path)


def test_forecast_history_enforces_release_lag_and_complete_month_grid():
    history = _history()
    values, missing, lineage = build_forecast_history(history, target_month="2025-01")
    assert values.shape == (1, 12, 5)
    assert not missing.any()
    assert len(lineage) == 60
    with pytest.raises(ValueError, match="T-14..T-3"):
        build_forecast_history(history.iloc[5:], target_month="2025-01")


def test_forecast_interface_returns_provenance_and_baselines(tmp_path):
    model = tmp_path / "model.joblib"
    _artifact(model)
    result = forecast_temperature(
        _history(), region_id="ecuador_pilot_bbox", initialization_month="2024-12",
        model_id="hist_gradient_boosting", horizon_months=1, model_path=model,
        gold_sha256="a" * 64, training_climatology={1: 22.5}, interval_radius_deg_c=0.5,
    )
    assert result["target_month"] == "2025-01"
    assert result["forecast_origin_max_feature_month"] == "2024-10"
    assert result["prediction_degC"] == 23.5
    assert result["baselines"]["persistence_degC"] == 22.0
    assert len(result["provenance"]["model_sha256"]) == 64
    with pytest.raises(ValueError, match=r"\+1-month"):
        forecast_temperature(
            _history(), region_id="ecuador_pilot_bbox", initialization_month="2024-12",
            model_id="hist_gradient_boosting", horizon_months=2, model_path=model,
            gold_sha256="a" * 64, training_climatology={1: 22.5},
        )


def test_analytical_interface_returns_coverage_trend_and_cache_metadata(tmp_path):
    rows = []
    for i, month in enumerate(pd.period_range("2000-01", periods=60, freq="M")):
        rows.append({
            "dataset_id": "M2", "variable_id": "merra2_t2m_c", "month": str(month),
            "region_id": "ecuador_pilot_bbox", "region_status": "PRELIMINARY_EXTRACTION_AREA",
            "value": 20.0 + 0.02 * i + np.sin(i * 2 * np.pi / 12), "units": "degC",
            "valid_area_fraction_bbox": 0.95, "qa_pass_fraction_of_candidates": 0.99,
            "source_sha256": "b" * 64, "contract_sha256": "c" * 64,
        })
    gold = pd.DataFrame(rows)
    gold_path = tmp_path / "gold.parquet"
    gold.to_parquet(gold_path, index=False)
    cache = GlobalAnalyticalCache(tmp_path / "cache")
    store = PartitionedGoldStore(gold_path, tmp_path / "partitions", cache=cache)
    store.materialize()
    result = query_analytical(
        store, region_id="ecuador_pilot_bbox", variable_id="merra2_t2m_c",
        start_month="2000-01", end_month="2004-12", trend_method="ols_hac", qa_threshold=0.8,
    )
    assert result["coverage"]["expected_months"] == 60
    assert result["coverage"]["eligible_months"] == 60
    assert result["trend"]["estimate_units"] == "degC per decade"
    assert result["cache"]["hit"] is False
    assert len(result["series"]) == 60
