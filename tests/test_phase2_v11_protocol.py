import pandas as pd
import pytest

from kuntur_data.phase2.augmentation import FEATURE_COLUMNS
from scripts.build_phase2_v11_strict_public_summary import parse_nonstandard_constant, sanitize
from scripts.run_phase2_v11_rolling_benchmark import FOLDS, _assert_t3_policy, _baseline_predictions


def test_v11_rolling_folds_are_chronological_development_only():
    assert FOLDS[0]["train_end"] < FOLDS[0]["validation_start"]
    assert FOLDS[1]["train_end"] < FOLDS[1]["validation_start"]
    assert FOLDS[2]["train_end"] < FOLDS[2]["validation_start"]
    assert FOLDS[-1]["validation_end"] == "2018-12"
    assert all(fold["validation_end"] < "2019-01" for fold in FOLDS)


def test_v11_forecast_rows_enforce_t_minus_three_and_origin_t_minus_one():
    valid = pd.DataFrame({
        "target_month": ["2003-03"],
        "history_end_month": ["2002-12"],
        "forecast_origin_month": ["2003-02"],
    })
    _assert_t3_policy(valid)

    invalid = valid.assign(history_end_month="2003-01")
    with pytest.raises(ValueError, match="T-3"):
        _assert_t3_policy(invalid)


def test_v11_climatology_uses_training_targets_only():
    train = pd.DataFrame({"target_month": pd.period_range("2010-01", periods=12, freq="M").astype(str), "target_value": range(1, 13)})
    val = pd.DataFrame({"target_month": ["2011-01"], "target_value": [999.0]})
    raw = pd.DataFrame([[0.0] * len(FEATURE_COLUMNS)], columns=FEATURE_COLUMNS).to_numpy().reshape(1, 12, 5)
    missing = raw.astype(bool)
    location = pd.Series([0.0] * 5).to_numpy()
    first = _baseline_predictions(train, val, raw, missing, location)["training_monthly_climatology"]
    val["target_value"] = -999.0
    second = _baseline_predictions(train, val, raw, missing, location)["training_monthly_climatology"]
    assert first.tolist() == second.tolist() == [1.0]


def test_v11_strict_json_converts_nonfinite_to_null_with_field_path():
    parsed = {"metric": parse_nonstandard_constant("NaN"), "ok": 1.0}
    output, validity = sanitize(parsed)
    assert output == {"metric": None, "ok": 1.0}
    assert validity == [{"json_pointer": "/metric", "original_token": "NaN"}]
