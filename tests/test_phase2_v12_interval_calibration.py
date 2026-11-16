import numpy as np
import pandas as pd
import pytest

from scripts.run_phase2_v12_round5_interval_calibration import _conformal_abs_quantile, _intervals, _winkler


def test_conformal_quantile_uses_finite_sample_rank():
    values = np.arange(1.0, 10.0)
    assert _conformal_abs_quantile(values, 0.8) == 8.0


def test_interval_methods_use_only_the_passed_calibration_rows():
    calibration = pd.DataFrame({
        "target_month": pd.period_range("2010-01", periods=36, freq="M").astype(str),
        "fold_id": ["past"] * 36,
        "y_true": np.arange(36, dtype=float) + 0.1,
        "y_pred": np.arange(36, dtype=float),
    })
    evaluation = pd.DataFrame({
        "target_month": pd.period_range("2013-01", periods=2, freq="M").astype(str),
        "fold_id": ["future"] * 2,
        "y_true": [36.2, 37.2], "y_pred": [36.0, 37.0],
    })
    rows = _intervals(calibration, evaluation)
    assert len(rows) == 12
    assert {row["calibration_end_month"] for row in rows} == {"2012-12"}
    assert {row["calibration_count"] for row in rows} == {36}
    assert {row["target_month"] for row in rows}.isdisjoint(set(calibration.target_month))


def test_winkler_penalizes_missed_observation_and_invalid_quantiles_reject():
    scores = _winkler(np.array([0.0, 4.0]), np.array([-1.0, -1.0]), np.array([1.0, 1.0]), alpha=0.1)
    assert scores[1] > scores[0]
    with pytest.raises(ValueError):
        _conformal_abs_quantile(np.array([np.nan]), 0.9)
