import numpy as np
import pandas as pd
import pytest

from scripts.score_phase2_v12_sealed_2025 import (
    PREDICTOR_VARIABLES,
    _paired_circular_block_ci,
    make_test_frame,
)


def _gold_rows(months):
    rows = []
    for month_index, month in enumerate(months):
        for variable_index, variable in enumerate(PREDICTOR_VARIABLES):
            rows.append({
                "month": str(month), "variable_id": variable,
                "value": float(month_index * 10 + variable_index),
                "valid_area_fraction_bbox": 0.95,
            })
    return pd.DataFrame(rows)


def test_2025_frame_has_exact_monthly_t14_to_t3_histories():
    historical = _gold_rows(pd.period_range("2023-11", "2024-12", freq="M").astype(str))
    sealed = _gold_rows(pd.period_range("2025-01", "2025-12", freq="M").astype(str))
    frame = make_test_frame(historical, sealed)
    assert len(frame) == 12
    assert frame.iloc[0].history_start_month == "2023-11"
    assert frame.iloc[0].history_end_month == "2024-10"
    assert frame.iloc[-1].history_start_month == "2024-10"
    assert frame.iloc[-1].history_end_month == "2025-09"
    assert frame.iloc[0]["merra2_t2m_c_lag_12"] == historical.iloc[0].value
    assert frame.iloc[-1].target_month == "2025-12"


def test_2025_frame_fails_closed_when_month_or_coverage_is_missing():
    historical = _gold_rows(pd.period_range("2023-11", "2024-12", freq="M").astype(str))
    sealed = _gold_rows(pd.period_range("2025-01", "2025-12", freq="M").astype(str))
    with pytest.raises(ValueError, match="exactly one row"):
        make_test_frame(historical, sealed[sealed.month != "2025-06"])
    bad = sealed.copy()
    bad.loc[(bad.month == "2025-02") & (bad.variable_id == "merra2_t2m_c"), "valid_area_fraction_bbox"] = 0.79
    with pytest.raises(ValueError, match="target failed"):
        make_test_frame(historical, bad)


def test_single_12_month_block_bootstrap_is_marked_degenerate():
    result = _paired_circular_block_ci(np.arange(12, dtype=float))
    assert result["degenerate_single_block_interval"] is True
    assert result["delta_mae_ci95"][0] == result["delta_mae_ci95"][1]
