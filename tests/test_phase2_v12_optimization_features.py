import numpy as np
import pandas as pd

from kuntur_data.phase2.augmentation import FEATURE_COLUMNS
from kuntur_data.phase2.optimization_features import FEATURE_GROUPS, build_feature_pair


def _frame(start: str, count: int, offset: int = 0) -> pd.DataFrame:
    months = pd.period_range(start, periods=count, freq="M")
    rows = []
    for i, month in enumerate(months):
        row = {
            "sample_id": f"s{offset+i:04d}",
            "target_month": str(month),
            "history_end_month": str(month - 3),
            "forecast_origin_month": str(month - 1),
            "target_value": 20.0 + 4.0 * np.sin(2 * np.pi * month.month / 12) + 0.01 * i,
        }
        for lag_i, lag in enumerate(range(12, 0, -1)):
            for channel_i, channel in enumerate(("merra2_t2m_c", "merra2_qv2m", "merra2_u2m", "merra2_v2m", "merra2_ps")):
                row[f"{channel}_lag_{lag:02d}"] = float(20 + channel_i + lag_i / 10 + np.sin(2 * np.pi * (month.month - lag) / 12))
        rows.append(row)
    return pd.DataFrame(rows)


def test_all_feature_groups_are_finite_and_keep_t3_alignment():
    train = _frame("2003-01", 84)
    validation = _frame("2010-01", 24, offset=84)
    for group in FEATURE_GROUPS:
        pair = build_feature_pair(train, validation, group)
        assert np.isfinite(pair.x_train).all()
        assert np.isfinite(pair.x_validation).all()
        assert len(pair.y_train) == len(train)
        assert len(pair.y_validation) == len(validation)
        assert np.all(pair.offset_train == 0) if group != "C_training_anomalies" else np.isfinite(pair.offset_train).all()
        target = pd.PeriodIndex(validation.target_month, freq="M")
        history_end = pd.PeriodIndex(validation.history_end_month, freq="M")
        assert np.all(history_end == target - 3)


def test_validation_targets_do_not_change_fold_fitted_feature_transforms():
    train = _frame("2003-01", 84)
    validation = _frame("2010-01", 24, offset=84)
    changed = validation.copy()
    changed["target_value"] += 1000
    for group in ("B_temperature_seasonal", "C_training_anomalies", "F_climatology_residual", "G_stable_channels"):
        original_pair = build_feature_pair(train, validation, group)
        changed_pair = build_feature_pair(train, changed, group)
        np.testing.assert_allclose(original_pair.x_validation, changed_pair.x_validation)
        np.testing.assert_allclose(original_pair.x_train, changed_pair.x_train)


def test_missingness_indicators_are_an_explicit_ablation():
    train = _frame("2003-01", 84)
    validation = _frame("2010-01", 24, offset=84)
    missing_column = FEATURE_COLUMNS[0]
    train.loc[0, missing_column] = np.nan
    with_mask = build_feature_pair(train, validation, "C_training_anomalies", include_missing_indicators=True)
    without_mask = build_feature_pair(train, validation, "C_training_anomalies", include_missing_indicators=False)
    assert with_mask.x_train.shape[1] - without_mask.x_train.shape[1] == 12
    assert np.isfinite(with_mask.x_train).all()
    assert np.isfinite(without_mask.x_train).all()


def test_weighted_feature_scaling_is_aligned_and_finite():
    train = _frame("2003-01", 84)
    validation = _frame("2010-01", 24, offset=84)
    weights = np.ones(len(train))
    weights[:12] = 0.5
    pair = build_feature_pair(train, validation, "C_training_anomalies", sample_weight=weights)
    assert np.isfinite(pair.x_train).all()
    assert np.isfinite(pair.x_validation).all()
