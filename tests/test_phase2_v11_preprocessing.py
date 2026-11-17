import numpy as np
import pandas as pd

from kuntur_data.phase2.augmentation import FEATURE_COLUMNS
from kuntur_data.phase2.predictive import fit_preprocessing_v1_1, make_clean_matrix_v1_1


def test_v11_scaler_is_fit_after_training_only_mean_imputation():
    values = np.arange(3 * len(FEATURE_COLUMNS), dtype=float).reshape(3, len(FEATURE_COLUMNS))
    values[0, 0] = np.nan
    train = pd.DataFrame(values, columns=FEATURE_COLUMNS)

    location, _, scaler = fit_preprocessing_v1_1(train)

    channel_mean = np.nanmean(values[:, 0::5])
    imputed_column = np.array([channel_mean, values[1, 0], values[2, 0]])
    assert np.isclose(location[0], channel_mean)
    assert np.isclose(scaler.mean_[0], imputed_column.mean())

    matrix, raw, missing = make_clean_matrix_v1_1(train, location, scaler)
    assert missing[0, 0, 0]
    assert np.isnan(raw[0, 0, 0])
    assert np.isfinite(matrix).all()
    assert matrix[0, 60] == 1.0


def test_v11_preprocessing_rejects_a_channel_missing_entirely_in_training():
    values = np.ones((2, len(FEATURE_COLUMNS)), dtype=float)
    values[:, 0::5] = np.nan
    train = pd.DataFrame(values, columns=FEATURE_COLUMNS)

    try:
        fit_preprocessing_v1_1(train)
    except ValueError as exc:
        assert "entirely missing" in str(exc)
    else:
        raise AssertionError("A fully missing training channel must be rejected")
