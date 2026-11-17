import warnings

import numpy as np

from scripts.run_phase2_v12_round3_augmentation import _smooth


def test_causal_smoother_handles_all_missing_windows_without_warnings():
    x = np.full((2, 12, 5), np.nan)
    x[0, :, 0] = np.arange(12, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        smoothed = _smooth(x, window=3)
    assert np.isnan(smoothed[1]).all()
    np.testing.assert_allclose(smoothed[0, :4, 0], [0.0, 0.5, 1.0, 2.0])


def test_causal_smoother_never_uses_later_history_values():
    x = np.zeros((1, 12, 5), dtype=float)
    changed = x.copy()
    changed[:, 9:, :] = 100.0
    first, second = _smooth(x, window=3), _smooth(changed, window=3)
    np.testing.assert_allclose(first[:, :9, :], second[:, :9, :])
