import numpy as np
import statsmodels.api as sm

from scripts.run_phase2_v12_round4_inference_calibration import (
    BASELINE,
    N,
    X,
    _block_ci,
    _cp_interval,
    _make_errors,
    _ols_hac,
    _seasonal_theil_sen,
)


def test_dgp_reproducible_from_seed_for_all_processes():
    processes = [
        "iid_gaussian", "ar1_m030", "ar1_p030", "ar1_060", "ar1_0844",
        "arma11", "seasonal_ar1_12", "heteroskedastic_iid", "student_t3",
    ]
    for process in processes:
        first = _make_errors(process, np.random.default_rng(817))
        second = _make_errors(process, np.random.default_rng(817))
        assert first.shape == (N,)
        np.testing.assert_array_equal(first, second)


def test_hac_estimate_and_covariance_match_statsmodels_reference():
    y = BASELINE + 0.35 * (np.arange(N, dtype=float) / 120.0) + np.random.default_rng(39).normal(0, 0.25, N)
    slope, low, high, _p, _se = _ols_hac(y, 12)
    fit = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 12, "use_correction": True})
    reference_low, reference_high = fit.conf_int(alpha=0.05)[1]
    assert abs(slope - fit.params[1]) < 1e-11
    np.testing.assert_allclose([low, high], [reference_low, reference_high], rtol=1e-8, atol=1e-10)


def test_seasonal_sen_recovers_linear_slope_without_noise():
    y = BASELINE + 0.5 * (np.arange(N, dtype=float) / 120.0)
    assert abs(_seasonal_theil_sen(y) - 0.5) < 1e-12


def test_bootstrap_is_seeded_and_mc_interval_is_valid():
    y = BASELINE + np.random.default_rng(24).normal(0, 0.25, N)
    first = _block_ci(y, np.random.default_rng(17), seasonal=True, draws=99)
    second = _block_ci(y, np.random.default_rng(17), seasonal=True, draws=99)
    assert first == second
    assert first[0] < first[1]
    lo, hi = _cp_interval(950, 1000)
    assert lo < 0.95 < hi
