from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2
import pandas as pd
import statsmodels.api as sm

from scripts.statistical_correction.interval_comparison_v1_3_2 import (
    critical_t_interval,
    fit_candidate,
    profile_likelihood_interval,
    residual_degrees_of_freedom,
    seasonal_ar12_profile_loglikelihood,
)
from scripts.statistical_correction.summarize_interval_comparison_v1_3_2 import summarize_preflight


def test_residual_df_accounts_for_regressors_and_dynamic_covariance_parameters():
    assert residual_degrees_of_freedom(276, 13, 1) == 262


def test_finite_sample_t_interval_and_test_are_matched():
    result = critical_t_interval(0.5, 0.1, df=262, alpha=0.05)
    assert result["ci_low"] < 0.5 < result["ci_high"]
    assert result["p_value"] > 0.0
    assert result["critical_value"] > 1.96
    assert math.isclose(result["ci_high"] - result["ci_low"], 2 * result["critical_value"] * 0.1)


def test_profile_likelihood_inverts_same_likelihood_ratio_test():
    beta_hat = 0.35
    se = 0.08
    llf_max = 3.0
    calls = []

    def profile(value: float):
        calls.append(value)
        return llf_max - 0.5 * ((value - beta_hat) / se) ** 2, True, None

    result = profile_likelihood_interval(beta_hat, llf_max, se, profile, alpha=0.05)
    expected_half_width = math.sqrt(3.841458820694124) * se
    assert math.isclose(result["ci_low"], beta_hat - expected_half_width, rel_tol=0, abs_tol=1e-7)
    assert math.isclose(result["ci_high"], beta_hat + expected_half_width, rel_tol=0, abs_tol=1e-7)
    null_lr = (beta_hat / se) ** 2
    assert math.isclose(result["profile_test_p_value"], float(chi2.sf(null_lr, 1)), rel_tol=0, abs_tol=1e-12)
    assert result["profile_evaluations"] == len(set(calls))


def test_profile_likelihood_clips_only_submilliloglikelihood_optimizer_tolerance():
    beta_hat = 0.2
    se = 0.1
    llf_max = 4.0

    def profile(value: float):
        return llf_max + 0.0005 - 0.5 * ((value - beta_hat) / se) ** 2, True, None

    result = profile_likelihood_interval(beta_hat, llf_max, se, profile, alpha=0.05)
    assert math.isclose(result["profile_log_likelihood_max"], llf_max + 0.0005, abs_tol=1e-12)


def test_profile_likelihood_rejects_material_maximum_mismatch():
    def profile(_value: float):
        return 4.01, True, None

    try:
        profile_likelihood_interval(0.0, 4.0, 0.1, profile, alpha=0.05)
    except RuntimeError as exc:
        assert "MISMATCH" in str(exc)
    else:
        raise AssertionError("material likelihood-maximum mismatch must fail visibly")


def test_preflight_does_not_authorize_development_when_any_candidate_has_failed_inference():
    methods = ("wald_normal", "wald_student_t", "profile_likelihood")
    slopes = (0.0, 0.25, 0.5)
    config = {
        "protocol_id": "test",
        "design": {"preflight_replicates_per_slope": 30, "development_replicates_per_slope": 1000, "slopes_per_decade": slopes},
        "resource_budget": {
            "preflight_projection_multiplier": 2.0,
            "development_hard_max_cpu_seconds": 7200,
            "development_hard_max_wall_seconds": 14400,
            "development_candidate_replicates": 9000,
            "preflight_minimum_candidate_inference_availability": 1.0,
        },
    }
    rows = []
    for slope in slopes:
        for replicate in range(30):
            for method in methods:
                rows.append({
                    "method": method,
                    "slope_per_decade": slope,
                    "cpu_seconds": 0.1,
                    "wall_seconds": 0.1,
                    "converged": True,
                    "inferential_output": True,
                    "profile_likelihood_evaluations": 8 if method == "profile_likelihood" else 0,
                })
    manifest = {"status": "COMPLETE", "resource_usage": {}}
    checkpoints = [{"status": "COMPLETE"} for _ in slopes]
    passing = summarize_preflight(config, "freeze", rows, manifest, checkpoints)
    assert passing["decision"] == "PROCEED"
    rows[17]["converged"] = False
    rows[17]["inferential_output"] = False
    failing = summarize_preflight(config, "freeze", rows, manifest, checkpoints)
    assert failing["decision"] == "NOT_VALIDATED_PREFLIGHT_NUMERICAL_FAILURE"


def _sample_series(seed: int = 481):
    n = 276
    periods = pd.period_range("2000-01", periods=n, freq="M")
    month = periods.month.to_numpy() - 1
    x = np.column_stack((np.ones(n), np.arange(n, dtype=float) / 120.0, np.eye(12)[month, 1:]))
    seasonal = 0.55 * np.sin(2 * np.pi * month / 12) + 0.18 * np.cos(4 * np.pi * month / 12)
    y = 12.0 + seasonal + np.random.default_rng(seed).normal(0.0, 0.25, n)
    return y, x


def test_analytic_profile_likelihood_matches_the_statsmodels_stationary_sarimax_likelihood():
    y, x = _sample_series()
    model = sm.tsa.SARIMAX(y, exog=x, order=(0, 0, 0), seasonal_order=(1, 0, 0, 12), trend="n", enforce_stationarity=True, enforce_invertibility=True)
    fit = model.fit(disp=False, maxiter=200)
    llf, converged, note = seasonal_ar12_profile_loglikelihood(y, x, float(fit.params[1]))
    assert converged, note
    assert math.isclose(llf, float(fit.llf), rel_tol=0, abs_tol=2e-4)


def test_interval_candidates_keep_the_same_point_model_and_slope_estimate():
    y, x = _sample_series(982)
    fits = [fit_candidate(y, x, method)[0] for method in ("wald_normal", "wald_student_t", "profile_likelihood")]
    assert all(fit.converged for fit in fits)
    estimates = [fit.slope for fit in fits]
    assert max(estimates) - min(estimates) < 1e-12
    assert all(fit.ci_low is not None and fit.ci_high is not None and fit.p_value is not None for fit in fits)
