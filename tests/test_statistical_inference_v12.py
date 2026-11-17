import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

import kuntur_data.phase2.statistical_v12 as statistical_v12
from kuntur_data.phase2.statistical_v12 import _seasonal_mk_block_p
from kuntur_data.phase2.statistical_v12 import (
    analyze_variable_v12,
    benjamini_yekutieli_v12,
    calendar_lag_acf,
)


def _gold(values, *, start="2000-01", coverage=None):
    months = pd.period_range(start=start, periods=len(values), freq="M").astype(str)
    if coverage is None:
        coverage = np.ones(len(values), dtype=float)
    return pd.DataFrame({
        "month": months,
        "variable_id": "x",
        "value": np.asarray(values, dtype=float),
        "valid_area_fraction_bbox": np.asarray(coverage, dtype=float),
        "units": "degree_Celsius",
    })


def test_constant_series_has_no_inferential_claim_or_residual_acf():
    result = analyze_variable_v12(_gold(np.full(60, 7.0)), "x", method="ols_hac")

    assert result["validity_status"] == "DEGENERATE_SERIES"
    assert result["slope_per_decade"] == 0.0
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None
    assert result["residual_diagnostics"]["acf_calendar_months"]["1"] is None
    assert result["residual_diagnostics"]["acf_calendar_months"]["12"] is None


def test_infinite_input_is_flagged_before_climatology_or_regression():
    values = np.linspace(1.0, 2.0, 60)
    values[17] = np.inf
    result = analyze_variable_v12(_gold(values), "x", method="ols_hac")

    assert result["validity_status"] == "NONFINITE_INPUT"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_ols_hac_with_missing_calendar_months_withholds_row_lag_inference():
    values = 12.0 + 0.01 * np.arange(120, dtype=float)
    values[40:49] = np.nan
    result = analyze_variable_v12(_gold(values), "x", method="ols_hac")

    assert result["validity_status"] == "IRREGULAR_TIME_SUPPORT"
    assert result["n_calendar_months"] == 120
    assert result["n_eligible_observations"] == 111
    assert result["max_missing_run_months"] == 9
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_calendar_lag_uses_timestamp_distance_across_missing_months():
    index = pd.period_range("2000-01", periods=24, freq="M")
    values = np.arange(24, dtype=float)
    values[10] = np.nan

    acf = calendar_lag_acf(pd.Series(values, index=index), max_lag_months=12)

    # 12-month pairs are determined by calendar positions, not compressed rows.
    assert acf["12"]["pair_count"] == 11
    assert acf["1"]["pair_count"] == 21


def test_state_space_ar1_retains_full_monthly_index_with_missing_observations():
    time = np.arange(120, dtype=float)
    values = 10.0 + 0.02 * time + 0.4 * np.sin(2 * np.pi * time / 12)
    values += np.random.default_rng(9).normal(0.0, 0.2, len(time))
    values[40:49] = np.nan
    result = analyze_variable_v12(_gold(values), "x", method="sarimax_ar1")

    assert result["method_id"] == "SARIMAX_AR1"
    assert result["n_calendar_months"] == 120
    assert result["n_eligible_observations"] == 111
    assert result["calendar_start"] == "2000-01"
    assert result["calendar_end"] == "2009-12"
    assert result["validity_status"] == "NOT_VALIDATED"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_duplicate_months_are_rejected_with_a_clear_error():
    gold = _gold(np.arange(60, dtype=float))
    gold.loc[1, "month"] = gold.loc[0, "month"]

    with pytest.raises(ValueError, match="duplicate calendar month"):
        analyze_variable_v12(gold, "x", method="ols_hac")


def test_out_of_order_rows_are_sorted_and_reported():
    gold = _gold(np.arange(60, dtype=float))
    gold = gold.iloc[::-1].reset_index(drop=True)
    result = analyze_variable_v12(gold, "x", method="ols_hac")

    assert result["calendar_start"] == "2000-01"
    assert result["calendar_end"] == "2004-12"
    assert result["input_was_sorted"] is False
    assert result["slope_per_decade"] == pytest.approx(120.0)


def test_multiple_regions_require_explicit_filter_and_return_region_identity():
    first = _gold(np.arange(60, dtype=float)).assign(region_id="ecuador_bbox")
    second = _gold(100.0 + 2.0 * np.arange(60, dtype=float)).assign(region_id="comparison")
    gold = pd.concat([first, second], ignore_index=True)

    with pytest.raises(ValueError, match="specify region_id explicitly"):
        analyze_variable_v12(gold, "x", method="ols_hac")

    result = analyze_variable_v12(gold, "x", region_id="comparison", method="ols_hac")
    assert result["region_id"] == "comparison"
    assert result["slope_per_decade"] == pytest.approx(240.0)


def test_complete_regular_hac_is_retained_only_as_unvalidated_reference():
    time = np.arange(120, dtype=float)
    values = 10.0 + 0.2 * time / 120.0 + np.sin(2 * np.pi * time / 12) * 0.2
    result = analyze_variable_v12(_gold(values), "x", method="ols_hac")

    assert result["validity_status"] == "UNDERCOVERAGE_DETECTED"
    assert result["slope_per_decade"] == pytest.approx(0.2, abs=0.01)
    assert result["standard_error"] is not None
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_inference_is_released_only_for_an_exact_frozen_registry_match():
    time = np.arange(60, dtype=float)
    values = 10.0 + 0.2 * time / 120.0 + np.random.default_rng(33).normal(0.0, 0.1, 60)
    registry = {
        "status": "FROZEN",
        "regimes": {
            "iid_gaussian": {
                "validity_status": "VALIDATED_FOR_DECLARED_REGIME",
                "method_id": "OLS_HAC_FROZEN_REFERENCE",
                "calendar_months_min": 60,
                "calendar_months_max": 60,
                "calendar_coverage_min": 1.0,
                "calendar_coverage_max": 1.0,
            }
        },
    }
    unmatched = analyze_variable_v12(_gold(values), "x", method="ols_hac")
    matched = analyze_variable_v12(
        _gold(values), "x", method="ols_hac", declared_regime="iid_gaussian",
        validation_registry=registry,
    )

    assert unmatched["p_value"] is None
    assert matched["validity_status"] == "VALIDATED_FOR_DECLARED_REGIME"
    assert matched["p_value"] is not None
    assert matched["confidence_interval_95"] is not None
    assert matched["provenance"]["declared_regime"] == "iid_gaussian"


@pytest.mark.parametrize(
    ("method", "expected_id"),
    [
        ("gls_ar1", "FEGLS_AR1_ITERATED"),
        ("sarimax_ar1", "SARIMAX_AR1"),
        ("arma11", "SARIMAX_ARMA11"),
        ("seasonal_ar12", "SARIMAX_SEASONAL_AR12"),
    ],
)
def test_parametric_dependence_candidates_return_common_gated_schema(method, expected_id):
    time = np.arange(96, dtype=float)
    values = 11.0 + 0.15 * time / 120.0 + 0.3 * np.sin(2 * np.pi * time / 12)
    values += np.random.default_rng(8).normal(0.0, 0.2, len(time))
    result = analyze_variable_v12(_gold(values), "x", method=method)

    assert result["method_id"] == expected_id
    assert result["slope_per_decade"] is not None
    assert result["convergence_status"] in {"CONVERGED", "PARTIAL_OR_WITHHELD"}
    assert "acf_calendar_lags" in result["residual_diagnostics"]
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None
    assert result["validity_status"] == "NOT_VALIDATED"


@pytest.mark.parametrize("method", ["moving_block_bootstrap", "seasonal_block_bootstrap"])
def test_residual_bootstrap_candidates_are_seeded_and_do_not_publish_unvalidated_inference(method):
    time = np.arange(120, dtype=float)
    values = 10.0 + 0.25 * time / 12.0 + 0.3 * np.sin(2 * np.pi * time / 12)
    gold = _gold(values)
    first = analyze_variable_v12(gold, "x", method=method, n_bootstrap=99, random_seed=47, block_length_months=24)
    second = analyze_variable_v12(gold, "x", method=method, n_bootstrap=99, random_seed=47, block_length_months=24)

    assert first["method_id"] == second["method_id"]
    assert first["standard_error"] == second["standard_error"]
    assert first["validity_status"] == "NOT_VALIDATED"
    assert first["p_value"] is None
    assert first["confidence_interval_95"] is None


def test_glsar_refuses_irregular_months_instead_of_compressing_time():
    values = 3.0 + np.arange(120, dtype=float) * 0.01
    values[40:49] = np.nan
    result = analyze_variable_v12(_gold(values), "x", method="gls_ar1")

    assert result["validity_status"] == "IRREGULAR_TIME_SUPPORT"
    assert result["n_calendar_months"] == 120
    assert result["n_eligible_observations"] == 111
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_glsar_iteration_cap_is_reported_as_numerical_failure(monkeypatch):
    class FakeGLSAR:
        def __init__(self, y, x, rho):
            self.rho = rho

        def iterative_fit(self, maxiter):
            params = np.zeros(13)
            params[1] = 0.2
            result = SimpleNamespace(
                iter=maxiter,
                model=SimpleNamespace(rho=np.array([0.4])),
                params=params,
                bse=np.ones(13),
                pvalues=np.ones(13),
                resid=np.random.default_rng(63).normal(size=59),
                conf_int=lambda alpha: np.column_stack((params - 0.1, params + 0.1)),
            )
            return result

    monkeypatch.setattr(statistical_v12.sm, "GLSAR", FakeGLSAR)
    result = analyze_variable_v12(_gold(np.linspace(1.0, 2.0, 60)), "x", method="gls_ar1")

    assert result["validity_status"] == "NUMERICAL_FAILURE"
    assert result["convergence_status"] == "PARTIAL_OR_WITHHELD"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_calendar_index_keeps_qa_missing_months_at_both_series_boundaries():
    values = 4.0 + 0.02 * np.arange(72, dtype=float)
    values[:4] = np.nan
    values[-5:] = np.nan
    result = analyze_variable_v12(_gold(values), "x", method="sarimax_ar1")

    assert result["n_calendar_months"] == 72
    assert result["n_eligible_observations"] == 63
    assert result["calendar_start"] == "2000-01"
    assert result["calendar_end"] == "2005-12"
    assert result["max_missing_run_months"] == 5


def test_non_numeric_values_are_not_silently_reclassified_as_missing():
    gold = _gold(np.linspace(0.0, 1.0, 60)).astype({"value": object})
    gold.loc[20, "value"] = "bad-value"
    result = analyze_variable_v12(gold, "x", method="ols_hac")

    assert result["validity_status"] == "NONFINITE_INPUT"
    assert result["p_value"] is None


def test_seasonal_mk_requires_multiple_supported_calendar_month_strata():
    values = np.linspace(1.0, 2.0, 48)
    coverage = np.zeros(48, dtype=float)
    coverage[np.arange(48) % 12 == 0] = 1.0
    result = analyze_variable_v12(
        _gold(values, coverage=coverage),
        "x",
        method="seasonal_mk_block",
        qa_policy={"min_valid_area_fraction_bbox": 0.8},
        n_bootstrap=49,
    )

    assert result["validity_status"] == "INSUFFICIENT_DATA"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_seasonal_bootstrap_rejects_partial_calendar_years_as_insufficient():
    values = 2.0 + 0.01 * np.arange(61, dtype=float)
    result = analyze_variable_v12(
        _gold(values), "x", method="seasonal_block_bootstrap", n_bootstrap=49
    )

    assert result["validity_status"] == "INSUFFICIENT_DATA"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_benjamini_yekutieli_handles_arbitrary_dependence_and_preserves_nan():
    adjusted = benjamini_yekutieli_v12([0.01, 0.04, np.nan, 0.03])

    assert adjusted[0] == pytest.approx(0.055)
    assert adjusted[1] == pytest.approx(0.0733333333333)
    assert np.isnan(adjusted[2])
    assert adjusted[3] == pytest.approx(0.0733333333333)


def test_unvalidated_dependence_aware_method_keeps_candidate_pvalue_suppressed():
    time = np.arange(48, dtype=float)
    values = 5.0 + 0.1 * time / 12.0 + np.sin(2 * np.pi * time / 12) * 0.2
    result = analyze_variable_v12(
        _gold(values), "x", method="seasonal_mk_block", n_bootstrap=99, random_seed=21
    )

    assert result["method_id"] == "SEASONAL_MK_YEAR_BLOCK_BOOTSTRAP"
    assert result["validity_status"] == "NOT_VALIDATED"
    assert result["slope_per_decade"] is not None
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_seasonal_mk_null_bootstrap_removes_injected_linear_trend_before_resampling(monkeypatch):
    periods = pd.period_range("2000-01", periods=240, freq="M")
    time = np.arange(len(periods), dtype=float)
    month_effect = 0.7 * np.sin(2 * np.pi * (periods.month.to_numpy() - 1) / 12)
    values = 8.0 + month_effect + 0.5 * time / 120.0
    captured = {}
    original = statistical_v12._seasonal_mk_s_many

    def capture_bootstrap_series(matrix):
        captured["matrix"] = matrix.copy()
        return original(matrix)

    monkeypatch.setattr(statistical_v12, "_seasonal_mk_s_many", capture_bootstrap_series)
    _s, p_value, converged = _seasonal_mk_block_p(
        values, periods, n_bootstrap=99, seed=25, block_years=2
    )
    year_means = np.mean(captured["matrix"], axis=2)

    assert converged
    assert p_value == pytest.approx(0.01)
    # The null resamples retain month means and dependence, but not the injected trend.
    assert np.max(np.std(year_means, axis=1)) < 1e-10
