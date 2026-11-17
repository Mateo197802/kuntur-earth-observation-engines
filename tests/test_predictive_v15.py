from __future__ import annotations

import pandas as pd
import pytest

from kuntur_data.phase2.predictive_v15 import (
    ALLOWED_GATE_FEATURES,
    chronological_inner_folds_v15,
    chronological_outer_folds_v15,
    complete_moving_blocks,
    history_months_for_target,
    history_published_by_issue,
    labels_published_by_issue,
    metric_summary_v15,
    retrospective_regime_summary_v15,
    gate_proxy_values,
    select_fixed_blend_strength,
    smooth_proxy_weights,
    validate_gate_feature_names,
)


def test_issue_history_is_exact_t_minus_14_through_t_minus_3():
    months = history_months_for_target("2018-02", width=12, latest_lag=3)

    assert months == [
        "2016-12", "2017-01", "2017-02", "2017-03", "2017-04", "2017-05",
        "2017-06", "2017-07", "2017-08", "2017-09", "2017-10", "2017-11",
    ]


def test_issue_history_rejects_a_granule_produced_after_the_issue_time():
    months = history_months_for_target("2018-02", width=12, latest_lag=3)
    produced = {month: "2017-11-15T00:00:00Z" for month in months}
    produced["2017-11"] = "2018-01-02T00:00:00Z"

    audit = history_published_by_issue(
        target_month="2018-02",
        issue_time_utc="2018-01-01T00:00:00Z",
        production_dates_utc=produced,
        width=12,
        latest_lag=3,
    )

    assert audit.eligible is False
    assert audit.unavailable_months == ("2017-11",)


def test_gate_rejects_target_error_and_future_fields():
    validate_gate_feature_names(["t2m_anomaly_lag_03", "t2m_anomaly_sd_06"])

    with pytest.raises(ValueError, match="not issue-time eligible"):
        validate_gate_feature_names(["t2m_anomaly_lag_03", "absolute_forecast_error"])

    assert "absolute_forecast_error" not in ALLOWED_GATE_FEATURES


def test_smooth_gate_uses_training_proxy_distribution_and_is_bounded():
    training_proxy = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4])
    validation_proxy = pd.Series([0.0, 0.2, 0.4])

    weights = smooth_proxy_weights(
        validation_proxy,
        training_proxy,
        strength=0.8,
        low_quantile=0.25,
        high_quantile=0.75,
    )

    assert weights.iloc[0] == 0.0
    assert weights.iloc[1] == pytest.approx(0.4)
    assert weights.iloc[2] == pytest.approx(0.8)
    assert ((weights >= 0) & (weights <= 0.8)).all()


def test_gate_proxy_magnitude_uses_only_the_named_allowed_issue_time_feature():
    values = gate_proxy_values(
        [[-2.0, 5.0], [1.0, 7.0]],
        ["t2m_anomaly_lag_03", "absolute_forecast_error"],
        "t2m_anomaly_lag_03",
    )

    assert values.tolist() == [2.0, 1.0]
    with pytest.raises(ValueError, match="not issue-time eligible"):
        gate_proxy_values([[2.0]], ["absolute_forecast_error"], "absolute_forecast_error")


def test_fixed_blend_selection_keeps_climatology_when_residual_hurts_inner_predictions():
    alpha, score = select_fixed_blend_strength(
        observations=[0.0, 0.0, 0.0, 0.0],
        climatology=[0.0, 0.0, 0.0, 0.0],
        residual_prediction=[2.0, -2.0, 2.0, -2.0],
        candidates=[0.0, 0.25, 0.5, 1.0],
    )

    assert alpha == 0.0
    assert score == 0.0


def test_paired_metrics_compare_same_months_and_keep_positive_bias_definition():
    metrics = metric_summary_v15(
        observations=[2.0, 0.0, 1.0],
        prediction=[1.0, 1.0, 2.0],
        baseline=[1.0, 1.0, 1.0],
    )

    assert metrics["sample_count"] == 3
    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["bias"] == pytest.approx(1.0 / 3.0)
    assert metrics["mae_skill_vs_baseline"] == pytest.approx(-0.5)


def test_nested_year_folds_are_chronological_and_do_not_include_consumed_tests():
    months = pd.period_range("2015-10", "2021-12", freq="M").astype(str)
    samples = pd.DataFrame(
        {
            "target_month": months,
            "issue_date_utc": [f"{pd.Period(m, freq='M') - 1}-01T00:00:00Z" for m in months],
        }
    )
    eligible = samples[samples.target_month <= "2021-08"].copy()

    outer = chronological_outer_folds_v15(eligible, validation_years=(2019, 2020, 2021))

    assert [fold.name for fold in outer] == ["outer_2019", "outer_2020", "outer_2021"]
    for fold in outer:
        assert fold.train.target_month.max() < fold.validation.target_month.min()
        assert fold.validation.target_month.str[:4].eq(fold.name[-4:]).all()
        assert fold.validation.target_month.max() <= "2021-12"
        inner = chronological_inner_folds_v15(fold.train, min_train_months=18)
        assert len(inner) >= 2
        for part in inner:
            assert part.train.target_month.max() < part.validation.target_month.min()
            assert part.validation.target_month.max() <= fold.train.target_month.max()


def test_outer_and_inner_training_labels_must_be_produced_by_first_validation_issue():
    months = pd.period_range("2015-01", "2021-12", freq="M").astype(str)
    samples = pd.DataFrame({
        "target_month": months,
        "issue_date_utc": [f"{pd.Period(month, freq='M') - 1}-01T00:00:00Z" for month in months],
    })
    production = {month: f"{month}-15T00:00:00Z" for month in months}
    production["2018-11"] = "2018-12-15T00:00:00Z"

    outer = chronological_outer_folds_v15(
        samples, validation_years=(2019, 2020, 2021), label_production_dates_utc=production
    )
    assert outer[0].train.target_month.max() == "2018-10"
    assert "2018-11" not in set(outer[0].train.target_month)
    assert "2018-12" not in set(outer[0].train.target_month)

    inner = chronological_inner_folds_v15(
        outer[0].train, min_train_months=18, label_production_dates_utc=production
    )
    final_inner = inner[-1]
    issue = final_inner.validation.sort_values("target_month").iloc[0].issue_date_utc
    eligible = labels_published_by_issue(final_inner.train, issue, production)
    assert set(final_inner.train.target_month) == set(eligible.target_month)
    assert not final_inner.train.empty


def test_moving_blocks_include_only_complete_calendar_contiguous_blocks():
    blocks = complete_moving_blocks(
        ["2020-01", "2020-02", "2020-03", "2020-04"], block_length=3
    )

    assert blocks == [(0, 1, 2), (1, 2, 3)]
    assert all(len(block) == 3 for block in blocks)


def test_outer_folds_reject_any_consumed_test_target():
    samples = pd.DataFrame(
        {"target_month": ["2019-01", "2020-01", "2021-01", "2022-01"]}
    )

    with pytest.raises(ValueError, match="2021-12 development cutoff"):
        chronological_outer_folds_v15(samples)


def test_retrospective_regime_metrics_use_a_training_derived_threshold_only():
    summary = retrospective_regime_summary_v15(
        train_target_anomaly=[0.0, 1.0, 2.0, 3.0],
        validation_target_anomaly=[0.5, 2.5, 3.0],
        observations=[10.5, 12.5, 13.0],
        prediction=[10.4, 12.0, 12.0],
        climatology=[10.0, 10.0, 10.0],
    )

    assert summary["large_anomaly_threshold"] == pytest.approx(2.25)
    assert summary["large_anomaly_count"] == 2
    assert summary["ordinary_count"] == 1
    assert summary["large_anomaly_mae"] == pytest.approx(0.75)
