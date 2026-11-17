from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import warnings

from kuntur_data.phase2.predictive_v14 import (
    augment_training_rows,
    load_fold_checkpoint,
    build_history_tensor,
    chronological_inner_folds,
    chronological_outer_folds,
    expanding_monthly_climatology,
    feature_pair,
    metric_summary,
    monthly_gold_wide,
    paired_moving_block_ci,
    select_shrinkage_alpha,
    save_fold_checkpoint,
    shrink_predictions,
    validate_forecast_timeline,
)


def _samples(start: str, end: str) -> pd.DataFrame:
    targets = pd.period_range(start, end, freq="M")
    rows = []
    for target in targets:
        rows.append(
            {
                "sample_id": f"sample:{target}",
                "target_month": str(target),
                "forecast_origin_month": str(target - 1),
                "issue_date_utc": f"{target - 1}-01T00:00:00Z",
                "history_start_month": str(target - 14),
                "history_end_month": str(target - 3),
            }
        )
    return pd.DataFrame(rows)


def test_timeline_requires_exact_t_minus_14_to_t_minus_3_history():
    frame = _samples("2003-03", "2003-04")
    validate_forecast_timeline(frame)

    frame.loc[0, "history_end_month"] = "2003-02"
    with pytest.raises(ValueError, match="history_end_month"):
        validate_forecast_timeline(frame)


def test_history_tensor_uses_only_t_minus_14_through_t_minus_3():
    sample = _samples("2003-03", "2003-03")
    months = pd.period_range("2002-01", "2003-03", freq="M")
    monthly = pd.DataFrame(
        {"merra2_t2m_c": np.arange(len(months), dtype=float)},
        index=months.astype(str),
    )
    tensor, month_labels = build_history_tensor(sample, monthly, ("merra2_t2m_c",))

    expected = pd.period_range("2002-01", "2002-12", freq="M").astype(str).tolist()
    assert tensor.shape == (1, 12, 1)
    assert month_labels[0] == expected
    assert tensor[0, :, 0].tolist() == list(range(12))
    # March 2003 is a target and must never enter its predictor window.
    assert "2003-03" not in month_labels[0]


def test_outer_and_inner_folds_are_strictly_chronological_and_development_only():
    frame = _samples("2003-03", "2021-12")
    outer = chronological_outer_folds(frame)

    assert [fold.name for fold in outer] == ["outer_2012_2014", "outer_2015_2017", "outer_2018_2021"]
    for fold in outer:
        assert fold.train.target_month.max() < fold.validation.target_month.min()
        assert fold.validation.target_month.max() <= "2021-12"
        inner = chronological_inner_folds(fold.train)
        assert len(inner) == 3
        for inner_fold in inner:
            assert inner_fold.train.target_month.max() < inner_fold.validation.target_month.min()
            assert inner_fold.validation.target_month.max() <= fold.train.target_month.max()


def test_outer_folds_reject_missing_development_months():
    frame = _samples("2003-03", "2021-12")
    frame = frame[frame.target_month != "2016-07"]
    with pytest.raises(ValueError, match="complete monthly chronology"):
        chronological_outer_folds(frame)


def test_outer_folds_reject_consumed_test_rows():
    frame = _samples("2003-03", "2024-12")
    with pytest.raises(ValueError, match="development cutoff"):
        chronological_outer_folds(frame)


def _monthly_source(start="2002-01", end="2006-12"):
    from kuntur_data.phase2.predictive_v14 import HISTORY_VARIABLES

    months = pd.period_range(start, end, freq="M")
    values = {}
    for index, variable in enumerate(HISTORY_VARIABLES):
        values[variable] = np.asarray(
            [index + 0.1 * i + np.sin(2 * np.pi * (month.month - 1) / 12) for i, month in enumerate(months)],
            dtype=float,
        )
    return pd.DataFrame(values, index=months.astype(str))


def _with_targets(samples: pd.DataFrame) -> pd.DataFrame:
    frame = samples.copy()
    periods = pd.PeriodIndex(frame.target_month, freq="M")
    frame["target_value"] = 20.0 + periods.month.to_numpy() / 10.0 + np.arange(len(frame)) / 100.0
    return frame


def test_feature_pair_uses_training_only_climatology_and_fold_local_scaling():
    samples = _with_targets(_samples("2003-03", "2005-12"))
    train = samples[samples.target_month <= "2004-12"].copy()
    validation = samples[samples.target_month >= "2005-01"].copy()
    monthly = _monthly_source()
    pair = feature_pair(train, validation, monthly, "temperature_only")

    assert pair.x_train.shape[1] == 20
    assert pair.x_validation.shape[1] == 20
    assert np.isfinite(pair.x_train).all() and np.isfinite(pair.x_validation).all()
    assert np.allclose(pair.x_train[:, :10].mean(axis=0), 0.0, atol=1e-10)
    assert set(pair.training_climatology) == set(range(1, 13))
    assert pair.feature_climatology["merra2_t2m_c"]
    assert pair.imputation_means.shape == (10,)
    assert pair.scaler_mean.shape == (10,)
    assert pair.scaler_scale.shape == (10,)
    assert np.allclose(pair.offset_validation, [pair.training_climatology[p.month] for p in pd.PeriodIndex(validation.target_month, freq="M")])


def test_feature_pair_does_not_use_source_months_after_latest_allowed_predictor():
    samples = _with_targets(_samples("2003-03", "2005-12"))
    train = samples[samples.target_month <= "2004-12"].copy()
    validation = samples[samples.target_month >= "2005-01"].copy()
    monthly = _monthly_source()
    altered = monthly.copy()
    # All validation histories end by September 2005, so later source changes are unavailable.
    altered.loc["2005-10":, :] = altered.loc["2005-10":, :] + 1e8

    original_pair = feature_pair(train, validation, monthly, "all_sources")
    altered_pair = feature_pair(train, validation, altered, "all_sources")
    assert np.allclose(original_pair.x_train, altered_pair.x_train)
    assert np.allclose(original_pair.x_validation, altered_pair.x_validation)
    assert original_pair.training_climatology == altered_pair.training_climatology


def test_monthly_gold_wide_rejects_nonvalidated_predictors():
    from kuntur_data.phase2.predictive_v14 import HISTORY_VARIABLES

    rows = []
    for variable in HISTORY_VARIABLES:
        rows.append({"month": "2002-01", "variable_id": variable, "value": 1.0, "quality_status": "VALID"})
    gold = pd.DataFrame(rows)
    monthly = monthly_gold_wide(gold)
    assert "merra2_wind_speed" in monthly.columns
    assert "imerg_log_precip" in monthly.columns
    gold.loc[0, "quality_status"] = "PROVISIONAL"
    with pytest.raises(ValueError, match="VALID Gold"):
        monthly_gold_wide(gold)


def test_shrinkage_is_climatology_first_and_alpha_is_bounded():
    climate = np.asarray([10.0, 20.0])
    residual = np.asarray([2.0, -4.0])
    assert shrink_predictions(climate, residual, 0.0).tolist() == [10.0, 20.0]
    assert shrink_predictions(climate, residual, 0.5).tolist() == [11.0, 18.0]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        shrink_predictions(climate, residual, 1.01)


def test_nested_alpha_selection_can_retain_climatology_fallback():
    y = np.asarray([10.0, 10.2, 9.8, 10.1])
    climate = np.asarray([10.0, 10.0, 10.0, 10.0])
    harmful_residual = np.asarray([2.0, -2.0, 2.0, -2.0])
    alpha, prediction, score = select_shrinkage_alpha(y, climate, harmful_residual)
    assert alpha == 0.0
    assert np.array_equal(prediction, climate)
    assert score["mae"] == pytest.approx(np.mean(np.abs(y - climate)))


def test_expanding_climatology_only_uses_observations_before_each_target():
    history = pd.DataFrame({"target_month": ["2003-01", "2004-01"], "target_value": [10.0, 12.0]})
    future = pd.DataFrame({"target_month": ["2005-01", "2006-01"], "target_value": [14.0, 9999.0]})
    prediction = expanding_monthly_climatology(history, future)
    assert prediction.tolist() == pytest.approx([11.0, 12.0])
    changed = future.copy()
    changed.loc[1, "target_value"] = -9999.0
    changed_prediction = expanding_monthly_climatology(history, changed)
    assert changed_prediction[0] == prediction[0]


def test_augmentation_keeps_each_observed_target_total_weight_one():
    x = np.asarray([[1.0, 2.0, 0.0, 0.0], [3.0, 4.0, 0.0, 0.0]])
    y = np.asarray([0.5, -0.5])
    names = ("target_month_sin", "x", "missing__target_month_sin", "missing__x")
    augmented_x, augmented_y, weights = augment_training_rows(
        x, y, names, sample_ids=("a", "b"), policy="mild_dropout", intensity=0.5, seed=7
    )
    assert augmented_x.shape == (4, 4)
    assert augmented_y.tolist() == [0.5, 0.5, -0.5, -0.5]
    assert weights.reshape(2, 2).sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert np.array_equal(weights, np.full(4, 0.5))
    assert np.array_equal(augmented_x[[0, 2]], x)
    assert np.all(augmented_x[[1, 3], 0] == x[:, 0])
    assert np.all(augmented_x[[1, 3], 2] == 0.0)
    assert np.all(augmented_x[[1, 3], 3] == 1.0)


def test_paired_moving_block_interval_is_seeded_and_metrics_pair_months():
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    model = np.asarray([1.2, 1.8, 3.2, 3.8])
    baseline = np.asarray([2.0, 2.0, 2.0, 2.0])
    metrics = metric_summary(y, model, baseline)
    assert metrics["sample_count"] == 4
    assert metrics["monthly_win_rate_vs_baseline"] == pytest.approx(0.75)
    assert metrics["mae_skill_vs_baseline"] > 0
    first = paired_moving_block_ci(np.abs(model - y) - np.abs(baseline - y), seed=12, replicates=200)
    second = paired_moving_block_ci(np.abs(model - y) - np.abs(baseline - y), seed=12, replicates=200)
    assert first == second
    assert first["block_length_months"] == 12
    assert first["sample_count"] == 4


def test_fold_checkpoint_is_atomic_and_reused_only_for_matching_inputs(tmp_path):
    path = tmp_path / "outer_2012_2014.joblib"
    payload = {"predictions": [1.0, 2.0], "registry": [{"status": "COMPLETED"}]}
    save_fold_checkpoint(path, payload, fingerprint="input-code-hash")
    assert path.is_file()
    assert not path.with_suffix(path.suffix + ".tmp").exists()
    assert load_fold_checkpoint(path, fingerprint="input-code-hash") == payload
    assert load_fold_checkpoint(path, fingerprint="changed-input-code-hash") is None


def test_histogram_boosting_fit_uses_bounded_native_threads_without_parallel_warning():
    from scripts.run_predictive_v14 import estimator_complexity, fit_estimator

    rng = np.random.default_rng(42)
    x = rng.normal(size=(48, 12))
    y = rng.normal(size=48)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        model, timing = fit_estimator(
            "hist_gradient_boosting",
            {"learning_rate": 0.05, "max_iter": 8, "max_leaf_nodes": 2, "min_samples_leaf": 5, "l2_regularization": 5.0},
            x,
            y,
            None,
            42,
        )

    warning_messages = [str(item.message) for item in recorded]
    assert not any("sklearn.utils.parallel.delayed" in message for message in warning_messages)
    assert np.isfinite(model.predict(x)).all()
    assert timing["fit_cpu_seconds"] >= 0
    complexity = estimator_complexity(model, "hist_gradient_boosting", x.shape[1])
    assert complexity["tree_node_count"] > 0
    assert complexity["complexity_indicator"] == complexity["tree_node_count"]
