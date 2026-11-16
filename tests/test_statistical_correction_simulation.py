import numpy as np
import pandas as pd
from types import SimpleNamespace

from scripts.statistical_correction import run_calibration as calibration
from scripts.statistical_correction.finalize_correction import _build_validation_registry
from scripts.statistical_correction.run_calibration import (
    _FitPlaceholder,
    _summarize_method,
    _paired_mean_difference_ci,
    clopper_pearson,
    make_series,
    _max_missing_run,
    scenario_seed,
)


def test_seeded_data_generator_repeats_exactly_for_each_stationary_process():
    for process in ("iid_gaussian", "ar1_0844", "arma11", "seasonal_ar1_12", "student_t3"):
        first, periods, mask = make_series(process, 120, 0.25, np.random.default_rng(710))
        second, periods2, mask2 = make_series(process, 120, 0.25, np.random.default_rng(710))

        np.testing.assert_array_equal(first, second)
        assert periods.equals(periods2)
        np.testing.assert_array_equal(mask, mask2)


def test_missingness_processes_preserve_full_calendar_and_explicit_nan_gaps():
    for process in ("ar1_060_mcar20", "ar1_060_seasonal_missing"):
        values, periods, observed = make_series(process, 276, 0.0, np.random.default_rng(19))

        assert len(values) == len(periods) == 276
        assert np.isnan(values[~observed]).all()
        assert np.isfinite(values[observed]).all()
        assert 0 < (~observed).sum() < 276


def test_missing_run_is_counted_in_calendar_months():
    observed = np.array([True, False, False, False, True, False, True])

    assert _max_missing_run(observed) == 3


def test_paired_metric_difference_uses_matched_replications():
    n, mean, low, high = _paired_mean_difference_ci(
        np.array([3.0, 4.0, 5.0]), np.array([1.0, 2.0, 3.0])
    )

    assert n == 3
    assert mean == 2.0
    assert low == high == 2.0


def test_registry_uses_public_method_identifier_and_requires_all_validation_effects():
    selected = {
        "decisions": {
            "iid_gaussian": {
                "selected": {"method": "OLS_HAC_LAG6", "method_id": "ols_hac", "calibration_score": 1.2}
            }
        }
    }
    rows = [
        {
            "process": "iid_gaussian", "n_months": 276, "method": "OLS_HAC_LAG6",
            "true_slope_per_decade": slope,
            "validity_status": "VALIDATED_FOR_DECLARED_REGIME",
            "min_observation_fraction": 1.0, "max_observation_fraction": 1.0,
        }
        for slope in (0.0, 0.1, 0.25, 0.5, -0.5)
    ]
    registry = _build_validation_registry(selected, pd.DataFrame(rows))

    assert registry["validated_regime_count"] == 1
    assert registry["regimes"]["iid_gaussian"]["method_id"] == "OLS_HAC_FROZEN_REFERENCE"

    incomplete = _build_validation_registry(selected, pd.DataFrame(rows[:-1]))
    assert incomplete["validated_regime_count"] == 0


def test_selection_and_final_validation_use_separate_seed_streams():
    selection = scenario_seed("selection", "ar1_0844", 0.0, 276)
    validation = scenario_seed("validation", "ar1_0844", 0.0, 276)

    assert selection != validation
    assert selection == scenario_seed("selection", "ar1_0844", 0.0, 276)


def test_monte_carlo_error_interval_uses_exact_two_sided_binomial_limits():
    low, high = clopper_pearson(50, 1000)

    assert 0.037 < low < 0.04
    assert 0.065 < high < 0.066


def test_structurally_unsupported_method_replicates_are_not_counted_as_optimizer_failures():
    fits = [
        _FitPlaceholder(False, support_eligible=False),
        SimpleNamespace(converged=True, slope=0.0, p_value=0.5, ci_low=-1.0, ci_high=1.0, warnings=("ConvergenceWarning: example",)),
        SimpleNamespace(converged=True, slope=0.1, p_value=0.5, ci_low=-1.0, ci_high=1.0),
    ]
    row = _summarize_method(
        phase="validation", process="ar1_060_mcar20", n_months=276, slope=0.0,
        method_spec={"name": "GLSAR", "method": "gls_ar1"}, reps_requested=3,
        seed=11, fits=fits, method_wall=0.1, method_cpu=0.1, rss_peak=10,
        oracle_widths=[1.0, 1.0], stress=False, bootstrap_inner=199,
    )

    assert row["replications_supported"] == 2
    assert row["support_fraction"] == 2 / 3
    assert row["convergence_failure_rate"] == 0.0
    assert row["validity_status"] == "UNSUPPORTED_REGIME"
    assert row["optimizer_warning_counts"] == {"ConvergenceWarning": 1}


def test_interrupted_scenario_resumes_from_atomic_replication_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "METHODS", ({"name": "TEST", "method": "ols_hac", "lag": 6, "block": 12},))
    monkeypatch.setattr(calibration, "CHECKPOINT_EVERY_REPS", 2)
    monkeypatch.setattr(calibration, "oracle_ci_width", lambda process, periods, observed: 1.0)
    calls = 0

    def interrupted_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("simulated interruption")
        return SimpleNamespace(
            slope=0.0, standard_error=1.0, ci_low=-1.0, ci_high=1.0,
            p_value=0.5, converged=True, support_eligible=True,
        )

    monkeypatch.setattr(calibration, "_fit_method", interrupted_fit)
    try:
        calibration._run_scenario("selection", "iid_gaussian", 24, 0.0, 4, 9, tmp_path)
        raise AssertionError("the injected interruption should propagate")
    except RuntimeError as exc:
        assert str(exc) == "simulated interruption"

    checkpoint = next((tmp_path / "checkpoints").glob("selection_iid_gaussian_n24_s0p0.json"))
    saved = __import__("json").loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["status"] == "IN_PROGRESS"
    assert saved["progress"]["completed_reps"] == 2

    calls = 0

    def resumed_fit(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(
            slope=0.0, standard_error=1.0, ci_low=-1.0, ci_high=1.0,
            p_value=0.5, converged=True, support_eligible=True,
        )

    monkeypatch.setattr(calibration, "_fit_method", resumed_fit)
    rows = calibration._run_scenario("selection", "iid_gaussian", 24, 0.0, 4, 9, tmp_path)

    assert calls == 2
    assert rows[0]["replications_requested"] == 4
    assert __import__("json").loads(checkpoint.read_text(encoding="utf-8"))["status"] == "COMPLETE"
