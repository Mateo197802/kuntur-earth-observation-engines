from __future__ import annotations

from scripts.statistical_correction.run_scoped_validation import validation_plan


def test_validation_scope_only_includes_selection_eligible_primary_processes() -> None:
    decisions = {
        "decisions": {
            "iid_gaussian": {"selected": {"method": "SEASONAL_MK_YEAR2"}},
            "ar1_0844": {"selected": None},
            "arma11": {"selected": None},
            "seasonal_ar1_12": {"selected": None},
            "ar1_060_mcar20": {"selected": {"method": "SEASONAL_MK_YEAR2"}},
        }
    }
    included, deferred = validation_plan(decisions)

    assert len(included) == 10
    assert {row["process"] for row in included} == {"iid_gaussian", "ar1_060_mcar20"}
    assert all(row["n_months"] == 276 for row in included)
    assert all(row["outer_replications"] == 1000 for row in included)
    assert all(row["bootstrap_inner_draws"] == 399 for row in included)
    assert len({row["seed"] for row in included}) == 10
    assert len(deferred) == 50
    reasons = {reason: sum(row["reason"] == reason for row in deferred) for reason in {row["reason"] for row in deferred}}
    assert reasons["NO_SELECTION_CANDIDATE_PASSED_SELECTION_GATES"] == 15
    assert reasons["SECONDARY_DGP_SCOPE_DEFERRED"] == 8
    assert reasons["SHORT_SERIES_SENSITIVITY_DEFERRED"] == 27


def test_validation_scope_refuses_an_empty_selection() -> None:
    decisions = {"decisions": {process: {"selected": None} for process in (
        "iid_gaussian", "ar1_0844", "arma11", "seasonal_ar1_12", "ar1_060_mcar20"
    )}}
    try:
        validation_plan(decisions)
    except ValueError as error:
        assert "No primary process passed selection" in str(error)
    else:
        raise AssertionError("empty candidate selection must not start an independent validation")
