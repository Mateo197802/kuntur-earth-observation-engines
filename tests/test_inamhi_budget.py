import pytest

from kuntur_data.phase2.inamhi_budget import (
    DEFAULT_TIME_BUDGET_SECONDS,
    MAX_TIME_BUDGET_SECONDS,
    InamhiTimeBudget,
    InamhiTimeBudgetExceeded,
)


def test_default_inamhi_budget_is_fifteen_minutes(monkeypatch):
    monkeypatch.delenv("KUNTUR_INAMHI_TIME_BUDGET_SECONDS", raising=False)
    budget = InamhiTimeBudget.from_environment()
    assert budget.seconds == DEFAULT_TIME_BUDGET_SECONDS == 900


def test_requested_budget_cannot_exceed_hard_cap(monkeypatch):
    monkeypatch.setenv("KUNTUR_INAMHI_TIME_BUDGET_SECONDS", "7200")
    budget = InamhiTimeBudget.from_environment()
    assert budget.seconds == MAX_TIME_BUDGET_SECONDS == 900


def test_expired_budget_stops_work_and_limits_operation_timeout():
    budget = InamhiTimeBudget(900)
    budget.deadline = budget.started - 1
    with pytest.raises(InamhiTimeBudgetExceeded, match="budget exhausted"):
        budget.check("parse another yearbook")
    with pytest.raises(InamhiTimeBudgetExceeded):
        budget.timeout(300)


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_invalid_budget_configuration_is_rejected(monkeypatch, value):
    monkeypatch.setenv("KUNTUR_INAMHI_TIME_BUDGET_SECONDS", value)
    with pytest.raises(ValueError):
        InamhiTimeBudget.from_environment()
