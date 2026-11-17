"""Bounded wall-clock budget for optional INAMHI acquisition and extraction."""

from __future__ import annotations

import os
import time


DEFAULT_TIME_BUDGET_SECONDS = 15 * 60
MAX_TIME_BUDGET_SECONDS = 15 * 60


class InamhiTimeBudgetExceeded(RuntimeError):
    """Raised when the bounded INAMHI acquisition/extraction window expires."""


class InamhiTimeBudget:
    """A monotonic deadline with a hard maximum, independent of wall-clock changes."""

    def __init__(self, seconds: int):
        if seconds < 1:
            raise ValueError("INAMHI time budget must be at least one second")
        self.seconds = min(int(seconds), MAX_TIME_BUDGET_SECONDS)
        self.started = time.monotonic()
        self.deadline = self.started + self.seconds

    @classmethod
    def from_environment(cls) -> "InamhiTimeBudget":
        configured = os.environ.get(
            "KUNTUR_INAMHI_TIME_BUDGET_SECONDS",
            str(DEFAULT_TIME_BUDGET_SECONDS),
        )
        try:
            seconds = int(configured)
        except ValueError as exc:
            raise ValueError("KUNTUR_INAMHI_TIME_BUDGET_SECONDS must be an integer") from exc
        if seconds < 1:
            raise ValueError("KUNTUR_INAMHI_TIME_BUDGET_SECONDS must be positive")
        return cls(seconds)

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def check(self, operation: str = "INAMHI work") -> None:
        if self.remaining() <= 0:
            raise InamhiTimeBudgetExceeded(f"INAMHI time budget exhausted before {operation}")

    def timeout(self, requested_seconds: float) -> float:
        self.check("network or page operation")
        return max(0.1, min(float(requested_seconds), self.remaining()))
