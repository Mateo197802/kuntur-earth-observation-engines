"""Issue-time availability and metric helpers for predictive experiment v1.5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


ALLOWED_GATE_FEATURES = frozenset(
    {
        "t2m_anomaly_lag_03",
        "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06",
        "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06",
        "target_month_sin",
        "target_month_cos",
    }
)


@dataclass(frozen=True)
class HistoryAvailabilityAudit:
    history_months: tuple[str, ...]
    eligible: bool
    unavailable_months: tuple[str, ...]


@dataclass(frozen=True)
class TemporalFoldV15:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame


def history_months_for_target(
    target_month: str, *, width: int = 12, latest_lag: int = 3
) -> list[str]:
    """Return a complete oldest-to-newest monthly history ending at T-latest_lag."""
    if width < 1 or latest_lag < 1:
        raise ValueError("history width and latest lag must be positive")
    try:
        target = pd.Period(str(target_month), freq="M")
    except Exception as exc:
        raise ValueError("target_month must be a YYYY-MM calendar month") from exc
    start = target - latest_lag - width + 1
    return [str(month) for month in pd.period_range(start, target - latest_lag, freq="M")]


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def history_published_by_issue(
    *,
    target_month: str,
    issue_time_utc: str | datetime,
    production_dates_utc: Mapping[str, str | datetime],
    width: int = 12,
    latest_lag: int = 3,
) -> HistoryAvailabilityAudit:
    """Check all required source-month production timestamps against a forecast issue."""
    history = history_months_for_target(target_month, width=width, latest_lag=latest_lag)
    issue = _utc_datetime(issue_time_utc)
    target = pd.Period(str(target_month), freq="M")
    expected_issue = (target - 1).start_time.to_pydatetime().replace(tzinfo=timezone.utc)
    if issue != expected_issue:
        raise ValueError("issue_time_utc must be 00:00 UTC on the first day of target_month minus one")
    unavailable: list[str] = []
    for month in history:
        raw_date = production_dates_utc.get(month)
        if raw_date is None:
            unavailable.append(month)
            continue
        try:
            production = _utc_datetime(raw_date)
        except (TypeError, ValueError):
            unavailable.append(month)
            continue
        if production > issue:
            unavailable.append(month)
    return HistoryAvailabilityAudit(
        history_months=tuple(history),
        eligible=not unavailable,
        unavailable_months=tuple(unavailable),
    )


def labels_published_by_issue(
    frame: pd.DataFrame,
    issue_time_utc: str | datetime,
    production_dates_utc: Mapping[str, str | datetime],
) -> pd.DataFrame:
    """Keep target labels whose CMR production time is known by the forecast issue.

    Missing or malformed production metadata fails closed and excludes that label.
    CMR production time is a conservative availability proxy, not proof of first public release.
    """
    if "target_month" not in frame:
        raise ValueError("training rows must contain target_month")
    issue = _utc_datetime(issue_time_utc)
    periods = pd.PeriodIndex(frame.target_month.astype(str), freq="M")
    available: list[bool] = []
    for period in periods:
        raw_date = production_dates_utc.get(str(period))
        if raw_date is None:
            available.append(False)
            continue
        try:
            available.append(_utc_datetime(raw_date) <= issue)
        except (TypeError, ValueError):
            available.append(False)
    return frame.loc[np.asarray(available, dtype=bool)].copy().reset_index(drop=True)


def complete_moving_blocks(
    target_months: Iterable[str], *, block_length: int
) -> list[tuple[int, ...]]:
    """Enumerate only full moving blocks contained in contiguous calendar segments."""
    if block_length < 1:
        raise ValueError("block_length must be positive")
    periods = pd.PeriodIndex([str(month) for month in target_months], freq="M")
    if periods.duplicated().any() or not periods.is_monotonic_increasing:
        raise ValueError("target_months must be unique and chronological for moving blocks")
    segments: list[list[int]] = []
    current: list[int] = []
    previous: pd.Period | None = None
    for index, period in enumerate(periods):
        if previous is None or period == previous + 1:
            current.append(index)
        else:
            segments.append(current)
            current = [index]
        previous = period
    if current:
        segments.append(current)
    return [
        tuple(segment[start : start + block_length])
        for segment in segments
        for start in range(max(0, len(segment) - block_length + 1))
    ]


def validate_gate_feature_names(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Fail closed if any gate input is not in the explicit issue-time allowlist."""
    names = tuple(str(name) for name in feature_names)
    if not names:
        raise ValueError("the regime gate requires at least one issue-time feature")
    forbidden = sorted(set(names) - ALLOWED_GATE_FEATURES)
    if forbidden:
        raise ValueError(f"gate feature(s) not issue-time eligible: {forbidden}")
    return names


def gate_proxy_values(
    feature_matrix: Iterable[Iterable[float]],
    feature_names: Iterable[str],
    proxy_name: str,
) -> np.ndarray:
    """Extract one allowlisted feature for inference-time gate decisions."""
    names = validate_gate_feature_names([proxy_name])
    columns = tuple(str(name) for name in feature_names)
    if proxy_name not in columns:
        raise ValueError(f"gate proxy {proxy_name!r} is absent from the feature matrix")
    matrix = np.asarray(feature_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(columns):
        raise ValueError("feature matrix width must match feature_names")
    result = matrix[:, columns.index(names[0])].copy()
    if proxy_name in {"t2m_anomaly_lag_03", "t2m_anomaly_mean_03", "t2m_anomaly_mean_06", "t2m_anomaly_slope_06"}:
        result = np.abs(result)
    return result


def smooth_proxy_weights(
    validation_proxy: Iterable[float],
    training_proxy: Iterable[float],
    *,
    strength: float,
    low_quantile: float = 0.5,
    high_quantile: float = 0.9,
) -> pd.Series:
    """Map one issue-time proxy to a bounded residual-blend weight using train-only quantiles."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be between zero and one")
    if not 0 <= low_quantile < high_quantile <= 1:
        raise ValueError("quantiles must satisfy 0 <= low < high <= 1")
    train = np.asarray(list(training_proxy), dtype=float)
    valid = np.asarray(list(validation_proxy), dtype=float)
    finite_train = train[np.isfinite(train)]
    if not finite_train.size:
        raise ValueError("training proxy must contain a finite value")
    low, high = np.quantile(finite_train, [low_quantile, high_quantile])
    if high <= low:
        scaled = np.where(valid > high, 1.0, 0.0)
    else:
        scaled = np.clip((valid - low) / (high - low), 0.0, 1.0)
    scaled[~np.isfinite(valid)] = 0.0
    return pd.Series(strength * scaled)


def select_fixed_blend_strength(
    *,
    observations: Iterable[float],
    climatology: Iterable[float],
    residual_prediction: Iterable[float],
    candidates: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, float]:
    """Select a fixed residual blend by inner-fold MAE, retaining alpha zero."""
    y = np.asarray(list(observations), dtype=float)
    climate = np.asarray(list(climatology), dtype=float)
    residual = np.asarray(list(residual_prediction), dtype=float)
    if y.shape != climate.shape or y.shape != residual.shape or not y.size:
        raise ValueError("inner observations, climatology and residuals must be aligned and nonempty")
    if not (np.isfinite(y).all() and np.isfinite(climate).all() and np.isfinite(residual).all()):
        raise ValueError("inner blend selection requires finite values")
    options = sorted(set([0.0, *(float(value) for value in candidates)]))
    if options[0] < 0 or options[-1] > 1 or not np.isfinite(options).all():
        raise ValueError("blend candidates must be finite values within [0, 1]")
    scored = []
    for alpha in options:
        error = climate + alpha * residual - y
        scored.append((float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error)))), alpha))
    mae, _, alpha = min(scored)
    return alpha, mae


def metric_summary_v15(
    observations: Iterable[float],
    prediction: Iterable[float],
    baseline: Iterable[float],
) -> dict[str, float | int]:
    """Calculate paired metrics after applying one identical finite-month mask."""
    y = np.asarray(list(observations), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    if y.shape != p.shape or y.shape != b.shape or not y.size:
        raise ValueError("observations, predictions and baseline must be aligned and nonempty")
    valid = np.isfinite(y) & np.isfinite(p) & np.isfinite(b)
    y, p, b = y[valid], p[valid], b[valid]
    if not y.size:
        raise ValueError("paired metrics require at least one finite month")
    error = p - y
    baseline_error = b - y
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(np.square(error))))
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    return {
        "sample_count": int(len(y)),
        "mae": mae,
        "rmse": rmse,
        "bias": float(np.mean(error)),
        "mae_skill_vs_baseline": float((baseline_mae - mae) / baseline_mae) if baseline_mae else 0.0,
        "rmse_skill_vs_baseline": float((baseline_rmse - rmse) / baseline_rmse) if baseline_rmse else 0.0,
        "monthly_win_rate_vs_baseline": float(np.mean(np.abs(error) < np.abs(baseline_error))),
    }


def chronological_outer_folds_v15(
    eligible_samples: pd.DataFrame,
    *,
    validation_years: Iterable[int] = (2019, 2020, 2021),
    development_end: str = "2021-12",
    label_production_dates_utc: Mapping[str, str | datetime] | None = None,
) -> list[TemporalFoldV15]:
    """Return expanding year-ahead folds and refuse all post-development rows."""
    if "target_month" not in eligible_samples:
        raise ValueError("eligible samples must contain target_month")
    ordered = eligible_samples.sort_values("target_month", kind="stable").reset_index(drop=True)
    months = ordered.target_month.astype(str)
    if ordered.empty or months.max() > development_end:
        raise ValueError("outer evaluation refuses rows beyond the 2021-12 development cutoff")
    if months.duplicated().any():
        raise ValueError("eligible sample target months must be unique")
    folds: list[TemporalFoldV15] = []
    for year in validation_years:
        start = f"{int(year):04d}-01"
        end = f"{int(year):04d}-12"
        train = ordered[months < start].copy().reset_index(drop=True)
        validation = ordered[(months >= start) & (months <= end)].copy().reset_index(drop=True)
        if train.empty or validation.empty:
            continue
        if train.target_month.max() >= validation.target_month.min():
            raise ValueError(f"outer fold {year} is not strictly chronological")
        if label_production_dates_utc is not None:
            issue_time = str(validation.sort_values("target_month", kind="stable").iloc[0].issue_date_utc)
            train = labels_published_by_issue(train, issue_time, label_production_dates_utc)
            if train.empty:
                raise ValueError(f"no target labels were produced by the first issue time in outer fold {year}")
        folds.append(TemporalFoldV15(f"outer_{int(year)}", train, validation))
    if not folds:
        raise ValueError("no chronological outer folds were formed")
    return folds


def chronological_inner_folds_v15(
    outer_train: pd.DataFrame,
    *,
    min_train_months: int = 18,
    label_production_dates_utc: Mapping[str, str | datetime] | None = None,
) -> list[TemporalFoldV15]:
    """Create expanding semiannual validation windows inside one outer training fold."""
    if "target_month" not in outer_train:
        raise ValueError("outer training rows must contain target_month")
    if min_train_months < 1:
        raise ValueError("min_train_months must be positive")
    ordered = outer_train.sort_values("target_month", kind="stable").reset_index(drop=True)
    months = ordered.target_month.astype(str)
    if ordered.empty or months.max() > "2021-12":
        raise ValueError("inner folds accept development training rows only through 2021-12")
    folds: list[TemporalFoldV15] = []
    first_year = int(months.min()[:4])
    last_year = int(months.max()[:4])
    for year in range(first_year, last_year + 1):
        for half, start_month, end_month in ((1, "01", "06"), (2, "07", "12")):
            start = f"{year:04d}-{start_month}"
            end = f"{year:04d}-{end_month}"
            train = ordered[months < start].copy().reset_index(drop=True)
            validation = ordered[(months >= start) & (months <= end)].copy().reset_index(drop=True)
            if len(train) < min_train_months or validation.empty:
                continue
            if train.target_month.max() >= validation.target_month.min():
                raise ValueError(f"inner window {year}-H{half} is not strictly chronological")
            if label_production_dates_utc is not None:
                issue_time = str(validation.sort_values("target_month", kind="stable").iloc[0].issue_date_utc)
                train = labels_published_by_issue(train, issue_time, label_production_dates_utc)
                if len(train) < min_train_months:
                    continue
            folds.append(TemporalFoldV15(f"inner_{year}_h{half}", train, validation))
    if len(folds) < 2:
        raise ValueError("at least two nested chronological inner folds are required")
    return folds


def retrospective_regime_summary_v15(
    *,
    train_target_anomaly: Iterable[float],
    validation_target_anomaly: Iterable[float],
    observations: Iterable[float],
    prediction: Iterable[float],
    climatology: Iterable[float],
) -> dict[str, float | int]:
    """Use training-q75 target anomaly only to label retrospective score strata."""
    train_anomaly = np.asarray(list(train_target_anomaly), dtype=float)
    validation_anomaly = np.asarray(list(validation_target_anomaly), dtype=float)
    y = np.asarray(list(observations), dtype=float)
    p = np.asarray(list(prediction), dtype=float)
    climate = np.asarray(list(climatology), dtype=float)
    if not train_anomaly.size or y.shape != p.shape or y.shape != climate.shape or y.shape != validation_anomaly.shape:
        raise ValueError("regime diagnostics require aligned nonempty target and forecast arrays")
    train_anomaly = train_anomaly[np.isfinite(train_anomaly)]
    if not train_anomaly.size:
        raise ValueError("training anomalies must contain finite values")
    threshold = float(np.quantile(np.abs(train_anomaly), 0.75))
    valid = np.isfinite(validation_anomaly) & np.isfinite(y) & np.isfinite(p) & np.isfinite(climate)
    large = np.abs(validation_anomaly[valid]) >= threshold
    error = np.abs(p[valid] - y[valid])
    baseline_error = np.abs(climate[valid] - y[valid])
    large_count = int(large.sum())
    ordinary_count = int((~large).sum())
    large_mae = float(error[large].mean()) if large_count else float("nan")
    ordinary_mae = float(error[~large].mean()) if ordinary_count else float("nan")
    large_baseline_mae = float(baseline_error[large].mean()) if large_count else float("nan")
    ordinary_baseline_mae = float(baseline_error[~large].mean()) if ordinary_count else float("nan")
    return {
        "large_anomaly_threshold": threshold,
        "large_anomaly_count": large_count,
        "ordinary_count": ordinary_count,
        "large_anomaly_mae": large_mae,
        "ordinary_mae": ordinary_mae,
        "large_anomaly_baseline_mae": large_baseline_mae,
        "ordinary_baseline_mae": ordinary_baseline_mae,
    }
