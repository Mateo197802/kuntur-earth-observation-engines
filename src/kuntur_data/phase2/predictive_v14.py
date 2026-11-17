"""Leakage-guarded helpers for KUNTUR predictive engine v1.4."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEVELOPMENT_START = "2003-03"
DEVELOPMENT_END = "2021-12"
SOURCE_RELEASE_LAG_MONTHS = 2
HISTORY_VARIABLES = (
    "merra2_t2m_c",
    "merra2_qv2m",
    "merra2_u2m",
    "merra2_v2m",
    "merra2_ps",
    "imerg_precip_monthly_total",
    "gldas_soil_moisture_0_10cm",
    "gldas_soil_moisture_10_40cm",
    "gldas_soil_moisture_40_100cm",
    "gldas_soil_moisture_100_200cm",
)

FEATURE_GROUP_COLUMNS = {
    "temperature_only": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06",
    ),
    "atmosphere": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "qv_anomaly_lag_03", "qv_anomaly_mean_03",
        "wind_speed_anomaly_lag_03", "wind_speed_anomaly_mean_03",
        "ps_anomaly_lag_03", "ps_anomaly_mean_03",
    ),
    "atmosphere_precipitation": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "qv_anomaly_lag_03", "qv_anomaly_mean_03",
        "wind_speed_anomaly_lag_03", "wind_speed_anomaly_mean_03",
        "ps_anomaly_lag_03", "ps_anomaly_mean_03", "precip_log_anomaly_lag_03",
        "precip_log_anomaly_mean_03", "precip_log_anomaly_mean_06",
    ),
    "atmosphere_soil": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "qv_anomaly_lag_03", "qv_anomaly_mean_03",
        "wind_speed_anomaly_lag_03", "wind_speed_anomaly_mean_03",
        "ps_anomaly_lag_03", "ps_anomaly_mean_03", "soil_0_10_anomaly_mean_03",
        "soil_10_40_anomaly_mean_03", "soil_40_100_anomaly_mean_03",
        "soil_100_200_anomaly_mean_03",
    ),
    "all_sources": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "qv_anomaly_lag_03", "qv_anomaly_mean_03",
        "wind_speed_anomaly_lag_03", "wind_speed_anomaly_mean_03",
        "ps_anomaly_lag_03", "ps_anomaly_mean_03", "precip_log_anomaly_lag_03",
        "precip_log_anomaly_mean_03", "precip_log_anomaly_mean_06",
        "soil_0_10_anomaly_mean_03", "soil_10_40_anomaly_mean_03",
        "soil_40_100_anomaly_mean_03", "soil_100_200_anomaly_mean_03",
    ),
    "all_without_atmosphere_extras": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "precip_log_anomaly_lag_03",
        "precip_log_anomaly_mean_03", "precip_log_anomaly_mean_06",
        "soil_0_10_anomaly_mean_03", "soil_10_40_anomaly_mean_03",
        "soil_40_100_anomaly_mean_03", "soil_100_200_anomaly_mean_03",
    ),
    "all_without_precipitation": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "qv_anomaly_lag_03", "qv_anomaly_mean_03",
        "wind_speed_anomaly_lag_03", "wind_speed_anomaly_mean_03",
        "ps_anomaly_lag_03", "ps_anomaly_mean_03", "soil_0_10_anomaly_mean_03",
        "soil_10_40_anomaly_mean_03", "soil_40_100_anomaly_mean_03",
        "soil_100_200_anomaly_mean_03",
    ),
    "all_without_soil": (
        "target_month_sin", "target_month_cos", "t2m_anomaly_lag_03",
        "t2m_anomaly_lag_04", "t2m_anomaly_lag_05", "t2m_anomaly_mean_03",
        "t2m_anomaly_mean_06", "t2m_anomaly_mean_12", "t2m_anomaly_sd_06",
        "t2m_anomaly_slope_06", "qv_anomaly_lag_03", "qv_anomaly_mean_03",
        "wind_speed_anomaly_lag_03", "wind_speed_anomaly_mean_03",
        "ps_anomaly_lag_03", "ps_anomaly_mean_03", "precip_log_anomaly_lag_03",
        "precip_log_anomaly_mean_03", "precip_log_anomaly_mean_06",
    ),
}


@dataclass(frozen=True)
class TemporalFold:
    name: str
    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class FeaturePair:
    x_train: np.ndarray
    x_validation: np.ndarray
    y_train_residual: np.ndarray
    y_validation: np.ndarray
    offset_train: np.ndarray
    offset_validation: np.ndarray
    feature_names: tuple[str, ...]
    training_climatology: dict[int, float]
    feature_climatology: dict[str, dict[int, float]]
    imputation_means: np.ndarray
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray


def _periods(series: pd.Series) -> pd.PeriodIndex:
    try:
        return pd.PeriodIndex(series.astype(str), freq="M")
    except Exception as exc:
        raise ValueError("target_month must contain YYYY-MM calendar months") from exc


def latest_available_label_month(issue_date_utc: str) -> pd.Period:
    """Latest monthly target assumed released by a forecast issue timestamp.

    The frozen two-month lag is an explicit modeling assumption; actual source
    publication timestamps were not established for this experiment.
    """
    issue = pd.to_datetime(issue_date_utc, utc=True, errors="raise")
    if issue.day != 1 or issue.hour != 0 or issue.minute != 0 or issue.second != 0:
        raise ValueError("issue_date_utc must be midnight UTC on the first of a month")
    issue_month = pd.Period(f"{issue.year}-{issue.month:02d}", freq="M")
    return issue_month - SOURCE_RELEASE_LAG_MONTHS


def labels_available_by_issue(frame: pd.DataFrame, issue_date_utc: str) -> pd.DataFrame:
    """Keep only monthly labels available at the specified forecast issue."""
    cutoff = latest_available_label_month(issue_date_utc)
    return frame.loc[_periods(frame["target_month"]) <= cutoff].copy().reset_index(drop=True)


def validate_training_labels_available(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Fail closed if any fit label postdates the first validation issue cutoff."""
    if train.empty or validation.empty:
        raise ValueError("training and validation windows must be nonempty")
    first_validation = validation.assign(_period=_periods(validation["target_month"])).sort_values(
        "_period", kind="stable"
    ).iloc[0]
    cutoff = latest_available_label_month(str(first_validation["issue_date_utc"]))
    unavailable = _periods(train["target_month"]) > cutoff
    if unavailable.any():
        latest = _periods(train.loc[unavailable, "target_month"]).max()
        raise ValueError(
            f"training labels include outcomes unavailable at first validation issue "
            f"{first_validation['issue_date_utc']}: latest target {latest}, allowed through {cutoff}"
        )


def validate_forecast_timeline(frame: pd.DataFrame) -> None:
    """Fail closed unless the frozen T, T-1, issue-time and T-14..T-3 contract holds."""
    required = {
        "sample_id", "target_month", "forecast_origin_month", "issue_date_utc",
        "history_start_month", "history_end_month",
    }
    absent = required - set(frame.columns)
    if absent:
        raise ValueError(f"forecast rows lack timeline fields: {sorted(absent)}")
    if frame.empty or frame["sample_id"].duplicated().any():
        raise ValueError("forecast rows must be nonempty and have unique sample_id values")
    targets = _periods(frame["target_month"])
    origins = _periods(frame["forecast_origin_month"])
    starts = _periods(frame["history_start_month"])
    ends = _periods(frame["history_end_month"])
    expected_issue = pd.Series([f"{p - 1}-01T00:00:00Z" for p in targets], index=frame.index)
    if not np.array_equal(origins.asi8, (targets - 1).asi8):
        raise ValueError("forecast_origin_month must equal target_month minus one month")
    if not frame["issue_date_utc"].astype(str).equals(expected_issue.astype(str)):
        raise ValueError("issue_date_utc must be 00:00 UTC on the first day of target_month minus one")
    if not np.array_equal(starts.asi8, (targets - 14).asi8):
        raise ValueError("history_start_month must equal target_month minus 14 months")
    if not np.array_equal(ends.asi8, (targets - 3).asi8):
        raise ValueError("history_end_month must equal target_month minus 3 months")
    if np.any(ends.asi8 >= origins.asi8):
        raise ValueError("all predictor months must precede the forecast-origin month")


def build_history_tensor(
    samples: pd.DataFrame,
    monthly: pd.DataFrame,
    variables: Iterable[str] = HISTORY_VARIABLES,
) -> tuple[np.ndarray, list[list[str]]]:
    """Return exact T-14..T-3 native-grid regional monthly histories."""
    validate_forecast_timeline(samples)
    variable_ids = tuple(variables)
    missing_columns = set(variable_ids) - set(monthly.columns)
    if missing_columns:
        raise ValueError(f"monthly Gold is missing predictor variables: {sorted(missing_columns)}")
    month_labels = [str(x) for x in monthly.index]
    if len(set(month_labels)) != len(month_labels):
        raise ValueError("monthly Gold must have unique month keys")
    lookup = monthly.copy()
    lookup.index = month_labels
    target_periods = _periods(samples["target_month"])
    tensors: list[np.ndarray] = []
    histories: list[list[str]] = []
    for target in target_periods:
        history = [str(p) for p in pd.period_range(target - 14, target - 3, freq="M")]
        absent = set(history) - set(lookup.index)
        if absent:
            raise ValueError(f"predictor history is incomplete for {target}: {sorted(absent)}")
        values = lookup.loc[history, list(variable_ids)].to_numpy(dtype=np.float64)
        if values.shape != (12, len(variable_ids)):
            raise ValueError(f"ambiguous predictor history for target {target}")
        tensors.append(values)
        histories.append(history)
    return np.stack(tensors), histories


def _assert_contiguous(frame: pd.DataFrame, *, expected_start: str, expected_end: str) -> None:
    targets = _periods(frame["target_month"])
    expected = pd.period_range(expected_start, expected_end, freq="M")
    if not targets.equals(expected):
        raise ValueError("development targets must have complete monthly chronology")


def chronological_outer_folds(frame: pd.DataFrame) -> list[TemporalFold]:
    """Three expanding outer folds; refuses rows from either consumed test period."""
    validate_forecast_timeline(frame)
    ordered = frame.sort_values("target_month", kind="stable").reset_index(drop=True)
    targets = _periods(ordered["target_month"])
    if targets.min() < pd.Period(DEVELOPMENT_START, freq="M") or targets.max() > pd.Period(DEVELOPMENT_END, freq="M"):
        raise ValueError("outer folds accept development data only through the 2021-12 development cutoff")
    _assert_contiguous(ordered, expected_start=DEVELOPMENT_START, expected_end=DEVELOPMENT_END)
    specs = (
        ("outer_2012_2014", "2011-12", "2012-01", "2014-12"),
        ("outer_2015_2017", "2014-12", "2015-01", "2017-12"),
        ("outer_2018_2021", "2017-12", "2018-01", "2021-12"),
    )
    result: list[TemporalFold] = []
    for name, train_end, validation_start, validation_end in specs:
        train = ordered[ordered.target_month <= train_end].copy().reset_index(drop=True)
        validation = ordered[ordered.target_month.between(validation_start, validation_end)].copy().reset_index(drop=True)
        if not len(train) or not len(validation):
            raise ValueError(f"invalid chronological boundary for {name}")
        first_issue = validation.sort_values("target_month", kind="stable").iloc[0].issue_date_utc
        train = labels_available_by_issue(train, str(first_issue))
        if not len(train) or train.target_month.max() >= validation.target_month.min():
            raise ValueError(f"invalid chronological boundary for {name}")
        result.append(TemporalFold(name, train, validation))
    return result


def chronological_inner_folds(train: pd.DataFrame) -> list[TemporalFold]:
    """Three expanding, one-year validation windows ending at the outer cutoff."""
    validate_forecast_timeline(train)
    ordered = train.sort_values("target_month", kind="stable").reset_index(drop=True)
    end = pd.Period(str(ordered.target_month.max()), freq="M")
    if end.month not in {10, 12} or end.year not in {2011, 2014, 2017}:
        raise ValueError("inner folds require an outer-training cutoff at October or December 2011, 2014, or 2017")
    years = range(end.year - 2, end.year + 1)
    result: list[TemporalFold] = []
    for year in years:
        start = pd.Period(f"{year}-01", freq="M")
        stop = end if year == end.year else pd.Period(f"{year}-12", freq="M")
        train_part = ordered[_periods(ordered.target_month) < start].copy().reset_index(drop=True)
        periods = _periods(ordered.target_month)
        validation = ordered[(periods >= start) & (periods <= stop)].copy().reset_index(drop=True)
        expected_validation_months = stop.month if year == end.year else 12
        if len(validation) != expected_validation_months or not len(train_part):
            raise ValueError(f"inner validation year {year} is not fully available and chronological")
        first_issue = validation.sort_values("target_month", kind="stable").iloc[0].issue_date_utc
        train_part = labels_available_by_issue(train_part, str(first_issue))
        if not len(train_part) or train_part.target_month.max() >= validation.target_month.min():
            raise ValueError(f"inner training labels are unavailable or nonchronological for {year}")
        result.append(TemporalFold(f"inner_{year}", train_part, validation))
    return result


def monthly_gold_wide(gold: pd.DataFrame) -> pd.DataFrame:
    """Pivot validated long-form Gold values to one row per calendar month."""
    required = {"month", "variable_id", "value", "quality_status"}
    absent = required - set(gold.columns)
    if absent:
        raise ValueError(f"Gold table lacks required fields: {sorted(absent)}")
    selected = gold[gold.variable_id.isin(HISTORY_VARIABLES)].copy()
    if set(selected.variable_id.unique()) != set(HISTORY_VARIABLES):
        raise ValueError("Gold table does not contain the full contracted predictor variable set")
    selected["month"] = selected.month.astype(str)
    if selected.duplicated(["month", "variable_id"]).any():
        raise ValueError("Gold predictor month/variable keys must be unique")
    if not selected.quality_status.astype(str).eq("VALID").all():
        raise ValueError("only VALID Gold predictor values may enter the v1.4 feature matrix")
    monthly = selected.pivot(index="month", columns="variable_id", values="value").sort_index()
    if monthly.isna().any().any():
        # Missing source values remain missing and are handled by fold-local imputers.
        pass
    monthly["merra2_wind_speed"] = np.hypot(monthly.merra2_u2m, monthly.merra2_v2m)
    monthly["imerg_log_precip"] = np.log1p(monthly.imerg_precip_monthly_total.clip(lower=0))
    return monthly


def _safe_mean(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(values, axis=axis)


def _safe_sd(values: np.ndarray, axis: int) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanstd(values, axis=axis, ddof=1)


def _safe_slope(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.shape[1], dtype=float)
    output = np.full(values.shape[0], np.nan)
    for row, series in enumerate(values):
        valid = np.isfinite(series)
        if valid.sum() >= 2:
            output[row] = np.polyfit(x[valid], series[valid], 1)[0]
    return output


def _feature_history_climatology(train_histories: list[list[str]], monthly: pd.DataFrame) -> dict[str, dict[int, float]]:
    unique_months = sorted({month for history in train_histories for month in history})
    selected = monthly.loc[unique_months, list(HISTORY_VARIABLES) + ["merra2_wind_speed", "imerg_log_precip"]].copy()
    month_number = pd.PeriodIndex(selected.index, freq="M").month
    climates: dict[str, dict[int, float]] = {}
    for variable in selected.columns:
        means = selected[variable].groupby(month_number).mean()
        climates[variable] = {int(k): float(v) for k, v in means.items() if np.isfinite(v)}
    return climates


def _anomaly_series(values: np.ndarray, months: list[list[str]], variable: str, climates: dict[str, dict[int, float]]) -> np.ndarray:
    climate = climates[variable]
    output = np.asarray(values, dtype=np.float64).copy()
    for i, labels in enumerate(months):
        month_number = pd.PeriodIndex(labels, freq="M").month
        for j, month in enumerate(month_number):
            mean = climate.get(int(month))
            output[i, j] = np.nan if mean is None or not np.isfinite(output[i, j]) else output[i, j] - mean
    return output


def _raw_features(samples: pd.DataFrame, histories: np.ndarray, labels: list[list[str]], climates: dict[str, dict[int, float]]) -> pd.DataFrame:
    source_index = {name: i for i, name in enumerate(HISTORY_VARIABLES)}
    values: dict[str, np.ndarray] = {}
    anomaly: dict[str, np.ndarray] = {}
    for name in HISTORY_VARIABLES:
        anomaly[name] = _anomaly_series(histories[:, :, source_index[name]], labels, name, climates)
    wind = np.hypot(histories[:, :, source_index["merra2_u2m"]], histories[:, :, source_index["merra2_v2m"]])
    values["merra2_wind_speed"] = wind
    values["imerg_log_precip"] = np.log1p(np.clip(histories[:, :, source_index["imerg_precip_monthly_total"]], 0.0, None))
    for name, arr in values.items():
        anomaly[name] = _anomaly_series(arr, labels, name, climates)

    result: dict[str, np.ndarray] = {}
    target_month = pd.PeriodIndex(samples.target_month.astype(str), freq="M").month.to_numpy(dtype=float)
    angle = 2.0 * np.pi * (target_month - 1.0) / 12.0
    result["target_month_sin"] = np.sin(angle)
    result["target_month_cos"] = np.cos(angle)
    temp = anomaly["merra2_t2m_c"]
    for lag, offset in ((3, 1), (4, 2), (5, 3)):
        result[f"t2m_anomaly_lag_{lag:02d}"] = temp[:, -offset]
    for width in (3, 6, 12):
        result[f"t2m_anomaly_mean_{width:02d}"] = _safe_mean(temp[:, -width:], axis=1)
    result["t2m_anomaly_sd_06"] = _safe_sd(temp[:, -6:], axis=1)
    result["t2m_anomaly_slope_06"] = _safe_slope(temp[:, -6:])
    for source, label in (("merra2_qv2m", "qv"), ("merra2_wind_speed", "wind_speed"), ("merra2_ps", "ps")):
        arr = anomaly[source]
        result[f"{label}_anomaly_lag_03"] = arr[:, -1]
        result[f"{label}_anomaly_mean_03"] = _safe_mean(arr[:, -3:], axis=1)
    precip = anomaly["imerg_log_precip"]
    result["precip_log_anomaly_lag_03"] = precip[:, -1]
    result["precip_log_anomaly_mean_03"] = _safe_mean(precip[:, -3:], axis=1)
    result["precip_log_anomaly_mean_06"] = _safe_mean(precip[:, -6:], axis=1)
    for source, label in (
        ("gldas_soil_moisture_0_10cm", "soil_0_10"),
        ("gldas_soil_moisture_10_40cm", "soil_10_40"),
        ("gldas_soil_moisture_40_100cm", "soil_40_100"),
        ("gldas_soil_moisture_100_200cm", "soil_100_200"),
    ):
        result[f"{label}_anomaly_mean_03"] = _safe_mean(anomaly[source][:, -3:], axis=1)
    return pd.DataFrame(result, index=samples.index)


def _target_climatology(train: pd.DataFrame) -> dict[int, float]:
    if "target_value" not in train:
        raise ValueError("target_value is required to fit the training-only climatology")
    months = pd.PeriodIndex(train.target_month.astype(str), freq="M").month
    grouped = train.assign(_month=months).groupby("_month").target_value.mean()
    if set(grouped.index.astype(int)) != set(range(1, 13)) or not np.isfinite(grouped.to_numpy()).all():
        raise ValueError("training window must provide finite monthly climatology for all 12 calendar months")
    return {int(month): float(value) for month, value in grouped.items()}


def feature_pair(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    monthly: pd.DataFrame,
    feature_group: str,
) -> FeaturePair:
    """Fit feature anomalies, imputation and scaling on train and transform validation."""
    if feature_group not in FEATURE_GROUP_COLUMNS:
        raise ValueError(f"unknown feature group {feature_group!r}")
    monthly = monthly.copy()
    if "merra2_wind_speed" not in monthly:
        monthly["merra2_wind_speed"] = np.hypot(monthly.merra2_u2m, monthly.merra2_v2m)
    if "imerg_log_precip" not in monthly:
        monthly["imerg_log_precip"] = np.log1p(monthly.imerg_precip_monthly_total.clip(lower=0))
    for frame in (train, validation):
        validate_forecast_timeline(frame)
    validate_training_labels_available(train, validation)
    train_history, train_labels = build_history_tensor(train, monthly)
    validation_history, validation_labels = build_history_tensor(validation, monthly)
    climates = _feature_history_climatology(train_labels, monthly)
    raw_train = _raw_features(train, train_history, train_labels, climates)
    raw_validation = _raw_features(validation, validation_history, validation_labels, climates)
    columns = FEATURE_GROUP_COLUMNS[feature_group]
    x_train_raw = raw_train.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    x_validation_raw = raw_validation.loc[:, columns].to_numpy(dtype=np.float64, copy=True)
    missing_train = ~np.isfinite(x_train_raw)
    missing_validation = ~np.isfinite(x_validation_raw)
    means = np.nanmean(np.where(missing_train, np.nan, x_train_raw), axis=0)
    if not np.isfinite(means).all():
        raise ValueError(f"feature group {feature_group} has a fully missing training feature")
    x_train_raw[missing_train] = np.broadcast_to(means, x_train_raw.shape)[missing_train]
    x_validation_raw[missing_validation] = np.broadcast_to(means, x_validation_raw.shape)[missing_validation]
    scaler = StandardScaler().fit(x_train_raw)
    x_train = np.concatenate([scaler.transform(x_train_raw), missing_train.astype(float)], axis=1)
    x_validation = np.concatenate([scaler.transform(x_validation_raw), missing_validation.astype(float)], axis=1)

    target_climate = _target_climatology(train)
    train_months = pd.PeriodIndex(train.target_month.astype(str), freq="M").month
    validation_months = pd.PeriodIndex(validation.target_month.astype(str), freq="M").month
    offset_train = np.asarray([target_climate[int(month)] for month in train_months], dtype=np.float64)
    offset_validation = np.asarray([target_climate[int(month)] for month in validation_months], dtype=np.float64)
    y_train = train.target_value.to_numpy(dtype=np.float64)
    y_validation = validation.target_value.to_numpy(dtype=np.float64)
    return FeaturePair(
        x_train=x_train,
        x_validation=x_validation,
        y_train_residual=y_train - offset_train,
        y_validation=y_validation,
        offset_train=offset_train,
        offset_validation=offset_validation,
        feature_names=tuple(columns) + tuple(f"missing__{column}" for column in columns),
        training_climatology=target_climate,
        feature_climatology=climates,
        imputation_means=means.copy(),
        scaler_mean=scaler.mean_.copy(),
        scaler_scale=scaler.scale_.copy(),
    )


def shrink_predictions(climatology: np.ndarray, residual_prediction: np.ndarray, alpha: float) -> np.ndarray:
    """Return climatology + alpha * residual prediction with a fixed bounded shrinkage."""
    if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha must be finite and constrained to [0, 1]")
    return np.asarray(climatology, dtype=float) + alpha * np.asarray(residual_prediction, dtype=float)


def select_shrinkage_alpha(
    observations: np.ndarray,
    climatology: np.ndarray,
    residual_prediction: np.ndarray,
    candidates: Iterable[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Choose alpha by validation MAE, with the climatology-only model in the grid."""
    y = np.asarray(observations, dtype=float)
    climate = np.asarray(climatology, dtype=float)
    residual = np.asarray(residual_prediction, dtype=float)
    if y.shape != climate.shape or y.shape != residual.shape or not y.size:
        raise ValueError("observations, climatology and residual predictions must be aligned and nonempty")
    if not (np.isfinite(y).all() and np.isfinite(climate).all() and np.isfinite(residual).all()):
        raise ValueError("alpha selection requires finite validation values")
    options = sorted(set(float(x) for x in candidates))
    if not options or options[0] < 0.0 or options[-1] > 1.0 or not np.isfinite(options).all():
        raise ValueError("alpha candidates must be finite values in [0, 1]")
    scored: list[tuple[float, float, float, np.ndarray]] = []
    for alpha in options:
        prediction = shrink_predictions(climate, residual, alpha)
        error = prediction - y
        scored.append((float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error**2))), alpha, prediction))
    mae, rmse, alpha, prediction = min(scored, key=lambda row: (row[0], row[1], row[2]))
    return alpha, prediction, {"mae": mae, "rmse": rmse}


def expanding_monthly_climatology(history: pd.DataFrame, future: pd.DataFrame) -> np.ndarray:
    """Prequential same-calendar-month means using only observations before each target."""
    required = {"target_month", "target_value"}
    for name, frame in (("history", history), ("future", future)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} requires target_month and target_value")
    if future.target_month.duplicated().any():
        raise ValueError("future targets must be unique")
    observed: list[tuple[pd.Period, float]] = []
    for row in history.itertuples(index=False):
        period = pd.Period(str(row.target_month), freq="M")
        value = float(row.target_value)
        if np.isfinite(value):
            observed.append((period, value))
    ordered = future.copy()
    ordered["_period"] = pd.PeriodIndex(ordered.target_month.astype(str), freq="M")
    ordered = ordered.sort_values("_period", kind="stable")
    predictions: dict[object, float] = {}
    for index, row in ordered.iterrows():
        period = row["_period"]
        prior = [value for old_period, value in observed if old_period < period and old_period.month == period.month]
        if not prior:
            raise ValueError(f"no earlier same-month observations are available for {period}")
        predictions[index] = float(np.mean(prior))
        actual = float(row["target_value"])
        if np.isfinite(actual):
            observed.append((period, actual))
    return np.asarray([predictions[index] for index in future.index], dtype=float)


def metric_summary(observations: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    """Paired forecast metrics on identical finite target months."""
    y = np.asarray(observations, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    reference = np.asarray(baseline, dtype=float)
    if y.shape != pred.shape or y.shape != reference.shape:
        raise ValueError("observations, candidate predictions and baseline must have the same shape")
    valid = np.isfinite(y) & np.isfinite(pred) & np.isfinite(reference)
    invalid = int((~np.isfinite(pred)).sum())
    if not valid.any():
        raise ValueError("no paired finite target months are available")
    y, pred, reference = y[valid], pred[valid], reference[valid]
    error = pred - y
    baseline_error = reference - y
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_error**2)))
    return {
        "sample_count": int(len(y)),
        "invalid_prediction_count": invalid,
        "mae": mae,
        "rmse": rmse,
        "mean_bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(np.abs(error))),
        "p90_absolute_error": float(np.quantile(np.abs(error), 0.90)),
        "monthly_win_rate_vs_baseline": float(np.mean(np.abs(error) < np.abs(baseline_error))),
        "mae_skill_vs_baseline": float(1.0 - mae / baseline_mae) if baseline_mae > 0 else float("nan"),
        "rmse_skill_vs_baseline": float(1.0 - rmse / baseline_rmse) if baseline_rmse > 0 else float("nan"),
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
    }


def paired_moving_block_ci(
    paired_difference: np.ndarray,
    *,
    replicates: int = 2000,
    seed: int = 20261117,
    block_length: int = 12,
) -> dict[str, object]:
    """Descriptive non-circular moving-block bootstrap for paired monthly loss changes."""
    values = np.asarray(paired_difference, dtype=float)
    values = values[np.isfinite(values)]
    if values.ndim != 1 or len(values) < 1 or replicates < 1 or block_length < 1:
        raise ValueError("bootstrap requires finite paired differences and positive settings")
    if len(values) < block_length:
        mean = float(np.mean(values))
        return {
            "mean_difference": mean,
            "ci95": [mean, mean],
            "sample_count": int(len(values)),
            "block_length_months": int(block_length),
            "replicates": int(replicates),
            "seed": int(seed),
            "interpretation": "degenerate: fewer observations than one declared moving block",
        }
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    starts = np.arange(len(values) - block_length + 1)
    for index in range(replicates):
        selected: list[int] = []
        while len(selected) < len(values):
            start = int(rng.choice(starts))
            selected.extend(range(start, start + block_length))
        draws[index] = float(np.mean(values[np.asarray(selected[: len(values)], dtype=int)]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean_difference": float(np.mean(values)),
        "ci95": [float(low), float(high)],
        "sample_count": int(len(values)),
        "block_length_months": int(block_length),
        "replicates": int(replicates),
        "seed": int(seed),
        "interpretation": "descriptive; serial dependence and model selection limit inferential claims",
    }


def augment_training_rows(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Iterable[str],
    *,
    sample_ids: Iterable[str],
    policy: str,
    intensity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add one training-only, label-preserving perturbation per original with total weight one."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    names = tuple(feature_names)
    ids = tuple(str(value) for value in sample_ids)
    if x.ndim != 2 or y.shape != (len(x),) or len(ids) != len(x) or len(names) != x.shape[1]:
        raise ValueError("features, labels, names and sample IDs must be aligned")
    if policy not in {"clean", "mild_gaussian", "mild_dropout"}:
        raise ValueError(f"unknown augmentation policy {policy!r}")
    if not np.isfinite(intensity) or intensity < 0 or intensity > 0.5:
        raise ValueError("augmentation intensity must be finite and between 0 and 0.5")
    if policy == "clean":
        return x.copy(), y.copy(), np.ones(len(y), dtype=float)
    if x.shape[1] % 2:
        raise ValueError("feature matrix must contain values followed by one missingness flag per feature")
    feature_count = x.shape[1] // 2
    value_names = names[:feature_count]
    dynamic = [i for i, name in enumerate(value_names) if not name.startswith("target_month_")]
    if not dynamic:
        raise ValueError("augmentation requires at least one nonseasonal feature")
    perturbed = x.copy()
    for row, sample_id in enumerate(ids):
        digest = hashlib.sha256(f"{seed}|{sample_id}|{policy}|{intensity:.6f}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:4], "big"))
        if policy == "mild_gaussian":
            perturbed[row, dynamic] += rng.normal(0.0, intensity, size=len(dynamic))
        else:
            count = max(1, int(round(intensity * len(dynamic)))) if intensity > 0 else 0
            chosen = rng.choice(dynamic, size=min(count, len(dynamic)), replace=False) if count else []
            for feature_index in chosen:
                perturbed[row, feature_index] = 0.0
                perturbed[row, feature_count + feature_index] = 1.0
    augmented_x = np.empty_like(np.repeat(x, 2, axis=0))
    augmented_y = np.empty_like(np.repeat(y, 2))
    for row in range(len(y)):
        augmented_x[2 * row] = x[row]
        augmented_x[2 * row + 1] = perturbed[row]
        augmented_y[2 * row : 2 * row + 2] = y[row]
    weights = np.full(len(augmented_y), 0.5, dtype=float)
    if not np.array_equal(augmented_y[0::2], y) or not np.array_equal(augmented_y[1::2], y):
        raise AssertionError("augmentation changed an authentic target label")
    if not np.allclose(weights.reshape(len(y), 2).sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise AssertionError("augmentation changed authentic target-month weight")
    return augmented_x, augmented_y, weights


def save_fold_checkpoint(path: Path, payload: dict, *, fingerprint: str) -> None:
    """Atomically persist one completed outer fold for safe resume."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump({"fingerprint": fingerprint, "payload": payload}, temp, compress=3)
    temp.replace(path)


def load_fold_checkpoint(path: Path, *, fingerprint: str) -> dict | None:
    """Return a checkpoint only when its input/code fingerprint matches."""
    path = Path(path)
    if not path.is_file():
        return None
    try:
        saved = joblib.load(path)
    except Exception:
        return None
    if not isinstance(saved, dict) or saved.get("fingerprint") != fingerprint:
        return None
    payload = saved.get("payload")
    return payload if isinstance(payload, dict) else None

