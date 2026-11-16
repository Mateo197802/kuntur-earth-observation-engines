"""Fold-local feature groups for finite v1.2 forecasting experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .augmentation import CHANNELS, history_array


FEATURE_GROUPS = {
    "A_temperature_lags": "T-14..T-3 MERRA-2 temperature lags only",
    "B_temperature_seasonal": "Temperature lags plus known target-month Fourier harmonics",
    "C_training_anomalies": "Temperature lag anomalies and target residuals using training-only monthly climatology",
    "D_past_rolling_temperature": "Temperature lags plus past-only 3/6/12-month means, variability and coverage",
    "E_full_meteorology": "All five MERRA-2 channels over the frozen T-14..T-3 history",
    "F_climatology_residual": "Raw temperature plus training-only target-month climatology and known harmonics",
    "G_stable_channels": "Temperature plus channels selected only in expanding subwindows of outer training data",
    "H_physical_combination": "Full meteorology plus known seasonality and past-only temperature summaries",
}


@dataclass(frozen=True)
class FeaturePair:
    x_train: np.ndarray
    x_validation: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    offset_train: np.ndarray
    offset_validation: np.ndarray
    sequence_channels: int
    static_features: int
    selected_channels: tuple[str, ...]
    feature_columns: tuple[str, ...]


def _target_month_number(frame: pd.DataFrame) -> np.ndarray:
    return pd.PeriodIndex(frame["target_month"].astype(str), freq="M").month.to_numpy(dtype=int)


def _season(month: np.ndarray) -> np.ndarray:
    angle = 2.0 * np.pi * (np.asarray(month, dtype=float) - 1.0) / 12.0
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _rolling_temperature(raw: np.ndarray) -> tuple[np.ndarray, list[str]]:
    temperature = raw[:, :, 0]
    values, names = [], []
    for width in (3, 6, 12):
        trailing = temperature[:, -width:]
        counts = np.isfinite(trailing).sum(axis=1)
        means = np.divide(np.nansum(trailing, axis=1), counts, out=np.full(len(raw), np.nan), where=counts > 0)
        centered = trailing - means[:, None]
        variance = np.divide(np.nansum(centered**2, axis=1), counts, out=np.full(len(raw), np.nan), where=counts > 0)
        values.extend([means, np.sqrt(variance), 1.0 - counts / width])
        names.extend([f"t2m_trailing_mean_{width}", f"t2m_trailing_sd_{width}", f"t2m_trailing_missing_fraction_{width}"])
    return np.column_stack(values), names


def _climatology(train: pd.DataFrame) -> dict[int, float]:
    month = _target_month_number(train)
    values = train.target_value.to_numpy(dtype=float)
    return {int(m): float(np.mean(values[month == m])) for m in range(1, 13) if np.isfinite(values[month == m]).any()}


def _stable_channel_selection(train: pd.DataFrame) -> tuple[int, ...]:
    """Rank channel blocks using expanding, training-only windows; always retain T2M."""
    raw = history_array(train)
    target = train.target_value.to_numpy(dtype=float)
    n = len(train)
    starts = sorted(set([max(48, int(n * 0.60)), max(48, int(n * 0.80)), n]))
    votes = np.zeros(len(CHANNELS), dtype=int)
    for end in starts:
        if end < 36:
            continue
        x = raw[:end].reshape(end, -1).copy()
        means = np.nanmean(x, axis=0)
        means = np.where(np.isfinite(means), means, 0.0)
        missing = ~np.isfinite(x)
        x[missing] = np.broadcast_to(means, x.shape)[missing]
        scaler = StandardScaler().fit(x)
        model = Ridge(alpha=10.0).fit(scaler.transform(x), target[:end])
        importance = np.abs(model.coef_).reshape(12, len(CHANNELS)).sum(axis=0)
        # Exclude T2M from the contest; it is retained by the physical contract.
        other = [i for i in range(len(CHANNELS)) if i != 0]
        top = sorted(other, key=lambda i: (-float(importance[i]), i))[:2]
        votes[top] += 1
    other = [i for i in range(len(CHANNELS)) if i != 0]
    selected = [i for i in other if votes[i] >= max(1, len(starts) - 1)]
    if len(selected) < 2:
        selected = sorted(other, key=lambda i: (-int(votes[i]), i))[:2]
    return tuple(sorted({0, *selected}))


def _raw_block(frame: pd.DataFrame, group: str, climate: dict[int, float], selected_channels: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    raw = history_array(frame)
    months = _target_month_number(frame)
    season = _season(months)
    # History columns are ordered T-14 through T-3, so lag offsets are exact.
    lag_offsets = np.arange(14, 2, -1, dtype=int)
    predictor_months = ((months[:, None] - lag_offsets[None, :] - 1) % 12) + 1
    column_names: list[str] = []
    static_parts: list[np.ndarray] = []
    static_names: list[str] = []

    if group in {"A_temperature_lags", "B_temperature_seasonal", "C_training_anomalies", "D_past_rolling_temperature"}:
        sequence = raw[:, :, [0]]
        channel_names = [CHANNELS[0]]
    elif group in {"E_full_meteorology", "F_climatology_residual", "H_physical_combination"}:
        sequence = raw[:, :, :]
        channel_names = list(CHANNELS)
    elif group == "G_stable_channels":
        sequence = raw[:, :, list(selected_channels)]
        channel_names = [CHANNELS[i] for i in selected_channels]
    else:
        raise ValueError(f"Unknown v1.2 feature group: {group}")

    if group in {"C_training_anomalies", "F_climatology_residual"}:
        target_climate = np.asarray([climate[int(m)] for m in months], dtype=float)
        if group == "C_training_anomalies":
            lag_climate = np.vectorize(lambda m: climate[int(m)])(predictor_months)
            sequence[:, :, 0] = sequence[:, :, 0] - lag_climate
            static_parts.append(season)
            static_names.extend(["target_month_sin", "target_month_cos"])
        else:
            static_parts.extend([target_climate[:, None], season])
            static_names.extend(["training_monthly_climatology", "target_month_sin", "target_month_cos"])
    if group in {"B_temperature_seasonal", "H_physical_combination"}:
        static_parts.append(season)
        static_names.extend(["target_month_sin", "target_month_cos"])
    if group in {"D_past_rolling_temperature", "H_physical_combination"}:
        rolling, rolling_names = _rolling_temperature(raw)
        static_parts.append(rolling)
        static_names.extend(rolling_names)
    static = np.column_stack(static_parts) if static_parts else np.empty((len(frame), 0), dtype=float)
    for lag in range(14, 2, -1):
        column_names.extend([f"{channel}_lag_{lag:02d}" for channel in channel_names])
    column_names.extend(static_names)
    return sequence, static, column_names


def _weighted_nanmean(values: np.ndarray, weights: np.ndarray | None, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if weights is None:
        return np.nanmean(values, axis=axis)
    if axis != 0 or values.shape[0] != len(weights):
        raise ValueError("Weighted feature statistics expect samples on axis zero")
    expanded = np.asarray(weights, dtype=float).reshape((-1,) + (1,) * (values.ndim - 1))
    valid = np.isfinite(values)
    numerator = np.nansum(values * expanded, axis=0)
    denominator = np.sum(valid * expanded, axis=0)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float), where=denominator > 0)


def build_feature_pair(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    group: str,
    *,
    climate_source: pd.DataFrame | None = None,
    selection_source: pd.DataFrame | None = None,
    sample_weight: np.ndarray | None = None,
    include_missing_indicators: bool = True,
) -> FeaturePair:
    """Fit all learned transforms inside ``train`` and transform its later validation block."""
    climate = _climatology(climate_source if climate_source is not None else train)
    selected_channels = _stable_channel_selection(selection_source if selection_source is not None else train) if group == "G_stable_channels" else (0, 1, 2, 3, 4)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)
    if weights is not None and (weights.shape != (len(train),) or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("Training sample weights must be nonnegative, aligned, and have positive sum")
    tr_seq, tr_static, names = _raw_block(train, group, climate, selected_channels)
    va_seq, va_static, _ = _raw_block(validation, group, climate, selected_channels)
    tr_missing, va_missing = ~np.isfinite(tr_seq), ~np.isfinite(va_seq)
    seq_means = _weighted_nanmean(tr_seq, weights, axis=0)
    if not np.isfinite(seq_means).all():
        raise ValueError(f"Training feature group {group} contains an all-missing sequence cell")
    tr_filled, va_filled = tr_seq.copy(), va_seq.copy()
    tr_fill = np.broadcast_to(seq_means[None, :, :], tr_filled.shape)
    va_fill = np.broadcast_to(seq_means[None, :, :], va_filled.shape)
    tr_filled[tr_missing] = tr_fill[tr_missing]
    va_filled[va_missing] = va_fill[va_missing]
    scaler = StandardScaler().fit(tr_filled.reshape(len(tr_filled), -1), sample_weight=weights)
    tr_scaled = scaler.transform(tr_filled.reshape(len(tr_filled), -1)).reshape(tr_filled.shape)
    va_scaled = scaler.transform(va_filled.reshape(len(va_filled), -1)).reshape(va_filled.shape)

    tr_static_missing, va_static_missing = ~np.isfinite(tr_static), ~np.isfinite(va_static)
    if tr_static.shape[1]:
        static_means = _weighted_nanmean(tr_static, weights, axis=0)
        if not np.isfinite(static_means).all():
            raise ValueError(f"Training feature group {group} contains an all-missing static feature")
        tr_static_fill, va_static_fill = tr_static.copy(), va_static.copy()
        tr_sf = np.broadcast_to(static_means[None, :], tr_static.shape)
        va_sf = np.broadcast_to(static_means[None, :], va_static.shape)
        tr_static_fill[tr_static_missing] = tr_sf[tr_static_missing]
        va_static_fill[va_static_missing] = va_sf[va_static_missing]
        static_scaler = StandardScaler().fit(tr_static_fill, sample_weight=weights)
        tr_static_scaled = static_scaler.transform(tr_static_fill)
        va_static_scaled = static_scaler.transform(va_static_fill)
        tr_static_all = np.concatenate([tr_static_scaled, tr_static_missing.astype(float)], axis=1)
        va_static_all = np.concatenate([va_static_scaled, va_static_missing.astype(float)], axis=1)
    else:
        tr_static_all, va_static_all = np.empty((len(train), 0)), np.empty((len(validation), 0))

    n_seq = int(np.prod(tr_scaled.shape[1:]))
    train_parts = [tr_scaled.reshape(len(train), -1)]
    validation_parts = [va_scaled.reshape(len(validation), -1)]
    if include_missing_indicators:
        train_parts.append(tr_missing.reshape(len(train), -1).astype(float))
        validation_parts.append(va_missing.reshape(len(validation), -1).astype(float))
    train_parts.append(tr_static_all)
    validation_parts.append(va_static_all)
    x_train = np.concatenate(train_parts, axis=1)
    x_validation = np.concatenate(validation_parts, axis=1)
    y_train, y_validation = train.target_value.to_numpy(dtype=float), validation.target_value.to_numpy(dtype=float)
    offset_train, offset_validation = np.zeros(len(train)), np.zeros(len(validation))
    if group == "C_training_anomalies":
        offset_train = np.asarray([climate[int(m)] for m in _target_month_number(train)], dtype=float)
        offset_validation = np.asarray([climate[int(m)] for m in _target_month_number(validation)], dtype=float)
        y_train, y_validation = y_train - offset_train, y_validation - offset_validation
    return FeaturePair(
        x_train=x_train, x_validation=x_validation, y_train=y_train, y_validation=y_validation,
        offset_train=offset_train, offset_validation=offset_validation,
        sequence_channels=int(tr_seq.shape[2]), static_features=int(tr_static_all.shape[1]),
        selected_channels=tuple(CHANNELS[i] for i in (selected_channels if group == "G_stable_channels" else range(tr_seq.shape[2]))),
        feature_columns=tuple(names),
    )
