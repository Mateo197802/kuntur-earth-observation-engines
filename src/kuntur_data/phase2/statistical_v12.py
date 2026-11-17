"""Versioned, validity-gated monthly trend inference for KUNTUR v1.2.

The v1 analytical implementation remains the historical reference. This
module preserves calendar time, rejects nonfinite inputs, exposes candidate
dependence-aware estimators, and withholds inferential output until a method
is validated for a declared error regime.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan


STATISTICAL_VERSION = "kuntur-statistical-inference-v1.2.0"
Z975 = float(norm.ppf(0.975))
METHOD_IDS = {
    "ols_hac": "OLS_HAC_FROZEN_REFERENCE",
    "ols_hac_calendar": "OLS_HAC_CALENDAR_DISTANCE",
    "gls_ar1": "FEGLS_AR1_ITERATED",
    "sarimax_ar1": "SARIMAX_AR1",
    "arma11": "SARIMAX_ARMA11",
    "seasonal_ar12": "SARIMAX_SEASONAL_AR12",
    "moving_block_bootstrap": "MOVING_RESIDUAL_BLOCK_BOOTSTRAP",
    "seasonal_block_bootstrap": "SEASONAL_YEAR_BLOCK_BOOTSTRAP",
    "seasonal_mk_block": "SEASONAL_MK_YEAR_BLOCK_BOOTSTRAP",
}


@dataclass(frozen=True)
class _Fit:
    slope: float | None
    standard_error: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    dependence_parameters: dict[str, float] | None = None
    residuals: np.ndarray | None = None
    converged: bool = True
    note: str | None = None
    warnings: tuple[str, ...] = ()


def _periods(values: pd.Series) -> pd.PeriodIndex:
    text = values.astype(str).str.slice(0, 7)
    return pd.PeriodIndex(text, freq="M")


def _design(periods: pd.PeriodIndex) -> np.ndarray:
    month_ordinal = periods.year.to_numpy(dtype=float) * 12 + periods.month.to_numpy(dtype=float) - 1
    elapsed_decades = (month_ordinal - month_ordinal[0]) / 120.0
    month = periods.month.to_numpy(dtype=int)
    dummies = np.eye(12, dtype=float)[month - 1, 1:]
    return np.column_stack((np.ones(len(periods)), elapsed_decades, dummies))


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observed = np.isfinite(y)
    if observed.sum() <= x.shape[1]:
        raise ValueError("insufficient degrees of freedom")
    beta = np.linalg.lstsq(x[observed], y[observed], rcond=None)[0]
    residual = np.full(len(y), np.nan, dtype=float)
    residual[observed] = y[observed] - x[observed] @ beta
    covariance = np.linalg.pinv(x[observed].T @ x[observed])
    return beta, residual, covariance


def _calendar_hac(y: np.ndarray, x: np.ndarray, ordinals: np.ndarray, max_lag: int) -> tuple[float, float, float, float, np.ndarray]:
    observed = np.isfinite(y)
    xo, yo, to = x[observed], y[observed], ordinals[observed]
    beta = np.linalg.lstsq(xo, yo, rcond=None)[0]
    residual = yo - xo @ beta
    scores = xo * residual[:, None]
    meat = scores.T @ scores
    for i in range(len(to)):
        for j in range(i + 1, len(to)):
            distance = int(to[j] - to[i])
            if distance > max_lag:
                break
            weight = 1.0 - distance / float(max_lag + 1)
            cross = np.outer(scores[j], scores[i])
            meat += weight * (cross + cross.T)
    bread = np.linalg.pinv(xo.T @ xo)
    cov = (len(to) / max(1, len(to) - x.shape[1])) * (bread @ meat @ bread)
    se = float(np.sqrt(max(0.0, cov[1, 1])))
    return float(beta[1]), se, float(beta[1] - Z975 * se), float(beta[1] + Z975 * se), np.where(observed, y - x @ beta, np.nan)


def calendar_lag_acf(series: pd.Series, *, max_lag_months: int = 24) -> dict[str, dict[str, float | int | None]]:
    """Compute pairwise ACF at exact calendar offsets without compressing gaps."""
    if max_lag_months < 1:
        raise ValueError("max_lag_months must be positive")
    if isinstance(series.index, pd.PeriodIndex):
        periods = series.index.asfreq("M")
    else:
        periods = pd.PeriodIndex(pd.to_datetime(series.index), freq="M")
    if periods.has_duplicates:
        raise ValueError("duplicate calendar month in residual index")
    ordinals = periods.year.to_numpy(dtype=int) * 12 + periods.month.to_numpy(dtype=int) - 1
    values = np.asarray(series, dtype=float)
    lookup = {int(month): float(value) for month, value in zip(ordinals, values) if np.isfinite(value)}
    result: dict[str, dict[str, float | int | None]] = {}
    for lag in range(1, max_lag_months + 1):
        left, right = [], []
        for month, value in lookup.items():
            other = lookup.get(month + lag)
            if other is not None:
                left.append(value)
                right.append(other)
        acf: float | None = None
        if len(left) >= 3 and np.std(left) > 0 and np.std(right) > 0:
            acf = float(np.corrcoef(left, right)[0, 1])
        result[str(lag)] = {"acf": acf, "pair_count": len(left)}
    return result


def benjamini_yekutieli_v12(p_values: list[float]) -> list[float]:
    """Adjust a fixed hypothesis family under arbitrary dependence.

    This controls FDR only when the supplied marginal p-values are valid. It
    does not calibrate an invalid trend test and is intentionally not applied
    to the frozen v1 exploratory family.
    """
    values = np.asarray(p_values, dtype=float)
    finite = np.isfinite(values)
    if np.any((values[finite] < 0.0) | (values[finite] > 1.0)):
        raise ValueError("finite p-values must be within [0, 1]")
    adjusted = np.full(values.shape, np.nan, dtype=float)
    p = values[finite]
    m = len(p)
    if m == 0:
        return [float("nan") for _ in values]
    order = np.argsort(p, kind="stable")
    ranks = np.arange(1, m + 1, dtype=float)
    harmonic = float(np.sum(1.0 / ranks))
    sorted_adjusted = p[order] * m * harmonic / ranks
    sorted_adjusted = np.minimum.accumulate(sorted_adjusted[::-1])[::-1]
    sorted_adjusted = np.clip(sorted_adjusted, 0.0, 1.0)
    restored = np.empty(m, dtype=float)
    restored[order] = sorted_adjusted
    adjusted[finite] = restored
    return [float(value) if np.isfinite(value) else float("nan") for value in adjusted]


def _validated_registry_entry(
    registry: dict[str, Any] | None, declared_regime: str | None, method: str,
    n_calendar_months: int, calendar_coverage: float,
) -> dict[str, Any] | None:
    if not registry or not declared_regime or registry.get("status") != "FROZEN":
        return None
    record = registry.get("regimes", {}).get(str(declared_regime))
    if not isinstance(record, dict) or record.get("validity_status") != "VALIDATED_FOR_DECLARED_REGIME":
        return None
    if record.get("method_id") != METHOD_IDS[method]:
        return None
    if not int(record.get("calendar_months_min", -1)) <= n_calendar_months <= int(record.get("calendar_months_max", -1)):
        return None
    if not float(record.get("calendar_coverage_min", -1.0)) <= calendar_coverage <= float(record.get("calendar_coverage_max", -1.0)):
        return None
    return record


def _seasonal_mk_s(y: np.ndarray, periods: pd.PeriodIndex) -> float:
    total = 0.0
    month = periods.month.to_numpy(dtype=int)
    year_ord = periods.year.to_numpy(dtype=int)
    for m in range(1, 13):
        idx = np.flatnonzero((month == m) & np.isfinite(y))
        idx = idx[np.argsort(year_ord[idx])]
        vals = y[idx]
        if len(vals) > 1:
            diff = vals[None, :] - vals[:, None]
            total += float(np.sign(diff[np.triu_indices(len(vals), k=1)]).sum())
    return total


def _seasonal_sen(y: np.ndarray, periods: pd.PeriodIndex) -> float | None:
    month = periods.month.to_numpy(dtype=int)
    ordinals = periods.year.to_numpy(dtype=int) * 12 + month - 1
    slopes: list[float] = []
    for m in range(1, 13):
        idx = np.flatnonzero((month == m) & np.isfinite(y))
        if len(idx) < 3:
            continue
        idx = idx[np.argsort(ordinals[idx])]
        i, j = np.triu_indices(len(idx), k=1)
        slopes.extend(((y[idx[j]] - y[idx[i]]) / ((ordinals[idx[j]] - ordinals[idx[i]]) / 120.0)).tolist())
    return float(np.median(slopes)) if slopes else None


def _residual_diagnostics(residual: np.ndarray, periods: pd.PeriodIndex, x: np.ndarray) -> dict[str, Any]:
    series = pd.Series(residual, index=periods)
    acf = calendar_lag_acf(series, max_lag_months=min(24, max(1, len(series) - 1)))
    observed = np.isfinite(residual)
    r = residual[observed]
    diag: dict[str, Any] = {
        "acf_calendar_months": {"1": acf.get("1", {}).get("acf"), "12": acf.get("12", {}).get("acf")},
        "acf_pair_counts": {k: v["pair_count"] for k, v in acf.items()},
        "acf_calendar_lags": acf,
        "ljung_box": None,
        "durbin_watson": None,
        "breusch_pagan": None,
        "jarque_bera": None,
        "monthly_mean_residual": {},
    }
    if len(r) >= 8:
        diag["jarque_bera"] = {"statistic": float(jarque_bera(r).statistic), "p_value": float(jarque_bera(r).pvalue)}
    if observed.all() and len(r) > x.shape[1] + 24:
        lb = acorr_ljungbox(r, lags=[1, 12, 24], return_df=True)
        diag["ljung_box"] = {str(k): {"statistic": float(lb.loc[k, "lb_stat"]), "p_value": float(lb.loc[k, "lb_pvalue"])} for k in lb.index}
        diag["durbin_watson"] = float(sm.stats.stattools.durbin_watson(r))
        try:
            bp = het_breuschpagan(r, x)
            diag["breusch_pagan"] = {"lm_statistic": float(bp[0]), "p_value": float(bp[1])}
        except (ValueError, np.linalg.LinAlgError):
            diag["breusch_pagan"] = {"status": "UNAVAILABLE"}
    for m in range(1, 13):
        vals = residual[(periods.month == m) & observed]
        diag["monthly_mean_residual"][str(m)] = float(np.mean(vals)) if len(vals) else None
    return diag


def _sample_circular_blocks(values: np.ndarray, rng: np.random.Generator, block_length: int, out_length: int) -> np.ndarray:
    if block_length < 1 or block_length > len(values):
        raise ValueError("block_length must be between 1 and the series length")
    n_blocks = int(np.ceil(out_length / block_length))
    starts = rng.integers(0, len(values), size=n_blocks)
    indexes = (starts[:, None] + np.arange(block_length)[None, :]) % len(values)
    return values[indexes.reshape(-1)[:out_length]]


def _circular_block_indices(n: int, draws: int, block_length: int, out_length: int, rng: np.random.Generator) -> np.ndarray:
    if n < 1 or block_length < 1 or block_length > n:
        raise ValueError("block_length must be between 1 and the number of available blocks")
    n_blocks = int(np.ceil(out_length / block_length))
    starts = rng.integers(0, n, size=(draws, n_blocks))
    indexes = (starts[:, :, None] + np.arange(block_length)[None, None, :]) % n
    return indexes.reshape(draws, -1)[:, :out_length]


def _bootstrap_fit(y: np.ndarray, x: np.ndarray, periods: pd.PeriodIndex, *, seasonal: bool, n_bootstrap: int, seed: int, block_length: int) -> _Fit:
    if not np.isfinite(y).all():
        return _Fit(None, converged=False, note="Residual block bootstrap currently requires a complete monthly series.")
    beta, residual, _ = _ols(y, x)
    slope = float(beta[1])
    rng = np.random.default_rng(seed)
    if seasonal:
        years = len(y) // 12
        if len(y) % 12 or years < block_length or periods[0].month != 1 or periods[-1].month != 12:
            return _Fit(None, converged=False, note="Seasonal year-block bootstrap requires complete January-to-December blocks.")
        resid_matrix = residual.reshape(years, 12)
        null_x = np.delete(x, 1, axis=1)
        null_beta = np.linalg.lstsq(null_x, y, rcond=None)[0]
        null_resid = y - null_x @ null_beta
        null_matrix = null_resid.reshape(years, 12)
        year_indexes = _circular_block_indices(years, n_bootstrap, block_length, years, rng)
        boot_resid = resid_matrix[year_indexes].reshape(n_bootstrap, len(y))
        boot_null = null_matrix[year_indexes].reshape(n_bootstrap, len(y))
    else:
        null_x = np.delete(x, 1, axis=1)
        null_beta = np.linalg.lstsq(null_x, y, rcond=None)[0]
        null_resid = y - null_x @ null_beta
        indexes = _circular_block_indices(len(y), n_bootstrap, block_length, len(y), rng)
        boot_resid = residual[indexes]
        boot_null = null_resid[indexes]
    slope_weights = np.linalg.pinv(x.T @ x) @ x.T
    ci_slopes = slope + boot_resid @ slope_weights[1]
    null_slopes = ((null_x @ null_beta)[None, :] + boot_null) @ slope_weights[1]
    low, high = np.quantile(ci_slopes, [0.025, 0.975])
    p = (1.0 + int(np.sum(np.abs(null_slopes) >= abs(slope)))) / (n_bootstrap + 1.0)
    return _Fit(slope, float(np.std(ci_slopes, ddof=1)), float(low), float(high), float(p), {"block_length_months": float(block_length)}, residual, True)


def _seasonal_mk_s_many(matrix: np.ndarray) -> np.ndarray:
    """Vectorized seasonal-MK S for draws x years x calendar-month arrays."""
    draws, years, _ = matrix.shape
    total = np.zeros(draws, dtype=float)
    for month in range(12):
        values = matrix[:, :, month]
        for left in range(years - 1):
            for right in range(left + 1, years):
                a, b = values[:, left], values[:, right]
                valid = np.isfinite(a) & np.isfinite(b)
                total += np.where(valid, np.sign(b - a), 0.0)
    return total


def _seasonal_mk_block_p(y: np.ndarray, periods: pd.PeriodIndex, *, n_bootstrap: int, seed: int, block_years: int) -> tuple[float, float, bool]:
    if block_years < 1:
        raise ValueError("block_years must be positive")
    first = pd.Period(f"{periods[0].year}-01", freq="M")
    last = pd.Period(f"{periods[-1].year}-12", freq="M")
    full = pd.period_range(first, last, freq="M")
    series = pd.Series(y, index=periods).reindex(full).to_numpy(dtype=float)
    years = len(full) // 12
    if years < max(3, block_years + 1):
        return _seasonal_mk_s(series, full), float("nan"), False
    matrix = series.reshape(years, 12)
    monthly_mean = np.nanmean(matrix, axis=0)
    if not np.isfinite(monthly_mean).all():
        return _seasonal_mk_s(series, full), float("nan"), False
    try:
        _beta, detrended_residual, _covariance = _ols(series, _design(full))
    except ValueError:
        return _seasonal_mk_s(series, full), float("nan"), False
    residual = detrended_residual.reshape(years, 12)
    observed_s = _seasonal_mk_s(series, full)
    rng = np.random.default_rng(seed)
    year_indices = _circular_block_indices(years, n_bootstrap, block_years, years, rng)
    boot_resid = residual[year_indices]
    boot = monthly_mean[None, None, :] + boot_resid
    boot_s = _seasonal_mk_s_many(boot)
    exceed = int(np.sum(np.abs(boot_s) >= abs(observed_s)))
    p = (1 + exceed) / (n_bootstrap + 1)
    return observed_s, float(p), True


def _fit_method(method: str, y: np.ndarray, x: np.ndarray, periods: pd.PeriodIndex, *, hac_lag_months: int, n_bootstrap: int, random_seed: int, block_length: int) -> _Fit:
    ordinals = periods.year.to_numpy(dtype=int) * 12 + periods.month.to_numpy(dtype=int) - 1
    observed = np.isfinite(y)
    regular = bool(observed.all() and (len(ordinals) < 2 or np.all(np.diff(ordinals) == 1)))
    if method == "ols_hac":
        if not regular:
            beta, residual, _ = _ols(y, x)
            return _Fit(float(beta[1]), residuals=residual, converged=False, note="HAC row-order inference is withheld because monthly support is irregular.")
        fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": min(hac_lag_months, max(1, len(y) // 4)), "use_correction": True})
        lo, hi = np.asarray(fit.conf_int(alpha=0.05)[1], dtype=float)
        return _Fit(float(fit.params[1]), float(fit.bse[1]), float(lo), float(hi), float(fit.pvalues[1]), residuals=np.asarray(fit.resid), note="Frozen v1-style Bartlett HAC reference; known undercoverage regime.")
    if method == "ols_hac_calendar":
        slope, se, lo, hi, residual = _calendar_hac(y, x, ordinals, hac_lag_months)
        p = float(2 * norm.sf(abs(slope / se))) if se > 0 else None
        return _Fit(slope, se, lo, hi, p, residuals=residual, note="Bartlett covariance weighted by exact month distance; calibration not established.")
    if method == "gls_ar1":
        if not regular:
            beta, residual, _ = _ols(y, x)
            return _Fit(float(beta[1]), residuals=residual, converged=False, note="GLSAR requires complete equally spaced monthly support.")
        try:
            fit = sm.GLSAR(y, x, rho=1).iterative_fit(maxiter=100)
        except Exception as exc:
            return _Fit(None, converged=False, note=f"GLSAR failed: {type(exc).__name__}.")
        rho = float(np.asarray(fit.model.rho).reshape(-1)[0])
        lo, hi = np.asarray(fit.conf_int(alpha=0.05)[1], dtype=float)
        iteration_count = getattr(fit, "iter", None)
        converged = bool(iteration_count is not None and int(iteration_count) < 100 and np.isfinite(rho) and abs(rho) < 1.0 and np.isfinite(fit.params).all())
        note = f"Feasible GLS AR(1); iterations={iteration_count}; converged={converged}; validation depends on diagnostics and calibrated regime."
        dependence = {"ar1_rho": rho}
        if iteration_count is not None:
            dependence["iterations"] = float(iteration_count)
        return _Fit(float(fit.params[1]), float(fit.bse[1]), float(lo), float(hi), float(fit.pvalues[1]), dependence, np.asarray(fit.resid), converged, note)
    if method in {"sarimax_ar1", "arma11", "seasonal_ar12"}:
        if method == "sarimax_ar1":
            order, seasonal_order = (1, 0, 0), (0, 0, 0, 0)
        elif method == "arma11":
            order, seasonal_order = (1, 0, 1), (0, 0, 0, 0)
        else:
            order, seasonal_order = (0, 0, 0), (1, 0, 0, 12)
        try:
            model = sm.tsa.SARIMAX(y, exog=x, order=order, seasonal_order=seasonal_order, trend="n", enforce_stationarity=True, enforce_invertibility=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = model.fit(disp=False, maxiter=200)
        except Exception as exc:
            return _Fit(None, converged=False, note=f"SARIMAX failed: {type(exc).__name__}.")
        params = np.asarray(fit.params, dtype=float)
        bse = np.asarray(fit.bse, dtype=float)
        ci = np.asarray(fit.conf_int(alpha=0.05), dtype=float)[1]
        names = list(fit.param_names)
        p = float(np.asarray(fit.pvalues, dtype=float)[1])
        dependence = {name: float(value) for name, value in zip(names[x.shape[1]:], params[x.shape[1]:]) if np.isfinite(value)}
        converged = bool(fit.mle_retvals.get("converged", False)) and np.isfinite(params[1]) and np.isfinite(bse[1])
        residual = np.asarray(fit.resid, dtype=float)
        warning_details = [f"{item.category.__name__}: {str(item.message)[:240]}" for item in caught]
        note = f"SARIMAX; warnings={warning_details}; optimizer={fit.mle_retvals.get('converged', False)}"
        return _Fit(float(params[1]), float(bse[1]), float(ci[0]), float(ci[1]), p, dependence, residual, converged, note, tuple(warning_details))
    if method == "moving_block_bootstrap":
        if not regular:
            beta, residual, _ = _ols(y, x)
            return _Fit(float(beta[1]), residuals=residual, converged=False, note="Moving monthly residual blocks require complete regular support.")
        return _bootstrap_fit(y, x, periods, seasonal=False, n_bootstrap=n_bootstrap, seed=random_seed, block_length=block_length)
    if method == "seasonal_block_bootstrap":
        if not regular:
            beta, residual, _ = _ols(y, x)
            return _Fit(float(beta[1]), residuals=residual, converged=False, note="Seasonal year blocks require complete regular support.")
        return _bootstrap_fit(y, x, periods, seasonal=True, n_bootstrap=n_bootstrap, seed=random_seed, block_length=max(1, block_length // 12))
    if method == "seasonal_mk_block":
        slope = _seasonal_sen(y, periods)
        _s, _p, converged = _seasonal_mk_block_p(y, periods, n_bootstrap=n_bootstrap, seed=random_seed, block_years=max(1, int(np.ceil(block_length / 12))))
        return _Fit(slope, p_value=_p if converged else None, converged=converged, note="Seasonal MK null bootstrap over moving calendar-year residual blocks; no interval implemented.")
    raise ValueError(f"Unsupported statistical method: {method}")


def analyze_variable_v12(
    gold: pd.DataFrame,
    variable_id: str,
    *,
    region_id: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    method: str = "ols_hac",
    qa_policy: dict[str, Any] | None = None,
    declared_regime: str | None = None,
    validation_registry: dict[str, Any] | None = None,
    hac_lag_months: int = 12,
    n_bootstrap: int = 999,
    random_seed: int = 20261123,
    block_length_months: int = 12,
) -> dict[str, Any]:
    """Analyze one Gold series while withholding unvalidated inference.

    NaN values and QA exclusions remain missing on the complete monthly index.
    Infinities are rejected. Candidate confidence intervals and p-values are
    intentionally suppressed because no v1.2 method-regime calibration has yet
    passed its declared acceptance gate.
    """
    if method not in METHOD_IDS:
        raise ValueError(f"Unsupported statistical method: {method}")
    required = {"month", "variable_id", "value"}
    absent = required - set(gold.columns)
    if absent:
        raise ValueError(f"Gold frame is missing columns: {sorted(absent)}")
    selected = gold.loc[gold.variable_id.astype(str) == str(variable_id)].copy()
    if region_id is not None:
        if "region_id" not in selected:
            raise ValueError("region_id was requested but is absent from the Gold frame")
        selected = selected.loc[selected.region_id.astype(str) == str(region_id)].copy()
    elif "region_id" in selected:
        regions = selected.region_id.dropna().astype(str).unique()
        if len(regions) > 1:
            raise ValueError("multiple regions are present; specify region_id explicitly")
        region_id = str(regions[0]) if len(regions) else None
    if selected.empty:
        raise ValueError(f"No Gold rows for variable_id={variable_id}, region_id={region_id}")
    selected["_period"] = _periods(selected.month)
    if start_month is not None:
        selected = selected.loc[selected._period >= pd.Period(start_month, freq="M")]
    if end_month is not None:
        selected = selected.loc[selected._period <= pd.Period(end_month, freq="M")]
    if selected.empty:
        raise ValueError("No rows fall inside the requested calendar interval")
    if selected._period.duplicated().any():
        raise ValueError("duplicate calendar month for variable_id")
    was_sorted = bool(selected._period.is_monotonic_increasing)
    selected = selected.sort_values("_period", kind="stable")
    if "valid_area_fraction_bbox" not in selected:
        selected["valid_area_fraction_bbox"] = 1.0
    numeric_value = pd.to_numeric(selected.value, errors="coerce")
    malformed = selected.value.notna().to_numpy() & numeric_value.isna().to_numpy()
    if malformed.any():
        return _invalid_result(method, "NONFINITE_INPUT", variable_id, "Non-numeric nonmissing values are not accepted.", region_id=region_id)
    value = numeric_value.to_numpy(dtype=float)
    raw_coverage = selected.valid_area_fraction_bbox
    numeric_coverage = pd.to_numeric(raw_coverage, errors="coerce")
    malformed_coverage = raw_coverage.notna().to_numpy() & numeric_coverage.isna().to_numpy()
    if malformed_coverage.any():
        return _invalid_result(method, "NONFINITE_INPUT", variable_id, "Non-numeric nonmissing coverage values are not accepted.", region_id=region_id)
    coverage = numeric_coverage.to_numpy(dtype=float)
    if np.isinf(value).any() or np.isinf(coverage).any():
        return _invalid_result(method, "NONFINITE_INPUT", variable_id, "Infinity is not accepted in values or coverage.", region_id=region_id)
    if not np.isfinite(coverage[~np.isnan(coverage)]).all() or (coverage[~np.isnan(coverage)] < 0).any() or (coverage[~np.isnan(coverage)] > 1).any():
        return _invalid_result(method, "NONFINITE_INPUT", variable_id, "Coverage must be finite and within [0, 1].", region_id=region_id)
    if qa_policy is None:
        threshold = 0.1 if str(variable_id).startswith("modis_") else 0.8
        policy = {"min_valid_area_fraction_bbox": threshold}
    else:
        policy = dict(qa_policy)
        threshold = float(policy.get("min_valid_area_fraction_bbox", 0.0))
    if not 0 <= threshold <= 1:
        raise ValueError("min_valid_area_fraction_bbox must be between 0 and 1")
    eligible = np.isfinite(value) & np.isfinite(coverage) & (coverage >= threshold)
    periods_observed = pd.PeriodIndex(selected.loc[eligible, "_period"], freq="M")
    if len(periods_observed) == 0:
        return _invalid_result(method, "INSUFFICIENT_DATA", variable_id, "No observations pass the QA policy.", region_id=region_id)
    start = pd.Period(start_month, freq="M") if start_month else selected["_period"].min()
    end = pd.Period(end_month, freq="M") if end_month else selected["_period"].max()
    if end < start:
        raise ValueError("end_month precedes start_month")
    full_periods = pd.period_range(start, end, freq="M")
    ordinals = full_periods.year.to_numpy(dtype=int) * 12 + full_periods.month.to_numpy(dtype=int) - 1
    y = pd.Series(np.nan, index=full_periods, dtype=float)
    coverage_series = pd.Series(np.nan, index=full_periods, dtype=float)
    for idx, (period, observed_value, observed_coverage) in enumerate(zip(selected["_period"], value, coverage)):
        if period in y.index:
            if eligible[idx]:
                y.loc[period] = observed_value
            coverage_series.loc[period] = observed_coverage
    yv = y.to_numpy(dtype=float)
    n_obs = int(np.isfinite(yv).sum())
    missing_mask = ~np.isfinite(yv)
    max_missing_run = _max_run(missing_mask)
    content = selected[["month", "variable_id", "value", "valid_area_fraction_bbox"]].copy()
    content_sha = hashlib.sha256(pd.util.hash_pandas_object(content, index=False).to_numpy(dtype=np.uint64).tobytes()).hexdigest()
    units = str(selected["units"].dropna().iloc[0]) if "units" in selected and selected["units"].notna().any() else "UNSPECIFIED"
    base: dict[str, Any] = {
        "statistical_version": STATISTICAL_VERSION,
        "region_id": str(region_id) if region_id is not None else None,
        "variable_id": str(variable_id),
        "method_id": METHOD_IDS[method],
        "slope_units": f"{units} per decade",
        "slope_per_decade": None,
        "standard_error": None,
        "standard_error_is_inferentially_valid": False,
        "confidence_interval_95": None,
        "p_value": None,
        "error_dependence_model": method,
        "time_lag_unit": "calendar_month",
        "fitted_dependence_parameters": {},
        "residual_diagnostics": {},
        "n_eligible_observations": n_obs,
        "n_calendar_months": int(len(full_periods)),
        "calendar_start": str(full_periods[0]),
        "calendar_end": str(full_periods[-1]),
        "observed_start": str(periods_observed.min()),
        "observed_end": str(periods_observed.max()),
        "calendar_coverage_fraction": float(n_obs / len(full_periods)),
        "missing_calendar_months": [str(p) for p, ok in zip(full_periods, np.isfinite(yv)) if not ok],
        "max_missing_run_months": max_missing_run,
        "input_was_sorted": was_sorted,
        "qa_policy": policy,
        "declared_regime": declared_regime,
        "convergence_status": "NOT_RUN",
        "validity_status": "NOT_VALIDATED",
        "provenance": {
            "input_series_sha256": content_sha,
            "region_id": str(region_id) if region_id is not None else None,
            "declared_regime": declared_regime,
            "validation_registry_sha256": hashlib.sha256(json.dumps(validation_registry, sort_keys=True, allow_nan=False).encode()).hexdigest() if validation_registry is not None else None,
            "start_month_requested": start_month,
            "end_month_requested": end_month,
            "method_configuration": {"hac_lag_months": hac_lag_months, "n_bootstrap": n_bootstrap, "random_seed": random_seed, "block_length_months": block_length_months},
            "inference_gate": "An exact frozen validation-registry method/regime match is required before p-values or intervals are emitted.",
        },
    }
    finite_y = yv[np.isfinite(yv)]
    if n_obs < 24:
        return {**base, "validity_status": "INSUFFICIENT_DATA", "convergence_status": "NOT_RUN"}
    if method == "seasonal_mk_block":
        month_counts = pd.Series(yv, index=full_periods).dropna().groupby(lambda p: p.month).size()
        if any(int(month_counts.get(month, 0)) < 3 for month in range(1, 13)):
            return {**base, "validity_status": "INSUFFICIENT_DATA", "convergence_status": "NOT_RUN", "method_note": "Seasonal MK requires at least three eligible observations in every calendar-month stratum."}
    scale = max(1.0, float(np.max(np.abs(finite_y))))
    if float(np.std(finite_y)) <= 1e-12 * scale:
        return {**base, "slope_per_decade": 0.0, "validity_status": "DEGENERATE_SERIES", "convergence_status": "NOT_APPLICABLE", "residual_diagnostics": {"acf_calendar_months": {"1": None, "12": None}, "status": "UNDEFINED_FOR_CONSTANT_SERIES"}}
    x = _design(full_periods)
    requires_regular = method in {"ols_hac", "gls_ar1", "moving_block_bootstrap", "seasonal_block_bootstrap"}
    if requires_regular and not np.isfinite(yv).all():
        # Retain the descriptive fit but categorically suppress row-index HAC inference.
        try:
            beta, residual, _ = _ols(yv, x)
            slope = float(beta[1])
        except ValueError:
            residual, slope = None, None
        reason = "Method requires complete, equally spaced monthly observations; gaps remain explicit and no compressed-row fit is used."
        status, fit = "IRREGULAR_TIME_SUPPORT", _Fit(slope, residuals=residual, converged=False, note=reason)
    else:
        fit = _fit_method(method, yv, x, full_periods, hac_lag_months=hac_lag_months, n_bootstrap=n_bootstrap, random_seed=random_seed, block_length=block_length_months)
        status = "UNDERCOVERAGE_DETECTED" if method == "ols_hac" else "NOT_VALIDATED"
        if method == "seasonal_mk_block" and not fit.converged:
            status = "INSUFFICIENT_DATA"
        if method == "seasonal_block_bootstrap" and fit.note and "complete January-to-December" in fit.note:
            status = "INSUFFICIENT_DATA"
        if not fit.converged and fit.slope is None and status != "INSUFFICIENT_DATA":
            status = "NUMERICAL_FAILURE"
        if method in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"} and not fit.converged:
            status = "NUMERICAL_FAILURE"
    diagnostics = {}
    if fit.residuals is not None:
        residual_values = np.asarray(fit.residuals, dtype=float)
        if len(residual_values) == len(full_periods):
            diagnostics = _residual_diagnostics(residual_values, full_periods, x)
        elif method == "gls_ar1" and len(residual_values) == len(full_periods) - 1:
            diagnostics = _residual_diagnostics(residual_values, full_periods[1:], x[1:])
            diagnostics["residual_alignment"] = "GLSAR whitened residuals align to months after the first AR lag"
        else:
            diagnostics = {"status": "UNAVAILABLE_RESIDUAL_ALIGNMENT", "residual_count": int(len(residual_values)), "calendar_month_count": int(len(full_periods))}
    registry_entry = _validated_registry_entry(
        validation_registry, declared_regime, method, len(full_periods), float(n_obs / len(full_periods))
    )
    if status == "NOT_VALIDATED" and method in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"}:
        lb = diagnostics.get("ljung_box") or {}
        if any(item.get("p_value", 1.0) < 0.01 for item in lb.values()):
            status = "DEPENDENCE_MISSPECIFICATION"
    if registry_entry is not None and fit.converged and status in {"NOT_VALIDATED", "UNDERCOVERAGE_DETECTED"}:
        if method in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"} and not diagnostics.get("ljung_box"):
            status = "NOT_VALIDATED"
        else:
            status = "VALIDATED_FOR_DECLARED_REGIME"
    interval = [fit.ci_low, fit.ci_high] if status == "VALIDATED_FOR_DECLARED_REGIME" and fit.ci_low is not None and fit.ci_high is not None else None
    p_value = fit.p_value if status == "VALIDATED_FOR_DECLARED_REGIME" else None
    provenance = dict(base["provenance"])
    if status == "VALIDATED_FOR_DECLARED_REGIME":
        provenance["inference_gate"] = "Matched a frozen independent-validation registry entry for the caller-declared regime; model diagnostics passed the runtime guard."
    elif validation_registry is not None:
        provenance["inference_gate"] = "The supplied registry did not validate this exact method, regime, calendar length and coverage support; p-values and intervals are withheld."
    return {
        **base,
        "slope_per_decade": fit.slope,
        "standard_error": fit.standard_error,
        "standard_error_is_inferentially_valid": status == "VALIDATED_FOR_DECLARED_REGIME" and fit.standard_error is not None,
        "confidence_interval_95": interval,
        "p_value": p_value,
        "fitted_dependence_parameters": fit.dependence_parameters or {},
        "residual_diagnostics": diagnostics,
        "convergence_status": "CONVERGED" if fit.converged else "PARTIAL_OR_WITHHELD",
        "validity_status": status,
        "method_note": fit.note,
        "provenance": provenance,
    }


def _max_run(mask: np.ndarray) -> int:
    best = current = 0
    for item in mask:
        current = current + 1 if bool(item) else 0
        best = max(best, current)
    return best


def _invalid_result(method: str, status: str, variable_id: str, reason: str, *, region_id: str | None = None) -> dict[str, Any]:
    return {
        "statistical_version": STATISTICAL_VERSION,
        "region_id": str(region_id) if region_id is not None else None,
        "variable_id": str(variable_id),
        "method_id": METHOD_IDS[method],
        "slope_units": "UNSPECIFIED per decade",
        "slope_per_decade": None,
        "standard_error": None,
        "standard_error_is_inferentially_valid": False,
        "confidence_interval_95": None,
        "p_value": None,
        "error_dependence_model": method,
        "time_lag_unit": "calendar_month",
        "fitted_dependence_parameters": {},
        "residual_diagnostics": {},
        "n_eligible_observations": 0,
        "n_calendar_months": 0,
        "calendar_start": None,
        "calendar_end": None,
        "observed_start": None,
        "observed_end": None,
        "calendar_coverage_fraction": 0.0,
        "missing_calendar_months": [],
        "max_missing_run_months": 0,
        "input_was_sorted": False,
        "qa_policy": {},
        "convergence_status": "NOT_RUN",
        "validity_status": status,
        "method_note": reason,
        "provenance": {"statistical_version": STATISTICAL_VERSION},
    }
