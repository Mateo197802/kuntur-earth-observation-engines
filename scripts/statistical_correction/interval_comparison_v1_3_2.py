"""IID seasonal-AR(12) interval comparison for KUNTUR v1.3.2 development."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import chi2, norm, t as student_t
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox


METHODS = ("wald_normal", "wald_student_t", "profile_likelihood")
PROFILE_MAXIMUM_LLF_TOLERANCE = 1e-3
MODEL_SPEC = {
    "order": (0, 0, 0),
    "seasonal_order": (1, 0, 0, 12),
    "trend": "n",
    "enforce_stationarity": True,
    "enforce_invertibility": True,
    "optimizer_maxiter": 200,
    "optimizer": "statsmodels SARIMAX default LBFGS",
    "exog_contract": "intercept, elapsed decades, and 11 monthly indicator columns",
}


@dataclass(frozen=True)
class CandidateFit:
    method: str
    slope: float | None
    standard_error: float | None
    ci_low: float | None
    ci_high: float | None
    p_value: float | None
    converged: bool
    residuals: np.ndarray | None
    log_likelihood: float | None
    residual_df: int | None
    profile_evaluations: int
    profile_log_likelihood_max: float | None
    note: str | None
    warnings: tuple[str, ...]


def residual_degrees_of_freedom(n_observations: int, exog_rank: int, estimated_dynamic_parameters: int) -> int:
    """n - rank(X) - AR/MA parameters; innovation scale is not subtracted."""
    values = (n_observations, exog_rank, estimated_dynamic_parameters)
    if any(int(value) != value or value < 0 for value in values):
        raise ValueError("degrees-of-freedom inputs must be nonnegative integers")
    df = int(n_observations) - int(exog_rank) - int(estimated_dynamic_parameters)
    if df <= 0:
        raise ValueError("nonpositive residual degrees of freedom")
    return df


def critical_t_interval(beta_hat: float, standard_error: float, df: int, alpha: float = 0.05) -> dict[str, float]:
    if not (math.isfinite(beta_hat) and math.isfinite(standard_error) and standard_error > 0):
        raise ValueError("finite estimate and positive standard error are required")
    if df < 1 or not (0 < alpha < 1):
        raise ValueError("df must be positive and alpha must be in (0, 1)")
    statistic = beta_hat / standard_error
    critical = float(student_t.ppf(1 - alpha / 2, df))
    p_value = float(2 * student_t.sf(abs(statistic), df))
    return {
        "ci_low": float(beta_hat - critical * standard_error),
        "ci_high": float(beta_hat + critical * standard_error),
        "p_value": p_value,
        "critical_value": critical,
        "test_statistic": float(statistic),
        "residual_df": int(df),
    }


def profile_likelihood_interval(
    beta_hat: float,
    llf_max: float,
    standard_error: float,
    profile_llf: Callable[[float], tuple[float, bool, str | None]],
    alpha: float = 0.05,
) -> dict[str, float | int]:
    """Invert the one-parameter likelihood-ratio test using constrained SARIMAX fits."""
    if not all(math.isfinite(value) for value in (beta_hat, llf_max, standard_error)) or standard_error <= 0:
        raise ValueError("finite MLE, log-likelihood and positive standard error are required")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")

    cutoff = float(chi2.ppf(1 - alpha, 1))
    cache: dict[float, float] = {}

    center_llf, center_converged, center_note = profile_llf(float(beta_hat))
    if not center_converged or not math.isfinite(float(center_llf)):
        raise RuntimeError(f"PROFILE_OPTIMIZATION_FAILED_AT_UNRESTRICTED_ESTIMATE:{center_note or 'unknown'}")
    if float(center_llf) > llf_max + PROFILE_MAXIMUM_LLF_TOLERANCE:
        raise RuntimeError("PROFILE_LIKELIHOOD_MAXIMUM_MISMATCH_EXCEEDS_NUMERICAL_TOLERANCE")
    profile_max = max(float(llf_max), float(center_llf))
    cache[float(beta_hat)] = min(float(center_llf), profile_max)

    def at(value: float) -> float:
        key = float(value)
        if key not in cache:
            llf, converged, note = profile_llf(key)
            if not converged or not math.isfinite(float(llf)):
                raise RuntimeError(f"PROFILE_OPTIMIZATION_FAILED:{note or 'unknown'}")
            llf = float(llf)
            if llf > profile_max + PROFILE_MAXIMUM_LLF_TOLERANCE:
                raise RuntimeError("PROFILE_LIKELIHOOD_MAXIMUM_MISMATCH_EXCEEDS_NUMERICAL_TOLERANCE")
            # Statsmodels' default optimizer and the analytic profile optimizer use
            # different stopping rules. Clip only a sub-1e-3 upward roundoff/tolerance
            # discrepancy; larger mismatches remain explicit numerical failures.
            cache[key] = min(llf, profile_max)
        return cache[key]

    llf_null = at(0.0)
    lr_null = max(0.0, 2.0 * (profile_max - llf_null))

    def root_for(direction: int) -> float:
        center = float(beta_hat)
        step = float(standard_error)

        def signed_distance(candidate: float) -> float:
            return 2.0 * (profile_max - at(candidate)) - cutoff

        endpoint = center + direction * step
        endpoint_value = signed_distance(endpoint)
        expansions = 0
        while endpoint_value < 0 and expansions < 12:
            step *= 2.0
            endpoint = center + direction * step
            endpoint_value = signed_distance(endpoint)
            expansions += 1
        if endpoint_value < 0:
            raise RuntimeError("PROFILE_INTERVAL_FAILED_TO_BRACKET_WITHIN_4096_STANDARD_ERRORS")
        left, right = (endpoint, center) if direction < 0 else (center, endpoint)
        xtol = max(1e-9, abs(standard_error) * 1e-4)
        return float(brentq(signed_distance, left, right, xtol=xtol, rtol=1e-6, maxiter=60))

    low = root_for(-1)
    high = root_for(1)
    if not (low <= beta_hat <= high):
        raise RuntimeError("PROFILE_INTERVAL_DOES_NOT_CONTAIN_MLE")
    return {
        "ci_low": low,
        "ci_high": high,
        "profile_test_p_value": float(chi2.sf(lr_null, 1)),
        "profile_lr_null": float(lr_null),
        "profile_lr_cutoff": cutoff,
        "profile_evaluations": len(cache),
        "profile_log_likelihood_max": profile_max,
    }


def residual_ljung_box(residuals: np.ndarray, alpha: float = 0.01) -> dict[str, Any]:
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or len(values) <= 37 or not np.isfinite(values).all():
        return {"kind": "LJUNG_BOX_UNAVAILABLE", "p_values": None, "failure": None}
    try:
        result = acorr_ljungbox(values, lags=[12, 24], return_df=True)
    except (ValueError, np.linalg.LinAlgError):
        return {"kind": "LJUNG_BOX_UNAVAILABLE", "p_values": None, "failure": None}
    p_values = {str(int(lag)): float(result.loc[lag, "lb_pvalue"]) for lag in result.index}
    return {
        "kind": "LJUNG_BOX_COMPLETE_SUPPORT",
        "p_values": p_values,
        "failure": bool(any(value < alpha for value in p_values.values())),
        "alpha": float(alpha),
        "lags": [12, 24],
    }


def seasonal_ar12_profile_loglikelihood(y: np.ndarray, x: np.ndarray, fixed_slope: float) -> tuple[float, bool, str | None]:
    """Exact stationary Gaussian SAR(12) likelihood profiled over nuisance means, phi, and scale."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or len(y) % 12:
        return float("nan"), False, "PROFILE_REQUIRES_COMPLETE_JANUARY_ALIGNED_YEARS"
    if not np.isfinite(y).all() or not np.isfinite(x).all() or x.shape[1] < 2:
        return float("nan"), False, "PROFILE_REQUIRES_FINITE_INPUTS_AND_SLOPE_COLUMN"
    if not math.isfinite(fixed_slope):
        return float("nan"), False, "PROFILE_SLOPE_NOT_FINITE"

    n = len(y)
    n_independent_month_chains = 12
    nuisance_columns = np.arange(x.shape[1]) != 1
    fixed_mean = y - float(fixed_slope) * x[:, 1]

    def transformed(values: np.ndarray, phi: float) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        is_vector = values.ndim == 1
        matrix = values[:, None] if is_vector else values
        pieces: list[np.ndarray] = []
        initial_scale = math.sqrt(1.0 - phi * phi)
        for month in range(12):
            chain = matrix[month::12]
            pieces.append(initial_scale * chain[:1])
            pieces.append(chain[1:] - phi * chain[:-1])
        output = np.concatenate(pieces, axis=0)
        return output[:, 0] if is_vector else output

    def objective(phi: float) -> float:
        if not (-1.0 < phi < 1.0):
            return float("inf")
        yw = transformed(fixed_mean, phi)
        xw = transformed(x[:, nuisance_columns], phi)
        coefficients, *_ = np.linalg.lstsq(xw, yw, rcond=None)
        residual = yw - xw @ coefficients
        ssr = float(residual @ residual)
        if not math.isfinite(ssr) or ssr <= 0:
            return float("inf")
        sigma2 = ssr / n
        log_likelihood = -0.5 * (
            n * (math.log(2.0 * math.pi) + 1.0 + math.log(sigma2))
            - n_independent_month_chains * math.log1p(-(phi * phi))
        )
        return -log_likelihood

    optimum = minimize_scalar(
        objective,
        method="bounded",
        bounds=(-0.9999, 0.9999),
        options={"xatol": 1e-6, "maxiter": 100},
    )
    if not optimum.success or not math.isfinite(float(optimum.fun)):
        return float("nan"), False, "PROFILE_SEASONAL_AR_PARAMETER_OPTIMIZATION_FAILED"
    if abs(float(optimum.x)) >= 0.999:
        return float("nan"), False, "PROFILE_SEASONAL_AR_PARAMETER_AT_SEARCH_BOUND"
    return -float(optimum.fun), True, None


def fit_candidate(
    y: np.ndarray,
    x: np.ndarray,
    method: str,
    *,
    alpha: float = 0.05,
    maxiter: int = 200,
) -> tuple[CandidateFit, dict[str, Any]]:
    """Fit the unchanged SARIMAX point model and derive one candidate interval."""
    if method not in METHODS:
        raise ValueError(f"unknown interval method: {method}")
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.ndim != 1 or x.ndim != 2 or len(y) != len(x) or not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("complete, finite IID series and aligned exogenous design are required")
    if x.shape[1] < 2 or not np.allclose(x[:, 0], 1.0):
        raise ValueError("exogenous column 0 must be the intercept and column 1 the slope")

    model = sm.tsa.SARIMAX(
        y,
        exog=x,
        order=MODEL_SPEC["order"],
        seasonal_order=MODEL_SPEC["seasonal_order"],
        trend=MODEL_SPEC["trend"],
        enforce_stationarity=MODEL_SPEC["enforce_stationarity"],
        enforce_invertibility=MODEL_SPEC["enforce_invertibility"],
    )
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    warning_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = model.fit(disp=False, maxiter=maxiter)
        warning_messages.extend(f"{item.category.__name__}:{str(item.message)[:200]}" for item in caught)
    except Exception as exc:  # keep diagnostics non-sensitive and deterministic
        elapsed = {"wall_seconds": time.perf_counter() - start_wall, "cpu_seconds": time.process_time() - start_cpu}
        failed = CandidateFit(method, None, None, None, None, None, False, None, None, None, 0, None, f"SARIMAX_FIT_EXCEPTION_{type(exc).__name__}", tuple(warning_messages))
        return failed, elapsed

    params = np.asarray(fit.params, dtype=float)
    bse = np.asarray(fit.bse, dtype=float)
    names = list(fit.param_names)
    try:
        slope_index = names.index("x1")
    except ValueError:
        slope_index = 1
    beta_hat = float(params[slope_index]) if len(params) > slope_index else float("nan")
    standard_error = float(bse[slope_index]) if len(bse) > slope_index else float("nan")
    base_converged = bool(fit.mle_retvals.get("converged", False)) and math.isfinite(beta_hat) and math.isfinite(standard_error) and standard_error > 0
    residual = np.asarray(fit.resid, dtype=float)
    llf_max = float(fit.llf)
    dynamic_parameter_count = sum(
        1 for name in names
        if name not in {"const", "sigma2"} and not name.startswith("x") and not name.endswith("(fixed)")
    )
    df = residual_degrees_of_freedom(len(y), int(np.linalg.matrix_rank(x)), dynamic_parameter_count)
    ci_low: float | None = None
    ci_high: float | None = None
    p_value: float | None = None
    profile_evaluations = 0
    profile_max: float | None = None
    note: str | None = None
    method_converged = base_converged

    if base_converged and method == "wald_normal":
        interval = np.asarray(fit.conf_int(alpha=alpha), dtype=float)[slope_index]
        ci_low, ci_high = float(interval[0]), float(interval[1])
        p_value = float(np.asarray(fit.pvalues, dtype=float)[slope_index])
    elif base_converged and method == "wald_student_t":
        interval = critical_t_interval(beta_hat, standard_error, df, alpha)
        ci_low, ci_high, p_value = interval["ci_low"], interval["ci_high"], interval["p_value"]
    elif base_converged and method == "profile_likelihood":
        def profile(value: float) -> tuple[float, bool, str | None]:
            return seasonal_ar12_profile_loglikelihood(y, x, float(value))

        try:
            profile_result = profile_likelihood_interval(beta_hat, llf_max, standard_error, profile, alpha)
            ci_low = float(profile_result["ci_low"])
            ci_high = float(profile_result["ci_high"])
            p_value = float(profile_result["profile_test_p_value"])
            profile_evaluations = int(profile_result["profile_evaluations"])
            profile_max = float(profile_result["profile_log_likelihood_max"])
        except Exception as exc:
            method_converged = False
            note = f"PROFILE_INTERVAL_FAILURE_{type(exc).__name__}:{str(exc)[:180]}"

    if not base_converged and note is None:
        note = "SARIMAX_POINT_FIT_NOT_CONVERGED_OR_NONFINITE"
    elif method_converged and (ci_low is None or ci_high is None or p_value is None or not all(map(math.isfinite, (ci_low, ci_high, p_value)))):
        method_converged = False
        note = "NONFINITE_INFERENTIAL_OUTPUT"
    elapsed = {"wall_seconds": time.perf_counter() - start_wall, "cpu_seconds": time.process_time() - start_cpu}
    return CandidateFit(
        method=method,
        slope=beta_hat if math.isfinite(beta_hat) else None,
        standard_error=standard_error if math.isfinite(standard_error) else None,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        converged=bool(method_converged),
        residuals=residual,
        log_likelihood=llf_max,
        residual_df=df,
        profile_evaluations=profile_evaluations,
        profile_log_likelihood_max=profile_max,
        note=note,
        warnings=tuple(warning_messages),
    ), elapsed
