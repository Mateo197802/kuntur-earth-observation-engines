"""Coverage-aware monthly climatology and trend estimates for KUNTUR Gold."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import theilslopes
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


ANALYTICAL_VERSION = "phase2-analytical-1.0.0"
ANALYTICAL_COMBINED = {
    "gaussian+drift": ("gaussian", "drift"),
    "gaussian+impulsive": ("gaussian", "impulsive"),
    "drift+missingness": ("drift", "missingness"),
    "gaussian+heteroscedastic": ("gaussian", "heteroscedastic"),
    "mixed_multi_family": ("gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness"),
}


def monthly_anomalies(
    gold: pd.DataFrame,
    variable_id: str,
    *,
    coverage_threshold: float = 0.8,
) -> pd.DataFrame:
    """Return a copy with calendar-month climatology/anomalies and coverage gating.

    The source Gold values are preserved in ``value``. Rows failing the declared
    valid-area threshold contribute neither to the climatology nor to inference.
    """
    if not 0.0 <= coverage_threshold <= 1.0:
        raise ValueError("coverage_threshold must be between 0 and 1")
    required = {"month", "variable_id", "value", "valid_area_fraction_bbox"}
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"Gold frame is missing columns: {sorted(missing)}")
    frame = gold.loc[gold.variable_id == variable_id].copy()
    if frame.empty:
        raise ValueError(f"No Gold rows for variable_id={variable_id}")
    frame["month_start"] = pd.to_datetime(frame["month"].astype(str) + "-01", errors="raise")
    if frame.month_start.duplicated().any():
        raise ValueError(f"Duplicate monthly rows for variable_id={variable_id}")
    frame = frame.sort_values("month_start", kind="stable").reset_index(drop=True)
    eligible = (
        frame["value"].notna()
        & frame["valid_area_fraction_bbox"].notna()
        & (frame["valid_area_fraction_bbox"] >= coverage_threshold)
    )
    frame["coverage_eligible"] = eligible
    frame["calendar_month"] = frame.month_start.dt.month
    frame["climatology"] = np.nan
    frame["anomaly"] = np.nan
    frame["climatology_sample_count"] = 0
    reference = frame.loc[eligible].groupby("calendar_month", sort=True)["value"]
    means = reference.mean()
    counts = reference.count()
    frame["climatology"] = frame.calendar_month.map(means)
    frame["climatology_sample_count"] = frame.calendar_month.map(counts).fillna(0).astype(int)
    frame.loc[eligible, "anomaly"] = frame.loc[eligible, "value"] - frame.loc[eligible, "climatology"]
    frame["coverage_threshold"] = float(coverage_threshold)
    frame["analytical_version"] = ANALYTICAL_VERSION
    return frame


def _seasonal_mann_kendall(frame: pd.DataFrame) -> dict:
    """Classic seasonal MK, stratified by calendar month with tie correction.

    This test assumes independent values within each same-calendar-month
    sequence. Serial dependence diagnostics are returned alongside the p-value.
    """
    from scipy.stats import norm

    total_s = 0.0
    total_var = 0.0
    strata = 0
    for _, group in frame.dropna(subset=["anomaly"]).groupby("calendar_month"):
        y = group.sort_values("month_start")["anomaly"].to_numpy(dtype=float)
        n = len(y)
        if n < 3:
            continue
        diff = y[np.newaxis, :] - y[:, np.newaxis]
        upper = diff[np.triu_indices(n, k=1)]
        total_s += float(np.sign(upper).sum())
        _, tie_counts = np.unique(y, return_counts=True)
        tie_term = float(np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5)))
        total_var += (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0
        strata += 1
    if total_var <= 0:
        return {"seasonal_mk_s": total_s, "seasonal_mk_variance": total_var, "seasonal_mk_z": None, "seasonal_mk_p_value": None, "seasonal_mk_strata": strata, "seasonal_mk_status": "INSUFFICIENT_OR_CONSTANT"}
    z = (total_s - np.sign(total_s)) / np.sqrt(total_var) if total_s else 0.0
    p = float(2.0 * norm.sf(abs(z)))
    return {"seasonal_mk_s": total_s, "seasonal_mk_variance": total_var, "seasonal_mk_z": float(z), "seasonal_mk_p_value": p, "seasonal_mk_strata": strata, "seasonal_mk_status": "CALCULATED_CLASSIC_INDEPENDENCE_ASSUMPTION"}


def analyze_variable(
    gold: pd.DataFrame,
    variable_id: str,
    *,
    coverage_threshold: float = 0.8,
    hac_maxlags: int = 12,
) -> dict:
    """Estimate seasonally-adjusted OLS-HAC, Theil–Sen, and seasonal MK trends."""
    ts = monthly_anomalies(gold, variable_id, coverage_threshold=coverage_threshold)
    valid = ts.dropna(subset=["anomaly"]).copy()
    if len(valid) < 24:
        raise ValueError(f"At least 24 eligible monthly observations are required; got {len(valid)}")
    month_number = valid.month_start.dt.year * 12 + valid.month_start.dt.month - 1
    valid["elapsed_years"] = (month_number - month_number.iloc[0]) / 12.0
    x = valid["elapsed_years"].to_numpy(dtype=float)
    y = valid["value"].to_numpy(dtype=float)
    month_effects = pd.get_dummies(valid.calendar_month.astype("category"), prefix="month", drop_first=True, dtype=float)
    X = pd.concat([pd.Series(1.0, index=valid.index, name="const"), pd.Series(x, index=valid.index, name="elapsed_years"), month_effects.set_axis(valid.index)], axis=1)
    lags = min(int(hac_maxlags), max(1, len(valid) // 4))
    fitted = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags, "use_correction": True})
    slope_year = float(fitted.params["elapsed_years"])
    ci_year = fitted.conf_int(alpha=0.05).loc["elapsed_years"].to_numpy(dtype=float)
    seasonal_sen = []
    for _, group in valid.groupby("calendar_month"):
        if len(group) >= 3:
            gy = group.value.to_numpy(dtype=float)
            gx = group.elapsed_years.to_numpy(dtype=float)
            seasonal_sen.append(theilslopes(gy, gx, alpha=0.95))
    if not seasonal_sen:
        raise ValueError("Insufficient within-calendar-month observations for seasonal Theil-Sen")
    sen_slope = float(np.median([item.slope for item in seasonal_sen]))
    sen_low = float(np.median([item.low_slope for item in seasonal_sen]))
    sen_high = float(np.median([item.high_slope for item in seasonal_sen]))
    residual = np.asarray(fitted.resid, dtype=float)
    if len(residual) > 1 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
        lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
    else:
        lag1 = None
    seasonal_lag12 = None
    if len(residual) > 12 and np.std(residual[:-12]) > 0 and np.std(residual[12:]) > 0:
        seasonal_lag12 = float(np.corrcoef(residual[:-12], residual[12:])[0, 1])
    mk = _seasonal_mann_kendall(valid)
    units = str(valid["units"].iloc[0]) if "units" in valid.columns else "UNSPECIFIED"
    trend = {
        "variable_id": variable_id,
        "units": units,
        "trend_units": f"{units} per decade",
        "n_months": int(len(valid)),
        "first_month": str(valid.month.iloc[0]),
        "last_month": str(valid.month.iloc[-1]),
        "coverage_threshold": float(coverage_threshold),
        "median_valid_area_fraction": float(valid.valid_area_fraction_bbox.median()),
        "ols_hac_slope_per_decade": slope_year * 10.0,
        "ols_hac_ci95_per_decade": [float(ci_year[0] * 10.0), float(ci_year[1] * 10.0)],
        "ols_hac_p_value": float(fitted.pvalues["elapsed_years"]),
        "hac_maxlags_months": int(lags),
        "theil_sen_slope_per_decade": sen_slope * 10.0,
        "theil_sen_seasonal_stratum_slope_range_per_decade": [float(min(item.slope for item in seasonal_sen) * 10.0), float(max(item.slope for item in seasonal_sen) * 10.0)],
        "ols_theil_sen_sign_concordance": bool(np.sign(slope_year) == np.sign(sen_slope)),
        "ols_seasonality_adjustment": "11 calendar-month fixed effects in original units; calendar month index is exact 12-month spacing",
        "theil_sen_method": "median of within-calendar-month Theil-Sen slopes; accompanying range is the min/max across seasonal strata, not a confidence interval",
        "residual_autocorrelation_lag1": lag1,
        "residual_autocorrelation_lag12": seasonal_lag12,
        "durbin_watson_residual": float(sm.stats.stattools.durbin_watson(residual)),
        "seasonal_mk_same_season_autocorrelation_assumption": "Classic seasonal MK is interpreted cautiously; lag-12 residual correlation approximates same-calendar-month dependence.",
        "trend_interpretation": "descriptive association over time; no causal attribution",
        **mk,
    }
    return {"time_series": ts, "trend": trend}


def cached_analyze_variable(
    gold: pd.DataFrame,
    variable_id: str,
    *,
    cache,
    coverage_threshold: float = 0.8,
    key_context: dict | None = None,
) -> dict:
    """Return a trend result from a scientific-input-aware shared cache.

    ``key_context`` may provide source/Gold SHA, contract and geometry hashes;
    observed source hashes, contract hashes, bbox coordinates and the canonical
    selected-series content hash are also included automatically.
    """
    selected = gold.loc[gold.variable_id == variable_id].copy()
    if selected.empty:
        raise ValueError(f"No Gold rows for variable_id={variable_id}")
    selected = selected.sort_values("month", kind="stable").reset_index(drop=True)
    hash_columns = [c for c in ("month", "variable_id", "value", "valid_area_fraction_bbox", "source_sha256", "contract_sha256") if c in selected.columns]
    row_hashes = pd.util.hash_pandas_object(selected[hash_columns], index=False).to_numpy(dtype=np.uint64).tobytes()
    selected_content_sha = hashlib.sha256(row_hashes).hexdigest()
    context = key_context or {}
    geometry_columns = [c for c in ("bbox_west_deg", "bbox_south_deg", "bbox_east_deg", "bbox_north_deg") if c in selected.columns]
    geometry_raw = selected[geometry_columns].drop_duplicates().sort_values(geometry_columns).to_json(orient="records") if geometry_columns else str(context.get("region_geometry_hash", "unspecified"))
    region_geometry_hash = context.get("region_geometry_hash") or hashlib.sha256(geometry_raw.encode()).hexdigest()
    params = {
        "dataset_ids": sorted(selected.dataset_id.dropna().astype(str).unique()) if "dataset_id" in selected else [],
        "source_sha256": sorted(selected.source_sha256.dropna().astype(str).unique()) if "source_sha256" in selected else [],
        "gold_sha256": context.get("gold_sha256", selected_content_sha),
        "selected_series_content_sha256": selected_content_sha,
        "contract_version": context.get("contract_version", sorted(selected.contract_sha256.dropna().astype(str).unique()) if "contract_sha256" in selected else "unspecified"),
        "region_geometry_hash": region_geometry_hash,
        "region_id": sorted(selected.region_id.dropna().astype(str).unique()) if "region_id" in selected else [],
        "variable": variable_id,
        "time_interval": [str(selected.month.min()), str(selected.month.max())],
        "qa_policy": f"valid_area_fraction_bbox >= {coverage_threshold:.6f}",
        "aggregation_method": sorted(selected.spatial_aggregation.dropna().astype(str).unique()) if "spatial_aggregation" in selected else "preaggregated Gold",
        "statistical_method": "monthly-climatology+OLS-HAC+seasonal-Theil-Sen+seasonal-MK",
        "code_version": ANALYTICAL_VERSION,
    }
    key = cache.key(params)
    cached = cache.get_json(key)
    if cached is not None:
        return {"trend": cached, "cache_hit": True, "cache_key": key}
    trend = analyze_variable(selected, variable_id, coverage_threshold=coverage_threshold)["trend"]
    cache.put_json(key, trend)
    return {"trend": trend, "cache_hit": False, "cache_key": key}


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Return FDR-adjusted p-values in input order, preserving missing entries."""
    values = np.asarray(p_values, dtype=float)
    valid = np.isfinite(values)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    if valid.any():
        adjusted[valid] = multipletests(values[valid], method="fdr_bh")[1]
    return [float(x) if np.isfinite(x) else float("nan") for x in adjusted]


def perturb_analytical_series(
    source: pd.DataFrame,
    variable_id: str,
    *,
    family: str,
    intensity: float,
    seed: int,
    coverage_threshold: float | None = None,
) -> pd.DataFrame:
    """Create a seeded derivative for analytical sensitivity tests; never edits Gold."""
    if family not in {"gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness", *ANALYTICAL_COMBINED}:
        raise ValueError(f"Unsupported analytical perturbation family: {family}")
    if not 0.0 <= intensity <= 0.5:
        raise ValueError("intensity must be in [0, 0.5]")
    out = source.loc[source.variable_id == variable_id].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(f"No rows for variable_id={variable_id}")
    y = out.value.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    finite = np.isfinite(y)
    eligible_mask = finite.copy()
    if coverage_threshold is not None:
        if "valid_area_fraction_bbox" not in out.columns:
            raise ValueError("Coverage-aware perturbation requires valid_area_fraction_bbox")
        eligible_mask &= out.valid_area_fraction_bbox.to_numpy(dtype=float) >= float(coverage_threshold)
    sd = float(np.std(y[eligible_mask], ddof=1)) if eligible_mask.sum() > 1 else 1.0
    sd = sd if np.isfinite(sd) and sd > 1e-12 else 1.0
    components = ANALYTICAL_COMBINED.get(family, (family,))
    missing_component = "missingness" in components
    eligible = np.flatnonzero(eligible_mask)
    missing = np.zeros_like(y, dtype=bool)
    if missing_component:
        count = int(round(intensity / np.sqrt(len(components)) * len(eligible))) if len(components) > 1 else int(round(intensity * len(eligible)))
        if intensity > 0 and count == 0 and len(eligible):
            count = 1
        if count:
            missing[rng.choice(eligible, size=min(count, len(eligible)), replace=False)] = True
    missing_fraction = float(missing.sum() / max(1, eligible_mask.sum()))
    additive_target = float(np.sqrt(max(0.0, intensity**2 - missing_fraction**2))) if missing_component else float(intensity)
    additive_components = [name for name in components if name != "missingness"]
    noise = np.zeros_like(y)
    component_budget = additive_target / np.sqrt(max(1, len(additive_components)))
    for component in additive_components:
        if component == "gaussian":
            noise += rng.normal(size=len(y)) * component_budget * sd
        elif component == "laplace":
            noise += rng.laplace(size=len(y)) * component_budget * sd / np.sqrt(2.0)
        elif component == "impulsive":
            active = rng.random(len(y)) < 0.05
            noise[active] += rng.choice([-1.0, 1.0], active.sum()) * component_budget * sd / np.sqrt(0.05)
        elif component == "drift":
            z = rng.normal(size=len(y))
            for i in range(1, len(z)):
                z[i] = 0.8 * z[i - 1] + np.sqrt(1 - 0.8**2) * z[i]
            z -= z.mean()
            rms = float(np.sqrt(np.mean(z**2)))
            if rms:
                noise += z * component_budget * sd / rms
        elif component == "heteroscedastic":
            hetero = np.sqrt(1.0 + np.abs((y - np.nanmean(y)) / sd))
            z = rng.normal(size=len(y)) * hetero
            z -= np.nanmean(z)
            rms = float(np.sqrt(np.nanmean(z**2)))
            if rms:
                noise += z * component_budget * sd / rms
    noise[~eligible_mask] = 0.0
    realized_raw_rms = float(np.sqrt(np.mean(noise[eligible_mask] ** 2))) if eligible_mask.any() else 0.0
    target_raw_rms = additive_target * sd
    if realized_raw_rms > 1e-12:
        noise *= target_raw_rms / realized_raw_rms
    realized_additive = float(np.sqrt(np.mean(noise[eligible_mask] ** 2)) / sd) if eligible_mask.any() else 0.0
    effective = float(np.sqrt(realized_additive**2 + missing_fraction**2))
    out["value"] = y + noise
    out.loc[missing, "value"] = np.nan
    out["analytical_noise_type"] = family
    out["analytical_noise_intensity"] = float(intensity)
    out["analytical_noise_seed"] = int(seed)
    out["analytical_noise_components"] = json.dumps(list(components))
    out["analytical_effective_intensity"] = effective
    out["analytical_missing_perturbation"] = missing
    out["analytical_noise_parameters_json"] = json.dumps({
        "intensity_definition": "additive RMS relative to this source series SD; missingness is fraction of eligible months",
        "realized_additive_rms_over_source_sd": realized_additive,
        "missing_months": int(missing.sum()),
        "eligible_months": int(eligible_mask.sum()),
        "coverage_threshold": coverage_threshold,
        "realized_effective_intensity": effective,
        "combined_budget_rule": "Per-component RMS budgets split by square root of component count; missing fraction and additive RMS combined by L2 norm.",
        "ar1_rho_for_drift": 0.8 if family == "drift" else None,
        "spike_duty_fraction": 0.05 if family == "impulsive" else None,
    }, sort_keys=True)
    out["derivative_version"] = f"phase2-analytical-robustness-{ANALYTICAL_VERSION}"
    return out
