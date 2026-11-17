from __future__ import annotations
import hashlib, json, math, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import theilslopes, norm
import psutil

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
SAVED = ROOT / "results/phase2/public/analytical_trends_v1.csv"
OUT = ROOT / "results/statistical_audit/real_trend_recalculation.csv"
MANIFEST = ROOT / "results/statistical_audit/real_trend_recalculation.manifest.json"
THRESHOLDS = {
    "merra2_t2m_c": .8,
    "imerg_precip_monthly_total": .8,
    "gldas_soil_moisture_0_10cm": .8,
    "gldas_soil_moisture_10_40cm": .8,
    "gldas_soil_moisture_40_100cm": .8,
    "gldas_soil_moisture_100_200cm": .8,
    "modis_ndvi": .1,
    "modis_evi": .1,
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def mk(frame: pd.DataFrame) -> dict:
    s_total = 0.0
    v_total = 0.0
    strata = 0
    for _, g in frame.groupby("calendar_month", sort=True):
        y = g["value"].to_numpy(dtype=float)
        n = len(y)
        if n < 3:
            continue
        d = y[np.newaxis, :] - y[:, np.newaxis]
        s_total += float(np.sign(d[np.triu_indices(n, k=1)]).sum())
        _, ties = np.unique(y, return_counts=True)
        tie = float(np.sum(ties * (ties - 1) * (2 * ties + 5)))
        v_total += float((n * (n - 1) * (2 * n + 5) - tie) / 18.0)
        strata += 1
    if v_total <= 0:
        return {"mk_s": s_total, "mk_variance": v_total, "mk_p": None, "mk_strata": strata}
    z = (s_total - np.sign(s_total)) / math.sqrt(v_total) if s_total else 0.0
    return {"mk_s": s_total, "mk_variance": v_total, "mk_p": float(2 * norm.sf(abs(z))), "mk_strata": strata}

def bh(p_values):
    p = np.asarray(p_values, float)
    ok = np.isfinite(p)
    q = np.full(len(p), np.nan)
    if ok.any():
        ix = np.flatnonzero(ok)
        order = ix[np.argsort(p[ix], kind="stable")]
        m = len(order)
        raw = p[order] * m / np.arange(1, m + 1)
        q_sorted = np.minimum.accumulate(raw[::-1])[::-1]
        q[order] = np.minimum(q_sorted, 1.0)
    return q

def fit_one(g: pd.DataFrame, threshold: float) -> dict:
    g = g.sort_values("month", kind="stable").copy()
    g = g.loc[g["value"].notna() & np.isfinite(g["value"].astype(float)) &
              g["valid_area_fraction_bbox"].notna() &
              (g["valid_area_fraction_bbox"].astype(float) >= threshold)].copy()
    if len(g) < 24:
        raise ValueError(f"only {len(g)} eligible rows for {g.variable_id.iloc[0]}")
    pidx = pd.PeriodIndex(g.month.astype(str), freq="M")
    ordinal = pidx.year * 12 + pidx.month - 1
    elapsed = (ordinal - ordinal[0]) / 12.0
    g["calendar_month"] = pidx.month
    month_effects = pd.get_dummies(pd.Series(pidx.month, index=g.index).astype("category"), drop_first=True, dtype=float).to_numpy()
    X = np.column_stack([np.ones(len(g)), elapsed, month_effects])
    y = g.value.to_numpy(dtype=float)
    bread = np.linalg.inv(X.T @ X)
    beta = bread @ X.T @ y
    resid = y - X @ beta
    score = X * resid[:, None]
    meat = score.T @ score
    lag = min(12, max(1, len(g) // 4))
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)
        cross = score[k:].T @ score[:-k]
        meat += w * (cross + cross.T)
    cov = bread @ meat @ bread
    cov *= len(g) / (len(g) - X.shape[1])
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    z = float(beta[1] / se) if se else None
    p = float(2 * norm.sf(abs(z))) if z is not None else None
    zcrit = 1.959963984540054
    ci = [(beta[1] - zcrit * se) * 10.0, (beta[1] + zcrit * se) * 10.0]
    seasonal_slopes = []
    seasonal_lows = []
    seasonal_highs = []
    g = g.assign(elapsed_years=elapsed)
    for _, s in g.groupby("calendar_month", sort=True):
        if len(s) >= 3:
            res = theilslopes(s.value.to_numpy(float), s.elapsed_years.to_numpy(float), alpha=.95)
            seasonal_slopes.append(float(res.slope) * 10.0)
            seasonal_lows.append(float(res.low_slope) * 10.0)
            seasonal_highs.append(float(res.high_slope) * 10.0)
    lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if len(resid) > 1 and np.std(resid[:-1]) > 0 and np.std(resid[1:]) > 0 else None
    lag12 = float(np.corrcoef(resid[:-12], resid[12:])[0, 1]) if len(resid) > 12 and np.std(resid[:-12]) > 0 and np.std(resid[12:]) > 0 else None
    return {
        "n_months": len(g), "first_month": str(pidx[0]), "last_month": str(pidx[-1]),
        "units": str(g.units.iloc[0]), "coverage_threshold": threshold,
        "median_valid_area_fraction": float(g.valid_area_fraction_bbox.median()),
        "ols_hac_slope_per_decade": float(beta[1] * 10),
        "ols_hac_se_per_decade": float(se * 10), "ols_hac_ci95_low": float(ci[0]),
        "ols_hac_ci95_high": float(ci[1]), "ols_hac_p_value": p,
        "hac_maxlags_rows": lag,
        "seasonal_theil_sen_per_decade": float(np.median(seasonal_slopes)),
        "seasonal_sen_stratum_range_low": min(seasonal_slopes),
        "seasonal_sen_stratum_range_high": max(seasonal_slopes),
        "residual_row_acf1": lag1, "residual_row_acf12": lag12,
        **mk(g),
    }

def main():
    start = time.perf_counter()
    if not GOLD.is_file() or not SAVED.is_file():
        raise SystemExit("Frozen Gold or saved trend CSV is absent; refusing recalculation.")
    gold_hash = sha256(GOLD)
    gold = pd.read_parquet(GOLD)
    if len(gold) != 5520 or gold.month.nunique() != 276:
        raise SystemExit("Frozen Gold dimensions differ from the recorded v1 input.")
    saved = pd.read_csv(SAVED).set_index("variable_id")
    rows = []
    for variable, threshold in THRESHOLDS.items():
        result = fit_one(gold.loc[gold.variable_id == variable], threshold)
        result["variable_id"] = variable
        result["saved_ols_hac_slope_per_decade"] = float(saved.loc[variable, "ols_hac_slope_per_decade"])
        result["saved_ols_hac_p_value"] = float(saved.loc[variable, "ols_hac_p_value"])
        result["saved_ols_hac_ci_low"] = float(saved.loc[variable, "ols_hac_ci95_per_decade"][1:-1].split(",")[0])
        result["saved_ols_hac_ci_high"] = float(saved.loc[variable, "ols_hac_ci95_per_decade"][1:-1].split(",")[1])
        result["slope_abs_difference"] = abs(result["ols_hac_slope_per_decade"] - result["saved_ols_hac_slope_per_decade"])
        result["p_abs_difference"] = abs(result["ols_hac_p_value"] - result["saved_ols_hac_p_value"])
        result["ci_low_abs_difference"] = abs(result["ols_hac_ci95_low"] - result["saved_ols_hac_ci_low"])
        result["ci_high_abs_difference"] = abs(result["ols_hac_ci95_high"] - result["saved_ols_hac_ci_high"])
        rows.append(result)
    q = bh([r["ols_hac_p_value"] for r in rows])
    mk_q = bh([r["mk_p"] if r["mk_p"] is not None else np.nan for r in rows])
    sq = bh([float(saved.loc[r["variable_id"], "ols_hac_p_value"]) for r in rows])
    for i, row in enumerate(rows):
        row["bh_q_recomputed"] = float(q[i])
        row["mk_bh_q_diagnostic_only"] = float(mk_q[i]) if np.isfinite(mk_q[i]) else None
        row["bh_q_saved_recomputed"] = float(sq[i])
        row["bh_q_abs_difference"] = abs(float(q[i] - sq[i]))
        row["gold_sha256"] = gold_hash
    pd.DataFrame(rows).to_csv(OUT, index=False)
    result = {
        "status": "INDEPENDENT_TREND_RECALCULATION_COMPLETE",
        "gold_sha256": gold_hash, "rows": len(rows),
        "max_slope_abs_difference": max(r["slope_abs_difference"] for r in rows),
        "max_p_abs_difference": max(r["p_abs_difference"] for r in rows),
        "max_ci_abs_difference": max(max(r["ci_low_abs_difference"], r["ci_high_abs_difference"]) for r in rows),
        "max_q_abs_difference": max(r["bh_q_abs_difference"] for r in rows),
        "output": OUT.relative_to(ROOT).as_posix(),
        "output_sha256": sha256(OUT),
        "elapsed_seconds": round(time.perf_counter() - start, 6),
        "peak_working_set_bytes": int(getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss)),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__
    }
    MANIFEST.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))

if __name__ == "__main__":
    main()
