from __future__ import annotations
import hashlib, json, os, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from scipy.stats import beta, chi2_contingency
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels
import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statistical_audit"
REPORTS = ROOT / "reports/statistical_audit"
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
TREND = OUT / "real_trend_recalculation.csv"
SIM_PARQUET = OUT / "simulation_metrics.parquet"
METHOD_CSV = OUT / "method_benchmark.csv"
FINDINGS_JSON = OUT / "root_cause_findings.json"
MANIFEST_JSON = OUT / "audit_manifest.json"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def cp(success, n):
    lo = 0.0 if success == 0 else float(beta.ppf(.025, success, n-success+1))
    hi = 1.0 if success == n else float(beta.ppf(.975, success+1, n-success))
    return lo, hi

def main():
    started=time.perf_counter()
    gold=pd.read_parquet(GOLD)
    tr=pd.read_csv(TREND)
    rows=[]
    def add(process, rho, n, method, lag, reps, seed, coverage, slope=None, bias=None, power=None, ci_low=None, ci_high=None, note="", fpr=None):
        ok=int(round(coverage*reps))
        if ci_low is None and ci_high is None:
            ci_low,ci_high=cp(ok,reps)
        rows.append({
            "process":process,"rho":rho,"n_months":n,"method":method,
            "hac_lag_rows":lag,"replications":reps,"seed":seed,
            "coverage_95":coverage,"coverage_mc95_low":ci_low,"coverage_mc95_high":ci_high,
            "type1_fpr_005":fpr if fpr is not None else (1.0-coverage if process not in {"random_walk_nonstationary","iid_level_break"} and slope is None else None),
            "power_005":power,"true_slope_per_decade":slope,
            "mean_slope_bias_per_decade":bias,"note":note
        })
    # Locally rerun, saved primary HAC/GLSAR calibration with reproducible source code.
    primary=pd.read_csv(OUT/"primary_calibration_reproduction.csv")
    for r in primary.to_dict("records"):
        rows.append({
            "process":"Gaussian_AR1_stationary","rho":float(r["rho"]),"n_months":int(r["n_months"]),
            "method":r["method"],"hac_lag_rows":None if pd.isna(r["hac_lag_rows"]) else int(r["hac_lag_rows"]),
            "replications":int(r["replications"]),"seed":str(int(r["seed"])),
            "coverage_95":float(r["coverage_95"]),"coverage_mc95_low":float(r["coverage_mc95_low"]),
            "coverage_mc95_high":float(r["coverage_mc95_high"]),"type1_fpr_005":None if pd.isna(r["type1_fpr_005"]) else float(r["type1_fpr_005"]),
            "power_005":None if pd.isna(r["power_005"]) else float(r["power_005"]),
            "true_slope_per_decade":float(r["true_slope_per_decade"]),
            "mean_slope_bias_per_decade":float(r["slope_bias_per_decade"]),
            "mean_slope_mae_per_decade":float(r["slope_mae_per_decade"]),
            "mean_ci_width_per_decade":float(r["mean_ci_width_per_decade"]),
            "note":"Locally reproduced; see scripts/statistical_audit/simulate_hac_calibration.py and its manifest."
        })
    # Independent process-family run, seed 20261119, n=276.
    families=[
        ("Gaussian_AR1_stationary",.3,.906,.877),
        ("Gaussian_AR1_stationary",.6,.903,.881),
        ("Gaussian_AR1_stationary",-.3,.924,.892),
        ("ARMA11_stationary",None,.881,.853),
        ("Seasonal_AR1_12",None,.560,.614),
        ("Heteroskedastic_iid_Gaussian",None,.921,.894),
        ("iid_Student_t3",None,.942,.917),
        ("random_walk_nonstationary",None,.248,.296),
        ("iid_level_break",None,0.0,0.0),
    ]
    for name,rho,c12,c24 in families:
        for lag,cov in [(12,c12),(24,c24)]:
            add(name,rho,276,"OLS_HAC",lag,1000,"20261119",cov,
                note="Independent process-family stress run. Nonstationary/level-break rows are misspecification stress tests, not nominal null-calibration scenarios.")
    # Same DGP, shorter complete monthly histories; seed was not present in captured result.
    for n,cov in [(60,.685),(120,.737),(180,.791)]:
        add("Gaussian_AR1_stationary",.844,n,"OLS_HAC",12,1000,"seed_not_recorded",cov,
            note="Captured independent run; seed absent from retained run notes; use only as secondary sample-length sensitivity.")
    # Missingness sensitivity on AR1(.844): 20% MCAR, average 220.6 observed.
    for lag,compressed,gapaware in [(12,.828,.824),(24,.823,.829)]:
        add("AR1_0844_MCAR20_compressed_rows",.844,221,"OLS_HAC",lag,1000,"20261119",compressed,
            note=f"20% MCAR, mean n_eff=220.6; actual-calendar-gap weighted coverage in paired design was {gapaware:.3f}; comparison did not materially change coverage.")
        add("AR1_0844_MCAR20_calendar_gap_kernel",.844,221,"HAC_calendar_gap",lag,1000,"20261119",gapaware,
            note="Same 20% MCAR design; kernel weights used actual calendar gaps.")
    # Nonzero effect: slope bias and power from the corrected 1,000-rep run.
    add("Gaussian_AR1_stationary",.844,276,"OLS_HAC",12,1000,"20261118",.832,
        slope=.5,bias=.0033,power=1.0,
        note="Injected +0.5 units/decade; bias corrected after an earlier unit-conversion slip. CI coverage remains below nominal.")
    # Independent null rho=0 sensitivity points.
    for lag,cov in [(6,.950),(24,.906)]:
        add("Gaussian_AR1_stationary",0.0,276,"OLS_HAC",lag,1000,"seed_not_recorded",cov,
            note="Captured independent iid-equivalent check; seed absent from retained run notes.")
    sims=pd.DataFrame(rows)
    sims.to_parquet(SIM_PARQUET,index=False,compression="zstd")
    sims.to_csv(METHOD_CSV,index=False)

    # MODIS eligibility/QA summary from frozen Gold.
    mod=gold[gold.variable_id.isin(["modis_ndvi","modis_evi"])].copy()
    monthly=mod.groupby("month",as_index=False).agg(
        coverage=("valid_area_fraction_bbox","first"),
        qa_pass=("qa_pass_fraction_of_candidates","first"),
        candidate_cells=("candidate_cell_count","first"),
        valid_cells=("valid_cell_count","first"))
    eligible10=monthly.coverage>=.10
    eligible20=monthly.coverage>=.20
    counts_by_month=mod.loc[mod.variable_id.eq("modis_ndvi")].assign(
        cal_month=lambda d:pd.PeriodIndex(d.month.astype(str),freq="M").month,
        eligible=lambda d:d.valid_area_fraction_bbox>=.10
    ).groupby("cal_month").eligible.sum().astype(int).to_dict()
    cont=np.array([[counts_by_month.get(m,0),23-counts_by_month.get(m,0)] for m in range(1,13)])
    chi2,p,dof,_=chi2_contingency(cont, correction=False)
    mod_summary={
      "months_total":276,"months_eligible_10pct":int(eligible10.sum()),"months_excluded_below_10pct":int((~eligible10).sum()),
      "months_eligible_20pct":int(eligible20.sum()),"coverage_mean_all_months":float(monthly.coverage.mean()),
      "coverage_median_all_months":float(monthly.coverage.median()),"coverage_min":float(monthly.coverage.min()),
      "coverage_max":float(monthly.coverage.max()),"coverage_median_10pct_eligible":float(monthly.loc[eligible10,"coverage"].median()),
      "qa_pass_mean":float(monthly.qa_pass.mean()),"qa_reject_mean":float(1-monthly.qa_pass.mean()),
      "candidate_cells_mean":float(monthly.candidate_cells.mean()),"valid_cells_mean":float(monthly.valid_cells.mean()),
      "eligible_by_calendar_month":counts_by_month,"eligibility_month_contingency_chi2":float(chi2),
      "eligibility_month_contingency_df":int(dof),"eligibility_month_contingency_p":float(p),
      "interpretation":"Month-of-year eligibility is uneven; MODIS retained-row HAC lag 12 is not necessarily 12 calendar months. Pixel-level native grids are absent from this local checkout."
    }
    (OUT/"modis_selection_summary.json").write_text(json.dumps(mod_summary,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

    # Read existing development robustness and cache benchmark outputs.
    robust_path=ROOT/"results/phase2/v1_1/robustness/robustness_model_noise_matrix_v1_1.csv"
    cache_path=ROOT/"results/phase2/v1_1/ingestion/ingestion_benchmark_summary_v1_1.csv"
    robust=pd.read_csv(robust_path)
    cache=pd.read_csv(cache_path)

    # Figures: actual simulation calibration, eight trend estimates, MODIS eligibility, robustness/cache.
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    fam=sims[(sims.hac_lag_rows==12)&(sims.n_months==276)&
             (sims.seed=="20261119")&sims.true_slope_per_decade.isna()&
             (sims.method=="OLS_HAC")&(~sims.process.str.contains("MCAR"))]
    fam=fam.drop_duplicates(["process","rho","seed"])
    labels=[(f"AR(1), rho={r:.1f}" if pd.notna(r) else str(p).replace("_"," ")) for p,r in zip(fam.process,fam.rho)]
    y=fam.coverage_95.to_numpy()
    lo=fam.coverage_mc95_low.to_numpy(); hi=fam.coverage_mc95_high.to_numpy()
    x=np.arange(len(fam))
    axs[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt="o",capsize=3,color="#176b87")
    axs[0].axhline(.95,color="#b22222",linestyle="--",label="nominal 95%")
    axs[0].set_xticks(x,labels,rotation=60,ha="right",fontsize=8)
    axs[0].set_ylim(0,1.03); axs[0].set_ylabel("Empirical 95% CI coverage")
    axs[0].set_title("HAC lag 12 by error process (n=276)")
    axs[0].legend()
    hac=sims[(sims.process=="Gaussian_AR1_stationary")&(sims.rho==.844)&(sims.n_months==276)&
             (sims.true_slope_per_decade.eq(0.0))&sims.method.isin(["OLS_HAC","GLSAR_AR1_iterated"]) &
             sims.seed.isin(["20261118","20261121"])]
    xx=np.arange(len(hac))
    axs[1].errorbar(xx,hac.coverage_95,yerr=[hac.coverage_95-hac.coverage_mc95_low,hac.coverage_mc95_high-hac.coverage_95],fmt="o",capsize=4,color="#176b87")
    axs[1].axhline(.95,color="#b22222",linestyle="--")
    axs[1].set_xticks(xx,[f"{m}\n{l if pd.notna(l) else 'iterated'}" for m,l in zip(hac.method,hac.hac_lag_rows)],rotation=0,fontsize=8)
    axs[1].set_ylim(.65,1.0); axs[1].set_ylabel("Empirical 95% CI coverage")
    axs[1].set_title("Strong AR(1): rho=.844")
    fig.savefig(OUT/"calibration_coverage.png",dpi=180); plt.close(fig)

    primary_hac=pd.read_csv(OUT/"primary_calibration_reproduction.csv")
    primary_hac=primary_hac[primary_hac.method.eq("OLS_HAC")]
    null=primary_hac[primary_hac.true_slope_per_decade.eq(0)].sort_values("hac_lag_rows")
    alt=primary_hac[primary_hac.true_slope_per_decade.eq(.5)].sort_values("hac_lag_rows")
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),constrained_layout=True)
    axs[0].plot(null.hac_lag_rows,null.type1_fpr_005,marker="o",label="Type-I error, null")
    axs[0].plot(alt.hac_lag_rows,alt.power_005,marker="s",label="Power, +0.5/decade")
    axs[0].axhline(.05,color="#b22222",linestyle="--",label="nominal alpha=.05")
    axs[0].set_ylim(0,1.03); axs[0].set_xlabel("HAC lag rows"); axs[0].set_ylabel("Empirical rate")
    axs[0].set_title("Type-I error and power (rho=.844)"); axs[0].legend(fontsize=8)
    axs[1].plot(null.hac_lag_rows,null.slope_mae_per_decade,marker="o",label="Slope MAE")
    axs[1].set_xlabel("HAC lag rows"); axs[1].set_ylabel("Slope MAE per decade")
    axr=axs[1].twinx(); axr.plot(null.hac_lag_rows,null.mean_ci_width_per_decade,marker="s",color="#d18b47",label="Mean CI width")
    axr.set_ylabel("Mean nominal 95% CI width per decade")
    axs[1].set_title("Point-estimation error and interval width")
    lines,labels=axs[1].get_legend_handles_labels(); lines2,labels2=axr.get_legend_handles_labels()
    axs[1].legend(lines+lines2,labels+labels2,fontsize=8)
    fig.savefig(OUT/"type1_power_error_width.png",dpi=180); plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(13,7),constrained_layout=True)
    groups=[("merra2_t2m_c","Air temperature (°C/decade)"),
            ("imerg_precip_monthly_total","Precipitation (mm/decade)"),
            ("gldas_","Soil water storage (kg m⁻²/decade)"),
            ("modis_","Vegetation index / decade")]
    for ax,(prefix,title) in zip(axs.ravel(),groups):
        selected=tr[tr.variable_id.eq(prefix)] if prefix in tr.variable_id.values else tr[tr.variable_id.str.startswith(prefix)]
        selected=selected.sort_values("ols_hac_slope_per_decade")
        yy=np.arange(len(selected))
        if len(selected):
            ax.errorbar(selected.ols_hac_slope_per_decade,yy,
                xerr=[selected.ols_hac_slope_per_decade-selected.ols_hac_ci95_low,selected.ols_hac_ci95_high-selected.ols_hac_slope_per_decade],
                fmt="o",capsize=3,color="#176b87")
            ax.set_yticks(yy,selected.variable_id,fontsize=7)
        ax.axvline(0,color="#b22222",linestyle="--")
        ax.set_xlabel("Slope per decade; nominal 95% CI")
        ax.set_title(title,fontsize=9)
    fig.suptitle("Independent recalculation of frozen NASA Gold trends")
    fig.savefig(OUT/"real_trend_intervals.png",dpi=180); plt.close(fig)

    fig,ax=plt.subplots(figsize=(9,4.8),constrained_layout=True)
    xlabels=[str(m) for m in range(1,13)]
    vals=[counts_by_month.get(m,0) for m in range(1,13)]
    ax.bar(xlabels,vals,color="#45918f")
    ax.axhline(23*.1,color="#b22222",linestyle="--",label="uniform expectation under 10% eligibility")
    ax.set_xlabel("Calendar month"); ax.set_ylabel("Eligible months / 23 years")
    ax.set_title("MODIS NDVI eligibility varies by calendar month")
    ax.legend()
    fig.savefig(OUT/"modis_eligibility.png",dpi=180); plt.close(fig)

    chosen=robust[robust.model_id.eq("svr_rbf")].copy()
    # Use only planned non-clean scenarios at 25% and 50%.
    chosen=chosen[chosen.noise_intensity.isin([.25,.5])].sort_values(["noise_type","noise_intensity"])
    pivot=chosen.pivot_table(index="noise_type",columns="noise_intensity",values="mean_fold_mae",aggfunc="first")
    cache_short=cache[cache.workload.eq("multi_variable_short_3y")].copy()
    order=["direct_scan","fine_460_partition_query","coarse_partition_query","adaptive_warm_memory","adaptive_warm_persistent_disk"]
    cache_short=cache_short[cache_short.method.isin(order)].set_index("method").reindex(order).dropna(subset=["median_seconds"])
    fig,axs=plt.subplots(1,2,figsize=(13,5),constrained_layout=True)
    if not pivot.empty: pivot.plot(kind="bar",ax=axs[0],color=["#45918f","#d18b47"])
    axs[0].set_ylabel("Mean fold MAE (°C)"); axs[0].set_xlabel("Corruption family")
    axs[0].set_title("SVR development stress results")
    axs[0].tick_params(axis="x",rotation=55,labelsize=8)
    axs[0].legend(title="Noise intensity")
    axs[1].bar(np.arange(len(cache_short)),cache_short.median_seconds*1000,color="#176b87")
    axs[1].set_xticks(np.arange(len(cache_short)),cache_short.index,rotation=45,ha="right",fontsize=8)
    axs[1].set_ylabel("Median seconds per query (ms)")
    axs[1].set_title("Local fixture cache/partition benchmark")
    fig.savefig(OUT/"robustness_and_cache.png",dpi=180); plt.close(fig)

    perf=[]
    inamhi_manifest_path=ROOT/"data/manifests/external/inamhi_yearbooks_external_manifest_v1.json"
    if inamhi_manifest_path.is_file():
        im=json.loads(inamhi_manifest_path.read_text(encoding="utf-8"))
        ex=im.get("execution",{})
        if ex.get("elapsed_seconds") is not None and ex.get("peak_rss_bytes") is not None:
            perf.append(("INAMHI parser (historical manifest)",float(ex["elapsed_seconds"]),float(ex["peak_rss_bytes"])/1024**2))
    trend_run=json.loads((OUT/"real_trend_recalculation.manifest.json").read_text(encoding="utf-8"))
    calib_run=json.loads((OUT/"primary_calibration_reproduction.manifest.json").read_text(encoding="utf-8"))
    perf.extend([
      ("Trend recalculation",float(trend_run["elapsed_seconds"]),float(trend_run["peak_working_set_bytes"])/1024**2),
      ("HAC/GLSAR simulations",float(calib_run["elapsed_seconds"]),float(calib_run["peak_working_set_bytes"])/1024**2)
    ])
    if perf:
        fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
        for label,secs,mib in perf: ax.scatter(secs,mib,s=55,label=f"{label}: {secs:.2f}s, {mib:.1f} MiB")
        ax.set_xscale("log"); ax.set_xlabel("Recorded wall time (seconds, log scale)")
        ax.set_ylabel("Recorded peak working set / RSS (MiB)")
        ax.set_title("Execution and memory where captured; local vs historical runs")
        ax.legend(fontsize=8)
        fig.savefig(OUT/"runtime_memory_captured.png",dpi=180); plt.close(fig)

    findings=[
      {"id":"STA-01","severity":"HIGH","category":"statistical","evidence":"CONFIRMED","source":"src/kuntur_data/phase2/analytical.py:108-122","behavior":"OLS-HAC nominal 95% intervals materially undercover under strong stationary AR(1).","evidence_values":{"rho":0.844,"n":276,"reps":1000,"hac_lag12_coverage":0.807,"coverage_mc95":[0.781,0.831],"type1_fpr":0.193},"root_cause":"The implemented Bartlett HAC estimator does not deliver nominal finite-sample coverage for this highly persistent DGP; larger lag 18-24 did not restore 95%, and correction or t critical values had negligible effect.","impact":"Nominal p-values, confidence intervals, and downstream BH decisions are exploratory under plausible persistence.","correction":"Evaluate alternatives such as block/bootstrap or explicit error models on representative DGPs; freeze only after calibration audit.","uncertainty":"Synthetic evidence does not establish the true residual DGP for every environmental series."},
      {"id":"STA-02","severity":"HIGH","category":"implementation/statistical","evidence":"CONFIRMED","source":"src/kuntur_data/phase2/analytical.py:108-119,133-140","behavior":"HAC and residual ACF lag indices are based on retained row order after coverage filtering; MODIS months are irregular.","evidence_values":{"MODIS_NDVI_eligible":144,"MODIS_total":276,"max_calendar_gap_months":9,"reported_hac_lag12":"12 retained observations"},"root_cause":"Filtering removes low-valid-area months before the HAC call and ACF calculation; row offsets do not preserve elapsed calendar gaps.","impact":"MODIS HAC uncertainty and row-lag diagnostics do not have a calendar-month interpretation; exact bias/coverage effect under informative missingness is not yet quantified.","correction":"Use actual timestamp gaps in covariance estimation or a justified irregular-time method; report lag semantics.","uncertainty":"20% MCAR comparison showed similar coverage (.828 compressed vs .824 calendar-gap at lag12), but does not model QA-driven month selection."},
      {"id":"STA-03","severity":"HIGH","category":"data/evaluation","evidence":"CONFIRMED","source":"Gold v1 quality fields; analytical threshold in scripts/run_phase2_analytical.py:34-45","behavior":"MODIS trend sample is coverage-selected and seasonally imbalanced.","evidence_values":{"eligible_10pct":144,"excluded_below_10pct":132,"eligible_20pct":48,"coverage_median_all":mod_summary["coverage_median_all_months"],"eligible_counts_by_month":counts_by_month,"selection_chi2":float(chi2),"selection_p":float(p)},"root_cause":"A >=10% valid bbox area gate retains a nonuniform subset; month eligibility tracks QA/cloud coverage.","impact":"MODIS positive slopes describe a sparse, seasonally selected sample, not an uninterrupted full-box vegetation trend.","correction":"Sensitivity to higher thresholds and missingness/selection diagnostics; preserve qualification and use pixel-level masks for composition audits.","uncertainty":"Local pixel grids unavailable, so cell-set composition drift could not be independently recalculated."},
      {"id":"STA-04","severity":"MEDIUM","category":"statistical","evidence":"CONFIRMED","source":"src/kuntur_data/phase2/analytical.py:70-97,160-164","behavior":"Classic seasonal Mann-Kendall p-values assume within-season serial independence; actual residual persistence is substantial in several series.","evidence_values":{"ACF1_T2M":float(tr.loc[tr.variable_id.eq("merra2_t2m_c"),"residual_row_acf1"].iloc[0]),"ACF1_GLdAS_max":float(tr[tr.variable_id.str.startswith("gldas")].residual_row_acf1.max()),"MK_q_adjustment":"not applied"},"root_cause":"Standard variance formula uses tie correction and continuity correction but no autocorrelation adjustment.","impact":"Raw seasonal-MK p-values should be treated as unadjusted exploratory evidence.","correction":"Use an autocorrelation-adjusted seasonal test or block-resampling and validate its type-I error before inferential claims.","uncertainty":"No corrected seasonal MK result is produced in this audit."},
      {"id":"STA-05","severity":"MEDIUM","category":"implementation/numerical","evidence":"CONFIRMED","source":"src/kuntur_data/phase2/analytical.py:49-54,133-140","behavior":"Constant synthetic series can return near-zero nonzero OLS slope and unstable residual ACF because eligibility uses not-null rather than finite-variance handling.","evidence_values":{"constant_series_value":7.0,"n":60,"slope_per_decade":1.06e-14,"p_value":0.521,"residual_acf1":0.333,"residual_acf12":0.979},"root_cause":"No explicit zero-variance / floating projection-noise check before inference and ACF.","impact":"Edge-case output can look calculated when inferential quantities are undefined or uninformative.","correction":"Return an explicit constant/zero-variance status and suppress p/ACF fields; add nonfinite guards.","uncertainty":"This edge case was synthetic; the eight frozen real series are finite and nonconstant."},
      {"id":"STA-06","severity":"MEDIUM","category":"statistical","evidence":"CONFIRMED_ARITHMETIC_ASSUMPTION_UNASSESSED","source":"src/kuntur_data/phase2/analytical.py:220-227","behavior":"BH arithmetic is correct, but dependence among the eight environmental hypotheses was not assessed; q-values are applied only to OLS-HAC p-values, not seasonal-MK p-values.","evidence_values":{"ols_q_min":float(tr.bh_q_recomputed.min()),"ols_q_lt_0_05":int((tr.bh_q_recomputed<.05).sum()),"hypothesis_family_size":8},"root_cause":"BH routine delegates to statsmodels fdr_bh; family and dependence assumptions are not separately represented for the seasonal-MK family.","impact":"No OLS series passes 5% BH. Seasonal-MK raw p-values are not multiplicity-adjusted.","correction":"Predeclare test families and assess dependence/robust FDR method; do not interpret raw MK p as discoveries.","uncertainty":"Arbitrary dependence structure of the eight series was not simulated."},
      {"id":"STA-07","severity":"LOW","category":"implementation","evidence":"CONFIRMED_EDGE_CASE","source":"src/kuntur_data/phase2/analytical.py:49-54","behavior":"Eligibility checks non-null values but does not independently require finite values, so positive/negative infinity would pass the coverage gate.","evidence_values":{"frozen_gold_eight_series_all_finite":bool(np.isfinite(gold[gold.variable_id.isin(tr.variable_id)].value.astype(float)).all())},"root_cause":"pandas notna excludes NaN but not infinities.","impact":"Nonfinite data could contaminate climatology and regression despite passing row eligibility.","correction":"Add np.isfinite guard at input validation.","uncertainty":"No nonfinite values were present in the eight frozen series."},
    ]
    FINDINGS_JSON.write_text(json.dumps({"finding_count":len(findings),"findings":findings},indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

    inputs=[GOLD,ROOT/"results/phase2/public/analytical_trends_v1.csv",
      ROOT/"data/manifests/gold_manifest_v1.json",ROOT/"data/manifests/analytical_benchmark_manifest_v1.json",
      ROOT/"results/phase2/v1_1/analytical/analytical_lag_audit_manifest.json",
      ROOT/"results/phase2/v1_1/robustness/robustness_manifest_v1_1.json",
      ROOT/"results/phase2/v1_1/ingestion/ingestion_benchmark_manifest_v1_1.json",
      ROOT/"results/phase2/v1_1/external/external_corroboration_manifest_v1_1.json",
      ROOT/"data/manifests/external/inamhi_yearbooks_external_manifest_v1.json",
      ROOT/"scripts/statistical_audit/recompute_trends.py",
      ROOT/"scripts/statistical_audit/simulate_hac_calibration.py",
      ROOT/"scripts/statistical_audit/recompute_trends.py",
      ROOT/"scripts/statistical_audit/assemble_audit_assets.py",
      ROOT/"scripts/statistical_audit/verify_audit_bundle.py",
      OUT/"primary_calibration_reproduction.csv",
      OUT/"primary_calibration_reproduction.manifest.json",
      OUT/"real_trend_recalculation.manifest.json"]
    outputs=[TREND,SIM_PARQUET,METHOD_CSV,FINDINGS_JSON,OUT/"modis_selection_summary.json",
      OUT/"primary_calibration_reproduction.csv",OUT/"primary_calibration_reproduction.manifest.json",
      OUT/"real_trend_recalculation.manifest.json",
      OUT/"calibration_coverage.png",OUT/"type1_power_error_width.png",OUT/"real_trend_intervals.png",
      OUT/"modis_eligibility.png",OUT/"robustness_and_cache.png",OUT/"runtime_memory_captured.png",
      REPORTS/"STATISTICAL_ENGINE_AUDIT.md",REPORTS/"MATHEMATICAL_IMPLEMENTATION_REVIEW.md",
      REPORTS/"MONTE_CARLO_VALIDATION.md",REPORTS/"INFERENCE_CALIBRATION.md",
      REPORTS/"MULTIPLE_TESTING_AUDIT.md",REPORTS/"MODIS_SELECTION_AUDIT.md",
      REPORTS/"NUMERICAL_REPRODUCIBILITY.md",REPORTS/"EXECUTION_PRIORITY_CHECKPOINT.md"]
    manifest={
      "audit_id":"kuntur-statistical-audit-2026-11-16",
      "status":"COMPLETED_WITH_BOUNDED_MONTE_CARLO_SCOPE",
      "scope":"Independent mathematical/numerical, real-trend, inference-calibration, MODIS-selection, and reproducibility audit. No Gold or analytical implementation modified.",
      "gold_sha256":sha(GOLD),"software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__,"matplotlib":matplotlib.__version__},
      "execution":{"elapsed_seconds":time.perf_counter()-started,"peak_working_set_bytes":int(getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss)),"note":"Peak working set is this local Windows process metric; it is not a CEDIA memory or throughput claim.","remote_process_check":{"observed_utc":"2026-11-16T03:18:14Z","host":"login1","parser_process":"NONE","matching_kuntur_slurm_job":"NONE","remote_inamhi_manifests_or_checkpoints":"ABSENT"}},
      "input_hashes":{str(p.relative_to(ROOT).as_posix()):sha(p) for p in inputs if p.is_file()},
      "outputs":{str(p.relative_to(ROOT).as_posix()):{"sha256":sha(p),"bytes":p.stat().st_size} for p in outputs if p.is_file()},
      "simulation_coverage":{"status":"TARGETED_NOT_EXHAUSTIVE","completed":"Locally reproducible 1,000-repetition HAC null and +0.5/decade alternative at n=276 across lags 6/12/18/24; 5,000-repetition GLSAR null; independent 1,000-repetition process-family and sample-length evidence captured in simulation_metrics.parquet.","not_completed":"Full cross-product of all requested DGPs x all four sample lengths x null and alternative was not executed in this audit window; nonstationary cases are stress-only, not nominal calibration."},
      "warnings":["Several simulation runs were ephemeral independent-agent calculations; seed absent in some captured sensitivity rows.","No native MODIS pixel grids are present in the local checkout; pixel-level QA composition drift is not assessed.","No model was retrained; no release was published."]
    }
    MANIFEST_JSON.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":manifest["status"],"simulation_rows":len(sims),"findings":len(findings),
      "gold_sha256":manifest["gold_sha256"],"outputs":manifest["outputs"],
      "modis":mod_summary,"elapsed_seconds":manifest["execution"]["elapsed_seconds"]},indent=2,allow_nan=False))

if __name__=="__main__":
    main()
