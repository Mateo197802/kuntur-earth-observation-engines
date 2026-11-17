from __future__ import annotations
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import statsmodels
import statsmodels.api as sm
import psutil
from scipy.stats import beta

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"results/statistical_audit/primary_calibration_reproduction.csv"
SEASONAL=np.tile(np.array([0.0,.5,1.0,.7,.3,-.1,-.4,-.3,-.1,.1,.3,.2]),23)
N=276
ELAPSED=np.arange(N,dtype=float)/12.0
PERIOD=pd.period_range("2002-01",periods=N,freq="M")
DUMMIES=pd.get_dummies(pd.Series(PERIOD.month).astype("category"),drop_first=True,dtype=float).to_numpy()
X=np.column_stack([np.ones(N),ELAPSED,DUMMIES])
LAGS=(6,12,18,24)
def cp(k,n):
    lo=0.0 if k==0 else float(beta.ppf(.025,k,n-k+1))
    hi=1.0 if k==n else float(beta.ppf(.975,k+1,n-k))
    return lo,hi
def errors(rng,rho,sigma=.25):
    eps=rng.normal(0.0,sigma,N)
    out=np.empty(N)
    out[0]=eps[0]
    scale=np.sqrt(1.0-rho*rho)
    for t in range(1,N): out[t]=rho*out[t-1]+scale*eps[t]
    return out
def run_hac(reps=1000,seed=20261118):
    rng=np.random.default_rng(seed)
    out=[]
    for true_slope in (0.0,.5):
        covered={lag:0 for lag in LAGS}; rejected={lag:0 for lag in LAGS}
        estimates={lag:[] for lag in LAGS}; widths={lag:[] for lag in LAGS}
        for _ in range(reps):
            y=SEASONAL+(true_slope/10.0)*ELAPSED+errors(rng,.844)
            for lag in LAGS:
                fit=sm.OLS(y,X).fit(cov_type="HAC",cov_kwds={"maxlags":lag,"use_correction":True})
                slope=float(fit.params[1]*10)
                ci=np.asarray(fit.conf_int(alpha=.05)[1])*10
                p=float(fit.pvalues[1])
                estimates[lag].append(slope)
                widths[lag].append(float(ci[1]-ci[0]))
                covered[lag]+=int(ci[0]<=true_slope<=ci[1])
                rejected[lag]+=int(p<.05)
        for lag in LAGS:
            low,high=cp(covered[lag],reps)
            out.append({"method":"OLS_HAC","rho":.844,"n_months":N,"true_slope_per_decade":true_slope,
                "hac_lag_rows":lag,"replications":reps,"seed":seed,
                "coverage_95":covered[lag]/reps,"coverage_mc95_low":low,"coverage_mc95_high":high,
                "type1_fpr_005":rejected[lag]/reps if true_slope==0 else None,
                "power_005":rejected[lag]/reps if true_slope!=0 else None,
                "mean_slope_per_decade":float(np.mean(estimates[lag])),
                "slope_bias_per_decade":float(np.mean(estimates[lag])-true_slope),
                "slope_mae_per_decade":float(np.mean(np.abs(np.asarray(estimates[lag])-true_slope))),
                "mean_ci_width_per_decade":float(np.mean(widths[lag]))})
    return out
def run_glsar(reps=5000,seed=20261121):
    rng=np.random.default_rng(seed)
    covered=rejected=0; widths=[]
    rhos=[]; slopes=[]
    for _ in range(reps):
        y=SEASONAL+errors(rng,.844)
        fit=sm.GLSAR(y,X,rho=1).iterative_fit(maxiter=100)
        ci=np.asarray(fit.conf_int(alpha=.05)[1])*10
        widths.append(float(ci[1]-ci[0]))
        p=float(fit.pvalues[1])
        covered+=int(ci[0]<=0.0<=ci[1]); rejected+=int(p<.05)
        rhos.append(float(np.asarray(fit.model.rho).reshape(-1)[0]))
        slopes.append(float(fit.params[1]*10))
    low,high=cp(covered,reps)
    return {"method":"GLSAR_AR1_iterated","rho":.844,"n_months":N,"true_slope_per_decade":0.0,
      "hac_lag_rows":None,"replications":reps,"seed":seed,"coverage_95":covered/reps,
      "coverage_mc95_low":low,"coverage_mc95_high":high,"type1_fpr_005":rejected/reps,
      "power_005":None,"mean_slope_per_decade":float(np.mean(slopes)),
      "slope_bias_per_decade":float(np.mean(slopes)),"slope_mae_per_decade":float(np.mean(np.abs(slopes))),
      "mean_ci_width_per_decade":float(np.mean(widths)),
      "mean_rho_hat":float(np.mean(rhos)),"sd_rho_hat":float(np.std(rhos,ddof=1))}
def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def main():
    start=time.perf_counter()
    rows=run_hac()
    rows.append(run_glsar())
    df=pd.DataFrame(rows)
    df.to_csv(OUT,index=False)
    manifest={"status":"REPRODUCED_PRIMARY_CALIBRATION","seed_hac":20261118,"reps_hac_per_effect":1000,
      "seed_glsar":20261121,"reps_glsar":5000,"dgp":"Seasonal fixed pattern plus stationary Gaussian AR(1), rho=.844, stationary SD=.25; independent null and +.5/decade alternatives for HAC.",
      "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"statsmodels":statsmodels.__version__},
      "output_sha256":sha(OUT),"output_bytes":OUT.stat().st_size,"elapsed_seconds":time.perf_counter()-start,
      "peak_working_set_bytes":int(getattr(psutil.Process().memory_info(),"peak_wset",psutil.Process().memory_info().rss))}
    (OUT.with_suffix(".manifest.json")).write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
