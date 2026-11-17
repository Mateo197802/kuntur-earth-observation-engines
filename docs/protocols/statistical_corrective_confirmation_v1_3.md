# KUNTUR v1.3 Independent Confirmation — IID Gaussian Candidate

**Status:** frozen design for a single independent confirmation. This confirmation is authorized by the user's supplied KUNTUR v1.3 execution objective only after the candidate freeze manifest is written and its hashes verified.

## Frozen candidate and scope

The completed v1.3 paired development expansion found a single development candidate for the IID Gaussian regime: `SARIMAX_SEASONAL_AR12`. No method was adequately supported for strong AR(1), ARMA(1,1), seasonal AR(1,12), AR(1)+20% MCAR, or AR(1)+seasonal-calendar missingness. This confirmation tests only the IID Gaussian candidate; it does not retest or promote a candidate for any failed regime.

The method source remains the exact hashed implementation in `src/kuntur_data/phase2/statistical_v12.py` that generated the development evidence. Its model is a regression with intercept, linear trend in units per decade, and eleven calendar-month indicators, with SARIMAX errors `order=(0,0,0)`, `seasonal_order=(1,0,0,12)`, no additional trend, stationarity and invertibility constraints enabled, and optimizer limit 200. It reports the fitted covariance-based two-sided 95% Wald interval and the two-sided asymptotic-normal Wald p-value for the trend coefficient. The exact module hash is recorded in the machine-readable freeze and run manifests.

## Simulation and independent seeds

- 276 complete consecutive monthly observations, January 2000 through December 2022.
- IID Gaussian errors from the existing frozen simulation generator; its source SHA-256 is frozen before execution.
- Trends: 0, +0.25, and +0.50 units per decade.
- 1,000 fresh outer replications per slope cell (3,000 fits total).
- There is one frozen candidate; no candidate search or tuning is performed on confirmation results.
- Seed namespace: `KUNTUR-STAT-CORRECTIVE-INDEPENDENT-CONFIRMATION-V1.3-2027-01-15`.
- Seeds use the same explicit 128-bit SHA-256 derivation as the v1.3 development expansion but a separate reserved namespace. The namespace did not occur in prior result/checkpoint records; only reservation text was present before the freeze. The run records every series and estimator seed and verifies within-run uniqueness.
- Checkpoints are atomic and committed every 50 replications. A resume is allowed only for matching frozen source/configuration hashes and a contiguous verified checkpoint prefix.

## Acceptance gates

The gates are unchanged from v1.2 and from the v1.3 development protocol:

1. Null Type-I rate at alpha .05 is in [.035,.065], and its exact two-sided 95% Clopper–Pearson interval contains .05.
2. Coverage of the nominal 95% interval is in [.935,.965] at each of the null, +.25, and +.50 slopes, and each exact binomial interval contains .95.
3. Absolute mean slope bias is at most .025 units per decade at each slope.
4. Numerical failures satisfy the integer gate `100 * failures <= supported`; inferential output availability is at least .99.
5. Residual dependence failures are at most .05 under the predeclared Ljung–Box diagnostic at lags 12 and 24 months, with p<.01 flagged.
6. Mean interval width is at most twice the known-covariance oracle width in every slope cell.

All gates must pass in every declared slope cell. The confirmation report also gives power, MAE, RMSE, point estimates, Monte Carlo standard errors, and uncertainty intervals; no candidate is chosen or tuned based on those results.

## Resource budget

The completed development expansion used 18,423.578 process-CPU seconds for 180,000 fits, or 0.102353 seconds per fit. The 3,000-fit confirmation is estimated at 307.06 CPU seconds. A 2x planning reserve is 614.12 CPU seconds. Hard caps are 3,600 CPU seconds (1 CPU hour), 7,200 active wall seconds (2 hours), 1 GiB RSS, one process, one BLAS thread, and zero SLURM jobs. The runner checkpoints before stopping at any cap.

## Release and inference scope

Until all independent gates pass, the method remains unvalidated and no registry entry is created. If it passes, a new v1.3 registry file may record only the exact IID synthetic regime, 276 complete months, and the frozen method. The historical v1.2 registry is never edited. Even a pass does not classify a real environmental series as IID Gaussian; the engine must return descriptive results unless the caller declares a regime and applicable diagnostics support the use. There are no confirmatory real-series trends or FDR-adjusted findings in this protocol.

## Reproduction

From the repository root:

```powershell
python scripts/statistical_correction/freeze_corrective_selection_v1_3_r1.py --verify
python scripts/statistical_correction/run_corrective_confirmation_v1_3.py --plan-only
python scripts/statistical_correction/run_corrective_confirmation_v1_3.py --execute
python scripts/statistical_correction/summarize_corrective_confirmation_v1_3.py
```

The `--execute` command refuses to run until the selection freeze manifest and all source hashes match. The summarizer validates checkpoint content before reporting any rates. It never alters the v1.2 closure, v1.2 registry, NASA Gold, historical scores, or the consumed 2025 hindcast.
