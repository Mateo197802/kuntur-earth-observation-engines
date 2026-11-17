# KUNTUR v1.3.1 — IID Seasonal AR(12) Interval Comparison

**Status:** frozen development protocol; no inference registry change. This work follows the completed v1.3 development expansion and preserves the failed v1.3 independent confirmation as historical evidence.

## Question and scope

The frozen `SARIMAX_SEASONAL_AR12` method passed its null-cell gates but its untouched v1.3 confirmation covered only 920/998 injected `+0.50` slopes (92.18%; exact 95% interval 90.34–93.77%), below the frozen 93.5% coverage floor. This comparison asks whether that shortfall is consistent with the interval approximation used by the same point-estimation model. It does not retune on the consumed confirmation rows and makes no claim for other error regimes or real environmental series.

The development design uses the existing 276-month complete IID Gaussian DGP, from January 2000 through December 2022, at slopes 0, +0.25, and +0.50 per decade. Each replicate's generated series is shared across all three candidates. Each candidate independently fits the same SARIMAX model: exogenous intercept, elapsed decades and eleven month indicators; order `(0,0,0)`; seasonal order `(1,0,0,12)`; no additional trend; stationarity and invertibility enforced; maximum 200 iterations.

## Predeclared interval candidates

1. **Wald normal comparator:** the unchanged two-sided 95% normal interval and normal Wald p-value from the unrestricted fit.
2. **Finite-sample t candidate:** same estimate and standard error; Student-t critical value and p-value with `df = n - rank(X) - k_dynamic`. For this design, `n=276`, `rank(X)=13`, and one seasonal AR parameter is estimated, so `df=262`. The innovation scale parameter is not subtracted. The interval and test use the same statistic and degrees of freedom.
3. **Profile-likelihood candidate:** a two-sided 95% likelihood-ratio interval and test from the same stationary Gaussian seasonal-AR(12) model. For each fixed slope, the likelihood profiles the intercept and monthly means by GLS, the seasonal AR coefficient by bounded scalar optimization, and innovation variance analytically. It inverts the same one-degree-of-freedom likelihood-ratio test, with the chi-square 95th percentile as cutoff. The exact likelihood calculation is tested against Statsmodels' SARIMAX likelihood before execution.

The point-estimation model, generated observations, diagnostic residuals, true covariance oracle, and replication IDs are identical across candidates. The Wald comparator is not eligible for a corrected-method freeze.

## Measures and acceptance

At the null slope, report Type-I error and exact Clopper–Pearson 95% interval. At every slope, report interval coverage and its exact interval, slope bias with Student-t Monte Carlo interval, power with exact interval, mean interval width and width/oracle ratio, convergence and inference availability, Ljung–Box residual-dependence failure rate, and profile-likelihood optimization diagnostics. The Ljung–Box checks use lags 12 and 24, with `p < .01` as a failure.

A corrected candidate must pass all applicable unchanged v1.3 gates in every slope cell: null Type-I between .035 and .065 with the exact interval containing .05; coverage between .935 and .965 with the exact interval containing .95 at all three slopes; absolute slope bias no more than .025 per decade; no more than 1% numerical failures under the integer count rule; at least 99% inferential availability; mean CI width no more than twice the known-covariance oracle; and residual-dependence failures no more than 5%. Power and all other Monte Carlo uncertainty are reported; no post-hoc power threshold is added.

Selection is based on all candidate gates across all three slopes. If only one corrected candidate passes, it is eligible for a freeze. If both pass, the predeclared lower-complexity tie-break selects the finite-sample t method for independent confirmation. Development eligibility is not validation and cannot change the statistical inference registry. If neither passes, the result is `NOT_VALIDATED` and confirmation is not started.

## Resource and checkpoint plan

Before the development run, execute 30 paired series per slope in a separate timing-only namespace: 90 series, 270 candidate fits. Preflight has hard limits of 1,800 CPU seconds, 3,600 active wall seconds and 1 GiB RSS. It contributes no scientific metrics. From measured per-candidate runtime, project the 9,000 candidate-replicates (3 methods × 3 slopes × 1,000) and apply a two-fold reserve. Start the main screen only if the two-fold CPU and wall projections fit within the frozen 7,200 CPU-second and 14,400 active-wall-second caps. One process, one BLAS thread, one GiB RSS, no SLURM job.

The development run writes an atomic checkpoint after each 50 paired series per slope, preserving replicate rows, series seeds, diagnostics, warnings, code/config hashes, and cumulative resources. Resume accepts only a sequential verified prefix whose source hashes and cell configuration match the freeze. The resource cap is checked after a complete three-candidate paired replicate so checkpoint evidence remains paired.

## Seeds and protected evidence

Preflight namespace: `KUNTUR-STAT-IID-INTERVAL-COMPARISON-V1.3.1-TIMING-PREFLIGHT-2026-11-17`.

Development namespace: `KUNTUR-STAT-IID-INTERVAL-COMPARISON-V1.3.1-DEV-2026-11-17`.

The frozen v1.2 validation, earlier v1.3 608-fit pilot, 180,000-fit development expansion, consumed v1.3 confirmation, 2025 predictive test, real Gold observations, and inference registry are excluded from seed reuse, method tuning and selection. The existing independent-confirmation failure and its source files are hash-protected in the pre-execution freeze.

## Reproduction

From the repository root, after the protocol, estimator, runner, summarizer and tests are complete:

```powershell
python -m pytest -q tests/test_interval_comparison_v13_1.py
python scripts/statistical_correction/freeze_interval_comparison_v1_3_1.py
python scripts/statistical_correction/run_interval_comparison_v1_3_1.py --phase preflight --plan-only
python scripts/statistical_correction/run_interval_comparison_v1_3_1.py --phase preflight
python scripts/statistical_correction/summarize_interval_comparison_v1_3_1.py --phase preflight
```

Run the main development phase only if the verified preflight summary says `PROCEED`:

```powershell
python scripts/statistical_correction/run_interval_comparison_v1_3_1.py --phase development --plan-only
python scripts/statistical_correction/run_interval_comparison_v1_3_1.py --phase development
python scripts/statistical_correction/summarize_interval_comparison_v1_3_1.py --phase development
```

Do not create an independent-confirmation plan unless a corrected candidate passes every development gate, and do not add any candidate to the inference registry without separate independent evidence.
