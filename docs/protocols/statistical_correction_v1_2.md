# KUNTUR v1.2 Statistical Correction Protocol

Status: frozen before running the v1.2 independent validation seed. The historical v1 audit and round-four exploratory calibration are preserved and are not overwritten.

## Frozen inputs and boundaries

- Frozen Gold SHA-256: `5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2`.
- Frozen v1 Gold manifest SHA-256: `c5bdac892d92670c3c52e2cf2fe268e00890e86b0e3cdbe07d5a040f139c5543`.
- Forecast split files and manifests remain immutable; their hashes are recorded by the correction manifest.
- The original `analytical.py`, trend CSV, and v1 statistical audit remain historical references. v1.2 is a separate engine and output family.
- No NASA real-series output is opened for method selection. Synthetic simulations are the only selection evidence. No method is tuned to make real trends significant.
- Candidate methods are accessed through a common result schema. Until independent calibration passes, public p-values and confidence intervals are withheld.

## Methods locked for comparison

1. Frozen OLS-HAC with row bandwidths 6, 12, 18, and 24 as a reference.
2. Calendar-distance Bartlett HAC, using actual month-ordinal differences.
3. Iterated feasible GLS AR(1), complete regular monthly support only.
4. State-space SARIMAX AR(1), with calendar gaps retained as missing observations.
5. State-space SARIMAX ARMA(1,1).
6. State-space seasonal AR(1,12).
7. Circular moving residual-block bootstrap, block lengths 6, 12, and 24 months.
8. Calendar-aligned moving year-block residual bootstrap, blocks of 2 and 3 years.
9. Dependence-aware seasonal Mann–Kendall p-value by null-constrained moving year-block residual resampling. It returns no confidence interval.

The v1.2 implementation begins with an empty validated-regime registry. No candidate method is designated inferentially valid merely because its optimizer converges or its real-data p-value is small.

GLSAR convergence requires Statsmodels to stop before the 100-iteration cap and return finite, stationary AR(1) coefficients. Reaching the iteration cap is a numerical failure even if the last parameter vector is finite. State-space convergence warnings are retained with their warning category and text.

## Monte Carlo design

The existing round-four run (selection seed 20261123) is a historical exploratory pilot. Its bootstrap p-values were interval-crossing proxies and its bootstrap outer replication count was 500 despite a 1,000 request; those bootstrap p-values are excluded from selection and validation.

- Corrected method-selection seed base: `20261126`.
- Independent final-validation seed base: `20261127`.
- The final seed must not be used to change methods, lag/block lengths, or thresholds.
- Selection uses 500 outer replications on n=276 for IID Gaussian, AR(1) rho .6/.844, seasonal AR(1,12), ARMA(1,1), and AR(1) rho .6 with 20% MCAR. Selection applies the locked candidate set using the per-DGP calibration gates and deterministic score specified below; it may not inspect real NASA outcomes. Selection results are never reported as final validation.
- All DGP innovations have documented marginal standard deviation 0.25 unless a DGP explicitly defines another scale; trend and calendar-month effects are fixed by the generator.
- Primary length is 276 months. The primary null DGPs are IID Gaussian, AR(1) rho .844, ARMA(1,1), seasonal AR(1,12), and AR(1) rho .6 with 20% MCAR gaps. Each applicable primary method/DGP uses at least 1,000 outer replications.
- Secondary n=276 null DGPs are AR(1) rho −0.3, 0.3, and 0.6; heteroscedastic Gaussian; Student-t3; seasonally patterned calendar missingness; and a level-break stress process. Use 1,000 outer replications for non-likelihood methods and record any reduced expensive-model count as incomplete. A random walk is an additional nonstationary stress case, not a stationary zero-slope null.
- Power and interval coverage use slopes 0, +0.1, +0.25, +0.5, and −0.5 units per decade. At n=276, the primary five DGPs use 1,000 outer replications at each effect for applicable candidates. The n=60, 120, and 180 secondary subset uses AR(1) rho .6 and .844 plus seasonal AR(1,12), with slopes 0, +0.25, and +0.5.
- Nested residual bootstraps use 199 inner draws during selection and 399 during independent final validation. Their p-value resolution and inner-bootstrap error are reported. If the predeclared replication counts or numerical precision are inadequate, the affected regime remains `NOT_VALIDATED`; reducing draws after looking at the final seed is prohibited.
- Fixed configurations, generators, all seeds, per-scenario terminal status, wall/CPU time, peak RSS, and content hashes are recorded. Each completed scenario is atomically checkpointed.

Structural breaks and random walks are stress tests. Their results are not interpreted as Type-I error for a stationary linear-trend null.

## Frozen selection rule and execution labels

Selection is applied separately to each of the five primary DGPs. A candidate is eligible only if it has 100% structural support, at most 1% numerical failure, null Type-I error in 0.035–0.065 with its exact 95% binomial interval containing 0.05, and (when it returns intervals) null and +0.25-slope coverage each in 0.935–0.965 with their exact Monte Carlo intervals containing 0.95, bias at most 0.025 units per decade, and mean interval width no more than twice the known-covariance oracle. If no candidate passes, no method is selected for that DGP. Passing candidates are ordered by the sum of standardized absolute deviations from nominal Type-I error, null/alternative coverage and zero bias; ties are ordered by mean interval-width/oracle ratio and then method identifier. The resulting selection file is hashed before final-validation outputs are inspected. Final validation evaluates every locked candidate; only the preselected method can be promoted for the corresponding DGP, and only if it independently passes all applicable gates. No real NASA outcomes enter this rule.

Selection-only rows are labeled `SELECTION_ONLY_NOT_VALIDATED`; structurally unsupported rows are labeled `UNSUPPORTED_REGIME`. A smoke run is stored separately and is never included in selection or validation outputs. Partial phase runs must identify their exact scenario coverage and cannot be labeled complete. Inferential output must be available for at least 99% of structurally supported replications.

For fitted AR/ARMA/GLSAR dependence models on complete regular support, a residual-dependence gate is predeclared: at most 5% of replications may reject Ljung–Box at either calendar lag 12 or 24 at p<0.01. Missing-support state-space scenarios use exact-calendar pairwise ACF diagnostics; row-order Ljung–Box is not used. Block-resampling methods do not require white residuals and are not subjected to that whitening gate.

## Acceptance gates fixed before final validation

For each claimed primary null regime with 1,000 outer replications:

- Empirical Type-I error must be 0.035–0.065 and its exact two-sided 95% Clopper–Pearson interval must contain 0.05.
- For confidence-interval methods, empirical 95% coverage must be 0.935–0.965 and its exact two-sided 95% interval must contain 0.95 under both null and injected-effect scenarios.
- Alternative slope bias must be no greater than 0.025 units per decade for the declared effects; report MAE and RMSE regardless of the gate.
- Optimizer/numerical failure rate must be at most 1% for a claimed regime. Convergence alone is insufficient.
- Mean interval width must not exceed twice the exact fixed-design Gaussian GLS-oracle width computed from the known DGP covariance and actual observation mask. If the DGP has no finite stationary covariance, this width gate is not applicable and the case is stress-only.
- A method is valid only in regimes that pass all applicable gates on the independent seed and show no material residual dependence misspecification. Passing one DGP does not establish a general method.
- For a method with no interval, only its applicable marginal-test calibration and slope-estimation metrics are evaluated; no interval coverage claim is made.

Monte Carlo intervals are reported; a point estimate alone is not a pass. Comparisons across methods use paired DGP draws and paired t confidence intervals (df = paired n − 1) for metric differences; all per-replication outcomes are retained so the comparisons can be reproduced. These comparative intervals are descriptive and are not multiplicity-adjusted. No bandwidth or method is selected by a favorable environmental result.

## Missingness and diagnostics

- Missing calendar months remain on the monthly index. No silent imputation is allowed.
- State-space models receive the full monthly endogenous vector with `NaN` at missing positions.
- GLSAR and block methods requiring complete support must report `IRREGULAR_TIME_SUPPORT` instead of compressing the series.
- Calendar ACF pairs are matched at exact calendar offsets. Ljung–Box and Durbin–Watson are emitted only for complete regular support; no compressed-row Ljung–Box is used for MODIS.
- MODIS QA coverage thresholds remain the frozen v1 policy. Eligibility counts by calendar month and valid-area distribution are reported as selection diagnostics.
- Historical classic seasonal-MK p-values remain descriptive reference outputs. Corrected multiple-testing q-values are emitted only if their marginal p-values pass calibration. Benjamini–Yekutieli is used for arbitrary dependence; it cannot repair invalid marginal tests.
- Seasonal-MK null resampling removes the fitted linear trend and calendar-month means before drawing calendar-aligned year residual blocks; the observed seasonal S statistic is compared with the resulting no-trend null distribution. The bootstrap preserves month-specific residual dependence and the missingness pattern within sampled calendar-year blocks.

## Reproduction and versioning

The v1.2 module is `src/kuntur_data/phase2/statistical_v12.py`; the historical v1 implementation is unchanged. Protocol and code hashes, frozen input hashes, environment versions, scenario outputs, and validation results are stored under `results/statistical_correction/`. No old audit or frozen NASA artifact is overwritten.
