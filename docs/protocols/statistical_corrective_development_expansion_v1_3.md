# KUNTUR Statistical Corrective Development Expansion v1.3

**Status:** Frozen before simulation. Development only. This is a separately versioned expansion of the completed 608-fit engineering pilot. It does not amend or reopen the v1.2 decisions, validation results, registry, real-series analysis, or 2025 hindcast.

## Objective and protected evidence

The experiment investigates why the v1.2 frozen selection procedure preferred two-year seasonal Mann–Kendall (Seasonal MK) for IID Gaussian and AR(1) rho=.6 with 20% MCAR missingness, then failed independent Type-I calibration. It compares a small fixed set of alternative dependence treatments under independent, strongly persistent, mixed ARMA, seasonal, and irregular-calendar conditions. The existing v1.2 final-validation seeds, the v1.3 608-fit pilot seeds, real NASA outcomes, and the consumed 2025 predictive test are not used. Every v1.2 selection and validation record remains immutable.

## Paired simulation design

- Series length is 276 consecutive calendar months. The frozen generator is reused without modification.
- Six predeclared regimes: IID Gaussian; AR(1) rho=.844; ARMA(1,1); seasonal AR(1,12); AR(1) rho=.6 with 20% MCAR gaps; and AR(1) rho=.6 with seasonal calendar gaps.
- Each regime uses slopes 0, +.25, and +.50 units per decade. For each regime/slope/replicate, every candidate receives the same generated series. Estimator/bootstrap streams are separate and deterministic.
- Each method-regime-slope cell has 1,000 outer replications. This yields 60 method-regime combinations, 180 cells, and 180,000 planned fits. Block bootstrap methods use 399 inner draws.
- Checkpoints are atomic and written every 50 replicates and at cell completion. A run resumes only from exact matching config, protocol, script, source-module, generator, and dependency hashes. Completed replicates in a verified cell are never recalculated. A single-process lock prevents simultaneous writers.
- Fresh development namespace: KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-EXPANSION-V1.3-2026-11-17. The previous engineering pilot has its own 2026-11-16 namespace. The reserved independent-confirmation namespace is not read to derive seeds or consumed by this run.

## Fixed candidates

Complete-calendar regimes compare Seasonal MK year blocks 1–4; OLS HAC lags 6 and 12; feasible GLS AR(1); SARIMAX AR(1), ARMA(1,1), and seasonal AR(1,12); and the seasonal year-block bootstrap (2-year blocks). Irregular-calendar regimes compare Seasonal MK year blocks 1–4; exact-calendar-distance OLS HAC lag 12; and the same three state-space models. Methods that require complete data are not compressed across gaps. Seasonal MK methods have no confidence interval in this implementation and are retained as p-value/slope comparators only.

## Outcomes and Monte Carlo uncertainty

For every cell, retain replicate-level convergence, inferential availability, slope, p-value, confidence interval, interval width, and method residual diagnostics. Also retain exact-calendar residual ACF and pair counts, plus known-DGP error ACF. For complete support, report Ljung–Box tests at calendar lags 12 and 24; a p-value below .01 is a residual-dependence failure. For irregular support, report pairwise exact-calendar ACF by lag and pair count as descriptive diagnostics only; do not call these calibrated tests or silently compress months.

At slope 0, report the Type-I rejection rate with exact two-sided 95% Clopper–Pearson interval. At all slopes, report interval coverage and exact intervals where applicable, convergence/failure rates, inferential availability, mean interval width and ratio to the known-covariance oracle, slope bias/MAE/RMSE, residual diagnostics, and power at +.25 and +.50. Continuous means include Student-t Monte Carlo intervals. Report paired method differences using replicate-matched uncertainty intervals against SARIMAX AR(1). Missing confidence intervals or dependence tests remain not applicable/unresolved rather than passes.

## Predeclared development gate and selection rule

The v1.2 independent acceptance criteria remain unchanged: Type-I rate in [.035,.065] and exact interval containing .05; coverage in [.935,.965] and exact interval containing .95 at null and both positive slopes; absolute slope bias no greater than .025; numerical failures at most 1% using the integer comparison 100 × failures ≤ supported; inferential availability at least 99%; mean interval width no more than twice the oracle; and residual-dependence failures at most 5% for an applicable calibrated diagnostic.

Selection is per declared regime. Only a method that produces intervals and passes every applicable gate across all three slopes is eligible as a full-inference candidate. If the residual-dependence gate is not calibrated for an irregular-calendar diagnostic, no candidate may be declared supported for that regime on the basis of descriptive ACF alone. If multiple candidates pass in a regime, the frozen tie-breaker is highest minimum power at +.25 and +.50, followed by lowest mean oracle-normalized width. This is a development choice only; the chosen code/configuration/gates must then be frozen before independent confirmation. No candidate is added to the inference registry from these results.

## Measured resource budget

Prior v1.3 engineering pilot: 608 fits, 54.516 process-CPU seconds, 54.706 wall seconds, and 188,235,776-byte peak RSS. Completed v1.2 validation: 160,000 fits and 11,748.5 process-CPU seconds. The expansion estimate uses the slower measured rate, 54.516/608 = 0.08966 CPU seconds/fit, projecting 16,140 seconds (4.483 CPU hours) for 180,000 fits. A 2× reserve produces a hard cap of 9 CPU hours. Hard limits are 10 active wall hours, 1 GiB RSS, one process, one BLAS thread, and zero SLURM jobs. The cap may be exceeded only by the duration of the single fit already underway when a limit is reached; that fit is checkpointed before stopping. Progress and accumulated resource use are written at each 50-replicate checkpoint. No automatic full-plan continuation occurs after a cap.

## Confirmation remains a separate authorization

The untouched namespace reserved for a later confirmation is KUNTUR-STAT-CORRECTIVE-INDEPENDENT-CONFIRMATION-V1.3-2027-01-15. It remains unused. Confirmation may be proposed only after development identifies a candidate, the candidate implementation, regime scope, acceptance gates, and confirmation seed derivation are frozen. The confirmation proposal uses 1,000 fresh outer replicates for each null and each declared effect cell, with method-paired series, exact intervals, matching diagnostics, and a separate resource cap. No confirmation run is launched by this protocol.

## Reproduction

From the repository root, use PowerShell:

    python scripts/statistical_correction/run_corrective_development_v1_3_expansion.py --plan-only
    python scripts/statistical_correction/run_corrective_development_v1_3_expansion.py
    python scripts/statistical_correction/summarize_corrective_development_v1_3_expansion.py

The runner refuses changed frozen inputs and resumes incomplete cells only under their original fingerprints. The summarizer verifies checkpoint hashes, replicate counts, seed uniqueness, matching paired replicates, and reports only development evidence. It never writes the validated registry, modifies v1.2 evidence, rescores the hindcast, or begins confirmation.
