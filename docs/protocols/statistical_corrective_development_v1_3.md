# KUNTUR Statistical Corrective Development Protocol v1.3

**Status:** Frozen before the bounded development pilot. This protocol is a new development-only version; it does not amend the v1.2 selection decisions, validation criteria, results, registry, real-series scores, or hindcast.

## Purpose and boundary

The v1.2 independent-validation experiment is closed as partial. This protocol investigates implementation changes against new, deterministic development simulations for the failures observed in v1.2: marginal Type-I error, interval coverage, residual dependence diagnostics, and numerical convergence. Its pilot is not a validation study and cannot select a method. The candidate grid and resource caps are in `configs/statistical_corrective_development_v1_3.json`.

No validation seed, 2025 hindcast target, NASA Gold outcome, or real environmental series may be used for development or tuning. v1.2's frozen gates remain unchanged. Development outcomes are diagnostic only, including when a pilot point estimate falls inside or outside a gate. No post-hoc winner is eligible for the validated registry.

## Demonstrated failure modes

1. **Type-I error:** both v1.2 frozen selections were `SEASONAL_MK_YEAR2`; independent null rejection rates were 0.072 for IID Gaussian and 0.080 for AR(1) 0.6 with 20% MCAR. Their exact two-sided 95% Monte Carlo intervals excluded 0.05.
2. **Confidence intervals:** across the 160 completed method-regime rows, 106 failed the applicable coverage gate. Coverage for a method without intervals is not applicable, not a pass.
3. **Residual dependence:** 20 rows failed the predeclared residual-dependence gate. The correction must preserve the calendar index and apply residual diagnostics only where the fitted covariance model and support make them valid.
4. **Numerical convergence:** 3 rows were classified `NUMERICAL_FAILURE`. The v1.2 matrix records failures of 17/1000 (AR(1) rho=.844, SARIMAX AR1), 10/1000 (heteroskedastic IID, SARIMAX ARMA11), and 12/1000 (IID, seasonal SARIMAX). The 10/1000 value is mathematically exactly 1%; v1.2's floating comparison represented it as slightly greater than .01. Frozen rows remain unchanged.

## Corrective development hypotheses

- **Type-I:** compare fixed seasonal year-block lengths of 1, 2, 3, and 4 years on the two failed selected regimes. Keep the plus-one Monte Carlo p-value and missingness within calendar-year blocks. Block length is not selected from real data. The development pilot is too small to estimate Type-I error precisely.
- **Interval coverage:** compare the fixed OLS-HAC lags with state-space and feasible-GLS candidates in the regimes that exposed undercoverage. Evaluate null and +0.25 slope cells, retain bias, interval width, and failures. No lag or method is selected by this pilot.
- **Residual dependence:** compare AR(1), ARMA(1,1), and seasonal AR(1,12) covariance candidates against the corresponding synthetic processes. Use Ljung–Box only on complete, regular residual support. For gapped state-space fits, retain calendar positions and record exact-offset pair counts and ACF diagnostics; never compress gaps or label the ACF screening proxy a calibrated test.
- **Convergence:** report fit failures and warnings separately from inferential performance. Future implementation may use bounded optimizer retries, but raw and retry outcomes must both be retained. Classify the 1% failure gate with integer counts (`failures * 100 <= supported`), not a floating-point rate comparison. The pilot checks the boundary behavior at 9, 10, and 11 failures per 1,000 without rewriting v1.2 results.

## Bounded pilot design

- 276 calendar months; development slopes 0 and +0.25 per decade.
- Eight paired outer series per declared regime/slope/method cell. Seed derivation is a new SHA-256 namespace, disjoint from the v1.2 selection and validation seeds. The pilot records the seed namespace and every cell seed.
- 199 inner draws for bootstrap candidates. Small outer counts intentionally limit the run; no Monte Carlo acceptance gate is evaluated.
- Candidate regimes and methods are fixed in the config. Missing calendar months remain on the calendar; methods that require complete support must record unsupported fits rather than compress time.
- One local process, one BLAS thread, 240-second hard wall cap, 1 GiB RSS stop, no SLURM submission. Per-cell outputs are atomic and resumable only under the identical config and source hashes. No automatic full-plan continuation is allowed.
- The pilot can stop at a cell boundary or preserve an `IN_PROGRESS` cell with its completed replicate count. Every completed cell, its raw replicate records, and the rolling manifest are content-hashed.

## Seed separation

Development namespace: `KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-V1.3-2026-11-16`. Seeds are derived deterministically by SHA-256 from the namespace, regime, slope, replicate, and stream. They are checked against the seed integers and substream ranges in existing selection and validation checkpoints before use.

Untouched confirmation namespace reserved for a later, separately frozen plan: `KUNTUR-STAT-CORRECTIVE-INDEPENDENT-CONFIRMATION-V1.3-2027-01-15`. The development script must not derive or consume any confirmation seed. Only after the candidate code, configuration, diagnostics, and gates are frozen may an independent confirmation plan use that namespace. It must use new series and estimator streams and must not be used to modify the candidate.

## Unchanged independent acceptance gates

Any later independent confirmation must retain every applicable v1.2 gate: 1,000 outer replications per declared primary null/effect cell; Type-I 0.035–0.065 and exact two-sided Clopper–Pearson interval containing .05; 95% interval coverage 0.935–0.965 and exact interval containing .95 under null and injected effects; absolute slope bias at most 0.025 units per decade; numerical failure at most 1%; inferential availability at least 99%; mean interval width at most twice the known-covariance oracle; and residual-dependence failures at most 5% for applicable covariance models. Each method is valid only for a declared regime that passes all applicable gates. An unavailable metric remains unavailable and cannot be counted as a pass.

The exact integer numerical gate is `100 * failure_count <= supported_count`. Thus 10 failures out of 1,000 pass at the boundary, while 11 fail. This corrects the comparison implementation only for future runs; it does not relabel any frozen v1.2 row.

## Resource estimate and reproduction

The bounded pilot covers 38 method-regime combinations, two slopes, and eight outer replications (608 fits), plus 199 inner draws only for block-bootstrap methods. Its hard maximum is 240 wall seconds, one process, one BLAS thread, and 1 GiB RSS. Actual elapsed time, CPU time, RSS, completed cells, and content hashes are written to the v1.3 development manifest.

The 10 existing complete v1.2 validation scenarios consumed approximately 11,748.5 process-CPU seconds (3.26 CPU-hours) across 160,000 method fits. A simple linear estimate for the 49 unstarted scenarios is about 16 CPU-hours at that same workload and hardware; this is a planning estimate, not an instruction to run them. Alternative-effect and other deferred scenarios would add cost.

Reproduce the audit and bounded pilot from the repository root:

```powershell
python scripts/statistical_correction/close_partial_validation.py --check-only
python scripts/statistical_correction/close_partial_validation.py
python scripts/statistical_correction/run_corrective_development_v1_3.py
```

The v1.3 output is stored under `results/statistical_correction/corrective_development_v1_3/`. Running this command never invokes v1.2 validation and never extends the 60-scenario plan.

## Evidence and interpretation

The pilot manifest must record protocol/config/script/v1.2 module hashes, dependency versions, all cell seeds, replicate-level fit results, warnings, missing-support status, convergence counts, coverage/rejection diagnostics, residual diagnostics and their method, resource usage, and hashes. All results carry `DEVELOPMENT_ONLY_NOT_VALIDATED`. No result is added to the v1.2 registry or used to revise the frozen selection. The 2025 hindcast score and its negative comparison against monthly climatology remain byte-for-byte preserved and unretuned.
