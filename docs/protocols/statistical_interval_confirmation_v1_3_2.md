# KUNTUR v1.3.2 — Independent Confirmation of the Finite-Sample t Interval

**Status:** separate confirmation protocol, frozen after development screening and before deriving confirmation seeds. This is a single-candidate, IID synthetic confirmation. It does not amend the v1.2 decisions or the failed v1.3 Wald confirmation.

## Candidate nomination and scope

The v1.3.2 development screen compared the unchanged normal-Wald comparator, a finite-sample Student-t interval, and a profile-likelihood interval using 1,000 paired series at each slope. Only `wald_student_t` was nominated as a corrected candidate after passing every predeclared gate at all three slopes. The unchanged comparator remains comparator-only, even though its development point estimates met the numerical gate thresholds; profile likelihood failed the null Type-I and +0.50 coverage gates.

The selected candidate changes the interval and matching test calibration only. The point estimator remains the exact frozen seasonal SARIMAX model: an intercept, elapsed decades, and 11 month indicators as exogenous regressors; order (0,0,0), seasonal order (1,0,0,12), no additional trend, stationarity/invertibility enforcement, and a maximum of 200 iterations. The t critical value uses `df = n - rank(X) - k_dynamic = 262`; the innovation scale is not subtracted. The p-value uses the two-sided Student-t distribution with the same df.

Scope is limited to a complete 276-month IID Gaussian synthetic series, January 2000 through December 2022, at slopes 0, +0.25, and +0.50 per decade. This does not validate strong AR(1), ARMA, seasonal-error, irregular-calendar, or real NASA series.

## Independent design and frozen gates

The confirmation uses 1,000 fresh series per slope (3,000 fits total) under the seed namespace in `configs/statistical_interval_confirmation_v1_3_2.json`. Each seed is derived deterministically from the frozen namespace, regime, slope, and replicate index. The freeze checks uniqueness across the confirmation inventory and verifies separation from the v1.3.2 preflight/development seed inventories. No confirmation observation is read before candidate nomination and freeze.

Each applicable gate is unchanged from the v1.3 criteria and must pass at every slope:

- At zero slope, Type-I point rate is 0.035–0.065 and its exact 95% Clopper–Pearson interval contains 0.05.
- At all slopes, coverage is 0.935–0.965 and its exact interval contains 0.95.
- Absolute mean slope bias is at most 0.025 per decade.
- Numerical failures satisfy the integer rule `100 * failures <= supported`.
- Inferential availability is at least 0.99.
- Mean interval width is no more than twice the known-covariance oracle width.
- Ljung–Box residual-dependence failures are no more than 0.05, using calendar lags 12 and 24 with 0.01 per-test threshold.

Power is reported with exact Monte Carlo intervals but has no added threshold. Rates use exact two-sided Clopper–Pearson 95% intervals; continuous means use Student-t Monte Carlo 95% intervals. A failed gate means the candidate remains unvalidated and the inference registry is not modified. A passing confirmation is evidence only for this declared IID synthetic scope; registry updates require a separate explicit review.

## Resource controls

The development run measured 3,000 `wald_student_t` fits at mean 0.232296875 CPU seconds and 0.233177321 active wall seconds per fit. A two-fold projection for 3,000 confirmation fits is 1,393.8 CPU seconds and 1,399.1 active wall seconds. Hard limits are 3,600 CPU seconds, 7,200 active wall seconds, 1 GiB RSS, one process, one BLAS thread, and zero SLURM jobs. Checkpoints are atomic and written every 50 completed fits. The confirmation runner resumes only from checkpoints matching the frozen configuration, code, development evidence, seed inventory, and input hashes.

## Reproduction

From the repository root, using the Python 3.12 environment described in the README:

```powershell
python -m pytest -q tests/test_interval_confirmation_v13_2.py
python scripts/statistical_correction/freeze_interval_confirmation_v1_3_2.py
python scripts/statistical_correction/run_interval_confirmation_v1_3_2.py --plan-only
python scripts/statistical_correction/run_interval_confirmation_v1_3_2.py
python scripts/statistical_correction/summarize_interval_confirmation_v1_3_2.py
```

The freeze command is a one-time operation; subsequent verification uses `--verify`. The runner performs no network access, does not read real Gold observations, does not rerun prior experiments, and never writes the validated-regime registry.
