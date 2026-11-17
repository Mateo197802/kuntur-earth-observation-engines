# KUNTUR v1.3.2 — Statistical Correction and Independent Confirmation

**Scientific result:** the finite-sample Student-t interval (`wald_student_t`) passed the frozen 3,000-fit independent confirmation for the declared IID Gaussian synthetic regime. The original v1.3 normal-Wald failure remains unchanged. This evidence does not validate inferential claims for dependent regimes or real NASA observations.

Aggregate handoff JSON SHA-256: `6e3450a1a914f99fdc958f3fb80aa55f4f5071eeb7619a27098a4ad8f17ac8cc`.
v1.3.2 development freeze SHA-256: `f7d3799f5ea56592bcd202a69a5c4759c0b0956954106e96d686b0a45ab96e4a`.
Independent-confirmation freeze SHA-256: `18bb4bcf7f6ccde1ad1b34e6039a2cc96589e6bd11f940dacff402e356ec30bd`.

## Why the frozen v1.2 selections failed later calibration

The original score reconstruction was exact; the score arithmetic was not wrong. The selector compared methods on unequal capabilities. In IID, Seasonal MK had no interval, so it was not penalized for null or alternative coverage, while interval methods were scored on coverage as well as Type-I error and bias. For AR(1) with 20% MCAR, exact-calendar residual diagnostics were prescribed but unavailable results were treated as failed gates, leaving Seasonal MK as the only eligible candidate. Selection used 500 replicates, giving an approximate Type-I Monte Carlo SE of 0.010 near 5%; the 500-to-1,000 replicate shifts were compatible with ordinary Monte Carlo variation and winner's curse. The saved selection decisions are not rewritten.

## Preserved original failure and v1.3.1 preflight correction

The original `SARIMAX_SEASONAL_AR12` normal-Wald confirmation covered 920/998 intervals at +0.50 per decade (92.18%; exact 95% interval 90.34–93.77%), below the frozen 93.5% floor. The v1.3.1 preflight then exposed a profile-likelihood implementation guard that treated a roughly 7.44e-6 optimizer tolerance difference as failure under a 1e-6 threshold. Its 37/90 affected rows are retained; no v1.3.1 main development ran. v1.3.2 corrected the numeric guard with an explicit 1e-3 tolerance and required 100% inferential output availability in all 270 preflight candidate rows.

## v1.3.2 paired development comparison

The comparison used 1,000 paired IID Gaussian series for each slope (0, +0.25, +0.50 per decade), shared across the unchanged normal-Wald comparator, finite-sample t interval, and profile-likelihood interval. All candidates use the same SARIMAX seasonal AR(12) point model. Exact binomial 95% intervals are shown for Type-I error, coverage, and power; bias intervals are Student-t Monte Carlo intervals.

| Method | Slope | Type-I / power (exact 95% interval) | Coverage (exact 95% interval) | Bias per decade (t-MC95) | Width/oracle | Numerical failures | Availability | Residual failures | Gates |
|---|---:|---|---|---|---:|---:|---:|---:|---|
| wald_normal | +0.00 | 0.0572 [0.0436, 0.0734] | 0.9428 [0.9266, 0.9564] | -0.001400 [-0.002798, -0.000002] | 0.984 | 3/1000 | 0.997 | 35/1000 | PASS |
| wald_student_t | +0.00 | 0.0562 [0.0427, 0.0723] | 0.9438 [0.9277, 0.9573] | -0.001400 [-0.002798, -0.000002] | 0.989 | 3/1000 | 0.997 | 35/1000 | PASS |
| profile_likelihood | +0.00 | 0.0702 [0.0551, 0.0879] | 0.9298 [0.9121, 0.9449] | -0.001400 [-0.002798, -0.000002] | 0.944 | 3/1000 | 0.997 | 35/1000 | FAIL |
| wald_normal | +0.25 | 1.0000 [0.9963, 1.0000] | 0.9509 [0.9355, 0.9634] | 0.000319 [-0.001068, 0.001706] | 0.985 | 3/1000 | 0.997 | 32/1000 | PASS |
| wald_student_t | +0.25 | 1.0000 [0.9963, 1.0000] | 0.9509 [0.9355, 0.9634] | 0.000319 [-0.001068, 0.001706] | 0.989 | 3/1000 | 0.997 | 32/1000 | PASS |
| profile_likelihood | +0.25 | 1.0000 [0.9963, 1.0000] | 0.9368 [0.9199, 0.9511] | 0.000319 [-0.001068, 0.001706] | 0.944 | 3/1000 | 0.997 | 32/1000 | PASS |
| wald_normal | +0.50 | 1.0000 [0.9963, 1.0000] | 0.9416 [0.9252, 0.9554] | -0.000653 [-0.002065, 0.000759] | 0.987 | 6/1000 | 0.994 | 21/1000 | PASS |
| wald_student_t | +0.50 | 1.0000 [0.9963, 1.0000] | 0.9416 [0.9252, 0.9554] | -0.000653 [-0.002065, 0.000759] | 0.991 | 6/1000 | 0.994 | 21/1000 | PASS |
| profile_likelihood | +0.50 | 1.0000 [0.9963, 1.0000] | 0.9316 [0.9141, 0.9465] | -0.000653 [-0.002065, 0.000759] | 0.943 | 6/1000 | 0.994 | 21/1000 | FAIL |

Only the finite-sample `wald_student_t` interval was eligible among corrected candidates. It passed every development gate at all three slopes. The unchanged normal-Wald comparator also met the metric thresholds in the new development sample, but it was designated comparator-only because it is the method that failed the earlier untouched confirmation. Profile likelihood failed null Type-I and +0.50 coverage gates. Development did not promote any method.

## Independent confirmation of `wald_student_t`

After freezing the candidate, acceptance criteria, implementation hashes, a fresh namespace, and measured budget, a separate 1,000-series confirmation was run at each slope. The exact 3,000-seed inventory was unique and had zero overlap with 207,906 saved seed values parsed from 342 readable prior statistical JSON artifacts. One known malformed historical recovery metadata receipt (no seed rows) was excluded by hash; it was not an analysis checkpoint. The protocol also verified no overlap with the v1.3.2 preflight/development inventories.

| Slope | Type-I / power (exact 95% interval) | Coverage (exact 95% interval) | Bias per decade (t-MC95) | Width/oracle | Numerical failures | Availability | Residual failures | Gates |
|---:|---|---|---|---:|---:|---:|---:|---|
| +0.00 | 0.0481 [0.0357, 0.0633] | 0.9519 [0.9367, 0.9643] | 0.000216 [-0.001167, 0.001600] | 0.989 | 3/1000 | 0.997 | 24/1000 | PASS |
| +0.25 | 1.0000 [0.9963, 1.0000] | 0.9418 [0.9254, 0.9555] | -0.000114 [-0.001533, 0.001305] | 0.986 | 4/1000 | 0.996 | 28/1000 | PASS |
| +0.50 | 1.0000 [0.9963, 1.0000] | 0.9507 [0.9353, 0.9633] | 0.000821 [-0.000611, 0.002252] | 0.993 | 7/1000 | 0.993 | 22/1000 | PASS |

All applicable gates passed. Actual resource use was 699.6 CPU seconds, 701.8 active wall seconds, and 196,091,904 bytes peak RSS; limits were 3,600 CPU seconds, 7,200 wall seconds, and 1 GiB RSS.

## What this enables and what remains withheld

This confirms the finite-sample t interval only for the stated synthetic IID Gaussian data-generating process, the 276-month support, slopes, model, and diagnostic policy. The `wald_student_t` result is not applicable to AR(1), ARMA, seasonal dependence, irregular missingness, or actual NASA climate series without separate development and independent confirmation in those regimes. The v1.3 runtime adapter has no frozen v1.3 registry; the existing v1.2 registry remains empty and hash-identical. Thus confirmatory p-values, confidence intervals, and FDR q-values remain withheld for real Gold/NASA analyses. No registry change is made by this handoff.

The original negative 2025 forecasting score is preserved and was not rerun or used for selection. v1.2 closure and original v1.3 confirmation checkpoints remain unchanged. No model training, trend analysis of NASA observations, publication merge, or PR merge occurred.

## Provenance and reproduction

- Public-safe aggregate JSON SHA-256: `6e3450a1a914f99fdc958f3fb80aa55f4f5071eeb7619a27098a4ad8f17ac8cc`.
- Development metrics SHA-256: `5dd7ec3b0e2ce236537dd51b58a2b10196d48bf15f256886d776229384b1cadf`; development report SHA-256: `d5da67b55dd5639ecac78f7d3657d65909ea9d06781a9a8b1e1df5f40584c155`.
- Independent-confirmation metrics SHA-256: `894dc5bd67b090c4ec7a9848bb1cba4480b3b4e63a8fa52fb249d6b09e6e3030`; report SHA-256: `b2594d064f0a2fdfd477db66d08a481ba54e0f20cc5de2d3d67ea8b2c39d29c4`.
- v1.2 registry SHA-256 (unchanged): `d68ddbf8825d98699198140593a871df307d71a3edb490ca02f992bae7f13580`.
- 2025 forecast score SHA-256 (unchanged): `a38af9f37962f948a0af7c27b06c829adb01c78311131f764bc344f1c64c80b7`.
- Machine-readable code/config/test hashes are in the aggregate JSON. No seed values or replicate-level rows are published.

From the repository root in the Python 3.12 environment from the README:

```powershell
python -m pytest -q tests/test_interval_comparison_v13_2.py tests/test_interval_confirmation_v13_2.py
python scripts/statistical_correction/freeze_interval_confirmation_v1_3_2.py --verify
python scripts/statistical_correction/audit_confirmation_seed_disjointness_v1_3_2.py
python scripts/statistical_correction/summarize_interval_comparison_v1_3_2.py --phase development
python scripts/statistical_correction/summarize_interval_confirmation_v1_3_2.py
python scripts/statistical_correction/export_interval_confirmation_public_v1_3_2.py
python scripts/statistical_correction/verify_v1_3_evidence_bundle.py
```

The result summary contains aggregate statistics only. Re-running simulations is a separate frozen workflow; the commands above verify and summarize the saved evidence without rerunning the simulations.
