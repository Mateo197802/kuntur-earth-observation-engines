# KUNTUR v1.2 Continuous Optimization and Sealed Evaluation Protocol

Status: frozen before acquisition or inspection of any 2025 MERRA-2 target values. This protocol is development-only until a single, separately logged sealed evaluation is executed.

## Immutable inputs and scope

- The v1.0 Gold Parquet, forecast features, and train/validation/test split files are immutable. Their recorded SHA-256 values remain the v1.0 release values.
- Development experiments may use only the pre-existing rolling-origin targets through 2018-12 and must retain the frozen T-3 feature availability rule. They must not open or score the already-used 2019-2024 partitions.
- External observations (NOAA GHCN-M, CHIRPS, and INAMHI) are evaluation-only. They are not predictors, transforms, tuning labels, or calibration data.
- INAMHI extraction is optional corroboration. Any further extraction run is capped at 900 seconds, is checkpointed by year, and must yield to predictive, analytical, robustness, external-evaluation, and reproducibility work.
- No new NASA collection or global-scale acquisition is in scope. A new Earthdata credential is used only for the predeclared future MERRA-2 test extension described below.

## Prediction contract

- Target: Ecuador pilot-bounding-box area-weighted monthly mean MERRA-2 `T2M`, in degrees Celsius, at target month T.
- Forecast issue time: month T-1. The latest permitted predictor month is T-3.
- Frozen history: 12 monthly values, T-14 through T-3, on the native MERRA-2 grid before the existing regional aggregation contract.
- No CHIRPS, vegetation, or external-source feature is admitted unless a source-specific release-time audit proves it was available by the same forecast origin. No such proof is currently recorded.
- All imputation, scaling, climatology, feature selection, augmentation scales, model fitting, and uncertainty calibration are fitted only within the applicable training fold.

## Development rounds and decision rules

The optimization is finite and may comprise at most six sequential, content-addressed rounds. Every run must record source hashes, code/config hashes, fixed seeds, exact target months, model configuration, elapsed wall/CPU time, peak RSS, predictions, metrics, warnings, and a terminal status. An identical completed run is skipped on resume.

1. **Architecture and regularization:** tune bounded, predeclared grids for Ridge, Random Forest, histogram gradient boosting, RBF-SVR, compact MLP, and temporal CNN on the three established expanding-window folds. All candidates use the same 108 targets. Stochastic models use three fixed seeds; scores first average their predictions by target month.
2. **Feature ablation:** test temperature-only, temperature plus known calendar harmonics, training-derived temperature anomalies, past-only rolling temperature summaries, and full MERRA-2 meteorology. Compare all retained architectures on the same folds and targets. Reject feature-selection procedures that use data outside each fold's training portion.
3. **Training augmentation:** compare clean-only, six individual physical-noise families, and the balanced mixed family at predeclared 5%, 10%, 15%, and 20% training intensities. Preserve originals and targets; scale synthetic weights so each authentic training month keeps the declared total weight. The 25% and 50% scenarios remain evaluation-only.
4. **Robustness and residual correction:** score the unchanged clean validation targets under each six single-noise and five combined-noise family at 25% and 50%. Any bias correction is trained from cross-fitted training-only residuals; validation labels cannot fit corrections.
5. **Uncertainty calibration:** compare rolling-origin and contiguous time-ordered development residual calibration. Report nominal 90% coverage, interval width, Winkler score, season and fold diagnostics. No exchangeability guarantee is claimed for dependent monthly errors.
6. **Freeze and audit:** select one candidate using the predeclared rule, record its full configuration and deterministic artifact hash, produce the development comparison and metric registry, audit inference calibration, and verify leakage/reproducibility. No later iteration may consult the sealed evaluation outcomes.

Primary development selection metric is mean rolling-origin MAE. Tie-breakers, in order, are pooled RMSE, worst-fold MAE, and a fixed candidate identifier. Baselines (persistence, seasonal naive, training-only monthly climatology) compete on the same target months. Baseline-relative MAE differences use paired 12-month circular block bootstrap intervals stratified by validation fold; an interval containing zero means uncertain improvement. Development scores after tuning remain selection evidence, not unbiased generalization estimates.

## Frozen new temporal evaluation extension

- Eligible test period: the complete calendar year 2025 only (12 monthly targets). The exact product is MERRA-2 `M2TMNXSLV.5.12.4`, collection concept `C1276812859-GES_DISC`, variable `T2M`, with the existing Ecuador pilot bounding box and existing native-grid aggregation/preprocessing semantics.
- NASA CMR metadata was checked before this freeze and showed one exact-product granule for each month January through December 2025. Metadata discovery does not expose target values. No 2026 month is eligible.
- Before any target values are read, freeze all development-selected model specifications, preprocessing code/configuration, random seeds, calibration method, baseline set, target metric definitions, artifact hashes, and the scoring implementation. Train the final candidate only on eligible pre-2025 development labels. The test scorer is a separate command that opens sealed 2025 regional targets once; optimization scripts must not read its path.
- Test predictors obey T-3 for every target. These are retrospective hindcasts unless contemporaneous product release timestamps are independently established for the applicable issue date.
- Report exact target dates, MAE, RMSE, bias, paired baseline differences, a 12-month block-bootstrap interval, and the limitation that twelve outcomes provide low precision. Do not merge sealed-test and development rows in a single unqualified leaderboard.
- If any required granule, compatible product definition, or source checksum cannot be verified, no 2025 score is reported and the frozen protocol remains unused.

## Statistical inference calibration

The analytical inference selector is chosen from a predeclared synthetic suite: independent errors; AR(1) rho -0.3, 0.3, 0.6, 0.844; ARMA(1,1); seasonal dependence; heteroscedasticity; and heavy tails, each under null and injected slopes. Primary scenarios use at least 1,000 replicates when runtime permits. Compare OLS-HAC lags 6/12/18/24, AR(1) GLS, ARMA error regression where estimable, prewhitening, moving and seasonal block bootstrap, and robust slopes. Report Monte Carlo confidence intervals for type-I error and 95% interval coverage, plus bias, slope RMSE, interval width, and power. No method is called universally calibrated based on only one generating process.

## Frozen v1.2 model and 2025 evaluation

The final development candidate is frozen from Round 3 before opening the 2025 target outcomes:

- Model: 3-seed Random Forest ensemble (`random_state` 20261115, 20261116, 20261117; `n_estimators=120`, `max_depth=4`, `min_samples_leaf=3`, `max_features=1.0`), with predictions averaged across seeds.
- Features: `C_training_anomalies`, using the twelve MERRA-2 temperature anomalies from T-14 through T-3 and known target-month sine/cosine terms. All climatology, imputation and scaling are fitted on model-training rows only.
- Training: all eligible targets through 2018-12 only; source forecast Gold SHA-256 `67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff`. The selected `heteroscedastic_only` training augmentation adds four deterministic input perturbations per original month at 5%, 10%, 15% and 20%, with balanced clean/synthetic sample weights; targets are unchanged. Perturbation scales are estimated from clean training histories only.
- Candidate specification SHA-256: `98876e3caf905353b8eadb3afc77f29fae766a2b1cf75fe8915b4c32791b54e8`.
- Forecast intervals: expanding symmetric absolute-residual quantile, nominal 90%, using the selected candidate's 108 out-of-fold development residuals over 2010-01–2018-12. The frozen 90th percentile is 0.5805256446 °C. Its observed development coverage was post-selection and serially dependent; no exchangeable conformal guarantee is claimed.
- Sealed evaluation: exact MERRA-2 `M2TMNXSLV.5.12.4`, T2M, January–December 2025, existing Ecuador pilot bbox, target valid-area fraction at least 0.80. Source granule identifiers are obtained from NASA CMR at execution; no target observations are assumed available before validation.
- Test features for target month T: all five contracted MERRA-2 channels for T-14 through T-3; no feature at T-2 or later is admissible. The evaluation is reported as a retrospective hindcast because contemporaneous MERRA-2 release timestamps have not been established.
- Scoring: exact target dates, MAE, RMSE, mean prediction-minus-observation bias, paired MAE difference against training-only calendar-month climatology, and interval coverage, width and Winkler score. A 12-month circular paired block-bootstrap on a 12-month test year is degenerate; report it as non-informative and make no precision claim.
- Scoring code: `scripts/score_phase2_v12_sealed_2025.py`; it checks the frozen model, protocol, data hashes, T-3 histories and coverage gate before reading the dedicated 2025 test Gold. It reads only November 2023–December 2024 monthly fields from historical Gold as past predictors for the 2025 targets; it does not read 2019–2024 forecast labels or the old validation/test split indices.
- The model bundle, 2025 regional Gold, month-level predictions and scored outcomes remain outside Git. Only aggregate, sanitized results and manifests may be published.

## Resource and publication rules

- The local CPU is the default for bounded model experiments. Record processor count, wall time, process CPU time, and peak RSS. No persistent/background work is claimed unless a real approved job was submitted.
- CEDIA work is restricted to the verified KUNTUR workspace and official Earthdata services; check the user's own queue and applicable limits before any job submission. Never affect other projects' jobs. No more than one KUNTUR SLURM job may be active.
- Keep credentials, source observations, restricted station records, NASA granules, and model binaries out of Git. Publish only source code, protocols, aggregate results, and sanitized manifests on the private optimization branch.
- A passing CI run or a lower development MAE does not constitute a v1.2 scientific release. Preserve v1.0.0 and report incomparable evaluation protocols explicitly.
