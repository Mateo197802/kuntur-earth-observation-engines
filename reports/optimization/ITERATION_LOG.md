# Optimization checkpoint

## Round 1 — architecture and regularization

**Root-cause hypothesis:** v1.1's strongest learned model had only a modest advantage over training-only climatology. A bounded regularization search may improve forecasting, while high-capacity candidates may overfit the small sample.

**Experiments executed:** 27 predeclared model configurations across Ridge, RBF-SVR, Random Forest, histogram gradient boosting, MLP and temporal CNN; all three expanding-window folds; three fixed seeds for stochastic models. The 108 monthly targets were identical for each complete candidate. No augmentation in this round.

**Dataset and split:** forecast Gold SHA-256 `67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff`; development targets 2003-03–2018-12; validation folds 2010-01–2012-12, 2013-01–2015-12, 2016-01–2018-12. Historical 2019–2024 partitions and sealed 2025 target values were not read.

**Previous metrics:** v1.1 SVR mean-fold MAE 0.3228 °C; strongest baseline training monthly climatology 0.3658 °C.

**New best development candidate:** svr_rbf `r1-svr_rbf-b1e9fb9a7a64a0df`, config C=1, epsilon=0.05, gamma=scale; mean-fold MAE 0.3176 °C, mean-fold RMSE 0.3950 °C, mean-fold bias 0.0309 °C, worst-fold MAE 0.3583 °C.

**Paired differences and uncertainty:** new SVR minus v1.1 SVR MAE = -0.0052 °C (95% 12-month circular block-bootstrap interval [-0.0112, 0.0001]); lower absolute error in 70/108 months and all 3/3 folds. The interval includes zero; improvement remains uncertain. Versus the strongest baseline, mean-fold delta = -0.0482 °C, 95% interval [-0.1048, 0.0046] °C, also including zero.

**Noise robustness:** not newly evaluated in Round 1. Existing v1.1 robustness results remain separate and are not attributed to these tuned candidates.

**Statistical validation:** no new inference calibration in Round 1. Existing v1.1 analytical audit is not counted as a v1.2 result.

**Computational cost:** 28.20 s wall time, 16 logical CPUs / 8 physical CPUs available; peak RSS across selected folds 427.0 MiB. No CEDIA or SLURM training was used.

**Model artifact hash:** candidate specification SHA-256 `b1e9fb9a7a64a0dfbbcacd020afcfd2197496c7e5fe0ed562702bb7394d4ba73`. This is a content-addressed specification; no final model binary is frozen in this round.

**Decision:** INCONCLUSIVE for scientific superiority. Retain SVR as the next development candidate because it has the lowest mean rolling MAE; do not claim proven improvement over climatology or v1.1 SVR.

**Next highest-value experiment:** controlled feature-group ablations with fold-local anomaly/climatology fitting and past-only rolling features, followed by training augmentation and 25%/50% robustness.


## Round 2 — fold-local feature ablation

**Root-cause hypothesis:** the full lagged feature vector may dilute the short-sample temperature signal; training-derived anomalies, seasonality, history-only summaries, or stable channel reduction may improve development error.

**Experiments executed:** eight predeclared feature groups × all six model families across three expanding-window folds (48 candidates, 144 completed candidate-folds; 108 identical targets each). Imputation, scaling, climatology, and the stability channel subset were fitted using only each fold's training window. The CNN used the same T-3 histories with variable sequence channels and static features joined after temporal pooling. No augmentation in this round.

**Dataset and split:** same frozen forecast Gold SHA-256 `67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff` and 108 targets across 2010-01–2018-12; 2025 target data remain sealed and unread.

**Previous metric:** Round 1 selected SVR mean-fold MAE 0.3176 °C; strongest baseline 0.3658 °C.

**New best development candidate:** random_forest with `C_training_anomalies` and configuration `{"max_depth": 4, "max_features": 1.0, "min_samples_leaf": 3, "n_estimators": 120}`; mean-fold MAE 0.2735 °C, mean-fold RMSE 0.3429 °C, bias 0.0271 °C, worst-fold MAE 0.3217 °C.

**Paired differences and uncertainty:** versus Round 1 SVR, MAE delta = -0.0440 °C, 95% 12-month block-bootstrap interval [-0.1136, 0.0237] °C, 65/108 months and 3/3 folds improved. uncertain; interval includes zero. Versus the strongest baseline the selected-candidate interval is [-0.1657, -0.0225] °C, but this winner was selected among 48 candidates on those same targets; that interval is optimistic and not confirmatory.

**Noise robustness:** not newly evaluated in Round 2. No augmentation or 25%/50% stress results are attributed to this candidate yet.

**Statistical validation:** no new inference calibration in Round 2.

**Computational cost:** 56.70 s wall time on local CPU; six model families × eight groups; all folds checkpointed. The 2025 test remained sealed.

**Model artifact hash:** candidate specification SHA-256 `51a649304ba43efa151893e425775719bf2c78abb1561aabf24cad11b708f705`; feature implementation SHA-256 `5e38313c7cc95bfa3523fd456d8058a12cdd5314d343fd09a79e08ed2d9c3c79`. The feature source hash supplements the run's script hash for reproducibility; no final model binary is frozen.

**Decision:** retain the anomaly-feature Random Forest as a provisional challenger and the Round 1 SVR as a reference. The apparent gain is selected from shared development folds; do not call it independently validated.

**Next highest-value experiment:** controlled training augmentation and 25%/50% corruption on the SVR and Random Forest challengers, using identical noise realizations and base-month-balanced weights.


## Round 3 — training augmentation and 25%/50% corruption robustness

**Experiments executed:** 14 predeclared training schemes × two Round 2 challenger families × three rolling folds (28 candidates, 84 completed candidate-folds); the same 108 targets were used. Every candidate also received 11 corruption scenarios at 25% and 50% intensity. Each scenario had 36 eligible months per fold. Target values were never augmented.

**Best selected development candidate:** random_forest / `heteroscedastic_only` using `C_training_anomalies`; mean-fold MAE 0.2578 °C, RMSE 0.3282 °C, bias 0.0435 °C, worst-fold MAE 0.3048 °C. The best is selected among 28 schemes on these same development folds.

**Paired comparison to same-model clean-only:** MAE delta -0.0180 °C; 95% 12-month block-bootstrap interval [-0.0348, -0.0013] °C; 62/108 target-months and 3/3 folds improved. This is post-selection development evidence, not a confirmatory interval.

**Corruption robustness for selected candidate:** at 25%, mean across 11 scenario MAEs 0.2804 °C and worst scenario 0.3130 °C; at 50%, mean 0.3213 °C and worst scenario 0.3647 °C. These synthetic perturbations measure sensitivity, not real sensor failure prevalence.

**Reproducibility:** source dataset SHA-256 `67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff`; round script SHA-256 `46abdbc8754b6bbeeac3d9d15057028df2fbc0c8e2770baf79b56aea5d54eb57`; runtime 128.27 s. The smoother's all-missing behavior was subsequently tested as warning-free and numerically unchanged for finite windows; the execution manifest retains the exact source hash used by the completed run.

**Decision:** retain the heteroscedastic-only Random Forest as a development candidate, without making a superiority claim before independent calibration and the sealed 2025 score.


## Round 4 — synthetic trend-inference calibration

**Experiments executed:** seeded null and +0.5 °C/decade scenarios across nine monthly error processes; 1,000 replicates per slope condition for HAC, feasible GLS, prewhitening and Sen estimators. All four HAC lags were compared under rho=0.844 and lag 12 elsewhere. Moving and seasonally aligned 12-month block bootstraps used 500 outer replicates × 299 inner draws for four representative processes. ARMA(1,1) likelihood intervals used 200 fits per slope condition; no convergence failures were recorded.

**Monte Carlo calibration:** with rho=0.844, OLS-HAC lag 12 covered 82.1% of null slopes (exact 95% Monte Carlo interval [79.6%, 84.4%]); its type-I rejection rate was 17.9% [15.6%, 20.4%]. Iterated AR(1) feasible GLS covered 91.9% and rejected 8.1%. Under seasonal AR(1) at lag 12, OLS-HAC coverage was 65.7% and type-I error 34.3%. On its matching ARMA(1,1) DGP, the bounded ARMA MLE null run covered 96.0% with 200 valid replicates.

**Interpretation:** no method calibrated at 95% across all dependence structures. Strong persistence and seasonal dependence produced substantial undercoverage for HAC, AR(1) GLS and block-bootstrap intervals. Theil–Sen iid intervals were calibrated for iid noise but failed under dependence. The process-specific ARMA fit performed well in the limited matching case, but 200 replicates give wider Monte Carlo uncertainty. The selector remains scenario-aware; none of these synthetic results proves the residual process of an observed environmental series.

**Reproducibility and scope:** seed 20261123; results SHA-256 `2c5b604ff9e85074cc51e434edf29b94a5655533c7d3774a8fe3d56bf1ec1344`; script SHA-256 `2592b741a85710a93820f1de1e3fcba8b1abbd4de38bddc21169a996fbcd81fa`; 237.6 s elapsed, peak resident memory 179.4 MiB. The simulation reads no KUNTUR observations or sealed 2025 targets.


## Round 5 — chronological forecast-interval calibration

**Scoring design:** the first 2010–2012 out-of-fold window supplied calibration residuals only; the 2013–2015 evaluation used 36 prior months, and the 2016–2018 evaluation used the earlier 72-month history for expanding methods or the latest 36 months for rolling methods. The audit confirms each calibration end date precedes its scored window. No 2019–2024 or 2025 targets were read.

**Methods compared:** expanding and rolling-36-month symmetric finite-sample absolute-residual quantiles, signed empirical residual quantiles, and bias-corrected Gaussian intervals. Overall post-selection development results across 72 targets:

| Method | 90% empirical coverage | Mean width (°C) | Winkler score (°C) |
|---|---:|---:|---:|
| expanding_symmetric_abs_residual | 91.7% | 1.212 | 1.452 |
| rolling36_symmetric_abs_residual | 93.1% | 1.279 | 1.519 |
| expanding_bias_corrected_gaussian | 81.9% | 1.045 | 1.865 |
| rolling36_bias_corrected_gaussian | 81.9% | 1.099 | 1.916 |
| expanding_signed_empirical_residual | 79.2% | 0.959 | 1.989 |
| rolling36_signed_empirical_residual | 79.2% | 0.992 | 2.038 |


**Provisional calibration choice:** expanding symmetric absolute-residual intervals, with 91.7% observed coverage, 1.212 °C mean width and 1.452 °C Winkler score. This choice minimizes the displayed development Winkler score among the compared methods, but it is not independent validation: the candidate was selected on the same folds and monthly residuals are dependent. Freeze it for the sealed evaluation without claiming nominal coverage.

**Artifacts:** interval output SHA-256 `cbb4bb01010dba78cb0c4f22e095f921e3af01fcd1c4366af2ef2da8ff5237f2`; summary SHA-256 `d6192bc544713ad9526b7ed364d4bef2b97cb3fee7daa2142315e2eab1cb0e6f`; chronological audit `results/optimization/round_05_interval_calibration/chronological_leakage_audit.json`.
