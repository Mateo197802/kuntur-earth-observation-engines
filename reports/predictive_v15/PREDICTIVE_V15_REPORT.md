# KUNTUR v1.5 Predictive Regime Specialization and Reliability

## Execution status

- Run ID: `20261117T210621Z`
- Development-only outer evaluation: **31 paired monthly forecasts**, 3 chronological year folds.
- Target months: `2019-01` through `2021-08`.
- Test isolation: **PASS**. Forecast and split rows were filtered to target months no later than 2021-12, and test-partition records were excluded before conversion to pandas. No 2022–2025 target values entered fitting, selection, scoring or diagnostics.
- Independent generalization claim: **not made**. This remains a development-only nested hindcast; it was informed by prior inspection of v1.4 development outcomes.

## v1.4 result reproduced

The v1.4 descriptive regime decomposition was recomputed from paired OOF rows in `results/predictive_v14/fold_predictions.parquet`. Its target-anomaly threshold is derived from each row's training-only q75 and is used **only after the forecast for retrospective score stratification**.

| Regime | n | Climatology MAE °C | v1.4 selected MAE °C | Relative change |
| --- | ---: | ---: | ---: | ---: |
| large_anomaly | 13 | 0.8621 | 0.6032 | -30.03% |
| ordinary | 107 | 0.2410 | 0.2689 | +11.56% |

## Source availability audit

The source guard combines CMR granule production timestamps with a conservative source-latency window and checks the corresponding validated Gold fields. NASA documents MERRA-2 monthly release usually by the 15th of the following month; IMERG Final is typically about 3.5 months; GLDAS-2.1 production streams normally take 3–4 months. The issue timestamp is the first instant of the month preceding the target. Therefore this run used a latest predictor lag of three months for MERRA-2 and six months for IMERG/GLDAS.

CMR `ProductionDateTime` means the exact time a granule was produced, while `ProviderDates.Insert` is the time it entered the provider database. Neither field is a guaranteed first-publication timestamp. The registry therefore labels these as **latency-and-production-guard eligible**, records `first_publication_timestamp_verified=false`, and does not overstate the positive availability proof. A granule produced after issue is rejected. Cross-source predictors were not scored unless a chronological, guard-eligible training set preceded 2019.

NASA references: [MERRA-2 system characteristics](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/system-characteristics_merra-2/), [IMERG Final latency](https://gpm.nasa.gov/resources/faq/what-determines-latency-imerg), [GLDAS-2.1 latency note](https://forum.earthdata.nasa.gov/viewtopic.php?t=7953), [CMR ProductionDateTime definition](https://wiki.earthdata.nasa.gov/spaces/CMR/pages/133727054/Production+Date+Time), [CMR ProviderDates definition](https://wiki.earthdata.nasa.gov/spaces/CMR/pages/135834714/Provider+Dates).

| source | version | eligible targets | range | pre-2019 training targets | cross-source scored |
| --- | --- | --- | --- | --- | --- |
| merra2_monthly_slv | 5.12.4 | 69 | 2015-10–2021-08 | 38 | MERRA-2 primary predictors |
| gpm_imerg_monthly_final | 07 | 0 |  | 0 | False |
| gldas_noah025_monthly | 2.1 | 13 | 2020-12–2021-12 | 0 | False |

## Nested chronological model comparison

Candidates: training-only monthly climatology; compact residual HGB; regularized Ridge anomaly; fixed smooth residual shrinkage; and an issue-time feature-only smooth gate over the compact residual forecast. Selection used inner expanding half-year folds and MAE, with climatology retained as an eligible candidate. The gate used lagged/rolling temperature-anomaly magnitude, recent persistence, variability, six-month slope, and seasonal sine/cosine state. Cross-source signals were assessed for availability but were not passed into a model unless the source eligibility gate was satisfied.

The lowest pooled outer-OOF MAE was `regime_mixture` (0.2354 °C). This is a development comparison after nested selection, not an untouched final test.

| strategy | sample_count | mae | rmse | bias | mae_skill_vs_baseline | rmse_skill_vs_baseline | worst_fold_mae_skill_vs_baseline | anomaly_calibration_slope | anomaly_calibration_intercept_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| climatology | 31 | 0.2454 | 0.3003 | -0.0120 | 0.0000 | 0.0000 | 0.0000 | — | — |
| compact_residual | 31 | 0.2715 | 0.3524 | -0.0526 | -0.1064 | -0.1736 | -0.2376 | -0.0572 | 0.0096 |
| regularized_anomaly | 31 | 0.2398 | 0.3145 | -0.0210 | 0.0226 | -0.0475 | -0.0539 | 0.2662 | 0.0144 |
| smooth_shrinkage | 31 | 0.2688 | 0.3489 | -0.0486 | -0.0953 | -0.1619 | -0.2376 | -0.0799 | 0.0090 |
| regime_mixture | 31 | 0.2354 | 0.3000 | -0.0206 | 0.0408 | 0.0010 | -0.0045 | 0.5190 | 0.0164 |
| nested_selected | 31 | 0.2392 | 0.3050 | -0.0333 | 0.0251 | -0.0159 | -0.0352 | 0.4352 | 0.0212 |

### Outer-fold selection and cost

| strategy | outer_folds_selected |
| --- | --- |
| regime_mixture | 2 |
| smooth_shrinkage | 1 |

| strategy | fit_count | fit_seconds_sum | fit_seconds_mean | fit_seconds_max |
| --- | --- | --- | --- | --- |
| compact_residual | 18 | 1.876053 | 0.104225 | 1.425524 |
| feature_pair | 15 | 6.520592 | 0.434706 | 1.676002 |
| regularized_anomaly | 3 | 0.003267 | 0.001089 | 0.001222 |
| ridge_alpha_0.1 | 15 | 0.022010 | 0.001467 | 0.001912 |
| ridge_alpha_1 | 15 | 0.019681 | 0.001312 | 0.001800 |
| ridge_alpha_10 | 15 | 0.018771 | 0.001251 | 0.001540 |
| ridge_alpha_100 | 15 | 0.018422 | 0.001228 | 0.001535 |

### Large-anomaly versus ordinary-month diagnostics

These labels are retrospective, using an outer-fold training-only q75 target-anomaly threshold after forecasts were fixed. They are not gate inputs.

| strategy | retrospective_regime | sample_count | mae | rmse | bias | mae_skill_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| climatology | ordinary | 25 | 0.1802 | 0.2141 | -0.0632 | 0.0000 |
| climatology | large_anomaly | 6 | 0.5169 | 0.5242 | 0.2014 | 0.0000 |
| compact_residual | ordinary | 25 | 0.1918 | 0.2447 | -0.0970 | -0.0642 |
| compact_residual | large_anomaly | 6 | 0.6037 | 0.6262 | 0.1321 | -0.1678 |
| regularized_anomaly | ordinary | 25 | 0.1718 | 0.2323 | -0.0686 | 0.0465 |
| regularized_anomaly | large_anomaly | 6 | 0.5232 | 0.5350 | 0.1771 | -0.0121 |
| smooth_shrinkage | ordinary | 25 | 0.1889 | 0.2431 | -0.0914 | -0.0483 |
| smooth_shrinkage | large_anomaly | 6 | 0.6015 | 0.6185 | 0.1299 | -0.1635 |
| regime_mixture | ordinary | 25 | 0.1659 | 0.2091 | -0.0716 | 0.0797 |
| regime_mixture | large_anomaly | 6 | 0.5250 | 0.5317 | 0.1921 | -0.0156 |
| nested_selected | ordinary | 25 | 0.1706 | 0.2180 | -0.0873 | 0.0532 |
| nested_selected | large_anomaly | 6 | 0.5250 | 0.5317 | 0.1921 | -0.0156 |

### Feature-gate behavior (diagnostic only)

Gate weights are fixed before outer labels are read. The split by retrospective anomaly category below is descriptive only.

| outer_fold | retrospective_regime_diagnostic_only | sample_count | mean_gate_weight | median_gate_weight | gate_weight_ge_0_5_rate | selected_strategy |
| --- | --- | --- | --- | --- | --- | --- |
| outer_2019 | ordinary | 12 | 0.2368 | 0.0000 | 0.3333 | smooth_shrinkage |
| outer_2020 | large_anomaly | 2 | 0.1247 | 0.1247 | 0.0000 | regime_mixture |
| outer_2020 | ordinary | 9 | 0.4487 | 0.4169 | 0.4444 | regime_mixture |
| outer_2021 | large_anomaly | 4 | 0.2055 | 0.2685 | 0.0000 | regime_mixture |
| outer_2021 | ordinary | 4 | 0.8009 | 0.9573 | 0.7500 | regime_mixture |

### Paired MAE skill uncertainty

Intervals resample 3-month blocks only within contiguous eligible calendar segments; no block crosses a month missing from the issue-time-eligible set.

| strategy | sample_count | mean_mae_improvement | ci_low | ci_high | block_length_months |
| --- | --- | --- | --- | --- | --- |
| climatology | 31 | 0.0000 | 0.0000 | 0.0000 | 3 |
| compact_residual | 31 | -0.0261 | -0.0792 | 0.0294 | 3 |
| regularized_anomaly | 31 | 0.0055 | -0.0378 | 0.0543 | 3 |
| smooth_shrinkage | 31 | -0.0234 | -0.0748 | 0.0289 | 3 |
| regime_mixture | 31 | 0.0100 | -0.0162 | 0.0392 | 3 |
| nested_selected | 31 | 0.0062 | -0.0215 | 0.0377 | 3 |

## Interpretation and recommendation

The lowest pooled outer-OOF MAE belongs to `regime_mixture`, but this is descriptive and does not override fold-local nested selection. The nested-selected strategy has MAE skill **+2.51%**, worst-fold MAE skill **-3.52%**, and paired 3-month block interval for MAE improvement **[-0.0215, +0.0377] °C**. On the predeclared reliability checks, the recommendation is: **RETAIN TRAINING-ONLY CLIMATOLOGY FOR DEPLOYMENT**. No independent generalization or production readiness is claimed. The primary baseline remained in every inner selection.

### Hypothesis outcomes

- **v1.4 retrospective regime result is reproducible:** confirmed from 120 paired OOF predictions (13 large-anomaly and 107 ordinary months).
- **Issue-time proxies identify months where residuals should be preferred:** not supported. The regime mixture's retrospective MAE skill was -1.56% on 6 large-anomaly months and +7.97% on 25 ordinary months; its pooled paired interval spans zero and no large-anomaly month received a gate weight of at least 0.5.
- **A MERRA-only residual forecast reliably beats climatology:** not confirmed. The nested-selected MAE gain is small, RMSE skill and worst-fold skill are negative, and the paired interval crosses zero.
- **IMERG/GLDAS historical signals can support the 2019–2021 outer evaluation:** not evaluable with the frozen issue-time contract. Neither source had a guard-eligible training target before 2019 (IMERG: 0; GLDAS: 0), so neither was scored as a cross-source model.
- **Seasonal state is a useful standalone gate:** not selected by any outer-fold nested selection; this does not establish that seasonality has no value as a forecast feature.

## Reproducibility and files

- Full experiment registry and input/configuration provenance: `results/predictive_v15/experiment_registry_v1_5.json`.
- Independent SHA-256 index for generated outputs: `results/predictive_v15/artifact_hashes_v1_5.json`.
- Independent verification command: `python scripts/verify_predictive_v15.py --data-root <existing read-only KUNTUR data root>`.
- Source eligibility audit and CMR snapshot: `results/predictive_v15/issue_time_availability_audit.csv`, `results/predictive_v15/source_availability_cmr_snapshot.json`.
- Outer OOF predictions: `results/predictive_v15/outer_fold_predictions.parquet`.
- Inner OOF predictions and all candidate parameter scores: `results/predictive_v15/inner_selection_oof.parquet`, `results/predictive_v15/inner_tuning_scores.csv`, `results/predictive_v15/inner_candidate_scores.csv`.
- Model and regime metrics: `results/predictive_v15/overall_metrics.csv`, `results/predictive_v15/regime_metrics.csv`.
- Leakage audit: `results/predictive_v15/leakage_audit.json`.
