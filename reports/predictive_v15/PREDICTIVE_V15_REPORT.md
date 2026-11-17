# KUNTUR v1.5 Predictive Regime Specialization and Reliability

## Execution status

- Run ID: `20261117T235039438415Z`
- Development-only outer evaluation: **31 paired monthly forecasts**, 3 chronological year folds.
- Target months: `2019-01` through `2021-08`.
- Test isolation: **PASS**. Forecast and split rows were filtered to target months no later than 2021-12, and test-partition records were excluded before conversion to pandas. No 2022–2025 target values entered fitting, selection, scoring or diagnostics.
- Independent generalization claim: **not made**. This remains a development-only nested hindcast; it was informed by prior inspection of v1.4 development outcomes.

## v1.4 result reproduced

The v1.4 descriptive regime decomposition was recomputed from paired OOF rows in `results/predictive_v14/fold_predictions.parquet`. Its target-anomaly threshold is derived from each row's training-only q75 and is used **only after the forecast for retrospective score stratification**.

| Regime | n | Climatology MAE °C | v1.4 selected MAE °C | Relative change |
| --- | ---: | ---: | ---: | ---: |
| large_anomaly | 13 | 0.8644 | 0.7287 | -15.69% |
| ordinary | 107 | 0.2411 | 0.2845 | +18.03% |

## Source availability audit

The source guard combines CMR granule production timestamps with a conservative source-latency window and checks the corresponding validated Gold fields. NASA documents MERRA-2 monthly release usually by the 15th of the following month; IMERG Final is typically about 3.5 months; GLDAS-2.1 production streams normally take 3–4 months. The issue timestamp is the first instant of the month preceding the target. Therefore this run used a latest predictor lag of three months for MERRA-2 and six months for IMERG/GLDAS.

CMR `ProductionDateTime` means the exact time a granule was produced, while `ProviderDates.Insert` is the time it entered the provider database. Neither field is a guaranteed first-publication timestamp. The registry therefore labels these as **latency-and-production-guard eligible**, records `first_publication_timestamp_verified=false`, and does not overstate the positive availability proof. A granule produced after issue is rejected. Cross-source predictors were not scored unless a chronological, guard-eligible training set preceded 2019.

The corrected folds also apply this production-time rule to **target labels used for fitting**. Every outer and inner model's training labels must have a MERRA-2 CMR production timestamp no later than the earliest issue in that validation window; missing timestamps are excluded. Fold-level cutoff/count evidence is recorded in `leakage_audit.json`.

NASA references: [MERRA-2 system characteristics](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/system-characteristics_merra-2/), [IMERG Final latency](https://gpm.nasa.gov/resources/faq/what-determines-latency-imerg), [GLDAS-2.1 latency note](https://forum.earthdata.nasa.gov/viewtopic.php?t=7953), [CMR ProductionDateTime definition](https://wiki.earthdata.nasa.gov/spaces/CMR/pages/133727054/Production+Date+Time), [CMR ProviderDates definition](https://wiki.earthdata.nasa.gov/spaces/CMR/pages/135834714/Provider+Dates).

| source | version | eligible targets | range | pre-2019 training targets | cross-source scored |
| --- | --- | --- | --- | --- | --- |
| merra2_monthly_slv | 5.12.4 | 69 | 2015-10–2021-08 | 38 | MERRA-2 primary predictors |
| gpm_imerg_monthly_final | 07 | 0 |  | 0 | False |
| gldas_noah025_monthly | 2.1 | 13 | 2020-12–2021-12 | 0 | False |

## Nested chronological model comparison

Candidates: training-only monthly climatology; compact residual HGB; regularized Ridge anomaly; fixed smooth residual shrinkage; and an issue-time feature-only smooth gate over the compact residual forecast. Selection used inner expanding half-year folds and MAE, with climatology retained as an eligible candidate. The gate used lagged/rolling temperature-anomaly magnitude, recent persistence, variability, six-month slope, and seasonal sine/cosine state. Cross-source signals were assessed for availability but were not passed into a model unless the source eligibility gate was satisfied.

The lowest pooled outer-OOF MAE was `regime_mixture` (0.2397 °C). This is a development comparison after nested selection, not an untouched final test.

| strategy | sample_count | mae | rmse | bias | mae_skill_vs_baseline | rmse_skill_vs_baseline | worst_fold_mae_skill_vs_baseline | anomaly_calibration_slope | anomaly_calibration_intercept_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| climatology | 31 | 0.2500 | 0.3027 | -0.0065 | 0.0000 | 0.0000 | 0.0000 | — | — |
| compact_residual | 31 | 0.2784 | 0.3547 | -0.0400 | -0.1136 | -0.1718 | -0.3545 | 0.0203 | 0.0072 |
| regularized_anomaly | 31 | 0.2516 | 0.3279 | -0.0086 | -0.0061 | -0.0832 | -0.1393 | 0.2093 | 0.0070 |
| smooth_shrinkage | 31 | 0.2767 | 0.3519 | -0.0348 | -0.1066 | -0.1624 | -0.3545 | -0.0030 | 0.0065 |
| regime_mixture | 31 | 0.2397 | 0.2999 | -0.0084 | 0.0414 | 0.0095 | 0.0144 | 0.5608 | 0.0076 |
| nested_selected | 31 | 0.2785 | 0.3500 | -0.0400 | -0.1137 | -0.1560 | -0.3545 | 0.0213 | 0.0073 |

### Outer-fold selection and cost

| strategy | outer_folds_selected |
| --- | --- |
| smooth_shrinkage | 2 |
| regime_mixture | 1 |

| strategy | fit_count | fit_seconds_sum | fit_seconds_mean | fit_seconds_max |
| --- | --- | --- | --- | --- |
| compact_residual | 15 | 1.754871 | 0.116991 | 1.388524 |
| feature_pair | 12 | 5.362631 | 0.446886 | 1.642228 |
| regularized_anomaly | 3 | 0.002753 | 0.000918 | 0.001067 |
| ridge_alpha_0.1 | 12 | 0.017332 | 0.001444 | 0.002179 |
| ridge_alpha_1 | 12 | 0.013933 | 0.001161 | 0.001384 |
| ridge_alpha_10 | 12 | 0.014104 | 0.001175 | 0.001635 |
| ridge_alpha_100 | 12 | 0.013295 | 0.001108 | 0.001289 |

### Large-anomaly versus ordinary-month diagnostics

These labels are retrospective, using an outer-fold training-only q75 target-anomaly threshold after forecasts were fixed. They are not gate inputs.

| strategy | retrospective_regime | sample_count | mae | rmse | bias | mae_skill_vs_baseline |
| --- | --- | --- | --- | --- | --- | --- |
| climatology | ordinary | 25 | 0.1860 | 0.2184 | -0.0564 | 0.0000 |
| climatology | large_anomaly | 6 | 0.5169 | 0.5242 | 0.2014 | 0.0000 |
| compact_residual | ordinary | 25 | 0.2006 | 0.2471 | -0.0762 | -0.0788 |
| compact_residual | large_anomaly | 6 | 0.6027 | 0.6291 | 0.1109 | -0.1659 |
| regularized_anomaly | ordinary | 25 | 0.1864 | 0.2543 | -0.0532 | -0.0021 |
| regularized_anomaly | large_anomaly | 6 | 0.5232 | 0.5350 | 0.1772 | -0.0122 |
| smooth_shrinkage | ordinary | 25 | 0.1983 | 0.2454 | -0.0699 | -0.0663 |
| smooth_shrinkage | large_anomaly | 6 | 0.6033 | 0.6236 | 0.1115 | -0.1670 |
| regime_mixture | ordinary | 25 | 0.1720 | 0.2107 | -0.0562 | 0.0753 |
| regime_mixture | large_anomaly | 6 | 0.5219 | 0.5288 | 0.1904 | -0.0095 |
| nested_selected | ordinary | 25 | 0.2008 | 0.2471 | -0.0760 | -0.0798 |
| nested_selected | large_anomaly | 6 | 0.6020 | 0.6151 | 0.1103 | -0.1646 |

### Feature-gate behavior (diagnostic only)

Gate weights are fixed before outer labels are read. The split by retrospective anomaly category below is descriptive only.

| outer_fold | retrospective_regime_diagnostic_only | sample_count | mean_gate_weight | median_gate_weight | gate_weight_ge_0_5_rate | selected_strategy |
| --- | --- | --- | --- | --- | --- | --- |
| outer_2019 | ordinary | 12 | 0.1583 | 0.0338 | 0.1667 | smooth_shrinkage |
| outer_2020 | large_anomaly | 2 | 0.0916 | 0.0916 | 0.0000 | smooth_shrinkage |
| outer_2020 | ordinary | 9 | 0.4232 | 0.3603 | 0.4444 | smooth_shrinkage |
| outer_2021 | large_anomaly | 4 | 0.1947 | 0.2560 | 0.0000 | regime_mixture |
| outer_2021 | ordinary | 4 | 0.7945 | 0.9518 | 0.7500 | regime_mixture |

### Paired MAE skill uncertainty

Intervals resample 3-month blocks only within contiguous eligible calendar segments; no block crosses a month missing from the issue-time-eligible set.

| strategy | sample_count | mean_mae_improvement | ci_low | ci_high | block_length_months |
| --- | --- | --- | --- | --- | --- |
| climatology | 31 | 0.0000 | 0.0000 | 0.0000 | 3 |
| compact_residual | 31 | -0.0284 | -0.0671 | 0.0380 | 3 |
| regularized_anomaly | 31 | -0.0015 | -0.0493 | 0.0602 | 3 |
| smooth_shrinkage | 31 | -0.0266 | -0.0647 | 0.0343 | 3 |
| regime_mixture | 31 | 0.0103 | -0.0035 | 0.0383 | 3 |
| nested_selected | 31 | -0.0284 | -0.0659 | 0.0243 | 3 |

## Interpretation and recommendation

The lowest pooled outer-OOF MAE belongs to `regime_mixture`, but this is descriptive and does not override fold-local nested selection. The nested-selected strategy has MAE skill **-11.37%**, worst-fold MAE skill **-35.45%**, and paired 3-month block interval for MAE improvement **[-0.0659, +0.0243] °C**. On the predeclared reliability checks, the recommendation is: **RETAIN TRAINING-ONLY CLIMATOLOGY FOR DEPLOYMENT**. No independent generalization or production readiness is claimed. The primary baseline remained in every inner selection.

### Hypothesis outcomes

- **v1.4 retrospective regime result is reproducible:** confirmed from 120 paired OOF predictions (13 large-anomaly and 107 ordinary months).
- **Issue-time proxies identify months where residuals should be preferred:** not supported. The regime mixture's retrospective MAE skill was -0.95% on 6 large-anomaly months and +7.53% on 25 ordinary months; its pooled paired interval spans zero and no large-anomaly month received a gate weight of at least 0.5.
- **A MERRA-only residual forecast reliably beats climatology:** not confirmed. The nested-selected MAE gain is small, RMSE skill and worst-fold skill are negative, and the paired interval crosses zero.
- **IMERG/GLDAS historical signals can support the 2019–2021 outer evaluation:** not evaluable with the frozen issue-time contract. Neither source had a guard-eligible training target before 2019 (IMERG: 0; GLDAS: 0), so neither was scored as a cross-source model.
- **Seasonal state is a useful standalone gate:** not selected by any outer-fold nested selection; this does not establish that seasonality has no value as a forecast feature.

## Reproducibility and files

- Full experiment registry and input/configuration provenance: `results/predictive_v15/experiment_registry_v1_5.json`.
- Independent SHA-256 index for generated outputs: `results/predictive_v15/artifact_hashes_v1_5.json`.
- Public aggregate verification: `python scripts/verify_predictive_v15.py --data-root <existing read-only KUNTUR data root>`.
- Full local metric recomputation: add `--private-output-root <private run directory>`; row-level files remain outside Git.
- Public source collection/production metadata: `results/predictive_v15/source_availability_cmr_snapshot.json`.
- Row-level availability audits and predictions are written only to the local private output root and excluded from the public repository. Pass the run directory to the verifier for the full local recomputation.
- Public inner candidate and parameter score summaries: `results/predictive_v15/inner_tuning_scores.csv`, `results/predictive_v15/inner_candidate_scores.csv`.
- Model and regime metrics: `results/predictive_v15/overall_metrics.csv`, `results/predictive_v15/regime_metrics.csv`.
- Leakage audit: `results/predictive_v15/leakage_audit.json`.
