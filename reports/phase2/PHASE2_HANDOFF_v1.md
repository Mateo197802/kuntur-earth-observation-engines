# KUNTUR Competition v1.0 — Phase 2 handoff

**Status:** verified local implementation and aggregate evidence; generated 2026-11-15T23:16:06.374750+00:00. Frozen v1 input files were not modified.

## Scope and immutable inputs

All evaluations use the Ecuador preliminary bounding box `[-81.5, -5.1, -75.0, 1.7]` (EPSG:4326). No Coast/Andes/Amazon mask or external administrative-region mask is claimed. Original v1 Gold and split inputs remain byte-identical to their recorded checksums.

| Artifact | Rows | SHA-256 |
|---|---:|---|
| Gold monthly v1 | 5,520 | `5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2` |
| Forecast +1 v1 | 262 | `67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff` |
| Original split indices v1 | 262 | `c2fbca4ac9636622a08e778913e2edfb23cc210453bf05114e1b3775b9b0726c` |
| Phase 2 split manifest | 262 | `89b2cd6e58e7f76107e4c743cd0f057a9cd75c5a4c1759f31e77264367fefe37` |

The Gold table has 20 variables across 276 months, 2002-01 through 2024-12. Forecast split counts are train/validation/test 190/36/36; the training-only internal model-fit/robustness split is 152/38. The latest permissible predictor is T-3. Exact target ranges are in `data/manifests/phase2_split_manifest_v1.json` and machine summary.

## Augmentation and validation

Training used 152 original model-fit months and 6,688 synthetic examples (6,840 rows total). Final refit used 190 original train months and 8,360 synthetic examples (8,550 rows total). Intensities: 5%, 10%, 15%, 20% for training; robustness used 25% and 50% after selection. Targets were never perturbed; authentic and synthetic sample weights were balanced per original month.

Six-model validation benchmark (validation selected the architecture; internal robustness rows were not used for selection):

| Model | Validation MAE (°C) | RMSE (°C) | Fit wall (s) | Peak RSS (MB) |
|---|---:|---:|---:|---:|
| hist_gradient_boosting | 0.2696 | 0.3269 | 6.58 | 429.0 |
| random_forest | 0.2973 | 0.3673 | 4.01 | 426.8 |
| svr_rbf | 0.3022 | 0.3566 | 1.26 | 553.9 |
| ridge | 0.4036 | 0.5158 | 0.01 | 412.8 |
| mlp | 0.8807 | 1.1026 | 4.38 | 423.8 |
| temporal_cnn | 1.0374 | 1.2265 | 7.38 | 494.4 |
| training_monthly_climatology (baseline) | 0.2607 | 0.3017 | — | — |
| seasonal_naive (baseline) | 0.3307 | 0.4182 | — | — |
| persistence (baseline) | 0.6285 | 0.7306 | — | — |

The selected model is `hist_gradient_boosting`. Its internal robustness diagnostic uses 38 unique chronological months; 207 aggregate rows cover nine IDs (six architectures plus three baselines), one clean condition, six independent noise families, and five combined families at 25% and 50%. Per-scenario aggregates for the selected model:

| Scenario | Level | MAE | RMSE | ΔMAE vs clean | Prediction stability | 90% interval coverage |
|---|---:|---:|---:|---:|---:|---:|
| combined / drift+missingness | 25% | 0.3014 | 0.3784 | 0.0151 | 0.1878 | 84.2% |
| combined / drift+missingness | 50% | 0.3192 | 0.4195 | 0.0329 | 0.2554 | 81.6% |
| combined / gaussian+drift | 25% | 0.3219 | 0.4029 | 0.0356 | 0.1339 | 84.2% |
| combined / gaussian+drift | 50% | 0.3410 | 0.4187 | 0.0547 | 0.1811 | 81.6% |
| combined / gaussian+heteroscedastic | 25% | 0.3316 | 0.4283 | 0.0452 | 0.1749 | 78.9% |
| combined / gaussian+heteroscedastic | 50% | 0.2619 | 0.3340 | -0.0245 | 0.1725 | 89.5% |
| combined / gaussian+impulsive | 25% | 0.3195 | 0.3865 | 0.0332 | 0.1355 | 86.8% |
| combined / gaussian+impulsive | 50% | 0.2845 | 0.3717 | -0.0018 | 0.2067 | 78.9% |
| combined / mixed_multi_family | 25% | 0.2729 | 0.3857 | -0.0134 | 0.1809 | 86.8% |
| combined / mixed_multi_family | 50% | 0.3381 | 0.4304 | 0.0517 | 0.2444 | 73.7% |
| independent / drift | 25% | 0.2910 | 0.3432 | 0.0047 | 0.1633 | 86.8% |
| independent / drift | 50% | 0.3690 | 0.4481 | 0.0827 | 0.2215 | 76.3% |
| independent / gaussian | 25% | 0.2981 | 0.3655 | 0.0117 | 0.1315 | 84.2% |
| independent / gaussian | 50% | 0.3117 | 0.3972 | 0.0254 | 0.2147 | 89.5% |
| independent / heteroscedastic | 25% | 0.2686 | 0.3309 | -0.0178 | 0.1427 | 89.5% |
| independent / heteroscedastic | 50% | 0.3607 | 0.4592 | 0.0743 | 0.2736 | 78.9% |
| independent / impulsive | 25% | 0.2906 | 0.3511 | 0.0042 | 0.0944 | 86.8% |
| independent / impulsive | 50% | 0.2797 | 0.3412 | -0.0066 | 0.0518 | 89.5% |
| independent / laplace | 25% | 0.2976 | 0.3676 | 0.0112 | 0.1361 | 81.6% |
| independent / laplace | 50% | 0.3776 | 0.4772 | 0.0912 | 0.2598 | 73.7% |
| independent / missingness | 25% | 0.2993 | 0.3943 | 0.0130 | 0.1634 | 84.2% |
| independent / missingness | 50% | 0.4093 | 0.5278 | 0.1229 | 0.2964 | 71.1% |

The full aggregate scenario table is `results/phase2/public/robustness_scenario_summary_v1.csv`. Repeated perturbed scenarios do not increase the 38 independent target months.

## Final test

The frozen final test was scored once on 36 months, 2022-01 through 2024-12. Selected model: MAE 0.4974 °C (month-bootstrap 95% CI [0.37900783851799186, 0.6185221008514847]), RMSE 0.6180 °C, bias -0.3501 °C. Invalid predictions: 0. Empirical coverage for the nominal 90% interval was 55.6%; it is descriptive and not a split-conformal guarantee.

| Model | MAE (°C) | RMSE (°C) | Bias (°C) |
|---|---:|---:|---:|
| hist_gradient_boosting | 0.4974 | 0.6180 | -0.3501 |
| persistence | 0.5770 | 0.7365 | -0.0492 |
| seasonal_naive | 0.6752 | 0.8469 | -0.3354 |
| training_monthly_climatology | 0.6477 | 0.7691 | -0.4314 |

Skill relative to persistence: MAE 13.8%, RMSE 16.1%; relative to seasonal naive: MAE 26.3%, RMSE 27.0%; relative to training monthly climatology: MAE 23.2%, RMSE 19.6%.

**Blinding limitation:** Earlier development code preloaded held-out rows before selection but did not fit, tune, select, predict, or report test metrics with them. This is recorded in phase2_execution_failures.jsonl; strict process blinding is not claimed.

## Trend estimates and analytical stress tests

OLS uses calendar-month fixed effects with 12-month HAC/Newey–West errors; the table below reports slopes in the physical units shown per decade. No OLS-HAC result is significant after BH correction across the eight series. Seasonal Mann–Kendall p-values are shown separately in the full CSV and assume seasonal-stratum independence; serial correlation diagnostics are retained. Results are descriptive, not causal.

| Variable | n | Slope/decade | 95% HAC CI | p | BH q |
|---|---:|---:|---:|---:|---:|
| merra2_t2m_c | 276 | 0.1484 degC | [-0.0976, 0.3945] | 0.2372 | 0.2372 |
| imerg_precip_monthly_total | 276 | -7.1670 mm | [-13.9272, -0.4069] | 0.03772 | 0.07543 |
| gldas_soil_moisture_0_10cm | 276 | -0.3156 kg m-2 | [-0.6636, 0.0323] | 0.07543 | 0.1207 |
| gldas_soil_moisture_10_40cm | 276 | -0.9459 kg m-2 | [-2.1000, 0.2082] | 0.1082 | 0.1442 |
| gldas_soil_moisture_40_100cm | 276 | -2.1190 kg m-2 | [-5.0423, 0.8043] | 0.1554 | 0.1776 |
| gldas_soil_moisture_100_200cm | 276 | -5.4793 kg m-2 | [-10.3114, -0.6471] | 0.02625 | 0.07001 |
| modis_ndvi | 144 | 0.0227 1 | [0.0063, 0.0390] | 0.0066 | 0.0528 |
| modis_evi | 144 | 0.0156 1 | [0.0029, 0.0283] | 0.01583 | 0.06334 |

Analytical robustness tested 46 perturbations per variable across 184 rows. Median absolute slope deviations by noise intensity: {'0.05': 0.0042982576908257505, '0.1': 0.0055155512962187, '0.15': 0.01006162787325445, '0.2': 0.015951373355953, '0.25': 0.020597526037391248, '0.5': 0.06584618344262061}. Synthetic reference checks are explicitly synthetic and appear in `results/phase2/public/analytical_synthetic_reference_checks_v1.json`.

## Ingestion and cache benchmark

Partitioning uses 460 year-bounded files for 5,520 rows. Ten repeats per workload returned numerically exact results versus full scans. Peak RSS was 395.9 MB; runtime was 17.66 s. Partitioned uncached reads were slower than a direct full-file scan at this small dataset size; warm cache reads were substantially faster.

| Workload | Full scan (s) | Partitioned uncached (s) | Warm cache (s) | Rows |
|---|---:|---:|---:|---:|
| single_variable_short_3y | 0.010019 | 0.017328 | 0.000268 | 36 |
| multi_variable_short_3y | 0.010462 | 0.065612 | 0.000380 | 144 |
| single_variable_full_23y | 0.010869 | 0.121006 | 0.000277 | 276 |
| multi_variable_full_23y | 0.012143 | 0.488028 | 0.000299 | 1104 |

## External datasets and descriptive comparisons

CHIRPS v3 final monthly: 276/276 months acquired and validated on the native 0.05° grid; median valid bbox pixel fraction 86.0%. The initial 21 rate-limit/read errors were recovered after the official server returned HTTP 200; per-month output hashes are in local sidecars and progress log. Full remote COG object hashes are not claimed because range reads were used.

NOAA GHCN-M v4 TAVG QCU: 275/276 months with at least three eligible stations; 14 stations contributed records. QCU missing sentinels and nonblank QC flags were rejected; point-network composites are not equivalent to the area-weighted grid.

INAMHI published yearbooks: 11 files acquired for 2002–2012; 11,412 extracted station-month rows. Yearbooks with extracted station rows: [2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]; available yearbooks without a validated station table: []. The resulting composites have 132 temperature months and 124 precipitation months with ≥3 stations.
The 2013–2024 yearbooks were not discoverable through the public media API during this run. Rights-reserved PDFs and extracted station data remain local and are excluded from GitHub.

| Source | Comparison | Months | r | Bias | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|
| NOAA_GHCN_M_v4_QCU | temperature_degC | 275 | 0.3824 | -2.9082 | 2.9082 | 3.0719 |
| NOAA_GHCN_M_v4_QCU | final_test_prediction:hist_gradient_boosting | 36 | 0.5804 | -3.5395 | 3.5395 | 3.6493 |
| CHIRPS_v3_final_monthly | precipitation_mm_month | 276 | 0.8541 | 31.5455 | 35.0158 | 43.4526 |
| INAMHI_yearbooks | temperature_degC | 132 | 0.6728 | -3.3527 | 3.3527 | 3.3857 |
| INAMHI_yearbooks | precipitation_mm_month | 124 | 0.6194 | -116.9682 | 117.7783 | 127.1424 |

NOAA and INAMHI are observational corroboration only. Station-point means and gridded bbox means have different spatial support, GHCN-M source inputs can overlap other products, and CHIRPS is a station-blended cross-product rather than independent ground truth. External data were not used for training, scaling, augmentation, selection, or calibration.

Landsat Collection 2: the bounded STAC asset request returned HTTP 200 with `text/html` (33,066 bytes), a login page rather than a science raster; no Landsat data or metric is claimed.

## Reproducibility and release

From a clean Python 3.12 environment, install `.[phase2,test]` and run `python -m pytest -q`. To reproduce the aggregate report with local frozen data, run `python scripts/evaluate_phase2_external.py` after external acquisition artifacts are present, then `python scripts/build_phase2_public_summary.py`. The verified manifests and exact checksums are listed in `reports/phase2/phase2_release_manifest_v1.json`.

Public release tables are aggregate metrics only. NASA granules, monthly regional observations, NOAA/INAMHI/CHIRPS observations, source PDFs, caches, manifests with source-specific paths, and fitted model binaries are ignored by Git. See README for references, license status, and data-use limits.

### Known limitations

- Only the preliminary Ecuador bounding box is available; Coast/Andes/Amazon and external comparison polygons await approved definitions.
- The test contains 36 months; the model beats the fixed baselines on aggregate test error but still undercovers its stated 90% interval.
- MLP candidates hit the declared 300-iteration limit; the six-model benchmark reports the actual results and warnings.
- INAMHI yearbooks have variable station support and no per-value QC flags; source PDFs reserve rights.
- CHIRPS source objects were accessed with byte ranges, so only the regional outputs—not full COG objects—have SHA-256 checksums.
- The Landsat data endpoint returned a login page rather than a valid raster.
- This repository has no open-source license pending contributor-ownership and third-party data-rights review.
