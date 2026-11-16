# KUNTUR Competition v1.2 — Phase 2 Handoff

**Prepared:** 2026-11-15 (America/Guayaquil)
**Release scope:** analysis and forecasting evaluation using the frozen Ecuador pilot bounding box.
**Region:** `ecuador_pilot_bbox`, EPSG:4326, west/south/east/north `[-81.5, -5.1, -75.0, 1.7]`. This rectangle is not an Ecuador administrative boundary or an approved Coast, Andes, Amazon, watershed, or external-comparison mask.

## 1. Data inputs and preserved artifacts

The historical v1.0 scientific inputs remain unchanged. The Gold table contains 5,520 rows (20 variables × 276 calendar months, January 2002–December 2024); its SHA-256 is `5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2`. The frozen forecast table, v1 split indices, split manifest, acquisition manifest, and regional outputs remain preserved. The acquisition and preliminary Ecuador bbox manifests each record 1,104 validated source granules or outputs; the failed geographic-mask attempt remains a separate pending record.

The immutable input contract and source history are documented in `configs/data_contract_v1.json`, `data/manifests/gold_manifest_v1.json`, `data/manifests/phase2_split_manifest_v1.json`, and `reports/phase2/PHASE2_HANDOFF_v1.md`. This v1.2 document supplements the v1.0 handoff; it does not rewrite its historical findings.

## 2. Actual-series v1.2 trend output

The v1.2 engine produced descriptive month-adjusted OLS slopes from the real frozen Gold data. These are estimates for the preliminary bbox and are not causal effects. The engine's reference HAC method is known to under-cover in some dependent-error regimes. The independently validated method-regime registry is not complete, so this output intentionally includes **no p-values, confidence intervals, significance labels, or adjusted q-values**.

QA eligibility thresholds were 0.80 valid bbox area for MERRA-2, IMERG, and GLDAS, and 0.10 for MODIS. All non-MODIS series retain 276/276 eligible months. MODIS NDVI and EVI each retain 144/276 (52.17%); the longest missing run is eight months.

| Variable | Unit | Eligible months | Descriptive slope per decade | v1.2 status |
|---|---|---:|---:|---|
| MERRA-2 2 m air temperature | °C | 276/276 | +0.1484 | `UNDERCOVERAGE_DETECTED`; inference withheld |
| IMERG monthly precipitation total | mm | 276/276 | −7.1670 | `UNDERCOVERAGE_DETECTED`; inference withheld |
| GLDAS soil moisture, 0–10 cm | kg m⁻² | 276/276 | −0.3156 | `UNDERCOVERAGE_DETECTED`; inference withheld |
| GLDAS soil moisture, 10–40 cm | kg m⁻² | 276/276 | −0.9459 | `UNDERCOVERAGE_DETECTED`; inference withheld |
| GLDAS soil moisture, 40–100 cm | kg m⁻² | 276/276 | −2.1190 | `UNDERCOVERAGE_DETECTED`; inference withheld |
| GLDAS soil moisture, 100–200 cm | kg m⁻² | 276/276 | −5.4793 | `UNDERCOVERAGE_DETECTED`; inference withheld |
| MODIS NDVI | 1 | 144/276 | +0.02266 | `IRREGULAR_TIME_SUPPORT`; inference withheld |
| MODIS EVI | 1 | 144/276 | +0.01561 | `IRREGULAR_TIME_SUPPORT`; inference withheld |

Machine-readable output: `results/statistical_correction/real_series_v12/descriptive_trend_estimates.csv` (8 rows, SHA-256 `43ee306cd8e1d93b541d5b2d3d7a6b6a5fd854b09788b315a271b58f4ee4ca60`). Provenance and configuration: `results/statistical_correction/real_series_v12/descriptive_trend_manifest.json`. These outputs are local ignored results and are not scientific source data committed to Git.

The selection simulation is complete: all ten locked process/slope scenarios reached 500/500, and `selection_decisions.json` is frozen before validation with status `FROZEN_BEFORE_INDEPENDENT_VALIDATION`, protocol SHA-256 `f2fdf0e7941eb548568b8e1979c89799433872df131a36ca56709bef395a65b8`, and validation seed base `20261127`. Independent validation is running across 60 locked scenarios. The first scenario (IID Gaussian, zero slope) completed 1,000/1,000 replicates with 399 nested draws. At 01:34:01, the second scenario (AR(1) ρ=0.844, zero slope) had reached 25/1,000. At 01:34:11, the validation worker had run 15:40 elapsed, used 940 CPU seconds, and used 214 MiB working set. These are progress records, not a phase-level validation result. Inferential trend outputs remain withheld until the full independent phase and acceptance gates are evaluated. Validation checkpoints are under `results/statistical_correction/validation/checkpoints/`.

## 3. Predictive evaluation

The v1.2 development process compared six architecture families with baselines on chronological development folds, then assessed noise robustness. These are selection and development results, not the final test:

- Round 1: 27 candidates, six families, three folds and 108 targets. Best RBF-SVR mean-fold MAE was 0.3176 °C. Its difference from the earlier SVR was −0.00520 °C, with a 12-month block interval [−0.01121, +0.000084] °C; the comparison was inconclusive.
- Round 2: 48 candidates and 144 candidate-fold evaluations; best random forest with anomaly features had mean-fold MAE 0.27355 °C.
- Round 3: 28 candidates and 84 candidate-fold evaluations across 11 perturbation scenarios at each of 25% and 50%. The selected development configuration was random forest with heteroscedastic-only augmentation: mean-fold MAE 0.25776 °C, RMSE 0.32818 °C, bias +0.04351 °C, and worst-fold MAE 0.30480 °C. Robustness findings are post-selection diagnostics.

The frozen Round 6 model was scored once on a separate 12-month January–December 2025 MERRA-2 Ecuador bbox extension. The model and scorer were frozen before test labels were opened. No model, interval method, or scorer was changed after scoring.

| 2025 retrospective hindcast | MAE (°C) | RMSE (°C) | Bias (°C) |
|---|---:|---:|---:|
| Frozen random forest | 0.32513 | 0.38214 | −0.26653 |
| Training-only monthly climatology | 0.16591 | 0.22177 | −0.13619 |

The model had lower absolute error than climatology in 0/12 months. The paired model-minus-baseline MAE was +0.15922 °C. The nominal 90% interval covered 10/12 months (83.3%), with mean width 1.16105 °C. Calibration residuals were post-selection and serially dependent; the observed one-year coverage is descriptive and does not guarantee coverage. The 12-month circular block diagnostic is non-informative because its block spans the entire test year. This is a retrospective hindcast; contemporaneous source release times were not established. **The frozen model did not beat the training-only climatology on the 2025 test and must not be retuned against that test.**

The original v1.0 2002–2024 split files are unchanged. The 2025 extension is MERRA-2 only and does not replace the v1.0 2022–2024 test. It is not an external observational validation.

## 4. 2025 source and regional artifacts

The CEDIA repository root is represented below as `<KUNTUR_CEDIA_REPO>`; relative paths are inside that existing workspace. The source granules remain full global NASA files. Each month is 78,841,049 bytes, for 946,092,588 bytes total. Catalog size estimates consistently differed from received byte counts (75,188,684 vs. 78,841,049 bytes); all files nevertheless passed HTTP, NetCDF4/HDF5 readability, variable, coordinate, time-bound, and checksum validation. The mismatch is recorded in each receipt and is not represented as an exact source-size match.

| Artifact | CEDIA path | Coverage / validation |
|---|---|---|
| 12 MERRA-2 source granules | `<KUNTUR_CEDIA_REPO>/data/bronze/merra2_monthly_slv/global_full_granule/2025/MERRA2_400.tavgM_2d_slv_Nx.YYYYMM.nc4` | Jan–Dec 2025, global full granules, 12/12 validated, 78,841,049 bytes each; per-file SHA-256 in acquisition receipts |
| Acquisition receipts | `<KUNTUR_CEDIA_REPO>/data/manifests/phase2_v12_sealed_2025_acquisition_receipts.jsonl` | 12 entries; source URL, granule ID, exact time bounds, validation details, size, and SHA-256 |
| Ecuador bbox HDF5 subsets | `<KUNTUR_CEDIA_REPO>/data/silver/regions/ecuador_pilot_bbox/merra2_monthly_slv/2025/MERRA2_400.tavgM_2d_slv_Nx.YYYYMM.h5` | Jan–Dec 2025, 14 × 11 native cells, 12/12 validated, 5,801,951 bytes total; each receipt links its source SHA to the subset SHA |
| Regional receipts | `<KUNTUR_CEDIA_REPO>/data/manifests/phase2_v12_sealed_2025_regional_receipts.jsonl` | 12 entries; coordinates, variable names, validation, source linkage, output size, and SHA-256 |

The additional CEDIA storage for these 2025 full granules plus bbox subsets is 951,894,539 bytes. Personal quota was not independently exposed in the environment; no quota value is inferred from shared-filesystem free space. No MODIS, IMERG, or GLDAS 2025 test extension is included.

The source and regional receipt manifests have SHA-256 values `af03884b123d482cb92eb3b52f8e649994726ccecc8d640fd029fe78df8424ee` and `4ea6ba9311003860f0cf440fa6f985f67ddc41a07638c508ccf4c084aad1ce7e`, respectively. The table gives each global source checksum and its validated regional output checksum; each global source file is 78,841,049 bytes.

| Month | Source `.nc4` SHA-256 | Bbox `.h5` bytes | Bbox `.h5` SHA-256 |
|---|---|---:|---|
| 2025-01 | `a535fee94e53a7843f59a9154a8de522bff95a68a4e63745f28f4bddf1ee553e` | 483,200 | `ee6ede1885463602a59e7da8fb8b11b35dd8fcbfe83e0167fc4122bc1a86827e` |
| 2025-02 | `a5cb2ee3c01ccf016da92e2a029524f8e6bfb5855ec31c04cea07ebf50d2010a` | 483,693 | `5e43a58759ef5b99ed602a5f6151dfe4ebc94e8633383d65f5121d4a75f9d669` |
| 2025-03 | `1bddb2659bb1431a17f2047b89e7bd9d5c975eb09905dab2d21184530a98452e` | 483,719 | `cefda0f12f8bb69e9d66d3fb9a1645e60186b0fc0ba991c17d1ffaf1993ac764` |
| 2025-04 | `b79af79b312cfaded031a342c98e2141baee9e9752ca52dd6977a0d5103c5b12` | 483,200 | `5dc6c54977a7d49216f5eefcaf1cb96993d5384e9639cf82af7650cb63387a0d` |
| 2025-05 | `17cd888df88f6264a53700531849241c559a2e95dbc7711896cd18cc6e56253c` | 483,200 | `4779e26b05340333e36d5e1a549f766c3307be7d1d138107ecaeefea0cec2a51` |
| 2025-06 | `0f3a65cea2554207687ee76632041674616c6ea1c18d2a0050cc17c21a83d5b1` | 483,200 | `6352a102d59802407acaea7bfcb16c90162b4f28ca94033bf0fd3f69cb105f0e` |
| 2025-07 | `0d4a294fa09c0aad01cba72a34ce04f5415ebd7467903cc6a32d2546aba9400a` | 483,200 | `02e19a76b7999ba043bf829356434117a398986a735302a446847f1e957c453b` |
| 2025-08 | `5a82cbd117a0b43e331da352aa15c407ae539d1f08ef563754bec57b78554638` | 485,248 | `e9bd98310a8bbb02e6790e168d1dd6f1fb1c5b2ad56beb1bb2b8c793e6dc56d5` |
| 2025-09 | `243acba46399b61fae03d30a41afa65698dbf8f186e68b3f1358184ca77818e5` | 483,691 | `c59d413e25b0a1de48f350fbf73a2cbd556ffd7880e0342d242b21171235da11` |
| 2025-10 | `4fd74e6f5e13d9ce84d1ac09f751f873117bbc298e5e3d472e5bb62263732b91` | 483,200 | `1f27085c114a5ce78a70b1ba1be6bdc398b057f73493c910c83940216af5a51d` |
| 2025-11 | `303c3f71d87452914dc14b4087439a53997f4f0891fd64b84be0cdaadf826af1` | 483,200 | `05f17752d42d0c6ac9e8378972e192f59abeb79774fff1e291228738e984f309` |
| 2025-12 | `f66607b96939558d52cbdf7756ceccde73d89ac283f6ee72ac238a8b40e0e1f1` | 483,200 | `4edc55f2dd97a0cd141ba8603afcce8aa87491cba103c6ebbc019faf0dfb1f14` |

The copied regional HDF5 files and receipt manifests used for local scoring remain in the local KUNTUR checkout. The 2025 Gold has 72 rows (12 months × six MERRA-2 variables), 22,741 bytes, and SHA-256 `76d2b7a6dd667590c34020ddc211debd30cbedc4d7b79dd771ceae6ba89f5710`. Its manifest is `data/manifests/phase2_v12_sealed_2025_gold_manifest.json`, SHA-256 `f21b519e59ef4f9d2267c3d1836da75f1b75f515b2dcf19a56245ce0b45dca20`; all six variables have 12/12 months and 100% valid bbox area. The score JSON is `results/optimization/round_06_2025_test_score/sealed_2025_score.json`, SHA-256 `5d524bacea80e4c6c42144e143d6096d5ebba249c661f8d82501ce0bda324f19`. The monthly predictions CSV is local and excluded from Git.

For team access, connect to CEDIA, change to the authorized KUNTUR workspace, and use the manifest paths above to read or copy files. The `.nc4` files are global source granules; use the `.h5` files for the preliminary Ecuador bbox extraction. These subsets are not Coast/Andes/Amazon or administrative-area masks.

## 5. External evidence and pending geographies

Previously validated external data remain separate from training and model selection: CHIRPS monthly precipitation covers 276/276 months, NOAA GHCN-M temperature has 275/276 months meeting the minimum station criterion, and extracted INAMHI yearbooks cover 2002–2012 (132 temperature and 124 precipitation comparison months meeting the prior comparison rule). The INAMHI extraction checkpoints are preserved. Station composites, blended precipitation products, reanalysis, and satellite products have different spatial support and dependencies; they are corroboration, not interchangeable ground truth. The v1.2 2025 test is not an external validation.

### Independently checked MERRA-2 native-grid point-to-grid comparison

The CEDIA KUNTUR workspace contains the full 2002-01–2024-12 comparison at `<KUNTUR_CEDIA_REPO>/data/external/processed/phase2_v12_native_grid_collocation/full_2002_2024/`. The directory is mode 700 and its files are mode 600. It uses the existing 276 validated MERRA-2 native-grid NPZs (14 × 11 cells); no NASA granules were downloaded again, and no regridding or elevation correction was applied. A separate verifier rehashed all 276 NPZ files, followed each to validated acquisition and regional-extraction checksums, checked the station-table / Parquet / source-manifest / CSV-sidecar checksums, recomputed all point-to-grid metrics, and confirmed the monthly coverage. Exact owner-only paths and access notes are kept in the local CEDIA handoff, which is excluded from Git.

| CEDIA artifact | Rows / bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `station_monthly_collocation.csv` | 12,945 / 3,049,960 | `c2f31333e22c1f3dca850cedc10cf2b323661be6d015d729e44455ae7f6498fd` | Eligible station-month pair detail; keep in CEDIA |
| `monthly_summary.csv` | 552 / 68,966 | `8c6374d29dea26b3983ed441807747f447e0a09c2fbe94732c7aad95a8b9a0af` | Two sources × 276 calendar months; keep in CEDIA |
| `summary.json` | 151,874 | `1766aec1e5a42fea2eb5a4a2773d51ebe5ad39c12f53197952ee393be1fecea9` | Aggregate metrics, input/output provenance and limitations |
| `validation.json` | 1,823 | `982eed21a7c0f1e3e8b15b24e7c7ccda11c191fa499f816443c8b9161730043d` | Independent validation receipt; status `PASSED` |

The validated comparison covered 2,446 eligible NOAA GHCN-M QCU station-months (14 stations, 276 months): MAE 1.9003 °C, RMSE 2.4125 °C, bias (MERRA-2 minus station) −0.8651 °C, Pearson r 0.9337, mean nearest-cell distance 22.15 km (p95 36.01 km). INAMHI covered 10,499 eligible station-months (151 stations, 132 months in 2002–2012): MAE 2.9823 °C, RMSE 3.7841 °C, bias −1.2330 °C, r 0.8052, mean nearest-cell distance 24.02 km (p95 37.05 km); 32 station IDs have coordinate variants, retained at the per-record level. These are descriptive point-to-grid agreements, not independent ground truth: MERRA-2 may assimilate station observations, stations and grid cells have different spatial support/elevation, NOAA station coverage is sparse, and INAMHI ends in 2012. The full station-level files and inputs remain only in the permission-restricted CEDIA directories.

Safe-to-publish aggregate copies are `results/optimization/native_grid_collocation_summary.json` and `results/optimization/native_grid_collocation_validation.json`; they are included in the experiment registry, metric history, and dashboard with their checksums. No CEDIA SLURM job was submitted for this bounded collocation; other project jobs were not touched.

The failed polygonal geographic-mask attempt is pending work. Coast, Andes, Amazon, and the team's external-comparison region require approved geographic definitions before regional series can be claimed. The point-to-grid comparison above uses only the Ecuador pilot bounding box; it does not create or validate those regional masks.

## 6. Reproduction and integrity

From the repository root with the project environment installed:

```powershell
python scripts/run_phase2_v12_descriptive_trends.py
python -m pytest -q tests/test_phase2_v12_descriptive_trends.py tests/test_phase2_v12_gold_preparation.py tests/test_phase2_v12_sealed_2025_scoring.py tests/test_statistical_inference_v12.py
```

The trend command verifies the frozen Gold SHA against its manifest before producing output. The focused reproducibility run for the v1.0 data-engineering handoff passed: **31 tests passed**. The full v1.2 repository suite passed **140 tests** on Nov 16, 2026. The sealed 2025 scorer was run once; do not rerun it as part of routine reproduction. Scientific data, source manifests with private paths, and the fitted model bundle are not included in the code release.

## 7. Release limitations

- v1.2 inferential trend claims remain withheld pending completed independent method calibration and validation.
- The frozen 2025 model performed worse than the training-only climatology; no test-driven retuning is authorized by this handoff.
- The bbox is preliminary; Coast, Andes, Amazon, and external comparison masks remain pending approved definitions.
- The 2025 extension is MERRA-2 only and a retrospective hindcast; source release-time availability was not established.
- CEDIA acquisition source granules and HDF5 subsets remain in the authorized KUNTUR workspace; they are not committed to Git.
