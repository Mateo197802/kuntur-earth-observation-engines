# KUNTUR Phase 2 v1.2 release-candidate audit

**Audit date:** 2026-11-16 (America/Guayaquil)

**Repository:** `Mateo197802/kuntur-earth-observation-engines` (public GitHub repository)

**License:** the existing all-rights-reserved notice remains unchanged.

## Results delivered

| Component | State | Evidence and scope |
|---|---|---|
| Trend engine | DONE — descriptive output only | Eight variable estimates in `results/statistical_correction/real_series_v12/descriptive_trend_estimates.csv`, SHA-256 `43ee306cd8e1d93b541d5b2d3d7a6b6a5fd854b09788b315a271b58f4ee4ca60`. Environmental p-values and intervals remain withheld pending independent method calibration. MODIS has 144/276 eligible months; the other six variables have 276/276. |
| Six architecture families and baselines | DONE — development comparison | 106 candidate aggregates across rounds 1–3. Best RBF-SVR in round 1: MAE 0.31760 °C; change vs v1.1 SVR −0.00520 °C, paired 95% block interval [−0.01121, +0.000084], inconclusive. |
| Feature and augmentation rounds | DONE — post-selection | Round 2: 48 candidates / 144 candidate-fold evaluations; best RF MAE 0.27355 °C. Round 3: 28 candidates / 84 candidate-fold evaluations, 11 noise scenarios at both 25% and 50%; selected heteroscedastic-only RF MAE 0.25776 °C, RMSE 0.32818 °C, bias +0.04351 °C, worst-fold MAE 0.30480 °C. |
| Frozen temporal test | DONE — negative result preserved | The frozen 2025 MERRA-2 retrospective hindcast had RF MAE 0.32513 °C versus training-only monthly climatology MAE 0.16591 °C; RF was better in 0/12 months. No test-driven retuning. The 12-month score is low precision and does not establish prospective skill. |
| Prediction intervals | DONE — descriptive, development-dependent | Development interval assessment: 72 targets, 91.7% observed coverage, 1.212 °C mean width, 1.452 °C Winkler score; post-selection and serial dependence limit interpretation. The 2025 interval covered 10/12 months (83.3%), also descriptive only. |
| External observations | DONE — descriptive point-to-grid comparison | Independently verified on CEDIA against 276 MERRA-2 native-grid months. NOAA: 2,446 eligible station-months, MAE 1.90033 °C, RMSE 2.41252 °C, bias −0.86508 °C, r 0.93374. INAMHI: 10,499 eligible station-months over 132 months (2002–2012), MAE 2.98229 °C, RMSE 3.78409 °C, bias −1.23304 °C, r 0.80522. This is not independent ground truth: reanalysis may assimilate stations, spatial support/elevation differ, NOAA coverage is sparse, and INAMHI coverage ends in 2012. |
| Partitioned ingestion and global cache | DONE — bounded local benchmark | Ten repeats on the 5,520-row Gold; exact results against full-file baseline; four concurrent 276-row queries passed; measured on Windows 11, not a cluster SLA. In the 3-year/single-variable case, warm cache was 64.7× faster than partitioned uncached median. Peak RSS 395,935,744 bytes. See the v1 ingestion/cache manifest and public aggregate CSV. |
| Reproduction tests | DONE | `python -m pytest -q`: **140 passed**. The round 2 and round 3 modules now support both direct-script and package imports. Frozen model smoke used only a synthetic 28-feature row; all three RF members returned finite predictions. |
| Calibration selection | DONE — decisions frozen | All ten locked process/slope scenarios completed 500/500. `selection_decisions.json` has status `FROZEN_BEFORE_INDEPENDENT_VALIDATION`; protocol SHA-256 `f2fdf0e7941eb548568b8e1979c89799433872df131a36ca56709bef395a65b8`; validation seed base 20261127. |
| Independent statistical validation | RUNNING | Sixty scenarios are locked; five completed 1,000/1,000 (IID Gaussian zero slope, AR(1) ρ=0.844 zero slope, ARMA(1,1) zero slope, seasonal AR(1) zero slope, and AR(1) ρ=0.6 with 20% MCAR zero slope). At 15:38:03 local time, scenario 6 (`ar1_m030`, zero slope) was at 75/1,000 outer replicates with 399 nested draws. Worker elapsed 53:21, CPU 3,222 seconds, working set 190 MiB. Progress is not phase-level validation. Inferential trend p-values, intervals, and q-values remain withheld until acceptance gates are evaluated. |
| Formal Codex Security scan | BLOCKED | The scan launcher returned: “The selected scan target changed while the scan was starting. Try again.” The scanner skill prohibits retry or substitution in this run; no formal security pass is claimed. A content-level public-path/credential check and automated repository tests passed, and the repository PR workflow runs Gitleaks. |
| GitHub pull request | DRAFT — OPEN | Public PR #1 is open on `Mateo197802/kuntur-earth-observation-engines`; it will remain a draft while statistical calibration and the formal scan are incomplete. Repository visibility and license were not changed. |

## External collocation artifacts

The station-month detail and monthly source summaries remain in the permission-restricted CEDIA workspace and are not committed. Only the aggregate summary and independent validation receipt are included in the release artifacts:

- `results/optimization/native_grid_collocation_summary.json` — SHA-256 `1766aec1e5a42fea2eb5a4a2773d51ebe5ad39c12f53197952ee393be1fecea9`.
- `results/optimization/native_grid_collocation_validation.json` — SHA-256 `982eed21a7c0f1e3e8b15b24e7c7ccda11c191fa499f816443c8b9161730043d`.
- The verifier checked 276 native-grid NPZ files and acquisition/extraction lineage links, 12,945 unique station-month comparisons, and 552 source-month summaries.

The preliminary Ecuador bounding box remains the only KUNTUR region. Coast, Andes, Amazon, and the external-comparison masks still require approved geographic definitions. No new NASA data were downloaded because the existing validated 2025 MERRA-2 granules and regional outputs were already present.

## Release decision

The predictive development, frozen-test report, benchmark, external point-to-grid check, aggregate registry, dashboard, and reproducibility suite are reviewable. The statistical release gate is **not complete**: selection decisions are frozen and independent validation is running. The results do not support releasing calibrated environmental inference yet. The GitHub repository is public; its visibility and all-rights-reserved license were not changed.
