# KUNTUR monthly data contract — v1.0.0

**Status:** Frozen for KUNTUR Competition Release v1.0
**Machine-readable contract:** [`configs/data_contract_v1.json`](../configs/data_contract_v1.json)
**Coverage:** January 2002–December 2024, subject to the validated source manifest
**Spatial scope:** preliminary Ecuador pilot bounding box, EPSG:4326, west/south/east/north = `[-81.5, -5.1, -75.0, 1.7]`.

This contract covers data engineering and dataset delivery. It does not approve Coast, Andes, Amazon, administrative, watershed, statistical-trend, model-training, or frontend outputs. The pilot rectangle is not an official Ecuador boundary.

## Source collections and native grids

| Dataset ID | Version | Product class | Native grid | Gold fields |
|---|---:|---|---|---:|
| `M2TMNXSLV.5.12.4` | 5.12.4 | MERRA-2 reanalysis | 0.5° × 0.625° | 6, including source T2M in K and derived T2M in °C |
| `GPM_3IMERGM.07` | 07 | IMERG Final satellite-gauge retrieval | 0.1° × 0.1° | 4, including source monthly mean rate and derived calendar-month total |
| `GLDAS_NOAH025_M.2.1` | 2.1 | GLDAS Noah land-surface model | 0.25° × 0.25° | 8 |
| `MOD13C2.061` | 061 | MODIS/Terra vegetation index | 0.05° geographic CMG | 2, NDVI and EVI |

Each source keeps its native grid and its own stable grid ID. No common-grid interpolation or resampling is performed. Cell centers and bounding-box-clipped spherical cell areas are stored in the processed native-grid Silver files. The bbox area is the denominator for regional coverage; `grid_area_coverage_fraction` shows the part of the bbox represented by the included grid cells.

## Missingness, valid ranges, and quality

- Source fill values, missing values, non-finite values, and source-declared valid ranges are applied before aggregation. Missing observations are never imputed.
- Where a source does not declare a useful range, the contract uses broad physical plausibility screens. These are labeled as contract screens, not NASA-declared ranges.
- Every Gold value carries valid area as a fraction of the bbox and of the represented native grid, fill and range-rejection fractions, cell counts, and QA context. A null value is retained when no valid cell remains.
- MOD13C2 raw int16 NDVI/EVI values are multiplied by `0.0001` per the C6.1 product guide. Raw values and QA arrays are retained in Silver. The strict mask requires MODLAND QA 0, VI usefulness 0 or 1, pixel reliability 0, land-only class, and no adjacent or mixed cloud flag. QA bits 14–15 are retained as geospatial-contribution context for this CMG product; they are not interpreted as snow/shadow flags. NASA guide: [MOD13 User Guide V6.1](https://lpdaac.usgs.gov/documents/621/MOD13_User_Guide_V61.pdf).
- IMERG `precipitationQualityIndex` is retained as an index and is not treated as a probability, confidence interval, or precipitation mask.
- GLDAS monthly `cell_methods` must contain `time: mean` for each contracted field. The four soil-moisture layers, evapotranspiration, and `Qs_acc`, `Qsb_acc`, `Qsm_acc` retain published units and monthly semantics; no field is summed or multiplied by month duration.

## Transformations and temporal meaning

- MERRA-2 `T2M` is preserved in K and additionally represented as `T2M_C = T2M_K - 273.15`.
- IMERG `precipitation` is retained as its monthly mean rate in mm hr⁻¹. The derived monthly accumulation is the rate multiplied by the exact number of proleptic-Gregorian calendar hours for that month (including leap February). Random error remains in mm hr⁻¹.
- GLDAS values retain source physical units and monthly time-mean semantics.
- MODIS NDVI/EVI retain their 0.0001 multiplier and strict QA policy.
- Gold temporal coverage is represented by exact half-open monthly bounds: `[month_start_utc, next_month_start_utc)`.

## Gold schema and lineage

Gold is a long-form Parquet table with one row per `(month, dataset_id, variable_id, region_id)`. Each row records the value and units, source/temporal/spatial meaning, native grid ID, bbox and coverage metrics, QA context, transformation ID, contract/configuration hashes, source granule SHA-256, regional-input SHA-256, and processed-native-grid path and SHA-256.

The complete field dictionary is available as [`data_dictionary_v1.json`](data_dictionary_v1.json) and [`data_dictionary_v1.csv`](data_dictionary_v1.csv). Silver retains the original regional bbox files unchanged and writes decoded, masked, scaled native-grid arrays to `data/silver/processed_native_grid/v1/<source_id>/<year>/<yyyymm>.npz`.

## Forecast dataset contract

The separate forecasting dataset is a leakage-safe, untrained feature table for +1-month regional mean MERRA-2 T2M in °C. At 00:00 UTC on the first day of a forecast-origin month, the latest allowed input is the month two calendar months earlier. Each row uses 12 consecutive complete months ending there; `lag_01` is the latest input month and `lag_12` the oldest. Exact issue date, target month bounds, feature months, target and feature lineage are stored per sample.

Partitions are frozen by target month:

- Train: target months in 2002–2018.
- Validation: target months in 2019–2021.
- Test: target months in 2022–2024.

Targets without the full 12-month history are excluded and enumerated in the split manifest. No random split, imputation, scaling fit, or training is performed.

## Output locations

- Gold: `data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet`.
- Forecast features: `data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet`.
- Forecast split membership: `data/gold/kuntur_forecast_split_indices_v1.parquet`.
- Per-input processing lineage: `data/manifests/preprocessing_manifest_v1.jsonl`.
- Gold and split integrity/coverage records: `data/manifests/gold_manifest_v1.json` and `data/manifests/forecast_split_manifest_v1.json`.

The acquisition and regional-extraction manifests remain the immutable Bronze/Silver inputs and are not rewritten by preprocessing.
