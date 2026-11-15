# KUNTUR Phase 2 Python interfaces

The interface module is `kuntur_data.phase2.api`. It returns plain Python dictionaries suitable for an API adapter; it does not create a web server.

## Analytical query

Create a `PartitionedGoldStore` using the locally staged frozen Gold table and its partition directory, and provide a `GlobalAnalyticalCache`. Call `store.materialize()` only when no matching partition manifest exists. Existing manifests are reused when their Gold SHA and contract version match.

```python
from pathlib import Path
from kuntur_data.phase2.ingestion import GlobalAnalyticalCache, PartitionedGoldStore
from kuntur_data.phase2.api import query_analytical

root = Path(".")
cache = GlobalAnalyticalCache(root / "data/cache/phase2_analytical_v1")
store = PartitionedGoldStore(
    root / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
    root / "data/derived/phase2/partitioned_gold_v1",
    cache=cache,
)
store.materialize()
result = query_analytical(
    store,
    region_id="ecuador_pilot_bbox",
    variable_id="merra2_t2m_c",
    start_month="2002-01",
    end_month="2024-12",
    trend_method="ols_hac",
    qa_threshold=0.8,
)
```

Supported trend methods are `ols_hac`, `seasonal_theil_sen`, and `seasonal_mann_kendall`. The response includes monthly values/anomalies, units, the selected estimate and method-appropriate uncertainty, autocorrelation diagnostics, coverage, cache metadata, Gold/contract provenance, and source hashes. The Theil–Sen seasonal-stratum range is not a confidence interval. The only approved v1 region is the preliminary Ecuador pilot bbox; other region IDs raise an error until masks are approved.

## +1-month temperature forecast

`forecast_temperature` accepts caller-supplied 12-month history in long form, one row per `month` and `variable_id`, with `value`, `valid_area_fraction_bbox`, and optional `source_sha256`. Required channels are `merra2_t2m_c`, `merra2_qv2m`, `merra2_u2m`, `merra2_v2m`, and `merra2_ps`. The 12 contiguous months must end at `T-3`; issue month is `T-1`. Values below 80% bbox coverage are treated as missing, training-only channel means and missingness indicators are used, and history with more than 20% missing cells is rejected.

The packaged inference model is the frozen final-refit histogram gradient-boosting artifact. Other model IDs and horizons are rejected. The output includes target and issue dates, prediction in °C, persistence/seasonal-naive/training-only climatology baselines, a validation-residual interval when supplied, input/model hashes, and aggregate final-test metrics path. The interval is descriptive: observed 90% interval coverage on the final test was 55.6%, so it is not a calibrated guarantee.

```python
from pathlib import Path
from kuntur_data.phase2.api import forecast_temperature

forecast = forecast_temperature(
    history=history_frame,
    region_id="ecuador_pilot_bbox",
    initialization_month="2024-12",
    model_id="hist_gradient_boosting",
    horizon_months=1,
    model_path=Path("results/phase2/models/final_v1/hist_gradient_boosting_final_refit_v1.joblib"),
    gold_sha256="<verified Gold SHA-256>",
    training_climatology=training_only_monthly_means,
    interval_radius_deg_c=0.5319413481041906,
)
```

The serving application must source current history and training-only climatology from approved, provenance-tracked inputs. It must not use validation/test targets to fit, select, or recalibrate a model.
