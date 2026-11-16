from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.run_phase2_v12_descriptive_trends import VARIABLES, build_trend_rows


def test_actual_trend_export_contract_withholds_unvalidated_inference() -> None:
    months = pd.period_range("2002-01", "2024-12", freq="M").astype(str)
    records = []
    for variable_index, variable_id in enumerate(VARIABLES):
        for index, month in enumerate(months):
            coverage = 0.5 if variable_id.startswith("modis_") else 0.9
            records.append({
                "month": month,
                "variable_id": variable_id,
                "region_id": "ecuador_pilot_bbox",
                "value": float(variable_index + index / 12.0 + np.sin(index * 2 * np.pi / 12)),
                "valid_area_fraction_bbox": coverage,
                "units": "test_unit",
            })
    rows = build_trend_rows(pd.DataFrame(records))
    assert len(rows) == 8
    for row in rows:
        assert row["n_calendar_months"] == 276
        assert row["n_eligible_observations"] == 276
        assert row["p_value"] is None
        assert row["confidence_interval_95"] is None
        assert row["inference_status"] == "WITHHELD_NO_INDEPENDENTLY_VALIDATED_METHOD_REGIME"
        assert row["slope_per_decade"] is not None
