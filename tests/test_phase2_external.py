import pandas as pd

from scripts.evaluate_phase2_external import summarize_pair


def test_external_summary_aligns_only_shared_months_and_reports_station_support():
    reference = pd.Series([20.0, 21.0, 22.0], index=["2022-01", "2022-02", "2022-03"])
    external = pd.Series([20.5, None, 21.5], index=["2022-01", "2022-02", "2022-03"])
    stations = pd.Series([4, 7, 5], index=reference.index)

    result = summarize_pair(
        "synthetic", "temperature", reference, external,
        caveat="synthetic fixture", station_count=stations,
    )

    assert result["n_months"] == 2
    assert result["first_month"] == "2022-01"
    assert result["last_month"] == "2022-03"
    assert result["bias_external_minus_reference"] == 0.0
    assert result["mae"] == 0.5
    assert result["station_count_min"] == 4
    assert result["station_count_max"] == 5


def test_external_summary_handles_no_temporal_overlap():
    reference = pd.Series([20.0], index=["2022-01"])
    external = pd.Series([20.5], index=["2023-01"])

    result = summarize_pair("synthetic", "temperature", reference, external, caveat="synthetic fixture")

    assert result["status"] == "NO_OVERLAP"
    assert result["n_months"] == 0
    assert result["limitation"] == "synthetic fixture"
