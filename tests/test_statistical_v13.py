from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import kuntur_data.phase2.statistical_v13 as statistical_v13
from kuntur_data.phase2.statistical_v13 import (
    FIXED_ANALYTICAL_FAMILY,
    adjust_fixed_family_fdr_v13,
    analyze_variable_v13,
)
from scripts.statistical_correction.render_confirmation_gate_correction_v1_3 import gate_label
from scripts.run_phase2_v13_descriptive_trends import VARIABLES, build_trend_rows


def _gold_with_calendar_gap() -> pd.DataFrame:
    months = pd.period_range("2000-01", "2003-12", freq="M")
    frame = pd.DataFrame({
        "month": months.astype(str),
        "variable_id": "fixture_temperature",
        "region_id": "fixture_region",
        "value": np.linspace(0.0, 4.0, len(months)) + np.sin(np.arange(len(months)) * 2 * np.pi / 12),
        "valid_area_fraction_bbox": 0.95,
        "units": "degC",
    })
    frame.loc[frame.month == "2001-07", "value"] = np.nan
    return frame


def test_v13_descriptive_mode_retains_calendar_gap_and_withholds_inference() -> None:
    result = analyze_variable_v13(
        _gold_with_calendar_gap(), "fixture_temperature", region_id="fixture_region",
        start_month="2000-01", end_month="2003-12", mode="descriptive",
    )

    assert result["statistical_version"] == "kuntur-statistical-inference-v1.3.0"
    assert result["n_calendar_months"] == 48
    assert result["n_eligible_observations"] == 47
    assert result["missing_calendar_months"] == ["2001-07"]
    assert result["slope_per_decade"] is not None
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None
    assert result["inference_status"] == "WITHHELD_NO_VALIDATED_METHOD_REGIME"
    assert result["provenance"]["regime_inferred_automatically"] is False


def test_v13_confirmatory_mode_without_registry_withholds_inference(tmp_path) -> None:
    result = analyze_variable_v13(
        _gold_with_calendar_gap(), "fixture_temperature", region_id="fixture_region",
        start_month="2000-01", end_month="2003-12", method="ols_hac",
        mode="confirmatory", declared_regime="iid_gaussian",
        registry_path=tmp_path / "missing_registry.json", data_scope="real_gold",
    )

    assert result["registry_gate_reason"] == "NO_INDEPENDENTLY_VALIDATED_V1_3_REGISTRY"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None
    assert result["standard_error_is_inferentially_valid"] is False


def test_v13_fixed_family_withholds_when_any_series_is_missing() -> None:
    rows = [{"variable_id": FIXED_ANALYTICAL_FAMILY[0], "p_value": 0.01, "inference_status": "VALIDATED_FOR_DECLARED_REGIME", "declared_regime": "r"}]
    result = adjust_fixed_family_fdr_v13(rows)

    assert result["status"] == "WITHHELD_INCOMPLETE_FAMILY"
    assert set(result["q_values"]) == set(FIXED_ANALYTICAL_FAMILY)
    assert all(value is None for value in result["q_values"].values())


def test_v13_fixed_family_applies_by_only_to_eight_valid_same_regime_p_values() -> None:
    rows = [
        {"variable_id": variable, "p_value": (index + 1) / 100.0,
         "inference_status": "VALIDATED_FOR_DECLARED_REGIME", "declared_regime": "approved_regime"}
        for index, variable in enumerate(FIXED_ANALYTICAL_FAMILY)
    ]
    result = adjust_fixed_family_fdr_v13(rows)

    assert result["status"] == "ADJUSTED_VALIDATED_COMPLETE_FAMILY"
    assert result["method"] == "Benjamini-Yekutieli"
    assert result["family_size"] == 8
    assert set(result["q_values"]) == set(FIXED_ANALYTICAL_FAMILY)
    assert all(0.0 <= value <= 1.0 for value in result["q_values"].values())


def test_v13_fixed_family_withholds_mixed_regimes() -> None:
    rows = [
        {"variable_id": variable, "p_value": 0.01,
         "inference_status": "VALIDATED_FOR_DECLARED_REGIME",
         "declared_regime": "regime_a" if index < 7 else "regime_b"}
        for index, variable in enumerate(FIXED_ANALYTICAL_FAMILY)
    ]
    result = adjust_fixed_family_fdr_v13(rows)

    assert result["status"] == "WITHHELD_FAMILY_HAS_UNVALIDATED_OR_MIXED_REGIMES"
    assert all(value is None for value in result["q_values"].values())


def test_v13_fixed_family_rejects_unknown_or_duplicate_variables() -> None:
    with pytest.raises(ValueError, match="Unexpected variable"):
        adjust_fixed_family_fdr_v13([{"variable_id": "not_in_family"}])
    with pytest.raises(ValueError, match="Duplicate result"):
        adjust_fixed_family_fdr_v13([
            {"variable_id": FIXED_ANALYTICAL_FAMILY[0]},
            {"variable_id": FIXED_ANALYTICAL_FAMILY[0]},
        ])


def _write_synthetic_registry(tmp_path, *, method: str = "ols_hac") -> Path:
    import hashlib
    import json

    confirmation_dir = tmp_path / "results" / "confirmation"
    confirmation_dir.mkdir(parents=True)
    run_manifest = confirmation_dir / "confirmation_manifest.json"
    run_manifest.write_text('{"status":"INDEPENDENT_CONFIRMATION_COMPLETE"}\n', encoding="utf-8")
    manifest_hash = hashlib.sha256(run_manifest.read_bytes()).hexdigest()
    metrics = {
        "status": "INDEPENDENT_CONFIRMATION_PASS_FOR_IID_SYNTHETIC_REGIME",
        "all_gates_pass": True,
        "candidate": {"method_key": method, "regime_id": "iid_gaussian"},
        "cells": [{"all_gates_pass": True}],
        "run_manifest_path": "results/confirmation/confirmation_manifest.json",
        "run_manifest_sha256": manifest_hash,
    }
    metrics_path = confirmation_dir / "confirmation_metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
    metrics_hash = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
    method_id = statistical_v13.METHOD_IDS[method]
    minimum_months = 48
    registry = {
        "schema_version": "kuntur-validated-regime-registry-v1.3",
        "status": "FROZEN",
        "evidence_scope": "independent simulation only",
        "source_confirmation_metrics_sha256": metrics_hash,
        "regimes": {
            "iid_gaussian": {
                "validity_status": "VALIDATED_FOR_DECLARED_REGIME",
                "method": method,
                "method_id": method_id,
                "n_calendar_months_min": minimum_months,
                "n_calendar_months_max": minimum_months,
                "calendar_coverage_min": 1.0,
                "calendar_coverage_max": 1.0,
                "confirmation_metrics_path": "results/confirmation/confirmation_metrics.json",
            }
        },
    }
    registry_path = tmp_path / "results" / "statistical_correction" / "validated_regime_registry_v1_3.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    return registry_path


def test_v13_accepts_current_frozen_registry_schema_for_synthetic_scope_only(tmp_path, monkeypatch) -> None:
    registry_path = _write_synthetic_registry(tmp_path)
    monkeypatch.setattr(statistical_v13, "PROJECT_ROOT", tmp_path)
    registry, reason = statistical_v13._load_confirmed_registry_entry(
        registry_path=registry_path, declared_regime="iid_gaussian", method="ols_hac",
        n_calendar_months=48, calendar_coverage=1.0, data_scope="synthetic_validation",
    )
    real_registry, real_reason = statistical_v13._load_confirmed_registry_entry(
        registry_path=registry_path, declared_regime="iid_gaussian", method="ols_hac",
        n_calendar_months=48, calendar_coverage=1.0, data_scope="real_gold",
    )

    assert registry is not None
    assert reason == "MATCHED_FROZEN_INDEPENDENT_CONFIRMATION"
    assert real_registry is None
    assert real_reason == "DATA_SCOPE_NOT_VALIDATED"


def test_v13_confirmatory_adapter_checks_frozen_metric_and_manifest_hashes(tmp_path, monkeypatch) -> None:
    registry_path = _write_synthetic_registry(tmp_path)
    monkeypatch.setattr(statistical_v13, "PROJECT_ROOT", tmp_path)
    months = pd.period_range("2000-01", periods=48, freq="M")
    data = pd.DataFrame({
        "month": months.astype(str),
        "variable_id": "fixture_temperature",
        "region_id": "fixture_region",
        "synthetic_dgp_id": "iid_gaussian",
        "value": 0.5 * np.arange(48) / 12.0 + np.sin(np.arange(48) * 2 * np.pi / 12),
        "valid_area_fraction_bbox": 1.0,
        "units": "degC",
    })
    result = analyze_variable_v13(
        data, "fixture_temperature", region_id="fixture_region", method="ols_hac",
        mode="confirmatory", declared_regime="iid_gaussian", data_scope="synthetic_validation",
        registry_path=registry_path,
    )
    real_result = analyze_variable_v13(
        data, "fixture_temperature", region_id="fixture_region", method="ols_hac",
        mode="confirmatory", declared_regime="iid_gaussian", data_scope="real_gold",
        registry_path=registry_path,
    )

    assert result["registry_gate_reason"] == "MATCHED_FROZEN_INDEPENDENT_CONFIRMATION"
    assert result["inference_status"] == "VALIDATED_FOR_DECLARED_REGIME"
    assert result["p_value"] is not None
    assert result["confidence_interval_95"] is not None
    assert real_result["registry_gate_reason"] == "DATA_SCOPE_NOT_VALIDATED"
    assert real_result["p_value"] is None
    assert real_result["confidence_interval_95"] is None


def test_v13_cannot_claim_synthetic_scope_without_row_level_dgp_marker(tmp_path) -> None:
    data = _gold_with_calendar_gap()
    result = analyze_variable_v13(
        data, "fixture_temperature", region_id="fixture_region", method="ols_hac",
        mode="confirmatory", declared_regime="iid_gaussian", data_scope="synthetic_validation",
        registry_path=tmp_path / "missing_registry.json",
    )

    assert result["registry_gate_reason"] == "SYNTHETIC_SCOPE_PROVENANCE_MISSING_OR_MISMATCHED"
    assert result["p_value"] is None
    assert result["confidence_interval_95"] is None


def test_confirmation_report_labels_nonzero_slope_type_i_as_not_applicable() -> None:
    assert gate_label("type_i_rate", None, 0.25) == "N/A (Type-I is evaluated only at zero slope)"
    assert gate_label("type_i_rate", True, 0.0) == "PASS"
    with pytest.raises(ValueError, match="must be null"):
        gate_label("type_i_rate", False, 0.5)


def test_public_handoff_preserves_original_confirmation_failure() -> None:
    import json
    from pathlib import Path

    public_path = Path(__file__).resolve().parents[1] / "results/phase2/public/statistical_validation_v1_3_2.json"
    public = json.loads(public_path.read_text(encoding="utf-8"))
    original = public["original_v1_3_wald_confirmation"]
    corrected = public["v1_3_2_independent_confirmation"]
    assert original["status"] == "FAILED_INDEPENDENT_CONFIRMATION"
    assert original["slope_0_50_available"] == 998
    assert original["slope_0_50_covered"] == 920
    assert original["coverage"] == pytest.approx(0.9218436873747495)
    assert corrected["decision"] == "INDEPENDENT_CONFIRMATION_PASS_FOR_IID_SYNTHETIC_SCOPE"
    assert public["registry_and_protected_inputs"]["inference_registry_modified"] is False


def test_v13_descriptive_export_contract_has_eight_slopes_and_no_inference() -> None:
    months = pd.period_range("2002-01", "2024-12", freq="M").astype(str)
    records = []
    for variable_index, variable_id in enumerate(VARIABLES):
        for index, month in enumerate(months):
            records.append({
                "month": month,
                "variable_id": variable_id,
                "region_id": "ecuador_pilot_bbox",
                "value": float(variable_index + index / 12.0 + np.sin(index * 2 * np.pi / 12)),
                "valid_area_fraction_bbox": 0.5 if variable_id.startswith("modis_") else 0.9,
                "units": "fixture_unit",
            })
    rows = build_trend_rows(pd.DataFrame(records))

    assert len(rows) == 8
    assert {row["variable_id"] for row in rows} == set(VARIABLES)
    assert all(row["n_calendar_months"] == 276 for row in rows)
    assert all(row["slope_per_decade"] is not None for row in rows)
    assert all(row["p_value"] is None and row["confidence_interval_95"] is None for row in rows)
    assert all(row["inference_status"] == "WITHHELD_NO_VALIDATED_METHOD_REGIME" for row in rows)
