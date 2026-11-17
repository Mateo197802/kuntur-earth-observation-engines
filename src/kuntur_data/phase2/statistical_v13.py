"""Validity-gated statistical interface for KUNTUR v1.3.

The v1.3 confirmation did not validate a method across its frozen IID design.
This module therefore exposes descriptive estimates by default and withholds
all p-values and confidence intervals unless a future, independently passing
confirmation registry is present and matches the exact method, regime, sample
support, and data scope. It never infers a real series' error regime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .statistical_v12 import METHOD_IDS, analyze_variable_v12, benjamini_yekutieli_v12


STATISTICAL_VERSION = "kuntur-statistical-inference-v1.3.0"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
V13_REGISTRY_PATH = PROJECT_ROOT / "results/statistical_correction/validated_regime_registry_v1_3.json"
FIXED_ANALYTICAL_FAMILY = (
    "merra2_t2m_c",
    "imerg_precip_monthly_total",
    "gldas_soil_moisture_0_10cm",
    "gldas_soil_moisture_10_40cm",
    "gldas_soil_moisture_40_100cm",
    "gldas_soil_moisture_100_200cm",
    "modis_ndvi",
    "modis_evi",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_project_file(relative_path: str) -> Path | None:
    candidate = (PROJECT_ROOT / relative_path).resolve()
    try:
        candidate.relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _load_confirmed_registry_entry(
    *, registry_path: Path, declared_regime: str | None, method: str,
    n_calendar_months: int, calendar_coverage: float, data_scope: str,
) -> tuple[dict[str, Any] | None, str]:
    """Return a verified registry and reason; fail closed on every mismatch."""
    if not registry_path.is_file():
        return None, "NO_INDEPENDENTLY_VALIDATED_V1_3_REGISTRY"
    try:
        Path(registry_path).resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return None, "REGISTRY_PATH_OUTSIDE_PROJECT"
    if not declared_regime:
        return None, "NO_CALLER_DECLARED_REGIME"
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "REGISTRY_UNREADABLE"
    if registry.get("schema_version") != "kuntur-validated-regime-registry-v1.3" or registry.get("status") != "FROZEN":
        return None, "REGISTRY_NOT_FROZEN_V1_3"
    entry = registry.get("regimes", {}).get(str(declared_regime))
    if not isinstance(entry, dict) or entry.get("validity_status") != "VALIDATED_FOR_DECLARED_REGIME":
        return None, "REGIME_NOT_IN_VALIDATED_REGISTRY"
    if entry.get("method_key", entry.get("method")) != method or entry.get("method_id") != METHOD_IDS[method]:
        return None, "METHOD_REGIME_MISMATCH"
    minimum = entry.get("calendar_months_min", entry.get("n_calendar_months_min"))
    maximum = entry.get("calendar_months_max", entry.get("n_calendar_months_max"))
    coverage_min = entry.get("calendar_coverage_min")
    coverage_max = entry.get("calendar_coverage_max")
    if not all(isinstance(value, (int, float)) for value in (minimum, maximum, coverage_min, coverage_max)):
        return None, "REGISTRY_SUPPORT_BOUNDS_MISSING"
    if not int(minimum) <= n_calendar_months <= int(maximum):
        return None, "CALENDAR_LENGTH_OUTSIDE_VALIDATED_SUPPORT"
    if not float(coverage_min) <= calendar_coverage <= float(coverage_max):
        return None, "CALENDAR_COVERAGE_OUTSIDE_VALIDATED_SUPPORT"
    allowed_scopes = entry.get("allowed_data_scopes")
    if allowed_scopes is None and "independent simulation only" in str(registry.get("evidence_scope", "")).lower():
        # The frozen confirmation schema validates simulated DGPs only. It
        # never supplies applicability evidence for real environmental data.
        allowed_scopes = ["synthetic_validation"]
    if data_scope not in (allowed_scopes or []):
        return None, "DATA_SCOPE_NOT_VALIDATED"

    evidence = entry.get("independent_confirmation", {})
    metrics_relative_path = evidence.get("metrics_path", entry.get("confirmation_metrics_path", ""))
    expected_hash = str(evidence.get("metrics_sha256", registry.get("source_confirmation_metrics_sha256", "")))
    metrics_path = _safe_project_file(str(metrics_relative_path))
    if metrics_path is None or len(expected_hash) != 64 or _sha256(metrics_path) != expected_hash:
        return None, "CONFIRMATION_METRICS_MISSING_OR_HASH_MISMATCH"
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "CONFIRMATION_METRICS_UNREADABLE"
    candidate = metrics.get("candidate", {})
    if not str(metrics.get("status", "")).startswith("INDEPENDENT_CONFIRMATION_PASS"):
        return None, "INDEPENDENT_CONFIRMATION_DID_NOT_PASS"
    if metrics.get("all_gates_pass") is not True or candidate.get("method_key") != method or candidate.get("regime_id") != declared_regime:
        return None, "CONFIRMATION_EVIDENCE_DOES_NOT_MATCH_REGISTRY_ENTRY"
    run_manifest_path = _safe_project_file(str(metrics.get("run_manifest_path", "")))
    run_manifest_hash = str(metrics.get("run_manifest_sha256", ""))
    if run_manifest_path is None or len(run_manifest_hash) != 64 or _sha256(run_manifest_path) != run_manifest_hash:
        return None, "CONFIRMATION_RUN_MANIFEST_MISSING_OR_HASH_MISMATCH"
    cells = metrics.get("cells", [])
    if not cells or any(cell.get("all_gates_pass") is not True for cell in cells):
        return None, "CONFIRMATION_HAS_A_FAILED_OR_MISSING_CELL"
    return registry, "MATCHED_FROZEN_INDEPENDENT_CONFIRMATION"


def analyze_variable_v13(
    gold: pd.DataFrame,
    variable_id: str,
    *,
    region_id: str | None = None,
    start_month: str | None = None,
    end_month: str | None = None,
    method: str = "ols_hac",
    qa_policy: dict[str, Any] | None = None,
    declared_regime: str | None = None,
    mode: Literal["descriptive", "confirmatory"] = "descriptive",
    data_scope: str = "real_gold",
    registry_path: Path = V13_REGISTRY_PATH,
    hac_lag_months: int = 12,
) -> dict[str, Any]:
    """Analyze a monthly Gold variable, emitting inference only on a strict match.

    Descriptive slopes and calendar/QA diagnostics are available without a
    registry. Confirmatory p-values and intervals require an independently
    passing v1.3 registry whose evidence hash is verified at runtime. A regime
    is never inferred from the observed values.
    """
    if mode not in {"descriptive", "confirmatory"}:
        raise ValueError("mode must be 'descriptive' or 'confirmatory'")
    if method not in METHOD_IDS:
        raise ValueError(f"Unsupported statistical method: {method}")

    registry: dict[str, Any] | None = None
    registry_reason = "DESCRIPTIVE_MODE_INFERENCE_WITHHELD"
    if mode == "confirmatory":
        required = {"month", "variable_id", "value"}
        absent = required - set(gold.columns)
        if absent:
            raise ValueError(f"Gold frame is missing columns: {sorted(absent)}")
        selected = gold.loc[gold.variable_id.astype(str) == str(variable_id)].copy()
        if region_id is not None:
            if "region_id" not in selected:
                raise ValueError("region_id was requested but is absent from the Gold frame")
            selected = selected.loc[selected.region_id.astype(str) == str(region_id)]
        if start_month is not None:
            selected = selected.loc[selected.month.astype(str).str.slice(0, 7) >= start_month]
        if end_month is not None:
            selected = selected.loc[selected.month.astype(str).str.slice(0, 7) <= end_month]
        if selected.empty:
            raise ValueError(f"No Gold rows for variable_id={variable_id}, region_id={region_id}")
        periods = pd.PeriodIndex(selected.month.astype(str).str.slice(0, 7), freq="M")
        if periods.has_duplicates:
            raise ValueError("duplicate calendar month for variable_id")
        start = pd.Period(start_month, freq="M") if start_month else periods.min()
        end = pd.Period(end_month, freq="M") if end_month else periods.max()
        full_count = len(pd.period_range(start, end, freq="M"))
        values = pd.to_numeric(selected.value, errors="coerce").to_numpy(dtype=float)
        if "valid_area_fraction_bbox" in selected:
            coverage = pd.to_numeric(selected.valid_area_fraction_bbox, errors="coerce").to_numpy(dtype=float)
        else:
            coverage = np.ones(len(selected), dtype=float)
        threshold = float((qa_policy or {}).get("min_valid_area_fraction_bbox", 0.1 if str(variable_id).startswith("modis_") else 0.8))
        eligible = np.isfinite(values) & np.isfinite(coverage) & (coverage >= threshold)
        calendar_coverage = float(eligible.sum() / full_count) if full_count else 0.0
        synthetic_scope_valid = True
        if data_scope == "synthetic_validation":
            marker = selected.get("synthetic_dgp_id")
            synthetic_scope_valid = bool(
                marker is not None
                and marker.notna().all()
                and declared_regime is not None
                and set(marker.astype(str).unique()) == {str(declared_regime)}
            )
            if not synthetic_scope_valid:
                registry_reason = "SYNTHETIC_SCOPE_PROVENANCE_MISSING_OR_MISMATCHED"
        if full_count and synthetic_scope_valid:
            registry, registry_reason = _load_confirmed_registry_entry(
                registry_path=Path(registry_path), declared_regime=declared_regime,
                method=method, n_calendar_months=full_count,
                calendar_coverage=calendar_coverage, data_scope=data_scope,
            )
        elif not full_count:
            registry_reason = "NO_CALENDAR_SUPPORT"

    # v1.2 supplies the established calendar/QA estimators. Its inference gate
    # receives only the registry already verified by this v1.3 layer.
    normalized_registry = None
    if registry is not None:
        entry = registry["regimes"][str(declared_regime)]
        normalized_entry = dict(entry)
        normalized_entry["calendar_months_min"] = entry.get("calendar_months_min", entry.get("n_calendar_months_min"))
        normalized_entry["calendar_months_max"] = entry.get("calendar_months_max", entry.get("n_calendar_months_max"))
        normalized_registry = {"status": "FROZEN", "regimes": {str(declared_regime): normalized_entry}}

    result = analyze_variable_v12(
        gold, variable_id, region_id=region_id, start_month=start_month,
        end_month=end_month, method=method, qa_policy=qa_policy,
        declared_regime=declared_regime, validation_registry=normalized_registry,
        hac_lag_months=hac_lag_months,
    )
    # v1.3 fails closed for any future malformed/incomplete registry result.
    inferentially_valid = bool(
        mode == "confirmatory"
        and registry is not None
        and result.get("validity_status") == "VALIDATED_FOR_DECLARED_REGIME"
        and result.get("p_value") is not None
        and result.get("confidence_interval_95") is not None
    )
    if not inferentially_valid:
        result = {**result, "p_value": None, "confidence_interval_95": None, "standard_error_is_inferentially_valid": False}
    result.update({
        "statistical_version": STATISTICAL_VERSION,
        "analysis_mode": mode,
        "data_scope": data_scope,
        "inference_status": "VALIDATED_FOR_DECLARED_REGIME" if inferentially_valid else "WITHHELD_NO_VALIDATED_METHOD_REGIME",
        "registry_gate_reason": registry_reason,
        "provenance": {
            **result.get("provenance", {}),
            "statistical_version": STATISTICAL_VERSION,
            "v1_3_registry_path": str(Path(registry_path)) if registry is not None else None,
            "v1_3_registry_sha256": _sha256(Path(registry_path)) if registry is not None else None,
            "regime_inferred_automatically": False,
            "confirmation_registry_verified": registry is not None,
        },
    })
    return result


def adjust_fixed_family_fdr_v13(
    results: list[dict[str, Any]],
    *,
    expected_variable_ids: tuple[str, ...] = FIXED_ANALYTICAL_FAMILY,
) -> dict[str, Any]:
    """Apply BY only to the complete, predeclared family of valid p-values."""
    if len(set(expected_variable_ids)) != len(expected_variable_ids):
        raise ValueError("expected_variable_ids must be unique")
    by_id: dict[str, dict[str, Any]] = {}
    for row in results:
        key = str(row.get("variable_id", ""))
        if key in by_id:
            raise ValueError(f"Duplicate result for variable_id={key}")
        if key not in expected_variable_ids:
            raise ValueError(f"Unexpected variable outside the frozen family: {key}")
        by_id[key] = row
    if set(by_id) != set(expected_variable_ids):
        return {
            "status": "WITHHELD_INCOMPLETE_FAMILY",
            "method": "Benjamini-Yekutieli",
            "family_size": len(expected_variable_ids),
            "q_values": {key: None for key in expected_variable_ids},
        }
    records = [by_id[key] for key in expected_variable_ids]
    regimes = {row.get("declared_regime") for row in records}
    valid = all(
        row.get("inference_status") == "VALIDATED_FOR_DECLARED_REGIME"
        and row.get("p_value") is not None
        and np.isfinite(float(row["p_value"]))
        for row in records
    ) and len(regimes) == 1
    if not valid:
        return {
            "status": "WITHHELD_FAMILY_HAS_UNVALIDATED_OR_MIXED_REGIMES",
            "method": "Benjamini-Yekutieli",
            "family_size": len(expected_variable_ids),
            "q_values": {key: None for key in expected_variable_ids},
        }
    adjusted = benjamini_yekutieli_v12([float(row["p_value"]) for row in records])
    return {
        "status": "ADJUSTED_VALIDATED_COMPLETE_FAMILY",
        "method": "Benjamini-Yekutieli",
        "family_size": len(expected_variable_ids),
        "declared_regime": next(iter(regimes)),
        "q_values": {key: float(value) for key, value in zip(expected_variable_ids, adjusted)},
    }
