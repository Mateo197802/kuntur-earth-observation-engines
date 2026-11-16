#!/usr/bin/env python3
"""Create a strict-JSON v1.1 derivative of the frozen v1 public summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/phase2/public/phase2_aggregate_summary_v1.json"
INAMHI = ROOT / "data/external/processed/inamhi_pilot_bbox_monthly_composite_2002_2024_v1.parquet"
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
OUT = ROOT / "results/phase2/v1_1/public/phase2_aggregate_summary_v1_1.json"
MANIFEST = ROOT / "results/phase2/v1_1/public/phase2_aggregate_summary_manifest_v1_1.json"
NONFINITE = {"__json_nonfinite__": True}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_nonstandard_constant(token: str):
    if token not in {"NaN", "Infinity", "-Infinity"}:
        raise ValueError(f"Unsupported JSON constant: {token}")
    return {**NONFINITE, "token": token}


def sanitize(value, pointer: str = "") -> tuple[object, list[dict]]:
    found: list[dict] = []
    if isinstance(value, dict):
        if value.get("__json_nonfinite__") is True:
            found.append({"json_pointer": pointer or "/", "original_token": value.get("token", "NaN")})
            return None, found
        output = {}
        for key, item in value.items():
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            safe_item, child_found = sanitize(item, pointer + "/" + escaped)
            output[key] = safe_item
            found.extend(child_found)
        return output, found
    if isinstance(value, list):
        output = []
        for index, item in enumerate(value):
            safe_item, child_found = sanitize(item, pointer + "/" + str(index))
            output.append(safe_item)
            found.extend(child_found)
        return output, found
    if isinstance(value, float) and not np.isfinite(value):
        found.append({"json_pointer": pointer or "/", "original_token": str(value)})
        return None, found
    return value, found


def inamhi_station_count_ranges() -> dict[str, dict]:
    composite = pd.read_parquet(INAMHI).set_index("month").sort_index()
    gold = pd.read_parquet(GOLD)
    result = {}
    for source_variable, external_value, count_column, threshold in (
        ("merra2_t2m_c", "inamhi_equal_station_tavg_c", "inamhi_tavg_station_count", 0.8),
        ("imerg_precip_monthly_total", "inamhi_equal_station_precipitation_mm_month", "inamhi_precipitation_station_count", 0.8),
    ):
        reference = gold[(gold.region_id == "ecuador_pilot_bbox") & (gold.variable_id == source_variable) & (gold.valid_area_fraction_bbox >= threshold)].set_index("month")
        matched = composite.join(reference[["value"]], how="inner").dropna(subset=[external_value, "value"])
        matched = matched[matched[count_column] >= 3]
        counts = matched[count_column]
        result[source_variable] = {
            "n_months": int(len(counts)),
            "station_count_min": int(counts.min()) if len(counts) else None,
            "station_count_median": float(counts.median()) if len(counts) else None,
            "station_count_max": int(counts.max()) if len(counts) else None,
            "manifest": "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json",
        }
    return result


def main() -> None:
    if not SOURCE.is_file():
        raise SystemExit(f"Frozen v1 aggregate summary missing: {SOURCE}")
    source_text = SOURCE.read_text(encoding="utf-8")
    parsed = json.loads(source_text, parse_constant=parse_nonstandard_constant)
    sanitized, nonfinite_fields = sanitize(parsed)
    if not isinstance(sanitized, dict):
        raise SystemExit("Expected an aggregate JSON object")

    validity = []
    for field in nonfinite_fields:
        pointer = field["json_pointer"]
        status = "not_measured"
        reason = "The v1 summary did not contain a finite measurement for this field."
        if "/external_evaluation/comparisons/" in pointer and "/station_count_" in pointer:
            comparison_index = int(pointer.split("/external_evaluation/comparisons/")[1].split("/")[0])
            comparisons = sanitized.get("external_evaluation", {}).get("comparisons", [])
            source_id = comparisons[comparison_index].get("source_id") if comparison_index < len(comparisons) else None
            if source_id == "CHIRPS_v3_final_monthly":
                status, reason = "not_applicable", "CHIRPS is gridded; station counts do not apply."
            elif source_id == "INAMHI_yearbooks":
                status, reason = "missing_in_v1_summary", "Yearly observation counts exist in the source composite; this legacy aggregate omitted them and no corrected per-month count is implied here."
        elif "/ingestion_cache/workloads/" in pointer:
            status, reason = "not_measured", "The v1 benchmark did not run analytical-query timing for this multi-variable workload."
        elif "/predictive_validation/baselines/" in pointer:
            status, reason = "not_applicable", "A deterministic baseline has no fitted estimator artifact, fit time, model file, or model hash."
        validity.append({**field, "status": status, "reason": reason})

    # Recover the omitted INAMHI station-count ranges from the validated local
    # composite and Gold months, without using model predictions or test scores.
    recovered = inamhi_station_count_ranges()
    for comparison in sanitized.get("external_evaluation", {}).get("comparisons", []):
        if comparison.get("source_id") != "INAMHI_yearbooks":
            continue
        variable = "merra2_t2m_c" if comparison.get("variable_id") == "temperature_degC" else "imerg_precip_monthly_total"
        item = recovered[variable]
        for stat in ("min", "median", "max"):
            comparison[f"station_count_{stat}"] = item[f"station_count_{stat}"]
        comparison["station_count_basis"] = "Eligible monthly INAMHI composite counts on exact matched NASA months; at least three stations."
        comparison["station_count_n_months"] = item["n_months"]
    for entry in validity:
        if entry.get("status") == "missing_in_v1_summary":
            entry["status"] = "recovered_from_validated_composite"
            entry["reason"] = "The value was recomputed on exact matched eligible months from the already validated local INAMHI composite; see station_count_basis on the comparison row."

    sanitized["serialization_metadata"] = {
        "json_standard": "RFC 8259 strict JSON",
        "nonfinite_policy": "Replace nonfinite numeric tokens with null and list each original field with a validity status and reason.",
        "field_validity": validity,
        "source_v1_summary_sha256": sha256(SOURCE),
        "historical_v1_summary_preserved": True,
    }
    # The source aggregate includes historical v1 model and final-test results.
    # Do not let a strict-JSON rewrite imply that those were recomputed in v1.1.
    sanitized["status"] = "STRICT_JSON_LEGACY_V1_DERIVATIVE"
    sanitized["report_scope"] = (
        "Strict-JSON serialization and recovery of omitted INAMHI station-count ranges "
        "from validated local inputs. No v1.1 model, trend, robustness, or final-test "
        "evaluation is performed by this script."
    )
    sanitized["evidence_provenance"] = {
        "source_summary_path": str(SOURCE.relative_to(ROOT)),
        "source_summary_sha256": sha256(SOURCE),
        "source_evidence_generation": "KUNTUR_PHASE2_V1_HISTORICAL",
        "preserved_legacy_sections": [
            section for section in (
                "augmentation", "predictive_validation", "final_test",
                "analytical", "ingestion_cache", "external_evaluation",
            ) if section in sanitized
        ],
        "final_test_policy": "Historical v1 final-test result is preserved as a once-used archival result; this derivative does not rescore it.",
        "v1_1_development_artifacts": [
            "results/phase2/v1_1/target_scaled_candidates/rolling_model_summary_development_2003_2018.json",
            "results/phase2/v1_1/robustness/robustness_manifest_v1_1.json",
            "results/phase2/v1_1/analytical/analytical_lag_audit_manifest.json",
            "results/phase2/v1_1/external/external_corroboration_manifest_v1_1.json",
            "results/phase2/v1_1/ingestion/ingestion_benchmark_manifest_v1_1.json",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(OUT)
    # Independent strict parse rejects NaN/Infinity constants.
    json.loads(OUT.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    manifest = {
        "status": "STRICT_JSON_VALIDATED",
        "source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(SOURCE),
        "output_path": str(OUT.relative_to(ROOT)),
        "output_sha256": sha256(OUT),
        "output_bytes": OUT.stat().st_size,
        "replaced_nonfinite_count": len(nonfinite_fields),
        "validity_entries": len(validity),
        "strict_parser_validation": "PASSED",
    }
    manifest_tmp = MANIFEST.with_suffix(".json.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    manifest_tmp.replace(MANIFEST)
    print(json.dumps(manifest, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
