#!/usr/bin/env python3
"""Independently verify native-grid collocation outputs and lineage checksums."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_MANIFEST = ROOT / "data/manifests/preprocessing_manifest_v1.jsonl"
ACQUISITION_MANIFEST = ROOT / "data/manifests/acquisition_manifest.jsonl"
REGIONAL_MANIFEST = ROOT / "data/manifests/regional_extraction_manifest.jsonl"
SOURCE_ID = "merra2_monthly_slv"
REGION_ID = "ecuador_pilot_bbox"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant {value}")

    result = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path.name}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {path.name}:{line_number}")
            rows.append(row)
    return rows


def safe_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    resolved_root = root.resolve()
    if path != resolved_root and resolved_root not in path.parents:
        raise ValueError("Manifest path escapes the KUNTUR workspace")
    return path


def month_range(start: str, end: str) -> list[str]:
    first_year, first_month = (int(part) for part in start.split("-"))
    last_year, last_month = (int(part) for part in end.split("-"))
    result = []
    year, month = first_year, first_month
    while (year, month) <= (last_year, last_month):
        result.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return result


def _linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot compute a quantile from no values")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def metric_summary(rows: list[dict[str, str]]) -> dict[str, Any]:
    if not rows:
        return {
            "matched_station_months": 0, "matched_months": 0, "matched_stations": 0,
            "unique_grid_cells": 0, "mae_c": None, "rmse_c": None, "bias_c": None,
            "pearson_r": None, "mean_cell_distance_km": None, "p95_cell_distance_km": None,
        }
    errors = [float(row["merra2_minus_station_c"]) for row in rows]
    station = [float(row["station_temperature_c"]) for row in rows]
    merra = [float(row["merra2_t2m_c"]) for row in rows]
    distances = [float(row["cell_center_distance_km"]) for row in rows]
    mean_station = statistics.fmean(station)
    mean_merra = statistics.fmean(merra)
    covariance = sum((x - mean_station) * (y - mean_merra) for x, y in zip(station, merra))
    variance_station = sum((x - mean_station) ** 2 for x in station)
    variance_merra = sum((y - mean_merra) ** 2 for y in merra)
    correlation = covariance / math.sqrt(variance_station * variance_merra) if variance_station > 0 and variance_merra > 0 else None
    return {
        "matched_station_months": len(rows),
        "matched_months": len({row["month"] for row in rows}),
        "matched_stations": len({row["station_id"] for row in rows}),
        "unique_grid_cells": len({(row["grid_lat_index"], row["grid_lon_index"]) for row in rows}),
        "mae_c": statistics.fmean(abs(value) for value in errors),
        "rmse_c": math.sqrt(statistics.fmean(value * value for value in errors)),
        "bias_c": statistics.fmean(errors),
        "pearson_r": correlation,
        "mean_cell_distance_km": statistics.fmean(distances),
        "p95_cell_distance_km": _linear_quantile(distances, 0.95),
    }


def verify(summary_dir: Path) -> dict[str, Any]:
    summary_path = summary_dir / "summary.json"
    summary = strict_json(summary_path)
    if summary.get("status") != "VALIDATED":
        raise ValueError("Collocation summary is not marked VALIDATED")
    if summary.get("region_id") != REGION_ID or summary.get("dataset_id") != "M2TMNXSLV.5.12.4":
        raise ValueError("Unexpected region or MERRA-2 dataset identifier")

    inputs = summary.get("inputs") or {}
    expected_manifest_hashes = {
        "preprocessing_manifest_sha256": PREPROCESSING_MANIFEST,
        "acquisition_manifest_sha256": ACQUISITION_MANIFEST,
        "regional_extraction_manifest_sha256": REGIONAL_MANIFEST,
    }
    for key, path in expected_manifest_hashes.items():
        if not path.is_file() or sha256_file(path) != inputs.get(key):
            raise ValueError(f"Input manifest checksum mismatch: {key}")

    for source, record in (inputs.get("station_tables") or {}).items():
        csv_path = safe_file(ROOT, str(record["input_table"]))
        parquet_path = safe_file(ROOT, str(record["source_parquet_table"]))
        manifest_name = "ghcnm_qcu_external_manifest_v1.json" if source == "NOAA_GHCN_M_QCU" else "inamhi_yearbooks_external_manifest_v1.json"
        manifest_path = csv_path.with_name(manifest_name)
        sidecar_path = Path(str(csv_path) + ".provenance.json")
        expected = {
            csv_path: record["input_sha256"],
            parquet_path: record["source_parquet_sha256"],
            manifest_path: record["source_manifest_sha256"],
            sidecar_path: record["csv_provenance_sidecar_sha256"],
        }
        for path, digest in expected.items():
            if not path.is_file() or sha256_file(path) != digest:
                raise ValueError(f"External station input checksum mismatch for {source}: {path.name}")

    outputs = summary.get("outputs") or {}
    output_paths = {}
    for name in ("station_monthly_collocation", "monthly_summary"):
        record = outputs.get(name) or {}
        path = safe_file(ROOT, str(record.get("path", "")))
        if not path.is_file() or path.stat().st_size != record.get("bytes") or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Output size/hash mismatch: {name}")
        output_paths[name] = path

    expected_months = month_range(summary["period_start"], summary["period_end"])
    if summary.get("requested_month_count") != len(expected_months):
        raise ValueError("Requested month count does not match period boundaries")

    processing_rows = read_jsonl(PREPROCESSING_MANIFEST)
    acquisition_rows = read_jsonl(ACQUISITION_MANIFEST)
    regional_rows = read_jsonl(REGIONAL_MANIFEST)
    acquisition_hashes = {row.get("sha256") for row in acquisition_rows if row.get("status") == "VALIDATED"}
    regional_hashes = {row.get("output_sha256") for row in regional_rows if row.get("status") == "VALIDATED"}
    processing_by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in processing_rows:
        if row.get("source_id") == SOURCE_ID and row.get("region_id") == REGION_ID and row.get("status") == "VALIDATED":
            processing_by_month[str(row.get("month"))].append(row)

    native_receipts = inputs.get("native_grid_receipts") or []
    receipt_by_month = {row.get("month"): row for row in native_receipts}
    if len(native_receipts) != len(expected_months) or set(receipt_by_month) != set(expected_months):
        raise ValueError("Native-grid receipt coverage does not match the full requested monthly period")
    for month in expected_months:
        receipt = receipt_by_month[month]
        native_path = safe_file(ROOT, str(receipt["native_grid_path"]))
        if not native_path.is_file() or native_path.stat().st_size != receipt.get("native_grid_bytes") or sha256_file(native_path) != receipt.get("native_grid_sha256"):
            raise ValueError(f"Native-grid artifact checksum mismatch: {month}")
        source_sha = receipt.get("source_sha256_from_validated_acquisition")
        regional_sha = receipt.get("regional_input_sha256_from_validated_extraction")
        if source_sha not in acquisition_hashes or regional_sha not in regional_hashes:
            raise ValueError(f"Upstream acquisition/extraction checksum is not validated for {month}")
        current_chain = [
            row for row in processing_by_month.get(month, [])
            if row.get("processed_path") == receipt.get("native_grid_path")
            and row.get("processed_sha256") == receipt.get("native_grid_sha256")
            and row.get("source_sha256") == source_sha
            and row.get("regional_input_sha256") == regional_sha
        ]
        if not current_chain:
            raise ValueError(f"No preprocessing record connects the native-grid output to validated inputs for {month}")
        current_record = current_chain[-1]
        regional_path_value = current_record.get("regional_input_path")
        if not regional_path_value:
            raise ValueError(f"Regional input path is missing from preprocessing lineage for {month}")
        regional_path = Path(str(regional_path_value))
        if not regional_path.is_absolute():
            regional_path = ROOT / regional_path
        regional_path = regional_path.resolve()
        if ROOT.resolve() not in regional_path.parents or not regional_path.is_file() or sha256_file(regional_path) != regional_sha:
            raise ValueError(f"Regional input on-disk SHA-256 mismatch for {month}")
        if not any(
            row.get("status") == "VALIDATED"
            and row.get("output_sha256") == regional_sha
            and row.get("output_path") == regional_path_value
            for row in regional_rows
        ):
            raise ValueError(f"Regional extraction manifest does not confirm the verified input for {month}")

    detailed_by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    unique_station_months: set[tuple[str, str, str]] = set()
    with output_paths["station_monthly_collocation"].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            source = row["source_id"]
            if source not in inputs.get("station_tables", {}):
                raise ValueError(f"Unexpected station source in detailed output: {source}")
            if row["month"] not in set(expected_months):
                raise ValueError(f"Collocation month outside requested period: {row['month']}")
            key = (source, row["station_id"], row["month"])
            if key in unique_station_months:
                raise ValueError(f"Duplicate source/station/month row: {key}")
            unique_station_months.add(key)
            detailed_by_source[source].append(row)
    verified_source_metrics = {}
    for source, recorded in (inputs.get("station_tables") or {}).items():
        observed = metric_summary(detailed_by_source[source])
        for metric, value in observed.items():
            recorded_value = recorded.get(metric)
            if value is None:
                if recorded_value is not None:
                    raise ValueError(f"Metric mismatch for {source}/{metric}: expected null")
            elif isinstance(value, float):
                if not math.isclose(value, float(recorded_value), rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"Metric mismatch for {source}/{metric}: recomputed={value} manifest={recorded_value}")
            elif value != recorded_value:
                raise ValueError(f"Count mismatch for {source}/{metric}: recomputed={value} manifest={recorded_value}")
        eligible = int(recorded["eligible_station_months_in_window"])
        if observed["matched_station_months"] != eligible:
            raise ValueError(f"Eligible/matched station-month count mismatch for {source}")
        verified_source_metrics[source] = observed

    expected_monthly_rows = len(expected_months) * len(inputs.get("station_tables", {}))
    monthly_rows = []
    monthly_seen = set()
    with output_paths["monthly_summary"].open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (row["source_id"], row["month"])
            if key in monthly_seen:
                raise ValueError(f"Duplicate monthly summary row: {key}")
            monthly_seen.add(key)
            monthly_rows.append(row)
    if len(monthly_rows) != expected_monthly_rows:
        raise ValueError(f"Expected {expected_monthly_rows} monthly rows, got {len(monthly_rows)}")
    for source in inputs.get("station_tables", {}):
        covered = {row["month"] for row in monthly_rows if row["source_id"] == source}
        if covered != set(expected_months):
            raise ValueError(f"Monthly summary coverage incomplete for {source}")

    result = {
        "validation_version": "kuntur-phase2-v12-native-grid-collocation-validation-0.1.0",
        "status": "PASSED",
        "summary_sha256": sha256_file(summary_path),
        "period": [summary["period_start"], summary["period_end"]],
        "native_grid_files_hash_verified": len(native_receipts),
        "upstream_acquisition_and_regional_hash_links_verified": len(native_receipts),
        "station_detail_rows_verified": len(unique_station_months),
        "monthly_summary_rows_verified": len(monthly_rows),
        "sources": verified_source_metrics,
        "output_hashes": {
            name: {"path": output_paths[name].relative_to(ROOT).as_posix(), "bytes": output_paths[name].stat().st_size, "sha256": sha256_file(output_paths[name])}
            for name in output_paths
        },
    }
    validation_path = summary_dir / "validation.json"
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".validation.", suffix=".tmp", dir=summary_dir)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, validation_path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    result["validation_path"] = validation_path.relative_to(ROOT).as_posix()
    result["validation_sha256"] = sha256_file(validation_path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, required=True)
    args = parser.parse_args()
    summary_dir = args.summary_dir.resolve()
    try:
        summary_dir.relative_to(ROOT.resolve())
    except ValueError as exc:
        parser.error("summary-dir must remain inside the KUNTUR workspace")
    result = verify(summary_dir)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
