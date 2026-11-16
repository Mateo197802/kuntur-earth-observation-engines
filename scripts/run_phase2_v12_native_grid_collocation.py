#!/usr/bin/env python3
"""Compare eligible Ecuador station-months to nearest native MERRA-2 cells.

This is a descriptive external-agreement diagnostic for the existing Ecuador
pilot bbox. It does not regrid, impute, retrain, or read the sealed 2025 test.
Station-level outputs should remain in the private KUNTUR data workspace.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "merra2_monthly_slv"
REGION_ID = "ecuador_pilot_bbox"
NATIVE_ROOT = ROOT / "data/silver/processed_native_grid/v1" / SOURCE_ID
PREPROCESSING_MANIFEST = ROOT / "data/manifests/preprocessing_manifest_v1.jsonl"
ACQUISITION_MANIFEST = ROOT / "data/manifests/acquisition_manifest.jsonl"
REGIONAL_MANIFEST = ROOT / "data/manifests/regional_extraction_manifest.jsonl"
EARTH_RADIUS_KM = 6371.0088


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required KUNTUR manifest missing: {path.relative_to(ROOT)}")
    result = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path.name} at line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object in {path.name} at line {line_number}")
            result.append(row)
    return result


def nearest_grid_cell(
    station_latitude: float,
    station_longitude: float,
    grid_latitudes: np.ndarray,
    grid_longitudes: np.ndarray,
) -> tuple[int, int, float]:
    """Return nearest native cell by great-circle distance and its km distance."""
    lat = np.asarray(grid_latitudes, dtype=float)
    lon = np.asarray(grid_longitudes, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("Native latitude/longitude coordinates must be finite one-dimensional axes")
    if not (-90 <= station_latitude <= 90 and -180 <= station_longitude <= 180):
        raise ValueError("Station coordinates are outside valid latitude/longitude ranges")
    lat_grid, lon_grid = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    station_lat = math.radians(float(station_latitude))
    station_lon = math.radians(float(station_longitude))
    dlat = lat_grid - station_lat
    dlon = (lon_grid - station_lon + math.pi) % (2 * math.pi) - math.pi
    haversine = np.sin(dlat / 2) ** 2 + math.cos(station_lat) * np.cos(lat_grid) * np.sin(dlon / 2) ** 2
    flat_index = int(np.argmin(haversine))
    row, column = np.unravel_index(flat_index, haversine.shape)
    distance_km = 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(float(np.clip(haversine[row, column], 0.0, 1.0))))
    return int(row), int(column), float(distance_km)


def _month_range(start: str, end: str) -> list[str]:
    first = pd.Period(start, freq="M")
    last = pd.Period(end, freq="M")
    if last < first:
        raise ValueError("end-month must not precede start-month")
    return [str(period) for period in pd.period_range(first, last, freq="M")]


def _safe_manifest_file(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    root = ROOT.resolve()
    if path != root and root not in path.parents:
        raise ValueError("Preprocessing manifest path escapes the KUNTUR workspace")
    return path


def _one_record_per_month(
    rows: list[dict[str, Any]],
    months: list[str],
    actual_sha_by_path: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], int]:
    chosen: dict[str, dict[str, Any]] = {}
    superseded = 0
    for month in months:
        candidates = [
            row for row in rows
            if row.get("source_id") == SOURCE_ID
            and row.get("region_id") == REGION_ID
            and row.get("month") == month
            and row.get("status") == "VALIDATED"
        ]
        current = [
            row for row in candidates
            if actual_sha_by_path.get(str(row.get("processed_path"))) == row.get("processed_sha256")
        ]
        unique = {
            (row.get("processed_path"), row.get("processed_sha256"), row.get("source_sha256"), row.get("regional_input_sha256"))
            for row in current
        }
        if len(unique) != 1:
            raise ValueError(f"Expected exactly one on-disk-hash-matched validated lineage for {month}; found {len(unique)}")
        chosen[month] = current[-1]
        superseded += len(candidates) - 1
    return chosen, superseded


def _check_input_schema(frame: pd.DataFrame, *, dataset: str) -> None:
    required = {"station_id", "month", "latitude", "longitude", "elevation_m"}
    required.add("value_c" if dataset == "NOAA_GHCN_M_QCU" else "tavg_c")
    required.add("eligible" if dataset == "NOAA_GHCN_M_QCU" else "tavg_eligible")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset} station table is missing required fields: {missing}")
    if frame.duplicated(["station_id", "month"]).any():
        raise ValueError(f"{dataset} contains duplicate station-month records; resolve provenance before collocation")


def _load_exported_station_table(
    csv_path: Path,
    source_table_path: Path,
    source_manifest_path: Path,
    *,
    source: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    sidecar_path = Path(str(csv_path) + ".provenance.json")
    if not csv_path.is_file() or not source_table_path.is_file() or not source_manifest_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"Station CSV, source Parquet, source manifest, or provenance sidecar missing for {source}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    csv_sha = sha256_file(csv_path)
    table_sha = sha256_file(source_table_path)
    manifest_sha = sha256_file(source_manifest_path)
    if sidecar.get("source_id") != source or sidecar.get("csv_sha256") != csv_sha:
        raise ValueError(f"Station CSV provenance mismatch for {source}")
    if sidecar.get("source_table_sha256") != table_sha or sidecar.get("source_manifest_sha256") != manifest_sha:
        raise ValueError(f"Station Parquet/source-manifest checksum mismatch for {source}")
    frame = pd.read_csv(csv_path, dtype={"station_id": "string", "month": "string"})
    if sidecar.get("row_count") != len(frame) or sidecar.get("columns") != list(frame.columns):
        raise ValueError(f"Station CSV schema/row-count mismatch for {source}")
    return frame, {
        "csv_sha256": csv_sha,
        "source_table_sha256": table_sha,
        "source_manifest_sha256": manifest_sha,
        "sidecar_sha256": sha256_file(sidecar_path),
    }


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    if not rows:
        return {"matched_station_months": 0, "matched_months": 0, "matched_stations": 0, "unique_grid_cells": 0, "mae_c": None, "rmse_c": None, "bias_c": None, "pearson_r": None, "mean_cell_distance_km": None, "p95_cell_distance_km": None}
    errors = np.asarray([float(row["merra2_minus_station_c"]) for row in rows], dtype=float)
    station_values = np.asarray([float(row["station_temperature_c"]) for row in rows], dtype=float)
    merra_values = np.asarray([float(row["merra2_t2m_c"]) for row in rows], dtype=float)
    corr = float(np.corrcoef(station_values, merra_values)[0, 1]) if len(rows) > 1 and np.std(station_values) > 0 and np.std(merra_values) > 0 else None
    return {
        "matched_station_months": int(len(rows)),
        "matched_months": int(len({row["month"] for row in rows if "month" in row})),
        "matched_stations": int(len({row["station_id"] for row in rows})),
        "unique_grid_cells": int(len({(row["grid_lat_index"], row["grid_lon_index"]) for row in rows})),
        "mae_c": float(np.mean(np.abs(errors))),
        "rmse_c": float(np.sqrt(np.mean(errors**2))),
        "bias_c": float(np.mean(errors)),
        "pearson_r": corr,
        "mean_cell_distance_km": float(np.mean([float(row["cell_center_distance_km"]) for row in rows])),
        "p95_cell_distance_km": float(np.quantile([float(row["cell_center_distance_km"]) for row in rows], 0.95)),
    }


def collocate(
    *,
    root: Path,
    months: list[str],
    station_files: dict[str, Path],
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory is not empty; choose a new versioned output directory")
    prep_rows = read_jsonl(root / PREPROCESSING_MANIFEST.relative_to(ROOT))
    acquisition_rows = read_jsonl(root / ACQUISITION_MANIFEST.relative_to(ROOT))
    regional_rows = read_jsonl(root / REGIONAL_MANIFEST.relative_to(ROOT))
    candidate_paths = {
        str(row.get("processed_path"))
        for row in prep_rows
        if row.get("source_id") == SOURCE_ID
        and row.get("region_id") == REGION_ID
        and row.get("month") in months
        and row.get("status") == "VALIDATED"
        and row.get("processed_path")
    }
    actual_sha_by_path = {}
    for relative_path in candidate_paths:
        candidate_file = _safe_manifest_file(relative_path)
        if candidate_file.is_file():
            actual_sha_by_path[relative_path] = sha256_file(candidate_file)
    processing_by_month, superseded_preprocessing_rows = _one_record_per_month(prep_rows, months, actual_sha_by_path)

    acquisition_hashes = {
        row.get("sha256") for row in acquisition_rows
        if row.get("status") == "VALIDATED" and row.get("sha256")
    }
    regional_hashes = {
        row.get("output_sha256") for row in regional_rows
        if row.get("status") == "VALIDATED" and row.get("output_sha256")
    }

    source_frames: dict[str, pd.DataFrame] = {}
    source_input_hashes: dict[str, str] = {}
    source_table_hashes: dict[str, str] = {}
    source_manifest_hashes: dict[str, str] = {}
    source_sidecar_hashes: dict[str, str] = {}
    for source, path in station_files.items():
        source_table_path = path.with_suffix(".parquet")
        source_manifest_path = path.with_name(
            "ghcnm_qcu_external_manifest_v1.json"
            if source == "NOAA_GHCN_M_QCU"
            else "inamhi_yearbooks_external_manifest_v1.json"
        )
        frame, verified_input_hashes = _load_exported_station_table(
            path, source_table_path, source_manifest_path, source=source
        )
        _check_input_schema(frame, dataset=source)
        frame["month"] = frame["month"].astype(str).str.slice(0, 7)
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if "VALIDATED" not in str(source_manifest.get("status", "")):
            raise ValueError(f"Source manifest is not validated for {source}")
        artifact_reference = f"data/external/processed/{source_table_path.name}"
        if source == "NOAA_GHCN_M_QCU":
            if artifact_reference not in source_manifest.get("artifacts", []):
                raise ValueError("NOAA source manifest does not reference the station-month table")
        else:
            expected_entry = (source_manifest.get("artifact_hashes") or {}).get(artifact_reference)
            expected_input_hash = expected_entry.get("sha256") if isinstance(expected_entry, dict) else expected_entry
            if expected_input_hash and expected_input_hash != verified_input_hashes["source_table_sha256"]:
                raise ValueError("INAMHI station-month table SHA-256 differs from its source manifest")
        source_manifest_hashes[source] = verified_input_hashes["source_manifest_sha256"]
        if source == "NOAA_GHCN_M_QCU":
            frame = frame[frame["eligible"].fillna(False).astype(bool)].copy()
            value_col = "value_c"
        else:
            frame = frame[frame["tavg_eligible"].fillna(False).astype(bool)].copy()
            value_col = "tavg_c"
        frame = frame[frame["month"].isin(months)].copy()
        frame["station_value_c"] = pd.to_numeric(frame[value_col], errors="coerce")
        frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
        frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
        frame["elevation_m"] = pd.to_numeric(frame["elevation_m"], errors="coerce")
        frame = frame[np.isfinite(frame["station_value_c"]) & np.isfinite(frame["latitude"]) & np.isfinite(frame["longitude"])].copy()
        source_frames[source] = frame
        source_input_hashes[source] = verified_input_hashes["csv_sha256"]
        source_table_hashes[source] = verified_input_hashes["source_table_sha256"]
        source_sidecar_hashes[source] = verified_input_hashes["sidecar_sha256"]

    matched: dict[str, list[dict[str, Any]]] = {source: [] for source in source_frames}
    grid_coordinates: tuple[np.ndarray, np.ndarray] | None = None
    native_receipts: list[dict[str, Any]] = []

    for month in months:
        record = processing_by_month[month]
        for label, digest, known_hashes in (
            ("source granule", record.get("source_sha256"), acquisition_hashes),
            ("regional input", record.get("regional_input_sha256"), regional_hashes),
        ):
            if not digest or digest not in known_hashes:
                raise ValueError(f"{month}: {label} checksum is absent from its validated upstream manifest")
        processed = _safe_manifest_file(str(record["processed_path"]))
        if not processed.is_file():
            raise FileNotFoundError(f"Validated processed native grid is missing for {month}")
        actual_sha = sha256_file(processed)
        if actual_sha != record.get("processed_sha256"):
            raise ValueError(f"Native-grid SHA-256 mismatch for {month}")
        with np.load(processed, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
            if metadata.get("source_sha256") != record.get("source_sha256"):
                raise ValueError(f"Source lineage mismatch inside native-grid artifact for {month}")
            if metadata.get("regional_input_sha256") != record.get("regional_input_sha256"):
                raise ValueError(f"Regional lineage mismatch inside native-grid artifact for {month}")
            if metadata.get("month") != month or metadata.get("region_id") != REGION_ID or metadata.get("source_id") != SOURCE_ID:
                raise ValueError(f"Native-grid identity metadata mismatch for {month}")
            latitude = np.asarray(archive["latitude"], dtype=float)
            longitude = np.asarray(archive["longitude"], dtype=float)
            if grid_coordinates is None:
                grid_coordinates = (latitude, longitude)
            elif not (np.array_equal(latitude, grid_coordinates[0]) and np.array_equal(longitude, grid_coordinates[1])):
                raise ValueError(f"Native MERRA-2 grid coordinates changed at {month}")
            if tuple(archive["t2m_c"].shape) != (len(latitude), len(longitude)):
                raise ValueError(f"T2M shape does not match native axes for {month}")
            field = np.asarray(archive["t2m_c"], dtype=float)
            valid = np.asarray(archive["valid_t2m"], dtype=bool) & np.isfinite(field)
            native_receipts.append({
                "month": month,
                "native_grid_path": str(processed.relative_to(root.resolve()).as_posix()),
                "native_grid_bytes": processed.stat().st_size,
                "native_grid_sha256": actual_sha,
                "source_sha256_from_validated_acquisition": record["source_sha256"],
                "regional_input_sha256_from_validated_extraction": record["regional_input_sha256"],
            })
            for source, frame in source_frames.items():
                month_frame = frame[frame["month"] == month]
                for station in month_frame.itertuples(index=False):
                    row_index, column_index, distance_km = nearest_grid_cell(
                        float(station.latitude), float(station.longitude), latitude, longitude
                    )
                    if not valid[row_index, column_index]:
                        continue
                    temperature = float(field[row_index, column_index])
                    station_value = float(station.station_value_c)
                    matched[source].append({
                        "source_id": source,
                        "station_id": str(station.station_id),
                        "month": month,
                        "station_temperature_c": station_value,
                        "merra2_t2m_c": temperature,
                        "merra2_minus_station_c": temperature - station_value,
                        "station_latitude": float(station.latitude),
                        "station_longitude": float(station.longitude),
                        "station_elevation_m": float(station.elevation_m) if np.isfinite(station.elevation_m) else None,
                        "station_coordinate_source_year": getattr(station, "coordinate_source_year", None),
                        "station_coordinate_source": getattr(station, "station_code_source", None),
                        "source_year": getattr(station, "source_year", None),
                        "grid_lat_index": row_index,
                        "grid_lon_index": column_index,
                        "grid_latitude": float(latitude[row_index]),
                        "grid_longitude": float(longitude[column_index]),
                        "cell_center_distance_km": distance_km,
                        "native_grid_sha256": actual_sha,
                    })

    if grid_coordinates is None:
        raise ValueError("No MERRA-2 native-grid files were read")

    detailed_columns = [
        "source_id", "station_id", "month", "station_temperature_c", "merra2_t2m_c",
        "merra2_minus_station_c", "station_latitude", "station_longitude", "station_elevation_m",
        "station_coordinate_source_year", "station_coordinate_source", "source_year",
        "grid_lat_index", "grid_lon_index", "grid_latitude", "grid_longitude",
        "cell_center_distance_km", "native_grid_sha256",
    ]
    monthly_columns = [
        "source_id", "month", "matched_station_months", "matched_stations", "unique_grid_cells",
        "matched_months", "mae_c", "rmse_c", "bias_c", "pearson_r", "mean_cell_distance_km", "p95_cell_distance_km",
    ]
    detail_rows = [row for source in matched for row in matched[source]]
    monthly_rows = []
    for source, rows in matched.items():
        by_month: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_month[row["month"]].append(row)
        for month in months:
            monthly_rows.append({"source_id": source, "month": month, **_metrics(by_month.get(month, []))})

    source_summaries = {}
    for source, frame in source_frames.items():
        rows = matched[source]
        source_summaries[source] = {
            "input_table": station_files[source].relative_to(root).as_posix(),
            "input_sha256": source_input_hashes[source],
            "source_parquet_table": station_files[source].with_suffix(".parquet").relative_to(root).as_posix(),
            "source_parquet_sha256": source_table_hashes[source],
            "source_manifest_sha256": source_manifest_hashes[source],
            "csv_provenance_sidecar_sha256": source_sidecar_hashes[source],
            "eligible_station_months_in_window": int(len(frame)),
            "eligible_stations_in_window": int(frame["station_id"].nunique()),
            "eligible_months_in_window": int(frame["month"].nunique()),
            "station_ids_with_coordinate_variants": int(
                frame.groupby("station_id")[["latitude", "longitude"]].nunique().gt(1).any(axis=1).sum()
            ),
            "match_fraction_of_eligible_station_months": (
                float(len(rows) / len(frame)) if len(frame) else None
            ),
            **_metrics(rows),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "station_monthly_collocation.csv"
    monthly_path = output_dir / "monthly_summary.csv"
    summary_path = output_dir / "summary.json"
    atomic_csv(detail_path, detail_rows, detailed_columns)
    atomic_csv(monthly_path, monthly_rows, monthly_columns)
    manifest = {
        "manifest_version": "kuntur-phase2-v12-native-grid-collocation-0.1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALIDATED" if all(source_summaries[s]["matched_station_months"] > 0 for s in source_summaries) else "PARTIAL",
        "scope": "Descriptive temperature agreement between eligible Ecuador station-months and their nearest native MERRA-2 grid-cell centers; not a forecast evaluation or independent ground truth.",
        "region_id": REGION_ID,
        "dataset_id": "M2TMNXSLV.5.12.4",
        "period_start": months[0],
        "period_end": months[-1],
        "requested_month_count": len(months),
        "native_grid_months_hash_verified": len(native_receipts),
        "superseded_preprocessing_rows_ignored_by_current_artifact_hash": superseded_preprocessing_rows,
        "grid_shape_lat_lon": [int(len(grid_coordinates[0])), int(len(grid_coordinates[1]))],
        "spatial_method": "Nearest existing native MERRA-2 cell center by haversine distance; no interpolation, regridding, or elevation adjustment.",
        "temperature_units": "degrees_Celsius",
        "eligibility_policies": {
            "NOAA_GHCN_M_QCU": "Existing table rows with eligible=true; provider QC, missing sentinel and physical range checks are preserved from the validated source table.",
            "INAMHI_yearbooks": "Existing table rows with tavg_eligible=true; source provenance and physical range checks are preserved from the validated yearbook extraction.",
            "MERRA2": "Existing preprocessing manifest status VALIDATED; monthly native-grid file SHA-256 and embedded source/regional checksums verified before sampling.",
        },
        "coordinate_policy": "Uses each eligible station-month row's stored coordinates; INAMHI coordinate variants are reported and retained rather than silently collapsed.",
        "inputs": {
            "preprocessing_manifest_sha256": sha256_file(root / PREPROCESSING_MANIFEST.relative_to(ROOT)),
            "acquisition_manifest_sha256": sha256_file(root / ACQUISITION_MANIFEST.relative_to(ROOT)),
            "regional_extraction_manifest_sha256": sha256_file(root / REGIONAL_MANIFEST.relative_to(ROOT)),
            "station_tables": source_summaries,
            "native_grid_receipts": native_receipts,
        },
        "outputs": {
            "station_monthly_collocation": {
                "path": detail_path.relative_to(root).as_posix(), "bytes": detail_path.stat().st_size,
                "sha256": sha256_file(detail_path), "rows": len(detail_rows),
                "retention": "Private KUNTUR workspace only; station-level records are excluded from Git.",
            },
            "monthly_summary": {
                "path": monthly_path.relative_to(root).as_posix(), "bytes": monthly_path.stat().st_size,
                "sha256": sha256_file(monthly_path), "rows": len(monthly_rows),
            },
        },
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
        "limitations": [
            "The pilot bbox includes land/ocean cells and is not an administrative or physiographic mask.",
            "Nearest-cell distances and elevation differences limit point-to-grid comparability.",
            "MERRA-2 is a reanalysis; station observations may not be independent of assimilated inputs.",
            "NOAA GHCN-M QC records can overlap other station networks; independence is not claimed.",
            "INAMHI yearbook tables provide eligible station months only for their existing 2002–2012 coverage.",
            "This comparison does not evaluate the regional-mean forecasting model and does not alter training or test partitions.",
        ],
    }
    summary_bytes = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    atomic_bytes(summary_path, summary_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-dir", type=Path, required=True, help="Private directory containing the existing NOAA and INAMHI station Parquet tables")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty output directory inside the KUNTUR workspace")
    parser.add_argument("--start-month", default="2002-01")
    parser.add_argument("--end-month", default="2024-12")
    parser.add_argument("--months", help="Optional comma-separated bounded sample, e.g. 2002-01,2002-02")
    args = parser.parse_args()
    months = [part.strip() for part in args.months.split(",") if part.strip()] if args.months else _month_range(args.start_month, args.end_month)
    if not months or any(pd.Period(month, freq="M").strftime("%Y-%m") != month for month in months):
        parser.error("months must be valid YYYY-MM values")
    inputs_dir = args.inputs_dir.resolve()
    output_dir = args.output_dir.resolve()
    try:
        inputs_dir.relative_to(ROOT.resolve())
        output_dir.relative_to(ROOT.resolve())
    except ValueError as exc:
        parser.error("Inputs and outputs must remain within the KUNTUR workspace")
    station_files = {
        "NOAA_GHCN_M_QCU": inputs_dir / "ghcnm_ecuador_bbox_station_monthly_v1.csv",
        "INAMHI_yearbook_temperature": inputs_dir / "inamhi_pilot_bbox_station_monthly_2002_2012_v1.csv",
    }
    result = collocate(root=ROOT, months=months, station_files=station_files, output_dir=output_dir)
    print(json.dumps({
        "status": result["status"],
        "period": [result["period_start"], result["period_end"]],
        "native_grid_months_hash_verified": result["native_grid_months_hash_verified"],
        "grid_shape_lat_lon": result["grid_shape_lat_lon"],
        "sources": result["inputs"]["station_tables"],
        "summary_path": str((output_dir / "summary.json").relative_to(ROOT)),
        "summary_sha256": sha256_file(output_dir / "summary.json"),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
