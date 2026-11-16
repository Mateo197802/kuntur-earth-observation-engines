#!/usr/bin/env python3
"""Audit MOD13C2 regional QA selection and spatial support without rewriting inputs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
try:
    from modis_spatial_selection import (
        area_weights_from_centers,
        build_quality_masks,
        geospatial_quality_fractions,
        temporal_support_summary,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "src"))
    from kuntur_data.phase2.modis_spatial_selection import (
        area_weights_from_centers,
        build_quality_masks,
        geospatial_quality_fractions,
        temporal_support_summary,
    )

REGIONAL_MANIFEST_REL = Path("data/manifests/regional_extraction_manifest.jsonl")
ACQUISITION_MANIFEST_REL = Path("data/manifests/acquisition_manifest.jsonl")
OUTPUT_REL = Path("results/statistical_correction/modis_spatial_selection_v1_1")
REGION_ID = "ecuador_pilot_bbox"
SOURCE_ID = "modis_mod13c2_monthly_ndvi"
BBOX_WSEN = (-81.5, -5.1, -75.0, 1.7)
EXPECTED_MONTHS = tuple(
    f"{year:04d}-{month:02d}"
    for year in range(2002, 2025)
    for month in range(1, 13)
)
ARRAY_NAMES = {
    "ndvi": "CMG_0.05_Deg_Monthly_NDVI",
    "evi": "CMG_0.05_Deg_Monthly_EVI",
    "vi_quality": "CMG_0.05_Deg_Monthly_VI_Quality",
    "pixel_reliability": "CMG_0.05_Deg_Monthly_pixel_reliability",
}
QA_TIERS = ("product_valid", "strict", "relaxed")
SCALE_FACTOR = 0.0001


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fieldnames = list(rows[0]) if rows else []
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def _safe_root_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Manifest path escapes KUNTUR root: {relative}")
    return path


def _month_from_record(record: dict[str, Any]) -> str:
    value = str(record.get("period_start", ""))
    try:
        parsed = dt.date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"Invalid MODIS period_start: {value!r}") from exc
    return parsed.strftime("%Y-%m")


def _validated_science_check(source_record: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    if source_record.get("status") != "VALIDATED" or source_record.get("sha256") != source_sha256:
        raise ValueError("Source granule status or SHA-256 disagrees with regional provenance")
    validation = source_record.get("validation") or {}
    if validation.get("scientific_content_status") != "VALIDATED":
        raise ValueError("Source acquisition manifest lacks validated HDF4 scientific content")
    required_variables = {"NDVI", "EVI", "VI Quality", "Pixel Reliability"}
    for check in validation.get("checks", []):
        if (
            check.get("status") == "VALIDATED"
            and check.get("sha256") == source_sha256
            and required_variables.issubset(set(check.get("variables_present", [])))
            and check.get("quality_fields_read") is True
        ):
            return check
    raise ValueError("No matching scientific HDF4 validation check was found")


def _resolve_inputs(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    regional_path = root / REGIONAL_MANIFEST_REL
    acquisition_path = root / ACQUISITION_MANIFEST_REL
    regional = read_jsonl(regional_path)
    acquisitions = read_jsonl(acquisition_path)

    outputs = [
        record
        for record in regional
        if record.get("region_id") == REGION_ID
        and record.get("source_id") == SOURCE_ID
        and record.get("output_path")
    ]
    acquisitions_by_sha = {
        str(record.get("sha256")): record
        for record in acquisitions
        if record.get("source_id") == SOURCE_ID and record.get("sha256")
    }
    outputs.sort(key=_month_from_record)

    months = [_month_from_record(record) for record in outputs]
    if months != list(EXPECTED_MONTHS):
        raise ValueError(
            "MODIS regional manifest must contain one validated record for every month "
            f"from 2002-01 through 2024-12; found {len(outputs)} rows and {len(set(months))} months"
        )
    for record in outputs:
        if record.get("status") != "VALIDATED":
            raise ValueError(f"Regional output is not validated for {_month_from_record(record)}")
        if record.get("validation", {}).get("status") != "VALIDATED":
            raise ValueError(f"Regional validation block failed for {_month_from_record(record)}")
        if tuple(record.get("region_bbox_wsen", ())) != BBOX_WSEN:
            raise ValueError(f"Unexpected MODIS region bounds for {_month_from_record(record)}")
        source = acquisitions_by_sha.get(str(record.get("source_sha256")))
        if not source:
            raise ValueError(f"No acquisition record matches {_month_from_record(record)} source hash")
        if source.get("object_id") != record.get("source_object_id"):
            raise ValueError(f"Granule ID mismatch for {_month_from_record(record)}")
        if source.get("local_path") != record.get("source_path"):
            raise ValueError(f"Source path mismatch for {_month_from_record(record)}")
        _validated_science_check(source, str(record.get("source_sha256")))
    return outputs, acquisitions_by_sha, regional


def _optional_peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes. CEDIA runs Linux.
        return peak * 1024 if sys.platform.startswith("linux") else peak
    except (ImportError, OSError, ValueError):
        return None


def run(root: Path, output_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    regional_path = root / REGIONAL_MANIFEST_REL
    acquisition_path = root / ACQUISITION_MANIFEST_REL
    if not regional_path.is_file() or not acquisition_path.is_file():
        raise FileNotFoundError("Required acquisition or regional extraction manifest is missing")

    output_records, acquisitions_by_sha, _ = _resolve_inputs(root)
    monthly_rows: list[dict[str, Any]] = []
    source_lineage: list[dict[str, Any]] = []
    masks_by_variable_tier: dict[tuple[str, str], list[np.ndarray]] = {
        (variable, tier): [] for variable in ("ndvi", "evi") for tier in QA_TIERS
    }
    expected_latitude: np.ndarray | None = None
    expected_longitude: np.ndarray | None = None
    cell_weights: np.ndarray | None = None
    bbox_area: float | None = None

    progress_path = output_dir / "modis_spatial_selection_progress_v1_1.json"
    for index, record in enumerate(output_records, 1):
        month = _month_from_record(record)
        regional_file = _safe_root_path(root, str(record["output_path"]))
        if not regional_file.is_file():
            raise FileNotFoundError(f"Regional MODIS file missing for {month}: {record['output_path']}")
        actual_bytes = regional_file.stat().st_size
        actual_sha256 = sha256_file(regional_file)
        if actual_bytes != int(record.get("output_bytes", -1)):
            raise ValueError(f"Regional MODIS file size mismatch for {month}")
        if actual_sha256 != record.get("output_sha256"):
            raise ValueError(f"Regional MODIS SHA-256 mismatch for {month}")

        with np.load(regional_file, allow_pickle=False) as data:
            required_arrays = {"latitude", "longitude", "time", "metadata_json", *ARRAY_NAMES.values()}
            missing_arrays = required_arrays - set(data.files)
            if missing_arrays:
                raise ValueError(f"{month} regional NPZ is missing arrays: {sorted(missing_arrays)}")
            latitude = np.asarray(data["latitude"], dtype=np.float64)
            longitude = np.asarray(data["longitude"], dtype=np.float64)
            time_values = np.asarray(data["time"]).reshape(-1)
            metadata = json.loads(str(data["metadata_json"].item()))
            ndvi = np.asarray(data[ARRAY_NAMES["ndvi"]])
            evi = np.asarray(data[ARRAY_NAMES["evi"]])
            vi_quality = np.asarray(data[ARRAY_NAMES["vi_quality"]])
            reliability = np.asarray(data[ARRAY_NAMES["pixel_reliability"]])

            if metadata.get("region_id") != REGION_ID:
                raise ValueError(f"Regional metadata region mismatch for {month}")
            if tuple(metadata.get("region_bbox_wsen", ())) != BBOX_WSEN:
                raise ValueError(f"Regional metadata bbox mismatch for {month}")
            if metadata.get("source_sha256") != record.get("source_sha256"):
                raise ValueError(f"Embedded source hash mismatch for {month}")
            if not str(metadata.get("source_period_start", "")).startswith(month):
                raise ValueError(f"Embedded temporal coverage mismatch for {month}")
            if len(time_values) != 1 or not str(time_values[0]).startswith(month):
                raise ValueError(f"NPZ time coordinate mismatch for {month}")
            if ndvi.shape != (latitude.size, longitude.size):
                raise ValueError(f"Science array dimensions do not match coordinates for {month}")
            if not (ndvi.shape == evi.shape == vi_quality.shape == reliability.shape):
                raise ValueError(f"MODIS science and QA array dimensions differ for {month}")
            if (
                not np.isfinite(latitude).all()
                or not np.isfinite(longitude).all()
                or not np.all(np.diff(latitude) > 0)
                or not np.all(np.diff(longitude) > 0)
            ):
                raise ValueError(f"MODIS regional coordinates are not finite and ascending for {month}")
            west, south, east, north = BBOX_WSEN
            if latitude.min() < south - 1e-9 or latitude.max() > north + 1e-9:
                raise ValueError(f"Latitude centers exceed the declared bbox for {month}")
            if longitude.min() < west - 1e-9 or longitude.max() > east + 1e-9:
                raise ValueError(f"Longitude centers exceed the declared bbox for {month}")

            if expected_latitude is None:
                expected_latitude = latitude.copy()
                expected_longitude = longitude.copy()
                cell_weights, bbox_area = area_weights_from_centers(latitude, longitude, BBOX_WSEN)
            elif not (
                np.array_equal(latitude, expected_latitude)
                and np.array_equal(longitude, expected_longitude)
            ):
                raise ValueError(f"MODIS native grid changed unexpectedly in {month}")

            masks = build_quality_masks(ndvi, evi, vi_quality, reliability)
            qa_uint = vi_quality.astype(np.uint32, copy=False)
            geospatial_quality = (qa_uint >> 14) & 0b11
            assert cell_weights is not None and bbox_area is not None

            for variable, raw, product_key in (
                ("ndvi", ndvi, "ndvi_product_valid"),
                ("evi", evi, "evi_product_valid"),
            ):
                for tier in QA_TIERS:
                    eligible = masks[product_key] if tier == "product_valid" else masks[f"{variable}_{tier}"]
                    eligible = np.asarray(eligible, dtype=bool)
                    selected_area = float(cell_weights[eligible].sum())
                    weighted_sum = float(np.sum(raw[eligible].astype(np.float64) * SCALE_FACTOR * cell_weights[eligible]))
                    value = weighted_sum / selected_area if selected_area > 0 else None
                    if value is not None and not math.isfinite(value):
                        raise ValueError(f"Nonfinite {variable} regional mean for {month}/{tier}")
                    quality_fractions = geospatial_quality_fractions(
                        eligible,
                        masks["qa_valid"],
                        geospatial_quality,
                        cell_weights,
                    )
                    monthly_rows.append(
                        {
                            "month": month,
                            "dataset_id": "MOD13C2.061",
                            "variable_id": variable.upper(),
                            "region_id": REGION_ID,
                            "qa_tier": tier,
                            "area_weighted_mean": value,
                            "units": "1",
                            "scale_factor_applied": SCALE_FACTOR,
                            "valid_pixel_count": int(np.count_nonzero(eligible)),
                            "valid_area_m2": selected_area,
                            "valid_area_fraction_bbox": selected_area / bbox_area,
                            "qa_valid_area_fraction_bbox": float(cell_weights[masks["qa_valid"]].sum() / bbox_area),
                            "strict_qa_area_fraction_bbox": float(cell_weights[masks["qa_strict"]].sum() / bbox_area),
                            "relaxed_qa_area_fraction_bbox": float(cell_weights[masks["qa_relaxed"]].sum() / bbox_area),
                            "reliability_valid_area_fraction_bbox": float(
                                cell_weights[masks["pixel_reliability_valid"]].sum() / bbox_area
                            ),
                            **quality_fractions,
                            "source_granule_id": record.get("source_object_id"),
                            "source_sha256": record.get("source_sha256"),
                            "regional_input_path": record.get("output_path"),
                            "regional_input_sha256": actual_sha256,
                        }
                    )
                    masks_by_variable_tier[(variable, tier)].append(eligible.reshape(-1).copy())

        source = acquisitions_by_sha[str(record["source_sha256"])]
        source_file = _safe_root_path(root, str(source["local_path"]))
        if not source_file.is_file():
            raise FileNotFoundError(f"Validated MODIS source granule missing for {month}: {source['local_path']}")
        source_bytes = source_file.stat().st_size
        expected_source_bytes = int(source.get("actual_download_size_bytes", source_bytes))
        if source_bytes != expected_source_bytes:
            raise ValueError(f"MODIS source granule byte-size mismatch for {month}")
        source_lineage.append(
            {
                "month": month,
                "source_granule_id": record.get("source_object_id"),
                "source_path": record.get("source_path"),
                "source_sha256": record.get("source_sha256"),
                "source_bytes": source_bytes,
                "source_hdf4_scientific_status": "VALIDATED",
                "regional_output_path": record.get("output_path"),
                "regional_output_bytes": actual_bytes,
                "regional_output_sha256": actual_sha256,
            }
        )
        if index % 25 == 0 or index == len(output_records):
            _atomic_json(
                progress_path,
                {
                    "status": "IN_PROGRESS",
                    "completed_months": index,
                    "total_months": len(output_records),
                    "last_completed_month": month,
                    "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )
            print(f"MODIS_SPATIAL_AUDIT_PROGRESS={index}/{len(output_records)} last={month}", flush=True)

    assert cell_weights is not None and bbox_area is not None
    support_rows: list[dict[str, Any]] = []
    for (variable, tier), month_masks in masks_by_variable_tier.items():
        support = temporal_support_summary(
            np.stack(month_masks, axis=0),
            cell_weights.reshape(-1),
            bbox_area=bbox_area,
            months=EXPECTED_MONTHS,
        )
        support_rows.append(
            {
                "dataset_id": "MOD13C2.061",
                "variable_id": variable.upper(),
                "region_id": REGION_ID,
                "qa_tier": tier,
                **support,
            }
        )

    monthly_name = "modis_spatial_selection_monthly_v1_1.csv"
    support_name = "modis_spatial_support_summary_v1_1.csv"
    lineage_name = "modis_spatial_selection_lineage_v1_1.json"
    manifest_name = "modis_spatial_selection_manifest_v1_1.json"
    _atomic_csv(output_dir / monthly_name, monthly_rows)
    _atomic_csv(output_dir / support_name, support_rows)
    _atomic_json(output_dir / lineage_name, {"months": source_lineage})

    elapsed_seconds = time.monotonic() - started
    outputs: dict[str, dict[str, Any]] = {}
    for name in (monthly_name, support_name, lineage_name):
        path = output_dir / name
        outputs[name] = {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    superseded_manifest_path = root / "results/statistical_correction/modis_spatial_selection_v1/modis_spatial_selection_manifest_v1.json"
    manifest = {
        "manifest_version": "1.0.0",
        "status": "VALIDATED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset": {"dataset_id": "MOD13C2.061", "source_id": SOURCE_ID, "region_id": REGION_ID},
        "coverage": {"start_month": EXPECTED_MONTHS[0], "end_month": EXPECTED_MONTHS[-1], "month_count": len(EXPECTED_MONTHS)},
        "spatial_support": {
            "bbox_wsen_degrees": list(BBOX_WSEN),
            "native_grid": "MODIS CMG geographic 0.05 degree; no interpolation or resampling",
            "area_method": "spherical cell areas clipped to the pilot bbox using adjacent coordinate centers",
            "earth_radius_m": 6_371_008.8,
            "grid_shape_lat_lon": [int(expected_latitude.size), int(expected_longitude.size)],
            "bbox_area_m2": bbox_area,
        },
        "preprocessing": {
            "raw_variables": ARRAY_NAMES,
            "scale_factor": SCALE_FACTOR,
            "raw_fill": -3000,
            "raw_valid_range": [-2000, 10000],
            "product_valid_tier": "valid-range and fill-value screen only; no QA filtering",
            "strict_tier": "frozen v1 contract: MODLAND=0, usefulness<=1, reliability=0, land-only, no adjacent or mixed cloud",
            "relaxed_tier": "sensitivity only: MODLAND<=1, usefulness<=4, reliability<=1, land-only, no adjacent or mixed cloud",
            "geospatial_quality": "bit-field class fractions 0..3 conditional on selected cells with non-fill VI Quality; QA-fill area is reported separately and is not assigned a class; not interpreted as snow/shadow flags",
        },
        "inputs": {
            "regional_manifest": {"path": REGIONAL_MANIFEST_REL.as_posix(), "sha256": sha256_file(root / REGIONAL_MANIFEST_REL)},
            "acquisition_manifest": {"path": ACQUISITION_MANIFEST_REL.as_posix(), "sha256": sha256_file(root / ACQUISITION_MANIFEST_REL)},
            "regional_files_verified": len(source_lineage),
            "source_granule_scientific_validations_referenced": len(source_lineage),
        },
        "validation": {
            "status": "PASSED",
            "checks": [
                "276 unique contiguous months from 2002-01 through 2024-12",
                "each regional NPZ byte size and SHA-256 match its regional extraction manifest",
                "each source granule links by object ID and SHA-256 to a scientific HDF4 validation record",
                "embedded region, bbox, source provenance, time coordinate and native-grid axes checked",
                "NDVI, EVI, VI Quality and Pixel Reliability arrays checked for matching dimensions",
                "native 0.05 degree coordinates remained unchanged across months",
                "QA masks, scaled area-weighted summaries and temporal support outputs computed",
            ],
        },
        "outputs": outputs,
        "supersedes": (
            {
                "manifest_path": superseded_manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": sha256_file(superseded_manifest_path),
                "reason": "The initial v1 QA summary included VI Quality fill words in geospatial class 3; v1.1 reports class fractions conditional on non-fill QA and retains QA coverage separately. The monthly means and support masks are otherwise unchanged.",
            }
            if superseded_manifest_path.is_file()
            else None
        ),
        "summary": {"monthly_series_rows": len(monthly_rows), "spatial_support_rows": len(support_rows)},
        "reproducibility": {
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "module_sha256": sha256_file(
                Path(__file__).with_name("modis_spatial_selection.py")
                if Path(__file__).with_name("modis_spatial_selection.py").is_file()
                else root / "src/kuntur_data/phase2/modis_spatial_selection.py"
            ),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "runtime_seconds": elapsed_seconds,
            "peak_rss_bytes": _optional_peak_rss_bytes(),
        },
        "limitations": [
            "The pilot bbox is not an administrative or Coast/Andes/Amazon mask.",
            "Relaxed QA results are sensitivity outputs and do not replace the frozen strict QA policy.",
            "Changing valid-pixel support can change the spatial population represented by a monthly mean.",
            "These descriptive diagnostics do not establish a vegetation trend or causal change.",
        ],
    }
    _atomic_json(output_dir / manifest_name, manifest)
    _atomic_json(
        progress_path,
        {
            "status": "COMPLETE",
            "completed_months": len(output_records),
            "total_months": len(output_records),
            "last_completed_month": EXPECTED_MONTHS[-1],
            "final_manifest": (output_dir / manifest_name).relative_to(root).as_posix(),
            "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="KUNTUR repository root")
    parser.add_argument("--output-dir", type=Path, help="Versioned output directory")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / OUTPUT_REL)).resolve()
    if not output_dir.is_relative_to(root):
        raise SystemExit("Output directory must be inside the KUNTUR repository")
    result = run(root, output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"],
                "summary": result["summary"],
                "outputs": result["outputs"],
                "runtime_seconds": result["reproducibility"]["runtime_seconds"],
                "peak_rss_bytes": result["reproducibility"]["peak_rss_bytes"],
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
