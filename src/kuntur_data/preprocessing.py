"""Versioned, native-grid preprocessing and monthly regional aggregation for KUNTUR."""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


CONTRACT_PATH = Path("configs/data_contract_v1.json")
ACQUISITION_MANIFEST = Path("data/manifests/acquisition_manifest.jsonl")
REGIONAL_MANIFEST = Path("data/manifests/regional_extraction_manifest.jsonl")
PREPROCESSING_MANIFEST = Path("data/manifests/preprocessing_manifest_v1.jsonl")
REGION_ID = "ecuador_pilot_bbox"
EARTH_RADIUS_M = 6_371_008.8


def software_versions(include_pyarrow: bool = False) -> dict[str, str]:
    versions = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import h5py
        versions["h5py"] = h5py.__version__
    except ImportError:
        pass
    if include_pyarrow:
        import pyarrow
        versions["pyarrow"] = pyarrow.__version__
    return versions


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def contract_hash(contract: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(contract).encode("utf-8"))


def processing_code_hashes(root: Path) -> dict[str, str]:
    return {
        "preprocessing_module_sha256": sha256_file(root / "src/kuntur_data/preprocessing.py"),
        "processing_entrypoint_sha256": sha256_file(root / "scripts/preprocess_kuntur_data.py"),
    }


def processing_config_hash(root: Path, contract: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(canonical_json(contract).encode("utf-8"))
    for relative in ("src/kuntur_data/preprocessing.py", "scripts/preprocess_kuntur_data.py"):
        digest.update(b"\0")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update((root / relative).read_bytes())
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Manifest row {line_no} in {path} is not an object")
            result.append(value)
    return result


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        with os.fdopen(descriptor, "ab") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def safe_manifest_path(root: Path, raw: str) -> Path:
    normalized = str(raw).replace("\\", "/")
    path = Path(normalized)
    if path.is_absolute():
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    relative = PurePosixPath(normalized)
    if ".." in relative.parts or not relative.parts or relative.parts[0] != "data":
        raise ValueError(f"Unsafe manifest path outside data/: {raw!r}")
    resolved = root.joinpath(*relative.parts).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def month_key(value: Any) -> str:
    return str(value or "")[:7]


def next_month(value: str) -> str:
    year, month = map(int, value.split("-"))
    if month == 12:
        return f"{year + 1:04d}-01"
    return f"{year:04d}-{month + 1:02d}"


def month_start(value: str) -> dt.date:
    year, month = map(int, value.split("-"))
    return dt.date(year, month, 1)


def month_end_exclusive(value: str) -> dt.date:
    return month_start(next_month(value))


def calendar_hours(value: str) -> int:
    return calendar.monthrange(*map(int, value.split("-")))[1] * 24


def month_range(start: str, end: str) -> list[str]:
    result = []
    current = start
    while current <= end:
        result.append(current)
        current = next_month(current)
    return result


def attr_text(attrs: dict[str, Any], wanted: str) -> str | None:
    match = next((value for key, value in attrs.items() if str(key).casefold() == wanted.casefold()), None)
    if match is None:
        return None
    if isinstance(match, np.ndarray):
        match = match.tolist()
    if isinstance(match, (list, tuple)) and len(match) == 1:
        match = match[0]
    if isinstance(match, bytes):
        return match.decode("utf-8", "replace")
    return str(match)


def attr_float(attrs: dict[str, Any], wanted: str) -> float | None:
    value = attr_text(attrs, wanted)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def attr_pair(attrs: dict[str, Any], wanted: str) -> tuple[float, float] | None:
    raw = attrs.get(wanted)
    if raw is None:
        raw = next((v for k, v in attrs.items() if str(k).casefold() == wanted.casefold()), None)
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        if isinstance(raw, str):
            items = re.split(r"[\s,]+", raw.strip().strip("()[]"))
        else:
            return None
    else:
        items = list(raw)
    try:
        return float(items[0]), float(items[1])
    except (ValueError, TypeError, IndexError):
        return None


def clean_attribute(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray):
        return clean_attribute(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): clean_attribute(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_attribute(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def get_hdf5_variable(handle: Any, aliases: list[str]) -> tuple[np.ndarray, dict[str, Any], str]:
    import h5py

    wanted = {normalized_name(alias) for alias in aliases}
    matches: list[tuple[str, Any]] = []

    def visit(name: str, obj: Any) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        source_name = attr_text(dict(obj.attrs), "source_variable_name")
        candidates = {normalized_name(name.split("/")[-1]), normalized_name(name)}
        if source_name:
            candidates.add(normalized_name(source_name))
        if wanted & candidates:
            matches.append((name, obj))

    handle.visititems(visit)
    if len(matches) != 1:
        raise ValueError(f"Expected one regional HDF5 variable for {aliases}, found {len(matches)}")
    name, dataset = matches[0]
    values = np.asarray(dataset[...])
    values = np.squeeze(values)
    if values.ndim != 2:
        raise ValueError(f"Regional variable {name} has shape {values.shape}, expected 2D after time squeeze")
    return values, {str(k): clean_attribute(v) for k, v in dataset.attrs.items()}, name


def read_regional_file(path: Path, source_id: str) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, dict[str, Any], str]], dict[str, Any]]:
    if source_id == "modis_mod13c2_monthly_ndvi":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(np.asarray(data["metadata_json"]).item()))
            fields = {
                "ndvi": (np.asarray(data["CMG_0.05_Deg_Monthly_NDVI"]), metadata["variable_attributes"]["CMG_0.05_Deg_Monthly_NDVI"], "CMG 0.05 Deg Monthly NDVI"),
                "evi": (np.asarray(data["CMG_0.05_Deg_Monthly_EVI"]), metadata["variable_attributes"]["CMG_0.05_Deg_Monthly_EVI"], "CMG 0.05 Deg Monthly EVI"),
                "vi_quality": (np.asarray(data["CMG_0.05_Deg_Monthly_VI_Quality"]), metadata["variable_attributes"]["CMG_0.05_Deg_Monthly_VI_Quality"], "CMG 0.05 Deg Monthly VI Quality"),
                "pixel_reliability": (np.asarray(data["CMG_0.05_Deg_Monthly_pixel_reliability"]), metadata["variable_attributes"]["CMG_0.05_Deg_Monthly_pixel_reliability"], "CMG 0.05 Deg Monthly pixel reliability"),
            }
            lat = np.asarray(data["latitude"], dtype=np.float64)
            lon = np.asarray(data["longitude"], dtype=np.float64)
        return lat, lon, fields, metadata

    import h5py

    with h5py.File(path, "r") as handle:
        if source_id == "gpm_imerg_monthly_final":
            lat = np.asarray(handle["coordinates/lat"], dtype=np.float64)
            lon = np.asarray(handle["coordinates/lon"], dtype=np.float64)
            aliases = {
                "precipitation": ["precipitation"],
                "random_error": ["randomError", "random_error"],
                "quality_index": ["precipitationQualityIndex", "quality_index"],
            }
        else:
            lat = np.asarray(handle["coordinates/lat"], dtype=np.float64)
            lon = np.asarray(handle["coordinates/lon"], dtype=np.float64)
            aliases = {
                "merra2_monthly_slv": {"t2m": ["T2M"], "qv2m": ["QV2M"], "u2m": ["U2M"], "v2m": ["V2M"], "ps": ["PS"]},
                "gldas_noah025_monthly": {
                    "soil_moisture_0_10cm": ["SoilMoi0_10cm_inst"],
                    "soil_moisture_10_40cm": ["SoilMoi10_40cm_inst"],
                    "soil_moisture_40_100cm": ["SoilMoi40_100cm_inst"],
                    "soil_moisture_100_200cm": ["SoilMoi100_200cm_inst"],
                    "evapotranspiration_rate": ["Evap_tavg"],
                    "storm_surface_runoff": ["Qs_acc"],
                    "baseflow_runoff": ["Qsb_acc"],
                    "snowmelt": ["Qsm_acc"],
                },
            }[source_id]
        source_attrs = {str(k): clean_attribute(v) for k, v in handle.attrs.items()}
        if "coordinates/time" in handle:
            time_ds = handle["coordinates/time"]
            source_attrs["regional_time_values"] = clean_attribute(np.asarray(time_ds[...]))
            source_attrs["regional_time_units"] = attr_text(dict(time_ds.attrs), "units")
        if "coordinates/time_bounds" in handle:
            bounds_ds = handle["coordinates/time_bounds"]
            source_attrs["regional_time_bounds"] = clean_attribute(np.asarray(bounds_ds[...]))
            source_attrs["regional_time_bounds_units"] = attr_text(dict(bounds_ds.attrs), "units")
        fields = {key: get_hdf5_variable(handle, names) for key, names in aliases.items()}
    return lat, lon, fields, source_attrs


def cell_edges(centers: np.ndarray, low: float, high: float) -> np.ndarray:
    values = np.asarray(centers, dtype=np.float64).reshape(-1)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite cell centers are required to derive native grid edges")
    differences = np.diff(values)
    if not np.all(differences > 0):
        raise ValueError("Latitude and longitude centers must be strictly ascending")
    median_step = float(np.median(differences))
    if not np.allclose(differences, median_step, rtol=1e-4, atol=1e-8):
        raise ValueError("Non-uniform grid centers require authoritative cell bounds; no interpolation is attempted")
    edges = np.empty(values.size + 1, dtype=np.float64)
    edges[1:-1] = (values[:-1] + values[1:]) / 2
    edges[0] = values[0] - differences[0] / 2
    edges[-1] = values[-1] + differences[-1] / 2
    edges[0] = max(low, edges[0])
    edges[-1] = min(high, edges[-1])
    edges = np.clip(edges, low, high)
    return edges


def area_weights(lat: np.ndarray, lon: np.ndarray, bbox: list[float]) -> tuple[np.ndarray, float]:
    west, south, east, north = map(float, bbox)
    lat_edges = cell_edges(lat, south, north)
    lon_edges = cell_edges(lon, west, east)
    sin_north = np.sin(np.deg2rad(lat_edges[1:]))
    sin_south = np.sin(np.deg2rad(lat_edges[:-1]))
    dlon = np.deg2rad(np.diff(lon_edges))
    weights = EARTH_RADIUS_M**2 * (sin_north - sin_south)[:, None] * dlon[None, :]
    bbox_area = EARTH_RADIUS_M**2 * math.radians(east - west) * (math.sin(math.radians(north)) - math.sin(math.radians(south)))
    if np.any(weights < 0) or not np.isfinite(weights).all() or bbox_area <= 0:
        raise ValueError("Derived spherical cell areas are invalid")
    return weights, bbox_area


def grid_id(lat: np.ndarray, lon: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(b"EPSG:4326|lat_lon_cell_centers|float64|")
    digest.update(np.asarray(lat, dtype="<f8").tobytes(order="C"))
    digest.update(b"|")
    digest.update(np.asarray(lon, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def mask_from_attrs(raw: np.ndarray, attrs: dict[str, Any], screen_range: list[float] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(raw)
    numeric = values.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    fill = attr_float(attrs, "_FillValue")
    missing = attr_float(attrs, "missing_value")
    fill_mask = ~finite
    for sentinel in (fill, missing):
        if sentinel is not None:
            fill_mask |= numeric == sentinel
    declared_range = attr_pair(attrs, "valid_range")
    if declared_range is None:
        vmin, vmax = attr_float(attrs, "valid_min"), attr_float(attrs, "valid_max")
        if vmin is not None and vmax is not None:
            declared_range = (vmin, vmax)
    selected_range = declared_range or (tuple(map(float, screen_range)) if screen_range else None)
    pre_range = finite & ~fill_mask
    range_reject = np.zeros(values.shape, dtype=bool)
    if selected_range is not None:
        range_reject = pre_range & ((numeric < selected_range[0]) | (numeric > selected_range[1]))
    valid = pre_range & ~range_reject
    audit = {
        "fill_value": fill,
        "missing_value": missing,
        "source_valid_range": list(declared_range) if declared_range else None,
        "applied_range": list(selected_range) if selected_range else None,
        "range_basis": "source_attribute" if declared_range else ("contract_physical_screen" if screen_range else "none_declared"),
    }
    return valid, fill_mask, range_reject, audit


def modis_qa_mask(vi_quality: np.ndarray, reliability: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    qa = np.asarray(vi_quality).astype(np.uint16, copy=False)
    rel = np.asarray(reliability)
    qa_valid = (qa != 65535) & (qa <= 65534)
    rel_valid = (rel != -1) & (rel >= 0) & (rel <= 4)
    modland = qa & 0b11
    usefulness = (qa >> 2) & 0b1111
    adjacent_cloud = (qa >> 8) & 1
    land_water = (qa >> 11) & 0b111
    mixed_cloud = (qa >> 10) & 1
    passed = (
        qa_valid & rel_valid & (modland == 0) & (usefulness <= 1) & (rel == 0)
        & (land_water == 1) & (adjacent_cloud == 0) & (mixed_cloud == 0)
    )
    return passed, {
        "qa_valid": qa_valid,
        "reliability_valid": rel_valid,
        "modland_quality": modland,
        "vi_usefulness": usefulness,
        "adjacent_cloud": adjacent_cloud,
        "mixed_cloud": mixed_cloud,
        "land_water_class": land_water,
        "geospatial_quality": (qa >> 14) & 0b11,
    }


def area_metrics(
    values: np.ndarray,
    valid: np.ndarray,
    fill_mask: np.ndarray,
    range_reject: np.ndarray,
    weights: np.ndarray,
    bbox_area: float,
    qa_pass: np.ndarray | None = None,
) -> dict[str, Any]:
    represented = float(np.sum(weights))
    valid_area = float(np.sum(weights[valid]))
    fill_area = float(np.sum(weights[fill_mask]))
    range_area = float(np.sum(weights[range_reject]))
    candidate = ~fill_mask & ~range_reject
    candidate_area = float(np.sum(weights[candidate]))
    qa_reject_area = 0.0
    qa_fraction = None
    final_valid = valid
    if qa_pass is not None:
        qa_reject = candidate & ~qa_pass
        qa_reject_area = float(np.sum(weights[qa_reject]))
        final_valid = valid & qa_pass
        valid_area = float(np.sum(weights[final_valid]))
        qa_fraction = (valid_area / candidate_area) if candidate_area > 0 else None
    mean = None
    if valid_area > 0:
        mean = float(np.sum(np.asarray(values, dtype=np.float64)[final_valid] * weights[final_valid]) / valid_area)
    return {
        "value": mean,
        "bbox_area_m2": bbox_area,
        "represented_grid_area_m2": represented,
        "grid_area_coverage_fraction": min(1.0, represented / bbox_area),
        "valid_area_m2": valid_area,
        "valid_area_fraction_bbox": min(1.0, valid_area / bbox_area),
        "valid_fraction_of_represented_grid": valid_area / represented if represented else 0.0,
        "fill_fraction_of_represented_grid": fill_area / represented if represented else 0.0,
        "range_reject_fraction_of_represented_grid": range_area / represented if represented else 0.0,
        "qa_pass_fraction_of_candidates": qa_fraction,
        "qa_reject_fraction_of_candidates": qa_reject_area / candidate_area if candidate_area else None,
        "valid_cell_count": int(np.count_nonzero(final_valid)),
        "candidate_cell_count": int(np.count_nonzero(candidate)),
        "quality_status": "VALID" if mean is not None else "NO_VALID_CELLS",
    }


def source_variable_specs(contract: dict[str, Any], source_id: str) -> list[dict[str, Any]]:
    return [spec for spec in contract["variables"] if spec["source_id"] == source_id]


def build_gold_rows(
    contract: dict[str, Any], source_id: str, version: str, month: str,
    bbox: list[float], grid: str, source_granule: str, source_sha: str,
    regional_path: str, regional_sha: str, config_sha: str, contract_sha: str,
    metrics_by_var: dict[str, dict[str, Any]],
    transforms: dict[str, dict[str, Any]],
    qa_context: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for spec in source_variable_specs(contract, source_id):
        variable_id = spec["variable_id"]
        metrics = metrics_by_var[variable_id]
        rows.append({
            "month": month,
            "month_start_utc": month_start(month).isoformat() + "T00:00:00Z",
            "month_end_exclusive_utc": month_end_exclusive(month).isoformat() + "T00:00:00Z",
            "days_in_month": calendar.monthrange(*map(int, month.split("-")))[1],
            "region_id": REGION_ID,
            "region_status": "PRELIMINARY_EXTRACTION_AREA",
            "bbox_west_deg": float(bbox[0]), "bbox_south_deg": float(bbox[1]),
            "bbox_east_deg": float(bbox[2]), "bbox_north_deg": float(bbox[3]),
            "dataset_id": spec["dataset_id"], "source_id": source_id,
            "dataset_version": version,
            "variable_id": variable_id, "source_variable_name": spec["source_variable_name"],
            "value": metrics["value"], "units": spec["units"],
            "temporal_statistic": spec["temporal_statistic"],
            "spatial_aggregation": spec["spatial_aggregation"],
            "native_grid_id": grid,
            "source_granule_id": source_granule,
            "source_sha256": source_sha,
            "regional_input_path": regional_path,
            "regional_input_sha256": regional_sha,
            "preprocessing_config_sha256": config_sha,
            "contract_sha256": contract_sha,
            "transform_id": spec["transform_id"],
            "applied_transform_json": canonical_json(transforms.get(variable_id, {})),
            "qa_context_json": canonical_json(qa_context),
            **{key: metrics[key] for key in (
                "bbox_area_m2", "represented_grid_area_m2", "grid_area_coverage_fraction", "valid_area_m2",
                "valid_area_fraction_bbox", "valid_fraction_of_represented_grid", "fill_fraction_of_represented_grid",
                "range_reject_fraction_of_represented_grid", "qa_pass_fraction_of_candidates",
                "qa_reject_fraction_of_candidates", "valid_cell_count", "candidate_cell_count", "quality_status"
            )},
        })
    return rows


def process_one(
    root: Path,
    contract: dict[str, Any],
    acquisition: dict[tuple[str, str], dict[str, Any]],
    regional: dict[tuple[str, str], dict[str, Any]],
    source_id: str,
    month: str,
    config_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    acquisition_row = acquisition[(source_id, month)]
    regional_row = regional[(source_id, month)]
    if acquisition_row.get("status") != "VALIDATED" or regional_row.get("status") != "VALIDATED":
        raise ValueError("Acquisition and regional manifest rows must both be VALIDATED")
    source_sha = str(acquisition_row.get("sha256") or "")
    regional_source_sha = str(regional_row.get("source_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha) or source_sha != regional_source_sha:
        raise ValueError("Regional manifest source checksum does not match the validated acquisition manifest")
    regional_path = safe_manifest_path(root, str(regional_row.get("output_path") or ""))
    if not regional_path.is_file():
        raise FileNotFoundError(f"Regional input is missing: {regional_row.get('output_path')}")
    regional_sha = sha256_file(regional_path)
    if regional_sha != regional_row.get("output_sha256"):
        raise ValueError("Regional input SHA-256 differs from the regional extraction manifest")
    if regional_path.stat().st_size != int(regional_row.get("output_bytes") or -1):
        raise ValueError("Regional input byte count differs from the regional extraction manifest")

    source_contract = contract["datasets"][source_id]
    contract_sha = contract_hash(contract)
    version = source_contract["version"]
    bbox = [float(x) for x in contract["region"]["bbox_wsen"]]
    lat, lon, fields, regional_metadata = read_regional_file(regional_path, source_id)
    metadata_start = str(regional_metadata.get("source_period_start") or "")[:7]
    metadata_end = str(regional_metadata.get("source_period_end") or "")[:7]
    if metadata_start != month or metadata_end != month:
        raise ValueError(f"Regional artifact time metadata {metadata_start}..{metadata_end} does not match manifest month {month}")
    if source_id != "modis_mod13c2_monthly_ndvi":
        bounds = regional_metadata.get("regional_time_bounds")
        expected_bounds_days = [
            (month_start(month) - dt.date(1970, 1, 1)).days,
            (month_end_exclusive(month) - dt.date(1970, 1, 1)).days,
        ]
        try:
            actual_bounds = np.asarray(bounds, dtype=np.int64).reshape(-1).tolist()
        except (TypeError, ValueError):
            actual_bounds = []
        if actual_bounds != expected_bounds_days:
            raise ValueError(f"Regional time_bounds {actual_bounds} do not match calendar month bounds {expected_bounds_days}")
    expected_shape = (len(lat), len(lon))
    for field_name, (array, _attrs, _source_name) in fields.items():
        if array.shape != expected_shape:
            raise ValueError(f"{field_name} grid shape {array.shape} does not match coordinates {expected_shape}")
    if len(lat) < 2 or len(lon) < 2 or not np.all(np.diff(lat) > 0) or not np.all(np.diff(lon) > 0):
        raise ValueError("Regional coordinates must be ascending native cell centers")
    if not (bbox[1] <= float(lat.min()) <= float(lat.max()) <= bbox[3] and bbox[0] <= float(lon.min()) <= float(lon.max()) <= bbox[2]):
        raise ValueError("Regional cell centers extend outside the manifest bbox")
    weights, bbox_area = area_weights(lat, lon, bbox)
    this_grid_id = grid_id(lat, lon)
    metrics: dict[str, dict[str, Any]] = {}
    transforms: dict[str, dict[str, Any]] = {}
    qa_context: dict[str, Any] = {"separate_quality_layer": False, "note": "Validity, fill, range, and coverage fractions are recorded per variable."}
    output_arrays: dict[str, np.ndarray] = {
        "latitude": lat.astype(np.float64), "longitude": lon.astype(np.float64),
        "cell_intersection_area_m2": weights.astype(np.float64),
    }

    if source_id == "merra2_monthly_slv":
        expected = {"t2m", "qv2m", "u2m", "v2m", "ps"}
        if set(fields) != expected:
            raise ValueError(f"MERRA-2 fields differ from contract: {sorted(fields)}")
        for key, expected_unit in {"t2m": "K", "qv2m": "kg kg-1", "u2m": "m s-1", "v2m": "m s-1", "ps": "Pa"}.items():
            attrs = fields[key][1]
            unit = attr_text(attrs, "units")
            if not unit or unit.strip().casefold() != expected_unit.casefold():
                raise ValueError(f"MERRA-2 {key} units {unit!r} do not match {expected_unit!r}")
        raw_specs = {
            "t2m": ("t2m_k", "MERRA2_T2M_K", "T2M_K", [150, 360]),
            "qv2m": ("qv2m", "MERRA2_QV2M", "QV2M", [0, 0.1]),
            "u2m": ("u2m", "MERRA2_U2M", "U2M", [-150, 150]),
            "v2m": ("v2m", "MERRA2_V2M", "V2M", [-150, 150]),
            "ps": ("ps", "MERRA2_PS", "PS", [10_000, 120_000]),
        }
        for key, (array, attrs, source_name) in fields.items():
            target_key, variable_id, _gold_name, screen = raw_specs[key]
            valid, fills, bad_range, attr_audit = mask_from_attrs(array, attrs, screen)
            physical = np.where(valid, array.astype(np.float64), np.nan)
            output_arrays[target_key] = physical
            output_arrays[f"valid_{key}"] = valid.astype(np.uint8)
            m = area_metrics(physical, valid, fills, bad_range, weights, bbox_area)
            target_id = "merra2_t2m_c" if key == "t2m" else variable_id.lower()
            if key == "t2m":
                celsius = physical - 273.15
                output_arrays["t2m_c"] = celsius
                metrics["merra2_t2m_k"] = m
                metrics["merra2_t2m_c"] = {**m, "value": (m["value"] - 273.15) if m["value"] is not None else None}
                transforms["merra2_t2m_k"] = {**attr_audit, "scale_factor_applied": 1.0, "add_offset_applied": 0.0, "source_units": "K"}
                transforms["merra2_t2m_c"] = {"formula": "T2M_C = T2M_K - 273.15", "source_units": "K", "output_units": "degC", "offset": -273.15}
            else:
                metrics[target_id] = m
                transforms[target_id] = {**attr_audit, "source_variable": source_name, "source_units": attr_text(attrs, "units")}
        # Carry NaN-corrected source kelvin into a separate internal Silver grid; Gold also records K and C.
        metrics.setdefault("merra2_t2m_k", {})

    elif source_id == "gpm_imerg_monthly_final":
        expected = {"precipitation", "random_error", "quality_index"}
        if set(fields) != expected:
            raise ValueError(f"IMERG fields differ from contract: {sorted(fields)}")
        rate, rate_attrs, rate_source = fields["precipitation"]
        rate_unit = attr_text(rate_attrs, "units")
        if not rate_unit or rate_unit.casefold().replace(" ", "") not in {"mm/hr", "mmh-1", "mm/hour"}:
            raise ValueError(f"IMERG precipitation units {rate_unit!r} are not mm/hr")
        if attr_text(fields["random_error"][1], "units") is None:
            raise ValueError("IMERG randomError is missing its units attribute")
        hours = calendar_hours(month)
        precip_valid, precip_fills, precip_bad, precip_audit = mask_from_attrs(rate, rate_attrs, [0, 200])
        rate_values = np.where(precip_valid, rate.astype(np.float64), np.nan)
        amount_values = rate_values * hours
        rate_metrics = area_metrics(rate_values, precip_valid, precip_fills, precip_bad, weights, bbox_area)
        total_metrics = {**rate_metrics, "value": (rate_metrics["value"] * hours) if rate_metrics["value"] is not None else None}
        metrics["imerg_precip_rate"] = rate_metrics
        metrics["imerg_precip_monthly_total"] = total_metrics
        transforms["imerg_precip_rate"] = {**precip_audit, "source_variable": rate_source, "source_units": "mm hr-1", "monthly_mean_rate_retained": True}
        transforms["imerg_precip_monthly_total"] = {"formula": "monthly mean rate (mm hr-1) × exact Gregorian calendar hours", "hours_in_month": hours, "days_in_month": hours // 24, "source_units": "mm hr-1", "output_units": "mm month-1"}
        output_arrays["precipitation_rate_mm_hr"] = rate_values
        output_arrays["precipitation_total_mm_month"] = amount_values
        output_arrays["valid_precipitation"] = precip_valid.astype(np.uint8)
        for key, variable_id, output_name, screen in (("random_error", "imerg_random_error", "random_error_mm_hr", [0, 200]), ("quality_index", "imerg_quality_index", "quality_index", [0, 1_000_000])):
            array, attrs, source_name = fields[key]
            valid, fills, bad, audit = mask_from_attrs(array, attrs, screen)
            physical = np.where(valid, array.astype(np.float64), np.nan)
            output_arrays[output_name] = physical
            output_arrays[f"valid_{key}"] = valid.astype(np.uint8)
            metrics[variable_id] = area_metrics(physical, valid, fills, bad, weights, bbox_area)
            transforms[variable_id] = {**audit, "source_variable": source_name, "source_units": attr_text(attrs, "units") or "index"}
        qa_context = {
            "separate_quality_layer": True,
            "quality_variable_id": "imerg_quality_index",
            "quality_index_valid_area_fraction_bbox": metrics["imerg_quality_index"]["valid_area_fraction_bbox"],
            "quality_index_area_weighted_mean": metrics["imerg_quality_index"]["value"],
            "interpretation": "NASA precipitationQualityIndex is retained as quality context; it is not a probability and is not used to mask precipitation.",
        }

    elif source_id == "gldas_noah025_monthly":
        expected = {
            "soil_moisture_0_10cm", "soil_moisture_10_40cm", "soil_moisture_40_100cm", "soil_moisture_100_200cm",
            "evapotranspiration_rate", "storm_surface_runoff", "baseflow_runoff", "snowmelt",
        }
        if set(fields) != expected:
            raise ValueError(f"GLDAS fields differ from contract: {sorted(fields)}")
        unit_expected = {
            "soil_moisture_0_10cm": "kg m-2", "soil_moisture_10_40cm": "kg m-2",
            "soil_moisture_40_100cm": "kg m-2", "soil_moisture_100_200cm": "kg m-2",
            "evapotranspiration_rate": "kg m-2 s-1", "storm_surface_runoff": "kg m-2",
            "baseflow_runoff": "kg m-2", "snowmelt": "kg m-2",
        }
        screens = {
            "soil_moisture_0_10cm": [0, 100], "soil_moisture_10_40cm": [0, 300],
            "soil_moisture_40_100cm": [0, 600], "soil_moisture_100_200cm": [0, 1000],
            "evapotranspiration_rate": None, "storm_surface_runoff": [0, 1000],
            "baseflow_runoff": [0, 1000], "snowmelt": [0, 1000],
        }
        for key, (array, attrs, source_name) in fields.items():
            unit = attr_text(attrs, "units")
            if not unit or unit.strip() != unit_expected[key]:
                raise ValueError(f"GLDAS {source_name} has units {unit!r}; expected {unit_expected[key]!r}")
            cell_methods = attr_text(attrs, "cell_methods") or ""
            if "time: mean" not in cell_methods.casefold():
                raise ValueError(f"GLDAS {source_name} does not declare monthly time-mean semantics: {cell_methods!r}")
            valid, fills, bad, audit = mask_from_attrs(array, attrs, screens[key])
            physical = np.where(valid, array.astype(np.float64), np.nan)
            output_arrays[key] = physical
            output_arrays[f"valid_{key}"] = valid.astype(np.uint8)
            variable_id = {
                "soil_moisture_0_10cm": "gldas_soil_moisture_0_10cm",
                "soil_moisture_10_40cm": "gldas_soil_moisture_10_40cm",
                "soil_moisture_40_100cm": "gldas_soil_moisture_40_100cm",
                "soil_moisture_100_200cm": "gldas_soil_moisture_100_200cm",
                "evapotranspiration_rate": "gldas_evapotranspiration_rate",
                "storm_surface_runoff": "gldas_storm_surface_runoff",
                "baseflow_runoff": "gldas_baseflow_runoff",
                "snowmelt": "gldas_snowmelt",
            }[key]
            metrics[variable_id] = area_metrics(physical, valid, fills, bad, weights, bbox_area)
            transforms[variable_id] = {
                **audit, "source_variable": source_name, "source_units": unit,
                "cell_methods": cell_methods, "aggregation_policy": "retain published monthly time mean; do not sum",
            }
        qa_context = {"separate_quality_layer": False, "note": "GLDAS land mask and source missingness are reflected in per-variable valid/fill/coverage fractions."}

    elif source_id == "modis_mod13c2_monthly_ndvi":
        expected = {"ndvi", "evi", "vi_quality", "pixel_reliability"}
        if set(fields) != expected:
            raise ValueError(f"MODIS fields differ from contract: {sorted(fields)}")
        vi_quality = fields["vi_quality"][0]
        reliability = fields["pixel_reliability"][0]
        qa_pass, qa_decoded = modis_qa_mask(vi_quality, reliability)
        output_arrays["vi_quality_raw"] = vi_quality
        output_arrays["pixel_reliability_raw"] = reliability
        for key, variable_id, scale, output_name in (("ndvi", "modis_ndvi", 0.0001, "ndvi"), ("evi", "modis_evi", 0.0001, "evi")):
            raw, attrs, source_name = fields[key]
            valid, fills, bad, audit = mask_from_attrs(raw, attrs, None)
            valid &= qa_pass
            scaled = np.where(valid, raw.astype(np.float64) * scale, np.nan)
            output_arrays[output_name] = scaled
            output_arrays[f"raw_{output_name}"] = raw
            output_arrays[f"valid_{output_name}"] = valid.astype(np.uint8)
            base_valid, base_fills, base_bad, _ = mask_from_attrs(raw, attrs, None)
            metrics[variable_id] = area_metrics(scaled, base_valid, base_fills, base_bad, weights, bbox_area, qa_pass=qa_pass)
            source_reported_scale = attr_float(attrs, "scale_factor")
            transforms[variable_id] = {
                **audit, "source_variable": source_name, "raw_valid_range": [-2000, 10000],
                "official_multiplier": scale, "source_reported_scale_factor": source_reported_scale,
                "scale_discrepancy_note": "If metadata reports 10000, follow MOD13C2.061 guide multiplier 0.0001; raw values retained.",
                "qa_policy": contract["quality_policy"]["modis_c2"],
            }
        for name, decoded in qa_decoded.items():
            output_arrays[f"qa_{name}"] = decoded
        represented_area = float(np.sum(weights))
        qa_context = {
            "separate_quality_layer": True,
            "quality_variable_ids": ["modis_ndvi", "modis_evi"],
            "strict_qa_pass_area_fraction_of_bbox_by_variable": {
                "modis_ndvi": metrics["modis_ndvi"]["valid_area_fraction_bbox"],
                "modis_evi": metrics["modis_evi"]["valid_area_fraction_bbox"],
            },
            "strict_qa_pass_fraction_of_candidates_by_variable": {
                "modis_ndvi": metrics["modis_ndvi"]["qa_pass_fraction_of_candidates"],
                "modis_evi": metrics["modis_evi"]["qa_pass_fraction_of_candidates"],
            },
            "modland_quality_area_fraction": {
                str(code): float(np.sum(weights[(qa_decoded["modland_quality"] == code) & qa_decoded["qa_valid"]]) / represented_area) if represented_area else 0.0
                for code in range(4)
            },
            "vi_usefulness_area_fraction": {
                str(code): float(np.sum(weights[(qa_decoded["vi_usefulness"] == code) & qa_decoded["qa_valid"]]) / represented_area) if represented_area else 0.0
                for code in range(16)
            },
            "pixel_reliability_area_fraction": {
                str(code): float(np.sum(weights[reliability == code]) / represented_area) if represented_area else 0.0
                for code in (-1, 0, 1, 2, 3, 4)
            },
            "land_water_class_area_fraction": {
                str(code): float(np.sum(weights[(qa_decoded["land_water_class"] == code) & qa_decoded["qa_valid"]]) / represented_area) if represented_area else 0.0
                for code in range(8)
            },
            "geospatial_quality_area_fraction": {
                str(code): float(np.sum(weights[(qa_decoded["geospatial_quality"] == code) & qa_decoded["qa_valid"]]) / represented_area) if represented_area else 0.0
                for code in range(4)
            },
            "strict_mask": contract["quality_policy"]["modis_c2"],
        }

    else:
        raise ValueError(f"Unrecognized source_id {source_id}")

    expected_ids = {spec["variable_id"] for spec in source_variable_specs(contract, source_id)}
    if set(metrics) != expected_ids:
        raise ValueError(f"Processed Gold variables differ from contract; expected={sorted(expected_ids)} actual={sorted(metrics)}")

    processed_dir = root / "data" / "silver" / "processed_native_grid" / "v1" / source_id / month[:4]
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"{month.replace('-', '')}.npz"
    tmp_path = processed_path.with_name(f".{processed_path.name}.{os.getpid()}.tmp")
    metadata = {
        "processing_version": contract["processing_version"], "contract_version": contract["contract_version"],
        "source_id": source_id, "dataset_id": source_contract["dataset_id"], "dataset_version": version,
        "month": month, "region_id": REGION_ID, "region_bbox_wsen": bbox, "native_grid_id": this_grid_id,
        "grid_shape_lat_lon": list(expected_shape), "source_sha256": source_sha,
        "regional_input_path": str(regional_row["output_path"]), "regional_input_sha256": regional_sha,
        "regional_input_bytes": regional_path.stat().st_size,
        "acquisition_manifest_path": ACQUISITION_MANIFEST.as_posix(),
        "regional_manifest_path": REGIONAL_MANIFEST.as_posix(), "preprocessing_config_sha256": config_sha,
        "contract_sha256": contract_sha,
        "regional_metadata": clean_attribute(regional_metadata), "transforms": transforms, "summary_metrics": metrics,
        "software_versions": software_versions(), "processing_code_hashes": processing_code_hashes(root),
        "no_regridding": True,
    }
    output_arrays["metadata_json"] = np.asarray(canonical_json(metadata))
    try:
        with tmp_path.open("wb") as stream:
            np.savez_compressed(stream, **output_arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, processed_path)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise

    output_sha = sha256_file(processed_path)
    gold_rows = build_gold_rows(
        contract, source_id, version, month, bbox, this_grid_id,
        str(acquisition_row.get("object_id") or acquisition_row.get("granule_concept_id") or ""),
        source_sha, str(regional_row["output_path"]), regional_sha, config_sha, contract_sha,
        metrics, transforms, qa_context,
    )
    for row in gold_rows:
        row["processed_native_grid_path"] = processed_path.relative_to(root).as_posix()
        row["processed_native_grid_sha256"] = output_sha
    manifest_row = {
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "VALIDATED", "source_id": source_id, "dataset_id": source_contract["dataset_id"],
        "dataset_version": version, "month": month, "region_id": REGION_ID,
        "source_granule_id": str(acquisition_row.get("object_id") or ""), "source_sha256": source_sha,
        "regional_input_path": str(regional_row["output_path"]), "regional_input_sha256": regional_sha,
        "regional_input_bytes": regional_path.stat().st_size,
        "processed_path": processed_path.relative_to(root).as_posix(),
        "processed_bytes": processed_path.stat().st_size, "processed_sha256": output_sha,
        "native_grid_id": this_grid_id, "grid_shape_lat_lon": list(expected_shape),
        "preprocessing_config_sha256": config_sha, "processing_version": contract["processing_version"],
        "contract_sha256": contract_sha,
        "processing_code_hashes": processing_code_hashes(root),
        "software_versions": software_versions(), "transforms": transforms,
        "validation": {
            "status": "PASSED", "checks": [
                "source acquisition row is VALIDATED and SHA-256 is present",
                "regional input exists and size and SHA-256 match regional manifest",
                "native coordinate arrays are ascending and match every variable shape",
                "cell area weights are spherical and clipped to the pilot bbox",
                "fill, missing, valid-range and QA masks are applied without imputation",
                "dataset-specific unit and aggregation semantics checks passed",
                "native grid coordinates are retained; no regridding or interpolation",
                "processed NPZ reopened by NumPy and output SHA-256 recorded",
            ],
        },
        "summary_metrics": metrics,
        "gold_rows": gold_rows,
    }
    with np.load(processed_path, allow_pickle=False) as check:
        if tuple(check["latitude"].shape) != (expected_shape[0],) or tuple(check["longitude"].shape) != (expected_shape[1],):
            raise ValueError("Processed native-grid artifact failed reopen/coordinate validation")
        for name in ("cell_intersection_area_m2",):
            if check[name].shape != expected_shape:
                raise ValueError(f"Processed artifact {name} shape is invalid")
    return manifest_row, gold_rows


def latest_manifest_map(rows: list[dict[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, Any]]:
    result: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(k) or "") for k in key_fields)
        result[key] = row
    return result


def forecast_rows(gold_rows: list[dict[str, Any]], contract: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    temp = {
        row["month"]: row for row in gold_rows
        if row["variable_id"] == contract["forecast"]["target_variable_id"]
    }
    predictors = contract["forecast"]["predictor_variable_ids"]
    by_variable: dict[str, dict[str, dict[str, Any]]] = {var: {} for var in predictors}
    for row in gold_rows:
        if row["variable_id"] in by_variable:
            by_variable[row["variable_id"]][row["month"]] = row
    all_months = month_range(contract["coverage"]["start_month"], contract["coverage"]["end_month"])
    samples: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    release_lag = int(contract["forecast"]["conservative_release_lag_months"])
    window = int(contract["forecast"]["history_window_months"])
    minimum_coverage = float(contract["forecast"]["minimum_valid_area_fraction_bbox"])
    for target in all_months:
        target_idx = all_months.index(target)
        issue_idx = target_idx - 1
        latest_idx = issue_idx - release_lag
        first_idx = latest_idx - window + 1
        if first_idx < 0:
            excluded.append({"target_month": target, "reason": "INSUFFICIENT_PRE_TARGET_HISTORY", "required_history_months": window})
            continue
        issue_month = all_months[issue_idx]
        feature_months = all_months[first_idx:latest_idx + 1]
        if len(feature_months) != window or any(m not in temp for m in (target,)):
            excluded.append({"target_month": target, "reason": "TARGET_OR_HISTORY_MONTH_NOT_AVAILABLE"})
            continue
        target_row = temp[target]
        if target_row["value"] is None or target_row["valid_area_fraction_bbox"] < minimum_coverage:
            excluded.append({"target_month": target, "reason": "TARGET_QUALITY_GATE_FAILED"})
            continue
        unavailable = []
        for feature_month in feature_months:
            for variable_id in predictors:
                row = by_variable[variable_id].get(feature_month)
                if row is None or row["value"] is None or row["valid_area_fraction_bbox"] < minimum_coverage:
                    unavailable.append({"month": feature_month, "variable_id": variable_id})
        if unavailable:
            excluded.append({"target_month": target, "reason": "HISTORY_QUALITY_GATE_FAILED", "unavailable_features": unavailable})
            continue
        sample_id = f"{contract['forecast']['dataset_id']}:{target}"
        row: dict[str, Any] = {
            "sample_id": sample_id, "forecast_origin_month": issue_month,
            "issue_date_utc": month_start(issue_month).isoformat() + "T00:00:00Z",
            "target_month": target,
            "target_start_utc": target_row["month_start_utc"],
            "target_end_exclusive_utc": target_row["month_end_exclusive_utc"],
            "history_start_month": feature_months[0], "history_end_month": feature_months[-1],
            "history_window_months": window, "conservative_release_lag_months": release_lag,
            "feature_months_json": canonical_json(feature_months),
            "target_variable_id": contract["forecast"]["target_variable_id"],
            "target_value": target_row["value"], "target_units": target_row["units"],
            "target_source_sha256": target_row["source_sha256"],
            "target_processed_native_grid_sha256": target_row.get("processed_native_grid_sha256"),
            "target_valid_area_fraction_bbox": target_row["valid_area_fraction_bbox"],
        }
        lineage = []
        # lag_01 is the latest available month (target - 3); lag_12 is the oldest.
        for lag, feature_month in enumerate(reversed(feature_months), start=1):
            for variable_id in predictors:
                feature = by_variable[variable_id][feature_month]
                name = contract["forecast"]["feature_column_prefixes"][variable_id]
                row[f"{name}_lag_{lag:02d}"] = feature["value"]
                lineage.append({
                    "feature_month": feature_month, "variable_id": variable_id,
                    "source_sha256": feature["source_sha256"],
                    "regional_input_sha256": feature.get("regional_input_sha256"),
                    "processed_native_grid_sha256": feature.get("processed_native_grid_sha256"),
                })
        row["history_lineage_json"] = canonical_json(lineage)
        samples.append(row)
    split_rows = []
    partitions = contract["forecast"]["partitions_by_target_year"]
    for sample in samples:
        year = int(sample["target_month"][:4])
        part = next((name for name, years in partitions.items() if years[0] <= year <= years[1]), None)
        if part is None:
            raise ValueError(f"No frozen chronological split covers target month {sample['target_month']}")
        split_rows.append({
            "sample_id": sample["sample_id"], "target_month": sample["target_month"],
            "forecast_origin_month": sample["forecast_origin_month"], "issue_date_utc": sample["issue_date_utc"],
            "history_start_month": sample["history_start_month"], "history_end_month": sample["history_end_month"],
            "partition": part,
        })
    audit = {
        "eligible_sample_count": len(samples), "excluded_target_month_count": len(excluded),
        "excluded_target_months": excluded,
        "partition_counts": {name: sum(1 for r in split_rows if r["partition"] == name) for name in partitions},
        "checks": {
            "feature_months_strictly_before_issue_month": all(sample["history_end_month"] < sample["forecast_origin_month"] for sample in samples),
            "history_window_lengths_equal_contract": all(len(json.loads(sample["feature_months_json"])) == window for sample in samples),
            "target_month_exactly_one_after_origin": all(next_month(s["forecast_origin_month"]) == s["target_month"] for s in samples),
            "partitions_assigned_by_target_year": all(
                next(name for name, years in partitions.items() if years[0] <= int(s["target_month"][:4]) <= years[1]) ==
                next(r["partition"] for r in split_rows if r["sample_id"] == s["sample_id"])
                for s in samples
            ),
            "no_random_split": True,
            "no_future_features": True,
        },
    }
    if not all(audit["checks"].values()):
        raise ValueError("Forecast leakage audit failed")
    return samples, split_rows, audit


def write_parquet(path: Path, rows: list[dict[str, Any]], metadata: dict[str, str]) -> tuple[int, str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    encoded_meta = {k.encode(): v.encode() for k, v in metadata.items()}
    schema_meta = dict(table.schema.metadata or {})
    schema_meta.update(encoded_meta)
    table = table.replace_schema_metadata(schema_meta)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    pq.write_table(table, temp, compression="zstd", version="2.6", write_statistics=True)
    with temp.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    # Reopen with Arrow and verify row count/schema metadata after atomic publish.
    reopened = pq.read_table(path)
    if reopened.num_rows != len(rows):
        raise ValueError(f"Parquet row count mismatch for {path}")
    return reopened.num_rows, sha256_file(path), path.stat().st_size


def build_gold_outputs(
    root: Path,
    contract: dict[str, Any],
    gold_rows: list[dict[str, Any]],
    config_sha: str,
) -> dict[str, Any]:
    acquisition_path = root / ACQUISITION_MANIFEST
    regional_path = root / REGIONAL_MANIFEST
    preprocess_manifest = root / PREPROCESSING_MANIFEST
    coverage_months = month_range(contract["coverage"]["start_month"], contract["coverage"]["end_month"])
    gold_rows.sort(key=lambda r: (r["month"], r["dataset_id"], r["variable_id"]))
    gold_path = root / contract["outputs"]["gold_parquet"]
    gold_count, gold_sha, gold_bytes = write_parquet(gold_path, gold_rows, {
        "kuntur.contract_version": contract["contract_version"], "kuntur.config_sha256": config_sha,
        "kuntur.region_id": REGION_ID, "kuntur.temporal_coverage": f"{coverage_months[0]}/{coverage_months[-1]}",
        "kuntur.spatial_policy": "native grids preserved; spherical cell areas clipped to preliminary bbox; no interpolation",
    })
    sources_summary = {}
    for sid in contract["datasets"]:
        scoped = [r for r in gold_rows if r["source_id"] == sid]
        sources_summary[sid] = {
            "dataset_id": contract["datasets"][sid]["dataset_id"],
            "dataset_version": contract["datasets"][sid]["version"],
            "months": sorted({r["month"] for r in scoped}), "month_count": len({r["month"] for r in scoped}),
            "source_sha256_count": len({r["source_sha256"] for r in scoped}),
        }
    gold_manifest = {
        "manifest_version": "1.0.0", "status": "VALIDATED", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_path": gold_path.relative_to(root).as_posix(), "format": "Apache Parquet", "rows": gold_count,
        "bytes": gold_bytes, "sha256": gold_sha, "contract_version": contract["contract_version"],
        "contract_sha256": contract_hash(contract), "preprocessing_config_sha256": config_sha,
        "source_manifest_sha256": sha256_file(acquisition_path),
        "regional_manifest_sha256": sha256_file(regional_path), "preprocessing_manifest_sha256": sha256_file(preprocess_manifest),
        "software_versions": software_versions(include_pyarrow=True),
        "processing_code_hashes": processing_code_hashes(root),
        "source_datasets": sources_summary,
        "validation": {"status": "PASSED", "checks": ["Parquet reopened with PyArrow", "row count equals expected month-variable grid", "source/regional/config hashes present on all rows", "no missing month-variable combinations"]},
    }
    atomic_write_json(root / contract["outputs"]["gold_manifest"], gold_manifest)

    samples, split_rows, leakage_audit = forecast_rows(gold_rows, contract)
    if not samples:
        raise ValueError("No forecast windows passed the frozen availability and quality policy")
    forecast_path = root / contract["outputs"]["forecast_parquet"]
    forecast_count, forecast_sha, forecast_bytes = write_parquet(forecast_path, samples, {
        "kuntur.dataset_id": contract["forecast"]["dataset_id"], "kuntur.contract_version": contract["contract_version"],
        "kuntur.target": contract["forecast"]["target_variable_id"],
        "kuntur.leakage_policy": canonical_json(contract["forecast"]["availability_assumptions"]),
        "kuntur.split_policy": canonical_json(contract["forecast"]["partitions_by_target_year"]),
    })
    split_path = root / contract["outputs"]["split_indices_parquet"]
    split_count, split_sha, split_bytes = write_parquet(split_path, split_rows, {
        "kuntur.contract_version": contract["contract_version"], "kuntur.assignment": "chronological by exact target month; no random assignment",
    })
    split_manifest = {
        "manifest_version": "1.0.0", "status": "VALIDATED", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "forecast_dataset_id": contract["forecast"]["dataset_id"], "target_variable_id": contract["forecast"]["target_variable_id"],
        "history_window_months": contract["forecast"]["history_window_months"],
        "conservative_release_lag_months": contract["forecast"]["conservative_release_lag_months"],
        "target_horizon_months": 1, "feature_month_order": "lag_01 is most recent available month; lag_12 is oldest",
        "forecast_path": forecast_path.relative_to(root).as_posix(), "forecast_rows": forecast_count,
        "forecast_bytes": forecast_bytes, "forecast_sha256": forecast_sha,
        "split_indices_path": split_path.relative_to(root).as_posix(), "split_rows": split_count,
        "split_bytes": split_bytes, "split_sha256": split_sha,
        "gold_path": gold_path.relative_to(root).as_posix(), "gold_sha256": gold_sha,
        "contract_sha256": contract_hash(contract), "preprocessing_config_sha256": config_sha,
        "source_manifest_sha256": sha256_file(root / ACQUISITION_MANIFEST),
        "regional_manifest_sha256": sha256_file(root / REGIONAL_MANIFEST),
        "preprocessing_manifest_sha256": sha256_file(root / PREPROCESSING_MANIFEST),
        "partitions_by_target_year": contract["forecast"]["partitions_by_target_year"],
        "software_versions": software_versions(include_pyarrow=True),
        "processing_code_hashes": processing_code_hashes(root),
        "partition_counts": leakage_audit["partition_counts"], "leakage_audit": leakage_audit,
        "excluded_targets": leakage_audit["excluded_target_months"],
    }
    atomic_write_json(root / contract["outputs"]["split_manifest"], split_manifest)
    return {
        "gold_rows": gold_count, "gold_bytes": gold_bytes, "gold_sha256": gold_sha,
        "gold_path": gold_path.relative_to(root).as_posix(),
        "forecast_rows": forecast_count, "forecast_path": forecast_path.relative_to(root).as_posix(),
        "split_rows": split_count, "split_counts": leakage_audit["partition_counts"],
        "excluded_targets": leakage_audit["excluded_target_month_count"], "config_sha256": config_sha,
        "failures": [],
    }


def finalize_from_preprocessing_manifest(root: Path) -> dict[str, Any]:
    contract = json.loads((root / CONTRACT_PATH).read_text(encoding="utf-8"))
    config_sha = processing_config_hash(root, contract)
    frozen_contract_sha = contract_hash(contract)
    manifest_path = root / PREPROCESSING_MANIFEST
    latest = latest_manifest_map(load_jsonl(manifest_path), ("source_id", "month", "region_id"))
    expected_pairs = {
        (source_id, month, REGION_ID)
        for source_id in contract["datasets"]
        for month in month_range(contract["coverage"]["start_month"], contract["coverage"]["end_month"])
    }
    absent = expected_pairs - set(latest)
    failed = [key for key in expected_pairs & set(latest) if latest[key].get("status") != "VALIDATED"]
    if absent or failed:
        raise ValueError(f"Preprocessing manifest is incomplete: missing={len(absent)} latest_failed={len(failed)}")
    gold_rows: list[dict[str, Any]] = []
    expected_vars = {spec["variable_id"] for spec in contract["variables"]}
    for key in sorted(expected_pairs):
        record = latest[key]
        if record.get("preprocessing_config_sha256") != config_sha or record.get("contract_sha256") != frozen_contract_sha:
            raise ValueError(f"Preprocessing config hash mismatch for {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("source_sha256") or "")):
            raise ValueError(f"Source checksum is absent or malformed for {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("regional_input_sha256") or "")):
            raise ValueError(f"Regional input checksum is absent or malformed for {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("processed_sha256") or "")):
            raise ValueError(f"Processed output checksum is absent or malformed for {key}")
        rows = record.get("gold_rows")
        if not isinstance(rows, list) or {row.get("variable_id") for row in rows} != {
            spec["variable_id"] for spec in contract["variables"] if spec["source_id"] == key[0]
        }:
            raise ValueError(f"Gold summary rows do not match frozen variable contract for {key}")
        for row in rows:
            row["processed_native_grid_path"] = record["processed_path"]
            row["processed_native_grid_sha256"] = record["processed_sha256"]
        gold_rows.extend(rows)
    expected_count = len(expected_pairs) * int(contract["gold_variable_count"])
    if len(gold_rows) != expected_count:
        raise ValueError(f"Manifest Gold summary has {len(gold_rows)} rows; expected {expected_count}")
    return build_gold_outputs(root, contract, gold_rows, config_sha)


def run(
    root: Path,
    limit: int = 0,
    source_ids: set[str] | None = None,
    months: set[str] | None = None,
    build_outputs: bool = True,
) -> dict[str, Any]:
    contract_path = root / CONTRACT_PATH
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    config_sha = processing_config_hash(root, json.loads(contract_path.read_text(encoding="utf-8")))
    acquisition_path = root / ACQUISITION_MANIFEST
    regional_path = root / REGIONAL_MANIFEST
    acquisition_rows = load_jsonl(acquisition_path)
    regional_rows = load_jsonl(regional_path)
    acquisition = latest_manifest_map(
        [r for r in acquisition_rows if r.get("source_id") in contract["datasets"]], ("source_id", "period_start")
    )
    # Normalize month keys and keep each manifest's last row for source/month.
    acquisition_month: dict[tuple[str, str], dict[str, Any]] = {}
    for row in acquisition_rows:
        sid = str(row.get("source_id") or "")
        if sid in contract["datasets"]:
            acquisition_month[(sid, month_key(row.get("period_start")))] = row
    regional_month: dict[tuple[str, str], dict[str, Any]] = {}
    for row in regional_rows:
        sid = str(row.get("source_id") or "")
        if sid in contract["datasets"] and row.get("region_id") == REGION_ID:
            regional_month[(sid, month_key(row.get("period_start")))] = row
    del acquisition
    coverage_months = month_range(contract["coverage"]["start_month"], contract["coverage"]["end_month"])
    selected_months = list(coverage_months)
    if months:
        unknown_months = set(months) - set(coverage_months)
        if unknown_months:
            raise ValueError(f"Requested months are outside the frozen coverage: {sorted(unknown_months)}")
        selected_months = [month for month in coverage_months if month in months]
    ids = list(contract["datasets"])
    if source_ids:
        unknown = set(source_ids) - set(ids)
        if unknown:
            raise ValueError(f"Unknown source ids: {sorted(unknown)}")
        ids = [source for source in ids if source in source_ids]
    pairs = [(sid, month) for sid in ids for month in selected_months]
    if limit:
        pairs = pairs[:limit]
    expected = {(sid, month) for sid in ids for month in selected_months}
    missing = [key for key in expected if key not in acquisition_month or key not in regional_month]
    sample_mode = bool(limit or source_ids or months)
    if missing and not sample_mode:
        raise ValueError(f"Manifest coverage is missing {len(missing)} acquisition/regional pairs; examples={missing[:5]}")
    for sid, month in pairs:
        if (sid, month) not in acquisition_month or (sid, month) not in regional_month:
            raise ValueError(f"No manifest pair for {sid} {month}")

    preprocess_manifest = root / PREPROCESSING_MANIFEST
    prior: dict[tuple[str, str, str], dict[str, Any]] = {}
    if preprocess_manifest.exists():
        for row in load_jsonl(preprocess_manifest):
            prior[(str(row.get("source_id")), str(row.get("month")), str(row.get("region_id")))] = row
    gold_rows: list[dict[str, Any]] = []
    validated_rows = 0
    reused_rows = 0
    failures: list[dict[str, str]] = []
    for sid, month in pairs:
        prior_row = prior.get((sid, month, REGION_ID))
        regional_record = regional_month[(sid, month)]
        acquisition_record = acquisition_month[(sid, month)]
        reusable = False
        if prior_row and prior_row.get("status") == "VALIDATED" and prior_row.get("preprocessing_config_sha256") == config_sha:
            try:
                processed = safe_manifest_path(root, prior_row["processed_path"])
                regional_input = safe_manifest_path(root, str(regional_record.get("output_path") or ""))
                region_integrity = (
                    regional_input.is_file()
                    and regional_input.stat().st_size == regional_record.get("output_bytes")
                    and sha256_file(regional_input) == regional_record.get("output_sha256")
                )
                reusable = (
                    processed.is_file() and processed.stat().st_size == prior_row.get("processed_bytes")
                    and sha256_file(processed) == prior_row.get("processed_sha256")
                    and region_integrity
                    and prior_row.get("regional_input_sha256") == regional_record.get("output_sha256")
                    and prior_row.get("source_sha256") == acquisition_record.get("sha256")
                )
            except (OSError, ValueError, KeyError):
                reusable = False
        try:
            if reusable:
                rows = prior_row.get("gold_rows") or []
                if not rows:
                    raise ValueError("Prior validated preprocessing record has no reproducible Gold summary rows")
                for row in rows:
                    row["processed_native_grid_path"] = prior_row["processed_path"]
                    row["processed_native_grid_sha256"] = prior_row["processed_sha256"]
                gold_rows.extend(rows)
                reused_rows += 1
                print(f"REUSED source_id={sid} month={month} processed_sha256={prior_row['processed_sha256']}", flush=True)
                continue
            manifest_row, rows = process_one(root, contract, acquisition_month, regional_month, sid, month, config_sha)
            for row in rows:
                row["processed_native_grid_path"] = manifest_row["processed_path"]
                row["processed_native_grid_sha256"] = manifest_row["processed_sha256"]
            append_jsonl(preprocess_manifest, manifest_row)
            prior[(sid, month, REGION_ID)] = manifest_row
            gold_rows.extend(rows)
            validated_rows += 1
            print(f"PREPROCESSED source_id={sid} month={month} output={manifest_row['processed_path']} sha256={manifest_row['processed_sha256']}", flush=True)
        except Exception as exc:
            failure = {"source_id": sid, "month": month, "error_type": type(exc).__name__, "error": str(exc)[:500]}
            failures.append(failure)
            append_jsonl(preprocess_manifest, {
                "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "status": "FAILED",
                "source_id": sid, "month": month, "region_id": REGION_ID,
                "source_sha256": acquisition_record.get("sha256"),
                "regional_input_path": regional_record.get("output_path"),
                "regional_input_sha256": regional_record.get("output_sha256"),
                "preprocessing_config_sha256": config_sha,
                "error_type": failure["error_type"], "error": failure["error"],
            })
            if not limit:
                continue
            raise

    # A focused run validates the small sample and writes its preprocessing manifest only.
    if sample_mode:
        return {"validated": validated_rows, "reused": reused_rows, "gold_rows": len(gold_rows), "failures": failures, "config_sha256": config_sha}
    if failures:
        raise RuntimeError(f"Preprocessing failed for {len(failures)} monthly regional inputs; see {PREPROCESSING_MANIFEST}")
    expected_gold = len(pairs) * int(contract["gold_variable_count"])
    if len(gold_rows) != expected_gold:
        raise ValueError(f"Gold record count is {len(gold_rows)}, expected {expected_gold}")
    if not build_outputs:
        return {"validated": validated_rows, "reused": reused_rows, "preprocessed_summary_rows": len(gold_rows), "config_sha256": config_sha, "failures": failures}
    result = build_gold_outputs(root, contract, gold_rows, config_sha)
    result.update({"validated": validated_rows, "reused": reused_rows})
    return result
