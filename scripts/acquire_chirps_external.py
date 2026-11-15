#!/usr/bin/env python3
"""Create resumable Ecuador-bbox CHIRPS v3 monthly crops from official COGs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.windows import from_bounds


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/cogs"
BBOX = (-81.5, -5.1, -75.0, 1.7)
FILL_VALUE = -9999.0
START_YEAR, END_YEAR = 2002, 2024
OUTPUT_DIR = ROOT / "data/external/processed/chirps_v3_ecuador_bbox_monthly"
ROWS_PATH = ROOT / "data/external/processed/chirps_v3_ecuador_bbox_monthly_v1.parquet"
CSV_PATH = ROOT / "data/external/processed/chirps_v3_ecuador_bbox_monthly_v1.csv"
JSONL_PATH = ROOT / "data/manifests/external/chirps_v3_monthly_progress_v1.jsonl"
MANIFEST_PATH = ROOT / "data/manifests/external/chirps_v3_external_manifest_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def process_month(month: str) -> dict[str, object]:
    year, month_number = month.split("-")
    filename = f"chirps-v3.0.{year}.{month_number}.cog"
    url = f"{BASE_URL}/{filename}"
    metadata = {"month": month, "source_url": url, "source_full_sha256": None, "source_checksum_note": "Remote COG was accessed by HTTP range reads; the full source object was not downloaded, so no full-object SHA-256 is claimed."}
    head = requests.head(url, timeout=(15, 30), allow_redirects=True)
    metadata.update({"http_status": head.status_code, "source_etag": head.headers.get("ETag"), "source_last_modified": head.headers.get("Last-Modified"), "source_content_length": head.headers.get("Content-Length")})
    head.raise_for_status()
    out_path = OUTPUT_DIR / f"chirps-v3.0.{year}.{month_number}.ecuador_bbox.tif"
    sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
    if out_path.exists() and sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if sidecar.get("source_url") == url and sidecar.get("source_etag") == metadata["source_etag"] and sidecar.get("output_sha256") == sha256(out_path):
            return sidecar
    vsi_url = "/vsicurl/" + url
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".cog", CPL_VSIL_CURL_USE_HEAD="NO"):
        with rasterio.open(vsi_url) as source:
            window = from_bounds(*BBOX, transform=source.transform).round_offsets().round_lengths()
            array = source.read(1, window=window).astype(np.float32, copy=False)
            transform = source.window_transform(window)
            x = transform.c + (np.arange(array.shape[1]) + 0.5) * transform.a
            y = transform.f + (np.arange(array.shape[0]) + 0.5) * transform.e
            inside = ((x >= BBOX[0]) & (x <= BBOX[2]))[None, :] & ((y >= BBOX[1]) & (y <= BBOX[3]))[:, None]
            valid = np.isfinite(array) & (array != FILL_VALUE) & (array >= 0.0) & inside
            rejected_negative = int((np.isfinite(array) & (array != FILL_VALUE) & (array < 0.0) & inside).sum())
            lat_weights = np.cos(np.deg2rad(np.broadcast_to(y[:, None], array.shape)))
            if int(valid.sum()) == 0:
                raise ValueError(f"No valid CHIRPS pixels in bbox for {month}")
            weights = lat_weights[valid]
            mean = float(np.average(array[valid], weights=weights))
            nodata_array = array.copy()
            nodata_array[~valid] = FILL_VALUE
            profile = {
                "driver": "GTiff", "height": array.shape[0], "width": array.shape[1], "count": 1,
                "dtype": "float32", "crs": source.crs, "transform": transform,
                "nodata": FILL_VALUE, "compress": "deflate", "predictor": 3,
            }
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with rasterio.open(tmp, "w", **profile) as dest:
                dest.write(nodata_array, 1)
                dest.set_band_description(1, "monthly_precipitation_total")
                dest.update_tags(
                    dataset_id="CHIRPS_v3.0_final_monthly", time=month, units="mm/month",
                    source_url=url, source_etag=metadata["source_etag"] or "",
                    bbox_wsen=",".join(str(v) for v in BBOX), source_fill_value=str(FILL_VALUE),
                    extraction="native 0.05-degree EPSG:4326 grid; pixel centers constrained to frozen pilot bbox",
                )
            tmp.replace(out_path)
    result = {
        **metadata,
        "status": "CROP_VALIDATED",
        "output_path": out_path.relative_to(ROOT).as_posix(),
        "output_bytes": out_path.stat().st_size,
        "output_sha256": sha256(out_path),
        "grid": {"crs": "EPSG:4326", "resolution_degrees": 0.05, "shape_y_x": list(array.shape), "transform": list(transform)[:6]},
        "units": "mm/month",
        "valid_pixel_count": int(valid.sum()),
        "bbox_pixel_count": int(inside.sum()),
        "valid_pixel_fraction_bbox": float(valid.sum() / inside.sum()),
        "rejected_negative_pixel_count": rejected_negative,
        "area_weighted_mean_mm_month": mean,
        "valid_min_mm_month": float(array[valid].min()),
        "valid_max_mm_month": float(array[valid].max()),
        "fill_value": FILL_VALUE,
        "validation": "native-grid GeoTIFF readable; CRS/transform preserved; bbox pixel centers masked; finite nonnegative precipitation only; SHA-256 verified",
    }
    atomic_json(sidecar_path, result)
    return result


def retry_delay(exc: Exception, attempt: int) -> float | None:
    """Return a bounded polite retry delay for rate limits and transient server errors."""
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    message = str(exc)
    retryable = status == 429 or (status is not None and status >= 500) or bool(re.search(r"(?:HTTP response code:\s*429|\b429\b)", message))
    # A failed range read can surface as a format/read error after the server
    # returns a short 429 body. Retry it once through the same official URL.
    if not retryable and "not recognized as being in a supported file format" in message:
        retryable = True
    if not retryable or attempt >= 4:
        return None
    header = getattr(response, "headers", {}).get("Retry-After") if response is not None else None
    try:
        requested = float(header) if header is not None else 10.0 * (2**attempt)
    except (TypeError, ValueError):
        requested = 10.0 * (2**attempt)
    return min(max(requested, 1.0), 300.0)


def main() -> None:
    started = time.perf_counter()
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    completed: dict[str, dict[str, object]] = {}
    progress_history: list[dict[str, object]] = []
    if JSONL_PATH.exists():
        for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                progress_history.append(record)
                if record.get("status") == "CROP_VALIDATED":
                    completed[record["month"]] = record
            except (json.JSONDecodeError, KeyError):
                continue
    failures: list[dict[str, object]] = []
    retry_events: list[dict[str, object]] = []
    expected = [str(period) for period in pd.period_range(f"{START_YEAR}-01", f"{END_YEAR}-12", freq="M")]
    with JSONL_PATH.open("a", encoding="utf-8", newline="\n") as log:
        for month in expected:
            out_path = OUTPUT_DIR / f"chirps-v3.0.{month[:4]}.{month[-2:]}.ecuador_bbox.tif"
            sidecar_path = out_path.with_suffix(out_path.suffix + ".json")
            if month in completed and out_path.is_file() and sidecar_path.is_file():
                try:
                    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    if sidecar.get("output_sha256") == sha256(out_path) and sidecar.get("month") == month:
                        continue
                except (OSError, json.JSONDecodeError):
                    pass
            attempt = 0
            try:
                while True:
                    try:
                        result = process_month(month)
                        if attempt:
                            result["retry_attempts"] = attempt
                            retry_events.append({"month": month, "retries": attempt, "status": "RECOVERED"})
                        completed[month] = result
                        log.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
                        log.flush()
                        break
                    except Exception as exc:
                        delay = retry_delay(exc, attempt)
                        if delay is None:
                            raise
                        retry_events.append({"month": month, "attempt": attempt + 1, "delay_seconds": delay, "error_type": type(exc).__name__, "status": "RETRY_WAIT"})
                        time.sleep(delay)
                        attempt += 1
            except Exception as exc:  # Independent monthly products; retain failure and continue.
                failure = {"month": month, "status": "FAILED", "attempts": attempt + 1, "error_type": type(exc).__name__, "error_message": str(exc)[:500], "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
                failures.append(failure)
                log.write(json.dumps(failure, sort_keys=True, separators=(",", ":")) + "\n")
                log.flush()
    rows = [completed[m] for m in expected if m in completed]
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_parquet(ROWS_PATH, index=False, compression="zstd")
        frame.to_csv(CSV_PATH, index=False)
    output_hashes = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in (ROWS_PATH, CSV_PATH) if p.exists()}
    earlier_failures = [r for r in progress_history if r.get("status") == "FAILED"]
    recovered_prior_failures = sorted({str(r["month"]) for r in earlier_failures if str(r.get("month")) in completed})
    manifest = {
        "manifest_version": "1.0.0", "status": "ACQUIRED_WITH_GAPS" if failures or len(rows) != len(expected) else "ACQUIRED_AND_VALIDATED",
        "run_id": "chirps-v3-ecuador-bbox-2002-2024", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"product": "CHIRPS v3.0 final monthly", "official_landing_page": "https://www.chc.ucsb.edu/data/chirps3", "official_readme": "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/README-CHIRPSv3.0.txt", "granules": "monthly COGs accessed via byte-range HTTP", "source_content_sha256": "not computed for full remote objects; each regional output is SHA-256 hashed"},
        "geography": {"type": "pilot bounding box; not an Ecuador administrative polygon", "bbox_wsen": list(BBOX)},
        "period": {"start": expected[0], "end": expected[-1], "expected_months": len(expected), "acquired_months": len(rows), "missing_months": [m for m in expected if m not in completed]},
        "method": {"monthly_values": "CHIRPS monthly precipitation total, mm/month", "source_fill_value": FILL_VALUE, "validity": "finite, non-fill, nonnegative values and pixel center inside bbox", "aggregation": "cos(latitude)-weighted mean on native 0.05-degree EPSG:4326 grid; no reprojection", "spatial_coverage": "valid bbox pixel fraction recorded per month", "outputs": "per-month native-grid clipped GeoTIFF plus monthly summary Parquet/CSV"},
        "counts": {"regional_crop_files": len(rows), "failed_months": len(failures), "median_valid_pixel_fraction_bbox": float(pd.DataFrame(rows).valid_pixel_fraction_bbox.median()) if rows else None},
        "failures": failures, "retry_events": retry_events,
        "retry_history": {"prior_failed_records": len(earlier_failures), "prior_failure_months_recovered": recovered_prior_failures},
        "artifact_hashes": output_hashes,
        "independence_caveat": "CHIRPS final blends station observations; station input overlap with GHCN and regional providers means it is cross-product corroboration, not a fully independent gauge validation. CHIRPS v3 daily satellite disaggregation uses IMERG Late V07, but this analysis uses the native monthly product.",
        "execution": {"elapsed_seconds": time.perf_counter() - started},
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps({"status": manifest["status"], "acquired_months": len(rows), "expected_months": len(expected), "failed_months": len(failures), "missing_months": manifest["period"]["missing_months"], "recovered_retry_months": sorted({e["month"] for e in retry_events if e.get("status") == "RECOVERED"}), "median_valid_pixel_fraction_bbox": manifest["counts"]["median_valid_pixel_fraction_bbox"], "summary_parquet": ROWS_PATH.relative_to(ROOT).as_posix() if ROWS_PATH.exists() else None, "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(), "elapsed_seconds": manifest["execution"]["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
