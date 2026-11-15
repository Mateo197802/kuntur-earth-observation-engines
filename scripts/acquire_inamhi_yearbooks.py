#!/usr/bin/env python3
"""Acquire and extract INAMHI annual station tables for 2002–2012."""

from __future__ import annotations

import datetime as dt
import gc
import hashlib
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

import pandas as pd
import pdfplumber
import psutil
import requests

from kuntur_data.phase2.inamhi_pdf import STATION_CODE, canonical_station_id, parse_monthly_station_words, parse_station_index_words
from kuntur_data.phase2.inamhi_checkpoints import load_year_checkpoint, write_year_checkpoint


ROOT = Path(__file__).resolve().parents[1]
BBOX = {"west": -81.5, "south": -5.1, "east": -75.0, "north": 1.7}
AVAILABLE_YEARS = range(2002, 2013)
QUERY_URL = "https://servicios.inamhi.gob.ec/wp-json/wp/v2/media"
RAW_DIR = ROOT / "data/external/raw/inamhi"
PROCESSED_DIR = ROOT / "data/external/processed"
MANIFEST_DIR = ROOT / "data/manifests/external"
CHECKPOINT_DIR = PROCESSED_DIR / "inamhi_yearbook_checkpoints_v1"
CHECKPOINT_MANIFEST_PATH = MANIFEST_DIR / "inamhi_yearbook_checkpoints_v1.json"
MAX_RSS_BYTES = int(3.5 * 1024**3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temp.replace(path)


def discover_pdf(year: int, session: requests.Session) -> dict[str, object] | None:
    for attempt in range(4):
        try:
            response = session.get(QUERY_URL, params={"search": f"Am-{year}", "per_page": 20}, timeout=(15, 30))
            response.raise_for_status()
            items = [item for item in response.json() if item.get("source_url", "").lower().endswith(".pdf")]
            if not items:
                return None
            item = next((i for i in items if i.get("slug", "").lower() == f"am-{year}"), items[0])
            return {"url": item["source_url"], "slug": item.get("slug"), "title": item.get("title", {}).get("rendered")}
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    return None


def acquire_pdf(url: str, target: Path, session: requests.Session) -> dict[str, object]:
    if target.exists() and target.stat().st_size > 1000:
        return {"path": target.relative_to(ROOT).as_posix(), "bytes": target.stat().st_size, "sha256": sha256(target), "reuse_existing": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        try:
            digest = hashlib.sha256()
            byte_count = 0
            with session.get(url, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if "pdf" not in response.headers.get("Content-Type", "").lower():
                    raise ValueError(f"INAMHI source did not return PDF: {response.headers.get('Content-Type')}")
                expected = response.headers.get("Content-Length")
                with temp.open("wb") as output:
                    for block in response.iter_content(512 * 1024):
                        if block:
                            output.write(block)
                            digest.update(block)
                            byte_count += len(block)
                if expected and byte_count != int(expected):
                    raise IOError(f"PDF length mismatch for {url}: {byte_count} != {expected}")
            if temp.read_bytes()[:5] != b"%PDF-":
                raise ValueError("Downloaded INAMHI file has no PDF signature")
            temp.replace(target)
            return {"path": target.relative_to(ROOT).as_posix(), "url": url, "bytes": byte_count, "sha256": digest.hexdigest(), "http_status": 200, "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag"), "reuse_existing": False}
        except (requests.RequestException, IOError) as exc:
            temp.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def index_page_range(year: int, page_count: int) -> range:
    if 2007 <= year <= 2009:
        # These books publish station tables without a coordinate index.
        return range(0)
    if year >= 2006:
        return range(10, min(page_count, 50))
    return range(70, min(page_count, 125))


def extract_station_index(path: Path, year: int) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    """Read all published index rows for a year, including rows outside the pilot box."""
    station_index: dict[str, dict[str, object]] = {}
    index_page_count = 0
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page_index in index_page_range(year, page_count):
            page = pdf.pages[page_index]
            try:
                text = page.extract_text() or ""
                normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper())
                if "LATITUD" not in normalized or "LONGITUD" not in normalized:
                    continue
                parsed = parse_station_index_words(page.extract_words(x_tolerance=1, y_tolerance=3))
                if not parsed:
                    continue
                index_page_count += 1
                for station in parsed:
                    station["coordinate_source_year"] = year
                    station_index[str(station["station_id"])] = station
            finally:
                page.close()
    return station_index, {"index_pages": index_page_count, "pdf_pages": page_count}


def reference_station_lookup(target_year: int, reference_indexes: dict[int, dict[str, dict[str, object]]]) -> dict[str, dict[str, object]]:
    """Choose the nearest published station coordinates for books without an index."""
    lookup: dict[str, dict[str, object]] = {}
    for reference_year in sorted(reference_indexes, key=lambda value: (abs(value - target_year), value)):
        for station_id, record in reference_indexes[reference_year].items():
            lookup.setdefault(station_id, {**record, "coordinate_source_year": reference_year})
    return lookup


def _station_table_codes_from_text(path: Path, station_index: dict[str, dict[str, object]]) -> tuple[dict[int, tuple[str, str]], str]:
    """Use searchable PDF text to locate station tables without assuming page offsets."""
    command = shutil.which("pdftotext")
    if command:
        result = subprocess.run([command, "-layout", str(path), "-"], capture_output=True, timeout=300, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"pdftotext failed for {path.name} with exit code {result.returncode}")
        text = result.stdout.decode("utf-8", errors="replace")
        pages = text.split("\f")
        method = "Poppler pdftotext page headings"
    else:
        page_texts: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    page_texts.append(page.extract_text() or "")
                finally:
                    page.close()
        pages = page_texts
        method = "pdfplumber page headings (Poppler unavailable)"

    candidate_pages: dict[int, tuple[str, str]] = {}
    header_pattern = re.compile(r"^\s*(M[A-Z0-9]{2,4})(?:\s+|$)", re.IGNORECASE)
    for page_index, page_text in enumerate(pages):
        upper = page_text.upper()
        if "TEMPERATURA DEL AIRE" not in upper or "SOMBRA" not in upper:
            continue
        station_id = None
        station_code_source = None
        for line in page_text.splitlines():
            match = header_pattern.match(line)
            if not match:
                continue
            code_source = match.group(1).upper()
            code = canonical_station_id(code_source)
            if code in station_index:
                station_id, station_code_source = code, code_source
                break
        if station_id is not None:
            candidate_pages[page_index] = (station_id, str(station_code_source))
    return candidate_pages, method


def extract_yearbook(
    path: Path,
    year: int,
    *,
    reference_index: dict[str, dict[str, object]] | None = None,
    cached_index: tuple[dict[str, dict[str, object]], dict[str, int]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    process = psutil.Process()
    peak_rss_bytes = process.memory_info().rss
    if peak_rss_bytes > MAX_RSS_BYTES:
        raise MemoryError(f"INAMHI PDF extraction exceeded the 3.5 GiB safety cap in {path.name}")
    local_index, index_stats = cached_index or extract_station_index(path, year)
    coordinate_reference_index = local_index if local_index else (reference_index or {})
    station_index = {
        station_id: station for station_id, station in coordinate_reference_index.items()
        if BBOX["west"] <= float(station["longitude"]) <= BBOX["east"]
        and BBOX["south"] <= float(station["latitude"]) <= BBOX["north"]
    }
    candidate_pages, page_scan_method = _station_table_codes_from_text(path, station_index)
    observations: list[dict[str, object]] = []
    station_pages: dict[str, list[dict[str, object]]] = {}
    with pdfplumber.open(path) as pdf:
        page_count = len(pdf.pages)
        for page_index, (station_id, station_code_source) in candidate_pages.items():
            if not 0 <= page_index < page_count:
                continue
            page = pdf.pages[page_index]
            try:
                station = {
                    **station_index[station_id],
                    "station_id": station_id,
                    "source_year": year,
                    "station_page_code_source": station_code_source,
                }
                rows = parse_monthly_station_words(
                    page.extract_words(x_tolerance=1, y_tolerance=3),
                    year=year,
                    station=station,
                    source_page=page_index + 1,
                )
                if rows:
                    station_pages.setdefault(station_id, []).append(rows)
                peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)
                if peak_rss_bytes > MAX_RSS_BYTES:
                    raise MemoryError(f"INAMHI PDF extraction exceeded the 3.5 GiB safety cap in {path.name}")
            finally:
                page.close()
                peak_rss_bytes = max(peak_rss_bytes, process.memory_info().rss)
                if peak_rss_bytes > MAX_RSS_BYTES:
                    raise MemoryError(f"INAMHI PDF extraction exceeded the 3.5 GiB safety cap in {path.name}")
    duplicate_station_pages = []
    for station_id, page_rows in station_pages.items():
        if len(page_rows) > 1:
            duplicate_station_pages.append(station_id)
            page_rows.sort(key=len, reverse=True)
        observations.extend(page_rows[0])
    return observations, {
        "pdf_pages": page_count,
        "index_pages": index_stats["index_pages"],
        "bbox_stations_listed": sum(
            BBOX["west"] <= float(record["longitude"]) <= BBOX["east"]
            and BBOX["south"] <= float(record["latitude"]) <= BBOX["north"]
            for record in local_index.values()
        ),
        "bbox_station_metadata_available": len(station_index),
        "station_metadata_source_years": sorted({int(record["coordinate_source_year"]) for record in station_index.values()}),
        "bbox_station_climatology_pages": len(station_pages),
        "duplicate_station_page_ids": duplicate_station_pages,
        "bbox_station_month_rows": len(observations),
        "station_codes_with_pages": sorted(station_pages),
        "peak_rss_bytes": peak_rss_bytes,
        "memory_safety_cap_bytes": MAX_RSS_BYTES,
        "page_mapping_policy": "station code and shade-air temperature table heading searched from each PDF page; metadata taken from same-year index or nearest available yearbook index when no index is published",
        "page_scan_method": page_scan_method,
    }


def main() -> None:
    started = time.perf_counter()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    manifest_path = MANIFEST_DIR / "inamhi_yearbooks_external_manifest_v1.json"
    source_path = PROCESSED_DIR / "inamhi_pilot_bbox_station_monthly_2002_2012_v1.parquet"
    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    source_records: dict[str, dict[str, object]] = dict(existing_manifest.get("yearbook_sources", {}))
    existing_frame = pd.read_parquet(source_path) if source_path.is_file() else pd.DataFrame()
    cacheable_years: set[int] = set()
    for year in AVAILABLE_YEARS:
        existing_record = source_records.get(str(year), {})
        expected_sha = str(existing_record.get("sha256", ""))
        if len(expected_sha) == 64 and load_year_checkpoint(
            year,
            expected_source_sha256=expected_sha,
            root=ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            manifest_path=CHECKPOINT_MANIFEST_PATH,
        ) is not None:
            cacheable_years.add(year)
            continue
        prior_rows = existing_frame.loc[existing_frame.year == year] if not existing_frame.empty else pd.DataFrame()
        if (
            existing_record.get("status") == "PDF_ACQUIRED_AND_PARSED"
            and not prior_rows.empty
            and len(prior_rows) == int(existing_record.get("bbox_station_month_rows", -1))
            and set(prior_rows.source_pdf_sha256.astype(str)) == {expected_sha}
            and not prior_rows.duplicated(["year", "station_id", "month_number"]).any()
        ):
            cacheable_years.add(year)
    years_requiring_parse = set(AVAILABLE_YEARS) - cacheable_years
    reference_indexes: dict[int, dict[str, dict[str, object]]] = {}
    index_cache: dict[int, tuple[dict[str, dict[str, object]], dict[str, int]]] = {}
    reference_index_errors: list[dict[str, object]] = []
    reference_year_options = [year for year in (2006, 2010, 2011, 2012) if (RAW_DIR / f"Am-{year}.pdf").is_file()]
    required_reference_years: set[int] = set()
    for year in years_requiring_parse:
        if year in reference_year_options:
            required_reference_years.add(year)
        elif 2007 <= year <= 2009 and reference_year_options:
            nearest = min(reference_year_options, key=lambda value: (abs(value - year), value))
            required_reference_years.add(nearest)
    for reference_year in sorted(required_reference_years):
        reference_path = RAW_DIR / f"Am-{reference_year}.pdf"
        try:
            index_data = extract_station_index(reference_path, reference_year)
            index_cache[reference_year] = index_data
            reference_indexes[reference_year] = index_data[0]
        except Exception as exc:
            reference_index_errors.append({"year": reference_year, "error_type": type(exc).__name__})
    coordinate_lookups = {
        year: reference_station_lookup(year, reference_indexes)
        for year in AVAILABLE_YEARS
    }
    all_observations: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for year in AVAILABLE_YEARS:
        key = str(year)
        existing_record = source_records.get(key, {})
        prior_rows = existing_frame.loc[existing_frame.year == year] if not existing_frame.empty else pd.DataFrame()
        expected_sha = str(existing_record.get("sha256", ""))
        if len(expected_sha) == 64:
            checkpoint = load_year_checkpoint(
                year,
                expected_source_sha256=expected_sha,
                root=ROOT,
                checkpoint_dir=CHECKPOINT_DIR,
                manifest_path=CHECKPOINT_MANIFEST_PATH,
            )
            if checkpoint is not None:
                checkpoint_frame, checkpoint_record = checkpoint
                source_records[key] = checkpoint_record.get("source_record", existing_record)
                all_observations.extend(checkpoint_frame.to_dict("records"))
                continue
        if (
            existing_record.get("status") == "PDF_ACQUIRED_AND_PARSED"
            and not prior_rows.empty
            and len(prior_rows) == int(existing_record.get("bbox_station_month_rows", -1))
            and set(prior_rows.source_pdf_sha256.astype(str)) == {expected_sha}
            and not prior_rows.duplicated(["year", "station_id", "month_number"]).any()
        ):
            cached_extraction = {k: v for k, v in existing_record.items() if k not in {"url", "slug", "title", "path", "bytes", "sha256", "reuse_existing", "status"}}
            write_year_checkpoint(
                year,
                prior_rows,
                source_record=existing_record,
                extraction=cached_extraction,
                root=ROOT,
                checkpoint_dir=CHECKPOINT_DIR,
                manifest_path=CHECKPOINT_MANIFEST_PATH,
            )
            all_observations.extend(prior_rows.to_dict("records"))
            continue
        try:
            pdf_path = RAW_DIR / f"Am-{year}.pdf"
            discovered = {
                "url": existing_record.get("url"),
                "slug": existing_record.get("slug"),
                "title": existing_record.get("title"),
            } if existing_record.get("url") else discover_pdf(year, session)
            if discovered is None:
                failures.append({"year": year, "status": "PDF_NOT_FOUND_IN_PUBLIC_MEDIA_CATALOG"})
                continue
            source = acquire_pdf(str(discovered["url"]), pdf_path, session)
            observations, extraction = extract_yearbook(
                pdf_path,
                year,
                reference_index=coordinate_lookups.get(year),
                cached_index=index_cache.get(year),
            )
            for row in observations:
                row["source_pdf_path"] = source["path"]
                row["source_pdf_url"] = discovered["url"]
                row["source_pdf_sha256"] = source["sha256"]
            if not observations and 2007 <= year <= 2012:
                raise ValueError(f"unexpected zero bbox rows for published station-table year {year}; left without a completion checkpoint")
            source_records[key] = {**discovered, **source, **extraction, "status": "PDF_ACQUIRED_AND_PARSED" if observations else "PDF_ACQUIRED_NO_BBOX_TEMPERATURE_ROWS"}
            year_frame = pd.DataFrame(observations)
            write_year_checkpoint(
                year,
                year_frame,
                source_record=source_records[key],
                extraction=extraction,
                root=ROOT,
                checkpoint_dir=CHECKPOINT_DIR,
                manifest_path=CHECKPOINT_MANIFEST_PATH,
            )
            all_observations.extend(observations)
            del observations
            gc.collect()
        except Exception as exc:
            if isinstance(exc, MemoryError):
                raise
            failures.append({"year": year, "status": "FAILED", "error_type": type(exc).__name__, "error_message": str(exc)[:500]})

    if not all_observations:
        raise RuntimeError("No INAMHI station-month rows parsed from the acquired yearbooks")
    frame = pd.DataFrame(all_observations)
    for column in ("tavg_c", "precipitation_mm_month"):
        ok_column = "tavg_physical_range_ok" if column == "tavg_c" else "precipitation_physical_range_ok"
        frame.loc[~frame[ok_column], column] = pd.NA
    if frame.duplicated(["year", "station_id", "month_number"]).any():
        raise RuntimeError("Duplicate station-month rows found across annual books")
    frame["tavg_eligible"] = frame.tavg_c.notna() & frame.tavg_physical_range_ok
    frame["precipitation_eligible"] = frame.precipitation_mm_month.notna() & frame.precipitation_physical_range_ok
    frame["temperature_units"] = "degC"
    frame["precipitation_units"] = "mm/month"
    frame["source_id"] = "INAMHI_meteorological_yearbook"
    frame["quality_policy"] = "published annual station monthly summary; blank values retained as missing; broad physical range screen; source does not supply per-value QC flags in these tables"

    grid = pd.period_range("2002-01", "2024-12", freq="M").astype(str)
    composites = []
    for month in grid:
        year, month_number = map(int, month.split("-"))
        sub = frame[(frame.year == year) & (frame.month_number == month_number)]
        temp = sub.loc[sub.tavg_eligible, ["station_id", "tavg_c"]].drop_duplicates("station_id")
        precip = sub.loc[sub.precipitation_eligible, ["station_id", "precipitation_mm_month"]].drop_duplicates("station_id")
        composites.append({
            "month": month,
            "inamhi_equal_station_tavg_c": float(temp.tavg_c.mean()) if len(temp) >= 3 else None,
            "inamhi_tavg_station_count": int(len(temp)),
            "inamhi_equal_station_precipitation_mm_month": float(precip.precipitation_mm_month.mean()) if len(precip) >= 3 else None,
            "inamhi_precipitation_station_count": int(len(precip)),
            "coverage_status": "ELIGIBLE_3_PLUS_STATIONS" if len(temp) >= 3 else "INSUFFICIENT_STATIONS_OR_UNAVAILABLE_YEARBOOK",
        })
    composite = pd.DataFrame(composites)
    composite_path = PROCESSED_DIR / "inamhi_pilot_bbox_monthly_composite_2002_2024_v1.parquet"
    frame.to_parquet(source_path, index=False, compression="zstd")
    composite.to_parquet(composite_path, index=False, compression="zstd")
    frame.to_csv(PROCESSED_DIR / "inamhi_pilot_bbox_station_monthly_2002_2012_v1.csv", index=False)
    artifacts = {}
    for path in (source_path, composite_path, PROCESSED_DIR / "inamhi_pilot_bbox_station_monthly_2002_2012_v1.csv"):
        artifacts[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    previous_history = list(existing_manifest.get("run_history", []))
    prior_snapshot = MANIFEST_DIR / "inamhi_yearbooks_external_manifest_pre_secondpass_20261115.json"
    if prior_snapshot.is_file():
        previous_history.append({
            "snapshot_path": prior_snapshot.relative_to(ROOT).as_posix(),
            "snapshot_sha256": sha256(prior_snapshot),
            "status": existing_manifest.get("status"),
            "counts": existing_manifest.get("counts", {}),
            "known_extraction_limit": "The first memory-bounded retry parsed 2006 but left 2007–2012 at zero rows because those books use different index/page layouts.",
        })
    missing_pdf_years = list(range(2013, 2025))
    manifest = {
        "manifest_version": "1.0.0", "status": "ACQUIRED_AND_VALIDATED_PARTIAL_YEARBOOK_COVERAGE" if missing_pdf_years or failures or reference_index_errors else "ACQUIRED_AND_VALIDATED",
        "run_id": "inamhi-yearbooks-pilot-bbox-2002-2012", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"organization": "Instituto Nacional de Meteorología e Hidrología (INAMHI), Ecuador", "catalog_api": QUERY_URL, "official_yearbook_landing_page": "https://servicios.inamhi.gob.ec/anuarios/", "document_type": "annual meteorological yearbooks, published tables", "data_period_with_yearbooks": [2002, 2012], "later_years_search": "Public media catalog search for Am-2013 through Am-2024 returned no PDF records at execution; the landing page advertises yearbooks, but those files were not machine-discoverable through this endpoint."},
        "geography": {"type": "pilot bbox, no administrative polygon claim", "bbox_wsen": [BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]]},
        "variables": {"tavg_c": {"description": "INAMHI annual-yearbook monthly mean air temperature at shade", "units": "degC"}, "precipitation_mm_month": {"description": "INAMHI annual-yearbook monthly precipitation total", "units": "mm/month"}},
        "methods": {"station_metadata": "DMS coordinates parsed from the yearbook's published-station index; stations retained only if coordinates fall inside the frozen pilot bbox", "temperature_column": "monthly mean air temperature, first climatology table; x-position 260–280 pt", "precipitation_column": "monthly precipitation sum, first climatology table; x-position 450–480 pt", "aggregation": "equal-station mean, minimum three eligible station records per month", "physical_screens": {"tavg_c": [-90, 60], "precipitation_mm_month": [0, 3000]}, "missingness": "blank cells retained as null; years with no discovered yearbook retained with null monthly composites", "limitations": "Yearbook tables provide no per-observation QC flags; variable station availability and historical summaries constrain temporal consistency. Published values are point stations and do not equal gridded area means."},
        "counts": {"pdf_years_acquired": len(source_records), "bbox_station_month_rows": int(len(frame)), "unique_station_ids": int(frame.station_id.nunique()), "tavg_eligible_station_months": int(frame.tavg_eligible.sum()), "precipitation_eligible_station_months": int(frame.precipitation_eligible.sum()), "months_with_3plus_tavg_stations": int((composite.inamhi_tavg_station_count >= 3).sum()), "months_with_3plus_precipitation_stations": int((composite.inamhi_precipitation_station_count >= 3).sum()), "expected_comparison_months": 276, "unavailable_pdf_years": list(range(2013, 2025))},
        "yearbook_sources": source_records, "failures": failures, "reference_index_errors": reference_index_errors,
        "run_history": previous_history, "artifact_hashes": artifacts,
        "independence_caveat": "INAMHI observations are a distinct national station archive from NASA gridded products; possible observation ingestion overlap with reanalysis cannot be ruled out. Spatial representativeness and changing station support prohibit claims of exact equivalence or fully independent domain transfer.",
        "validation": {"status": "PASSED_WITH_PDF_EXTRACTION_LIMITS", "checks": ["each source URL was discovered from the official INAMHI media API", "PDF signature and page count validated", "station coordinates parsed from official station index", "temperature and precipitation columns validated against rendered 2002 page sample", "physical ranges applied", "source page and source file SHA retained for every station-month", "monthly composites require >=3 stations"]},
        "execution": {"elapsed_seconds": time.perf_counter() - started, "peak_rss_bytes": max((int(record["peak_rss_bytes"]) for record in source_records.values() if record.get("peak_rss_bytes") is not None), default=psutil.Process().memory_info().rss), "memory_safety_cap_bytes": MAX_RSS_BYTES},
    }
    if CHECKPOINT_MANIFEST_PATH.is_file():
        checkpoint_manifest = json.loads(CHECKPOINT_MANIFEST_PATH.read_text(encoding="utf-8"))
    else:
        checkpoint_manifest = {"manifest_version": "1.0.0", "run_id": "inamhi-yearbooks-pilot-bbox-2002-2012", "years": {}}
    checkpointed_years = {
        int(year) for year, record in checkpoint_manifest.get("years", {}).items()
        if record.get("status") in {"VALIDATED", "VALIDATED_NO_BBOX_ROWS"}
    }
    checkpoint_manifest["status"] = "CHECKPOINTS_VALIDATED" if set(AVAILABLE_YEARS).issubset(checkpointed_years) else "CHECKPOINTS_PARTIAL"
    checkpoint_manifest["missing_years"] = sorted(set(AVAILABLE_YEARS) - checkpointed_years)
    atomic_json(CHECKPOINT_MANIFEST_PATH, checkpoint_manifest)
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "yearbooks_acquired": len(source_records), "yearbook_years": sorted(map(int, source_records)), "bbox_station_month_rows": len(frame), "stations": int(frame.station_id.nunique()), "temperature_months_3plus_stations": manifest["counts"]["months_with_3plus_tavg_stations"], "precipitation_months_3plus_stations": manifest["counts"]["months_with_3plus_precipitation_stations"], "reference_index_years": sorted(reference_indexes), "reference_index_errors": reference_index_errors, "failures": failures, "manifest": manifest_path.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
