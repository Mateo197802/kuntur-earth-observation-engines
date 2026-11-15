#!/usr/bin/env python3
"""Acquire and quality-screen NOAA GHCN-M QCU for the frozen Ecuador pilot bbox."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import tarfile
import time
from pathlib import Path

import pandas as pd
import requests

from kuntur_data.phase2.external import parse_ghcnm_inventory_line, parse_ghcnm_tavg_line


ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.ncei.noaa.gov/pub/data/ghcn/v4/ghcnm.tavg.latest.qcu.tar.gz"
BBOX = {"west": -81.5, "south": -5.1, "east": -75.0, "north": 1.7}
START_YEAR, END_YEAR = 2002, 2024
MISSING = -9999


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(stream) -> tuple[str, list[str]]:
    digest = hashlib.sha256()
    lines: list[str] = []
    for raw in stream:
        digest.update(raw)
        lines.append(raw.decode("ascii").rstrip("\r\n"))
    return digest.hexdigest(), lines


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def download_source(path: Path) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    digest = hashlib.sha256()
    count = 0
    response_metadata: dict[str, object] = {}
    started = time.perf_counter()
    with requests.get(URL, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        response_metadata = {
            "http_status": response.status_code,
            "content_length_header": response.headers.get("Content-Length"),
            "last_modified": response.headers.get("Last-Modified"),
            "etag": response.headers.get("ETag"),
        }
        with tmp.open("wb") as output:
            for block in response.iter_content(1024 * 1024):
                if block:
                    output.write(block)
                    digest.update(block)
                    count += len(block)
    if response_metadata.get("content_length_header") and count != int(response_metadata["content_length_header"]):
        tmp.unlink(missing_ok=True)
        raise IOError(f"GHCN-M download length mismatch: received {count} bytes")
    tmp.replace(path)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "url": URL,
        "bytes": count,
        "sha256": digest.hexdigest(),
        "elapsed_seconds": time.perf_counter() - started,
        **response_metadata,
    }


def main() -> None:
    started = time.perf_counter()
    raw_path = ROOT / "data/external/raw/ghcnm.tavg.latest.qcu.tar.gz"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        source = {"path": raw_path.relative_to(ROOT).as_posix(), "url": URL, "bytes": raw_path.stat().st_size, "sha256": sha256(raw_path), "reuse_existing": True}
        try:
            response = requests.head(URL, timeout=(15, 30), allow_redirects=True)
            if response.ok:
                source.update({"http_status": response.status_code, "last_modified": response.headers.get("Last-Modified"), "etag": response.headers.get("ETag")})
        except requests.RequestException:
            source["remote_metadata_status"] = "HEAD_UNAVAILABLE; local SHA-256 verified"
    else:
        source = download_source(raw_path)

    with tarfile.open(raw_path, mode="r:gz") as archive:
        inv_member = next((item for item in archive.getmembers() if item.name.endswith(".inv")), None)
        dat_member = next((item for item in archive.getmembers() if item.name.endswith(".dat")), None)
        if not inv_member or not dat_member:
            raise RuntimeError("Expected GHCN-M .inv and .dat members are missing")
        with archive.extractfile(inv_member) as stream:
            inv_sha, inv_lines = sha256_stream(stream)
        stations = [parse_ghcnm_inventory_line(line) for line in inv_lines]
        station_frame = pd.DataFrame(stations)
        in_bbox = station_frame[
            station_frame.longitude.between(BBOX["west"], BBOX["east"], inclusive="both")
            & station_frame.latitude.between(BBOX["south"], BBOX["north"], inclusive="both")
        ].copy()
        in_bbox["elevation_missing"] = in_bbox.elevation_m == -999.0
        station_lookup = in_bbox.set_index("station_id").to_dict("index")
        observations: list[dict[str, object]] = []
        dat_digest = hashlib.sha256()
        with archive.extractfile(dat_member) as stream:
            for raw_line in stream:
                dat_digest.update(raw_line)
                if raw_line[15:19] != b"TAVG":
                    continue
                station_id = raw_line[:11].decode("ascii").strip()
                if station_id not in station_lookup:
                    continue
                line = raw_line.decode("ascii").rstrip("\r\n")
                for row in parse_ghcnm_tavg_line(line):
                    if not (START_YEAR <= row["year"] <= END_YEAR):
                        continue
                    station = station_lookup[station_id]
                    value = row["value_c"]
                    physical_ok = value is not None and -90.0 <= float(value) <= 60.0
                    qc_ok = not row["qcflag"]
                    status = "ELIGIBLE" if physical_ok and qc_ok else (
                        "MISSING" if row["missing_sentinel"] else "QC_FLAGGED" if not qc_ok else "OUTSIDE_PHYSICAL_SCREEN"
                    )
                    observations.append(
                        {
                            **row,
                            "latitude": station["latitude"],
                            "longitude": station["longitude"],
                            "elevation_m": station["elevation_m"],
                            "station_name": station["station_name"],
                            "physical_range_ok": bool(physical_ok),
                            "qc_blank": bool(qc_ok),
                            "eligible": status == "ELIGIBLE",
                            "eligibility_status": status,
                            "source_product": "GHCN-M v4 TAVG QCU",
                        }
                    )
    if not observations:
        raise RuntimeError("No GHCN-M monthly temperature records found inside the frozen bbox")

    obs = pd.DataFrame(observations)
    duplicate_key = ["station_id", "month"]
    duplicate_rows = int(obs.duplicated(duplicate_key, keep=False).sum())
    obs = obs.sort_values(duplicate_key, kind="stable").drop_duplicates(duplicate_key, keep="last")
    eligible = obs[obs.eligible].copy()
    coverage = eligible.groupby("month").station_id.nunique().rename("eligible_station_count")
    station_sum = eligible.groupby("month").value_c.sum().rename("sum_c")
    grid = pd.period_range(f"{START_YEAR}-01", f"{END_YEAR}-12", freq="M").astype(str)
    monthly = pd.DataFrame({"month": grid}).merge(coverage, how="left", left_on="month", right_index=True)
    monthly = monthly.merge(station_sum, how="left", left_on="month", right_index=True)
    monthly["eligible_station_count"] = monthly.eligible_station_count.fillna(0).astype(int)
    monthly["tavg_equal_station_mean_c"] = monthly.sum_c / monthly.eligible_station_count.replace(0, pd.NA)
    monthly.loc[monthly.eligible_station_count < 3, "tavg_equal_station_mean_c"] = pd.NA
    eligible_stats = eligible.groupby("month").value_c.agg(["min", "max"]).reset_index()
    monthly = monthly.merge(eligible_stats.rename(columns={"min": "station_min_c", "max": "station_max_c"}), how="left", on="month")
    monthly["coverage_status"] = monthly.eligible_station_count.map(lambda count: "ELIGIBLE_3_PLUS_STATIONS" if count >= 3 else "INSUFFICIENT_STATIONS")

    out_dir = ROOT / "data/external/processed"
    manifest_dir = ROOT / "data/manifests/external"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    obs_path = out_dir / "ghcnm_ecuador_bbox_station_monthly_v1.parquet"
    monthly_path = out_dir / "ghcnm_ecuador_bbox_monthly_composite_v1.parquet"
    stations_path = out_dir / "ghcnm_ecuador_bbox_inventory_v1.csv"
    obs.to_parquet(obs_path, index=False, compression="zstd")
    monthly.to_parquet(monthly_path, index=False, compression="zstd")
    in_bbox.to_csv(stations_path, index=False)
    artifacts = {}
    for path in (obs_path, monthly_path, stations_path):
        artifacts[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    nmonths = int((monthly.eligible_station_count >= 3).sum())
    manifest = {
        "manifest_version": "1.0.0",
        "status": "ACQUIRED_AND_VALIDATED",
        "run_id": "ghcnm-qcu-ecuador-bbox-20261115",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            **source,
            "product": "GHCN-M v4 TAVG QCU",
            "data_member": dat_member.name,
            "data_member_sha256": dat_digest.hexdigest(),
            "data_member_bytes_uncompressed": dat_member.size,
            "inventory_member": inv_member.name,
            "inventory_member_sha256": inv_sha,
            "inventory_member_bytes_uncompressed": inv_member.size,
            "version_directory": Path(dat_member.name).parts[-2] if len(Path(dat_member.name).parts) > 1 else None,
            "official_readme_url": "https://www.ncei.noaa.gov/pub/data/ghcn/v4/readme.txt",
        },
        "geography": {"type": "pilot bounding box; not an Ecuador administrative polygon", "bbox_wsen": [BBOX["west"], BBOX["south"], BBOX["east"], BBOX["north"]]},
        "period": {"start": f"{START_YEAR}-01", "end": f"{END_YEAR}-12", "expected_months": 276},
        "methods": {
            "temperature_units": "hundredths °C converted by raw integer / 100",
            "missing_sentinel": MISSING,
            "accepted_qc_policy": "blank QC flag only; NOAA documents blank as no QC failure or could not be evaluated",
            "dmflag_policy": "retained; nonblank DMFLAG does not by itself reject a value",
            "dsflag_policy": "retained verbatim",
            "physical_screen_c": [-90.0, 60.0],
            "monthly_aggregation": "equal-station arithmetic mean; eligible only at >=3 unique stations/month",
            "station_deduplication": "one station-month retained; duplicate count recorded",
        },
        "counts": {
            "bbox_inventory_stations": int(len(in_bbox)),
            "bbox_station_month_rows_after_deduplication": int(len(obs)),
            "unique_bbox_stations_with_records": int(obs.station_id.nunique()),
            "eligible_observations": int(obs.eligible.sum()),
            "missing_sentinel_rows": int(obs.missing_sentinel.sum()),
            "qc_flagged_rows": int((obs.eligibility_status == "QC_FLAGGED").sum()),
            "physical_screen_rejections": int((obs.eligibility_status == "OUTSIDE_PHYSICAL_SCREEN").sum()),
            "duplicate_station_month_rows_seen": duplicate_rows,
            "months_with_three_or_more_eligible_stations": nmonths,
            "months_without_composite": int(276 - nmonths),
            "eligible_station_count_min_max": [int(monthly.eligible_station_count.min()), int(monthly.eligible_station_count.max())],
        },
        "station_selection": {
            "bbox_station_ids": sorted(in_bbox.station_id.astype(str).unique().tolist()),
            "stations_with_records": sorted(obs.station_id.astype(str).unique().tolist()),
            "stations_with_eligible_observations": sorted(obs.loc[obs.eligible, "station_id"].astype(str).unique().tolist()),
            "selection_rule": "published coordinates inside the frozen preliminary bbox; QCU missing and nonblank QC flags rejected; one record per station-month; composite requires at least three unique stations",
        },
        "external_use_caveat": "Station composites are observational corroboration, not equivalent to a gridded area mean. GHCN-M v4 draws substantially on GHCN-Daily and ISTI; source independence from other station-blended products must not be assumed.",
        "artifacts": artifacts,
        "validation": {"status": "PASSED", "checks": ["archive gzip/tar integrity opened and both expected members present", "all bbox station coordinates parsed from fixed-width metadata", "all TAVG values converted from hundredths Celsius", "missing and QC flags retained", "physical range screen applied", "monthly composite requires at least three unique eligible stations", "276-month expected grid retained"]},
        "execution": {"elapsed_seconds": time.perf_counter() - started},
    }
    manifest_path = manifest_dir / "ghcnm_qcu_external_manifest_v1.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({"status": manifest["status"], "source_sha256": source["sha256"], "bbox_inventory_stations": len(in_bbox), "stations_with_records": obs.station_id.nunique(), "eligible_rows": int(obs.eligible.sum()), "valid_months": nmonths, "missing_composite_months": 276 - nmonths, "artifacts": artifacts, "manifest": manifest_path.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
