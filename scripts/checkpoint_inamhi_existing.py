#!/usr/bin/env python3
"""Validate and atomically checkpoint existing INAMHI yearbook outputs without reparsing PDFs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kuntur_data.phase2.inamhi_checkpoints import write_year_checkpoint, atomic_json


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/processed/inamhi_pilot_bbox_station_monthly_2002_2012_v1.parquet"
MANIFEST = ROOT / "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json"
CHECKPOINT_DIR = ROOT / "data/external/processed/inamhi_yearbook_checkpoints_v1"
CHECKPOINT_MANIFEST = ROOT / "data/manifests/external/inamhi_yearbook_checkpoints_v1.json"
YEARS = range(2002, 2013)


def main() -> None:
    if not SOURCE.is_file() or not MANIFEST.is_file():
        raise SystemExit("Existing INAMHI source Parquet and manifest are required")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frame = pd.read_parquet(SOURCE)
    if frame.duplicated(["year", "station_id", "month_number"]).any():
        raise SystemExit("Existing INAMHI station-month output contains duplicate records")
    results: dict[str, int] = {}
    missing: list[int] = []
    for year in YEARS:
        key = str(year)
        source_record = manifest.get("yearbook_sources", {}).get(key)
        if not source_record or source_record.get("status") not in {"PDF_ACQUIRED_AND_PARSED", "PDF_ACQUIRED_NO_BBOX_TEMPERATURE_ROWS"}:
            missing.append(year)
            continue
        year_frame = frame.loc[frame.year == year].copy()
        expected_rows = int(source_record.get("bbox_station_month_rows", -1))
        if len(year_frame) != expected_rows:
            raise SystemExit(f"Year {year} Parquet rows do not match the acquisition manifest ({len(year_frame)} != {expected_rows})")
        expected_sha = str(source_record.get("sha256", ""))
        if len(year_frame) and set(year_frame.source_pdf_sha256.astype(str)) != {expected_sha}:
            raise SystemExit(f"Year {year} source PDF SHA does not match station-month provenance")
        if len(year_frame):
            if not year_frame.month_number.between(1, 12).all():
                raise SystemExit(f"Year {year} has an invalid month number")
            if source_record.get("status") != "PDF_ACQUIRED_AND_PARSED":
                raise SystemExit(f"Year {year} contains rows but manifest status is not parsed")
        elif source_record.get("status") == "PDF_ACQUIRED_AND_PARSED":
            raise SystemExit(f"Year {year} is marked parsed but has no station-month rows")
        extraction = {k: v for k, v in source_record.items() if k not in {"url", "slug", "title", "path", "bytes", "sha256", "reuse_existing", "status"}}
        record = write_year_checkpoint(
            year,
            year_frame,
            source_record=source_record,
            extraction=extraction,
            root=ROOT,
            checkpoint_dir=CHECKPOINT_DIR,
            manifest_path=CHECKPOINT_MANIFEST,
        )
        results[key] = int(record["rows"])
    if CHECKPOINT_MANIFEST.is_file():
        checkpoint_manifest = json.loads(CHECKPOINT_MANIFEST.read_text(encoding="utf-8"))
    else:
        checkpoint_manifest = {"manifest_version": "1.0.0", "run_id": "inamhi-yearbooks-pilot-bbox-2002-2012", "years": {}}
    missing = sorted(set(missing) | (set(YEARS) - {int(year) for year in checkpoint_manifest.get("years", {})}))
    checkpoint_manifest["status"] = "CHECKPOINTS_VALIDATED" if not missing else "CHECKPOINTS_PARTIAL"
    checkpoint_manifest["missing_years"] = missing
    atomic_json(CHECKPOINT_MANIFEST, checkpoint_manifest)
    print(json.dumps({"status": checkpoint_manifest["status"], "year_rows": results, "missing_years": missing, "manifest": CHECKPOINT_MANIFEST.relative_to(ROOT).as_posix()}, indent=2))


if __name__ == "__main__":
    main()
