#!/usr/bin/env python3
"""Independently validate local INAMHI PDF provenance and per-year outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = Path("data/manifests/external/inamhi_yearbooks_external_manifest_v1.json")
CHECKPOINT_MANIFEST = Path("data/manifests/external/inamhi_yearbook_checkpoints_v1.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, object]:
    source_path = root / SOURCE_MANIFEST
    checkpoint_path = root / CHECKPOINT_MANIFEST
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    checkpoint_manifest = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint_manifest.get("status") != "CHECKPOINTS_VALIDATED":
        raise ValueError("INAMHI checkpoint manifest is not marked CHECKPOINTS_VALIDATED")

    records: list[dict[str, object]] = []
    for year in range(2002, 2013):
        source = source_manifest.get("yearbook_sources", {}).get(str(year))
        checkpoint = checkpoint_manifest.get("years", {}).get(str(year))
        if not source or not checkpoint:
            raise ValueError(f"Missing source or checkpoint manifest record for {year}")
        if source.get("status") != "PDF_ACQUIRED_AND_PARSED" or checkpoint.get("status") != "VALIDATED":
            raise ValueError(f"Year {year} is not marked parsed and validated in both manifests")

        pdf = root / str(source["path"])
        parquet = root / str(checkpoint["path"])
        if not pdf.is_file() or sha256(pdf) != source.get("sha256"):
            raise ValueError(f"Year {year} PDF is missing or its SHA-256 differs from the source manifest")
        if not parquet.is_file() or sha256(parquet) != checkpoint.get("sha256"):
            raise ValueError(f"Year {year} checkpoint is missing or its SHA-256 differs from the checkpoint manifest")
        frame = pd.read_parquet(parquet)
        expected_rows = int(source.get("bbox_station_month_rows", -1))
        if len(frame) != expected_rows or len(frame) != int(checkpoint.get("rows", -2)):
            raise ValueError(f"Year {year} row count disagrees with one or both manifests")
        if set(frame["year"].astype(int)) != {year}:
            raise ValueError(f"Year {year} checkpoint contains another year")
        if set(frame["source_pdf_sha256"].astype(str)) != {source["sha256"]}:
            raise ValueError(f"Year {year} rows do not retain the source PDF hash")
        if frame.duplicated(["year", "station_id", "month_number"]).any():
            raise ValueError(f"Year {year} contains duplicate station-month records")
        if not frame["month_number"].between(1, 12).all():
            raise ValueError(f"Year {year} contains an invalid month number")
        station_count = int(frame["station_id"].nunique())
        if station_count != int(checkpoint.get("stations", -1)):
            raise ValueError(f"Year {year} station count disagrees with its checkpoint manifest")

        extraction = checkpoint.get("extraction", {})
        records.append({
            "year": year,
            "status": "INDEPENDENTLY_VALIDATED",
            "rows": len(frame),
            "stations": station_count,
            "pdf_pages_scanned": extraction.get("pdf_pages"),
            "station_table_pages": extraction.get("bbox_station_climatology_pages"),
            "source_pdf_sha256": source["sha256"],
            "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_bytes": parquet.stat().st_size,
            "checkpoint_mtime_utc": pd.Timestamp(parquet.stat().st_mtime, unit="s", tz="UTC").isoformat(),
        })

    return {
        "status": "INDEPENDENT_YEAR_CHECKS_PASSED",
        "years_validated": len(records),
        "station_month_rows": sum(int(record["rows"]) for record in records),
        "years": records,
        "source_manifest_sha256": sha256(source_path),
        "checkpoint_manifest_sha256": sha256(checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="Local KUNTUR repository root")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; otherwise print to stdout")
    args = parser.parse_args()
    result = validate(args.root.resolve())
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(output)
    print(serialized, end="")


if __name__ == "__main__":
    main()
