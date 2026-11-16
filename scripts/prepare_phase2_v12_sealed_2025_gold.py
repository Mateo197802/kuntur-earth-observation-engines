#!/usr/bin/env python3
"""Create isolated, resumable Ecuador MERRA-2 Gold rows for the frozen 2025 test."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuntur_data import preprocessing as prep  # noqa: E402


SOURCE_ID = "merra2_monthly_slv"
MONTHS = [str(p) for p in __import__("pandas").period_range("2025-01", "2025-12", freq="M")]
OUTPUT = ROOT / "data/gold/phase2_v12_sealed_2025_merra2_bbox.parquet"
MANIFEST = ROOT / "data/manifests/phase2_v12_sealed_2025_gold_manifest.json"
ACQUISITION_MANIFEST = ROOT / "data/manifests/acquisition_manifest.jsonl"
REGIONAL_MANIFEST = ROOT / "data/manifests/regional_extraction_manifest.jsonl"
CONTRACT_PATH = ROOT / "configs/data_contract_v1.json"


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _sha(path: Path) -> str:
    return prep.sha256_file(path)


def _latest_month(rows: list[dict], id_key: str, month_key: str) -> dict[tuple[str, str], dict]:
    output = {}
    for row in rows:
        source_id = str(row.get(id_key) or "")
        month = prep.month_key(row.get(month_key))
        if source_id and month:
            output[(source_id, month)] = row
    return output


def _manifest_path(value: str | None, default: Path) -> Path:
    supplied = Path(value) if value else default
    path = supplied if supplied.is_absolute() else ROOT / supplied
    path = path.resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("Receipt manifests must remain inside the KUNTUR repository.")
    if not path.is_file():
        raise FileNotFoundError(f"Receipt manifest is missing: {path.relative_to(ROOT)}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition-manifest", help="Validated source receipt JSONL; defaults to the full acquisition manifest")
    parser.add_argument("--regional-manifest", help="Validated regional receipt JSONL; defaults to the full regional manifest")
    args = parser.parse_args()
    acquisition_manifest = _manifest_path(args.acquisition_manifest, ACQUISITION_MANIFEST)
    regional_manifest = _manifest_path(args.regional_manifest, REGIONAL_MANIFEST)
    # process_one records these provenance paths in every processed Silver artifact.
    prep.ACQUISITION_MANIFEST = acquisition_manifest.relative_to(ROOT)
    prep.REGIONAL_MANIFEST = regional_manifest.relative_to(ROOT)

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    config_sha = prep.processing_config_hash(ROOT, contract)
    acquisition_rows = prep.load_jsonl(acquisition_manifest)
    regional_rows = prep.load_jsonl(regional_manifest)
    acquisition = _latest_month(acquisition_rows, "source_id", "period_start")
    regional = _latest_month(regional_rows, "source_id", "period_start")
    prior = {}
    if MANIFEST.is_file():
        prior = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if prior.get("status") == "VALIDATED" and prior.get("months") == MONTHS and OUTPUT.is_file() and _sha(OUTPUT) == prior.get("output_sha256"):
        print(json.dumps({"status": "REUSED_VALIDATED_OUTPUT", "path": OUTPUT.relative_to(ROOT).as_posix(), "sha256": _sha(OUTPUT), "months": len(MONTHS)}, indent=2))
        return

    records = prior.get("records", {}) if prior.get("contract_sha256") == prep.contract_hash(contract) else {}
    gold_rows = []
    source_receipts = []
    for month in MONTHS:
        key = (SOURCE_ID, month)
        if key not in acquisition or key not in regional:
            raise ValueError(f"Missing validated acquisition/regional manifest pair for {month}")
        acquisition_row = acquisition[key]
        regional_row = regional[key]
        source_sha = str(acquisition_row.get("sha256") or "")
        regional_sha = str(regional_row.get("output_sha256") or "")
        existing = records.get(month)
        if existing and existing.get("source_sha256") == source_sha and existing.get("regional_output_sha256") == regional_sha:
            gold_rows.extend(existing["gold_rows"])
            source_receipts.append(existing)
            continue
        # Avoid replacing an existing processed artifact whose provenance is
        # not already represented in this isolated 2025 handoff manifest.
        processed_path = ROOT / "data/silver/processed_native_grid/v1" / SOURCE_ID / month[:4] / f"{month.replace('-', '')}.npz"
        if processed_path.exists():
            raise ValueError(f"Existing processed artifact lacks a matching handoff checkpoint: {processed_path.relative_to(ROOT)}")
        processed_record, month_gold = prep.process_one(
            ROOT, contract, acquisition, regional, SOURCE_ID, month, config_sha,
        )
        record = {
            "month": month,
            "source_sha256": source_sha,
            "regional_output_path": regional_row.get("output_path"),
            "regional_output_sha256": regional_sha,
            "processed_path": processed_record["processed_path"],
            "processed_sha256": processed_record["processed_sha256"],
            "processing_status": processed_record["status"],
            "gold_rows": month_gold,
        }
        records[month] = record
        gold_rows.extend(month_gold)
        source_receipts.append(record)
        _atomic_json(MANIFEST, {
            "manifest_version": "kuntur-phase2-v12-sealed-2025-gold-0.1.0",
            "status": "PROCESSING", "region_id": "ecuador_pilot_bbox",
            "source_id": SOURCE_ID, "dataset_id": "M2TMNXSLV.5.12.4",
            "months": MONTHS, "contract_sha256": prep.contract_hash(contract),
            "processing_config_sha256": config_sha, "records": records,
        })
        print(f"MONTH_VALIDATED={month} GOLD_ROWS={len(month_gold)}", flush=True)

    gold_rows.sort(key=lambda row: (row["month"], row["variable_id"]))
    expected = 12 * len(prep.source_variable_specs(contract, SOURCE_ID))
    if len(gold_rows) != expected or {row["month"] for row in gold_rows} != set(MONTHS):
        raise ValueError(f"Expected {expected} Gold rows covering twelve 2025 months; got {len(gold_rows)}")
    rows, output_sha, output_bytes = prep.write_parquet(OUTPUT, gold_rows, {
        "kuntur.contract_sha256": prep.contract_hash(contract),
        "kuntur.processing_config_sha256": config_sha,
        "kuntur.source_collection": "M2TMNXSLV.5.12.4",
        "kuntur.region_id": "ecuador_pilot_bbox",
        "kuntur.temporal_coverage": "2025-01/2025-12",
        "kuntur.sealed_test": "true",
    })
    result = {
        "manifest_version": "kuntur-phase2-v12-sealed-2025-gold-0.1.0",
        "status": "VALIDATED", "scope": "Separate 2025 MERRA-2 Ecuador bbox evaluation input; does not modify v1.0 Gold.",
        "region_id": "ecuador_pilot_bbox", "source_id": SOURCE_ID,
        "dataset_id": "M2TMNXSLV.5.12.4", "months": MONTHS,
        "contract_sha256": prep.contract_hash(contract), "processing_config_sha256": config_sha,
        "source_receipt_manifests": {
            "acquisition": {"path": acquisition_manifest.relative_to(ROOT).as_posix(), "sha256": _sha(acquisition_manifest)},
            "regional": {"path": regional_manifest.relative_to(ROOT).as_posix(), "sha256": _sha(regional_manifest)},
        },
        "row_count": rows, "variable_ids": sorted({row["variable_id"] for row in gold_rows}),
        "output_path": OUTPUT.relative_to(ROOT).as_posix(), "output_sha256": output_sha, "output_bytes": output_bytes,
        "source_receipts": [{key: value for key, value in row.items() if key != "gold_rows"} for row in source_receipts],
        "quality_summary": {
            variable_id: {
                "min_valid_area_fraction_bbox": float(min(row["valid_area_fraction_bbox"] for row in gold_rows if row["variable_id"] == variable_id)),
                "max_valid_area_fraction_bbox": float(max(row["valid_area_fraction_bbox"] for row in gold_rows if row["variable_id"] == variable_id)),
                "valid_months": int(sum(row["value"] is not None and row["valid_area_fraction_bbox"] >= 0.8 for row in gold_rows if row["variable_id"] == variable_id)),
            }
            for variable_id in sorted({row["variable_id"] for row in gold_rows})
        },
        "validation_checks": ["all twelve validated granules independently matched acquisition and regional SHA-256 records", "monthly temporal bounds matched exact calendar month", "source units and MERRA-2 physical plausibility screens passed", "native grid coordinates retained with area-weighted bbox summaries", "no regridding or interpolation", "Parquet reopened and row count verified"],
    }
    _atomic_json(MANIFEST, result)
    print(json.dumps({"status": result["status"], "path": result["output_path"], "months": len(MONTHS), "rows": rows, "sha256": output_sha, "bytes": output_bytes, "quality_summary": result["quality_summary"]}, indent=2))


if __name__ == "__main__":
    main()
