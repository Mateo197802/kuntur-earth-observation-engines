#!/usr/bin/env python3
"""Audit preprocessing lineage and build the KUNTUR v1 Gold Parquet outputs.

The preprocessing worker's terminal summary has a historical row-count
formula that multiplies the per-month Gold variable count by the number of
source-month records. This finalizer validates the correct source-specific
variable grid, then builds Gold from the already validated manifest rows.
It does not reopen, redownload, or modify source and regional input granules.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuntur_data.preprocessing import (  # noqa: E402
    ACQUISITION_MANIFEST,
    PREPROCESSING_MANIFEST,
    REGION_ID,
    REGIONAL_MANIFEST,
    atomic_write_json,
    build_gold_outputs,
    contract_hash,
    latest_manifest_map,
    load_jsonl,
    month_range,
    processing_code_hashes,
    processing_config_hash,
    sha256_file,
)


def by_source_month(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    indexed = {}
    for row in rows:
        start = str(row.get("period_start") or "")
        source_id = str(row.get("source_id") or "")
        if source_id and len(start) >= 7:
            indexed[(source_id, start[:7])] = row
    return indexed


def expected_gold_rows(contract: dict[str, Any], month_count: int) -> int:
    variable_count_by_source = collections.Counter(spec["source_id"] for spec in contract["variables"])
    return month_count * sum(variable_count_by_source.values())


def audit_preprocessing(root: Path, verify_processed_files: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = json.loads((root / "configs/data_contract_v1.json").read_text(encoding="utf-8"))
    months = month_range(contract["coverage"]["start_month"], contract["coverage"]["end_month"])
    expected_sources = set(contract["datasets"])
    expected_pairs = {(source_id, month, REGION_ID) for source_id in expected_sources for month in months}

    acquisition = by_source_month(load_jsonl(root / ACQUISITION_MANIFEST))
    regional = by_source_month(load_jsonl(root / REGIONAL_MANIFEST))
    processing_path = root / PREPROCESSING_MANIFEST
    processing_rows = load_jsonl(processing_path)
    latest = latest_manifest_map(processing_rows, ("source_id", "month", "region_id"))

    missing = sorted(expected_pairs - set(latest))
    invalid = [key for key in sorted(expected_pairs & set(latest)) if latest[key].get("status") != "VALIDATED"]
    unexpected = sorted(set(latest) - expected_pairs)
    if missing or invalid or unexpected:
        raise ValueError(f"Preprocessing coverage mismatch: missing={len(missing)} invalid={len(invalid)} unexpected={len(unexpected)}")

    frozen_contract_sha = contract_hash(contract)
    config_shas = {str(row.get("preprocessing_config_sha256") or "") for row in latest.values()}
    recorded_code_hashes = {json.dumps(row.get("processing_code_hashes"), sort_keys=True) for row in latest.values()}
    expected_code_hashes = processing_code_hashes(root)
    expected_code_hashes_json = json.dumps(expected_code_hashes, sort_keys=True)
    current_config_sha = processing_config_hash(root, contract)
    if len(config_shas) != 1 or current_config_sha not in config_shas:
        raise ValueError("Preprocessing configuration hash is not consistent with the current frozen code and contract")
    if len(recorded_code_hashes) != 1 or expected_code_hashes_json not in recorded_code_hashes:
        raise ValueError("Preprocessing code hashes are not consistent with the current preprocessing module and entrypoint")

    expected_by_source = {
        source_id: {spec["variable_id"] for spec in contract["variables"] if spec["source_id"] == source_id}
        for source_id in expected_sources
    }
    gold_rows: list[dict[str, Any]] = []
    processed_bad: list[str] = []
    for key in sorted(expected_pairs):
        source_id, month, region_id = key
        record = latest[key]
        if record.get("contract_sha256") != frozen_contract_sha:
            raise ValueError(f"Contract hash mismatch in preprocessing row {key}")
        source_record = acquisition.get((source_id, month))
        regional_record = regional.get((source_id, month))
        if not source_record or source_record.get("status") != "VALIDATED":
            raise ValueError(f"No validated acquisition manifest entry for {source_id} {month}")
        if not regional_record or regional_record.get("status") != "VALIDATED":
            raise ValueError(f"No validated regional manifest entry for {source_id} {month}")
        if source_record.get("sha256") != record.get("source_sha256"):
            raise ValueError(f"Source checksum does not match acquisition manifest for {source_id} {month}")
        if regional_record.get("source_sha256") != record.get("source_sha256"):
            raise ValueError(f"Regional source checksum does not match acquisition manifest for {source_id} {month}")
        if regional_record.get("output_sha256") != record.get("regional_input_sha256"):
            raise ValueError(f"Regional input checksum does not match regional manifest for {source_id} {month}")
        if regional_record.get("output_path") != record.get("regional_input_path"):
            raise ValueError(f"Regional input path does not match regional manifest for {source_id} {month}")
        if int(regional_record.get("output_bytes") or 0) != int(record.get("regional_input_bytes") or 0):
            raise ValueError(f"Regional input size does not match regional manifest for {source_id} {month}")

        processed_path_text = str(record.get("processed_path") or "")
        processed_sha = str(record.get("processed_sha256") or "")
        if not processed_path_text or not re.fullmatch(r"[0-9a-f]{64}", processed_sha):
            raise ValueError(f"Processed output path or SHA-256 is malformed for {source_id} {month}")
        if verify_processed_files:
            processed_path = root / processed_path_text
            if not processed_path.is_file():
                processed_bad.append(processed_path_text)
            elif processed_path.stat().st_size != int(record.get("processed_bytes") or -1):
                processed_bad.append(processed_path_text)
            elif sha256_file(processed_path) != processed_sha:
                processed_bad.append(processed_path_text)

        source_rows = record.get("gold_rows")
        if not isinstance(source_rows, list) or {row.get("variable_id") for row in source_rows} != expected_by_source[source_id]:
            raise ValueError(f"Gold variables differ from contract for {source_id} {month}")
        for row in source_rows:
            if row.get("source_id") != source_id or row.get("month") != month:
                raise ValueError(f"Gold row key differs from preprocessing manifest key {key}")
            row["processed_native_grid_path"] = processed_path_text
            row["processed_native_grid_sha256"] = processed_sha
            gold_rows.append(row)

    if processed_bad:
        raise ValueError(f"Processed native-grid hash/size checks failed for {len(processed_bad)} files: {processed_bad[:5]}")

    expected_count = expected_gold_rows(contract, len(months))
    observed_keys = {(row["month"], row["variable_id"]) for row in gold_rows}
    expected_keys = {(month, spec["variable_id"]) for month in months for spec in contract["variables"]}
    if len(gold_rows) != expected_count or len(observed_keys) != expected_count or observed_keys != expected_keys:
        raise ValueError(f"Gold row grid mismatch: actual={len(gold_rows)} unique={len(observed_keys)} expected={expected_count}")

    summary = {
        "status": "PASSED",
        "contract_version": contract["contract_version"],
        "contract_sha256": frozen_contract_sha,
        "preprocessing_config_sha256": next(iter(config_shas)),
        "processing_code_hashes": expected_code_hashes,
        "acquisition_manifest_sha256": sha256_file(root / ACQUISITION_MANIFEST),
        "regional_manifest_sha256": sha256_file(root / REGIONAL_MANIFEST),
        "preprocessing_manifest_sha256": sha256_file(processing_path),
        "monthly_source_records": len(latest),
        "months_per_source": len(months),
        "source_counts": dict(collections.Counter(key[0] for key in latest)),
        "gold_rows": len(gold_rows),
        "gold_variables_per_month": len(contract["variables"]),
        "processed_native_grid_files_hash_verified": len(latest) if verify_processed_files else 0,
        "validation_checks": [
            "complete source-month coverage with no latest failed records",
            "acquisition, regional, and preprocessing checksums agree",
            "contract, preprocessing configuration, and code hashes agree",
            "source-specific Gold variables match the frozen contract",
            "all month-variable keys are unique and complete",
            "processed native-grid file checksums and sizes verified" if verify_processed_files else "processed native-grid checksums are present in manifest",
        ],
    }
    if verify_processed_files:
        summary["processed_native_grid_file_verification"] = {
            "status": "PASSED",
            "files_verified": len(latest),
            "verified_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "verification_root": str(root),
            "method": "actual file byte size and SHA-256 compared with preprocessing manifest",
        }
    return summary, gold_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="KUNTUR repository/workspace root")
    parser.add_argument("--verify-only", action="store_true", help="Audit manifests without creating Gold outputs")
    parser.add_argument("--verify-processed-files", action="store_true", help="Rehash all processed native-grid files; run on CEDIA")
    parser.add_argument("--verification-receipt", type=Path, help="Use a CEDIA processed-grid verification receipt when building Gold locally")
    args = parser.parse_args()
    root = args.root.resolve()
    summary, gold_rows = audit_preprocessing(root, verify_processed_files=args.verify_processed_files)

    if args.verification_receipt:
        receipt = json.loads(args.verification_receipt.read_text(encoding="utf-8"))
        verification = receipt.get("processed_native_grid_file_verification")
        expected_facts = {
            "acquisition_manifest_sha256": summary["acquisition_manifest_sha256"],
            "regional_manifest_sha256": summary["regional_manifest_sha256"],
            "preprocessing_manifest_sha256": summary["preprocessing_manifest_sha256"],
            "contract_sha256": summary["contract_sha256"],
            "preprocessing_config_sha256": summary["preprocessing_config_sha256"],
        }
        if not isinstance(verification, dict) or verification.get("status") != "PASSED":
            raise ValueError("Processed-grid verification receipt does not show PASSED")
        if int(verification.get("files_verified") or 0) != summary["monthly_source_records"]:
            raise ValueError("Processed-grid verification receipt has the wrong file count")
        if not str(verification.get("verification_root") or "").endswith("/proyecto-kuntur.y8myK4sW/repo"):
            raise ValueError("Processed-grid verification receipt does not identify the KUNTUR CEDIA workspace")
        for key, expected in expected_facts.items():
            if receipt.get(key) != expected:
                raise ValueError(f"Processed-grid verification receipt mismatch for {key}")
        summary["processed_native_grid_files_hash_verified"] = int(verification["files_verified"])
        summary["processed_native_grid_file_verification"] = verification
        summary["validation_checks"][-1] = "all processed native-grid file sizes and SHA-256 hashes verified on CEDIA"

    if not args.verify_only:
        contract = json.loads((root / "configs/data_contract_v1.json").read_text(encoding="utf-8"))
        result = build_gold_outputs(root, contract, gold_rows, summary["preprocessing_config_sha256"])
        finalizer_sha = sha256_file(Path(__file__).resolve())
        for manifest_key in ("gold_manifest", "split_manifest"):
            manifest_path = root / contract["outputs"][manifest_key]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["gold_finalizer_entrypoint_sha256"] = finalizer_sha
            manifest["manifest_audit"] = summary
            atomic_write_json(manifest_path, manifest)
        summary["outputs"] = result
        summary["gold_finalizer_entrypoint_sha256"] = finalizer_sha
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
