#!/usr/bin/env python
"""Guarded sequential transfer runner; dry-run unless explicitly enabled."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuntur_data.download.http import AuthenticationRequired, DownloadError, RateLimiter, download_file  # noqa: E402
from kuntur_data.validation.files import inspect_container  # noqa: E402


REQUIRED_AUTHORIZATION_FLAGS = (
    "cedia_general_research_access_user_attested",
    "academic_research_scope_user_attested",
    "private_workspace_verified",
    "storage_quota_verified",
    "earthdata_authentication_verified",
    "required_endpoint_connectivity_verified",
)


class TransferPlanError(ValueError):
    """Raised when a manifest row is not explicitly safe to transfer."""


def resolve_transfer_plan(
    row: dict[str, Any],
    pilot: dict[str, Any],
    source: dict[str, Any],
    remaining_budget: int,
) -> dict[str, Any]:
    """Select a validated subset result or explicitly approved full-object fallback."""
    use_subset = bool(
        row.get("planned_transfer_mode") == "SUBSET_RESULT_READY"
        and row.get("subset_request_status") == "RESPONSE_VALIDATED"
        and row.get("subset_request_hash")
        and row.get("subset_result_request_hash") == row.get("subset_request_hash")
        and row.get("validated_subset_download_url")
    )
    subset_available = bool(
        row.get("spatial_subsetting_advertised")
        or row.get("harmony_bbox_time_variable_subset_supported")
    )
    if pilot.get("require_spatial_subset_when_available") and subset_available and not use_subset:
        raise TransferPlanError("A submitted and validated subset result is required.")

    if use_subset:
        url = row["validated_subset_download_url"]
        expected_size = row.get("subset_validated_output_size_bytes")
        if type(expected_size) is not int or expected_size <= 0:
            raise TransferPlanError("A validated subset result needs a positive exact size.")
        allowed_hosts = set(source.get("allowed_hosts", [])) | {"harmony.earthdata.nasa.gov"}
    else:
        if (
            pilot.get("fallback_to_full_granules_only_if_within_budget") is not True
            or row.get("planned_transfer_mode") != "FULL_OBJECT_FALLBACK_APPROVED"
        ):
            raise TransferPlanError("An explicit in-budget full-object fallback plan is required.")
        url = row.get("source_url")
        expected_size = row.get("expected_full_granule_size_bytes")
        if type(expected_size) is not int or expected_size <= 0 or expected_size > remaining_budget:
            raise TransferPlanError("The full-object fallback size is unknown or exceeds the remaining budget.")
        allowed_hosts = set(source.get("allowed_hosts", []))

    if not url:
        raise TransferPlanError("The selected transfer route has no provider URL.")
    if expected_size > remaining_budget:
        raise TransferPlanError("The selected transfer exceeds the remaining budget.")
    return {
        "url": url,
        "expected_size_bytes": expected_size,
        "allowed_hosts": allowed_hosts,
        "transfer_mode": "SUBSET_RESULT_READY" if use_subset else "FULL_OBJECT_FALLBACK_APPROVED",
    }


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON mapping in {path}.")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at line {line_number}.") from exc
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="perform transfers; requires all config gates")
    args = parser.parse_args()
    acquisition = load_json(PROJECT_ROOT / "configs" / "acquisition.yaml")
    source_config = load_json(PROJECT_ROOT / "configs" / "sources.yaml")
    auth = acquisition.get("authorization", {})
    missing = [flag for flag in REQUIRED_AUTHORIZATION_FLAGS if auth.get(flag) is not True]
    if not args.execute:
        print("DRY RUN: no bytes will be requested.")
        print(f"execute_downloads_config={acquisition.get('execute_downloads')}")
        print(f"authorization_flags_not_verified={','.join(missing) if missing else 'none'}")
        return 0
    if acquisition.get("execute_downloads") is not True or missing:
        print("BLOCKED: download execution requires the CLI flag, execute_downloads=true, and every authorization flag verified.")
        print(f"authorization_flags_not_verified={','.join(missing) if missing else 'none'}")
        return 2

    manifest_path = PROJECT_ROOT / "data" / "manifests" / "acquisition_manifest.jsonl"
    records = load_jsonl(manifest_path)
    if any(item.get("status") == "BLOCKED" for item in records):
        print("BLOCKED: the current manifest contains BLOCKED rows. Re-run metadata planning after the preflight gates pass.")
        return 2
    pending = [item for item in records if item.get("status") == "PLANNED"]
    if not pending:
        print("No PLANNED transfer rows found; nothing to do.")
        return 0

    pilot = acquisition["pilot"]
    if pilot.get("max_concurrent_downloads") != 1:
        print("BLOCKED: this workflow requires max_concurrent_downloads=1.")
        return 2
    source_lookup = {item["source_id"]: item for item in source_config["sources"]}
    remaining_budget = int(pilot["budget_bytes"])
    limiter = RateLimiter(float(pilot.get("minimum_request_interval_seconds", 1.0)))
    for row in pending:
        source = source_lookup[row["source_id"]]
        try:
            transfer_plan = resolve_transfer_plan(row, pilot, source, remaining_budget)
        except TransferPlanError as error:
            print(f"BLOCKED: {error} Object={row.get('object_id')}.")
            return 2
        destination = PROJECT_ROOT / Path(row["local_path"])
        try:
            result = download_file(
                transfer_plan["url"],
                destination,
                allowed_hosts=transfer_plan["allowed_hosts"],
                expected_size_bytes=transfer_plan["expected_size_bytes"],
                max_bytes=remaining_budget,
                token=os.environ.get(str(pilot.get("earthdata_token_env", "EARTHDATA_TOKEN"))),
                timeout_seconds=float(pilot.get("timeout_seconds", 60)),
                max_retries=int(pilot.get("max_retries", 3)),
                rate_limiter=limiter,
            )
            row.update(
                {
                    "status": "DOWNLOADED",
                    "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
                    "actual_download_size_bytes": result["bytes"],
                    "sha256": result["sha256"],
                    "local_path": str(destination),
                    "errors": [],
                }
            )
            suffix = destination.suffix.casefold()
            format_hint = "HDF5" if suffix in {".h5", ".hdf5", ".nc4"} else "NetCDF"
            try:
                byte_check = inspect_container(destination, format_hint)
                row["validation"] = {"status": "BYTE_CONTAINER_CHECK_PASSED", "checks": [byte_check], "scientific_validation": "NOT_STARTED"}
            except (OSError, ValueError) as validation_error:
                row["validation"] = {"status": "BYTE_CONTAINER_CHECK_FAILED", "checks": [], "scientific_validation": "NOT_STARTED"}
                row["errors"] = [f"BYTE_CONTAINER_CHECK_FAILED:{type(validation_error).__name__}"]
            remaining_budget -= int(result["bytes"])
            save_jsonl(manifest_path, records)
            print(f"{row['status']} {row['object_id']} bytes={result['bytes']} sha256={result['sha256']}")
        except AuthenticationRequired:
            row["status"] = "BLOCKED"
            row["errors"] = ["AUTHENTICATION_REQUIRED; use only legitimate Earthdata access"]
            save_jsonl(manifest_path, records)
            print(f"BLOCKED authentication for {row.get('object_id')}; no alternate identity was tried.")
            return 3
        except (DownloadError, OSError) as error:
            row["status"] = "FAILED"
            row["errors"] = [f"{type(error).__name__}; see local operator report"]
            save_jsonl(manifest_path, records)
            print(f"FAILED {row.get('object_id')} ({type(error).__name__}); partial bytes, if any, were retained.")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
