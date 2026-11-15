#!/usr/bin/env python3
"""Build KUNTUR's versioned native-grid Silver and regional monthly Gold outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kuntur_data.preprocessing import canonical_json, finalize_from_preprocessing_manifest, run  # noqa: E402


def write_dictionary(workspace: Path) -> None:
    contract_path = workspace / "configs" / "data_contract_v1.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    dictionary = {
        "dictionary_version": contract["contract_version"],
        "contract_sha256": hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest(),
        "region": contract["region"],
        "variables": contract["variables"],
    }
    json_path = workspace / contract["outputs"]["variable_dictionary_json"]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dictionary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    csv_path = workspace / contract["outputs"]["variable_dictionary_csv"]
    columns = [
        "variable_id", "dataset_id", "source_id", "source_variable_name", "name", "units",
        "temporal_statistic", "spatial_aggregation", "transform_id", "valid_range_screen",
        "screen_type", "scale_factor", "fill_value", "domain", "quality_policy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in contract["variables"]:
            prepared = {key: row.get(key) for key in columns}
            for key in ("valid_range_screen",):
                if prepared[key] is not None:
                    prepared[key] = json.dumps(prepared[key], separators=(",", ":"))
            writer.writerow(prepared)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=PROJECT_ROOT, help="Existing KUNTUR workspace root")
    parser.add_argument("--month", action="append", help="Process only this YYYY-MM; may be repeated (sample mode)")
    parser.add_argument("--source-id", action="append", help="Process only this source ID; may be repeated (sample mode)")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N source-month inputs (sample mode)")
    parser.add_argument("--preprocess-only", action="store_true", help="Build native-grid Silver files and the manifest, leaving Parquet finalization for a PyArrow-enabled workstation")
    parser.add_argument("--finalize-from-manifest", action="store_true", help="Build Parquet Gold and forecast outputs from the completed preprocessing manifest")
    parser.add_argument("--write-dictionary-only", action="store_true", help="Generate the frozen contract's JSON and CSV data dictionaries, then exit")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be nonnegative")
    if args.month:
        for month in args.month:
            if len(month) != 7 or month[4] != "-":
                parser.error(f"Invalid --month {month!r}; expected YYYY-MM")
    try:
        workspace = args.workspace.resolve()
        write_dictionary(workspace)
        if args.write_dictionary_only:
            print("DATA_DICTIONARY=WRITTEN", flush=True)
            return 0
        if args.preprocess_only and args.finalize_from_manifest:
            parser.error("--preprocess-only and --finalize-from-manifest are mutually exclusive")
        if args.finalize_from_manifest:
            result = finalize_from_preprocessing_manifest(workspace)
        else:
            result = run(
                workspace, limit=args.limit, source_ids=set(args.source_id or []), months=set(args.month or []),
                build_outputs=not args.preprocess_only,
            )
    except Exception as exc:
        print(f"PREPROCESSING_FAILED={type(exc).__name__}: {str(exc)[:600]}", file=sys.stderr, flush=True)
        return 2
    print("PREPROCESSING_RESULT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0 if not result.get("failures") else 2


if __name__ == "__main__":
    raise SystemExit(main())
