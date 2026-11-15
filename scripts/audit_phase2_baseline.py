#!/usr/bin/env python3
"""Audit immutable KUNTUR v1 artifacts before Phase 2 experiments."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/manifests/phase2_baseline_audit_v1.json"
EXPECTED = {
    "configs/data_contract_v1.json": "0769d821ca09b248f2eae671f1397ba4d935c3962eb8490abd3ce08f66556b8a",
    "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet": "5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2",
    "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet": "67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff",
    "data/gold/kuntur_forecast_split_indices_v1.parquet": "c2fbca4ac9636622a08e778913e2edfb23cc210453bf05114e1b3775b9b0726c",
    "data/manifests/acquisition_manifest.jsonl": "76909ed69b9101a344d0e481d0298ae9132e95586a72c27e978f781ecb9df8bd",
    "data/manifests/regional_extraction_manifest.jsonl": "bafd0aefcc89aa940075c34ff7f89eb406feb627d71fb0b670aae623515cc8e8",
    "data/manifests/preprocessing_manifest_v1.jsonl": "a74491fa2351c6680dc5aac9d873cf33689eb0aa1c32dbd65f288620e8e5cd3c",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def month_of(row: dict) -> str | None:
    for key in ("month", "period_start", "time_start", "start_time", "target_month"):
        value = row.get(key)
        if value:
            return str(value)[:7]
    return None


def main() -> None:
    files = {}
    for rel, expected_hash in EXPECTED.items():
        path = ROOT / rel
        actual = sha256(path)
        if actual != expected_hash:
            raise SystemExit(f"Immutable baseline hash mismatch: {rel}")
        files[rel] = {"sha256": actual, "bytes": path.stat().st_size, "baseline_match": True}

    contract = json.loads((ROOT / "configs/data_contract_v1.json").read_text(encoding="utf-8"))
    acquisition = read_jsonl(ROOT / "data/manifests/acquisition_manifest.jsonl")
    regional = read_jsonl(ROOT / "data/manifests/regional_extraction_manifest.jsonl")
    preprocessing = read_jsonl(ROOT / "data/manifests/preprocessing_manifest_v1.jsonl")
    gold_path = ROOT / contract["outputs"]["gold_parquet"]
    forecast_path = ROOT / contract["outputs"]["forecast_parquet"]
    split_path = ROOT / contract["outputs"]["split_indices_parquet"]
    gold = pd.read_parquet(gold_path)
    forecast = pd.read_parquet(forecast_path)
    split = pd.read_parquet(split_path)

    expected_vars = {v["variable_id"] for v in contract["variables"]}
    months = sorted(gold["month"].unique().tolist())
    if len(gold) != 5520 or len(expected_vars) != 20 or len(months) != 276:
        raise SystemExit("Gold row/variable/month baseline differs from frozen v1")
    if gold["value"].isna().any() or set(gold["variable_id"]) != expected_vars:
        raise SystemExit("Gold has missing values or unexpected variables")

    acquisition_counts = collections.Counter(r["source_id"] for r in acquisition if r.get("status") == "VALIDATED")
    regional_counts = collections.Counter(r["source_id"] for r in regional if r.get("status") == "VALIDATED")
    latest_preproc = {}
    for record in preprocessing:
        key = (record["source_id"], record["month"])
        if key not in latest_preproc or record.get("recorded_at_utc", "") > latest_preproc[key].get("recorded_at_utc", ""):
            latest_preproc[key] = record
    preproc_counts = collections.Counter(k[0] for k, r in latest_preproc.items() if r.get("status") == "VALIDATED")
    blocked_region_events = [r for r in regional if r.get("status") != "VALIDATED"]
    if len(acquisition) != 1104 or len(regional_counts) != 4 or min(regional_counts.values()) != 276:
        raise SystemExit("Acquisition/regional manifest baseline differs from v1")
    if len(latest_preproc) != 1104 or min(preproc_counts.values()) != 276:
        raise SystemExit("Latest preprocessing manifest does not have 1,104 validated source-month rows")

    source_hashes = {(r["source_id"], month_of(r)): r.get("sha256") for r in acquisition}
    unmatched_sources = []
    for row in gold[["source_id", "month", "source_sha256"]].drop_duplicates().to_dict("records"):
        recorded = source_hashes.get((row["source_id"], row["month"]))
        if recorded and recorded != row["source_sha256"]:
            unmatched_sources.append((row["source_id"], row["month"]))
    if unmatched_sources:
        raise SystemExit(f"Gold/source checksum lineage mismatch: {unmatched_sources[:3]}")

    if len(forecast) != 262 or len(split) != 262 or forecast["sample_id"].duplicated().any():
        raise SystemExit("Forecast sample count/identity differs from frozen v1")
    if set(forecast["sample_id"]) != set(split["sample_id"]):
        raise SystemExit("Forecast and split sample IDs disagree")
    counts = collections.Counter(split["partition"])
    if dict(counts) != {"train": 190, "validation": 36, "test": 36}:
        raise SystemExit(f"Frozen split counts differ from v1: {dict(counts)}")

    variable_prefixes = contract["forecast"]["feature_column_prefixes"]
    feature_columns = [f"{prefix}_lag_{lag:02d}" for lag in range(1, 13) for prefix in variable_prefixes.values()]
    if len(feature_columns) != 60 or any(c not in forecast.columns for c in feature_columns):
        raise SystemExit("Forecast model-input schema differs from the frozen contract")
    forbidden = {"target_value", "target_month", "sample_id", "history_lineage_json", "target_source_sha256"}
    if forbidden.intersection(feature_columns):
        raise SystemExit("Metadata or target leaked into predictor columns")
    leak_errors = []
    for record in forecast.to_dict("records"):
        target = pd.Period(record["target_month"], freq="M")
        expected_history = [str(target - offset) for offset in range(14, 2, -1)]
        actual_history = json.loads(record["feature_months_json"])
        if actual_history != expected_history:
            leak_errors.append(record["sample_id"])
        if pd.Timestamp(record["issue_date_utc"]) >= pd.Timestamp(record["target_start_utc"]):
            leak_errors.append(record["sample_id"])
    if leak_errors:
        raise SystemExit(f"Forecast availability audit failed for {len(leak_errors)} rows")

    result = {
        "audit_version": "1.0.0",
        "status": "PASSED",
        "audited_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "immutable_baseline": files,
        "contract_version": contract["contract_version"],
        "region_id": contract["region"]["region_id"],
        "coverage_months": {"start": months[0], "end": months[-1], "count": len(months)},
        "gold": {"rows": len(gold), "variables": len(expected_vars), "source_month_records": len(acquisition), "null_aggregate_values": int(gold["value"].isna().sum())},
        "acquisition_rows_by_source": dict(acquisition_counts),
        "regional_validated_rows_by_source": dict(regional_counts),
        "regional_nonvalidated_event_rows": [{"status": r.get("status"), "event_type": r.get("event_type"), "recorded_at_utc": r.get("recorded_at_utc"), "outputs": r.get("validated_regional_outputs")} for r in blocked_region_events],
        "preprocessing_history_rows": len(preprocessing),
        "latest_preprocessing_rows_by_source": dict(preproc_counts),
        "forecast": {"rows": len(forecast), "predictor_columns": len(feature_columns), "target": contract["forecast"]["target_variable_id"], "target_start": str(forecast["target_month"].min()), "target_end": str(forecast["target_month"].max()), "original_partition_counts": dict(counts), "leakage_errors": 0},
        "source_lineage_mismatches": len(unmatched_sources),
        "verification_scope": "Local Gold and manifest audit. Full source/regional/processed-grid byte verification remains evidenced by the CEDIA verification receipt.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result["audit_receipt_sha256"] = sha256(OUT)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
