#!/usr/bin/env python3
"""Independently verify frozen inputs and sanitized Phase 2 release files."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> dict:
    actual = sha256(path)
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {label}: expected {expected}, got {actual}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": actual, "bytes": path.stat().st_size}


def main() -> None:
    required = [
        "README.md", "LICENSE", "CITATION.cff", "pyproject.toml",
        "configs/data_contract_v1.json", "docs/data_dictionary_v1.csv",
        "docs/phase2_api.md", "docs/protocols/predictive_model_protocol_v1.md",
        "docs/protocols/external_validation_protocol_v1.md",
        "reports/phase2/PHASE2_HANDOFF_v1.md",
        "results/phase2/public/phase2_aggregate_summary_v1.json",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    if missing:
        raise SystemExit(f"Missing required release artifacts: {missing}")

    gold_manifest = json.loads((ROOT / "data/manifests/gold_manifest_v1.json").read_text(encoding="utf-8"))
    gold = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
    gold_record = require_hash(gold, gold_manifest["sha256"], "frozen Gold v1")
    gold_frame = pd.read_parquet(gold)
    if (len(gold_frame), gold_frame.month.nunique(), gold_frame.variable_id.nunique()) != (5520, 276, 20):
        raise SystemExit("Frozen Gold shape/coverage does not match its manifest")

    forecast_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    split_path = ROOT / "data/gold/kuntur_forecast_split_indices_v1.parquet"
    final = json.loads((ROOT / "data/manifests/final_test_manifest_v1.json").read_text(encoding="utf-8"))
    freeze_path = ROOT / "data/manifests/model_selection_freeze_v1.json"
    if final.get("status") != "FINAL_TEST_EVALUATED_ONCE" or final.get("test_read_count") != 1:
        raise SystemExit("Final test must remain completed exactly once")
    if sha256(freeze_path) != final.get("selection_freeze_sha256"):
        raise SystemExit("Final test no longer matches the frozen model selection")
    forecast_record = require_hash(forecast_path, final["source_forecast_sha256"], "frozen forecast v1")
    split_record = require_hash(split_path, final["source_split_sha256"], "frozen split v1")

    external_manifest_path = ROOT / "data/manifests/external/external_evaluation_manifest_v1.json"
    external = json.loads(external_manifest_path.read_text(encoding="utf-8"))
    if external.get("status") != "EVALUATED_WITH_SOURCE_LIMITS":
        raise SystemExit("External evaluation has not completed")
    inamhi_audit = None
    inamhi_manifest_path = ROOT / "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json"
    inamhi_checkpoint_path = ROOT / "data/manifests/external/inamhi_yearbook_checkpoints_v1.json"
    if inamhi_manifest_path.is_file() and inamhi_checkpoint_path.is_file():
        inamhi_manifest = json.loads(inamhi_manifest_path.read_text(encoding="utf-8"))
        checkpoint_manifest = json.loads(inamhi_checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_manifest.get("status") != "CHECKPOINTS_VALIDATED":
            raise SystemExit("INAMHI per-year checkpoints are incomplete")
        source_rows = 0
        year_evidence = []
        for year in range(2002, 2013):
            source_record = inamhi_manifest.get("yearbook_sources", {}).get(str(year), {})
            checkpoint = checkpoint_manifest.get("years", {}).get(str(year), {})
            if source_record.get("status") != "PDF_ACQUIRED_AND_PARSED" or checkpoint.get("status") != "VALIDATED":
                raise SystemExit(f"INAMHI year {year} is not validated with station rows")
            checkpoint_file = ROOT / checkpoint["path"]
            if not checkpoint_file.is_file() or sha256(checkpoint_file) != checkpoint.get("sha256"):
                raise SystemExit(f"INAMHI year {year} checkpoint SHA-256 mismatch")
            frame = pd.read_parquet(checkpoint_file)
            if len(frame) != int(source_record.get("bbox_station_month_rows", -1)) or len(frame) != int(checkpoint.get("rows", -2)):
                raise SystemExit(f"INAMHI year {year} row counts disagree across checkpoints and source manifest")
            if frame.empty or set(frame.year.astype(int)) != {year} or set(frame.source_pdf_sha256.astype(str)) != {source_record.get("sha256")}:
                raise SystemExit(f"INAMHI year {year} checkpoint provenance is invalid")
            pdf_path = ROOT / source_record["path"]
            if not pdf_path.is_file() or sha256(pdf_path) != source_record.get("sha256"):
                raise SystemExit(f"INAMHI year {year} source PDF SHA-256 mismatch")
            source_rows += len(frame)
            year_evidence.append({"year": year, "rows": len(frame), "checkpoint_sha256": checkpoint["sha256"], "source_pdf_sha256": source_record["sha256"]})
        if source_rows != int(inamhi_manifest.get("counts", {}).get("bbox_station_month_rows", -1)):
            raise SystemExit("INAMHI checkpoint row total disagrees with its acquisition manifest")
        inamhi_audit = {"status": "INDEPENDENT_YEAR_CHECKS_PASSED", "years": year_evidence, "station_month_rows": source_rows, "checkpoint_manifest_sha256": sha256(inamhi_checkpoint_path)}
    public = ROOT / "results/phase2/public"
    report = ROOT / "reports/phase2/PHASE2_HANDOFF_v1.md"
    candidates = [*public.glob("*.csv"), *public.glob("*.json"), report]
    private_path_pattern = re.compile(
        r"(?i)(diego\.almeida|/home/" + "diego" + r"|[A-Z]:\\Users\\Pc|earthdata\.token|Bearer\s+[A-Za-z0-9._-]{24,})"
    )
    for path in candidates:
        if path.suffix.lower() in {".parquet", ".tif", ".tiff", ".h5", ".hdf", ".nc", ".nc4", ".pdf", ".joblib", ".pt", ".zip", ".tar", ".gz"}:
            raise SystemExit(f"Scientific data/model artifact must not be in release staging: {path.name}")
        content = path.read_text(encoding="utf-8", errors="ignore")
        if private_path_pattern.search(content):
            raise SystemExit(f"Private path or credential-like content found in release artifact: {path.relative_to(ROOT)}")

    release_files = [p for p in candidates if p.is_file()]
    file_records = {p.relative_to(ROOT).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(set(release_files))}
    summary_path = public / "phase2_aggregate_summary_v1.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for relative_path, record in summary.get("artifacts", {}).items():
        path = ROOT / relative_path
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise SystemExit(f"Public summary artifact hash mismatch: {relative_path}")
    payload = {
        "manifest_version": "1.0.0", "status": "PASSED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "frozen_inputs": {"gold": gold_record, "forecast": forecast_record, "original_split": split_record},
        "final_test": {"status": final["status"], "read_count": final["test_read_count"], "manifest_sha256": sha256(ROOT / "data/manifests/final_test_manifest_v1.json")},
        "external_evaluation": {"status": external["status"], "manifest_sha256": sha256(external_manifest_path)},
        "inamhi_local_validation": inamhi_audit,
        "release_files": file_records,
        "checks": ["frozen input hashes and dimensions match", "final test evaluated once and model selection freeze unchanged", "external evaluation manifest validated", "INAMHI yearbook rows/checkpoint hashes/source PDF hashes reconcile independently where supplied", "no local external datasets, NASA datasets, model binaries, credential files, or CEDIA paths in release outputs", "aggregate output checksums match the public summary"],
    }
    out = ROOT / "reports/phase2/phase2_release_manifest_v1.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gold_sha256": gold_record["sha256"], "forecast_sha256": forecast_record["sha256"], "split_sha256": split_record["sha256"], "release_files": len(file_records), "manifest": out.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(out)}, indent=2))


if __name__ == "__main__":
    main()
