#!/usr/bin/env python3
"""Write a hash inventory for the completed KUNTUR v1.3 evidence handoff."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/statistical_correction/v1_3_evidence_handoff_manifest.json"
ARTIFACT_PATHS = (
    "reports/statistical_correction/v1_3_corrective_evidence_handoff_20261117.md",
    "reports/statistical_correction/v1_3_confirmation_gate_display_correction_addendum_20261117.md",
    "reports/statistical_correction/v1_3_development_candidate_selection_20261117.md",
    "reports/statistical_correction/v1_3_independent_confirmation_iid_20261117.md",
    "reports/statistical_correction/selection_root_cause_v1_3_20261117.md",
    "reports/statistical_correction/v1_2_report_reconciliation_addendum_v1_3.md",
    "src/kuntur_data/phase2/statistical_v13.py",
    "scripts/run_phase2_v13_descriptive_trends.py",
    "scripts/statistical_correction/render_confirmation_gate_correction_v1_3.py",
    "scripts/statistical_correction/verify_v1_3_evidence_bundle.py",
    "scripts/statistical_correction/write_v1_3_evidence_handoff_manifest.py",
    "tests/test_statistical_v13.py",
    "tests/test_statistical_confirmation_v13.py",
    "configs/statistical_corrective_confirmation_v1_3.json",
    "docs/protocols/statistical_corrective_confirmation_v1_3.md",
    "docs/audits/statistical_failure_to_correction_matrix_v1_3.csv",
    "results/statistical_correction/selection_reconstruction_v1_3.json",
    "results/statistical_correction/corrective_development_v1_3/selection_freeze_iid_v1_3_r1.json",
    "results/statistical_correction/corrective_development_v1_3/selection_freeze_receipt_iid_v1_3_r1.json",
    "results/statistical_correction/corrective_development_v1_3/expanded_20261117/development_manifest.json",
    "results/statistical_correction/corrective_development_v1_3/expanded_20261117/development_metrics.json",
    "results/statistical_correction/corrective_development_v1_3/expanded_20261117/summary_receipt.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_manifest.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_metrics.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/summary_receipt.json",
    "results/statistical_correction/real_series_v13/descriptive_trend_estimates_v1_3.csv",
    "results/statistical_correction/real_series_v13/descriptive_trend_manifest_v1_3.json",
    "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
    "data/manifests/gold_manifest_v1.json",
    "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json",
    "results/statistical_correction/validated_regime_registry.json",
    "results/optimization/round_06_2025_test_score/sealed_2025_score.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict:
    records = []
    for relative in ARTIFACT_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required handoff artifact is missing: {relative}")
        records.append({
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    v13_registry = ROOT / "results/statistical_correction/validated_regime_registry_v1_3.json"
    if v13_registry.exists():
        raise RuntimeError("The independent confirmation failed; refusing to seal a handoff with a v1.3 registry")
    return {
        "schema_version": "kuntur-v1.3-evidence-handoff-manifest-v1",
        "status": "DEVELOPMENT_COMPLETE_CONFIRMATION_FAILED_INFERENCE_WITHHELD",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(records),
        "v1_3_inference_registry_created": False,
        "v1_2_registry_modified": False,
        "sealed_2025_score_touched": False,
        "publication_or_merge_performed": False,
        "verification": {
            "command": "python scripts/statistical_correction/verify_v1_3_evidence_bundle.py",
            "last_observed_status": "READ_ONLY_EVIDENCE_VERIFICATION_PASSED",
            "files_written_by_verifier": 0,
        },
        "artifacts": records,
    }


def main() -> None:
    manifest = build_manifest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "manifest_path": str(OUTPUT.relative_to(ROOT)).replace("\\", "/"),
        "artifact_count": manifest["artifact_count"],
        "manifest_sha256": sha256(OUTPUT),
    }, indent=2))


if __name__ == "__main__":
    main()
