#!/usr/bin/env python3
"""Read-only verification of frozen v1.2/v1.3 statistical evidence outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.statistical_correction import summarize_corrective_confirmation_v1_3 as confirmation  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {path}")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load verification source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify() -> dict:
    checked: list[str] = []

    # Reconstruct v1.2 selector scores in memory and compare with the preserved
    # JSON; the historical output file is never rewritten.
    reconstruction_path = ROOT / "results/statistical_correction/selection_reconstruction_v1_3.json"
    reconstruction = load_json(reconstruction_path)
    recon_module = load_module("kuntur_v13_selection_reconstruction", ROOT / "scripts/statistical_correction/reconstruct_frozen_selection_v1_2.py")
    rebuilt = recon_module.build()
    if rebuilt != reconstruction:
        raise RuntimeError("Read-only v1.2 score reconstruction differs from its saved result")
    checked.append("v1.2_selection_reconstruction")

    # Verify the completed expansion and every checkpoint listed in its saved
    # metrics inventory without running any simulation.
    dev_metrics_path = ROOT / "results/statistical_correction/corrective_development_v1_3/expanded_20261117/development_metrics.json"
    dev_metrics = load_json(dev_metrics_path)
    dev_receipt_path = dev_metrics_path.parent / "summary_receipt.json"
    dev_receipt = load_json(dev_receipt_path)
    dev_manifest_path = ROOT / dev_metrics["manifest_path"]
    assert_sha(dev_manifest_path, dev_metrics["manifest_sha256"], "development run manifest")
    assert_sha(dev_metrics_path, dev_receipt["metrics_sha256"], "development metrics")
    dev_report_path = ROOT / dev_receipt["report_path"]
    assert_sha(dev_report_path, dev_receipt["report_sha256"], "development report")
    if dev_metrics.get("run_status") != "DEVELOPMENT_EXPANSION_COMPLETE" or dev_metrics.get("completed_cells") != 180 or dev_metrics.get("completed_outer_fits") != 180000:
        raise RuntimeError("Development expansion is not complete at the expected frozen size")
    for item in dev_metrics["checkpoint_inventory"]:
        checkpoint = ROOT / item["path"]
        assert_sha(checkpoint, item["sha256"], "development checkpoint")
        if checkpoint.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Development checkpoint size mismatch: {checkpoint}")
    checked.append(f"development_metrics_and_{len(dev_metrics['checkpoint_inventory'])}_checkpoints")

    # The confirmation helper's verify() is read-only. Recompute summaries in
    # memory and compare them to the saved metrics; do not call its main().
    config, run_manifest, cells = confirmation.verify()
    metric_path = confirmation.METRICS_PATH
    confirm_metrics = load_json(metric_path)
    criteria = config["acceptance_criteria"]
    summaries = [
        confirmation.summarize_cell(cell, float(criteria["type_i_alpha"]), float(criteria["coverage_nominal"]))
        for cell in cells
    ]
    if summaries != confirm_metrics["cells"]:
        raise RuntimeError("Read-only confirmation summary differs from frozen metrics")
    confirm_receipt = load_json(confirmation.RECEIPT_PATH)
    assert_sha(metric_path, confirm_receipt["metrics_sha256"], "confirmation metrics")
    assert_sha(confirmation.REPORT_PATH, confirm_receipt["report_sha256"], "original confirmation report")
    if confirm_metrics.get("status") != "INDEPENDENT_CONFIRMATION_FAIL_NOT_VALIDATED" or confirm_metrics.get("registry_created") is not False:
        raise RuntimeError("Confirmation outcome or registry state differs from the recorded failure")
    if confirmation.V13_REGISTRY_PATH.exists():
        raise RuntimeError("A v1.3 registry exists although the candidate failed confirmation")
    checked.append("independent_confirmation_metrics_and_checkpoints")

    # Verify the actual descriptive export and untouched v1.2 protected inputs.
    trend_manifest_path = ROOT / "results/statistical_correction/real_series_v13/descriptive_trend_manifest_v1_3.json"
    trend_manifest = load_json(trend_manifest_path)
    gold_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
    trend_csv_path = trend_manifest_path.parent / trend_manifest["output"]["path"]
    assert_sha(gold_path, trend_manifest["input"]["sha256"], "frozen Gold input")
    assert_sha(trend_csv_path, trend_manifest["output"]["sha256"], "v1.3 descriptive trend export")
    closure_path = ROOT / "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json"
    closure = load_json(closure_path)
    registry_path = ROOT / "results/statistical_correction/validated_regime_registry.json"
    score_path = ROOT / "results/optimization/round_06_2025_test_score/sealed_2025_score.json"
    assert_sha(registry_path, "d68ddbf8825d98699198140593a871df307d71a3edb490ca02f992bae7f13580", "v1.2 registry")
    assert_sha(score_path, "a38af9f37962f948a0af7c27b06c829adb01c78311131f764bc344f1c64c80b7", "sealed 2025 score")
    if closure.get("closure_status") != "PARTIAL_EXPERIMENT_CLOSED" or closure.get("not_a_successful_validation") is not True:
        raise RuntimeError("Unexpected authoritative v1.2 closure state")
    checked.append("real_series_export_and_protected_v1_2_inputs")

    return {
        "status": "READ_ONLY_EVIDENCE_VERIFICATION_PASSED",
        "checked": checked,
        "development_outer_fits": dev_metrics["completed_outer_fits"],
        "confirmation_outer_fits": run_manifest["completed_outer_fits"],
        "confirmation_status": confirm_metrics["status"],
        "v1_3_registry_created": False,
        "files_written": 0,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
