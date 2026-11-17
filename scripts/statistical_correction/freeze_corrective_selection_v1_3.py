"""Freeze the completed v1.3 development choice before independent confirmation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/statistical_correction"))

import run_calibration as calibration

OUTPUT = ROOT / "results/statistical_correction/corrective_development_v1_3"
FREEZE_PATH = OUTPUT / "selection_freeze_iid_v1_3.json"
RECEIPT_PATH = OUTPUT / "selection_freeze_receipt_iid_v1_3.json"
CONFIG_PATH = ROOT / "configs/statistical_corrective_confirmation_v1_3.json"
PROTOCOL_PATH = ROOT / "docs/protocols/statistical_corrective_confirmation_v1_3.md"
RUNNER_PATH = ROOT / "scripts/statistical_correction/run_corrective_confirmation_v1_3.py"
SUMMARIZER_PATH = ROOT / "scripts/statistical_correction/summarize_corrective_confirmation_v1_3.py"
TEST_PATH = ROOT / "tests/test_statistical_confirmation_v13.py"
MODULE_PATH = ROOT / "src/kuntur_data/phase2/statistical_v12.py"
HELPER_PATH = ROOT / "scripts/statistical_correction/run_corrective_development_v1_3_expansion.py"
DEVELOPMENT_CONFIG_PATH = ROOT / "configs/statistical_corrective_development_expansion_v1_3.json"
DEVELOPMENT_MANIFEST_PATH = OUTPUT / "expanded_20261117/development_manifest.json"
METRICS_PATH = OUTPUT / "expanded_20261117/development_metrics.json"
SUMMARY_RECEIPT_PATH = OUTPUT / "expanded_20261117/summary_receipt.json"
REPORT_PATH = ROOT / "reports/statistical_correction/v1_3_development_candidate_selection_20261117.md"
CLOSURE_PATH = ROOT / "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json"
V12_REGISTRY_PATH = ROOT / "results/statistical_correction/validated_regime_registry.json"
HINDCAST_PATH = ROOT / "results/optimization/round_06_2025_test_score/sealed_2025_score.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def require_files() -> None:
    missing = [str(path.relative_to(ROOT)) for path in (
        CONFIG_PATH, PROTOCOL_PATH, RUNNER_PATH, SUMMARIZER_PATH, TEST_PATH, MODULE_PATH, HELPER_PATH,
        DEVELOPMENT_CONFIG_PATH, DEVELOPMENT_MANIFEST_PATH, METRICS_PATH,
        SUMMARY_RECEIPT_PATH, REPORT_PATH, CLOSURE_PATH, V12_REGISTRY_PATH,
        HINDCAST_PATH, Path(calibration.__file__).resolve(),
    ) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required freeze evidence is missing: " + ", ".join(missing))


def build_freeze() -> dict[str, Any]:
    require_files()
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    development = json.loads(DEVELOPMENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    summary_receipt = json.loads(SUMMARY_RECEIPT_PATH.read_text(encoding="utf-8"))
    dev_config = json.loads(DEVELOPMENT_CONFIG_PATH.read_text(encoding="utf-8"))
    confirm_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    if metrics.get("run_status") != "DEVELOPMENT_EXPANSION_COMPLETE" or metrics.get("completed_cells") != 180 or metrics.get("completed_outer_fits") != 180000:
        raise RuntimeError("The development expansion is not independently verified complete")
    if metrics.get("independent_confirmation_started") is not False:
        raise RuntimeError("Development metrics indicate that confirmation may already have started")
    if metrics.get("registered_inference_registry_modified") is not False:
        raise RuntimeError("Development phase unexpectedly changed an inference registry")
    selection = metrics.get("candidate_selection", {})
    iid = selection.get("iid_gaussian", {})
    candidate = iid.get("independent_confirmation_candidate")
    if candidate != "SARIMAX_SEASONAL_AR12" or iid.get("selection_status") != "CANDIDATE_READY_FOR_FREEZE_BEFORE_CONFIRMATION":
        raise RuntimeError("The frozen v1.3 rule did not nominate the expected IID candidate")
    if not any(item.get("method") == candidate and item.get("passed_all_development_gates") is True for item in iid.get("methods", [])):
        raise RuntimeError("The IID candidate did not pass every recorded development gate")
    for regime, record in selection.items():
        if regime != "iid_gaussian" and record.get("independent_confirmation_candidate") is not None:
            raise RuntimeError(f"Unexpected independent-confirmation candidate in {regime}")
    if dev_config.get("reserved_confirmation_seed_namespace_used") is not False:
        raise RuntimeError("The reserved confirmation namespace is marked as used")
    namespace = confirm_config.get("design", {}).get("series_seed_namespace")
    if namespace != dev_config.get("reserved_confirmation_seed_namespace"):
        raise RuntimeError("Confirmation namespace differs from the reserved untouched namespace")
    if confirm_config.get("protected_inputs", {}).get("v1_2_final_validation_seeds_used") is not False:
        raise RuntimeError("Confirmation config does not exclude v1.2 final-validation seeds")
    if confirm_config.get("protected_inputs", {}).get("v1_3_development_expansion_seeds_used") is not False:
        raise RuntimeError("Confirmation config does not exclude development seeds")
    if metrics.get("closure_manifest_sha256") != sha256(CLOSURE_PATH):
        raise RuntimeError("Authoritative v1.2 closure hash differs from the completed development receipt")
    if metrics.get("protected_v1_2_evidence", {}).get("validated_registry_sha256_current") != sha256(V12_REGISTRY_PATH):
        raise RuntimeError("The v1.2 registry has changed since development evidence was summarized")
    if metrics.get("protected_v1_2_evidence", {}).get("sealed_2025_score_sha256_current") != sha256(HINDCAST_PATH):
        raise RuntimeError("The sealed 2025 hindcast score has changed since development evidence was summarized")
    if summary_receipt.get("metrics_sha256") != sha256(METRICS_PATH) or summary_receipt.get("report_sha256") != sha256(REPORT_PATH):
        raise RuntimeError("Development summary receipt does not match current metrics/report")

    source_paths = {
        "confirmation_config": CONFIG_PATH,
        "confirmation_protocol": PROTOCOL_PATH,
        "confirmation_runner": RUNNER_PATH,
        "confirmation_summarizer": SUMMARIZER_PATH,
        "confirmation_unit_tests": TEST_PATH,
        "freeze_script": Path(__file__).resolve(),
        "statistical_module": MODULE_PATH,
        "calibration_generator": Path(calibration.__file__).resolve(),
        "development_helpers": HELPER_PATH,
        "development_config": DEVELOPMENT_CONFIG_PATH,
        "development_manifest": DEVELOPMENT_MANIFEST_PATH,
        "development_metrics": METRICS_PATH,
        "development_summary_receipt": SUMMARY_RECEIPT_PATH,
        "development_report": REPORT_PATH,
        "v1_2_closure_manifest": CLOSURE_PATH,
        "v1_2_registry": V12_REGISTRY_PATH,
        "sealed_2025_score": HINDCAST_PATH,
    }
    source_hashes = {key: sha256(path) for key, path in source_paths.items()}
    resources = metrics.get("resource_usage", {})
    payload: dict[str, Any] = {
        "schema_version": "kuntur-v1.3-development-selection-freeze-v1",
        "freeze_id": "KUNTUR-V1.3-IID-GAUSSIAN-SARIMAX-SEASONAL-AR12",
        "status": "FROZEN_BEFORE_INDEPENDENT_CONFIRMATION",
        "frozen_utc": utc_now(),
        "authorization_basis": "The user-provided KUNTUR v1.3 execution objective authorizes independent confirmation only after candidate, scope, gates, and seed namespace are frozen.",
        "signature_type": "SHA256_CHECKSUM_NOT_DIGITAL_SIGNATURE",
        "development_evidence": {
            "manifest_status": development.get("status"),
            "completed_cells": metrics.get("completed_cells"),
            "completed_outer_fits": metrics.get("completed_outer_fits"),
            "metrics_sha256": source_hashes["development_metrics"],
            "report_sha256": source_hashes["development_report"],
            "metrics_candidate_selection": iid,
            "cpu_seconds": resources.get("cpu_seconds"),
            "active_wall_seconds": resources.get("active_wall_seconds"),
            "peak_rss_bytes": resources.get("peak_rss_bytes"),
        },
        "frozen_candidate": confirm_config["candidate"],
        "frozen_design": confirm_config["design"],
        "frozen_acceptance_criteria": confirm_config["acceptance_criteria"],
        "frozen_resource_budget": confirm_config["resource_budget"],
        "interpretation_limits": confirm_config["interpretation_limits"],
        "confirmation_authorized": True,
        "source_hashes": source_hashes,
        "runtime_environment": {
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "scipy": importlib.metadata.version("scipy"),
            "statsmodels": importlib.metadata.version("statsmodels"),
        },
        "protected_history": {
            "v1_2_closure_sha256": sha256(CLOSURE_PATH),
            "v1_2_registry_sha256": sha256(V12_REGISTRY_PATH),
            "sealed_2025_score_sha256": sha256(HINDCAST_PATH),
            "v1_2_decisions_modified": False,
            "v1_2_registry_modified": False,
            "sealed_2025_test_rerun": False,
            "real_series_used_for_selection": False,
        },
        "selection_rule": "Among IID full-inference candidates that passed all three slope cells and every frozen gate, maximize minimum power; ties minimize mean oracle-normalized interval width. This rule selected SARIMAX_SEASONAL_AR12.",
        "not_selected_regimes": [
            {"regime": regime, "status": selection[regime].get("selection_status"), "candidate": None}
            for regime in selection if regime != "iid_gaussian"
        ],
    }
    payload["freeze_payload_sha256"] = canonical_hash(payload)
    return payload


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze", action="store_true", help="write the immutable selection-freeze manifest")
    group.add_argument("--verify", action="store_true", help="verify the previously frozen manifest")
    args = parser.parse_args()

    if args.verify:
        if not FREEZE_PATH.is_file() or not RECEIPT_PATH.is_file():
            raise FileNotFoundError("No v1.3 IID candidate freeze exists")
        current = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
        if current.get("status") != "FROZEN_BEFORE_INDEPENDENT_CONFIRMATION":
            raise RuntimeError("Selection freeze status is invalid")
        if receipt.get("selection_manifest_sha256") != sha256(FREEZE_PATH):
            raise RuntimeError("Selection freeze receipt hash mismatch")
        print(json.dumps({"status": current["status"], "freeze_path": str(FREEZE_PATH.relative_to(ROOT)), "selection_manifest_sha256": sha256(FREEZE_PATH)}, indent=2))
        return

    if not args.freeze:
        parser.error("specify --freeze or --verify")
    if FREEZE_PATH.exists() or RECEIPT_PATH.exists():
        raise FileExistsError("A freeze artifact already exists; it is immutable and will not be overwritten")
    payload = build_freeze()
    write_exclusive(FREEZE_PATH, payload)
    receipt = {
        "schema_version": "kuntur-v1.3-selection-freeze-receipt-v1",
        "selection_manifest_path": str(FREEZE_PATH.relative_to(ROOT)).replace("\\", "/"),
        "selection_manifest_sha256": sha256(FREEZE_PATH),
        "freeze_payload_sha256": payload["freeze_payload_sha256"],
        "frozen_utc": payload["frozen_utc"],
        "status": payload["status"],
    }
    write_exclusive(RECEIPT_PATH, receipt)
    print(json.dumps({"status": payload["status"], "selection_manifest_path": str(FREEZE_PATH.relative_to(ROOT)), "selection_manifest_sha256": receipt["selection_manifest_sha256"], "receipt_path": str(RECEIPT_PATH.relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
