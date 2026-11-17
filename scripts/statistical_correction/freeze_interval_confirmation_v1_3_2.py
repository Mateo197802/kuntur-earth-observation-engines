from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/statistical_interval_confirmation_v1_3_2.json"
PROTOCOL_PATH = ROOT / "docs/protocols/statistical_interval_confirmation_v1_3_2.md"
RESULTS = ROOT / "results/statistical_correction/interval_confirmation_v1_3_2"
FREEZE_PATH = RESULTS / "protocol_freeze.json"

SOURCE_PATHS = (
    "configs/statistical_interval_confirmation_v1_3_2.json",
    "docs/protocols/statistical_interval_confirmation_v1_3_2.md",
    "scripts/statistical_correction/freeze_interval_confirmation_v1_3_2.py",
    "scripts/statistical_correction/run_interval_confirmation_v1_3_2.py",
    "scripts/statistical_correction/summarize_interval_confirmation_v1_3_2.py",
    "scripts/statistical_correction/interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/run_calibration.py",
    "scripts/statistical_correction/run_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/summarize_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/freeze_interval_comparison_v1_3_2.py",
    "tests/test_interval_confirmation_v13_2.py",
    "results/statistical_correction/interval_comparison_v1_3_2/protocol_freeze.json",
    "results/statistical_correction/interval_comparison_v1_3_2/development/development_metrics.json",
    "results/statistical_correction/interval_comparison_v1_3_2/development/development_report.md",
    "results/statistical_correction/interval_comparison_v1_3_2/development/summary_receipt.json",
    "results/statistical_correction/validated_regime_registry.json",
    "results/statistical_correction/validation_registry_manifest.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_manifest.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_metrics.json",
    "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json",
    "results/optimization/round_06_2025_test_score/sealed_2025_score.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for(namespace: str, slope: float, replicate: int) -> int:
    text = f"{namespace}|series|iid_gaussian|{slope:.8f}|{replicate}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:16], "big", signed=False)


def source_hashes() -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input is missing: {relative}")
        result[relative] = sha256_file(path)
    return result


def _candidate_checkpoint_rows() -> list[dict[str, Any]]:
    directory = ROOT / "results/statistical_correction/interval_comparison_v1_3_2/development/checkpoints"
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("cell_iid_gaussian_n276_s*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"Development checkpoint is incomplete: {path.name}")
        rows.extend(row for row in payload.get("replicates", []) if row.get("method") == "wald_student_t")
    if len(rows) != 3000:
        raise RuntimeError(f"Expected 3000 selected-candidate development rows, found {len(rows)}")
    return rows


def _development_projection(config: dict[str, Any]) -> dict[str, Any]:
    rows = _candidate_checkpoint_rows()
    cpu = [float(row["cpu_seconds"]) for row in rows if row.get("cpu_seconds") is not None]
    wall = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds") is not None]
    if len(cpu) != 3000 or len(wall) != 3000:
        raise RuntimeError("Selected-candidate development runtime measurements are incomplete")
    multiplier = float(config["resource_budget"].get("reserve_multiplier", 2.0))
    fits = int(config["resource_budget"]["candidate_fit_count"])
    cpu_mean, wall_mean = statistics.mean(cpu), statistics.mean(wall)
    projection = {
        "measured_development_fit_count": len(rows),
        "mean_cpu_seconds_per_fit": cpu_mean,
        "mean_wall_seconds_per_fit": wall_mean,
        "two_fold_projected_cpu_seconds": multiplier * cpu_mean * fits,
        "two_fold_projected_wall_seconds": multiplier * wall_mean * fits,
        "reserve_multiplier": multiplier,
    }
    declared = config["resource_budget"]["measured_development_candidate_runtime"]
    for key in ("mean_cpu_seconds_per_fit", "mean_wall_seconds_per_fit", "two_fold_projected_cpu_seconds", "two_fold_projected_wall_seconds"):
        if abs(float(declared[key]) - float(projection[key])) > 1e-10:
            raise RuntimeError(f"The measured development runtime for {key} does not match the frozen config")
    budget = config["resource_budget"]
    if projection["two_fold_projected_cpu_seconds"] >= int(budget["hard_max_cpu_seconds"]):
        raise RuntimeError("Measured two-fold CPU projection exceeds the frozen confirmation cap")
    if projection["two_fold_projected_wall_seconds"] >= int(budget["hard_max_active_wall_seconds"]):
        raise RuntimeError("Measured two-fold wall projection exceeds the frozen confirmation cap")
    return projection


def seed_inventory(config: dict[str, Any]) -> dict[str, Any]:
    design = config["design"]
    confirm = [
        seed_for(design["series_seed_namespace"], float(slope), replicate)
        for slope in design["slopes_per_decade"]
        for replicate in range(int(design["replicates_per_slope"]))
    ]
    development_freeze = json.loads(
        (ROOT / "results/statistical_correction/interval_comparison_v1_3_2/protocol_freeze.json").read_text(encoding="utf-8")
    )
    development_config = json.loads(
        (ROOT / "configs/statistical_interval_comparison_v1_3_2.json").read_text(encoding="utf-8")
    )
    old_design = development_config["design"]
    old = []
    for namespace, key in (
        (old_design["preflight_seed_namespace"], "preflight_replicates_per_slope"),
        (old_design["series_seed_namespace"], "development_replicates_per_slope"),
    ):
        old.extend(
            seed_for(namespace, float(slope), replicate)
            for slope in old_design["slopes_per_decade"]
            for replicate in range(int(old_design[key]))
        )
    if len(confirm) != len(set(confirm)) or set(confirm) & set(old):
        raise RuntimeError("Confirmation seed inventory is duplicated or overlaps the v1.3.2 preflight/development seeds")
    canonical = "\n".join(str(seed) for seed in confirm).encode("ascii")
    old_inventory = development_freeze.get("seed_inventory", {})
    return {
        "confirmation_seed_count": len(confirm),
        "confirmation_unique_seed_count": len(set(confirm)),
        "confirmation_seed_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "v1_3_2_preflight_and_development_seed_count": len(old),
        "v1_3_2_seed_inventory_sha256": old_inventory.get("seed_inventory_sha256"),
        "no_overlap_with_v1_3_2_preflight_or_development": True,
    }


def build_payload() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    metrics_path = ROOT / "results/statistical_correction/interval_comparison_v1_3_2/development/development_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    selection = metrics.get("candidate_selection", {})
    decisions = metrics.get("candidate_gate_decisions", {})
    if metrics.get("status") != "COMPLETE_DEVELOPMENT_ONLY":
        raise RuntimeError("A complete v1.3.2 development comparison is required before confirmation freeze")
    if selection.get("development_candidate_nominated_for_independent_freeze") != "wald_student_t":
        raise RuntimeError("The nominated development candidate is not wald_student_t")
    if selection.get("corrected_candidates_passing_all_gates") != ["wald_student_t"]:
        raise RuntimeError("The frozen candidate set differs from the development evidence")
    if not decisions.get("wald_student_t", {}).get("all_three_slopes_pass"):
        raise RuntimeError("The nominated method did not pass every development gate")
    if selection.get("independent_confirmation_started") or selection.get("inference_registry_modified"):
        raise RuntimeError("Development evidence indicates a confirmation or registry mutation already occurred")
    projection = _development_projection(config)
    return {
        "schema_version": "kuntur-statistical-interval-confirmation-freeze-v1.3.2",
        "protocol_id": config["protocol_id"],
        "status": "FROZEN_BEFORE_INDEPENDENT_CONFIRMATION",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_sha256": sha256_file(CONFIG_PATH),
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "source_hashes": source_hashes(),
        "candidate": {
            "id": config["scope"]["candidate_method"],
            "point_model": config["scope"]["point_model"],
            "interval_and_test": config["scope"]["candidate_role"],
            "applicable_regime": config["scope"]["regime"],
            "development_candidate_selection": "The only corrected candidate that passed every v1.3.2 development gate at all three slopes.",
        },
        "development_evidence": {
            "metrics_sha256": sha256_file(metrics_path),
            "report_sha256": sha256_file(ROOT / "results/statistical_correction/interval_comparison_v1_3_2/development/development_report.md"),
            "candidate_gate_decisions_sha256": hashlib.sha256(json.dumps(decisions, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
        },
        "seed_inventory": seed_inventory(config),
        "measured_resource_projection": projection,
        "acceptance_criteria": config["acceptance_criteria"],
        "registry_policy": "The inference registry is not changed by the confirmation runner. A passing result is limited to this IID synthetic regime and requires separate review before any registry update.",
        "protected_evidence_unchanged": {
            "v1_2_validation_and_partial_closure": True,
            "v1_3_original_failed_confirmation": True,
            "v1_3_1_failed_engineering_preflight": True,
            "v1_3_2_preflight_and_development": True,
            "2025_predictive_test": True,
            "real_gold": True,
            "inference_registry": True,
        },
        "environment_at_freeze": {"python": sys.version, "platform": platform.platform()},
    }


def verify_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("status") != "FROZEN_BEFORE_INDEPENDENT_CONFIRMATION":
        raise RuntimeError("Unexpected confirmation freeze status")
    if payload.get("source_hashes") != source_hashes():
        raise RuntimeError("A frozen protocol, estimator, runner, test, or evidence input changed")
    if payload.get("configuration_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("Confirmation configuration hash does not match its freeze")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("protocol_sha256") != sha256_file(PROTOCOL_PATH):
        raise RuntimeError("Confirmation protocol hash does not match its freeze")
    if payload.get("seed_inventory") != seed_inventory(config):
        raise RuntimeError("Confirmation seed inventory does not match its freeze")
    if payload.get("measured_resource_projection") != _development_projection(config):
        raise RuntimeError("Measured candidate runtime or resource projection changed after freeze")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        verify_payload(payload)
        print(json.dumps({"status": "FREEZE_VERIFIED", "freeze_sha256": sha256_file(FREEZE_PATH), "source_count": len(payload["source_hashes"]), "seed_inventory": payload["seed_inventory"], "measured_resource_projection": payload["measured_resource_projection"]}, indent=2))
        return
    if FREEZE_PATH.exists():
        raise SystemExit(f"Confirmation freeze already exists and is immutable: {FREEZE_PATH.relative_to(ROOT)}")
    payload = build_payload()
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = FREEZE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(FREEZE_PATH)
    print(json.dumps({"status": payload["status"], "freeze_sha256": sha256_file(FREEZE_PATH), "source_count": len(payload["source_hashes"]), "seed_inventory": payload["seed_inventory"], "measured_resource_projection": payload["measured_resource_projection"]}, indent=2))


if __name__ == "__main__":
    main()
