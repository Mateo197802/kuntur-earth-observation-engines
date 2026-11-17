"""Reconcile and close the existing v1.2 validation as a partial experiment.

This is read-only with respect to frozen results. It writes a new closure
manifest and never rewrites the legacy calibration manifest or checkpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/statistical_correction/validation"
LEGACY_MANIFEST = BASE / "calibration_manifest.json"
MATRIX = BASE / "execution_reconciliation_method_matrix_20261116.csv"
SELECTION = ROOT / "results/statistical_correction/selection/selection_decisions.json"
REGISTRY = ROOT / "results/statistical_correction/validated_regime_registry.json"
CORRECTION_MANIFEST = ROOT / "results/statistical_correction/correction_manifest.json"
SCORE_2025 = ROOT / "results/optimization/round_06_2025_test_score/sealed_2025_score.json"
DEFAULT_OUTPUT = BASE / "partial_validation_closure_v1_2_20261116.json"
METHODS_PER_SCENARIO = 16
EXPECTED_SCENARIOS = 60


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def reconcile() -> dict[str, Any]:
    manifest = read_json(LEGACY_MANIFEST)
    selection = read_json(SELECTION)
    registry = read_json(REGISTRY)
    matrix_bytes = MATRIX.read_bytes()
    matrix_rows = list(csv.DictReader(matrix_bytes.decode("utf-8").splitlines()))
    checkpoint_paths = sorted((BASE / "checkpoints").glob("validation_*.json"))

    if manifest.get("phase") != "validation" or int(manifest.get("planned_scenario_count", -1)) != EXPECTED_SCENARIOS:
        raise ValueError("Legacy validation manifest scope does not match the frozen 60-scenario plan")
    if len(matrix_rows) != 160:
        raise ValueError(f"Expected 160 method-regime rows, found {len(matrix_rows)}")

    missing_names = set(manifest.get("missing_scenarios", []))
    actual_names = {path.name for path in checkpoint_paths}
    expected_names = missing_names | actual_names
    if len(expected_names) != EXPECTED_SCENARIOS:
        raise ValueError(f"Cannot reconcile the 60-scenario universe: inferred {len(expected_names)} names")

    inventory: list[dict[str, Any]] = []
    complete_paths: set[str] = set()
    complete_count = in_progress_count = 0
    for path in checkpoint_paths:
        record = read_json(path)
        config = record.get("config", {})
        progress = record.get("progress", {})
        status = str(record.get("status", "UNKNOWN"))
        requested = int(config.get("reps", -1))
        completed = int(progress.get("completed_reps", requested if status == "COMPLETE" else 0))
        digest = sha256(path)
        relative = path.relative_to(ROOT).as_posix()
        linked_rows = [row for row in matrix_rows if row.get("source_checkpoint_path") == relative]
        unique_hashes = {row.get("source_checkpoint_sha256") for row in linked_rows}
        if status == "COMPLETE":
            complete_count += 1
            complete_paths.add(path.name)
            if completed != requested or requested != 1000:
                raise ValueError(f"Completed checkpoint has unexpected replication count: {path.name}")
            if len(linked_rows) != METHODS_PER_SCENARIO or unique_hashes != {digest}:
                raise ValueError(f"Completed checkpoint/matrix linkage failed: {path.name}")
        elif status == "IN_PROGRESS":
            in_progress_count += 1
            if len(linked_rows) != 0:
                raise ValueError(f"Partial checkpoint must not appear in completed matrix: {path.name}")
        else:
            raise ValueError(f"Unexpected checkpoint status {status!r}: {path.name}")
        inventory.append({
            "path": relative,
            "status": status,
            "completed_reps": completed,
            "requested_reps": requested,
            "sha256": digest,
            "bytes": path.stat().st_size,
            "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
            "matrix_rows": len(linked_rows),
            "matrix_hash_matches": bool(linked_rows) and unique_hashes == {digest},
        })

    if complete_count != 10 or in_progress_count != 1:
        raise ValueError(f"Expected 10 complete plus 1 in-progress checkpoint; found {complete_count}+{in_progress_count}")
    unlisted_complete = sorted(complete_paths & missing_names)
    legacy_complete_names = sorted(expected_names - missing_names)
    partial_names = sorted(path.name for path in checkpoint_paths if read_json(path).get("status") == "IN_PROGRESS")
    not_started = sorted(expected_names - actual_names)
    if (len(legacy_complete_names), len(unlisted_complete), len(not_started)) != (7, 3, 49):
        raise ValueError("Legacy manifest/checkpoint reconciliation did not yield 7 recorded, 3 subsequently complete, and 49 not started")

    status_counts: dict[str, int] = {}
    for row in matrix_rows:
        status = row.get("validity_status", "UNKNOWN")
        status_counts[status] = status_counts.get(status, 0) + 1
    passing = [row for row in matrix_rows if row.get("validity_status") == "VALIDATED_FOR_DECLARED_REGIME"]
    selected_rows = [row for row in matrix_rows if _bool(row.get("frozen_selected_method"))]
    if len(passing) != 18 or any(_bool(row.get("frozen_selected_method")) for row in passing):
        raise ValueError("The 18 passing method-regimes must be distinct from all frozen-preselected rows")

    selected_methods: list[dict[str, Any]] = []
    for regime, decision in selection.get("decisions", {}).items():
        selected = decision.get("selected")
        if selected:
            matching = [row for row in selected_rows if row.get("process") == regime and row.get("method") == selected.get("method")]
            if len(matching) != 1:
                raise ValueError(f"Frozen selected method is absent or duplicated in matrix: {regime}")
            row = matching[0]
            selected_methods.append({
                "regime": regime,
                "method": selected["method"],
                "validation_status": row["validity_status"],
                "type1_fpr": float(row["type1_fpr_005"]),
                "type1_mc95": [float(row["type1_mc95_low"]), float(row["type1_mc95_high"])],
                "coverage_95": float(row["coverage_95"]) if row.get("coverage_95") else None,
                "type1_gate_pass": _bool(row.get("type1_gate_pass")),
                "coverage_gate_pass": _bool(row.get("coverage_gate_pass")) if row.get("coverage_gate_pass") else None,
            })
    if len(selected_methods) != 2 or any(item["type1_gate_pass"] for item in selected_methods):
        raise ValueError("Expected two frozen selections, both failing independent Type-I calibration")

    matrix_path_hashes = sorted({row.get("source_checkpoint_path") for row in matrix_rows})
    complete_matrix_scenarios = {Path(item).name for item in matrix_path_hashes}
    if complete_matrix_scenarios != complete_paths:
        raise ValueError("Method matrix does not cover exactly the ten complete checkpoints")

    slopes = sorted({float(row["true_slope_per_decade"]) for row in matrix_rows})
    registry_entries = registry.get("regimes", {})
    if len(registry_entries) != 0:
        raise ValueError("Validated regime registry is no longer empty; re-evaluate closure conditions")

    validator_source = Path(__file__)
    return {
        "schema_version": "1.0.0",
        "experiment_id": "kuntur-statistical-inference-v1.2-independent-validation",
        "closure_status": "PARTIAL_EXPERIMENT_CLOSED",
        "closure_time_utc": datetime.now(timezone.utc).isoformat(),
        "not_a_successful_validation": True,
        "legacy_manifest": {
            "path": LEGACY_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": sha256(LEGACY_MANIFEST),
            "status": manifest.get("status"),
            "recorded_complete_scenarios": int(manifest.get("scenario_count", 0)),
            "planned_scenarios": int(manifest.get("planned_scenario_count", 0)),
            "recorded_completed_names": legacy_complete_names,
            "recorded_missing_count": len(missing_names),
            "manifest_missing_but_now_complete": unlisted_complete,
            "in_progress_still_listed_missing": partial_names,
        },
        "actual_checkpoint_counts": {
            "complete_hash_verified": complete_count,
            "in_progress_hash_verified": in_progress_count,
            "not_started": len(not_started),
            "not_started_names": not_started,
            "planned": EXPECTED_SCENARIOS,
        },
        "checkpoint_inventory": inventory,
        "execution_matrix": {
            "path": MATRIX.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(matrix_bytes).hexdigest(),
            "rows": len(matrix_rows),
            "rows_per_completed_checkpoint": METHODS_PER_SCENARIO,
            "completed_scenario_count": len(complete_matrix_scenarios),
            "source_checkpoint_hashes_all_match": True,
            "status_counts": status_counts,
            "individually_passing_method_regime_count": len(passing),
            "individually_passing_method_regimes": [
                {"regime": row["process"], "method": row["method"], "frozen_preselected": _bool(row.get("frozen_selected_method"))}
                for row in passing
            ],
            "frozen_preselected_validation_rows": selected_methods,
            "true_slope_values_in_completed_matrix": slopes,
        },
        "selection_decisions": {
            "path": SELECTION.relative_to(ROOT).as_posix(),
            "sha256": sha256(SELECTION),
            "status": selection.get("status"),
            "protocol_sha256": selection.get("protocol_sha256"),
            "seed_bases": {"selection": selection.get("selection_seed_base"), "validation": selection.get("validation_seed_base")},
            "frozen_selected_methods": selected_methods,
            "primary_regimes_without_a_frozen_selection": sorted(
                regime for regime, record in selection.get("decisions", {}).items() if not record.get("selected")
            ),
        },
        "validated_registry": {
            "path": REGISTRY.relative_to(ROOT).as_posix(),
            "sha256": sha256(REGISTRY),
            "status": registry.get("status"),
            "promoted_regime_count": len(registry_entries),
        },
        "correction_manifest": {
            "path": CORRECTION_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": sha256(CORRECTION_MANIFEST),
            "status": read_json(CORRECTION_MANIFEST).get("status"),
            "inferential_real_series_rows_released": read_json(CORRECTION_MANIFEST).get("inferential_real_series_rows_released"),
        },
        "frozen_2025_hindcast": {
            "path": SCORE_2025.relative_to(ROOT).as_posix(),
            "sha256": sha256(SCORE_2025),
            "preserved_without_rescoring": True,
            "negative_result_retained": True,
        } if SCORE_2025.is_file() else {"status": "NOT_FOUND"},
        "closure_script_sha256": sha256(validator_source),
        "interpretation": [
            "The legacy calibration manifest is a self-consistent earlier snapshot at 7/60; it is not the current checkpoint inventory.",
            "Three complete checkpoints and one 50/1000 checkpoint were still listed as missing by the legacy manifest.",
            "The 160-row matrix has exactly 16 rows for each of the 10 complete, hash-verified checkpoints; the in-progress checkpoint has no matrix rows.",
            "Both frozen-selected seasonal-MK methods fail their independent Type-I gate; the 18 individually passing rows are all unselected and do not qualify for promotion.",
            "Only zero-slope scenarios are in the completed matrix, so alternative-slope coverage, bias, and power are not independently established.",
            "The current validation is closed as partial. Frozen criteria, seeds, selection decisions, validated registry, and 2025 hindcast are not rewritten.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true", help="validate and print counts without writing the closure manifest")
    args = parser.parse_args()
    payload = reconcile()
    print(json.dumps({
        "closure_status": payload["closure_status"],
        "legacy_manifest_scenarios": payload["legacy_manifest"]["recorded_complete_scenarios"],
        "actual_complete": payload["actual_checkpoint_counts"]["complete_hash_verified"],
        "actual_in_progress": payload["actual_checkpoint_counts"]["in_progress_hash_verified"],
        "actual_not_started": payload["actual_checkpoint_counts"]["not_started"],
        "matrix_rows": payload["execution_matrix"]["rows"],
        "individual_passes": payload["execution_matrix"]["individually_passing_method_regime_count"],
        "promoted": payload["validated_registry"]["promoted_regime_count"],
    }, indent=2))
    if not args.check_only:
        if args.output.exists():
            raise FileExistsError(f"Refusing to overwrite closure artifact: {args.output}")
        atomic_json(args.output, payload)
        print(f"CLOSURE_MANIFEST={args.output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
