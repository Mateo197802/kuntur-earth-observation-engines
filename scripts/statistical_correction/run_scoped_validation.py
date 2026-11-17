from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

try:
    from scripts.statistical_correction import run_calibration as calibration
except ModuleNotFoundError:
    from statistical_correction import run_calibration as calibration


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "results/statistical_correction"
SCOPE_PATH = OUT_ROOT / "validation/validation_scope_v1_2.json"
VALIDATION_SLOPES = (0.0, 0.1, 0.25, 0.5, -0.5)
OUTER_REPLICATIONS = 1000
BOOTSTRAP_INNER = 399
MAX_WALL_SECONDS = 6 * 60 * 60


def _case_key(case: tuple[str, int, float, int, int]) -> tuple[str, int, float]:
    return str(case[0]), int(case[1]), float(case[2])


def selected_primary_processes(decisions: dict[str, Any]) -> list[str]:
    selected = decisions.get("decisions", {})
    return [
        process for process in calibration.PRIMARY_PROCESSES
        if selected.get(process, {}).get("selected") is not None
    ]


def validation_plan(decisions: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the frozen independent scope from selection-eligible primary regimes."""
    selected_processes = selected_primary_processes(decisions)
    if not selected_processes:
        raise ValueError("No primary process passed selection; there is no candidate to independently validate.")

    included: list[dict[str, Any]] = []
    for process in selected_processes:
        for slope in VALIDATION_SLOPES:
            included.append({
                "process": process,
                "n_months": 276,
                "true_slope_per_decade": slope,
                "outer_replications": OUTER_REPLICATIONS,
                "bootstrap_inner_draws": BOOTSTRAP_INNER,
                "seed": calibration.scenario_seed("validation", process, slope, 276),
            })

    included_keys = {
        (case["process"], case["n_months"], float(case["true_slope_per_decade"]))
        for case in included
    }
    deferred: list[dict[str, Any]] = []
    selection_map = decisions.get("decisions", {})
    for process, n_months, slope, reps, inner in calibration._scenario_plan("validation"):
        key = (process, int(n_months), float(slope))
        if key in included_keys:
            continue
        if process in calibration.PRIMARY_PROCESSES and n_months == 276 and slope in VALIDATION_SLOPES:
            reason = "NO_SELECTION_CANDIDATE_PASSED_SELECTION_GATES" if selection_map.get(process, {}).get("selected") is None else "OUTSIDE_FROZEN_SCOPE"
        elif n_months != 276:
            reason = "SHORT_SERIES_SENSITIVITY_DEFERRED"
        else:
            reason = "SECONDARY_DGP_SCOPE_DEFERRED"
        deferred.append({
            "process": process,
            "n_months": int(n_months),
            "true_slope_per_decade": float(slope),
            "protocol_outer_replications": int(reps),
            "protocol_bootstrap_inner_draws": int(inner),
            "reason": reason,
        })
    return included, deferred


def _scope_payload(decisions: dict[str, Any], selection_manifest: dict[str, Any], selection_phase_manifest: dict[str, Any]) -> dict[str, Any]:
    included, deferred = validation_plan(decisions)
    selected_processes = selected_primary_processes(decisions)
    selected_methods = {
        process: decisions["decisions"][process]["selected"]["method"]
        for process in selected_processes
    }
    protocol_path = ROOT / "docs/protocols/statistical_correction_v1_2.md"
    runner_path = ROOT / "scripts/statistical_correction/run_calibration.py"
    module_path = ROOT / "src/kuntur_data/phase2/statistical_v12.py"
    return {
        "schema_version": "kuntur-statistical-validation-scope-v1.2",
        "status": "FROZEN_BEFORE_INDEPENDENT_VALIDATION",
        "scope_id": "selected_primary_regimes_n276_all_effects_v1_2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "status": selection_manifest.get("status"),
            "selection_manifest_sha256": calibration.sha256(OUT_ROOT / "selection/selection_manifest.json"),
            "selection_decisions_sha256": calibration.sha256(OUT_ROOT / "selection/selection_decisions.json"),
            "selection_matrix_sha256": calibration.sha256(OUT_ROOT / "selection/method_validation_matrix.csv"),
            "selection_phase_manifest_sha256": calibration.sha256(OUT_ROOT / "selection/calibration_manifest.json"),
            "selected_processes": selected_processes,
            "selected_method_by_process": selected_methods,
            "processes_with_no_selection_candidate": [
                process for process in calibration.PRIMARY_PROCESSES if process not in selected_processes
            ],
        },
        "frozen_code_and_protocol": {
            "protocol_sha256": calibration.sha256(protocol_path),
            "run_calibration_sha256": calibration.sha256(runner_path),
            "statistical_module_sha256": calibration.sha256(module_path),
            "scope_runner_sha256": calibration.sha256(Path(__file__).resolve()),
        },
        "execution_scope": {
            "reason": "Independently validate only primary regimes with a method that passed the frozen selection gates; regimes with no selected candidate cannot be promoted. For included regimes, validate every locked candidate method to retain fair independent comparison.",
            "all_locked_candidate_methods_evaluated": True,
            "candidate_method_count": len(calibration.METHODS),
            "included_scenario_count": len(included),
            "included_scenarios": included,
            "deferred_full_protocol_scenario_count": len(deferred),
            "deferred_full_protocol_scenarios": deferred,
            "full_protocol_scenario_count": len(calibration._scenario_plan("validation")),
            "primary_regime_claims_restricted_to": selected_processes,
            "unselected_primary_regimes_remain": "NO_SELECTION_CANDIDATE_PASSED_SELECTION_GATES; no inferential promotion",
            "secondary_and_short_series_regimes_remain": "NOT_VALIDATED",
        },
        "validation_configuration": {
            "validation_seed_base": 20261127,
            "outer_replications_per_included_scenario": OUTER_REPLICATIONS,
            "bootstrap_inner_draws": BOOTSTRAP_INNER,
            "max_wall_seconds": MAX_WALL_SECONDS,
            "scenario_seed_function": "scenario_seed(phase='validation', process, slope, n_months)",
            "slope_units": "units per decade",
            "methods": [spec["name"] for spec in calibration.METHODS],
            "confidence_interval_gate": "Not applicable to seasonal MK methods that do not return an interval; report marginal calibration, effect metrics, failures, and available power only.",
        },
        "limitations": [
            "This is a predeclared scope subset of the 60-scenario full protocol matrix, not a claim that the full matrix was completed.",
            "No method is valid outside its exact validated process, 276-month length, effect, and temporal-support regime.",
            "No real NASA outcome or p-value was used to choose the scope or method.",
        ],
    }


def freeze_scope() -> dict[str, Any]:
    selection_dir = OUT_ROOT / "selection"
    decisions = calibration._verify_frozen_selection(selection_dir)
    selection_manifest = json.loads((selection_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    phase_manifest = json.loads((selection_dir / "calibration_manifest.json").read_text(encoding="utf-8"))
    if phase_manifest.get("status") != "COMPLETED_PHASE" or int(phase_manifest.get("scenario_count", 0)) != 10:
        raise RuntimeError("The complete ten-scenario selection phase is required before freezing validation scope.")
    if phase_manifest.get("script_sha256") != calibration.sha256(ROOT / "scripts/statistical_correction/run_calibration.py"):
        raise RuntimeError("Calibration runner changed after selection; refusing to freeze a mismatched scope.")
    if phase_manifest.get("statistical_module_sha256") != calibration.sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py"):
        raise RuntimeError("Statistical module changed after selection; refusing to freeze a mismatched scope.")
    payload = _scope_payload(decisions, selection_manifest, phase_manifest)
    calibration.atomic_json(SCOPE_PATH, payload)
    return payload


def _load_and_verify_scope() -> tuple[dict[str, Any], list[tuple[str, int, float, int, int]]]:
    if not SCOPE_PATH.is_file():
        raise FileNotFoundError("Validation scope must be frozen before independent validation starts.")
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    if scope.get("status") != "FROZEN_BEFORE_INDEPENDENT_VALIDATION":
        raise RuntimeError("Validation scope is not frozen.")
    selection_dir = OUT_ROOT / "selection"
    decisions = calibration._verify_frozen_selection(selection_dir)
    selection_manifest_path = selection_dir / "selection_manifest.json"
    phase_manifest_path = selection_dir / "calibration_manifest.json"
    expected_hashes = {
        "selection_manifest_sha256": calibration.sha256(selection_manifest_path),
        "selection_decisions_sha256": calibration.sha256(selection_dir / "selection_decisions.json"),
        "selection_matrix_sha256": calibration.sha256(selection_dir / "method_validation_matrix.csv"),
        "selection_phase_manifest_sha256": calibration.sha256(phase_manifest_path),
    }
    for key, expected in expected_hashes.items():
        if scope.get("selection", {}).get(key) != expected:
            raise RuntimeError(f"Frozen selection artifact changed after validation scope creation: {key}")
    expected_code = {
        "protocol_sha256": calibration.sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
        "run_calibration_sha256": calibration.sha256(ROOT / "scripts/statistical_correction/run_calibration.py"),
        "statistical_module_sha256": calibration.sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py"),
        "scope_runner_sha256": calibration.sha256(Path(__file__).resolve()),
    }
    for key, expected in expected_code.items():
        if scope.get("frozen_code_and_protocol", {}).get(key) != expected:
            raise RuntimeError(f"Validation scope code/protocol hash mismatch: {key}")
    current_cases, current_deferred = validation_plan(decisions)
    if scope.get("execution_scope", {}).get("included_scenarios") != current_cases:
        raise RuntimeError("Recomputed validation cases do not match the frozen scope.")
    if scope.get("execution_scope", {}).get("deferred_full_protocol_scenarios") != current_deferred:
        raise RuntimeError("Recomputed deferred scenarios do not match the frozen scope.")
    plan = [
        (
            str(case["process"]), int(case["n_months"]), float(case["true_slope_per_decade"]),
            int(case["outer_replications"]), int(case["bootstrap_inner_draws"]),
        )
        for case in current_cases
    ]
    return scope, plan


def _write_scoped_manifest(scope: dict[str, Any], plan: list[tuple[str, int, float, int, int]], start: float, budget_exhausted: bool) -> dict[str, Any]:
    output_dir = OUT_ROOT / "validation"
    expected_paths = {
        calibration._scenario_checkpoint_path(output_dir, "validation", process, n_months, slope).name
        for process, n_months, slope, _reps, _inner in plan
    }
    complete_paths: set[str] = set()
    checkpoint_dir = output_dir / "checkpoints"
    if checkpoint_dir.exists():
        complete_paths = {
            path.name for path in checkpoint_dir.glob("validation_*.json")
            if json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETE"
        }
    missing_scope = sorted(expected_paths - complete_paths)
    scope_complete = not missing_scope
    all_protocol_cases = calibration._scenario_plan("validation")
    scope_keys = {_case_key(case) for case in plan}
    deferred_protocol = [case for case in all_protocol_cases if _case_key(case) not in scope_keys]
    if complete_paths:
        base = calibration._write_phase_results(output_dir, "validation", expected_cases=plan)
    else:
        base = {
            "phase": "validation",
            "status": "PARTIAL_PREDECLARED_SCOPE",
            "execution_scope": "SELECTED_PRIMARY_REGIMES_N276_ALL_EFFECTS",
            "planned_scenario_count": len(plan),
            "scenario_count": 0,
            "rows": 0,
            "completed_checkpoints": 0,
            "missing_scenarios": sorted(expected_paths),
            "outputs": {},
        }
    base.update({
        "status": "COMPLETED_PREDECLARED_SCOPE" if scope_complete else "PARTIAL_PREDECLARED_SCOPE",
        "execution_scope": scope["scope_id"],
        "scope_complete": scope_complete,
        "scope_record_path": SCOPE_PATH.relative_to(ROOT).as_posix(),
        "scope_record_sha256": calibration.sha256(SCOPE_PATH),
        "scope_planned_scenario_count": len(plan),
        "scope_completed_scenario_count": len(expected_paths & complete_paths),
        "scope_missing_scenarios": missing_scope,
        "full_protocol_status": "INCOMPLETE_OUTSIDE_PREDECLARED_SCOPE" if deferred_protocol else "COMPLETED",
        "full_protocol_scenario_count": len(all_protocol_cases),
        "full_protocol_deferred_scenario_count": len(deferred_protocol),
        "full_protocol_deferred_scenarios": scope["execution_scope"]["deferred_full_protocol_scenarios"],
        "full_protocol_missing_scenarios": [
            calibration._scenario_checkpoint_path(output_dir, "validation", *case[:3]).name
            for case in deferred_protocol
        ],
        "max_wall_seconds": int(scope["validation_configuration"]["max_wall_seconds"]),
        "run_wall_seconds": round(time.monotonic() - start, 3),
        "wall_budget_exhausted": bool(budget_exhausted),
        "limitations": [
            "Only the exact selection-eligible primary regimes and 276-month support in the frozen scope are evaluated.",
            "All other primary, secondary, and short-series regimes remain NOT_VALIDATED.",
            "Structural-break and random-walk processes are stress diagnostics, not stationary zero-slope calibration claims.",
        ],
    })
    calibration.atomic_json(output_dir / "calibration_manifest.json", base)
    return base


def run_validation() -> dict[str, Any]:
    scope, plan = _load_and_verify_scope()
    output_dir = OUT_ROOT / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    deadline = start + int(scope["validation_configuration"]["max_wall_seconds"])
    budget_exhausted = False
    try:
        for index, (process, n_months, slope, reps, inner) in enumerate(plan, start=1):
            checkpoint = calibration._scenario_checkpoint_path(output_dir, "validation", process, n_months, slope)
            if checkpoint.exists():
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                if payload.get("status") == "COMPLETE":
                    print(f"SCENARIO_REUSED phase=validation process={process} n={n_months} slope={slope:g} {index}/{len(plan)}", flush=True)
                    continue
            if time.monotonic() >= deadline:
                budget_exhausted = True
                break
            rows = calibration._run_scenario("validation", process, n_months, slope, reps, inner, output_dir)
            print(f"SCENARIO_COMPLETE phase=validation process={process} n={n_months} slope={slope:g} methods={len(rows)} {index}/{len(plan)}", flush=True)
    finally:
        result = _write_scoped_manifest(scope, plan, start, budget_exhausted)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and run the predeclared selection-eligible v1.2 validation scope.")
    parser.add_argument("--freeze-scope", action="store_true", help="Freeze the validation scope from the completed selection artifacts; do not run validation.")
    parser.add_argument("--run", action="store_true", help="Run or safely resume the frozen independent validation scope.")
    args = parser.parse_args()
    if args.freeze_scope == args.run:
        parser.error("choose exactly one of --freeze-scope or --run")
    if args.freeze_scope:
        result = freeze_scope()
        print(json.dumps({
            "status": result["status"],
            "scope_path": SCOPE_PATH.relative_to(ROOT).as_posix(),
            "scope_scenarios": result["execution_scope"]["included_scenario_count"],
            "deferred_full_protocol_scenarios": result["execution_scope"]["deferred_full_protocol_scenario_count"],
            "selected_method_by_process": result["execution_scope"]["selected_method_by_process"],
            "scope_runner_sha256": result["frozen_code_and_protocol"]["scope_runner_sha256"],
        }, indent=2, sort_keys=True))
    else:
        run_validation()


if __name__ == "__main__":
    main()
