from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from datetime import datetime, timezone
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.statistical_correction.freeze_interval_comparison_v1_3_2 import verify_payload
from scripts.statistical_correction.provenance_interval_comparison_v1_3_2 import FREEZE_PATH, RESULTS, sha256_file, source_hashes
from scripts.statistical_correction.run_interval_comparison_v1_3_2 import checkpoint_path, load_config
from scripts.statistical_correction.summarize_corrective_confirmation_v1_3 import cp_interval, mean_summary, summarize_cell
from scripts.statistical_correction.interval_comparison_v1_3_2 import METHODS


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_frozen() -> tuple[dict[str, Any], dict[str, Any], str]:
    config = load_config()
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    verify_payload(freeze)
    return config, freeze, sha256_file(FREEZE_PATH)


def load_rows(phase: str, config: dict[str, Any], freeze_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    run_dir = RESULTS / phase
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("phase") != phase or manifest.get("protocol_freeze_sha256") != freeze_hash:
        raise RuntimeError(f"{phase} run manifest does not match the frozen protocol")
    if manifest.get("source_hashes") != source_hashes():
        raise RuntimeError(f"{phase} sources changed after the run")
    rows: list[dict[str, Any]] = []
    checkpoint_inventory: list[dict[str, Any]] = []
    for slope in config["design"]["slopes_per_decade"]:
        path = checkpoint_path(run_dir, float(slope))
        if not path.is_file():
            continue
        cell = json.loads(path.read_text(encoding="utf-8"))
        digest = sha256_file(path)
        if cell.get("protocol_freeze_sha256") != freeze_hash or cell.get("source_hashes") != source_hashes():
            raise RuntimeError(f"checkpoint hash lineage does not match the frozen protocol: {path.name}")
        if cell.get("cell", {}).get("slope_per_decade") != float(slope):
            raise RuntimeError(f"checkpoint slope mismatch: {path.name}")
        rows.extend(cell.get("replicates", []))
        checkpoint_inventory.append({
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "bytes": path.stat().st_size,
            "completed_paired_replicates": int(cell.get("completed_paired_replicates", 0)),
            "status": cell.get("status"),
        })
    return rows, manifest, checkpoint_inventory


def summarize_preflight(config: dict[str, Any], freeze_hash: str, rows: list[dict[str, Any]], manifest: dict[str, Any], checkpoints: list[dict[str, Any]]) -> dict[str, Any]:
    reps_per_slope = int(config["design"]["preflight_replicates_per_slope"])
    expected = reps_per_slope * len(config["design"]["slopes_per_decade"]) * len(METHODS)
    complete = manifest.get("status") == "COMPLETE" and len(rows) == expected and all(item["status"] == "COMPLETE" for item in checkpoints)
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in rows:
        if row.get("method") not in by_method:
            raise RuntimeError("Unexpected method in preflight checkpoint")
        by_method[row["method"]].append(row)
    per_method: dict[str, Any] = {}
    projected_cpu = projected_wall = 0.0
    minimum_availability = float(config["resource_budget"]["preflight_minimum_candidate_inference_availability"])
    future_candidate_replicates = int(config["design"]["development_replicates_per_slope"]) * len(config["design"]["slopes_per_decade"])
    for method in METHODS:
        method_rows = by_method[method]
        if len(method_rows) != reps_per_slope * len(config["design"]["slopes_per_decade"]):
            complete = False
        cpu = [float(row["cpu_seconds"]) for row in method_rows if row.get("cpu_seconds") is not None]
        wall = [float(row["wall_seconds"]) for row in method_rows if row.get("wall_seconds") is not None]
        cpu_mean = float(sum(cpu) / len(cpu)) if cpu else None
        wall_mean = float(sum(wall) / len(wall)) if wall else None
        cpu_ci = mean_summary(cpu) if cpu else None
        wall_ci = mean_summary(wall) if wall else None
        candidates = sum(1 for row in method_rows if row.get("converged"))
        inference_rows = sum(1 for row in method_rows if row.get("inferential_output"))
        metric = {
            "candidate_replicates": len(method_rows),
            "point_model_converged": candidates,
            "convergence_fraction": candidates / len(method_rows) if method_rows else None,
            "inferential_rows": inference_rows,
            "inferential_availability": inference_rows / len(method_rows) if method_rows else None,
            "profile_likelihood_evaluations_mean": float(sum(int(row.get("profile_likelihood_evaluations") or 0) for row in method_rows) / len(method_rows)) if method_rows else None,
            "mean_candidate_cpu_seconds": cpu_ci,
            "mean_candidate_wall_seconds": wall_ci,
            "projected_cpu_seconds_for_3000": cpu_mean * future_candidate_replicates if cpu_mean is not None else None,
            "projected_wall_seconds_for_3000": wall_mean * future_candidate_replicates if wall_mean is not None else None,
        }
        per_method[method] = metric
        if cpu_mean is not None:
            projected_cpu += cpu_mean * future_candidate_replicates
        if wall_mean is not None:
            projected_wall += wall_mean * future_candidate_replicates
    multiplier = float(config["resource_budget"]["preflight_projection_multiplier"])
    reserved_cpu = projected_cpu * multiplier
    reserved_wall = projected_wall * multiplier
    cpu_cap = int(config["resource_budget"]["development_hard_max_cpu_seconds"])
    wall_cap = int(config["resource_budget"]["development_hard_max_wall_seconds"])
    technical_pass = complete and all(
        per_method[method]["inferential_availability"] is not None
        and per_method[method]["inferential_availability"] >= minimum_availability
        for method in METHODS
    )
    resource_pass = reserved_cpu <= cpu_cap and reserved_wall <= wall_cap
    decision = "PROCEED" if complete and technical_pass and resource_pass else "NOT_VALIDATED_PREFLIGHT_NUMERICAL_FAILURE" if complete and not technical_pass else "NOT_VALIDATED_RESOURCE_BOUNDED" if complete else "INCOMPLETE_PREFLIGHT"
    return {
        "schema_version": "kuntur-v1.3.2-interval-preflight-summary-v1",
        "protocol_id": config["protocol_id"],
        "phase": "preflight",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": manifest.get("status"),
        "decision": decision,
        "protocol_freeze_sha256": freeze_hash,
        "completed_candidate_replicates": len(rows),
        "expected_candidate_replicates": expected,
        "resource_usage": manifest.get("resource_usage"),
        "candidate_runtime": per_method,
        "preflight_integrity_gate": {
            "minimum_inferential_availability": minimum_availability,
            "all_candidates_meet_gate": technical_pass,
            "interpretation": "Execution-readiness check only; not a scientific acceptance gate or method-selection result.",
        },
        "projected_development": {
            "planned_candidate_replicates": int(config["resource_budget"]["development_candidate_replicates"]),
            "unreserved_cpu_seconds": projected_cpu,
            "unreserved_active_wall_seconds": projected_wall,
            "reserve_multiplier": multiplier,
            "reserved_cpu_seconds": reserved_cpu,
            "reserved_active_wall_seconds": reserved_wall,
            "cpu_hard_cap_seconds": cpu_cap,
            "active_wall_hard_cap_seconds": wall_cap,
        },
        "checkpoint_inventory": checkpoints,
        "protected_inputs_unchanged": True,
    }


def summarize_development(config: dict[str, Any], freeze_hash: str, rows: list[dict[str, Any]], manifest: dict[str, Any], checkpoints: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    reps = int(config["design"]["development_replicates_per_slope"])
    expected = reps * len(config["design"]["slopes_per_decade"]) * len(METHODS)
    full = manifest.get("status") == "COMPLETE" and len(rows) == expected and all(item["status"] == "COMPLETE" for item in checkpoints)
    indexed: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["method"]), float(row["slope_per_decade"]))
        indexed.setdefault(key, []).append(row)
    cell_metrics: list[dict[str, Any]] = []
    for slope in [float(value) for value in config["design"]["slopes_per_decade"]]:
        pairing: dict[str, list[str]] = {}
        for method in METHODS:
            method_rows = indexed.get((method, slope), [])
            method_rows.sort(key=lambda item: int(item["replicate"]))
            pairing[method] = [str(row["series_seed"]) for row in method_rows]
            expected_ids = list(range(len(method_rows)))
            if [int(row["replicate"]) for row in method_rows] != expected_ids:
                raise RuntimeError(f"Replicate IDs are missing or duplicated: {method} slope {slope}")
        if len(set(tuple(v) for v in pairing.values())) != 1:
            raise RuntimeError(f"Candidate comparisons are not paired at slope {slope}")
        for method in METHODS:
            method_rows = indexed.get((method, slope), [])
            if len(method_rows) != reps:
                full = False
            metrics = summarize_cell({"cell": {"slope_per_decade": slope, "method_name": method}, "replicates": method_rows}, 0.05, 0.95)
            metrics["replications_requested"] = reps
            metrics["replications_completed"] = len(method_rows)
            metrics["profile_likelihood_evaluations_mean"] = float(sum(int(row.get("profile_likelihood_evaluations") or 0) for row in method_rows) / len(method_rows)) if method_rows else None
            cell_metrics.append(metrics)

    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for row in cell_metrics:
        by_method[row["method"]].append(row)
    method_decisions: dict[str, Any] = {}
    corrected_pass = []
    for method in METHODS:
        method_cells = sorted(by_method[method], key=lambda row: row["slope_per_decade"])
        passes = len(method_cells) == 3 and all(row["all_gates_pass"] for row in method_cells)
        eligible = method != "wald_normal" and passes
        if eligible:
            corrected_pass.append(method)
        method_decisions[method] = {
            "all_three_slopes_pass": passes,
            "eligible_corrected_candidate": eligible,
            "per_slope_gate_results": [{"slope_per_decade": row["slope_per_decade"], "gates": row["acceptance_gates"], "all_gates_pass": row["all_gates_pass"]} for row in method_cells],
        }
    selected = next((method for method in ("wald_student_t", "profile_likelihood") if method in corrected_pass), None)
    if full and selected:
        decision = "CANDIDATE_ELIGIBLE_FOR_FREEZE_AND_INDEPENDENT_CONFIRMATION"
        status = "COMPLETE_DEVELOPMENT_ONLY"
    elif full:
        decision = "NOT_VALIDATED"
        status = "COMPLETE_DEVELOPMENT_ONLY"
    else:
        decision = "NOT_VALIDATED_RESOURCE_BOUNDED" if manifest.get("status") == "BOUNDED_PARTIAL" else "INCOMPLETE_DEVELOPMENT"
        status = manifest.get("status")

    metrics = {
        "schema_version": "kuntur-v1.3.2-interval-development-metrics-v1",
        "protocol_id": config["protocol_id"],
        "status": status,
        "decision": decision,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_freeze_sha256": freeze_hash,
        "scope": {"regime": "iid_gaussian", "calendar_months": 276, "slopes_per_decade": config["design"]["slopes_per_decade"], "replications_per_slope_candidate": reps},
        "candidate_selection": {
            "wald_normal": "UNCHANGED_COMPARATOR_ONLY",
            "corrected_candidates_passing_all_gates": corrected_pass,
            "development_candidate_nominated_for_independent_freeze": selected,
            "independent_confirmation_started": False,
            "inference_registry_modified": False,
        },
        "candidate_gate_decisions": method_decisions,
        "cells": cell_metrics,
        "resource_usage": manifest.get("resource_usage"),
        "checkpoint_inventory": checkpoints,
        "protected_inputs": {
            "v1_2_validation_or_selection_changed": False,
            "v1_3_608_fit_pilot_changed_or_repeated": False,
            "v1_3_280000_fit_development_expansion_changed_or_repeated": False,
            "prior_v1_3_independent_confirmation_changed_or_reused": False,
            "2025_predictive_test_rerun": False,
            "real_gold_used_for_selection": False,
        },
    }
    report = build_development_report(config, metrics)
    return metrics, report


def build_development_report(config: dict[str, Any], metrics: dict[str, Any]) -> str:
    lines = [
        "# KUNTUR v1.3.2 IID Interval Development Comparison",
        "",
        f"**Status:** `{metrics['status']}`; decision `{metrics['decision']}`. Development evidence only; no inference registry entry is created.",
        "",
        f"Protocol: `{config['protocol_id']}`; freeze SHA-256: `{metrics['protocol_freeze_sha256']}`.",
        "",
        "## Frozen design",
        "",
        f"- Regime: `{metrics['scope']['regime']}`, 276 complete monthly observations, January 2000–December 2022.",
        f"- Slopes: {', '.join(f'{float(value):+.2f}' for value in metrics['scope']['slopes_per_decade'])} per decade.",
        f"- Replications: {metrics['scope']['replications_per_slope_candidate']:,} paired series per slope; {len(METHODS)} interval candidates.",
        "- Every candidate independently fits the same seasonal SARIMAX point model. Paired candidates use the same generated series at each slope and replicate.",
        "- The predeclared development gates are unchanged from the frozen v1.3 confirmation; Monte Carlo uncertainty uses exact Clopper–Pearson intervals for rates and Student-t intervals for means.",
        "",
        "## Development metrics",
        "",
        "| Method | Slope | Type-I (CP95) / power (CP95) | Coverage (CP95) | Bias (t-MC95) | Mean CI width / oracle | Convergence failures | Availability | Ljung–Box failures | Gates |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in metrics["cells"]:
        slope = float(cell["slope_per_decade"])
        reject = cell["type_i_rate"] if slope == 0 else cell["power"]
        reject_ci = cell["type_i_cp95"] if slope == 0 else cell["power_cp95"]
        rejection_text = "—" if reject is None else f"{reject:.4f} [{reject_ci[0]:.4f}, {reject_ci[1]:.4f}]"
        coverage_ci = cell["coverage_cp95"]
        coverage_text = "—" if cell["coverage_rate"] is None else f"{cell['coverage_rate']:.4f} [{coverage_ci[0]:.4f}, {coverage_ci[1]:.4f}]"
        bias = cell["slope_bias_per_decade"]
        bias_text = "—" if bias is None else f"{bias['mean']:.5f} [{bias['ci95_t'][0]:.5f}, {bias['ci95_t'][1]:.5f}]"
        residual = "—" if cell["residual_ljung_box_failure_rate"] is None else f"{cell['residual_ljung_box_failures']}/{cell['residual_ljung_box_tested_count']} ({cell['residual_ljung_box_failure_rate']:.3f})"
        ratio = cell["mean_ci_width_over_oracle"]
        width_text = "—" if ratio is None else f"{ratio['mean']:.3f}"
        numeric = f"{cell['numerical_failures']}/{cell['supported_count']}"
        gates = "PASS" if cell["all_gates_pass"] else "FAIL"
        lines.append(f"| `{cell['method']}` | {slope:+.2f} | {rejection_text} | {coverage_text} | {bias_text} | {width_text} | {numeric} | {cell['inferential_availability']:.3f} | {residual} | {gates} |")
    lines.extend([
        "",
        "## Method-selection decision",
        "",
        f"- Unchanged Wald comparator: not eligible for a corrected-method freeze. All three slopes pass: `{metrics['candidate_gate_decisions']['wald_normal']['all_three_slopes_pass']}`.",
        f"- Corrected candidates passing every applicable gate across all three slopes: `{metrics['candidate_selection']['corrected_candidates_passing_all_gates']}`.",
        f"- Candidate nominated for a separate independent-confirmation freeze: `{metrics['candidate_selection']['development_candidate_nominated_for_independent_freeze']}`.",
        "- Passing development gates would make a candidate eligible for a separately frozen independent confirmation. It would not establish validity or authorize registry promotion.",
        "- The inference registry was not changed. Other dependence/missingness regimes and real NASA series remain outside this IID-only comparison.",
        "",
        "## Resources and artifacts",
        "",
        f"- Resources: {metrics['resource_usage']['cumulative_process_cpu_seconds']:.1f} CPU s; {metrics['resource_usage']['cumulative_active_wall_seconds']:.1f} active wall s; {metrics['resource_usage']['peak_rss_bytes_observed']:,} peak RSS bytes.",
        "- Atomic checkpoints were written per 50 paired replications and their SHA-256 values are recorded in the metrics manifest.",
        "- Earlier v1.2 decisions, the 608-fit pilot, 180,000-fit expansion, v1.3 independent-confirmation failure, Gold data, and 2025 hindcast remain preserved and unchanged.",
        "",
    ])
    return "\n".join(lines)


def run(phase: str) -> dict[str, Any]:
    config, _freeze, freeze_hash = load_frozen()
    rows, manifest, checkpoints = load_rows(phase, config, freeze_hash)
    if phase == "preflight":
        summary = summarize_preflight(config, freeze_hash, rows, manifest, checkpoints)
        output_path = RESULTS / "preflight" / "preflight_summary.json"
        write_json(output_path, summary)
        print(json.dumps({"status": summary["status"], "decision": summary["decision"], "completed_candidate_replicates": summary["completed_candidate_replicates"], "projected_development": summary["projected_development"], "summary": output_path.relative_to(ROOT).as_posix()}, indent=2))
        return summary

    metrics, report = summarize_development(config, freeze_hash, rows, manifest, checkpoints)
    results_path = RESULTS / "development" / "development_metrics.json"
    report_path = RESULTS / "development" / "development_report.md"
    write_json(results_path, metrics)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text(report, encoding="utf-8")
    temporary.replace(report_path)
    report_hash = sha256_file(report_path)
    result_hash = sha256_file(results_path)
    receipt = {
        "schema_version": "kuntur-v1.3.2-interval-development-summary-receipt-v1",
        "decision": metrics["decision"],
        "metrics_path": results_path.relative_to(ROOT).as_posix(),
        "metrics_sha256": result_hash,
        "report_path": report_path.relative_to(ROOT).as_posix(),
        "report_sha256": report_hash,
        "protocol_freeze_sha256": freeze_hash,
        "source_hashes": source_hashes(),
    }
    write_json(RESULTS / "development" / "summary_receipt.json", receipt)
    print(json.dumps({"status": metrics["status"], "decision": metrics["decision"], "candidate_selection": metrics["candidate_selection"], "metrics_sha256": result_hash, "report_sha256": report_hash, "metrics": results_path.relative_to(ROOT).as_posix(), "report": report_path.relative_to(ROOT).as_posix()}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("preflight", "development"))
    args = parser.parse_args()
    run(args.phase)


if __name__ == "__main__":
    main()
