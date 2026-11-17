from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import beta, t as student_t

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.statistical_correction.freeze_interval_confirmation_v1_3_2 import (
    CONFIG_PATH,
    FREEZE_PATH,
    RESULTS,
    ROOT as PROJECT_ROOT,
    seed_for,
    seed_inventory,
    sha256_file,
    source_hashes,
    verify_payload,
)
from scripts.statistical_correction.run_interval_confirmation_v1_3_2 import checkpoint_path

RUN_DIR = RESULTS / "confirmation"
MANIFEST_PATH = RUN_DIR / "run_manifest.json"
METRICS_PATH = RUN_DIR / "confirmation_metrics.json"
REPORT_PATH = RUN_DIR / "confirmation_report.md"
RECEIPT_PATH = RUN_DIR / "summary_receipt.json"


def cp_interval(successes: int, trials: int) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    lower = 0.0 if successes == 0 else float(beta.ppf(0.025, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(0.975, successes + 1, trials - successes))
    return lower, upper


def mean_summary(values: list[float]) -> dict[str, Any] | None:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    mean = float(np.mean(array))
    if len(array) < 2:
        return {"n": len(array), "mean": mean, "mcse": None, "ci95_t": None}
    mcse = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    margin = float(student_t.ppf(0.975, len(array) - 1) * mcse)
    return {"n": len(array), "mean": mean, "mcse": mcse, "ci95_t": [mean - margin, mean + margin]}


def load_verified_rows() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError("Confirmation run manifest is missing")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    verify_payload(frozen)
    freeze_hash = sha256_file(FREEZE_PATH)
    sources = source_hashes()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE":
        raise RuntimeError("Confirmation run is not complete; no acceptance decision can be made")
    if manifest.get("protocol_freeze_sha256") != freeze_hash or manifest.get("source_hashes") != sources:
        raise RuntimeError("Confirmation manifest does not match the frozen protocol and sources")
    if manifest.get("seed_inventory") != seed_inventory(config):
        raise RuntimeError("Confirmation manifest seed inventory differs from the frozen protocol")
    rows: list[dict[str, Any]] = []
    seen_seeds: set[str] = set()
    for slope in config["design"]["slopes_per_decade"]:
        slope = float(slope)
        path = checkpoint_path(slope)
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        expected_cell = {
            "phase": "confirmation",
            "regime": config["scope"]["regime"],
            "slope_per_decade": slope,
            "replicates": int(config["design"]["replicates_per_slope"]),
            "seed_namespace": config["design"]["series_seed_namespace"],
            "candidate_method": config["scope"]["candidate_method"],
        }
        if checkpoint.get("status") != "COMPLETE" or checkpoint.get("cell") != expected_cell:
            raise RuntimeError(f"Confirmation checkpoint is incomplete or has changed design: {path.name}")
        if checkpoint.get("source_hashes") != sources or checkpoint.get("protocol_freeze_sha256") != freeze_hash:
            raise RuntimeError(f"Confirmation checkpoint provenance mismatch: {path.name}")
        if sha256_file(path) != next((cell.get("checkpoint_sha256") for cell in manifest["cells"] if float(cell["slope_per_decade"]) == slope), None):
            raise RuntimeError(f"Confirmation checkpoint hash mismatch: {path.name}")
        cell_rows = checkpoint.get("replicates", [])
        reps = int(config["design"]["replicates_per_slope"])
        if len(cell_rows) != reps or [int(row.get("replicate", -1)) for row in cell_rows] != list(range(reps)):
            raise RuntimeError(f"Confirmation checkpoint row count or sequence is invalid: {path.name}")
        for row in cell_rows:
            if row.get("method") != config["scope"]["candidate_method"] or row.get("phase") != "confirmation":
                raise RuntimeError("Confirmation row method or phase differs from the frozen candidate")
            if str(row.get("series_seed")) in seen_seeds:
                raise RuntimeError("Duplicate series seed across confirmation cells")
            seen_seeds.add(str(row.get("series_seed")))
        rows.extend(cell_rows)
    expected_total = int(config["design"]["replicates_per_slope"]) * len(config["design"]["slopes_per_decade"])
    if len(rows) != expected_total or len(seen_seeds) != expected_total:
        raise RuntimeError("Confirmation row or unique seed inventory is incomplete")
    return config, manifest, rows


def summarize_cell(rows: list[dict[str, Any]], slope: float, criteria: dict[str, Any]) -> dict[str, Any]:
    supported = [row for row in rows if row.get("supported") is True]
    inferred = [row for row in supported if row.get("inferential_output") is True and row.get("p_value") is not None]
    intervals = [row for row in supported if row.get("converged") is True and row.get("ci_low_per_decade") is not None and row.get("ci_high_per_decade") is not None]
    coverage_count = sum(bool(row.get("ci_covers_true_slope")) for row in intervals)
    coverage_rate = coverage_count / len(intervals) if intervals else None
    coverage_cp = cp_interval(coverage_count, len(intervals))
    rejects = sum(float(row["p_value"]) < float(criteria["type_i_alpha"]) for row in inferred)
    type_i_or_power = rejects / len(inferred) if inferred else None
    reject_cp = cp_interval(rejects, len(inferred))
    errors = [float(row["slope_estimate_per_decade"]) - slope for row in inferred if row.get("slope_estimate_per_decade") is not None]
    widths = [float(row["ci_width"]) for row in intervals if row.get("ci_width") is not None]
    width_ratios = [float(row["ci_width_over_oracle"]) for row in intervals if row.get("ci_width_over_oracle") is not None]
    residual_rows = [row.get("residual_diagnostic", {}) for row in supported]
    residual_tested = [row for row in residual_rows if row.get("kind") == "LJUNG_BOX_COMPLETE_SUPPORT" and row.get("ljung_box_failure") is not None]
    residual_failures = sum(bool(row["ljung_box_failure"]) for row in residual_tested)
    residual_rate = residual_failures / len(residual_tested) if residual_tested else None
    convergence_failures = len(supported) - sum(bool(row.get("converged")) for row in supported)
    availability = len(inferred) / len(supported) if supported else None
    bias = mean_summary(errors)
    width_summary = mean_summary(widths)
    width_ratio_summary = mean_summary(width_ratios)

    is_null = float(slope) == 0.0
    type_i_gate = None
    if is_null:
        type_i_gate = (
            type_i_or_power is not None
            and float(criteria["type_i_rate_range"][0]) <= type_i_or_power <= float(criteria["type_i_rate_range"][1])
            and reject_cp[0] is not None
            and reject_cp[0] <= float(criteria["type_i_alpha"]) <= reject_cp[1]
        )
    coverage_gate = (
        coverage_rate is not None
        and float(criteria["coverage_rate_range"][0]) <= coverage_rate <= float(criteria["coverage_rate_range"][1])
        and coverage_cp[0] is not None
        and coverage_cp[0] <= float(criteria["coverage_nominal"]) <= coverage_cp[1]
    )
    gates = {
        "type_i_rate": type_i_gate,
        "coverage_rate_and_exact_interval": coverage_gate,
        "slope_bias": bias is not None and abs(float(bias["mean"])) <= float(criteria["max_absolute_slope_bias_per_decade"]),
        "numerical_failures_integer_gate": 100 * convergence_failures <= len(supported),
        "inferential_availability": availability is not None and availability >= float(criteria["minimum_inferential_availability"]),
        "ci_width_over_oracle": width_ratio_summary is not None and float(width_ratio_summary["mean"]) <= float(criteria["maximum_mean_ci_width_over_known_covariance_oracle"]),
        "residual_dependence": residual_rate is not None and residual_rate <= float(criteria["max_residual_dependence_failure_rate"]),
    }
    all_gates = all(value is True or (name == "type_i_rate" and value is None) for name, value in gates.items())
    return {
        "regime": "iid_gaussian",
        "method": "wald_student_t",
        "slope_per_decade": float(slope),
        "supported_count": len(supported),
        "inferential_output_count": len(inferred),
        "numerical_failures": convergence_failures,
        "numerical_failure_rate": convergence_failures / len(supported) if supported else None,
        "inferential_availability": availability,
        "type_i_rejections": rejects if is_null else None,
        "type_i_rate": type_i_or_power if is_null else None,
        "type_i_cp95": list(reject_cp) if is_null else None,
        "power": type_i_or_power if not is_null else None,
        "power_cp95": list(reject_cp) if not is_null else None,
        "coverage_count": coverage_count,
        "coverage_rate": coverage_rate,
        "coverage_cp95": list(coverage_cp),
        "slope_bias_per_decade": bias,
        "mean_ci_width": width_summary,
        "mean_ci_width_over_oracle": width_ratio_summary,
        "residual_ljung_box_failures": residual_failures,
        "residual_ljung_box_tested_count": len(residual_tested),
        "residual_ljung_box_failure_rate": residual_rate,
        "acceptance_gates": gates,
        "all_gates_pass": all_gates,
    }


def build_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# KUNTUR v1.3.2 Independent Confirmation — Finite-Sample t Interval",
        "",
        f"**Status:** `{metrics['status']}`. Scope is one frozen IID Gaussian synthetic regime.",
        "",
        f"Protocol: `{metrics['protocol_id']}`; freeze SHA-256 `{metrics['protocol_freeze_sha256']}`.",
        "",
        "## Frozen design and resource use",
        "",
        f"- Candidate: `{metrics['candidate_method']}`; point model remains seasonal SARIMAX AR(12).",
        f"- Completed: {metrics['completed_replicates']:,}/{metrics['target_replicates']:,} independent series across slopes 0, +0.25, and +0.50 per decade.",
        f"- Seed namespace: `{metrics['seed_namespace']}`; inventory SHA-256 `{metrics['seed_inventory']['confirmation_seed_inventory_sha256']}`.",
        f"- Resource use: {metrics['resource_usage']['cumulative_process_cpu_seconds']:.1f} CPU s; {metrics['resource_usage']['cumulative_active_wall_seconds']:.1f} active wall s; {metrics['resource_usage']['peak_rss_bytes_observed']:,} peak RSS bytes.",
        "- Exact binomial Clopper–Pearson 95% intervals are used for Type-I error, coverage, and power; continuous Monte Carlo means use Student-t 95% intervals.",
        "",
        "## Independent confirmation metrics",
        "",
        "| Slope | Type-I / power (exact 95% interval) | Coverage (exact 95% interval) | Bias (t-MC95) | Mean CI/oracle | Convergence failures | Availability | Ljung–Box failures | Gates |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in metrics["cells"]:
        rate = cell["type_i_rate"] if cell["type_i_rate"] is not None else cell["power"]
        ci = cell["type_i_cp95"] if cell["type_i_cp95"] is not None else cell["power_cp95"]
        rate_text = "NA" if rate is None else f"{rate:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        coverage_ci = cell["coverage_cp95"]
        coverage_text = "NA" if cell["coverage_rate"] is None else f"{cell['coverage_rate']:.4f} [{coverage_ci[0]:.4f}, {coverage_ci[1]:.4f}]"
        bias = cell["slope_bias_per_decade"]
        bias_text = "NA" if bias is None else f"{bias['mean']:.6f} [{bias['ci95_t'][0]:.6f}, {bias['ci95_t'][1]:.6f}]"
        width = cell["mean_ci_width_over_oracle"]
        width_text = "NA" if width is None else f"{width['mean']:.4f}"
        residual = f"{cell['residual_ljung_box_failures']}/{cell['residual_ljung_box_tested_count']} ({cell['residual_ljung_box_failure_rate']:.3f})" if cell["residual_ljung_box_failure_rate"] is not None else "NA"
        passed = "PASS" if cell["all_gates_pass"] else "FAIL"
        lines.append(f"| {cell['slope_per_decade']:+.2f} | {rate_text} | {coverage_text} | {bias_text} | {width_text} | {cell['numerical_failures']}/{cell['supported_count']} | {cell['inferential_availability']:.4f} | {residual} | {passed} |")
    lines.extend([
        "",
        "## Decision and limitations",
        "",
        f"- Decision: `{metrics['decision']}`.",
        "- The confirmation runner did not modify the inference registry. A pass would apply only to the declared IID Gaussian synthetic regime; no claim is made for AR(1), ARMA, seasonal-error, irregular-calendar, or real NASA series.",
        "- The original v1.3 normal-Wald confirmation failure (920/998 coverage at +0.50) remains preserved. This is a separate corrected interval method and a fresh seed namespace, not a rewrite of that result.",
        "",
    ])
    return "\n".join(lines)


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def run() -> dict[str, Any]:
    config, manifest, rows = load_verified_rows()
    cells = []
    for slope in config["design"]["slopes_per_decade"]:
        slope_rows = [row for row in rows if float(row["slope_per_decade"]) == float(slope)]
        cells.append(summarize_cell(slope_rows, float(slope), config["acceptance_criteria"]))
    complete = len(rows) == int(config["design"]["candidate_fit_count"])
    passed = complete and all(cell["all_gates_pass"] for cell in cells)
    decision = "INDEPENDENT_CONFIRMATION_PASS_FOR_IID_SYNTHETIC_SCOPE" if passed else "FAILED_INDEPENDENT_CONFIRMATION"
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    metrics = {
        "schema_version": "kuntur-v1.3.2-independent-confirmation-metrics-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETE_INDEPENDENT_CONFIRMATION",
        "decision": decision,
        "protocol_id": config["protocol_id"],
        "protocol_freeze_sha256": sha256_file(FREEZE_PATH),
        "candidate_method": config["scope"]["candidate_method"],
        "scope": config["scope"],
        "seed_namespace": config["design"]["series_seed_namespace"],
        "seed_inventory": freeze["seed_inventory"],
        "completed_replicates": len(rows),
        "target_replicates": int(config["design"]["candidate_fit_count"]),
        "cells": cells,
        "all_applicable_gates_pass": passed,
        "inference_registry_modified": False,
        "resource_usage": manifest["resource_usage"],
        "protected_evidence_unchanged": freeze["protected_evidence_unchanged"],
        "frozen_source_hashes": freeze["source_hashes"],
    }
    report = build_report(metrics)
    atomic_write(METRICS_PATH, json.dumps(metrics, sort_keys=True, indent=2, allow_nan=False) + "\n")
    atomic_write(REPORT_PATH, report)
    receipt = {
        "schema_version": "kuntur-v1.3.2-independent-confirmation-summary-receipt-v1",
        "protocol_id": config["protocol_id"],
        "protocol_freeze_sha256": sha256_file(FREEZE_PATH),
        "metrics_sha256": sha256_file(METRICS_PATH),
        "report_sha256": sha256_file(REPORT_PATH),
        "decision": decision,
        "candidate_method": config["scope"]["candidate_method"],
        "completed_replicates": len(rows),
        "all_applicable_gates_pass": passed,
        "inference_registry_modified": False,
    }
    atomic_write(RECEIPT_PATH, json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": metrics["status"], "decision": decision, "completed_replicates": len(rows), "metrics_sha256": receipt["metrics_sha256"], "report_sha256": receipt["report_sha256"], "metrics": METRICS_PATH.relative_to(ROOT).as_posix(), "report": REPORT_PATH.relative_to(ROOT).as_posix()}, indent=2))
    return metrics


def main() -> None:
    run()


if __name__ == "__main__":
    main()
