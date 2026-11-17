"""Verify v1.3 independent-confirmation checkpoints and apply frozen gates."""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import beta, t as student_t

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from kuntur_data.phase2.statistical_v12 import METHOD_IDS

CONFIG_PATH = ROOT / "configs/statistical_corrective_confirmation_v1_3.json"
FREEZE_PATH = ROOT / "results/statistical_correction/corrective_development_v1_3/selection_freeze_iid_v1_3_r1.json"
FREEZE_RECEIPT_PATH = ROOT / "results/statistical_correction/corrective_development_v1_3/selection_freeze_receipt_iid_v1_3_r1.json"
OUTPUT = ROOT / "results/statistical_correction/independent_confirmation_v1_3_iid"
CHECKPOINTS = OUTPUT / "checkpoints"
RUN_MANIFEST_PATH = OUTPUT / "confirmation_manifest.json"
STATE_PATH = OUTPUT / "run_state.json"
METRICS_PATH = OUTPUT / "confirmation_metrics.json"
REPORT_PATH = ROOT / "reports/statistical_correction/v1_3_independent_confirmation_iid_20261117.md"
RECEIPT_PATH = OUTPUT / "summary_receipt.json"
V13_REGISTRY_PATH = ROOT / "results/statistical_correction/validated_regime_registry_v1_3.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def verify() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not RUN_MANIFEST_PATH.is_file() or not STATE_PATH.is_file():
        raise FileNotFoundError("Independent-confirmation run manifest or state is missing")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    freeze_receipt = json.loads(FREEZE_RECEIPT_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(RUN_MANIFEST_PATH.read_text(encoding="utf-8"))
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    if freeze_receipt.get("selection_manifest_sha256") != sha256(FREEZE_PATH):
        raise RuntimeError("Selection freeze receipt no longer matches its manifest")
    if manifest.get("selection_freeze_sha256") != sha256(FREEZE_PATH):
        raise RuntimeError("Run manifest does not reference the current frozen selection")
    if manifest.get("status") != "INDEPENDENT_CONFIRMATION_COMPLETE" or state.get("status") != "INDEPENDENT_CONFIRMATION_COMPLETE":
        raise RuntimeError("Confirmation did not complete all frozen cells within its budget")
    if manifest.get("candidate") != config.get("candidate") or manifest.get("acceptance_criteria") != config.get("acceptance_criteria"):
        raise RuntimeError("Run manifest candidate or gates differ from frozen config")
    if manifest.get("design", {}).get("series_seed_namespace") != config.get("design", {}).get("series_seed_namespace"):
        raise RuntimeError("Run manifest uses an unexpected confirmation seed namespace")
    if manifest.get("completed_cells") != 3 or manifest.get("completed_outer_fits") != 3000:
        raise RuntimeError("Expected 3 full cells and 3,000 fits")
    expected_hashes = manifest.get("source_hashes", {})
    live_hashes = {
        "confirmation_config": sha256(CONFIG_PATH),
        "confirmation_protocol": sha256(ROOT / "docs/protocols/statistical_corrective_confirmation_v1_3.md"),
        "confirmation_runner": sha256(ROOT / "scripts/statistical_correction/run_corrective_confirmation_v1_3.py"),
        "selection_freeze_script": sha256(ROOT / "scripts/statistical_correction/freeze_corrective_selection_v1_3_r1.py"),
        "confirmation_summarizer": sha256(ROOT / "scripts/statistical_correction/summarize_corrective_confirmation_v1_3.py"),
        "confirmation_unit_tests": sha256(ROOT / "tests/test_statistical_confirmation_v13.py"),
        "statistical_module": sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py"),
        "calibration_generator": sha256(ROOT / "scripts/statistical_correction/run_calibration.py"),
        "development_helpers": sha256(ROOT / "scripts/statistical_correction/run_corrective_development_v1_3_expansion.py"),
        "selection_freeze_manifest": sha256(FREEZE_PATH),
        "selection_freeze_receipt": sha256(FREEZE_RECEIPT_PATH),
    }
    if any(expected_hashes.get(key) != value for key, value in live_hashes.items()):
        raise RuntimeError("A frozen source hash changed after independent confirmation")
    cells: list[dict[str, Any]] = []
    seeds: list[str] = []
    for slope in config["design"]["slopes_per_decade"]:
        label = f"{float(slope):.2f}".replace("-", "m").replace(".", "p")
        path = CHECKPOINTS / f"cell_iid_gaussian_n276_s{label}_sarimax_seasonal_ar12.json"
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("status") != "COMPLETE" or int(cell.get("completed_replicates", -1)) != 1000:
            raise RuntimeError(f"Incomplete independent-confirmation checkpoint: {path.name}")
        if cell.get("source_hashes") != expected_hashes or cell.get("selection_freeze_sha256") != sha256(FREEZE_PATH):
            raise RuntimeError(f"Checkpoint provenance mismatch: {path.name}")
        records = cell.get("replicates", [])
        if len(records) != 1000 or [row.get("replicate") for row in records] != list(range(1000)):
            raise RuntimeError(f"Checkpoint replicate IDs are missing or duplicated: {path.name}")
        if any(row.get("supported") is not True for row in records):
            raise RuntimeError(f"Unexpected unsupported confirmation row: {path.name}")
        seeds.extend([row["series_seed"] for row in records])
        seeds.extend([row["estimator_seed"] for row in records])
        cells.append(cell)
    if len(seeds) != 6000 or len(set(seeds)) != len(seeds):
        raise RuntimeError("Confirmation series and estimator seeds are not globally unique")
    return config, manifest, cells


def summarize_cell(cell: dict[str, Any], alpha: float, nominal: float) -> dict[str, Any]:
    slope = float(cell["cell"]["slope_per_decade"])
    rows = cell["replicates"]
    supported = [row for row in rows if row.get("supported")]
    inferred = [row for row in supported if row.get("inferential_output") and row.get("p_value") is not None]
    intervals = [row for row in supported if row.get("converged") and row.get("ci_low_per_decade") is not None and row.get("ci_high_per_decade") is not None]
    coverage_count = sum(bool(row.get("ci_covers_true_slope")) for row in intervals)
    coverage_rate = coverage_count / len(intervals) if intervals else None
    coverage_ci = cp_interval(coverage_count, len(intervals))
    rejection_count = sum(float(row["p_value"]) < alpha for row in inferred)
    rejection_rate = rejection_count / len(inferred) if inferred else None
    rejection_ci = cp_interval(rejection_count, len(inferred))
    estimates = [float(row["slope_estimate_per_decade"]) for row in inferred if row.get("slope_estimate_per_decade") is not None]
    errors = [value - slope for value in estimates]
    mae_values = [abs(value) for value in errors]
    widths = [float(row["ci_width"]) for row in intervals if row.get("ci_width") is not None]
    width_ratios = [float(row["ci_width_over_oracle"]) for row in intervals if row.get("ci_width_over_oracle") is not None]
    residual_rows = [row.get("residual_diagnostic", {}) for row in supported]
    tested_residual = [row for row in residual_rows if row.get("kind") == "LJUNG_BOX_COMPLETE_SUPPORT" and row.get("ljung_box_failure") is not None]
    residual_failures = sum(bool(row.get("ljung_box_failure")) for row in tested_residual)
    residual_rate = residual_failures / len(tested_residual) if tested_residual else None
    convergence_failures = len(supported) - sum(bool(row.get("converged")) for row in supported)
    availability = len(inferred) / len(supported) if supported else None
    bias = mean_summary(errors)
    abs_error = mean_summary(mae_values)
    width = mean_summary(widths)
    width_ratio = mean_summary(width_ratios)
    power = rejection_rate if slope > 0 else None
    criteria = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))["acceptance_criteria"]
    type_i_gate = None
    if slope == 0.0:
        type_i_gate = (
            rejection_rate is not None
            and criteria["type_i_rate_range"][0] <= rejection_rate <= criteria["type_i_rate_range"][1]
            and rejection_ci[0] is not None and rejection_ci[0] <= alpha <= rejection_ci[1]
        )
    coverage_gate = (
        coverage_rate is not None
        and criteria["coverage_rate_range"][0] <= coverage_rate <= criteria["coverage_rate_range"][1]
        and coverage_ci[0] is not None and coverage_ci[0] <= nominal <= coverage_ci[1]
    )
    bias_gate = bias is not None and abs(float(bias["mean"])) <= float(criteria["max_absolute_slope_bias_per_decade"])
    numerical_gate = 100 * convergence_failures <= len(supported)
    availability_gate = availability is not None and availability >= float(criteria["minimum_inferential_availability"])
    width_gate = width_ratio is not None and width_ratio["mean"] <= float(criteria["maximum_mean_ci_width_over_known_covariance_oracle"])
    residual_gate = residual_rate is not None and residual_rate <= float(criteria["max_residual_dependence_failure_rate"])
    gates = {
        "type_i_rate": type_i_gate,
        "coverage_rate_and_exact_interval": coverage_gate,
        "slope_bias": bias_gate,
        "numerical_failures_integer_gate": numerical_gate,
        "inferential_availability": availability_gate,
        "ci_width_over_oracle": width_gate,
        "residual_dependence": residual_gate,
    }
    all_gates = all(value is True or (name == "type_i_rate" and value is None) for name, value in gates.items())
    return {
        "regime": "iid_gaussian",
        "method": cell["cell"]["method_name"],
        "slope_per_decade": slope,
        "supported_count": len(supported),
        "inferential_output_count": len(inferred),
        "numerical_failures": convergence_failures,
        "numerical_failure_rate": convergence_failures / len(supported) if supported else None,
        "inferential_availability": availability,
        "type_i_rejections": rejection_count if slope == 0.0 else None,
        "type_i_rate": rejection_rate if slope == 0.0 else None,
        "type_i_cp95": list(rejection_ci) if slope == 0.0 else None,
        "coverage_count": coverage_count,
        "coverage_rate": coverage_rate,
        "coverage_cp95": list(coverage_ci),
        "power": power,
        "power_cp95": list(rejection_ci) if slope > 0 else None,
        "slope_bias_per_decade": bias,
        "slope_mae_per_decade": abs_error,
        "slope_rmse_per_decade": float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        "mean_ci_width": width,
        "mean_ci_width_over_oracle": width_ratio,
        "residual_ljung_box_failures": residual_failures,
        "residual_ljung_box_tested_count": len(tested_residual),
        "residual_ljung_box_failure_rate": residual_rate,
        "acceptance_gates": gates,
        "all_gates_pass": all_gates,
    }


def build_report(metrics: dict[str, Any]) -> str:
    lines = [
        "# KUNTUR v1.3 Independent Confirmation — IID Gaussian",
        "",
        f"**Status:** `{metrics['status']}`. This report covers one frozen synthetic regime only.",
        "",
        f"Generated UTC: {metrics['generated_utc']}",
        "",
        "## Frozen design",
        "",
        f"- Candidate: `{metrics['candidate']['method_name']}` in `{metrics['candidate']['regime_id']}`.",
        f"- Replications: {metrics['completed_outer_fits']:,} ({metrics['completed_cells']} slope cells × 1,000), paired only within this single candidate's predeclared DGP design.",
        f"- Seed namespace: `{metrics['design']['series_seed_namespace']}`.",
        f"- Resource use: {metrics['resource_usage']['cumulative_cpu_seconds']:.1f} CPU seconds; {metrics['resource_usage']['cumulative_active_wall_seconds']:.1f} active wall seconds; {metrics['resource_usage']['peak_rss_bytes_observed']:,} peak RSS bytes.",
        "- The v1.2 acceptance gates were applied exactly as frozen. All rates use exact Clopper–Pearson 95% intervals; continuous Monte Carlo means use Student-t 95% intervals.",
        "",
        "## Calibration and effect results",
        "",
        "| Slope | Type-I / power | Exact 95% binomial interval | Coverage | Exact 95% coverage interval | Bias (95% MC interval) | Mean CI/oracle | Residual failures | Gate result |",
        "|---:|---:|---|---:|---|---|---:|---:|---|",
    ]
    for row in metrics["cells"]:
        rejection = "—" if row["type_i_rate"] is None else f"{row['type_i_rate']:.4f}"
        reject_ci = "—" if row["type_i_cp95"] is None else f"[{row['type_i_cp95'][0]:.4f}, {row['type_i_cp95'][1]:.4f}]"
        power = "—" if row["power"] is None else f"{row['power']:.4f}"
        width = row["mean_ci_width_over_oracle"]
        bias = row["slope_bias_per_decade"]
        residual = "NA" if row["residual_ljung_box_failure_rate"] is None else f"{row['residual_ljung_box_failures']}/{row['residual_ljung_box_tested_count']} ({row['residual_ljung_box_failure_rate']:.3f})"
        cov_ci = row["coverage_cp95"]
        bias_ci = bias["ci95_t"] if bias else None
        lines.append(
            f"| {row['slope_per_decade']:.2f} | {rejection if rejection != '—' else power} | {reject_ci} | {row['coverage_rate']:.4f} | [{cov_ci[0]:.4f}, {cov_ci[1]:.4f}] | {bias['mean']:.5f} [{bias_ci[0]:.5f}, {bias_ci[1]:.5f}] | {width['mean']:.4f} | {residual} | {'PASS' if row['all_gates_pass'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## Gate decisions",
        "",
    ])
    for row in metrics["cells"]:
        lines.append(f"- Slope {row['slope_per_decade']:.2f}: " + ", ".join(f"{name}={'PASS' if value else 'FAIL'}" for name, value in row["acceptance_gates"].items()))
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "A pass supports only this exact synthetic IID Gaussian design with 276 complete consecutive monthly observations and the frozen SARIMAX parameterization. It does not validate AR(1), ARMA, seasonal dependence, calendar missingness, heavy tails, heteroskedasticity, or real environmental series. Real KUNTUR series remain descriptive unless a justified applicable error-regime assessment is available. No v1.2 decision, NASA Gold artifact, v1.2 registry, real-series score, or 2025 predictive result was altered or reused.",
        "",
        "## Provenance",
        "",
        f"- Raw run manifest SHA-256: `{metrics['run_manifest_sha256']}`.",
        f"- Frozen selection SHA-256: `{metrics['selection_freeze_sha256']}`.",
        f"- Confirmation metrics SHA-256 is recorded in `{metrics['receipt_path']}`.",
        "- Cell checkpoint hashes and source hashes are in the run manifest and per-cell checkpoint files.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    config, manifest, cells = verify()
    criteria = config["acceptance_criteria"]
    summaries = [summarize_cell(cell, float(criteria["type_i_alpha"]), float(criteria["coverage_nominal"])) for cell in cells]
    overall = all(cell["all_gates_pass"] for cell in summaries)
    metrics: dict[str, Any] = {
        "schema_version": "kuntur-v1.3-independent-confirmation-metrics-v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INDEPENDENT_CONFIRMATION_PASS_FOR_IID_SYNTHETIC_REGIME" if overall else "INDEPENDENT_CONFIRMATION_FAIL_NOT_VALIDATED",
        "classification": "INDEPENDENT_CONFIRMATION_EVIDENCE",
        "candidate": config["candidate"],
        "design": config["design"],
        "acceptance_criteria": criteria,
        "completed_cells": manifest["completed_cells"],
        "completed_outer_fits": manifest["completed_outer_fits"],
        "cells": summaries,
        "all_gates_pass": overall,
        "registry_created": False,
        "registry_path": None,
        "resource_usage": manifest["resource_usage"],
        "run_manifest_path": str(RUN_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
        "run_manifest_sha256": sha256(RUN_MANIFEST_PATH),
        "receipt_path": str(RECEIPT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "selection_freeze_sha256": sha256(FREEZE_PATH),
        "source_hashes": manifest["source_hashes"],
        "v1_2_registry_modified": False,
        "sealed_2025_score_touched": False,
    }
    report = build_report(metrics)
    if overall:
        registry = {
            "schema_version": "kuntur-validated-regime-registry-v1.3",
            "registry_version": "1.3.0",
            "status": "FROZEN",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "evidence_scope": "independent simulation confirmation only; no automatic real-series error-regime classifier",
            "source_confirmation_metrics_sha256": None,
            "regime_count": 1,
            "regimes": {
                "iid_gaussian": {
                    "validity_status": "VALIDATED_FOR_DECLARED_REGIME",
                    "method": "seasonal_ar12",
                    "method_id": METHOD_IDS["seasonal_ar12"],
                    "n_calendar_months_min": 276,
                    "n_calendar_months_max": 276,
                    "calendar_coverage_min": 1.0,
                    "calendar_coverage_max": 1.0,
                    "simulation_scope": "276 complete consecutive monthly observations with iid Gaussian errors and deterministic monthly seasonal effects",
                    "acceptance_gates_passed": True,
                    "confirmation_seed_namespace": config["design"]["series_seed_namespace"],
                    "confirmation_replicates_per_slope": 1000,
                    "confirmation_slopes_per_decade": config["design"]["slopes_per_decade"],
                    "method_spec": config["candidate"]["method_spec"],
                    "confirmation_metrics_path": str(METRICS_PATH.relative_to(ROOT)).replace("\\", "/"),
                    "confirmation_run_manifest_path": str(RUN_MANIFEST_PATH.relative_to(ROOT)).replace("\\", "/"),
                }
            },
            "unsupported_regimes": ["ar1_0844", "arma11", "seasonal_ar1_12", "ar1_060_mcar20", "ar1_060_seasonal_missing", "real_series_without justified regime evidence"],
            "v1_2_registry_sha256_preserved": sha256(ROOT / "results/statistical_correction/validated_regime_registry.json"),
        }
        registry["source_confirmation_metrics_sha256"] = sha256(METRICS_PATH) if METRICS_PATH.exists() else "PENDING_METRICS_WRITE"
        if V13_REGISTRY_PATH.exists():
            existing = json.loads(V13_REGISTRY_PATH.read_text(encoding="utf-8"))
            if existing.get("schema_version") != registry["schema_version"]:
                raise FileExistsError("A different v1.3 registry already exists; refusing overwrite")
        else:
            write_json(V13_REGISTRY_PATH, registry)
        metrics["registry_created"] = True
        metrics["registry_path"] = str(V13_REGISTRY_PATH.relative_to(ROOT)).replace("\\", "/")
        # The registry stores a source metrics digest. Rewrite it after metrics are first serialized below.
    write_json(METRICS_PATH, metrics)
    if overall:
        registry = json.loads(V13_REGISTRY_PATH.read_text(encoding="utf-8"))
        registry["source_confirmation_metrics_sha256"] = sha256(METRICS_PATH)
        write_json(V13_REGISTRY_PATH, registry)
    report = build_report(metrics)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    receipt = {
        "schema_version": "kuntur-v1.3-independent-confirmation-summary-receipt-v1",
        "status": metrics["status"],
        "metrics_path": str(METRICS_PATH.relative_to(ROOT)).replace("\\", "/"),
        "metrics_sha256": sha256(METRICS_PATH),
        "report_path": str(REPORT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "report_sha256": sha256(REPORT_PATH),
        "run_manifest_sha256": sha256(RUN_MANIFEST_PATH),
        "selection_freeze_sha256": sha256(FREEZE_PATH),
        "registry_created": metrics["registry_created"],
        "registry_path": metrics["registry_path"],
    }
    write_json(RECEIPT_PATH, receipt)
    print(json.dumps({"status": metrics["status"], "completed_cells": metrics["completed_cells"], "completed_fits": metrics["completed_outer_fits"], "metrics_sha256": receipt["metrics_sha256"], "report_sha256": receipt["report_sha256"], "registry_created": metrics["registry_created"], "registry_path": metrics["registry_path"]}, indent=2))


if __name__ == "__main__":
    main()
