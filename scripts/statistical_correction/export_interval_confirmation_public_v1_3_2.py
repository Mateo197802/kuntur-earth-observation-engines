from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.statistical_correction.audit_confirmation_seed_disjointness_v1_3_2 import audit as audit_seeds
from scripts.statistical_correction.freeze_interval_confirmation_v1_3_2 import ROOT, sha256_file

PRE_ROOT = ROOT / "results/statistical_correction/interval_comparison_v1_3_2"
CONF_ROOT = ROOT / "results/statistical_correction/interval_confirmation_v1_3_2"
PRE_FREEZE = PRE_ROOT / "protocol_freeze.json"
CONF_FREEZE = CONF_ROOT / "protocol_freeze.json"
PREFLIGHT = PRE_ROOT / "preflight/preflight_summary.json"
DEV_METRICS = PRE_ROOT / "development/development_metrics.json"
DEV_REPORT = PRE_ROOT / "development/development_report.md"
DEV_RECEIPT = PRE_ROOT / "development/summary_receipt.json"
CONF_METRICS = CONF_ROOT / "confirmation/confirmation_metrics.json"
CONF_REPORT = CONF_ROOT / "confirmation/confirmation_report.md"
CONF_RECEIPT = CONF_ROOT / "confirmation/summary_receipt.json"
PUBLIC_JSON = ROOT / "results/phase2/public/statistical_validation_v1_3_2.json"
PUBLIC_REPORT = ROOT / "docs/audits/statistical_validation_v1_3_2_2026-11-17.md"

PUBLIC_SOURCE_PATHS = (
    "configs/statistical_interval_comparison_v1_3_2.json",
    "configs/statistical_interval_confirmation_v1_3_2.json",
    "docs/protocols/statistical_interval_comparison_v1_3_2.md",
    "docs/protocols/statistical_interval_confirmation_v1_3_2.md",
    "scripts/statistical_correction/interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/freeze_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/run_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/summarize_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/freeze_interval_confirmation_v1_3_2.py",
    "scripts/statistical_correction/run_interval_confirmation_v1_3_2.py",
    "scripts/statistical_correction/summarize_interval_confirmation_v1_3_2.py",
    "scripts/statistical_correction/audit_confirmation_seed_disjointness_v1_3_2.py",
    "scripts/statistical_correction/export_interval_confirmation_public_v1_3_2.py",
    "tests/test_interval_comparison_v13_2.py",
    "tests/test_interval_confirmation_v13_2.py",
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(content)
    temporary.replace(path)


def _git_text_sha256(path: Path) -> str:
    """Hash the LF-normalized bytes stored for public text sources in Git."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _public_cell(cell: dict[str, Any]) -> dict[str, Any]:
    allowed = (
        "method", "regime", "slope_per_decade", "replications_requested", "replications_completed",
        "supported_count", "inferential_output_count", "type_i_rejections", "type_i_rate", "type_i_cp95",
        "power", "power_cp95", "coverage_count", "coverage_rate", "coverage_cp95", "slope_bias_per_decade",
        "mean_ci_width_over_oracle", "numerical_failures", "numerical_failure_rate", "inferential_availability",
        "residual_ljung_box_failures", "residual_ljung_box_tested_count", "residual_ljung_box_failure_rate",
        "acceptance_gates", "all_gates_pass",
    )
    return {key: cell[key] for key in allowed if key in cell}


def _confirmation_cell(cell: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "regime", "method", "slope_per_decade", "supported_count", "inferential_output_count",
        "type_i_rejections", "type_i_rate", "type_i_cp95", "power", "power_cp95", "coverage_count",
        "coverage_rate", "coverage_cp95", "slope_bias_per_decade", "mean_ci_width_over_oracle",
        "numerical_failures", "numerical_failure_rate", "inferential_availability",
        "residual_ljung_box_failures", "residual_ljung_box_tested_count", "residual_ljung_box_failure_rate",
        "acceptance_gates", "all_gates_pass",
    )
    return {key: cell[key] for key in fields if key in cell}


def _public_code_hashes(freeze: dict[str, Any]) -> dict[str, str]:
    frozen = freeze["source_hashes"]
    result = {}
    for path in PUBLIC_SOURCE_PATHS:
        file_path = ROOT / path
        if not file_path.is_file():
            raise FileNotFoundError(f"Public provenance source is missing: {path}")
        # The two publishing/audit helpers were written after the statistical freeze.
        if path in frozen and frozen[path] != sha256_file(file_path):
            raise RuntimeError(f"Frozen confirmation source changed: {path}")
        result[path] = _git_text_sha256(file_path)
    return result


def build_payload() -> dict[str, Any]:
    pre_freeze = _read(PRE_FREEZE)
    conf_freeze = _read(CONF_FREEZE)
    preflight = _read(PREFLIGHT)
    development = _read(DEV_METRICS)
    development_receipt = _read(DEV_RECEIPT)
    confirmation = _read(CONF_METRICS)
    confirmation_receipt = _read(CONF_RECEIPT)
    if development.get("status") != "COMPLETE_DEVELOPMENT_ONLY" or len(development.get("cells", [])) != 9:
        raise RuntimeError("A complete nine-cell v1.3.2 development result is required")
    if development_receipt.get("metrics_sha256") != sha256_file(DEV_METRICS) or development_receipt.get("report_sha256") != sha256_file(DEV_REPORT):
        raise RuntimeError("Development report or metrics do not match the verified summary receipt")
    if confirmation.get("decision") != "INDEPENDENT_CONFIRMATION_PASS_FOR_IID_SYNTHETIC_SCOPE" or confirmation.get("all_applicable_gates_pass") is not True:
        raise RuntimeError("The independent confirmation did not pass all frozen gates")
    if confirmation.get("inference_registry_modified") is not False:
        raise RuntimeError("Confirmation metrics report an unexpected inference-registry mutation")
    if confirmation_receipt.get("metrics_sha256") != sha256_file(CONF_METRICS) or confirmation_receipt.get("report_sha256") != sha256_file(CONF_REPORT):
        raise RuntimeError("Confirmation report or metrics do not match the verified summary receipt")

    registry_path = ROOT / "results/statistical_correction/validated_regime_registry.json"
    registry_hash = sha256_file(registry_path)
    registry_manifest = _read(ROOT / "results/statistical_correction/validation_registry_manifest.json")
    if registry_hash != registry_manifest.get("registry_sha256"):
        raise RuntimeError("The existing inference-registry file no longer matches its recorded manifest")
    if registry_hash != "d68ddbf8825d98699198140593a871df307d71a3edb490ca02f992bae7f13580":
        raise RuntimeError("The historical v1.2 empty-registry hash has changed")

    source_hashes = _public_code_hashes(conf_freeze)
    seed_audit = audit_seeds()
    if seed_audit.get("status") != "SEED_DISJOINTNESS_VERIFIED" or seed_audit.get("exact_overlap_count") != 0:
        raise RuntimeError("Independent confirmation seed-disjointness audit failed")
    prior_sources = pre_freeze.get("source_hashes", {})
    prior_2025 = "results/optimization/round_06_2025_test_score/sealed_2025_score.json"
    prior_registry = "results/statistical_correction/validated_regime_registry.json"
    prior_closure = "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json"
    original_confirmation = "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_metrics.json"
    protected_hashes = {
        prior_2025: "a38af9f37962f948a0af7c27b06c829adb01c78311131f764bc344f1c64c80b7",
        prior_registry: "d68ddbf8825d98699198140593a871df307d71a3edb490ca02f992bae7f13580",
        prior_closure: "8b0677caead3ae6e3fe22999f0c6b078f9d2f596051ac670f4de2d7d313825e5",
        original_confirmation: "1566ca78fd69d1869740bff1a7c4178f264e656389f2f7a4e7ca9a2afdc40c67",
    }
    for name, expected_hash in protected_hashes.items():
        current_hash = sha256_file(ROOT / name)
        frozen_hash = prior_sources.get(name)
        if current_hash != expected_hash or (frozen_hash is not None and frozen_hash != current_hash):
            raise RuntimeError(f"A protected historical evidence file changed: {name}")

    development_cells = [_public_cell(cell) for cell in development["cells"]]
    confirmation_cells = [_confirmation_cell(cell) for cell in confirmation["cells"]]
    return {
        "schema_version": "kuntur-public-statistical-validation-v1.3.2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "statistical_scope": "Synthetic IID Gaussian only; no real NASA series, dependent-error regime, or irregular calendar is validated.",
        "historical_v1_2_reconstruction": {
            "frozen_selections_unchanged": {
                "iid_gaussian": "SEASONAL_MK_YEAR2",
                "ar1_rho_0_6_with_20pct_mcar": "SEASONAL_MK_YEAR2",
            },
            "score_arithmetic_reconstructed_without_error": True,
            "root_causes": [
                "The IID score was capability-asymmetric: Seasonal MK had no interval, so coverage penalties were omitted from its score while interval-producing candidates were scored on additional terms.",
                "The AR(1) with MCAR residual-dependence gate treated unavailable exact-calendar diagnostics as failed, excluding otherwise plausible SARIMAX candidates.",
                "Five hundred selection replications left Monte Carlo standard error near 0.01 for a 5% rejection rate; independent validation shifts were compatible with sampling variation and winner's curse.",
                "The saved score arithmetic reproduced exactly; a score arithmetic implementation error and model misspecification as the sole cause were not established.",
            ],
            "selection_root_cause_report_sha256": sha256_file(ROOT / "reports/statistical_correction/selection_root_cause_v1_3_20261117.md"),
        },
        "original_v1_3_wald_confirmation": {
            "method": "SARIMAX_SEASONAL_AR12 with normal-Wald interval",
            "status": "FAILED_INDEPENDENT_CONFIRMATION",
            "slope_0_50_covered": 920,
            "slope_0_50_available": 998,
            "coverage": 920 / 998,
            "coverage_exact_cp95": [0.9034, 0.9377],
            "metrics_sha256": protected_hashes[original_confirmation],
        },
        "v1_3_1_profile_preflight": {
            "status": "ABORTED_ENGINEERING_PREFLIGHT; main development not started",
            "failure": "Strict 1e-6 profile maximum check rejected small optimizer stopping-precision discrepancies; an audited row differed by about 7.44e-6.",
            "v1_3_2_fix": "Evaluate the exact profile at the unrestricted estimate; use the higher of two close maxima and clip only differences up to 1e-3; reject larger mismatches.",
        },
        "v1_3_2_preflight": {
            "status": preflight["status"],
            "decision": preflight["decision"],
            "candidate_replicates": preflight["completed_candidate_replicates"],
            "integrity_gate": preflight["preflight_integrity_gate"],
            "runtime_projection": preflight["projected_development"],
            "preflight_summary_sha256": sha256_file(PREFLIGHT),
        },
        "v1_3_2_development": {
            "status": development["status"],
            "decision": development["decision"],
            "paired_series": 3000,
            "candidate_fits": 9000,
            "candidate_gate_decisions": development["candidate_gate_decisions"],
            "cells": development_cells,
            "resource_usage": development["resource_usage"],
            "metrics_sha256": sha256_file(DEV_METRICS),
            "report_sha256": sha256_file(DEV_REPORT),
            "protocol_freeze_sha256": development["protocol_freeze_sha256"],
        },
        "v1_3_2_independent_confirmation": {
            "status": confirmation["status"],
            "decision": confirmation["decision"],
            "candidate_method": confirmation["candidate_method"],
            "scope": confirmation["scope"],
            "replications": confirmation["completed_replicates"],
            "all_applicable_gates_pass": confirmation["all_applicable_gates_pass"],
            "cells": confirmation_cells,
            "resource_usage": confirmation["resource_usage"],
            "seed_namespace": confirmation["seed_namespace"],
            "seed_inventory_sha256": confirmation["seed_inventory"]["confirmation_seed_inventory_sha256"],
            "development_seed_inventory": {
                "count": confirmation["seed_inventory"]["v1_3_2_preflight_and_development_seed_count"],
                "sha256": confirmation["seed_inventory"]["v1_3_2_seed_inventory_sha256"],
                "disjoint_from_confirmation": confirmation["seed_inventory"]["no_overlap_with_v1_3_2_preflight_or_development"],
            },
            "seed_disjointness_audit": seed_audit,
            "metrics_sha256": sha256_file(CONF_METRICS),
            "report_sha256": sha256_file(CONF_REPORT),
            "protocol_freeze_sha256": confirmation["protocol_freeze_sha256"],
        },
        "registry_and_protected_inputs": {
            "inference_registry_modified": False,
            "v1_2_registry_status": registry_manifest["status"],
            "v1_2_registry_sha256": registry_hash,
            "v1_2_partial_closure_sha256": protected_hashes[prior_closure],
            "original_v1_3_failure_preserved": True,
            "2025_predictive_test_rerun": False,
            "2025_predictive_score_sha256": protected_hashes[prior_2025],
            "original_v1_3_confirmation_metrics_sha256": protected_hashes[original_confirmation],
        },
        "public_code_sha256": source_hashes,
        "delivery_policy": "Aggregate metrics and code only. No row-level simulation checkpoints, NASA data, private paths, credentials, or raw source records are included.",
        "public_code_hash_policy": "SHA-256 of LF-normalized text bytes, matching the Git blobs under the repository text=auto/eol=lf policy.",
    }


def _report(payload: dict[str, Any], public_json_sha256: str) -> str:
    lines = [
        "# KUNTUR v1.3.2 — Statistical Correction and Independent Confirmation",
        "",
        "**Scientific result:** the finite-sample Student-t interval (`wald_student_t`) passed the frozen 3,000-fit independent confirmation for the declared IID Gaussian synthetic regime. The original v1.3 normal-Wald failure remains unchanged. This evidence does not validate inferential claims for dependent regimes or real NASA observations.",
        "",
        f"Aggregate handoff JSON SHA-256: `{public_json_sha256}`.",
        f"v1.3.2 development freeze SHA-256: `{payload['v1_3_2_development']['protocol_freeze_sha256']}`.",
        f"Independent-confirmation freeze SHA-256: `{payload['v1_3_2_independent_confirmation']['protocol_freeze_sha256']}`.",
        "",
        "## Why the frozen v1.2 selections failed later calibration",
        "",
        "The original score reconstruction was exact; the score arithmetic was not wrong. The selector compared methods on unequal capabilities. In IID, Seasonal MK had no interval, so it was not penalized for null or alternative coverage, while interval methods were scored on coverage as well as Type-I error and bias. For AR(1) with 20% MCAR, exact-calendar residual diagnostics were prescribed but unavailable results were treated as failed gates, leaving Seasonal MK as the only eligible candidate. Selection used 500 replicates, giving an approximate Type-I Monte Carlo SE of 0.010 near 5%; the 500-to-1,000 replicate shifts were compatible with ordinary Monte Carlo variation and winner's curse. The saved selection decisions are not rewritten.",
        "",
        "## Preserved original failure and v1.3.1 preflight correction",
        "",
        "The original `SARIMAX_SEASONAL_AR12` normal-Wald confirmation covered 920/998 intervals at +0.50 per decade (92.18%; exact 95% interval 90.34–93.77%), below the frozen 93.5% floor. The v1.3.1 preflight then exposed a profile-likelihood implementation guard that treated a roughly 7.44e-6 optimizer tolerance difference as failure under a 1e-6 threshold. Its 37/90 affected rows are retained; no v1.3.1 main development ran. v1.3.2 corrected the numeric guard with an explicit 1e-3 tolerance and required 100% inferential output availability in all 270 preflight candidate rows.",
        "",
        "## v1.3.2 paired development comparison",
        "",
        "The comparison used 1,000 paired IID Gaussian series for each slope (0, +0.25, +0.50 per decade), shared across the unchanged normal-Wald comparator, finite-sample t interval, and profile-likelihood interval. All candidates use the same SARIMAX seasonal AR(12) point model. Exact binomial 95% intervals are shown for Type-I error, coverage, and power; bias intervals are Student-t Monte Carlo intervals.",
        "",
        "| Method | Slope | Type-I / power (exact 95% interval) | Coverage (exact 95% interval) | Bias per decade (t-MC95) | Width/oracle | Numerical failures | Availability | Residual failures | Gates |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---|",
    ]
    for cell in payload["v1_3_2_development"]["cells"]:
        rate = cell.get("type_i_rate") if cell.get("type_i_rate") is not None else cell.get("power")
        ci = cell.get("type_i_cp95") if cell.get("type_i_cp95") is not None else cell.get("power_cp95")
        rate_text = "—" if rate is None else f"{rate:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        coverage = cell["coverage_rate"]
        cov_ci = cell["coverage_cp95"]
        coverage_text = f"{coverage:.4f} [{cov_ci[0]:.4f}, {cov_ci[1]:.4f}]" if coverage is not None else "—"
        bias = cell.get("slope_bias_per_decade")
        bias_text = "—" if bias is None else f"{bias['mean']:.6f} [{bias['ci95_t'][0]:.6f}, {bias['ci95_t'][1]:.6f}]"
        width = cell.get("mean_ci_width_over_oracle", {}).get("mean")
        residual = f"{cell['residual_ljung_box_failures']}/{cell['residual_ljung_box_tested_count']}"
        pass_text = "PASS" if cell.get("all_gates_pass") else "FAIL"
        lines.append(f"| {cell['method']} | {cell['slope_per_decade']:+.2f} | {rate_text} | {coverage_text} | {bias_text} | {width:.3f} | {cell['numerical_failures']}/{cell['supported_count']} | {cell['inferential_availability']:.3f} | {residual} | {pass_text} |")
    lines.extend([
        "",
        "Only the finite-sample `wald_student_t` interval was eligible among corrected candidates. It passed every development gate at all three slopes. The unchanged normal-Wald comparator also met the metric thresholds in the new development sample, but it was designated comparator-only because it is the method that failed the earlier untouched confirmation. Profile likelihood failed null Type-I and +0.50 coverage gates. Development did not promote any method.",
        "",
        "## Independent confirmation of `wald_student_t`",
        "",
        "After freezing the candidate, acceptance criteria, implementation hashes, a fresh namespace, and measured budget, a separate 1,000-series confirmation was run at each slope. The exact 3,000-seed inventory was unique and had zero overlap with 207,906 saved seed values parsed from 342 readable prior statistical JSON artifacts. One known malformed historical recovery metadata receipt (no seed rows) was excluded by hash; it was not an analysis checkpoint. The protocol also verified no overlap with the v1.3.2 preflight/development inventories.",
        "",
        "| Slope | Type-I / power (exact 95% interval) | Coverage (exact 95% interval) | Bias per decade (t-MC95) | Width/oracle | Numerical failures | Availability | Residual failures | Gates |",
        "|---:|---|---|---|---:|---:|---:|---:|---|",
    ])
    for cell in payload["v1_3_2_independent_confirmation"]["cells"]:
        rate = cell.get("type_i_rate") if cell.get("type_i_rate") is not None else cell.get("power")
        ci = cell.get("type_i_cp95") if cell.get("type_i_cp95") is not None else cell.get("power_cp95")
        rate_text = "—" if rate is None else f"{rate:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
        cov_ci = cell["coverage_cp95"]
        coverage_text = f"{cell['coverage_rate']:.4f} [{cov_ci[0]:.4f}, {cov_ci[1]:.4f}]"
        bias = cell["slope_bias_per_decade"]
        bias_text = f"{bias['mean']:.6f} [{bias['ci95_t'][0]:.6f}, {bias['ci95_t'][1]:.6f}]"
        width = cell["mean_ci_width_over_oracle"]["mean"]
        residual = f"{cell['residual_ljung_box_failures']}/{cell['residual_ljung_box_tested_count']}"
        pass_text = "PASS" if cell["all_gates_pass"] else "FAIL"
        lines.append(f"| {cell['slope_per_decade']:+.2f} | {rate_text} | {coverage_text} | {bias_text} | {width:.3f} | {cell['numerical_failures']}/{cell['supported_count']} | {cell['inferential_availability']:.3f} | {residual} | {pass_text} |")
    lines.extend([
        "",
        f"All applicable gates passed. Actual resource use was {payload['v1_3_2_independent_confirmation']['resource_usage']['cumulative_process_cpu_seconds']:.1f} CPU seconds, {payload['v1_3_2_independent_confirmation']['resource_usage']['cumulative_active_wall_seconds']:.1f} active wall seconds, and {payload['v1_3_2_independent_confirmation']['resource_usage']['peak_rss_bytes_observed']:,} bytes peak RSS; limits were 3,600 CPU seconds, 7,200 wall seconds, and 1 GiB RSS.",
        "",
        "## What this enables and what remains withheld",
        "",
        "This confirms the finite-sample t interval only for the stated synthetic IID Gaussian data-generating process, the 276-month support, slopes, model, and diagnostic policy. The `wald_student_t` result is not applicable to AR(1), ARMA, seasonal dependence, irregular missingness, or actual NASA climate series without separate development and independent confirmation in those regimes. The v1.3 runtime adapter has no frozen v1.3 registry; the existing v1.2 registry remains empty and hash-identical. Thus confirmatory p-values, confidence intervals, and FDR q-values remain withheld for real Gold/NASA analyses. No registry change is made by this handoff.",
        "",
        "The original negative 2025 forecasting score is preserved and was not rerun or used for selection. v1.2 closure and original v1.3 confirmation checkpoints remain unchanged. No model training, trend analysis of NASA observations, publication merge, or PR merge occurred.",
        "",
        "## Provenance and reproduction",
        "",
        f"- Public-safe aggregate JSON SHA-256: `{public_json_sha256}`.",
        f"- Development metrics SHA-256: `{payload['v1_3_2_development']['metrics_sha256']}`; development report SHA-256: `{payload['v1_3_2_development']['report_sha256']}`.",
        f"- Independent-confirmation metrics SHA-256: `{payload['v1_3_2_independent_confirmation']['metrics_sha256']}`; report SHA-256: `{payload['v1_3_2_independent_confirmation']['report_sha256']}`.",
        f"- v1.2 registry SHA-256 (unchanged): `{payload['registry_and_protected_inputs']['v1_2_registry_sha256']}`.",
        f"- 2025 forecast score SHA-256 (unchanged): `{payload['registry_and_protected_inputs']['2025_predictive_score_sha256']}`.",
        "- Machine-readable code/config/test hashes are in the aggregate JSON. No seed values or replicate-level rows are published.",
        "",
        "From the repository root in the Python 3.12 environment from the README:",
        "",
        "```powershell",
        "python -m pytest -q tests/test_interval_comparison_v13_2.py tests/test_interval_confirmation_v13_2.py",
        "python scripts/statistical_correction/freeze_interval_confirmation_v1_3_2.py --verify",
        "python scripts/statistical_correction/audit_confirmation_seed_disjointness_v1_3_2.py",
        "python scripts/statistical_correction/summarize_interval_comparison_v1_3_2.py --phase development",
        "python scripts/statistical_correction/summarize_interval_confirmation_v1_3_2.py",
        "python scripts/statistical_correction/export_interval_confirmation_public_v1_3_2.py",
        "python scripts/statistical_correction/verify_v1_3_evidence_bundle.py",
        "```",
        "",
        "The result summary contains aggregate statistics only. Re-running simulations is a separate frozen workflow; the commands above verify and summarize the saved evidence without rerunning the simulations.",
        "",
    ])
    return "\n".join(lines)


def export() -> dict[str, Any]:
    payload = build_payload()
    public_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _atomic_write(PUBLIC_JSON, public_json)
    json_hash = sha256_file(PUBLIC_JSON)
    report = _report(payload, json_hash)
    _atomic_write(PUBLIC_REPORT, report)
    return {
        "status": "PUBLIC_SAFE_AGGREGATES_EXPORTED",
        "public_json": PUBLIC_JSON.relative_to(ROOT).as_posix(),
        "public_json_sha256": json_hash,
        "public_report": PUBLIC_REPORT.relative_to(ROOT).as_posix(),
        "public_report_sha256": sha256_file(PUBLIC_REPORT),
        "replicate_rows_published": 0,
        "registry_modified": False,
    }


def main() -> None:
    print(json.dumps(export(), indent=2))


if __name__ == "__main__":
    main()
