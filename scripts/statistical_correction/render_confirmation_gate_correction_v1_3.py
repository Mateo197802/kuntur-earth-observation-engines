#!/usr/bin/env python3
"""Render an additive correction for the v1.3 confirmation gate display.

The original confirmation metrics, report, run manifest, and checkpoints are
read-only inputs. This utility writes a separate addendum and never edits them.
"""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
METRICS_PATH = ROOT / "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_metrics.json"
ORIGINAL_REPORT_PATH = ROOT / "reports/statistical_correction/v1_3_independent_confirmation_iid_20261117.md"
OUTPUT_PATH = ROOT / "reports/statistical_correction/v1_3_confirmation_gate_display_correction_addendum_20261117.md"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gate_label(name: str, value: bool | None, slope: float) -> str:
    if name == "type_i_rate" and slope != 0.0:
        if value is not None:
            raise ValueError("Type-I gate must be null at nonzero slopes")
        return "N/A (Type-I is evaluated only at zero slope)"
    return "PASS" if value is True else "FAIL"


def render(metrics: dict) -> str:
    if metrics.get("status") != "INDEPENDENT_CONFIRMATION_FAIL_NOT_VALIDATED":
        raise ValueError("Unexpected confirmation status; refusing to render correction")
    lines = [
        "# Addendum — v1.3 Confirmation Gate Display Correction",
        "",
        "**Scope:** This is an additive rendering correction. The original confirmation metrics, report, run manifest, selection freeze, checkpoints, and v1.2 historical records are unchanged.",
        "",
        "The original report's ‘Gate decisions’ line formatted every falsey value as `FAIL`. In the metrics JSON, `type_i_rate` and its gate are `null` for nonzero injected slopes because Type-I error is only evaluated at the zero-slope null. Those cells must be reported as not applicable, not failed.",
        "",
        "## Corrected gate display from the immutable metrics JSON",
        "",
        "| Slope per decade | Type-I gate | Coverage gate | Other frozen gates | Cell decision |",
        "|---:|---|---|---|---|",
    ]
    for row in metrics["cells"]:
        slope = float(row["slope_per_decade"])
        gates = row["acceptance_gates"]
        type_i = gate_label("type_i_rate", gates.get("type_i_rate"), slope)
        coverage = gate_label("coverage_rate_and_exact_interval", gates.get("coverage_rate_and_exact_interval"), slope)
        other_names = ("slope_bias", "numerical_failures_integer_gate", "inferential_availability", "ci_width_over_oracle", "residual_dependence")
        others = ", ".join(f"{name}={gate_label(name, gates.get(name), slope)}" for name in other_names)
        decision = "PASS" if row.get("all_gates_pass") is True else "FAIL"
        lines.append(f"| {slope:.2f} | {type_i} | {coverage} | {others} | {decision} |")
    failed = [row for row in metrics["cells"] if row.get("all_gates_pass") is not True]
    if len(failed) != 1 or float(failed[0]["slope_per_decade"]) != 0.5:
        raise ValueError("Observed failure pattern differs from the recorded confirmation evidence")
    failed_coverage = failed[0]
    lo, hi = failed_coverage["coverage_cp95"]
    coverage_denominator = int(failed_coverage["inferential_output_count"])
    if abs(failed_coverage["coverage_count"] / coverage_denominator - failed_coverage["coverage_rate"]) > 1e-12:
        raise ValueError("Coverage count and inferential-output denominator disagree")
    lines.extend([
        "",
        "## Scientific result",
        "",
        f"The candidate still fails independent confirmation. The +0.50 slope cell has coverage {failed_coverage['coverage_count']}/{coverage_denominator} = {100 * failed_coverage['coverage_rate']:.2f}% (exact 95% interval {100 * lo:.2f}%–{100 * hi:.2f}%), below both the frozen point-rate floor and the required nominal-coverage interval criterion. Two of the 1,000 supported fits did not yield usable inferential output; coverage is conditional on the {coverage_denominator} usable intervals. The zero-slope Type-I cell and +0.25 cell pass their applicable gates. No method is promoted, and no v1.3 inference registry exists.",
        "",
        "## Immutable evidence provenance",
        "",
        f"- Metrics JSON: `{METRICS_PATH.relative_to(ROOT).as_posix()}` — SHA-256 `{sha256(METRICS_PATH)}`.",
        f"- Original report: `{ORIGINAL_REPORT_PATH.relative_to(ROOT).as_posix()}` — SHA-256 `{sha256(ORIGINAL_REPORT_PATH)}`.",
        f"- Run manifest SHA-256 from metrics: `{metrics['run_manifest_sha256']}`.",
        f"- Frozen-selection manifest SHA-256 from metrics: `{metrics['selection_freeze_sha256']}`.",
        "- This addendum does not rewrite the original report or its recorded hash.",
        "",
    ])
    return "\n".join(lines)


def main(output_path: Path = OUTPUT_PATH) -> None:
    if not METRICS_PATH.is_file() or not ORIGINAL_REPORT_PATH.is_file():
        raise FileNotFoundError("Expected immutable confirmation metrics or report is missing")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    content = render(metrics)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "ADDITIVE_CORRECTION_WRITTEN",
        "path": str(output_path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(output_path),
        "metrics_sha256": sha256(METRICS_PATH),
        "original_report_sha256": sha256(ORIGINAL_REPORT_PATH),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    main(args.output)
