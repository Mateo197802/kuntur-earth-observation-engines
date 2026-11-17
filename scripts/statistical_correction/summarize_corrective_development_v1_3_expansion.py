"""Verify and summarize the frozen KUNTUR v1.3 development expansion."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from kuntur_data.phase2.statistical_v12 import _design

CONFIG_PATH = ROOT / "configs/statistical_corrective_development_expansion_v1_3.json"
PROTOCOL_PATH = ROOT / "docs/protocols/statistical_corrective_development_expansion_v1_3.md"
RUNNER_PATH = ROOT / "scripts/statistical_correction/run_corrective_development_v1_3_expansion.py"
OUTPUT = ROOT / "results/statistical_correction/corrective_development_v1_3/expanded_20261117"
CHECKPOINTS = OUTPUT / "checkpoints"
MANIFEST_PATH = OUTPUT / "development_manifest.json"
METRICS_PATH = OUTPUT / "development_metrics.json"
REPORT_PATH = ROOT / "reports/statistical_correction/v1_3_development_candidate_selection_20261117.md"
CLOSURE_PATH = ROOT / "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mean_ci(values: Iterable[float]) -> dict[str, Any]:
    data = np.asarray([float(value) for value in values if value is not None and np.isfinite(float(value))], dtype=float)
    if not len(data):
        return {"n": 0, "mean": None, "mcse": None, "ci95_t": None}
    mean = float(data.mean())
    if len(data) == 1:
        return {"n": 1, "mean": mean, "mcse": None, "ci95_t": None}
    sd = float(data.std(ddof=1))
    mcse = sd / math.sqrt(len(data))
    half = float(student_t.ppf(0.975, len(data) - 1)) * mcse
    return {"n": int(len(data)), "mean": mean, "mcse": mcse, "ci95_t": [mean - half, mean + half]}


def cp_contains(interval: list[float] | None, nominal: float) -> bool | None:
    if not interval or len(interval) != 2:
        return None
    return bool(interval[0] <= nominal <= interval[1])


def numeric_values(records: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in records if row.get(key) is not None and np.isfinite(float(row[key]))]


def summarize_cell(cell: dict[str, Any]) -> dict[str, Any]:
    records = cell["replicates"]
    config = cell["config"]
    slope = float(config["slope_per_decade"])
    eligible = [row for row in records if row.get("support_eligible")]
    converged = [row for row in eligible if row.get("converged")]
    inferential = [row for row in converged if row.get("inferential_output")]
    intervals = [row for row in converged if row.get("ci_low") is not None and row.get("ci_high") is not None]
    failures = len(eligible) - len(converged)
    rejects = sum(bool(row.get("reject_at_005")) for row in inferential)
    type_i = slope == 0.0
    rejection_interval = None
    rejection_rate = None
    if type_i and inferential:
        rejection_rate = rejects / len(inferential)
        interval = cell.get("cp95") or [None, None]
        rejection_interval = list(interval)
    cover_count = sum(bool(row.get("ci_covers_true_slope")) for row in intervals)
    coverage = cover_count / len(intervals) if intervals else None
    residual_lb = [
        row["residual_diagnostic"].get("ljung_box_failure")
        for row in eligible
        if row.get("residual_diagnostic", {}).get("kind") == "LJUNG_BOX_COMPLETE_SUPPORT"
        and row["residual_diagnostic"].get("ljung_box_failure") is not None
    ]
    gap_acf: dict[str, Any] = {}
    for lag in ("1", "12", "24"):
        vals = [
            row["residual_diagnostic"].get("acf_lags_1_12_24", {}).get(lag, {}).get("acf")
            for row in eligible
            if row.get("residual_diagnostic", {}).get("kind") == "EXACT_CALENDAR_ACF_DESCRIPTIVE_ONLY"
            and row["residual_diagnostic"].get("acf_lags_1_12_24", {}).get(lag, {}).get("acf") is not None
        ]
        pairs = [
            row["residual_diagnostic"].get("acf_lags_1_12_24", {}).get(lag, {}).get("pair_count")
            for row in eligible
            if row.get("residual_diagnostic", {}).get("kind") == "EXACT_CALENDAR_ACF_DESCRIPTIVE_ONLY"
            and row["residual_diagnostic"].get("acf_lags_1_12_24", {}).get(lag, {}).get("acf") is not None
        ]
        gap_acf[lag] = {**mean_ci(vals), "mean_pair_count": float(np.mean(pairs)) if pairs else None}
    estimates = numeric_values(converged, "estimate_per_decade")
    error = [value - slope for value in estimates]
    abs_error = [abs(value) for value in error]
    width = numeric_values(intervals, "ci_width")
    width_ratio = numeric_values(intervals, "ci_width_over_oracle")
    out: dict[str, Any] = {
        "cell_key": cell["cell_key"],
        "regime": config["regime"],
        "slope_per_decade": slope,
        "method": config["method_name"],
        "status": cell["status"],
        "outer_replicates_completed": len(records),
        "support_eligible": len(eligible),
        "converged": len(converged),
        "numerical_failures": failures,
        "numerical_failure_rate": failures / len(eligible) if eligible else None,
        "integer_gate_at_1pct": bool(failures * 100 <= len(eligible)) if eligible else None,
        "convergence_cp95": None,
        "inferential_output_count": len(inferential),
        "inferential_availability": len(inferential) / len(eligible) if eligible else None,
        "inferential_availability_cp95": cell.get("availability_cp95"),
        "type_i_rejections": rejects if type_i else None,
        "type_i_rate": rejection_rate,
        "type_i_cp95": rejection_interval,
        "type_i_cp95_contains_0p05": cp_contains(rejection_interval, 0.05) if type_i else None,
        "interval_count": len(intervals),
        "coverage_count": cover_count if intervals else None,
        "coverage_rate": coverage,
        "coverage_cp95": cell.get("coverage_cp95"),
        "coverage_cp95_contains_0p95": cp_contains(cell.get("coverage_cp95"), 0.95) if intervals else None,
        "slope_bias_per_decade": mean_ci(error),
        "slope_mae_per_decade": mean_ci(abs_error),
        "slope_rmse_per_decade": float(np.sqrt(np.mean(np.square(error)))) if error else None,
        "mean_ci_width": mean_ci(width),
        "mean_ci_width_over_oracle": mean_ci(width_ratio),
        "power": rejects / len(inferential) if not type_i and inferential else None,
        "power_cp95": cell.get("power_cp95") if not type_i else None,
        "residual_ljung_box_applicable_replicates": len(residual_lb),
        "residual_ljung_box_failures": int(sum(bool(flag) for flag in residual_lb)) if residual_lb else None,
        "residual_ljung_box_failure_rate": float(np.mean(residual_lb)) if residual_lb else None,
        "residual_ljung_box_cp95": cell.get("residual_cp95") if residual_lb else None,
        "gap_exact_calendar_acf_descriptive_only": gap_acf,
        "known_dgp_error_acf_lags_1_12_24": {
            lag: mean_ci([
                row.get("known_dgp_error_acf_lags_1_12_24", {}).get(lag, {}).get("acf")
                for row in records
                if row.get("known_dgp_error_acf_lags_1_12_24", {}).get(lag, {}).get("acf") is not None
            ])
            for lag in ("1", "12", "24")
        },
        "development_only_not_validated": True,
    }
    # Populate intervals from replicate outcomes; stored manifest summaries are
    # deliberately not trusted as a substitute for checkpoint reconstruction.
    if eligible:
        successes = len(converged)
        from scipy.stats import beta
        out["convergence_cp95"] = [
            0.0 if successes == 0 else float(beta.ppf(0.025, successes, len(eligible) - successes + 1)),
            1.0 if successes == len(eligible) else float(beta.ppf(0.975, successes + 1, len(eligible) - successes)),
        ]
    if inferential:
        from scipy.stats import beta
        successes = len(inferential)
        out["inferential_availability_cp95"] = [
            0.0 if successes == 0 else float(beta.ppf(0.025, successes, len(eligible) - successes + 1)),
            1.0 if successes == len(eligible) else float(beta.ppf(0.975, successes + 1, len(eligible) - successes)),
        ]
    if intervals:
        from scipy.stats import beta
        successes = cover_count
        n = len(intervals)
        out["coverage_cp95"] = [
            0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1)),
            1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes)),
        ]
    if type_i and inferential:
        from scipy.stats import beta
        n = len(inferential)
        out["type_i_cp95"] = [
            0.0 if rejects == 0 else float(beta.ppf(0.025, rejects, n - rejects + 1)),
            1.0 if rejects == n else float(beta.ppf(0.975, rejects + 1, n - rejects)),
        ]
        out["type_i_cp95_contains_0p05"] = cp_contains(out["type_i_cp95"], 0.05)
    elif not type_i and inferential:
        from scipy.stats import beta
        n = len(inferential)
        out["power_cp95"] = [
            0.0 if rejects == 0 else float(beta.ppf(0.025, rejects, n - rejects + 1)),
            1.0 if rejects == n else float(beta.ppf(0.975, rejects + 1, n - rejects)),
        ]
    if residual_lb:
        from scipy.stats import beta
        successes = int(sum(bool(flag) for flag in residual_lb))
        n = len(residual_lb)
        out["residual_ljung_box_cp95"] = [
            0.0 if successes == 0 else float(beta.ppf(0.025, successes, n - successes + 1)),
            1.0 if successes == n else float(beta.ppf(0.975, successes + 1, n - successes)),
        ]
    return out


def paired_difference(left: list[float], right: list[float]) -> dict[str, Any]:
    length = min(len(left), len(right))
    return mean_ci([left[index] - right[index] for index in range(length)])


def build_paired(summaries: list[dict[str, Any]], cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(s["regime"], s["slope_per_decade"], s["method"]): s for s in summaries}
    output: list[dict[str, Any]] = []
    for summary in summaries:
        regime, slope, method = summary["regime"], summary["slope_per_decade"], summary["method"]
        baseline_key = (regime, slope, "SARIMAX_AR1")
        if baseline_key not in lookup:
            continue
        baseline = cells[lookup[baseline_key]["cell_key"]]
        candidate = cells[summary["cell_key"]]
        left_map = {row["replicate"]: row for row in candidate["replicates"]}
        right_map = {row["replicate"]: row for row in baseline["replicates"]}
        common = sorted(set(left_map) & set(right_map))
        if any(left_map[index]["series_seed"] != right_map[index]["series_seed"] for index in common):
            raise RuntimeError(f"Paired methods used different generated series: {regime}, {slope}, {method}")
        error_diff: list[float] = []
        reject_diff: list[float] = []
        coverage_diff: list[float] = []
        width_diff: list[float] = []
        for index in common:
            a, b = left_map[index], right_map[index]
            ea, eb = a.get("estimate_per_decade"), b.get("estimate_per_decade")
            if a.get("converged") and b.get("converged") and ea is not None and eb is not None:
                error_diff.append(abs(float(ea) - slope) - abs(float(eb) - slope))
            if a.get("inferential_output") and b.get("inferential_output"):
                reject_diff.append(float(bool(a.get("reject_at_005"))) - float(bool(b.get("reject_at_005"))))
            if a.get("ci_covers_true_slope") is not None and b.get("ci_covers_true_slope") is not None and a.get("ci_low") is not None and b.get("ci_low") is not None:
                coverage_diff.append(float(bool(a["ci_covers_true_slope"])) - float(bool(b["ci_covers_true_slope"])))
            if a.get("ci_width_over_oracle") is not None and b.get("ci_width_over_oracle") is not None:
                width_diff.append(float(a["ci_width_over_oracle"]) - float(b["ci_width_over_oracle"]))
        output.append({
            "regime": regime,
            "slope_per_decade": slope,
            "method": method,
            "comparator": "SARIMAX_AR1",
            "paired_series_count": len(common),
            "paired_abs_slope_error_difference": paired_difference(error_diff, [0.0] * len(error_diff)),
            "paired_rejection_or_power_difference": paired_difference(reject_diff, [0.0] * len(reject_diff)),
            "paired_interval_coverage_difference": paired_difference(coverage_diff, [0.0] * len(coverage_diff)),
            "paired_oracle_normalized_width_difference": paired_difference(width_diff, [0.0] * len(width_diff)),
        })
    return output


def gate_evaluation(config: dict[str, Any], summaries: list[dict[str, Any]]) -> dict[str, Any]:
    gates = config["frozen_selection_rule"]
    by = {(item["regime"], item["method"], item["slope_per_decade"]): item for item in summaries}
    report: dict[str, Any] = {}
    for regime, methods in config["regimes"].items():
        regime_rows = [
            by.get((regime, method, float(slope)))
            for method in methods
            for slope in config["slopes_per_decade"]
        ]
        if any(
            row is None
            or row["status"] != "COMPLETE"
            or row["outer_replicates_completed"] != int(config["outer_replicates_per_cell"])
            for row in regime_rows
        ):
            report[regime] = {
                "selected_development_candidate": None,
                "independent_confirmation_candidate": None,
                "selection_status": "NOT_EVALUATED_REGIME_INCOMPLETE",
                "methods": [],
            }
            continue
        method_results: list[dict[str, Any]] = []
        for method in methods:
            rows = [by[(regime, method, float(slope))] for slope in config["slopes_per_decade"]]
            has_ci = all(row["interval_count"] > 0 for row in rows)
            reasons: list[str] = []
            if not has_ci:
                reasons.append("NO_CONFIDENCE_INTERVAL_IMPLEMENTATION")
            null = rows[0]
            trange = gates["type_i_rate_range"]
            if null["type_i_rate"] is None or not (trange[0] <= null["type_i_rate"] <= trange[1]):
                reasons.append("TYPE_I_POINT_RATE_OUTSIDE_GATE")
            if null["type_i_cp95_contains_0p05"] is not True:
                reasons.append("TYPE_I_EXACT_INTERVAL_DOES_NOT_CONTAIN_0P05")
            crange = gates["coverage_range"]
            for row in rows:
                if row["coverage_rate"] is None or not (crange[0] <= row["coverage_rate"] <= crange[1]):
                    reasons.append(f"COVERAGE_POINT_RATE_OUTSIDE_GATE_AT_SLOPE_{row['slope_per_decade']:.2f}")
                if row["coverage_cp95_contains_0p95"] is not True:
                    reasons.append(f"COVERAGE_EXACT_INTERVAL_DOES_NOT_CONTAIN_0P95_AT_SLOPE_{row['slope_per_decade']:.2f}")
                bias = row["slope_bias_per_decade"]["mean"]
                if bias is None or abs(bias) > float(gates["max_absolute_slope_bias_per_decade"]):
                    reasons.append(f"SLOPE_BIAS_OUTSIDE_GATE_AT_SLOPE_{row['slope_per_decade']:.2f}")
                if row["integer_gate_at_1pct"] is not True:
                    reasons.append(f"NUMERICAL_FAILURE_GATE_FAIL_AT_SLOPE_{row['slope_per_decade']:.2f}")
                availability = row["inferential_availability"]
                if availability is None or availability < float(gates["min_inferential_availability"]):
                    reasons.append(f"INFERENTIAL_AVAILABILITY_FAIL_AT_SLOPE_{row['slope_per_decade']:.2f}")
                ratio = row["mean_ci_width_over_oracle"]["mean"]
                if ratio is None or ratio > float(gates["max_mean_ci_width_over_oracle"]):
                    reasons.append(f"INTERVAL_WIDTH_RATIO_FAIL_AT_SLOPE_{row['slope_per_decade']:.2f}")
            residual_applicable = method in {"FEGLS_AR1", "SARIMAX_AR1", "SARIMAX_ARMA11", "SARIMAX_SEASONAL_AR12"}
            if residual_applicable:
                if regime in {"ar1_060_mcar20", "ar1_060_seasonal_missing"}:
                    reasons.append("IRREGULAR_SUPPORT_RESIDUAL_DEPENDENCE_GATE_UNCALIBRATED")
                else:
                    for row in rows:
                        rate = row["residual_ljung_box_failure_rate"]
                        if rate is None:
                            reasons.append(f"RESIDUAL_DIAGNOSTIC_UNAVAILABLE_AT_SLOPE_{row['slope_per_decade']:.2f}")
                        elif rate > float(gates["max_residual_dependence_failure_rate"]):
                            reasons.append(f"RESIDUAL_DEPENDENCE_FAILURE_RATE_OUTSIDE_GATE_AT_SLOPE_{row['slope_per_decade']:.2f}")
            passed = not reasons
            min_power = min((row["power"] for row in rows[1:] if row["power"] is not None), default=None)
            avg_width = float(np.mean([row["mean_ci_width_over_oracle"]["mean"] for row in rows if row["mean_ci_width_over_oracle"]["mean"] is not None])) if has_ci else None
            method_results.append({
                "method": method,
                "full_inference_candidate": has_ci,
                "passed_all_development_gates": passed,
                "gate_failures_or_unresolved": sorted(set(reasons)),
                "minimum_power_plus_025_plus_050": min_power,
                "mean_oracle_normalized_width_across_slopes": avg_width,
            })
        eligible = [item for item in method_results if item["passed_all_development_gates"]]
        selected = None
        if eligible:
            eligible.sort(
                key=lambda item: (
                    -(item["minimum_power_plus_025_plus_050"] if item["minimum_power_plus_025_plus_050"] is not None else -1.0),
                    item["mean_oracle_normalized_width_across_slopes"] if item["mean_oracle_normalized_width_across_slopes"] is not None else float("inf"),
                    item["method"],
                )
            )
            selected = eligible[0]["method"]
        report[regime] = {
            "selected_development_candidate": selected,
            "independent_confirmation_candidate": selected,
            "selection_status": "CANDIDATE_READY_FOR_FREEZE_BEFORE_CONFIRMATION" if selected else "NO_ADEQUATELY_SUPPORTED_CANDIDATE",
            "methods": method_results,
        }
    return report


def markdown_report(metrics: dict[str, Any]) -> str:
    rows = metrics["cell_summaries"]
    lines = [
        "# KUNTUR v1.3 Corrective Development Candidate Selection",
        "",
        f"**Status:** {metrics['run_status']} — development evidence only. No statistical inference registry entry was created and no independent confirmation was launched.",
        "",
        f"Generated: {metrics['generated_utc']}",
        "",
        "## Execution",
        "",
        f"- Planned: {metrics['plan']['planned_fit_count']:,} outer fits across {metrics['plan']['cell_count']} paired cells and {metrics['plan']['method_regime_combinations']} method-regime combinations.",
        f"- Completed: {metrics['completed_outer_fits']:,} fits in {metrics['completed_cells']}/{metrics['plan']['cell_count']} complete cells.",
        f"- Accumulated process CPU: {metrics['resource_usage']['cpu_seconds']:.1f} s ({metrics['resource_usage']['cpu_hours']:.3f} h); active wall: {metrics['resource_usage']['active_wall_seconds']:.1f} s; peak RSS: {metrics['resource_usage']['peak_rss_bytes']:,} bytes.",
        "- Sampling is paired by regime, slope and replicate. Exact binomial intervals are Clopper–Pearson 95% intervals; continuous metric intervals are t-based Monte Carlo intervals.",
        "",
        "## Candidate decision",
        "",
    ]
    for regime, decision in metrics["candidate_selection"].items():
        lines.append(f"- **{regime}: {decision['selection_status']}**" + (f" — {decision['selected_development_candidate']}" if decision["selected_development_candidate"] else ""))
    if not any(item["selected_development_candidate"] for item in metrics["candidate_selection"].values()):
        incomplete = any(item["selection_status"] == "NOT_EVALUATED_REGIME_INCOMPLETE" for item in metrics["candidate_selection"].values())
        if incomplete:
            lines.extend(["", "No independent confirmation is proposed or started. At least one regime is incomplete, so its selection gates were not evaluated; completed regimes are identified separately above."])
        else:
            lines.extend(["", "No candidate met every predeclared development gate, so independent confirmation is not proposed or started."])
    else:
        lines.extend(["", "Development candidates are not validated. Freeze their exact code, regime scope, gates and independent namespace before any confirmation run."])
    lines.extend([
        "",
        "## Type-I calibration and coverage",
        "",
        "| Regime | Method | Slope | Type-I / power | 95% binomial interval | Coverage | 95% coverage interval |",
        "|---|---|---:|---:|---|---:|---|",
    ])
    for row in rows:
        if row["slope_per_decade"] == 0.0:
            test_rate, test_ci = row["type_i_rate"], row["type_i_cp95"]
        else:
            test_rate, test_ci = row["power"], row["power_cp95"]
        fmt = lambda value: "NA" if value is None else f"{value:.3f}"
        ci_fmt = lambda value: "NA" if value is None else f"[{value[0]:.3f}, {value[1]:.3f}]"
        lines.append(
            f"| {row['regime']} | {row['method']} | {row['slope_per_decade']:.2f} | {fmt(test_rate)} | {ci_fmt(test_ci)} | {fmt(row['coverage_rate'])} | {ci_fmt(row['coverage_cp95'])} |"
        )
    lines.extend([
        "",
        "## Convergence, bias and interval efficiency",
        "",
        "| Regime | Method | Slope | Failures / supported | Availability | Bias (95% MC interval) | Mean width/oracle |",
        "|---|---|---:|---:|---:|---|---:|",
    ])
    for row in rows:
        bias = row["slope_bias_per_decade"]
        bias_text = "NA" if bias["mean"] is None else f"{bias['mean']:.4f} [{bias['ci95_t'][0]:.4f}, {bias['ci95_t'][1]:.4f}]" if bias["ci95_t"] else f"{bias['mean']:.4f}"
        ratio = row["mean_ci_width_over_oracle"]["mean"]
        ratio_text = "NA" if ratio is None else f"{ratio:.3f}"
        availability = row["inferential_availability"]
        availability_text = "NA" if availability is None else f"{availability:.3f}"
        lines.append(
            f"| {row['regime']} | {row['method']} | {row['slope_per_decade']:.2f} | {row['numerical_failures']}/{row['support_eligible']} | {availability_text} | {bias_text} | {ratio_text} |"
        )
    lines.extend([
        "",
        "## Residual dependence",
        "",
        "Complete-support dependence-model diagnostics use Ljung–Box at calendar lags 12 and 24, with p < .01 marked as a residual-dependence failure. Irregular-calendar exact-offset ACFs and pair counts are descriptive and do not pass the residual-dependence gate.",
        "",
        "| Regime | Method | Slope | Ljung–Box failures / tested | Exact-calendar ACF (lag 1 / 12 / 24) |",
        "|---|---|---:|---:|---|",
    ])
    for row in rows:
        acf = row["gap_exact_calendar_acf_descriptive_only"]
        acf_text = "NA" if not any(acf[lag]["n"] for lag in ("1", "12", "24")) else " / ".join(
            "NA" if acf[lag]["mean"] is None else f"{acf[lag]['mean']:.3f} (n={acf[lag]['n']})" for lag in ("1", "12", "24")
        )
        failures = row["residual_ljung_box_failures"]
        tested = row["residual_ljung_box_applicable_replicates"]
        lines.append(f"| {row['regime']} | {row['method']} | {row['slope_per_decade']:.2f} | {failures if failures is not None else 'NA'} / {tested} | {acf_text} |")
    lines.extend([
        "",
        "## Paired comparisons",
        "",
        "The machine-readable metrics file gives paired mean differences versus SARIMAX AR(1) by regime and slope, with t-based 95% Monte Carlo intervals for absolute slope error, rejection/power, interval coverage and oracle-normalized interval width. Only generated-series streams are shared across methods; estimator streams are method-specific.",
        "",
        "## v1.2 selection root cause",
        "",
        "The original reconstruction is preserved in results/statistical_correction/selection_reconstruction_v1_3.json. It found no arithmetic score error. The score was capability-asymmetric because methods without confidence intervals skipped coverage penalties; only one candidate passed the MCAR selection gates because exact-calendar diagnostics prescribed for missing support were not implemented and unavailable diagnostics were treated as failed. With 500 selection replications, Monte Carlo standard error near a 5% rate was about .010. Selection versus independent validation differences were compatible with sampling variability (two-sided Fisher exact p=.225 for IID and p=.112 for MCAR), but variability does not negate the independent failures. Misspecification was not established as a sole cause.",
        "",
        "## Frozen evidence and reproducibility",
        "",
        f"- Expanded run manifest: {metrics['manifest_path']} (SHA-256 {metrics['manifest_sha256']}).",
        f"- Metrics JSON: {METRICS_PATH.relative_to(ROOT).as_posix()}; the exact metrics and report SHA-256 values are recorded in summary_receipt.json.",
        f"- Protocol SHA-256: {metrics['source_hashes']['protocol']}; config: {metrics['source_hashes']['configuration']}; runner: {metrics['source_hashes']['script']}; summarizer: {metrics['source_hashes']['summarizer']}.",
        f"- v1.2 closure manifest SHA-256: {metrics['closure_manifest_sha256']}. Older reports remain historical snapshots and are reconciled only by {metrics['v1_2_reconciliation_addendum']}.",
        "- v1.2 historical evidence, validated inference registry, real-series scores and the 2025 hindcast were not changed. No publication or merge occurred.",
        "- Reproduce/verify with: python scripts/statistical_correction/run_corrective_development_v1_3_expansion.py --plan-only; python scripts/statistical_correction/run_corrective_development_v1_3_expansion.py; python scripts/statistical_correction/summarize_corrective_development_v1_3_expansion.py.",
        "",
        "## Bounded independent-confirmation proposal",
        "",
        "Do not run confirmation unless this report identifies a candidate and its exact code, regime scope and acceptance gates are frozen. If a future candidate qualifies, use only the untouched namespace KUNTUR-STAT-CORRECTIVE-INDEPENDENT-CONFIRMATION-V1.3-2027-01-15, 1,000 paired outer series for null, +.25 and +.50 per-decade cells, exact two-sided intervals, the unchanged v1.2 gates, a 2× measured pilot allowance capped at 9 CPU-hours, 10 active wall-hours and 1 GiB RSS, 50-replicate atomic checkpoints, and no reuse of development or v1.2 streams. This proposal is not authorization to start the phase.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    if not MANIFEST_PATH.exists():
        raise SystemExit(f"Development manifest not found: {MANIFEST_PATH}")
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("source_hashes", {}).get("configuration") != sha256(CONFIG_PATH):
        raise RuntimeError("Manifest config hash no longer matches")
    if manifest.get("source_hashes", {}).get("protocol") != sha256(PROTOCOL_PATH):
        raise RuntimeError("Manifest protocol hash no longer matches")
    if manifest.get("source_hashes", {}).get("script") != sha256(RUNNER_PATH):
        raise RuntimeError("Manifest runner hash no longer matches")

    cell_data: dict[str, dict[str, Any]] = {}
    checkpoint_rows: list[dict[str, Any]] = []
    expected = 0
    for regime, methods in config["regimes"].items():
        for slope in config["slopes_per_decade"]:
            for method in methods:
                expected += 1
                slope_label = f"{float(slope):.2f}".replace("-", "m").replace(".", "p")
                path = CHECKPOINTS / f"cell_{regime}_n276_s{slope_label}_{method.lower()}.json"
                if not path.exists():
                    continue
                cell = json.loads(path.read_text(encoding="utf-8"))
                if cell.get("classification") != "DEVELOPMENT_ONLY_NOT_VALIDATED":
                    raise RuntimeError(f"Unexpected classification: {path.name}")
                if cell.get("source_hashes") != manifest.get("source_hashes"):
                    raise RuntimeError(f"Source hash mismatch: {path.name}")
                if cell.get("completed_replicates") != len(cell.get("replicates", [])):
                    raise RuntimeError(f"Replicate count mismatch: {path.name}")
                if [row.get("replicate") for row in cell["replicates"]] != list(range(len(cell["replicates"]))):
                    raise RuntimeError(f"Replicate IDs are not contiguous: {path.name}")
                seeds = [row["series_seed"] for row in cell["replicates"]]
                if len(seeds) != len(set(seeds)):
                    raise RuntimeError(f"Duplicate series seed within cell: {path.name}")
                cell["cp95"] = None
                inferential = [row for row in cell["replicates"] if row.get("converged") and row.get("inferential_output")]
                if float(slope) == 0.0 and inferential:
                    from scipy.stats import beta
                    rejections = sum(bool(row.get("reject_at_005")) for row in inferential)
                    n = len(inferential)
                    cell["cp95"] = [
                        0.0 if rejections == 0 else float(beta.ppf(0.025, rejections, n - rejections + 1)),
                        1.0 if rejections == n else float(beta.ppf(0.975, rejections + 1, n - rejections)),
                    ]
                intervals = [
                    row for row in cell["replicates"]
                    if row.get("converged") and row.get("ci_low") is not None and row.get("ci_high") is not None
                ]
                if intervals:
                    from scipy.stats import beta
                    covered = sum(bool(row.get("ci_covers_true_slope")) for row in intervals)
                    n = len(intervals)
                    cell["coverage_cp95"] = [
                        0.0 if covered == 0 else float(beta.ppf(0.025, covered, n - covered + 1)),
                        1.0 if covered == n else float(beta.ppf(0.975, covered + 1, n - covered)),
                    ]
                cell["availability_cp95"] = None
                eligible = [row for row in cell["replicates"] if row.get("support_eligible")]
                if eligible:
                    from scipy.stats import beta
                    available = sum(bool(row.get("inferential_output")) for row in eligible)
                    n = len(eligible)
                    cell["availability_cp95"] = [
                        0.0 if available == 0 else float(beta.ppf(0.025, available, n - available + 1)),
                        1.0 if available == n else float(beta.ppf(0.975, available + 1, n - available)),
                    ]
                residuals = [
                    row["residual_diagnostic"].get("ljung_box_failure")
                    for row in cell["replicates"]
                    if row.get("residual_diagnostic", {}).get("kind") == "LJUNG_BOX_COMPLETE_SUPPORT"
                    and row["residual_diagnostic"].get("ljung_box_failure") is not None
                ]
                cell["residual_cp95"] = None
                if residuals:
                    from scipy.stats import beta
                    n = len(residuals)
                    positive = sum(bool(value) for value in residuals)
                    cell["residual_cp95"] = [
                        0.0 if positive == 0 else float(beta.ppf(0.025, positive, n - positive + 1)),
                        1.0 if positive == n else float(beta.ppf(0.975, positive + 1, n - positive)),
                    ]
                key = cell["cell_key"]
                cell_data[key] = cell
                checkpoint_rows.append({
                    "path": path.relative_to(ROOT).as_posix(),
                    "status": cell["status"],
                    "completed_replicates": len(cell["replicates"]),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "cell_key": key,
                })

    summaries = [summarize_cell(cell) for cell in cell_data.values()]
    summaries.sort(key=lambda item: (item["regime"], item["method"], item["slope_per_decade"]))
    paired = build_paired(summaries, cell_data)
    decisions = gate_evaluation(config, summaries)
    closure_hash = sha256(CLOSURE_PATH) if CLOSURE_PATH.exists() else None
    closure = json.loads(CLOSURE_PATH.read_text(encoding="utf-8")) if CLOSURE_PATH.exists() else {}
    registry_record = closure.get("validated_registry", {})
    hindcast_record = closure.get("frozen_2025_hindcast", {})
    registry_path = ROOT / str(registry_record.get("path", ""))
    hindcast_path = ROOT / str(hindcast_record.get("path", ""))
    registry_actual_hash = sha256(registry_path) if registry_path.exists() else None
    hindcast_actual_hash = sha256(hindcast_path) if hindcast_path.exists() else None
    if registry_actual_hash != registry_record.get("sha256"):
        raise RuntimeError("Validated inference registry changed from the v1.2 closure receipt")
    if hindcast_actual_hash != hindcast_record.get("sha256"):
        raise RuntimeError("The sealed 2025 hindcast score changed from the v1.2 closure receipt")
    metric_payload = {
        "schema_version": "kuntur-v1.3-development-expansion-metrics-v1",
        "generated_utc": utc_now(),
        "run_status": manifest.get("status"),
        "plan": manifest.get("plan"),
        "completed_cells": manifest.get("completed_cells"),
        "completed_outer_fits": manifest.get("completed_outer_fits"),
        "resource_usage": {
            "cpu_seconds": manifest.get("cumulative_process_cpu_seconds", 0.0),
            "cpu_hours": float(manifest.get("cumulative_process_cpu_seconds", 0.0)) / 3600.0,
            "active_wall_seconds": manifest.get("cumulative_active_wall_seconds", 0.0),
            "peak_rss_bytes": manifest.get("peak_rss_bytes_observed", 0),
        },
        "source_hashes": {
            **manifest.get("source_hashes", {}),
            "summarizer": sha256(Path(__file__).resolve()),
        },
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "checkpoint_inventory": checkpoint_rows,
        "candidate_selection": decisions,
        "cell_summaries": summaries,
        "paired_vs_sarimax_ar1": paired,
        "closure_manifest_sha256": closure_hash,
        "v1_2_closure_status": closure.get("closure_status"),
        "protected_v1_2_evidence": {
            "validated_registry_path": registry_record.get("path"),
            "validated_registry_sha256_at_closure": registry_record.get("sha256"),
            "validated_registry_sha256_current": registry_actual_hash,
            "sealed_2025_score_path": hindcast_record.get("path"),
            "sealed_2025_score_sha256_at_closure": hindcast_record.get("sha256"),
            "sealed_2025_score_sha256_current": hindcast_actual_hash,
        },
        "v1_2_reconciliation_addendum": "reports/statistical_correction/v1_2_report_reconciliation_addendum_v1_3.md",
        "registered_inference_registry_modified": False,
        "2025_predictive_test_rescored": False,
        "independent_confirmation_started": False,
        "development_only_not_validated": True,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metric_payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report_text = markdown_report(metric_payload)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report_text, encoding="utf-8")
    # The metrics payload is written before this digest is known; attach an
    # external receipt so its final hash can be checked without self-reference.
    receipt = {
        "metrics_path": METRICS_PATH.relative_to(ROOT).as_posix(),
        "metrics_sha256": sha256(METRICS_PATH),
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
        "report_sha256": sha256(REPORT_PATH),
        "generated_utc": utc_now(),
    }
    (OUTPUT / "summary_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": metric_payload["run_status"],
        "completed_cells": metric_payload["completed_cells"],
        "expected_cells": expected,
        "completed_fits": metric_payload["completed_outer_fits"],
        "candidate_selection": {
            regime: result["selection_status"]
            for regime, result in decisions.items()
        },
        "metrics_sha256": receipt["metrics_sha256"],
        "report_sha256": receipt["report_sha256"],
        "metrics_path": METRICS_PATH.relative_to(ROOT).as_posix(),
        "report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
