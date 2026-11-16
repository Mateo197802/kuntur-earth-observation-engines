from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels

from kuntur_data.phase2.statistical_v12 import METHOD_IDS, analyze_variable_v12
from scripts.statistical_correction.run_calibration import METHODS, sha256


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statistical_correction"
REPORTS = ROOT / "reports/statistical_correction"
EXPECTED_GOLD_SHA256 = "5883fe2f7af8a8d0a58dd87c6bc90340a157c71e0d3fed51fff3a850eb2368b2"
EXPECTED_FROZEN_ARTIFACT_HASHES = {
    "data/manifests/gold_manifest_v1.json": "c5bdac892d92670c3c52e2cf2fe268e00890e86b0e3cdbe07d5a040f139c5543",
    "results/phase2/public/analytical_trends_v1.csv": "6387fa1aa63ea737aacd0c25a000fd2e9d054c57931b290c1d28de223a810608",
    "docs/data_dictionary_v1.json": "6b4a0cf6221161fb6cab3d30648abe0f0e17067cdda91e8396a58757a8b69287",
    "docs/monthly_data_contract.md": "25138bb005d18739344403023d7f8479691bf29799aea9a8428199b20fb89a78",
}
EXPECTED_FROZEN_SPLIT_HASHES = {
    "data/derived/phase2/forecast_phase2_split_v1.parquet": "0a64f2b78bbfe99937b95f2dc817cc037e3583625740e1b84b058a22bbf9bf4d",
    "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet": "67c69236a8516b11b900b90f63926c1047aae3af991b726c567b5c85020ce2ff",
    "data/gold/kuntur_forecast_split_indices_v1.parquet": "c2fbca4ac9636622a08e778913e2edfb23cc210453bf05114e1b3775b9b0726c",
    "data/manifests/forecast_split_manifest_v1.json": "6b72ab2f5db8107e9980f5e0435f94a3fe1220a4de8e055d5f6dae5e6332258e",
    "data/manifests/phase2_split_manifest_v1.json": "89b2cd6e58e7f76107e4c743cd0f057a9cd75c5a4c1759f31e77264367fefe37",
}
PRIMARY_PROCESSES = ("iid_gaussian", "ar1_0844", "arma11", "seasonal_ar1_12", "ar1_060_mcar20")
SLOPES_VALIDATION = (0.0, 0.1, 0.25, 0.5, -0.5)
ROOT_CAUSE_FIXES = {
    "STA-01": (
        "Severe inference undercoverage",
        "The frozen OLS-HAC reference remains exploratory; candidate methods are accepted only for DGP regimes that pass independent calibration.",
        "test_complete_regular_hac_is_retained_only_as_unvalidated_reference",
    ),
    "STA-02": (
        "Calendar-gap lag semantics",
        "Retain a complete monthly PeriodIndex, use calendar-distance HAC, and keep missing periods explicit for state-space fitting.",
        "test_calendar_lag_uses_timestamp_distance_across_missing_months",
    ),
    "STA-03": (
        "MODIS QA sample-selection sensitivity",
        "Keep QA eligibility and the original threshold explicit; do not infer stable pixel support or vegetation change from the aggregate series alone.",
        "test_temporal_support_reports_stable_union_and_early_late_composition",
    ),
    "STA-04": (
        "Seasonal Mann-Kendall dependence",
        "Preserve classical seasonal-MK as a descriptive reference and gate trend-free, null-constrained year-block inference on its declared regime.",
        "test_seasonal_mk_null_bootstrap_removes_injected_linear_trend_before_resampling",
    ),
    "STA-05": (
        "Degenerate-series residual diagnostics",
        "Detect zero-variance series before inference and report residual ACF as undefined rather than floating-point noise.",
        "test_constant_series_has_no_inferential_claim_or_residual_acf",
    ),
    "STA-06": (
        "Multiple-testing assumptions",
        "Retain the historical BH family, compute BY only from calibrated marginal p-values, and withhold adjusted inference when marginals are invalid.",
        "test_benjamini_yekutieli_handles_arbitrary_dependence_and_preserves_nan",
    ),
    "STA-07": (
        "Nonfinite values pass missingness checks",
        "Require finite numeric observations before coverage checks or model fitting.",
        "test_infinite_input_is_flagged_before_climatology_or_regression",
    ),
}


def pytest_checkpoint_note() -> str:
    """Describe the recorded local test run without freezing mutable remote CI state."""
    return (
        "Recorded local workspace full suite: 151 passed on 2026-11-16; "
        "verify GitHub CI against the current remote commit when reporting."
    )


def modis_spatial_selection_receipt_note(receipt: dict[str, Any] | None) -> str:
    """Render a sanitized, hash-linked receipt for the native-grid CEDIA audit."""
    if receipt is None:
        return (
            "No native MODIS pixel grid is asserted by this local finalizer. "
            "Spatial pixel-composition drift is unresolved here; aggregate Gold alone cannot "
            "establish stable pixel support. See `results/statistical_audit/MODIS_SELECTION_AUDIT.md`."
        )
    if receipt.get("status") != "VALIDATED_ON_CEDIA":
        raise ValueError("MODIS spatial-selection receipt is not marked VALIDATED_ON_CEDIA")
    coverage = receipt.get("coverage", {})
    validation = receipt.get("validation", {})
    if (
        coverage.get("start_month") != "2002-01"
        or coverage.get("end_month") != "2024-12"
        or int(coverage.get("month_count", 0)) != 276
        or validation.get("manifest_validation_status") != "PASSED"
        or int(validation.get("source_granules_referenced", 0)) != 276
        or int(validation.get("regional_files_verified", 0)) != 276
        or int(validation.get("monthly_summary_rows", 0)) != 1656
        or receipt.get("spatial_grid", {}).get("native_grid")
        != "MODIS CMG geographic 0.05 degree; no interpolation or resampling"
    ):
        raise ValueError("MODIS spatial-selection receipt does not prove the frozen 276-month scope")
    artifacts = receipt.get("artifacts", {})
    for key in (
        "manifest_sha256",
        "monthly_summary_sha256",
        "lineage_sha256",
        "spatial_support_sha256",
        "acquisition_manifest_sha256",
        "regional_manifest_sha256",
    ):
        digest = str(artifacts.get(key, ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"MODIS spatial-selection receipt has an invalid {key}")
    for key in ("manifest_path", "monthly_summary_path", "lineage_path", "spatial_support_path"):
        relative_path = str(artifacts.get(key, ""))
        if not relative_path.startswith("results/statistical_correction/modis_spatial_selection_v1_1/"):
            raise ValueError(f"MODIS spatial-selection receipt has an unexpected {key}")
    strict = receipt.get("spatial_support", {}).get("strict", {})
    eligibility = receipt.get("qa_eligibility", {})
    return (
        "## Separate CEDIA native-grid spatial-selection audit\n\n"
        "A CEDIA audit of the native MOD13C2.061 CMG 0.05-degree grid covers 276 contiguous months "
        "from 2002-01 through 2024-12. Its validated manifest references 276 regional inputs and "
        "276 source-granule scientific-validation records. This report uses a sanitized receipt; "
        "raw HDF4 files and regional arrays remain in CEDIA, and this local finalizer did not rehash them.\n\n"
        "CEDIA project-relative audit files: manifest `" + str(artifacts["manifest_path"]) + "`; monthly summary `"
        + str(artifacts["monthly_summary_path"]) + "`; lineage `" + str(artifacts["lineage_path"])
        + "`; spatial-support summary `" + str(artifacts["spatial_support_path"]) + "`.\n\n"
        f"Under the frozen strict QA policy, {int(eligibility['strict_months_at_10pct'])}/276 months meet 10% "
        f"valid-area coverage and {int(eligibility['strict_months_at_20pct'])}/276 meet 20%. "
        f"At a 50% temporal-support frequency threshold, the strict tier has a stable-pixel count of "
        f"{int(strict['stable_pixel_count'])}; the union covers {float(strict['union_area_fraction_bbox']):.6f} "
        f"of bbox area. Early/late support Jaccard area is "
        f"{float(strict['early_late_jaccard_area']):.6f}, and "
        f"{100*float(strict['early_late_area_fraction_changed_by_10pp']):.3f}% of bbox area changes "
        "support frequency by at least 10 percentage points. These are spatial-selection diagnostics, "
        "not evidence of a vegetation trend or causal change.\n\n"
        "CEDIA artifact SHA-256: manifest `" + str(artifacts["manifest_sha256"]) + "`; monthly summary `"
        + str(artifacts["monthly_summary_sha256"]) + "`; lineage `" + str(artifacts["lineage_sha256"])
        + "`; spatial-support summary `" + str(artifacts["spatial_support_sha256"]) + "`.\n"
    )


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_text(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def root_cause_matrix_rows(findings: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Join each frozen audit ID to its corresponding correction and regression evidence."""
    rows = []
    unknown_ids = []
    for finding in findings:
        finding_id = str(finding.get("id", ""))
        fix = ROOT_CAUSE_FIXES.get(finding_id)
        if fix is None:
            unknown_ids.append(finding_id or "<missing>")
            continue
        defect, correction, regression = fix
        rows.append({
            "id": finding_id,
            "defect": defect,
            "root_cause": str(finding["root_cause"]),
            "correction": correction,
            "regression_evidence": regression,
        })
    if unknown_ids:
        raise ValueError(f"Root-cause audit contains unmapped finding IDs: {sorted(unknown_ids)}")
    return rows


def stable_phase_output_paths(phase_dir: Path, phase: str, phase_frame: pd.DataFrame) -> list[Path]:
    """List aggregate and completed-scenario files represented by a phase matrix."""
    completed_stems: set[str] = set()
    if not phase_frame.empty:
        scenario_columns = ["process", "n_months", "true_slope_per_decade"]
        missing = set(scenario_columns).difference(phase_frame.columns)
        if missing:
            raise ValueError(f"{phase} phase matrix is missing scenario columns: {sorted(missing)}")
        scenarios = phase_frame[scenario_columns].drop_duplicates()
        for row in scenarios.itertuples(index=False, name=None):
            process, n_months, slope = row
            slope_tag = str(float(slope)).replace("-", "m").replace(".", "p")
            completed_stems.add(f"{phase}_{process}_n{int(n_months)}_s{slope_tag}")

    stable_paths: list[Path] = []
    for path in sorted(phase_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".tmp"):
            continue
        relative_parts = path.relative_to(phase_dir).parts
        if relative_parts and relative_parts[0] in {"checkpoints", "replicates"}:
            if path.stem not in completed_stems:
                continue
        stable_paths.append(path)
    return stable_paths


def _validation_scope_complete(validation_manifest: dict[str, Any] | None) -> bool:
    if not validation_manifest:
        return False
    if validation_manifest.get("status") == "COMPLETED_PHASE":
        return True
    if validation_manifest.get("status") != "COMPLETED_PREDECLARED_SCOPE" or not validation_manifest.get("scope_complete"):
        return False
    scope_relative = validation_manifest.get("scope_record_path")
    scope_digest = validation_manifest.get("scope_record_sha256")
    if not scope_relative or not scope_digest:
        return False
    scope_path = ROOT / Path(str(scope_relative))
    return scope_path.is_file() and sha256(scope_path) == scope_digest


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def _read_phase(name: str) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    directory = OUT / name
    csv_path = directory / "method_validation_matrix.csv"
    manifest_path = directory / "calibration_manifest.json"
    if not csv_path.exists() or not manifest_path.exists():
        return pd.DataFrame(), None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(csv_path)
    if manifest.get("outputs", {}).get(csv_path.name, {}).get("sha256") != sha256(csv_path):
        raise RuntimeError(f"{name} calibration matrix hash does not match its manifest")
    return frame, manifest


def _build_validation_registry(
    selection: dict[str, Any] | None,
    validation: pd.DataFrame,
    validation_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    decisions = (selection or {}).get("decisions", {})
    for process in PRIMARY_PROCESSES:
        selected = decisions.get(process, {}).get("selected")
        if not selected:
            continue
        if validation.empty:
            continue
        chosen_method = str(selected["method"])
        engine_method = next(spec["method"] for spec in METHODS if spec["name"] == chosen_method)
        subset = validation.loc[
            (validation.process == process)
            & (validation.n_months == 276)
            & (validation.method == chosen_method)
            & validation.true_slope_per_decade.isin(SLOPES_VALIDATION)
        ]
        by_slope = {float(row.true_slope_per_decade): row for row in subset.itertuples(index=False)}
        if set(by_slope) != set(SLOPES_VALIDATION):
            continue
        if not all(row.validity_status == "VALIDATED_FOR_DECLARED_REGIME" for row in by_slope.values()):
            continue
        rows = list(by_slope.values())
        regimes[process] = {
            "validity_status": "VALIDATED_FOR_DECLARED_REGIME",
            "method": chosen_method,
            "method_id": METHOD_IDS[engine_method],
            "calendar_months_min": 276,
            "calendar_months_max": 276,
            "calendar_coverage_min": min(float(row.min_observation_fraction) for row in rows),
            "calendar_coverage_max": max(float(row.max_observation_fraction) for row in rows),
            "validation_slopes_per_decade": list(SLOPES_VALIDATION),
            "selection_calibration_score": float(selected["calibration_score"]),
            "selection_seed_base": 20261126,
            "independent_validation_seed_base": 20261127,
        }
    return {
        "schema_version": "kuntur-statistical-validation-registry-v1",
        "status": (
            "FROZEN_FOR_DECLARED_SCOPE"
            if validation_manifest
            and validation_manifest.get("status") == "COMPLETED_PREDECLARED_SCOPE"
            and _validation_scope_complete(validation_manifest)
            else "FROZEN"
            if validation_manifest
            and validation_manifest.get("status") == "COMPLETED_PHASE"
            and _validation_scope_complete(validation_manifest)
            else "PARTIAL_OR_EMPTY"
        ),
        "protocol_sha256": sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
        "selection_manifest_sha256": sha256(OUT / "selection/selection_manifest.json") if (OUT / "selection/selection_manifest.json").exists() else None,
        "validation_matrix_sha256": sha256(OUT / "validation/method_validation_matrix.csv") if (OUT / "validation/method_validation_matrix.csv").exists() else None,
        "validated_regime_count": len(regimes),
        "regimes": regimes,
        "limitation": "Only exact declared DGP, method, 276-month length and simulated temporal-coverage envelope matches are eligible. This registry does not identify the true error process of a NASA series.",
    }


def _load_original_trends() -> pd.DataFrame:
    path = ROOT / "results/phase2/public/analytical_trends_v1.csv"
    return pd.read_csv(path)


def _build_trend_before_after(gold_path: Path, gold_sha: str) -> pd.DataFrame:
    gold = pd.read_parquet(gold_path)
    original = _load_original_trends().set_index("variable_id")
    rows: list[dict[str, Any]] = []
    region_values = gold.region_id.dropna().astype(str).unique() if "region_id" in gold else []
    region_id = str(region_values[0]) if len(region_values) == 1 else None
    for variable_index, variable in enumerate(original.index.astype(str)):
        historical = original.loc[variable]
        threshold = float(historical.coverage_threshold)
        try:
            ci = json.loads(historical.ols_hac_ci95_per_decade)
        except (TypeError, json.JSONDecodeError):
            ci = None
        for method_index, spec in enumerate(METHODS):
            result = analyze_variable_v12(
                gold, variable, region_id=region_id, method=spec["method"],
                qa_policy={"min_valid_area_fraction_bbox": threshold},
                hac_lag_months=int(spec["lag"]), n_bootstrap=399,
                random_seed=20261128 + variable_index * 101 + method_index,
                block_length_months=int(spec["block"]),
            )
            rows.append({
                "region_id": result.get("region_id"),
                "variable_id": variable,
                "units": historical.units,
                "historical_method": "v1 OLS-HAC (exploratory; frozen result)",
                "historical_slope_per_decade": float(historical.ols_hac_slope_per_decade),
                "historical_nominal_ci95_low": float(ci[0]) if ci else None,
                "historical_nominal_ci95_high": float(ci[1]) if ci else None,
                "historical_nominal_p_value": float(historical.ols_hac_p_value),
                "historical_bh_q_value": float(historical.ols_hac_p_value_fdr_bh),
                "candidate_method": spec["name"],
                "candidate_method_id": result["method_id"],
                "candidate_slope_per_decade": result.get("slope_per_decade"),
                "candidate_standard_error": result.get("standard_error"),
                "candidate_standard_error_inferentially_valid": result.get("standard_error_is_inferentially_valid"),
                "candidate_ci95_low": (result.get("confidence_interval_95") or [None, None])[0],
                "candidate_ci95_high": (result.get("confidence_interval_95") or [None, None])[1],
                "candidate_p_value": result.get("p_value"),
                "candidate_q_value": None,
                "candidate_validity_status": result.get("validity_status"),
                "candidate_convergence_status": result.get("convergence_status"),
                "n_eligible_observations": result.get("n_eligible_observations"),
                "n_calendar_months": result.get("n_calendar_months"),
                "calendar_start": result.get("calendar_start"),
                "calendar_end": result.get("calendar_end"),
                "calendar_coverage_fraction": result.get("calendar_coverage_fraction"),
                "max_missing_run_months": result.get("max_missing_run_months"),
                "qa_threshold": threshold,
                "input_gold_sha256": gold_sha,
                "input_series_sha256": result.get("provenance", {}).get("input_series_sha256"),
                "random_seed": 20261128 + variable_index * 101 + method_index,
                "inference_note": "Candidate point estimates are descriptive unless an exact frozen independent-validation registry entry is explicitly matched; this report does not declare an error DGP for NASA series.",
            })
    return pd.DataFrame(rows)


def _save_figures(validation: pd.DataFrame, trends: pd.DataFrame) -> list[Path]:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    produced: list[Path] = []

    def save(fig: plt.Figure, name: str) -> None:
        path = figure_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        produced.append(path)

    primary = validation.loc[validation.process.isin(PRIMARY_PROCESSES)] if not validation.empty else validation
    null = primary.loc[primary.true_slope_per_decade == 0.0] if not primary.empty else primary
    if not primary.empty:
        available = primary.loc[(primary.n_months == 276) & (primary.true_slope_per_decade == 0.0) & primary.coverage_95.notna()]
        if not available.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            for process, group in available.groupby("process"):
                by_method = group.groupby("method", as_index=False).coverage_95.mean()
                ax.plot(by_method.method, by_method.coverage_95, marker="o", label=process)
            ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("Empirical 95% CI coverage")
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=65)
            ax.legend(fontsize=7, ncol=2)
            save(fig, "ci_coverage_by_method.png")
        type1 = null.loc[(null.n_months == 276) & null.type1_fpr_005.notna()]
        if not type1.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            for process, group in type1.groupby("process"):
                ax.plot(group.method, group.type1_fpr_005, marker="o", label=process)
            ax.axhline(0.05, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("Empirical Type-I error at 0.05")
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=65)
            ax.legend(fontsize=7, ncol=2)
            save(fig, "false_positive_rate_by_method.png")
        alternatives = primary.loc[(primary.n_months == 276) & (primary.true_slope_per_decade != 0.0) & primary.power_005.notna()]
        if not alternatives.empty:
            processes = [name for name in PRIMARY_PROCESSES if name in set(alternatives.process)]
            fig, axes = plt.subplots(2, 3, figsize=(16, 9))
            for ax, process in zip(axes.flat, processes):
                group = alternatives.loc[alternatives.process == process]
                for method, method_group in group.groupby("method"):
                    avg = method_group.groupby("true_slope_per_decade", as_index=False).power_005.mean()
                    ax.plot(avg.true_slope_per_decade, avg.power_005, marker="o", linewidth=0.8, label=method)
                ax.set_title(process)
                ax.set_ylim(0, 1)
                ax.set_xlabel("Slope per decade")
                ax.set_ylabel("Power")
            if axes.flat[0].get_legend_handles_labels()[0]:
                axes.flat[0].legend(fontsize=5, ncol=2)
            for ax in axes.flat[len(processes):]:
                ax.set_visible(False)
            save(fig, "power_curves.png")
        fig, ax = plt.subplots(figsize=(12, 5))
        if "slope_rmse_per_decade" in primary:
            pivot = primary.pivot_table(index="method", columns="process", values="slope_rmse_per_decade", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Slope RMSE (units per decade)")
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=60)
            ax.legend(fontsize=7, ncol=2)
            save(fig, "slope_error_by_method.png")
        widths = primary.loc[(primary.n_months == 276) & primary.mean_ci_width_over_oracle.notna()]
        if not widths.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            pivot = widths.pivot_table(index="method", columns="process", values="mean_ci_width_over_oracle", aggfunc="mean")
            pivot.plot(kind="bar", ax=ax)
            ax.axhline(2.0, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("Mean interval width / GLS oracle width")
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=60)
            ax.legend(fontsize=7, ncol=2)
            save(fig, "interval_width_vs_oracle.png")
        season = primary.loc[(primary.n_months == 276) & primary.process.isin(["iid_gaussian", "ar1_0844", "seasonal_ar1_12"]) & (primary.true_slope_per_decade == 0.0) & primary.coverage_95.notna()]
        if not season.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            for process, group in season.groupby("process"):
                ax.plot(group.method, group.coverage_95, marker="o", label=process)
            ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
            ax.tick_params(axis="x", rotation=65)
            ax.set_ylabel("Null CI coverage")
            ax.legend(fontsize=8)
            save(fig, "seasonal_dependence_sensitivity.png")
        miss = validation.loc[validation.process.isin(["ar1_0844", "ar1_060_mcar20", "ar1_060_seasonal_missing"]) & (validation.n_months == 276) & (validation.true_slope_per_decade == 0.0) & validation.coverage_95.notna()] if not validation.empty else validation
        if not miss.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            for process, group in miss.groupby("process"):
                ax.plot(group.method, group.coverage_95, marker="o", label=process)
            ax.axhline(0.95, color="black", linestyle="--", linewidth=1)
            ax.tick_params(axis="x", rotation=65)
            ax.set_ylabel("Null CI coverage")
            ax.legend(fontsize=8)
            save(fig, "missingness_sensitivity.png")
        runtimes = primary.loc[primary.runtime_seconds_per_requested_series.notna()]
        if not runtimes.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            runtimes.groupby("method").runtime_seconds_per_requested_series.mean().sort_values().plot(kind="bar", ax=ax)
            ax.set_ylabel("Mean runtime per simulated series (s)")
            ax.set_xlabel("Method")
            ax.tick_params(axis="x", rotation=60)
            save(fig, "computational_cost.png")
    if not trends.empty:
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        for ax, (variable, group) in zip(axes.flat, trends.groupby("variable_id", sort=True)):
            x = np.arange(len(group))
            ax.plot(x, group.candidate_slope_per_decade.astype(float), marker="o", linewidth=1, label="v1.2 candidate slope")
            ax.axhline(float(group.historical_slope_per_decade.iloc[0]), color="tab:red", linestyle="--", label="Frozen v1 OLS-HAC")
            ax.set_title(variable)
            ax.set_xticks(x)
            ax.set_xticklabels(group.candidate_method, rotation=75, fontsize=6)
            ax.set_ylabel(f"Slope per decade ({group.units.iloc[0]})")
        axes.flat[0].legend(fontsize=7)
        save(fig, "environmental_estimates_before_after.png")
    return produced


def _write_reports(
    selection_manifest: dict[str, Any] | None, validation_manifest: dict[str, Any] | None,
    validation: pd.DataFrame, trends: pd.DataFrame, gold_sha: str,
) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    complete_validation = _validation_scope_complete(validation_manifest)
    validated_rows = int((validation.validity_status == "VALIDATED_FOR_DECLARED_REGIME").sum()) if not validation.empty else 0
    selection_status = selection_manifest.get("status") if selection_manifest else "NOT_STARTED"
    validation_status = validation_manifest.get("status") if validation_manifest else "NOT_STARTED"
    atomic_text(REPORTS / "CORRECTION_IMPLEMENTATION.md", f"""# KUNTUR statistical correction v1.2 — implementation

## State

- Frozen Gold SHA-256: `{gold_sha}`. The Gold input and frozen forecast splits were read-only during this work.
- Selection phase: `{selection_status}`.
- Independent validation phase: `{validation_status}`.
- Independent validation rows that passed every applicable gate: {validated_rows}.
- Eligible real-series candidate records: {len(trends)}. Candidate slopes are descriptive unless an exact frozen regime registry match is supplied.
- Statistical source: `src/kuntur_data/phase2/statistical_v12.py`; calibration runner: `scripts/statistical_correction/run_calibration.py`.

The v1 analytical implementation and original trend CSV remain the historical reference. The v1.2 engine preserves monthly calendar gaps, rejects nonfinite values, provides dependence-aware candidates, records model warnings and numerical status, and suppresses p-values and intervals unless a frozen registry entry matches the explicit method, regime, calendar length and temporal coverage.

Methods compared: OLS-HAC row bandwidths 6/12/18/24; calendar-distance HAC; iterated GLSAR(1); state-space SARIMAX AR(1), ARMA(1,1), seasonal AR(1,12); moving residual blocks of 6/12/24 months; calendar-year blocks of 2/3 years; and trend-free, null-constrained seasonal-MK year-block inference.

## Verification

{pytest_checkpoint_note()} Focused statistical tests cover constant and nonfinite inputs, irregular calendar support, calendar-lag diagnostics, registry gating, GLSAR iteration cap, trend-free seasonal-MK null resampling, and the frozen validation scope.

See [METHOD_VALIDATION.md](METHOD_VALIDATION.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for completed scenario coverage and artifact hashes.
""")

    root_cause = json.loads((ROOT / "results/statistical_audit/root_cause_findings.json").read_text(encoding="utf-8"))
    matrix_rows = root_cause_matrix_rows(root_cause.get("findings", []))
    matrix_markdown = "\n".join(
        f"| {row['id']} | {row['defect']} | {row['root_cause']} | {row['correction']} | `{row['regression_evidence']}` |"
        for row in matrix_rows
    )
    atomic_text(REPORTS / "ROOT_CAUSE_TO_FIX_MATRIX.md", """# Root cause to correction matrix

| Finding | Defect | Confirmed root cause | v1.2 correction | Regression evidence |
|---|---|---|---|---|
""" + matrix_markdown + "\n\nOriginal audit evidence remains in `reports/statistical_audit/`; this table does not replace it.\n")

    if validation.empty:
        method_body = "No independent validation scenarios have completed. No method is validated."
    else:
        metrics = [column for column in ("type1_fpr_005", "coverage_95", "slope_bias_per_decade", "power_005", "mean_ci_width_over_oracle", "convergence_failure_rate", "residual_dependence_failure_rate") if column in validation]
        summary = validation.groupby(["method", "validity_status"], dropna=False).agg(
            scenarios=("process", "size"),
            **{metric: (metric, "mean") for metric in metrics},
        ).reset_index()
        if len(summary) > 80:
            summary = summary.head(80)
        summary = markdown_table(summary)
        method_body = summary
    if validation_manifest and validation_manifest.get("status") == "COMPLETED_PHASE":
        scope_summary = (
            f"The complete frozen protocol completed {validation_manifest.get('scenario_count', 0)} of "
            f"{validation_manifest.get('planned_scenario_count', 0)} scenarios; "
            f"{validation_manifest.get('completed_checkpoints', 0)} scenario checkpoints are complete."
        )
    elif validation_manifest and validation_manifest.get("status") == "COMPLETED_PREDECLARED_SCOPE":
        scope_summary = (
            f"The frozen validation scope completed {validation_manifest.get('scope_completed_scenario_count', 0)} of "
            f"{validation_manifest.get('scope_planned_scenario_count', 0)} scenarios. The original protocol matrix contains "
            f"{validation_manifest.get('full_protocol_scenario_count', 0)} scenarios; "
            f"{validation_manifest.get('full_protocol_deferred_scenario_count', 0)} remain deferred and unvalidated. "
            f"Scope record: `{validation_manifest.get('scope_record_path')}` "
            f"(SHA-256 `{validation_manifest.get('scope_record_sha256')}`)."
        )
    elif validation_manifest:
        scope_summary = (
            f"Validation status: `{validation_manifest.get('status')}`; "
            f"{validation_manifest.get('completed_checkpoints', validation_manifest.get('scope_completed_scenario_count', 0))} of "
            f"{validation_manifest.get('planned_scenario_count', validation_manifest.get('scope_planned_scenario_count', 0))} planned scenarios have complete checkpoints."
        )
    else:
        scope_summary = "No independent validation has run."
    atomic_text(REPORTS / "METHOD_VALIDATION.md", f"""# Method validation

Selection status: `{selection_status}`. Independent validation status: `{validation_status}`.

{scope_summary}

Protocol: [`docs/protocols/statistical_correction_v1_2.md`](../../docs/protocols/statistical_correction_v1_2.md). Selection seed base 20261126 is distinct from validation seed base 20261127. Selection outputs are not final validation. Selection, validation, smoke, partial runs and stress-only rows remain separate.

## Independent validation matrix

{method_body}

Rows labeled `VALIDATED_FOR_DECLARED_REGIME` are restricted to their named DGP and sample support. `NOT_VALIDATED`, `NUMERICAL_FAILURE`, `UNSUPPORTED_REGIME`, and stress-only rows do not release inferential output. See paired differences in `results/statistical_correction/validation/paired_method_differences.csv`.
""")

    if trends.empty:
        trend_body = "No real-series candidate recomputation was completed."
    else:
        displayed = trends.groupby("variable_id", as_index=False).agg(
            units=("units", "first"),
            frozen_v1_ols_hac_slope_per_decade=("historical_slope_per_decade", "first"),
            v1_2_candidate_slope_median=("candidate_slope_per_decade", "median"),
            v1_2_candidate_slope_min=("candidate_slope_per_decade", "min"),
            v1_2_candidate_slope_max=("candidate_slope_per_decade", "max"),
            candidates_with_released_p=("candidate_p_value", lambda values: int(values.notna().sum())),
        )
        trend_body = markdown_table(displayed)
    atomic_text(REPORTS / "STATISTICAL_BEFORE_AFTER.md", f"""# Statistical results before and after v1.2

The frozen v1 results are preserved unchanged and remain nominal/exploratory where their calibration assumptions are unsupported. The v1.2 candidates below report point estimates; p-values, corrected intervals and q-values stay null unless a matching frozen validation registry entry is explicitly declared. The finalizer does not infer the environmental error process from a favorable p-value.

{trend_body}

The full table contains one row per frozen environmental series and locked v1.2 candidate method. For every row the original v1 OLS-HAC slope, nominal interval, p-value and BH q-value are repeated for comparison. Candidate confidence limits and p-values are deliberately withheld when the regime is not established.
""")

    modis = json.loads((ROOT / "results/statistical_audit/modis_selection_summary.json").read_text(encoding="utf-8"))
    modis_counts = modis.get("eligible_by_calendar_month", {})
    count_rows = "\n".join(f"| {month} | {count} |" for month, count in sorted(modis_counts.items(), key=lambda item: int(item[0])))
    spatial_receipt_path = OUT / "modis_spatial_selection_cedia_summary_v1_1.json"
    spatial_receipt = json.loads(spatial_receipt_path.read_text(encoding="utf-8")) if spatial_receipt_path.is_file() else None
    spatial_note = modis_spatial_selection_receipt_note(spatial_receipt)
    atomic_text(REPORTS / "MODIS_CALENDAR_CORRECTION.md", f"""# MODIS calendar and selection correction

The frozen MODIS QA policy remains at valid-area fraction 0.10. The pre-existing audit found {modis.get('months_eligible_10pct', 'not reported')}/276 eligible months at that threshold, {modis.get('months_eligible_20pct', 'not reported')}/276 at 0.20, median valid-area fraction {modis.get('coverage_median_all_months', 'not reported')}, and a maximum calendar gap of 9 months. Missing months remain explicit on the monthly PeriodIndex; calendar ACF pairs use exact month offsets. Row-compressed HAC, GLSAR and block methods return `IRREGULAR_TIME_SUPPORT`; state-space models retain NaN months; calendar-distance HAC uses actual month ordinals.

| Calendar month | Eligible years |
|---:|---:|
{count_rows}

{spatial_note}
""")

    atomic_text(REPORTS / "REPRODUCIBILITY.md", f"""# Reproducibility and artifact provenance

## Commands

```powershell
$env:PYTHONPATH='src;scripts'
python -m pytest -q
python scripts/statistical_correction/run_calibration.py --phase selection
python scripts/statistical_correction/run_calibration.py --phase validation
# Or, instead of the full protocol above, freeze and run only the predeclared selection-eligible scope:
python scripts/statistical_correction/run_scoped_validation.py --freeze-scope
python scripts/statistical_correction/run_scoped_validation.py --run
python scripts/statistical_correction/finalize_correction.py
```

The calibration runner checkpoints every 25 outer replications and each completed scenario atomically. It records independent seeds, protocol/script/module hashes, software versions, wall/CPU time, peak RSS, per-replication outputs, paired method comparisons and phase-level artifact SHA-256 values. Smoke outputs are isolated from selection and validation.

## Frozen source hashes

- Gold: `{gold_sha}`
- Gold manifest: `{sha256(ROOT / 'data/manifests/gold_manifest_v1.json')}`
- v1 analytical trends: `{sha256(ROOT / 'results/phase2/public/analytical_trends_v1.csv')}`
- v1 data dictionary: `{sha256(ROOT / 'docs/data_dictionary_v1.json')}`
- v1 monthly contract: `{sha256(ROOT / 'docs/monthly_data_contract.md')}`
- v1.2 protocol: `{sha256(ROOT / 'docs/protocols/statistical_correction_v1_2.md')}`

Additional frozen references:

{chr(10).join(f"- `{path}`: `{digest}`" for path, digest in EXPECTED_FROZEN_ARTIFACT_HASHES.items())}

Frozen forecast split artifacts (rechecked before finalization):

{chr(10).join(f"- `{path}`: `{digest}`" for path, digest in EXPECTED_FROZEN_SPLIT_HASHES.items())}

Native NASA source granules, frozen Gold, historical audits and forecast splits were not modified. No downloads or SLURM jobs were started. This is a local Windows benchmark; it is not an HPC performance claim.
""")


def main() -> None:
    gold_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
    if sha256(gold_path) != EXPECTED_GOLD_SHA256:
        raise RuntimeError("Frozen Gold SHA-256 differs from the verified v1 input; refusing finalization.")
    frozen_artifact_hashes = {relative: sha256(ROOT / relative) for relative in EXPECTED_FROZEN_ARTIFACT_HASHES}
    mismatched_frozen = [relative for relative, expected in EXPECTED_FROZEN_ARTIFACT_HASHES.items() if frozen_artifact_hashes[relative] != expected]
    if mismatched_frozen:
        raise RuntimeError(f"Frozen historical artifacts changed; refusing finalization: {mismatched_frozen}")
    frozen_split_hashes = {relative: sha256(ROOT / relative) for relative in EXPECTED_FROZEN_SPLIT_HASHES}
    mismatched_splits = [relative for relative, expected in EXPECTED_FROZEN_SPLIT_HASHES.items() if frozen_split_hashes[relative] != expected]
    if mismatched_splits:
        raise RuntimeError(f"Frozen forecast split hashes changed; refusing finalization: {mismatched_splits}")
    selection, selection_manifest = _read_phase("selection")
    validation, validation_manifest = _read_phase("validation")
    selection_decisions = None
    selection_path = OUT / "selection/selection_decisions.json"
    if selection_path.exists():
        selection_decisions = json.loads(selection_path.read_text(encoding="utf-8"))
    phase_frames = [frame for frame in (selection, validation) if not frame.empty]
    if not phase_frames:
        raise RuntimeError("No completed selection or validation scenarios are available.")
    method_frame = pd.concat(phase_frames, ignore_index=True).sort_values(
        ["phase", "process", "n_months", "true_slope_per_decade", "method"]
    ).reset_index(drop=True)
    type1 = method_frame.loc[(method_frame.true_slope_per_decade == 0.0) & method_frame.type1_fpr_005.notna()].copy()
    coverage = method_frame.loc[method_frame.coverage_95.notna()].copy()
    atomic_csv(OUT / "method_validation_matrix.csv", method_frame)
    atomic_parquet(OUT / "monte_carlo_results.parquet", method_frame)
    atomic_csv(OUT / "type1_error_comparison.csv", type1)
    atomic_csv(OUT / "coverage_comparison.csv", coverage)
    gold_sha = sha256(gold_path)
    trends = _build_trend_before_after(gold_path, gold_sha)
    atomic_csv(OUT / "trend_before_after.csv", trends)
    registry = _build_validation_registry(selection_decisions, validation, validation_manifest)
    atomic_json(OUT / "validated_regime_registry.json", registry)
    atomic_json(OUT / "validation_registry_manifest.json", {
        "registry_sha256": sha256(OUT / "validated_regime_registry.json"),
        "protocol_sha256": registry["protocol_sha256"],
        "validated_regime_count": registry["validated_regime_count"],
        "status": registry["status"],
    })
    figures = _save_figures(validation, trends)
    _write_reports(selection_manifest, validation_manifest, validation, trends, gold_sha)
    correction_log = REPORTS / "CORRECTION_LOG.md"
    selection_hash = sha256(OUT / "selection/selection_manifest.json") if (OUT / "selection/selection_manifest.json").exists() else "none"
    validation_hash = sha256(OUT / "validation/calibration_manifest.json") if (OUT / "validation/calibration_manifest.json").exists() else "none"
    finalization_id = hashlib.sha256(f"{sha256(ROOT / 'docs/protocols/statistical_correction_v1_2.md')}|{selection_hash}|{validation_hash}".encode()).hexdigest()[:16]
    if not correction_log.exists():
        atomic_text(correction_log, "# Statistical correction log (append-only)\n\nHistorical v1 outputs and frozen Gold are preserved. Each entry records a v1.2 finalization state; prior entries are retained.\n")
    prior_log = correction_log.read_text(encoding="utf-8")
    if finalization_id not in prior_log:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        with correction_log.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"\n- {timestamp} — finalization `{finalization_id}`: selection `{selection_manifest.get('status') if selection_manifest else 'NOT_STARTED'}`, independent validation `{validation_manifest.get('status') if validation_manifest else 'NOT_STARTED'}`, validated method-regimes `{registry['validated_regime_count']}`, real-series inferential rows released `{int(trends.candidate_p_value.notna().sum())}`. Manifest: `results/statistical_correction/correction_manifest.json`.\n")

    output_paths = [
        OUT / "method_validation_matrix.csv", OUT / "monte_carlo_results.parquet",
        OUT / "type1_error_comparison.csv", OUT / "coverage_comparison.csv",
        OUT / "trend_before_after.csv", OUT / "validated_regime_registry.json",
        OUT / "validation_registry_manifest.json",
        REPORTS / "CORRECTION_IMPLEMENTATION.md", REPORTS / "ROOT_CAUSE_TO_FIX_MATRIX.md",
        REPORTS / "METHOD_VALIDATION.md", REPORTS / "STATISTICAL_BEFORE_AFTER.md",
        REPORTS / "MODIS_CALENDAR_CORRECTION.md", REPORTS / "REPRODUCIBILITY.md",
        correction_log,
        *figures,
    ]
    if (OUT / "modis_spatial_selection_cedia_summary_v1_1.json").is_file():
        output_paths.append(OUT / "modis_spatial_selection_cedia_summary_v1_1.json")
    for phase_name, phase_frame in (("selection", selection), ("validation", validation)):
        phase_dir = OUT / phase_name
        if phase_dir.exists():
            output_paths.extend(stable_phase_output_paths(phase_dir, phase_name, phase_frame))
    manifest = {
        "correction_version": "kuntur-statistical-inference-v1.2.0",
        "status": (
            "COMPLETED_WITH_INDEPENDENT_VALIDATION"
            if validation_manifest and validation_manifest.get("status") == "COMPLETED_PHASE"
            else "SCOPED_VALIDATION_COMPLETE_WITH_DEFERRED_REGIMES"
            if validation_manifest
            and validation_manifest.get("status") == "COMPLETED_PREDECLARED_SCOPE"
            and _validation_scope_complete(validation_manifest)
            else "PARTIAL_VALIDATION"
        ),
        "gold_sha256": gold_sha,
        "additional_frozen_artifact_hashes": frozen_artifact_hashes,
        "frozen_forecast_split_hashes": frozen_split_hashes,
        "original_trends_sha256": sha256(ROOT / "results/phase2/public/analytical_trends_v1.csv"),
        "protocol_sha256": sha256(ROOT / "docs/protocols/statistical_correction_v1_2.md"),
        "code_hashes": {
            "statistical_v12.py": sha256(ROOT / "src/kuntur_data/phase2/statistical_v12.py"),
            "run_calibration.py": sha256(ROOT / "scripts/statistical_correction/run_calibration.py"),
            "finalize_correction.py": sha256(Path(__file__).resolve()),
        },
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "selection_status": selection_manifest.get("status") if selection_manifest else "NOT_STARTED",
        "validation_status": validation_manifest.get("status") if validation_manifest else "NOT_STARTED",
        "validation_scope": {
            "scope_record_path": validation_manifest.get("scope_record_path") if validation_manifest else None,
            "scope_record_sha256": validation_manifest.get("scope_record_sha256") if validation_manifest else None,
            "scope_complete": _validation_scope_complete(validation_manifest),
            "scoped_scenarios_completed": validation_manifest.get("scope_completed_scenario_count", 0) if validation_manifest else 0,
            "scoped_scenarios_planned": validation_manifest.get("scope_planned_scenario_count", 0) if validation_manifest else 0,
            "full_protocol_scenarios_deferred": validation_manifest.get("full_protocol_deferred_scenario_count", 0) if validation_manifest else 0,
        },
        "selection_scenarios": int(selection[["process", "n_months", "true_slope_per_decade"]].drop_duplicates().shape[0]) if not selection.empty else 0,
        "validation_scenarios": int(validation[["process", "n_months", "true_slope_per_decade"]].drop_duplicates().shape[0]) if not validation.empty else 0,
        "validation_rows": int(len(validation)),
        "full_pytest_checkpoint": pytest_checkpoint_note() if len(method_frame) > 0 else "NOT_RUN",
        "validated_method_regimes": int(registry["validated_regime_count"]),
        "real_series_candidate_rows": int(len(trends)),
        "inferential_real_series_rows_released": int(trends.candidate_p_value.notna().sum()),
        "figures": [path.relative_to(ROOT).as_posix() for path in figures],
        "outputs": {path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in output_paths},
        "frozen_inputs_unchanged": True,
        "limitations": [
            "The registry does not identify the true error process of an environmental series; explicit regime declaration is required.",
            "The validated CEDIA MODIS spatial-selection audit is represented locally by a sanitized aggregate receipt; raw HDF4 and regional arrays remain remote.",
            "Paired method-difference confidence intervals are descriptive and not multiplicity-adjusted.",
        ],
    }
    atomic_json(OUT / "correction_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
