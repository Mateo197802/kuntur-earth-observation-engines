#!/usr/bin/env python3
"""Independently verify KUNTUR predictive v1.5 artifacts and development metrics."""

from __future__ import annotations

import hashlib
import argparse
import json
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from kuntur_data.phase2.predictive_v15 import ALLOWED_GATE_FEATURES


DEV_END = "2021-12"
STRATEGIES = (
    "climatology",
    "compact_residual",
    "regularized_anomaly",
    "smooth_shrinkage",
    "regime_mixture",
    "nested_selected",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_metrics(y: np.ndarray, prediction: np.ndarray, baseline: np.ndarray) -> dict[str, float | int]:
    error = prediction - y
    baseline_error = baseline - y
    baseline_mae = float(np.mean(np.abs(baseline_error)))
    baseline_rmse = float(np.sqrt(np.mean(np.square(baseline_error))))
    return {
        "sample_count": int(len(y)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "bias": float(np.mean(error)),
        "mae_skill_vs_baseline": float((baseline_mae - float(np.mean(np.abs(error)))) / baseline_mae) if baseline_mae else 0.0,
        "rmse_skill_vs_baseline": float((baseline_rmse - float(np.sqrt(np.mean(np.square(error))))) / baseline_rmse) if baseline_rmse else 0.0,
        "monthly_win_rate_vs_baseline": float(np.mean(np.abs(error) < np.abs(baseline_error))),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def resolve_input_path(label: str, repo_root: Path, data_root: Path) -> Path:
    if label.startswith("external_data_root/"):
        return data_root / label.removeprefix("external_data_root/")
    return repo_root / label


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Read-only existing KUNTUR data root containing gold/ and manifests/")
    parser.add_argument("--private-output-root", type=Path, help="Optional local run directory containing private row-level outputs for full metric recomputation")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    private_output_root = args.private_output_root.resolve() if args.private_output_root else None
    result_dir = repo_root / "results" / "predictive_v15"
    registry_path = result_dir / "experiment_registry_v1_5.json"
    hash_path = result_dir / "artifact_hashes_v1_5.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    hash_index = json.loads(hash_path.read_text(encoding="utf-8"))

    hash_failures: list[str] = []
    seen_paths: set[str] = set()
    for record in hash_index["files"]:
        rel = str(record["path"])
        if rel in seen_paths:
            hash_failures.append(f"duplicate path: {rel}")
            continue
        seen_paths.add(rel)
        path = repo_root / Path(rel)
        if not path.is_file():
            hash_failures.append(f"missing: {rel}")
            continue
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != record["sha256"] or actual_size != int(record["size_bytes"]):
            hash_failures.append(f"hash or size mismatch: {rel}")
    require(not hash_failures, f"output hash index failed: {hash_failures}")

    input_hash_failures: list[str] = []
    for key, record in registry["input_files"].items():
        label = str(record["path"])
        if Path(label).is_absolute() or PureWindowsPath(label).is_absolute() or "\\Users\\" in label:
            input_hash_failures.append(f"machine-specific absolute path leaked: {key}")
            continue
        path = resolve_input_path(label, repo_root, data_root)
        if not path.is_file() and key == "v14_fold_predictions" and private_output_root is None:
            # The source OOF rows are intentionally not part of the public release.
            continue
        if not path.is_file() or sha256_file(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            input_hash_failures.append(key)
    require(not input_hash_failures, f"input/code hash check failed: {input_hash_failures}")

    public_row_level_names = {
        "eligible_development_samples.csv", "issue_time_availability_audit.csv",
        "outer_fold_predictions.parquet", "inner_selection_oof.parquet",
    }
    require(
        not any((result_dir / name).exists() for name in public_row_level_names),
        "row-level data or predictions are present in the public results directory",
    )
    private_records = registry.get("private_output_artifacts", [])
    require(
        {str(record["path"]) for record in private_records} == public_row_level_names,
        "private output hash index does not cover the expected row-level artifacts",
    )
    if private_output_root is not None:
        try:
            private_output_root.relative_to(repo_root.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError("private row-level output root must remain outside the repository")
        require(private_output_root.name == str(registry["run_id"]), "private output directory does not match the registry run_id")

    leakage = json.loads((result_dir / "leakage_audit.json").read_text(encoding="utf-8"))
    guard = leakage.get("target_label_availability_guard", {})
    require(guard.get("status") == "PASS", "training target labels lack the issue-time CMR production guard")
    cmr_snapshot = json.loads((result_dir / "source_availability_cmr_snapshot.json").read_text(encoding="utf-8"))
    merra_records = cmr_snapshot["source_collections"]["merra2_monthly_slv"]["records"]
    merra_production = {
        str(record["month"]): record.get("production_date_time_utc")
        for record in merra_records
        if record.get("month")
    }
    for summary in guard.get("fold_training_summaries", []):
        latest_month = str(summary["latest_training_label_month"])
        production_time = merra_production.get(latest_month)
        require(production_time is not None, f"CMR production time missing for guarded label month {latest_month}")
        require(
            str(production_time) == str(summary["latest_label_cmr_production_time_utc"]),
            f"recorded latest label production time differs from the CMR snapshot for {latest_month}",
        )
        require(
            pd.to_datetime(production_time, utc=True) <= pd.to_datetime(summary["first_validation_issue_time_utc"], utc=True),
            f"training target label {latest_month} postdates its first validation issue",
        )

    if private_output_root is None:
        metrics = pd.read_csv(result_dir / "overall_metrics.csv")
        require(set(metrics.strategy.astype(str)) == set(STRATEGIES), "public aggregate metrics do not contain the frozen strategy set")
        require(metrics.sample_count.astype(int).nunique() == 1, "public strategies are not scored on the same target count")
        require(np.isfinite(metrics[["mae", "rmse", "bias", "mae_skill_vs_baseline", "rmse_skill_vs_baseline"]].to_numpy(dtype=float)).all(), "public aggregate metrics contain non-finite scores")
        require(registry["evaluation"]["outer_fold_count"] == 3, "public evaluation does not record all three frozen outer folds")
        print(json.dumps({
            "status": "PASS",
            "verification_scope": "PUBLIC_AGGREGATES_ONLY",
            "public_output_files_hash_verified": len(hash_index["files"]),
            "input_and_implementation_hash_verified": len(registry["input_files"]),
            "private_row_level_files_listed_by_hash": len(private_records),
            "row_level_files_in_public_results": 0,
            "outer_oof_rows": int(metrics.sample_count.iloc[0]),
            "target_range": registry["evaluation"]["target_range"],
            "training_label_issue_time_guard": "PASS",
            "note": "Supply --private-output-root for local row-level and independent metric recomputation.",
        }, indent=2, sort_keys=True))
        return 0

    private_hash_failures: list[str] = []
    for record in private_records:
        path = private_output_root / str(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"] or path.stat().st_size != int(record["size_bytes"]):
            private_hash_failures.append(str(record["path"]))
    require(not private_hash_failures, f"private output hash check failed: {private_hash_failures}")

    predictions = pd.read_parquet(private_output_root / "outer_fold_predictions.parquet")
    target = predictions.target_month.astype(str)
    require(not predictions.empty, "outer predictions are empty")
    require(target.max() <= DEV_END, "outer predictions include a post-development target")
    require(not target.duplicated().any(), "outer target months are duplicated")
    require(set(predictions.partition.astype(str)) == {"validation"}, "outer OOF table contains a non-validation partition")
    expected_issue = target.map(lambda month: f"{pd.Period(month, freq='M') - 1}-01T00:00:00Z")
    require(predictions.issue_date_utc.astype(str).reset_index(drop=True).equals(expected_issue.reset_index(drop=True)), "issue timestamps do not match the frozen contract")
    expected_folds = {"outer_2019", "outer_2020", "outer_2021"}
    require(set(predictions.outer_fold.astype(str)) == expected_folds, "outer fold labels differ from the frozen design")
    require(np.isfinite(predictions[[f"prediction__{name}" for name in STRATEGIES]].to_numpy(dtype=float)).all(), "non-finite outer forecast")
    require(predictions.gate_weight.between(0, 1).all(), "gate weights are outside [0,1]")

    fold_selections = pd.read_csv(result_dir / "fold_selections.csv")
    selected_by_fold = fold_selections.set_index("outer_fold").selected_strategy.astype(str).to_dict()
    for fold_name, group in predictions.groupby("outer_fold", sort=False):
        strategy = selected_by_fold[str(fold_name)]
        require(group.selected_strategy.astype(str).eq(strategy).all(), f"selection record mismatch in {fold_name}")
        require(
            np.allclose(group.prediction__nested_selected, group[f"prediction__{strategy}"], atol=1e-12, rtol=0),
            f"nested selected prediction does not match its fold-local selected strategy in {fold_name}",
        )
        expected_unit = np.where(
            group.gate_proxy_training_q75.to_numpy(dtype=float) > group.gate_proxy_training_q25.to_numpy(dtype=float),
            np.clip(
                (group.gate_proxy_value.to_numpy(dtype=float) - group.gate_proxy_training_q25.to_numpy(dtype=float))
                / (group.gate_proxy_training_q75.to_numpy(dtype=float) - group.gate_proxy_training_q25.to_numpy(dtype=float)),
                0.0,
                1.0,
            ),
            (group.gate_proxy_value.to_numpy(dtype=float) > group.gate_proxy_training_q75.to_numpy(dtype=float)).astype(float),
        )
        require(np.allclose(group.gate_unit_weight.to_numpy(dtype=float), expected_unit, atol=1e-12, rtol=0), f"outer gate weights cannot be reproduced in {fold_name}")
        require(np.allclose(group.gate_weight, group.gate_strength * group.gate_unit_weight, atol=1e-12, rtol=0), f"outer gate strength mismatch in {fold_name}")

    inner_oof = pd.read_parquet(private_output_root / "inner_selection_oof.parquet")
    require(not inner_oof.empty and inner_oof.target_month.astype(str).max() <= DEV_END, "inner selection OOF includes empty or post-cutoff rows")
    require(not inner_oof.duplicated(["outer_fold", "target_month"]).any(), "inner selection OOF target repeats within an outer fold")
    for fold_name, group in inner_oof.groupby("outer_fold", sort=False):
        outer_year = int(str(fold_name).split("_")[-1])
        require((group.inner_train_end_month.astype(str) < group.target_month.astype(str)).all(), f"inner training reaches validation in {fold_name}")
        require((group.target_month.astype(str) < f"{outer_year:04d}-01").all(), f"inner validation reaches the outer validation year in {fold_name}")
        require((group.inner_train_sample_count.astype(int) >= 18).all(), f"inner training is below its declared minimum in {fold_name}")
        for proxy in registry["configuration"]["gate_proxies"]:
            value = group[f"gate_value__{proxy}"].to_numpy(dtype=float)
            low = group[f"gate_train_q25__{proxy}"].to_numpy(dtype=float)
            high = group[f"gate_train_q75__{proxy}"].to_numpy(dtype=float)
            expected_unit = np.where(high > low, np.clip((value - low) / (high - low), 0.0, 1.0), (value > high).astype(float))
            require(np.allclose(group[f"gate_unit__{proxy}"].to_numpy(dtype=float), expected_unit, atol=1e-12, rtol=0), f"inner gate proxy weights cannot be reproduced for {proxy}")

    tuning = pd.read_csv(result_dir / "inner_tuning_scores.csv")
    expected_tuning_counts = {"ridge_alpha": 4 * 3, "smooth_blend": 5 * 3, "gate_proxy_strength": len(registry["configuration"]["gate_proxies"]) * 5 * 3}
    tuning_counts = tuning.tuning_family.value_counts().to_dict()
    require(tuning_counts == expected_tuning_counts, f"nested parameter score inventory differs: {tuning_counts}")
    require(set(tuning.loc[tuning.tuning_family.eq("gate_proxy_strength"), "gate_proxy"].dropna().astype(str)) <= ALLOWED_GATE_FEATURES, "inner tuning contains a gate field outside the allowlist")

    metrics = pd.read_csv(result_dir / "overall_metrics.csv").set_index("strategy")
    y = predictions.observed.to_numpy(dtype=float)
    baseline = predictions.climatology.to_numpy(dtype=float)
    metric_failures: list[str] = []
    for strategy in STRATEGIES:
        actual = paired_metrics(y, predictions[f"prediction__{strategy}"].to_numpy(dtype=float), baseline)
        for name, value in actual.items():
            if not np.isclose(float(metrics.loc[strategy, name]), float(value), atol=1e-12, rtol=1e-10):
                metric_failures.append(f"{strategy}.{name}")
    require(not metric_failures, f"independent overall metric recomputation differs: {metric_failures}")

    # Filter v1.4 OOF rows before pandas conversion to remain inside development.
    v14_record = registry["input_files"]["v14_fold_predictions"]
    v14_path = resolve_input_path(str(v14_record["path"]), repo_root, data_root)
    v14 = ds.dataset(v14_path, format="parquet").to_table(filter=ds.field("target_month") <= DEV_END).to_pandas()
    selected = v14[v14.model_id.eq("nested_selected_pipeline")].copy()
    climate = v14[v14.model_id.eq("training_monthly_climatology")].copy()
    paired = selected.merge(climate[["sample_id", "prediction_t2m_c"]], on="sample_id", suffixes=("", "_baseline"), validate="one_to_one")
    require(not paired.empty and paired.target_month.astype(str).max() <= DEV_END, "v1.4 paired OOF verification escaped the development period")
    saved_v14 = json.loads((result_dir / "v14_regime_recomputed.json").read_text(encoding="utf-8"))
    threshold = paired.training_anomaly_q75_c.to_numpy(dtype=float)
    large = np.abs(paired.absolute_anomaly_c.to_numpy(dtype=float)) >= threshold
    v14_failures: list[str] = []
    for name, mask in (("large_anomaly", large), ("ordinary", ~large)):
        computed_model = float(np.mean(np.abs(paired.loc[mask, "prediction_t2m_c"] - paired.loc[mask, "observed_t2m_c"])))
        computed_base = float(np.mean(np.abs(paired.loc[mask, "prediction_t2m_c_baseline"] - paired.loc[mask, "observed_t2m_c"])))
        saved = saved_v14["regimes"][name]
        if int(mask.sum()) != int(saved["sample_count"]):
            v14_failures.append(f"{name}.sample_count")
        if not np.isclose(computed_model, saved["nested_selected_pipeline_mae_c"], atol=1e-12):
            v14_failures.append(f"{name}.model_mae")
        if not np.isclose(computed_base, saved["training_climatology_mae_c"], atol=1e-12):
            v14_failures.append(f"{name}.baseline_mae")
    require(not v14_failures, f"independent v1.4 recomputation differs: {v14_failures}")

    availability = pd.read_csv(private_output_root / "issue_time_availability_audit.csv")
    require(availability.target_month.astype(str).max() <= DEV_END, "availability audit includes post-development forecast targets")
    require(not availability.first_publication_timestamp_verified.astype(bool).any(), "CMR production metadata was misrepresented as first-publication proof")
    require(registry["source_availability"]["first_publication_timestamp_verified"] is False, "registry overstates publication-time verification")
    require(registry["v1_4_immutability"]["status"] == "PASS", "v1.4 immutability audit did not pass")
    require(registry["v1_4_immutability"]["hashes_before"] == registry["v1_4_immutability"]["hashes_after"], "v1.4 artifact hashes changed")
    require(registry["evaluation"]["outer_oof_sample_count"] == len(predictions), "registry outer sample count differs from predictions")
    require(registry["evaluation"]["target_range"] == [target.min(), target.max()], "registry target range differs from predictions")

    print(json.dumps({
        "status": "PASS",
        "output_files_hash_verified": len(hash_index["files"]),
        "input_and_implementation_hash_verified": len(registry["input_files"]),
        "outer_oof_rows": len(predictions),
        "target_range": [target.min(), target.max()],
        "outer_folds": sorted(expected_folds),
        "independent_metric_recomputations": len(STRATEGIES),
        "v14_development_regime_rows": len(paired),
        "v1_4_regime_recomputation": "PASS",
        "v1_4_immutability": "PASS",
        "private_row_level_output_hashes_verified": len(private_records),
        "post_2021_target_values_read": False,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
