#!/usr/bin/env python3
"""Chronological calibration of development-only forecast intervals.

Uses only the selected Round 3 model's saved out-of-fold residuals through
2018. Earlier folds calibrate later contiguous folds; no later target is used
to create an interval for an earlier target.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
OPT = ROOT / "results/optimization"
ROUND3 = OPT / "round_03_augmentation_robustness"
OUT = OPT / "round_05_interval_calibration"
NOMINAL = 0.90
ROLLING_MONTHS = 36


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _conformal_abs_quantile(residuals: np.ndarray, coverage: float) -> float:
    values = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Calibration residuals must be a nonempty finite vector")
    rank = int(math.ceil((len(values) + 1) * coverage))
    return float(values[min(rank, len(values)) - 1])


def _winkler(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    width = upper - lower
    return width + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)


def _intervals(calibration: pd.DataFrame, evaluation: pd.DataFrame) -> list[dict]:
    residual = calibration.y_true.to_numpy(dtype=float) - calibration.y_pred.to_numpy(dtype=float)
    if len(residual) < 9:
        raise ValueError("At least nine prior residuals are required for finite 90% calibration")
    alpha = 1.0 - NOMINAL
    q = _conformal_abs_quantile(residual, NOMINAL)
    signed_low, signed_high = np.quantile(residual, [alpha / 2, 1.0 - alpha / 2], method="higher")
    center = float(np.mean(residual))
    gaussian_half = float(norm.ppf(1.0 - alpha / 2) * np.std(residual, ddof=1))
    methods = {
        "expanding_symmetric_abs_residual": (-q, q),
        "expanding_signed_empirical_residual": (float(signed_low), float(signed_high)),
        "expanding_bias_corrected_gaussian": (center - gaussian_half, center + gaussian_half),
    }
    rolling = calibration.sort_values("target_month").tail(ROLLING_MONTHS)
    rolling_residual = rolling.y_true.to_numpy(dtype=float) - rolling.y_pred.to_numpy(dtype=float)
    rolling_q = _conformal_abs_quantile(rolling_residual, NOMINAL)
    rolling_low, rolling_high = np.quantile(rolling_residual, [alpha / 2, 1.0 - alpha / 2], method="higher")
    rolling_center = float(np.mean(rolling_residual))
    rolling_half = float(norm.ppf(1.0 - alpha / 2) * np.std(rolling_residual, ddof=1))
    methods.update({
        "rolling36_symmetric_abs_residual": (-rolling_q, rolling_q),
        "rolling36_signed_empirical_residual": (float(rolling_low), float(rolling_high)),
        "rolling36_bias_corrected_gaussian": (rolling_center - rolling_half, rolling_center + rolling_half),
    })
    rows: list[dict] = []
    for method, (offset_low, offset_high) in methods.items():
        for row in evaluation.itertuples(index=False):
            rows.append({
                "method": method, "target_month": row.target_month, "fold_id": row.fold_id,
                "y_true": float(row.y_true), "y_pred": float(row.y_pred),
                "lower": float(row.y_pred + offset_low), "upper": float(row.y_pred + offset_high),
                "calibration_count": int(len(rolling_residual) if method.startswith("rolling36") else len(residual)),
                "calibration_end_month": str(rolling.target_month.max() if method.startswith("rolling36") else calibration.target_month.max()),
                "calibration_start_month": str(rolling.target_month.min() if method.startswith("rolling36") else calibration.target_month.min()),
            })
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r3_manifest = json.loads((ROUND3 / "round_manifest.json").read_text(encoding="utf-8"))
    best = r3_manifest["best_candidate"]
    registry = [json.loads(line) for line in (ROUND3 / "fold_registry.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in registry if r.get("status") == "COMPLETED" and r.get("model_id") == best["model_id"] and r.get("training_scheme") == best["training_scheme"]]
    if len(records) != 3:
        raise ValueError(f"Expected one completed record for each of three folds; got {len(records)}")
    predictions = []
    for record in records:
        for prediction in record["predictions"]:
            predictions.append({
                "fold_id": record["fold_id"], "sample_id": prediction["sample_id"],
                "target_month": prediction["target_month"], "y_true": float(prediction["y_true"]),
                "y_pred": float(prediction["y_pred"]),
            })
    frame = pd.DataFrame(predictions).sort_values("target_month", kind="stable").reset_index(drop=True)
    if len(frame) != 108 or frame.target_month.duplicated().any():
        raise ValueError("Expected 108 unique, chronological development predictions")
    frame["target_date"] = pd.PeriodIndex(frame.target_month, freq="M").to_timestamp(how="start")
    frame["month_of_year"] = frame.target_date.dt.month
    frame["residual"] = frame.y_true - frame.y_pred
    fold_order = frame.groupby("fold_id").target_date.min().sort_values().index.tolist()
    all_interval_rows = []
    leakage_audit = []
    for i in range(1, len(fold_order)):
        prior = frame[frame.fold_id.isin(fold_order[:i])].copy()
        current = frame[frame.fold_id == fold_order[i]].copy()
        if not prior.target_date.max() < current.target_date.min():
            raise AssertionError(f"Temporal leakage detected: calibration ends {prior.target_month.max()}, evaluation starts {current.target_month.min()}")
        all_interval_rows.extend(_intervals(prior, current))
        leakage_audit.append({
            "evaluation_fold": fold_order[i], "evaluation_start": str(current.target_month.min()),
            "evaluation_end": str(current.target_month.max()), "calibration_folds": fold_order[:i],
            "calibration_start": str(prior.target_month.min()), "calibration_end": str(prior.target_month.max()),
            "calibration_targets": int(len(prior)), "strictly_chronological": True,
        })
    intervals = pd.DataFrame(all_interval_rows)
    if intervals.empty or len(intervals) != 72 * 6:
        raise ValueError(f"Expected six interval methods for 72 later-fold targets, got {len(intervals)} rows")
    intervals["covered"] = (intervals.y_true >= intervals.lower) & (intervals.y_true <= intervals.upper)
    intervals["width"] = intervals.upper - intervals.lower
    intervals["winkler"] = _winkler(intervals.y_true.to_numpy(), intervals.lower.to_numpy(), intervals.upper.to_numpy(), 1.0 - NOMINAL)
    summaries = []
    for grouping, keys in (("overall", ["method"]), ("fold", ["method", "fold_id"]), ("calendar_month", ["method", "month_of_year"])):
        intervals["month_of_year"] = pd.PeriodIndex(intervals.target_month, freq="M").month
        group = intervals.groupby(keys, sort=True, dropna=False)
        for key, values in group:
            key_values = key if isinstance(key, tuple) else (key,)
            summaries.append({
                "group_type": grouping, **dict(zip(keys, key_values)), "target_count": int(len(values)),
                "coverage_90": float(values.covered.mean()), "mean_interval_width_c": float(values.width.mean()),
                "mean_winkler_score_c": float(values.winkler.mean()),
                "calibration_count_min": int(values.calibration_count.min()), "calibration_count_max": int(values.calibration_count.max()),
            })
    interval_path = OUT / "development_prediction_intervals.csv"
    summary_path = OUT / "interval_calibration_summary.csv"
    interval_path.write_text(intervals.to_csv(index=False), encoding="utf-8")
    summary_path.write_text(pd.DataFrame(summaries).to_csv(index=False), encoding="utf-8")
    (OUT / "chronological_leakage_audit.json").write_text(json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "manifest_version": "kuntur-phase2-v1.2-round-5-interval-calibration-0.1.0",
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "selected_candidate": {"model_id": best["model_id"], "training_scheme": best["training_scheme"], "candidate_spec_sha256": best["candidate_spec_sha256"]},
        "source_registry_sha256": _sha(ROUND3 / "fold_registry.jsonl"),
        "source_round_manifest_sha256": _sha(ROUND3 / "round_manifest.json"),
        "source_target_months": int(len(frame)), "evaluated_target_months": int(intervals.target_month.nunique()),
        "evaluated_range": [str(intervals.target_month.min()), str(intervals.target_month.max())],
        "target_years": [2013, 2018], "nominal_coverage": NOMINAL,
        "frozen_interval_calibration": {
            "method": "expanding_symmetric_abs_residual",
            "nominal_coverage": NOMINAL,
            "absolute_residual_quantile_c": _conformal_abs_quantile(frame.residual.to_numpy(dtype=float), NOMINAL),
            "residual_count": int(len(frame)),
            "residual_target_range": [str(frame.target_month.min()), str(frame.target_month.max())],
            "source_candidate_spec_sha256": best["candidate_spec_sha256"],
        },
        "methods": ["expanding_symmetric_abs_residual", "expanding_signed_empirical_residual", "expanding_bias_corrected_gaussian", "rolling36_symmetric_abs_residual", "rolling36_signed_empirical_residual", "rolling36_bias_corrected_gaussian"],
        "chronological_audit": leakage_audit,
        "limitations": ["The first 2010-2012 fold supplies calibration residuals and is not scored.", "Only later folds 2013-2018 are scored (72 target months).", "The candidate and augmentation scheme were selected using all three development folds; these interval results are post-selection development evidence.", "Monthly residuals are dependent, so the finite-sample quantile is not claimed to have exchangeable conformal coverage.", "No 2019-2024 partition or sealed 2025 target was read."],
        "artifacts": {
            "development_prediction_intervals": {"path": interval_path.relative_to(ROOT).as_posix(), "sha256": _sha(interval_path), "rows": int(len(intervals))},
            "interval_calibration_summary": {"path": summary_path.relative_to(ROOT).as_posix(), "sha256": _sha(summary_path), "rows": int(len(summaries))},
            "chronological_leakage_audit": {"path": "results/optimization/round_05_interval_calibration/chronological_leakage_audit.json", "sha256": _sha(OUT / "chronological_leakage_audit.json")},
        },
    }
    (OUT / "round_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "evaluated_months": manifest["evaluated_target_months"], "methods": manifest["methods"], "artifacts": manifest["artifacts"], "limitations": manifest["limitations"]}, indent=2))


if __name__ == "__main__":
    main()
