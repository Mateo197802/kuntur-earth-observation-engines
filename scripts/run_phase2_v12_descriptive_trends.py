#!/usr/bin/env python3
"""Write actual-series v1.2 descriptive slopes without unvalidated inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuntur_data.phase2.statistical_v12 import analyze_variable_v12  # noqa: E402

GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
GOLD_MANIFEST = ROOT / "data/manifests/gold_manifest_v1.json"
PROTOCOL = ROOT / "docs/protocols/statistical_correction_v1_2.md"
OUTPUT_DIR = ROOT / "results/statistical_correction/real_series_v12"
VARIABLES = (
    "merra2_t2m_c",
    "imerg_precip_monthly_total",
    "gldas_soil_moisture_0_10cm",
    "gldas_soil_moisture_10_40cm",
    "gldas_soil_moisture_40_100cm",
    "gldas_soil_moisture_100_200cm",
    "modis_ndvi",
    "modis_evi",
)
REGION_ID = "ecuador_pilot_bbox"
START_MONTH = "2002-01"
END_MONTH = "2024-12"
METHOD = "ols_hac"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def _threshold(variable_id: str) -> float:
    return 0.1 if variable_id.startswith("modis_") else 0.8


def build_trend_rows(gold: pd.DataFrame) -> list[dict]:
    required = {"month", "variable_id", "value", "valid_area_fraction_bbox", "units", "region_id"}
    missing = required - set(gold.columns)
    if missing:
        raise ValueError(f"Frozen Gold is missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for variable_id in VARIABLES:
        series = gold.loc[
            (gold.variable_id.astype(str) == variable_id)
            & (gold.region_id.astype(str) == REGION_ID)
        ].copy()
        if len(series) != 276 or series.month.astype(str).duplicated().any():
            raise ValueError(f"Expected 276 unique monthly rows for {variable_id}; found {len(series)}")
        threshold = _threshold(variable_id)
        result = analyze_variable_v12(
            gold,
            variable_id,
            region_id=REGION_ID,
            start_month=START_MONTH,
            end_month=END_MONTH,
            method=METHOD,
            qa_policy={"min_valid_area_fraction_bbox": threshold},
            declared_regime="REAL_SERIES_DESCRIPTIVE_ONLY",
            validation_registry=None,
        )
        if result.get("p_value") is not None or result.get("confidence_interval_95") is not None:
            raise AssertionError("v1.2 descriptive output must withhold unvalidated p-values and intervals")
        if result.get("validity_status") == "VALIDATED_FOR_DECLARED_REGIME":
            raise AssertionError("No real-series inferential regime is validated in this run")
        coverage = pd.to_numeric(series.valid_area_fraction_bbox, errors="coerce")
        values = pd.to_numeric(series.value, errors="coerce")
        eligible = values.notna() & coverage.notna() & (coverage >= threshold)
        if int(eligible.sum()) != int(result["n_eligible_observations"]):
            raise AssertionError(f"QA eligible count disagrees with the v1.2 engine for {variable_id}")
        rows.append({
            "region_id": REGION_ID,
            "variable_id": variable_id,
            "units": str(series.units.dropna().iloc[0]),
            "slope_method": "OLS with calendar-month fixed effects; descriptive slope only",
            "slope_per_decade": result.get("slope_per_decade"),
            "slope_units": result.get("slope_units"),
            "calendar_start": result.get("calendar_start"),
            "calendar_end": result.get("calendar_end"),
            "observed_start": result.get("observed_start"),
            "observed_end": result.get("observed_end"),
            "n_calendar_months": result.get("n_calendar_months"),
            "n_eligible_observations": result.get("n_eligible_observations"),
            "calendar_coverage_fraction": result.get("calendar_coverage_fraction"),
            "qa_min_valid_area_fraction_bbox": threshold,
            "median_valid_area_fraction_bbox": float(coverage.median()),
            "max_missing_run_months": result.get("max_missing_run_months"),
            "validity_status": result.get("validity_status"),
            "convergence_status": result.get("convergence_status"),
            "p_value": None,
            "confidence_interval_95": None,
            "inference_status": "WITHHELD_NO_INDEPENDENTLY_VALIDATED_METHOD_REGIME",
            "method_note": result.get("method_note"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--gold-manifest", type=Path, default=GOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    gold_path = args.gold.resolve()
    manifest_path = args.gold_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gold_sha = sha256_file(gold_path)
    if manifest.get("sha256") != gold_sha:
        raise ValueError("Frozen v1.0 Gold checksum does not match its manifest")
    gold = pd.read_parquet(gold_path)
    rows = build_trend_rows(gold)
    csv_bytes = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n").encode("utf-8")
    output_dir = args.output_dir.resolve()
    csv_path = output_dir / "descriptive_trend_estimates.csv"
    _atomic_write(csv_path, csv_bytes)
    engine_path = ROOT / "src/kuntur_data/phase2/statistical_v12.py"
    script_path = Path(__file__).resolve()
    manifest_out = {
        "manifest_version": "kuntur-phase2-v12-real-series-descriptive-trends-0.1.0",
        "status": "DESCRIPTIVE_ESTIMATES_COMPLETE_INFERENCE_WITHHELD",
        "scope": "Eight predeclared real-series variables from the frozen Ecuador pilot-bbox v1.0 Gold; no model selection or inferential claims.",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "temporal_coverage": {"start_month": START_MONTH, "end_month": END_MONTH, "calendar_months": 276},
        "region_id": REGION_ID,
        "input": {"path": gold_path.name, "sha256": gold_sha, "manifest_sha256": sha256_file(manifest_path)},
        "protocol": {"path": PROTOCOL.relative_to(ROOT).as_posix(), "sha256": sha256_file(PROTOCOL)},
        "engine": {"path": engine_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(engine_path)},
        "entrypoint": {"path": script_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(script_path)},
        "method": METHOD,
        "qa_thresholds": {"default": 0.8, "modis": 0.1},
        "inference_policy": "p-values, confidence intervals, and multiplicity-adjusted results are omitted until an independently validated method-regime registry passes the frozen acceptance gates.",
        "rows": len(rows),
        "software_versions": {"python": platform.python_version(), "pandas": pd.__version__},
        "output": {"path": csv_path.name, "sha256": hashlib.sha256(csv_bytes).hexdigest(), "bytes": len(csv_bytes)},
    }
    manifest_bytes = (json.dumps(manifest_out, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    _atomic_write(output_dir / "descriptive_trend_manifest.json", manifest_bytes)
    print(json.dumps({"status": manifest_out["status"], "rows": len(rows), "output": str(csv_path), "output_sha256": manifest_out["output"]["sha256"], "manifest": str(output_dir / "descriptive_trend_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
