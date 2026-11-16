#!/usr/bin/env python3
"""Recompute available monthly external corroboration without reading test forecasts."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
NOAA = ROOT / "data/external/processed/ghcnm_ecuador_bbox_monthly_composite_v1.parquet"
NOAA_STATIONS = ROOT / "data/external/processed/ghcnm_ecuador_bbox_station_monthly_v1.parquet"
CHIRPS = ROOT / "data/external/processed/chirps_v3_ecuador_bbox_monthly_v1.parquet"
OUT = ROOT / "results/phase2/v1_1/external"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def pair_metrics(source: str, variable: str, nasa: pd.Series, ref: pd.Series) -> tuple[dict, pd.DataFrame]:
    paired = pd.concat([nasa.rename("nasa"), ref.rename("reference")], axis=1).dropna()
    if paired.empty:
        return {"source": source, "variable": variable, "status": "NO_OVERLAP", "n_months": 0}, paired
    error = paired.reference.to_numpy(float) - paired.nasa.to_numpy(float)
    corr = float(np.corrcoef(paired.nasa, paired.reference)[0, 1]) if len(paired) >= 3 and paired.nasa.std() and paired.reference.std() else None
    summary = {
        "source": source,
        "variable": variable,
        "status": "MONTHLY_AGGREGATE_CORROBORATION_ONLY",
        "n_months": int(len(paired)),
        "first_month": str(paired.index.min()),
        "last_month": str(paired.index.max()),
        "pearson_r": corr,
        "bias_reference_minus_nasa": float(error.mean()),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }
    paired["error_reference_minus_nasa"] = error
    return summary, paired


def seasonal_rows(source: str, variable: str, paired: pd.DataFrame) -> list[dict]:
    if paired.empty:
        return []
    month = pd.PeriodIndex(paired.index.astype(str), freq="M").month
    # Fixed local-season grouping; DJF spans year boundaries but each month is
    # assigned by its calendar month for stable coverage accounting.
    seasons = np.select([month.isin([12, 1, 2]), month.isin([3, 4, 5]), month.isin([6, 7, 8])], ["DJF", "MAM", "JJA"], default="SON")
    result = []
    for season in ("DJF", "MAM", "JJA", "SON"):
        group = paired.loc[seasons == season]
        if group.empty:
            continue
        error = group.error_reference_minus_nasa.to_numpy(float)
        result.append({"source": source, "variable": variable, "season": season, "n_months": int(len(group)), "bias_reference_minus_nasa": float(error.mean()), "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error**2)))})
    return result


def read_gold(path: Path, variable: str, threshold: float = 0.8) -> pd.Series:
    frame = pd.read_parquet(path)
    frame = frame[(frame.region_id == "ecuador_pilot_bbox") & (frame.variable_id == variable) & (frame.valid_area_fraction_bbox >= threshold)].copy()
    if frame.month.duplicated().any():
        raise ValueError(f"Duplicate NASA Gold month for {variable}")
    return frame.set_index(frame.month.astype(str)).value.sort_index()


def main() -> None:
    inputs = (GOLD, NOAA, NOAA_STATIONS, CHIRPS)
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        raise SystemExit(f"Required validated external input missing: {missing}")
    gold_manifest = json.loads((ROOT / "data/manifests/gold_manifest_v1.json").read_text(encoding="utf-8"))
    if sha256(GOLD) != gold_manifest["sha256"]:
        raise SystemExit("Frozen NASA Gold hash does not match its engineering manifest")
    noaa_manifest = json.loads((ROOT / "data/manifests/external/ghcnm_qcu_external_manifest_v1.json").read_text(encoding="utf-8"))
    chirps_manifest = json.loads((ROOT / "data/manifests/external/chirps_v3_external_manifest_v1.json").read_text(encoding="utf-8"))
    if noaa_manifest.get("status") != "ACQUIRED_AND_VALIDATED" or chirps_manifest.get("status") != "ACQUIRED_AND_VALIDATED":
        raise SystemExit("NOAA/CHIRPS acquisition manifests are not validated")
    noaa = pd.read_parquet(NOAA).set_index("month").sort_index()
    station = pd.read_parquet(NOAA_STATIONS)
    chirps = pd.read_parquet(CHIRPS).set_index("month").sort_index()
    if noaa.index.duplicated().any() or chirps.index.duplicated().any():
        raise SystemExit("Duplicate monthly external summary keys")
    rows, seasons = [], []

    merra = read_gold(GOLD, "merra2_t2m_c", 0.8)
    noaa_ref = noaa.tavg_equal_station_mean_c.where(noaa.eligible_station_count >= 3)
    summary, paired = pair_metrics("NOAA_GHCN_M_v4_QCU", "monthly_temperature_degC", merra, noaa_ref)
    summary.update({"eligible_station_month_records": int(station.eligible.sum()), "distinct_stations_with_records": int(station.loc[station.eligible, "station_id"].nunique()), "station_elevation_min_m": float(station.loc[station.eligible, "elevation_m"].min()), "station_elevation_median_m": float(station.loc[station.eligible, "elevation_m"].median()), "station_elevation_max_m": float(station.loc[station.eligible, "elevation_m"].max()), "station_level_collocation": "BLOCKED_NO_NATIVE_NASA_GRID_CELL_VALUES_IN_LOCAL_WORKSPACE"})
    rows.append(summary)
    seasons.extend(seasonal_rows("NOAA_GHCN_M_v4_QCU", "monthly_temperature_degC", paired))

    imerg = read_gold(GOLD, "imerg_precip_monthly_total", 0.8)
    chirps_ref = chirps.area_weighted_mean_mm_month.where(chirps.valid_pixel_fraction_bbox >= 0.8)
    summary, paired = pair_metrics("CHIRPS_v3_final_monthly", "monthly_precipitation_mm", imerg, chirps_ref)
    summary.update({"median_chirps_valid_pixel_fraction_on_source": float(chirps.valid_pixel_fraction_bbox.median()), "independence_class": "CROSS_PRODUCT_CORROBORATION_NOT_INDEPENDENT_GROUND_TRUTH"})
    rows.append(summary)
    seasons.extend(seasonal_rows("CHIRPS_v3_final_monthly", "monthly_precipitation_mm", paired))

    inamhi_path = ROOT / "data/external/processed/inamhi_pilot_bbox_monthly_composite_2002_2024_v1.parquet"
    inamhi_manifest_path = ROOT / "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json"
    if inamhi_path.is_file() and inamhi_manifest_path.is_file():
        inamhi_manifest = json.loads(inamhi_manifest_path.read_text(encoding="utf-8"))
        inamhi = pd.read_parquet(inamhi_path).set_index("month").sort_index()
        if inamhi_manifest.get("status") == "ACQUIRED_AND_VALIDATED_PARTIAL_YEARBOOK_COVERAGE":
            inamhi_ref = inamhi.inamhi_equal_station_tavg_c.where(inamhi.inamhi_tavg_station_count >= 3)
            summary, paired = pair_metrics("INAMHI_yearbooks_partial", "monthly_temperature_degC", merra, inamhi_ref)
            summary.update({"external_manifest_coverage": "2002-01 through 2012-12 yearbooks only; 2013-2024 unavailable", "station_level_collocation": "BLOCKED_NO_NATIVE_NASA_GRID_CELL_VALUES_IN_LOCAL_WORKSPACE"})
            rows.append(summary)
            seasons.extend(seasonal_rows("INAMHI_yearbooks_partial", "monthly_temperature_degC", paired))

    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "monthly_external_corroboration_v1_1.csv"
    season_path = OUT / "seasonal_external_corroboration_v1_1.csv"
    manifest_path = OUT / "external_corroboration_manifest_v1_1.json"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    pd.DataFrame(seasons).to_csv(season_path, index=False)
    manifest = {
        "version": "kuntur-external-corroboration-v1.1.0",
        "status": "COMPLETED_WITH_SPATIAL_SUPPORT_LIMITATION",
        "inputs": {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in inputs},
        "gold_manifest_sha256": sha256(ROOT / "data/manifests/gold_manifest_v1.json"),
        "external_manifests": {"NOAA": sha256(ROOT / "data/manifests/external/ghcnm_qcu_external_manifest_v1.json"), "CHIRPS": sha256(ROOT / "data/manifests/external/chirps_v3_external_manifest_v1.json")},
        "policies": {"nasa_coverage_threshold": 0.8, "noaa_month_eligibility": "at least three unique QC-eligible stations", "chirps_month_eligibility": "at least 0.8 valid pixel fraction", "station_level_collocation": "not calculated because local NASA artifacts contain only regional monthly Gold summaries; no station-cell source-grid values are available in this workspace", "forecast_test_access": "none; no forecast/test predictions are read", "interpretation": "descriptive time-series corroboration only; NOAA point network and NASA bbox mean have different spatial support; CHIRPS is a gridded cross-product, not independent ground truth"},
        "comparisons": rows,
        "artifacts": {str(p.relative_to(ROOT)): {"sha256": sha256(p), "bytes": p.stat().st_size} for p in (summary_path, season_path)},
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temp = manifest_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(manifest_path)
    print(json.dumps(json_safe({"status": manifest["status"], "comparisons": rows, "artifacts": manifest["artifacts"], "manifest": str(manifest_path)}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
