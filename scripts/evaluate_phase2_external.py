#!/usr/bin/env python3
"""Descriptive, frozen-protocol comparison to public external products."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
START, END = "2002-01", "2024-12"
REGION = "ecuador_pilot_bbox"
METRICS = ROOT / "results/phase2/metrics"
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
PROTOCOL = ROOT / "docs/protocols/external_validation_protocol_v1.md"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(manifest: dict, path: Path, artifact_key: str, label: str) -> str:
    record = manifest.get(artifact_key, {}).get(path.relative_to(ROOT).as_posix())
    if not record or record.get("sha256") != sha256(path):
        raise SystemExit(f"{label} file does not match its acquisition manifest: {path.relative_to(ROOT)}")
    return record["sha256"]


def summarize_pair(source: str, variable: str, left: pd.Series, right: pd.Series, *, caveat: str, station_count: pd.Series | None = None) -> dict:
    pair = pd.concat([left.rename("reference"), right.rename("external")], axis=1).dropna()
    if pair.empty:
        return {"source_id": source, "variable_id": variable, "status": "NO_OVERLAP", "n_months": 0, "limitation": caveat}
    delta = pair.external.to_numpy(dtype=float) - pair.reference.to_numpy(dtype=float)
    corr = None
    if len(pair) >= 3 and np.std(pair.reference.to_numpy()) > 0 and np.std(pair.external.to_numpy()) > 0:
        corr = float(np.corrcoef(pair.reference.to_numpy(dtype=float), pair.external.to_numpy(dtype=float))[0, 1])
    row = {
        "source_id": source, "variable_id": variable, "status": "DESCRIPTIVE_COMPARISON",
        "n_months": int(len(pair)), "first_month": str(pair.index.min()), "last_month": str(pair.index.max()),
        "pearson_r": corr, "bias_external_minus_reference": float(delta.mean()),
        "mae": float(np.abs(delta).mean()), "rmse": float(np.sqrt(np.mean(delta**2))),
        "limitation": caveat,
    }
    if station_count is not None:
        counts = station_count.reindex(pair.index).dropna()
        row["station_count_min"] = int(counts.min()) if len(counts) else None
        row["station_count_median"] = float(counts.median()) if len(counts) else None
        row["station_count_max"] = int(counts.max()) if len(counts) else None
    return row


def gold_series(gold: pd.DataFrame, variable: str, threshold: float = 0.8) -> pd.DataFrame:
    frame = gold[(gold.region_id == REGION) & (gold.variable_id == variable)].copy()
    frame = frame[frame.valid_area_fraction_bbox >= threshold]
    frame["month"] = frame.month.astype(str)
    return frame.set_index("month").sort_index()


def main() -> None:
    final_path = ROOT / "data/manifests/final_test_manifest_v1.json"
    freeze_path = ROOT / "data/manifests/model_selection_freeze_v1.json"
    if not final_path.is_file() or not freeze_path.is_file():
        raise SystemExit("Frozen model selection and final-test manifests are required before external scoring")
    final = json.loads(final_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if final.get("status") != "FINAL_TEST_EVALUATED_ONCE" or freeze.get("status") != "FROZEN_BEFORE_FINAL_TEST":
        raise SystemExit("Model selection/final evaluation state does not pass the external-evaluation gate")
    forecast_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    if sha256(forecast_path) != final["source_forecast_sha256"]:
        raise SystemExit("The forecast input no longer matches the one-time final-test manifest")
    if sha256(freeze_path) != final["selection_freeze_sha256"]:
        raise SystemExit("The final-test result does not point to the unchanged selection freeze")
    gold_manifest_path = ROOT / "data/manifests/gold_manifest_v1.json"
    gold_manifest = json.loads(gold_manifest_path.read_text(encoding="utf-8"))
    if sha256(GOLD) != gold_manifest["sha256"]:
        raise SystemExit("Frozen Gold SHA does not match its engineering manifest")
    gold = pd.read_parquet(GOLD)
    rows: list[dict] = []
    sources: dict[str, dict] = {}

    # NOAA GHCN-M v4 QCU: equal-station bbox composite versus gridded mean and forecasts.
    ghcn_manifest_path = ROOT / "data/manifests/external/ghcnm_qcu_external_manifest_v1.json"
    ghcn_path = ROOT / "data/external/processed/ghcnm_ecuador_bbox_monthly_composite_v1.parquet"
    ghcn_manifest = json.loads(ghcn_manifest_path.read_text(encoding="utf-8"))
    if ghcn_manifest.get("status") != "ACQUIRED_AND_VALIDATED":
        raise SystemExit("NOAA GHCN-M manifest is not validated")
    ghcn_sha = verify_artifact(ghcn_manifest, ghcn_path, "artifacts", "NOAA GHCN-M composite")
    ghcn = pd.read_parquet(ghcn_path).set_index("month").sort_index()
    ghcn.index = ghcn.index.astype(str)
    merra = gold_series(gold, "merra2_t2m_c")
    ghcn_station_count = ghcn.eligible_station_count
    ghcn_temperature = ghcn.tavg_equal_station_mean_c.where(ghcn_station_count >= 3)
    sources["NOAA_GHCN_M_v4_QCU"] = {"manifest": ghcn_manifest_path.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(ghcn_manifest_path), "data_sha256": ghcn_sha, "eligible_months": int(ghcn_temperature.notna().sum()), "eligible_station_count": len(ghcn_manifest.get("station_selection", {}).get("stations_with_eligible_observations", [])), "distinct_station_records": ghcn_manifest["counts"]["unique_bbox_stations_with_records"], "qc_flagged_rows_rejected": ghcn_manifest["counts"]["qc_flagged_rows"]}
    rows.append(summarize_pair("NOAA_GHCN_M_v4_QCU", "temperature_degC", merra.value, ghcn_temperature, caveat="Equal-station monthly composite versus area-weighted gridded bbox mean; not collocated; NOAA QCU source streams overlap other networks and no independence is claimed.", station_count=ghcn_station_count))

    predictions_path = METRICS / "final_test_selected_model_v1.parquet"
    predictions = pd.read_parquet(predictions_path)
    selected_predictions = predictions.loc[predictions.model_id == final["selected_architecture"]]
    if len(selected_predictions) != 36 or selected_predictions.target_month.duplicated().any():
        raise SystemExit("Expected exactly 36 unique one-time final-test predictions for the frozen selected architecture")
    series = selected_predictions.set_index("target_month").y_pred_degC.sort_index()
    rows.append(summarize_pair("NOAA_GHCN_M_v4_QCU", f"final_test_prediction:{final['selected_architecture']}", series, ghcn_temperature, caveat="External station corroboration only; these values did not enter model fitting, selection, calibration, or the final-test score. Point-network and gridded-bbox support differ.", station_count=ghcn_station_count))

    # CHIRPS v3 final monthly total versus IMERG monthly total.
    chirps_manifest_path = ROOT / "data/manifests/external/chirps_v3_external_manifest_v1.json"
    chirps_path = ROOT / "data/external/processed/chirps_v3_ecuador_bbox_monthly_v1.parquet"
    chirps_manifest = json.loads(chirps_manifest_path.read_text(encoding="utf-8"))
    if chirps_manifest.get("status") != "ACQUIRED_AND_VALIDATED" or chirps_manifest["period"]["missing_months"]:
        raise SystemExit("CHIRPS does not have complete validated monthly coverage")
    chirps_sha = verify_artifact(chirps_manifest, chirps_path, "artifact_hashes", "CHIRPS regional summary")
    chirps = pd.read_parquet(chirps_path).set_index("month").sort_index()
    chirps.index = chirps.index.astype(str)
    imerg = gold_series(gold, "imerg_precip_monthly_total")
    chirps_series = chirps.loc[chirps.valid_pixel_fraction_bbox >= 0.8, "area_weighted_mean_mm_month"]
    rows.append(summarize_pair("CHIRPS_v3_final_monthly", "precipitation_mm_month", imerg.value, chirps_series, caveat="Cross-product comparison, not independent ground truth: CHIRPS blends station observations and its satellite inputs share parts of the broader GPM/IMERG ecosystem."))
    sources["CHIRPS_v3_final_monthly"] = {"manifest": chirps_manifest_path.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(chirps_manifest_path), "summary_sha256": chirps_sha, "acquired_months": chirps_manifest["period"]["acquired_months"], "missing_months": chirps_manifest["period"]["missing_months"], "failed_months": chirps_manifest["counts"]["failed_months"], "median_valid_pixel_fraction_bbox": chirps_manifest["counts"]["median_valid_pixel_fraction_bbox"], "remote_cog_full_sha_claimed": False}

    # INAMHI published yearbook tables: use only months with >=3 eligible stations.
    inamhi_manifest_path = ROOT / "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json"
    inamhi_path = ROOT / "data/external/processed/inamhi_pilot_bbox_monthly_composite_2002_2024_v1.parquet"
    if inamhi_manifest_path.is_file() and inamhi_path.is_file():
        inamhi_manifest = json.loads(inamhi_manifest_path.read_text(encoding="utf-8"))
        inamhi_sha = verify_artifact(inamhi_manifest, inamhi_path, "artifact_hashes", "INAMHI composite")
        inamhi = pd.read_parquet(inamhi_path).set_index("month").sort_index()
        inamhi.index = inamhi.index.astype(str)
        temp = inamhi.loc[inamhi.inamhi_tavg_station_count >= 3, "inamhi_equal_station_tavg_c"]
        ppt = inamhi.loc[inamhi.inamhi_precipitation_station_count >= 3, "inamhi_equal_station_precipitation_mm_month"]
        rows.append(summarize_pair("INAMHI_yearbooks", "temperature_degC", merra.value, temp, caveat="Published station means versus area-weighted gridded bbox mean; changing station support and no value-level QC flags; PDFs have rights reserved; do not redistribute the extracted observations."))
        rows.append(summarize_pair("INAMHI_yearbooks", "precipitation_mm_month", imerg.value, ppt, caveat="Equal-station monthly total versus area-weighted satellite/gridded total; available yearbooks only, changing station support and no value-level QC flags; do not redistribute extracted observations."))
        sources["INAMHI_yearbooks"] = {"manifest": inamhi_manifest_path.relative_to(ROOT).as_posix(), "manifest_sha256": sha256(inamhi_manifest_path), "composite_sha256": inamhi_sha, "yearbooks_acquired": inamhi_manifest["counts"]["pdf_years_acquired"], "station_month_rows": inamhi_manifest["counts"]["bbox_station_month_rows"], "comparable_temp_months_3plus": inamhi_manifest["counts"]["months_with_3plus_tavg_stations"], "comparable_precip_months_3plus": inamhi_manifest["counts"]["months_with_3plus_precipitation_stations"], "missing_pdf_years": inamhi_manifest["counts"]["unavailable_pdf_years"], "rights_note": "INAMHI yearbook pages reserve rights; source PDFs and extracted station values are not included in the GitHub release."}
    else:
        sources["INAMHI_yearbooks"] = {"status": "NOT_AVAILABLE", "limitation": "No validated station composite or manifest was produced."}

    output = pd.DataFrame(rows)
    METRICS.mkdir(parents=True, exist_ok=True)
    csv_path = METRICS / "external_comparison_summary_v1.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(csv_path, index=False)
    artifacts = {csv_path.relative_to(ROOT).as_posix(): {"bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)}}
    manifest = {
        "manifest_version": "1.0.0", "status": "EVALUATED_WITH_SOURCE_LIMITS",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "region_id": REGION, "region_status": "PRELIMINARY_PILOT_BOUNDING_BOX",
        "period_expected": [START, END], "external_protocol_path": PROTOCOL.relative_to(ROOT).as_posix(), "external_protocol_sha256": sha256(PROTOCOL),
        "selection_freeze_path": freeze_path.relative_to(ROOT).as_posix(), "selection_freeze_sha256": sha256(freeze_path),
        "final_test_manifest_path": final_path.relative_to(ROOT).as_posix(), "final_test_manifest_sha256": sha256(final_path),
        "source_gold_sha256": sha256(GOLD), "external_sources": sources,
        "evaluation_code_sha256": sha256(Path(__file__).resolve()),
        "comparisons": rows, "artifacts": artifacts,
        "interpretation": "All metrics are descriptive external corroboration only. External sources were held out of model fitting, scaling, augmentation, selection, calibration, and training climatology. Station and gridded products have unequal spatial support; shared station/input streams preclude full independence claims.",
        "landsat": {"status": "BLOCKED_NO_SCIENCE_RASTER", "evidence": "The selected public USGS STAC item resolved to an HTTP 200 HTML login page (Content-Type text/html; 33,066 bytes), not a Landsat science file. No Landsat metric is reported."},
        "validation": {"status": "PASSED_DESCRIPTIVE_ONLY", "checks": ["frozen selection and final-test manifests required", "source summary checksums match acquisition manifests", "only >=0.8 valid-area CHIRPS months compared", "NOAA QCU QC-screened composite reused without refitting and only months with at least three eligible stations included", "INAMHI months require at least three eligible station records", "comparison support/independence limitations retained", "no external source used for model selection"]},
    }
    manifest_path = ROOT / "data/manifests/external/external_evaluation_manifest_v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    print(json.dumps({"status": manifest["status"], "comparisons": rows, "external_sources": sources, "artifact": artifacts, "manifest": manifest_path.relative_to(ROOT).as_posix()}, indent=2, default=str))


if __name__ == "__main__":
    main()
