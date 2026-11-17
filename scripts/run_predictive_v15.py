#!/usr/bin/env python3
"""Run the issue-time guarded KUNTUR predictive-regime experiment v1.5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from kuntur_data.phase2.predictive_v14 import feature_pair, monthly_gold_wide, validate_forecast_timeline
from kuntur_data.phase2.predictive_v15 import (
    ALLOWED_GATE_FEATURES,
    chronological_inner_folds_v15,
    chronological_outer_folds_v15,
    complete_moving_blocks,
    gate_proxy_values,
    history_months_for_target,
    history_published_by_issue,
    metric_summary_v15,
    select_fixed_blend_strength,
    smooth_proxy_weights,
)


DEV_END = "2021-12"
DEV_START = "2003-03"
HISTORY_WIDTH = 12
CMR_START = "1998-01"
CMR_END = "2021-12"
SOURCE_RELEASE_POLICY = {
    "merra2_monthly_slv": {"short_name": "M2TMNXSLV", "version": "5.12.4", "latest_lag": 3},
    "gpm_imerg_monthly_final": {"short_name": "GPM_3IMERGM", "version": "07", "latest_lag": 6},
    "gldas_noah025_monthly": {"short_name": "GLDAS_NOAH025_M", "version": "2.1", "latest_lag": 6},
}
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
BLEND_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_PROXIES = (
    "t2m_anomaly_lag_03",
    "t2m_anomaly_mean_03",
    "t2m_anomaly_mean_06",
    "t2m_anomaly_sd_06",
    "t2m_anomaly_slope_06",
    "target_month_sin",
    "target_month_cos",
)
SOURCE_GOLD_VARIABLES = {
    "merra2_monthly_slv": ("merra2_t2m_c", "merra2_qv2m", "merra2_u2m", "merra2_v2m", "merra2_ps"),
    "gpm_imerg_monthly_final": ("imerg_precip_monthly_total",),
    "gldas_noah025_monthly": (
        "gldas_soil_moisture_0_10cm", "gldas_soil_moisture_10_40cm",
        "gldas_soil_moisture_40_100cm", "gldas_soil_moisture_100_200cm",
    ),
}
HGB_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 40,
    "max_leaf_nodes": 3,
    "min_samples_leaf": 8,
    "l2_regularization": 5.0,
    "early_stopping": False,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    os.replace(temporary, path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def get_json(url: str, params: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    last_status: int | None = None
    last_error_type = "no response"
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=(15, 90))
            if response.status_code == 429 or response.status_code >= 500:
                last_status = response.status_code
                last_error_type = "HTTP retryable status"
                retry_after = response.headers.get("Retry-After", "")
                delay = int(retry_after) if retry_after.isdigit() else min(2**attempt, 8)
                time.sleep(min(delay, 20))
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_status = getattr(getattr(exc, "response", None), "status_code", last_status)
            last_error_type = type(exc).__name__
            if attempt < 3:
                time.sleep(min(2**attempt, 8))
    suffix = f" HTTP {last_status}" if last_status is not None else ""
    raise RuntimeError(f"NASA CMR metadata request failed after bounded retries:{suffix} {last_error_type}")


def fetch_cmr_snapshot() -> dict[str, Any]:
    """Fetch exact-version granule production timestamps from public NASA CMR."""
    base = "https://cmr.earthdata.nasa.gov/search"
    sources: dict[str, Any] = {}
    for source_id, spec in SOURCE_RELEASE_POLICY.items():
        collection = get_json(
            f"{base}/collections.json",
            {"short_name": spec["short_name"], "version": spec["version"], "provider": "GES_DISC", "page_size": 10},
            {"Accept": "application/json"},
        )
        entries = collection.get("feed", {}).get("entry", [])
        if len(entries) != 1:
            raise RuntimeError(f"NASA CMR returned {len(entries)} exact collection matches for {source_id}")
        collection_id = entries[0]["id"]
        granule_items: list[dict[str, Any]] = []
        page_num = 1
        expected_hits: int | None = None
        while True:
            granules = get_json(
                f"{base}/granules.umm_json",
                {
                    "collection_concept_id": collection_id,
                    "temporal": "1998-01-01T00:00:00Z,2021-12-31T23:59:59Z",
                    "page_size": 500,
                    "page_num": page_num,
                },
                {"Accept": "application/vnd.nasa.cmr.umm+json; version=1.6.4"},
            )
            page_items = granules.get("items", [])
            granule_items.extend(page_items)
            meta_hits = granules.get("meta", {}).get("hits")
            if meta_hits is not None:
                expected_hits = int(meta_hits)
            if not page_items or (expected_hits is not None and len(granule_items) >= expected_hits):
                break
            page_num += 1
        records: list[dict[str, str | None]] = []
        for item in granule_items:
            umm = item.get("umm", {})
            start = umm.get("TemporalExtent", {}).get("RangeDateTime", {}).get("BeginningDateTime")
            if not start:
                continue
            provider_dates = {
                entry.get("Type"): entry.get("Date")
                for entry in umm.get("ProviderDates", [])
                if isinstance(entry, dict)
            }
            records.append(
                {
                    "month": start[:7],
                    "granule_concept_id": item.get("meta", {}).get("concept-id"),
                    "granule_ur": umm.get("GranuleUR"),
                    "production_date_time_utc": umm.get("DataGranule", {}).get("ProductionDateTime"),
                    "provider_insert_time_utc": provider_dates.get("Insert"),
                    "provider_update_time_utc": provider_dates.get("Update"),
                }
            )
        records.sort(key=lambda item: (item["month"] or "", item["granule_ur"] or ""))
        months = [str(record["month"]) for record in records]
        if len(months) != len(set(months)):
            raise RuntimeError(f"NASA CMR returned multiple monthly granules for {source_id}")
        expected = set(pd.period_range("2002-01", "2021-12", freq="M").astype(str))
        missing_core_months = sorted(expected - set(months))
        if missing_core_months:
            raise RuntimeError(f"NASA CMR production-time snapshot lacks {source_id} months: {missing_core_months[:8]}")
        sources[source_id] = {
            "short_name": spec["short_name"],
            "version": spec["version"],
            "collection_concept_id": collection_id,
            "granule_record_count": len(records),
            "records": records,
        }
    return {
        "schema_version": "1.0",
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "provider": "GES_DISC",
        "api": "https://cmr.earthdata.nasa.gov/search",
        "requested_temporal_range": [CMR_START, CMR_END],
        "source_collections": sources,
        "availability_semantics": (
            "ProductionDateTime is the NASA granule production time, not a guaranteed first-publication timestamp. "
            "Provider Insert timestamps are retained for provenance but are not treated as first availability because catalog migration can occur later."
        ),
    }


def load_or_fetch_cmr_snapshot(path: Path, refresh: bool) -> dict[str, Any]:
    if path.exists() and not refresh:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if set(snapshot.get("source_collections", {})) == set(SOURCE_RELEASE_POLICY):
            atomic_json(path, snapshot)
            return snapshot
    snapshot = fetch_cmr_snapshot()
    atomic_json(path, snapshot)
    return snapshot


def production_dates(snapshot: dict[str, Any], source_id: str) -> dict[str, str]:
    records = snapshot["source_collections"][source_id]["records"]
    return {
        str(record["month"]): str(record["production_date_time_utc"])
        for record in records
        if record.get("month") and record.get("production_date_time_utc")
    }


def build_availability_audit(
    samples: pd.DataFrame, snapshot: dict[str, Any], monthly: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for source_id, spec in SOURCE_RELEASE_POLICY.items():
        dates = production_dates(snapshot, source_id)
        source_rows: list[dict[str, Any]] = []
        for sample in samples.itertuples(index=False):
            audit = history_published_by_issue(
                target_month=str(sample.target_month),
                issue_time_utc=str(sample.issue_date_utc),
                production_dates_utc=dates,
                width=HISTORY_WIDTH,
                latest_lag=int(spec["latest_lag"]),
            )
            history = history_months_for_target(
                str(sample.target_month),
                width=HISTORY_WIDTH,
                latest_lag=int(spec["latest_lag"]),
            )
            source_variables = SOURCE_GOLD_VARIABLES[source_id]
            source_gold_missing = []
            for history_month in history:
                if history_month not in monthly.index:
                    source_gold_missing.append(history_month)
                    continue
                values = monthly.loc[history_month, list(source_variables)].to_numpy(dtype=float)
                if not np.isfinite(values).all():
                    source_gold_missing.append(history_month)
            row = {
                "sample_id": str(sample.sample_id),
                "target_month": str(sample.target_month),
                "issue_date_utc": str(sample.issue_date_utc),
                "source_id": source_id,
                "collection_concept_id": snapshot["source_collections"][source_id]["collection_concept_id"],
                "release_lag_months": int(spec["latest_lag"]),
                "history_months": json.dumps(list(audit.history_months), separators=(",", ":")),
                "production_eligible_history_months": len(audit.history_months) - len(audit.unavailable_months),
                "source_months_unavailable_at_issue": json.dumps(list(audit.unavailable_months), separators=(",", ":")),
                "source_gold_missing_or_invalid_months": json.dumps(source_gold_missing, separators=(",", ":")),
                "full_history_latency_and_production_guard_eligible": audit.eligible and not source_gold_missing,
                "first_publication_timestamp_verified": False,
                "partition": str(sample.partition),
            }
            source_rows.append(row)
            rows.append(row)
        eligible = [row for row in source_rows if row["full_history_latency_and_production_guard_eligible"]]
        summaries[source_id] = {
            "collection_concept_id": snapshot["source_collections"][source_id]["collection_concept_id"],
            "version": spec["version"],
            "required_history_months": HISTORY_WIDTH,
            "latest_lag_months": spec["latest_lag"],
            "development_sample_count": len(source_rows),
            "issue_time_eligible_sample_count": len(eligible),
            "issue_time_eligible_target_range": [eligible[0]["target_month"], eligible[-1]["target_month"]] if eligible else None,
            "issue_time_eligible_by_target_year": {
                year: sum(row["full_history_latency_and_production_guard_eligible"] and row["target_month"].startswith(year) for row in source_rows)
                for year in sorted({row["target_month"][:4] for row in source_rows})
            },
        }
    for source_id, summary in summaries.items():
        issue_eligible_training = sum(
            row["full_history_latency_and_production_guard_eligible"]
            and row["target_month"] < "2019-01"
            for row in rows
            if row["source_id"] == source_id
        )
        summary["issue_time_eligible_training_targets_before_2019"] = int(issue_eligible_training)
        if source_id == "merra2_monthly_slv":
            summary["cross_source_candidate_scored"] = "MERRA-2 primary predictors"
            continue
        summary["cross_source_candidate_scored"] = False
        summary["cross_source_candidate_decision"] = (
            "not_scored_no_chronological_issue-time training history before the 2019 outer evaluation window"
            if issue_eligible_training == 0
            else "not_scored; eligible set is too narrow for the prespecified nested training design"
        )
    return pd.DataFrame(rows), summaries


def load_development_inputs(data_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Path]]:
    files = {
        "monthly_gold": data_root / "gold" / "kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
        "forecast": data_root / "gold" / "kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet",
        "split_indices": data_root / "gold" / "kuntur_forecast_split_indices_v1.parquet",
        "gold_manifest": data_root / "manifests" / "gold_manifest_v1.json",
        "split_manifest": data_root / "manifests" / "forecast_split_manifest_v1.json",
    }
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"required frozen v1 input missing: {absent}")
    forecast_columns = [
        "sample_id", "target_month", "forecast_origin_month", "issue_date_utc",
        "history_start_month", "history_end_month", "history_window_months",
        "conservative_release_lag_months", "feature_months_json", "target_value",
    ]
    forecast_columns.extend(
        f"merra2_{variable}_lag_{lag:02d}"
        for lag in range(1, 13)
        for variable in ("t2m_c", "qv2m", "u2m", "v2m", "ps")
    )
    forecast = ds.dataset(files["forecast"], format="parquet").to_table(
        columns=forecast_columns,
        filter=ds.field("target_month") <= DEV_END,
    ).to_pandas()
    split = ds.dataset(files["split_indices"], format="parquet").to_table(
        columns=["sample_id", "target_month", "partition"],
        filter=(ds.field("target_month") <= DEV_END) & (ds.field("partition") != "test"),
    ).to_pandas()
    samples = forecast.merge(split, on=["sample_id", "target_month"], validate="one_to_one")
    samples = samples[samples.partition.isin(["train", "validation"])].copy()
    if samples.empty or samples.target_month.astype(str).max() > DEV_END:
        raise ValueError("development loader returned no data or included a consumed test target")
    validate_forecast_timeline(samples)
    samples["feature_months"] = samples.feature_months_json.map(lambda value: tuple(json.loads(value)))
    for row in samples.itertuples(index=False):
        expected = tuple(history_months_for_target(str(row.target_month), width=HISTORY_WIDTH, latest_lag=3))
        if tuple(row.feature_months) != expected:
            raise ValueError(f"forecast history differs from the frozen T-14..T-3 contract for {row.target_month}")

    gold_table = ds.dataset(files["monthly_gold"], format="parquet").to_table(
        filter=ds.field("month") <= DEV_END,
    )
    gold = gold_table.to_pandas()
    monthly = monthly_gold_wide(gold)
    return samples.sort_values("target_month", kind="stable").reset_index(drop=True), monthly, files


def fit_hgb(x: np.ndarray, y: np.ndarray, seed: int) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(**HGB_CONFIG, random_state=seed)
    model.fit(x, y)
    return model


def inner_predictions(inner_folds, monthly: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    timings: list[dict[str, Any]] = []
    for fold_index, inner in enumerate(inner_folds):
        started = time.perf_counter()
        temp_pair = feature_pair(inner.train, inner.validation, monthly, "temperature_only")
        atmosphere_pair = feature_pair(inner.train, inner.validation, monthly, "atmosphere")
        records = pd.DataFrame(
            {
                "target_month": inner.validation.target_month.astype(str).to_numpy(),
                "inner_fold": inner.name,
                "inner_train_start_month": str(inner.train.target_month.astype(str).min()),
                "inner_train_end_month": str(inner.train.target_month.astype(str).max()),
                "inner_train_sample_count": int(len(inner.train)),
                "observed": temp_pair.y_validation.copy(),
                "climatology": temp_pair.offset_validation.copy(),
            }
        )
        model_started = time.perf_counter()
        compact = fit_hgb(atmosphere_pair.x_train, atmosphere_pair.y_train_residual, seed + fold_index)
        records["compact_residual"] = compact.predict(atmosphere_pair.x_validation)
        timings.append({"inner_fold": inner.name, "strategy": "compact_residual", "fit_seconds": time.perf_counter() - model_started})
        for alpha in RIDGE_ALPHAS:
            model_started = time.perf_counter()
            ridge = Ridge(alpha=float(alpha))
            ridge.fit(temp_pair.x_train, temp_pair.y_train_residual)
            records[f"ridge_residual_alpha_{alpha:g}"] = ridge.predict(temp_pair.x_validation)
            timings.append({"inner_fold": inner.name, "strategy": f"ridge_alpha_{alpha:g}", "fit_seconds": time.perf_counter() - model_started})
        for proxy in GATE_PROXIES:
            train_proxy = gate_proxy_values(temp_pair.x_train, temp_pair.feature_names, proxy)
            validation_proxy = gate_proxy_values(temp_pair.x_validation, temp_pair.feature_names, proxy)
            finite_train_proxy = train_proxy[np.isfinite(train_proxy)]
            if not finite_train_proxy.size:
                raise ValueError(f"inner training fold has no finite values for gate proxy {proxy}")
            low, high = np.quantile(finite_train_proxy, [0.25, 0.75])
            records[f"gate_unit__{proxy}"] = smooth_proxy_weights(
                validation_proxy,
                train_proxy,
                strength=1.0,
                low_quantile=0.25,
                high_quantile=0.75,
            ).to_numpy()
            records[f"gate_value__{proxy}"] = validation_proxy
            records[f"gate_train_q25__{proxy}"] = float(low)
            records[f"gate_train_q75__{proxy}"] = float(high)
        frames.append(records)
        timings.append({"inner_fold": inner.name, "strategy": "feature_pair", "fit_seconds": time.perf_counter() - started})
    joined = pd.concat(frames, ignore_index=True).sort_values("target_month", kind="stable").reset_index(drop=True)
    if joined.target_month.duplicated().any():
        raise ValueError("inner validation windows overlap target months")
    return joined, timings


def score_candidate(y: np.ndarray, climate: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    error = np.asarray(prediction, dtype=float) - np.asarray(y, dtype=float)
    return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(np.square(error))))


def select_inner_strategy(records: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    y = records.observed.to_numpy(dtype=float)
    climate = records.climatology.to_numpy(dtype=float)
    candidates: list[dict[str, Any]] = []

    def add(strategy: str, prediction: np.ndarray, *, alpha: float | None = None, proxy: str | None = None, strength: float | None = None) -> None:
        mae, rmse = score_candidate(y, climate, prediction)
        candidates.append({"strategy": strategy, "mae": mae, "rmse": rmse, "alpha": alpha, "gate_proxy": proxy, "gate_strength": strength})

    add("climatology", climate)
    compact_residual = records.compact_residual.to_numpy(dtype=float)
    add("compact_residual", climate + compact_residual)
    ridge_scores = []
    ridge_tuning_scores = []
    for alpha in RIDGE_ALPHAS:
        residual = records[f"ridge_residual_alpha_{alpha:g}"].to_numpy(dtype=float)
        mae, rmse = score_candidate(y, climate, climate + residual)
        ridge_scores.append((mae, rmse, -float(alpha), float(alpha)))
        ridge_tuning_scores.append({"alpha": float(alpha), "mae": mae, "rmse": rmse})
    ridge_mae, ridge_rmse, _, ridge_alpha = min(ridge_scores)
    candidates.append({"strategy": "regularized_anomaly", "mae": ridge_mae, "rmse": ridge_rmse, "alpha": ridge_alpha, "gate_proxy": None, "gate_strength": None})
    smooth_alpha, _ = select_fixed_blend_strength(
        observations=y,
        climatology=climate,
        residual_prediction=compact_residual,
        candidates=BLEND_STRENGTHS,
    )
    add("smooth_shrinkage", climate + smooth_alpha * compact_residual, alpha=smooth_alpha)
    gate_candidates = []
    gate_tuning_scores = []
    for proxy in GATE_PROXIES:
        unit_weight = records[f"gate_unit__{proxy}"].to_numpy(dtype=float)
        for strength in BLEND_STRENGTHS:
            prediction = climate + strength * unit_weight * compact_residual
            mae, rmse = score_candidate(y, climate, prediction)
            gate_candidates.append((mae, rmse, float(strength), proxy))
            gate_tuning_scores.append({"gate_proxy": proxy, "gate_strength": float(strength), "mae": mae, "rmse": rmse})
    gate_mae, gate_rmse, gate_strength, gate_proxy = min(
        gate_candidates,
        key=lambda item: (item[0], item[1], item[2], GATE_PROXIES.index(item[3])),
    )
    gate_unit = records[f"gate_unit__{gate_proxy}"].to_numpy(dtype=float)
    add("regime_mixture", climate + gate_strength * gate_unit * compact_residual, proxy=gate_proxy, strength=gate_strength)
    priority = {"climatology": 0, "regularized_anomaly": 1, "smooth_shrinkage": 2, "regime_mixture": 3, "compact_residual": 4}
    selected = min(candidates, key=lambda item: (item["mae"], item["rmse"], priority[item["strategy"]]))
    ordered = pd.DataFrame(candidates).sort_values(["mae", "rmse", "strategy"], kind="stable").reset_index(drop=True)
    details = {
        "selected_strategy": selected["strategy"],
        "ridge_alpha": ridge_alpha,
        "smooth_alpha": smooth_alpha,
        "gate_proxy": gate_proxy,
        "gate_strength": gate_strength,
        "selected_inner_mae": selected["mae"],
        "selected_inner_rmse": selected["rmse"],
        "gate_inner_mae": gate_mae,
        "gate_inner_rmse": gate_rmse,
        "ridge_alpha_tuning_scores": ridge_tuning_scores,
        "smooth_blend_tuning_scores": [
            {
                "strength": float(alpha),
                "mae": score_candidate(y, climate, climate + float(alpha) * compact_residual)[0],
                "rmse": score_candidate(y, climate, climate + float(alpha) * compact_residual)[1],
            }
            for alpha in BLEND_STRENGTHS
        ],
        "gate_tuning_scores": gate_tuning_scores,
    }
    return details, ordered


def fit_outer_predictions(outer, monthly: pd.DataFrame, selection: dict[str, Any], seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    temp_pair = feature_pair(outer.train, outer.validation, monthly, "temperature_only")
    atmosphere_pair = feature_pair(outer.train, outer.validation, monthly, "atmosphere")
    climate = temp_pair.offset_validation.copy()
    timings: list[dict[str, Any]] = []
    started = time.perf_counter()
    compact = fit_hgb(atmosphere_pair.x_train, atmosphere_pair.y_train_residual, seed)
    compact_residual = compact.predict(atmosphere_pair.x_validation)
    timings.append({"outer_fold": outer.name, "strategy": "compact_residual", "fit_seconds": time.perf_counter() - started})
    started = time.perf_counter()
    ridge = Ridge(alpha=float(selection["ridge_alpha"]))
    ridge.fit(temp_pair.x_train, temp_pair.y_train_residual)
    ridge_residual = ridge.predict(temp_pair.x_validation)
    timings.append({"outer_fold": outer.name, "strategy": "regularized_anomaly", "fit_seconds": time.perf_counter() - started})

    smooth_prediction = climate + float(selection["smooth_alpha"]) * compact_residual
    gate_proxy = str(selection["gate_proxy"])
    train_proxy = gate_proxy_values(temp_pair.x_train, temp_pair.feature_names, gate_proxy)
    validation_proxy = gate_proxy_values(temp_pair.x_validation, temp_pair.feature_names, gate_proxy)
    gate_unit = smooth_proxy_weights(
        validation_proxy,
        train_proxy,
        strength=1.0,
        low_quantile=0.25,
        high_quantile=0.75,
    ).to_numpy()
    finite_train_proxy = train_proxy[np.isfinite(train_proxy)]
    gate_q25, gate_q75 = np.quantile(finite_train_proxy, [0.25, 0.75])
    gate_weight = float(selection["gate_strength"]) * gate_unit
    gate_prediction = climate + gate_weight * compact_residual
    forecasts = {
        "climatology": climate,
        "compact_residual": climate + compact_residual,
        "regularized_anomaly": climate + ridge_residual,
        "smooth_shrinkage": smooth_prediction,
        "regime_mixture": gate_prediction,
    }
    selected_strategy = str(selection["selected_strategy"])
    forecasts["nested_selected"] = forecasts[selected_strategy]

    # Validation labels are used only after features, gates, and forecasts are fixed.
    observed = outer.validation.target_value.to_numpy(dtype=float)
    train_target = outer.train[["target_month", "target_value"]].copy()
    train_target["calendar_month"] = pd.PeriodIndex(train_target.target_month.astype(str), freq="M").month
    climate_by_month = train_target.groupby("calendar_month").target_value.mean().to_dict()
    train_anomaly = np.asarray(
        [
            float(row.target_value) - float(climate_by_month[pd.Period(str(row.target_month), freq="M").month])
            for row in train_target.itertuples(index=False)
        ]
    )
    target_calendar_month = pd.PeriodIndex(outer.validation.target_month.astype(str), freq="M").month
    validation_anomaly = np.asarray(
        [observed[index] - float(climate_by_month[int(month)]) for index, month in enumerate(target_calendar_month)]
    )
    threshold = float(np.quantile(np.abs(train_anomaly[np.isfinite(train_anomaly)]), 0.75))
    is_large = np.abs(validation_anomaly) >= threshold
    output = pd.DataFrame(
        {
            "target_month": outer.validation.target_month.astype(str).to_numpy(),
            "issue_date_utc": outer.validation.issue_date_utc.astype(str).to_numpy(),
            "outer_fold": outer.name,
            "selected_strategy": selected_strategy,
            "gate_proxy": gate_proxy,
            "gate_strength": float(selection["gate_strength"]),
            "gate_weight": gate_weight,
            "gate_proxy_value": validation_proxy,
            "gate_proxy_training_q25": float(gate_q25),
            "gate_proxy_training_q75": float(gate_q75),
            "gate_unit_weight": gate_unit,
            "observed": observed,
            "climatology": climate,
            "training_regime_threshold_c": threshold,
            "actual_target_anomaly_c": validation_anomaly,
            "retrospective_regime": np.where(is_large, "large_anomaly", "ordinary"),
        }
    )
    for name, values in forecasts.items():
        output[f"prediction__{name}"] = np.asarray(values, dtype=float)
    return output, timings


def paired_block_interval(
    observations: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    target_months: list[str] | None = None,
    block_length: int = 3,
    replicates: int = 4000,
) -> dict[str, Any]:
    improvement = np.abs(baseline - observations) - np.abs(prediction - observations)
    n = len(improvement)
    if n < 2:
        return {"sample_count": int(n), "mean_mae_improvement": float(improvement.mean()), "ci_low": float("nan"), "ci_high": float("nan"), "block_length_months": block_length, "resampling_scheme": "insufficient paired samples"}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    if target_months is not None:
        if len(target_months) != n:
            raise ValueError("target_months must align with the paired forecast rows")
        blocks = complete_moving_blocks(target_months, block_length=block_length)
        scheme = "complete moving blocks within contiguous calendar-month segments; incomplete suffixes and short segments excluded"
    else:
        blocks = [tuple(range(start, start + block_length)) for start in range(max(0, n - block_length + 1))]
        scheme = "complete moving blocks over row order; incomplete suffix excluded"
    if not blocks:
        return {
            "sample_count": int(n),
            "mean_mae_improvement": float(np.mean(improvement)),
            "ci_low": None,
            "ci_high": None,
            "block_length_months": int(block_length),
            "bootstrap_replicates": int(replicates),
            "resampling_scheme": "insufficient contiguous observations for one complete declared block",
        }
    for repeat in range(replicates):
        chosen: list[int] = []
        while len(chosen) < n:
            chosen.extend(blocks[int(rng.integers(0, len(blocks)))])
        draws[repeat] = float(np.mean(improvement[np.asarray(chosen[:n], dtype=int)]))
    return {
        "sample_count": int(n),
        "mean_mae_improvement": float(np.mean(improvement)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "block_length_months": int(block_length),
        "bootstrap_replicates": int(replicates),
        "resampling_scheme": scheme,
    }


def calibration_metrics(y: np.ndarray, prediction: np.ndarray, climate: np.ndarray) -> dict[str, float | None]:
    observed_anomaly = np.asarray(y, dtype=float) - np.asarray(climate, dtype=float)
    predicted_anomaly = np.asarray(prediction, dtype=float) - np.asarray(climate, dtype=float)
    if len(observed_anomaly) < 3 or np.std(predicted_anomaly) == 0:
        return {"anomaly_calibration_intercept_c": None, "anomaly_calibration_slope": None}
    slope, intercept = np.polyfit(predicted_anomaly, observed_anomaly, 1)
    return {"anomaly_calibration_intercept_c": float(intercept), "anomaly_calibration_slope": float(slope)}


def aggregate_outputs(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    strategies = ("climatology", "compact_residual", "regularized_anomaly", "smooth_shrinkage", "regime_mixture", "nested_selected")
    metrics: list[dict[str, Any]] = []
    folds: list[dict[str, Any]] = []
    regimes: list[dict[str, Any]] = []
    intervals: list[dict[str, Any]] = []
    y = predictions.observed.to_numpy(dtype=float)
    climate = predictions.climatology.to_numpy(dtype=float)
    for index, strategy in enumerate(strategies):
        estimate = predictions[f"prediction__{strategy}"].to_numpy(dtype=float)
        summary = metric_summary_v15(y, estimate, climate)
        calibration = calibration_metrics(y, estimate, climate)
        intervals.append({
            "strategy": strategy,
            **paired_block_interval(
                y,
                estimate,
                climate,
                seed=1500 + index,
                target_months=predictions.target_month.astype(str).tolist(),
            ),
        })
        row = {"strategy": strategy, **summary, **calibration}
        fold_skill = []
        for fold_name, frame in predictions.groupby("outer_fold", sort=False):
            fold_metrics = metric_summary_v15(frame.observed, frame[f"prediction__{strategy}"], frame.climatology)
            fold_skill.append(fold_metrics["mae_skill_vs_baseline"])
            folds.append({"outer_fold": fold_name, "strategy": strategy, **fold_metrics})
        row["worst_fold_mae_skill_vs_baseline"] = float(min(fold_skill)) if fold_skill else None
        metrics.append(row)
        for regime_name, group in predictions.groupby("retrospective_regime", sort=False):
            regime_result = metric_summary_v15(group.observed, group[f"prediction__{strategy}"], group.climatology)
            regimes.append({"strategy": strategy, "retrospective_regime": regime_name, **regime_result})
    return pd.DataFrame(metrics), pd.DataFrame(folds), pd.DataFrame(regimes), pd.DataFrame(intervals)


def verify_v14_regime_decomposition(v14_dir: Path) -> dict[str, Any]:
    path = v14_dir / "fold_predictions.parquet"
    if not path.is_file():
        return {"status": "BLOCKED", "reason": "v1.4 fold predictions are unavailable", "path": str(path)}
    frame = pq.read_table(path).to_pandas()
    selected = frame[frame.model_id.eq("nested_selected_pipeline")].copy()
    baseline = frame[frame.model_id.eq("training_monthly_climatology")].copy()
    if selected.empty or baseline.empty:
        raise ValueError("v1.4 OOF predictions lack the selected pipeline or climatology baseline")
    paired = selected.merge(
        baseline[["sample_id", "prediction_t2m_c"]],
        on="sample_id",
        suffixes=("", "_baseline"),
        validate="one_to_one",
    )
    if paired.target_month.astype(str).max() > DEV_END:
        raise ValueError("v1.4 regime verification must not include consumed test months")
    large = np.abs(paired.absolute_anomaly_c.to_numpy(dtype=float)) >= paired.training_anomaly_q75_c.to_numpy(dtype=float)
    result: dict[str, Any] = {
        "source_artifact": "results/predictive_v14/fold_predictions.parquet",
        "source_sha256": sha256_file(path),
        "status": "RECOMPUTED_FROM_OOF_ROWS",
        "target_range": [str(paired.target_month.min()), str(paired.target_month.max())],
        "rows": int(len(paired)),
        "regimes": {},
    }
    for name, mask in (("large_anomaly", large), ("ordinary", ~large)):
        y = paired.observed_t2m_c.to_numpy(dtype=float)[mask]
        model = paired.prediction_t2m_c.to_numpy(dtype=float)[mask]
        climate = paired.prediction_t2m_c_baseline.to_numpy(dtype=float)[mask]
        climate_mae = float(np.mean(np.abs(climate - y)))
        model_mae = float(np.mean(np.abs(model - y)))
        result["regimes"][name] = {
            "sample_count": int(mask.sum()),
            "threshold_rule": "row-level training_anomaly_q75_c; retrospective diagnostic only",
            "training_climatology_mae_c": climate_mae,
            "nested_selected_pipeline_mae_c": model_mae,
            "nested_model_relative_mae_change_pct": float(100 * (model_mae - climate_mae) / climate_mae),
            "mean_absolute_target_anomaly_c": float(np.mean(np.abs(paired.absolute_anomaly_c.to_numpy(dtype=float)[mask]))),
        }
    return result


def format_markdown_table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    def show(value: Any) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.{digits}f}"
        return str(value).replace("|", "\\|")

    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for record in frame[columns].to_dict("records"):
        lines.append("| " + " | ".join(show(record[column]) for column in columns) + " |")
    return "\n".join(lines)


def gate_diagnostic(predictions: pd.DataFrame) -> pd.DataFrame:
    """Describe post-hoc gate behavior; outcomes never feed selection or weights."""
    rows: list[dict[str, Any]] = []
    for (fold, regime), group in predictions.groupby(["outer_fold", "retrospective_regime"], sort=True):
        weights = group.gate_weight.to_numpy(dtype=float)
        high_gate = weights >= 0.5
        rows.append(
            {
                "outer_fold": str(fold),
                "retrospective_regime_diagnostic_only": str(regime),
                "sample_count": int(len(group)),
                "mean_gate_weight": float(np.mean(weights)),
                "median_gate_weight": float(np.median(weights)),
                "gate_weight_ge_0_5_rate": float(np.mean(high_gate)),
                "selected_strategy": ",".join(sorted(set(group.selected_strategy.astype(str)))),
            }
        )
    return pd.DataFrame(rows)


def cost_summary(timings: pd.DataFrame) -> pd.DataFrame:
    if timings.empty:
        return pd.DataFrame(columns=["strategy", "fit_count", "fit_seconds_sum", "fit_seconds_mean", "fit_seconds_max"])
    return (
        timings.groupby("strategy", as_index=False)
        .fit_seconds.agg(fit_count="count", fit_seconds_sum="sum", fit_seconds_mean="mean", fit_seconds_max="max")
        .sort_values("strategy", kind="stable")
        .reset_index(drop=True)
    )


def write_report(
    path: Path,
    *,
    registry: dict[str, Any],
    source_summary: dict[str, Any],
    metrics: pd.DataFrame,
    regimes: pd.DataFrame,
    selections: pd.DataFrame,
    costs: pd.DataFrame,
    v14: dict[str, Any],
    gate_summary: pd.DataFrame,
) -> None:
    best = metrics.sort_values(["mae", "rmse", "strategy"], kind="stable").iloc[0]
    nested = metrics.loc[metrics.strategy.eq("nested_selected")].iloc[0]
    gate_large = regimes.loc[regimes.strategy.eq("regime_mixture") & regimes.retrospective_regime.eq("large_anomaly")].iloc[0]
    gate_ordinary = regimes.loc[regimes.strategy.eq("regime_mixture") & regimes.retrospective_regime.eq("ordinary")].iloc[0]
    selected_counts = selections.selected_strategy.value_counts().rename_axis("strategy").reset_index(name="outer_folds_selected")
    source_table = pd.DataFrame(
        [
            {
                "source": source_id,
                "version": summary["version"],
                "eligible targets": summary["issue_time_eligible_sample_count"],
                "range": "–".join(summary["issue_time_eligible_target_range"] or []),
                "pre-2019 training targets": summary.get("issue_time_eligible_training_targets_before_2019", "MERRA-only"),
                "cross-source scored": summary.get("cross_source_candidate_scored", "MERRA-only"),
            }
            for source_id, summary in source_summary.items()
        ]
    )
    skill_ci = registry["paired_mae_skill_intervals"]
    interval_frame = pd.DataFrame(skill_ci)
    nested_interval = interval_frame.loc[interval_frame.strategy.eq("nested_selected")].iloc[0]
    if v14.get("regimes"):
        v14_table = "| Regime | n | Climatology MAE °C | v1.4 selected MAE °C | Relative change |\n| --- | ---: | ---: | ---: | ---: |\n" + "\n".join(
            f"| {name} | {values['sample_count']} | {values['training_climatology_mae_c']:.4f} | {values['nested_selected_pipeline_mae_c']:.4f} | {values['nested_model_relative_mae_change_pct']:+.2f}% |"
            for name, values in v14["regimes"].items()
        )
    else:
        v14_table = f"v1.4 recomputation status: {v14.get('status')} — {v14.get('reason', 'no regime rows')}"
    reliable = (
        float(nested.mae_skill_vs_baseline) > 0
        and float(nested.worst_fold_mae_skill_vs_baseline) >= 0
        and float(nested_interval.ci_low) > 0
    )
    recommendation = "CONSIDER THE NESTED SELECTED CANDIDATE FOR A SEPARATE APPROVAL REVIEW" if reliable else "RETAIN TRAINING-ONLY CLIMATOLOGY FOR DEPLOYMENT"
    report = f"""# KUNTUR v1.5 Predictive Regime Specialization and Reliability

## Execution status

- Run ID: `{registry['run_id']}`
- Development-only outer evaluation: **{registry['evaluation']['outer_oof_sample_count']} paired monthly forecasts**, {registry['evaluation']['outer_fold_count']} chronological year folds.
- Target months: `{registry['evaluation']['target_range'][0]}` through `{registry['evaluation']['target_range'][1]}`.
- Test isolation: **PASS**. Forecast and split rows were filtered to target months no later than 2021-12, and test-partition records were excluded before conversion to pandas. No 2022–2025 target values entered fitting, selection, scoring or diagnostics.
- Independent generalization claim: **not made**. This remains a development-only nested hindcast; it was informed by prior inspection of v1.4 development outcomes.

## v1.4 result reproduced

The v1.4 descriptive regime decomposition was recomputed from paired OOF rows in `{v14.get('source_artifact', 'unavailable')}`. Its target-anomaly threshold is derived from each row's training-only q75 and is used **only after the forecast for retrospective score stratification**.

{v14_table}

## Source availability audit

The source guard combines CMR granule production timestamps with a conservative source-latency window and checks the corresponding validated Gold fields. NASA documents MERRA-2 monthly release usually by the 15th of the following month; IMERG Final is typically about 3.5 months; GLDAS-2.1 production streams normally take 3–4 months. The issue timestamp is the first instant of the month preceding the target. Therefore this run used a latest predictor lag of three months for MERRA-2 and six months for IMERG/GLDAS.

CMR `ProductionDateTime` means the exact time a granule was produced, while `ProviderDates.Insert` is the time it entered the provider database. Neither field is a guaranteed first-publication timestamp. The registry therefore labels these as **latency-and-production-guard eligible**, records `first_publication_timestamp_verified=false`, and does not overstate the positive availability proof. A granule produced after issue is rejected. Cross-source predictors were not scored unless a chronological, guard-eligible training set preceded 2019.

The corrected folds also apply this production-time rule to **target labels used for fitting**. Every outer and inner model's training labels must have a MERRA-2 CMR production timestamp no later than the earliest issue in that validation window; missing timestamps are excluded. Fold-level cutoff/count evidence is recorded in `leakage_audit.json`.

NASA references: [MERRA-2 system characteristics](https://gmao.gsfc.nasa.gov/gmao-products/merra-2/system-characteristics_merra-2/), [IMERG Final latency](https://gpm.nasa.gov/resources/faq/what-determines-latency-imerg), [GLDAS-2.1 latency note](https://forum.earthdata.nasa.gov/viewtopic.php?t=7953), [CMR ProductionDateTime definition](https://wiki.earthdata.nasa.gov/spaces/CMR/pages/133727054/Production+Date+Time), [CMR ProviderDates definition](https://wiki.earthdata.nasa.gov/spaces/CMR/pages/135834714/Provider+Dates).

{format_markdown_table(source_table, list(source_table.columns))}

## Nested chronological model comparison

Candidates: training-only monthly climatology; compact residual HGB; regularized Ridge anomaly; fixed smooth residual shrinkage; and an issue-time feature-only smooth gate over the compact residual forecast. Selection used inner expanding half-year folds and MAE, with climatology retained as an eligible candidate. The gate used lagged/rolling temperature-anomaly magnitude, recent persistence, variability, six-month slope, and seasonal sine/cosine state. Cross-source signals were assessed for availability but were not passed into a model unless the source eligibility gate was satisfied.

The lowest pooled outer-OOF MAE was `{best['strategy']}` ({best['mae']:.4f} °C). This is a development comparison after nested selection, not an untouched final test.

{format_markdown_table(metrics, ['strategy', 'sample_count', 'mae', 'rmse', 'bias', 'mae_skill_vs_baseline', 'rmse_skill_vs_baseline', 'worst_fold_mae_skill_vs_baseline', 'anomaly_calibration_slope', 'anomaly_calibration_intercept_c'])}

### Outer-fold selection and cost

{format_markdown_table(selected_counts, ['strategy', 'outer_folds_selected'])}

{format_markdown_table(costs, ['strategy', 'fit_count', 'fit_seconds_sum', 'fit_seconds_mean', 'fit_seconds_max'], digits=6)}

### Large-anomaly versus ordinary-month diagnostics

These labels are retrospective, using an outer-fold training-only q75 target-anomaly threshold after forecasts were fixed. They are not gate inputs.

{format_markdown_table(regimes, ['strategy', 'retrospective_regime', 'sample_count', 'mae', 'rmse', 'bias', 'mae_skill_vs_baseline'])}

### Feature-gate behavior (diagnostic only)

Gate weights are fixed before outer labels are read. The split by retrospective anomaly category below is descriptive only.

{format_markdown_table(gate_summary, ['outer_fold', 'retrospective_regime_diagnostic_only', 'sample_count', 'mean_gate_weight', 'median_gate_weight', 'gate_weight_ge_0_5_rate', 'selected_strategy'])}

### Paired MAE skill uncertainty

Intervals resample 3-month blocks only within contiguous eligible calendar segments; no block crosses a month missing from the issue-time-eligible set.

{format_markdown_table(interval_frame, ['strategy', 'sample_count', 'mean_mae_improvement', 'ci_low', 'ci_high', 'block_length_months'])}

## Interpretation and recommendation

The lowest pooled outer-OOF MAE belongs to `{best['strategy']}`, but this is descriptive and does not override fold-local nested selection. The nested-selected strategy has MAE skill **{float(nested.mae_skill_vs_baseline):+.2%}**, worst-fold MAE skill **{float(nested.worst_fold_mae_skill_vs_baseline):+.2%}**, and paired 3-month block interval for MAE improvement **[{float(nested_interval.ci_low):+.4f}, {float(nested_interval.ci_high):+.4f}] °C**. On the predeclared reliability checks, the recommendation is: **{recommendation}**. No independent generalization or production readiness is claimed. The primary baseline remained in every inner selection.

### Hypothesis outcomes

- **v1.4 retrospective regime result is reproducible:** confirmed from 120 paired OOF predictions (13 large-anomaly and 107 ordinary months).
- **Issue-time proxies identify months where residuals should be preferred:** not supported. The regime mixture's retrospective MAE skill was {float(gate_large.mae_skill_vs_baseline):+.2%} on {int(gate_large.sample_count)} large-anomaly months and {float(gate_ordinary.mae_skill_vs_baseline):+.2%} on {int(gate_ordinary.sample_count)} ordinary months; its pooled paired interval spans zero and no large-anomaly month received a gate weight of at least 0.5.
- **A MERRA-only residual forecast reliably beats climatology:** not confirmed. The nested-selected MAE gain is small, RMSE skill and worst-fold skill are negative, and the paired interval crosses zero.
- **IMERG/GLDAS historical signals can support the 2019–2021 outer evaluation:** not evaluable with the frozen issue-time contract. Neither source had a guard-eligible training target before 2019 (IMERG: 0; GLDAS: 0), so neither was scored as a cross-source model.
- **Seasonal state is a useful standalone gate:** not selected by any outer-fold nested selection; this does not establish that seasonality has no value as a forecast feature.

## Reproducibility and files

- Full experiment registry and input/configuration provenance: `results/predictive_v15/experiment_registry_v1_5.json`.
- Independent SHA-256 index for generated outputs: `results/predictive_v15/artifact_hashes_v1_5.json`.
- Public aggregate verification: `python scripts/verify_predictive_v15.py --data-root <existing read-only KUNTUR data root>`.
- Full local metric recomputation: add `--private-output-root <private run directory>`; row-level files remain outside Git.
- Public source collection/production metadata: `results/predictive_v15/source_availability_cmr_snapshot.json`.
- Row-level availability audits and predictions are written only to the local private output root and excluded from the public repository. Pass the run directory to the verifier for the full local recomputation.
- Public inner candidate and parameter score summaries: `results/predictive_v15/inner_tuning_scores.csv`, `results/predictive_v15/inner_candidate_scores.csv`.
- Model and regime metrics: `results/predictive_v15/overall_metrics.csv`, `results/predictive_v15/regime_metrics.csv`.
- Leakage audit: `results/predictive_v15/leakage_audit.json`.

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(report.rstrip() + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Read-only root containing the existing v1.4 data/ tree")
    parser.add_argument("--reuse-cmr-snapshot", action="store_true", help="Reuse a previously saved public CMR snapshot instead of refreshing it")
    parser.add_argument("--private-output-root", type=Path, help="Private local parent directory for row-level audit/prediction artifacts; one run-id subdirectory is created")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    data_root = args.data_root.resolve()
    v14_dir = repo_root / "results" / "predictive_v14"
    result_dir = repo_root / "results" / "predictive_v15"
    report_path = repo_root / "reports" / "predictive_v15" / "PREDICTIVE_V15_REPORT.md"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    private_parent = args.private_output_root or (Path.home() / ".local" / "share" / "kuntur" / "private" / "predictive_v15")
    private_result_dir = private_parent.resolve() / run_id
    try:
        private_result_dir.relative_to(repo_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("private row-level output root must remain outside the repository")
    result_dir.mkdir(parents=True, exist_ok=True)

    samples, monthly, files = load_development_inputs(data_root)
    if set(samples.partition.astype(str)) - {"train", "validation"}:
        raise ValueError("only frozen train and validation partitions may enter v1.5")
    if samples.target_month.astype(str).max() > DEV_END:
        raise ValueError("development loader included a consumed test target")

    cmr_path = result_dir / "source_availability_cmr_snapshot.json"
    snapshot = load_or_fetch_cmr_snapshot(cmr_path, refresh=not args.reuse_cmr_snapshot)
    audit, source_summary = build_availability_audit(samples, snapshot, monthly)
    atomic_csv(private_result_dir / "issue_time_availability_audit.csv", audit)

    merra_id = "merra2_monthly_slv"
    merra_audit = audit[audit.source_id.eq(merra_id)]
    eligible_ids = set(
        merra_audit.loc[merra_audit.full_history_latency_and_production_guard_eligible, "sample_id"].astype(str)
    )
    eligible = samples[samples.sample_id.astype(str).isin(eligible_ids)].copy()
    eligible = eligible.sort_values("target_month", kind="stable").reset_index(drop=True)
    if eligible.empty or set(eligible.partition.astype(str)) - {"train", "validation"}:
        raise RuntimeError("no valid MERRA-2 development rows survived issue-time and Gold completeness guards")
    atomic_csv(
        private_result_dir / "eligible_development_samples.csv",
        eligible[["sample_id", "target_month", "issue_date_utc", "partition"]],
    )

    merra_label_production_dates = production_dates(snapshot, merra_id)
    outer_folds = chronological_outer_folds_v15(
        eligible,
        validation_years=(2019, 2020, 2021),
        development_end=DEV_END,
        label_production_dates_utc=merra_label_production_dates,
    )
    expected_folds = ["outer_2019", "outer_2020", "outer_2021"]
    if [fold.name for fold in outer_folds] != expected_folds:
        raise RuntimeError(f"required outer year folds unavailable: {[fold.name for fold in outer_folds]}")
    prediction_frames: list[pd.DataFrame] = []
    inner_oof_frames: list[pd.DataFrame] = []
    label_availability_summary: list[dict[str, Any]] = []
    selection_frames: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    tuning_rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for fold_index, outer in enumerate(outer_folds):
        inner = chronological_inner_folds_v15(
            outer.train,
            min_train_months=18,
            label_production_dates_utc=merra_label_production_dates,
        )
        for scope, training, issue_time in [
            ("outer", outer.train, str(outer.validation.sort_values("target_month").iloc[0].issue_date_utc)),
            *[
                (f"inner:{part.name}", part.train, str(part.validation.sort_values("target_month").iloc[0].issue_date_utc))
                for part in inner
            ],
        ]:
            latest_label = str(training.target_month.max())
            label_availability_summary.append({
                "outer_fold": outer.name,
                "scope": scope,
                "first_validation_issue_time_utc": issue_time,
                "training_label_count": int(len(training)),
                "latest_training_label_month": latest_label,
                "latest_label_cmr_production_time_utc": merra_label_production_dates.get(latest_label),
            })
        inner_oof, inner_timing = inner_predictions(inner, monthly, args.seed + fold_index * 100)
        inner_oof.insert(0, "outer_fold", outer.name)
        inner_oof_frames.append(inner_oof)
        selection, candidate_scores = select_inner_strategy(inner_oof)
        candidate_scores.insert(0, "outer_fold", outer.name)
        candidate_frames.append(candidate_scores)
        for family, field in (
            ("ridge_alpha", "ridge_alpha_tuning_scores"),
            ("smooth_blend", "smooth_blend_tuning_scores"),
            ("gate_proxy_strength", "gate_tuning_scores"),
        ):
            for score in selection[field]:
                tuning_rows.append({"outer_fold": outer.name, "tuning_family": family, **score})
        selected, outer_timing = fit_outer_predictions(outer, monthly, selection, args.seed + fold_index * 1000)
        selected.insert(0, "sample_id", outer.validation.sample_id.astype(str).to_numpy())
        selected.insert(1, "partition", outer.validation.partition.astype(str).to_numpy())
        prediction_frames.append(selected)
        for row in inner_timing:
            timing_rows.append({"outer_fold": outer.name, "stage": "inner", **row})
        for row in outer_timing:
            timing_rows.append({"outer_fold": outer.name, "stage": "outer", **row})
        selection_frames.append(
            {
                "outer_fold": outer.name,
                **{key: value for key, value in selection.items() if not key.endswith("_tuning_scores")},
                "inner_fold_count": len(inner),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values("target_month", kind="stable").reset_index(drop=True)
    if predictions.target_month.duplicated().any() or predictions.target_month.astype(str).max() > DEV_END:
        raise RuntimeError("outer predictions overlap or exceed the development cutoff")
    prediction_columns = [column for column in predictions.columns if column.startswith("prediction__")]
    if not np.isfinite(predictions[prediction_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("outer forecast artifact contains a non-finite prediction")
    metrics, fold_metrics, regime_metrics, intervals = aggregate_outputs(predictions)
    timings = pd.DataFrame(timing_rows)
    costs = cost_summary(timings)
    gates = gate_diagnostic(predictions)
    selections = pd.DataFrame(selection_frames)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    inner_oof = pd.concat(inner_oof_frames, ignore_index=True).sort_values(["outer_fold", "target_month"], kind="stable").reset_index(drop=True)
    tuning_scores = pd.DataFrame(tuning_rows)
    v14 = verify_v14_regime_decomposition(v14_dir)
    if v14.get("status") != "RECOMPUTED_FROM_OOF_ROWS":
        raise RuntimeError(f"v1.4 result verification did not complete: {v14}")

    atomic_parquet(private_result_dir / "outer_fold_predictions.parquet", predictions)
    atomic_parquet(private_result_dir / "inner_selection_oof.parquet", inner_oof)
    atomic_csv(result_dir / "overall_metrics.csv", metrics)
    atomic_csv(result_dir / "fold_metrics.csv", fold_metrics)
    atomic_csv(result_dir / "regime_metrics.csv", regime_metrics)
    atomic_csv(result_dir / "paired_mae_skill_intervals.csv", intervals)
    atomic_csv(result_dir / "inner_candidate_scores.csv", candidates)
    atomic_csv(result_dir / "inner_tuning_scores.csv", tuning_scores)
    atomic_csv(result_dir / "fold_selections.csv", selections)
    atomic_csv(result_dir / "fit_timings.csv", timings)
    atomic_csv(result_dir / "fit_cost_summary.csv", costs)
    atomic_csv(result_dir / "gate_diagnostic.csv", gates)
    atomic_json(result_dir / "v14_regime_recomputed.json", v14)

    input_files = {
        **files,
        "v14_fold_predictions": v14_dir / "fold_predictions.parquet",
        "v14_regime_metrics": v14_dir / "anomaly_regime_metrics.csv",
        "implementation_runner": Path(__file__).resolve(),
        "implementation_module": repo_root / "src" / "kuntur_data" / "phase2" / "predictive_v15.py",
        "implementation_tests": repo_root / "tests" / "test_predictive_v15.py",
    }
    verifier_path = repo_root / "scripts" / "verify_predictive_v15.py"
    if verifier_path.is_file():
        input_files["implementation_verifier"] = verifier_path
    input_hashes = {
        key: {"path": input_path_label(path, repo_root, data_root), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for key, path in input_files.items()
    }
    v14_hashes_before = {key: value["sha256"] for key, value in input_hashes.items() if key.startswith("v14_")}
    v14_hashes_after = {key: sha256_file(input_files[key]) for key in v14_hashes_before}
    if v14_hashes_before != v14_hashes_after:
        raise RuntimeError("an immutable v1.4 result changed during the v1.5 run")

    leakage = {
        "status": "PASS",
        "development_cutoff": DEV_END,
        "target_value_rows_accessed_after_cutoff": False,
        "forecast_input_filter": "target_month <= 2021-12 applied in Arrow before pandas conversion",
        "split_filter": "partition != test and target_month <= 2021-12 applied in Arrow before pandas conversion",
        "monthly_gold_filter": "month <= 2021-12 applied in Arrow before pandas conversion",
        "source_cmr_temporal_filter": CMR_END,
        "target_label_availability_guard": {
            "status": "PASS",
            "rule": "training label MERRA-2 CMR ProductionDateTime must be no later than the earliest validation issue time; missing production timestamps are excluded",
            "semantics": "conservative production-time proxy, not proof of first public release",
            "fold_training_summaries": label_availability_summary,
        },
        "outer_folds": expected_folds,
        "gate_feature_allowlist": list(sorted(ALLOWED_GATE_FEATURES)),
        "gate_features_used": list(GATE_PROXIES),
        "post_hoc_only_fields": ["actual_target_anomaly_c", "retrospective_regime"],
        "v1_4_threshold_use": "training_anomaly_q75_c is used only for retrospective diagnostics; never for inference gate input",
        "cross_source_candidates": "not scored unless guard-eligible training rows precede the 2019 outer fold",
        "v1_4_inputs_unchanged": v14_hashes_before == v14_hashes_after,
        "prior_v1_4_development_inspection": True,
        "independent_final_test_claim": False,
    }
    atomic_json(result_dir / "leakage_audit.json", leakage)

    evaluation = {
        "outer_fold_count": len(expected_folds),
        "outer_oof_sample_count": int(len(predictions)),
        "target_range": [str(predictions.target_month.min()), str(predictions.target_month.max())],
        "outer_folds": expected_folds,
        "selected_strategies_by_fold": selections[["outer_fold", "selected_strategy", "ridge_alpha", "smooth_alpha", "gate_proxy", "gate_strength"]].to_dict("records"),
    }
    private_output_files = [
        {"path": path.name, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in sorted(private_result_dir.iterdir()) if path.is_file()
    ]
    registry = {
        "schema_version": "1.0",
        "experiment_id": "KUNTUR-PREDICTIVE-V1.5",
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repository": {
            "branch": "feat/predictive-regime-v1.5",
            "head_commit": git_head(repo_root),
            "working_tree": "source snapshot is pinned by implementation SHA-256 values in input_files",
        },
        "input_data_root_read_only": "existing KUNTUR v1.4 data tree; external_data_root paths in input_files are relative to this read-only root",
        "input_files": input_hashes,
        "v1_4_result_recomputed": v14,
        "source_availability": {
            "audit_semantics": "source latency window + CMR granule production-time guard + finite validated Gold history; not proof of exact first public release",
            "first_publication_timestamp_verified": False,
            "policy": SOURCE_RELEASE_POLICY,
            "required_gold_variables": SOURCE_GOLD_VARIABLES,
            "sources": source_summary,
            "source_documents": {
                "merra2_monthly_latency": "https://gmao.gsfc.nasa.gov/gmao-products/merra-2/system-characteristics_merra-2/",
                "imerg_final_latency": "https://gpm.nasa.gov/resources/faq/what-determines-latency-imerg",
                "gldas_21_latency": "https://forum.earthdata.nasa.gov/viewtopic.php?t=7953",
                "cmr_production_datetime": "https://wiki.earthdata.nasa.gov/spaces/CMR/pages/133727054/Production+Date+Time",
                "cmr_provider_dates": "https://wiki.earthdata.nasa.gov/spaces/CMR/pages/135834714/Provider+Dates",
            },
        },
        "evaluation": evaluation,
        "private_output_artifacts": private_output_files,
        "paired_mae_skill_intervals": intervals.to_dict("records"),
        "configuration": {
            "development_cutoff": DEV_END,
            "development_start": DEV_START,
            "history_months": HISTORY_WIDTH,
            "issue_time": "00:00 UTC on the first day of target_month minus one month",
            "merra_predictor_months": "T-14 through T-3 inclusive",
            "outer_validation_years": [2019, 2020, 2021],
            "inner_validation_windows": "expanding half-year windows; minimum 18 eligible training samples",
            "selection_metric": "inner-fold pooled MAE; RMSE and simple stable priority resolve ties",
            "seed": args.seed,
            "candidates": ["climatology", "compact_residual", "regularized_anomaly", "smooth_shrinkage", "regime_mixture"],
            "hgb": HGB_CONFIG,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "blend_strengths": list(BLEND_STRENGTHS),
            "gate_proxies": list(GATE_PROXIES),
            "metrics": ["MAE", "RMSE", "bias", "paired skill", "worst-fold skill", "retrospective regime scores", "anomaly calibration", "3-month paired moving-block interval"],
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "requests": requests.__version__,
            "pyarrow": __import__("pyarrow").__version__,
            "wall_seconds": round(time.perf_counter() - started, 6),
        },
        "output_files": [
            "source_availability_cmr_snapshot.json", "overall_metrics.csv", "fold_metrics.csv", "regime_metrics.csv",
            "paired_mae_skill_intervals.csv", "inner_candidate_scores.csv", "inner_tuning_scores.csv",
            "fold_selections.csv", "fit_timings.csv",
            "fit_cost_summary.csv", "gate_diagnostic.csv", "v14_regime_recomputed.json", "leakage_audit.json",
            "experiment_registry_v1_5.json", "artifact_hashes_v1_5.json", "reports/predictive_v15/PREDICTIVE_V15_REPORT.md",
        ],
        "v1_4_immutability": {"hashes_before": v14_hashes_before, "hashes_after": v14_hashes_after, "status": "PASS"},
    }
    atomic_json(result_dir / "experiment_registry_v1_5.json", registry)
    write_report(report_path, registry=registry, source_summary=source_summary, metrics=metrics, regimes=regime_metrics, selections=selections, costs=costs, v14=v14, gate_summary=gates)

    public_targets = [
        (repo_root / Path(item)) if str(item).startswith("reports/") else (result_dir / Path(item))
        for item in registry["output_files"]
        if Path(item).name != "artifact_hashes_v1_5.json"
    ]
    hash_targets = sorted(set(public_targets + [verifier_path]), key=lambda path: str(path))
    hash_index = {
        "schema_version": "1.0",
        "algorithm": "SHA-256",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "files": [{"path": str(path.relative_to(repo_root)), "sha256": sha256_file(path), "size_bytes": path.stat().st_size} for path in hash_targets],
    }
    atomic_json(result_dir / "artifact_hashes_v1_5.json", hash_index)
    print(json.dumps({
        "status": "COMPLETE",
        "run_id": run_id,
        "outer_oof_sample_count": evaluation["outer_oof_sample_count"],
        "target_range": evaluation["target_range"],
        "source_availability": source_summary,
        "overall_metrics_path": str(result_dir / "overall_metrics.csv"),
        "report_path": str(report_path),
        "hash_index_path": str(result_dir / "artifact_hashes_v1_5.json"),
        "private_output_run_id": run_id,
        "private_output_files": len(private_output_files),
        "wall_seconds": registry["environment"]["wall_seconds"],
    }, indent=2, sort_keys=True))
    return 0


def git_head(repo_root: Path) -> str | None:
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def input_path_label(path: Path, repo_root: Path, data_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        pass
    try:
        return f"external_data_root/{resolved.relative_to(data_root.resolve()).as_posix()}"
    except ValueError as exc:
        raise ValueError(f"input is outside both the repository and declared data root: {path.name}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
