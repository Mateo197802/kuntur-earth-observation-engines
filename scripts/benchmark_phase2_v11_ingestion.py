#!/usr/bin/env python3
"""Compare direct, coarse, legacy fine partitions, and adaptive cache reads."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import psutil

from kuntur_data.phase2.ingestion import AdaptiveGoldReader, GlobalAnalyticalCache, PartitionedGoldStore, sha256_file


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
FINE_ROOT = ROOT / "data/derived/phase2/partitioned_gold_v1"
COARSE_ROOT = ROOT / "data/derived/phase2/partitioned_gold_v1_1_coarse"
CACHE_ROOT = ROOT / "data/cache/phase2_v1_1_adaptive"
OUT = ROOT / "results/phase2/v1_1/ingestion"
VARIABLES = ["merra2_t2m_c", "imerg_precip_monthly_total", "gldas_soil_moisture_0_10cm", "modis_ndvi"]
COLUMNS = ["dataset_id", "variable_id", "month", "region_id", "value", "units", "valid_area_fraction_bbox", "source_sha256"]
REPEATS = 30
DIRECT_MAX_BYTES = 5 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def timing(fn, repeats=REPEATS):
    values = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        values.append(time.perf_counter() - start)
    return result, {"n": len(values), "median_seconds": float(np.median(values)), "p95_seconds": float(np.quantile(values, .95)), "mean_seconds": float(np.mean(values))}


def direct_query(case: dict) -> pd.DataFrame:
    filters = [
        ("variable_id", "in", case["variables"]), ("region_id", "=", "ecuador_pilot_bbox"),
        ("month", ">=", case["start"]), ("month", "<=", case["end"]),
    ]
    frame = pd.read_parquet(GOLD, columns=COLUMNS, filters=filters)
    return frame.sort_values(["dataset_id", "variable_id", "month"], kind="stable").reset_index(drop=True)


def build_coarse_partitions(gold: pd.DataFrame, source_hash: str) -> dict:
    manifest_path = COARSE_ROOT / "partition_manifest.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("source_gold_sha256") == source_hash and prior.get("status") == "VALIDATED":
            return prior
        raise RuntimeError("Coarse partition path exists with a different source; refusing overwrite")
    records = []
    count = rows = 0
    for (dataset, variable, region), group in gold.groupby(["dataset_id", "variable_id", "region_id"], sort=True, dropna=False):
        parts = [f"dataset_id={quote(str(dataset), safe='')}", f"variable_id={quote(str(variable), safe='')}", f"region_id={quote(str(region), safe='')}"]
        out_dir = COARSE_ROOT.joinpath(*parts)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "part.parquet"
        temp = target.with_suffix(".tmp.parquet")
        group.to_parquet(temp, index=False, compression="zstd")
        os.replace(temp, target)
        records.append({"path": target.relative_to(ROOT).as_posix(), "rows": len(group), "bytes": target.stat().st_size, "sha256": sha256(target)})
        rows += len(group)
        count += 1
    manifest = {"status": "VALIDATED", "partition_scheme": ["dataset_id", "variable_id", "region_id"], "partition_count": count, "row_count": rows, "source_gold_sha256": source_hash, "partitions": records}
    atomic_json(manifest_path, manifest)
    return manifest


def coarse_query(case: dict) -> pd.DataFrame:
    frames = []
    for path in COARSE_ROOT.rglob("part.parquet"):
        rel = path.relative_to(COARSE_ROOT).parts
        variable = next((x.split("=", 1)[1] for x in rel if x.startswith("variable_id=")), None)
        region = next((x.split("=", 1)[1] for x in rel if x.startswith("region_id=")), None)
        if variable not in case["variables"] or region != "ecuador_pilot_bbox":
            continue
        frame = pd.read_parquet(path, columns=COLUMNS)
        frame = frame[(frame.month >= case["start"]) & (frame.month <= case["end"])]
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)
    return frame.sort_values(["dataset_id", "variable_id", "month"], kind="stable").reset_index(drop=True)


def fine_match_count(case: dict) -> int:
    first_year, last_year = int(case["start"][:4]), int(case["end"][:4])
    paths = set()
    for variable in case["variables"]:
        for year in range(first_year, last_year + 1):
            paths.update(FINE_ROOT.glob(f"dataset_id=*/variable_id={variable}/year={year}/region_id=ecuador_pilot_bbox/contract_version=1.0.0/part.parquet"))
    return len(paths)


def main() -> None:
    started = time.perf_counter()
    if not GOLD.is_file():
        raise SystemExit("Frozen Gold input missing")
    gold_hash = sha256(GOLD)
    gold = pd.read_parquet(GOLD)
    if len(gold) != 5520:
        raise SystemExit("Gold row count changed")
    if not (FINE_ROOT / "partition_manifest.json").is_file():
        raise SystemExit("Existing v1 fine partition manifest missing")
    fine_manifest = json.loads((FINE_ROOT / "partition_manifest.json").read_text(encoding="utf-8"))
    if fine_manifest.get("source_gold_sha256") != gold_hash or fine_manifest.get("partition_count") != 460:
        raise SystemExit("Existing fine partitions do not match frozen Gold")
    coarse_manifest = build_coarse_partitions(gold, gold_hash)
    fine = PartitionedGoldStore(GOLD, FINE_ROOT, contract_version="1.0.0")
    adaptive_cache = GlobalAnalyticalCache(CACHE_ROOT, max_memory_bytes=64 * 1024 * 1024, max_disk_bytes=64 * 1024 * 1024, ttl_seconds=7 * 86400)
    adaptive = AdaptiveGoldReader(GOLD, contract_version="1.1.0", direct_scan_max_bytes=DIRECT_MAX_BYTES, cache=adaptive_cache)
    cases = [
        {"name": "single_variable_short_3y", "variables": ["merra2_t2m_c"], "start": "2022-01", "end": "2024-12"},
        {"name": "multi_variable_short_3y", "variables": VARIABLES, "start": "2022-01", "end": "2024-12"},
        {"name": "single_variable_full_23y", "variables": ["merra2_t2m_c"], "start": "2002-01", "end": "2024-12"},
        {"name": "multi_variable_full_23y", "variables": VARIABLES, "start": "2002-01", "end": "2024-12"},
    ]
    rows = []
    peak_rss = psutil.Process().memory_info().rss
    for case in cases:
        expected = direct_query(case)
        measurements = {}
        for name, fn in {
            "direct_scan": lambda: direct_query(case),
            "fine_460_partition_query": lambda: fine.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=COLUMNS),
            "coarse_partition_query": lambda: coarse_query(case),
        }.items():
            got, timing_row = timing(fn)
            pd.testing.assert_frame_equal(got, expected, check_exact=True)
            measurements[name] = timing_row
        adaptive_cache.invalidate_all()
        cold_got, cold = timing(lambda: adaptive.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=COLUMNS), repeats=1)
        pd.testing.assert_frame_equal(cold_got, expected, check_exact=True)
        warm_got, warm = timing(lambda: adaptive.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=COLUMNS))
        pd.testing.assert_frame_equal(warm_got, expected, check_exact=True)
        warm_memory_hits = adaptive_cache.stats()["memory_hits"]
        disk_cache = GlobalAnalyticalCache(CACHE_ROOT, max_memory_bytes=0, max_disk_bytes=64 * 1024 * 1024, ttl_seconds=7 * 86400)
        disk_reader = AdaptiveGoldReader(GOLD, contract_version="1.1.0", direct_scan_max_bytes=DIRECT_MAX_BYTES, cache=disk_cache)
        _, disk_cold = timing(lambda: disk_reader.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=COLUMNS), repeats=1)
        disk_reader2 = AdaptiveGoldReader(GOLD, contract_version="1.1.0", direct_scan_max_bytes=DIRECT_MAX_BYTES, cache=GlobalAnalyticalCache(CACHE_ROOT, max_memory_bytes=0, max_disk_bytes=64 * 1024 * 1024, ttl_seconds=7 * 86400))
        disk_warm_frame, disk_warm = timing(lambda: disk_reader2.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=COLUMNS), repeats=1)
        pd.testing.assert_frame_equal(disk_warm_frame, expected, check_exact=True)
        measurements.update({"adaptive_cold": cold, "adaptive_warm_memory": warm, "adaptive_warm_persistent_disk": disk_warm, "persistent_disk_miss_then_write": disk_cold})
        rows.append({"workload": case["name"], "result_rows": len(expected), "source_gold_bytes": GOLD.stat().st_size, "fine_partition_files": fine_manifest["partition_count"], "coarse_partition_files": coarse_manifest["partition_count"], "matching_fine_files": fine_match_count(case), "warm_memory_cache_hits": warm_memory_hits, "measurements": measurements, "exact_numeric_equivalence": True, "adaptive_strategy": "direct_parquet_scan" if GOLD.stat().st_size <= DIRECT_MAX_BYTES else "validated_partition_store"})
        peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
        print(json.dumps({"event": "INGESTION_CASE_COMPLETE", "workload": case["name"], "rows": len(expected), "elapsed_seconds": round(time.perf_counter() - started, 3)}, allow_nan=False), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "ingestion_benchmark_summary_v1_1.csv"
    manifest_path = OUT / "ingestion_benchmark_manifest_v1_1.json"
    flat_rows = []
    for row in rows:
        for method, stats in row["measurements"].items():
            flat_rows.append({"workload": row["workload"], "method": method, "result_rows": row["result_rows"], "median_seconds": stats["median_seconds"], "p95_seconds": stats["p95_seconds"], "n": stats["n"], "source_gold_bytes": row["source_gold_bytes"], "fine_partition_files": row["fine_partition_files"], "coarse_partition_files": row["coarse_partition_files"], "exact_numeric_equivalence": row["exact_numeric_equivalence"]})
    pd.DataFrame(flat_rows).to_csv(csv_path, index=False)
    manifest = {
        "version": "kuntur-ingestion-cache-v1.1.0",
        "status": "COMPLETED_LOCAL_BENCHMARK",
        "source_gold": "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
        "source_gold_sha256": gold_hash,
        "source_gold_bytes": GOLD.stat().st_size,
        "source_rows": len(gold),
        "existing_fine_partitions": {"path": "data/derived/phase2/partitioned_gold_v1", "partition_count": fine_manifest["partition_count"], "status": fine_manifest["status"]},
        "new_coarse_partitions": {"path": str(COARSE_ROOT.relative_to(ROOT)), "scheme": coarse_manifest["partition_scheme"], "partition_count": coarse_manifest["partition_count"], "manifest_sha256": sha256(COARSE_ROOT / "partition_manifest.json")},
        "adaptive_direct_scan_threshold_bytes": DIRECT_MAX_BYTES,
        "repeats_per_method": REPEATS,
        "workloads": rows,
        "decision": "For the current 332 KB/5,520-row Gold file, use direct Parquet scans for one-off queries. Cache only known repeated queries in memory; persistent disk caching is approximately equal to direct-read latency here. Retain an optional validated partition fallback for larger future Gold releases. Do not materialize 460 tiny year partitions for this dataset size.",
        "cache_policy": "Content-addressed key includes source SHA, contract, region, time, variables, columns, QA policy and aggregation method; this benchmark uses a fresh v1.1 cache root.",
        "benchmark_code_sha256": sha256(Path(__file__).resolve()),
        "warm_cache_hits": {row["workload"]: row["warm_memory_cache_hits"] for row in rows},
        "peak_rss_bytes": peak_rss,
        "software": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "artifacts": {str(csv_path.relative_to(ROOT)): {"sha256": sha256(csv_path), "bytes": csv_path.stat().st_size}},
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(json_safe({"status": manifest["status"], "source_gold_bytes": manifest["source_gold_bytes"], "fine_partitions": fine_manifest["partition_count"], "coarse_partitions": coarse_manifest["partition_count"], "peak_rss_bytes": peak_rss, "manifest": str(manifest_path), "elapsed_seconds": manifest["elapsed_seconds"]}), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
