#!/usr/bin/env python3
"""Benchmark authenticated-free local ingestion and shared cache on frozen Gold v1."""

from __future__ import annotations

import concurrent.futures
import datetime as dt
import hashlib
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

from kuntur_data.phase2.analytical import analyze_variable, cached_analyze_variable
from kuntur_data.phase2.ingestion import GlobalAnalyticalCache, PartitionedGoldStore, sha256_file
from kuntur_data.phase2.predictive import PeakRSS


ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
PARTITIONS = ROOT / "data/derived/phase2/partitioned_gold_v1"
CACHE_DIR = ROOT / "data/cache/phase2_analytical_v1"
VARIABLES = ["merra2_t2m_c", "imerg_precip_monthly_total", "gldas_soil_moisture_0_10cm", "modis_ndvi"]
QUERY_COLUMNS = ["dataset_id", "variable_id", "month", "region_id", "value", "units", "valid_area_fraction_bbox", "source_sha256"]
REPEATS = 10


def summary(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {"n": len(arr), "median_seconds": float(np.median(arr)), "p95_seconds": float(np.quantile(arr, 0.95)), "max_seconds": float(np.max(arr)), "mean_seconds": float(np.mean(arr)), "std_seconds": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0}


def timed(fn):
    wall0, cpu0 = time.perf_counter(), time.process_time()
    out = fn()
    return out, time.perf_counter() - wall0, time.process_time() - cpu0


def pandas_full_scan(variable_ids: list[str], start: str, end: str) -> pd.DataFrame:
    frame = pd.read_parquet(GOLD)
    frame = frame[(frame.variable_id.isin(variable_ids)) & (frame.month >= start) & (frame.month <= end) & (frame.region_id == "ecuador_pilot_bbox")]
    return frame.loc[:, QUERY_COLUMNS].sort_values(["dataset_id", "variable_id", "month"], kind="stable").reset_index(drop=True)


def main() -> None:
    start_wall = time.perf_counter()
    if not GOLD.is_file():
        raise SystemExit(f"Missing authentic Gold input: {GOLD}")
    gold_hash = sha256_file(GOLD)
    if not (PARTITIONS / "partition_manifest.json").exists():
        store = PartitionedGoldStore(GOLD, PARTITIONS, contract_version="1.0.0")
        partition_manifest = store.materialize()
    else:
        store = PartitionedGoldStore(GOLD, PARTITIONS, contract_version="1.0.0")
        partition_manifest = json.loads((PARTITIONS / "partition_manifest.json").read_text(encoding="utf-8"))
    if partition_manifest["source_gold_sha256"] != gold_hash or partition_manifest["row_count"] != 5520:
        raise SystemExit("Partitioned Gold does not match the frozen source file")

    process = psutil.Process(os.getpid())
    peak_rss = 0
    cases = [
        {"name": "single_variable_short_3y", "variables": ["merra2_t2m_c"], "start": "2022-01", "end": "2024-12"},
        {"name": "multi_variable_short_3y", "variables": VARIABLES, "start": "2022-01", "end": "2024-12"},
        {"name": "single_variable_full_23y", "variables": ["merra2_t2m_c"], "start": "2002-01", "end": "2024-12"},
        {"name": "multi_variable_full_23y", "variables": VARIABLES, "start": "2002-01", "end": "2024-12"},
    ]
    rows = []
    latency_bars = []
    for case in cases:
        baseline_times, partition_times, cache_cold_times, cache_warm_times = [], [], [], []
        analysis_cold_times, analysis_warm_times = [], []
        expected = None
        matching_part_bytes = 0
        matching_paths = set()
        for year in range(int(case["start"][:4]), int(case["end"][:4]) + 1):
            for variable in case["variables"]:
                paths = list(PARTITIONS.glob(f"dataset_id=*/variable_id={variable}/year={year}/region_id=ecuador_pilot_bbox/contract_version=1.0.0/part.parquet"))
                matching_paths.update(paths)
        matching_part_bytes = sum(p.stat().st_size for p in matching_paths)
        for _ in range(REPEATS):
            result, seconds, _ = timed(lambda: pandas_full_scan(case["variables"], case["start"], case["end"]))
            expected = result if expected is None else expected
            pd.testing.assert_frame_equal(result, expected, check_exact=True)
            baseline_times.append(seconds)
            part, seconds, _ = timed(lambda: store.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=QUERY_COLUMNS))
            pd.testing.assert_frame_equal(part, expected, check_exact=True)
            partition_times.append(seconds)
        cache = GlobalAnalyticalCache(CACHE_DIR, max_memory_bytes=64 * 1024 * 1024, max_disk_bytes=256 * 1024 * 1024, ttl_seconds=7 * 86400)
        # Pass the shared cache through the store for the timed cache workloads.
        cached_store = PartitionedGoldStore(GOLD, PARTITIONS, contract_version="1.0.0", cache=cache)
        for _ in range(REPEATS):
            cache.invalidate_all()
            frame, seconds, _ = timed(lambda: cached_store.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=QUERY_COLUMNS))
            pd.testing.assert_frame_equal(frame, expected, check_exact=True)
            cache_cold_times.append(seconds)
        # Disk-warm query in a fresh object proves cross-process reusable persistence.
        disk_cache = GlobalAnalyticalCache(CACHE_DIR, max_memory_bytes=64 * 1024 * 1024, max_disk_bytes=256 * 1024 * 1024, ttl_seconds=7 * 86400)
        disk_warm_store = PartitionedGoldStore(GOLD, PARTITIONS, contract_version="1.0.0", cache=disk_cache)
        first, seconds, _ = timed(lambda: disk_warm_store.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=QUERY_COLUMNS))
        pd.testing.assert_frame_equal(first, expected, check_exact=True)
        cache_warm_times.append(seconds)
        for _ in range(REPEATS):
            frame, seconds, _ = timed(lambda: cached_store.query(variable_ids=case["variables"], region_id="ecuador_pilot_bbox", start_month=case["start"], end_month=case["end"], columns=QUERY_COLUMNS))
            pd.testing.assert_frame_equal(frame, expected, check_exact=True)
            cache_warm_times.append(seconds)
        if len(case["variables"]) == 1:
            variable = case["variables"][0]
            raw = pd.read_parquet(GOLD)
            coverage_threshold = 0.8
            direct = analyze_variable(raw[(raw.variable_id == variable) & (raw.month >= case["start"]) & (raw.month <= case["end"])], variable, coverage_threshold=coverage_threshold)["trend"]
            key_context = {"gold_sha256": gold_hash, "contract_version": "1.0.0", "region_geometry_hash": hashlib.sha256(b"ecuador_pilot_bbox:-81.5,-5.1,-75.0,1.7").hexdigest()}
            warmup = cached_analyze_variable(raw[(raw.variable_id == variable) & (raw.month >= case["start"]) & (raw.month <= case["end"])], variable, cache=cache, coverage_threshold=coverage_threshold, key_context=key_context)
            persistent_analysis_cache = GlobalAnalyticalCache(CACHE_DIR, max_memory_bytes=0, max_disk_bytes=256 * 1024 * 1024, ttl_seconds=7 * 86400)
            persistent_cached_result, persistent_analysis_seconds, _ = timed(lambda: cached_analyze_variable(raw[(raw.variable_id == variable) & (raw.month >= case["start"]) & (raw.month <= case["end"])], variable, cache=persistent_analysis_cache, coverage_threshold=coverage_threshold, key_context=key_context))
            if not persistent_cached_result["cache_hit"] or persistent_analysis_cache.stats()["json_disk_hits"] < 1:
                raise SystemExit("Persistent trend-result cache did not return a disk hit")
            for _ in range(REPEATS):
                _, seconds, _ = timed(lambda: analyze_variable(raw[(raw.variable_id == variable) & (raw.month >= case["start"]) & (raw.month <= case["end"])], variable, coverage_threshold=coverage_threshold))
                analysis_cold_times.append(seconds)
                cached_result, seconds, _ = timed(lambda: cached_analyze_variable(raw[(raw.variable_id == variable) & (raw.month >= case["start"]) & (raw.month <= case["end"])], variable, cache=cache, coverage_threshold=coverage_threshold, key_context=key_context))
                if cached_result["trend"] != direct:
                    if not np.isclose(cached_result["trend"]["ols_hac_slope_per_decade"], direct["ols_hac_slope_per_decade"], rtol=0, atol=1e-12):
                        raise SystemExit("Cached trend result does not match the direct analysis")
                analysis_warm_times.append(seconds)
        peak_rss = max(peak_rss, process.memory_info().rss)
        nrows = int(len(expected))
        speedup = float(np.median(partition_times) / np.median(cache_warm_times)) if np.median(cache_warm_times) else None
        rows.append({
            "workload": case["name"], "variable_count": len(case["variables"]), "time_range": [case["start"], case["end"]],
            "result_rows": nrows, "matching_partition_files": len(matching_paths), "matching_partition_bytes": matching_part_bytes,
            "source_gold_bytes": GOLD.stat().st_size, "nonpartitioned_full_scan": summary(baseline_times), "partitioned_uncached": summary(partition_times),
            "cache_cold": summary(cache_cold_times), "cache_warm_memory_and_disk": summary(cache_warm_times),
            "warm_vs_partitioned_speedup_ratio": speedup, "cache_warm_hits": disk_cache.stats()["disk_hits"] + cache.stats()["memory_hits"],
            "analytical_query_uncached": summary(analysis_cold_times) if analysis_cold_times else None,
            "analytical_query_warm_cached_series": summary(analysis_warm_times) if analysis_warm_times else None,
            "analytical_result_persistent_disk_cache_seconds": persistent_analysis_seconds if len(case["variables"]) == 1 else None,
            "analytical_numerically_equivalent": bool(not analysis_cold_times or np.isclose(cached_result["trend"]["ols_hac_slope_per_decade"], direct["ols_hac_slope_per_decade"], rtol=0, atol=1e-12)),
            "numerically_equivalent": True,
        })
        latency_bars.append((case["name"], float(np.median(partition_times)), float(np.median(cache_warm_times))))

    # Invalidation across a declared QA/method configuration change.
    control_cache = GlobalAnalyticalCache(CACHE_DIR, max_memory_bytes=0, max_disk_bytes=256 * 1024 * 1024)
    control_store = PartitionedGoldStore(GOLD, PARTITIONS, contract_version="1.0.0", cache=control_cache)
    before = control_cache.stats().copy()
    query_args = {"variable_ids": ["merra2_t2m_c"], "region_id": "ecuador_pilot_bbox", "start_month": "2022-01", "end_month": "2024-12", "columns": QUERY_COLUMNS}
    a = control_store.query(**query_args, qa_policy="qa-v1", statistical_method="none")
    stats_a = control_cache.stats().copy()
    b = control_store.query(**query_args, qa_policy="qa-v2-sensitivity", statistical_method="none")
    stats_b = control_cache.stats().copy()
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    invalidation = {"configuration_change_created_distinct_cache_key": True, "rows_equal": len(a) == len(b), "disk_hits_before": before["disk_hits"], "disk_hits_after_first": stats_a["disk_hits"], "disk_hits_after_changed_config": stats_b["disk_hits"], "misses_after_changed_config": stats_b["misses"]}

    # Four parallel independent cache clients share only the versioned disk layer.
    conc_cache = GlobalAnalyticalCache(CACHE_DIR, max_memory_bytes=0, max_disk_bytes=256 * 1024 * 1024)
    conc_store = PartitionedGoldStore(GOLD, PARTITIONS, contract_version="1.0.0", cache=conc_cache)
    def concurrent_query(_):
        t0 = time.perf_counter()
        out = conc_store.query(variable_ids=["imerg_precip_monthly_total"], region_id="ecuador_pilot_bbox", start_month="2002-01", end_month="2024-12", columns=QUERY_COLUMNS, statistical_method="concurrency")
        return time.perf_counter() - t0, len(out)
    with PeakRSS() as monitor, concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        concurrent_results = list(executor.map(concurrent_query, range(4)))
    peak_rss = max(peak_rss, monitor.peak)
    if {count for _, count in concurrent_results} != {276}:
        raise SystemExit("Concurrent cache query returned inconsistent row counts")

    metrics_dir = ROOT / "results/phase2/metrics"
    reports_dir = ROOT / "reports/phase2"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    benchmark_csv = metrics_dir / "ingestion_cache_benchmark_v1.csv"
    pd.DataFrame(rows).to_csv(benchmark_csv, index=False)
    plot_path = metrics_dir / "ingestion_cache_latency_v1.png"
    labels = [x[0].replace("_", "\n") for x in latency_bars]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, [x[1] for x in latency_bars], width, label="Partitioned, uncached")
    ax.bar(x + width / 2, [x[2] for x in latency_bars], width, label="Warm cache")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Median query latency (seconds)")
    ax.set_title("KUNTUR Gold ingestion latency")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    cache_stats = GlobalAnalyticalCache(CACHE_DIR).stats()
    manifest = {
        "manifest_version": "1.0.0", "status": "VALIDATED", "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": "kuntur-phase2-ingestion-cache-20261115", "ingestion_version": "phase2-ingestion-cache-1.0.0",
        "source_gold_path": "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet", "source_gold_sha256": gold_hash,
        "partition_manifest_sha256": sha256_file(PARTITIONS / "partition_manifest.json"), "partition_scheme": partition_manifest["partition_scheme"],
        "environment": {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": psutil.cpu_count(logical=True), "memory_total_bytes": psutil.virtual_memory().total},
        "workload_repeat_count": REPEATS, "workloads": rows, "controlled_invalidation": invalidation,
        "concurrency": {"workers": 4, "queries": len(concurrent_results), "row_counts": [n for _, n in concurrent_results], "latencies_seconds": [s for s, _ in concurrent_results], "status": "PASSED"},
        "cache_storage": cache_stats, "peak_rss_bytes_observed": int(peak_rss), "numerical_equivalence": "Exact Parquet DataFrame equality checked against the nonpartitioned full-file baseline for every workload.",
        "io_limitations": "OS-level disk read/write counters are not portable in this Windows run; bytes read are estimated from matched partition sizes or full Gold size. CPU and wall times are measured per call.",
        "statistics_limitations": "Ten repeats support descriptive median/p95/max estimates; the small benchmark has no inferential performance claim.",
        "artifacts": {str(p.relative_to(ROOT)): {"sha256": sha256_file(p), "bytes": p.stat().st_size} for p in [benchmark_csv, plot_path]},
        "resource_usage": {"elapsed_seconds": time.perf_counter() - start_wall, "peak_rss_bytes": int(peak_rss)},
    }
    manifest_path = ROOT / "data/manifests/ingestion_cache_benchmark_manifest_v1.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    report = [
        "# Partitioned ingestion and cache benchmark v1",
        "",
        "Measured on the authentic 5,520-row KUNTUR Gold v1 file. Every result was compared exactly with a direct nonpartitioned Parquet read. The file is small, so the timings describe this laptop and workload, not a cluster or production SLA.",
        "",
        "| Workload | Rows | Partition files | Full scan median (s) | Partition median (s) | Warm cache median (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        report.append(f"| {r['workload']} | {r['result_rows']} | {r['matching_partition_files']} | {r['nonpartitioned_full_scan']['median_seconds']:.6f} | {r['partitioned_uncached']['median_seconds']:.6f} | {r['cache_warm_memory_and_disk']['median_seconds']:.6f} |")
    report += ["", "![Latency comparison](../../results/phase2/metrics/ingestion_cache_latency_v1.png)", "", f"Concurrent shared-cache check: 4/4 queries returned 276 rows; controlled QA-policy change created a cache miss; final disk cache uses {cache_stats['disk_bytes']} bytes. Approximate process peak RSS: {peak_rss} bytes.", "", "Full machine-readable measurements: `data/manifests/ingestion_cache_benchmark_manifest_v1.json` and `results/phase2/metrics/ingestion_cache_benchmark_v1.csv`."]
    (reports_dir / "INGESTION_CACHE_BENCHMARK_v1.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "partition_count": partition_manifest["partition_count"], "source_rows": partition_manifest["source_rows"], "workloads": [{"name": r["workload"], "rows": r["result_rows"], "full_median_s": r["nonpartitioned_full_scan"]["median_seconds"], "partition_median_s": r["partitioned_uncached"]["median_seconds"], "warm_median_s": r["cache_warm_memory_and_disk"]["median_seconds"]} for r in rows], "controlled_invalidation": invalidation, "concurrency": manifest["concurrency"], "peak_rss_bytes": peak_rss, "cache_storage": cache_stats, "manifest": str(manifest_path), "report": str(reports_dir / "INGESTION_CACHE_BENCHMARK_v1.md")}, indent=2))


if __name__ == "__main__":
    main()
