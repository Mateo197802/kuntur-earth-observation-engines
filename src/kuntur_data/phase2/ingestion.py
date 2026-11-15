"""Hash-aware partitioned Gold ingestion and a process-shared disk cache."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote

import pandas as pd


INGESTION_VERSION = "phase2-ingestion-cache-1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class GlobalAnalyticalCache:
    """LRU memory layer plus atomic Parquet disk entries shared by processes."""

    _locks_guard = threading.Lock()
    _locks: dict[str, threading.RLock] = {}

    def __init__(
        self,
        root: Path,
        *,
        max_memory_bytes: int = 64 * 1024 * 1024,
        max_disk_bytes: int = 512 * 1024 * 1024,
        ttl_seconds: float = 7 * 24 * 3600,
        lock_timeout_seconds: float = 15.0,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_memory_bytes = int(max_memory_bytes)
        self.max_disk_bytes = int(max_disk_bytes)
        self.ttl_seconds = float(ttl_seconds)
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self._memory: OrderedDict[str, tuple[pd.DataFrame, int, float]] = OrderedDict()
        self._memory_bytes = 0
        self._json_memory: OrderedDict[str, tuple[dict, int, float]] = OrderedDict()
        self._memory_json_bytes = 0
        self._stats = {"memory_hits": 0, "disk_hits": 0, "json_memory_hits": 0, "json_disk_hits": 0, "misses": 0, "writes": 0, "json_writes": 0, "invalidations": 0}

    @staticmethod
    def key(parameters: dict) -> str:
        """Create a content-addressed key from canonical scientific parameters."""
        encoded = json.dumps(parameters, sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _entry(self, key: str) -> tuple[Path, Path, Path]:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Cache keys must be lowercase SHA-256 hex strings")
        return self.root / f"{key}.parquet", self.root / f"{key}.json", self.root / f"{key}.lock"

    @contextmanager
    def _key_lock(self, key: str):
        with self._locks_guard:
            lock = self._locks.setdefault(key, threading.RLock())
        with lock:
            _, _, lock_path = self._entry(key)
            deadline = time.monotonic() + self.lock_timeout_seconds
            fd = None
            while fd is None:
                try:
                    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for cache key lock {key}")
                    time.sleep(0.025)
            try:
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                fd = None
                yield
            finally:
                if fd is not None:
                    os.close(fd)
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _remember(self, key: str, frame: pd.DataFrame) -> None:
        size = int(frame.memory_usage(index=True, deep=True).sum())
        if size > self.max_memory_bytes or self.max_memory_bytes <= 0:
            return
        old = self._memory.pop(key, None)
        if old:
            self._memory_bytes -= old[1]
        self._memory[key] = (frame.copy(deep=True), size, time.time())
        self._memory_bytes += size
        self._evict_memory()

    def _remember_json(self, key: str, value: dict) -> None:
        size = len(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())
        if size > self.max_memory_bytes or self.max_memory_bytes <= 0:
            return
        old = self._json_memory.pop(key, None)
        if old:
            self._memory_json_bytes -= old[1]
        self._json_memory[key] = (json.loads(json.dumps(value, default=str)), size, time.time())
        self._memory_json_bytes += size
        self._evict_memory()

    def _evict_memory(self) -> None:
        while self._memory_bytes + self._memory_json_bytes > self.max_memory_bytes:
            frame_old = next(iter(self._memory.items()), None)
            json_old = next(iter(self._json_memory.items()), None)
            if frame_old is None and json_old is None:
                break
            if json_old is None or (frame_old is not None and frame_old[1][2] <= json_old[1][2]):
                _, (_, removed, _) = self._memory.popitem(last=False)
                self._memory_bytes -= removed
            else:
                _, (_, removed, _) = self._json_memory.popitem(last=False)
                self._memory_json_bytes -= removed

    def get_frame(self, key: str) -> pd.DataFrame | None:
        parquet, metadata, _ = self._entry(key)
        now = time.time()
        cached = self._memory.get(key)
        if cached is not None:
            frame, size, created = cached
            if now - created <= self.ttl_seconds:
                self._memory.move_to_end(key)
                self._memory[key] = (frame, size, now)
                self._stats["memory_hits"] += 1
                return frame.copy(deep=True)
            self._memory_bytes -= size
            del self._memory[key]
        if not parquet.is_file() or not metadata.is_file():
            self._stats["misses"] += 1
            return None
        try:
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            if now - float(meta["created_epoch"]) > self.ttl_seconds:
                self.invalidate(key)
                self._stats["misses"] += 1
                return None
            if sha256_file(parquet) != meta["sha256"]:
                self.invalidate(key)
                self._stats["misses"] += 1
                return None
            frame = pd.read_parquet(parquet)
            os.utime(parquet, None)
            self._remember(key, frame)
            self._stats["disk_hits"] += 1
            return frame
        except (OSError, ValueError, KeyError, json.JSONDecodeError, Exception):
            # A partial/corrupt cache is a miss; source Gold remains authoritative.
            self.invalidate(key)
            self._stats["misses"] += 1
            return None

    def put_frame(self, key: str, frame: pd.DataFrame) -> None:
        parquet, metadata, _ = self._entry(key)
        with self._key_lock(key):
            temp = parquet.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp.parquet")
            try:
                frame.to_parquet(temp, index=False, compression="zstd")
                digest = sha256_file(temp)
                os.replace(temp, parquet)
                payload = {"key": key, "created_epoch": time.time(), "sha256": digest, "bytes": parquet.stat().st_size, "rows": len(frame), "columns": list(frame.columns), "cache_version": INGESTION_VERSION}
                meta_tmp = metadata.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp.json")
                meta_tmp.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
                os.replace(meta_tmp, metadata)
            finally:
                temp.unlink(missing_ok=True)
            self._remember(key, frame)
            self._stats["writes"] += 1
        self._evict_disk()

    def get_json(self, key: str) -> dict | None:
        result_path = self.root / f"{key}.result.json"
        metadata_path = self.root / f"{key}.result.meta.json"
        now = time.time()
        cached = self._json_memory.get(key)
        if cached is not None:
            value, size, created = cached
            if now - created <= self.ttl_seconds:
                self._json_memory.move_to_end(key)
                self._json_memory[key] = (value, size, now)
                self._stats["json_memory_hits"] += 1
                return json.loads(json.dumps(value, default=str))
            self._memory_json_bytes -= size
            del self._json_memory[key]
        if not result_path.is_file() or not metadata_path.is_file():
            self._stats["misses"] += 1
            return None
        try:
            payload = result_path.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if now - float(metadata["created_epoch"]) > self.ttl_seconds or hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
                self.invalidate(key)
                self._stats["misses"] += 1
                return None
            value = json.loads(payload.decode("utf-8"))
            self._remember_json(key, value)
            os.utime(result_path, None)
            self._stats["json_disk_hits"] += 1
            return value
        except Exception:
            self.invalidate(key)
            self._stats["misses"] += 1
            return None

    def put_json(self, key: str, value: dict) -> None:
        result_path = self.root / f"{key}.result.json"
        metadata_path = self.root / f"{key}.result.meta.json"
        with self._key_lock(key):
            encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
            result_tmp = result_path.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp.result.json")
            result_tmp.write_bytes(encoded)
            os.replace(result_tmp, result_path)
            metadata = {"key": key, "created_epoch": time.time(), "sha256": hashlib.sha256(encoded).hexdigest(), "bytes": len(encoded), "cache_version": INGESTION_VERSION}
            metadata_tmp = metadata_path.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp.result.meta.json")
            metadata_tmp.write_text(json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8")
            os.replace(metadata_tmp, metadata_path)
            self._remember_json(key, value)
            self._stats["json_writes"] += 1
        self._evict_disk()

    def invalidate(self, key: str) -> None:
        parquet, metadata, _ = self._entry(key)
        cached = self._memory.pop(key, None)
        if cached:
            self._memory_bytes -= cached[1]
        cached_json = self._json_memory.pop(key, None)
        if cached_json:
            self._memory_json_bytes -= cached_json[1]
        parquet.unlink(missing_ok=True)
        metadata.unlink(missing_ok=True)
        (self.root / f"{key}.result.json").unlink(missing_ok=True)
        (self.root / f"{key}.result.meta.json").unlink(missing_ok=True)
        self._stats["invalidations"] += 1

    def invalidate_all(self) -> None:
        keys = {p.stem for p in self.root.glob("*.parquet")}
        keys.update(p.name.removesuffix(".result.json") for p in self.root.glob("*.result.json"))
        for key in keys:
            self.invalidate(key)

    def _evict_disk(self) -> None:
        entries = list(self.root.glob("*.parquet")) + list(self.root.glob("*.result.json"))
        total = sum(p.stat().st_size for p in entries)
        for parquet in sorted(entries, key=lambda p: p.stat().st_atime):
            if total <= self.max_disk_bytes:
                break
            size = parquet.stat().st_size
            if parquet.name.endswith(".result.json"):
                self.root.joinpath(parquet.name.removesuffix(".result.json") + ".result.meta.json").unlink(missing_ok=True)
            else:
                parquet.with_suffix(".json").unlink(missing_ok=True)
            parquet.unlink(missing_ok=True)
            total -= size

    def stats(self) -> dict:
        disk_files = list(self.root.glob("*.parquet")) + list(self.root.glob("*.result.json"))
        return {**self._stats, "memory_entries": len(self._memory) + len(self._json_memory), "memory_frame_bytes": self._memory_bytes, "memory_json_bytes": self._memory_json_bytes, "memory_bytes": self._memory_bytes + self._memory_json_bytes, "disk_entries": len(disk_files), "disk_bytes": sum(p.stat().st_size for p in disk_files), "root": str(self.root)}


class PartitionedGoldStore:
    """Year-bounded Parquet partitions with hash/version-aware filtered queries."""

    def __init__(
        self,
        gold_path: Path,
        partition_root: Path,
        *,
        contract_version: str = "1.0.0",
        cache: GlobalAnalyticalCache | None = None,
    ):
        self.gold_path = Path(gold_path)
        self.partition_root = Path(partition_root)
        self.contract_version = str(contract_version)
        self.cache = cache

    def materialize(self) -> dict:
        if not self.gold_path.is_file():
            raise FileNotFoundError(self.gold_path)
        source_hash = sha256_file(self.gold_path)
        manifest_path = self.partition_root / "partition_manifest.json"
        if manifest_path.is_file():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("source_gold_sha256") == source_hash and existing.get("contract_version") == self.contract_version and existing.get("status") == "VALIDATED":
                return existing
            raise RuntimeError("Partition root contains a different frozen source/version; choose a new versioned path")
        frame = pd.read_parquet(self.gold_path)
        required = {"dataset_id", "variable_id", "month", "region_id", "value", "contract_sha256"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Gold input missing partition columns: {sorted(missing)}")
        frame["year"] = frame.month.astype(str).str.slice(0, 4)
        frame["time_block"] = "12-month-year"
        rows_written = 0
        partition_count = 0
        for (dataset, variable, year, region), group in frame.groupby(["dataset_id", "variable_id", "year", "region_id"], sort=True, dropna=False):
            parts = [
                f"dataset_id={quote(str(dataset), safe='')}",
                f"variable_id={quote(str(variable), safe='')}",
                f"year={quote(str(year), safe='')}",
                f"region_id={quote(str(region), safe='')}",
                f"contract_version={quote(self.contract_version, safe='')}",
            ]
            out_dir = self.partition_root.joinpath(*parts)
            out_dir.mkdir(parents=True, exist_ok=True)
            data_path = out_dir / "part.parquet"
            temp = data_path.with_suffix(".tmp.parquet")
            group.drop(columns=["year", "time_block"]).to_parquet(temp, index=False, compression="zstd")
            os.replace(temp, data_path)
            rows_written += len(group)
            partition_count += 1
        manifest = {
            "manifest_version": "1.0.0", "status": "VALIDATED", "ingestion_version": INGESTION_VERSION,
            "source_gold_path": self.gold_path.name, "source_gold_sha256": source_hash,
            "contract_version": self.contract_version, "partition_scheme": ["dataset_id", "variable_id", "year (bounded 12-month block)", "region_id", "contract_version"],
            "native_spatial_grids": "Preserved as precomputed Gold regional values; no cross-collection resampling performed.",
            "row_count": rows_written, "partition_count": partition_count,
            "source_rows": len(frame), "partitioned_rows": rows_written,
        }
        temp_manifest = manifest_path.with_suffix(".tmp")
        temp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp_manifest, manifest_path)
        return manifest

    def query(
        self,
        *,
        variable_ids: list[str] | None = None,
        dataset_ids: list[str] | None = None,
        region_id: str | None = None,
        start_month: str | None = None,
        end_month: str | None = None,
        columns: list[str] | None = None,
        coverage_threshold: float | None = None,
        qa_policy: str = "v1 Gold quality_status and valid-area field",
        aggregation_method: str = "precomputed Gold spherical_cell_area_weighted_mean",
        statistical_method: str = "none",
    ) -> pd.DataFrame:
        manifest_path = self.partition_root / "partition_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("Call materialize() before querying partitioned Gold")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_hash = manifest["source_gold_sha256"]
        columns_key = sorted(columns) if columns is not None else None
        params = {
            "dataset_id": sorted(dataset_ids or []), "source_gold_sha256": source_hash,
            "contract_version": self.contract_version, "region_geometry": region_id or "all-regions-as-stored",
            "variable": sorted(variable_ids or []), "time_interval": [start_month, end_month],
            "qa_policy": qa_policy, "coverage_threshold": coverage_threshold,
            "aggregation_method": aggregation_method, "statistical_method": statistical_method,
            "columns": columns_key, "code_version": INGESTION_VERSION,
        }
        cache_key = self.cache.key(params) if self.cache else None
        if cache_key:
            cached = self.cache.get_frame(cache_key)
            if cached is not None:
                return cached
        paths = []
        base = self.partition_root
        dataset_parts = dataset_ids or [None]
        variable_parts = variable_ids or [None]
        years: list[str | None] = [None]
        if start_month or end_month:
            lo = int((start_month or "0000-01")[:4])
            hi = int((end_month or "9999-12")[:4])
            years = [str(y) for y in range(lo, hi + 1)]
        for ds in dataset_parts:
            for var in variable_parts:
                for year in years:
                    pattern = (
                        f"dataset_id={quote(str(ds), safe='')}" if ds is not None else "dataset_id=*"
                    ) + "/" + (
                        f"variable_id={quote(str(var), safe='')}" if var is not None else "variable_id=*"
                    ) + "/" + (f"year={year}" if year is not None else "year=*") + "/" + (
                        f"region_id={quote(region_id, safe='')}" if region_id is not None else "region_id=*"
                    ) + f"/contract_version={quote(self.contract_version, safe='')}/part.parquet"
                    paths.extend(base.glob(pattern))
        frames = []
        for path in sorted(set(paths)):
            read_columns = None
            if columns is not None:
                internal = set(columns) | {"month", "variable_id", "dataset_id", "region_id"}
                if coverage_threshold is not None:
                    internal.add("valid_area_fraction_bbox")
                read_columns = list(internal)
            part = pd.read_parquet(path, columns=read_columns)
            if variable_ids is not None:
                part = part[part.variable_id.isin(variable_ids)]
            if dataset_ids is not None:
                part = part[part.dataset_id.isin(dataset_ids)]
            if region_id is not None:
                part = part[part.region_id == region_id]
            if start_month is not None:
                part = part[part.month >= start_month]
            if end_month is not None:
                part = part[part.month <= end_month]
            if coverage_threshold is not None:
                part = part[part.valid_area_fraction_bbox >= float(coverage_threshold)]
            if columns is not None:
                part = part.loc[:, columns]
            frames.append(part)
        result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
        if not result.empty and {"dataset_id", "variable_id", "month"}.issubset(result.columns):
            result = result.sort_values(["dataset_id", "variable_id", "month"], kind="stable").reset_index(drop=True)
        if cache_key:
            self.cache.put_frame(cache_key, result)
        return result
