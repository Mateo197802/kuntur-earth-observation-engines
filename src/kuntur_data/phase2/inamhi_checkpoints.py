"""Atomic, checksum-verified per-year checkpoints for INAMHI station tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _manifest(manifest_path: Path) -> dict:
    if not manifest_path.is_file():
        return {"manifest_version": "1.0.0", "status": "CHECKPOINTS_IN_PROGRESS", "years": {}}
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("years"), dict):
        raise ValueError("INAMHI checkpoint manifest has no year map")
    return payload


def _validate_frame(frame: pd.DataFrame, year: int, source_sha256: str) -> None:
    required = {"year", "station_id", "month_number", "source_pdf_sha256"}
    if not required.issubset(frame.columns):
        raise ValueError(f"INAMHI year {year} checkpoint is missing required columns")
    if len(frame) and (set(frame.year.astype(int)) != {year} or set(frame.source_pdf_sha256.astype(str)) != {source_sha256}):
        raise ValueError(f"INAMHI year {year} checkpoint provenance does not match")
    if frame.duplicated(["year", "station_id", "month_number"]).any():
        raise ValueError(f"INAMHI year {year} checkpoint contains duplicate station-month rows")


def write_year_checkpoint(
    year: int,
    frame: pd.DataFrame,
    *,
    source_record: dict,
    extraction: dict,
    root: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
) -> dict:
    """Write one year atomically, validate it, then atomically advance its manifest."""
    source_sha256 = str(source_record.get("sha256", ""))
    if len(source_sha256) != 64:
        raise ValueError("INAMHI source PDF SHA-256 is required for a checkpoint")
    _validate_frame(frame, year, source_sha256)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if len(frame):
        path = checkpoint_dir / f"inamhi_{year}_station_monthly_v1.parquet"
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False, compression="zstd")
        temporary.replace(path)
        stored = pd.read_parquet(path)
        _validate_frame(stored, year, source_sha256)
        record = {
            "status": "VALIDATED",
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "rows": int(len(stored)),
            "stations": int(stored.station_id.nunique()),
        }
    else:
        record = {"status": "VALIDATED_NO_BBOX_ROWS", "path": None, "bytes": 0, "sha256": None, "rows": 0, "stations": 0}

    record.update({
        "year": int(year),
        "source_pdf_sha256": source_sha256,
        "source_pdf_path": source_record.get("path"),
        "extraction": extraction,
        "source_record": source_record,
    })
    payload = _manifest(manifest_path)
    payload["status"] = "CHECKPOINTS_IN_PROGRESS"
    payload.setdefault("run_id", "inamhi-yearbooks-pilot-bbox-2002-2012")
    payload["years"][str(year)] = record
    atomic_json(manifest_path, payload)
    return record


def load_year_checkpoint(
    year: int,
    *,
    expected_source_sha256: str,
    root: Path,
    checkpoint_dir: Path,
    manifest_path: Path,
) -> tuple[pd.DataFrame, dict] | None:
    """Return a checkpoint only when source hash, data hash, grain and year all validate."""
    if not manifest_path.is_file():
        return None
    try:
        record = _manifest(manifest_path).get("years", {}).get(str(year))
        if not record or record.get("source_pdf_sha256") != expected_source_sha256:
            return None
        if record.get("status") == "VALIDATED_NO_BBOX_ROWS" and int(record.get("rows", -1)) == 0:
            return pd.DataFrame(), record
        if record.get("status") != "VALIDATED" or not record.get("path"):
            return None
        path = root / Path(record["path"])
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            return None
        frame = pd.read_parquet(path)
        _validate_frame(frame, year, expected_source_sha256)
        if len(frame) != int(record.get("rows", -1)):
            return None
        return frame, record
    except (OSError, ValueError, KeyError, json.JSONDecodeError, ImportError):
        return None
