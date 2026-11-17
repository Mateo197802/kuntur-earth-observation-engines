#!/usr/bin/env python3
"""Export validated compact station tables to checksummed CSV for CEDIA."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data/external/processed/phase2_v12_native_grid_collocation_inputs"
INPUTS = {
    "NOAA_GHCN_M_QCU": {
        "table": ROOT / "data/external/processed/ghcnm_ecuador_bbox_station_monthly_v1.parquet",
        "manifest": ROOT / "data/manifests/external/ghcnm_qcu_external_manifest_v1.json",
        "expected_manifest_status": "ACQUIRED_AND_VALIDATED",
    },
    "INAMHI_yearbook_temperature": {
        "table": ROOT / "data/external/processed/inamhi_pilot_bbox_station_monthly_2002_2012_v1.parquet",
        "manifest": ROOT / "data/manifests/external/inamhi_yearbooks_external_manifest_v1.json",
        "expected_manifest_status": "ACQUIRED_AND_VALIDATED_PARTIAL_YEARBOOK_COVERAGE",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _output_path_for(source: str, table: Path) -> Path:
    if source == "NOAA_GHCN_M_QCU":
        return OUTPUT_DIR / "ghcnm_ecuador_bbox_station_monthly_v1.csv"
    return OUTPUT_DIR / "inamhi_pilot_bbox_station_monthly_2002_2012_v1.csv"


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {OUTPUT_DIR.relative_to(ROOT)}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for source, spec in INPUTS.items():
        table_path = spec["table"]
        manifest_path = spec["manifest"]
        if not table_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"Missing validated input or provenance manifest for {source}")
        source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_status = source_manifest.get("status")
        if manifest_status != spec["expected_manifest_status"]:
            raise ValueError(f"Unexpected source-manifest status for {source}: {manifest_status}")
        artifact_reference = f"data/external/processed/{table_path.name}"
        if source == "NOAA_GHCN_M_QCU":
            if artifact_reference not in source_manifest.get("artifacts", []):
                raise ValueError("NOAA manifest does not reference the validated station-month table")
        else:
            artifact_entry = (source_manifest.get("artifact_hashes") or {}).get(artifact_reference)
            expected_hash = artifact_entry.get("sha256") if isinstance(artifact_entry, dict) else artifact_entry
            if expected_hash != sha256_file(table_path):
                raise ValueError("INAMHI station-month table SHA-256 does not match its source manifest")

        frame = pd.read_parquet(table_path)
        if frame.duplicated(["station_id", "month"]).any():
            raise ValueError(f"Duplicate station-month rows found in {source}")
        csv_path = _output_path_for(source, table_path)
        if csv_path.exists() or Path(str(csv_path) + ".provenance.json").exists():
            raise FileExistsError(f"Refusing to overwrite existing transfer input for {source}")
        csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        atomic_write(csv_path, csv_bytes)
        sidecar = {
            "manifest_version": "kuntur-phase2-v12-station-table-export-0.1.0",
            "source_id": source,
            "source_table_path": table_path.relative_to(ROOT).as_posix(),
            "source_table_sha256": sha256_file(table_path),
            "source_manifest_path": manifest_path.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": sha256_file(manifest_path),
            "source_manifest_status": manifest_status,
            "csv_path": csv_path.relative_to(ROOT).as_posix(),
            "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "row_count": int(len(frame)),
            "columns": list(frame.columns),
            "conversion": "pandas read_parquet followed by to_csv(index=False, lineterminator=LF); values are not transformed.",
            "pandas_version": pd.__version__,
            "exporter_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        sidecar_path = Path(str(csv_path) + ".provenance.json")
        sidecar_bytes = (json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        atomic_write(sidecar_path, sidecar_bytes)
        summaries[source] = {
            "rows": int(len(frame)),
            "csv_path": csv_path.relative_to(ROOT).as_posix(),
            "csv_bytes": len(csv_bytes),
            "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
            "source_table_sha256": sidecar["source_table_sha256"],
            "source_manifest_sha256": sidecar["source_manifest_sha256"],
            "sidecar_sha256": sha256_file(sidecar_path),
        }
    print(json.dumps({"status": "EXPORTED_AND_HASHED", "outputs": summaries}, indent=2))


if __name__ == "__main__":
    main()
