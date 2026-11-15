"""Leakage-safe Phase 2 chronological split derived from immutable v1."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def phase2_split(split_frame: pd.DataFrame) -> pd.DataFrame:
    """Return sample-level roles; 152 fit + 38 internal holdout within v1 train."""
    required = {"sample_id", "target_month", "forecast_origin_month", "issue_date_utc", "history_start_month", "history_end_month", "partition"}
    if not required.issubset(split_frame.columns):
        raise ValueError(f"Missing split columns: {sorted(required - set(split_frame.columns))}")
    if split_frame["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id in frozen v1 split")
    frame = split_frame.copy()
    frame["target_month"] = frame["target_month"].astype(str)
    frame = frame.sort_values(["target_month", "sample_id"], kind="stable").reset_index(drop=True)
    train = frame[frame["partition"] == "train"].copy()
    if len(train) != 190:
        raise ValueError(f"Expected 190 original training samples, got {len(train)}")
    roles = {sid: "model_fit" for sid in train.iloc[:152]["sample_id"]}
    roles.update({sid: "internal_robustness_holdout" for sid in train.iloc[152:]["sample_id"]})
    frame["phase2_role"] = [roles.get(sid, partition) for sid, partition in zip(frame["sample_id"], frame["partition"])]
    if (frame.loc[frame["partition"] != "train", "phase2_role"] != frame.loc[frame["partition"] != "train", "partition"]).any():
        raise AssertionError("Validation/test role changed")
    if set(frame[frame.phase2_role == "model_fit"].sample_id) & set(frame[frame.phase2_role == "internal_robustness_holdout"].sample_id):
        raise AssertionError("Model-fit and robustness samples overlap")
    fit = frame[frame.phase2_role == "model_fit"]
    robust = frame[frame.phase2_role == "internal_robustness_holdout"]
    if len(fit) != 152 or len(robust) != 38 or fit.target_month.max() >= robust.target_month.min():
        raise AssertionError("Phase 2 split is not the declared chronological 152/38 partition")
    return frame


def write_phase2_split(root: Path) -> dict:
    split_path = root / "data/gold/kuntur_forecast_split_indices_v1.parquet"
    forecast_path = root / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    v1_manifest_path = root / "data/manifests/forecast_split_manifest_v1.json"
    output = root / "data/derived/phase2/forecast_phase2_split_v1.parquet"
    manifest_path = root / "data/manifests/phase2_split_manifest_v1.json"
    v1 = pd.read_parquet(split_path)
    phase = phase2_split(v1)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    phase.to_parquet(temp, index=False, compression="zstd")
    temp.replace(output)
    manifest = {
        "manifest_version": "1.0.0",
        "status": "VALIDATED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "policy": "chronological 80/20 split within original v1 train only; no augmentation before split",
        "role_note": "The 38-sample internal robustness holdout is a development diagnostic, not external validation or an independent domain. The original 2022-2024 test remains untouched.",
        "source_forecast_path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet",
        "source_forecast_sha256": sha256_file(forecast_path),
        "source_v1_split_path": "data/gold/kuntur_forecast_split_indices_v1.parquet",
        "source_v1_split_sha256": sha256_file(split_path),
        "source_v1_split_manifest_sha256": sha256_file(v1_manifest_path),
        "output_path": "data/derived/phase2/forecast_phase2_split_v1.parquet",
        "output_sha256": sha256_file(output),
        "rows": len(phase),
        "counts": {k: int(v) for k, v in phase.phase2_role.value_counts().to_dict().items()},
        "exact_target_ranges": {k: {"start": str(g.target_month.min()), "end": str(g.target_month.max()), "count": int(len(g))} for k, g in phase.groupby("phase2_role", sort=True)},
        "validation": {"status": "PASSED", "checks": ["v1 input hashes recorded", "v1 partition column unchanged", "80/20 division uses original train only", "152/38 split chronological", "unique sample IDs", "no overlap among phases", "original validation and test retained"]},
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
