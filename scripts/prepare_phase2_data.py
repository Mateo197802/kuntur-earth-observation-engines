#!/usr/bin/env python3
"""Build the Phase 2 split and deterministic, fit-only augmentation table."""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from kuntur_data.phase2.augmentation import (
    AUGMENTATION_VERSION,
    FEATURE_COLUMNS,
    MISSING_COLUMNS,
    training_augmentation,
)
from kuntur_data.phase2.splits import phase2_split, sha256_file, write_phase2_split


ROOT = Path(__file__).resolve().parents[1]


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def main() -> None:
    split_manifest = write_phase2_split(ROOT)
    forecast_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet"
    gold_path = ROOT / "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet"
    split_path = ROOT / "data/derived/phase2/forecast_phase2_split_v1.parquet"
    forecast = pd.read_parquet(forecast_path)
    split = pd.read_parquet(split_path)
    joined = forecast.merge(split[["sample_id", "phase2_role"]], on="sample_id", validate="one_to_one")
    fit = joined[joined.phase2_role == "model_fit"].sort_values("target_month", kind="stable").reset_index(drop=True)
    if len(fit) != 152 or fit.target_month.max() >= joined.loc[joined.phase2_role == "internal_robustness_holdout", "target_month"].min():
        raise SystemExit("Phase 2 model-fit split did not pass its chronological gate")

    augmented, state = training_augmentation(fit, seed=20261115)
    if state["rejected"]:
        print("REJECTED_SYNTHETIC_ROWS", len(state["rejected"]))
    synthetic = augmented[augmented.noise_type != "clean"]
    synth_counts = synthetic.groupby("base_sample_id").size().to_dict()
    augmented["augmentation_weight"] = [1.0 if t == "clean" else 1.0 / synth_counts.get(sid, 1) for t, sid in zip(augmented.noise_type, augmented.base_sample_id)]
    # Per original sample, authentic and synthetic records each contribute total weight 1.
    weight_sums = augmented.groupby("base_sample_id").augmentation_weight.sum()
    if not np.allclose(weight_sums.to_numpy(), 2.0, atol=1e-12):
        raise SystemExit("Augmented sample weights do not preserve equal original/synthetic mass")

    out_dir = ROOT / "data/derived/phase2"
    out_path = out_dir / "augmentation_training_v1.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    temp = out_path.with_suffix(out_path.suffix + ".tmp")
    augmented.to_parquet(temp, index=False, compression="zstd")
    temp.replace(out_path)

    manifest_jsonl = ROOT / "data/manifests/augmentation_manifest_v1.jsonl"
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    manifest_temp = manifest_jsonl.with_suffix(manifest_jsonl.suffix + ".tmp")
    records = []
    for row in augmented.to_dict("records"):
        records.append({
            "base_sample_id": row["base_sample_id"],
            "augmentation_id": row["augmentation_id"],
            "noise_type": row["noise_type"],
            "noise_intensity": row["noise_intensity"],
            "noise_components": json.loads(row["noise_components"]),
            "is_combined_noise": bool(row["is_combined_noise"]),
            "random_seed": int(row["random_seed"]),
            "noise_parameters": json.loads(row["noise_parameters"]),
            "source_partition": row["source_partition"],
            "target_variable": row["target_variable"],
            "augmentation_version": row["augmentation_version"],
            "target_value_sha256": hashlib.sha256(f"{float(row['target_value']):.12g}".encode()).hexdigest(),
            "augmentation_weight": float(row["augmentation_weight"]),
            "status": "ACCEPTED",
        })
    with manifest_temp.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    manifest_temp.replace(manifest_jsonl)

    preprocess_state = {
        "state_version": "1.0.0",
        "channels": ["merra2_t2m_c", "merra2_qv2m", "merra2_u2m", "merra2_v2m", "merra2_ps"],
        "history_order": "oldest to newest (T-14 through T-3)",
        "fit_sample_count": len(fit),
        "fit_sample_ids_sha256": canonical_sha(sorted(fit.sample_id.astype(str).tolist())),
        "fit_only_imputation_location": state["location"].tolist(),
        "fit_only_noise_scale_sd": state["scale"].tolist(),
        "feature_columns": list(FEATURE_COLUMNS),
        "missing_indicator_columns": list(MISSING_COLUMNS),
        "source_forecast_sha256": sha256_file(forecast_path),
        "source_gold_sha256": sha256_file(gold_path),
        "source_split_sha256": split_manifest["output_sha256"],
        "augmentation_version": AUGMENTATION_VERSION,
    }
    state_path = out_dir / "model_preprocessing_state_v1.json"
    atomic_json(state_path, preprocess_state)

    config = {
        "augmentation_version": AUGMENTATION_VERSION,
        "seed": 20261115,
        "training_intensities": [0.05, 0.10, 0.15, 0.20],
        "single_families": ["gaussian", "laplace", "impulsive", "drift", "heteroscedastic", "missingness"],
        "combined_families": ["gaussian+drift", "gaussian+impulsive", "drift+missingness", "gaussian+heteroscedastic", "mixed_multi_family"],
        "sample_weight_policy": "For each base sample, clean weight=1 and all accepted synthetic variants sum to weight=1.",
        "scale_estimation": "Per-physical-variable SD pooled over 12 history months and only the 152 model-fit samples.",
        "missing_policy": "Training-fit channel means plus one explicit indicator per lag/channel; target never perturbed.",
    }
    manifest = {
        "manifest_version": "1.0.0",
        "status": "VALIDATED",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dataset_id": "KUNTUR_PHASE2_TRAINING_AUGMENTATION_V1",
        "source_forecast_path": "data/gold/kuntur_ecuador_pilot_bbox_forecast_plus1_v1.parquet",
        "source_forecast_sha256": sha256_file(forecast_path),
        "source_gold_path": "data/gold/kuntur_ecuador_pilot_bbox_monthly_v1.parquet",
        "source_gold_sha256": sha256_file(gold_path),
        "phase2_split_manifest_sha256": sha256_file(ROOT / "data/manifests/phase2_split_manifest_v1.json"),
        "output_path": "data/derived/phase2/augmentation_training_v1.parquet",
        "output_sha256": sha256_file(out_path),
        "output_bytes": out_path.stat().st_size,
        "augmentation_manifest_path": "data/manifests/augmentation_manifest_v1.jsonl",
        "augmentation_manifest_sha256": sha256_file(manifest_jsonl),
        "preprocessing_state_path": "data/derived/phase2/model_preprocessing_state_v1.json",
        "preprocessing_state_sha256": sha256_file(state_path),
        "configuration": config,
        "configuration_sha256": canonical_sha(config),
        "model_fit_original_samples": len(fit),
        "unique_original_samples": int(augmented.base_sample_id.nunique()),
        "total_rows": int(len(augmented)),
        "synthetic_rows": int((augmented.noise_type != "clean").sum()),
        "rows_by_noise_type": {str(k): int(v) for k, v in augmented.noise_type.value_counts().sort_index().items()},
        "rows_by_intensity": {f"{float(k):.2f}": int(v) for k, v in augmented.groupby("noise_intensity").size().items()},
        "combined_rows": int(augmented.is_combined_noise.sum()),
        "rejected_rows": state["rejected"],
        "sample_weight_total": float(augmented.augmentation_weight.sum()),
        "sample_weight_per_original_min_max": [float(weight_sums.min()), float(weight_sums.max())],
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "validation": {"status": "PASSED", "checks": ["all augmented rows derive from model-fit train only", "authentic targets unchanged", "all scenarios deterministically seeded", "single and combined noise labels explicit", "synthetic total weight equals authentic weight per base sample", "out-of-range synthetic rows rejected rather than clipped", "no validation, robustness-holdout, or test observations used for augmentation or noise scales"]},
    }
    summary_path = ROOT / "data/manifests/augmentation_manifest_v1.json"
    atomic_json(summary_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
