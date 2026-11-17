from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/statistical_correction/interval_comparison_v1_3_2"
FREEZE_PATH = RESULTS / "protocol_freeze.json"

SOURCE_PATHS = (
    "configs/statistical_interval_comparison_v1_3_2.json",
    "docs/protocols/statistical_interval_comparison_v1_3_2.md",
    "scripts/statistical_correction/interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/provenance_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/freeze_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/run_interval_comparison_v1_3_2.py",
    "scripts/statistical_correction/summarize_interval_comparison_v1_3_2.py",
    "tests/test_interval_comparison_v13_2.py",
    "scripts/statistical_correction/run_calibration.py",
    "src/kuntur_data/phase2/statistical_v12.py",
    "scripts/statistical_correction/summarize_corrective_confirmation_v1_3.py",
    "configs/statistical_corrective_confirmation_v1_3.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_manifest.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_metrics.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/checkpoints/cell_iid_gaussian_n276_s0p00_sarimax_seasonal_ar12.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/checkpoints/cell_iid_gaussian_n276_s0p25_sarimax_seasonal_ar12.json",
    "results/statistical_correction/independent_confirmation_v1_3_iid/checkpoints/cell_iid_gaussian_n276_s0p50_sarimax_seasonal_ar12.json",
    "results/statistical_correction/corrective_development_v1_3/selection_freeze_iid_v1_3_r1.json",
    "results/statistical_correction/validated_regime_registry.json",
    "results/statistical_correction/validation_registry_manifest.json",
    "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json",
    "configs/statistical_interval_comparison_v1_3_1.json",
    "docs/protocols/statistical_interval_comparison_v1_3_1.md",
    "scripts/statistical_correction/interval_comparison_v1_3_1.py",
    "scripts/statistical_correction/provenance_interval_comparison_v1_3_1.py",
    "scripts/statistical_correction/freeze_interval_comparison_v1_3_1.py",
    "scripts/statistical_correction/run_interval_comparison_v1_3_1.py",
    "scripts/statistical_correction/summarize_interval_comparison_v1_3_1.py",
    "tests/test_interval_comparison_v13_1.py",
    "results/statistical_correction/interval_comparison_v1_3_1/protocol_freeze.json",
    "reports/statistical_correction/v1_3_1_preflight_failure_addendum.md",
    "results/statistical_correction/interval_comparison_v1_3_1/preflight/run_manifest.json",
    "results/statistical_correction/interval_comparison_v1_3_1/preflight/preflight_summary.json",
    "results/statistical_correction/interval_comparison_v1_3_1/preflight/checkpoints/cell_iid_gaussian_n276_s0p00.json",
    "results/statistical_correction/interval_comparison_v1_3_1/preflight/checkpoints/cell_iid_gaussian_n276_s0p25.json",
    "results/statistical_correction/interval_comparison_v1_3_1/preflight/checkpoints/cell_iid_gaussian_n276_s0p50.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(*, allow_missing_protected: bool = False) -> dict[str, str]:
    output: dict[str, str] = {}
    missing: list[str] = []
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        output[relative] = sha256_file(path)
    if missing and not allow_missing_protected:
        raise FileNotFoundError("Required freeze inputs missing: " + ", ".join(missing))
    return output
