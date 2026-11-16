import json
from pathlib import Path

import pandas as pd

from scripts.statistical_correction import finalize_correction


ROOT = Path(__file__).resolve().parents[1]


def test_root_cause_matrix_keeps_frozen_sta_ids_attached_to_their_defects():
    findings = json.loads(
        (ROOT / "results/statistical_audit/root_cause_findings.json").read_text(encoding="utf-8")
    )["findings"]

    build_rows = getattr(finalize_correction, "root_cause_matrix_rows", None)
    assert callable(build_rows), "finalizer must expose the root-cause mapping used by its report"
    rows = build_rows(findings)

    assert {row["id"]: row["defect"] for row in rows} == {
        "STA-01": "Severe inference undercoverage",
        "STA-02": "Calendar-gap lag semantics",
        "STA-03": "MODIS QA sample-selection sensitivity",
        "STA-04": "Seasonal Mann-Kendall dependence",
        "STA-05": "Degenerate-series residual diagnostics",
        "STA-06": "Multiple-testing assumptions",
        "STA-07": "Nonfinite values pass missingness checks",
    }
    sta03 = next(row for row in rows if row["id"] == "STA-03")
    assert sta03["regression_evidence"] == "test_temporal_support_reports_stable_union_and_early_late_composition"


def test_frozen_monthly_contract_matches_its_release_guard():
    relative = "docs/monthly_data_contract.md"
    actual = finalize_correction.sha256(ROOT / relative)

    assert actual == finalize_correction.EXPECTED_FROZEN_ARTIFACT_HASHES[relative]


def test_partial_validation_registry_accepts_no_completed_validation_rows():
    selection = {
        "decisions": {
            "iid_gaussian": {
                "selected": {"method": "SEASONAL_MK_YEAR2", "calibration_score": 0.1}
            }
        }
    }

    registry = finalize_correction._build_validation_registry(
        selection, pd.DataFrame(), {"status": "PARTIAL_PHASE_RUN"}
    )

    assert registry["status"] == "PARTIAL_OR_EMPTY"
    assert registry["validated_regime_count"] == 0
    assert registry["regimes"] == {}


def test_partial_manifest_excludes_checkpoints_not_in_the_aggregated_phase(tmp_path):
    build_paths = getattr(finalize_correction, "stable_phase_output_paths", None)
    assert callable(build_paths), "finalizer must filter mutable, unaggregated scenario files"

    phase_dir = tmp_path / "validation"
    completed = "validation_iid_gaussian_n276_s0p0"
    in_progress = "validation_ar1_p030_n276_s0p0"
    for directory in (phase_dir / "checkpoints", phase_dir / "replicates"):
        directory.mkdir(parents=True, exist_ok=True)
    (phase_dir / "checkpoints" / f"{completed}.json").write_text("{}", encoding="utf-8")
    (phase_dir / "replicates" / f"{completed}.parquet").write_bytes(b"complete")
    (phase_dir / "checkpoints" / f"{in_progress}.json").write_text("{}", encoding="utf-8")
    (phase_dir / "replicates" / f"{in_progress}.parquet").write_bytes(b"partial")

    frame = pd.DataFrame([{"process": "iid_gaussian", "n_months": 276, "true_slope_per_decade": 0.0}])
    paths = build_paths(phase_dir, "validation", frame)
    relative = {path.relative_to(phase_dir).as_posix() for path in paths}

    assert f"checkpoints/{completed}.json" in relative
    assert f"replicates/{completed}.parquet" in relative
    assert f"checkpoints/{in_progress}.json" not in relative
    assert f"replicates/{in_progress}.parquet" not in relative


def test_pytest_checkpoint_does_not_freeze_a_stale_remote_pr_head():
    note = finalize_correction.pytest_checkpoint_note()

    assert "150 passed on 2026-11-16" in note
    assert "current remote commit" in note
    assert "ca17129f0e2e1ae3b53053e34b2dbbb76948b0fe" not in note
