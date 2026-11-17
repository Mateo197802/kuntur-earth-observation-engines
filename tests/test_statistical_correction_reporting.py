import json
from pathlib import Path

import pandas as pd

from scripts.statistical_correction import finalize_correction


ROOT = Path(__file__).resolve().parents[1]


def test_root_cause_matrix_keeps_frozen_sta_ids_attached_to_their_defects():
    findings = [
        {"id": finding_id, "root_cause": f"fixture root cause for {finding_id}"}
        for finding_id in (
            "STA-01",
            "STA-02",
            "STA-03",
            "STA-04",
            "STA-05",
            "STA-06",
            "STA-07",
        )
    ]

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

    assert "152 passed on 2026-11-16" in note
    assert "current remote commit" in note
    assert "ca17129f0e2e1ae3b53053e34b2dbbb76948b0fe" not in note


def test_modis_spatial_receipt_reports_remote_native_grid_evidence_without_private_paths():
    receipt = {
        "status": "VALIDATED_ON_CEDIA",
        "coverage": {"start_month": "2002-01", "end_month": "2024-12", "month_count": 276},
        "validation": {
            "manifest_validation_status": "PASSED",
            "source_granules_referenced": 276,
            "regional_files_verified": 276,
            "monthly_summary_rows": 1656,
        },
        "spatial_grid": {"native_grid": "MODIS CMG geographic 0.05 degree; no interpolation or resampling"},
        "qa_eligibility": {"strict_months_at_10pct": 144, "strict_months_at_20pct": 48},
        "spatial_support": {
            "strict": {
                "stable_pixel_count": 0,
                "union_area_fraction_bbox": 0.7984141351641939,
                "early_late_jaccard_area": 0.7588053463215233,
                "early_late_area_fraction_changed_by_10pp": 0.05343198940517315,
            }
        },
        "artifacts": {
            "manifest_sha256": "a" * 64,
            "monthly_summary_sha256": "b" * 64,
            "lineage_sha256": "c" * 64,
            "spatial_support_sha256": "d" * 64,
            "acquisition_manifest_sha256": "e" * 64,
            "regional_manifest_sha256": "f" * 64,
            "manifest_path": "results/statistical_correction/modis_spatial_selection_v1_1/modis_spatial_selection_manifest_v1_1.json",
            "monthly_summary_path": "results/statistical_correction/modis_spatial_selection_v1_1/modis_spatial_selection_monthly_v1_1.csv",
            "lineage_path": "results/statistical_correction/modis_spatial_selection_v1_1/modis_spatial_selection_lineage_v1_1.json",
            "spatial_support_path": "results/statistical_correction/modis_spatial_selection_v1_1/modis_spatial_support_summary_v1_1.csv",
        },
    }

    note = finalize_correction.modis_spatial_selection_receipt_note(receipt)

    assert "native MOD13C2.061 CMG 0.05-degree grid" in note
    assert "144/276" in note
    assert "48/276" in note
    assert "stable-pixel count of 0" in note
    assert "0.758805" in note
    assert "/home/" not in note
    assert "regional_manifest_sha256" not in note


def test_finalizer_records_verified_git_source_commit(monkeypatch):
    expected = "a" * 40
    calls = {}

    def fake_check_output(args, *, cwd, text):
        calls["args"] = args
        calls["cwd"] = cwd
        calls["text"] = text
        return expected + "\n"

    monkeypatch.setattr(finalize_correction.subprocess, "check_output", fake_check_output)

    assert finalize_correction.git_source_commit() == expected
    assert calls["args"][-2:] == ["rev-parse", "HEAD"]
    assert calls["cwd"] == finalize_correction.ROOT
    assert calls["text"] is True
