import pandas as pd

from kuntur_data.phase2.inamhi_checkpoints import load_year_checkpoint, write_year_checkpoint


def _frame(year: int, source_sha: str) -> pd.DataFrame:
    return pd.DataFrame([
        {"year": year, "station_id": "M001", "month_number": 1, "source_pdf_sha256": source_sha, "tavg_c": 22.0},
        {"year": year, "station_id": "M001", "month_number": 2, "source_pdf_sha256": source_sha, "tavg_c": 22.5},
    ])


def test_year_checkpoint_round_trips_and_requires_matching_source(tmp_path):
    source_sha = "a" * 64
    checkpoint_dir = tmp_path / "data" / "external" / "processed" / "inamhi_yearbook_checkpoints_v1"
    manifest_path = tmp_path / "data" / "manifests" / "external" / "inamhi_yearbook_checkpoints_v1.json"
    source = {"path": "data/external/raw/inamhi/Am-2002.pdf", "sha256": source_sha}
    record = write_year_checkpoint(2002, _frame(2002, source_sha), source_record=source, extraction={"pages": 2}, root=tmp_path, checkpoint_dir=checkpoint_dir, manifest_path=manifest_path)

    loaded = load_year_checkpoint(2002, expected_source_sha256=source_sha, root=tmp_path, checkpoint_dir=checkpoint_dir, manifest_path=manifest_path)
    assert loaded is not None
    recovered, recovered_record = loaded
    assert len(recovered) == 2
    assert recovered_record["sha256"] == record["sha256"]
    assert load_year_checkpoint(2002, expected_source_sha256="b" * 64, root=tmp_path, checkpoint_dir=checkpoint_dir, manifest_path=manifest_path) is None


def test_empty_year_checkpoint_is_reusable_with_source_provenance(tmp_path):
    source_sha = "c" * 64
    checkpoint_dir = tmp_path / "data" / "external" / "processed" / "inamhi_yearbook_checkpoints_v1"
    manifest_path = tmp_path / "data" / "manifests" / "external" / "inamhi_yearbook_checkpoints_v1.json"
    source = {"path": "data/external/raw/inamhi/Am-2013.pdf", "sha256": source_sha}
    write_year_checkpoint(2013, pd.DataFrame(columns=["year", "station_id", "month_number", "source_pdf_sha256"]), source_record=source, extraction={"pages": 1}, root=tmp_path, checkpoint_dir=checkpoint_dir, manifest_path=manifest_path)

    loaded = load_year_checkpoint(2013, expected_source_sha256=source_sha, root=tmp_path, checkpoint_dir=checkpoint_dir, manifest_path=manifest_path)
    assert loaded is not None
    assert loaded[0].empty
    assert loaded[1]["status"] == "VALIDATED_NO_BBOX_ROWS"
