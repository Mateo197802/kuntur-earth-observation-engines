from __future__ import annotations

import pandas as pd

from scripts import acquire_inamhi_yearbooks as acquisition


def test_resume_planning_loads_each_validated_checkpoint_once(monkeypatch, tmp_path):
    source_records = {
        "2007": {"sha256": "a" * 64},
        "2008": {"sha256": "b" * 64},
        "2009": {"sha256": "invalid"},
    }
    frame = pd.DataFrame([{"year": 2007, "station_id": "M001"}])
    expected_record = {"status": "VALIDATED", "rows": 1}
    valid_checkpoint = (frame, expected_record)
    calls: list[tuple[int, str]] = []

    def fake_load(year, *, expected_source_sha256, root, checkpoint_dir, manifest_path):
        calls.append((year, expected_source_sha256))
        return valid_checkpoint if year == 2007 else None

    monkeypatch.setattr(acquisition, "load_year_checkpoint", fake_load)

    cached = acquisition.load_validated_year_checkpoints(
        range(2007, 2010),
        source_records,
        root=tmp_path,
        checkpoint_dir=tmp_path / "checkpoints",
        manifest_path=tmp_path / "manifest.json",
    )

    assert calls == [(2007, "a" * 64), (2008, "b" * 64)]
    assert set(cached) == {2007}
    assert cached[2007][0] is frame
    assert cached[2007][1] is expected_record
