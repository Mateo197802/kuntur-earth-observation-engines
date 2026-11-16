from pathlib import Path

import pytest

import scripts.prepare_phase2_v12_sealed_2025_gold as preparation


def test_gold_receipt_manifest_path_resolves_inside_repository(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "ROOT", tmp_path)
    manifest = tmp_path / "data/manifests/gold_manifest_v1.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    path = preparation._manifest_path("data/manifests/gold_manifest_v1.json", Path("unused.json"))
    assert path == manifest


def test_gold_receipt_manifest_path_rejects_paths_outside_repository(tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the KUNTUR repository"):
        preparation._manifest_path(str(outside), Path("unused.json"))
