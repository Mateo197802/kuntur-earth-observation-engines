from pathlib import Path

import pytest

from scripts.prepare_phase2_v12_sealed_2025_gold import ROOT, _manifest_path


def test_gold_receipt_manifest_path_resolves_inside_repository():
    path = _manifest_path("data/manifests/gold_manifest_v1.json", Path("unused.json"))
    assert path == ROOT / "data/manifests/gold_manifest_v1.json"


def test_gold_receipt_manifest_path_rejects_paths_outside_repository(tmp_path):
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inside the KUNTUR repository"):
        _manifest_path(str(outside), Path("unused.json"))
