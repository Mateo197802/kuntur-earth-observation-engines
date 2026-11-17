from __future__ import annotations

import hashlib

from scripts.statistical_correction.export_interval_confirmation_public_v1_3_2 import (
    _atomic_write,
    _git_text_sha256,
    _public_cell,
)


def test_public_cell_keeps_aggregate_metrics_and_drops_replicate_identifiers() -> None:
    source = {
        "method": "wald_student_t",
        "slope_per_decade": 0.5,
        "coverage_rate": 0.95,
        "coverage_cp95": [0.93, 0.97],
        "series_seed": "12345678901234567890",
        "replicate": 17,
        "residual_diagnostic": {"p_values": {"12": 0.5}},
        "warnings": ["replicate-specific detail"],
    }
    public = _public_cell(source)
    assert public["method"] == "wald_student_t"
    assert public["coverage_rate"] == 0.95
    assert "series_seed" not in public
    assert "replicate" not in public
    assert "residual_diagnostic" not in public
    assert "warnings" not in public


def test_public_hash_matches_lf_normalized_git_text_and_export_is_lf(tmp_path) -> None:
    source = tmp_path / "source.py"
    source.write_bytes(b"first\r\nsecond\r\n")
    assert _git_text_sha256(source) == hashlib.sha256(b"first\nsecond\n").hexdigest()

    output = tmp_path / "artifact.json"
    _atomic_write(output, "{\n  \"ok\": true\n}\n")
    assert output.read_bytes() == b"{\n  \"ok\": true\n}\n"
