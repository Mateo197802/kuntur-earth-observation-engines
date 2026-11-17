from __future__ import annotations

import json
from pathlib import Path

from scripts.statistical_correction.freeze_interval_confirmation_v1_3_2 import ROOT, seed_inventory
from scripts.statistical_correction.summarize_interval_confirmation_v1_3_2 import cp_interval, summarize_cell


def _criteria() -> dict:
    return json.loads((ROOT / "configs/statistical_interval_confirmation_v1_3_2.json").read_text(encoding="utf-8"))["acceptance_criteria"]


def _rows(slope: float, *, reject_count: int = 5, coverage_count: int = 95, residual_failures: int = 5) -> list[dict]:
    result = []
    for i in range(100):
        covers = i < coverage_count
        residual_failure = i < residual_failures
        result.append({
            "supported": True,
            "inferential_output": True,
            "converged": True,
            "slope_estimate_per_decade": slope,
            "p_value": 0.001 if i < reject_count else 0.5,
            "ci_low_per_decade": slope - 0.5 if covers else slope + 0.01,
            "ci_high_per_decade": slope + 0.5,
            "ci_covers_true_slope": covers,
            "ci_width": 1.0,
            "ci_width_over_oracle": 1.0,
            "residual_diagnostic": {"kind": "LJUNG_BOX_COMPLETE_SUPPORT", "ljung_box_failure": residual_failure},
        })
    return result


def test_confirmation_seed_namespace_is_unique_and_disjoint_from_v132_dev() -> None:
    config = json.loads((ROOT / "configs/statistical_interval_confirmation_v1_3_2.json").read_text(encoding="utf-8"))
    inventory = seed_inventory(config)
    assert inventory["confirmation_seed_count"] == 3000
    assert inventory["confirmation_unique_seed_count"] == 3000
    assert inventory["no_overlap_with_v1_3_2_preflight_or_development"] is True
    assert len(inventory["confirmation_seed_inventory_sha256"]) == 64


def test_exact_clopper_pearson_interval_contains_rate_at_null_gate() -> None:
    low, high = cp_interval(5, 100)
    assert low is not None and high is not None
    assert low < 0.05 < high


def test_all_frozen_gates_pass_when_each_synthetic_cell_meets_criteria() -> None:
    criteria = _criteria()
    null = summarize_cell(_rows(0.0), 0.0, criteria)
    positive = summarize_cell(_rows(0.25, reject_count=100), 0.25, criteria)
    assert null["all_gates_pass"] is True
    assert null["acceptance_gates"]["type_i_rate"] is True
    assert positive["all_gates_pass"] is True
    assert positive["acceptance_gates"]["type_i_rate"] is None
    assert positive["power"] == 1.0


def test_confirmation_fails_when_coverage_point_gate_fails() -> None:
    failed = summarize_cell(_rows(0.5, coverage_count=90), 0.5, _criteria())
    assert failed["acceptance_gates"]["coverage_rate_and_exact_interval"] is False
    assert failed["all_gates_pass"] is False
