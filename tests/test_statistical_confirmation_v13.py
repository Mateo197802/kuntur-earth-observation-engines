from __future__ import annotations

import json
from pathlib import Path

from scripts.statistical_correction.run_corrective_confirmation_v1_3 import (
    planned_seed_inventory,
    seed_for,
)
from scripts.statistical_correction.summarize_corrective_confirmation_v1_3 import (
    cp_interval,
    mean_summary,
    summarize_cell,
)


ROOT = Path(__file__).resolve().parents[1]


def _cell(slope: float, rejections: int = 50, coverage: int = 950) -> dict:
    rows = []
    for replicate in range(1000):
        rows.append({
            "replicate": replicate,
            "supported": True,
            "converged": True,
            "inferential_output": True,
            "slope_estimate_per_decade": slope,
            "p_value": 0.01 if replicate < rejections else 0.50,
            "reject_at_005": replicate < rejections,
            "ci_low_per_decade": slope - 0.1,
            "ci_high_per_decade": slope + 0.1,
            "ci_covers_true_slope": replicate < coverage,
            "ci_width": 0.2,
            "ci_width_over_oracle": 1.0,
            "residual_diagnostic": {
                "kind": "LJUNG_BOX_COMPLETE_SUPPORT",
                "ljung_box_failure": False,
                "p_values": {"12": 0.5, "24": 0.5},
            },
        })
    return {
        "cell": {"slope_per_decade": slope, "method_name": "SARIMAX_SEASONAL_AR12"},
        "replicates": rows,
    }


def test_confirmation_seed_inventory_is_unique_and_uses_reserved_namespace():
    config = json.loads((ROOT / "configs/statistical_corrective_confirmation_v1_3.json").read_text(encoding="utf-8"))
    inventory = planned_seed_inventory(config)
    assert inventory["seed_count"] == 6000
    assert inventory["unique_seed_count"] == 6000
    assert inventory["all_seeds_at_least_64_bit"] is True
    assert config["design"]["series_seed_namespace"] == "KUNTUR-STAT-CORRECTIVE-INDEPENDENT-CONFIRMATION-V1.3-2027-01-15"
    assert seed_for(config["design"]["series_seed_namespace"], "series", "iid_gaussian", "0.00000000", 0) != seed_for(
        "KUNTUR-STAT-CORRECTIVE-DEVELOPMENT-EXPANSION-V1.3-2026-11-17", "series", "iid_gaussian", "0.00000000", 0
    )


def test_exact_monte_carlo_intervals_cover_nominal_rates():
    type_i = cp_interval(50, 1000)
    coverage = cp_interval(950, 1000)
    assert type_i[0] <= 0.05 <= type_i[1]
    assert coverage[0] <= 0.95 <= coverage[1]


def test_continuous_metric_monte_carlo_interval_is_finite():
    result = mean_summary([0.9, 1.0, 1.1, 1.0])
    assert result is not None
    assert result["n"] == 4
    assert result["mcse"] > 0
    assert result["ci95_t"][0] < result["mean"] < result["ci95_t"][1]


def test_candidate_must_pass_all_three_predeclared_slope_cells():
    null = summarize_cell(_cell(0.0), 0.05, 0.95)
    plus_025 = summarize_cell(_cell(0.25, rejections=900), 0.05, 0.95)
    plus_050 = summarize_cell(_cell(0.5, rejections=980), 0.05, 0.95)
    assert null["all_gates_pass"] is True
    assert plus_025["all_gates_pass"] is True
    assert plus_050["all_gates_pass"] is True
    assert null["type_i_rate"] == 0.05
    assert plus_025["power"] == 0.9


def test_candidate_fails_when_null_type_i_rate_exceeds_frozen_gate():
    failed = summarize_cell(_cell(0.0, rejections=70), 0.05, 0.95)
    assert failed["acceptance_gates"]["type_i_rate"] is False
    assert failed["all_gates_pass"] is False


def test_candidate_fails_when_residual_dependence_gate_exceeds_five_percent():
    cell = _cell(0.25, rejections=900)
    for row in cell["replicates"][:60]:
        row["residual_diagnostic"]["ljung_box_failure"] = True
    failed = summarize_cell(cell, 0.05, 0.95)
    assert failed["residual_ljung_box_failure_rate"] == 0.06
    assert failed["acceptance_gates"]["residual_dependence"] is False
    assert failed["all_gates_pass"] is False


def test_missing_inferential_outputs_cannot_be_counted_as_valid_inference():
    cell = _cell(0.25, rejections=900)
    for row in cell["replicates"][:12]:
        row["converged"] = False
        row["inferential_output"] = False
        row["ci_low_per_decade"] = None
        row["ci_high_per_decade"] = None
        row["ci_width"] = None
        row["ci_width_over_oracle"] = None
    result = summarize_cell(cell, 0.05, 0.95)
    assert result["inferential_availability"] == 0.988
    assert result["acceptance_gates"]["inferential_availability"] is False
    assert result["all_gates_pass"] is False
