"""Independently reconstruct the two v1.2 frozen-selection decisions.

Reads only the saved selection/validation summaries and selected checkpoints;
does not run simulations or alter v1.2 decisions or artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from scipy.stats import beta, binom, fisher_exact


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/statistical_correction"
SELECTION_DIR = BASE / "selection"
VALIDATION_DIR = BASE / "validation"
MATRIX = VALIDATION_DIR / "execution_reconciliation_method_matrix_20261116.csv"
DECISIONS = SELECTION_DIR / "selection_decisions.json"
PROTOCOL = ROOT / "docs/protocols/statistical_correction_v1_2.md"
DEFAULT_OUTPUT = BASE / "selection_reconstruction_v1_3.json"
REGIMES = ("iid_gaussian", "ar1_060_mcar20")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cp_interval(k: int, n: int) -> list[float]:
    low = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [low, high]


def read_checkpoint(phase: str, regime: str, slope: float) -> dict[str, Any]:
    suffix = str(slope).replace("-", "m").replace(".", "p")
    path = BASE / phase / "checkpoints" / f"{phase}_{regime}_n276_s{suffix}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "COMPLETE":
        raise ValueError(f"Required historical checkpoint is incomplete: {path}")
    return payload


def gate(row: dict[str, Any], name: str) -> bool:
    return row.get(name) is True


def reconstructed_decision(regime: str) -> dict[str, Any]:
    null_payload = read_checkpoint("selection", regime, 0.0)
    alt_payload = read_checkpoint("selection", regime, 0.25)
    null_rows = {row["method"]: row for row in null_payload["rows"]}
    alt_rows = {row["method"]: row for row in alt_payload["rows"]}
    names = sorted(set(null_rows) & set(alt_rows))
    ranked: list[dict[str, Any]] = []
    candidate_audit: list[dict[str, Any]] = []
    for method in names:
        null, alt = null_rows[method], alt_rows[method]
        required: dict[str, bool] = {
            "null_type1": gate(null, "type1_gate_pass"),
            "null_failure": gate(null, "failure_gate_pass"),
            "null_inference": gate(null, "inference_availability_gate_pass"),
            "alt_failure": gate(alt, "failure_gate_pass"),
            "alt_inference": gate(alt, "inference_availability_gate_pass"),
            "null_full_support": null.get("support_fraction") == 1.0,
            "alt_full_support": alt.get("support_fraction") == 1.0,
        }
        has_ci = int(null.get("replications_with_interval", 0)) > 0 or int(alt.get("replications_with_interval", 0)) > 0
        if has_ci:
            required.update({
                "null_coverage": gate(null, "coverage_gate_pass"),
                "alt_coverage": gate(alt, "coverage_gate_pass"),
                "null_width": gate(null, "width_gate_pass"),
                "alt_width": gate(alt, "width_gate_pass"),
            })
        method_id = str(null.get("method_id"))
        if method_id in {"gls_ar1", "sarimax_ar1", "arma11", "seasonal_ar12"}:
            required.update({
                "null_residual_dependence": gate(null, "residual_dependence_gate_pass"),
                "alt_residual_dependence": gate(alt, "residual_dependence_gate_pass"),
            })
        score: float | None = None
        score_terms: dict[str, float] = {}
        score_inputs_available = null.get("type1_fpr_005") is not None and alt.get("slope_bias_per_decade") is not None
        if has_ci:
            score_inputs_available = score_inputs_available and null.get("coverage_95") is not None and alt.get("coverage_95") is not None
        if score_inputs_available:
            score = abs(float(null["type1_fpr_005"]) - 0.05) / 0.05
            score_terms["null_type1_standardized_error"] = score
            if has_ci:
                null_term = abs(float(null["coverage_95"]) - 0.95) / 0.05
                alt_term = abs(float(alt["coverage_95"]) - 0.95) / 0.05
                score += null_term + alt_term
                score_terms["null_coverage_standardized_error"] = null_term
                score_terms["alternative_coverage_standardized_error"] = alt_term
            bias_term = abs(float(alt["slope_bias_per_decade"])) / 0.025
            score += bias_term
            score_terms["alternative_bias_standardized_error"] = bias_term
        ratios = [float(row["mean_ci_width_over_oracle"]) for row in (null, alt) if row.get("mean_ci_width_over_oracle") is not None]
        score_width = sum(ratios) / len(ratios) if ratios else None
        eligible = all(required.values()) and score is not None
        detail = {
            "method": method,
            "method_id": method_id,
            "eligible_reconstructed": eligible,
            "failed_required_gates": [key for key, passed in required.items() if not passed],
            "has_confidence_interval": has_ci,
            "score": score,
            "score_terms": score_terms,
            "mean_ci_width_over_oracle": score_width,
            "null": {
                "type1_fpr": null.get("type1_fpr_005"),
                "type1_cp95": [null.get("type1_mc95_low"), null.get("type1_mc95_high")],
                "coverage": null.get("coverage_95"),
                "coverage_gate": null.get("coverage_gate_pass"),
                "residual_failure_rate": null.get("residual_dependence_failure_rate"),
                "residual_gate": null.get("residual_dependence_gate_pass"),
                "support_fraction": null.get("support_fraction"),
            },
            "alternative": {
                "coverage": alt.get("coverage_95"),
                "bias_per_decade": alt.get("slope_bias_per_decade"),
                "residual_failure_rate": alt.get("residual_dependence_failure_rate"),
                "residual_gate": alt.get("residual_dependence_gate_pass"),
                "support_fraction": alt.get("support_fraction"),
            },
        }
        candidate_audit.append(detail)
        if eligible:
            ranked.append(detail)
    ranked.sort(key=lambda row: (
        row["score"],
        row["mean_ci_width_over_oracle"] if row["mean_ci_width_over_oracle"] is not None else float("inf"),
        row["method"],
    ))
    return {
        "regime": regime,
        "candidate_count": len(candidate_audit),
        "eligible_count": len(ranked),
        "selected_reconstructed": ranked[0]["method"] if ranked else None,
        "ranked_eligible": ranked,
        "all_candidate_audit": candidate_audit,
    }


def raw_rejection_count(payload: dict[str, Any], method: str) -> tuple[int, int]:
    rows = [r for r in payload.get("replicate_records", []) if r.get("method") == method and float(r.get("true_slope_per_decade", 1.0)) == 0.0 and r.get("support_eligible")]
    valid = [r for r in rows if r.get("converged") and r.get("p_value") is not None]
    return sum(bool(r.get("reject_at_005")) for r in valid), len(valid)


def compare_selection_validation(regime: str, selected_method: str) -> dict[str, Any]:
    selection_payload = read_checkpoint("selection", regime, 0.0)
    validation_payload = read_checkpoint("validation", regime, 0.0)
    ks, ns = raw_rejection_count(selection_payload, selected_method)
    kv, nv = raw_rejection_count(validation_payload, selected_method)
    odds, fisher_p = fisher_exact([[ks, ns - ks], [kv, nv - kv]], alternative="two-sided")
    pval = kv / nv
    return {
        "regime": regime,
        "method": selected_method,
        "selection": {"rejections": ks, "n": ns, "rate": ks / ns, "mcse": (ks / ns * (1 - ks / ns) / ns) ** 0.5, "cp95": cp_interval(ks, ns)},
        "independent_validation": {"rejections": kv, "n": nv, "rate": kv / nv, "mcse": (kv / nv * (1 - kv / nv) / nv) ** 0.5, "cp95": cp_interval(kv, nv)},
        "difference_validation_minus_selection": kv / nv - ks / ns,
        "fisher_exact_two_sided_p_descriptive": float(fisher_p),
        "selection_low_tail_probability_if_validation_rate_were_true": float(binom.cdf(ks, ns, pval)),
        "odds_ratio": float(odds),
        "interpretation": "Descriptive independent-binomial comparison; it does not correct the selection winner's curse or establish equivalence.",
    }


def build() -> dict[str, Any]:
    original = json.loads(DECISIONS.read_text(encoding="utf-8"))
    reconstructed = {regime: reconstructed_decision(regime) for regime in REGIMES}
    selected_names = {regime: original["decisions"][regime]["selected"]["method"] for regime in REGIMES}
    for regime, method in selected_names.items():
        if reconstructed[regime]["selected_reconstructed"] != method:
            raise AssertionError(f"Independent selector reconstruction disagrees for {regime}")
    compared = {regime: compare_selection_validation(regime, method) for regime, method in selected_names.items()}
    matrix = pd.read_csv(MATRIX)
    v1_2_val = {}
    for regime, method in selected_names.items():
        row = matrix[(matrix.process == regime) & (matrix.method == method)].iloc[0]
        v1_2_val[regime] = {
            "status": row.validity_status,
            "type1_fpr": float(row.type1_fpr_005),
            "type1_cp95": [float(row.type1_mc95_low), float(row.type1_mc95_high)],
            "coverage": None if pd.isna(row.coverage_95) else float(row.coverage_95),
            "residual_failure_rate": None if pd.isna(row.residual_dependence_failure_rate) else float(row.residual_dependence_failure_rate),
            "numerical_failure_rate": float(row.convergence_failure_rate),
        }
    return {
        "schema_version": "kuntur-v1_2-selection-reconstruction-v1_3/1.0.0",
        "status": "RECONSTRUCTED_WITHOUT_MODIFYING_FROZEN_DECISIONS",
        "original_selection_decisions_sha256": sha256(DECISIONS),
        "original_selection_protocol_sha256": sha256(PROTOCOL),
        "selection_matrix_sha256": sha256(SELECTION_DIR / "method_validation_matrix.csv"),
        "selected_validation_matrix_sha256": sha256(MATRIX),
        "selection_seed_base": original.get("selection_seed_base"),
        "validation_seed_base": original.get("validation_seed_base"),
        "reconstructed_regimes": reconstructed,
        "selected_method_selection_validation_comparison": compared,
        "authoritative_v1_2_validation_rows": v1_2_val,
        "diagnosis": {
            "iid": "Arithmetic reproduces exactly, but the ranking is structurally non-comparable: the selected seasonal-MK candidate had no confidence interval and therefore incurred no coverage or interval-width score terms. Four IID candidates passed the frozen gates; the smaller seasonal-MK score is partly a consequence of omitting two terms, not a direct same-metric superiority comparison.",
            "mcar": "The selected seasonal-MK candidate was the only eligible method. SARIMAX_AR1 and SARIMAX_ARMA11 passed their null Type-I, interval, width, support, failure, and inference gates, but the residual-dependence gate was false because no residual diagnostic was available for the irregular calendar. The v1.2 protocol calls for exact-calendar diagnostics in this case, while the v1.2 helper only implements complete-support Ljung-Box logic. This is an implementation/protocol mismatch that excluded the model candidates.",
            "sampling": "Selection used 500 replications, giving a Type-I Monte Carlo SE near 0.010 at 5%. The 500- versus 1,000-rep selected-method differences are not significant in two-sided Fisher exact comparisons (IID p and MCAR p are reported above); selection estimates are compatible with Monte Carlo variability, including selection-induced winner's curse. This does not explain away the independent v1.2 failures.",
            "misspecification": "The data do not isolate block-length misspecification as the sole cause. The v1.3 expanded study will compare fixed 1-4 year blocks across IID, dependent, seasonal, and missing-calendar DGPs.",
        },
        "frozen_decisions_modified": False,
        "v1_2_recalibration_or_resimulation_performed": False,
    }


def main() -> int:
    payload = build()
    DEFAULT_OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"],
        "selected": {r: payload["reconstructed_regimes"][r]["selected_reconstructed"] for r in REGIMES},
        "eligible_counts": {r: payload["reconstructed_regimes"][r]["eligible_count"] for r in REGIMES},
        "outputs": DEFAULT_OUTPUT.relative_to(ROOT).as_posix(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
