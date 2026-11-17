from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.statistical_correction.freeze_interval_confirmation_v1_3_2 import ROOT, seed_for

CONFIG_PATH = ROOT / "configs/statistical_interval_confirmation_v1_3_2.json"
RUN_DIR = ROOT / "results/statistical_correction/interval_confirmation_v1_3_2/confirmation"
OLD_EVIDENCE_ROOT = ROOT / "results/statistical_correction"
KNOWN_MALFORMED_METADATA = {
    "results/statistical_correction/corrective_development_v1_3/expanded_20261117/checkpoint_recovery_receipt.json": "a87fea7442dcc9755f928cf557f012f3be7a1f14e68916b58ffb65884c5471d4"
}
SEED_KEYS = {"series_seed", "estimator_seed", "replicate_seed", "rng_seed", "seed"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_seed_values(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SEED_KEYS and isinstance(item, (str, int)):
                text = str(item)
                if text.isdigit():
                    output.add(text)
            _collect_seed_values(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_seed_values(item, output)


def audit() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    design = config["design"]
    namespace = design["series_seed_namespace"]
    expected: set[str] = set()
    for slope in design["slopes_per_decade"]:
        for replicate in range(int(design["replicates_per_slope"])):
            expected.add(str(seed_for(namespace, float(slope), replicate)))

    actual: set[str] = set()
    for slope in design["slopes_per_decade"]:
        label = f"{float(slope):.2f}".replace("-", "m").replace(".", "p")
        checkpoint = RUN_DIR / "checkpoints" / f"cell_iid_gaussian_n276_s{label}_wald_student_t.json"
        payload = json.loads(checkpoint.read_text(encoding="utf-8"))
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"Confirmation checkpoint is incomplete: {checkpoint.name}")
        for row in payload.get("replicates", []):
            actual.add(str(row.get("series_seed")))
    if actual != expected:
        raise RuntimeError("Executed confirmation series seeds do not match the frozen seed inventory")

    old_values: set[str] = set()
    parsed_files = 0
    excluded_receipts: list[dict[str, str]] = []
    for path in OLD_EVIDENCE_ROOT.rglob("*.json"):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("results/statistical_correction/interval_confirmation_v1_3_2/"):
            continue
        expected_receipt_hash = KNOWN_MALFORMED_METADATA.get(relative)
        if expected_receipt_hash is not None:
            current = sha256_file(path)
            if current != expected_receipt_hash:
                raise RuntimeError("Known legacy metadata receipt changed; review it before accepting the seed audit")
            excluded_receipts.append({"sha256": current, "reason": "known malformed non-checkpoint recovery receipt; contains no seed rows"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unexpected unreadable prior JSON artifact: {relative}") from exc
        _collect_seed_values(payload, old_values)
        parsed_files += 1

    overlap = expected & old_values
    if overlap:
        raise RuntimeError(f"Confirmation seed overlap found: {len(overlap)}")
    return {
        "schema_version": "kuntur-v1.3.2-confirmation-seed-disjointness-audit-v1",
        "protocol_id": config["protocol_id"],
        "confirmation_seed_namespace": namespace,
        "confirmation_expected_seed_count": len(expected),
        "confirmation_unique_executed_seed_count": len(actual),
        "prior_json_artifacts_parsed": parsed_files,
        "prior_seed_values_found": len(old_values),
        "exact_overlap_count": 0,
        "known_nonseed_metadata_receipts_excluded": excluded_receipts,
        "status": "SEED_DISJOINTNESS_VERIFIED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="print a compact JSON result")
    args = parser.parse_args()
    result = audit()
    print(json.dumps(result, sort_keys=True, indent=2 if not args.json else None))


if __name__ == "__main__":
    main()
