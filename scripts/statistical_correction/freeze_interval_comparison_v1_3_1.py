from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from .provenance_interval_comparison_v1_3_1 import FREEZE_PATH, ROOT, RESULTS, sha256_file, source_hashes
except ImportError:  # direct `python scripts/.../freeze_*.py` execution
    from provenance_interval_comparison_v1_3_1 import FREEZE_PATH, ROOT, RESULTS, sha256_file, source_hashes


CONFIG_PATH = ROOT / "configs/statistical_interval_comparison_v1_3_1.json"


def seed_for(namespace: str, slope: float, replicate: int) -> int:
    text = f"{namespace}|series|iid_gaussian|{slope:.8f}|{replicate}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:16], "big", signed=False)


def seed_inventory(config: dict) -> dict:
    design = config["design"]
    preflight = [
        seed_for(design["preflight_seed_namespace"], float(slope), replicate)
        for slope in design["slopes_per_decade"]
        for replicate in range(int(design["preflight_replicates_per_slope"]))
    ]
    development = [
        seed_for(design["series_seed_namespace"], float(slope), replicate)
        for slope in design["slopes_per_decade"]
        for replicate in range(int(design["development_replicates_per_slope"]))
    ]
    combined = preflight + development
    if len(combined) != len(set(combined)):
        raise RuntimeError("Preflight and development seed inventories overlap")
    canonical = "\n".join(str(seed) for seed in combined).encode("ascii")
    return {
        "preflight_series_seed_count": len(preflight),
        "development_series_seed_count": len(development),
        "all_series_seed_count": len(combined),
        "unique_series_seed_count": len(set(combined)),
        "all_series_seeds_unique_across_new_namespaces": True,
        "seed_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def build_payload() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    hashes = source_hashes()
    closure = json.loads((ROOT / "results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json").read_text(encoding="utf-8"))
    if closure.get("closure_status") != "PARTIAL_EXPERIMENT_CLOSED":
        raise RuntimeError("Authoritative v1.2 partial-closure manifest status is unexpected")
    registry_hash = hashes["results/statistical_correction/validated_regime_registry.json"]
    expected_registry = json.loads((ROOT / "results/statistical_correction/corrective_development_v1_3/selection_freeze_iid_v1_3_r1.json").read_text(encoding="utf-8"))
    expected_registry_hash = expected_registry.get("protected_history", {}).get("v1_2_registry_sha256")
    if expected_registry_hash and registry_hash != expected_registry_hash:
        raise RuntimeError("v1.2 inference registry differs from its preserved freeze hash")
    return {
        "schema_version": "kuntur-statistical-protocol-freeze-v1.3.1",
        "protocol_id": config["protocol_id"],
        "status": "FROZEN_BEFORE_PREFLIGHT_AND_DEVELOPMENT",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "source_hashes": hashes,
        "config_sha256": sha256_file(CONFIG_PATH),
        "seed_inventory": seed_inventory(config),
        "protected_historical_evidence": {
            "v1_3_confirmation_manifest_sha256": hashes["results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_manifest.json"],
            "v1_3_confirmation_metrics_sha256": hashes["results/statistical_correction/independent_confirmation_v1_3_iid/confirmation_metrics.json"],
            "v1_3_608_fit_pilot_seed_namespace_reused": False,
            "v1_3_180000_fit_expansion_reused": False,
            "v1_2_registry_sha256": registry_hash,
            "authoritative_v1_2_partial_closure_manifest_sha256": hashes["results/statistical_correction/validation/partial_validation_closure_v1_2_20261116.json"],
            "prior_confirmation_checkpoints": {
                Path(path).name: digest
                for path, digest in hashes.items()
                if "/checkpoints/" in path.replace("\\", "/")
            },
        },
        "environment_at_freeze": {
            "python": sys.version,
            "platform": platform.platform(),
        },
    }


def verify_payload(payload: dict) -> dict:
    current = source_hashes()
    frozen = payload.get("source_hashes", {})
    if current != frozen:
        changed = sorted(set(current) | set(frozen))
        changed = [path for path in changed if current.get(path) != frozen.get(path)]
        raise RuntimeError("Frozen protocol/code/evidence changed: " + ", ".join(changed))
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload.get("config_sha256") != sha256_file(CONFIG_PATH):
        raise RuntimeError("Frozen protocol configuration hash mismatch")
    if payload.get("seed_inventory") != seed_inventory(config):
        raise RuntimeError("Frozen seed inventory mismatch")
    if payload.get("status") != "FROZEN_BEFORE_PREFLIGHT_AND_DEVELOPMENT":
        raise RuntimeError("Unexpected freeze status")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="verify an existing immutable freeze")
    args = parser.parse_args()
    if args.verify:
        payload = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        verify_payload(payload)
        print(json.dumps({"status": "FREEZE_VERIFIED", "freeze_sha256": sha256_file(FREEZE_PATH), "source_count": len(payload["source_hashes"])}, indent=2))
        return
    if FREEZE_PATH.exists():
        raise SystemExit(f"Freeze already exists and is immutable: {FREEZE_PATH.relative_to(ROOT)}")
    payload = build_payload()
    RESULTS.mkdir(parents=True, exist_ok=True)
    temporary = FREEZE_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(FREEZE_PATH)
    print(json.dumps({"status": payload["status"], "freeze_sha256": sha256_file(FREEZE_PATH), "source_count": len(payload["source_hashes"]), "seed_inventory": payload["seed_inventory"]}, indent=2))


if __name__ == "__main__":
    main()
