"""Minimal public-checkout support for tests that reference local-only freezes."""

from __future__ import annotations

import json
from pathlib import Path


def pytest_sessionstart(session) -> None:
    """Provide only the public development seed hash when private run files are absent.

    The frozen seed-inventory helper recomputes the confirmation seeds and checks
    them against the public preflight/development namespaces. It also reads the
    recorded development inventory hash from a local freeze receipt. Public CI
    intentionally does not include that private receipt, so create a clearly
    marked metadata-only shim from the already-published aggregate audit. This
    is not a protocol freeze and contains no replicate-level seed values.
    """
    repo = Path(__file__).resolve().parents[1]
    local_freeze = repo / "results/statistical_correction/interval_comparison_v1_3_2/protocol_freeze.json"
    if local_freeze.exists():
        return

    public_metrics = repo / "results/phase2/public/statistical_validation_v1_3_2.json"
    public = json.loads(public_metrics.read_text(encoding="utf-8"))
    inventory_hash = public["v1_3_2_independent_confirmation"]["seed_disjointness_audit"][
        "v1_3_2_seed_inventory_sha256"
    ]
    if not isinstance(inventory_hash, str) or len(inventory_hash) != 64:
        raise RuntimeError("Public aggregate evidence lacks the v1.3.2 development seed inventory hash")

    local_freeze.parent.mkdir(parents=True, exist_ok=True)
    shim = {
        "test_fixture_kind": "PUBLIC_AGGREGATE_REFERENCE_ONLY_NOT_A_PROTOCOL_FREEZE",
        "seed_inventory": {"seed_inventory_sha256": inventory_hash},
    }
    try:
        with local_freeze.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(shim, stream, sort_keys=True)
            stream.write("\n")
    except FileExistsError:
        # Another test runner created the same ephemeral support file first.
        return
