"""Reproduce one L15 native-evidence check from an oracle payload.

The command recorded in every qualified record's ``native_evidence.commands``::

    .venv/bin/python -m benchmark.plugins.task_family.l15.verify_cli \
        --oracle bank/staging_20k/l15/oracle/orl15_<hash>.json

Recomputes the reference answer from the pinned store (never from the stored
reference), then checks: stored reference equals fresh recomputation;
``verify_terminal(stored_reference)`` passes; the stored mutation fails.
Exit code 0 iff every check passes. Output is JSON on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from benchmark.plugins.task_family.l15.families import BY_ID, LANE, PROFILE
from benchmark.plugins.verifier.l15.verifier import ReactomeNativeVerifier, _canon
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask


def _grounded(payload: dict) -> GroundedTask:
    ast = payload["ast"]
    family = payload.get("family_id", "")
    spec = BY_ID.get(family)
    cand = CandidateTaskAST(
        lane_id=LANE,
        family_id=family,
        base_goal_id=payload.get("ast", {}).get("base_goal_id", "reconstructed"),
        instance_id=payload["instance_id"],
        regime=spec.regime if spec else "application",
        semantic_profile=PROFILE,
        ast=ast,
        source_unit_ids=(),
    )
    return GroundedTask(ast=cand, public={}, oracle_ref=payload["oracle_handle"],
                        native_evidence={})


def verify_oracle(oracle_path: Path) -> dict:
    payload = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    proof = payload.get("proof") or {}
    m_only = (proof.get("differential_agree") == "NOT_RUN:biopax_snapshot_unavailable")
    verifier = ReactomeNativeVerifier(allow_m_only=m_only)
    grounded = _grounded(payload)
    checks: dict = {}
    fresh = verifier.reference_submission(payload["ast"])
    checks["reference_matches_fresh"] = _canon(fresh) == _canon(payload["reference_submission"])
    positive = verifier.verify_terminal(grounded, payload["reference_submission"])
    checks["positive_accepted"] = bool(positive.valid)
    mutated = payload.get("mutation_submission")
    if mutated is not None:
        negative = verifier.verify_terminal(grounded, mutated)
        checks["mutation_rejected"] = not negative.valid
    else:
        checks["mutation_rejected"] = False
    proof = verifier.prove_exists(grounded)
    checks["prove_exists"] = bool(proof.exists)
    ok = all(checks.values())
    return {"ok": ok, "checks": checks, "task_id": payload.get("task_id"),
            "family_id": payload.get("family_id")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    args = ap.parse_args()
    result = verify_oracle(Path(args.oracle))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
