"""L06 native-evidence re-verification CLI.

Recomputes a task's verdict from the pinned GO index + GOA snapshot without
trusting any stored evidence: rebuilds the GroundedTask from the oracle
payload's AST, re-derives the public view, re-runs the three verifier gates
(prove_exists / certify_semantic_relevance / verify_terminal on both the
reference and the wrong submission), and compares the result with the
self-check stored at build time.

Usage (from the repo root):

    .venv/bin/python -m benchmark.plugins.verifier.l06.reverify \
        --oracle-file bank/staging/l06/oracle/<handle>.json

Exit code 0 iff every recomputed verdict matches the stored self-check.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l06.build import (
    _reference_submission,
    _wrong_submission,
)
from benchmark.plugins.task_family.l06.families import BY_ID
from benchmark.plugins.task_family.l06.propose import L06TaskFamilyPlugin
from benchmark.plugins.verifier.l06.fragment import induce
from benchmark.plugins.verifier.l06.verifier import L06NativeVerifier
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l06_go import RELEASE_ID

DEFAULT_INDEX = Path("external_knowledge/indexes/gene-ontology") / RELEASE_ID


def reverify(oracle_file: Path, index_dir: Path) -> dict[str, Any]:
    payload = json.loads(Path(oracle_file).read_text(encoding="utf-8"))
    ast = payload["ast"]
    spec = BY_ID[payload["family_id"]]
    cand = CandidateTaskAST(
        lane_id=payload["lane_id"],
        family_id=payload["family_id"],
        base_goal_id=payload["base_goal_id"],
        instance_id=payload["instance_id"],
        regime=payload["regime"],
        semantic_profile=payload["semantic_profile"],
        ast=ast,
        source_unit_ids=tuple(payload["source_unit_ids"]),
    )
    plugin = L06TaskFamilyPlugin(index_dir)
    verifier = L06NativeVerifier(index_dir, index=plugin.index)
    grounded = plugin.ground(cand)
    frag = induce(ast["universe_ids"], verifier.graph)
    ref_sub = _reference_submission(grounded, verifier, frag)
    proof = verifier.prove_exists(grounded)
    witness = verifier.certify_semantic_relevance(grounded)
    pos = verifier.verify_terminal(grounded, ref_sub)
    neg = verifier.verify_terminal(grounded, _wrong_submission(grounded, verifier, frag))
    stored = payload["self_check"]
    handle_ok = grounded.oracle_ref == payload["oracle_handle"]
    ref_ok = ref_sub == payload["reference_submission"]
    out = {
        "oracle_handle": payload["oracle_handle"],
        "instance_id": payload["instance_id"],
        "family_id": payload["family_id"],
        "release_id": RELEASE_ID,
        "recomputed": {
            "prove_exists": proof.exists,
            "witness_ok": witness.ok,
            "positive_valid": pos.valid,
            "negative_valid": neg.valid,
        },
        "stored_self_check": stored,
        "reference_submission_matches": ref_ok,
        "oracle_handle_matches": handle_ok,
    }
    out["verdict"] = bool(
        proof.exists and witness.ok and pos.valid and not neg.valid
        and ref_ok and handle_ok
        and stored["positive"] is True and stored["negative_valid"] is False)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Re-verify one L06 oracle payload")
    ap.add_argument("--oracle-file", required=True)
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    args = ap.parse_args(argv)
    result = reverify(Path(args.oracle_file), Path(args.index))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
