"""L06 20K-expansion native-evidence re-verification CLI.

Recomputes a task's verdict from the pinned GO index + GOA evidence index
without trusting any stored evidence: rebuilds the GroundedTask from the
oracle payload's AST, re-derives the public view, re-runs the verifier gates
(prove_exists with the owlrl dual-leg / certify_semantic_relevance /
verify_terminal on both the reference and the wrong submission), and
compares the result with the self-check stored at build time. The GAF
evidence split is rebuilt from the pinned annotations file — the AST never
carries evidence data.

Usage (from the repo root):

    .venv/bin/python -m benchmark.expansion.l06.reverify20k \
        --oracle-file bank/staging_20k/l06/oracle/<handle>.json

Exit code 0 iff every recomputed verdict matches the stored self-check.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmark.expansion.l06.build20k import (
    _reference_submission20k,
    _wrong_submission20k,
)
from benchmark.expansion.l06.families20k import BY_ID_20K
from benchmark.expansion.l06.propose20k import L06ExpansionProposer
from benchmark.expansion.l06.verify20k import L06ExpansionVerifier
from benchmark.plugins.verifier.l06.fragment import induce
from benchmark.schema.plugin_protocols import CandidateTaskAST
from source_registry.fetchers.l06_go import RELEASE_ID

DEFAULT_INDEX = Path("external_knowledge/indexes/gene-ontology") / RELEASE_ID


def reverify20k(oracle_file: Path, index_dir: Path) -> dict[str, Any]:
    payload = json.loads(Path(oracle_file).read_text(encoding="utf-8"))
    ast = payload["ast"]
    spec = BY_ID_20K[payload["family_id"]]
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
    proposer = L06ExpansionProposer(index_dir)
    verifier = L06ExpansionVerifier(index_dir, index=proposer.index,
                                     evidence=proposer.evidence)
    grounded = proposer.ground(cand)
    frag = induce(ast["universe_ids"], verifier.graph)
    ref_sub = _reference_submission20k(grounded, verifier, frag)
    proof = verifier.prove_exists(grounded)
    witness = verifier.certify_semantic_relevance(grounded)
    pos = verifier.verify_terminal(grounded, ref_sub)
    neg = verifier.verify_terminal(
        grounded, _wrong_submission20k(grounded, verifier, frag))
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
    ap = argparse.ArgumentParser(
        description="Re-verify one L06 20K-expansion oracle payload")
    ap.add_argument("--oracle-file", required=True)
    ap.add_argument("--index", default=str(DEFAULT_INDEX))
    args = ap.parse_args(argv)
    result = reverify20k(Path(args.oracle_file), Path(args.index))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verdict"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
