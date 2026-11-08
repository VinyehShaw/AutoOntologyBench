"""Single-instance re-verification CLI for L05 (independent native re-check).

Rebuilds the task from a shipped ``qualified.jsonl`` record + its private
oracle payload, then *recomputes* every gate with the pinned RDKit engine:
differential index-vs-native agreement, counterfactual semantic witness,
reference terminal replay (must pass) and a seeded wrong submission (must
fail). The stored reference answer, record evidence hash and qualification
flags are compared against the recomputation; stored claims are never trusted.

Usage (repo root, ``.venv/bin/python``)::

    .venv/bin/python -m benchmark.plugins.verifier.l05.recheck \
      --index external_knowledge/indexes/rdkit-molecules/rdkit-Release_2026_09_1+chebi-rel255 \
      --bank bank/staging/l05 --instance l05i-...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.verifier.l05.verifier import RDKitNativeVerifier, SUPPORTED_PROFILE
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

REQUIRED_FIELDS = (
    "schema_version", "dataset_id", "lane_id", "family_id", "base_goal_id",
    "instance_id", "task_id", "task_regime", "complexity", "semantic_profile",
    "public", "oracle_handle", "status", "qualification", "native_evidence",
    "source_snapshot_ids", "source_unit_ids",
)
QUALIFICATION_FLAGS = (
    "native_verified", "solvable", "nontrivial", "semantic_relevance",
    "leakage_clean", "release_eligible",
)


def answer_sha(reference: Any) -> str:
    return canonical_sha({"reference": reference})


def recheck_command(instance_id: str, *, index_rel: str,
                    bank_rel: str = "bank/staging/l05") -> str:
    return (f".venv/bin/python -m benchmark.plugins.verifier.l05.recheck "
            f"--index {index_rel} --bank {bank_rel} --instance {instance_id}")


def locate(bank: Path, instance_id: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    for shard in sorted(Path(bank).glob("shard_*")):
        qpath = shard / "qualified.jsonl"
        if not qpath.is_file():
            continue
        for line in qpath.open(encoding="utf-8"):
            record = json.loads(line)
            if record.get("instance_id") == instance_id:
                opath = shard / "oracle" / f"{instance_id}.json"
                return record, json.loads(opath.read_text()), shard
    raise SystemExit(f"instance not found: {instance_id}")


def recheck_record(record: dict[str, Any], oracle: dict[str, Any],
                   verifier: RDKitNativeVerifier) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    structural = [f for f in REQUIRED_FIELDS if f not in record]
    checks["required_fields"] = not structural
    checks["missing_fields"] = structural

    qual = record.get("qualification") or {}
    checks["qualification_flags"] = all(qual.get(f) is True for f in QUALIFICATION_FLAGS)
    checks["status_qualified"] = record.get("status") == "QUALIFIED"
    checks["schema_version"] = record.get("schema_version") == "aob-task/5.0"
    checks["semantic_profile"] = record.get("semantic_profile") == SUPPORTED_PROFILE

    ast = oracle["ast"]
    cand = CandidateTaskAST(
        lane_id=record["lane_id"], family_id=record["family_id"],
        base_goal_id=record["base_goal_id"], instance_id=record["instance_id"],
        regime=record["task_regime"], semantic_profile=record["semantic_profile"],
        ast=ast, source_unit_ids=tuple(record.get("source_unit_ids") or ()),
    )
    grounded = GroundedTask(ast=cand, public=record["public"],
                            oracle_ref=record["oracle_handle"], native_evidence={})

    proof = verifier.prove_exists(grounded)
    checks["prove_exists"] = proof.exists
    checks["prove_reason"] = proof.reason
    witness = verifier.certify_semantic_relevance(grounded)
    checks["semantic_witness"] = witness.ok

    reference = verifier.reference_submission(ast)
    checks["reference_matches_oracle"] = reference == oracle["reference_submission"]
    stored_sha = ((record.get("native_evidence") or {})
                  .get("prove_exists", {}).get("reference_answer_sha256"))
    checks["reference_sha_matches_record"] = stored_sha == answer_sha(reference)

    pos = verifier.verify_terminal(grounded, reference)
    neg = verifier.verify_terminal(grounded, verifier.wrong_submission(ast))
    checks["terminal_positive_pass"] = pos.valid
    checks["terminal_negative_rejected"] = not neg.valid
    checks["leakage_clean"] = "private_" not in json.dumps(record["public"])
    passed = (all(v for k, v in checks.items() if isinstance(v, bool))
              and not structural)
    return {"instance_id": record["instance_id"], "family_id": record["family_id"],
            "verdict": "pass" if passed else "fail", "checks": checks,
            "oracle_handle": record["oracle_handle"]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-verify one L05 qualified instance")
    ap.add_argument("--index", required=True, help="pinned RDKit+ChEBI index dir")
    ap.add_argument("--bank", default="bank/staging/l05")
    ap.add_argument("--instance", required=True)
    args = ap.parse_args()
    verifier = RDKitNativeVerifier(Path(args.index))
    record, oracle, shard = locate(Path(args.bank), args.instance)
    result = recheck_record(record, oracle, verifier)
    result["shard"] = str(shard)
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())