"""L14 NativeVerifierPlugin: dual-path SQL/Python reference + witnesses.

Reference answers are recomputed natively against the pinned ChEMBL 37 SQLite:
  * fast path — the family SQL query (families.py),
  * clean-parser path — pyclean.py: raw row extraction only in SQL; all
    filtering/join/normalization logic re-implemented in pure Python.
Both must be exactly equal or the candidate is rejected. Witness certification
(witness.py) re-runs the clean-parser answer on a mutated in-memory slice:
the governed probe must flip and the control probe must stay unchanged.

Evidence rows carry: family, params, answer sha256/count, db manifest sha256,
engine version, witness payload — every verdict is independently recomputable
from the pinned bytes via the reverify CLI. No live API is ever consulted.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)
from benchmark.plugins.source.l14.kb import L14KB
from benchmark.plugins.verifier.l14 import pyclean, witness as W

_ENGINE = f"sqlite3 {sqlite3.sqlite_version} / python {sys.version_info.major}." \
          f"{sys.version_info.minor}"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def db_manifest_sha(pin_root: Path) -> str:
    pin = Path(pin_root) / "chembl_37_db.sha256"
    db = Path(pin_root) / "chembl_37" / "chembl_37_sqlite" / "chembl_37.db"
    return _sha({
        "db_sha256": pin.read_text().strip() if pin.is_file() else None,
        "db_bytes": db.stat().st_size if db.is_file() else None,
        "engine": _ENGINE,
    })


def verify_reference(pin_root, family_id: str, params: dict[str, Any],
                     expected: tuple[str, ...], kb: L14KB) -> dict[str, Any]:
    """Dual-path check: fast-path SQL answer (already in ``expected``) must
    equal the clean-parser recomputation from the pinned bytes."""
    con: sqlite3.Connection = kb.connection()
    try:
        got = pyclean.ANSWER_BY_FAMILY[family_id](con, params)
    except Exception as exc:
        con.close()
        return {"verdict": "ERROR", "reason": f"reference error: {exc}",
                "engine": _ENGINE}
    con.close()
    expected_set = frozenset(expected)
    verdict = "PASS" if got == expected_set else "FAIL"
    return {
        "verdict": verdict,
        "engine": _ENGINE,
        "family": family_id,
        "params_sha256": _sha(params),
        "answer_sha256": _sha(sorted(expected_set)),
        "answer_count": len(expected_set),
        "clean_parser_count": len(got),
        "db_manifest_sha": db_manifest_sha(Path(pin_root)),
        "reference_path": "pyclean/1 (pure-Python predicate over raw rows)",
    }


def certify_witness(kb: L14KB, family_id: str, witness_kind: str,
                    params: dict[str, Any], expected: tuple[str, ...],
                    seed: int) -> dict[str, Any]:
    con = kb.connection()
    try:
        return W.certify(con, family_id, params, expected, witness_kind, seed)
    finally:
        con.close()


class L14Verifier:
    """NativeVerifierPlugin surface (FINAL-BUILD-ORDER-v2 §3.2)."""

    lane_id = "L14"

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast
        family_id = ast.family_id.split(".", 1)[-1]
        answer = tuple(sorted(set(ast.ast.get("answer", []))))
        if not answer:
            return ProofOrCounterexample(
                exists=False, reason="empty reference answer")
        return ProofOrCounterexample(exists=True, evidence={
            "answer_sha256": _sha(sorted(answer)),
            "answer_count": len(answer),
            "family": family_id,
        })

    def verify_terminal(self, task: GroundedTask,
                        submitted_state_or_answer: Any) -> NativeVerdict:
        expected = {str(x) for x in task.ast.ast.get("answer", [])}
        got = {str(x) for x in (submitted_state_or_answer or [])}
        return NativeVerdict(valid=expected == got, detail={
            "expected_count": len(expected), "submitted_count": len(got),
            "missing_sample": sorted(expected - got)[:5],
            "extra_sample": sorted(got - expected)[:5],
        })

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        return SemanticWitness(ok=True, witness={
            "kind": task.ast.ast.get("witness_kind"),
            "note": "witness certified at build time; see oracle payload"})


PLUGIN = L14Verifier()
