"""L16 NativeVerifierPlugin facade (FINAL-BUILD-ORDER-v2 §3.2).

``prove_exists`` recomputes the reference answer from the independent
XML-parser SQLite store and requires exact agreement with the fast-path answer
carried by the AST; ``verify_terminal`` is the exact-set terminal check;
``certify_semantic_relevance`` reruns the counterfactual witness.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)
from benchmark.plugins.verifier.l16 import reference as ref
from benchmark.plugins.verifier.l16 import witness as wit

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "uniprot"
                    / "uniprot-2026_03-sprot")


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _xml_meta(root: Path) -> dict[str, str]:
    db = ref._db(Path(root))
    return dict(db.execute("SELECT key, value FROM meta").fetchall())


def verify_reference(pin_root: Path, family_id: str, params: dict[str, Any],
                     expected: tuple[str, ...], kb: Any = None) -> dict[str, Any]:
    return ref.verify_reference(Path(pin_root), family_id, params, expected,
                                _xml_meta(Path(pin_root)))


def certify_witness(kb: Any, family_id: str, witness_kind: str,
                    params: dict[str, Any], seed: int) -> dict[str, Any]:
    from benchmark.plugins.task_family.l16.families import FAMILY_BY_ID
    fam = FAMILY_BY_ID.get(family_id)
    if fam is None:
        return {"ok": False, "reason": f"unknown family {family_id}"}
    return wit.witness_for(fam, kb, params, seed)


def _family_params(task: Any) -> tuple[str, dict[str, Any], tuple[str, ...], int]:
    ast = task.ast if isinstance(task, GroundedTask) else task
    fam_id = ast.family_id.split(".", 1)[-1]
    payload = ast.ast
    return (fam_id, dict(payload["params"]),
            tuple(payload.get("answer", ())), int(payload.get("seed", 0)))


class L16NativeVerifier:
    """§3.2 facade over the L16 dual-store stack."""

    lane_id = "L16"

    def __init__(self, pin_root: str | Path | None = None):
        self.pin_root = Path(pin_root) if pin_root else DEFAULT_PIN_ROOT
        self._kb: Any = None

    @property
    def kb(self):
        if self._kb is None:
            from benchmark.plugins.source.l16.kb import load_kb
            self._kb = load_kb(str(self.pin_root))
        return self._kb

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        fam_id, params, expected, _seed = _family_params(grounded_task)
        if not expected:
            from benchmark.plugins.task_family.l16.families import FAMILY_BY_ID
            fam = FAMILY_BY_ID[fam_id]
            expected = fam.answer(self.kb, params)
        ev = verify_reference(self.pin_root, fam_id, params, expected)
        ok = ev.get("verdict") == "PASS" and len(set(expected)) > 0
        return ProofOrCounterexample(
            ok, evidence=ev, reason="" if ok else "reference disagreement")

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any
                        ) -> NativeVerdict:
        fam_id, _params, expected, _seed = _family_params(task)
        answer = submitted_state_or_answer
        if isinstance(answer, dict):
            answer = answer.get("answer")
        if not (isinstance(answer, (list, tuple))
                and all(isinstance(x, str) for x in answer)):
            return NativeVerdict(False,
                                 {"reason": "submitted answer must be list[str]"})
        got, want = set(answer), set(expected)
        return NativeVerdict(got == want, {
            "family": fam_id,
            "expected_count": len(want), "submitted_count": len(got),
            "missing_sample": sorted(want - got)[:5],
            "extra_sample": sorted(got - want)[:5],
            "expected_sha256": _sha(sorted(want)),
            "submitted_sha256": _sha(sorted(got)),
            "engine": "l16 exact-set (pinned dual store)",
        })

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        fam_id, params, _expected, seed = _family_params(task)
        w = certify_witness(self.kb, fam_id, "", params, seed)
        return SemanticWitness(bool(w.get("ok")), witness=w)


PLUGIN = L16NativeVerifier()
