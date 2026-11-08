"""L15 native verifier: recompute reference answers from the pinned Reactome
store and gate qualification.

Truth discipline: answers are recomputed per call from ``reactome.db`` (itself
a deterministic build over the pinned files, sha256-checked). For B+M families
the BioPAX and mapping paths must agree or the task is rejected
(``PROFILE_UNSUPPORTED:path_disagree``). No LLM output is gold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.plugins.source.l15.kb import L15KB
import os
from benchmark.plugins.source.l15.plugin import STORE_DB
from benchmark.plugins.task_family.l15.families import BY_ID
from benchmark.plugins.task_family.l15 import goals as G
from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)


def _canon(payload: Any) -> str:
    def norm(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: norm(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return sorted({json.dumps(x, sort_keys=True) for x in
                           (norm(i) for i in o)}) if all(
                isinstance(i, str) for i in o) else [norm(i) for i in o]
        return o
    return json.dumps(norm(payload), sort_keys=True, separators=(",", ":"))


class ReactomeNativeVerifier:
    def __init__(self, db_path: str | Path | None = None, *,
                 allow_m_only: bool = False) -> None:
        self.kb = L15KB(db_path or os.environ.get("L15_STORE_DB") or STORE_DB)
        self.allow_m_only = allow_m_only

    def reference_submission(self, ast: dict[str, Any]) -> Any:
        return G.KINDS[ast["kind"]].answer(self.kb, ast)

    def _answer_size(self, ref: Any) -> int:
        if "answer_ids" in ref:
            return len(ref["answer_ids"])
        if "pairs" in ref:
            return len(ref["pairs"])
        if "assignment" in ref:
            return len(ref["assignment"])
        return 0

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        kind = ast["kind"]
        spec = BY_ID[grounded_task.ast.family_id]
        ev: dict[str, Any] = {"kind": kind, "store": str(self.kb.db_path),
                              "family_id": grounded_task.ast.family_id}
        skip_agree = False
        if not self.kb.has_biopax() and spec.truth_path.startswith("B"):
            if self.allow_m_only and spec.truth_path == "B+M":
                skip_agree = True
                ev["differential_agree"] = "NOT_RUN:biopax_snapshot_unavailable"
            else:
                return ProofOrCounterexample(False, ev, "PROFILE_UNSUPPORTED:no_biopax_store")
        try:
            ref = self.reference_submission(ast)
        except Exception as exc:
            return ProofOrCounterexample(False, ev, f"PROFILE_UNSUPPORTED:{exc}"[:200])
        size = self._answer_size(ref)
        ev["answer_size"] = size
        if size < spec.min_answer or size > spec.max_answer:
            return ProofOrCounterexample(
                False, ev, f"TRIVIAL:size_{size}_outside_[{spec.min_answer},{spec.max_answer}]")
        if spec.truth_path == "B+M" and not skip_agree:
            ops = G.KINDS[kind]
            if ops.agree is not None:
                try:
                    agree = ops.agree(self.kb, ast)
                except Exception as exc:
                    return ProofOrCounterexample(
                        False, ev, f"NATIVE_ERROR:agree:{type(exc).__name__}"[:200])
                ev["differential_agree"] = bool(agree)
                if not agree:
                    return ProofOrCounterexample(
                        False, ev, "PROFILE_UNSUPPORTED:path_disagree")
        if kind == "direction_repair":
            if not self._repair_evidence_ok(ast):
                return ProofOrCounterexample(
                    False, ev, "PROFILE_UNSUPPORTED:repair_evidence_missing")
        ev["solvable"] = True
        return ProofOrCounterexample(True, ev)

    def _repair_evidence_ok(self, ast: dict[str, Any]) -> bool:
        """direction_repair needs the swapped PE to carry a UniProt accession
        that has a reaction-role row (m_reaction_roles, from the pinned
        reactome_reaction_exporter.txt) for the reaction; otherwise the public
        role evidence cannot determine the correct side."""
        swapped = ast["swapped_pe"]
        rows = self.kb._q(
            """SELECT x.accession FROM b_xrefs x
               JOIN b_reaction_sides s ON s.pe_uri = x.pe_uri
               WHERE s.reaction=? AND x.db LIKE 'UniProt%'""",
            (ast["reaction"],))
        sides = G.reaction_sides(self.kb, ast["reaction"])
        target_uri = None
        for side in ("left", "right"):
            for u in sides[side]:
                if G._tail(u) == swapped:
                    target_uri = u
        if target_uri is None:
            return False
        accs = {x["accession"] for x in self.kb.b_pe_xrefs(target_uri)
                if x["db"].lower().startswith("uniprot")}
        if not accs:
            return False
        for acc in accs:
            r = self.kb._q(
                "SELECT role FROM m_reaction_roles WHERE accession=? AND reaction_id=?",
                (acc, ast["reaction"]))
            if r:
                return True
        return False

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict:
        ast = task.ast.ast
        try:
            ref = self.reference_submission(ast)
        except Exception as exc:
            return NativeVerdict(False, {"error": f"reference:{exc}"[:200]})
        ok = _canon(submitted_state_or_answer) == _canon(ref)
        return NativeVerdict(ok, {"match": ok})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        kind = ast["kind"]
        base: dict[str, Any] = {"kind": kind, "family_id": task.ast.family_id}
        try:
            ref = self.reference_submission(ast)
            wit = G.KINDS[kind].counter(self.kb, ast, ref)
        except Exception as exc:
            return SemanticWitness(False, dict(base, error=f"{type(exc).__name__}: {exc}"[:200]))
        if wit is None:
            return SemanticWitness(False, dict(base, error="no_counterfactual_found"))
        return SemanticWitness(True, dict(base, **wit))

    def mutation_submission(self, ast: dict[str, Any], reference: Any) -> Any:
        return G.KINDS[ast["kind"]].mutate(self.kb, ast, reference)
