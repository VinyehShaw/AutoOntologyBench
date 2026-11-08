"""L12 20K expansion verifier: pinned-pm4py native verification, new oracle dir.

Subclass of the legacy ``Pm4pyNativeVerifier``:
- oracles persist under ``bank/staging_20k/l12/oracle/`` (legacy
  ``bank/staging/l12/oracle/`` is never touched);
- the two new exact-artifact operations (``footprints``, ``footprints_conf``)
  are proved / terminal-verified / counterfactually witnessed through
  ``expand20k.compute`` (the same single source of truth the generators use,
  so exact equality cannot drift);
- every other goal kind / operation delegates to the legacy verifier logic
  unchanged (strict token replay, log skeleton, DECLARE, dfg / variants /
  endpoints / nonconformant_trace_set);
- the ``reproduction`` evidence string is rewritten to this module's audit
  command so each record carries its exact repro command.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)
from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.expand20k import compute
from benchmark.plugins.task_family.l12.expand20k.compute import NEW_EXACT_OPS
from benchmark.plugins.task_family.l12.expand20k.registry import STAGING_20K
from benchmark.plugins.verifier.l12.pm4py_verifier import Pm4pyNativeVerifier

ORACLE_DIR_20K = STAGING_20K / "oracle"

_REPRO_TMPL = (
    "PYTHONPATH=. .venv/bin/python -m "
    "benchmark.plugins.task_family.l12.expand20k.pipeline --audit --task-id l12-%s"
)


def _repro_cmd(instance_id: str) -> str:
    return _REPRO_TMPL % instance_id


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _is_new_exact(ast: dict[str, Any]) -> bool:
    return ast.get("goal_kind") == "exact_artifact" and ast.get("operation") in NEW_EXACT_OPS


class Pm4pyVerifier20k(Pm4pyNativeVerifier):
    """NativeVerifierPlugin for the L12 20K expansion."""

    plugin_id = "l12.pm4py-verifier-20k"

    # -- prove_exists -----------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        if _is_new_exact(ast):
            return self._prove_exact20k(grounded_task)
        proof = super().prove_exists(grounded_task)
        if isinstance(proof.evidence, dict):
            proof.evidence["reproduction"] = _repro_cmd(grounded_task.ast.instance_id)
        return proof

    def _prove_exact20k(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        answer = compute.compute_answer(
            ast["operation"], ast["log_rel"], ast["projection"]["first_n"],
            ast.get("net_rel"),
        )
        evidence = {
            "engine": engine.ENGINE,
            "engine_version": engine.PM4PY_VERSION,
            "source_commit": engine.PM4PY_COMMIT,
            "operation": ast["operation"],
            "log_sha256": engine.sha256_file(engine.PINNED_ROOT / ast["log_rel"]),
            "projection": ast["projection"],
            "reference_answer_sha256": engine.sha256_json(answer),
            "answer_size": len(json.dumps(answer, sort_keys=True)),
            "reproduction": _repro_cmd(grounded_task.ast.instance_id),
        }
        if ast.get("net_rel"):
            evidence["net_sha256"] = engine.sha256_file(
                engine.PINNED_ROOT / ast["net_rel"])
        if answer != ast["reference_answer"]:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="UNSOLVABLE: native recompute disagrees with generator (nondeterminism)",
            )
        if not compute.nontrivial(ast["operation"], answer):
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: answer artifact below nontriviality floor",
            )
        evidence["reference_answer"] = answer
        return ProofOrCounterexample(exists=True, evidence=evidence)

    # -- verify_terminal ----------------------------------------------------------
    def verify_terminal(
        self, task: GroundedTask, submitted_state_or_answer: Any
    ) -> NativeVerdict:
        ast = task.ast.ast
        if _is_new_exact(ast):
            reference = compute.compute_answer(
                ast["operation"], ast["log_rel"], ast["projection"]["first_n"],
                ast.get("net_rel"),
            )
            submitted_ans = (submitted_state_or_answer or {}).get("answer")
            ok = submitted_ans == reference
            detail: dict[str, Any] = {
                "check": "exact_artifact_recompute",
                "operation": ast["operation"],
                "match": ok,
                "reference_answer_sha256": engine.sha256_json(reference),
            }
            if not ok and isinstance(submitted_ans, dict):
                detail["submitted_answer_sha256"] = engine.sha256_json(submitted_ans)
            return NativeVerdict(valid=ok, detail=detail)
        return super().verify_terminal(task, submitted_state_or_answer)

    # -- certify_semantic_relevance --------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        if _is_new_exact(ast):
            reference = compute.compute_answer(
                ast["operation"], ast["log_rel"], ast["projection"]["first_n"],
                ast.get("net_rel"),
            )
            mutated = self._mutate_answer20k(ast["operation"], reference)
            failing: list[dict[str, Any]] = []
            if mutated is not None and mutated != reference:
                failing.append({
                    "mode": "mutated_answer",
                    "note": "single-element mutation fails exact native recompute equality",
                    "mutated_sha256": engine.sha256_json(mutated),
                    "reference_sha256": engine.sha256_json(reference),
                })
            return SemanticWitness(ok=bool(failing), witness={"counterfactuals": failing})
        return super().certify_semantic_relevance(task)

    @staticmethod
    def _mutate_answer20k(operation: str, reference: dict[str, Any]) -> dict[str, Any] | None:
        import copy

        mutated = copy.deepcopy(reference)
        if operation == "footprints":
            if mutated.get("sequence"):
                mutated["sequence"] = mutated["sequence"][1:]
                return mutated
            if mutated.get("activities"):
                mutated["activities"] = mutated["activities"][:-1]
                return mutated
            return None
        if operation == "footprints_conf":
            idx = set(mutated.get("violating_trace_indices") or [])
            idx.discard(0) if 0 in idx else idx.add(0)
            mutated["violating_trace_indices"] = sorted(idx)
            return mutated
        return None

    # -- oracle persistence (new dir only; legacy never touched) -----------------
    def write_oracle(self, grounded_task: GroundedTask, extra: dict[str, Any]) -> Path:
        cand = grounded_task.ast
        ast = cand.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "family_id": cand.family_id,
            "goal_kind": ast["goal_kind"],
            "operation": ast["operation"],
            "dataset_id": "autoontobench-bank-20000-candidate-dev1",
            "candidate": {
                "lane_id": cand.lane_id,
                "family_id": cand.family_id,
                "base_goal_id": cand.base_goal_id,
                "instance_id": cand.instance_id,
                "regime": cand.regime,
                "semantic_profile": cand.semantic_profile,
                "source_unit_ids": list(cand.source_unit_ids),
            },
            "ast": ast,
            "reference_terminal": extra.get("reference_terminal"),
            "reference_answer": extra.get("reference_answer"),
            "reference_sha256": extra.get("reference_terminal_sha256")
            or extra.get("reference_answer_sha256"),
            "engine": engine.ENGINE,
            "engine_version": engine.PM4PY_VERSION,
            "source_commit": engine.PM4PY_COMMIT,
        }
        path = ORACLE_DIR_20K / ("%s.json" % grounded_task.oracle_ref)
        _write_json_atomic(path, payload)
        return path

    def _oracle(self, task: GroundedTask) -> dict[str, Any] | None:
        path = ORACLE_DIR_20K / ("%s.json" % task.oracle_ref)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
