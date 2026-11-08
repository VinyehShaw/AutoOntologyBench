"""L12 NativeVerifierPlugin: pinned-pm4py conformance verification.

- ``prove_exists``: native-prove the corrupted initial artifact fails the
  family check (strict token replay / log skeleton / DECLARE) and the reference
  repair passes, or natively recompute the exact audit artifact.
- ``verify_terminal``: recompute/replay the *submitted* answer from the pinned
  source artifacts (never trusts stored state).
- ``certify_semantic_relevance``: counterfactual witness — at least one
  plausible wrong submission natively fails the same check.

Oracle payloads live at ``bank/staging/l12/oracle/<oracle_ref>.json`` (staging
is the private build area). Every check is a pm4py 2.7.23.11 native call over
the pinned snapshot (commit 0e077848); no LLM output is used as gold.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)
from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT

REPO_ROOT = Path(__file__).resolve().parents[4]
ORACLE_DIR = REPO_ROOT / "bank" / "staging" / "l12" / "oracle"

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


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


def _model_for(operation: str, log_rel: str) -> Any:
    key = (operation, log_rel)
    if key not in _MODEL_CACHE:
        traces = RT.traces(log_rel)
        if operation == "log_skeleton":
            _MODEL_CACHE[key] = RT.skeleton(traces)
        else:
            _MODEL_CACHE[key] = RT.declare_model(traces)
    return _MODEL_CACHE[key]


def _check_trace(operation: str, ast: dict[str, Any], trace: list[str]) -> dict[str, Any]:
    """One native verdict for a trace under the family check."""
    if operation == "token_replay_strict":
        return RT.replay([trace], ast["net_rel"])[0]
    model = _model_for(operation, ast["log_rel"])
    if operation == "log_skeleton":
        return RT.skeleton_check([trace], model)[0]
    return RT.declare_check([trace], model)[0]


def _passes(operation: str, verdict: dict[str, Any]) -> bool:
    if operation == "token_replay_strict":
        return bool(verdict["trace_is_fit"])
    return bool(verdict["is_fit"])


def _summarize_replay(verdicts: list[dict[str, Any]]) -> dict[str, Any]:
    bad = [i for i, v in enumerate(verdicts) if not v["trace_is_fit"]]
    return {
        "traces": len(verdicts),
        "fit_traces": len(verdicts) - len(bad),
        "nonconformant_trace_indices": bad[:50],
        "all_fit": not bad,
    }


def _recompute_exact(ast: dict[str, Any]) -> dict[str, Any]:
    from benchmark.plugins.task_family.l12.propose import _compute_answer

    return _compute_answer(
        ast["operation"], ast["log_rel"], ast["projection"]["first_n"], ast.get("net_rel")
    )


def _nontrivial_exact(operation: str, ans: dict[str, Any]) -> bool:
    if operation == "dfg":
        return len(ans["edges"]) >= 3
    if operation == "variants":
        return len(ans["variants"]) >= 2
    if operation == "endpoints":
        return len(ans["activity_counts"]) >= 2
    if operation == "nonconformant_trace_set":
        return 1 <= len(ans["nonconformant_trace_indices"]) < ans["traces_total"]
    return False


class Pm4pyNativeVerifier:
    """NativeVerifierPlugin for lane L12."""

    plugin_id = "l12.pm4py-verifier"

    # -- prove_exists -----------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        kind = ast["goal_kind"]
        base = {
            "engine": engine.ENGINE,
            "engine_version": engine.PM4PY_VERSION,
            "source_commit": engine.PM4PY_COMMIT,
            "operation": ast["operation"],
            "log_sha256": engine.sha256_file(engine.PINNED_ROOT / ast["log_rel"]),
            "reproduction": (
                "PYTHONPATH=. .venv/bin/python -m "
                "benchmark.plugins.task_family.l12.pipeline --audit --task-id l12-%s"
                % grounded_task.ast.instance_id
            ),
        }
        if ast.get("net_rel"):
            base["net_sha256"] = engine.sha256_file(engine.PINNED_ROOT / ast["net_rel"])

        if kind == "repair_trace":
            corrupted = ast["corrupted_trace"]
            reference = ast["reference_trace"]
            initial = _check_trace(ast["operation"], ast, corrupted)
            ref = _check_trace(ast["operation"], ast, reference)
            evidence = {
                **base,
                "initial_verdict": initial,
                "reference_verdict": ref,
                "corrupted_trace_sha256": engine.sha256_json(corrupted),
                "reference_terminal_sha256": engine.sha256_json(reference),
            }
            if _passes(ast["operation"], initial):
                return ProofOrCounterexample(
                    exists=False, evidence=evidence,
                    reason="TRIVIAL: corrupted trace already passes the native check",
                )
            if not _passes(ast["operation"], ref):
                return ProofOrCounterexample(
                    exists=False, evidence=evidence,
                    reason="UNSOLVABLE: reference trace fails the native check",
                )
            evidence["reference_terminal"] = {"trace": reference}
            return ProofOrCounterexample(exists=True, evidence=evidence)

        if kind == "repair_model":
            traces = RT.traces(ast["log_rel"])
            corrupted = _summarize_replay(
                RT.replay_on_relabeled(traces, ast["net_rel"], ast["corrupted_tids"])
            )
            reference = _summarize_replay(
                RT.replay_on_relabeled(traces, ast["net_rel"], ast["reference_mapping"])
            )
            evidence = {
                **base,
                "initial_verdict": corrupted,
                "reference_verdict": reference,
                "corrupted_tids": ast["corrupted_tids"],
                "reference_terminal_sha256": engine.sha256_json(ast["reference_mapping"]),
            }
            if corrupted["all_fit"]:
                return ProofOrCounterexample(
                    exists=False, evidence=evidence,
                    reason="TRIVIAL: obscured net already replays the log perfectly",
                )
            if not reference["all_fit"]:
                return ProofOrCounterexample(
                    exists=False, evidence=evidence,
                    reason="UNSOLVABLE: reference mapping does not restore perfect replay",
                )
            evidence["reference_terminal"] = {"mapping": ast["reference_mapping"]}
            return ProofOrCounterexample(exists=True, evidence=evidence)

        # exact_artifact
        answer = _recompute_exact(ast)
        evidence = {
            **base,
            "projection": ast["projection"],
            "reference_answer_sha256": engine.sha256_json(answer),
            "answer_size": len(json.dumps(answer, sort_keys=True)),
        }
        if answer != ast["reference_answer"]:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="UNSOLVABLE: native recompute disagrees with generator (nondeterminism)",
            )
        if not _nontrivial_exact(ast["operation"], answer):
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
        oracle = self._oracle(task)
        if oracle is None:
            return NativeVerdict(valid=False, detail={"reason": "oracle_missing"})
        submitted = submitted_state_or_answer or {}
        checks: list[dict[str, Any]] = []

        if ast["goal_kind"] == "repair_trace":
            trace = submitted.get("trace")
            if not isinstance(trace, list) or not all(isinstance(a, str) for a in trace):
                return NativeVerdict(valid=False, detail={"reason": "answer_not_a_trace"})
            corrupted = ast["corrupted_trace"]
            for guard in ast.get("guards", []):
                if guard == "differs_from_initial":
                    checks.append({"check": guard, "pass": trace != corrupted})
                elif guard == "same_multiset":
                    checks.append({"check": guard, "pass": sorted(trace) == sorted(corrupted)})
                elif guard == "same_length":
                    checks.append({"check": guard, "pass": len(trace) == len(corrupted)})
                elif guard == "longer_than_initial":
                    checks.append({"check": guard, "pass": len(trace) > len(corrupted)})
                elif guard == "shorter_than_initial":
                    checks.append({"check": guard, "pass": len(trace) < len(corrupted)})
            verdict = _check_trace(ast["operation"], ast, trace)
            checks.append({
                "check": ast["operation"],
                "pass": _passes(ast["operation"], verdict),
                "verdict": verdict,
            })
            ok = all(c["pass"] for c in checks)
            return NativeVerdict(valid=ok, detail={"checks": checks})

        if ast["goal_kind"] == "repair_model":
            mapping = submitted.get("mapping")
            if not isinstance(mapping, dict):
                return NativeVerdict(valid=False, detail={"reason": "answer_not_a_mapping"})
            checks.append({
                "check": "mapping_keys_cover_obscured",
                "pass": set(mapping.keys()) == set(ast["corrupted_tids"].keys()),
            })
            checks.append({
                "check": "labels_are_strings",
                "pass": all(isinstance(v, str) and v for v in mapping.values()),
            })
            traces = RT.traces(ast["log_rel"])
            summary = _summarize_replay(
                RT.replay_on_relabeled(traces, ast["net_rel"], mapping)
            )
            checks.append({
                "check": "token_replay_strict",
                "pass": summary["all_fit"],
                "verdict": summary,
            })
            ok = all(c["pass"] for c in checks)
            return NativeVerdict(valid=ok, detail={"checks": checks})

        # exact_artifact: independent native recompute, exact equality
        reference = _recompute_exact(ast)
        submitted_ans = submitted.get("answer")
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

    # -- certify_semantic_relevance --------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        failing: list[dict[str, Any]] = []

        if ast["goal_kind"] == "repair_trace":
            corrupted = ast["corrupted_trace"]
            verdict = _check_trace(ast["operation"], ast, corrupted)
            if not _passes(ast["operation"], verdict):
                failing.append({
                    "mode": "no_repair",
                    "verdict": verdict,
                    "note": "initial corrupted trace fails the native check",
                })
        elif ast["goal_kind"] == "repair_model":
            traces = RT.traces(ast["log_rel"])
            summary = _summarize_replay(
                RT.replay_on_relabeled(traces, ast["net_rel"], ast["corrupted_tids"])
            )
            if not summary["all_fit"]:
                failing.append({
                    "mode": "no_repair",
                    "verdict": summary,
                    "note": "obscured net fails strict replay on the log",
                })
            alt = {tid: "zzz_alt_%02d" % i for i, tid in enumerate(ast["corrupted_tids"])}
            summary2 = _summarize_replay(
                RT.replay_on_relabeled(traces, ast["net_rel"], alt)
            )
            if not summary2["all_fit"]:
                failing.append({
                    "mode": "wrong_label_substitution",
                    "verdict": summary2,
                    "note": "plausible-but-wrong relabeling fails strict replay",
                })
        else:
            reference = _recompute_exact(ast)
            mutated = self._mutate_answer(ast["operation"], reference)
            if mutated is not None and mutated != reference:
                failing.append({
                    "mode": "mutated_answer",
                    "note": "single-element mutation fails exact native recompute equality",
                    "mutated_sha256": engine.sha256_json(mutated),
                    "reference_sha256": engine.sha256_json(reference),
                })
        return SemanticWitness(ok=bool(failing), witness={"counterfactuals": failing})

    @staticmethod
    def _mutate_answer(operation: str, reference: dict[str, Any]) -> dict[str, Any] | None:
        import copy

        mutated = copy.deepcopy(reference)
        if operation == "dfg" and mutated.get("edges"):
            mutated["edges"] = mutated["edges"][1:]
            return mutated
        if operation == "variants" and mutated.get("variants"):
            mutated["variants"] = mutated["variants"][:-1]
            return mutated
        if operation == "endpoints" and mutated.get("activity_counts"):
            key = sorted(mutated["activity_counts"])[0]
            mutated["activity_counts"][key] += 1
            return mutated
        if operation == "nonconformant_trace_set":
            idx = set(mutated.get("nonconformant_trace_indices") or [])
            idx.discard(0) if 0 in idx else idx.add(0)
            mutated["nonconformant_trace_indices"] = sorted(idx)
            return mutated
        return None

    # -- oracle persistence ------------------------------------------------------
    def write_oracle(self, grounded_task: GroundedTask, extra: dict[str, Any]) -> Path:
        cand: CandidateTaskAST = grounded_task.ast
        ast = cand.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "family_id": cand.family_id,
            "goal_kind": ast["goal_kind"],
            "operation": ast["operation"],
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
        path = ORACLE_DIR / ("%s.json" % grounded_task.oracle_ref)
        _write_json_atomic(path, payload)
        return path

    def _oracle(self, task: GroundedTask) -> dict[str, Any] | None:
        path = ORACLE_DIR / ("%s.json" % task.oracle_ref)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
