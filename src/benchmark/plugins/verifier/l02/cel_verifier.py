"""L02 NativeVerifierPlugin: pinned cel-go evaluation of VAP/CEL task artifacts.

- ``prove_exists``: replay the candidate's reference solution / compute the exact
  reference answer natively (cel_helper eval per object per policy expression)
  and prove the initial state is NOT already solved (nontrivial start).
- ``verify_terminal``: check a submitted terminal state / answer against the
  persisted oracle, recomputing native verdicts where the goal kind allows.
- ``certify_semantic_relevance``: counterfactual witness — under perturbed
  params, a policy subset, or a perturbed object, at least one native verdict
  or message flips. Recorded with the same command+sha evidence discipline.

Oracle payloads live at ``bank/staging/l02/oracle/<oracle_ref>.json`` (staging
is the private build area; release isolation is the release lane's job).
"""
from __future__ import annotations

import copy
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
from benchmark.plugins.task_family.l02 import engine
from benchmark.plugins.task_family.l02.engine import PolicyUnit

REPO_ROOT = Path(__file__).resolve().parents[4]
ORACLE_DIR = REPO_ROOT / "bank" / "staging" / "l02" / "oracle"


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


def _public(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if not k.startswith("__")}


def _units(grounded: GroundedTask) -> list[tuple[dict[str, Any], PolicyUnit]]:
    policies = engine.load_policies()
    return [(p, policies[p["policy_id"]]) for p in grounded.ast.ast["policy_configs"]]


def _inputs(obj: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    return {"object": _public(obj), "params": dict(params)}


def _decision(
    cfg: dict[str, Any], unit: PolicyUnit, obj: dict[str, Any], ctx: str,
    *, params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rec = engine.policy_decision(
        unit, _inputs(obj, cfg["params"] if params is None else params), input_id=ctx)
    rec["handle"] = cfg["handle"]
    return rec


def _object_set_decisions(
    configs: list[tuple[dict[str, Any], PolicyUnit]],
    objects: dict[str, Any],
    ctx: str,
    *,
    params: dict[str, Any] | None = None,
) -> tuple[bool, list[dict[str, Any]]]:
    """(all_allow, per-policy eval records) for every object x policy."""
    evals: list[dict[str, Any]] = []
    all_allow = True
    for oid, obj in objects.items():
        for cfg, unit in configs:
            rec = _decision(cfg, unit, obj, "%s:%s" % (ctx, oid), params=params)
            evals.append(rec)
            if rec["decision"] != "ALLOW":
                all_allow = False
    return all_allow, evals


class KubernetesCelVerifier:
    """NativeVerifierPlugin for lane L02."""

    plugin_id = "l02.cel-verifier"

    # -- prove_exists -----------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        goal_kind = ast["goal_kind"]
        handler = {
            "repair_compliance": self._prove_repair,
            "param_tuning": self._prove_param_tuning,
            "boundary_probe": self._prove_boundary_probe,
            "exact_decision_set": self._prove_decision_set,
            "message_expression": self._prove_message_expression,
            "match_gate": self._prove_match_gate,
        }.get(goal_kind)
        if handler is None:
            return ProofOrCounterexample(
                exists=False, reason="PROFILE_UNSUPPORTED: goal_kind %r" % goal_kind)
        return handler(grounded_task)

    def _prove_repair(self, grounded: GroundedTask) -> ProofOrCounterexample:
        ast = grounded.ast.ast
        configs = _units(grounded)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        reference = {oid: _public(o) for oid, o in ast["reference_terminal"].items()}
        ctx = grounded.ast.instance_id
        evidence: dict[str, Any] = {"goal_kind": ast["goal_kind"], "evals": []}
        init_ok, init_evals = _object_set_decisions(configs, objects, "%s:init" % ctx)
        evidence["evals"].extend(init_evals)
        if init_ok:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: initial state already compliant")
        ref_ok, ref_evals = _object_set_decisions(configs, reference, "%s:ref" % ctx)
        evidence["evals"].extend(ref_evals)
        if not ref_ok:
            bad = [e["handle"] for e in ref_evals if e["decision"] != "ALLOW"]
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="UNSOLVABLE: reference terminal violates %s" % sorted(set(bad)))
        evidence["reference_terminal_sha256"] = {
            oid: engine.sha256_json(o) for oid, o in reference.items()}
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def _prove_param_tuning(self, grounded: GroundedTask) -> ProofOrCounterexample:
        ast = grounded.ast.ast
        configs = _units(grounded)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        current = dict(ast["current_params"])
        reference = dict(ast["reference_params"])
        ctx = grounded.ast.instance_id
        evidence: dict[str, Any] = {"goal_kind": ast["goal_kind"], "evals": []}
        cur_ok, cur_evals = _object_set_decisions(configs, objects, "%s:cur" % ctx,
                                                  params=current)
        evidence["evals"].extend(cur_evals)
        if cur_ok:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: current params already admit every object")
        ref_ok, ref_evals = _object_set_decisions(configs, objects, "%s:ref" % ctx,
                                                  params=reference)
        evidence["evals"].extend(ref_evals)
        if not ref_ok:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="UNSOLVABLE: reference params do not admit every object")
        below = {**reference, "maxReplicas": int(reference["maxReplicas"]) - 1}
        below_ok, below_evals = _object_set_decisions(configs, objects, "%s:min" % ctx,
                                                      params=below)
        evidence["evals"].extend(below_evals)
        if below_ok:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="UNSOLVABLE: reference params are not minimal (maxReplicas-1 also admits)")
        evidence["reference_params"] = reference
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def _prove_boundary_probe(self, grounded: GroundedTask) -> ProofOrCounterexample:
        ast = grounded.ast.ast
        configs = _units(grounded)
        probes = ast["probe_objects"]
        boundaries = {k: int(v) for k, v in ast["reference_answer"]["boundaries"].items()}
        ctx = grounded.ast.instance_id
        evidence: dict[str, Any] = {"goal_kind": ast["goal_kind"], "evals": []}
        for cfg, unit in configs:
            handle = cfg["handle"]
            boundary = boundaries[handle]
            base = probes[handle]
            at = _with_replicas(base, boundary)
            beyond = _with_replicas(base, boundary + 1)
            below = _with_replicas(base, max(boundary - 1, 0))
            rec_at = _decision(cfg, unit, at, "%s:%s:at" % (ctx, handle))
            rec_beyond = _decision(cfg, unit, beyond, "%s:%s:beyond" % (ctx, handle))
            rec_below = _decision(cfg, unit, below, "%s:%s:below" % (ctx, handle))
            evidence["evals"].extend([rec_at, rec_beyond, rec_below])
            if rec_at["decision"] != "ALLOW" or rec_beyond["decision"] != "DENY":
                return ProofOrCounterexample(
                    exists=False, evidence=evidence,
                    reason="UNSOLVABLE: boundary %d for %s does not straddle ALLOW/DENY"
                           % (boundary, handle))
            if rec_below["decision"] != "ALLOW":
                return ProofOrCounterexample(
                    exists=False, evidence=evidence,
                    reason="TRIVIAL: boundary %d for %s is degenerate (below-boundary denies)"
                           % (boundary, handle))
        if not boundaries or any(b < 1 for b in boundaries.values()):
            return ProofOrCounterexample(
                exists=False, evidence=evidence, reason="TRIVIAL: empty/zero boundary set")
        evidence["reference_answer"] = {"boundaries": boundaries}
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def _prove_decision_set(self, grounded: GroundedTask) -> ProofOrCounterexample:
        ast = grounded.ast.ast
        configs = _units(grounded)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        ctx = grounded.ast.instance_id
        evidence: dict[str, Any] = {"goal_kind": ast["goal_kind"], "evals": []}
        answer: dict[str, dict[str, str]] = {}
        for oid, obj in objects.items():
            per_policy: dict[str, str] = {}
            for cfg, unit in configs:
                rec = _decision(cfg, unit, obj, "%s:%s" % (ctx, oid))
                evidence["evals"].append(rec)
                per_policy[cfg["handle"]] = rec["decision"]
            answer[oid] = per_policy
        decisions = [d for per in answer.values() for d in per.values()]
        if not answer or len(set(decisions)) < 2:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: decision set has <2 distinct outcomes")
        evidence["reference_answer"] = {"decisions": answer}
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def _prove_message_expression(self, grounded: GroundedTask) -> ProofOrCounterexample:
        ast = grounded.ast.ast
        configs = _units(grounded)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        ctx = grounded.ast.instance_id
        evidence: dict[str, Any] = {"goal_kind": ast["goal_kind"], "evals": []}
        messages: dict[str, dict[str, str]] = {}
        for oid, obj in objects.items():
            per_policy: dict[str, str] = {}
            for cfg, unit in configs:
                rec = _decision(cfg, unit, obj, "%s:%s:gate" % (ctx, oid))
                evidence["evals"].append(rec)
                if unit.message_kind == "violation" and rec["decision"] != "DENY":
                    return ProofOrCounterexample(
                        exists=False, evidence=evidence,
                        reason="TRIVIAL: object %s does not violate %s (no violation message)"
                               % (oid, cfg["handle"]))
                for expr in unit.message_expressions[:1]:
                    out = engine.cel_eval(expr, _inputs(obj, cfg["params"]),
                                          input_id="%s:%s:msg" % (ctx, oid))
                    evidence["evals"].append({
                        "handle": cfg["handle"],
                        "expression_sha256": out["expression_sha256"],
                        "input_sha256": out["input_sha256"],
                        "output_type": out["output_type"],
                        "value": out["value"],
                        "exit_code": out["exit_code"],
                        "expr_path": out["expr_path"],
                        "input_path": out["input_path"],
                        "engine": out["engine"],
                        "engine_version": out["engine_version"],
                        "mode": "message_expression",
                    })
                    if not isinstance(out["value"], str) or not out["value"]:
                        return ProofOrCounterexample(
                            exists=False, evidence=evidence,
                            reason="UNSOLVABLE: messageExpression did not yield a string")
                    per_policy[cfg["handle"]] = out["value"]
            messages[oid] = per_policy
        evidence["reference_answer"] = {"messages": messages}
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def _prove_match_gate(self, grounded: GroundedTask) -> ProofOrCounterexample:
        ast = grounded.ast.ast
        configs = _units(grounded)
        requests = ast["requests"]
        ctx = grounded.ast.instance_id
        evidence: dict[str, Any] = {"goal_kind": ast["goal_kind"], "evals": []}
        answer: dict[str, Any] = {}
        applies_seen: set[bool] = set()
        for rid, req in requests.items():
            per_condition: dict[str, bool] = {}
            for cfg, unit in configs:
                for name, expr in cfg["match_conditions"]:
                    rec = engine.eval_bool(
                        expr, {"object": {}, "params": dict(cfg["params"]), "request": req},
                        input_id="%s:%s:%s" % (ctx, rid, name))
                    evidence["evals"].append({
                        "handle": cfg["handle"], "condition": name,
                        "expression_sha256": rec["expression_sha256"],
                        "input_sha256": rec["input_sha256"],
                        "value": rec["value"], "exit_code": rec["exit_code"],
                        "expr_path": rec["expr_path"], "input_path": rec["input_path"],
                        "engine": rec["engine"], "engine_version": rec["engine_version"],
                    })
                    per_condition[name] = bool(rec["value"])
            applies = all(per_condition.values())
            applies_seen.add(applies)
            answer[rid] = {"applies": applies, "conditions": per_condition}
        if len(applies_seen) < 2:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: match gate has <2 distinct applies outcomes")
        evidence["reference_answer"] = {"match": answer}
        return ProofOrCounterexample(exists=True, evidence=evidence)

    # -- verify_terminal ----------------------------------------------------------
    def verify_terminal(
        self, task: GroundedTask, submitted_state_or_answer: Any
    ) -> NativeVerdict:
        ast = task.ast.ast
        goal_kind = ast["goal_kind"]
        oracle = self._oracle(task)
        if oracle is None:
            return NativeVerdict(valid=False, detail={"reason": "oracle_missing"})
        if goal_kind == "repair_compliance":
            return self._verify_repair(task, submitted_state_or_answer)
        if goal_kind == "param_tuning":
            return self._verify_param_tuning(task, submitted_state_or_answer)
        if goal_kind == "boundary_probe":
            return self._verify_boundary_probe(task, submitted_state_or_answer)
        if goal_kind == "exact_decision_set":
            return self._verify_mapping(task, submitted_state_or_answer, "decisions")
        if goal_kind == "message_expression":
            return self._verify_mapping(task, submitted_state_or_answer, "messages")
        if goal_kind == "match_gate":
            return self._verify_mapping(task, submitted_state_or_answer, "match")
        return NativeVerdict(valid=False, detail={"reason": "unknown_goal_kind"})

    def _verify_repair(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        configs = _units(task)
        submitted_objects = (submitted or {}).get("objects") or {}
        ctx = task.ast.instance_id
        checks: list[dict[str, Any]] = []
        ok = True
        for oid in ast["reference_terminal"]:
            obj = submitted_objects.get(oid)
            if obj is None:
                checks.append({"object": oid, "pass": False, "reason": "missing"})
                ok = False
                continue
            all_allow, evals = _object_set_decisions(
                configs, {oid: _public(obj)}, "%s:verify" % ctx)
            checks.append({
                "object": oid, "pass": all_allow,
                "units": {e["handle"]: e["decision"] for e in evals},
                "evidence": [v["input_sha256"] for e in evals for v in e["verdicts"]],
            })
            ok = ok and all_allow
        initial_objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        for guard in ast.get("guards", []):
            oid = guard["object"]
            for path in guard.get("paths", []):
                before = _get_path(initial_objects.get(oid, {}), path)
                after = _get_path(submitted_objects.get(oid, {}), path)
                passed = before == after and before is not None
                checks.append({"check": "guard", "object": oid, "path": path, "pass": passed})
                ok = ok and passed
        return NativeVerdict(valid=ok, detail={"checks": checks})

    def _verify_param_tuning(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        configs = _units(task)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        params = ((submitted or {}).get("params") or {})
        ctx = task.ast.instance_id
        ceiling = params.get("maxReplicas")
        if not isinstance(ceiling, int) or ceiling < 0:
            return NativeVerdict(valid=False, detail={"reason": "params_not_integral_maxReplicas"})
        admit_ok, evals = _object_set_decisions(configs, objects, "%s:vt" % ctx, params=params)
        detail: dict[str, Any] = {
            "admits_all": admit_ok,
            "evidence": [v["input_sha256"] for e in evals for v in e["verdicts"]],
        }
        if not admit_ok:
            return NativeVerdict(valid=False, detail=detail)
        below = {**params, "maxReplicas": ceiling - 1}
        below_ok, below_evals = _object_set_decisions(
            configs, objects, "%s:vt-min" % ctx, params=below)
        detail["minimal"] = not below_ok
        detail["minimality_evidence"] = [
            v["input_sha256"] for e in below_evals for v in e["verdicts"]]
        return NativeVerdict(valid=admit_ok and not below_ok, detail=detail)

    def _verify_boundary_probe(self, task: GroundedTask, submitted: Any) -> NativeVerdict:
        ast = task.ast.ast
        configs = _units(task)
        probes = ast["probe_objects"]
        claimed = ((submitted or {}).get("boundaries") or {})
        ctx = task.ast.instance_id
        checks: list[dict[str, Any]] = []
        ok = True
        for cfg, unit in configs:
            handle = cfg["handle"]
            b = claimed.get(handle)
            if not isinstance(b, int):
                checks.append({"handle": handle, "pass": False, "reason": "missing_or_not_int"})
                ok = False
                continue
            base = probes[handle]
            rec_at = _decision(cfg, unit, _with_replicas(base, b), "%s:vb:%s:at" % (ctx, handle))
            rec_beyond = _decision(cfg, unit, _with_replicas(base, b + 1),
                                   "%s:vb:%s:beyond" % (ctx, handle))
            passed = rec_at["decision"] == "ALLOW" and rec_beyond["decision"] == "DENY"
            checks.append({
                "handle": handle, "pass": passed, "claimed": b,
                "at": rec_at["decision"], "beyond": rec_beyond["decision"],
                "evidence": [v["input_sha256"]
                             for e in (rec_at, rec_beyond) for v in e["verdicts"]],
            })
            ok = ok and passed
        return NativeVerdict(valid=ok, detail={"checks": checks})

    def _verify_mapping(self, task: GroundedTask, submitted: Any, key: str) -> NativeVerdict:
        oracle = self._oracle(task) or {}
        reference = (oracle.get("reference_answer") or {}).get(key)
        if reference is None:
            return NativeVerdict(valid=False, detail={"reason": "oracle_missing_answer"})
        got = (submitted or {}).get(key)
        ok = got == reference
        detail: dict[str, Any] = {"check": key, "match": ok}
        if not ok and isinstance(got, dict):
            detail["submitted_sha256"] = engine.sha256_json(got)
            detail["reference_sha256"] = engine.sha256_json(reference)
        return NativeVerdict(valid=ok, detail=detail)

    # -- certify_semantic_relevance --------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        goal_kind = ast["goal_kind"]
        try:
            if goal_kind in ("repair_compliance", "param_tuning"):
                return self._witness_param_flip(task)
            if goal_kind == "boundary_probe":
                return self._witness_boundary_flip(task)
            if goal_kind == "exact_decision_set":
                return self._witness_policy_subset(task)
            if goal_kind == "message_expression":
                return self._witness_message_shift(task)
            if goal_kind == "match_gate":
                return self._witness_condition_subset(task)
        except engine.CelRunError as exc:
            return SemanticWitness(ok=False, witness={"reason": "cel error: %s" % exc})
        return SemanticWitness(ok=False, witness={"reason": "no witness for %s" % goal_kind})

    def _witness_param_flip(self, task: GroundedTask) -> SemanticWitness:
        """Param policies: perturbed params flip a governed object's verdict."""
        ast = task.ast.ast
        configs = _units(task)
        objects = {oid: _public(o) for oid, o in (ast.get("objects") or {}).items()}
        ctx = task.ast.instance_id
        for cfg, unit in configs:
            params = dict(cfg["params"])
            if "maxReplicas" not in params:
                continue
            for oid, obj in objects.items():
                correct = _decision(cfg, unit, obj, "%s:sem-correct" % ctx)
                wrong_params = {**params, "maxReplicas": int(params["maxReplicas"]) + 10_000}
                wrong = _decision(cfg, unit, obj, "%s:sem-wrong" % ctx, params=wrong_params)
                if correct["decision"] != wrong["decision"]:
                    return SemanticWitness(ok=True, witness={
                        "ok": True,
                        "counterfactuals": [{
                            "object": oid,
                            "policy_id": cfg["policy_id"],
                            "mode": "perturbed_parameters",
                            "correct_decision": correct["decision"],
                            "wrong_decision": wrong["decision"],
                            "correct_input_sha256": correct["verdicts"][0]["input_sha256"],
                            "wrong_input_sha256": wrong["verdicts"][0]["input_sha256"],
                        }],
                    })
        return self._witness_policy_dropped(task)

    def _witness_boundary_flip(self, task: GroundedTask) -> SemanticWitness:
        """Boundary tasks: perturbed params/fields move the boundary itself."""
        ast = task.ast.ast
        configs = _units(task)
        probes = ast["probe_objects"]
        boundaries = {k: int(v) for k, v in ast["reference_answer"]["boundaries"].items()}
        ctx = task.ast.instance_id
        for cfg, unit in configs:
            handle = cfg["handle"]
            boundary = boundaries[handle]
            at = _with_replicas(probes[handle], boundary)
            params = dict(cfg["params"])
            if "maxReplicas" in params:
                correct = _decision(cfg, unit, at, "%s:semb-at" % ctx)
                wrong_params = {**params, "maxReplicas": boundary - 1}
                wrong = _decision(cfg, unit, at, "%s:semb-wrong" % ctx, params=wrong_params)
            elif "maxReplicas" in probes[handle]:
                correct = _decision(cfg, unit, at, "%s:semb-at" % ctx)
                perturbed = copy.deepcopy(at)
                perturbed["maxReplicas"] = boundary - 1
                wrong = _decision(cfg, unit, perturbed, "%s:semb-wrong" % ctx)
            else:
                beyond = _with_replicas(probes[handle], boundary + 1)
                correct = _decision(cfg, unit, beyond, "%s:semb-beyond" % ctx)
                if correct["decision"] != "DENY":
                    continue
                return SemanticWitness(ok=True, witness={
                    "ok": True,
                    "counterfactuals": [{
                        "handle": handle,
                        "policy_id": cfg["policy_id"],
                        "mode": "policy_dropped",
                        "correct_decision": correct["decision"],
                        "wrong_decision": "ALLOW",
                        "correct_input_sha256": correct["verdicts"][0]["input_sha256"],
                        "note": "definitional: empty validation set admits everything",
                    }],
                })
            if correct["decision"] != wrong["decision"]:
                return SemanticWitness(ok=True, witness={
                    "ok": True,
                    "counterfactuals": [{
                        "handle": handle,
                        "policy_id": cfg["policy_id"],
                        "mode": ("perturbed_parameters" if "maxReplicas" in params
                                 else "perturbed_probe_object"),
                        "correct_decision": correct["decision"],
                        "wrong_decision": wrong["decision"],
                        "correct_input_sha256": correct["verdicts"][0]["input_sha256"],
                        "wrong_input_sha256": wrong["verdicts"][0]["input_sha256"],
                    }],
                })
        return SemanticWitness(ok=False, witness={"reason": "no boundary flip found"})

    def _witness_policy_subset(self, task: GroundedTask) -> SemanticWitness:
        """Decision-set: dropping a policy flips an object's overall verdict."""
        ast = task.ast.ast
        configs = _units(task)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        ctx = task.ast.instance_id
        if len(configs) < 2:
            return self._witness_policy_dropped(task)
        for drop_idx in range(len(configs)):
            dropped = configs[drop_idx]
            subset = [c for i, c in enumerate(configs) if i != drop_idx]
            for oid, obj in objects.items():
                full_decisions = []
                for cfg, unit in configs:
                    full_decisions.append(_decision(cfg, unit, obj, "%s:sem-f" % ctx)["decision"])
                sub_decisions = []
                for cfg, unit in subset:
                    sub_decisions.append(_decision(cfg, unit, obj, "%s:sem-sub" % ctx)["decision"])
                full_all = all(d == "ALLOW" for d in full_decisions)
                sub_all = all(d == "ALLOW" for d in sub_decisions)
                if full_all != sub_all:
                    return SemanticWitness(ok=True, witness={
                        "ok": True,
                        "counterfactuals": [{
                            "object": oid,
                            "policy_id": dropped[0]["policy_id"],
                            "mode": "policy_subset",
                            "correct_decision": "ALLOW" if full_all else "DENY",
                            "wrong_decision": "ALLOW" if sub_all else "DENY",
                            "correct_input_sha256": None,
                            "wrong_input_sha256": None,
                        }],
                    })
        return self._witness_policy_dropped(task)

    def _witness_message_shift(self, task: GroundedTask) -> SemanticWitness:
        """Message tasks: a perturbed object yields a different message string."""
        ast = task.ast.ast
        configs = _units(task)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        ctx = task.ast.instance_id
        for cfg, unit in configs:
            if not unit.message_expressions:
                continue
            expr = unit.message_expressions[0]
            for oid, obj in objects.items():
                correct = engine.cel_eval(expr, _inputs(obj, cfg["params"]),
                                          input_id="%s:semm-correct" % ctx)
                perturbed = copy.deepcopy(obj)
                spec = perturbed.get("spec")
                if isinstance(spec, dict) and isinstance(spec.get("replicas"), int):
                    spec["replicas"] = spec["replicas"] + 1
                else:
                    continue
                wrong = engine.cel_eval(expr, _inputs(perturbed, cfg["params"]),
                                        input_id="%s:semm-wrong" % ctx)
                if correct["value"] != wrong["value"]:
                    return SemanticWitness(ok=True, witness={
                        "ok": True,
                        "counterfactuals": [{
                            "object": oid,
                            "policy_id": cfg["policy_id"],
                            "mode": "perturbed_object",
                            "correct_value": correct["value"],
                            "wrong_value": wrong["value"],
                            "correct_input_sha256": correct["input_sha256"],
                            "wrong_input_sha256": wrong["input_sha256"],
                        }],
                    })
        return SemanticWitness(ok=False, witness={"reason": "no message shift found"})

    def _witness_condition_subset(self, task: GroundedTask) -> SemanticWitness:
        """Match gate: dropping one matchCondition flips a request's applies."""
        ast = task.ast.ast
        configs = _units(task)
        requests = ast["requests"]
        ctx = task.ast.instance_id
        for cfg, unit in configs:
            conditions = list(cfg["match_conditions"])
            if len(conditions) < 2:
                continue
            for drop_idx, (drop_name, _expr) in enumerate(conditions):
                subset = [c for i, c in enumerate(conditions) if i != drop_idx]
                for rid, req in requests.items():
                    full = all(
                        engine.eval_bool(e, {"object": {}, "params": dict(cfg["params"]),
                                             "request": req},
                                         input_id="%s:semc-f" % ctx)["value"]
                        for _n, e in conditions)
                    sub = all(
                        engine.eval_bool(e, {"object": {}, "params": dict(cfg["params"]),
                                             "request": req},
                                         input_id="%s:semc-s" % ctx)["value"]
                        for _n, e in subset)
                    if full != sub:
                        return SemanticWitness(ok=True, witness={
                            "ok": True,
                            "counterfactuals": [{
                                "request": rid,
                                "policy_id": cfg["policy_id"],
                                "mode": "condition_subset",
                                "dropped_condition": drop_name,
                                "correct_applies": bool(full),
                                "wrong_applies": bool(sub),
                            }],
                        })
        return SemanticWitness(ok=False, witness={"reason": "no condition flip found"})

    def _witness_policy_dropped(self, task: GroundedTask) -> SemanticWitness:
        """Definitional counterfactual: empty policy set admits everything."""
        ast = task.ast.ast
        objects = {oid: _public(o) for oid, o in (ast.get("objects") or {}).items()}
        configs = _units(task)
        if not objects or not configs:
            return SemanticWitness(ok=False, witness={"reason": "no objects for witness"})
        ctx = task.ast.instance_id
        for oid in sorted(objects):
            for cfg, unit in configs:
                correct = _decision(cfg, unit, objects[oid], "%s:semd" % ctx)
                if correct["decision"] != "DENY":
                    continue
                return SemanticWitness(ok=True, witness={
                    "ok": True,
                    "counterfactuals": [{
                        "object": oid,
                        "policy_id": cfg["policy_id"],
                        "mode": "policy_dropped",
                        "correct_decision": correct["decision"],
                        "wrong_decision": "ALLOW",
                        "correct_input_sha256": correct["verdicts"][0]["input_sha256"],
                        "note": "definitional: empty validation set admits everything",
                    }],
                })
        return SemanticWitness(ok=False, witness={"reason": "no denying object for witness"})

    # -- oracle persistence ------------------------------------------------------
    def write_oracle(self, grounded_task: GroundedTask, extra: dict[str, Any]) -> Path:
        ast = grounded_task.ast.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "goal_kind": ast["goal_kind"],
            "policy_configs": ast["policy_configs"],
            "guards": ast.get("guards", []),
            "reference_terminal_sha256": extra.get("reference_terminal_sha256"),
            "reference_params": extra.get("reference_params"),
            "reference_answer": extra.get("reference_answer"),
            "engine": engine.ENGINE,
            "engine_version": engine.ENGINE_VERSION,
            "source_commit": engine.UPSTREAM_COMMIT,
        }
        path = ORACLE_DIR / ("%s.json" % grounded_task.oracle_ref)
        _write_json_atomic(path, payload)
        return path

    def _oracle(self, task: GroundedTask) -> dict[str, Any] | None:
        path = ORACLE_DIR / ("%s.json" % task.oracle_ref)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _get_path(obj: Any, path: str) -> Any:
    cur = obj
    for seg in path.split("."):
        if not isinstance(cur, dict) or seg not in cur:
            return None
        cur = cur[seg]
    return cur


def _with_replicas(obj: dict[str, Any], replicas: int) -> dict[str, Any]:
    """Probe object with the replicas field set (spec.replicas or top-level)."""
    new = copy.deepcopy(obj)
    if isinstance(new.get("spec"), dict):
        new["spec"]["replicas"] = replicas
    else:
        new["replicas"] = replicas
    return new
