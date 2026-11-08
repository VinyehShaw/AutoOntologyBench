"""L01 NativeVerifierPlugin: pinned OPA evaluation of Gatekeeper task artifacts.

- ``prove_exists``: replay the candidate's reference solution / compute the exact
  reference answer natively (opa eval per governed object per unit) and prove the
  initial state is NOT already solved (nontrivial start).
- ``verify_terminal``: check a submitted terminal state (``{"objects": ...}``) or
  decision-set answer (``{"decisions": ...}``) against the persisted oracle.
- ``certify_semantic_relevance``: counterfactual witness — under a perturbed
  (wrong) constraint configuration, or with the policy removed entirely, at
  least one governed object's native verdict flips. Recorded with the same
  command+sha evidence as the main verdicts.

Oracle payloads live at ``bank/staging/l01/oracle/<oracle_ref>.json`` (staging
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
from benchmark.plugins.task_family.l01 import engine
from benchmark.plugins.task_family.l01.engine import LoadedUnit

REPO_ROOT = Path(__file__).resolve().parents[4]
ORACLE_DIR = REPO_ROOT / "bank" / "staging" / "l01" / "oracle"


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


def _load_units(grounded: GroundedTask) -> list[LoadedUnit]:
    return [
        engine.load_unit(p["unit"], p["sample"])
        for p in grounded.ast.ast["policy_units"]
    ]


def _public(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if not k.startswith("__")}


def _inventory_for(objects: dict[str, Any], exclude_oid: str) -> list[dict[str, Any]]:
    return [_public(o) for oid, o in objects.items() if oid != exclude_oid]


class GatekeeperOpaVerifier:
    """NativeVerifierPlugin for lane L01."""

    plugin_id = "l01.opa-verifier"

    # -- prove_exists -----------------------------------------------------------
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        goal_kind = ast["goal_kind"]
        units = _load_units(grounded_task)
        objects = {oid: _public(o) for oid, o in ast["objects"].items()}
        ctx = grounded_task.ast.instance_id
        evidence: dict[str, Any] = {"evals": [], "goal_kind": goal_kind}
        if goal_kind == "exact_decision_set":
            answer: dict[str, dict[str, str]] = {}
            for oid, obj in objects.items():
                per_unit: dict[str, str] = {}
                for p, loaded in zip(ast["policy_units"], units):
                    if obj.get("kind") not in loaded.match_kinds:
                        continue
                    rec = engine.opa_eval_object(loaded, obj, context="%s:prove" % ctx)
                    evidence["evals"].append(rec)
                    per_unit[p["rule_unit_id"]] = rec["decision"]
                answer[oid] = per_unit
            decisions = [d for per in answer.values() for d in per.values()]
            if not answer or len(set(decisions)) < 2:
                return ProofOrCounterexample(
                    exists=False,
                    evidence=evidence,
                    reason="TRIVIAL: decision set has <2 distinct outcomes",
                )
            evidence["reference_answer"] = answer
            return ProofOrCounterexample(exists=True, evidence=evidence)

        reference_terminal = {
            oid: _public(o) for oid, o in (ast.get("reference_terminal") or {}).items()
        }
        cluster_objects = ast.get("cluster_inventory") or None
        initial_violations = 0
        for oid, obj in objects.items():
            inv = _inventory_for(objects, oid) if ast.get("inventory_required") else None
            rec_set = engine.eval_object_set(units, obj, context="%s:init" % ctx,
                                             inventory=inv, cluster_objects=cluster_objects)
            evidence["evals"].extend(rec_set["verdicts"])
            if not rec_set["compliant"]:
                initial_violations += 1
        if initial_violations == 0:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: initial state already compliant",
            )
        terminal_bad: list[str] = []
        for oid, obj in reference_terminal.items():
            inv = (
                _inventory_for(reference_terminal, oid)
                if ast.get("inventory_required")
                else None
            )
            rec_set = engine.eval_object_set(units, obj, context="%s:ref" % ctx,
                                             inventory=inv, cluster_objects=cluster_objects)
            evidence["evals"].extend(rec_set["verdicts"])
            if not rec_set["compliant"]:
                terminal_bad.append(oid)
        if terminal_bad:
            return ProofOrCounterexample(
                exists=False,
                evidence=evidence,
                reason="UNSOLVABLE: reference terminal violates %s" % terminal_bad,
            )
        evidence["reference_terminal_sha256"] = {
            oid: engine.sha256_json(o) for oid, o in reference_terminal.items()
        }
        return ProofOrCounterexample(exists=True, evidence=evidence)

    # -- verify_terminal ----------------------------------------------------------
    def verify_terminal(
        self, task: GroundedTask, submitted_state_or_answer: Any
    ) -> NativeVerdict:
        ast = task.ast.ast
        oracle = self._oracle(task)
        if ast["goal_kind"] == "exact_decision_set":
            submitted = (submitted_state_or_answer or {}).get("decisions")
            reference = (oracle or {}).get("reference_answer")
            if reference is None:
                return NativeVerdict(valid=False, detail={"reason": "oracle_missing"})
            ok = submitted == reference
            detail: dict[str, Any] = {"check": "exact_decision_set", "match": ok}
            if not ok and isinstance(submitted, dict):
                diffs = {
                    oid: {u: (submitted.get(oid, {}).get(u), ref.get(u))
                          for u in ref if submitted.get(oid, {}).get(u) != ref.get(u)}
                    for oid, ref in reference.items()
                }
                detail["mismatches"] = {k: v for k, v in diffs.items() if v}
            return NativeVerdict(valid=ok, detail=detail)

        units = _load_units(task)
        submitted_objects = (submitted_state_or_answer or {}).get("objects") or {}
        reference = (oracle or {}).get("reference_terminal_sha256") or {}
        ctx = task.ast.instance_id
        checks: list[dict[str, Any]] = []
        ok = True
        cluster_objects = ast.get("cluster_inventory") or None
        for oid in reference:
            obj = submitted_objects.get(oid)
            if obj is None:
                checks.append({"object": oid, "pass": False, "reason": "missing"})
                ok = False
                continue
            inv = (
                [o for k, o in submitted_objects.items() if k != oid]
                if ast.get("inventory_required")
                else None
            )
            rec_set = engine.eval_object_set(units, _public(obj), context="%s:verify" % ctx,
                                             inventory=inv, cluster_objects=cluster_objects)
            checks.append({
                "object": oid, "pass": rec_set["compliant"],
                "units": {v["rule_unit_id"]: v["decision"] for v in rec_set["verdicts"]},
                "evidence": [v["input_sha256"] for v in rec_set["verdicts"]],
            })
            ok = ok and rec_set["compliant"]
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

    # -- certify_semantic_relevance --------------------------------------------
    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        units = _load_units(task)
        initial = {oid: _public(o) for oid, o in ast["objects"].items()}
        reference = {oid: _public(o) for oid, o in (ast.get("reference_terminal") or {}).items()}
        ctx = task.ast.instance_id
        cluster_objects = ast.get("cluster_inventory") or None
        witness: dict[str, Any] = {"counterfactuals": []}
        phases = [("initial", initial)]
        if reference:
            phases.append(("reference", reference))
        for phase, objects in phases:
            for idx, (p, loaded) in enumerate(zip(ast["policy_units"], units)):
                mech = p.get("mechanism", "")
                for oid, obj in objects.items():
                    if obj.get("kind") not in loaded.match_kinds:
                        continue
                    inv = _inventory_for(objects, oid) if ast.get("inventory_required") else None
                    correct = engine.opa_eval_object(
                        loaded, obj, context="%s:sem-correct" % ctx, inventory=inv,
                        cluster_objects=cluster_objects)
                    wrong = self._counterfactual_verdict(
                        mech, p, loaded, obj, objects, oid, ctx, idx, inv,
                        cluster_objects=cluster_objects)
                    if wrong is None:
                        continue
                    if correct["decision"] != wrong["decision"]:
                        witness["counterfactuals"].append({
                            "phase": phase,
                            "object": oid,
                            "rule_unit_id": p["rule_unit_id"],
                            "mode": wrong["mode"],
                            "correct_decision": correct["decision"],
                            "wrong_decision": wrong["decision"],
                            "correct_evidence": correct,
                            "wrong_evidence": {k: v for k, v in wrong.items() if k != "mode"},
                        })
                        witness["ok"] = True
                        return SemanticWitness(ok=True, witness=witness)
        return SemanticWitness(ok=False, witness={**witness, "reason": "no verdict flip found"})

    def _counterfactual_verdict(
        self,
        mech: str,
        payload: dict[str, Any],
        loaded: LoadedUnit,
        obj: dict[str, Any],
        objects: dict[str, Any],
        oid: str,
        ctx: str,
        idx: int,
        inv: list[dict[str, Any]] | None,
        cluster_objects: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        perturbed = _perturb_params(mech, payload.get("parameters") or {}, obj)
        if perturbed is not None:
            wrong_unit = engine.load_unit_with_params(loaded, perturbed)
            rec = engine.opa_eval_object(
                wrong_unit, obj, context="%s:sem-wrong" % ctx, inventory=inv,
                cluster_objects=cluster_objects)
            rec["mode"] = "perturbed_parameters"
            return rec
        # No parameters to perturb: the counterfactual is "policy absent from the
        # constraint set", which is definitionally ALLOW (no unit -> no violation).
        return {
            "mode": "policy_dropped",
            "decision": "ALLOW",
            "violation_count": 0,
            "violations": [],
            "input_sha256": None,
            "command": [],
            "exit_code": None,
            "log_path": None,
            "engine": engine.ENGINE,
            "engine_version": engine.OPA_VERSION,
            "rule_unit_id": payload.get("rule_unit_id", ""),
            "note": "definitional: empty constraint set admits everything",
        }

    # -- oracle persistence ------------------------------------------------------
    def write_oracle(self, grounded_task: GroundedTask, extra: dict[str, Any]) -> Path:
        ast = grounded_task.ast.ast
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "policy_units": ast["policy_units"],
            "guards": ast.get("guards", []),
            "reference_terminal_sha256": extra.get("reference_terminal_sha256"),
            "reference_answer": extra.get("reference_answer"),
            "engine": engine.ENGINE,
            "engine_version": engine.OPA_VERSION,
            "source_commit": engine.GK_COMMIT,
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


def _perturb_params(
    mech: str, params: dict[str, Any], obj: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    """Plausible-but-wrong parameter set per mechanism (counterfactual)."""
    p = copy.deepcopy(params)
    if mech in ("allowedrepos", "disallowedrepos"):
        p["repos"] = ["counterfactual.invalid/"]
        return p
    if mech == "disallowedtags":
        p["tags"] = ["zzz-nonexistent-tag"]
        return p
    if mech == "externalip":
        # Misconfiguration direction that flips a verdict: allow the very IPs the
        # correct policy forbids for this object (overly permissive allowlist).
        ips = ((obj or {}).get("spec") or {}).get("externalIPs") or []
        if ips:
            p["allowedIPs"] = sorted(set(ips))
            return p
        p["allowedIPs"] = ["198.51.100.99"]
        return p
    if mech == "replicalimits":
        ranges = p.get("ranges") or [{"min_replicas": 1, "max_replicas": 2}]
        p["ranges"] = [{**r, "max_replicas": max(int(r.get("min_replicas", 1)) - 1, 0)
                        if int(r.get("max_replicas", 2)) > 0 else 1} for r in ranges]
        if all(int(r.get("max_replicas", 0)) >= int(r.get("min_replicas", 0)) for r in p["ranges"]):
            p["ranges"] = [{"min_replicas": 9, "max_replicas": 9}]
        return p
    if mech in ("requiredlabels", "requiredannotations"):
        key = "labels" if mech == "requiredlabels" else "annotations"
        specs = p.get(key) or []
        if specs:
            p[key] = [{**s, "allowedRegex": "^zzz-never-matches$"} if "allowedRegex" in s
                      else {**s, "key": "zzz-absent-key"} for s in specs]
            return p
        return None
    if mech in ("containerlimits", "containerrequests"):
        p["cpu"] = "1m"
        p["memory"] = "1Mi"
        return p
    if mech == "containerresourceratios":
        p["ratio"] = "0.0001" if str(p.get("ratio")) != "0.0001" else "1000000"
        return p
    if mech == "ephemeralstoragelimit":
        p["ephemeral-storage"] = "1Mi"
        return p
    if mech == "storageclass":
        if p.get("allowedStorageClasses"):
            p["allowedStorageClasses"] = ["zzz-never-class"]
            return p
        return None
    return None
