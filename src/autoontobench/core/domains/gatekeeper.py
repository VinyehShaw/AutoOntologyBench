"""Gatekeeper NativeDomain (Review 5B §11.1/§11.2). Frozen ``NativeDomain`` over the
deterministic OFFLINE state machine of SIMULATION_ASSUMPTIONS.md (our design, never a live
cluster); legality = pinned `opa eval` via GatekeeperNativeOracle. Never calls OntologyTool.

W12 containment (Review-5B G2 finding B): ``legal_action``/``step`` contain the
known crash-class inputs — ``create_object`` without ``value``, patch/remove
paths walking into missing intermediates, and projection-shaped or unprojected
subtree paths — converting the raw ``KeyError``/``IndexError``/``TypeError``/
``ValueError`` into a RECORDED error outcome (``legal=None`` /
``feedback.status == "error"``, ``detail.reason == "domain_error"`` carrying the
raw exception class). A contained action is never applied, never counts as
legal/illegal native truth, never consumes its idempotence key, and never kills
the run; the error is persisted to the step ledger. Any other exception still
propagates (real bugs stay loud).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from autoontobench.contracts.models import GoalResult, TaskSpecV5
from autoontobench.core.native_oracle import GatekeeperNativeOracle, sha256_json
from autoontobench.core.protocols import (
    NativeDecision, NativeState, NativeStateAndFeedback, PermissionSet, PublicObservation,
    TypedAction,
)

ALLOWED_OPS = ("patch_object", "create_object", "delete_object")
OBSERVE_RIGHT = "read:state"
REPO_ROOT = Path(__file__).resolve().parents[3]
_SEGMENT = re.compile(r"^([^\[\]]+)(?:\[(\d+)\])?$")

#: Contained crash classes (W12 finding B): exactly the exceptions the
#: dotted-path machinery and the create-object body read can raise on
#: malformed action shapes. Anything else propagates.
_CRASH_CLASSES = (KeyError, IndexError, TypeError, ValueError)


def _domain_error_detail(exc: Exception) -> Dict[str, Any]:
    """Recorded ERROR outcome for a contained crash-class input (never a silent pass)."""
    return {
        "reason": "domain_error",
        "error_class": type(exc).__name__,
        "error": str(exc)[:200],
    }


def _resolve(ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else REPO_ROOT / p


# ---- dotted-path helpers (state machine mechanics, not policy) ----------------
def _parse_path(path: str) -> List[Any]:
    out: List[Any] = []
    for seg in path.split("."):
        m = _SEGMENT.match(seg)
        if not m:
            raise ValueError("bad path segment: %r" % seg)
        out.append(m.group(1))
        if m.group(2) is not None:
            out.append(int(m.group(2)))
    return out


def get_path(obj: Any, path: str) -> Any:
    cur = obj
    for part in _parse_path(path):
        if isinstance(part, int):
            if not isinstance(cur, list) or part >= len(cur):
                raise KeyError(path)
            cur = cur[part]
        else:
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(path)
            cur = cur[part]
    return cur


def set_path(obj: Any, path: str, value: Any) -> None:
    parts = _parse_path(path)
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def del_path(obj: Any, path: str) -> None:
    parts = _parse_path(path)
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    last = parts[-1]
    if isinstance(last, int):
        cur.pop(last)
    else:
        cur.pop(last, None)


def snapshot_sha(state: NativeState) -> str:
    """Snapshot identity = sha256 over the objects map only (episode meta/ledger excluded)."""
    return sha256_json(state.payload.get("objects", {}))


class GatekeeperDomain:
    """NativeDomain over typed k8s-like objects; legality via pinned OPA."""

    def __init__(self, oracle: GatekeeperNativeOracle | None = None,
                 log_root: Path | None = None) -> None:
        root = Path(log_root).resolve() if log_root else (
            REPO_ROOT / "data" / "erule" / "review5" / "software" / "logs"
        )
        self.log_root = root
        self.oracle = oracle if oracle is not None else GatekeeperNativeOracle(root)
        self._episode_oracles: Dict[str, GatekeeperNativeOracle] = {}
        (root / "steps").mkdir(parents=True, exist_ok=True)

    # -- protocol -------------------------------------------------------------
    def reset(self, spec: TaskSpecV5, seed: int) -> NativeState:
        """Pure function of spec refs; seed recorded for protocol conformance only (§5)."""
        initial = json.loads(_resolve(spec.initial_state_ref).read_text(encoding="utf-8"))
        goal = json.loads(_resolve(spec.private_goal_verifier_ref).read_text(encoding="utf-8"))
        payload = {
            "objects": copy.deepcopy(initial["objects"]),
            "ledger": [],
            "applied_keys": [],
            "_episode": {
                "task_id": spec.task_id,
                "seed": seed,
                "forbidden_mutations": goal.get("forbidden_mutations", []),
                "parameter_overrides": goal.get("parameter_overrides", {}),
            },
        }
        return NativeState(payload=payload)

    def observe(self, state: NativeState, rights: PermissionSet) -> PublicObservation:
        """Allowlisted public projection: kinds, names, governed fields only."""
        if OBSERVE_RIGHT not in rights:
            return PublicObservation(payload={"objects": {}})
        view: Dict[str, Any] = {}
        for oid, obj in state.payload["objects"].items():
            meta, spec = obj.get("metadata") or {}, obj.get("spec") or {}
            entry: Dict[str, Any] = {
                "kind": obj.get("kind"), "apiVersion": obj.get("apiVersion"),
                "name": meta.get("name"), "namespace": meta.get("namespace", "default"),
                "labels": copy.deepcopy(meta.get("labels") or {}),
            }
            if "replicas" in spec:
                entry["replicas"] = spec["replicas"]
            if "type" in spec:
                entry["service_type"] = spec["type"]
            for field in ("containers", "initContainers", "ephemeralContainers"):
                if spec.get(field):
                    entry[field] = [{"name": c.get("name"), "image": c.get("image")}
                                    for c in spec[field]]
            view[oid] = entry
        return PublicObservation(payload={"objects": view})

    def legal_action(self, state: NativeState, action: TypedAction) -> NativeDecision:
        """LEGAL iff state-machine guards pass AND the post-state object is natively compliant."""
        op = action.get("op")
        oid = action.get("object")
        if op not in ALLOWED_OPS:
            return NativeDecision(legal=False, detail={"reason": "op_not_allowlisted", "op": op})
        if not action.get("idempotence_key"):
            return NativeDecision(legal=False, detail={"reason": "missing_idempotence_key"})
        if action["idempotence_key"] in state.payload["applied_keys"]:
            return NativeDecision(legal=True, detail={"duplicate": True})
        objects = state.payload["objects"]
        if op in ("patch_object", "delete_object") and oid not in objects:
            return NativeDecision(legal=False, detail={"reason": "object_not_found", "object": oid})
        if op == "create_object" and oid in objects:
            return NativeDecision(legal=False, detail={"reason": "object_exists", "object": oid})
        for forb in state.payload["_episode"]["forbidden_mutations"]:
            if forb.get("object") == oid and op in (forb.get("ops") or []):
                return NativeDecision(
                    legal=False, detail={"reason": "forbidden_mutation", "object": oid, "op": op}
                )
        if op == "delete_object":
            return NativeDecision(legal=True, detail={"admission": "not_applicable"})
        try:
            post = self._post_object(state, action)
        except _CRASH_CLASSES as exc:
            return NativeDecision(legal=None, detail=_domain_error_detail(exc))
        verdict = self._oracle_for(state).evaluate_object(
            post, context="%s:legal" % state.payload["_episode"]["task_id"]
        )
        return NativeDecision(legal=verdict.compliant, detail={
            "evidence_sha256": verdict.evidence_sha256,
            "units": {v.rule_unit_id: v.decision for v in verdict.verdicts}})

    def step(self, state: NativeState, action: TypedAction) -> NativeStateAndFeedback:
        """Apply a legal action; persist before/after sha256 + idempotence key to the ledger."""
        payload = copy.deepcopy(state.payload)
        key = action.get("idempotence_key")
        task_id = payload["_episode"]["task_id"]
        if key in payload["applied_keys"]:
            return NativeStateAndFeedback(
                state=state, feedback={"status": "duplicate", "idempotence_key": key}, cost=0.0
            )
        decision = self.legal_action(state, action)
        before = snapshot_sha(state)
        entry = {"idempotence_key": key, "op": action.get("op"), "object": action.get("object"),
                 "before_sha256": before, "legal": decision.legal}
        contained = decision.legal is None and decision.detail.get("reason") == "domain_error"
        if decision.legal:
            if action["op"] == "patch_object":
                new_obj = self._post_object(state, action)
                payload["objects"][action["object"]] = new_obj
            elif action["op"] == "create_object":
                payload["objects"][action["object"]] = copy.deepcopy(action["value"])
            else:
                del payload["objects"][action["object"]]
        if contained:
            # the idempotence key stays unconsumed so a corrected retry is evaluated fresh
            entry["error_class"] = decision.detail.get("error_class")
        else:
            payload["applied_keys"].append(key)
        entry["after_sha256"] = sha256_json(payload["objects"])
        payload["ledger"].append(entry)
        self._persist_step(task_id, entry)
        status = "error" if contained else ("applied" if decision.legal else "denied")
        feedback = {"status": status, "detail": decision.detail}
        return NativeStateAndFeedback(state=NativeState(payload=payload),
                                      feedback=feedback, cost=1.0)

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        """Independent final-state verification: native compliance + exact values + guards + forbidden."""
        goal = json.loads(_resolve(spec.private_goal_verifier_ref).read_text(encoding="utf-8"))
        initial = json.loads(_resolve(spec.initial_state_ref).read_text(encoding="utf-8"))
        objects = state.payload["objects"]
        checks: List[Dict[str, Any]] = []
        oracle = self._oracle_for(state)

        ok = True
        for oid in goal.get("compliant_objects", []):
            obj = objects.get(oid)
            if obj is None:
                checks.append({"check": "compliant", "object": oid, "pass": False,
                               "reason": "missing"})
                ok = False
                continue
            verdict = oracle.evaluate_object(obj, context="%s:verify" % spec.task_id)
            checks.append({"check": "compliant", "object": oid, "pass": verdict.compliant,
                           "evidence_sha256": verdict.evidence_sha256})
            ok &= verdict.compliant
        for vc in goal.get("value_checks", []):
            try:
                actual = get_path(objects.get(vc["object"], {}), vc["path"])
            except KeyError:
                actual = None
            passed = actual == vc["equals"] and actual is not None
            checks.append({"check": "value", "object": vc["object"], "path": vc["path"],
                           "pass": passed, "actual": actual})
            ok &= passed
        for guard in goal.get("guards", []):
            oid = guard["object"]
            for path in guard.get("paths", []):
                try:
                    now = get_path(objects.get(oid, {}), path)
                    was = get_path(initial["objects"].get(oid, {}), path)
                    passed = sha256_json(now) == sha256_json(was)
                except KeyError:
                    passed = False
                checks.append({"check": "guard", "object": oid, "path": path, "pass": passed})
                ok &= passed
        forb_hits = [
            e for e in state.payload["ledger"]
            if any(f.get("object") == e["object"] and e["op"] in (f.get("ops") or [])
                   and e.get("legal")
                   for f in goal.get("forbidden_mutations", []))
        ]
        checks.append({"check": "forbidden_mutations", "pass": not forb_hits,
                       "violations": forb_hits})
        ok &= not forb_hits
        return GoalResult(goal_reached=ok, evidence={
            "checks": checks,
            "terminal_state_sha256": snapshot_sha(state),
            "evidence_sha256": sha256_json(checks),
        })

    # -- internals --------------------------------------------------------------
    def _oracle_for(self, state: NativeState) -> GatekeeperNativeOracle:
        overrides = state.payload.get("_episode", {}).get("parameter_overrides") or {}
        key = sha256_json(overrides)
        if key not in self._episode_oracles:
            self._episode_oracles[key] = self.oracle.with_overrides(overrides)
        return self._episode_oracles[key]

    def _post_object(self, state: NativeState, action: TypedAction) -> Dict[str, Any]:
        if action["op"] == "create_object":
            return copy.deepcopy(action["value"])
        obj = copy.deepcopy(state.payload["objects"][action["object"]])
        for path in action.get("remove") or []:
            del_path(obj, path)
        for path, value in (action.get("set") or {}).items():
            set_path(obj, path, value)
        return obj

    def _persist_step(self, task_id: str, entry: Dict[str, Any]) -> None:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id)
        path = self.log_root / "steps" / ("%s.steps.jsonl" % safe)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
