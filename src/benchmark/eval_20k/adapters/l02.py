"""20K harness adapter for lane L02 (Kubernetes VAP / pinned cel-go v0.26.1).

The L02 oracle payload is lane-specific (``goal_kind``, ``policy_configs``,
``guards``, reference hashes/answers) and has no top-level ``ast``.  The native
``KubernetesCelVerifier.verify_terminal`` needs an AST split between the public
row and the oracle:

* ``goal_kind`` / ``policy_configs`` / ``guards`` from the oracle payload;
* ``objects`` (and, for ``match_gate``, ``requests``) from ``row.public``;
* ``reference_terminal`` *keys* from ``reference_terminal_sha256`` (the repair
  terminal check iterates keys; reference values are not in the oracle);
* ``probe_objects`` for ``boundary_probe`` are not published in either the
  public row or the oracle, so they are reconstructed by replaying the lane's
  deterministic proposal stream (the exact path used by
  ``bank/audits/aob-20000-audit-v3/l02/scripts/audit_v3.py``).

Positive self-checks that need full reference terminal values use the same
proposal replay via :meth:`reference_submission`.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import (
    LaneAdapter,
    LaneNotSupported,
    Verdict,
    _json_safe,
    build_grounded,
)
from benchmark.eval_20k.oracle import OracleNotFound

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Seeds used by the pinned L02 build/audit (legacy 8K then 20K expansion).
LEGACY_SEED = 20261028
NEW_SEED = 20261101

_CANDIDATE_CACHE: dict[str, Any] = {}
_FAMILIES_LOADED: set[str] = set()


def _make_verifier_class() -> Any:
    from benchmark.plugins.verifier.l02.cel_verifier import KubernetesCelVerifier

    class L02Verifier(KubernetesCelVerifier):
        """Native verifier with harness-injected oracle payloads.

        ``KubernetesCelVerifier._oracle`` is hardcoded to the legacy 8K staging
        directory; 20K-expansion payloads live under ``bank/staging_20k`` (or an
        external ``--oracle`` mount), so the adapter registers what it loaded.
        """

        def __init__(self) -> None:
            self._payloads: dict[str, dict[str, Any]] = {}

        def _oracle(self, task: Any) -> dict[str, Any] | None:
            payload = self._payloads.get(task.oracle_ref)
            if payload is not None:
                return payload
            return super()._oracle(task)

    return L02Verifier


def _candidate_for(task: Any) -> Any:
    """Regenerate one L02 candidate via the lane's own proposal stream."""
    try:
        from benchmark.plugins.task_family.l02 import expansion_20k as exp
        from benchmark.plugins.task_family.l02 import families as fam
        from source_registry.fetchers import l02_kubernetes_cel as fetcher
    except ImportError as exc:
        raise LaneNotSupported(
            "L02 proposal replay dependencies not importable: %s" % exc) from exc

    family_id = task.family_id
    if family_id not in _FAMILIES_LOADED:
        is_expansion = family_id in set(exp.list_families())
        if not is_expansion and family_id not in fam.FAMILY_BY_ID:
            raise LaneNotSupported(
                "L02 family %r is not in the pinned L02 registry" % family_id)
        spec = fam.FAMILY_BY_ID.get(family_id)
        if spec is None and is_expansion:
            try:
                exp.register()
            except Exception as exc:  # noqa: BLE001 - environment gap
                raise LaneNotSupported(
                    "L02 could not register the 20K expansion families: %s: %s"
                    % (type(exc).__name__, exc)) from exc
            spec = fam.FAMILY_BY_ID.get(family_id)
        if spec is None:
            raise LaneNotSupported(
                "L02 family %r is not in the pinned L02 registry" % family_id)
        seed = NEW_SEED if is_expansion else LEGACY_SEED
        plugin = (exp.KubernetesCelTaskFamilies20K() if is_expansion
                  else fam.KubernetesCelTaskFamilies())
        snapshot = fetcher.load_snapshot()
        try:
            for candidate in plugin.propose(
                snapshot, family=family_id, seed=seed,
                constraints={"quota_scale": 1.0, "instances_per_goal": 5},
            ):
                _CANDIDATE_CACHE[candidate.instance_id] = candidate
        except Exception as exc:  # noqa: BLE001 - environment gap
            raise LaneNotSupported(
                "L02 could not replay the deterministic proposal stream for "
                "family %s: %s: %s"
                % (family_id, type(exc).__name__, exc)) from exc
        _FAMILIES_LOADED.add(family_id)
    candidate = _CANDIDATE_CACHE.get(task.instance_id)
    if candidate is None:
        raise LaneNotSupported(
            "L02 proposal replay did not reproduce instance %r (family %s)"
            % (task.instance_id, family_id))
    return candidate


class Adapter(LaneAdapter):
    lane_id = "L02"

    # -- construction --------------------------------------------------------
    def build_verifier(self) -> Any:
        try:
            from autoontobench.core.domains import cel
            from benchmark.plugins.task_family.l02 import engine
        except ImportError as exc:
            raise LaneNotSupported("L02 lane engine not importable: %s" % exc) from exc

        helper = Path(cel.HELPER)
        if not helper.is_file() or not os.access(str(helper), os.X_OK):
            raise LaneNotSupported(
                "L02 requires the pinned cel-go %s helper binary; missing or "
                "not executable at %s" % (cel.CEL_GO_VERSION, helper))
        if not engine.PINNED_ROOT.is_dir():
            raise LaneNotSupported(
                "L02 requires the pinned VAP/CEL source tree at %s"
                % engine.PINNED_ROOT)
        try:
            # load_policies compile-checks every expression with the pinned
            # helper, so a successful load is a real runtime support proof.
            engine.load_policies()
        except Exception as exc:  # noqa: BLE001 - environment gap
            raise LaneNotSupported(
                "L02 CEL runtime / pinned policy load failed: %s: %s"
                % (type(exc).__name__, exc)) from exc
        try:
            cls = _make_verifier_class()
            return cls()
        except ImportError as exc:
            raise LaneNotSupported(
                "L02 verifier not importable: %s" % exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise LaneNotSupported(
                "L02 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc

    # -- AST rebuild ---------------------------------------------------------
    def build_ast(self, task: Any, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        goal_kind = oracle_payload.get("goal_kind")
        if not goal_kind:
            raise LaneNotSupported(
                "L02 oracle payload for %s carries no goal_kind"
                % task.instance_id)
        objects = task.public.get("objects") or {}
        ast: dict[str, Any] = {
            "goal_kind": goal_kind,
            "policy_configs": oracle_payload.get("policy_configs") or [],
            "guards": oracle_payload.get("guards") or [],
            "objects": objects,
            "reference_answer": oracle_payload.get("reference_answer"),
            "reference_params": oracle_payload.get("reference_params"),
        }
        if goal_kind == "repair_compliance":
            sha = oracle_payload.get("reference_terminal_sha256") or {}
            # Only the key set is consulted by _verify_repair.
            ast["reference_terminal"] = {oid: {} for oid in sha}
        elif goal_kind == "param_tuning":
            ast["current_params"] = task.public.get("current_params") or {}
        elif goal_kind == "boundary_probe":
            candidate = _candidate_for(task)
            ast["probe_objects"] = candidate.ast.get("probe_objects") or {}
            ast["reference_answer"] = (oracle_payload.get("reference_answer")
                                       or candidate.ast.get("reference_answer"))
        elif goal_kind == "match_gate":
            ast["requests"] = task.public.get("requests") or {}
        ref = oracle_payload.get("oracle_ref") or task.oracle_handle
        self.verifier._payloads[ref] = oracle_payload
        self.verifier._payloads[task.oracle_handle] = oracle_payload
        return ast

    # -- scoring -------------------------------------------------------------
    def evaluate(self, task: Any, prediction: Any, oracle_store: Any) -> Verdict:
        try:
            verifier = self.verifier
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        try:
            oracle_path = str(oracle_store.locate(
                task.lane_id, task.instance_id, task.oracle_handle))
            payload = oracle_store.load(
                task.lane_id, task.instance_id, task.oracle_handle)
        except OracleNotFound as exc:
            return Verdict(status="ORACLE_MISSING", valid=None, reason=str(exc))
        except (OSError, ValueError) as exc:
            return Verdict(status="ERROR", valid=None,
                           reason="oracle payload unreadable: %s" % exc)
        try:
            ast = self.build_ast(task, payload)
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None,
                           reason=str(exc), oracle_path=oracle_path)
        grounded = build_grounded(task.raw, ast)
        try:
            verdict = verifier.verify_terminal(grounded, prediction.answer)
        except Exception as exc:  # noqa: BLE001 - native failure is data
            return Verdict(status="ERROR", valid=None,
                           reason="verify_terminal raised %s: %s"
                                  % (type(exc).__name__, exc),
                           verifier_id=self.verifier_id(verifier),
                           oracle_path=oracle_path)
        valid = bool(getattr(verdict, "valid", False))
        detail = getattr(verdict, "detail", {}) or {}
        if not isinstance(detail, dict):
            detail = {"detail": str(detail)}
        if detail.get("reason") in self.ORACLE_MISSING_REASONS:
            return Verdict(status="ORACLE_MISSING", valid=None,
                           reason="lane verifier reported its own oracle "
                                  "material unreachable: %s" % detail.get("reason"),
                           verifier_id=self.verifier_id(verifier),
                           oracle_path=oracle_path)
        return Verdict(status="SCORED", valid=valid, detail=_json_safe(detail),
                       verifier_id=self.verifier_id(verifier),
                       oracle_path=oracle_path)

    # -- reference / self-check helpers -------------------------------------
    def reference_submission(self, task: Any,
                             oracle_payload: dict[str, Any]) -> Any:
        kind = oracle_payload["goal_kind"]
        if kind == "repair_compliance":
            candidate = _candidate_for(task)
            return {"objects": {
                oid: {k: v for k, v in obj.items() if not k.startswith("__")}
                for oid, obj in
                (candidate.ast.get("reference_terminal") or {}).items()
            }}
        if kind == "param_tuning":
            return {"params": dict(oracle_payload["reference_params"])}
        ref = oracle_payload.get("reference_answer") or {}
        if kind == "boundary_probe":
            return {"boundaries": dict(ref["boundaries"])}
        if kind == "exact_decision_set":
            return {"decisions": ref["decisions"]}
        if kind == "message_expression":
            return {"messages": ref["messages"]}
        if kind == "match_gate":
            return {"match": ref["match"]}
        raise LaneNotSupported("L02 unknown goal_kind %r" % kind)

    def reference_answer(self, oracle_payload: dict[str, Any],
                         task: Any = None) -> Any:
        kind = oracle_payload.get("goal_kind")
        if kind == "repair_compliance":
            if task is None:
                raise LaneNotSupported(
                    "L02 repair oracle carries only reference_terminal sha256s; "
                    "pass the task row to Adapter.reference_submission(task, payload)")
            return self.reference_submission(task, oracle_payload)
        if kind == "param_tuning":
            return {"params": dict(oracle_payload["reference_params"])}
        ref = oracle_payload.get("reference_answer")
        if ref is not None:
            return ref
        raise LaneNotSupported(
            "L02 oracle payload for goal_kind %r exposes no reference answer"
            % kind)