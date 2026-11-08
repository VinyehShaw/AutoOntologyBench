"""20K harness adapter for lane L01 (Gatekeeper / pinned OPA 0.70.0).

The L01 private oracle payload is lane-specific (``policy_units``, ``guards``,
``reference_terminal_sha256``, ``reference_answer``); it carries no top-level
``ast``.  The native ``GatekeeperOpaVerifier.verify_terminal`` only needs the
rebuilt AST fields that are split between the *public* task row (``objects``)
and the mounted oracle (``policy_units`` / ``guards`` / reference hashes).  This
adapter rebuilds that AST exactly the way the lane's own generators/audit do:

* ``goal_kind`` comes from the pinned family registry (legacy + 20K expansion);
* ``objects`` come from ``row.public``;
* ``policy_units`` / ``guards`` come from the mounted oracle payload;
* ``inventory_required`` is true only for the ``organize_fleet`` family, and
  ``cluster_inventory`` is reloaded from the pinned inventory fixtures via
  ``engine.load_inventory_fixtures`` (the same call the generator uses).

Reference terminal *values* are intentionally not in the oracle (only their
sha256s); for positive self-checks that need them, :meth:`reference_submission`
replays the lane's deterministic proposal stream (same seeds and code path as
``bank/audits/aob-20000-audit-v3/l01/scripts/audit_v3.py``).
"""
from __future__ import annotations

import os
import subprocess
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

#: Seeds used by the pinned L01 build/audit (legacy 8K then 20K expansion).
LEGACY_SEED = 20261027
NEW_SEED = 20261101


#: Lazily regenerated candidate ASTs, keyed by instance_id; one family is
#: proposed at a time so a self-check does not pay for the whole lane.
_CANDIDATE_CACHE: dict[str, Any] = {}
_FAMILIES_LOADED: set[str] = set()


def _family_spec(family_id: str) -> Any:
    try:
        from benchmark.plugins.task_family.l01 import expansion_20k as exp
        from benchmark.plugins.task_family.l01 import families as fam
    except ImportError as exc:
        raise LaneNotSupported(
            "L01 task-family registry not importable: %s" % exc) from exc

    spec = fam.FAMILY_BY_ID.get(family_id)
    if spec is not None:
        return spec
    return next((s for s in exp.NEW_FAMILIES if s.family_id == family_id), None)


def _make_verifier_class() -> Any:
    from benchmark.plugins.verifier.l01.opa_verifier import GatekeeperOpaVerifier

    class L01Verifier(GatekeeperOpaVerifier):
        """Native verifier with harness-injected oracle payloads.

        ``GatekeeperOpaVerifier._oracle`` is hardcoded to the legacy 8K staging
        directory; 20K-expansion payloads live under ``bank/staging_20k`` (or an
        external ``--oracle`` mount).  The adapter registers the payload it
        already loaded, so scoring never depends on the verifier's disk path.
        """

        def __init__(self) -> None:
            self._payloads: dict[str, dict[str, Any]] = {}

        def _oracle(self, task: Any) -> dict[str, Any] | None:
            payload = self._payloads.get(task.oracle_ref)
            if payload is not None:
                return payload
            return super()._oracle(task)

    return L01Verifier


def _candidate_for(task: Any) -> Any:
    """Regenerate one L01 candidate via the lane's own proposal stream."""
    try:
        from benchmark.plugins.task_family.l01 import expansion_20k as exp
        from benchmark.plugins.task_family.l01 import families as fam
        from source_registry.fetchers import l01_gatekeeper as fetcher
    except ImportError as exc:
        raise LaneNotSupported(
            "L01 proposal replay dependencies not importable: %s" % exc) from exc

    family_id = task.family_id
    if family_id not in _FAMILIES_LOADED:
        spec = _family_spec(family_id)
        if spec is None:
            raise LaneNotSupported(
                "L01 family %r is not in the pinned L01 registry" % family_id)
        is_expansion = any(s.family_id == family_id for s in exp.NEW_FAMILIES)
        if is_expansion:
            try:
                exp.register()
            except Exception as exc:  # noqa: BLE001 - environment gap
                raise LaneNotSupported(
                    "L01 could not register the 20K expansion families: %s: %s"
                    % (type(exc).__name__, exc)) from exc
        seed = NEW_SEED if is_expansion else LEGACY_SEED
        snapshot = fetcher.load_snapshot()
        plugin = fam.GatekeeperTaskFamilies()
        try:
            for candidate in plugin.propose(
                snapshot, family=family_id, seed=seed,
                constraints={"quota_scale": 1.0, "instances_per_goal": 5},
            ):
                _CANDIDATE_CACHE[candidate.instance_id] = candidate
        except Exception as exc:  # native sample load is a real environment gap
            raise LaneNotSupported(
                "L01 could not replay the deterministic proposal stream for "
                "family %s: %s: %s"
                % (family_id, type(exc).__name__, exc)) from exc
        _FAMILIES_LOADED.add(family_id)
    candidate = _CANDIDATE_CACHE.get(task.instance_id)
    if candidate is None:
        raise LaneNotSupported(
            "L01 proposal replay did not reproduce instance %r (family %s)"
            % (task.instance_id, family_id))
    return candidate


class Adapter(LaneAdapter):
    lane_id = "L01"

    # -- construction --------------------------------------------------------
    def build_verifier(self) -> Any:
        try:
            from benchmark.plugins.task_family.l01 import engine
        except ImportError as exc:
            raise LaneNotSupported("L01 lane engine not importable: %s" % exc) from exc

        opa = (self.lane_kw.get("opa_bin")
               or os.environ.get("AOB_L01_OPA")
               or engine.OPA_BIN)
        opa_path = Path(opa)
        if not opa_path.is_file() or not os.access(str(opa_path), os.X_OK):
            raise LaneNotSupported(
                "L01 requires the pinned OPA %s binary; missing or not "
                "executable at %s (pass --lane-kw L01.opa_bin=<path> or set "
                "AOB_L01_OPA)" % (engine.OPA_VERSION, opa_path))
        if str(opa_path) != engine.OPA_BIN:
            # engine.opa_eval_object reads the module global at call time.
            engine.OPA_BIN = str(opa_path)
        if not engine.GK_ROOT.is_dir():
            raise LaneNotSupported(
                "L01 requires the pinned Gatekeeper source tree at %s"
                % engine.GK_ROOT)
        try:
            proc = subprocess.run(
                [str(opa_path), "version"], capture_output=True, text=True,
                timeout=30)
        except Exception as exc:  # noqa: BLE001 - reported as environment gap
            raise LaneNotSupported(
                "L01 OPA binary failed to run (%s): %s: %s"
                % (opa_path, type(exc).__name__, exc)) from exc
        if proc.returncode != 0:
            raise LaneNotSupported(
                "L01 OPA binary %s exited %d: %s"
                % (opa_path, proc.returncode, (proc.stderr or proc.stdout)[:200]))
        if ("Version: %s" % engine.OPA_VERSION) not in proc.stdout:
            raise LaneNotSupported(
                "L01 requires OPA %s; %s reports %r"
                % (engine.OPA_VERSION, opa_path,
                   (proc.stdout or "").strip().splitlines()[:2]))
        try:
            cls = _make_verifier_class()
            return cls()
        except ImportError as exc:
            raise LaneNotSupported(
                "L01 verifier not importable: %s" % exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise LaneNotSupported(
                "L01 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc

    # -- AST rebuild ---------------------------------------------------------
    def build_ast(self, task: Any, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        from benchmark.plugins.task_family.l01 import engine

        spec = _family_spec(task.family_id)
        if spec is None:
            raise LaneNotSupported(
                "L01 family %r is not in the pinned L01 registry"
                % task.family_id)
        units = oracle_payload.get("policy_units")
        if not isinstance(units, list) or not units:
            raise LaneNotSupported(
                "L01 oracle payload for %s carries no policy_units"
                % task.instance_id)
        objects = task.public.get("objects")
        if not isinstance(objects, dict):
            raise LaneNotSupported(
                "L01 public row for %s carries no objects" % task.instance_id)
        ast: dict[str, Any] = {
            "goal_kind": spec.goal_kind,
            "policy_units": units,
            "objects": objects,
            "guards": oracle_payload.get("guards") or [],
            "reference_terminal_sha256":
                oracle_payload.get("reference_terminal_sha256"),
            "reference_answer": oracle_payload.get("reference_answer"),
        }
        if spec.goal_kind == "organize_fleet":
            ast["inventory_required"] = True
        cluster: list[dict[str, Any]] = []
        for unit in units:
            try:
                cluster.extend(engine.load_inventory_fixtures(
                    unit["unit"], unit["sample"]))
            except Exception as exc:  # noqa: BLE001 - environment gap
                raise LaneNotSupported(
                    "L01 could not load pinned inventory fixtures for %s/%s: "
                    "%s: %s" % (unit.get("unit"), unit.get("sample"),
                                type(exc).__name__, exc)) from exc
        if cluster:
            ast["cluster_inventory"] = cluster
        # Make the native verifier consult the payload the harness loaded
        # (release_v3 mixes legacy 8K and 20K-expansion oracle directories).
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
        if oracle_payload.get("reference_answer") is not None:
            return {"decisions": oracle_payload["reference_answer"]}
        candidate = _candidate_for(task)
        return {"objects": {
            oid: {k: v for k, v in obj.items() if not k.startswith("__")}
            for oid, obj in (candidate.ast.get("reference_terminal") or {}).items()
        }}

    def reference_answer(self, oracle_payload: dict[str, Any],
                         task: Any = None) -> Any:
        ref = oracle_payload.get("reference_answer")
        if ref is not None:
            return {"decisions": ref}
        if task is None:
            raise LaneNotSupported(
                "L01 oracle payload carries only reference_terminal sha256s; "
                "pass the task row to Adapter.reference_submission(task, payload)")
        return self.reference_submission(task, oracle_payload)