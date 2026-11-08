"""Per-lane dispatch to the existing ``NativeVerifierPlugin`` implementations.

The harness never re-implements lane semantics: scoring a prediction means
calling the lane's own ``verify_terminal(grounded_task, answer)`` from
``benchmark.plugins.verifier.<lane>`` on a ``GroundedTask`` rebuilt from the
public task row plus the mounted private oracle payload (the same rebuild the
lanes' own ``recheck`` CLIs perform).

Lane support is *honest by construction*:

* a lane is ``SUPPORTED`` only if its verifier module imports, constructs with
  the provided resources (index dirs, stores, binaries), and its oracle
  payloads carry the task AST needed to rebuild the ``GroundedTask``;
* anything else is reported ``NOT_SUPPORTED`` with the precise reason (missing
  native dependency, missing index, oracle layout without a task AST, ...).
  No lane is ever scored partially or with a stand-in metric.

Per-lane resources are passed with ``--lane-kw L05.index_dir=...`` or
environment variables (e.g. ``AOB_L05_INDEX``); see the CLI help.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

from .oracle import OracleStore
from .submission import Prediction
from .tasks import TaskRecord

REPO_ROOT = Path(__file__).resolve().parents[2]

#: All lanes present in the 20K build (lane order of the build manifest).
ALL_LANES: tuple[str, ...] = tuple(f"L{i:02d}" for i in range(1, 17))


class LaneNotSupported(RuntimeError):
    """Raised when a lane cannot be evaluated in the current environment."""


@dataclass(frozen=True)
class Verdict:
    """Normalised per-task verifier outcome."""

    status: str  # SCORED | NOT_SUPPORTED | ORACLE_MISSING | ERROR
    valid: bool | None
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    verifier_id: str = ""
    oracle_path: str = ""


def build_grounded(row: dict[str, Any], ast: dict[str, Any]) -> GroundedTask:
    """Rebuild the GroundedTask exactly like the lanes' recheck CLIs do."""
    candidate = CandidateTaskAST(
        lane_id=row["lane_id"],
        family_id=row["family_id"],
        base_goal_id=row["base_goal_id"],
        instance_id=row["instance_id"],
        regime=row["task_regime"],
        semantic_profile=row["semantic_profile"],
        ast=ast,
        source_unit_ids=tuple(row.get("source_unit_ids") or ()),
    )
    return GroundedTask(
        ast=candidate,
        public=row["public"],
        oracle_ref=row["oracle_handle"],
        native_evidence={},
    )


class LaneAdapter:
    """Base adapter: dispatch ``verify_terminal`` for one lane."""

    lane_id: str = ""

    #: Verdict detail reasons meaning "the lane verifier could not reach its
    #: own oracle material" — surfaced as ORACLE_MISSING (environment gap),
    #: never as a task failure.
    ORACLE_MISSING_REASONS: tuple[str, ...] = ("oracle_missing",)

    def __init__(self, lane_kw: dict[str, str] | None = None) -> None:
        self.lane_kw = dict(lane_kw or {})
        self._verifier: Any = None
        self._support_error: str | None = None

    # -- construction --------------------------------------------------------

    def build_verifier(self) -> Any:
        """Construct (and memoize) the lane verifier or raise LaneNotSupported."""
        raise NotImplementedError

    @property
    def verifier(self) -> Any:
        if self._verifier is not None:
            return self._verifier
        if self._support_error is not None:
            raise LaneNotSupported(self._support_error)
        try:
            self._verifier = self.build_verifier()
        except LaneNotSupported as exc:
            self._support_error = str(exc)
            raise
        return self._verifier

    def support(self) -> tuple[bool, str, str]:
        """(supported, verifier_id, reason) — construction attempted once."""
        try:
            verifier = self.verifier
            return True, self.verifier_id(verifier), ""
        except LaneNotSupported as exc:
            return False, "", str(exc)

    # -- oracle ---------------------------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        ast = oracle_payload.get("ast")
        if not isinstance(ast, dict):
            raise LaneNotSupported(
                "oracle payload for lane %s carries no task AST "
                "(cannot rebuild GroundedTask from the mounted oracle)"
                % self.lane_id)
        return ast

    # -- scoring ---------------------------------------------------------------

    def evaluate(self, task: TaskRecord, prediction: Prediction,
                 oracle_store: OracleStore) -> Verdict:
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
            ast = self.extract_ast(payload)
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        grounded = build_grounded(task.raw, ast)
        try:
            verdict = verifier.verify_terminal(grounded, prediction.answer)
        except Exception as exc:  # native engine failure is data, not a crash
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

    def verifier_id(self, verifier: Any) -> str:
        return "%s.%s" % (type(verifier).__module__, type(verifier).__qualname__)

    # -- reference/self-check helpers (used by tests and diagnostics) ---------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """Best-effort extraction of the oracle's own reference submission.

        Used only by the harness self-check/fixtures: the value comes from the
        mounted private oracle, never from the harness itself.
        """
        for key in ("reference_submission", "reference_answer", "expected_answer"):
            if key in oracle_payload:
                return oracle_payload[key]
        raise LaneNotSupported(
            "oracle payload for lane %s exposes no reference submission"
            % self.lane_id)


def _json_safe(obj: Any, *, _depth: int = 0) -> Any:
    if _depth > 8:
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v, _depth=_depth + 1)
                for k, v in list(obj.items())[:64]}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_json_safe(v, _depth=_depth + 1) for v in list(obj)[:64]]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# L05: RDKit substructure lane (fully wired; smoke-tested)
# ---------------------------------------------------------------------------

class L05Adapter(LaneAdapter):
    lane_id = "L05"

    DEFAULT_INDEX_GLOB = ("external_knowledge", "indexes", "rdkit-molecules")

    def _default_index_dir(self) -> Path | None:
        base = REPO_ROOT.joinpath(*self.DEFAULT_INDEX_GLOB)
        if not base.is_dir():
            return None
        candidates = sorted(p for p in base.iterdir() if p.is_dir())
        return candidates[-1] if candidates else None

    def build_verifier(self) -> Any:
        index = (self.lane_kw.get("index_dir")
                 or os.environ.get("AOB_L05_INDEX")
                 or self._default_index_dir())
        if index is None:
            raise LaneNotSupported(
                "L05 requires the pinned RDKit index dir; pass "
                "--lane-kw L05.index_dir=<dir> or set AOB_L05_INDEX")
        try:
            from benchmark.plugins.verifier.l05.verifier import RDKitNativeVerifier
        except ImportError as exc:
            raise LaneNotSupported("L05 verifier not importable: %s" % exc) from exc
        try:
            return RDKitNativeVerifier(Path(index))
        except Exception as exc:
            raise LaneNotSupported(
                "L05 verifier construction failed (index=%s): %s: %s"
                % (index, type(exc).__name__, exc)) from exc

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = oracle_payload.get("reference_submission")
        if ref is None:
            raise LaneNotSupported("L05 oracle payload has no reference_submission")
        return ref

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """The lane verifier's own seeded wrong submission (negative control)."""
        ast = self.extract_ast(oracle_payload)
        return self.verifier.wrong_submission(ast)


# ---------------------------------------------------------------------------
# Reflection adapter for lanes whose oracle payloads carry a task AST
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReflectionSpec:
    module: str
    class_name: str
    #: constructor kwargs resolvable via --lane-kw <LANE>.<key>=<value>
    ctor_kwargs: tuple[str, ...] = ()
    #: env vars consulted for the first ctor kwarg, in order
    env: tuple[str, ...] = ()
    #: short doc of what the lane needs to be live
    needs: str = ""


class ReflectionAdapter(LaneAdapter):
    """Generic adapter: import + construct the lane's own verifier class and
    dispatch ``verify_terminal`` on a GroundedTask rebuilt from the oracle
    payload's AST. Any failure becomes an explicit NOT_SUPPORTED verdict."""

    def __init__(self, lane_id: str, spec: ReflectionSpec,
                 lane_kw: dict[str, str] | None = None) -> None:
        super().__init__(lane_kw)
        self.lane_id = lane_id
        self.spec = spec

    def build_verifier(self) -> Any:
        if not self.spec.class_name:
            raise LaneNotSupported(
                "lane %s: %s" % (self.lane_id, self.spec.needs))
        try:
            module = importlib.import_module(self.spec.module)
            cls = getattr(module, self.spec.class_name)
        except (ImportError, AttributeError) as exc:
            raise LaneNotSupported(
                "lane %s verifier %s.%s not importable: %s"
                % (self.lane_id, self.spec.module, self.spec.class_name, exc)) from exc
        kwargs: dict[str, Any] = {}
        for i, key in enumerate(self.spec.ctor_kwargs):
            value = self.lane_kw.get(key)
            if value is None and i == 0:
                for env in self.spec.env:
                    if os.environ.get(env):
                        value = os.environ[env]
                        break
            if value is not None:
                kwargs[key] = value
        missing = [k for k in self.spec.ctor_kwargs[:1] if k not in kwargs]
        if missing and self.spec.ctor_kwargs:
            raise LaneNotSupported(
                "lane %s requires %s (pass --lane-kw %s.%s=<value>%s); needs: %s"
                % (self.lane_id, self.spec.ctor_kwargs[0], self.lane_id,
                   self.spec.ctor_kwargs[0],
                   (" or env " + "/".join(self.spec.env)) if self.spec.env else "",
                   self.spec.needs))
        try:
            return cls(**kwargs)
        except Exception as exc:
            raise LaneNotSupported(
                "lane %s verifier construction failed: %s: %s (needs: %s)"
                % (self.lane_id, type(exc).__name__, exc, self.spec.needs)) from exc


#: Registry of lanes wired for GroundedTask-rebuild dispatch. Lanes not in
#: this registry evaluate to NOT_SUPPORTED("no harness adapter") — explicit,
#: never faked. Reflection entries are best-effort: they go live only when
#: their declared resources are provided.
REFLECTION_SPECS: dict[str, ReflectionSpec] = {
    "L06": ReflectionSpec(
        module="benchmark.plugins.verifier.l06.verifier",
        class_name="L06NativeVerifier",
        ctor_kwargs=("index_dir",),
        env=("AOB_L06_INDEX",),
        needs="pinned GO/rdflib index dir; oracle payload AST",
    ),
    "L10": ReflectionSpec(
        module="benchmark.plugins.verifier.l10.bddl_verifier",
        class_name="BddlNativeVerifier",
        ctor_kwargs=("oracle_dir",),
        env=("AOB_L10_ORACLE",),
        needs="BDDL oracle dir; oracle payload AST",
    ),
    "L12": ReflectionSpec(
        module="benchmark.plugins.verifier.l12.pm4py_verifier",
        class_name="Pm4pyNativeVerifier",
        ctor_kwargs=(),
        env=(),
        needs="pm4py runtime; oracle payload AST",
    ),
}


class _BrokenAdapter(LaneAdapter):
    """A dynamic adapter module exists but is unusable (bad import, bad
    constructor, or not a ``LaneAdapter``). Construction reports the precise
    reason as NOT_SUPPORTED — a broken adapter must never crash the harness
    probe for the other lanes, and is never silently scored."""

    def __init__(self, lane_id: str, reason: str,
                 lane_kw: dict[str, str] | None = None) -> None:
        super().__init__(lane_kw)
        self.lane_id = lane_id
        self._reason = reason

    def build_verifier(self) -> Any:
        raise LaneNotSupported(self._reason)


def _dynamic_adapter(lane_id: str, lane_kw: dict[str, str] | None
                     ) -> LaneAdapter | None:
    """Try ``benchmark.eval_20k.adapters.<lane_lower>.Adapter``.

    Returns ``None`` — so the caller falls back to the built-in L05 /
    reflection registry — when the module or its ``Adapter`` attribute is
    genuinely absent. A module that *exists* but fails to import (its own
    missing dependency), whose ``Adapter`` is not a ``LaneAdapter``, or whose
    constructor rejects ``lane_kw`` yields a :class:`_BrokenAdapter` carrying
    the exact reason as NOT_SUPPORTED.
    """
    module_name = "%s.adapters.%s" % (__package__, lane_id.lower())
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        # absent only if the adapter module itself (or its parent package,
        # e.g. adapters/ not created yet) is what is missing
        if module_name == missing or module_name.startswith(missing + "."):
            return None
        return _BrokenAdapter(
            lane_id,
            "adapter module %s failed to import (missing dependency %r): %s"
            % (module_name, missing, exc), lane_kw)
    except ImportError as exc:
        return _BrokenAdapter(
            lane_id, "adapter module %s failed to import: %s"
            % (module_name, exc), lane_kw)
    cls = getattr(module, "Adapter", None)
    if cls is None:
        return None
    if not (isinstance(cls, type) and issubclass(cls, LaneAdapter)):
        return _BrokenAdapter(
            lane_id, "adapter %s.Adapter is not a LaneAdapter subclass"
            % module_name, lane_kw)
    try:
        return cls(lane_kw)
    except Exception as exc:
        return _BrokenAdapter(
            lane_id, "adapter %s.Adapter construction failed: %s: %s"
            % (module_name, type(exc).__name__, exc), lane_kw)


def get_adapter(lane_id: str, lane_kw: dict[str, str] | None = None) -> LaneAdapter:
    """Return the adapter for a lane (never None; unsupported lanes get a
    reflection adapter whose construction will report the precise reason).

    Discovery order: (1) ``benchmark.eval_20k.adapters.<lane_lower>.Adapter``
    instantiated with ``lane_kw``; (2) the built-in L05 adapter; (3) the
    reflection registry; (4) an explicit NOT_SUPPORTED placeholder.
    """
    lane_id = lane_id.upper()
    dynamic = _dynamic_adapter(lane_id, lane_kw)
    if dynamic is not None:
        return dynamic
    if lane_id == "L05":
        return L05Adapter(lane_kw)
    spec = REFLECTION_SPECS.get(lane_id)
    if spec is not None:
        return ReflectionAdapter(lane_id, spec, lane_kw)
    return ReflectionAdapter(
        lane_id,
        ReflectionSpec(
            module="benchmark.plugins.verifier.%s" % lane_id.lower(),
            class_name="",
            needs="no harness adapter registered for this lane",
        ),
        lane_kw,
    )


def probe_lanes(lane_kw: dict[str, dict[str, str]] | None = None
                ) -> dict[str, dict[str, str]]:
    """Attempt verifier construction for every lane; report support status."""
    report: dict[str, dict[str, str]] = {}
    for lane_id in ALL_LANES:
        adapter = get_adapter(lane_id, (lane_kw or {}).get(lane_id))
        try:
            verifier = adapter.verifier
            report[lane_id] = {
                "status": "SUPPORTED",
                "verifier": adapter.verifier_id(verifier),
                "reason": "",
            }
        except LaneNotSupported as exc:
            report[lane_id] = {
                "status": "NOT_SUPPORTED",
                "verifier": "",
                "reason": str(exc),
            }
    return report
