"""L09 eval adapter: SMART-ANC structural verification lane.

The lane's native ``SmartAncStructuralVerifier`` (families f01..f06) and
``SmartAncExpansionStructuralVerifier`` (families f07..f14, the 20K
expansion) recompute every verdict in pure Python from the pinned SMART-ANC
snapshot. This adapter only resolves the private oracle payload and rebuilds
the ``GroundedTask``/``ast`` those verifiers expect -- no lane semantics are
re-implemented here.

L09 oracle payloads deliberately carry **no task ``ast``**. They carry what the
native verifier reads at scoring time::

    {"mechanism": ..., "artifact_format": ..., "artifact_path": ...,
     "slots": [...], "guards": [...], "reference_artifact_sha256": ...}

The adapter rebuilds a minimal AST from those fields and points
``GroundedTask.oracle_ref`` at the *absolute* oracle file so the native
verifier's own ``_oracle()`` loader reads exactly the payload mounted here.
Lexicographic ``artifact_path`` values are resolved against the pinned
``external_knowledge/pinned/smart-anc/commit-301a6b8b9043`` snapshot by the
lane engine.

Oracle layout: ``bank/staging_20k/l09/oracle/l09x-oracle-*.json`` (600 20K
tasks) and ``bank/staging/l09/oracle/l09-oracle-*.json`` (200 legacy tasks).
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import (
    LaneAdapter,
    LaneNotSupported,
    Verdict,
    build_grounded,
)
from benchmark.eval_20k.oracle import OracleNotFound
from benchmark.eval_20k.tasks import TaskRecord
from benchmark.eval_20k.submission import Prediction

#: Mechanisms handled by the legacy native verifier.
LEGACY_MECHANISMS: tuple[str, ...] = (
    "check_condition_identifiers",
    "check_cql_includes",
    "check_cql_valueset_refs",
    "check_valueset_expansion",
    "check_plan_references",
)


class _L09Dispatch:
    """Dispatch to the lane's own legacy / 20K-expansion structural verifiers."""

    def __init__(self, legacy: Any, expansion: Any) -> None:
        self.legacy = legacy
        self.expansion = expansion

    def verify_terminal(self, task: Any, submitted: Any) -> Any:
        mechanism = str((task.ast.ast or {}).get("mechanism") or "")
        verifier = (self.legacy if mechanism in LEGACY_MECHANISMS
                    else self.expansion)
        return verifier.verify_terminal(task, submitted)


def _find_in_root(root: Path, keys: tuple[str, ...]) -> Path | None:
    """Find a L09 oracle JSON below one mounted root (flat or shard layout)."""
    bases: list[Path] = []
    lane_dir = root / "l09"
    if lane_dir.is_dir():
        bases.append(lane_dir)
    if root.is_dir():
        bases.append(root)
    for base in bases:
        oracle_dir = base / "oracle"
        if oracle_dir.is_dir():
            for key in keys:
                candidate = oracle_dir / ("%s.json" % key)
                if candidate.is_file():
                    return candidate
            for sub in sorted(oracle_dir.iterdir()):
                if not sub.is_dir():
                    continue
                for key in keys:
                    candidate = sub / ("%s.json" % key)
                    if candidate.is_file():
                        return candidate
        for key in keys:
            candidate = base / ("%s.json" % key)
            if candidate.is_file():
                return candidate
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            nested = sub / "oracle"
            if not nested.is_dir():
                continue
            for key in keys:
                candidate = nested / ("%s.json" % key)
                if candidate.is_file():
                    return candidate
    return None


def _locate_l09(oracle_store: Any, task: TaskRecord) -> Path:
    try:
        return Path(oracle_store.locate(
            task.lane_id, task.instance_id, task.oracle_handle))
    except OracleNotFound:
        pass
    keys = tuple(dict.fromkeys(
        k for k in (task.instance_id, task.oracle_handle) if k))
    for root in oracle_store.roots:
        found = _find_in_root(Path(root), keys)
        if found is not None:
            return found
    raise OracleNotFound(
        "no L09 oracle payload for instance=%s handle=%s in roots: %s"
        % (task.instance_id, task.oracle_handle,
           ", ".join(str(r) for r in oracle_store.roots))
    )


def _artifact_submission(artifact_format: str | None, artifact: Any) -> dict[str, Any]:
    if artifact_format == "cql-text":
        return {"artifact": {"format": "cql-text", "text": artifact}}
    return {"artifact": {"content": artifact}}


def _oracle_ref(path: Path) -> str:
    """Oracle ref that the native ``_oracle`` loader resolves back to ``path``.

    Both native verifiers load ``ORACLE_DIR / ("%s.json" % oracle_ref)`` and an
    absolute ``oracle_ref`` makes pathlib discard ``ORACLE_DIR``. Passing the
    absolute path *without* its ``.json`` suffix therefore makes the native
    loader read the exact mounted payload, wherever the oracle root lives.
    """
    return str(path.with_suffix("")) if path.suffix == ".json" else str(path)


class Adapter(LaneAdapter):
    lane_id = "L09"

    # -- construction -------------------------------------------------------

    def build_verifier(self) -> Any:
        try:
            from benchmark.plugins.task_family.l09 import engine as legacy_engine
            from benchmark.plugins.verifier.l09.structural_verifier import (
                SmartAncStructuralVerifier,
            )
            from benchmark.plugins.verifier.l09x.structural_verifier import (
                SmartAncExpansionStructuralVerifier,
            )
        except ImportError as exc:
            raise LaneNotSupported("L09 verifier not importable: %s" % exc) from exc
        if not legacy_engine.ANC_ROOT.is_dir():
            raise LaneNotSupported(
                "L09 needs the pinned SMART-ANC snapshot at %s"
                % legacy_engine.ANC_ROOT)
        try:
            return _L09Dispatch(
                SmartAncStructuralVerifier(),
                SmartAncExpansionStructuralVerifier(),
            )
        except Exception as exc:
            raise LaneNotSupported(
                "L09 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc

    def verifier_id(self, verifier: Any) -> str:
        if isinstance(verifier, _L09Dispatch):
            return "%s.%s|%s.%s" % (
                type(verifier.legacy).__module__,
                type(verifier.legacy).__qualname__,
                type(verifier.expansion).__module__,
                type(verifier.expansion).__qualname__,
            )
        return super().verifier_id(verifier)

    # -- oracle / AST rebuild ----------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        mechanism = oracle_payload.get("mechanism")
        if not isinstance(mechanism, str) or not mechanism:
            raise LaneNotSupported(
                "L09 oracle payload carries no mechanism (cannot rebuild the "
                "structural verifier's GroundedTask)")
        ast: dict[str, Any] = {"mechanism": mechanism}
        for key in ("artifact_format", "artifact_path",
                    "reference_artifact_sha256", "semantic_profile"):
            if key in oracle_payload:
                ast[key] = oracle_payload[key]
        slots = oracle_payload.get("slots")
        if isinstance(slots, list):
            ast["slots"] = slots
        guards = oracle_payload.get("guards")
        if isinstance(guards, list):
            ast["guards"] = guards
        return ast

    def _locate(self, oracle_store: Any, task: TaskRecord) -> Path:
        return _locate_l09(oracle_store, task)

    # -- scoring ------------------------------------------------------------

    def evaluate(self, task: TaskRecord, prediction: Prediction,
                 oracle_store: Any) -> Verdict:
        try:
            verifier = self.verifier
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        try:
            oracle_path = self._locate(oracle_store, task)
        except OracleNotFound as exc:
            return Verdict(status="ORACLE_MISSING", valid=None, reason=str(exc))
        try:
            payload = json.loads(oracle_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return Verdict(status="ERROR", valid=None,
                           reason="oracle payload unreadable: %s" % exc,
                           oracle_path=str(oracle_path))
        try:
            ast = self.extract_ast(payload)
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        grounded = replace(build_grounded(task.raw, ast),
                           oracle_ref=_oracle_ref(Path(oracle_path).resolve()))
        try:
            verdict = verifier.verify_terminal(grounded, prediction.answer)
        except Exception as exc:
            return Verdict(status="ERROR", valid=None,
                           reason="verify_terminal raised %s: %s"
                                  % (type(exc).__name__, exc),
                           verifier_id=self.verifier_id(verifier),
                           oracle_path=str(oracle_path))
        valid = bool(getattr(verdict, "valid", False))
        detail = getattr(verdict, "detail", {}) or {}
        if not isinstance(detail, dict):
            detail = {"detail": str(detail)}
        if detail.get("reason") in self.ORACLE_MISSING_REASONS:
            return Verdict(status="ORACLE_MISSING", valid=None,
                           reason="lane verifier reported its own oracle "
                                  "material unreachable: %s" % detail.get("reason"),
                           verifier_id=self.verifier_id(verifier),
                           oracle_path=str(oracle_path))
        return Verdict(status="SCORED", valid=valid, detail=detail,
                       verifier_id=self.verifier_id(verifier),
                       oracle_path=str(oracle_path))

    # -- reference / negative helpers --------------------------------------

    def _reference_artifact(self, ast: dict[str, Any]) -> Any:
        from benchmark.plugins.task_family.l09 import engine

        path = ast["artifact_path"]
        if ast.get("artifact_format") == "cql-text":
            return engine.ANC_ROOT.joinpath(path).read_text(encoding="utf-8")
        return engine.load_artifact(path)

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ast = self.extract_ast(oracle_payload)
        if "artifact_path" not in ast:
            raise LaneNotSupported(
                "L09 oracle payload has no artifact_path (cannot rebuild the "
                "pristine reference artifact)")
        reference = self._reference_artifact(ast)
        return _artifact_submission(ast.get("artifact_format"), reference)

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """Reconstruct the build's corrupted artifact from the oracle slots.

        Mirrors the audit's independent tamper reconstruction: replace each
        corrupted slot with its recorded ``wrong_value``. This is the same
        construction Gate C uses as a negative witness.
        """
        from benchmark.plugins.task_family.l09 import engine

        ast = self.extract_ast(oracle_payload)
        reference = self._reference_artifact(ast)
        slots = oracle_payload.get("slots") or []
        artifact_format = ast.get("artifact_format")
        if artifact_format == "cql-text":
            corrupted = reference
            for slot in slots:
                find = slot.get("find")
                if find and "wrong_value" in slot:
                    corrupted = corrupted.replace(find, slot["wrong_value"], 1)
        else:
            corrupted = copy.deepcopy(reference)
            for slot in slots:
                path = slot.get("path")
                if path and "wrong_value" in slot:
                    engine.set_path(corrupted, path, slot["wrong_value"])
        return _artifact_submission(artifact_format, corrupted)