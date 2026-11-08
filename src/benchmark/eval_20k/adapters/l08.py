"""L08 eval adapter: CQL/FHIR rule evaluation lane.

The lane's native verifiers rebuild the rule's ELM from the typed rule and run
it through the vendored ``cql-execution`` engine, so this adapter wires the
Node engine and rebuilds the verifier's ``GroundedTask`` from the lane's own
oracle + evidence artifacts. No lane semantics are re-implemented here.

Oracle layout (unlike L05/L07) uses *nested shards*::

    bank/staging_20k/l08/oracle/shard-0000/<instance_id>.json   (250 20K tasks)
    bank/staging_20k/l08/evidence/shard-0000/<instance_id>.json
    bank/staging/l08/oracle/shard-0000/<instance_id>.json       (250 legacy tasks)
    bank/staging/l08/evidence/shard-0000/<instance_id>.json

``OracleStore``'s generic index only looks at ``<lane>/oracle/*.json`` and
``<lane>/<sub>/oracle/*.json``, so it cannot see ``oracle/shard-*/*.json``.
This adapter therefore performs the lane-local lookup: oracle payload for the
reference answer/rule, evidence payload for the synthetic FHIR panel, exactly
as ``benchmark.plugins.verifier.l08.recheck_20k`` does. The 20K verifier
(``L0820kNativeVerifier``) is used for ``...-candidate-dev1`` tasks and the
legacy verifier (``L08NativeVerifier``) for ``...-8000-rc1`` tasks; both are the
lane's own native implementations.
"""
from __future__ import annotations

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

DATASET_8000 = "autoontobench-bank-8000-rc1"


class _L08Dispatch:
    """Small dispatcher over the lane's two native verifier generations."""

    def __init__(self, legacy: Any, expansion: Any) -> None:
        self.legacy = legacy
        self.expansion = expansion

    def verify_terminal(self, task: Any, submitted: Any) -> Any:
        dataset_id = str((task.ast.ast or {}).get("dataset_id") or "")
        verifier = self.legacy if dataset_id == DATASET_8000 else self.expansion
        return verifier.verify_terminal(task, submitted)


def _keys(task: TaskRecord) -> tuple[str, ...]:
    keys = [task.instance_id]
    if task.oracle_handle and task.oracle_handle not in keys:
        keys.append(task.oracle_handle)
    return tuple(keys)


def _find_in_root(root: Path, kind: str, keys: tuple[str, ...]) -> Path | None:
    """Find ``<kind>/.../<key>.json`` below one mounted root."""
    bases: list[Path] = []
    lane_dir = root / "l08"
    if lane_dir.is_dir():
        bases.append(lane_dir)
    if root.is_dir():
        bases.append(root)
    for base in bases:
        kind_dir = base / kind
        if kind_dir.is_dir():
            for key in keys:
                candidate = kind_dir / ("%s.json" % key)
                if candidate.is_file():
                    return candidate
            # shard layout: <kind>/shard-XXXX/<key>.json
            for sub in sorted(kind_dir.iterdir()):
                if not sub.is_dir():
                    continue
                for key in keys:
                    candidate = sub / ("%s.json" % key)
                    if candidate.is_file():
                        return candidate
        # flat lane layout: <lane>/<key>.json
        for key in keys:
            candidate = base / ("%s.json" % key)
            if candidate.is_file():
                return candidate
        # shard layout without a kind dir: <lane>/<shard>/<kind>/<key>.json
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            nested = sub / kind
            if not nested.is_dir():
                continue
            for key in keys:
                candidate = nested / ("%s.json" % key)
                if candidate.is_file():
                    return candidate
    return None


def _locate_l08(oracle_store: Any, task: TaskRecord) -> tuple[Path, Path]:
    """Resolve the oracle payload and the matching evidence job file."""
    keys = _keys(task)
    oracle_path: Path | None = None
    evidence_path: Path | None = None
    for root in oracle_store.roots:
        found_oracle = _find_in_root(Path(root), "oracle", keys)
        if found_oracle is None:
            continue
        found_evidence = _find_in_root(Path(root), "evidence", keys)
        if found_evidence is not None:
            return found_oracle, found_evidence
        oracle_path = oracle_path or found_oracle
    if oracle_path is None:
        # Last resort: the generic store (flat mounted layouts).
        try:
            oracle_path = Path(oracle_store.locate(
                task.lane_id, task.instance_id, task.oracle_handle))
        except OracleNotFound as exc:
            raise OracleNotFound(
                "no L08 oracle payload for instance=%s handle=%s in roots: %s"
                % (task.instance_id, task.oracle_handle,
                   ", ".join(str(r) for r in oracle_store.roots))
            ) from exc
    if evidence_path is None:
        # Oracle and evidence may be mounted under different roots; search
        # every root for the panel before declaring it missing.
        for root in oracle_store.roots:
            evidence_path = _find_in_root(Path(root), "evidence", keys)
            if evidence_path is not None:
                break
    if evidence_path is None:
        raise OracleNotFound(
            "L08 oracle payload found at %s but no evidence/<instance>.json "
            "panel under roots: %s (the verifier needs the synthetic FHIR "
            "panel from the lane's evidence layout)"
            % (oracle_path, ", ".join(str(r) for r in oracle_store.roots))
        )
    return oracle_path, evidence_path


class Adapter(LaneAdapter):
    lane_id = "L08"

    # -- construction -------------------------------------------------------

    def build_verifier(self) -> Any:
        try:
            from benchmark.plugins.verifier.l08.runner import (
                NodeEngineRunner,
                engine_versions,
            )
            from benchmark.plugins.verifier.l08.verifier import L08NativeVerifier
            from benchmark.plugins.verifier.l08.verifier_20k import (
                L0820kNativeVerifier,
            )
        except ImportError as exc:
            raise LaneNotSupported("L08 verifier not importable: %s" % exc) from exc
        try:
            runner = NodeEngineRunner()
            versions = engine_versions()
        except Exception as exc:
            raise LaneNotSupported(
                "L08 needs the pinned Node runtime + cql-execution closure "
                "(tools/node + external_knowledge/pinned/cql-clinical/npm): "
                "%s: %s" % (type(exc).__name__, exc)
            ) from exc
        verifier = _L08Dispatch(
            L08NativeVerifier(runner=runner),
            L0820kNativeVerifier(runner=runner),
        )
        verifier.engine_versions = versions  # provenance for diagnostics
        return verifier

    def verifier_id(self, verifier: Any) -> str:
        if isinstance(verifier, _L08Dispatch):
            return "%s.%s|%s.%s" % (
                type(verifier.expansion).__module__,
                type(verifier.expansion).__qualname__,
                type(verifier.legacy).__module__,
                type(verifier.legacy).__qualname__,
            )
        return super().verifier_id(verifier)

    # -- oracle / scoring ---------------------------------------------------

    def evaluate(self, task: TaskRecord, prediction: Prediction,
                 oracle_store: Any) -> Verdict:
        try:
            verifier = self.verifier
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        try:
            oracle_path, evidence_path = _locate_l08(oracle_store, task)
        except OracleNotFound as exc:
            return Verdict(status="ORACLE_MISSING", valid=None, reason=str(exc))
        try:
            oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return Verdict(status="ERROR", valid=None,
                           reason="oracle/evidence unreadable: %s" % exc,
                           oracle_path=str(oracle_path))
        rule = oracle.get("rule")
        patients = evidence.get("patients")
        if not isinstance(rule, dict) or not isinstance(patients, list) or not patients:
            return Verdict(
                status="NOT_SUPPORTED", valid=None,
                reason="L08 oracle/evidence payload lacks the typed rule or "
                       "synthetic FHIR panel",
                oracle_path=str(oracle_path))
        if evidence.get("rule") is not None and evidence.get("rule") != rule:
            return Verdict(
                status="ERROR", valid=None,
                reason="L08 evidence rule differs from oracle rule "
                       "(cannot rebuild the verifier's GroundedTask)",
                oracle_path=str(oracle_path))
        ast = {
            "rule": rule,
            "patients": patients,
            "dataset_id": task.raw.get("dataset_id", ""),
        }
        grounded = replace(build_grounded(task.raw, ast),
                           oracle_ref=str(Path(oracle_path).resolve()))
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

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = oracle_payload.get("reference_submission")
        if isinstance(ref, dict):
            return ref
        answer = oracle_payload.get("reference_answer")
        if isinstance(answer, list):
            return {"answer_ids": list(answer)}
        raise LaneNotSupported("L08 oracle payload has no reference submission/answer")

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = self.reference_answer(oracle_payload)
        if not isinstance(ref, dict):
            return {}
        if "op" in ref:  # repair regime: corrupt the parameters
            mutated = dict(ref)
            flipped = {"<=": "<", "<": "<=", ">=": ">", ">": ">="}.get(
                mutated.get("op"), mutated.get("op"))
            mutated["op"] = flipped
            if mutated.get("op") == ref.get("op"):
                try:
                    mutated["threshold"] = float(mutated.get("threshold", 0)) + 1.0
                except (TypeError, ValueError):
                    mutated["threshold"] = 1.0
            return mutated
        answer_ids = list(ref.get("answer_ids") or [])
        if answer_ids:
            return {"answer_ids": answer_ids[:-1]}
        return {"answer_ids": ["synthetic-unknown-9999"]}