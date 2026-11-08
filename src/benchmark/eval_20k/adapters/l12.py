"""L12 (workflow-mining / pm4py) adapter for the AutoOntologyBench 20K harness.

The lane has two native verifier generations over the same pinned pm4py
snapshot:

* ``benchmark.plugins.verifier.l12.pm4py_verifier.Pm4pyNativeVerifier`` -- the
  legacy families (dfg / variants / endpoints / nonconformant_trace_set and
  the trace/model repairs);
* ``benchmark.plugins.task_family.l12.expand20k.verifier.Pm4pyVerifier20k`` --
  a subclass that adds the 20K-expansion exact-artifact operations
  ``footprints`` and ``footprints_conf`` and delegates every other goal kind
  to the legacy logic unchanged.

Oracle payloads carry ``ast`` directly (``ast.goal_kind`` / ``ast.operation``
select the code path); the adapter rebuilds the ``GroundedTask`` through the
base class and only dispatches ``verify_terminal``. No lane semantics are
re-implemented here.

The only harness-side adaptation is oracle resolution. Both verifier
generations resolve ``_oracle(task)`` as ``<their ORACLE_DIR> / oracle_ref +
".json"`` (legacy ``bank/staging/l12/oracle``; 20K
``bank/staging_20k/l12/oracle``). A 20K task therefore reports
``oracle_missing`` when scored through the legacy resolution. This adapter
resolves the payload through the harness ``OracleStore`` and points
``oracle_ref`` at the *absolute* mounted file (without the ``.json`` suffix),
exactly like the L09 adapter: pathlib discards ``ORACLE_DIR`` for an absolute
ref, so either verifier generation reads the exact mounted payload.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import (
    LaneAdapter,
    LaneNotSupported,
    Verdict,
    _json_safe,
    build_grounded,
)
from benchmark.eval_20k.oracle import OracleNotFound, OracleStore
from benchmark.eval_20k.submission import Prediction
from benchmark.eval_20k.tasks import TaskRecord


def _oracle_ref(path: Path) -> str:
    """Oracle ref that the native ``_oracle`` loader resolves back to ``path``.

    Both native verifiers load ``ORACLE_DIR / ("%s.json" % oracle_ref)``; an
    absolute ``oracle_ref`` makes pathlib discard ``ORACLE_DIR``. Passing the
    absolute path *without* its ``.json`` suffix therefore makes the native
    loader read the exact mounted payload, wherever the oracle root lives.
    """
    return str(path.with_suffix("")) if path.suffix == ".json" else str(path)


class Adapter(LaneAdapter):
    lane_id = "L12"

    # -- construction --------------------------------------------------------

    def build_verifier(self) -> Any:
        try:
            from benchmark.plugins.task_family.l12.expand20k.verifier import (
                Pm4pyVerifier20k,
            )
        except ImportError as exc:
            raise LaneNotSupported(
                "L12 native verifier not importable "
                "(benchmark.plugins.task_family.l12.expand20k.verifier): %s"
                % exc) from exc
        try:
            return Pm4pyVerifier20k()
        except Exception as exc:
            raise LaneNotSupported(
                "L12 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc

    # -- scoring -------------------------------------------------------------

    def evaluate(self, task: TaskRecord, prediction: Prediction,
                 oracle_store: OracleStore) -> Verdict:
        try:
            verifier = self.verifier
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        try:
            oracle_path = Path(oracle_store.locate(
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
        grounded = replace(build_grounded(task.raw, ast),
                           oracle_ref=_oracle_ref(oracle_path.resolve()))
        try:
            verdict = verifier.verify_terminal(grounded, prediction.answer)
        except (AttributeError, TypeError, KeyError) as exc:
            # A malformed answer shape (e.g. a stringified artifact) is a
            # model failure, not a harness failure: keep it in the TSR
            # denominator instead of emitting an ERROR row.
            return Verdict(
                status="SCORED", valid=False,
                detail={"check": "answer_shape",
                        "error": "%s: %s" % (type(exc).__name__, exc)},
                reason="answer_shape_error: %s: %s"
                       % (type(exc).__name__, exc),
                verifier_id=self.verifier_id(verifier),
                oracle_path=str(oracle_path))
        except Exception as exc:  # native engine failure is data, not a crash
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
        return Verdict(status="SCORED", valid=valid, detail=_json_safe(detail),
                       verifier_id=self.verifier_id(verifier),
                       oracle_path=str(oracle_path))

    # -- reference / negative helpers (self-checks) --------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """The oracle payload's stored reference answer (exact-artifact tasks)."""
        ref = oracle_payload.get("reference_answer")
        if ref is None:
            raise LaneNotSupported(
                "L12 oracle payload exposes no reference_answer")
        return {"answer": ref}


__all__ = ["Adapter"]