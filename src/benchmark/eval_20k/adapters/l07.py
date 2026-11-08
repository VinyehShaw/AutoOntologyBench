"""L07 eval adapter: SBML simulation / structural lane.

The lane's own ``SbmlNativeVerifier20K`` superset verifier recomputes truth from
the pinned SBML bytes with libSBML + pinned-CVODE RoadRunner, so this adapter
only wires resources and answer extraction -- no lane semantics are
re-implemented here.

Resources (first match wins):

* ``--lane-kw L07.index_dir=<dir>``
* ``AOB_L07_INDEX``
* ``external_knowledge/indexes/biomodels/<release>/`` (the pinned index that
  carries ``models.jsonl`` + ``manifest.json``)

Oracle layout: ``bank/staging_20k/l07/oracle/<handle>.json`` (550 20K tasks)
and ``bank/staging/l07/oracle/<handle>.json`` (400 legacy 8K tasks); every
payload carries ``ast``, ``reference_submission`` and the solver evidence.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import (
    LaneAdapter,
    LaneNotSupported,
    Verdict,
    build_grounded,
)
from benchmark.eval_20k.oracle import OracleNotFound

from .l07_safe import (
    AnswerShapeError,
    SHAPE_EXCEPTIONS,
    normalize_answer,
    verify_isolated,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

#: Pinned BioModels index root (release id is part of the directory name).
DEFAULT_INDEX_ROOT = (
    REPO_ROOT / "external_knowledge" / "indexes" / "biomodels" / "api-20261028"
)

#: libpython locations tried when the interpreter's own library is not on the
#: default loader path (the RoadRunner wheel links against it as a shared lib).
_LIBPYTHON_CANDIDATES = (
    "/opt/conda/lib/libpython3.11.so.1.0",
    "/usr/lib/x86_64-linux-gnu/libpython3.11.so.1.0",
    "/usr/local/lib/libpython3.11.so.1.0",
)


def _default_index_dir() -> Path | None:
    """Newest ``biomodels/<release>`` directory carrying a models index."""
    base = DEFAULT_INDEX_ROOT.parent
    if DEFAULT_INDEX_ROOT.is_dir() and (DEFAULT_INDEX_ROOT / "models.jsonl").is_file():
        return DEFAULT_INDEX_ROOT
    if base.is_dir():
        for candidate in sorted((p for p in base.iterdir() if p.is_dir()),
                                reverse=True):
            if (candidate / "models.jsonl").is_file():
                return candidate
    return None


def _preload_libpython() -> None:
    """Make libpython visible to the dynamically linked RoadRunner extension.

    ``benchmark.plugins.source.l07.simulate.load_roadrunner`` already does the
    same for its normal path; doing it here keeps ``support()`` honest even
    when the verifier is imported through this adapter first.
    """
    try:
        import roadrunner  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in _LIBPYTHON_CANDIDATES:
        if not os.path.isfile(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=getattr(ctypes, "RTLD_GLOBAL", 0))
        except OSError:
            continue
        try:
            import roadrunner  # noqa: F401
            return
        except ImportError:
            continue


class Adapter(LaneAdapter):
    lane_id = "L07"

    # -- construction -------------------------------------------------------

    def build_verifier(self) -> Any:
        index = (
            self.lane_kw.get("index_dir")
            or os.environ.get("AOB_L07_INDEX")
            or _default_index_dir()
        )
        if index is None:
            raise LaneNotSupported(
                "L07 requires the pinned BioModels index dir; pass "
                "--lane-kw L07.index_dir=<dir> or set AOB_L07_INDEX"
            )
        index_path = Path(index)
        if not (index_path / "models.jsonl").is_file():
            raise LaneNotSupported(
                "L07 index dir %s is missing models.jsonl (needs the pinned "
                "BioModels index: external_knowledge/indexes/biomodels/<release>)"
                % index_path
            )
        _preload_libpython()
        try:
            from benchmark.plugins.source.l07.simulate import load_roadrunner
        except ImportError as exc:  # pragma: no cover - import guard
            raise LaneNotSupported("L07 simulation runtime not importable: %s" % exc) from exc
        try:
            load_roadrunner()
        except Exception as exc:
            raise LaneNotSupported(
                "L07 needs libSBML + libRoadRunner (pinned CVODE): %s: %s"
                % (type(exc).__name__, exc)
            ) from exc
        try:
            from benchmark.plugins.verifier.l07.verifier_20k import (
                SbmlNativeVerifier20K,
            )
        except ImportError as exc:
            raise LaneNotSupported("L07 verifier not importable: %s" % exc) from exc
        try:
            return SbmlNativeVerifier20K(index_path)
        except Exception as exc:
            raise LaneNotSupported(
                "L07 verifier construction failed (index=%s): %s: %s"
                % (index_path, type(exc).__name__, exc)
            ) from exc

    # -- scoring -----------------------------------------------------------

    def _timeout_s(self) -> float:
        raw = self.lane_kw.get("timeout") or os.environ.get("AOB_L07_TIMEOUT")
        try:
            return float(raw) if raw else 600.0
        except (TypeError, ValueError):
            return 600.0

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
            ast = self.extract_ast(payload)
        except LaneNotSupported as exc:
            return Verdict(status="NOT_SUPPORTED", valid=None, reason=str(exc))
        grounded = build_grounded(task.raw, ast)
        verifier_id = self.verifier_id(verifier)
        try:
            answer, note = normalize_answer(prediction.answer, payload)
        except AnswerShapeError as exc:
            return Verdict(
                status="SCORED", valid=False,
                detail={"error": str(exc), "answer_shape_invalid": True},
                reason="invalid answer shape: %s" % exc,
                verifier_id=verifier_id, oracle_path=oracle_path)
        envelope = verify_isolated(
            verifier, grounded, answer, timeout=self._timeout_s())
        detail: dict[str, Any] = {}
        if envelope.get("ok"):
            raw_detail = envelope.get("detail") or {}
            detail = raw_detail if isinstance(raw_detail, dict) else {
                "detail": str(raw_detail)}
            if note is not None:
                detail["answer_normalized"] = note
            return Verdict(
                status="SCORED", valid=bool(envelope.get("valid")),
                detail=detail, verifier_id=verifier_id,
                oracle_path=oracle_path)
        reason = envelope.get("reason") or (
            "verify_terminal raised %s: %s"
            % (envelope.get("error_type", "Exception"),
               envelope.get("error", "unknown error")))
        if note is not None:
            detail["answer_normalized"] = note
        if envelope.get("error_type") in SHAPE_EXCEPTIONS:
            detail["error"] = reason
            return Verdict(
                status="SCORED", valid=False, detail=detail,
                reason="answer shape rejected: %s" % reason,
                verifier_id=verifier_id, oracle_path=oracle_path)
        if envelope.get("crash"):
            detail["native_crash"] = {
                "signal": envelope.get("signal_name") or envelope.get("signal"),
                "evidence": reason,
            }
        return Verdict(
            status="ERROR", valid=None, detail=detail, reason=reason,
            verifier_id=verifier_id, oracle_path=oracle_path)

    # -- reference / negative helpers --------------------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = oracle_payload.get("reference_submission")
        if ref is None:
            raise LaneNotSupported("L07 oracle payload has no reference_submission")
        return ref

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """A deterministic mutation of the oracle reference (negative control).

        Every branch either removes an element, reorders a ranking or shifts a
        repaired value; the native verifier rejects it. The mutation is not
        used for scoring -- only by harness self-checks.
        """
        ref = self.reference_answer(oracle_payload)
        if not isinstance(ref, dict):
            return {}
        if "answer_ids" in ref:
            ids = list(ref["answer_ids"])
            if ids:
                return {"answer_ids": ids[:-1]}
            return {"answer_ids": ["__not_in_universe__"]}
        if "ranking" in ref:
            ranking = list(ref["ranking"])
            if len(ranking) > 1:
                ranking.reverse()
            else:
                ranking = ranking + ["__not_in_universe__"]
            return {"ranking": ranking}
        if "accepted_values" in ref:
            values = list(ref["accepted_values"])
            if values:
                return {"accepted_values": values[:-1]}
            return {"accepted_values": [0.0]}
        if "assignment" in ref:
            assignment = dict(ref["assignment"])
            if assignment:
                key = next(iter(assignment))
                assignment[key] = "__wrong__"
            return {"assignment": assignment}
        for key in ("parameter_value", "initial_value"):
            if key in ref:
                value = ref[key]
                if isinstance(value, (int, float)):
                    return {key: float(value) * 1.5 + 1.0}
                return {key: value}
        if "repaired_formula" in ref:
            formula = ref["repaired_formula"]
            return {"repaired_formula": "0.0" if formula != "0.0" else "1.0"}
        return {}