"""20K harness adapter for lane L04 (RDFLib / SPARQL + pinned Rhea/ChEBI).

The L04 oracle payload is lane-specific (``family``, ``params``,
``expected_answer``, ``sparql_reference``, ``witness``) and carries no top-level
``ast``.  The native ``L04NativeVerifier.verify_terminal`` reads only
``task.ast.ast`` fields ``params`` / ``answer`` / ``seed`` (via
``_family_params``) and the row's ``family_id``; the exact-set comparison is
against the oracle's ``expected_answer``.  The AST is therefore rebuilt directly
from the public row + mounted oracle payload, exactly as
``benchmark.plugins.verifier.l04.reverify_v2`` consumes the oracle.
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


class Adapter(LaneAdapter):
    lane_id = "L04"

    # -- construction --------------------------------------------------------
    def build_verifier(self) -> Any:
        try:
            from benchmark.plugins.verifier.l04.plugin import (
                DEFAULT_PIN_ROOT,
                L04NativeVerifier,
            )
        except ImportError as exc:
            raise LaneNotSupported(
                "L04 verifier not importable (needs rdflib): %s" % exc) from exc
        pin = (self.lane_kw.get("pin_root")
               or self.lane_kw.get("pin-root")
               or os.environ.get("AOB_L04_PIN")
               or DEFAULT_PIN_ROOT)
        pin_root = Path(pin)
        if not pin_root.is_dir():
            raise LaneNotSupported(
                "L04 requires the pinned Rhea/ChEBI graph root; missing at %s "
                "(pass --lane-kw L04.pin_root=<dir> or set AOB_L04_PIN)"
                % pin_root)
        if not (pin_root / "fetch_manifest.json").is_file():
            raise LaneNotSupported(
                "L04 pinned graph root %s has no fetch_manifest.json "
                "(incomplete snapshot)" % pin_root)
        try:
            import rdflib  # noqa: F401 - the lane's native engine dependency
        except ImportError as exc:
            raise LaneNotSupported(
                "L04 requires rdflib: %s" % exc) from exc
        try:
            return L04NativeVerifier(pin_root)
        except Exception as exc:  # noqa: BLE001
            raise LaneNotSupported(
                "L04 verifier construction failed (pin_root=%s): %s: %s"
                % (pin_root, type(exc).__name__, exc)) from exc

    # -- AST rebuild ---------------------------------------------------------
    def build_ast(self, task: Any, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        params = oracle_payload.get("params")
        expected = oracle_payload.get("expected_answer")
        if not isinstance(params, dict):
            raise LaneNotSupported(
                "L04 oracle payload for %s carries no params"
                % task.instance_id)
        if not isinstance(expected, (list, tuple)):
            raise LaneNotSupported(
                "L04 oracle payload for %s carries no expected_answer"
                % task.instance_id)
        try:
            seed = int(task.raw.get("seed") or 0)
        except (TypeError, ValueError):
            seed = 0
        return {
            "params": dict(params),
            "answer": list(expected),
            "seed": seed,
            "reference_answer": list(expected),
        }

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
        expected = oracle_payload.get("expected_answer")
        if expected is None:
            raise LaneNotSupported(
                "L04 oracle payload for %s exposes no expected_answer"
                % task.instance_id)
        return {"answer": list(expected)}

    def reference_answer(self, oracle_payload: dict[str, Any],
                         task: Any = None) -> Any:
        expected = oracle_payload.get("expected_answer")
        if expected is None:
            raise LaneNotSupported("L04 oracle payload exposes no expected_answer")
        return {"answer": list(expected)}