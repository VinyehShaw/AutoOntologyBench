"""20K harness adapter for lane L03 (static Camunda DMN; legacy + 20K).

L03 is the one lane whose native verifier reads the private oracle itself: both
``L03NativeVerifier.verify_terminal`` and ``L03NativeVerifier20K.verify_terminal``
load ``<oracle_ref>.json`` and rebuild the DMN AST from that payload with the
lane's own ``_ast_from_oracle`` helpers.  The oracle payload therefore carries
every field the verifier needs; the public row contributes only identity.

This adapter wraps the two verifier generations in one object (legacy 8K goal
kinds and 20K expansion goal kinds are disjoint) and injects the oracle payload
loaded by the harness by ``oracle_ref``.  That keeps the adapter correct even
when the oracle is mounted through the harness ``--oracle`` root rather than the
in-repo staging tree.
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

#: Submission key per goal kind (copied from the lane audit's SUBMIT_KEY).
SUBMIT_KEY: dict[str, str] = {
    "exact_output": "output",
    "rule_audit": "matched_rule_ids",
    "rule_repair": "table",
    "hit_policy_repair": "table",
    "chain_eval": "output",
    "binding_repair": "tables",
    "collect_result": "output",
    "input_synthesis": "input",
    "tck_output": "output",
    "suite_audit": "inconsistent_case_id",
    "output_repair": "table",
    "aggregation_repair": "table",
    "rule_deletion_repair": "table",
    "chain_synthesis": "input",
    "ordered_output": "output",
}


def _default_oracle_dir() -> Path:
    for rel in (("bank", "staging_20k", "l03", "oracle"),
                ("bank", "staging", "l03", "oracle")):
        candidate = REPO_ROOT.joinpath(*rel)
        if candidate.is_dir():
            return candidate
    return REPO_ROOT


def _make_verifier_class() -> Any:
    from benchmark.plugins.verifier.l03.plugin import L03NativeVerifier
    from benchmark.plugins.verifier.l03.plugin_20k import (
        L03NativeVerifier20K,
        _TERMINAL_20K,
    )

    class L03Verifier(L03NativeVerifier20K):
        """One facade over the legacy and 20K L03 native verifiers."""

        def __init__(self, oracle_dir: str | Path) -> None:
            # Do not call L03NativeVerifier.__init__: it would chmod/mkdir the
            # (read-only) oracle directory.  Payloads are injected by the
            # adapter before verify_terminal runs.
            self.oracle_dir = Path(oracle_dir)
            self._payloads: dict[str, dict[str, Any]] = {}

        def load_oracle(self, oracle_ref: str) -> dict[str, Any]:
            payload = self._payloads.get(oracle_ref)
            if payload is not None:
                return payload
            return super().load_oracle(oracle_ref)

        def verify_terminal(self, task: Any, submitted_state_or_answer: Any) -> Any:
            oracle = self.load_oracle(task.oracle_ref)
            if oracle.get("goal_kind") in _TERMINAL_20K:
                return super().verify_terminal(task, submitted_state_or_answer)
            return L03NativeVerifier.verify_terminal(
                self, task, submitted_state_or_answer)

    return L03Verifier


class Adapter(LaneAdapter):
    lane_id = "L03"

    # -- construction --------------------------------------------------------
    def build_verifier(self) -> Any:
        pinned = REPO_ROOT / "external_knowledge" / "pinned"
        if not (pinned / "camunda-dmn-examples").is_dir():
            raise LaneNotSupported(
                "L03 requires the pinned Camunda DMN source tree at %s"
                % (pinned / "camunda-dmn-examples"))
        if not (pinned / "dmn-tck").is_dir():
            raise LaneNotSupported(
                "L03 requires the pinned DMN TCK source tree at %s"
                % (pinned / "dmn-tck"))
        oracle_dir = (self.lane_kw.get("oracle_dir")
                      or self.lane_kw.get("oracle")
                      or os.environ.get("AOB_L03_ORACLE"))
        directory = Path(oracle_dir) if oracle_dir else _default_oracle_dir()
        try:
            cls = _make_verifier_class()
            return cls(directory)
        except ImportError as exc:
            raise LaneNotSupported(
                "L03 verifier not importable: %s" % exc) from exc
        except Exception as exc:  # noqa: BLE001
            raise LaneNotSupported(
                "L03 verifier construction failed (oracle_dir=%s): %s: %s"
                % (directory, type(exc).__name__, exc)) from exc

    # -- AST rebuild ---------------------------------------------------------
    def build_ast(self, task: Any, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        try:
            from benchmark.plugins.verifier.l03.plugin import (
                _TERMINAL,
                _ast_from_oracle,
            )
            from benchmark.plugins.verifier.l03.plugin_20k import (
                _TERMINAL_20K,
                _ast_from_oracle_20k,
            )
        except ImportError as exc:
            raise LaneNotSupported(
                "L03 verifier modules not importable: %s" % exc) from exc

        oracle_ref = oracle_payload.get("oracle_ref")
        if oracle_ref:
            self.verifier._payloads[oracle_ref] = oracle_payload
        self.verifier._payloads[task.oracle_handle] = oracle_payload
        kind = oracle_payload.get("goal_kind")
        if kind in _TERMINAL_20K:
            return _ast_from_oracle_20k(oracle_payload)
        if kind in _TERMINAL:
            return _ast_from_oracle(oracle_payload)
        raise LaneNotSupported(
            "L03 oracle payload for %s has unsupported goal_kind %r"
            % (task.instance_id, kind))

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
        reference = oracle_payload["reference"]
        if kind == "partial_input_audit":
            return dict(reference)
        key = SUBMIT_KEY.get(kind)
        if key is None:
            raise LaneNotSupported("L03 unknown goal_kind %r" % kind)
        return {key: reference}

    def reference_answer(self, oracle_payload: dict[str, Any],
                         task: Any = None) -> Any:
        kind = oracle_payload.get("goal_kind")
        reference = oracle_payload.get("reference")
        if reference is None:
            raise LaneNotSupported(
                "L03 oracle payload for goal_kind %r exposes no reference" % kind)
        if kind == "partial_input_audit":
            return dict(reference)
        key = SUBMIT_KEY.get(kind)
        if key is None:
            raise LaneNotSupported("L03 unknown goal_kind %r" % kind)
        return {key: reference}