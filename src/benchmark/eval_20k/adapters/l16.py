"""L16 eval adapter: UniProt Swiss-Prot 2026_03 native truth.

The 20K L16 oracle payloads carry ``family`` / ``params`` /
``expected_answer`` (plus ``witness`` / ``xml_store_reference``) but no
top-level ``ast``.  ``extract_ast`` rebuilds the AST shape produced by the
lane's own ``benchmark.plugins.task_family.l16.plugin.L16TaskFamilies.propose``
from the oracle's recomputable fields; ``L16NativeVerifier.verify_terminal``
consumes ``ast.ast["answer"]`` (the recorded fast-path answer, whose XML-store
recomputation is the lane's own ``reverify`` CLI / audit gate B).

Resource: the pinned UniProt tree
(``external_knowledge/pinned/uniprot/uniprot-2026_03-sprot``) including the
DAT/XML SQLite stores.  Support is gated on that pin being present.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "uniprot"
                    / "uniprot-2026_03-sprot")


def _resolve_pin_root(lane_kw: dict[str, str]) -> Path:
    value = (lane_kw.get("pin_root")
             or os.environ.get("AOB_L16_PIN")
             or str(DEFAULT_PIN_ROOT))
    return Path(value)


class Adapter(LaneAdapter):
    lane_id = "L16"

    # -- verifier --------------------------------------------------------------

    def build_verifier(self) -> Any:
        pin_root = _resolve_pin_root(self.lane_kw)
        xml_store = pin_root / "store" / "l16_uniprot_xml.sqlite"
        if not pin_root.is_dir():
            raise LaneNotSupported(
                "L16 requires the pinned UniProt tree at %s" % pin_root)
        if not xml_store.is_file():
            raise LaneNotSupported(
                "L16 requires the XML-parser store at %s" % xml_store)
        try:
            from benchmark.plugins.verifier.l16.plugin import L16NativeVerifier
        except ImportError as exc:
            raise LaneNotSupported(
                "L16 verifier not importable: %s" % exc) from exc
        try:
            return L16NativeVerifier(pin_root)
        except Exception as exc:
            raise LaneNotSupported(
                "L16 verifier construction failed (pin=%s): %s: %s"
                % (pin_root, type(exc).__name__, exc)) from exc

    # -- AST rebuild -----------------------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        ast = oracle_payload.get("ast")
        if isinstance(ast, dict):
            return ast
        family = str(oracle_payload.get("family") or "")
        answer = oracle_payload.get("expected_answer")
        if not family or not isinstance(answer, list):
            raise LaneNotSupported(
                "L16 oracle payload cannot rebuild its task AST "
                "(family=%r, expected_answer=%s)"
                % (family, type(answer).__name__))
        try:
            from benchmark.plugins.task_family.l16.families import (
                FAMILY_BY_ID,
                STABLE_SEED,
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise LaneNotSupported(
                "L16 family metadata not importable: %s" % exc) from exc
        spec = FAMILY_BY_ID.get(family)
        witness = oracle_payload.get("witness") or {}
        # Same shape as L16TaskFamilies.propose(); ``answer`` is the oracle's
        # recorded expected set (the only field the terminal check reads).
        return {
            "params": dict(oracle_payload.get("params") or {}),
            "answer": list(answer),
            "state": dict(oracle_payload.get("state") or {}),
            "complexity": spec.complexity if spec else "unknown",
            "witness_kind": (witness.get("kind")
                             or (spec.witness_kind if spec else "")),
            "seed": witness.get("seed", STABLE_SEED),
            "generator": "l16-families/1.0",
        }

    # -- reference / negative helpers -----------------------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        answer = oracle_payload.get("expected_answer")
        if not isinstance(answer, list):
            raise LaneNotSupported(
                "L16 oracle payload exposes no expected_answer list")
        return list(answer)

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = self.reference_answer(oracle_payload)
        if len(ref) > 1:
            return list(ref[:-1])          # dropped member
        return list(ref) + ["__aob_decoy__"]  # added spurious member