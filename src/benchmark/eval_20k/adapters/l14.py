"""L14 eval adapter: ChEMBL 37 exact-set / artifact-repair truth.

The 20K L14 oracle payloads carry ``family`` / ``params`` /
``expected_answer`` (plus the witness/reference evidence) but no top-level
``ast``.  ``extract_ast`` rebuilds the AST shape produced by the lane's own
``benchmark.plugins.task_family.l14.plugin.L14TaskFamilies.propose`` from the
oracle's recomputable fields; ``L14Verifier.verify_terminal`` consumes
``ast.ast["answer"]`` (the recorded oracle answer, itself re-derived from the
pinned SQLite by the lane's ``reverify`` CLI / audit gates).

Resource: the pinned ChEMBL 37 SQLite
(``external_knowledge/pinned/chembl/chembl-37``).  Support is gated on the pin
and the SQLite file being present, so a lane that claims SUPPORTED here is one
whose native reverify path is actually runnable in this environment.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "chembl"
                    / "chembl-37")


def _resolve_pin_root(lane_kw: dict[str, str]) -> Path:
    value = (lane_kw.get("pin_root")
             or os.environ.get("AOB_L14_PIN")
             or str(DEFAULT_PIN_ROOT))
    return Path(value)


def _db_path(pin_root: Path) -> Path:
    try:
        from source_registry.fetchers.l14_chembl import db_path
        return db_path(pin_root)
    except Exception:  # pragma: no cover - fall back to the known layout
        return pin_root / "chembl_37" / "chembl_37_sqlite" / "chembl_37.db"


class Adapter(LaneAdapter):
    lane_id = "L14"

    # -- verifier --------------------------------------------------------------

    def build_verifier(self) -> Any:
        pin_root = _resolve_pin_root(self.lane_kw)
        db = _db_path(pin_root)
        if not pin_root.is_dir():
            raise LaneNotSupported(
                "L14 requires the pinned ChEMBL 37 tree at %s" % pin_root)
        if not db.is_file():
            raise LaneNotSupported(
                "L14 requires the pinned ChEMBL 37 SQLite at %s" % db)
        try:
            from benchmark.plugins.verifier.l14.plugin import L14Verifier
        except ImportError as exc:
            raise LaneNotSupported(
                "L14 verifier not importable: %s" % exc) from exc
        try:
            verifier = L14Verifier()
        except Exception as exc:
            raise LaneNotSupported(
                "L14 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc
        verifier.pin_root = pin_root  # carried for diagnostics/self-checks
        return verifier

    # -- AST rebuild -----------------------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        ast = oracle_payload.get("ast")
        if isinstance(ast, dict):
            return ast
        family = str(oracle_payload.get("family") or "")
        answer = oracle_payload.get("expected_answer")
        if not family or not isinstance(answer, list):
            raise LaneNotSupported(
                "L14 oracle payload cannot rebuild its task AST "
                "(family=%r, expected_answer=%s)"
                % (family, type(answer).__name__))
        try:
            from benchmark.plugins.task_family.l14.families import (
                FAMILY_BY_ID,
                STABLE_SEED,
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise LaneNotSupported(
                "L14 family metadata not importable: %s" % exc) from exc
        spec = FAMILY_BY_ID.get(family)
        witness = oracle_payload.get("witness") or {}
        # Same shape as L14TaskFamilies.propose(); the fields the terminal
        # verifier reads (``answer``) are exactly the oracle's answer, and the
        # remaining keys are the lane's own family/witness metadata.
        return {
            "params": dict(oracle_payload.get("params") or {}),
            "answer": list(answer),
            "state": dict(oracle_payload.get("state") or {}),
            "complexity": spec.complexity if spec else "unknown",
            "witness_kind": (witness.get("kind")
                             or (spec.witness_kind if spec else "")),
            "seed": witness.get("seed", STABLE_SEED),
            "generator": "l14-families/1.0",
        }

    # -- reference / negative helpers -----------------------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        answer = oracle_payload.get("expected_answer")
        if not isinstance(answer, list):
            raise LaneNotSupported(
                "L14 oracle payload exposes no expected_answer list")
        return list(answer)

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = self.reference_answer(oracle_payload)
        if len(ref) > 1:
            return list(ref[:-1])          # dropped member
        return list(ref) + ["__aob_decoy__"]  # added spurious member