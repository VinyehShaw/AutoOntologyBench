"""L15 eval adapter: Reactome release-98 native truth.

Unlike L13/L14/L16, the shipped L15 oracle payloads carry their task ``ast``
(1,500/1,500 in ``bank/staging_20k/l15/oracle``); ``extract_ast`` uses it
verbatim and falls back to the lane's own ``verify_cli._grounded`` shape only
for a JSON-encoded AST.  A payload with neither (a non-task "summary" file such
as ``oracle/manifest.json``, or a legacy summary row) cannot rebuild a
``GroundedTask`` and is reported NOT_SUPPORTED with the precise reason rather
than silently scored.

Native verifier: ``ReactomeNativeVerifier`` recomputes every reference answer
from the pinned ``reactome.db`` (``external_knowledge/indexes/reactome/
release-98/reactome.db``); the stored oracle reference is never trusted.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DB = (REPO_ROOT / "external_knowledge" / "indexes" / "reactome"
              / "release-98" / "reactome.db")


def _resolve_db(lane_kw: dict[str, str]) -> Path:
    value = (lane_kw.get("db_path")
             or os.environ.get("L15_STORE_DB")
             or str(DEFAULT_DB))
    return Path(value)


class Adapter(LaneAdapter):
    lane_id = "L15"

    # -- verifier --------------------------------------------------------------

    def build_verifier(self) -> Any:
        db = _resolve_db(self.lane_kw)
        if not db.is_file():
            raise LaneNotSupported(
                "L15 requires the pinned Reactome store at %s "
                "(pass --lane-kw L15.db_path=<path> or set L15_STORE_DB)" % db)
        try:
            from benchmark.plugins.verifier.l15.verifier import (
                ReactomeNativeVerifier,
            )
        except ImportError as exc:
            raise LaneNotSupported(
                "L15 verifier not importable: %s" % exc) from exc
        allow_m_only = str(self.lane_kw.get("allow_m_only", "")).lower() in (
            "1", "true", "yes", "on")
        try:
            return ReactomeNativeVerifier(db, allow_m_only=allow_m_only)
        except Exception as exc:
            raise LaneNotSupported(
                "L15 verifier construction failed (db=%s): %s: %s"
                % (db, type(exc).__name__, exc)) from exc

    # -- AST rebuild -----------------------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        ast = oracle_payload.get("ast")
        if isinstance(ast, dict):
            return ast
        if isinstance(ast, str):
            try:
                decoded = json.loads(ast)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                return decoded
        # Non-task / summary rows (e.g. oracle/manifest.json) have no AST.
        raise LaneNotSupported(
            "L15 oracle payload for %s carries no task AST (summary/non-task "
            "row); cannot rebuild GroundedTask"
            % oracle_payload.get("oracle_handle", "<unknown>"))

    # -- reference / negative helpers -----------------------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = oracle_payload.get("reference_submission")
        if ref is None:
            raise LaneNotSupported(
                "L15 oracle payload exposes no reference_submission")
        return ref

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        stored = oracle_payload.get("mutation_submission")
        if stored is not None:
            return stored
        return self.verifier.mutation_submission(
            self.extract_ast(oracle_payload),
            self.reference_answer(oracle_payload))