"""20K harness adapter for lane L10 (BDDL symbolic checks).

The lane has two native verifier generations over the same pinned
BEHAVIOR-1K BDDL snapshot:

* ``benchmark.plugins.verifier.l10.bddl_verifier.BddlNativeVerifier`` -- the
  legacy families (``obligation_query`` ... ``defect_audit``);
* ``benchmark.plugins.verifier.l10.bddl_verifier_20k.BddlVerifier20K`` -- the
  20K-expansion families (``NEW_KINDS``: ``quant_inventory`` ...
  ``predicate_usage``).

Oracle payloads carry ``ast`` directly (``ast.goal_kind`` selects the owning
verifier generation); the adapter rebuilds the ``GroundedTask`` through the
base class and only dispatches ``verify_terminal``.  No lane semantics are
re-implemented here.

Oracle directories are resolved (read-only) from:

1. ``--lane-kw L10.oracle_dir=<dir>``
2. ``$AOB_L10_ORACLE``
3. the in-repo staging trees ``bank/staging_20k/l10/oracle`` (20K) and
   ``bank/staging/l10/oracle`` (legacy).

Neither constructor writes to or chmods the oracle directory (only
``write_oracle`` does, and the harness never calls it), so the staging trees
are left untouched.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported

REPO_ROOT = Path(__file__).resolve().parents[3]
STAGING_20K_ORACLE = REPO_ROOT / "bank" / "staging_20k" / "l10" / "oracle"
STAGING_LEGACY_ORACLE = REPO_ROOT / "bank" / "staging" / "l10" / "oracle"


def _has_payloads(directory: Path) -> bool:
    return directory.is_dir() and any(directory.glob("*.json"))


def _explicit_oracle_dir(lane_kw: dict[str, str]) -> Path | None:
    value = lane_kw.get("oracle_dir") or os.environ.get("AOB_L10_ORACLE")
    return Path(value) if value else None


def _resolve_oracle_dirs(
        lane_kw: dict[str, str]) -> tuple[Path | None, Path | None]:
    """(legacy_dir, expansion_dir); an explicit override is used for both."""
    explicit = _explicit_oracle_dir(lane_kw)
    if explicit is not None:
        if not _has_payloads(explicit):
            raise LaneNotSupported(
                "lane L10 oracle dir has no *.json payloads: %s" % explicit)
        return explicit, explicit
    legacy = STAGING_LEGACY_ORACLE if _has_payloads(STAGING_LEGACY_ORACLE) \
        else None
    expansion = STAGING_20K_ORACLE if _has_payloads(STAGING_20K_ORACLE) \
        else None
    # A single repo tree can serve both generations (the verifiers' material
    # comes from the pinned snapshot; the dir is only used for diagnostics).
    legacy = legacy or expansion
    expansion = expansion or legacy
    return legacy, expansion


class _L10Dispatch:
    """Dispatch to the legacy / 20K native verifier by ``ast.goal_kind``."""

    def __init__(self, legacy: Any, expansion: Any) -> None:
        self.legacy = legacy
        self.expansion = expansion

    def verifier_for(self, ast: dict[str, Any]) -> Any:
        from benchmark.plugins.verifier.l10.bddl_verifier_20k import NEW_KINDS

        return (self.expansion if ast.get("goal_kind") in NEW_KINDS
                else self.legacy)

    def verify_terminal(self, task: Any, submitted: Any) -> Any:
        return self.verifier_for(task.ast.ast).verify_terminal(task, submitted)


class Adapter(LaneAdapter):
    lane_id = "L10"

    # -- construction ---------------------------------------------------------

    def build_verifier(self) -> Any:
        try:
            from benchmark.plugins.verifier.l10.bddl_verifier import (
                BddlNativeVerifier,
            )
            from benchmark.plugins.verifier.l10.bddl_verifier_20k import (
                BddlVerifier20K,
            )
            from benchmark.plugins.task_family.l10 import engine
        except ImportError as exc:
            raise LaneNotSupported(
                "L10 native verifier not importable: %s" % exc) from exc

        if not Path(engine.PINNED_ROOT).is_dir():
            raise LaneNotSupported(
                "lane L10 needs the pinned BEHAVIOR-1K BDDL snapshot at %s"
                % engine.PINNED_ROOT)

        legacy_dir, expansion_dir = _resolve_oracle_dirs(self.lane_kw)
        if legacy_dir is None or expansion_dir is None:
            raise LaneNotSupported(
                "lane L10 requires oracle_dir (--lane-kw "
                "L10.oracle_dir=... or AOB_L10_ORACLE); no in-repo staging "
                "oracle found at %s or %s"
                % (STAGING_20K_ORACLE, STAGING_LEGACY_ORACLE))
        try:
            legacy = BddlNativeVerifier(legacy_dir)
            expansion = BddlVerifier20K(expansion_dir)
        except Exception as exc:
            raise LaneNotSupported(
                "L10 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc
        return _L10Dispatch(legacy, expansion)

    def verifier_id(self, verifier: Any) -> str:
        if isinstance(verifier, _L10Dispatch):
            return ("benchmark.plugins.verifier.l10.bddl_verifier."
                    "BddlNativeVerifier|benchmark.plugins.verifier.l10."
                    "bddl_verifier_20k.BddlVerifier20K")
        return super().verifier_id(verifier)

    # -- reference / negative helpers (self-checks) ---------------------------

    def _verifier_for_payload(self, oracle_payload: dict[str, Any]) -> Any:
        ast = self.extract_ast(oracle_payload)
        verifier = self.verifier
        return verifier.verifier_for(ast)

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """The verifier's own recomputed reference submission.

        L10 oracle payloads deliberately store no reference answer; the native
        verifier recomputes it from the pinned BDDL bytes.
        """
        ast = self.extract_ast(oracle_payload)
        return self._verifier_for_payload(oracle_payload).reference_submission(ast)

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """The verifier's own seeded wrong submission (negative witness)."""
        verifier = self._verifier_for_payload(oracle_payload)
        material = verifier._material(self.extract_ast(oracle_payload))
        return verifier._wrong_submission(material)


__all__ = ["Adapter"]