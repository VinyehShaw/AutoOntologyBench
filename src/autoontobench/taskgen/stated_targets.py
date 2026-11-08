"""W15 stated-targets seam for the T5 public checks (SOLVABILITY_AUDIT §3/§5).

A blueprint whose ``goal_mode`` is "stated_targets" carries goal value_checks
that are DELIBERATELY public — the wording IS the goal statement, so the stated
(object, path, equals) triples are not hidden terms. For those tasks T5.9
verifies the entailment direction (wording states every triple) instead of the
exposure direction; compliance-only tasks keep the strict exposure check
unchanged. Kept in its own module so checks_public stays reviewable.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

STATED_GOAL_MODE = "stated_targets"


def stated_value_checks(blueprint: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Goal value_checks that are public by design (W15 stated-targets repair)."""
    blueprint = blueprint or {}
    if str(blueprint.get("goal_mode") or "") != STATED_GOAL_MODE:
        return []
    return [vc for vc in (blueprint.get("goal") or {}).get("value_checks", [])
            if isinstance(vc, Mapping)]


def wording_states_target(text: str, vc: Mapping[str, Any]) -> bool:
    """True iff ``text`` states object + path + exact value of a goal value_check
    (token-exact for the value: quoted/backticked strings, word-boundary numbers —
    mirrors the W15 audit's ``wording_mentions``; case-sensitive, never substrings)."""
    oid, path, equals = str(vc.get("object")), str(vc.get("path")), vc.get("equals")
    if oid not in text or path not in text:
        return False
    if isinstance(equals, str):
        return any(form in text
                   for form in (f'"{equals}"', f"`{equals}`", f"'{equals}'"))
    return re.search(r"(?<![\d.])%s(?!\d|\.\d)" % re.escape(str(equals)), text) is not None
