"""License / redistribution decision matrix (owner: S11).

FINAL-BUILD-ORDER-v2 §2.1/§6.3: redistribution decisions are per source and
must be auditable. Restricted sources ship as `fetch-recipe + attribution +
hash`, never as bundled content.

Decision sources, in precedence order:
1. `source_registry/licenses/source_redistribution_matrix.csv` (authoritative,
   per-source human verdicts written by the lanes/orchestrator)
2. `source_registry/sources.yaml` `redistribution_policy` field (conservative
   mapping below; `verify_per_asset` defaults to HOLD_REVIEW, never to OK)
3. unknown source -> HOLD_REVIEW
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import yaml


class Decision(str, Enum):
    REDISTRIBUTE_OK = "REDISTRIBUTE_OK"  # content may ship in the public release
    REFERENCE_ONLY = "REFERENCE_ONLY"  # cite + summarize, no content bundling
    FETCH_RECIPE_ONLY = "FETCH_RECIPE_ONLY"  # ship recipe + hash; user fetches
    METADATA_ONLY = "METADATA_ONLY"  # metadata index only, no content
    HOLD_REVIEW = "HOLD_REVIEW"  # unresolved: blocks release eligibility


# decisions under which a task's public artifacts may still ship (with content
# replaced by fetch recipe where applicable)
SHIPPABLE: frozenset[Decision] = frozenset(
    {Decision.REDISTRIBUTE_OK, Decision.REFERENCE_ONLY, Decision.FETCH_RECIPE_ONLY,
     Decision.METADATA_ONLY}
)
# decisions under which raw source content may be bundled
CONTENT_OK: frozenset[Decision] = frozenset({Decision.REDISTRIBUTE_OK})

_POLICY_FALLBACK: dict[str, Decision] = {
    "verify_per_asset": Decision.HOLD_REVIEW,
    "docs_reference_only": Decision.REFERENCE_ONLY,
    "metadata_only_until_authorized": Decision.METADATA_ONLY,
    "metadata_index_only": Decision.METADATA_ONLY,
    "fetch_recipe_only": Decision.FETCH_RECIPE_ONLY,
    "reference_only": Decision.REFERENCE_ONLY,
    "redistribute_ok": Decision.REDISTRIBUTE_OK,
}

_CSV_COLUMNS = ("source_id", "license", "decision", "reviewed_by", "reason")


@dataclass(frozen=True)
class LicenseDecision:
    source_id: str
    license: str
    decision: Decision
    basis: str  # "matrix_csv" | "sources_yaml" | "default_hold"
    reason: str = ""

    @property
    def shippable(self) -> bool:
        return self.decision in SHIPPABLE

    @property
    def content_ok(self) -> bool:
        return self.decision in CONTENT_OK


class LicenseMatrix:
    """Lookup table of per-source redistribution decisions."""

    def __init__(self, decisions: dict[str, LicenseDecision]) -> None:
        self._decisions = dict(decisions)

    @classmethod
    def load(
        cls,
        *,
        matrix_csv: Path | None = None,
        sources_yaml: Path | None = None,
    ) -> "LicenseMatrix":
        decisions: dict[str, LicenseDecision] = {}
        yaml_rows: dict[str, dict[str, str]] = {}
        if sources_yaml is not None and sources_yaml.exists():
            doc = yaml.safe_load(sources_yaml.read_text(encoding="utf-8")) or {}
            for row in doc.get("sources", []) or []:
                sid = row.get("source_id")
                if sid:
                    yaml_rows[sid] = {
                        "license": str(row.get("license", "")),
                        "policy": str(row.get("redistribution_policy", "")),
                    }
        for sid, info in yaml_rows.items():
            policy = info["policy"]
            decision = _POLICY_FALLBACK.get(policy, Decision.HOLD_REVIEW)
            decisions[sid] = LicenseDecision(
                source_id=sid, license=info["license"], decision=decision,
                basis="sources_yaml",
                reason="" if policy in _POLICY_FALLBACK else f"unknown policy '{policy}'",
            )
        if matrix_csv is not None and matrix_csv.exists():
            with matrix_csv.open("r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    sid = (row.get("source_id") or "").strip()
                    if not sid:
                        continue
                    raw = (row.get("decision") or "").strip().upper()
                    try:
                        decision = Decision(raw)
                    except ValueError:
                        decision = Decision.HOLD_REVIEW
                    decisions[sid] = LicenseDecision(
                        source_id=sid,
                        license=(row.get("license") or yaml_rows.get(sid, {}).get("license", "")),
                        decision=decision,
                        basis="matrix_csv",
                        reason=(row.get("reason") or "").strip(),
                    )
        return cls(decisions)

    def decide(self, source_id: str) -> LicenseDecision:
        known = self._decisions.get(source_id)
        if known is not None:
            return known
        return LicenseDecision(
            source_id=source_id, license="", decision=Decision.HOLD_REVIEW,
            basis="default_hold", reason="source not present in matrix or registry",
        )

    def gate_task(self, source_ids: list[str]) -> tuple[bool, list[LicenseDecision]]:
        """Release gate: a task is eligible iff every source decision is shippable."""
        verdicts = [self.decide(sid) for sid in source_ids]
        ok = all(v.shippable for v in verdicts)
        return ok, verdicts

    def to_rows(self) -> list[dict[str, str]]:
        return [
            {
                "source_id": d.source_id,
                "license": d.license,
                "decision": d.decision.value,
                "basis": d.basis,
                "reason": d.reason,
            }
            for d in sorted(self._decisions.values(), key=lambda x: x.source_id)
        ]
