"""Canonical goal dedup (T4/T5.12). Paraphrases collapse to ONE goal.

Dedup key = canonical_sha over a semantics-only payload: wording fields
(user_goal_en/zh, surface_style, prompt_hash, plausible_reason, ...) are excluded
by construction, so 20 near-identical paraphrases keep the same hash and count
as a single `base_goal_id` (Implementation section 3 + 4 T4 counting rule).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from autoontobench.contracts.models import canonical_sha

SEMANTIC_KEYS = (
    "blueprint_id",
    "domain",
    "track",
    "goal_type",
    "goal_core",
    "public_entities",
    "constraints",
    "source_rule_unit_id",
    "source_rule_unit_ids",
    "mechanism_group_id",
    "task_family",
)


def canonical_goal_payload(obj: Mapping[str, Any]) -> dict[str, Any]:
    """Semantics-only projection. Unknown wording fields never enter the hash."""
    return {k: _normalize(v) for k, v in ((k, obj.get(k)) for k in SEMANTIC_KEYS)
            if v is not None}


def _normalize(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return sorted(str(x) for x in v)
    if isinstance(v, Mapping):
        return {k: _normalize(v[k]) for k in sorted(v)}
    return v


def goal_sha(obj: Mapping[str, Any]) -> str:
    return canonical_sha(canonical_goal_payload(obj))


@dataclass(frozen=True)
class DedupeResult:
    unique: list[Any]
    duplicate_indices: list[int] = field(default_factory=list)
    kept_index_by_sha: dict[str, int] = field(default_factory=dict)

    @property
    def n_unique_goals(self) -> int:
        return len(self.kept_index_by_sha)


def dedupe_by_goal(items: Iterable[Any], payload_of) -> DedupeResult:
    """Collapse items by canonical goal sha. `payload_of(item)` -> semantic mapping.

    First occurrence wins; later same-goal items are paraphrase duplicates.
    """
    unique: list[Any] = []
    dup_idx: list[int] = []
    kept: dict[str, int] = {}
    for idx, item in enumerate(items):
        sha = goal_sha(payload_of(item))
        if sha in kept:
            dup_idx.append(idx)
            continue
        kept[sha] = len(unique)
        unique.append(item)
    return DedupeResult(unique=unique, duplicate_indices=dup_idx,
                        kept_index_by_sha=kept)
