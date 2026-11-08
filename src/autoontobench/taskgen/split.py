"""T6 source-lineage splits (skeleton).

Grouping keys: source_rule_unit_id ancestry + mechanism group hash. Seeds,
paraphrases and variants of one unit stay in ONE lineage group so hidden-source /
hidden-mechanism holdouts are truly unseen (Implementation section 4 T6). Final
split assignment locks at preregistration (G4); this module only groups.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable

from autoontobench.contracts.models import TaskSpecV5


@dataclass(frozen=True)
class LineageGroup:
    group_key: str
    source_rule_unit_ids: tuple[str, ...]
    mechanism_group_id: str
    task_ids: list[str] = field(default_factory=list)


def lineage_key(spec: TaskSpecV5) -> str:
    units = ",".join(sorted(r.source_rule_unit_id for r in spec.source_refs))
    blob = f"{units}|{spec.mechanism_group_id}"
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def group_by_lineage(specs: Iterable[TaskSpecV5]) -> dict[str, LineageGroup]:
    groups: dict[str, LineageGroup] = {}
    members: dict[str, list[str]] = {}
    for spec in specs:
        key = lineage_key(spec)
        members.setdefault(key, []).append(spec.task_id)
        if key not in groups:
            groups[key] = LineageGroup(
                group_key=key,
                source_rule_unit_ids=tuple(sorted({r.source_rule_unit_id
                                                   for r in spec.source_refs})),
                mechanism_group_id=spec.mechanism_group_id,
                task_ids=members[key],
            )
    return groups
