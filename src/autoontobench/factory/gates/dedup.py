"""Canonical dedup, lineage grouping, and whole-group split assignment (owner: S11).

FINAL-BUILD-ORDER-v2 §4.4 / §1:
- dedup runs on canonical signatures from `canonicalize.py` (family / base-goal /
  instance), never on wording.
- `source_lineage_group` merges tasks that share lineage, mechanism group, or an
  identical base-goal signature (cross-source similar mechanisms dedup together).
- train/dev/test assignment is by WHOLE GROUP: a lineage group never crosses a
  split (leakage prevention at the split layer).
- duplicate-rate statistics are reported per lane and overall.

Idempotent: same input records -> byte-identical decisions and stats.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

SPLITS: tuple[str, ...] = ("train", "dev", "test")
DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "dev": 0.15, "test": 0.15}


@dataclass(frozen=True)
class DedupRecord:
    """One candidate's canonical identity (hashes precomputed by canonicalize)."""

    task_id: str
    lane_id: str
    family_id: str
    base_goal_id: str
    instance_sig: str
    base_goal_sig: str
    family_sig: str
    lineage_group: str
    mechanism_group_id: str


@dataclass(frozen=True)
class DedupDecision:
    task_id: str
    is_duplicate: bool
    duplicate_of: str | None
    level: str | None  # "instance" | "base_goal" | None


@dataclass(frozen=True)
class DedupStats:
    lane_id: str
    total: int
    unique_instances: int
    unique_base_goals: int
    families: int
    mechanism_groups: int
    lineage_groups: int
    instance_duplicate_rate: float
    base_goal_duplicate_rate: float
    max_family_share: float


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        root = x
        while self.parent.get(root, root) != root:
            root = self.parent[root]
        while self.parent.get(x, x) != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class SignatureIndex:
    """First-writer-wins duplicate detection at instance and base-goal level."""

    def __init__(self) -> None:
        self._by_instance: dict[str, str] = {}
        self._by_goal: dict[str, str] = {}

    def add(self, record: DedupRecord) -> DedupDecision:
        keeper = self._by_instance.get(record.instance_sig)
        if keeper is not None:
            return DedupDecision(record.task_id, True, keeper, "instance")
        keeper = self._by_goal.get(record.base_goal_sig)
        if keeper is not None:
            return DedupDecision(record.task_id, True, keeper, "base_goal")
        self._by_instance[record.instance_sig] = record.task_id
        self._by_goal[record.base_goal_sig] = record.task_id
        return DedupDecision(record.task_id, False, None, None)


def assign_lineage_groups(records: list[DedupRecord]) -> dict[str, str]:
    """Union tasks sharing lineage hash, mechanism group, or base-goal signature.

    Returns task_id -> group_id (deterministic: group id = smallest member key).
    """
    uf = _UnionFind()
    seen_keys: dict[str, str] = {}  # grouping key -> first task_id seen with it
    for rec in sorted(records, key=lambda r: r.task_id):
        for key in (
            f"lineage:{rec.lineage_group}",
            f"mech:{rec.mechanism_group_id}",
            f"goal:{rec.base_goal_sig}",
        ):
            other = seen_keys.get(key)
            if other is None:
                seen_keys[key] = rec.task_id
            else:
                uf.union(other, rec.task_id)
    return {rec.task_id: uf.find(rec.task_id) for rec in records}


def assign_splits(
    groups: dict[str, str],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Whole-group deterministic split assignment (no group crosses splits)."""
    table = ratios or DEFAULT_SPLIT_RATIOS
    buckets: list[tuple[str, float]] = []
    acc = 0.0
    for split in SPLITS:
        acc += table.get(split, 0.0)
        buckets.append((split, acc))
    out: dict[str, str] = {}
    for task_id, group_id in sorted(groups.items()):
        digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()
        draw = int(digest[:8], 16) / 0xFFFFFFFF
        split = SPLITS[-1]
        for name, upper in buckets:
            if draw <= upper:
                split = name
                break
        out[task_id] = split
    return out


def duplicate_rate_stats(
    records: list[DedupRecord], groups: dict[str, str]
) -> list[DedupStats]:
    """Per-lane + TOTAL duplicate-rate and diversity statistics."""

    def lane_stats(lane: str, rows: list[DedupRecord]) -> DedupStats:
        total = len(rows)
        uniq_i = len({r.instance_sig for r in rows})
        uniq_g = len({r.base_goal_sig for r in rows})
        family_counts: dict[str, int] = {}
        for r in rows:
            family_counts[r.family_id] = family_counts.get(r.family_id, 0) + 1
        return DedupStats(
            lane_id=lane,
            total=total,
            unique_instances=uniq_i,
            unique_base_goals=uniq_g,
            families=len(family_counts),
            mechanism_groups=len({r.mechanism_group_id for r in rows}),
            lineage_groups=len({groups[r.task_id] for r in rows}),
            instance_duplicate_rate=(total - uniq_i) / total if total else 0.0,
            base_goal_duplicate_rate=(total - uniq_g) / total if total else 0.0,
            max_family_share=(max(family_counts.values()) / total) if total else 0.0,
        )

    by_lane: dict[str, list[DedupRecord]] = {}
    for rec in records:
        by_lane.setdefault(rec.lane_id, []).append(rec)
    stats = [lane_stats(lane, by_lane[lane]) for lane in sorted(by_lane)]
    if records:
        stats.append(lane_stats("TOTAL", records))
    return stats


def run_dedup(
    records: list[DedupRecord], *, seed: int
) -> tuple[dict[str, DedupDecision], dict[str, str], dict[str, str], list[DedupStats]]:
    """Full pass: duplicates -> lineage groups -> splits -> stats. Idempotent."""
    index = SignatureIndex()
    decisions: dict[str, DedupDecision] = {}
    for rec in sorted(records, key=lambda r: r.task_id):
        decisions[rec.task_id] = index.add(rec)
    groups = assign_lineage_groups(records)
    splits = assign_splits(groups, seed=seed)
    stats = duplicate_rate_stats(records, groups)
    return decisions, groups, splits, stats
