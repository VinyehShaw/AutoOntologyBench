"""L05 sampling machinery: group pools, universe sampler, SMARTS mutators.

All randomness flows through a seeded ``random.Random`` — every artifact is
reproducible from ``(family, base_seq, instance_seq, seed)``. Universes are
built positive-first: the true positive set of the candidate query over the
full corpus is computed from the index, up to ``k_pos`` positives are drawn,
and the remainder is filled with corpus-uniform negatives. Qualification
gates (nontriviality, interaction) run later in the verifier plugin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from random import Random

from rdkit import Chem


@dataclass(frozen=True)
class GroupPool:
    """Corpus-statistics-bucketed group labels (built once per build run)."""

    frequent: list[str]            # 200..20000 corpus hits
    medium: list[str]              # 30..20000
    mixed: list[str]               # 30..20000 incl. frequent
    parents_with_children: list[str]   # >= 1 child (parent-minus-child family)
    parents_multi: list[str]           # >= 2 children (descendant-union family)
    count_variable: list[str]      # groups whose per-molecule match count varies
    children: dict[str, tuple[str, ...]]  # parent label -> child labels

    def children_of(self, label: str) -> list[str]:
        return list(self.children.get(label, ()))

    @staticmethod
    def build(ctx, corpus_ids: list[str]) -> "GroupPool":
        corpus = frozenset(corpus_ids)
        size = {g: len(ctx.membership.get(g, frozenset()) & corpus)
                for g in ctx.smarts}
        frequent = sorted(g for g, n in size.items() if 200 <= n <= 20000)
        medium = sorted(g for g, n in size.items() if 30 <= n <= 20000)
        mixed = sorted(set(frequent) | set(medium))
        parents = sorted(p for p, ch in ctx.children.items()
                         if len(ch) >= 1 and 200 <= size.get(p, 0) <= 30000)
        parents_multi = sorted(p for p, ch in ctx.children.items()
                               if len(ch) >= 2 and 200 <= size.get(p, 0) <= 30000)
        count_variable = sorted(
            g for g in medium
            if len({ctx.match_counts.get(g, {}).get(m, 0)
                    for m in list(ctx.membership.get(g, ()))[:2000]}) >= 3)
        return GroupPool(frequent=frequent, medium=medium, mixed=mixed,
                         parents_with_children=parents, parents_multi=parents_multi,
                         count_variable=count_variable or medium,
                         children={p: tuple(sorted(ch)) for p, ch in ctx.children.items()})


def sample_universe(rng: Random, positives: list[str], corpus_ids: list[str],
                    *, size: int, k_pos: int) -> list[str] | None:
    """k_pos positives + corpus-uniform fill; None when not enough positives."""
    if len(positives) < k_pos:
        return None
    pos = rng.sample(sorted(positives), k_pos)
    rest = [m for m in corpus_ids if m not in set(pos)]
    fill = rng.sample(rest, size - k_pos)
    universe = pos + fill
    rng.shuffle(universe)
    return universe


def sample_universe_per_label(rng: Random, labels: list[str], ctx,
                              corpus_ids: list[str], *, size: int,
                              k_per_label: int) -> list[str] | None:
    """Guarantee >= k_per_label members of every label in the universe."""
    pos: set[str] = set()
    for label in labels:
        members = sorted(ctx.membership.get(label, frozenset()))
        if len(members) < k_per_label:
            return None
        pos |= set(rng.sample(members, k_per_label))
    if len(pos) >= size:
        return None
    rest = [m for m in corpus_ids if m not in pos]
    universe = sorted(pos) + rng.sample(rest, size - len(pos))
    rng.shuffle(universe)
    return universe


# ---------------------------------------------------------------------------
# SMARTS mutators (repair families). All mutators are deterministic in rng.
# ---------------------------------------------------------------------------

def mutate_syntax(smarts: str, rng: Random) -> str | None:
    """Break bracket/paren balance. The result must fail to parse AND must not
    contain the golden SMARTS as a substring (otherwise the repair is a
    trivial string edit, not a chemistry task)."""
    ops = []
    if "]" in smarts:
        ops.append("drop_close_bracket")
    if "(" in smarts:
        ops.append("drop_open_paren")
    ops.append("insert_open_paren")
    for _ in range(12):
        op = rng.choice(ops)
        if op == "drop_close_bracket":
            i = rng.choice([m.start() for m in re.finditer(re.escape("]"), smarts)])
            broken = smarts[:i] + smarts[i + 1:]
        elif op == "drop_open_paren":
            i = rng.choice([m.start() for m in re.finditer(re.escape("("), smarts)])
            broken = smarts[:i] + smarts[i + 1:]
        else:
            i = rng.randrange(1, len(smarts))
            broken = smarts[:i] + "(" + smarts[i:]
        if smarts in broken:
            continue
        if Chem.MolFromSmarts(broken) is None:
            return broken
    return None


_ATOM_SWAPS = (("N", "S"), ("O", "N"), ("C", "N"), ("c", "n"), ("S", "O"),
               ("Cl", "F"), ("#6", "#7"), ("#7", "#8"), ("#8", "#16"))


def mutate_semantic(smarts: str, rng: Random) -> str | None:
    """Swap one atom primitive; result must parse (behavior check is downstream)."""
    candidates = [(a, b) for a, b in _ATOM_SWAPS if a in smarts]
    rng.shuffle(candidates := list(candidates))
    for a, b in candidates:
        positions = [m.start() for m in re.finditer(re.escape(a), smarts)]
        rng.shuffle(positions)
        for i in positions:
            broken = smarts[:i] + b + smarts[i + len(a):]
            if broken != smarts and Chem.MolFromSmarts(broken) is not None:
                return broken
    return None


_EXCLUSION = re.compile(r"!\$\([^()]*\)")
_CONSTRAINT = re.compile(r";[A-Z0-9!$(),=\-\[\]#]+")


def mutate_overgeneral(smarts: str, rng: Random) -> str | None:
    """Remove one constraint/exclusion; result must parse and differ textually."""
    ops: list[str] = []
    m = _EXCLUSION.search(smarts)
    if m:
        ops.append(smarts[:m.start()] + smarts[m.end():])
    for m2 in _CONSTRAINT.finditer(smarts):
        cand = smarts[:m2.start()] + smarts[m2.end():]
        if cand != smarts:
            ops.append(cand)
    rng.shuffle(ops)
    for cand in ops:
        if Chem.MolFromSmarts(cand) is not None:
            return cand
    return None


def derangement(rng: Random, labels: list[str]) -> dict[str, str]:
    """Seeded derangement: label -> the label whose set it is shown with."""
    n = len(labels)
    for _ in range(200):
        perm = rng.sample(labels, n)
        if all(a != b for a, b in zip(labels, perm)):
            return dict(zip(labels, perm))
    raise RuntimeError("no derangement found")
