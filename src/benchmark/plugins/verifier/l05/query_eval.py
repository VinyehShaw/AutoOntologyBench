"""L05 query DSL evaluators — dual implementation for the differential gate.

The query AST is a small closed DSL over the molecule universe:

    {"op": "group", "group": label}                  membership set of a group
    {"op": "and"/"or", "args": [q, ...]}
    {"op": "xor", "args": [q, ...]}                  exactly one of the args
    {"op": "not", "arg": q}
    {"op": "count_range", "group": g, "lo": a, "hi": b}   substructure match count
    {"op": "heavy_band", "lo": a, "hi": b}                heavy atoms in [lo, hi]
    {"op": "profile", "pos": [g...], "neg": [g...]}       exact boolean profile
    {"op": "descendants_of", "group": p}                  union of strict descendants
    {"op": "group_minus_descendants", "group": p}

``eval_index`` answers from the precomputed membership/match-count index.
``eval_native`` answers by compiling SMARTS and matching with RDKit directly.
Qualification requires both to agree on the universe (second-parser rule,
Build-order §4.3); both are pure functions of (query, universe, context).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem

from benchmark.plugins.source.l05.features import molecule_features


class QueryError(ValueError):
    pass


@dataclass(frozen=True)
class GroupContext:
    """Hierarchy facts needed to evaluate queries (immutable)."""

    smarts: dict[str, str]
    children: dict[str, frozenset]  # label -> direct child labels
    heavy: dict[str, int]           # chebi_id -> heavy atoms
    membership: dict[str, frozenset]   # label -> chebi_ids (match_count >= 1)
    match_counts: dict[str, dict[str, int]]
    # 20K expansion: canonical SMILES + lazily computed descriptor cache. The
    # cache is an *index* path; eval_native recomputes from Mol objects.
    smiles: dict[str, str] = field(default_factory=dict)
    _features: dict[str, dict[str, int]] = field(default_factory=dict,
                                                 repr=False, compare=False)

    def feature(self, chebi_id: str) -> dict[str, int]:
        """Descriptor vector from the cached index path (computed on demand)."""
        cached = self._features.get(chebi_id)
        if cached is not None:
            return cached
        smi = self.smiles.get(chebi_id)
        if smi is None:
            raise QueryError(f"no canonical SMILES for {chebi_id!r}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise QueryError(f"unparseable canonical SMILES for {chebi_id!r}")
        feats = molecule_features(mol)
        self._features[chebi_id] = feats
        return feats

    def descendant_labels(self, label: str) -> frozenset:
        out: set[str] = set()
        stack = list(self.children.get(label, ()))
        while stack:
            cur = stack.pop()
            out.add(cur)
            stack.extend(self.children.get(cur, ()))
        return frozenset(out)

    @staticmethod
    def from_index(index: dict) -> "GroupContext":
        groups = index["hierarchy"]
        children: dict[str, set] = {}
        for g in groups:
            children.setdefault(g["parent_label"] or "", set())
            if g["parent_label"]:
                children.setdefault(g["parent_label"], set()).add(g["label"])
        return GroupContext(
            smarts={g["label"]: g["smarts"] for g in groups},
            children={k: frozenset(v) for k, v in children.items() if k},
            heavy={m["chebi_id"]: m["heavy_atoms"] for m in index["molecules"]},
            membership={k: frozenset(v) for k, v in index["membership"].items()},
            match_counts=index["match_counts"],
            smiles={m["chebi_id"]: m["canonical_smiles"] for m in index["molecules"]},
        )


def group_leaves(query: dict, positive_only: bool = False,
                 path: tuple = ()) -> list[tuple[tuple, str]]:
    """(path, label) for each group leaf. Path steps: ("args", i) | ("arg",) |
    ("self",) | ("pos", i) | ("neg", i) — navigable by replace_leaf."""
    op = query.get("op")
    if op in ("group", "count_range", "descendants_of", "group_minus_descendants"):
        return [(path + (("self",),), query["group"])]
    if op == "prop_band_with_group":
        return [(path + (("group",),), query["group"])]
    if op == "count_range_dual":
        return [(path + (("groups", i),), g)
                for i, g in enumerate(query["groups"])]
    if op == "not" and positive_only:
        return []
    out: list[tuple[tuple, str]] = []
    if op in ("and", "or", "xor"):
        for i, a in enumerate(query["args"]):
            out.extend(group_leaves(a, positive_only, path + (("args", i),)))
    elif op == "not":
        out.extend(group_leaves(query["arg"], positive_only, path + (("arg",),)))
    elif op == "profile":
        for i, g in enumerate(query.get("pos", [])):
            out.append((path + (("pos", i),), g))
        if not positive_only:
            for i, g in enumerate(query.get("neg", [])):
                out.append((path + (("neg", i),), g))
    return out


def replace_leaf(query: dict, path: tuple, new_label: str) -> dict:
    import copy
    q = copy.deepcopy(query)
    node = q
    *steps, last = path
    for step in steps:
        node = node["args"][step[1]] if step[0] == "args" else node[step[0]]
    if last[0] in ("self", "group"):
        node["group"] = new_label
    elif last[0] == "groups":
        node["groups"][last[1]] = new_label
    elif last[0] == "args":
        node["args"][last[1]]["group"] = new_label
    elif last[0] == "arg":
        node["arg"]["group"] = new_label
    else:  # pos / neg: plain string list entries
        node[last[0]][last[1]] = new_label
    return q


def _group_set_index(ctx: GroupContext, label: str) -> frozenset:
    if label not in ctx.smarts:
        raise QueryError(f"unknown group label {label!r}")
    return ctx.membership.get(label, frozenset())


def eval_index(query: dict, ctx: GroupContext, universe: frozenset) -> frozenset:
    op = query.get("op")
    if op == "group":
        return _eval_group_index(query, ctx, universe)
    if op == "and":
        args = [eval_index(a, ctx, universe) for a in query["args"]]
        return frozenset.intersection(*args) if args else frozenset()
    if op == "or":
        out: set[str] = set()
        for a in query["args"]:
            out |= eval_index(a, ctx, universe)
        return frozenset(out)
    if op == "xor":
        args = [eval_index(a, ctx, universe) for a in query["args"]]
        return frozenset(m for m in universe if sum(m in s for s in args) == 1)
    if op == "not":
        return frozenset(universe - eval_index(query["arg"], ctx, universe))
    if op == "count_range":
        counts = ctx.match_counts.get(query["group"], {})
        return frozenset(m for m in universe
                         if query["lo"] <= counts.get(m, 0) <= query["hi"])
    if op == "count_range_dual":
        ranges = [tuple(r) for r in query["ranges"]]
        return frozenset(
            m for m in universe
            if all(lo <= ctx.match_counts.get(g, {}).get(m, 0) <= hi
                   for g, (lo, hi) in zip(query["groups"], ranges)))
    if op == "prop_band":
        return _prop_band_index(query["prop"], query["lo"], query["hi"], ctx, universe)
    if op == "prop_band_with_group":
        band = _prop_band_index(query["prop"], query["lo"], query["hi"], ctx, universe)
        return band & _group_set_index(ctx, query["group"]) & universe
    if op == "element_band":
        return _prop_band_index(f"elem_{query['element']}", query["lo"], query["hi"],
                                ctx, universe)
    if op == "heavy_band":
        return frozenset(m for m in universe
                         if query["lo"] <= ctx.heavy.get(m, 0) <= query["hi"])
    if op == "profile":
        pos = frozenset.intersection(
            *[_group_set_index(ctx, g) for g in query["pos"]]) if query["pos"] else universe
        neg: set[str] = set()
        for g in query["neg"]:
            neg |= _group_set_index(ctx, g)
        return frozenset((universe & pos) - neg)
    if op == "descendants_of":
        out: set[str] = set()
        for label in ctx.descendant_labels(query["group"]):
            out |= _group_set_index(ctx, label)
        return frozenset(out & universe)
    if op == "group_minus_descendants":
        base = _group_set_index(ctx, query["group"])
        desc: set[str] = set()
        for label in ctx.descendant_labels(query["group"]):
            desc |= _group_set_index(ctx, label)
        return frozenset((base - desc) & universe)
    raise QueryError(f"unknown op {op!r}")


def _eval_group_index(query: dict, ctx: GroupContext, universe: frozenset) -> frozenset:
    return frozenset(_group_set_index(ctx, query["group"]) & universe)


def _prop_band_index(prop: str, lo: int, hi: int, ctx: GroupContext,
                     universe: frozenset) -> frozenset:
    return frozenset(m for m in universe if lo <= ctx.feature(m).get(prop, 0) <= hi)


def _group_set_native(ctx: GroupContext, label: str, mols: dict[str, Chem.Mol]) -> frozenset:
    if label not in ctx.smarts:
        raise QueryError(f"unknown group label {label!r}")
    return frozenset(match_smarts(ctx.smarts[label], mols))


def match_smarts(smarts: str, mols: dict[str, Chem.Mol]) -> list[str]:
    """Direct native substructure match; raises QueryError on bad SMARTS."""
    pat = Chem.MolFromSmarts(smarts)
    if pat is None:
        raise QueryError(f"unparsable SMARTS {smarts!r}")
    return sorted(mid for mid, mol in mols.items() if mol.HasSubstructMatch(pat))


def eval_native(query: dict, ctx: GroupContext, mols: dict[str, Chem.Mol]) -> frozenset:
    universe = frozenset(mols)
    op = query.get("op")
    if op == "group":
        return _group_set_native(ctx, query["group"], mols)
    if op == "and":
        args = [eval_native(a, ctx, mols) for a in query["args"]]
        return frozenset.intersection(*args) if args else frozenset()
    if op == "or":
        out: set[str] = set()
        for a in query["args"]:
            out |= eval_native(a, ctx, mols)
        return frozenset(out)
    if op == "xor":
        args = [eval_native(a, ctx, mols) for a in query["args"]]
        return frozenset(m for m in universe if sum(m in s for s in args) == 1)
    if op == "not":
        return frozenset(universe - eval_native(query["arg"], ctx, mols))
    if op == "count_range":
        pat = Chem.MolFromSmarts(ctx.smarts[query["group"]])
        if pat is None:
            raise QueryError(f"unparsable group SMARTS for {query['group']!r}")
        return frozenset(m for m, mol in mols.items()
                         if query["lo"] <= len(mol.GetSubstructMatches(pat, maxMatches=1000))
                         <= query["hi"])
    if op == "count_range_dual":
        pats = []
        for group in query["groups"]:
            pat = Chem.MolFromSmarts(ctx.smarts[group])
            if pat is None:
                raise QueryError(f"unparsable group SMARTS for {group!r}")
            pats.append(pat)
        return frozenset(
            m for m, mol in mols.items()
            if all(lo <= len(mol.GetSubstructMatches(pat, maxMatches=1000)) <= hi
                   for pat, (lo, hi) in zip(pats, query["ranges"])))
    if op == "prop_band":
        return _prop_band_native(query["prop"], query["lo"], query["hi"], mols)
    if op == "prop_band_with_group":
        band = _prop_band_native(query["prop"], query["lo"], query["hi"], mols)
        return band & _group_set_native(ctx, query["group"], mols)
    if op == "element_band":
        return _prop_band_native(f"elem_{query['element']}", query["lo"], query["hi"],
                                 mols)
    if op == "heavy_band":
        return frozenset(m for m, mol in mols.items()
                         if query["lo"] <= mol.GetNumHeavyAtoms() <= query["hi"])
    if op == "profile":
        pos = frozenset.intersection(
            *[_group_set_native(ctx, g, mols) for g in query["pos"]]) if query["pos"] else universe
        neg: set[str] = set()
        for g in query["neg"]:
            neg |= _group_set_native(ctx, g, mols)
        return frozenset((universe & pos) - neg)
    if op == "descendants_of":
        out: set[str] = set()
        for label in ctx.descendant_labels(query["group"]):
            out |= _group_set_native(ctx, label, mols)
        return frozenset(out & universe)
    if op == "group_minus_descendants":
        base = _group_set_native(ctx, query["group"], mols)
        desc: set[str] = set()
        for label in ctx.descendant_labels(query["group"]):
            desc |= _group_set_native(ctx, label, mols)
        return frozenset(base - desc)
    raise QueryError(f"unknown op {op!r}")


def _prop_band_native(prop: str, lo: int, hi: int,
                      mols: dict[str, Chem.Mol]) -> frozenset:
    return frozenset(m for m, mol in mols.items()
                     if lo <= molecule_features(mol).get(prop, 0) <= hi)
