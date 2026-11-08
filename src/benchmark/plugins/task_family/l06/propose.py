"""L06 task-family plugin: propose CandidateTaskASTs over the GO index and ground them.

Pipeline position: source index -> propose (this file) -> ground -> native
verifier gates -> shard writer (build.py).

Determinism: every draw is seeded ``Random(f"{family}|base|{seq}|{seed}")`` /
``Random(f"{family}|inst|{base_goal_id}|{seq}|{seed}")``; the stable seed is
20261027. All answers are defined on the task fragment under the declared
profile go-rel-closure/1 and recomputed by the verifier.

The public view carries the fragment (term ids + names + namespaces + induced
edge list) and the question; it never carries the serialized answer. For
repair families the public edge list is the *tampered* fragment.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from random import Random
from typing import Any, Iterator

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.source.l06.closure import REGULATES_KINDS
from benchmark.plugins.source.l06.plugin import L06SourcePlugin
from benchmark.plugins.task_family.l06.families import (
    BY_ID,
    FRAGMENT_CAP,
    FamilySpec,
)
from benchmark.plugins.verifier.l06.fragment import (
    induce,
    lowest_common_ancestors,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l06_go import RELEASE_ID

LANE = "L06"
PROFILE = "go-rel-closure/1"
SEED = 20261027


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _derangement(rng: Random, items: list[str]) -> dict[str, str]:
    """Deterministic derangement (no fixed points) of items."""
    for _ in range(200):
        perm = items[:]
        rng.shuffle(perm)
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))
    n = len(items)
    return {items[i]: items[(i + 1) % n] for i in range(n)}


class L06TaskFamilyPlugin:
    """TaskFamilyPlugin for the 17 L06 families (families.FAMILY_TABLE)."""

    def __init__(self, index_dir: str | Path, *, seed: int = SEED,
                 index: dict[str, Any] | None = None) -> None:
        self.index = index if index is not None else L06SourcePlugin.load_index(
            Path(index_dir))
        self.graph = self.index["graph"]
        self.closures = self.index["closures"]
        self.terms = self.index["terms"]
        self.depth_global = self.index["depth"]
        self.gene_terms = self.index["gene_terms"]
        self.seed = seed
        self._pools: dict[str, Any] | None = None

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    # ------------------------------------------------------------------ pools

    def _build_pools(self) -> dict[str, Any]:
        isa_anc = self.closures["isa_anc"]
        desc_count: dict[str, int] = {}
        for t, anc in isa_anc.items():
            for a in anc:
                desc_count[a] = desc_count.get(a, 0) + 1
        live = sorted(t for t, m in self.terms.items() if not m["obsolete"])
        by_ns: dict[str, list[str]] = {}
        for t in live:
            by_ns.setdefault(self.terms[t]["namespace"], []).append(t)
        partof_anc = self.closures["partof_anc"]
        reg_anc = self.closures["reg_anc"]
        genes_bp = {
            g: by["P"] for g, by in self.gene_terms.items()
            if len(by.get("P", [])) >= 3
        }
        direct_regs: dict[str, set[str]] = {}
        for s, by_pred in self.graph["rel_out"].items():
            for k in REGULATES_KINDS:
                for o in by_pred.get(k, ()):
                    direct_regs.setdefault(o, set()).add(s)
        pools = {
            "live": live,
            "by_ns": by_ns,
            "desc_count": desc_count,
            "anc_seeds": [t for t in live
                          if 3 <= len(isa_anc.get(t, [])) <= 80
                          and self.depth_global.get(t, 0) >= 3],
            "desc_seeds": [t for t in live if 3 <= desc_count.get(t, 0) <= 400],
            "partof_seeds": [t for t in live if len(partof_anc.get(t, [])) >= 3],
            "reg_seeds": [t for t in live if len(reg_anc.get(t, [])) >= 3],
            "reg_direct_seeds": sorted(t for t, rs in direct_regs.items()
                                       if len(rs) >= 2 and t in live),
            "direct_regs": direct_regs,
            "pair_roots": [t for t in live
                           if 150 <= desc_count.get(t, 0) <= 3000
                           and self.depth_global.get(t, 0) >= 3],
            "root_pool": [t for t in live
                          if 200 <= desc_count.get(t, 0) <= 4000
                          and self.depth_global.get(t, 0) >= 3],
            "swap_edges": [],   # filled lazily per instance draw
            "genes_bp": genes_bp,
            "distractor_roots": [t for t in live
                                 if 30 <= desc_count.get(t, 0) <= 400
                                 and self.depth_global.get(t, 0) >= 2],
        }
        return pools

    @property
    def pools(self) -> dict[str, Any]:
        if self._pools is None:
            self._pools = self._build_pools()
        return self._pools

    # -------------------------------------------------------------- fragments

    def fragment_terms(self, root: str, cap: int) -> list[str]:
        """Deterministic BFS over is_a children from root, truncated to cap."""
        children = self.graph["isa_children"]
        out: list[str] = []
        seen = {root}
        queue = [root]
        while queue and len(out) < cap:
            node = queue.pop(0)
            out.append(node)
            for c in sorted(children.get(node, [])):
                if c not in seen and len(out) + len(queue) < cap * 4:
                    seen.add(c)
                    queue.append(c)
        return sorted(seen)

    def _fragment_for_seed(self, seed_term: str, cap: int, *,
                           required: tuple[str, ...] = (),
                           idx: int = 0) -> list[str] | None:
        """Universe = descendant cone of an ancestor of seed_term (fragment
        semantics are fragment-local; see fragment.py). The root must be an
        ancestor of seed_term AND of every term in ``required`` (so pair/repair
        anchors are inside the fragment); roots are ordered by global
        descendant count and ``idx`` picks one deterministically. Once ``idx``
        cycles past the root pool, a sibling branch of the root is pruned
        (rotating by idx), giving each instance of a base goal a distinct
        fragment even when the viable root pool is small."""
        desc_count = self.pools["desc_count"]
        isa_anc = self.closures["isa_anc"]
        cands = [a for a in isa_anc.get(seed_term, [])
                 if 40 <= desc_count.get(a, 0) <= 6000
                 and all(a == r or a in isa_anc.get(r, []) for r in required)]
        if not cands:
            return None
        cands.sort(key=lambda a: (desc_count.get(a, 0), a))
        root = cands[idx % len(cands)]
        prune = idx // len(cands)
        terms = self.fragment_terms(root, cap)
        if seed_term not in terms:
            return None
        if any(r not in terms for r in required):
            return None
        return self._prune_variant(root, seed_term, required, terms, prune)

    def _prune_variant(self, root: str, seed_term: str,
                       required: tuple[str, ...], terms: list[str],
                       prune: int) -> list[str] | None:
        """Remove the sub-cones of the child branches of ``root`` selected by
        the bits of ``prune`` (branch i is removed when bit i is set), skipping
        branches that lead to the seed or any required term; None if the seed
        or a required term would be lost. Bit selection gives 2^n - 1 distinct
        pruned variants instead of the n a rotating index would cycle through."""
        if prune <= 0:
            return terms
        isa_anc = self.closures["isa_anc"]
        tset = set(terms)
        children = sorted(c for c in self.graph["isa_children"].get(root, [])
                          if c in tset)
        keep: set[str] = set()
        for c in children:
            if c == seed_term or c in isa_anc.get(seed_term, []):
                keep.add(c)
            if any(c == r or c in isa_anc.get(r, []) for r in required):
                keep.add(c)
        prunable = [c for c in children if c not in keep]
        if not prunable:
            return terms
        removed: set[str] = set()
        p = prune
        bit = 0
        while p and bit < len(prunable):
            if p & 1:
                victim = prunable[bit]
                removed.add(victim)
                stack = [victim]
                while stack:
                    node = stack.pop()
                    for ch in self.graph["isa_children"].get(node, []):
                        if ch in tset and ch not in removed:
                            removed.add(ch)
                            stack.append(ch)
            p >>= 1
            bit += 1
        if not removed:
            return terms
        out = tset - removed
        if seed_term not in out or any(r not in out for r in required):
            return None
        return sorted(out)

    def _rooted_universe(self, root: str, cap: int, *, idx: int = 0) -> list[str] | None:
        """Universe = cone of ``root`` (fixed by the base goal) with the
        per-instance bit-pruned variant; the root is part of the goal params so
        depth-band / multi-parent goals have a non-degenerate goal space."""
        terms = self.fragment_terms(root, cap)
        if len(terms) < 40:
            return None
        pruned = self._prune_variant(root, root, (), terms, idx)
        if pruned is None or len(pruned) < 40:
            return None
        return pruned

    # ---------------------------------------------------------------- propose

    def iter_base_goals(self, family: str, *, max_goals: int,
                        seed: int | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield (base_goal_id, base) pairs: distinct base goals for ``family``,
        drawn deterministically until ``max_goals`` distinct goals or the draw
        budget (60 draws per goal) is exhausted."""
        spec = BY_ID[family]
        base_seed = self.seed if seed is None else seed
        seen_goals: set[str] = set()
        base_seq = 0
        while len(seen_goals) < max_goals and base_seq < max_goals * 60:
            rng = Random(f"{family}|base|{base_seq}|{base_seed}")
            base_seq += 1
            base = self._propose_base(spec, rng)
            if base is None:
                continue
            base_goal_id = "bg-" + canonical_sha(
                {"lane": LANE, "family": family, "params": base["goal_params"]})[:20]
            if base_goal_id in seen_goals:
                continue
            seen_goals.add(base_goal_id)
            yield base_goal_id, base

    def instantiate(self, spec: FamilySpec, base: dict[str, Any],
                    base_goal_id: str, inst_seq: int,
                    base_seed: int) -> CandidateTaskAST | None:
        """Public per-instance construction (the builder drives fill-or-replace
        per base goal)."""
        return self._instantiate(spec, base, base_goal_id, inst_seq, base_seed)

    def propose(self, snapshot: Any = None, *, family: str, seed: int | None = None,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        spec = BY_ID[family]
        base_seed = self.seed if seed is None else seed
        max_goals = (constraints or {}).get("max_base_goals", spec.base_goals)
        overdraw = (constraints or {}).get("instance_overdraw", 1)
        for base_goal_id, base in self.iter_base_goals(
                family, max_goals=max_goals, seed=base_seed):
            for inst_seq in range(spec.instances_per_goal * overdraw):
                cand = self._instantiate(spec, base, base_goal_id, inst_seq, base_seed)
                if cand is not None:
                    yield cand

    # ------------------------------------------------------- base-goal draws

    def _propose_base(self, spec: FamilySpec, rng: Random) -> dict[str, Any] | None:
        fid = spec.family_id
        p = self.pools
        if fid == "l06.isa_ancestors_exact":
            t = rng.choice(p["anc_seeds"])
            return {"kind": "isa_ancestors", "goal_params": {"seed": t}}
        if fid == "l06.isa_descendants_exact":
            t = rng.choice(p["desc_seeds"])
            return {"kind": "isa_descendants", "goal_params": {"seed": t}}
        if fid == "l06.isa_common_ancestors":
            return self._draw_pair(rng, "common_ancestors")
        if fid == "l06.isa_lowest_common_ancestor":
            return self._draw_pair(rng, "lca")
        if fid == "l06.partof_ancestors_closure":
            t = rng.choice(p["partof_seeds"])
            return {"kind": "partof_ancestors", "goal_params": {"seed": t}}
        if fid == "l06.regulates_closure":
            if not p["reg_direct_seeds"]:
                return None
            t = rng.choice(p["reg_direct_seeds"])
            return {"kind": "regulators", "goal_params": {"seed": t}}
        if fid == "l06.cross_isa_partof_intersection":
            return self._draw_cross(rng)
        if fid == "l06.branch_difference":
            return self._draw_branch_pair(rng)
        if fid == "l06.depth_band_terms":
            lo = rng.choice([1, 2, 3])
            hi = lo + rng.choice([1, 2])
            root = rng.choice(p["root_pool"])
            return {"kind": "depth_band",
                    "goal_params": {"lo": lo, "hi": hi, "root": root}}
        if fid == "l06.multi_parent_terms":
            root = rng.choice(p["root_pool"])
            return {"kind": "multi_parent", "goal_params": {"root": root}}
        if fid == "l06.repair_isa_parent_swap":
            return self._draw_repair_edge(rng, "swap")
        if fid == "l06.repair_missing_isa_edge":
            return self._draw_repair_edge(rng, "missing")
        if fid == "l06.repair_relation_type_confusion":
            return self._draw_repair_edge(rng, "reltype")
        if fid == "l06.repair_orphan_reconnect":
            return self._draw_repair_edge(rng, "orphan")
        if fid == "l06.org_align_gene_aspect_annotations":
            return self._draw_align(rng)
        if fid == "l06.org_annotation_scope_closure":
            return self._draw_annotation_closure(rng)
        if fid == "l06.org_shared_ancestors_multigene":
            return self._draw_shared(rng)
        raise ValueError(f"unknown family {fid}")

    def _draw_pair(self, rng: Random, kind: str) -> dict[str, Any] | None:
        """Cousins under a mid-level root: pick M (pair_roots), two distinct
        children of M, and one grandchild under each. The pair is guaranteed
        close relatives, so a common-ancestor fragment root exists."""
        if not self.pools["pair_roots"]:
            return None
        M = rng.choice(self.pools["pair_roots"])
        children = sorted(self.graph["isa_children"].get(M, []))
        if len(children) < 2:
            return None
        c1, c2 = rng.sample(children, 2)

        def pick(c: str) -> str:
            gs = sorted(self.graph["isa_children"].get(c, []))
            return rng.choice(gs) if gs else c

        a, b = pick(c1), pick(c2)
        if a == b:
            return None
        shared = set(self.closures["isa_anc"].get(a, [])) & set(
            self.closures["isa_anc"].get(b, []))
        if len(shared) < 2:
            return None
        if kind == "lca":
            lca = lowest_common_ancestors(frozenset(self.closures["isa_anc"][a]),
                                          frozenset(self.closures["isa_anc"][b]),
                                          self.graph["isa_parents"])
            if not lca:
                return None
        return {"kind": kind, "goal_params": {"a": a, "b": b}}

    def _draw_cross(self, rng: Random) -> dict[str, Any] | None:
        """isa_root = an is_a ancestor of b whose subtree contains >= 2 of b's
        part_of ancestors — the intersection the task asks for is guaranteed
        non-degenerate, and the fragment (cone of isa_root + b's partof
        context) contains both query anchors."""
        desc_count = self.pools["desc_count"]
        isa_anc = self.closures["isa_anc"]
        b = rng.choice(self.pools["partof_seeds"])
        po = set(self.closures["partof_anc"].get(b, []))
        cands = []
        for a in isa_anc.get(b, []):
            if not (150 <= desc_count.get(a, 0) <= 2000):
                continue
            inter = sum(1 for x in po if a in isa_anc.get(x, []))
            if inter >= 2:
                cands.append(a)
        if not cands:
            return None
        a = rng.choice(sorted(cands))
        return {"kind": "cross_isa_partof",
                "goal_params": {"isa_root": a, "partof_target": b}}

    def _draw_branch_pair(self, rng: Random) -> dict[str, Any] | None:
        desc_count = self.pools["desc_count"]
        b = rng.choice(self.pools["desc_seeds"])
        parents = self.graph["isa_parents"].get(b, [])
        if not parents:
            return None
        anc_cands = [a for a in self.closures["isa_anc"].get(b, [])
                     if 40 <= desc_count.get(a, 0) <= 1400
                     and desc_count.get(a, 0) >= desc_count.get(b, 0) + 6]
        if not anc_cands:
            return None
        a = rng.choice(sorted(anc_cands))
        return {"kind": "branch_difference", "goal_params": {"parent": a, "child": b}}

    def _draw_repair_edge(self, rng: Random, mode: str) -> dict[str, Any] | None:
        live = self.pools["live"]
        isa_parents = self.graph["isa_parents"]
        rel_out = self.graph["rel_out"]
        for _ in range(30):
            t = rng.choice(live)
            if self.depth_global.get(t, 0) < 3:
                continue
            if mode in ("swap", "missing", "orphan"):
                ps = isa_parents.get(t, [])
                if mode == "swap" and len(ps) >= 1:
                    p1 = rng.choice(sorted(ps))
                    if not self._viable_repair_root(t, (p1,)):
                        continue
                    return {"kind": "repair_isa_parent_swap",
                            "goal_params": {"term": t, "correct_parent": p1}}
                if mode == "missing" and len(ps) == 1:
                    if not self._viable_repair_root(t, (ps[0],)):
                        continue
                    return {"kind": "repair_missing_isa_edge",
                            "goal_params": {"term": t, "removed_parent": ps[0]}}
                if mode == "orphan" and 1 <= len(ps) <= 3:
                    if not self._viable_repair_root(t, tuple(sorted(ps))):
                        continue
                    return {"kind": "repair_orphan_reconnect",
                            "goal_params": {"term": t, "removed_parents": sorted(ps)}}
            else:
                by = rel_out.get(t, {})
                options: list[tuple[str, str, str]] = []
                for o in by.get("part_of", []):
                    options.append((t, "part_of", o))
                for o in by.get("regulates", []):
                    options.append((t, "regulates", o))
                viable = [opt for opt in options
                          if self._viable_confusion_root(*opt)]
                if viable:
                    s, pred, o = rng.choice(sorted(viable))
                    wrong = "regulates" if pred == "part_of" else "part_of"
                    return {"kind": "repair_relation_type_confusion",
                            "goal_params": {"subject": s, "object": o,
                                            "correct_pred": pred, "wrong_pred": wrong}}
        return None

    def _viable_confusion_root(self, s: str, _pred: str, o: str) -> bool:
        """A fragment root exists that contains both edge endpoints: a common
        is_a ancestor of s and o with a moderate descendant count."""
        isa_anc = self.closures["isa_anc"]
        desc_count = self.pools["desc_count"]
        anc_s = set(isa_anc.get(s, [])) | {s}
        anc_o = set(isa_anc.get(o, [])) | {o}
        return any(150 <= desc_count.get(a, 0) <= 6000 for a in anc_s & anc_o)

    def _viable_repair_root(self, term: str, anchors: tuple[str, ...]) -> bool:
        """A fragment root exists that contains the term and every anchor: an
        is_a ancestor of term that is an ancestor-or-self of each anchor, with
        a moderate descendant count."""
        isa_anc = self.closures["isa_anc"]
        desc_count = self.pools["desc_count"]
        for a in isa_anc.get(term, []):
            if 40 <= desc_count.get(a, 0) <= 6000 and all(
                    a == r or a in isa_anc.get(r, []) for r in anchors):
                return True
        return False

    def _draw_align(self, rng: Random) -> dict[str, Any] | None:
        genes = sorted(self.pools["genes_bp"])
        if len(genes) < 4:
            return None
        for _ in range(40):
            pick = rng.sample(genes, 4)
            sets = [frozenset(self.pools["genes_bp"][g]) for g in pick]
            if any(len(s) < 3 for s in sets) or len(set(sets)) != 4:
                continue
            return {"kind": "align_gene_terms",
                    "goal_params": {"genes": sorted(pick), "aspect": "P"}}
        return None

    def _draw_annotation_closure(self, rng: Random) -> dict[str, Any] | None:
        isa_anc = self.closures["isa_anc"]
        partof_anc = self.closures["partof_anc"]
        genes = sorted(self.pools["genes_bp"])
        rng.shuffle(genes)
        for g in genes[:200]:
            direct = self.pools["genes_bp"][g]
            closure: set[str] = set(direct)
            for t in direct:
                closure.update(isa_anc.get(t, []))
                closure.update(partof_anc.get(t, []))
        if 3 <= len(direct) and 10 <= len(closure) <= 300:
            return {"kind": "annotation_closure",
                    "goal_params": {"gene": g, "direct_terms": sorted(direct)}}
        return None

    def _draw_shared(self, rng: Random) -> dict[str, Any] | None:
        isa_anc = self.closures["isa_anc"]
        partof_anc = self.closures["partof_anc"]
        genes = sorted(g for g, terms in self.pools["genes_bp"].items()
                       if len(terms) >= 2)
        rng.shuffle(genes)
        roots = {"GO:0008150", "GO:0003674", "GO:0005575"}
        for i in range(0, min(600, len(genes) - 2), 3):
            trio = genes[i:i + 3]
            shared: set[str] | None = None
            union: set[str] = set()
            for g in trio:
                prop: set[str] = set()
                for t in self.pools["genes_bp"][g]:
                    prop.add(t)
                    prop.update(isa_anc.get(t, []))
                    prop.update(partof_anc.get(t, []))
                union |= prop
                shared = prop if shared is None else shared & prop
            shared = (shared or set()) - roots
            if shared and len(shared) >= 2 and len(union) <= 400:
                return {"kind": "shared_ancestors",
                        "goal_params": {"genes": sorted(trio)}}
        return None

    # ------------------------------------------------------------ instantiate

    def _instantiate(self, spec: FamilySpec, base: dict[str, Any], base_goal_id: str,
                     inst_seq: int, base_seed: int) -> CandidateTaskAST | None:
        rng = Random(f"{spec.family_id}|inst|{base_goal_id}|{inst_seq}|{base_seed}")
        cap = FRAGMENT_CAP[spec.complexity]
        kind = base["kind"]
        gp = dict(base["goal_params"])
        if kind in ("annotation_closure", "shared_ancestors"):
            universe = self._universe_from_annotations(kind, gp, idx=inst_seq)
            if universe is None:
                return None
        elif kind in ("align_gene_terms",):
            universe = sorted({t for g in gp["genes"]
                               for t in self.pools["genes_bp"][g]})
        elif kind in ("depth_band", "multi_parent"):
            universe = self._rooted_universe(gp["root"], cap, idx=inst_seq)
            if universe is None:
                return None
        elif kind in ("cross_isa_partof",):
            universe = self._cross_universe(gp["isa_root"], gp["partof_target"],
                                            cap, idx=inst_seq)
            if universe is None:
                return None
        elif kind == "partof_ancestors":
            universe = self._partof_universe(gp["seed"], cap, idx=inst_seq)
            if universe is None:
                return None
        elif kind == "regulators":
            universe = self._regulator_universe(gp["seed"], cap, idx=inst_seq)
            if universe is None:
                return None
        elif kind == "branch_difference":
            universe = self._fragment_for_seed(gp["child"], cap,
                                               required=(gp["parent"],), idx=inst_seq)
            if universe is None:
                return None
        elif kind.startswith("repair_"):
            focus = gp.get("term") or gp.get("subject")
            if kind == "repair_isa_parent_swap":
                required = (gp["correct_parent"],)
            elif kind == "repair_missing_isa_edge":
                required = (gp["removed_parent"],)
            elif kind == "repair_orphan_reconnect":
                required = tuple(gp["removed_parents"])
            else:
                required = (gp["object"],)
            universe = self._fragment_for_seed(focus, cap, required=required,
                                               idx=inst_seq)
            if universe is None:
                return None
        else:
            seed_term = gp.get("seed") or gp.get("a")
            required = (gp["b"],) if kind in ("common_ancestors", "lca") else ()
            universe = self._fragment_for_seed(seed_term, cap, required=required,
                                               idx=inst_seq)
            if universe is None:
                return None
        ast: dict[str, Any] = {"kind": kind, "universe_ids": sorted(universe),
                               "complexity": spec.complexity}
        ast.update(gp)
        if kind == "align_gene_terms":
            ast["derangement"] = _derangement(rng, sorted(gp["genes"]))
        if kind == "repair_isa_parent_swap":
            wrong = self._pick_wrong_parent(rng, ast)
            if wrong is None:
                return None
            ast["wrong_parent"] = wrong
        if kind == "shared_ancestors":
            ast["direct_terms_by_gene"] = {
                g: sorted(self.pools["genes_bp"][g]) for g in gp["genes"]}
        instance_id = "in-" + canonical_sha(
            {"family": spec.family_id, "base": base_goal_id, "ast": ast})[:20]
        src_units = [f"goterm:{t}" for t in ast["universe_ids"][:8]]
        if kind in ("align_gene_terms", "annotation_closure", "shared_ancestors"):
            src_units.append("gaf:goa_human.v238")
        return CandidateTaskAST(
            lane_id=LANE, family_id=spec.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=spec.regime, semantic_profile=PROFILE,
            ast=ast, source_unit_ids=tuple(src_units))

    def _universe_from_annotations(self, kind: str, gp: dict[str, Any], *,
                                   idx: int = 0) -> list[str] | None:
        """Universe = propagated closure of the genes' direct annotations plus a
        per-instance distractor branch (an unrelated GO subtree). The answer is
        unchanged by distractors — they only vary the fragment context, so
        instances of one base goal differ while the gold answer stays fixed."""
        isa_anc = self.closures["isa_anc"]
        partof_anc = self.closures["partof_anc"]
        genes = [gp["gene"]] if kind == "annotation_closure" else gp["genes"]
        terms: set[str] = set()
        for g in genes:
            for t in self.pools["genes_bp"].get(g, []):
                terms.add(t)
                terms.update(isa_anc.get(t, []))
                terms.update(partof_anc.get(t, []))
        if len(terms) > 400:
            return None
        want = max(30, len(terms) // 2)
        if len(terms) + want > 500:
            want = 500 - len(terms)
        if want < 10:
            return None
        distractors = self._distractor_branch(idx, frozenset(terms), want)
        if not distractors:
            return None
        return sorted(terms | distractors)

    def _partof_universe(self, seed: str, cap: int, *, idx: int = 0) -> list[str] | None:
        """Universe = isa-cone of an ancestor of seed UNION the global part_of
        closure of seed. part_of targets live outside the isa-cone (cross
        branch), so the closure context is added explicitly; the fragment-local
        partof answer then equals the global closure restricted to the union."""
        po = set(self.closures["partof_anc"].get(seed, []))
        if len(po) < 2:
            return None
        base = self._fragment_for_seed(seed, cap, idx=idx)
        if base is None:
            return None
        return sorted(set(base) | po)

    def _cross_universe(self, isa_root: str, b: str, cap: int,
                        *, idx: int = 0) -> list[str] | None:
        """Universe = isa-cone rooted at (or above) isa_root containing b,
        UNION b's global part_of closure and b's full is_a ancestor chain —
        partof paths from b jump to targets whose isa prefixes reach above the
        cone root, so the ancestor chain keeps every path intact inside the
        fragment."""
        po = set(self.closures["partof_anc"].get(b, []))
        base = self._fragment_for_seed(b, cap, required=(isa_root,), idx=idx)
        if base is None:
            return None
        return sorted(set(base) | po | set(self.closures["isa_anc"].get(b, [])))

    def _regulator_universe(self, seed: str, cap: int, *, idx: int = 0) -> list[str] | None:
        """Universe = isa-cone of an ancestor of seed (contains seed's subtree,
        the R7 regulation targets) UNION each direct regulator with a small
        descendant cone (the R6 subjects). Regulators live in the mirror
        'regulation of ...' branch, never inside the seed's isa-cone, so they
        are added explicitly."""
        direct = sorted(self.pools["direct_regs"].get(seed, ()))
        if not direct:
            return None
        base = self._fragment_for_seed(seed, cap, idx=idx)
        if base is None:
            return None
        ctx = set(base)
        for r in direct[:4]:
            ctx |= set(self.fragment_terms(r, 12))
        return sorted(ctx)

    def _distractor_branch(self, idx: int, exclude: frozenset[str],
                           want: int) -> set[str] | None:
        """A coherent unrelated subtree (cone of a small-branch root) of about
        ``want`` terms, disjoint from ``exclude``; ``idx`` rotates the root."""
        pool = self.pools["distractor_roots"]
        out: set[str] = set()
        for k in range(len(pool)):
            root = pool[(idx + k) % len(pool)]
            if root in exclude:
                continue
            branch = set(self.fragment_terms(root, max(10, want // 4)))
            if branch & exclude:
                continue
            out |= branch
            if len(out) >= want:
                break
        return out or None

    def _pick_wrong_parent(self, rng: Random, ast: dict[str, Any]) -> str | None:
        """A wrong parent: same namespace, in fragment, not an is_a
        ancestor/descendant of the term, not already its parent."""
        t = ast["term"]
        u = set(ast["universe_ids"])
        ns = self.terms[t]["namespace"]
        anc = set(self.closures["isa_anc"].get(t, []))
        parents = set(self.graph["isa_parents"].get(t, []))
        cands = [x for x in sorted(u)
                 if self.terms[x]["namespace"] == ns and x != t
                 and x not in anc and x not in parents
                 and t not in set(self.closures["isa_anc"].get(x, []))]
        if not cands:
            return None
        return rng.choice(cands)

    # ---------------------------------------------------------------- ground

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        universe = ast["universe_ids"]
        frag = induce(universe, self.graph)
        edges = [{"s": s, "p": p, "o": o}
                 for s, p, o in self._public_edges(candidate, frag)]
        public: dict[str, Any] = {
            "user_goal": self._user_goal(candidate),
            "fragment": {
                "terms": [{"go_id": t, "name": self.terms[t]["name"],
                           "namespace": self.terms[t]["namespace"]}
                          for t in universe],
                "edges": edges,
                "profile": PROFILE,
                "source_release": RELEASE_ID,
            },
            "allowed_tools": ["go_fragment.read", "go_closure.compute",
                              "gaf_annotations.read"],
            "budget": {"probes": 6, "actions": 14},
            "semantic_profile": PROFILE,
            "complexity": ast["complexity"],
        }
        kind = ast["kind"]
        if kind == "align_gene_terms":
            correct = {g: sorted(self.pools["genes_bp"][g]) for g in ast["genes"]}
            public["misaligned_table"] = {
                g: correct[wrong] for g, wrong in ast["derangement"].items()}
            public["aspect"] = ast["aspect"]
        if kind in ("annotation_closure",):
            public["gene"] = ast["gene"]
            public["direct_terms"] = ast["direct_terms"]
        if kind == "shared_ancestors":
            public["genes"] = ast["genes"]
            public["direct_terms_by_gene"] = {
                g: sorted(self.pools["genes_bp"][g]) for g in ast["genes"]}
        oracle_ref = "l06-oracle-" + _sha(candidate.instance_id + "|oracle")[:16]
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                            native_evidence={"profile": PROFILE,
                                             "release_id": RELEASE_ID})

    def _public_edges(self, candidate: CandidateTaskAST,
                      frag: dict[str, Any]) -> list[tuple[str, str, str]]:
        """Edges shown to the agent; tampered for repair families."""
        ast = candidate.ast
        kind = ast["kind"]
        ip = {s: list(ps) for s, ps in frag["isa_parents"].items()}
        ro = {s: {p: list(ts) for p, ts in by.items()}
              for s, by in frag["rel_out"].items()}
        if kind == "repair_isa_parent_swap":
            t = ast["term"]
            ip[t] = [p for p in ip.get(t, []) if p != ast["correct_parent"]]
            ip[t].append(ast["wrong_parent"])
            ip[t].sort()
        elif kind == "repair_missing_isa_edge":
            t = ast["term"]
            ip[t] = [p for p in ip.get(t, []) if p != ast["removed_parent"]]
        elif kind == "repair_orphan_reconnect":
            ip[ast["term"]] = []
        elif kind == "repair_relation_type_confusion":
            s, o = ast["subject"], ast["object"]
            correct, wrong = ast["correct_pred"], ast["wrong_pred"]
            if s in ro and correct in ro.get(s, {}):
                ro[s][correct] = [x for x in ro[s][correct] if x != o]
                if not ro[s][correct]:
                    del ro[s][correct]
                ro.setdefault(s, {}).setdefault(wrong, [])
                if o not in ro[s][wrong]:
                    ro[s][wrong].append(o)
                    ro[s][wrong].sort()
        edges = [(s, "is_a", o) for s, ps in ip.items() for o in ps]
        for s, by in ro.items():
            for pred, targets in by.items():
                edges.extend((s, pred, o) for o in targets)
        return sorted(set(edges))

    # ---------------------------------------------------------------- wording

    def _user_goal(self, candidate: CandidateTaskAST) -> str:
        ast = candidate.ast
        kind = ast["kind"]
        name = lambda t: f"{self.terms[t]['name']} ({t})"  # noqa: E731
        if kind == "isa_ancestors":
            return ("In the provided Gene Ontology fragment, return the exact set of all "
                    f"GO terms that are is_a ancestors of {name(ast['seed'])} — every term "
                    "reachable by one or more is_a steps. Submit the complete set as "
                    "answer_ids; partial or superset answers are scored as failure.")
        if kind == "isa_descendants":
            return ("In the provided Gene Ontology fragment, return the exact set of all "
                    f"GO terms that are is_a descendants of {name(ast['seed'])} — every "
                    "term that has this term as an is_a ancestor. Submit as answer_ids.")
        if kind == "common_ancestors":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    f"terms that are is_a ancestors of BOTH {name(ast['a'])} AND "
                    f"{name(ast['b'])}. Submit as answer_ids.")
        if kind == "lca":
            return ("In the provided Gene Ontology fragment, return the exact set of "
                    f"LOWEST common is_a ancestors of {name(ast['a'])} and "
                    f"{name(ast['b'])} — shared ancestors that have no descendant among "
                    "the shared ancestors. Submit as answer_ids.")
        if kind == "partof_ancestors":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    f"terms that {name(ast['seed'])} is part of, under the declared "
                    "profile go-rel-closure/1: part_of chains propagate over is_a on "
                    "both sides and part_of is transitive. Submit as answer_ids.")
        if kind == "regulators":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    f"terms that regulate {name(ast['seed'])} under go-rel-closure/1: "
                    "regulates/positively_regulates/negatively_regulates all count, "
                    "regulation propagates over is_a on both sides, and regulates is "
                    "NOT transitive. Submit as answer_ids.")
        if kind == "cross_isa_partof":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    f"terms that are BOTH is_a descendants of {name(ast['isa_root'])} "
                    f"AND stand in the part_of closure to {name(ast['partof_target'])} "
                    "(profile go-rel-closure/1). Submit as answer_ids.")
        if kind == "branch_difference":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    f"terms that are is_a descendants of {name(ast['parent'])} but NOT "
                    f"is_a descendants of {name(ast['child'])}. Submit as answer_ids.")
        if kind == "depth_band":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    f"terms whose fragment-local is_a depth (longest is_a path to a "
                    f"fragment root) lies between {ast['lo']} and {ast['hi']} inclusive. "
                    "Submit as answer_ids.")
        if kind == "multi_parent":
            return ("In the provided Gene Ontology fragment, return the exact set of GO "
                    "terms that have two or more distinct is_a parents inside the "
                    "fragment. Submit as answer_ids.")
        if kind == "repair_isa_parent_swap":
            return ("The provided Gene Ontology fragment was corrupted: the term "
                    f"{name(ast['term'])} had one is_a parent edge replaced by a wrong "
                    "parent, so its inferred ancestor closure no longer matches the "
                    "reference Gene Ontology release. Identify the correct is_a parent "
                    "and submit it as {'term': <term>, 'parent': <correct_parent>}.")
        if kind == "repair_missing_isa_edge":
            return ("The provided Gene Ontology fragment was corrupted: exactly one is_a "
                    f"parent edge of {name(ast['term'])} was deleted, so part of its "
                    "inferred ancestry is missing. Identify the deleted edge and submit "
                    "it as {'term': <term>, 'parent': <missing_parent>}.")
        if kind == "repair_relation_type_confusion":
            return ("The provided Gene Ontology fragment was corrupted: the relation "
                    f"between {name(ast['subject'])} and {name(ast['object'])} was "
                    "relabeled with the wrong predicate (part_of vs regulates). Restore "
                    "the correct predicate and submit {'subject': ..., 'object': ..., "
                    "'predicate': ...}.")
        if kind == "repair_orphan_reconnect":
            return ("The provided Gene Ontology fragment was corrupted: ALL is_a parent "
                    f"edges of {name(ast['term'])} were removed, orphaning it from the "
                    "hierarchy. Restore the complete correct parent set and submit "
                    "{'term': <term>, 'parents': [...]}.")
        if kind == "align_gene_terms":
            return ("The provided table assigns each gene its direct GO "
                    "biological-process annotation term set (from the pinned GOA human "
                    "GAF), but every row was attached to the wrong gene. Using the GO "
                    "fragment and the annotation evidence, re-align every gene to its "
                    "correct term set and submit the mapping as "
                    "{'alignment': {gene: [go_ids]}}.")
        if kind == "annotation_closure":
            return (f"The gene {ast['gene']} has the provided direct GO "
                    "biological-process annotations (pinned GOA human GAF). Return the "
                    "exact FULL annotation set implied by the Gene Ontology fragment "
                    "under go-rel-closure/1: the direct terms plus every is_a and "
                    "part_of propagated ancestor. Submit as answer_ids.")
        if kind == "shared_ancestors":
            genes = ", ".join(ast["genes"])
            return ("The provided genes each carry direct GO biological-process "
                    "annotations (pinned GOA human GAF). Return the exact set of "
                    f"biological-process terms — excluding the namespace root — that "
                    f"annotate ALL of {{{genes}}} directly or by is_a/part_of "
                    "propagation (profile go-rel-closure/1). Submit as answer_ids.")
        raise ValueError(f"no wording for kind {kind!r}")

    # convenience for the verifier: reference alignment for org tasks
    def reference_alignment(self, genes: list[str]) -> dict[str, list[str]]:
        return {g: sorted(self.pools["genes_bp"][g]) for g in genes}
