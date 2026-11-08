"""L06 20K-expansion task-family proposer: new mechanisms over the pinned
GO + GOA index, seeded with the 20K seed base 20261101.

Pipeline position: pinned source index + GAF evidence index (evidence20k)
-> propose (this file) -> ground -> native verifier gates (verify20k)
-> shard writer (build20k).

Determinism: every draw is seeded ``Random(f"{family}|base|{seq}|{seed}")``
/ ``Random(f"{family}|inst|{base_goal_id}|{seq}|{seed}")`` with the stable
20K seed base 20261101 (distinct from the legacy 8K seed 20261027; the
collision check in build20k additionally proves no task_id / instance_id /
base_goal_id / public-blob overlap with the legacy 949 records).

The public view carries the fragment (term ids + names + namespaces +
induced edge list — *tampered* for the four new repair kinds) and the
question; it never carries the serialized answer. Organization kinds carry
only gene / term / candidate identifiers: the evidence split is always
recomputed from the pinned GAF by the verifier, never stored in the AST.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from random import Random
from typing import Any, Iterator

from autoontobench.contracts.models import canonical_sha
from benchmark.expansion.l06.evidence20k import load_evidence
from benchmark.expansion.l06.families20k import (
    FRAGMENT_CAP_20K,
    PROFILE,
    SEED_20K,
    FamilySpec20k,
)
from benchmark.plugins.source.l06.closure import REGULATES_KINDS
from benchmark.plugins.source.l06.plugin import L06SourcePlugin
from benchmark.plugins.verifier.l06.fragment import induce
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l06_go import RELEASE_ID

LANE = "L06"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


class L06ExpansionProposer:
    """TaskFamilyPlugin for the 15 20K-expansion L06 families."""

    def __init__(self, index_dir: str | Path, *, seed: int = SEED_20K,
                 index: dict[str, Any] | None = None,
                 evidence: dict[str, Any] | None = None) -> None:
        self.index = index if index is not None else L06SourcePlugin.load_index(
            Path(index_dir))
        self.index_dir = Path(index_dir)
        self.graph = self.index["graph"]
        self.closures = self.index["closures"]
        self.terms = self.index["terms"]
        self.depth_global = self.index["depth"]
        self.gene_terms = self.index["gene_terms"]
        self.evidence = evidence if evidence is not None else load_evidence(
            self.index_dir, self.closures)
        self.seed = seed
        self._pools: dict[str, Any] | None = None

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    # ------------------------------------------------------------------ pools

    def _build_pools(self) -> dict[str, Any]:
        isa_anc = self.closures["isa_anc"]
        partof_anc = self.closures["partof_anc"]
        reg_anc = self.closures["reg_anc"]
        reg_pos_anc = self.closures["reg_pos_anc"]
        reg_neg_anc = self.closures["reg_neg_anc"]
        desc_count: dict[str, int] = {}
        for t, anc in isa_anc.items():
            for a in anc:
                desc_count[a] = desc_count.get(a, 0) + 1
        live = sorted(t for t, m in self.terms.items() if not m["obsolete"])
        by_ns: dict[str, list[str]] = {}
        for t in live:
            by_ns.setdefault(self.terms[t]["namespace"], []).append(t)
        rel_out = self.graph["rel_out"]
        rel_in: dict[str, dict[str, set[str]]] = {}
        for s, by_pred in rel_out.items():
            for k, targets in by_pred.items():
                for o in targets:
                    rel_in.setdefault(k, {}).setdefault(o, set()).add(s)
        reg_subjects = {s for s, by in rel_out.items()
                        if any(by.get(k) for k in REGULATES_KINDS)}
        partof_targets = {s: by.get("part_of", [])
                          for s, by in rel_out.items()}
        # partonomy descendants of t (terms whose part_of closure contains t)
        partof_desc: dict[str, set[str]] = {}
        for x, anc in partof_anc.items():
            for a in anc:
                partof_desc.setdefault(a, set()).add(x)
        gene_ev = self.evidence["gene_ev"]
        term_genes = self.evidence["term_genes"]
        term_genes_exp = self.evidence["term_genes_exp"]
        isa_desc: dict[str, set[str]] = {}
        for t, anc in isa_anc.items():
            for a in anc:
                isa_desc.setdefault(a, set()).add(t)

        def propagated(terms: set[str]) -> set[str]:
            out = set(terms)
            for t in terms:
                out.update(isa_anc.get(t, []))
                out.update(partof_anc.get(t, []))
            return out

        exp_genes = sorted(g for g, by in gene_ev.items()
                           if len(by["exp"]) >= 3)
        disc_genes: list[str] = []
        for g, by in gene_ev.items():
            if len(by["exp"]) < 2 or len(by["other"]) < 2:
                continue
            d = propagated(by["exp"]) - propagated(by["other"])
            if 2 <= len(d) <= 200:
                disc_genes.append(g)
        mid_terms = sorted(t for t, v in term_genes.items()
                           if 5 <= len(v) <= 60)
        mid_exp_terms = sorted(t for t, v in term_genes_exp.items()
                               if 3 <= len(v) <= 40)
        term_genes_prop = self.evidence["term_genes_prop"]
        term_genes_exp_prop = self.evidence["term_genes_exp_prop"]
        mid_prop_terms = sorted(t for t, v in term_genes_prop.items()
                                if 2 <= len(v) <= 60)
        mid_exp_prop_terms = sorted(t for t, v in term_genes_exp_prop.items()
                                    if 2 <= len(v) <= 40)
        # partof-rich cones for the partonomy-aware depth family
        partof_rich_roots: list[str] = []
        root_scan = [t for t in live
                     if 200 <= desc_count.get(t, 0) <= 4000
                     and self.depth_global.get(t, 0) >= 3]
        for t in root_scan[:3000]:
            cone = self.fragment_terms(t, 350)
            n_pe = sum(1 for s in cone if s in partof_targets
                       for o in partof_targets[s] if o in cone)
            if n_pe >= 5:
                partof_rich_roots.append(t)
        # roots whose subtree contains several regulation-edge subjects
        reg_subject_roots = sorted(
            t for t in live
            if 3 <= len(isa_desc.get(t, frozenset()) & reg_subjects) <= 150
            and 40 <= desc_count.get(t, 0) <= 4000
            and self.depth_global.get(t, 0) >= 2)
        # part_of / polarity edges with a viable common-ancestor fragment
        po_edges = sorted(
            (s, o) for s, by in rel_out.items() for o in by.get("part_of", [])
            if self._viable_confusion_root(s, o, desc_count))
        pol_edges = sorted(
            (s, k, o) for s, by in rel_out.items()
            for k in ("positively_regulates", "negatively_regulates")
            for o in by.get(k, [])
            if self._viable_confusion_root(s, o, desc_count))
        pools = {
            "live": live, "by_ns": by_ns, "desc_count": desc_count,
            "isa_anc": isa_anc, "partof_anc": partof_anc,
            "reg_anc": reg_anc, "reg_pos_anc": reg_pos_anc,
            "reg_neg_anc": reg_neg_anc, "isa_desc": isa_desc,
            "rel_out": rel_out, "rel_in": rel_in,
            "reg_subjects": reg_subjects, "partof_targets": partof_targets,
            "partof_desc": partof_desc,
            "anc_seeds": [t for t in live
                          if 3 <= len(isa_anc.get(t, [])) <= 80
                          and self.depth_global.get(t, 0) >= 3],
            "desc_seeds": [t for t in live if 3 <= desc_count.get(t, 0) <= 400],
            "partof_seeds": [t for t in live
                             if len(partof_anc.get(t, [])) >= 3],
            "pair_roots": [t for t in live
                           if 150 <= desc_count.get(t, 0) <= 3000
                           and self.depth_global.get(t, 0) >= 3],
            "root_pool": [t for t in live
                          if 200 <= desc_count.get(t, 0) <= 4000
                          and self.depth_global.get(t, 0) >= 3],
            "distractor_roots": [t for t in live
                                 if 30 <= desc_count.get(t, 0) <= 400
                                 and self.depth_global.get(t, 0) >= 2],
            "pos_direct_seeds": sorted(
                t for t, rs in rel_in.get("positively_regulates", {}).items()
                if rs and t in live and self.depth_global.get(t, 0) >= 3),
            "neg_direct_seeds": sorted(
                t for t, rs in rel_in.get("negatively_regulates", {}).items()
                if rs and t in live and self.depth_global.get(t, 0) >= 3),
            "generic_direct_seeds": sorted(
                t for t, rs in rel_in.get("regulates", {}).items()
                if rs and t in live and self.depth_global.get(t, 0) >= 3),
            "partof_isa_seeds": sorted(
                t for t in live
                if len(set(partof_anc.get(t, []))
                       & set(isa_anc.get(t, []))) >= 2),
            "reg_subject_roots": reg_subject_roots,
            "partof_rich_roots": partof_rich_roots,
            "po_edges": po_edges,
            "pol_edges": pol_edges,
            "exp_genes": exp_genes,
            "disc_genes": sorted(disc_genes),
            "mid_terms": mid_terms,
            "mid_exp_terms": mid_exp_terms,
            "mid_prop_terms": mid_prop_terms,
            "mid_exp_prop_terms": mid_exp_prop_terms,
            "gene_ev": gene_ev,
            "term_genes": term_genes,
            "term_genes_exp": term_genes_exp,
            "term_genes_prop": term_genes_prop,
            "term_genes_exp_prop": term_genes_exp_prop,
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
        """Universe = descendant cone of an ancestor of seed_term (identical
        construction contract to the legacy proposer: root ordered by global
        descendant count, ``idx`` rotates, past-cycle bit-prunes sibling
        branches so instances of one base goal get distinct fragments)."""
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
        """Remove sub-cones of the root's child branches selected by the bits
        of ``prune`` (identical contract to the legacy proposer)."""
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

    def _rooted_universe(self, root: str, cap: int, *, idx: int = 0
                         ) -> list[str] | None:
        terms = self.fragment_terms(root, cap)
        if len(terms) < 40:
            return None
        pruned = self._prune_variant(root, root, (), terms, idx)
        if pruned is None or len(pruned) < 40:
            return None
        return pruned

    def _viable_confusion_root(self, s: str, o: str,
                               desc_count: dict[str, int] | None = None) -> bool:
        """A fragment root exists that contains both edge endpoints: a common
        is_a ancestor of s and o with a moderate descendant count (identical
        contract to the legacy proposer)."""
        if desc_count is None:
            desc_count = self.pools["desc_count"]
        isa_anc = self.closures["isa_anc"]
        anc_s = set(isa_anc.get(s, [])) | {s}
        anc_o = set(isa_anc.get(o, [])) | {o}
        return any(150 <= desc_count.get(a, 0) <= 6000 for a in anc_s & anc_o)

    # ------------------------------------------------- new universe builders

    def _polarity_reg_universe(self, seed: str, cap: int, kind_key: str,
                               *, idx: int = 0) -> list[str] | None:
        """isa-cone of an ancestor of seed UNION small cones of the seed's
        DIRECT regulators of the requested polarity kind (the mirror
        'regulation of ...' branch lives outside the seed's isa-cone)."""
        rel_in = self.pools["rel_in"]
        direct = sorted(rel_in.get(kind_key, {}).get(seed, ()))
        if not direct:
            return None
        base = self._fragment_for_seed(seed, cap, idx=idx)
        if base is None:
            return None
        ctx = set(base)
        for r in direct[:4]:
            ctx |= set(self.fragment_terms(r, 12))
        return sorted(ctx)

    def _dual_target_universe(self, a: str, b: str, cap: int,
                              *, idx: int = 0) -> list[str] | None:
        """Cone containing both targets UNION small cones of each target's
        direct regulators (any polarity)."""
        rel_in = self.pools["rel_in"]
        base = self._fragment_for_seed(a, cap, required=(b,), idx=idx)
        if base is None:
            return None
        ctx = set(base)
        for target in (a, b):
            direct = sorted(
                set().union(*(rel_in.get(k, {}).get(target, frozenset())
                              for k in REGULATES_KINDS)))
            for r in direct[:4]:
                ctx |= set(self.fragment_terms(r, 12))
        return sorted(ctx)

    def _partof_isa_universe(self, seed: str, cap: int, *,
                             idx: int = 0) -> list[str] | None:
        """Cone rooted at (or above) one of the seed's dual isa+partof
        ancestors UNION the seed's global part_of closure context."""
        isa_anc = self.closures["isa_anc"]
        partof_anc = self.closures["partof_anc"]
        dual = sorted(set(partof_anc.get(seed, []))
                      & set(isa_anc.get(seed, [])))
        if not dual:
            return None
        dual.sort(key=lambda t: (self.depth_global.get(t, 0), t))
        anchor = dual[idx % len(dual)]
        base = self._fragment_for_seed(seed, cap, required=(anchor,), idx=idx)
        if base is None:
            return None
        return sorted(set(base) | set(partof_anc.get(seed, [])))

    def _reg_subject_universe(self, root: str, cap: int, *,
                              idx: int = 0) -> list[str] | None:
        """Cone of the root UNION the objects of every regulation edge whose
        subject lies in the cone (the regulation targets live outside the
        root's isa-cone — a genuinely new fragment shape)."""
        rel_out = self.graph["rel_out"]
        terms = self.fragment_terms(root, cap)
        if len(terms) < 40:
            return None
        pruned = self._prune_variant(root, root, (), terms, idx)
        if pruned is None or len(pruned) < 40:
            return None
        cone = set(pruned)
        objects: set[str] = set()
        for s in cone:
            by = rel_out.get(s, {})
            for k in REGULATES_KINDS:
                for o in by.get(k, ()):
                    if o in self.terms and not self.terms[o]["obsolete"]:
                        objects.add(o)
        if not objects:
            return None
        return sorted(cone | objects)

    def _universe_from_evidence(self, gene: str, *, idx: int = 0,
                                include_other: bool = False
                                ) -> list[str] | None:
        """Universe = propagated closure of the gene's experimentally
        evidenced direct annotations (plus the non-experimental closure when
        ``include_other``) UNION a per-instance distractor branch. The answer
        is unchanged by distractors — they only vary the fragment context."""
        isa_anc = self.closures["isa_anc"]
        partof_anc = self.closures["partof_anc"]
        by = self.pools["gene_ev"].get(gene, {})
        terms: set[str] = set()
        for t in by.get("exp", ()):
            terms.add(t)
            terms.update(isa_anc.get(t, []))
            terms.update(partof_anc.get(t, []))
        if include_other:
            for t in by.get("other", ()):
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

    def _term_reverse_universe(self, term: str, cap: int, candidates: list[str],
                                *, idx: int = 0) -> list[str] | None:
        """Universe = cone of an ancestor of the term (contains the term's
        subtree plus sibling-branch context) UNION the term's full
        isa-descendant and partonomy-descendant sets (capped — the downward
        partonomy closure is a new fragment shape) UNION the direct
        annotation terms of the candidate genes that lie below the term.
        The candidate gene panel is the public puzzle input; the gene->term
        join is the task, so including candidate direct terms leaks nothing
        (the mapping lives in the pinned GAF)."""
        isa_desc = self.pools["isa_desc"]
        partof_desc = self.pools["partof_desc"]
        gene_ev = self.pools["gene_ev"]
        base = self._fragment_for_seed(term, cap, idx=idx)
        if base is None:
            return None
        below = set(base) | {term}
        below |= set(sorted(isa_desc.get(term, set()))[:150])
        below |= set(sorted(partof_desc.get(term, set()))[:150])
        for g in candidates:
            for t in gene_ev.get(g, {}).get("all", ()):
                if (t in isa_desc.get(term, set())
                        or t in partof_desc.get(term, set()) or t == term):
                    below.add(t)
        if len(below) > 500:
            return None
        return sorted(below)

    def _distractor_branch(self, idx: int, exclude: frozenset[str],
                           want: int) -> set[str] | None:
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

    # ---------------------------------------------------------------- propose

    def iter_base_goals(self, family: str, *, max_goals: int,
                        seed: int | None = None
                        ) -> Iterator[tuple[str, dict[str, Any]]]:
        spec = self._spec(family)
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
                {"lane": LANE, "family": family,
                 "params": base["goal_params"], "seed": base_seed})[:20]
            if base_goal_id in seen_goals:
                continue
            seen_goals.add(base_goal_id)
            yield base_goal_id, base

    def instantiate(self, spec: FamilySpec20k, base: dict[str, Any],
                    base_goal_id: str, inst_seq: int,
                    base_seed: int) -> CandidateTaskAST | None:
        return self._instantiate(spec, base, base_goal_id, inst_seq, base_seed)

    def propose(self, snapshot: Any = None, *, family: str,
                seed: int | None = None,
                constraints: dict[str, Any] | None = None
                ) -> Iterator[CandidateTaskAST]:
        from benchmark.expansion.l06.families20k import BY_ID_20K
        spec = BY_ID_20K[family]
        base_seed = self.seed if seed is None else seed
        max_goals = (constraints or {}).get(
            "max_base_goals", spec.base_goals)
        overdraw = (constraints or {}).get("instance_overdraw", 1)
        for base_goal_id, base in self.iter_base_goals(
                family, max_goals=max_goals, seed=base_seed):
            for inst_seq in range(spec.instances_per_goal * overdraw):
                cand = self._instantiate(spec, base, base_goal_id, inst_seq,
                                         base_seed)
                if cand is not None:
                    yield cand

    def _spec(self, family: str) -> FamilySpec20k:
        from benchmark.expansion.l06.families20k import BY_ID_20K
        return BY_ID_20K[family]

    # ------------------------------------------------------- base-goal draws

    def _propose_base(self, spec: FamilySpec20k,
                      rng: Random) -> dict[str, Any] | None:
        fid = spec.family_id
        p = self.pools
        if fid == "l06.posreg_closure":
            if not p["pos_direct_seeds"]:
                return None
            return {"kind": "posregulators",
                    "goal_params": {"seed": rng.choice(p["pos_direct_seeds"])}}
        if fid == "l06.negreg_closure":
            if not p["neg_direct_seeds"]:
                return None
            return {"kind": "negregulators",
                    "goal_params": {"seed": rng.choice(p["neg_direct_seeds"])}}
        if fid == "l06.reg_polarity_exclusive":
            if not p["generic_direct_seeds"]:
                return None
            return {"kind": "generic_only_regulators",
                    "goal_params": {"seed": rng.choice(p["generic_direct_seeds"])}}
        if fid == "l06.reg_dual_target":
            return self._draw_dual_pair(rng)
        if fid == "l06.partof_isa_redundant_ancestors":
            if not p["partof_isa_seeds"]:
                return None
            return {"kind": "partof_isa_overlap",
                    "goal_params": {"seed": rng.choice(p["partof_isa_seeds"])}}
        if fid == "l06.reg_subject_descendants":
            if not p["reg_subject_roots"]:
                return None
            return {"kind": "reg_subject_descendants",
                    "goal_params": {"root": rng.choice(p["reg_subject_roots"])}}
        if fid == "l06.partof_depth_band":
            if not p["partof_rich_roots"]:
                return None
            lo = rng.choice([1, 2, 3])
            hi = lo + rng.choice([1, 2])
            return {"kind": "partof_depth_band",
                    "goal_params": {"lo": lo, "hi": hi,
                                    "root": rng.choice(p["partof_rich_roots"])}}
        if fid == "l06.repair_partof_edge_deletion":
            if not p["po_edges"]:
                return None
            s, o = rng.choice(p["po_edges"])
            return {"kind": "repair_partof_edge_deletion",
                    "goal_params": {"subject": s, "object": o}}
        if fid == "l06.repair_spurious_isa_edge":
            return {"kind": "repair_spurious_isa_edge",
                    "goal_params": {"term": rng.choice(p["anc_seeds"])}}
        if fid == "l06.repair_regulation_polarity_swap":
            if not p["pol_edges"]:
                return None
            s, k, o = rng.choice(p["pol_edges"])
            wrong = ("negatively_regulates"
                     if k == "positively_regulates" else "positively_regulates")
            return {"kind": "repair_regulation_polarity_swap",
                    "goal_params": {"subject": s, "object": o,
                                    "correct_pred": k, "wrong_pred": wrong}}
        if fid == "l06.repair_spurious_partof_edge":
            return {"kind": "repair_spurious_partof_edge",
                    "goal_params": {"subject": rng.choice(p["partof_seeds"])}}
        if fid == "l06.org_exp_evidence_closure":
            if not p["exp_genes"]:
                return None
            return {"kind": "exp_annotation_closure",
                    "goal_params": {"gene": rng.choice(p["exp_genes"])}}
        if fid == "l06.org_evidence_discriminated_ancestors":
            if not p["disc_genes"]:
                return None
            return {"kind": "evidence_discriminated",
                    "goal_params": {"gene": rng.choice(p["disc_genes"])}}
        if fid == "l06.org_term_annotated_genes":
            if not p["mid_prop_terms"]:
                return None
            return {"kind": "term_annotated_genes",
                    "goal_params": {"term": rng.choice(p["mid_prop_terms"])}}
        if fid == "l06.org_exp_genes_for_term":
            if not p["mid_exp_prop_terms"]:
                return None
            return {"kind": "exp_term_genes",
                    "goal_params": {"term": rng.choice(p["mid_exp_prop_terms"])}}
        raise ValueError(f"unknown family {fid}")

    def _draw_dual_pair(self, rng: Random) -> dict[str, Any] | None:
        """Cousins under a mid-level root whose regulator sets intersect in
        at least 2 terms (global screening; the fragment-local answer is
        gated by the verifier)."""
        p = self.pools
        reg_anc = p["reg_anc"]
        if not p["pair_roots"]:
            return None
        for _ in range(30):
            m = rng.choice(p["pair_roots"])
            children = sorted(self.graph["isa_children"].get(m, []))
            if len(children) < 2:
                continue
            c1, c2 = rng.sample(children, 2)

            def pick(c: str) -> str:
                gs = sorted(self.graph["isa_children"].get(c, []))
                return rng.choice(gs) if gs else c

            a, b = pick(c1), pick(c2)
            if a == b:
                continue
            shared = set(reg_anc.get(a, [])) & set(reg_anc.get(b, []))
            if len(shared) < 2:
                continue
            return {"kind": "dual_regulators",
                    "goal_params": {"a": a, "b": b}}
        return None

    # ------------------------------------------------------------ instantiate

    def _instantiate(self, spec: FamilySpec20k, base: dict[str, Any],
                     base_goal_id: str, inst_seq: int,
                     base_seed: int) -> CandidateTaskAST | None:
        rng = Random(f"{spec.family_id}|inst|{base_goal_id}|{inst_seq}|{base_seed}")
        cap = FRAGMENT_CAP_20K[spec.complexity]
        kind = base["kind"]
        gp = dict(base["goal_params"])
        universe: list[str] | None
        if kind in ("posregulators", "negregulators"):
            key = ("positively_regulates" if kind == "posregulators"
                   else "negatively_regulates")
            universe = self._polarity_reg_universe(gp["seed"], cap, key,
                                                   idx=inst_seq)
        elif kind == "generic_only_regulators":
            universe = self._polarity_reg_universe(
                gp["seed"], cap, "regulates", idx=inst_seq)
        elif kind == "dual_regulators":
            universe = self._dual_target_universe(gp["a"], gp["b"], cap,
                                                   idx=inst_seq)
        elif kind == "partof_isa_overlap":
            universe = self._partof_isa_universe(gp["seed"], cap, idx=inst_seq)
        elif kind == "reg_subject_descendants":
            universe = self._reg_subject_universe(gp["root"], cap,
                                                  idx=inst_seq)
        elif kind == "partof_depth_band":
            universe = self._rooted_universe(gp["root"], cap, idx=inst_seq)
        elif kind == "repair_partof_edge_deletion":
            universe = self._fragment_for_seed(gp["subject"], cap,
                                               required=(gp["object"],),
                                               idx=inst_seq)
        elif kind == "repair_spurious_isa_edge":
            universe = self._fragment_for_seed(gp["term"], cap, idx=inst_seq)
            if universe is not None:
                wrong = self._pick_spurious_parent(rng, gp["term"], universe)
                if wrong is None:
                    universe = None
                else:
                    gp["spurious_parent"] = wrong
        elif kind == "repair_regulation_polarity_swap":
            universe = self._fragment_for_seed(gp["subject"], cap,
                                               required=(gp["object"],),
                                               idx=inst_seq)
        elif kind == "repair_spurious_partof_edge":
            # part_of targets of the subject live outside its isa-cone
            # (cross-branch), so the universe is the cone UNION the subject's
            # global part_of closure context (mirrors the legacy partof
            # universe construction) — the true partof edges are then inside
            # the fragment, giving both the signature diff and a meaningful
            # wrong-fix candidate.
            base = self._fragment_for_seed(gp["subject"], cap, idx=inst_seq)
            if base is None:
                universe = None
            else:
                universe = sorted(
                    set(base)
                    | set(self.closures["partof_anc"].get(gp["subject"], [])))
                if len(universe) > 500:
                    universe = None
            if universe is not None:
                wrong = self._pick_spurious_partof_target(
                    rng, gp["subject"], universe)
                if wrong is None:
                    universe = None
                else:
                    gp["spurious_target"] = wrong
        elif kind == "exp_annotation_closure":
            universe = self._universe_from_evidence(gp["gene"], idx=inst_seq)
        elif kind == "evidence_discriminated":
            universe = self._universe_from_evidence(
                gp["gene"], idx=inst_seq, include_other=True)
        elif kind in ("term_annotated_genes", "exp_term_genes"):
            candidates = self._draw_candidate_panel(rng, gp["term"], kind)
            if candidates is None:
                universe = None
            else:
                gp["candidate_genes"] = candidates
                universe = self._term_reverse_universe(
                    gp["term"], cap, candidates, idx=inst_seq)
        else:
            raise ValueError(f"unknown kind {kind!r}")
        if universe is None:
            return None
        ast: dict[str, Any] = {"kind": kind, "universe_ids": sorted(universe),
                               "complexity": spec.complexity}
        ast.update(gp)
        instance_id = "in-" + canonical_sha(
            {"family": spec.family_id, "base": base_goal_id, "ast": ast,
             "seed": base_seed})[:20]
        src_units = [f"goterm:{t}" for t in ast["universe_ids"][:8]]
        if kind in ("exp_annotation_closure", "evidence_discriminated",
                    "term_annotated_genes", "exp_term_genes"):
            src_units.append("gaf:goa_human.v238")
        return CandidateTaskAST(
            lane_id=LANE, family_id=spec.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=spec.regime,
            semantic_profile=PROFILE, ast=ast, source_unit_ids=tuple(src_units))

    def _draw_candidate_panel(self, rng: Random, term: str,
                              kind: str) -> list[str] | None:
        """Candidate gene panel: a mix of genes propagated-annotated to the
        term (drawn from the propagated-only reverse sets so the answer
        depends on the fragment's propagation edges, never on direct lookup)
        and genes with P annotations that are not (the false side), shuffled
        deterministically. For the evidence-gated variant the false pool
        deliberately includes genes annotated without experimental
        evidence."""
        p = self.pools
        gene_ev = p["gene_ev"]
        if kind == "exp_term_genes":
            true_pool = sorted(p["term_genes_exp_prop"].get(term, set()))
            false_pool = sorted(
                g for g, by in gene_ev.items()
                if by.get("all") and g not in p["term_genes_exp"].get(term, set()))
        else:
            true_pool = sorted(p["term_genes_prop"].get(term, set()))
            false_pool = sorted(
                g for g, by in gene_ev.items()
                if by.get("all") and g not in p["term_genes"].get(term, set()))
        if len(true_pool) < 2 or len(false_pool) < 2:
            return None
        k_true = min(len(true_pool), rng.choice([2, 3, 4]))
        k_false = min(len(false_pool), rng.choice([2, 3, 4]))
        trues = rng.sample(true_pool, k_true)
        falses = rng.sample(false_pool, k_false)
        panel = trues + falses
        rng.shuffle(panel)
        return panel

    def _pick_spurious_parent(self, rng: Random, term: str,
                              universe: list[str]) -> str | None:
        """A spurious is_a parent: same namespace, in fragment, not an
        is_a ancestor/descendant of the term, not already its parent."""
        t = term
        u = set(universe)
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

    def _pick_spurious_partof_target(self, rng: Random, subject: str,
                                     universe: list[str]) -> str | None:
        """A spurious part_of target: same namespace, in fragment, not
        already part_of-reachable from the subject, not the subject itself.
        The subject must keep at least one TRUE part_of edge inside the
        fragment so the negative self-check has a meaningful wrong fix."""
        s = subject
        u = set(universe)
        ns = self.terms[s]["namespace"]
        po = set(self.closures["partof_anc"].get(s, []))
        true_targets = set(self.pools["partof_targets"].get(s, []))
        if not (true_targets & u):
            return None
        cands = [x for x in sorted(u)
                 if self.terms[x]["namespace"] == ns and x != s and x not in po]
        if not cands:
            return None
        return rng.choice(cands)

    # ---------------------------------------------------------------- ground

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        universe = ast["universe_ids"]
        frag = induce(universe, self.graph)
        edges = self._public_edges(candidate, frag)
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
        name = lambda t: f"{self.terms[t]['name']} ({t})"  # noqa: E731
        if kind in ("exp_annotation_closure", "evidence_discriminated"):
            public["gene"] = ast["gene"]
        if kind in ("term_annotated_genes", "exp_term_genes"):
            public["term"] = {"go_id": ast["term"],
                              "name": self.terms[ast["term"]]["name"]}
            public["candidate_genes"] = list(ast["candidate_genes"])
        oracle_ref = "l06x-oracle-" + _sha(candidate.instance_id + "|oracle")[:16]
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                            native_evidence={"profile": PROFILE,
                                             "release_id": RELEASE_ID})

    def _public_edges(self, candidate: CandidateTaskAST,
                      frag: dict[str, Any]) -> list[dict[str, str]]:
        """Edges shown to the agent; tampered for the four new repair kinds
        (apply_tamper20k is the single source of tamper truth, shared with
        the verifier)."""
        from benchmark.expansion.l06.fragment20k import apply_tamper20k
        from benchmark.plugins.verifier.l06.fragment import fragment_edges
        ast = candidate.ast
        if ast["kind"] in {"repair_partof_edge_deletion",
                           "repair_spurious_isa_edge",
                           "repair_regulation_polarity_swap",
                           "repair_spurious_partof_edge"}:
            broken = apply_tamper20k(ast, frag)
            edges = fragment_edges(broken)
        else:
            edges = fragment_edges(frag)
        return [{"s": s, "p": p, "o": o} for s, p, o in edges]

    def _user_goal(self, candidate: CandidateTaskAST) -> str:
        ast = candidate.ast
        kind = ast["kind"]
        name = lambda t: f"{self.terms[t]['name']} ({t})"  # noqa: E731
        if kind == "posregulators":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    f"of GO terms that POSITIVELY regulate {name(ast['seed'])} under "
                    "the declared profile go-rel-closure/1: only "
                    "positively_regulates edges count (positively_regulates "
                    "propagates over is_a on both sides; the generic regulates "
                    "rollup and negatively_regulates edges do NOT count). Submit "
                    "the complete set as answer_ids; partial or superset answers "
                    "are scored as failure.")
        if kind == "negregulators":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    f"of GO terms that NEGATIVELY regulate {name(ast['seed'])} under "
                    "the declared profile go-rel-closure/1: only "
                    "negatively_regulates edges count (negatively_regulates "
                    "propagates over is_a on both sides; the generic regulates "
                    "rollup and positively_regulates edges do NOT count). Submit "
                    "the complete set as answer_ids.")
        if kind == "generic_only_regulators":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    f"of GO terms that regulate {name(ast['seed'])} using ONLY the "
                    "generic regulates relation under go-rel-closure/1: a term "
                    "qualifies when some path (with is_a propagation on both "
                    "sides) uses a generic regulates edge, and NO path uses a "
                    "positively_regulates or negatively_regulates edge to reach "
                    "the target. Terms that regulate the target via a polarity "
                    "edge are excluded even if they also reach it generically. "
                    "Submit the complete set as answer_ids.")
        if kind == "dual_regulators":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    f"of GO terms that regulate BOTH {name(ast['a'])} AND "
                    f"{name(ast['b'])} under go-rel-closure/1 (regulates, "
                    "positively_regulates and negatively_regulates all count, "
                    "with is_a propagation on both sides; regulates is NOT "
                    "transitive). Submit the complete set as answer_ids.")
        if kind == "partof_isa_overlap":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    f"of GO terms that are simultaneously is_a ancestors of "
                    f"{name(ast['seed'])} AND members of its part_of closure under "
                    "go-rel-closure/1 (terms reachable by a path of is_a and "
                    "part_of edges that uses at least one part_of edge, with "
                    "part_of transitive and propagating over is_a on both "
                    "sides). Submit the complete set as answer_ids.")
        if kind == "reg_subject_descendants":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    f"of GO terms that are is_a descendants of {name(ast['root'])} "
                    "AND stand in the regulates relation to some term of the "
                    "fragment under go-rel-closure/1: the regulation may be "
                    "asserted directly (regulates, positively_regulates or "
                    "negatively_regulates edge with both endpoints in the "
                    "fragment) or inherited by subsumption (a term whose is_a "
                    "ancestor within the fragment regulates something). Submit "
                    "the complete set as answer_ids.")
        if kind == "partof_depth_band":
            return ("In the provided Gene Ontology fragment, return the exact set "
                    "of GO terms whose partonomy-aware fragment depth — the "
                    "longest path length to a fragment root over BOTH is_a and "
                    "part_of edges (child->parent, part->whole) — lies between "
                    f"{ast['lo']} and {ast['hi']} inclusive. Submit the complete "
                    "set as answer_ids.")
        if kind == "repair_partof_edge_deletion":
            return ("The provided Gene Ontology fragment was corrupted: exactly "
                    f"one part_of edge of {name(ast['subject'])} was deleted, so "
                    "part of its inferred partonomy is missing. Identify the "
                    "deleted edge and submit it as "
                    "{'subject': <term>, 'object': <part_of_target>}.")
        if kind == "repair_spurious_isa_edge":
            return ("The provided Gene Ontology fragment was corrupted: one "
                    "SPURIOUS is_a edge was added between a term and a class "
                    "that is not its true parent, so the inferred subsumption "
                    "closure no longer matches the reference Gene Ontology "
                    "release. Identify the spurious edge and submit "
                    "{'term': <term>, 'remove_parent': <spurious_parent>}.")
        if kind == "repair_regulation_polarity_swap":
            return ("The provided Gene Ontology fragment was corrupted: the "
                    f"regulation edge between {name(ast['subject'])} and "
                    f"{name(ast['object'])} was relabelled with the wrong "
                    "polarity (positively_regulates <-> negatively_regulates). "
                    "Restore the correct predicate and submit "
                    "{'subject': ..., 'object': ..., 'predicate': ...}.")
        if kind == "repair_spurious_partof_edge":
            return ("The provided Gene Ontology fragment was corrupted: one "
                    "SPURIOUS part_of edge was added, asserting a partonomy "
                    "relation that does not exist in the reference Gene Ontology "
                    "release. Identify the spurious edge and submit "
                    "{'subject': <term>, 'remove_object': <spurious_target>}.")
        if kind == "exp_annotation_closure":
            return (f"The gene {ast['gene']} is annotated in the pinned GOA human "
                    "GAF (aspect P). Return the exact set of GO terms in the "
                    "fragment that are DIRECT annotations of this gene supported "
                    "by EXPERIMENTAL evidence (evidence codes EXP, IDA, IMP, IGI, "
                    "IEP, HTP, HDA, HMP, HGI, HEP) or is_a/part_of propagated "
                    "ancestors of such experimentally evidenced annotations "
                    "(profile go-rel-closure/1). Annotations with other evidence "
                    "codes do not contribute. Submit the complete set as "
                    "answer_ids.")
        if kind == "evidence_discriminated":
            return (f"The gene {ast['gene']} is annotated in the pinned GOA human "
                    "GAF (aspect P). Return the exact set of GO terms in the "
                    "fragment that are is_a/part_of propagated ancestors of the "
                    "gene's EXPERIMENTALLY evidenced annotations (evidence codes "
                    "EXP, IDA, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP) but are "
                    "NOT ancestors of any of its annotations with other evidence "
                    "codes (profile go-rel-closure/1). Submit the complete set "
                    "as answer_ids.")
        if kind == "term_annotated_genes":
            tname = f"{self.terms[ast['term']]['name']} ({ast['term']})"
            genes = ", ".join(ast["candidate_genes"])
            return (f"Which of the candidate genes {{{genes}}} are annotated, "
                    f"directly or by is_a/part_of propagation, to {tname} in the "
                    "pinned GOA human GAF (aspect P)? A gene counts when one of "
                    "its direct annotation terms is the given term itself or "
                    "propagates to it over the fragment's is_a/part_of edges "
                    "(profile go-rel-closure/1). Submit the exact subset as "
                    "answer_genes.")
        if kind == "exp_term_genes":
            tname = f"{self.terms[ast['term']]['name']} ({ast['term']})"
            genes = ", ".join(ast["candidate_genes"])
            return (f"Which of the candidate genes {{{genes}}} carry an "
                    "EXPERIMENTALLY evidenced annotation (evidence codes EXP, "
                    "IDA, IMP, IGI, IEP, HTP, HDA, HMP, HGI, HEP) to "
                    f"{tname}, directly or by is_a/part_of propagation, in the "
                    "pinned GOA human GAF (aspect P)? Submit the exact subset "
                    "as answer_genes.")
        raise ValueError(f"no wording for kind {kind!r}")
