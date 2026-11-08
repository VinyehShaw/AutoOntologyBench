"""L05 task-family plugin: propose CandidateTaskASTs and ground them.

Pipeline position: source index -> propose (this file) -> ground -> native
verifier gates (prove_exists / certify_semantic_relevance / verify_terminal
self-checks) -> shard writer (build.py).

Grounding computes the private oracle payload (reference answer / alignment,
recomputed natively) and assembles the public view. The public view contains
molecule identities + structures + the group library + the *question*; it
never contains the answer set (leakage is scanned for in the builder).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any, Iterator

from rdkit import Chem, __version__ as RDKIT_VERSION

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.source.l05.features import PROP_LABELS
from benchmark.plugins.source.l05.plugin import L05SourcePlugin
from benchmark.plugins.task_family.l05.families import (
    BY_ID, FAMILY_TABLE, FamilySpec, propose_query,
)
from benchmark.plugins.task_family.l05.plus_families import (
    PLUS_BY_ID, PLUS_QUERY_FAMILIES, PLUS_REPAIR_FAMILIES,
    mutate_exclusion_flip, mutate_overspecific, mutate_wrong_branch,
    propose_plus_query,
)
from benchmark.plugins.task_family.l05.sample import (
    GroupPool, derangement, mutate_overgeneral, mutate_semantic, mutate_syntax,
    sample_universe, sample_universe_per_label,
)
from benchmark.plugins.verifier.l05.query_eval import (
    GroupContext, eval_index, group_leaves, match_smarts,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

LANE = "L05"
DATASET_ID = "autoontobench-bank-8000-rc1"
PROFILE = "rdkit-smarts-native/1"
UNIVERSE_SIZE = {"low": 256, "medium": 384, "high": 512}
ALIGN_LABELS = 4


def _ast_group_labels(ast: dict) -> list[str]:
    kind = ast["kind"]
    if kind == "query":
        return [label for _, label in _walk_group_leaves(ast["query"])]
    if kind == "repair":
        return [ast["label"]]
    return list(ast["labels"])


def _walk_group_leaves(query: dict) -> list[tuple[tuple, str]]:
    op = query.get("op")
    if op in ("group", "count_range", "descendants_of", "group_minus_descendants"):
        return [((), query["group"])]
    if op == "prop_band_with_group":
        return [((), query["group"])]
    if op == "count_range_dual":
        return [((), g) for g in query["groups"]]
    out: list[tuple[tuple, str]] = []
    if op in ("and", "or", "xor"):
        for i, a in enumerate(query["args"]):
            out.extend(((i,) + p, l) for p, l in _walk_group_leaves(a))
    elif op == "not":
        out.extend((("arg",) + p, l) for p, l in _walk_group_leaves(query["arg"]))
    elif op == "profile":
        out.extend((("pos", i), g) for i, g in enumerate(query.get("pos", [])))
        out.extend((("neg", i), g) for i, g in enumerate(query.get("neg", [])))
    return out


def _describe_query(query: dict, names: dict[str, str]) -> str:
    def gname(label: str) -> str:
        return f"{names.get(label, label)} (label '{label}')"

    op = query.get("op")
    if op == "group":
        return f"molecule contains the group {gname(query['group'])}"
    if op == "and":
        return " ALL of: (" + "; ".join(_describe_query(a, names) for a in query["args"]) + ")"
    if op == "or":
        return " ANY of: (" + "; ".join(_describe_query(a, names) for a in query["args"]) + ")"
    if op == "xor":
        return (" EXACTLY ONE of: (" + "; ".join(_describe_query(a, names)
                                                 for a in query["args"]) + ")")
    if op == "not":
        return " NOT (" + _describe_query(query["arg"], names) + ")"
    if op == "count_range":
        return (f"the group {gname(query['group'])} occurs between {query['lo']} and "
                f"{query['hi']} times (inclusive) as distinct substructure matches")
    if op == "count_range_dual":
        parts = []
        for group, (lo, hi) in zip(query["groups"], query["ranges"]):
            parts.append(f"{gname(group)} occurs between {lo} and {hi} times")
        return " BOTH conditions hold: " + "; ".join(parts)
    if op == "prop_band":
        label = PROP_LABELS.get(query["prop"], query["prop"])
        return f"{label} is between {query['lo']} and {query['hi']} (inclusive)"
    if op == "prop_band_with_group":
        label = PROP_LABELS.get(query["prop"], query["prop"])
        return (f"{label} is between {query['lo']} and {query['hi']} (inclusive) "
                f"AND the molecule contains the group {gname(query['group'])}")
    if op == "element_band":
        return (f"the number of {query['element']} atoms is between {query['lo']} "
                f"and {query['hi']} (inclusive)")
    if op == "heavy_band":
        return f"heavy-atom count between {query['lo']} and {query['hi']} (inclusive)"
    if op == "profile":
        pos = ", ".join(gname(g) for g in query["pos"])
        neg = ", ".join(gname(g) for g in query["neg"])
        return f"molecule contains ALL of [{pos}] and NONE of [{neg}]"
    if op == "descendants_of":
        return (f"molecule matches at least one descendant (more specific child) of the "
                f"parent group {gname(query['group'])} in the library hierarchy")
    if op == "group_minus_descendants":
        return (f"molecule matches the parent group {gname(query['group'])} but NONE of "
                "its more specific descendants in the library hierarchy")
    raise ValueError(f"cannot describe op {op!r}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()



class L05TaskFamilyPlugin:
    """TaskFamilyPlugin for all 16 L05 families (see families.FAMILY_TABLE)."""

    def __init__(self, index_dir: str | Path, *, seed: int = 20261027) -> None:
        self.index = L05SourcePlugin.load_index(Path(index_dir))
        self.ctx = GroupContext.from_index(self.index)
        self.smiles_by_id = {m["chebi_id"]: m["canonical_smiles"]
                             for m in self.index["molecules"]}
        self.corpus_ids = sorted(
            mid for mid in self.ctx.heavy
            if Chem.MolFromSmiles(self.smiles_by_id[mid]) is not None)
        self.corpus_set = frozenset(self.corpus_ids)
        self.pool = GroupPool.build(self.ctx, self.corpus_ids)
        self.name_by_label = {g["label"]: g["name"] for g in self.index["hierarchy"]}
        self.smarts_by_label = dict(self.ctx.smarts)
        self.seed = seed
        self._screen_mols: dict[str, Any] | None = None

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    # -- screening subsample for validating repair mutations ----------------

    def _screening_mols(self) -> dict[str, Any]:
        if self._screen_mols is None:
            from rdkit import Chem
            rng = Random(f"screen|{self.seed}")
            ids = rng.sample(self.corpus_ids, min(4000, len(self.corpus_ids)))
            self._screen_mols = {i: Chem.MolFromSmiles(self.smiles_by_id[i]) for i in ids}
        return self._screen_mols

    def _feature_distribution(self, prop: str) -> list[int]:
        if not hasattr(self, "_feature_dists"):
            self._feature_dists: dict[str, list[int]] = {}
        if prop not in self._feature_dists:
            vals = [self.ctx.feature(mid).get(prop, 0) for mid in self.corpus_ids]
            self._feature_dists[prop] = sorted(vals)
        return self._feature_dists[prop]

    # -- propose -------------------------------------------------------------

    def propose(self, snapshot: Any = None, *, family: str, seed: int | None = None,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        spec = PLUS_BY_ID.get(family) or BY_ID[family]
        base_seed = self.seed if seed is None else seed
        max_goals = (constraints or {}).get("max_base_goals", spec.base_goals)
        seen_goals: set[str] = set()
        base_seq = 0
        while len(seen_goals) < max_goals and base_seq < max_goals * 40:
            rng = Random(f"{family}|base|{base_seq}|{base_seed}")
            base_seq += 1
            ast_params = self._propose_base(spec, rng)
            if ast_params is None:
                continue
            prefix = "l05p-" if family in PLUS_BY_ID else "l05b-"
            base_goal_id = prefix + canonical_sha(
                {"family": family, "params": ast_params["goal_params"]})[:12]
            if base_goal_id in seen_goals:
                continue
            seen_goals.add(base_goal_id)
            for inst_seq in range(spec.instances_per_goal):
                cand = self._instantiate(spec, ast_params, base_goal_id, inst_seq,
                                         base_seed)
                if cand is not None:
                    yield cand

    def _propose_base(self, spec: FamilySpec, rng: Random) -> dict[str, Any] | None:
        fid = spec.family_id
        if fid in PLUS_REPAIR_FAMILIES:
            return self._propose_repair_plus(spec, rng)
        if fid in PLUS_QUERY_FAMILIES:
            return self._propose_plus_query(spec, rng)
        if fid.startswith("l05.smarts_repair"):
            return self._propose_repair(spec, rng)
        if fid == "l05.org_align_group_molecule_sets":
            return self._propose_align(spec, rng)
        params = propose_query(spec, rng, self.pool)
        if params is None:
            return None
        corpus = frozenset(self.corpus_ids)
        positives = sorted(eval_index(params["query"], self.ctx, corpus))
        k_pos = min(len(positives), max(spec.min_answer + 2,
                                        UNIVERSE_SIZE[spec.complexity] // 10))
        if k_pos < spec.min_answer:
            return None
        if params.get("require_interaction"):
            # Mirrors the verifier's universe-level interaction gate (fail cheap here).
            for _, label in group_leaves(params["query"], positive_only=True):
                if len(self.ctx.membership.get(label, frozenset()) & corpus) <= len(positives):
                    return None
        return {"kind": "query", "goal_params": params["query"],
                "require_interaction": params.get("require_interaction", False),
                "positives": positives, "k_pos": k_pos}

    def _propose_plus_query(self, spec: FamilySpec, rng: Random) -> dict[str, Any] | None:
        params = propose_plus_query(spec, rng, pool=self.pool,
                                    dist=self._feature_distribution)
        if params is None:
            return None
        corpus = frozenset(self.corpus_ids)
        positives = sorted(eval_index(params["query"], self.ctx, corpus))
        k_pos = min(len(positives), max(spec.min_answer + 2,
                                        UNIVERSE_SIZE[spec.complexity] // 10))
        if k_pos < spec.min_answer:
            return None
        return {"kind": "query", "goal_params": params["query"],
                "require_interaction": params.get("require_interaction", False),
                "positives": positives, "k_pos": k_pos}

    def _propose_repair_plus(self, spec: FamilySpec, rng: Random) -> dict[str, Any] | None:
        fid = spec.family_id
        screen = self._screening_mols()
        corpus = self.corpus_set
        if fid == "l05plus.smarts_repair_wrong_branch":
            labels = sorted(set(self.ctx.children) | {
                child for children in self.ctx.children.values() for child in children})
        elif fid == "l05plus.smarts_repair_exclusion_flip":
            labels = [l for l in self.pool.medium if "!" in self.smarts_by_label.get(l, "")]
        else:
            labels = list(self.pool.frequent)
        rng.shuffle(labels)
        for label in labels[:40]:
            golden = self.smarts_by_label[label]
            golden_screen = frozenset(match_smarts(golden, screen))
            if not golden_screen:
                continue
            for _ in range(10):
                if fid == "l05plus.smarts_repair_overspecific":
                    broken = mutate_overspecific(golden, rng)
                elif fid == "l05plus.smarts_repair_wrong_branch":
                    broken = mutate_wrong_branch(golden, label, self.ctx, rng)
                else:
                    broken = mutate_exclusion_flip(golden, rng)
                if broken is None or broken == golden or golden in broken:
                    continue
                try:
                    broken_screen = frozenset(match_smarts(broken, screen))
                except Exception:
                    continue
                if not broken_screen:
                    continue
                if fid == "l05plus.smarts_repair_overspecific":
                    if not broken_screen < golden_screen:
                        continue
                elif broken_screen == golden_screen:
                    continue
                positives = sorted(self.ctx.membership.get(label, frozenset()) & corpus)
                if len(positives) < spec.min_answer:
                    continue
                return {"kind": "repair",
                        "goal_params": {"label": label, "golden_smarts": golden},
                        "broken_smarts": broken, "positives": positives,
                        "k_pos": min(len(positives), spec.min_answer + 4)}
        return None

    def _propose_repair(self, spec: FamilySpec, rng: Random) -> dict[str, Any] | None:
        fid = spec.family_id
        pool = self.pool.frequent if fid == "l05.smarts_repair_syntax" else self.pool.medium
        labels = list(pool)
        rng.shuffle(labels)
        screen = self._screening_mols()
        for label in labels[:24]:
            golden = self.smarts_by_label[label]
            if fid == "l05.smarts_repair_syntax":
                broken = mutate_syntax(golden, rng)
            elif fid == "l05.smarts_repair_semantic_wrongset":
                broken = mutate_semantic(golden, rng)
            else:
                broken = mutate_overgeneral(golden, rng)
            if broken is None or broken == golden:
                continue
            golden_screen = frozenset(match_smarts(golden, screen))
            if not golden_screen:
                continue
            try:
                broken_screen: frozenset | None = frozenset(match_smarts(broken, screen))
            except Exception:
                broken_screen = None  # syntax defect: golden parses, broken does not
            if fid == "l05.smarts_repair_syntax":
                if broken_screen is not None:
                    continue  # mutator contract: must fail to parse
            elif fid == "l05.smarts_repair_semantic_wrongset":
                if broken_screen is None or broken_screen == golden_screen:
                    continue
            else:  # overgeneral: strict superset on the screening sample
                if broken_screen is None or not golden_screen < broken_screen:
                    continue
            positives = sorted(self.ctx.membership.get(label, frozenset())
                               & self.corpus_set)
            if len(positives) < spec.min_answer:
                continue
            return {"kind": "repair",
                    "goal_params": {"label": label, "golden_smarts": golden},
                    "broken_smarts": broken,
                    "positives": positives,
                    "k_pos": min(len(positives), spec.min_answer + 4)}
        return None

    def _propose_align(self, spec: FamilySpec, rng: Random) -> dict[str, Any] | None:
        labels = [l for l in self.pool.medium if not l.startswith("flat:")]
        rng.shuffle(labels)
        chosen: list[str] = []
        sets: list[frozenset] = []
        corpus = frozenset(self.corpus_ids)
        for label in labels:
            s = frozenset(self.ctx.membership.get(label, frozenset()) & corpus)
            if not s or s in sets:
                continue
            chosen.append(label)
            sets.append(s)
            if len(chosen) == ALIGN_LABELS:
                break
        if len(chosen) < ALIGN_LABELS:
            return None
        perm = derangement(rng, chosen)
        positives = sorted(set().union(*sets))
        return {"kind": "align",
                "goal_params": {"labels": chosen, "derangement": perm},
                "positives": positives,
                "k_pos": min(len(positives), ALIGN_LABELS * (spec.min_answer + 2))}

    # -- instance construction ------------------------------------------------

    def _instantiate(self, spec: FamilySpec, base: dict[str, Any], base_goal_id: str,
                     inst_seq: int, base_seed: int) -> CandidateTaskAST | None:
        rng = Random(f"{spec.family_id}|inst|{base_goal_id}|{inst_seq}|{base_seed}")
        size = UNIVERSE_SIZE[spec.complexity]
        if base["kind"] == "align":
            universe = sample_universe_per_label(
                rng, base["goal_params"]["labels"], self.ctx, self.corpus_ids,
                size=size, k_per_label=spec.min_answer + 2)
        else:
            universe = sample_universe(rng, base["positives"], self.corpus_ids,
                                       size=size, k_pos=base["k_pos"])
        if universe is None:
            return None
        ast: dict[str, Any] = {"kind": base["kind"], "universe_ids": sorted(universe),
                               "complexity": spec.complexity}
        if base["kind"] == "query":
            ast["query"] = base["goal_params"]
            ast["require_interaction"] = base["require_interaction"]
        elif base["kind"] == "repair":
            ast.update(base["goal_params"])
            ast["broken_smarts"] = base["broken_smarts"]
        else:
            labels = base["goal_params"]["labels"]
            ast["labels"] = labels
            ast["misassign_pair"] = self._first_swap_pair(base["goal_params"]["derangement"])
            ast["derangement"] = base["goal_params"]["derangement"]
        inst_prefix = "l05x-" if spec.family_id in PLUS_BY_ID else "l05i-"
        instance_id = inst_prefix + canonical_sha(
            {"family": spec.family_id, "base": base_goal_id, "ast": ast})[:16]
        return CandidateTaskAST(
            lane_id=LANE, family_id=spec.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=spec.regime, semantic_profile=PROFILE,
            ast=ast,
            source_unit_ids=tuple(
                [f"fg:{l}" for l in _ast_group_labels(ast)] +
                [f"mol:{m}" for m in universe[:8]]))

    @staticmethod
    def _first_swap_pair(perm: dict[str, str]) -> tuple[str, str]:
        for a in sorted(perm):
            b = perm[a]
            if perm.get(b) == a and a < b:
                return (a, b)
        items = sorted(perm.items())
        return (items[0][0], items[0][1])

    # -- ground ---------------------------------------------------------------

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        universe = [{"chebi_id": mid, "canonical_smiles": self.smiles_by_id[mid],
                     "heavy_atoms": self.ctx.heavy[mid]} for mid in ast["universe_ids"]]
        public: dict[str, Any] = {
            "user_goal": self._user_goal(candidate),
            "universe": universe,
            "group_library": self._public_library(ast),
            "allowed_tools": ["molecule_inventory.read", "functional_group_library.read",
                              "rdkit.substructure_match"],
            "budget": {"probes": 4, "actions": 16},
            "semantic_profile": PROFILE,
            "complexity": ast["complexity"],
        }
        if ast["kind"] == "repair":
            public["broken_smarts"] = ast["broken_smarts"]
            public["intended_group"] = {"label": ast["label"],
                                        "name": self.name_by_label.get(ast["label"], ast["label"])}
        if ast["kind"] == "align":
            public["misaligned_table"] = self._deranged_table(candidate)
        oracle_ref = "orl05_" + _sha(candidate.instance_id + "|oracle")[:16]
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                            native_evidence={"rdkit_version": RDKIT_VERSION,
                                             "profile": PROFILE})

    def _public_library(self, ast: dict) -> list[dict[str, str]]:
        """Full group library as public context. For repair tasks the golden
        label's true SMARTS is withheld (it IS the answer); the broken variant
        is published separately as ``broken_smarts``."""
        withheld = ast.get("label") if ast["kind"] == "repair" else None
        return [{"label": l, "smarts": self.smarts_by_label[l],
                 "name": self.name_by_label.get(l, l)}
                for l in sorted(self.smarts_by_label) if l != withheld]

    def _deranged_table(self, candidate: CandidateTaskAST) -> dict[str, list[str]]:
        ast = candidate.ast
        universe = frozenset(ast["universe_ids"])
        correct = {l: sorted(self.ctx.membership.get(l, frozenset()) & universe)
                   for l in ast["labels"]}
        return {label: correct[wrong] for label, wrong in ast["derangement"].items()}

    # -- wording (deterministic templates; LLM paraphrase is a later, optional pass)

    def _user_goal(self, candidate: CandidateTaskAST) -> str:
        ast = candidate.ast
        kind = ast["kind"]
        if kind == "query":
            cond = _describe_query(ast["query"], self.name_by_label)
            return ("From the provided ChEBI molecule inventory, return the exact set of "
                    "ChEBI identifiers of all molecules satisfying this structural "
                    f"condition: {cond}. Submit the complete set as answer_ids; partial "
                    "or superset answers are scored as failure.")
        if kind == "repair":
            return ("The SMARTS rule "
                    f"'{ast['broken_smarts']}' was intended to identify the functional "
                    f"group '{self.name_by_label.get(ast['label'], ast['label'])}' "
                    f"(library label '{ast['label']}'; its reference definition is "
                    "withheld from the provided library) in the provided ChEBI molecule "
                    "inventory, but it does not reproduce the intended behavior. Repair "
                    "the rule and submit it as repaired_smarts; it must match exactly the "
                    "intended molecule set over the inventory.")
        return ("The provided table assigns each RDKit functional-group label a ChEBI "
                "molecule-ID set, but every row was attached to the wrong label. Using "
                "the molecule inventory and the group library, re-align every label to "
                "its correct molecule set and submit the full corrected mapping as "
                "alignment {label: [chebi_ids]}.")
