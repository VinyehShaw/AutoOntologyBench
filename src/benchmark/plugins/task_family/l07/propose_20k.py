"""L07 20K-candidate plugin: propose/ground the new mechanism families.

Wraps the legacy ``L07TaskFamilyPlugin`` so the pinned index, SimRunner and
source plugin are reused unchanged. The model pool is filtered to exclude
every model already used by the legacy 8K L07 lane, which is what makes the
550 new tasks genuinely new-model tasks rather than relabelled legacy ones.
"""
from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any, Iterator

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.task_family.l07 import goals_20k, ground_20k
from benchmark.plugins.task_family.l07.families_20k import BY_ID_20K, LANE, PROFILE
from benchmark.plugins.task_family.l07.propose import L07TaskFamilyPlugin
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

DEFAULT_LEGACY_ROOT = Path("bank/staging/l07")

_PROPOSERS = {
    "retention_set": goals_20k.propose_retention_set,
    "peak_time_ranking": goals_20k.propose_peak_time_ranking,
    "pulse_set": goals_20k.propose_pulse_set,
    "init_threshold_set": goals_20k.propose_init_threshold_set,
    "init_window_set": goals_20k.propose_init_window_set,
    "reaction_degree_query": goals_20k.propose_reaction_degree_query,
}


def legacy_models(root: Path = DEFAULT_LEGACY_ROOT) -> set[str]:
    """Every model id appearing in a legacy L07 candidate shard."""
    models: set[str] = set()
    cand_dir = Path(root) / "candidates"
    if not cand_dir.is_dir():
        return models
    for path in sorted(cand_dir.glob("shard-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line:
                models.add(json.loads(line)["model_id"])
    return models


class L07TaskFamilyPlugin20K(L07TaskFamilyPlugin):
    def __init__(self, index_dir: str | Path, *, seed: int = 20261101,
                 exclude_models: set[str] | None = None,
                 legacy_root: Path = DEFAULT_LEGACY_ROOT) -> None:
        super().__init__(index_dir, seed=seed)
        self.exclude_models = (set(exclude_models) if exclude_models is not None
                               else legacy_models(legacy_root))

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    def _capable(self, spec: Any) -> list[str]:
        capable = self.sim.capable_models(
            needs_sim=spec.needs_sim, min_floating=spec.min_floating_species,
            min_reactions=spec.min_reactions, min_params=spec.min_settable_params)
        return [m for m in capable if m not in self.exclude_models]

    def propose(self, snapshot: Any = None, *, family: str, seed: int | None = None,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        spec = BY_ID_20K[family]
        base_seed = self.seed if seed is None else seed
        max_goals = (constraints or {}).get("max_base_goals", spec.base_goals)
        proposer = _PROPOSERS[spec.kind]
        goals_done = 0
        for model_id in self._capable(spec):
            if goals_done >= max_goals:
                break
            rng = Random(f"{family}|base|{model_id}|{base_seed}")
            base = proposer(self.sim, model_id, rng)
            if base is None:
                continue
            base_goal_id = "l07b20-" + canonical_sha(
                {"family": family, "params": base["goal_params"]})[:12]
            instances = [
                cand for inst_seq in range(spec.instances_per_goal)
                if (cand := self._instantiate(spec, base, base_goal_id,
                                              inst_seq, base_seed)) is not None
            ]
            if len(instances) != spec.instances_per_goal:
                continue
            if len({canonical_sha(i.ast) for i in instances}) != spec.instances_per_goal:
                continue
            goals_done += 1
            yield from instances

    def _instantiate(self, spec: Any, base: dict[str, Any], base_goal_id: str,
                     inst_seq: int, base_seed: int) -> CandidateTaskAST | None:
        rng = Random(f"{spec.family_id}|inst|{base_goal_id}|{inst_seq}|{base_seed}")
        gp = base["goal_params"]
        kind = spec.kind
        ast: dict[str, Any] = {"kind": kind, "model_id": gp["model_id"],
                               "complexity": spec.complexity}
        if kind == "retention_set":
            inst = goals_20k.instantiate_retention_set(base, rng, spec.universe_size)
            if inst is None:
                return None
            ast.update(inst)
            ast["retention_threshold"] = gp["retention_threshold"]
        elif kind == "peak_time_ranking":
            inst = goals_20k.instantiate_peak_time_ranking(base, rng,
                                                           spec.universe_size)
            if inst is None:
                return None
            ast.update(inst)
            ast["k"] = gp["k"]
        elif kind == "pulse_set":
            inst = goals_20k.instantiate_pulse_set(base, rng, spec.universe_size)
            if inst is None:
                return None
            ast.update(inst)
            ast["level"] = gp["level"]
        elif kind == "init_threshold_set":
            ast.update(goals_20k.instantiate_init_threshold_set(base, inst_seq))
            ast.update({k: gp[k] for k in gp if k != "model_id"})
        elif kind == "init_window_set":
            ast.update(goals_20k.instantiate_init_window_set(base, inst_seq))
            ast.update({k: gp[k] for k in gp if k != "model_id"})
        elif kind == "reaction_degree_query":
            inst = goals_20k.instantiate_reaction_degree_query(
                base, rng, spec.universe_size)
            if inst is None:
                return None
            ast.update(inst)
            ast["degree_threshold"] = gp["degree_threshold"]
        else:
            raise ValueError(f"unknown kind {kind!r}")
        instance_id = "l07i20-" + canonical_sha(
            {"family": spec.family_id, "base": base_goal_id, "ast": ast})[:16]
        return CandidateTaskAST(
            lane_id=LANE, family_id=spec.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=spec.regime, semantic_profile=PROFILE,
            ast=ast, source_unit_ids=(f"biomodels/{gp['model_id']}",))

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        return ground_20k.ground(candidate, self.sim, self.source)