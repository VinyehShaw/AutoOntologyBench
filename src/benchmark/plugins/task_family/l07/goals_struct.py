"""L07 structural base-goal proposers (libsbml-only families f08/f09)."""
from __future__ import annotations

from random import Random
from typing import Any

from benchmark.plugins.task_family.l07.support import (
    SimRunner, relation_set, sample_universe, species_roles,
)

RELATIONS = ("consumes", "produces", "modulates")


def propose_role_query(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    reaction_ids = [r.rid for r in struct.reactions]
    species = [s.sid for s in struct.species]
    combos = [(s, rel) for s in species for rel in RELATIONS]
    rng.shuffle(combos)
    for target, relation in combos[:48]:
        hits = relation_set(struct, relation, target)
        if not hits or len(hits) > len(reaction_ids) - 4:
            continue
        return {"goal_params": {"model_id": model_id, "target_species": target,
                                "relation": relation},
                "reaction_ids": reaction_ids, "hits": sorted(hits)}
    return None


def instantiate_role_query(base: dict[str, Any], rng: Random) -> dict[str, Any] | None:
    hits = sorted(base["hits"])
    misses = sorted(set(base["reaction_ids"]) - set(hits))
    if not hits or len(misses) < 4:
        return None
    take_hits = min(len(hits), 4)
    universe = sorted(rng.sample(hits, take_hits)
                      + rng.sample(misses, min(len(misses), 8 - take_hits)))
    if len(universe) < 5:
        return None
    answer = sorted(set(universe) & set(hits))
    if not answer or len(answer) == len(universe):
        return None
    return {"universe_reaction_ids": universe}


def propose_species_role_org(sim: SimRunner, model_id: str, rng: Random) -> dict[str, Any] | None:
    struct = sim.struct(model_id)
    roles = species_roles(struct)
    if len(set(roles.values())) < 3:
        return None
    return {"goal_params": {"model_id": model_id}, "roles": roles}


def instantiate_species_role_org(base: dict[str, Any], rng: Random,
                                 size: int) -> dict[str, Any] | None:
    roles = base["roles"]
    for _ in range(24):
        universe = sample_universe(rng, sorted(roles), size)
        if universe is None:
            return None
        if len({roles[s] for s in universe}) >= 3:
            return {"universe_ids": universe}
    return None
