"""L07 proposal support: cached native simulations over the pinned snapshot.

The verifier never reads this cache — it recomputes from pinned bytes. This
module exists so proposal-time screening (parameter-effect probes, candidate
ladders, corruption checks) does not re-run identical CVODE integrations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from random import Random
from typing import Any

from benchmark.plugins.source.l07.plugin import L07SourcePlugin, load_struct, screening_ok
from benchmark.plugins.source.l07.sbml_edit import ParameterEdit, apply_edit
from benchmark.plugins.source.l07.sbml_struct import ModelStruct
from benchmark.plugins.source.l07.simulate import SimulationError, simulate_checked


class SimRunner:
    """Deterministic sims keyed by (model_id, edit, probe tuple)."""

    def __init__(self, index: dict[str, Any], plugin: L07SourcePlugin) -> None:
        self.index = index
        self.plugin = plugin
        self._struct: dict[str, ModelStruct] = {}
        self._sim: dict[str, dict[str, Any]] = {}

    def struct(self, model_id: str) -> ModelStruct:
        if model_id not in self._struct:
            self._struct[model_id] = load_struct(self.index["models"][model_id])
        return self._struct[model_id]

    def raw(self, model_id: str) -> bytes:
        return self.plugin.read_model(model_id)

    def sim(self, model_id: str, probes: tuple[str, ...],
            edit: ParameterEdit | None = None) -> dict[str, Any]:
        key_parts: list[Any] = [model_id, list(probes)]
        if edit is not None:
            key_parts.append([edit.kind, edit.parameter_id, edit.species_id,
                              edit.reaction_id, repr(edit.new_value), edit.formula])
        key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode()).hexdigest()
        if key not in self._sim:
            sbml = self.raw(model_id)
            if edit is not None:
                sbml = apply_edit(sbml, edit)
            self._sim[key] = simulate_checked(sbml.decode("utf-8"), list(probes))
        return self._sim[key]

    def capable_models(self, *, needs_sim: bool, min_floating: int,
                       min_reactions: int, min_params: int) -> list[str]:
        out: list[str] = []
        for mid, rec in sorted(self.index["models"].items()):
            scr = rec.get("screening") or {}
            if not scr.get("parse_ok"):
                continue
            if needs_sim and not screening_ok(scr):
                continue
            if scr.get("n_floating_species", 0) < min_floating:
                continue
            if scr.get("n_reactions", 0) < min_reactions:
                continue
            if scr.get("n_settable_parameters", 0) < min_params:
                continue
            out.append(mid)
        return out


def sample_universe(rng: Random, pool: list[str], size: int) -> list[str] | None:
    pool = sorted(pool)
    if len(pool) < size:
        return None
    return sorted(rng.sample(pool, size))


def rel_diff(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def fmt_float(v: float) -> str:
    """Full round-trip repr so public payloads carry exact values."""
    return repr(float(v))


def species_roles(struct: ModelStruct) -> dict[str, str]:
    """Deterministic reaction-graph role per species (family f09 vocabulary)."""
    reactant: set[str] = set()
    product: set[str] = set()
    modifier: set[str] = set()
    for r in struct.reactions:
        reactant.update(s for s, _ in r.reactants)
        product.update(s for s, _ in r.products)
        modifier.update(r.modifiers)
    roles: dict[str, str] = {}
    for s in struct.species:
        sid = s.sid
        if s.boundary:
            roles[sid] = "boundary"
        elif sid in reactant and sid in product:
            roles[sid] = "intermediate"
        elif sid in reactant:
            roles[sid] = "source"
        elif sid in product:
            roles[sid] = "sink"
        elif sid in modifier:
            roles[sid] = "catalyst"
        else:
            roles[sid] = "isolated"
    return roles


def relation_set(struct: ModelStruct, relation: str, target: str) -> set[str]:
    """Reaction ids satisfying a structural relation to ``target`` (family f08)."""
    out: set[str] = set()
    for r in struct.reactions:
        if relation == "consumes" and any(s == target for s, _ in r.reactants):
            out.add(r.rid)
        elif relation == "produces" and any(s == target for s, _ in r.products):
            out.add(r.rid)
        elif relation == "modulates" and target in r.modifiers:
            out.add(r.rid)
    return out
