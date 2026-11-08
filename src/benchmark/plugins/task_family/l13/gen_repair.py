"""L13 repair-regime builders: f01 required-param, f02 param-type, f03 body, f07 enum.

All four produce ``param_contract``/``body_contract`` ASTs: a public contract
excerpt, a public failing initial state, a private repaired reference, guards
pinning the conforming prefix, and a ``relaxed_contract`` witness recipe.
"""
from __future__ import annotations

from typing import Any, Iterator

from benchmark.plugins.source.l13 import contracts, oas, synth
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13.families import (
    FamilySpec, build_ast, instance_rng, make_id,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, SourceSnapshot

MAX_SPECS_SCANNED = 250


def _synth_assignment(params: list[dict[str, Any]], rng) -> dict[str, Any] | None:
    assignment: dict[str, Any] = {}
    for p in params:
        if not p["required"] and rng.randrange(3) == 0:
            continue  # optional params: included with p=2/3
        try:
            assignment[p["name"]] = synth.minimal_instance(p["schema"], rng, hint=p["name"])
        except OasError:
            return None
    return assignment


def _emit(family: FamilySpec, unit: dict[str, Any], coordinate: dict[str, Any],
          contract: dict[str, Any], initials: list[tuple[Any, Any, list, dict]],
          seed: int) -> Iterator[CandidateTaskAST]:
    """One base goal; ``initials`` are per-variant (initial, reference, guards, witness)."""
    base_goal_id = make_id("bg", family.family_id, unit["unit_id"], coordinate)
    for variant, (initial, reference, guards, witness) in enumerate(initials):
        instance_id = make_id("in", base_goal_id, variant, seed)
        ast = build_ast(family, unit=unit, coordinate=coordinate, contract=contract,
                        initial=initial, reference=reference, guards=guards,
                        constraints={}, witness=witness, variant=variant)
        yield CandidateTaskAST(
            lane_id="L13", family_id=family.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=family.regime,
            semantic_profile="openapi-3.0-contract", ast=ast,
            source_unit_ids=(unit["unit_id"],),
        )


def _param_sites(plugin, snapshot: SourceSnapshot, family: FamilySpec, seed: int,
                 instances_per_goal: int, mode: str) -> Iterator[CandidateTaskAST]:
    found, scanned = 0, 0
    for unit in plugin.units(snapshot):
        if found >= family.base_goals or scanned >= MAX_SPECS_SCANNED:
            break
        scanned += 1
        try:
            spec = plugin.source.load_spec(unit)
        except OasError:
            continue
        for site in oas.iter_operations(spec):
            params = contracts.task_params(site, spec)
            required = [p for p in params if p["required"]]
            if len(params) < 2 or not required:
                continue
            target_pool = {
                "required_param": list(required),
                "type_param": params,
                "enum_param": [p for p in params if isinstance(p["schema"].get("enum"), list)],
            }[mode]
            if not target_pool:
                continue
            coordinate = {"path": site.path, "method": site.method,
                          "operation_key": site.operation_key}
            initials = []
            ok = True
            for variant in range(instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, variant))
                assignment = _synth_assignment(params, rng)
                if assignment is None:
                    ok = False
                    break
                target = target_pool[variant % len(target_pool)]
                if mode == "required_param":
                    if target["name"] not in assignment:
                        ok = False
                        break
                    initial = {k: v for k, v in assignment.items() if k != target["name"]}
                    reference = dict(initial)
                    reference[target["name"]] = synth.minimal_instance(target["schema"], rng, hint=target["name"])
                    witness = {"kind": "relaxed_contract",
                               "relax": {"remove_required": [target["name"]]}}
                else:
                    try:
                        bad = synth.violate_value(target["schema"], rng)
                    except OasError:
                        ok = False
                        break
                    initial = dict(assignment)
                    initial[target["name"]] = bad
                    reference = dict(initial)
                    reference[target["name"]] = synth.minimal_instance(target["schema"], rng, hint=target["name"])
                    witness = {"kind": "relaxed_contract",
                               "relax": {"replace_schema": {target["name"]: {}}}}
                guards = [{"kind": "param", "name": n, "expect": "unchanged"}
                          for n in sorted(initial) if n != target["name"]]
                initials.append((initial, reference, guards, witness))
            if not ok or len(initials) != instances_per_goal:
                break
            yield from _emit(family, unit, coordinate, contracts.param_contract(params), initials, seed)
            found += 1
            break  # one base goal per spec


def required_param(plugin, snapshot, family, *, seed, instances_per_goal):
    yield from _param_sites(plugin, snapshot, family, seed, instances_per_goal, "required_param")


def type_param(plugin, snapshot, family, *, seed, instances_per_goal):
    yield from _param_sites(plugin, snapshot, family, seed, instances_per_goal, "type_param")


def enum_param(plugin, snapshot, family, *, seed, instances_per_goal):
    yield from _param_sites(plugin, snapshot, family, seed, instances_per_goal, "enum_param")


def body(plugin, snapshot: SourceSnapshot, family: FamilySpec, *, seed: int,
         instances_per_goal: int) -> Iterator[CandidateTaskAST]:
    found, scanned = 0, 0
    for unit in plugin.units(snapshot):
        if found >= family.base_goals or scanned >= MAX_SPECS_SCANNED:
            break
        scanned += 1
        try:
            spec = plugin.source.load_spec(unit)
        except OasError:
            continue
        for site in oas.iter_operations(spec):
            if site.request_body_schema is None:
                continue
            try:
                schema = oas.normalize_schema(site.request_body_schema, spec)
            except OasError:
                continue
            props = schema.get("properties") or {}
            required = sorted(schema.get("required") or [])
            if schema.get("type") != "object" or len(props) < 2 or not required:
                continue
            coordinate = {"path": site.path, "method": site.method,
                          "operation_key": site.operation_key}
            contract = {"kind": "body_contract", "body_schema": schema}
            initials = []
            ok = True
            for variant in range(instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, variant))
                try:
                    full = synth.minimal_instance(schema, rng, hint="body")
                except OasError:
                    ok = False
                    break
                if not isinstance(full, dict):
                    ok = False
                    break
                target = required[variant % len(required)]
                initial = dict(full)
                if variant % 2 == 0:
                    del initial[target]
                    reference = dict(initial)
                    reference[target] = synth.minimal_instance(
                        props.get(target) or {}, rng, hint=target)
                    witness = {"kind": "relaxed_contract",
                               "relax": {"body_remove_required": [target]}}
                else:
                    try:
                        initial[target] = synth.violate_value(props.get(target) or {}, rng)
                    except OasError:
                        ok = False
                        break
                    reference = dict(initial)
                    reference[target] = synth.minimal_instance(
                        props.get(target) or {}, rng, hint=target)
                    witness = {"kind": "relaxed_contract",
                               "relax": {"body_replace_schema": {target: {}}}}
                guards = [{"kind": "body_prop", "name": n, "expect": "unchanged"}
                          for n in sorted(initial) if n != target]
                initials.append((initial, reference, guards, witness))
            if not ok or len(initials) != instances_per_goal:
                break
            yield from _emit(family, unit, coordinate, contract, initials, seed)
            found += 1
            break
