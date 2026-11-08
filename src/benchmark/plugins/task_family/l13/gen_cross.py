"""L13 f05 API version migration (repair regime, cross-version).

Pairs two pinned versions of the same ``api_path``: the initial payload
validates under the older contract and fails the newer one; the reference is
the minimal repair under the newer contract, keeping every value the newer
contract still accepts unchanged (guards).
"""
from __future__ import annotations

import re
from typing import Any, Iterator

from benchmark.plugins.source.l13 import contracts, oas, synth
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13.families import (
    FamilySpec, build_ast, instance_rng, make_id,
)
from benchmark.plugins.task_family.l13.gen_repair import _synth_assignment
from benchmark.schema.plugin_protocols import CandidateTaskAST, SourceSnapshot


def _vkey(version: str) -> tuple:
    return tuple((0, int(p)) if p.isdigit() else (1, p)
                 for p in re.split(r"[^0-9A-Za-z]+", version) if p)


def _repair_params(contract_b: dict[str, Any], initial: dict[str, Any], rng) -> dict[str, Any] | None:
    out: dict[str, Any] = {}
    try:
        for p in contract_b["parameters"]:
            name = p["name"]
            if name in initial and oas.validate_instance(p["schema"], initial[name]) == []:
                out[name] = initial[name]
            else:
                out[name] = synth.minimal_instance(p["schema"], rng, hint=name)
    except OasError:
        return None
    return out


def _repair_body(schema_b: dict[str, Any], initial: Any, rng) -> dict[str, Any] | None:
    if not isinstance(initial, dict):
        return None
    props = schema_b.get("properties") or {}
    out: dict[str, Any] = {}
    for name, val in initial.items():
        sub = props.get(name)
        if sub is not None and oas.validate_instance(sub, val) == []:
            out[name] = val
    try:
        for name in schema_b.get("required") or []:
            if name not in out:
                out[name] = synth.minimal_instance(props.get(name) or {}, rng, hint=name)
    except OasError:
        return None
    return out


def version_migration(plugin, snapshot: SourceSnapshot, family: FamilySpec, *, seed: int,
                      instances_per_goal: int) -> Iterator[CandidateTaskAST]:
    by_api: dict[str, list[dict[str, Any]]] = {}
    for unit in plugin.units(snapshot):
        api_path = unit["metadata"].get("api_path")
        if api_path:
            by_api.setdefault(api_path, []).append(unit)
    pairs = []
    for api_path in sorted(by_api):
        units = sorted(by_api[api_path], key=lambda u: _vkey(u["metadata"]["api_version"]))
        if len(units) >= 2:
            pairs.append((units[0], units[-1]))
    found = 0
    for unit_a, unit_b in pairs:
        if found >= family.base_goals:
            break
        try:
            spec_a = plugin.source.load_spec(unit_a)
            spec_b = plugin.source.load_spec(unit_b)
        except OasError:
            continue
        ops_a = {(s.path, s.method): s for s in oas.iter_operations(spec_a)}
        ops_b = {(s.path, s.method): s for s in oas.iter_operations(spec_b)}
        for key in sorted(ops_a):
            site_a, site_b = ops_a.get(key), ops_b.get(key)
            if site_b is None:
                continue
            mode, contract_a, contract_b = _pick_mode(spec_a, spec_b, site_a, site_b, seed, unit_b, key)
            if mode is None:
                continue
            initials = _build_variants(family, unit_b, key, mode, contract_a, contract_b,
                                       site_a, site_b, spec_a, spec_b, seed, instances_per_goal)
            if initials is None:
                break
            base_goal_id = make_id("bg", family.family_id, unit_b["unit_id"],
                                   {"path": key[0], "method": key[1], "mode": mode})
            coordinate = {"path": key[0], "method": key[1],
                          "operation_key": site_b.operation_key, "mode": mode}
            for variant, (initial, reference, guards) in enumerate(initials):
                instance_id = make_id("in", base_goal_id, variant, seed)
                ast = build_ast(family, unit=unit_b, coordinate=coordinate,
                                contract=contract_b, initial=initial, reference=reference,
                                guards=guards, constraints={},
                                witness={"kind": "version_contrast"},
                                variant=variant, unit_b=unit_a, contract_b=contract_a)
                yield CandidateTaskAST(
                    lane_id="L13", family_id=family.family_id, base_goal_id=base_goal_id,
                    instance_id=instance_id, regime=family.regime,
                    semantic_profile="openapi-3.0-contract", ast=ast,
                    source_unit_ids=(unit_a["unit_id"], unit_b["unit_id"]),
                )
            found += 1
            break


def _pick_mode(spec_a, spec_b, site_a, site_b, seed, unit_b, key):
    """Return (mode, contract_a, contract_b) or (None, None, None)."""
    params_a = contracts.task_params(site_a, spec_a)
    params_b = contracts.task_params(site_b, spec_b)
    if params_a and params_b:
        contract_a, contract_b = contracts.param_contract(params_a), contracts.param_contract(params_b)
        probe_rng = instance_rng(seed, make_id("probe", "l13-f05", unit_b["unit_id"], key))
        probe = _synth_assignment(params_a, probe_rng)
        if probe is not None and contracts.check_param_assignment(contract_a, probe) == [] \
                and contracts.check_param_assignment(contract_b, probe) != []:
            return "params", contract_a, contract_b
    if site_a.request_body_schema is not None and site_b.request_body_schema is not None:
        try:
            sch_a = oas.normalize_schema(site_a.request_body_schema, spec_a)
            sch_b = oas.normalize_schema(site_b.request_body_schema, spec_b)
            probe_rng = instance_rng(seed, make_id("probe", "l13-f05", unit_b["unit_id"], key))
            probe = synth.minimal_instance(sch_a, probe_rng, hint="body")
        except OasError:
            return None, None, None
        if oas.validate_instance(sch_a, probe) == [] and oas.validate_instance(sch_b, probe) != []:
            return ("body", {"kind": "body_contract", "body_schema": sch_a},
                    {"kind": "body_contract", "body_schema": sch_b})
    return None, None, None


def _build_variants(family, unit_b, key, mode, contract_a, contract_b,
                    site_a, site_b, spec_a, spec_b, seed, instances_per_goal):
    initials = []
    for variant in range(instances_per_goal):
        rng = instance_rng(seed, make_id("in", family.family_id, unit_b["unit_id"], key, variant))
        if mode == "params":
            params_a = contracts.task_params(site_a, spec_a)
            params_b = contracts.task_params(site_b, spec_b)
            initial = _synth_assignment(params_a, rng)
            if initial is None or contracts.check_param_assignment(contract_a, initial) != [] \
                    or contracts.check_param_assignment(contract_b, initial) == []:
                return None
            reference = _repair_params(contract_b, initial, rng)
            if reference is None or contracts.check_param_assignment(contract_b, reference) != []:
                return None
            valid_b = {p["name"]: p["schema"] for p in params_b}
            guards = [{"kind": "param", "name": n, "expect": "unchanged"}
                      for n in sorted(initial) if n in valid_b
                      and oas.validate_instance(valid_b[n], initial[n]) == []]
        else:
            try:
                initial = synth.minimal_instance(contract_a["body_schema"], rng, hint="body")
            except OasError:
                return None
            if oas.validate_instance(contract_a["body_schema"], initial) != [] \
                    or oas.validate_instance(contract_b["body_schema"], initial) == []:
                return None
            reference = _repair_body(contract_b["body_schema"], initial, rng)
            if reference is None or oas.validate_instance(contract_b["body_schema"], reference) != []:
                return None
            props_b = contract_b["body_schema"].get("properties") or {}
            guards = [{"kind": "body_prop", "name": n, "expect": "unchanged"}
                      for n in sorted(initial) if isinstance(initial, dict) and n in props_b
                      and oas.validate_instance(props_b[n], initial[n]) == []]
        initials.append((initial, reference, guards))
    return initials
