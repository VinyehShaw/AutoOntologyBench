"""L13 f06 cross-API schema mapping (organization regime).

Pairs a target component schema with a source component schema from a
*different provider* whose properties cover the target's required properties
(case-insensitive name match, type-compatible). The public task ships both
normalized schemas, the field mapping, and a source instance; the reference is
the mapped instance, which must validate against the target schema and contain
exactly the mapped keys. The unmapped source instance must fail the target
schema (else the mapping is vacuous and the site is skipped).
"""
from __future__ import annotations

from typing import Any, Iterator

from benchmark.plugins.source.l13 import oas, synth
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13.families import (
    FamilySpec, build_ast, instance_rng, make_id,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, SourceSnapshot

POOL_CAP = 120
PER_PROVIDER_CAP = 3


def _prop_type(schema: dict[str, Any]) -> Any:
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    return typ


def _compatible(target: dict[str, Any], source: dict[str, Any]) -> bool:
    if "enum" in target or "const" in target:
        return False
    tt, st = _prop_type(target), _prop_type(source)
    if tt is None or st is None:
        return True
    if tt != st:
        return False
    if tt == "array":
        return _compatible(target.get("items") or {}, source.get("items") or {})
    return True


def _eligible_schema(spec: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    schemas = (spec.get("components") or {}).get("schemas") or {}
    for name in sorted(schemas):
        try:
            norm = oas.normalize_schema(schemas[name], spec)
        except OasError:
            continue
        props = norm.get("properties") or {}
        required = sorted(norm.get("required") or [])
        if norm.get("type") != "object" or not (1 <= len(required) <= 4) or len(props) < 2:
            continue
        if any("enum" in (props.get(r) or {}) or "const" in (props.get(r) or {}) for r in required):
            continue
        return name, norm
    return None


def _build_pool(plugin, snapshot: SourceSnapshot) -> list[tuple[dict[str, Any], str, dict[str, Any]]]:
    by_provider: dict[str, list[dict[str, Any]]] = {}
    for unit in plugin.units(snapshot):
        by_provider.setdefault(str(unit["metadata"].get("provider")), []).append(unit)
    pool: list[tuple[dict[str, Any], str, dict[str, Any]]] = []
    for provider in sorted(by_provider):
        taken = 0
        for unit in by_provider[provider]:
            try:
                spec = plugin.source.load_spec(unit)
            except OasError:
                continue
            eligible = _eligible_schema(spec)
            if eligible is None:
                continue
            pool.append((unit, eligible[0], eligible[1]))
            taken += 1
            if taken >= PER_PROVIDER_CAP:
                break
        if len(pool) >= POOL_CAP:
            break
    return pool


def schema_mapping(plugin, snapshot: SourceSnapshot, family: FamilySpec, *, seed: int,
                   instances_per_goal: int) -> Iterator[CandidateTaskAST]:
    pool = _build_pool(plugin, snapshot)
    found = 0
    for unit_t, name_t, schema_t in pool:
        if found >= family.base_goals:
            break
        required_t = sorted(schema_t.get("required") or [])
        props_t = schema_t.get("properties") or {}
        for unit_s, name_s, schema_s in pool:
            if unit_s["metadata"].get("provider") == unit_t["metadata"].get("provider"):
                continue
            props_s = schema_s.get("properties") or {}
            lower_s = {k.lower(): k for k in props_s}
            mapping: dict[str, str] = {}
            ok = True
            for r in required_t:
                src = lower_s.get(r.lower())
                if src is None or not _compatible(props_t.get(r) or {}, props_s.get(src) or {}):
                    ok = False
                    break
                mapping[r] = src
            if not ok:
                continue
            initials = []
            for variant in range(instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit_t["unit_id"], name_t, variant))
                try:
                    source_instance = synth.minimal_instance(schema_s, rng, hint="src")
                except OasError:
                    ok = False
                    break
                for src_name in sorted(set(mapping.values())):
                    if src_name in source_instance:
                        continue
                    try:
                        source_instance[src_name] = synth.minimal_instance(
                            props_s.get(src_name) or {}, rng, hint=src_name)
                    except OasError:
                        ok = False
                        break
                if not ok:
                    break
                if oas.validate_instance(schema_t, source_instance) == []:
                    ok = False
                    break
                reference = {t: source_instance[s] for t, s in mapping.items()}
                if oas.validate_instance(schema_t, reference) != []:
                    ok = False
                    break
                initials.append((source_instance, reference))
            if not ok or len(initials) != instances_per_goal:
                continue
            coordinate = {"schema_a": name_s, "schema_b": name_t}
            contract = {"kind": "schema_mapping", "target_schema": schema_t}
            contract_b = {"kind": "schema_mapping", "source_schema": schema_s}
            base_goal_id = make_id("bg", family.family_id, unit_t["unit_id"],
                                   unit_s["unit_id"], coordinate)
            for variant, (initial, reference) in enumerate(initials):
                instance_id = make_id("in", base_goal_id, variant, seed)
                ast = build_ast(family, unit=unit_t, coordinate=coordinate,
                                contract=contract, initial=initial, reference=reference,
                                guards=[], constraints={"mapping": mapping, "exact_keys": True},
                                witness={"kind": "unmapped_fails"}, variant=variant,
                                unit_b=unit_s, contract_b=contract_b)
                yield CandidateTaskAST(
                    lane_id="L13", family_id=family.family_id, base_goal_id=base_goal_id,
                    instance_id=instance_id, regime=family.regime,
                    semantic_profile="openapi-3.0-contract", ast=ast,
                    source_unit_ids=(unit_s["unit_id"], unit_t["unit_id"]),
                )
            found += 1
            break
