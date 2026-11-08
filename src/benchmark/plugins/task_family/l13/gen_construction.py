"""L13 construction-regime builders: f04 response instance, f08 operation
selection, f09 security requirement enumeration.

Construction tasks publish the contract evidence and constraints but no initial
state; the reference answer is computed from the pinned spec and verified
deterministically (jsonschema for instances, exact-set recompute for
selection/enumeration).
"""
from __future__ import annotations

from typing import Any, Iterator

from benchmark.plugins.source.l13 import contracts, oas, synth
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13.families import (
    FamilySpec, build_ast, instance_rng, make_id,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, SourceSnapshot

MAX_SPECS_SCANNED = 300
# Security requirements are heavily deduplicated per spec (most operations of a
# spec share one requirement), so eligible specs with >=6 *distinct* answers are
# sparse in the sorted inventory. Scan the whole inventory for this family.
SECURITY_MAX_SPECS_SCANNED = 4000
RESPONSE_CODE_PREF = ("200", "201", "202", "203", "206", "2XX", "default")


def _emit(family: FamilySpec, unit: dict[str, Any], coordinate: dict[str, Any],
          contract: dict[str, Any], variants: list[tuple[Any, dict, dict]],
          seed: int) -> Iterator[CandidateTaskAST]:
    """``variants`` are (reference, constraints, witness) tuples."""
    base_goal_id = make_id("bg", family.family_id, unit["unit_id"], coordinate)
    for variant, (reference, constraints, witness) in enumerate(variants):
        instance_id = make_id("in", base_goal_id, variant, seed)
        ast = build_ast(family, unit=unit, coordinate=coordinate, contract=contract,
                        initial=None, reference=reference, guards=[],
                        constraints=constraints, witness=witness, variant=variant)
        yield CandidateTaskAST(
            lane_id="L13", family_id=family.family_id, base_goal_id=base_goal_id,
            instance_id=instance_id, regime=family.regime,
            semantic_profile="openapi-3.0-contract", ast=ast,
            source_unit_ids=(unit["unit_id"],),
        )


def response_instance(plugin, snapshot: SourceSnapshot, family: FamilySpec, *, seed: int,
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
            code = next((c for c in RESPONSE_CODE_PREF if c in site.response_schemas), None)
            if code is None:
                continue
            try:
                schema = oas.normalize_schema(site.response_schemas[code], spec, direction="response")
            except OasError:
                continue
            props = schema.get("properties") or {}
            required = sorted(schema.get("required") or [])
            if schema.get("type") != "object" or not required or len(props) < 2:
                continue
            if oas.validate_instance(schema, {}) == []:
                continue  # contract without teeth
            coordinate = {"path": site.path, "method": site.method,
                          "operation_key": site.operation_key, "code": code}
            contract = {"kind": "response_construction", "code": code,
                        "response_schema": schema}
            optional = [p for p in sorted(props) if p not in required]
            arrays = [p for p in sorted(props)
                      if props[p].get("type") == "array" and isinstance(props[p].get("items"), dict)]
            variants: list[tuple[Any, dict, dict]] = []
            ok = True
            for v in range(instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                try:
                    ref = synth.minimal_instance(schema, rng, hint="resp")
                except OasError:
                    ok = False
                    break
                constraints: dict[str, Any] = {"extra": []}
                mode = v % 3
                if mode == 1 and optional:
                    target = optional[(v // 3) % len(optional)]
                    try:
                        ref[target] = synth.minimal_instance(props[target], rng, hint=target)
                    except OasError:
                        ok = False
                        break
                    constraints["extra"].append({"kind": "include_property", "name": target})
                elif mode == 2 and arrays:
                    target = arrays[(v // 3) % len(arrays)]
                    item_schema = props[target].get("items") or {}
                    try:
                        ref[target] = [synth.minimal_instance(item_schema, rng, hint="%s[%d]" % (target, i))
                                       for i in range(2)]
                    except OasError:
                        ok = False
                        break
                    constraints["extra"].append({"kind": "array_length", "name": target, "length": 2})
                if oas.validate_instance(schema, ref) != []:
                    ok = False
                    break
                witness = {"kind": "relaxed_contract", "relax": {"replace_schema": {}}}
                variants.append((ref, constraints, witness))
            if not ok or len(variants) != instances_per_goal:
                break
            yield from _emit(family, unit, coordinate, contract, variants, seed)
            found += 1
            break


def operation_selection(plugin, snapshot: SourceSnapshot, family: FamilySpec, *, seed: int,
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
        feats = [contracts.op_features(s) for s in oas.iter_operations(spec)]
        if len(feats) < max(6, instances_per_goal):
            continue
        tokens = sorted({t for f in feats for t in f["segments"]})
        predicates: list[dict[str, Any]] = []
        for token in tokens:
            for method in ("get", "post", "put", "delete", "patch"):
                for flag in (True, False):
                    pred = {"path_segment": token, "method": method,
                            "has_required_query_param": flag}
                    answer = contracts.apply_predicate(feats, pred)
                    relaxed = contracts.apply_predicate(feats, pred, relaxed=True)
                    if 1 <= len(answer) <= min(6, len(feats) - 1) and set(answer) < set(relaxed):
                        predicates.append({**pred, "answer": answer, "relaxed_answer": relaxed})
        if len(predicates) < instances_per_goal:
            continue
        coordinate = {"scope": "document"}
        contract = {"kind": "operation_selection",
                    "operations": [{k: f[k] for k in ("operation_key", "path", "method", "parameters")}
                                   for f in feats]}
        variants: list[tuple[Any, dict, dict]] = []
        for v in range(instances_per_goal):
            pred = predicates[v % len(predicates)]
            constraints = {"predicate": {k: pred[k] for k in
                                         ("path_segment", "method", "has_required_query_param")}}
            witness = {"kind": "relaxed_predicate", "drop": "has_required_query_param"}
            variants.append((sorted(pred["answer"]), constraints, witness))
        yield from _emit(family, unit, coordinate, contract, variants, seed)
        found += 1


def security_requirement(plugin, snapshot: SourceSnapshot, family: FamilySpec, *, seed: int,
                         instances_per_goal: int) -> Iterator[CandidateTaskAST]:
    found, scanned = 0, 0
    for unit in plugin.units(snapshot):
        if found >= family.base_goals or scanned >= SECURITY_MAX_SPECS_SCANNED:
            break
        scanned += 1
        try:
            spec = plugin.source.load_spec(unit)
        except OasError:
            continue
        eligible: list[tuple[Any, list[dict[str, list[str]]]]] = []
        seen_ans: set[str] = set()
        for s in oas.iter_operations(spec):
            if not s.security_schemes:
                continue
            ans = contracts.normalized_security(s.security)
            if not ans:
                continue
            key = oas.sha256_json(ans)
            if key in seen_ans:
                continue
            seen_ans.add(key)
            eligible.append((s, ans))
        if len(eligible) < instances_per_goal:
            continue
        base_goal_id = make_id("bg", family.family_id, unit["unit_id"], "security")
        for v in range(instances_per_goal):
            site, ans = eligible[v % len(eligible)]
            instance_id = make_id("in", base_goal_id, v, seed)
            coordinate = {"path": site.path, "method": site.method,
                          "operation_key": site.operation_key}
            contract = {"kind": "security_requirement",
                        "security_schemes": site.security_schemes,
                        "operation_security": site.operation.get("security"),
                        "global_security": spec.get("security")}
            witness = {"kind": "relaxed_contract", "relax": {"strip_security": True}}
            ast = build_ast(family, unit=unit, coordinate=coordinate, contract=contract,
                            initial=None, reference=ans, guards=[],
                            constraints={}, witness=witness, variant=v)
            yield CandidateTaskAST(
                lane_id="L13", family_id=family.family_id, base_goal_id=base_goal_id,
                instance_id=instance_id, regime=family.regime,
                semantic_profile="openapi-3.0-contract", ast=ast,
                source_unit_ids=(unit["unit_id"],),
            )
        found += 1
