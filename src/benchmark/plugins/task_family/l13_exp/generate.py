"""Deterministic generators for the L13 20K expansion families.

All value synthesis is seeded by sha256 of ``(seed, instance_id)``; sites are
found by sorted scans of the pinned snapshot. No LLM and no network are
involved. Every generator stops once it has emitted its family target, so a
partial final base goal is allowed (variants are independent instances).

Units already used by the legacy 240 are excluded, which is what makes the 910
additions new API surfaces rather than re-instantiations of legacy specs.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Iterator

from benchmark.plugins.source.l13 import contracts, oas, synth
from benchmark.plugins.source.l13.oas import OasError
from benchmark.plugins.task_family.l13_exp import extract
from benchmark.plugins.task_family.l13_exp.families import (
    ExpFamily, emit, instance_rng, make_id,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST

_PREFERRED_CODES = ("200", "201", "202", "203", "204", "206", "2XX", "default")

_MARKER_CACHE: dict[str, str] = {}
_ADDPROPS_MARKERS = (
    "additionalProperties: false", "additionalProperties:false",
    '"additionalProperties": false', '"additionalProperties":false',
)
_UNION_MARKERS = ("oneOf:", '"oneOf"', "anyOf:", '"anyOf"')
_ARRAY_MARKERS = ("type: array", '"type": "array"', '"type":"array"')


def _unit_text(unit: dict[str, Any]) -> str:
    uid = unit["unit_id"]
    if uid not in _MARKER_CACHE:
        from source_registry.fetchers.l13_apis_guru import PINNED_ROOT
        try:
            _MARKER_CACHE[uid] = (PINNED_ROOT / unit["payload_ref"]).read_text(
                encoding="utf-8", errors="replace")
        except OSError:
            _MARKER_CACHE[uid] = ""
    return _MARKER_CACHE[uid]


def _matches(unit: dict[str, Any], markers: tuple[str, ...]) -> bool:
    text = _unit_text(unit)
    return any(m in text for m in markers)

_STRING_POOL = [
    "a", "abc", "Abc123", "abc123", "123", "12345", "test", "A1", "v1",
    "2026-01-01", "2026-01-01T00:00:00Z", "a@b.co", "https://example.com/x",
    "00000000-0000-4000-8000-000000000000", "user_name", "ABC", "0", "x",
]


# -- value synthesis -------------------------------------------------------------
def _pattern_probe(pattern: str) -> str:
    """Best-effort literal string for an ECMA-262 pattern (candidate, not proof)."""
    p = pattern
    if p.startswith("^"):
        p = p[1:]
    if p.endswith("$"):
        p = p[:-1]
    p = p.replace("\\d", "1").replace("\\w", "a").replace("\\s", " ")
    p = re.sub(r"\[([^\]]+)\]", lambda m: _class_pick(m.group(1)), p)
    p = re.sub(r"\{[0-9]+(?:,[0-9]*)?\}", "", p)
    p = p.replace("+", "").replace("*", "").replace("?", "")
    p = p.replace("\\.", ".").replace("\\-", "-").replace("\\/", "/")
    p = p.replace("\\", "")
    return p


def _class_pick(body: str) -> str:
    if body.startswith("^"):
        for cand in "abcXYZ0123456789_-. ":
            if cand not in body[1:]:
                return cand
        return "a"
    m = re.match(r"([A-Za-z0-9])-([A-Za-z0-9])", body)
    if m:
        return m.group(1)
    return body[0] if body else "a"


def _string_candidates(schema: dict[str, Any], rng) -> list[str]:
    out: list[str] = []
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        out.append(_pattern_probe(pattern))
    out.extend(_STRING_POOL)
    for k in range(1, 25):
        out.extend(["1" * k, "0" * k, "a" * k, "A" * k, "x" * k])
    n = int(schema.get("minLength") or 0)
    m = schema.get("maxLength")
    salt = rng.getrandbits(32)
    out.append(("v%08x" % salt))
    for base in list(out):
        if len(base) < n:
            out.append(base + ("x" * (n - len(base))))
        out.append(base * 2)
    if m is not None:
        out = [s for s in out if len(s) <= int(m)] + [s[: int(m)] for s in out]
    fmt = schema.get("format")
    extra = {
        "date-time": "2026-01-01T00:00:00Z", "date": "2026-01-01",
        "email": "a@b.co", "uuid": "00000000-0000-4000-8000-000000000000",
        "uri": "https://example.com/x", "url": "https://example.com/x",
    }.get(fmt)
    if extra:
        out.insert(0, extra)
    return out


def minimal(schema: dict[str, Any], rng, *, hint: str = "") -> Any:
    """Pattern-aware recursive minimal instance (superset of synth.minimal_instance)."""
    if not isinstance(schema, dict):
        raise OasError("schema_not_a_mapping")
    if "const" in schema:
        return schema["const"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][rng.randrange(len(schema["enum"]))]
    for comb in ("anyOf", "oneOf"):
        if isinstance(schema.get(comb), list) and schema[comb]:
            return minimal(schema[comb][0], rng, hint=hint)
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "null")
    if typ == "string" and isinstance(schema.get("pattern"), str):
        for cand in _string_candidates(schema, rng):
            if oas.validate_instance(schema, cand) == []:
                return cand
        raise OasError("no pattern candidate for %r" % schema["pattern"][:80])
    if typ == "object" or (typ is None and isinstance(schema.get("properties"), dict)):
        props = schema.get("properties") or {}
        out: dict[str, Any] = {}
        for name in sorted(schema.get("required") or []):
            out[name] = minimal(props.get(name) or {}, rng, hint="%s.%s" % (hint, name))
        for name in sorted(props):
            if name not in out and rng.randrange(2):
                try:
                    out[name] = minimal(props[name], rng, hint="%s.%s" % (hint, name))
                except OasError:
                    continue
        return out
    if typ == "array":
        item_schema = schema.get("items") or {}
        count = min(max(int(schema.get("minItems") or 0), 1), 2)
        return [minimal(item_schema, rng, hint="%s[%d]" % (hint, i)) for i in range(count)]
    return synth.minimal_instance(schema, rng, hint=hint)


def violate(schema: dict[str, Any], rng) -> Any:
    """Return a value the schema rejects (pattern/format/bound aware)."""
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return synth.violate_value(schema, rng)
    cands: list[Any] = []
    if isinstance(schema.get("pattern"), str):
        cands.extend(["", "!", " ", "###", "__invalid__", "a" * 3])
    if schema.get("minimum") is not None and typ in ("integer", "number"):
        cands.append(schema["minimum"] - (1 if typ == "integer" else 0.5))
    if schema.get("maximum") is not None and typ in ("integer", "number"):
        cands.append(schema["maximum"] + (1 if typ == "integer" else 0.5))
    if schema.get("maxLength") is not None and typ == "string":
        cands.append("x" * (int(schema["maxLength"]) + 2))
    if int(schema.get("minLength") or 0) > 0 and typ == "string":
        cands.append("")
    if schema.get("format") == "email" and typ == "string":
        cands.extend(["not-an-email", "a@"])
    if schema.get("format") == "date" and typ == "string":
        cands.append("not-a-date")
    if schema.get("format") == "date-time" and typ == "string":
        cands.append("not-a-date-time")
    if schema.get("format") == "uuid" and typ == "string":
        cands.append("not-a-uuid")
    if typ == "array":
        return _violate_array(schema, rng)
    cands.extend({
        "integer": ["not-an-integer", 1.5, None],
        "number": ["not-a-number", None],
        "string": [12345, None],
        "boolean": ["not-a-bool", None],
        "object": ["not-an-object", [], None],
        "null": ["not-null"],
    }.get(typ, [None, "not-valid"]))
    for cand in cands:
        if oas.validate_instance(schema, cand) != []:
            return cand
    try:
        return synth.violate_value(schema, rng)
    except OasError as exc:
        raise OasError("no violation strategy for %s" % sorted(schema)[:6]) from exc


def _violate_array(schema: dict[str, Any], rng) -> Any:
    cands: list[Any] = []
    if int(schema.get("minItems") or 0) > 0:
        cands.append([])
    if schema.get("maxItems") is not None:
        item = minimal(schema.get("items") or {}, rng)
        cands.append([item] * (int(schema["maxItems"]) + 2))
    if schema.get("uniqueItems"):
        item = minimal(schema.get("items") or {}, rng)
        cands.append([item, item])
    items = schema.get("items")
    if isinstance(items, dict):
        try:
            cands.append([violate(items, rng)])
        except OasError:
            pass
    cands.extend([{}, None, "not-an-array"])
    for cand in cands:
        if oas.validate_instance(schema, cand) != []:
            return cand
    raise OasError("no array violation strategy")


# -- shared helpers --------------------------------------------------------------
def _load(plugin, unit):
    try:
        return plugin.source.load_spec(unit)
    except OasError:
        return None


def _units(plugin, exclude_units: set[str]) -> Iterator[dict[str, Any]]:
    for unit in plugin.units():
        if unit["unit_id"] in exclude_units:
            continue
        yield unit


def _coordinate(site) -> dict[str, Any]:
    return {"path": site.path, "method": site.method, "operation_key": site.operation_key}


def _assignment(params: list[dict[str, Any]], rng) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in params:
        if not p["required"] and rng.randrange(3) == 0:
            continue
        out[p["name"]] = minimal(p["schema"], rng, hint=p["name"])
    return out


def _emit(family, unit, coordinate, contract, variants, seed, *, unit_b=None,
          contract_b=None, base_goal=None):
    variants = [v for v in variants if v]
    yield from emit(family, unit, coordinate, contract, variants, seed,
                    unit_b=unit_b, contract_b=contract_b, base_goal=base_goal)


def _truncate(variants: list, remaining: int) -> list:
    return variants[:remaining]


# -- x01 pattern parameter repair -------------------------------------------------
def pattern_param(plugin, family: ExpFamily, *, seed: int, target_records: int,
                  exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            params = contracts.task_params(site, spec)
            pats = [p for p in params if isinstance(p["schema"].get("pattern"), str)]
            if len(params) < 2 or not pats:
                continue
            coordinate = _coordinate(site)
            contract = contracts.param_contract(params)
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                target = pats[v % len(pats)]
                try:
                    assignment = _assignment(params, rng)
                    bad = violate(target["schema"], rng)
                    initial = dict(assignment)
                    initial[target["name"]] = bad
                    reference = dict(initial)
                    reference[target["name"]] = minimal(target["schema"], rng, hint=target["name"])
                except OasError:
                    ok = False
                    break
                if (contracts.check_param_assignment(contract, initial) == []
                        or contracts.check_param_assignment(contract, reference) != []):
                    ok = False
                    break
                guards = [{"kind": "param", "name": n, "expect": "unchanged"}
                          for n in sorted(initial) if n != target["name"]]
                variants.append({"initial": initial, "reference": reference, "guards": guards,
                                 "witness": {"kind": "relaxed_contract",
                                             "relax": {"replace_schema": {target["name"]: {}}}}})
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- x02 additionalProperties body repair -----------------------------------------
def _has_addprops_false(schema: dict[str, Any]) -> bool:
    return schema.get("additionalProperties") is False


def additional_property_body(plugin, family: ExpFamily, *, seed: int, target_records: int,
                             exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        if not _matches(unit, _ADDPROPS_MARKERS):
            continue
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            if site.request_body_schema is None:
                continue
            try:
                schema = oas.normalize_schema(site.request_body_schema, spec)
            except OasError:
                continue
            props = schema.get("properties") or {}
            if schema.get("type") != "object" or not props or not _has_addprops_false(schema):
                continue
            coordinate = _coordinate(site)
            contract = {"kind": "body_contract", "body_schema": schema}
            extra = "__undeclared_extra__"
            if extra in props:
                continue
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                try:
                    full = minimal(schema, rng, hint="body")
                except OasError:
                    ok = False
                    break
                if not isinstance(full, dict):
                    ok = False
                    break
                reference = copy.deepcopy(full)
                initial = copy.deepcopy(full)
                initial[extra] = "extra-%d" % v
                if (oas.validate_instance(schema, initial) == []
                        or oas.validate_instance(schema, reference) != []):
                    ok = False
                    break
                guards = [{"kind": "body_prop", "name": n, "expect": "unchanged"}
                          for n in sorted(reference)]
                variants.append({"initial": initial, "reference": reference, "guards": guards,
                                 "witness": {"kind": "relaxed_contract",
                                             "relax": {"body_drop_additional_properties": True}}})
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- x03 nested required body repair ---------------------------------------------
def _nested_required(schema: dict[str, Any]) -> tuple[str, str] | None:
    props = schema.get("properties") or {}
    for name in sorted(props):
        sub = props[name]
        if not isinstance(sub, dict) or sub.get("type") != "object":
            continue
        sub_props = sub.get("properties") or {}
        req = sorted(sub.get("required") or [])
        if req and sub_props:
            return name, req[0]
    return None


def nested_required_body(plugin, family: ExpFamily, *, seed: int, target_records: int,
                         exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            if site.request_body_schema is None:
                continue
            try:
                schema = oas.normalize_schema(site.request_body_schema, spec)
            except OasError:
                continue
            if schema.get("type") != "object" or len(schema.get("properties") or {}) < 2:
                continue
            nested = _nested_required(schema)
            if nested is None:
                continue
            outer, inner = nested
            outer_schema = schema["properties"][outer]
            coordinate = _coordinate(site)
            contract = {"kind": "body_contract", "body_schema": schema}
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                try:
                    full = minimal(schema, rng, hint="body")
                except OasError:
                    ok = False
                    break
                if not isinstance(full, dict):
                    ok = False
                    break
                if not isinstance(full.get(outer), dict):
                    try:
                        full[outer] = minimal(outer_schema, rng, hint=outer)
                    except OasError:
                        ok = False
                        break
                if not isinstance(full.get(outer), dict) or inner not in full[outer]:
                    try:
                        full[outer][inner] = minimal(
                            (outer_schema.get("properties") or {}).get(inner) or {},
                            rng, hint=inner)
                    except OasError:
                        ok = False
                        break
                reference = copy.deepcopy(full)
                initial = copy.deepcopy(full)
                del initial[outer][inner]
                if (oas.validate_instance(schema, initial) == []
                        or oas.validate_instance(schema, reference) != []):
                    ok = False
                    break
                guards = [{"kind": "body_prop", "name": n, "expect": "unchanged"}
                          for n in sorted(reference) if n != outer]
                variants.append({
                    "initial": initial, "reference": reference, "guards": guards,
                    "witness": {"kind": "relaxed_contract",
                                "relax": {"body_nested_remove_required": {"path": [outer, inner]}}},
                })
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- selection helpers -----------------------------------------------------------
def _relax_required(schema: dict[str, Any], instance: Any) -> tuple[dict[str, Any], list[str]] | None:
    req = list(schema.get("required") or [])
    if not req:
        return None
    relaxed = copy.deepcopy(schema)
    relaxed["required"] = []
    if oas.validate_instance(relaxed, instance) == [] and oas.validate_instance(schema, instance) != []:
        return relaxed, req
    return None


def _pick_probe(schemas: dict[str, dict[str, Any]], rng, offset: int):
    names = sorted(schemas)
    for idx in range(len(names)):
        name = names[(offset + idx) % len(names)]
        try:
            inst = minimal(schemas[name], rng, hint="probe")
        except OasError:
            continue
        matches = sorted(k for k, s in schemas.items() if oas.validate_instance(s, inst) == [])
        if 1 <= len(matches) < len(names):
            relax = None
            for other in names:
                if other in matches:
                    continue
                relax = _relax_required(schemas[other], inst)
                if relax is not None:
                    break
            if relax is not None:
                return inst, matches, other, relax[1]
    return None


# -- x04 response status selection -----------------------------------------------
def response_status_selection(plugin, family: ExpFamily, *, seed: int, target_records: int,
                              exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            schemas = extract.response_schema_map(site, spec)
            if len(schemas) < 2:
                continue
            coordinate = _coordinate(site)
            contract = {"kind": "status_selection", "response_schemas": schemas}
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                probe = _pick_probe(schemas, rng, v)
                if probe is None:
                    ok = False
                    break
                inst, matches, other, dropped = probe
                variants.append({
                    "initial": inst, "reference": matches, "guards": [],
                    "constraints": {},
                    "witness": {"kind": "status_relax", "code": other, "drop_required": dropped},
                })
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- x05 request media-type selection --------------------------------------------
def media_type_selection(plugin, family: ExpFamily, *, seed: int, target_records: int,
                         exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            schemas = extract.request_media_schemas(site, spec)
            if len(schemas) < 2:
                continue
            coordinate = _coordinate(site)
            contract = {"kind": "media_type_selection", "media_schemas": schemas}
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                probe = _pick_probe(schemas, rng, v)
                if probe is None:
                    ok = False
                    break
                inst, matches, other, dropped = probe
                variants.append({
                    "initial": inst, "reference": matches, "guards": [],
                    "constraints": {},
                    "witness": {"kind": "media_relax", "media": other, "drop_required": dropped},
                })
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- x06 response header construction --------------------------------------------
def response_header_construction(plugin, family: ExpFamily, *, seed: int, target_records: int,
                                 exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            for code in _PREFERRED_CODES:
                if code not in site.response_schemas:
                    continue
                headers = extract.response_headers(site, spec, code)
                for name in sorted(headers):
                    if emitted >= target_records:
                        break
                    schema = headers[name]
                    coordinate = {**_coordinate(site), "code": code, "header": name}
                    contract = {"kind": "header_construction", "header_name": name,
                                "header_schema": schema}
                    variants, ok = [], True
                    for v in range(family.instances_per_goal):
                        rng = instance_rng(seed, make_id("in", family.family_id,
                                                         unit["unit_id"], coordinate, v))
                        try:
                            reference = minimal(schema, rng, hint=name)
                            bad = violate(schema, rng)
                        except OasError:
                            ok = False
                            break
                        if (oas.validate_instance(schema, reference) != []
                                or oas.validate_instance(schema, bad) == []):
                            ok = False
                            break
                        variants.append({
                            "initial": None, "reference": reference, "guards": [],
                            "constraints": {},
                            "witness": {"kind": "header_relax", "bad_value": bad},
                        })
                    if ok and variants:
                        variants = _truncate(variants, target_records - emitted)
                        yield from _emit(family, unit, coordinate, contract, variants, seed)
                        emitted += len(variants)
                        break
                if emitted >= target_records:
                    break
            if emitted >= target_records:
                break


# -- x07 union response construction ---------------------------------------------
def union_response_construction(plugin, family: ExpFamily, *, seed: int, target_records: int,
                                exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        if not _matches(unit, _UNION_MARKERS):
            continue
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            schemas = extract.response_schema_map(site, spec)
            for code in _PREFERRED_CODES:
                schema = schemas.get(code)
                if not isinstance(schema, dict):
                    continue
                if oas.validate_instance(schema, {}) == [] or oas.validate_instance(schema, None) == []:
                    continue
                branches = schema.get("oneOf") or schema.get("anyOf")
                if not isinstance(branches, list):
                    continue
                branches = [b for b in branches
                            if isinstance(b, dict) and b.get("type") == "object" and b.get("required")]
                if len(branches) < 2:
                    continue
                coordinate = {**_coordinate(site), "code": code}
                contract = {"kind": "response_construction", "code": code,
                            "response_schema": schema}
                variants, ok = [], True
                for v in range(family.instances_per_goal):
                    rng = instance_rng(seed, make_id("in", family.family_id,
                                                     unit["unit_id"], coordinate, v))
                    branch = branches[v % len(branches)]
                    try:
                        inst = minimal(branch, rng, hint="union")
                    except OasError:
                        ok = False
                        break
                    matches = [b for b in branches if oas.validate_instance(b, inst) == []]
                    if oas.validate_instance(schema, inst) != [] or len(matches) != 1:
                        ok = False
                        break
                    variants.append({
                        "initial": None, "reference": inst, "guards": [],
                        "constraints": {},
                        "witness": {"kind": "relaxed_contract",
                                    "relax": {"replace_schema": {}}},
                    })
                if ok and variants:
                    variants = _truncate(variants, target_records - emitted)
                    yield from _emit(family, unit, coordinate, contract, variants, seed)
                    emitted += len(variants)
                    break
            if emitted >= target_records:
                break


# -- x08 response version migration ----------------------------------------------
def _vkey(version: str) -> tuple:
    return tuple((0, int(p)) if p.isdigit() else (1, p)
                 for p in re.split(r"[^0-9A-Za-z]+", version) if p)


def _repair_response(schema: dict[str, Any], initial: Any, rng) -> Any:
    if not isinstance(initial, dict) or schema.get("type") != "object":
        return minimal(schema, rng, hint="resp")
    props = schema.get("properties") or {}
    out: dict[str, Any] = {}
    for name, val in initial.items():
        sub = props.get(name)
        if sub is not None and oas.validate_instance(sub, val) == []:
            out[name] = val
    for name in schema.get("required") or []:
        if name not in out:
            out[name] = minimal(props.get(name) or {}, rng, hint=name)
    if oas.validate_instance(schema, out) != []:
        return minimal(schema, rng, hint="resp")
    return out


def response_version_migration(plugin, family: ExpFamily, *, seed: int, target_records: int,
                               exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    by_api: dict[str, list[dict[str, Any]]] = {}
    for unit in _units(plugin, exclude_units):
        api_path = unit["metadata"].get("api_path")
        if api_path:
            by_api.setdefault(api_path, []).append(unit)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for api_path in sorted(by_api):
        units = sorted(by_api[api_path], key=lambda u: _vkey(u["metadata"]["api_version"]))
        for i in range(len(units) - 1):
            pairs.append((units[i], units[i + 1]))
        if len(units) > 2:
            pairs.append((units[0], units[-1]))
    seen_pairs: set[tuple[str, str]] = set()
    ordered: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for a, b in pairs:
        k = (a["unit_id"], b["unit_id"])
        if k in seen_pairs:
            continue
        seen_pairs.add(k)
        ordered.append((a, b))
    for unit_a, unit_b in ordered:
        if emitted >= target_records:
            break
        spec_a, spec_b = _load(plugin, unit_a), _load(plugin, unit_b)
        if spec_a is None or spec_b is None:
            continue
        ops_a = {(s.path, s.method): s for s in oas.iter_operations(spec_a)}
        ops_b = {(s.path, s.method): s for s in oas.iter_operations(spec_b)}
        for key in sorted(ops_a):
            if emitted >= target_records:
                break
            site_a, site_b = ops_a[key], ops_b.get(key)
            if site_b is None:
                continue
            codes = [c for c in _PREFERRED_CODES
                     if c in site_a.response_schemas and c in site_b.response_schemas]
            for code in codes:
                if emitted >= target_records:
                    break
                try:
                    sch_a = oas.normalize_schema(site_a.response_schemas[code], spec_a,
                                                 direction="response")
                    sch_b = oas.normalize_schema(site_b.response_schemas[code], spec_b,
                                                 direction="response")
                except OasError:
                    continue
                coordinate = {**_coordinate(site_b), "mode": "response", "code": code}
                contract = {"kind": "response_construction", "code": code,
                            "response_schema": sch_b}
                contract_b = {"kind": "response_construction", "code": code,
                              "response_schema": sch_a}
                variants, ok = [], True
                for v in range(family.instances_per_goal):
                    rng = instance_rng(seed, make_id("in", family.family_id,
                                                     unit_b["unit_id"], coordinate, v))
                    try:
                        initial = minimal(sch_a, rng, hint="resp")
                        if (oas.validate_instance(sch_a, initial) != []
                                or oas.validate_instance(sch_b, initial) == []):
                            ok = False
                            break
                        reference = _repair_response(sch_b, initial, rng)
                    except OasError:
                        ok = False
                        break
                    if oas.validate_instance(sch_b, reference) != []:
                        ok = False
                        break
                    props_b = sch_b.get("properties") or {}
                    guards = [{"kind": "body_prop", "name": n, "expect": "unchanged"}
                              for n in sorted(initial) if isinstance(initial, dict)
                              and n in props_b
                              and oas.validate_instance(props_b[n], initial[n]) == []]
                    variants.append({"initial": initial, "reference": reference,
                                     "guards": guards,
                                     "witness": {"kind": "version_contrast"}})
                if ok and variants:
                    variants = _truncate(variants, target_records - emitted)
                    yield from _emit(family, unit_b, coordinate, contract, variants, seed,
                                     unit_b=unit_a, contract_b=contract_b)
                    emitted += len(variants)
                    break  # first usable response code for this operation


# -- x09 array parameter repair ---------------------------------------------------
def array_param(plugin, family: ExpFamily, *, seed: int, target_records: int,
                exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        if not _matches(unit, _ARRAY_MARKERS):
            continue
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            params = contracts.task_params(site, spec)
            arrays = [p for p in params if p["schema"].get("type") == "array"]
            if len(params) < 2 or not arrays:
                continue
            coordinate = _coordinate(site)
            contract = contracts.param_contract(params)
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                target = arrays[v % len(arrays)]
                try:
                    assignment = _assignment(params, rng)
                    bad = _violate_array(target["schema"], rng)
                    initial = dict(assignment)
                    initial[target["name"]] = bad
                    reference = dict(initial)
                    reference[target["name"]] = minimal(target["schema"], rng, hint=target["name"])
                except OasError:
                    ok = False
                    break
                if (contracts.check_param_assignment(contract, initial) == []
                        or contracts.check_param_assignment(contract, reference) != []):
                    ok = False
                    break
                guards = [{"kind": "param", "name": n, "expect": "unchanged"}
                          for n in sorted(initial) if n != target["name"]]
                variants.append({"initial": initial, "reference": reference, "guards": guards,
                                 "witness": {"kind": "relaxed_contract",
                                             "relax": {"replace_schema": {target["name"]: {}}}}})
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- x11 compound parameter repair ------------------------------------------------
def compound_param(plugin, family: ExpFamily, *, seed: int, target_records: int,
                   exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            params = contracts.task_params(site, spec)
            required = [p for p in params if p["required"]]
            if len(params) < 3 or not required:
                continue
            coordinate = _coordinate(site)
            contract = contracts.param_contract(params)
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                req = required[v % len(required)]
                others = [p for p in params if p["name"] != req["name"]]
                bad_target = others[v % len(others)]
                try:
                    assignment = _assignment(params, rng)
                    bad = violate(bad_target["schema"], rng)
                    reference = dict(assignment)
                    reference[req["name"]] = minimal(req["schema"], rng, hint=req["name"])
                    reference[bad_target["name"]] = minimal(bad_target["schema"], rng,
                                                           hint=bad_target["name"])
                    initial = dict(reference)
                    initial.pop(req["name"], None)
                    initial[bad_target["name"]] = bad
                except OasError:
                    ok = False
                    break
                errs = contracts.check_param_assignment(contract, initial)
                if not any(e.startswith("missing:%s" % req["name"]) for e in errs):
                    ok = False
                    break
                if not any(e.startswith("%s:" % bad_target["name"]) for e in errs):
                    ok = False
                    break
                if contracts.check_param_assignment(contract, reference) != []:
                    ok = False
                    break
                guards = [{"kind": "param", "name": n, "expect": "unchanged"}
                          for n in sorted(reference)
                          if n not in (req["name"], bad_target["name"])]
                variants.append({
                    "initial": initial, "reference": reference, "guards": guards,
                    "witness": {"kind": "relaxed_contract",
                                "relax": {"remove_required": [req["name"]],
                                          "replace_schema": {bad_target["name"]: {}}}},
                })
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


# -- x12 array-constrained body repair --------------------------------------------
def _array_property(schema: dict[str, Any]) -> str | None:
    props = schema.get("properties") or {}
    for name in sorted(props):
        sub = props[name]
        if not isinstance(sub, dict) or sub.get("type") != "array":
            continue
        if any(k in sub for k in ("minItems", "maxItems", "uniqueItems", "items")):
            return name
    return None


def array_body(plugin, family: ExpFamily, *, seed: int, target_records: int,
               exclude_units: set[str]) -> Iterator[CandidateTaskAST]:
    emitted = 0
    for unit in _units(plugin, exclude_units):
        if emitted >= target_records:
            break
        if not _matches(unit, _ARRAY_MARKERS):
            continue
        spec = _load(plugin, unit)
        if spec is None:
            continue
        for site in oas.iter_operations(spec):
            if emitted >= target_records:
                break
            if site.request_body_schema is None:
                continue
            try:
                schema = oas.normalize_schema(site.request_body_schema, spec)
            except OasError:
                continue
            props = schema.get("properties") or {}
            if schema.get("type") != "object" or len(props) < 2:
                continue
            target = _array_property(schema)
            if target is None:
                continue
            coordinate = _coordinate(site)
            contract = {"kind": "body_contract", "body_schema": schema}
            variants, ok = [], True
            for v in range(family.instances_per_goal):
                rng = instance_rng(seed, make_id("in", family.family_id, unit["unit_id"], coordinate, v))
                try:
                    full = minimal(schema, rng, hint="body")
                    bad = _violate_array(props[target], rng)
                except OasError:
                    ok = False
                    break
                if not isinstance(full, dict):
                    ok = False
                    break
                reference = copy.deepcopy(full)
                initial = copy.deepcopy(full)
                initial[target] = bad
                if (oas.validate_instance(schema, initial) == []
                        or oas.validate_instance(schema, reference) != []):
                    ok = False
                    break
                guards = [{"kind": "body_prop", "name": n, "expect": "unchanged"}
                          for n in sorted(reference) if n != target]
                variants.append({"initial": initial, "reference": reference, "guards": guards,
                                 "witness": {"kind": "relaxed_contract",
                                             "relax": {"body_replace_schema": {target: {}}}}})
            if ok and variants:
                variants = _truncate(variants, target_records - emitted)
                yield from _emit(family, unit, coordinate, contract, variants, seed)
                emitted += len(variants)
                break


__all__ = [
    "minimal", "violate", "pattern_param", "additional_property_body",
    "nested_required_body", "response_status_selection", "media_type_selection",
    "response_header_construction", "union_response_construction",
    "response_version_migration", "array_param", "compound_param", "array_body",
]