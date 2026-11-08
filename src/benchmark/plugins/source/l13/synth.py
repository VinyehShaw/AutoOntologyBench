"""L13 deterministic value synthesis over normalized OpenAPI schemas.

``minimal_instance`` builds the smallest valid instance the schema admits
(deterministic given the seeded RNG); ``violate_value`` builds a value the
schema must reject. Both raise :class:`OasError` on shapes outside the honest
subset (e.g. pattern-constrained strings), and callers skip such sites.
"""
from __future__ import annotations

import hashlib
from typing import Any

from benchmark.plugins.source.l13.oas import OasError, sha256_json


def minimal_instance(schema: dict[str, Any], rng, *, hint: str = "") -> Any:
    if not isinstance(schema, dict):
        raise OasError("schema_not_a_mapping")
    if "const" in schema:
        return schema["const"]
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return schema["enum"][rng.randrange(len(schema["enum"]))]
    for comb in ("anyOf", "oneOf"):
        if isinstance(schema.get(comb), list) and schema[comb]:
            return minimal_instance(schema[comb][0], rng, hint=hint)
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), "null")
    if typ is None:
        if isinstance(schema.get("properties"), dict):
            typ = "object"
        elif isinstance(schema.get("items"), dict):
            typ = "array"
        elif not schema:
            return {}
        else:
            raise OasError("schema without inferable type (keys=%s)" % sorted(schema)[:6])
    if typ == "null":
        return None
    if typ == "boolean":
        return bool(rng.randrange(2))
    if typ in ("integer", "number"):
        return _numeric(schema, rng, integer=(typ == "integer"))
    if typ == "string":
        return _string(schema, rng, hint)
    if typ == "array":
        item_schema = schema.get("items") or {}
        count = min(max(int(schema.get("minItems") or 0), 1), 2)
        return [minimal_instance(item_schema, rng, hint="%s[%d]" % (hint, i)) for i in range(count)]
    if typ == "object":
        props = schema.get("properties") or {}
        out: dict[str, Any] = {}
        for name in sorted(schema.get("required") or []):
            out[name] = minimal_instance(props.get(name) or {}, rng, hint="%s.%s" % (hint, name))
        for name in sorted(props):
            if name not in out and rng.randrange(2):
                try:
                    out[name] = minimal_instance(props[name], rng, hint="%s.%s" % (hint, name))
                except OasError:
                    continue  # optional unsynthesizable property: skip
        return out
    raise OasError("unsupported type %r" % typ)


def _numeric(schema: dict[str, Any], rng, *, integer: bool) -> Any:
    lo = schema.get("minimum")
    if "exclusiveMinimum" in schema:
        lo = schema["exclusiveMinimum"] + (1 if integer else 1e-6)
    hi = schema.get("maximum")
    if "exclusiveMaximum" in schema:
        hi = schema["exclusiveMaximum"] - (1 if integer else 1e-6)
    base = lo if lo is not None else 0
    if hi is not None and base > hi:
        base = hi
    spread = 0 if hi is None else max(0, min(int(hi - base), 3))
    val = base + (rng.randrange(spread + 1) if spread else 0)
    if lo is not None and val < lo:
        val = lo
    if hi is not None and val > hi:
        val = hi
    return int(val) if integer else float(val)


def _string(schema: dict[str, Any], rng, hint: str) -> str:
    if schema.get("pattern"):
        raise OasError("pattern-constrained string unsupported")
    n = int(schema.get("minLength") or 0)
    m = schema.get("maxLength")
    if m is not None and n > m:
        raise OasError("minLength>maxLength")
    fmt = schema.get("format")
    if fmt == "date-time":
        s = "2026-01-0%dT00:00:00Z" % (1 + rng.randrange(9))
    elif fmt == "date":
        s = "2026-01-0%d" % (1 + rng.randrange(9))
    elif fmt == "email":
        s = "a@b.co"
    elif fmt == "uuid":
        s = "00000000-0000-4000-8000-%012d" % rng.randrange(10**12)
    elif fmt in ("uri", "url"):
        s = "https://example.com/x"
    else:
        salt = rng.getrandbits(64)
        s = ("v" + hashlib.sha256(("%s:%d" % (hint, salt)).encode()).hexdigest())[: max(n, 8)]
    while len(s) < n:
        s += "x"
    if m is not None:
        s = s[:m]
    if len(s) < n:
        raise OasError("format value shorter than minLength")
    return s


def violate_value(schema: dict[str, Any], rng) -> Any:
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        bad = "__not_in_enum_%s__" % sha256_json(schema["enum"])[:8]
        if bad in schema["enum"]:
            raise OasError("cannot violate enum")
        return bad
    typ = schema.get("type")
    if isinstance(typ, list):
        typ = next((t for t in typ if t != "null"), None)
    if "minimum" in schema and typ in ("integer", "number"):
        return schema["minimum"] - (1 if typ == "integer" else 0.5)
    if "maxLength" in schema and typ == "string":
        return "x" * (int(schema["maxLength"]) + 2)
    if "minLength" in schema and int(schema["minLength"]) > 0 and typ == "string":
        return ""
    table = {
        "integer": "not-an-integer", "number": "not-a-number", "string": 12345,
        "boolean": "not-a-bool", "object": "not-an-object",
        "array": "not-an-array", "null": "not-null",
    }
    if typ in table:
        return table[typ]
    raise OasError("no violation strategy for schema keys=%s" % sorted(schema)[:6])
