"""L13 OpenAPI 3.0.x deterministic primitives: parse, resolve, normalize, validate.

Shared, side-effect-free tooling over pinned specs: local ``$ref`` resolution,
schema normalization into a jsonschema-checkable subset, operation extraction,
and instance validation (jsonschema is the independent engine). Value synthesis
lives in ``synth.py``; contract-level extraction/checks live in ``contracts.py``.

Scope is honest: only ``openapi: 3.0.x`` documents are eligible (3.1 and
swagger 2.0 units are indexed but not used for tasks). Unsupported shapes raise
:class:`OasError` and the site is skipped at propose time.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import jsonschema
import yaml


class OasError(Exception):
    """Deterministic, expected failure: unsupported or malformed spec shape."""


HTTP_METHODS = ("get", "put", "post", "delete", "patch", "head", "options", "trace")
_OAS_ONLY_KEYS = {"discriminator", "xml", "example", "examples", "deprecated", "externalDocs"}


def load_spec(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        doc = yaml.safe_load(text)
    except (yaml.YAMLError, ValueError, TypeError) as exc:
        # PyYAML raises ValueError for out-of-range implicit timestamps (e.g.
        # ``seconds: 60``) rather than a YAMLError; treat both as malformed.
        raise OasError("yaml_parse: %s" % str(exc)[:120]) from exc
    if not isinstance(doc, dict):
        raise OasError("spec_not_a_mapping")
    ver = str(doc.get("openapi", ""))
    if not ver.startswith("3.0"):
        raise OasError("unsupported openapi version %r (3.0.x only)" % ver)
    if not isinstance(doc.get("paths"), dict):
        raise OasError("no paths object")
    return doc


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_ref(ref: str, root: Any) -> Any:
    if not ref.startswith("#/"):
        raise OasError("non-local ref %r" % ref)
    node = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise OasError("dangling ref %r" % ref)
        node = node[part]
    return node


def resolve_refs(node: Any, root: Any, *, max_depth: int = 40, _seen: frozenset = frozenset()) -> Any:
    """Return a new tree with local ``$ref`` objects inlined (cycle-safe)."""
    if max_depth <= 0:
        raise OasError("ref depth exceeded")
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str):
                raise OasError("non-string $ref")
            if ref in _seen:
                return {}  # cyclic schema: permissive placeholder
            target = _resolve_ref(ref, root)
            return resolve_refs(target, root, max_depth=max_depth - 1, _seen=_seen | {ref})
        return {k: resolve_refs(v, root, max_depth=max_depth, _seen=_seen) for k, v in node.items()}
    if isinstance(node, list):
        return [resolve_refs(v, root, max_depth=max_depth, _seen=_seen) for v in node]
    return node


def normalize_schema(schema: Any, root: Any, *, direction: str = "request") -> dict[str, Any]:
    """Resolve refs and normalize an OpenAPI 3.0 schema to a jsonschema subset.

    ``nullable: true`` becomes ``type: [t, "null"]``; ``allOf`` is merged;
    ``anyOf``/``oneOf`` are kept (each branch normalized); OpenAPI-only keys are
    dropped; writeOnly/readOnly properties cannot satisfy required in
    request/response direction respectively.
    """
    if not isinstance(schema, dict):
        raise OasError("schema_not_a_mapping")
    normalized = _norm(resolve_refs(schema, root), direction)
    _assert_compilable_patterns(normalized)
    return normalized


def _assert_compilable_patterns(node: Any) -> None:
    """OpenAPI patterns are ECMA-262; jsonschema compiles them with Python ``re``.

    A pattern that Python cannot compile would make validation raise instead of
    returning errors, so such schemas are rejected as unsupported up front.
    """
    if isinstance(node, dict):
        pattern = node.get("pattern")
        if isinstance(pattern, str):
            try:
                re.compile(pattern)
            except re.error as exc:
                raise OasError("invalid_pattern: %s" % str(exc)[:120]) from exc
        for value in node.values():
            _assert_compilable_patterns(value)
    elif isinstance(node, list):
        for value in node:
            _assert_compilable_patterns(value)


def _norm(node: Any, direction: str) -> Any:
    if isinstance(node, list):
        return [_norm(v, direction) for v in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, val in node.items():
        if key in _OAS_ONLY_KEYS:
            continue
        out[key] = _norm(val, direction)
    if out.pop("nullable", False) and isinstance(out.get("type"), str):
        out["type"] = [out["type"], "null"]
    if "allOf" in out:
        subs = out.pop("allOf")
        merged: dict[str, Any] = {}
        for sub in subs:
            if not isinstance(sub, dict):
                raise OasError("allOf branch not a mapping")
            merged = _merge_schema(merged, sub)
        out = _merge_schema(merged, out)
    if isinstance(out.get("properties"), dict):
        props = out["properties"]
        req = list(out.get("required") or [])
        drop_flag = "writeOnly" if direction == "request" else "readOnly"
        req = [p for p in req if not (isinstance(props.get(p), dict) and props[p].get(drop_flag))]
        if "required" in out or req:
            out["required"] = req
    return out


def _merge_schema(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    for key, val in b.items():
        if key == "properties" and isinstance(val, dict) and isinstance(out.get(key), dict):
            merged = dict(out[key])
            merged.update(val)
            out[key] = merged
        elif key == "required" and isinstance(val, list):
            out[key] = sorted(set(out.get(key) or []) | set(val))
        else:
            out[key] = val
    return out


@dataclass(frozen=True)
class OperationSite:
    path: str
    method: str
    operation_key: str
    operation: dict[str, Any]
    parameters: tuple[dict[str, Any], ...]  # merged path+op level, refs resolved
    request_body_schema: dict[str, Any] | None
    response_schemas: dict[str, dict[str, Any]]
    security: Any  # effective security requirement array (raw, refs resolved)
    security_schemes: dict[str, Any]


def _json_media(content: dict[str, Any]) -> Any:
    media = content.get("application/json")
    if media is None:
        media = next((v for k, v in sorted(content.items()) if "json" in k), None)
    return media


def iter_operations(spec: dict[str, Any]) -> Iterator[OperationSite]:
    schemes = resolve_refs((spec.get("components") or {}).get("securitySchemes") or {}, spec)
    global_sec = spec.get("security")
    for path in sorted(spec["paths"]):
        item = spec["paths"][path]
        if not isinstance(item, dict):
            continue
        path_params = [p for p in (item.get("parameters") or []) if isinstance(p, dict)]
        for method in HTTP_METHODS:
            op = item.get(method)
            if not isinstance(op, dict):
                continue
            merged: dict[tuple[str, str], dict[str, Any]] = {}
            for raw in path_params + [p for p in (op.get("parameters") or []) if isinstance(p, dict)]:
                try:
                    p = resolve_refs(raw, spec)
                except OasError:
                    continue  # malformed/dangling parameter ref: skip that param
                if not isinstance(p, dict) or "name" not in p or "in" not in p:
                    continue
                merged[(p["name"], p["in"])] = p  # op-level overrides path-level
            body_schema = None
            rb = op.get("requestBody")
            if isinstance(rb, dict):
                try:
                    rb_resolved = resolve_refs(rb, spec)
                except OasError:
                    rb_resolved = {}
                media = _json_media(rb_resolved.get("content") or {})
                if isinstance(media, dict) and isinstance(media.get("schema"), dict):
                    body_schema = media["schema"]
            responses: dict[str, dict[str, Any]] = {}
            for code, resp in (op.get("responses") or {}).items():
                if not isinstance(resp, dict):
                    continue
                try:
                    resp_resolved = resolve_refs(resp, spec)
                except OasError:
                    continue  # dangling response ref: that response is unusable
                media = _json_media(resp_resolved.get("content") or {})
                if isinstance(media, dict) and isinstance(media.get("schema"), dict):
                    responses[str(code)] = media["schema"]
            sec = op["security"] if "security" in op else global_sec
            key = op.get("operationId") or ("%s %s" % (method.upper(), path))
            yield OperationSite(
                path=path, method=method, operation_key=str(key), operation=op,
                parameters=tuple(merged[k] for k in sorted(merged)),
                request_body_schema=body_schema, response_schemas=responses,
                security=sec, security_schemes=schemes,
            )


def param_schema(param: dict[str, Any], root: Any) -> dict[str, Any]:
    if "content" in param:
        raise OasError("content parameter unsupported")
    schema = param.get("schema")
    if not isinstance(schema, dict):
        raise OasError("param %r has no schema" % param.get("name"))
    return normalize_schema(schema, root)


def validate_instance(schema: dict[str, Any], instance: Any) -> list[str]:
    """Sorted jsonschema error strings; empty list == valid."""
    validator = jsonschema.Draft7Validator(schema)
    return sorted(
        ("%s: %s" % ("/".join(str(p) for p in e.absolute_path), e.message))
        for e in validator.iter_errors(instance)
    )
