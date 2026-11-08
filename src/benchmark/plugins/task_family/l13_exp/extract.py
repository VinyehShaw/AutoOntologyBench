"""Deterministic OpenAPI surface extractors shared by the L13 expansion
generators and verifier.

Everything here is recomputed from the pinned spec bytes. The verifier calls
the same functions as the generator, so a contract cannot drift between the
public task and the private oracle without the digest check catching it.

Only OpenAPI 3.0.x JSON-media surfaces that :mod:`benchmark.plugins.source.l13.oas`
can normalize are returned; unsupported shapes raise ``OasError`` and callers
skip the site.
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.source.l13 import oas
from benchmark.plugins.source.l13.oas import OasError, OperationSite


def _resolve(node: Any, spec: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(node, dict):
        return {}
    try:
        resolved = oas.resolve_refs(node, spec)
    except OasError:
        return {}
    return resolved if isinstance(resolved, dict) else {}


def response_schema_map(site: OperationSite, spec: dict[str, Any],
                        *, direction: str = "response") -> dict[str, dict[str, Any]]:
    """Normalized schema per response status code (JSON media only)."""
    out: dict[str, dict[str, Any]] = {}
    for code in sorted(site.response_schemas):
        try:
            out[code] = oas.normalize_schema(site.response_schemas[code], spec,
                                             direction=direction)
        except OasError:
            continue
    return out


def request_media_schemas(site: OperationSite, spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalized schema per request media type (all media types with a schema)."""
    rb = _resolve(site.operation.get("requestBody"), spec)
    content = rb.get("content") or {}
    if not isinstance(content, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for media in sorted(content):
        media_obj = content.get(media)
        if not isinstance(media_obj, dict):
            continue
        schema = media_obj.get("schema")
        if not isinstance(schema, dict):
            continue
        try:
            out[str(media)] = oas.normalize_schema(schema, spec, direction="request")
        except OasError:
            continue
    return out


def response_headers(site: OperationSite, spec: dict[str, Any],
                     code: str) -> dict[str, dict[str, Any]]:
    """Normalized schema per response header for one status code."""
    responses = site.operation.get("responses") or {}
    if not isinstance(responses, dict):
        return {}
    resp = _resolve(responses.get(code), spec)
    headers = resp.get("headers") or {}
    if not isinstance(headers, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for name in sorted(headers):
        hdr = _resolve(headers.get(name), spec)
        schema = hdr.get("schema")
        if not isinstance(schema, dict):
            continue
        try:
            out[str(name)] = oas.normalize_schema(schema, spec, direction="response")
        except OasError:
            continue
    return out


__all__ = [
    "response_schema_map",
    "request_media_schemas",
    "response_headers",
]