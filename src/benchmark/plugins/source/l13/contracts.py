"""L13 contract-level extraction and checks shared by generators and verifier.

The verifier re-runs these from the pinned spec bytes (digest-checked) so the
contract in every oracle is recomputed, never trusted from the record. All
functions are deterministic and side-effect free.
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.source.l13 import oas
from benchmark.plugins.source.l13.oas import OasError, OperationSite

PARAM_LOCS = ("query", "path")


def task_params(site: OperationSite, spec: dict[str, Any]) -> list[dict[str, Any]]:
    """(query|path) parameters with normalized schemas, in (name, in) order."""
    out = []
    for p in site.parameters:
        if p.get("in") not in PARAM_LOCS:
            continue
        try:
            schema = oas.param_schema(p, spec)
        except OasError:
            continue
        out.append({"name": p["name"], "in": p["in"],
                    "required": bool(p.get("required")), "schema": schema})
    return out


def param_contract(params: list[dict[str, Any]]) -> dict[str, Any]:
    return {"kind": "param_contract",
            "parameters": [{k: p[k] for k in ("name", "in", "required", "schema")} for p in params]}


def check_param_assignment(contract: dict[str, Any], assignment: Any) -> list[str]:
    """Deterministic contract check: sorted violation strings, empty == pass."""
    if not isinstance(assignment, dict):
        return ["assignment_not_an_object"]
    errs = []
    declared = {p["name"] for p in contract["parameters"]}
    for name in sorted(assignment):
        if name not in declared:
            errs.append("undeclared:%s" % name)
    for p in contract["parameters"]:
        if p["required"] and p["name"] not in assignment:
            errs.append("missing:%s" % p["name"])
        elif p["name"] in assignment:
            errs.extend("%s:%s" % (p["name"], e)
                        for e in oas.validate_instance(p["schema"], assignment[p["name"]]))
    return sorted(errs)


def op_features(site: OperationSite) -> dict[str, Any]:
    """The selection-relevant operation surface used by f08 predicates."""
    return {
        "operation_key": site.operation_key,
        "path": site.path,
        "method": site.method,
        "segments": [s for s in site.path.split("/") if s and not s.startswith("{")],
        "has_required_query_param": any(
            p.get("in") == "query" and p.get("required") is True for p in site.parameters),
        "parameters": [{"name": p["name"], "in": p["in"], "required": bool(p.get("required"))}
                       for p in site.parameters if p.get("in") in PARAM_LOCS],
    }


def apply_predicate(features: list[dict[str, Any]], predicate: dict[str, Any],
                    *, relaxed: bool = False) -> list[str]:
    """Sorted operation keys matching the predicate; ``relaxed`` drops the
    required-query-param constraint (the f08 witness contrast)."""
    out = []
    for f in features:
        if predicate["path_segment"] not in f["segments"]:
            continue
        if predicate["method"] != f["method"]:
            continue
        if not relaxed and f["has_required_query_param"] != predicate["has_required_query_param"]:
            continue
        out.append(f["operation_key"])
    return sorted(out)


def normalized_security(raw: Any) -> list[dict[str, list[str]]]:
    """Sorted list of alternatives; each alternative maps scheme -> sorted scopes."""
    if not isinstance(raw, list):
        return []
    out: list[dict[str, list[str]]] = []
    for req in raw:
        if not isinstance(req, dict):
            continue
        alt = {str(k): sorted(str(s) for s in (v or [])) for k, v in req.items()}
        out.append(dict(sorted(alt.items())))
    out.sort(key=lambda a: oas.sha256_json(a))
    return out
