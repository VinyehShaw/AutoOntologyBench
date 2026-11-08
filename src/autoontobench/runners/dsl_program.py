"""``gk-dsl/1`` — the CODE_DSL arm's restricted constraint language (owner: W13).

A CODE_DSL artifact is ONE JSON document (``program.json`` in the
``code-dsl/1`` bundle): a list of rules over the SAME W6 serialized
Gatekeeper vocabulary v1 facts that ``shacl-rule/1`` consumes. The
interpreter is fixed (this module), pure, deterministic, stdlib-only, and is
executed inside ``autoontobench.safety.sandbox`` via ``dsl_worker``.

    {"schema": "gk-dsl/1",
     "rules": [{"name": str,
                "match": {"action_kind": str, "object_type": str},   # both optional
                "require": [CONDITION, ...]}], ...}                  # 1..8 conditions

    CONDITION = {"op": "eq",     "field": F, "value": V}
              | {"op": "prefix", "field": F, "value": str}
              | {"op": "in",     "field": F, "values": [V, ...]}
              | {"op": "range",  "field": F, "min": num, "max": num}
              | {"op": "count",  "field": F, "min": int}

The operator set equals the shacl-rule/1 fragment the W13 gold-O uses
(equality / prefix / set-membership / integer range / count — the degenerate
presence form for the key-only requiredlabels clause); there is NO negation
operator, and an absent effective value FAILS every condition (uniform
defensive semantics). Effective values mirror the gold-O post-patch
admission clauses: the action's set-override when present, else the object's
current state field (fixed v1 resolution table in ``_override_key``).

Decision contract mirrors the SHACL gold-O structure (target-class coverage
vs in-query applicability): a rule COVERS an action when ``match.action_kind``
fits; a covering rule APPLIES when ``match.object_type`` also fits; only
applied rules evaluate conditions. Any failed condition -> DENY; >= 1
covering rule, none failed -> ALLOW (an inapplicable-but-covering rule
therefore ALLOWs, like the SHACL shape whose Pod clause does not fire for a
Service patch); no covering rule -> UNKNOWN (e.g. ``delete_object``).

PARITY scope and honest limits (multi-container pods, remove paths,
create_object values, regex, cross-object constraints CANNOT be expressed;
ONE documented divergence: absent ``service_type`` passes the gold SHACL
nodeport rule but fails the DSL ``in`` condition) are pinned by
``tests/agent/test_code_dsl.py`` and recorded in ``agent_eval/REPORT.md``.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from autoontobench.runners.dsl_schema import DSL_SCHEMA, ProgramRejected


#: Field -> action-override key resolution (the fixed v1 table).
def _override_key(field: str) -> str | None:
    if field.startswith("image_") and field[6:].isdigit():
        return "set_spec_containers_%s_image" % field[6:]
    if field.startswith("label_"):
        return "set_metadata_labels_%s" % field[6:]
    if field == "replicas":
        return "set_spec_replicas"
    if field == "service_type":
        return "set_spec_type"
    return None


def _effective(record: Mapping[str, Any] | None, action: Mapping[str, Any], field: str) -> Any:
    key = _override_key(field)
    if key is not None and key in action and action[key] is not None:
        return action[key]
    if record is not None:
        return record.get(field)
    return None


def _holds(cond: Mapping[str, Any], effective: Any) -> bool:
    op = cond["op"]
    if op == "count":
        present = 0 if effective is None else 1
        return present >= cond["min"]
    if effective is None:
        return False  # uniform absent-fails (module docstring)
    if op == "eq":
        return bool(effective == cond["value"])
    if op == "prefix":
        return isinstance(effective, str) and effective.startswith(cond["value"])
    if op == "in":
        return effective in cond["values"]
    if op == "range":
        return (
            isinstance(effective, (int, float))
            and not isinstance(effective, bool)
            and cond["min"] <= effective <= cond["max"]
        )
    raise ProgramRejected(["unknown op %r at evaluation (bug)" % (op,)])


def evaluate_program(
    program: Mapping[str, Any],
    facts_state: Mapping[str, Any],
    facts_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate a validated program over serialized facts -> decision record.

    Returns ``{"decision": "ALLOW"|"DENY"|"UNKNOWN", "status": "OK",
    "violations": [...], "unknowns": [...], "report_hash": sha256}``.
    """
    kind = facts_action.get("kind")
    ref = facts_action.get("object")
    oid = ref.get("$ref") if isinstance(ref, Mapping) else None
    record = facts_state.get(oid) if isinstance(oid, str) else None
    if isinstance(record, Mapping) and record is not None:
        object_type = record.get("type")
    else:
        object_type = None
    covering = []
    applied = []
    for rule in program["rules"]:
        match = rule.get("match") or {}
        if match.get("action_kind") is not None and match["action_kind"] != kind:
            continue
        covering.append(rule)
        if match.get("object_type") is not None and match["object_type"] != object_type:
            continue
        applied.append(rule)
    if not covering:
        unknowns = [
            "no rule covers action kind %r (object type %r; uncovered)"
            % (kind, object_type)
        ]
        return {
            "decision": "UNKNOWN",
            "status": "OK",
            "violations": [],
            "unknowns": unknowns,
            "report_hash": _report_hash("UNKNOWN", [], unknowns),
        }
    violations: list[dict[str, Any]] = []
    for rule in applied:
        for cond in rule["require"]:
            effective = _effective(record, facts_action, cond["field"])
            if not _holds(cond, effective):
                violations.append(
                    {
                        "rule": rule["name"],
                        "field": cond["field"],
                        "condition": {k: cond[k] for k in sorted(cond)},
                        "effective": effective,
                        "object": oid,
                    }
                )
    if violations:
        return {
            "decision": "DENY",
            "status": "OK",
            "violations": violations,
            "unknowns": [],
            "report_hash": _report_hash("DENY", violations, []),
        }
    return {
        "decision": "ALLOW",
        "status": "OK",
        "violations": [],
        "unknowns": [],
        "report_hash": _report_hash("ALLOW", [], []),
    }


def _report_hash(decision: str, violations: list[Any], unknowns: list[str]) -> str:
    canonical = json.dumps(
        {"decision": decision, "violations": violations, "unknowns": unknowns},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- the DSL encoding of the four gold-O units (parity partner of gold_o) ----

#: One DSL rule per pinned unit, decision-equivalent to the gold_o.py SHACL
#: clause on the supported fragment (asserted by tests/agent/test_code_dsl.py).
DSL_UNIT_RULES: dict[str, dict[str, Any]] = {
    "allowedrepos": {
        "name": "allowedrepos",
        "match": {"action_kind": "patch_object", "object_type": "Pod"},
        "require": [{"op": "prefix", "field": "image_0", "value": "openpolicyagent/"}],
    },
    "requiredlabels": {
        "name": "requiredlabels",
        "match": {"action_kind": "patch_object", "object_type": "Pod"},
        "require": [{"op": "count", "field": "label_pizza", "min": 1}],
    },
    "replicalimits": {
        "name": "replicalimits",
        "match": {"action_kind": "patch_object", "object_type": "Deployment"},
        "require": [{"op": "range", "field": "replicas", "min": 3, "max": 50}],
    },
    "block-nodeport-services": {
        "name": "block-nodeport-services",
        "match": {"action_kind": "patch_object", "object_type": "Service"},
        "require": [
            {"op": "in", "field": "service_type",
             "values": ["ClusterIP", "LoadBalancer", "ExternalName"]}
        ],
    },
}


def program_for_units(units: tuple[str, ...] | list[str]) -> dict[str, Any]:
    """The composed DSL program for the given units (same composition as gold-O)."""
    from autoontobench.runners.gold_o import UNITS

    unknown = sorted(set(units) - set(UNITS))
    if unknown:
        raise ValueError("unknown units %s (declared: %s)" % (unknown, list(UNITS)))
    return {
        "schema": DSL_SCHEMA,
        "rules": [DSL_UNIT_RULES[u] for u in units],
    }


def program_text_for_units(units: tuple[str, ...] | list[str]) -> str:
    return json.dumps(program_for_units(units), indent=2, sort_keys=True, ensure_ascii=False)


#: The code_dsl arm's initial artifact: an empty program (every action UNKNOWN).
DSL_STUB_TEXT = json.dumps({"schema": DSL_SCHEMA, "rules": []}, indent=2, sort_keys=True)
