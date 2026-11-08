"""``gk-dsl/1`` schema gate (owner: W13): fail-closed validation of the
CODE_DSL arm's agent-authored ``program.json`` BEFORE any evaluation — the
same contract position as ``erule.contract.validator`` for shacl-rule/1.
The language grammar and parity scope are documented in
``autoontobench.runners.dsl_program`` (the interpreter); this module owns only
the structural gate. Anything outside the grammar is rejected with reasons,
never executed.
"""
from __future__ import annotations

from typing import Any, Mapping

DSL_SCHEMA = "gk-dsl/1"
MAX_RULES = 32
MAX_CONDITIONS = 8


class ProgramRejected(ValueError):
    """The DSL program failed schema validation (fail-closed; reasons listed)."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("dsl program rejected: " + "; ".join(self.reasons))


def _check_condition(cond: Mapping[str, Any], reasons: list[str], where: str) -> None:
    op = cond.get("op")
    if op not in ("eq", "prefix", "in", "range", "count"):
        reasons.append("%s: op must be one of eq/prefix/in/range/count, got %r" % (where, op))
        return
    field = cond.get("field")
    if not isinstance(field, str) or not field.replace("_", "").isalnum():
        reasons.append("%s: field must be a lower-alnum/underscore name" % where)
    extra = set(cond) - {"op", "field", "value", "values", "min", "max"}
    if extra:
        reasons.append("%s: unknown keys %s" % (where, sorted(extra)))
    if op in ("eq", "prefix") and "value" not in cond:
        reasons.append("%s: %s requires 'value'" % (where, op))
    if op == "prefix" and "value" in cond and not isinstance(cond["value"], str):
        reasons.append("%s: prefix value must be a string" % where)
    if op == "in":
        values = cond.get("values")
        if not isinstance(values, list) or not values:
            reasons.append("%s: 'in' requires a non-empty 'values' list" % where)
    if op in ("range", "count"):
        bound = cond.get("min")
        if not isinstance(bound, (int, float)) or isinstance(bound, bool):
            reasons.append("%s: %s requires numeric 'min'" % (where, op))
    if op == "range":
        high = cond.get("max")
        if not isinstance(high, (int, float)) or isinstance(high, bool):
            reasons.append("%s: range requires numeric 'max'" % where)
        elif isinstance(cond.get("min"), (int, float)) and high < cond["min"]:
            reasons.append("%s: range min %r exceeds max %r" % (where, cond["min"], high))


def validate_program(doc: Any) -> dict[str, Any]:
    """Fail-closed schema gate; returns the validated program document."""
    reasons: list[str] = []
    if not isinstance(doc, dict):
        raise ProgramRejected(["program must be a JSON object"])
    if doc.get("schema") != DSL_SCHEMA:
        reasons.append("schema must be %r" % DSL_SCHEMA)
    unknown = set(doc) - {"schema", "rules"}
    if unknown:
        reasons.append("unknown top-level keys %s" % sorted(unknown))
    rules = doc.get("rules")
    if not isinstance(rules, list) or len(rules) > MAX_RULES:
        reasons.append("rules must be a list of 0..%d entries" % MAX_RULES)
        rules = []
    names: set[str] = set()
    for index, rule in enumerate(rules):
        where = "rules[%d]" % index
        if not isinstance(rule, dict):
            reasons.append("%s: must be an object" % where)
            continue
        if set(rule) - {"name", "match", "require"}:
            reasons.append("%s: unknown keys %s" % (where, sorted(set(rule) - {"name", "match", "require"})))
        name = rule.get("name")
        if not isinstance(name, str) or not name:
            reasons.append("%s: name must be a non-empty string" % where)
        elif name in names:
            reasons.append("%s: duplicate rule name %r" % (where, name))
        else:
            names.add(name)
        match = rule.get("match", {})
        if not isinstance(match, dict) or set(match) - {"action_kind", "object_type"}:
            reasons.append("%s: match may carry only action_kind/object_type" % where)
        require = rule.get("require")
        if not isinstance(require, list) or not 0 < len(require) <= MAX_CONDITIONS:
            reasons.append("%s: require must be a list of 1..%d conditions" % (where, MAX_CONDITIONS))
            continue
        for cindex, cond in enumerate(require):
            if not isinstance(cond, dict):
                reasons.append("%s.require[%d]: must be an object" % (where, cindex))
            else:
                _check_condition(cond, reasons, "%s.require[%d]" % (where, cindex))
    if reasons:
        raise ProgramRejected(reasons)
    return {"schema": DSL_SCHEMA, "rules": rules}
