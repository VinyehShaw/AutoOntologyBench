"""DMN model + parser for the L03 pinned static runtime (``aob-dmn-static``).

Parses a DMN XML document (namespace-agnostic by ``local-name``; Camunda ships
DMN 1.0/1.1 files) into an immutable model: decisions, decision tables (hit
policy, typed input clauses, output clauses, rules), DRG information
requirements and input data. Only the profile the lane honestly supports is
parsed for evaluation; everything else raises :class:`DmnUnsupportedError` so
the source plugin can mark the unit ``profile_unsupported`` instead of
guessing semantics.

Engine identity: ``aob-dmn-static/1.0.0`` (pure Python, deterministic; the
Camunda Java DMN runtime is unavailable per build rules - no Java/Docker).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree

ENGINE_ID = "aob-dmn-static"
ENGINE_VERSION = "1.0.0"

DMN_SUFFIXES = (".dmn", ".dmn11.xml")
SUPPORTED_HIT_POLICIES = frozenset(
    {"UNIQUE", "FIRST", "ANY", "PRIORITY", "RULE ORDER", "OUTPUT ORDER", "COLLECT"}
)
SUPPORTED_AGGREGATIONS = frozenset({"SUM", "MIN", "MAX", "COUNT"})
SUPPORTED_TYPE_REFS = frozenset({"string", "integer", "long", "double", "boolean", "number"})

_SIMPLE_VARIABLE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


class DmnParseError(ValueError):
    """Malformed DMN XML (unreadable document or missing required slots)."""


class DmnUnsupportedError(ValueError):
    """Valid DMN outside the supported profile (JUEL-EL, dates, FEEL fns)."""


def sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class InputClause:
    id: str
    label: str
    expression: str  # variable name bound at evaluation time
    type_ref: str


@dataclass(frozen=True)
class OutputClause:
    id: str
    label: str
    name: str
    type_ref: str
    values: tuple[str, ...] = ()  # ordered output values for PRIORITY/OUTPUT ORDER


@dataclass(frozen=True)
class Rule:
    id: str
    input_entries: tuple[str, ...]
    output_entries: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class DecisionTable:
    hit_policy: str
    aggregation: str  # "" when absent (only meaningful for COLLECT)
    inputs: tuple[InputClause, ...]
    outputs: tuple[OutputClause, ...]
    rules: tuple[Rule, ...]


@dataclass(frozen=True)
class Decision:
    id: str
    name: str
    table: DecisionTable | None
    required_decision_ids: tuple[str, ...] = ()
    required_input_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Definitions:
    id: str
    name: str
    namespace: str
    decisions: dict[str, Decision]
    input_data: dict[str, str]  # id -> label name
    digest: str
    source_path: str = ""

    def decision_table(self, decision_id: str) -> DecisionTable:
        dec = self.decisions[decision_id]
        if dec.table is None:
            raise DmnUnsupportedError("decision %r has no decisionTable" % decision_id)
        return dec.table


def _local(el: etree._Element) -> str:
    return etree.QName(el).localname


def _children(el: etree._Element, name: str) -> list[etree._Element]:
    return [c for c in el if _local(c) == name]


def _child(el: etree._Element, name: str) -> etree._Element | None:
    for c in el:
        if _local(c) == name:
            return c
    return None


def _text_of(el: etree._Element | None) -> str:
    if el is None:
        return ""
    t = _child(el, "text")
    if t is None or t.text is None:
        return ""
    return t.text.strip()


def parse_definitions(path: str | Path) -> Definitions:
    path = Path(path)
    try:
        # Element-only tree is an invariant of _local/_children; TCK files carry XML comments.
        parser = etree.XMLParser(remove_comments=True, remove_pis=True)
        tree = etree.parse(str(path), parser)
    except (etree.XMLSyntaxError, OSError) as exc:
        raise DmnParseError("%s: %s" % (path, exc)) from exc
    root = tree.getroot()
    if _local(root) != "definitions":
        raise DmnParseError("%s: root is <%s>, not <definitions>" % (path, _local(root)))
    return _build_definitions(root, path)


def _build_definitions(root: etree._Element, path: Path) -> Definitions:
    decisions: dict[str, Decision] = {}
    input_data: dict[str, str] = {}
    for el in root:
        tag = _local(el)
        if tag == "decision":
            dec = _parse_decision(el, path)
            if dec.id in decisions:
                raise DmnParseError("%s: duplicate decision id %r" % (path, dec.id))
            decisions[dec.id] = dec
        elif tag == "inputData":
            input_data[el.get("id") or ""] = el.get("name") or ""
    digest = sha256_json({
        "decisions": {
            did: {
                "name": d.name,
                "required": sorted(d.required_decision_ids),
                "table": _table_digest_payload(d.table) if d.table else None,
            }
            for did, d in sorted(decisions.items())
        },
        "input_data": input_data,
    })
    return Definitions(
        id=root.get("id") or "",
        name=root.get("name") or "",
        namespace=root.nsmap.get(None) or "",
        decisions=decisions,
        input_data=input_data,
        digest=digest,
        source_path=str(path),
    )


def _table_payload(table: DecisionTable) -> dict[str, object]:
    return {
        "hit_policy": table.hit_policy,
        "aggregation": table.aggregation,
        "inputs": [(i.expression, i.type_ref) for i in table.inputs],
        "outputs": [(o.name, o.type_ref, list(o.values)) for o in table.outputs],
        "rules": [
            {"in": list(r.input_entries), "out": list(r.output_entries), "id": r.id}
            for r in table.rules
        ],
    }


def _table_digest_payload(table: DecisionTable) -> str:
    return sha256_json(_table_payload(table))


def table_digest(table: DecisionTable) -> str:
    return _table_digest_payload(table)


def table_from_payload(payload: dict[str, Any]) -> DecisionTable:
    """Rebuild a DecisionTable from the canonical JSON rendering (index-row
    shape: ``inputs``/``outputs``/``rules``/``hit_policy``/``aggregation``).
    This rendering is the agent-visible artifact format for L03 tasks; the
    pinned source XML remains the provenance anchor (``table_digest``)."""

    def _inp(d: dict[str, Any]) -> InputClause:
        return InputClause(
            id=d.get("id", ""), label=d.get("label", ""),
            expression=d["expression"], type_ref=d.get("type_ref", "string"),
        )

    def _out(d: dict[str, Any]) -> OutputClause:
        return OutputClause(
            id=d.get("id", ""), label=d.get("label", ""), name=d.get("name", ""),
            type_ref=d.get("type_ref", "string"),
            values=tuple(d.get("values") or ()),
        )

    def _rule(d: dict[str, Any]) -> Rule:
        return Rule(
            id=d.get("id", ""),
            input_entries=tuple(d.get("input_entries") or ()),
            output_entries=tuple(d.get("output_entries") or ()),
            description=d.get("description", ""),
        )

    hit = (payload.get("hit_policy") or "UNIQUE").strip()
    if hit not in SUPPORTED_HIT_POLICIES:
        raise DmnUnsupportedError("hitPolicy %r not in supported profile" % hit)
    aggregation = (payload.get("aggregation") or "").strip()
    if aggregation and aggregation not in SUPPORTED_AGGREGATIONS:
        raise DmnUnsupportedError("aggregation %r unsupported" % aggregation)
    table = DecisionTable(
        hit_policy=hit,
        aggregation=aggregation,
        inputs=tuple(_inp(d) for d in payload.get("inputs", ())),
        outputs=tuple(_out(d) for d in payload.get("outputs", ())),
        rules=tuple(_rule(d) for d in payload.get("rules", ())),
    )
    for rule in table.rules:
        if len(rule.input_entries) != len(table.inputs) or \
                len(rule.output_entries) != len(table.outputs):
            raise DmnParseError("rule %s arity mismatch" % rule.id)
    return table


def table_to_payload(table: DecisionTable) -> dict[str, Any]:
    """Canonical JSON rendering of a DecisionTable (inverse of
    :func:`table_from_payload`)."""
    return {
        "hit_policy": table.hit_policy,
        "aggregation": table.aggregation,
        "inputs": [
            {"id": c.id, "label": c.label, "expression": c.expression,
             "type_ref": c.type_ref}
            for c in table.inputs
        ],
        "outputs": [
            {"id": o.id, "label": o.label, "name": o.name,
             "type_ref": o.type_ref, "values": list(o.values)}
            for o in table.outputs
        ],
        "rules": [
            {"id": r.id, "input_entries": list(r.input_entries),
             "output_entries": list(r.output_entries), "description": r.description}
            for r in table.rules
        ],
    }


def _parse_decision(el: etree._Element, path: Path) -> Decision:
    required_decisions: list[str] = []
    required_inputs: list[str] = []
    table: DecisionTable | None = None
    for child in el:
        tag = _local(child)
        if tag == "informationRequirement":
            for req in child:
                rtag = _local(req)
                href = (req.get("href") or "").lstrip("#")
                if rtag == "requiredDecision":
                    required_decisions.append(href)
                elif rtag == "requiredInput":
                    required_inputs.append(href)
        elif tag == "decisionTable":
            table = _parse_table(child, path, el.get("id") or "?")
    return Decision(
        id=el.get("id") or "",
        name=el.get("name") or "",
        table=table,
        required_decision_ids=tuple(dict.fromkeys(required_decisions)),
        required_input_ids=tuple(dict.fromkeys(required_inputs)),
    )


def _parse_table(el: etree._Element, path: Path, decision_id: str) -> DecisionTable:
    hit = (el.get("hitPolicy") or "UNIQUE").strip()
    if hit not in SUPPORTED_HIT_POLICIES:
        raise DmnUnsupportedError(
            "%s#%s: hitPolicy %r not in supported profile" % (path, decision_id, hit)
        )
    aggregation = (el.get("aggregation") or "").strip()
    if aggregation and aggregation not in SUPPORTED_AGGREGATIONS:
        raise DmnUnsupportedError(
            "%s#%s: aggregation %r unsupported" % (path, decision_id, aggregation)
        )
    inputs = tuple(_parse_input(c, path, decision_id) for c in _children(el, "input"))
    outputs = tuple(_parse_output(c, path, decision_id) for c in _children(el, "output"))
    rules = tuple(_parse_rule(c, path, decision_id) for c in _children(el, "rule"))
    for rule in rules:
        if len(rule.input_entries) != len(inputs) or len(rule.output_entries) != len(outputs):
            raise DmnParseError(
                "%s#%s: rule %s arity mismatch" % (path, decision_id, rule.id)
            )
    if not outputs:
        raise DmnUnsupportedError("%s#%s: table without output clause" % (path, decision_id))
    return DecisionTable(
        hit_policy=hit, aggregation=aggregation,
        inputs=inputs, outputs=outputs, rules=rules,
    )


def _parse_input(el: etree._Element, path: Path, decision_id: str) -> InputClause:
    ie = _child(el, "inputExpression")
    expr = _text_of(ie)
    type_ref = (ie.get("typeRef") or "string") if ie is not None else "string"
    if not expr:
        raise DmnUnsupportedError(
            "%s#%s: empty inputExpression" % (path, decision_id)
        )
    if not _SIMPLE_VARIABLE.match(expr):
        raise DmnUnsupportedError(
            "%s#%s: inputExpression %r outside profile (JUEL/complex)"
            % (path, decision_id, expr)
        )
    if type_ref not in SUPPORTED_TYPE_REFS:
        raise DmnUnsupportedError(
            "%s#%s: typeRef %r outside profile" % (path, decision_id, type_ref)
        )
    return InputClause(
        id=el.get("id") or "", label=el.get("label") or "",
        expression=expr, type_ref=type_ref,
    )


def _parse_output(el: etree._Element, path: Path, decision_id: str) -> OutputClause:
    values_el = _child(el, "outputValues")
    values: tuple[str, ...] = ()
    if values_el is not None:
        raw = _text_of(values_el)
        if raw:
            values = tuple(v.strip() for v in raw.split(",") if v.strip())
    name = el.get("name") or el.get("label") or el.get("id") or ""
    return OutputClause(
        id=el.get("id") or "", label=el.get("label") or "", name=name,
        type_ref=el.get("typeRef") or "string", values=values,
    )


def _parse_rule(el: etree._Element, path: Path, decision_id: str) -> Rule:
    in_entries = tuple(_text_of(c) for c in _children(el, "inputEntry"))
    out_entries = tuple(_text_of(c) for c in _children(el, "outputEntry"))
    desc_el = _child(el, "description")
    return Rule(
        id=el.get("id") or "",
        input_entries=in_entries,
        output_entries=out_entries,
        description=(desc_el.text or "").strip() if desc_el is not None else "",
    )
