"""L08 ELM authoring + CQL-style text rendering.

Builds ELM JSON libraries for the lane's restricted clinical-rule language and
renders the matching CQL-style text. The ELM JSON is the executable artifact
fed to the vendored cql-execution engine (Node); the text is the agent-facing
rule artifact.

Convention (declared in every task and in the lane REPORT): rules are
expressed over explicit FHIR R4 JSON element paths (e.g.
``Observation.valueQuantity.value``). Choice-typed elements carry exactly one
variant in lane data; FHIR primitives compare via their ``value`` element;
date comparisons are day-precision; no FHIRHelpers implicit conversions are
in scope (the Java CQL-to-ELM translator is absent on the build host, so the
lane pins rule semantics to the explicit-path form and differentially
verifies ELM execution against an independent Python recomputation).

Rule AST (``rule`` dict) kinds:
- ``obs_threshold``  exists Observation(code) with valueQuantity op threshold
- ``obs_recency``    obs_threshold restricted to effectiveDateTime >= cutoff
- ``obs_latest``     the most-recent Observation(code) satisfies op threshold
- ``med_active``     exists MedicationRequest(med, status active, authoredOn >= cutoff)
- ``cond_active``    exists Condition(cond, clinicalStatus active[, onset >= cutoff])
- ``enc_count``      Count(Encounter(class, start >= cutoff)) >= min_count
- ``combined``       obs_threshold AND cond_active AND NOT cond_active(excl)
- ``recency_excl``   obs_recency AND NOT cond_active(excl)  (repair family base)
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

FHIR_URI = "http://hl7.org/fhir"
FHIR = "{%s}" % FHIR_URI
SYS = "{urn:hl7-org:elm-types:r1}"

OPS = {">=": "GreaterOrEqual", ">": "Greater", "<=": "LessOrEqual", "<": "Less", "=": "Equal"}
OP_PHRASE = {">=": "at or above", ">": "strictly above", "<=": "at or below", "<": "strictly below"}

REF_DATE = "2026-10-01"  # frozen lane reference date (no wall-clock expressions)


def cutoff_date(rule: dict[str, Any]) -> str:
    ref = date.fromisoformat(rule.get("ref_date", REF_DATE))
    return (ref - timedelta(days=int(rule["lookback_days"]))).isoformat()


# ---------------------------------------------------------------- ELM helpers

def _lit(value: Any, value_type: str) -> dict[str, Any]:
    return {"valueType": "%s%s" % (SYS, value_type), "value": str(value), "type": "Literal"}


def _int_lit(v: int) -> dict[str, Any]:
    return _lit(v, "Integer")


def _dec_lit(v: float | int) -> dict[str, Any]:
    return _lit(v, "Decimal")


def _str_lit(v: str) -> dict[str, Any]:
    return _lit(v, "String")


def _dt_lit(iso: str) -> dict[str, Any]:
    y, m, d = (int(x) for x in iso.split("-"))
    return {"type": "DateTime", "year": _int_lit(y), "month": _int_lit(m), "day": _int_lit(d)}


def _prop(path: str, source: dict[str, Any]) -> dict[str, Any]:
    return {"path": path, "type": "Property", "source": source}


def _alias(name: str) -> dict[str, Any]:
    return {"name": name, "type": "AliasRef"}


def _retrieve(res: str) -> dict[str, Any]:
    return {"type": "Retrieve", "dataType": "%s%s" % (FHIR, res)}


def _coded_retrieve(res: str, code_name: str) -> dict[str, Any]:
    return {
        "type": "Retrieve",
        "dataType": "%s%s" % (FHIR, res),
        "codeProperty": "code",
        "codeComparator": "in",
        "codes": {"type": "ToList", "operand": {"type": "CodeRef", "name": code_name}},
    }


def _query(sources: list[dict[str, Any]], where: dict[str, Any] | None = None) -> dict[str, Any]:
    q: dict[str, Any] = {"type": "Query", "source": sources}
    if where is not None:
        q["where"] = where
    return q


def _src(alias: str, expression: dict[str, Any]) -> dict[str, Any]:
    return {"alias": alias, "expression": expression}


def _cmp(op: str, a: dict[str, Any], b: dict[str, Any], precision: str | None = None) -> dict[str, Any]:
    e: dict[str, Any] = {"type": OPS[op], "operand": [a, b]}
    if precision:
        e["precision"] = precision
    return e


def _exists(operand: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Exists", "operand": operand}


def _count(source: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Count", "source": source}


def _and(*ops: dict[str, Any]) -> dict[str, Any]:
    return {"type": "And", "operand": list(ops)}


def _not(op: dict[str, Any]) -> dict[str, Any]:
    return {"type": "Not", "operand": op}


# ------------------------------------------------------------- rule fragments

def _obs_value_cmp(rule: dict[str, Any], alias: str) -> dict[str, Any]:
    obs = rule["obs"]
    return _and(
        _cmp(rule["op"], _prop("valueQuantity.value.value", _alias(alias)), _dec_lit(rule["threshold"])),
        _cmp("=", _prop("valueQuantity.unit.value", _alias(alias)), _str_lit(obs["unit"])),
    )


def _obs_threshold_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _exists(_query([_src("O", _coded_retrieve("Observation", "TargetObs"))],
                          _obs_value_cmp(rule, "O")))


def _obs_recency_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _exists(_query([_src("O", _coded_retrieve("Observation", "TargetObs"))],
                          _and(_obs_value_cmp(rule, "O"),
                               _cmp(">=", _prop("effectiveDateTime.value", _alias("O")),
                                    _dt_lit(cutoff_date(rule))))))


def _obs_latest_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _exists(_query(
        [_src("O", _coded_retrieve("Observation", "TargetObs"))],
        _and(_obs_value_cmp(rule, "O"),
             _not(_exists(_query(
                 [_src("Later", _coded_retrieve("Observation", "TargetObs"))],
                 _cmp(">", _prop("effectiveDateTime.value", _alias("Later")),
                      _prop("effectiveDateTime.value", _alias("O")), "Day")))))))


def _med_active_expr(rule: dict[str, Any]) -> dict[str, Any]:
    med = rule["med"]
    return _exists(_query(
        [_src("M", _retrieve("MedicationRequest"))],
        _and(
            _exists(_query([_src("MC", _prop("coding", _prop("medicationCodeableConcept", _alias("M"))))],
                           _and(_cmp("=", _prop("code.value", _alias("MC")), _str_lit(med["code"])),
                                _cmp("=", _prop("system.value", _alias("MC")), _str_lit(med["system"]))))),
            _cmp("=", _prop("status.value", _alias("M")), _str_lit("active")),
            _cmp(">=", _prop("authoredOn.value", _alias("M")), _dt_lit(cutoff_date(rule))),
        )))


def _cond_active_expr(cond: dict[str, Any], *, with_onset_cutoff: str | None,
                      code_name: str = "TargetCond") -> dict[str, Any]:
    clauses = [_exists(_query([_src("CS", _prop("coding", _prop("clinicalStatus", _alias("C"))))],
                              _cmp("=", _prop("code.value", _alias("CS")), _str_lit("active"))))]
    if with_onset_cutoff is not None:
        clauses.append(_cmp(">=", _prop("onsetDateTime.value", _alias("C")), _dt_lit(with_onset_cutoff)))
    return _exists(_query([_src("C", _coded_retrieve("Condition", code_name))], _and(*clauses)))


def _cond_active_from_rule(rule: dict[str, Any]) -> dict[str, Any]:
    cutoff = cutoff_date(rule) if rule.get("lookback_days") is not None else None
    return _cond_active_expr(rule["cond"], with_onset_cutoff=cutoff)


def _enc_count_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _cmp(">=", _count(_query(
        [_src("E", _retrieve("Encounter"))],
        _and(_cmp("=", _prop("class.code.value", _alias("E")), _str_lit(rule["enc_class"])),
             _cmp(">=", _prop("period.start.value", _alias("E")), _dt_lit(cutoff_date(rule)))))),
        _int_lit(rule["min_count"]))


def _combined_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _and(
        _obs_threshold_expr(rule),
        _cond_active_expr(rule["cond"], with_onset_cutoff=None),
        _not(_cond_active_expr(rule["excl_cond"], with_onset_cutoff=None, code_name="ExclCond")),
    )


def _recency_excl_expr(rule: dict[str, Any]) -> dict[str, Any]:
    expr = _obs_recency_expr(rule)
    if rule.get("exclude", True):
        expr = _and(expr, _not(_cond_active_expr(rule["excl_cond"], with_onset_cutoff=None,
                                                 code_name="ExclCond")))
    return expr


_EXPR_BY_KIND = {
    "obs_threshold": _obs_threshold_expr,
    "obs_recency": _obs_recency_expr,
    "obs_latest": _obs_latest_expr,
    "med_active": _med_active_expr,
    "cond_active": _cond_active_from_rule,
    "enc_count": _enc_count_expr,
    "combined": _combined_expr,
    "recency_excl": _recency_excl_expr,
}


def rule_expression(rule: dict[str, Any]) -> dict[str, Any]:
    return _EXPR_BY_KIND[rule["kind"]](rule)


def _rule_codes(rule: dict[str, Any]) -> list[dict[str, Any]]:
    """Code definitions needed by the library (deduped, deterministic order)."""
    out: dict[str, dict[str, Any]] = {}

    def add(code: dict[str, Any], name: str) -> None:
        if name not in out:
            out[name] = {"name": name, "id": code["code"], "display": code["display"],
                         "codeSystem": {"name": code["cs_name"]}, "accessLevel": "Public"}

    if rule.get("obs"):
        add(rule["obs"], "TargetObs")
    if rule.get("cond"):
        add(rule["cond"], "TargetCond")
    if rule.get("excl_cond"):
        add(rule["excl_cond"], "ExclCond")
    return [out[k] for k in sorted(out)]


def _rule_code_systems(rule: dict[str, Any]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}

    def add(code: dict[str, Any]) -> None:
        name = code["cs_name"]
        if name not in seen:
            seen[name] = {"name": name, "id": code["system"], "accessLevel": "Public"}

    for key in ("obs", "cond", "excl_cond"):
        if rule.get(key):
            add(rule[key])
    return [seen[k] for k in sorted(seen)]


def build_elm(rule: dict[str, Any], *, library_id: str = "EligibilityCheck") -> dict[str, Any]:
    """Full ELM JSON library for the rule (executable by cql-execution)."""
    return {
        "library": {
            "identifier": {"id": library_id, "version": "1.0.0"},
            "schemaIdentifier": {"id": "urn:hl7-org:elm", "version": "r1"},
            "usings": {"def": [
                {"localIdentifier": "System", "uri": "urn:hl7-org:elm-types:r1"},
                {"localIdentifier": "FHIR", "uri": FHIR_URI, "version": "4.0.1"},
            ]},
            "codeSystems": {"def": _rule_code_systems(rule)},
            "codes": {"def": _rule_codes(rule)},
            "statements": {"def": [
                {"name": "Patient", "context": "Patient",
                 "expression": {"type": "SingletonFrom", "operand": _retrieve("Patient")}},
                {"name": "Eligible", "context": "Patient", "accessLevel": "Public",
                 "expression": rule_expression(rule)},
            ]},
        }
    }


# ------------------------------------------------------------ CQL text render

def _code_decl(code: dict[str, Any], name: str) -> str:
    return 'code "%s": \'%s\' from "%s" display \'%s\'' % (
        name, code["code"], code["cs_name"], code["display"])


def _obs_where(rule: dict[str, Any], alias: str) -> str:
    obs = rule["obs"]
    return ("%s.valueQuantity.value %s %s\n"
            "      and %s.valueQuantity.unit.value = '%s'" % (
                alias, rule["op"], _fmt_num(rule["threshold"]), alias, obs["unit"]))


def _fmt_num(v: float | int) -> str:
    return str(v)


def render_cql(rule: dict[str, Any]) -> str:
    """CQL-style text over explicit FHIR JSON element paths (see module docstring)."""
    lines: list[str] = []
    kind = rule["kind"]
    ref = rule.get("ref_date", REF_DATE)
    lines.append('library "EligibilityCheck" version \'1.0.0\'')
    lines.append("using FHIR version '4.0.1'")
    lines.append("")
    for cs in _rule_code_systems(rule):
        lines.append('codesystem "%s": \'%s\'' % (cs["name"], cs["id"]))
    for code, name in _code_pairs(rule):
        lines.append(_code_decl(code, name))
    lines.append("")
    lines.append("context Patient")
    lines.append("")

    if kind == "obs_threshold":
        lines.append("// Semantics: day-precision date comparisons; unit must match.")
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O").replace("\n      and", "\n      and"))
        lines.append(")")
    elif kind == "obs_recency":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Window: effectiveDateTime on or after %s (day precision)." % cutoff_date(rule))
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O"))
        lines.append("      and O.effectiveDateTime.value >= @%s" % cutoff_date(rule))
        lines.append(")")
    elif kind == "obs_latest":
        lines.append("// Semantics: the MOST RECENT Observation of the target code (day-precision")
        lines.append("// dates; ties both count as most recent) must satisfy the value condition.")
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O"))
        lines.append("      and not exists ([Observation: \"TargetObs\"] Later")
        lines.append("        where Later.effectiveDateTime.value > O.effectiveDateTime.value))")
    elif kind == "med_active":
        med = rule["med"]
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append('define "Eligible":')
        lines.append("  exists ([MedicationRequest] M")
        lines.append("    where exists (M.medicationCodeableConcept.coding MC")
        lines.append("          where MC.code.value = '%s' and MC.system.value = '%s')" % (
            med["code"], med["system"]))
        lines.append("      and M.status.value = 'active'")
        lines.append("      and M.authoredOn.value >= @%s)" % cutoff_date(rule))
    elif kind == "cond_active":
        cond = rule["cond"]
        lines.append("// clinicalStatus must contain a coding with code 'active'.")
        if rule.get("lookback_days") is not None:
            lines.append("// Reference date: %s. Lookback: %d days (inclusive onset start)." % (
                ref, rule["lookback_days"]))
        lines.append('define "Eligible":')
        lines.append("  exists ([Condition: \"TargetCond\"] C")
        lines.append("    where exists (C.clinicalStatus.coding CS where CS.code.value = 'active')")
        if rule.get("lookback_days") is not None:
            lines.append("      and C.onsetDateTime.value >= @%s" % cutoff_date(rule))
        lines.append(")")
    elif kind == "enc_count":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append('define "Eligible":')
        lines.append("  Count([Encounter] E")
        lines.append("    where E.class.code.value = '%s'" % rule["enc_class"])
        lines.append("      and E.period.start.value >= @%s) >= %d" % (
            cutoff_date(rule), rule["min_count"]))
    elif kind == "combined":
        lines.append("// Inclusion lab criterion AND active inclusion condition AND NOT active")
        lines.append("// exclusion condition (clinicalStatus coding code 'active').")
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where %s)" % _obs_where(rule, "O"))
        lines.append("    and exists ([Condition: \"TargetCond\"] C")
        lines.append("      where exists (C.clinicalStatus.coding CS where CS.code.value = 'active'))")
        lines.append("    and not exists ([Condition: \"ExclCond\"] X")
        lines.append("      where exists (X.clinicalStatus.coding XS where XS.code.value = 'active'))")
    elif kind == "recency_excl":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        if rule.get("exclude", True):
            lines.append("// Exclusion: active %s conditions are excluded." % rule["excl_cond"]["display"])
        lines.append('define "Flagged":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O"))
        lines.append("      and O.effectiveDateTime.value >= @%s)" % cutoff_date(rule))
        if rule.get("exclude", True):
            lines.append("    and not exists ([Condition: \"ExclCond\"] C")
            lines.append("      where exists (C.clinicalStatus.coding CS where CS.code.value = 'active'))")
    else:  # pragma: no cover
        raise ValueError("unknown rule kind %r" % kind)
    return "\n".join(lines) + "\n"


def _code_pairs(rule: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    pairs: list[tuple[dict[str, Any], str]] = []
    if rule.get("obs"):
        pairs.append((rule["obs"], "TargetObs"))
    if rule.get("cond"):
        pairs.append((rule["cond"], "TargetCond"))
    if rule.get("excl_cond"):
        pairs.append((rule["excl_cond"], "ExclCond"))
    return pairs


def intent_text(rule: dict[str, Any]) -> str:
    """Natural-language clinical intent for repair tasks (fully determines params)."""
    obs = rule["obs"]
    ref = rule.get("ref_date", REF_DATE)
    base = ("Patients are flagged when they have a %s result %s %s %s recorded "
            "on or after %s (the start of the %d-day lookback window before the "
            "reference date %s)." % (
                obs["display"], OP_PHRASE[rule["op"]], _fmt_num(rule["threshold"]), obs["unit"],
                cutoff_date(rule), rule["lookback_days"], ref))
    if rule.get("exclude", True):
        base += (" Patients with an active %s condition (clinicalStatus coding code "
                 "'active') are excluded even when the lab criterion holds." % rule["excl_cond"]["display"])
    else:
        base += " There is no condition-based exclusion in this rule."
    return base
