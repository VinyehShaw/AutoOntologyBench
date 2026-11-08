"""L08 20K-expansion ELM authoring + CQL rendering (NEW rule structures).

This module authors ELM JSON libraries for the ten NEW L08 rule kinds
introduced by the 20,000-task expansion (dataset
``autoontobench-bank-20000-candidate-dev1``). It shares the pinned Node /
cql-execution runtime with the legacy lane but the rule structures here are
genuinely new mechanisms, not re-parameterizations of the legacy eight:

- ``obs_all``     universal quantification: EVERY target obs must pass (legacy
                  used existential only) and at least one must exist
- ``obs_count``   counting over observations (value+window filtered); legacy
                  counted only Encounters
- ``obs_trend``   temporal join between two target observations (failing side
                  strictly before passing side, or the reverse); legacy only
                  inspected the single latest value
- ``med_pair``    two-medication conjunction (polypharmacy window); legacy
                  matched a single medication
- ``cond_resolved`` resolved-status semantics over Condition.abatementDateTime
                  (a FHIR element the legacy lane never used; legacy matched
                  'active' only)
- ``enc_gap``     negation-of-existence (care gap: NO class-X encounter in the
                  window); legacy used Count >= min
- ``age_cond``    Patient demographic criterion (birthDate vs an age cutoff)
                  conjoined with an active condition; legacy never read
                  Patient fields
- ``obs_range``   two-sided closed band [lo, hi]; legacy was one-sided
- ``combo3``      three-mechanism conjunction (lab recency AND active
                  medication AND NOT active exclusion condition); the legacy
                  'combined' family was lab + condition + exclusion
- ``obs_count_repair`` repair regime over the NEW obs_count structure (the
                  legacy repair family mutated a recency+exclusion rule)

Conventions are identical to the legacy lane (explicit FHIR R4 JSON element
paths, single-variant choices, day-precision dates, no FHIRHelpers, no
wall-clock expressions). The frozen reference date for this expansion is
``REF_DATE_20K`` (distinct from the legacy lane's date so the synthetic
scenario calendars differ).

This module deliberately does NOT export ``PLUGIN``/``get_plugin()`` so the
8K plugin discovery (which resolves ``benchmark.plugins.verifier.l08`` to the
legacy ``verifier.PLUGIN``) is unaffected.
"""
from __future__ import annotations

from datetime import date
from typing import Any

FHIR_URI = "http://hl7.org/fhir"
FHIR = "{%s}" % FHIR_URI
SYS = "{urn:hl7-org:elm-types:r1}"

OPS = {">=": "GreaterOrEqual", ">": "Greater", "<=": "LessOrEqual", "<": "Less", "=": "Equal"}
OP_PHRASE = {">=": "at or above", ">": "strictly above", "<=": "at or below", "<": "strictly below"}

#: frozen reference date for the 20K expansion (no wall-clock expressions)
REF_DATE_20K = "2026-06-30"

TREND_DIRECTIONS = ("improve", "worsen")


def cutoff_date_20k(rule: dict[str, Any]) -> str:
    ref = date.fromisoformat(rule.get("ref_date", REF_DATE_20K))
    from datetime import timedelta
    return (ref - timedelta(days=int(rule["lookback_days"]))).isoformat()


def birth_cutoff_date(rule: dict[str, Any]) -> str:
    """Latest birthDate for age >= min_age_years at the reference date."""
    ref = date.fromisoformat(rule.get("ref_date", REF_DATE_20K))
    years = int(rule["min_age_years"])
    return date(ref.year - years, ref.month, ref.day).isoformat()


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


def _date_lit(iso: str) -> dict[str, Any]:
    """Date literal (day precision) — FHIR date primitives (e.g. Patient.birthDate)
    compare against Date literals, NOT DateTime literals (engine-verified)."""
    y, m, d = (int(x) for x in iso.split("-"))
    return {"type": "Date", "year": _int_lit(y), "month": _int_lit(m), "day": _int_lit(d)}


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


def _obs_in_window(rule: dict[str, Any], alias: str) -> dict[str, Any]:
    return _cmp(">=", _prop("effectiveDateTime.value", _alias(alias)),
                _dt_lit(cutoff_date_20k(rule)))


def _obs_all_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _and(
        _exists(_query([_src("O", _coded_retrieve("Observation", "TargetObs"))])),
        _not(_exists(_query([_src("F", _coded_retrieve("Observation", "TargetObs"))],
                             _not(_obs_value_cmp(rule, "F"))))),
    )


def _obs_count_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _cmp(">=", _count(_query(
        [_src("O", _coded_retrieve("Observation", "TargetObs"))],
        _and(_obs_value_cmp(rule, "O"), _obs_in_window(rule, "O")))),
        _int_lit(rule["min_count"]))


def _obs_trend_expr(rule: dict[str, Any]) -> dict[str, Any]:
    improve = rule.get("direction", "improve") == "improve"
    first_cmp = _not(_obs_value_cmp(rule, "First")) if improve else _obs_value_cmp(rule, "First")
    later_cmp = _obs_value_cmp(rule, "Later") if improve else _not(_obs_value_cmp(rule, "Later"))
    return _exists(_query(
        [_src("First", _coded_retrieve("Observation", "TargetObs"))],
        _and(
            first_cmp,
            _exists(_query(
                [_src("Later", _coded_retrieve("Observation", "TargetObs"))],
                _and(
                    later_cmp,
                    _cmp(">", _prop("effectiveDateTime.value", _alias("Later")),
                         _prop("effectiveDateTime.value", _alias("First")), "Day"),
                ))),
        )))


def _med_active_expr_20k(med: dict[str, Any], rule: dict[str, Any], alias: str) -> dict[str, Any]:
    return _exists(_query(
        [_src(alias, _retrieve("MedicationRequest"))],
        _and(
            _exists(_query([_src(alias + "C", _prop("coding", _prop("medicationCodeableConcept", _alias(alias))))],
                           _and(_cmp("=", _prop("code.value", _alias(alias + "C")), _str_lit(med["code"])),
                                _cmp("=", _prop("system.value", _alias(alias + "C")), _str_lit(med["system"]))))),
            _cmp("=", _prop("status.value", _alias(alias)), _str_lit("active")),
            _cmp(">=", _prop("authoredOn.value", _alias(alias)), _dt_lit(cutoff_date_20k(rule))),
        )))


def _med_pair_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _and(
        _med_active_expr_20k(rule["med_a"], rule, "MA"),
        _med_active_expr_20k(rule["med_b"], rule, "MB"),
    )


def _cond_status_expr(cond: dict[str, Any], *, status_code: str,
                      code_name: str = "TargetCond",
                      abatement_cutoff: str | None = None) -> dict[str, Any]:
    clauses = [_exists(_query([_src("CS", _prop("coding", _prop("clinicalStatus", _alias("C"))))],
                              _cmp("=", _prop("code.value", _alias("CS")), _str_lit(status_code))))]
    if abatement_cutoff is not None:
        clauses.append(_cmp(">=", _prop("abatementDateTime.value", _alias("C")),
                            _dt_lit(abatement_cutoff)))
    return _exists(_query([_src("C", _coded_retrieve("Condition", code_name))], _and(*clauses)))


def _cond_resolved_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _cond_status_expr(rule["cond"], status_code="resolved",
                             abatement_cutoff=cutoff_date_20k(rule))


def _enc_gap_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _not(_exists(_query(
        [_src("E", _retrieve("Encounter"))],
        _and(_cmp("=", _prop("class.code.value", _alias("E")), _str_lit(rule["enc_class"])),
             _cmp(">=", _prop("period.start.value", _alias("E")), _dt_lit(cutoff_date_20k(rule)))))))


def _age_cond_expr(rule: dict[str, Any]) -> dict[str, Any]:
    return _and(
        _exists(_query([_src("P", _retrieve("Patient"))],
                       _cmp("<=", _prop("birthDate.value", _alias("P")),
                            _date_lit(birth_cutoff_date(rule))))),
        _cond_status_expr(rule["cond"], status_code="active"),
    )


def _obs_range_expr(rule: dict[str, Any]) -> dict[str, Any]:
    obs = rule["obs"]
    return _exists(_query(
        [_src("O", _coded_retrieve("Observation", "TargetObs"))],
        _and(
            _cmp(">=", _prop("valueQuantity.value.value", _alias("O")), _dec_lit(rule["lo"])),
            _cmp("<=", _prop("valueQuantity.value.value", _alias("O")), _dec_lit(rule["hi"])),
            _cmp("=", _prop("valueQuantity.unit.value", _alias("O")), _str_lit(obs["unit"])),
        )))


def _obs_recency_expr_20k(rule: dict[str, Any]) -> dict[str, Any]:
    return _exists(_query([_src("O", _coded_retrieve("Observation", "TargetObs"))],
                          _and(_obs_value_cmp(rule, "O"), _obs_in_window(rule, "O"))))


def _combo3_expr(rule: dict[str, Any]) -> dict[str, Any]:
    expr = _and(
        _obs_recency_expr_20k(rule),
        _med_active_expr_20k(rule["med"], rule, "M"),
    )
    if rule.get("exclude", True):
        expr = _and(expr, _not(_cond_status_expr(rule["excl_cond"], status_code="active",
                                                 code_name="ExclCond")))
    return expr


_EXPR_BY_KIND_20K = {
    "obs_all": _obs_all_expr,
    "obs_count": _obs_count_expr,
    "obs_trend": _obs_trend_expr,
    "med_pair": _med_pair_expr,
    "cond_resolved": _cond_resolved_expr,
    "enc_gap": _enc_gap_expr,
    "age_cond": _age_cond_expr,
    "obs_range": _obs_range_expr,
    "combo3": _combo3_expr,
    "obs_count_repair": _obs_count_expr,
}


def rule_expression_20k(rule: dict[str, Any]) -> dict[str, Any]:
    return _EXPR_BY_KIND_20K[rule["kind"]](rule)


def _rule_codes(rule: dict[str, Any]) -> list[dict[str, Any]]:
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


def build_elm_20k(rule: dict[str, Any], *, library_id: str = "EligibilityCheck") -> dict[str, Any]:
    """Full ELM JSON library for a 20K-expansion rule (cql-execution ready)."""
    statement_name = "Flagged" if rule["kind"] == "obs_count_repair" else "Eligible"
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
                {"name": statement_name, "context": "Patient", "accessLevel": "Public",
                 "expression": rule_expression_20k(rule)},
            ]},
        }
    }


# ------------------------------------------------------------ CQL text render

def _fmt_num(v: float | int) -> str:
    return str(v)


def _code_decl(code: dict[str, Any], name: str) -> str:
    return 'code "%s": \'%s\' from "%s" display \'%s\'' % (
        name, code["code"], code["cs_name"], code["display"])


def _code_pairs(rule: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    pairs: list[tuple[dict[str, Any], str]] = []
    if rule.get("obs"):
        pairs.append((rule["obs"], "TargetObs"))
    if rule.get("cond"):
        pairs.append((rule["cond"], "TargetCond"))
    if rule.get("excl_cond"):
        pairs.append((rule["excl_cond"], "ExclCond"))
    return pairs


def _obs_where(rule: dict[str, Any], alias: str) -> str:
    obs = rule["obs"]
    return ("%s.valueQuantity.value %s %s\n"
            "      and %s.valueQuantity.unit.value = '%s'" % (
                alias, rule["op"], _fmt_num(rule["threshold"]), alias, obs["unit"]))


def _med_where(med: dict[str, Any], rule: dict[str, Any], alias: str) -> str:
    return ("exists ([MedicationRequest] %s\n"
            "    where exists (%s.medicationCodeableConcept.coding %sC\n"
            "          where %sC.code.value = '%s' and %sC.system.value = '%s')\n"
            "      and %s.status.value = 'active'\n"
            "      and %s.authoredOn.value >= @%s)" % (
                alias, alias, alias, alias, med["code"], alias, med["system"],
                alias, alias, cutoff_date_20k(rule)))


def render_cql_20k(rule: dict[str, Any]) -> str:
    """CQL-style text over explicit FHIR element paths for the new kinds."""
    lines: list[str] = []
    kind = rule["kind"]
    ref = rule.get("ref_date", REF_DATE_20K)
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

    if kind == "obs_all":
        lines.append("// Semantics: EVERY Observation of the target code must satisfy the value")
        lines.append("// condition (matching unit); the patient must have at least one such")
        lines.append("// Observation. Day-precision date comparisons; unit must match.")
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("  and not exists ([Observation: \"TargetObs\"] F")
        lines.append("    where not (%s))" % _obs_where(rule, "F").replace("\n      and", "\n      and"))
    elif kind == "obs_count":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Window: effectiveDateTime on or after %s (day precision)." % cutoff_date_20k(rule))
        lines.append("// Semantics: at least %d qualifying results inside the window." % rule["min_count"])
        lines.append('define "Eligible":')
        lines.append("  Count([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O"))
        lines.append("      and O.effectiveDateTime.value >= @%s) >= %d" % (
            cutoff_date_20k(rule), rule["min_count"]))
    elif kind == "obs_trend":
        improve = rule.get("direction", "improve") == "improve"
        if improve:
            lines.append("// Semantics: a target-code result on the FAILING side of the threshold")
            lines.append("// followed STRICTLY LATER (day precision) by a result on the passing")
            lines.append("// side (same-day pairs do not count).")
        else:
            lines.append("// Semantics: a target-code result on the passing side of the threshold")
            lines.append("// followed STRICTLY LATER (day precision) by a result on the FAILING")
            lines.append("// side (same-day pairs do not count).")
        lines.append('define "Eligible":')
        if improve:
            lines.append("  exists ([Observation: \"TargetObs\"] First")
            lines.append("    where not (%s)" % _obs_where(rule, "First"))
            lines.append("      and exists ([Observation: \"TargetObs\"] Later")
            lines.append("        where %s" % _obs_where(rule, "Later"))
            lines.append("          and Later.effectiveDateTime.value > First.effectiveDateTime.value))")
        else:
            lines.append("  exists ([Observation: \"TargetObs\"] First")
            lines.append("    where %s" % _obs_where(rule, "First"))
            lines.append("      and exists ([Observation: \"TargetObs\"] Later")
            lines.append("        where not (%s)" % _obs_where(rule, "Later"))
            lines.append("          and Later.effectiveDateTime.value > First.effectiveDateTime.value))")
    elif kind == "med_pair":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Semantics: BOTH medications must have an active request inside the")
        lines.append("// window (polypharmacy check).")
        lines.append('define "Eligible":')
        lines.append("  %s" % _med_where(rule["med_a"], rule, "MA"))
        lines.append("  and %s" % _med_where(rule["med_b"], rule, "MB"))
    elif kind == "cond_resolved":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Semantics: target condition whose clinicalStatus coding is 'resolved'")
        lines.append("// AND whose abatementDateTime is on or after %s (day precision)." % cutoff_date_20k(rule))
        lines.append('define "Eligible":')
        lines.append("  exists ([Condition: \"TargetCond\"] C")
        lines.append("    where exists (C.clinicalStatus.coding CS where CS.code.value = 'resolved')")
        lines.append("      and C.abatementDateTime.value >= @%s)" % cutoff_date_20k(rule))
    elif kind == "enc_gap":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Semantics: care gap — NO encounter of the target class started within")
        lines.append("// the window (encounters of other classes and older ones are ignored).")
        lines.append('define "Eligible":')
        lines.append("  not exists ([Encounter] E")
        lines.append("    where E.class.code.value = '%s'" % rule["enc_class"])
        lines.append("      and E.period.start.value >= @%s)" % cutoff_date_20k(rule))
    elif kind == "age_cond":
        lines.append("// Semantics: patient is at least %d years old at the reference date %s" % (
            rule["min_age_years"], ref))
        lines.append("// (birthDate on or before %s, day precision) AND has an active target" % birth_cutoff_date(rule))
        lines.append("// condition (clinicalStatus coding code 'active').")
        lines.append('define "Eligible":')
        lines.append("  exists ([Patient] P")
        lines.append("    where P.birthDate.value <= @%s)" % birth_cutoff_date(rule))
        lines.append("    and exists ([Condition: \"TargetCond\"] C")
        lines.append("      where exists (C.clinicalStatus.coding CS where CS.code.value = 'active'))")
    elif kind == "obs_range":
        obs = rule["obs"]
        lines.append("// Semantics: at least one target-code result inside the CLOSED band")
        lines.append("// [%s, %s] %s (bounds included; unit must match)." % (
            _fmt_num(rule["lo"]), _fmt_num(rule["hi"]), obs["unit"]))
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where O.valueQuantity.value >= %s" % _fmt_num(rule["lo"]))
        lines.append("      and O.valueQuantity.value <= %s" % _fmt_num(rule["hi"]))
        lines.append("      and O.valueQuantity.unit.value = '%s')" % obs["unit"])
    elif kind == "combo3":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Semantics: qualifying lab result inside the window AND an active")
        if rule.get("exclude", True):
            lines.append("// %s request inside the window AND NO active %s condition." % (
                rule["med"]["display"], rule["excl_cond"]["display"]))
        else:
            lines.append("// %s request inside the window." % rule["med"]["display"])
        lines.append('define "Eligible":')
        lines.append("  exists ([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O"))
        lines.append("      and O.effectiveDateTime.value >= @%s)" % cutoff_date_20k(rule))
        lines.append("  and %s" % _med_where(rule["med"], rule, "M"))
        if rule.get("exclude", True):
            lines.append("  and not exists ([Condition: \"ExclCond\"] X")
            lines.append("    where exists (X.clinicalStatus.coding XS where XS.code.value = 'active'))")
    elif kind == "obs_count_repair":
        lines.append("// Reference date: %s. Lookback: %d days (inclusive start)." % (
            ref, rule["lookback_days"]))
        lines.append("// Window: effectiveDateTime on or after %s (day precision)." % cutoff_date_20k(rule))
        lines.append("// Semantics: at least %d qualifying results inside the window." % rule["min_count"])
        lines.append('define "Flagged":')
        lines.append("  Count([Observation: \"TargetObs\"] O")
        lines.append("    where %s" % _obs_where(rule, "O"))
        lines.append("      and O.effectiveDateTime.value >= @%s) >= %d" % (
            cutoff_date_20k(rule), rule["min_count"]))
    else:  # pragma: no cover
        raise ValueError("unknown 20K rule kind %r" % kind)
    return "\n".join(lines) + "\n"


def intent_text_20k(rule: dict[str, Any]) -> str:
    """Natural-language clinical intent for the repair family (fully determines params)."""
    obs = rule["obs"]
    ref = rule.get("ref_date", REF_DATE_20K)
    return ("Patients are flagged when they have at least %d %s results %s %s %s recorded "
            "on or after %s (the start of the %d-day lookback window before the "
            "reference date %s)." % (
                rule["min_count"], obs["display"], OP_PHRASE[rule["op"]],
                _fmt_num(rule["threshold"]), obs["unit"],
                cutoff_date_20k(rule), rule["lookback_days"], ref))
