"""L08 20K-expansion independent recomputation (engine-independent path).

Second verification path for the ten NEW rule kinds: the same rule that
cql-execution evaluates over ELM is recomputed here by plain Python logic
over the FHIR bundle JSON (no ELM, no Node, no shared code with the engine).
Qualification requires exact agreement between both paths (differential gate
in verifier_20k).

Semantics mirror elm_20k exactly: explicit element paths, single-variant
choices, day-precision date comparisons, (system, code) terminology matching
for coded retrieves, string equality for status/class/code primitives, and
Date-literal comparison for Patient.birthDate.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from benchmark.plugins.verifier.l08.elm_20k import (
    REF_DATE_20K, birth_cutoff_date, cutoff_date_20k,
)

_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
}


def _entries(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    return [e["resource"] for e in bundle.get("entry", [])
            if e.get("resource", {}).get("resourceType") == resource_type]


def _patient_id(bundle: dict[str, Any]) -> str:
    for e in bundle.get("entry", []):
        r = e.get("resource", {})
        if r.get("resourceType") == "Patient":
            return r["id"]
    raise ValueError("bundle has no Patient resource")


def _coding_matches(concept: dict[str, Any] | None, system: str, code: str) -> bool:
    if not concept:
        return False
    return any(c.get("system") == system and c.get("code") == code
               for c in concept.get("coding", []))


def _obs_target(bundle: dict[str, Any], obs: dict[str, Any]) -> list[dict[str, Any]]:
    return [o for o in _entries(bundle, "Observation")
            if _coding_matches(o.get("code"), obs["system"], obs["code"])]


def _obs_value_ok(obs_resource: dict[str, Any], rule: dict[str, Any]) -> bool:
    vq = obs_resource.get("valueQuantity")
    if not vq or vq.get("value") is None:
        return False
    if vq.get("unit") != rule["obs"]["unit"]:
        return False
    return _OPS[rule["op"]](float(vq["value"]), float(rule["threshold"]))


def _obs_in_window(obs_resource: dict[str, Any], rule: dict[str, Any]) -> bool:
    eff = obs_resource.get("effectiveDateTime")
    if not eff:
        return False
    return date.fromisoformat(eff[:10]) >= date.fromisoformat(cutoff_date_20k(rule))


def _obs_all(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    targets = _obs_target(bundle, rule["obs"])
    if not targets:
        return False
    return all(_obs_value_ok(o, rule) for o in targets)


def _obs_count(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    n = sum(1 for o in _obs_target(bundle, rule["obs"])
            if _obs_value_ok(o, rule) and _obs_in_window(o, rule))
    return n >= int(rule["min_count"])


def _obs_trend(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    improve = rule.get("direction", "improve") == "improve"
    dated = [(date.fromisoformat(o["effectiveDateTime"][:10]), o)
             for o in _obs_target(bundle, rule["obs"]) if o.get("effectiveDateTime")]
    for d1, o1 in dated:
        first_ok = (not _obs_value_ok(o1, rule)) if improve else _obs_value_ok(o1, rule)
        if not first_ok:
            continue
        for d2, o2 in dated:
            if d2 <= d1:
                continue  # must be STRICTLY later (day precision)
            later_ok = _obs_value_ok(o2, rule) if improve else (not _obs_value_ok(o2, rule))
            if later_ok:
                return True
    return False


def _med_active(med: dict[str, Any], rule: dict[str, Any],
                bundle: dict[str, Any]) -> bool:
    cutoff = date.fromisoformat(cutoff_date_20k(rule))
    for m in _entries(bundle, "MedicationRequest"):
        if m.get("status") != "active":
            continue
        if not _coding_matches(m.get("medicationCodeableConcept"), med["system"], med["code"]):
            continue
        authored = m.get("authoredOn")
        if authored and date.fromisoformat(authored[:10]) >= cutoff:
            return True
    return False


def _med_pair(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    return (_med_active(rule["med_a"], rule, bundle)
            and _med_active(rule["med_b"], rule, bundle))


def _cond_with_status(cond: dict[str, Any], bundle: dict[str, Any], *, status_code: str,
                      abatement_cutoff: date | None = None) -> bool:
    for c in _entries(bundle, "Condition"):
        if not _coding_matches(c.get("code"), cond["system"], cond["code"]):
            continue
        clin = c.get("clinicalStatus") or {}
        if not any(coding.get("code") == status_code for coding in clin.get("coding", [])):
            continue
        if abatement_cutoff is not None:
            abate = c.get("abatementDateTime")
            if not abate or date.fromisoformat(abate[:10]) < abatement_cutoff:
                continue
        return True
    return False


def _cond_resolved(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    cutoff = date.fromisoformat(cutoff_date_20k(rule))
    return _cond_with_status(rule["cond"], bundle, status_code="resolved",
                             abatement_cutoff=cutoff)


def _enc_gap(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    cutoff = date.fromisoformat(cutoff_date_20k(rule))
    for e in _entries(bundle, "Encounter"):
        if (e.get("class") or {}).get("code") != rule["enc_class"]:
            continue
        start = e.get("period", {}).get("start")
        if start and date.fromisoformat(start[:10]) >= cutoff:
            return False  # an in-window target encounter exists -> no gap
    return True


def _age_cond(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    cutoff = date.fromisoformat(birth_cutoff_date(rule))
    for p in _entries(bundle, "Patient"):
        bd = p.get("birthDate")
        if bd and date.fromisoformat(bd[:10]) <= cutoff:
            return _cond_with_status(rule["cond"], bundle, status_code="active")
    return False


def _obs_range(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    lo, hi = float(rule["lo"]), float(rule["hi"])
    for o in _obs_target(bundle, rule["obs"]):
        vq = o.get("valueQuantity")
        if not vq or vq.get("value") is None:
            continue
        if vq.get("unit") != rule["obs"]["unit"]:
            continue
        v = float(vq["value"])
        if lo <= v <= hi:
            return True
    return False


def _obs_recency(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    return any(_obs_value_ok(o, rule) and _obs_in_window(o, rule)
               for o in _obs_target(bundle, rule["obs"]))


def _combo3(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    if not _obs_recency(rule, bundle):
        return False
    if not _med_active(rule["med"], rule, bundle):
        return False
    if rule.get("exclude", True):
        return not _cond_with_status(rule["excl_cond"], bundle, status_code="active")
    return True


_EVAL_BY_KIND_20K = {
    "obs_all": _obs_all,
    "obs_count": _obs_count,
    "obs_trend": _obs_trend,
    "med_pair": _med_pair,
    "cond_resolved": _cond_resolved,
    "enc_gap": _enc_gap,
    "age_cond": _age_cond,
    "obs_range": _obs_range,
    "combo3": _combo3,
    "obs_count_repair": _obs_count,
}


def recompute_20k(rule: dict[str, Any], patients: list[dict[str, Any]]) -> dict[str, bool]:
    """Independent recomputation: {patient_id: Eligible} for each FHIR bundle."""
    if rule.get("ref_date", REF_DATE_20K) != REF_DATE_20K:
        raise ValueError("20K lane rules must use the frozen reference date %s" % REF_DATE_20K)
    out: dict[str, bool] = {}
    for bundle in patients:
        out[_patient_id(bundle)] = _EVAL_BY_KIND_20K[rule["kind"]](rule, bundle)
    return out
