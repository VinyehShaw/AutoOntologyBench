"""L08 independent recomputation: rule semantics directly over FHIR JSON.

This is the SECOND, engine-independent verification path required by the lane
contract: the same rule that cql-execution evaluates over ELM is recomputed
here by plain Python logic over the FHIR bundle JSON (no ELM, no Node, no
shared code with the engine). Qualification requires exact agreement between
both paths (differential gate in the verifier).

Semantics mirror the ELM exactly (see elm.py docstring for the pinned
conventions): explicit element paths, single-variant choices, day-precision
date comparisons, (system, code) terminology matching for coded retrieves,
string equality for status/class/code primitives.
"""
from __future__ import annotations

from datetime import date
from typing import Any

from benchmark.plugins.verifier.l08.elm import REF_DATE, cutoff_date

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


def _obs_threshold(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    return any(_obs_value_ok(o, rule) for o in _obs_target(bundle, rule["obs"]))


def _obs_recency(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    cutoff = date.fromisoformat(cutoff_date(rule))
    for o in _obs_target(bundle, rule["obs"]):
        eff = o.get("effectiveDateTime")
        if not eff:
            continue
        if date.fromisoformat(eff[:10]) >= cutoff and _obs_value_ok(o, rule):
            return True
    return False


def _obs_latest(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    obs_list = _obs_target(bundle, rule["obs"])
    dated = [(date.fromisoformat(o["effectiveDateTime"][:10]), o)
             for o in obs_list if o.get("effectiveDateTime")]
    if not dated:
        return False
    latest_day = max(d for d, _ in dated)
    return any(_obs_value_ok(o, rule) for d, o in dated if d == latest_day)


def _med_active(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    med = rule["med"]
    cutoff = date.fromisoformat(cutoff_date(rule))
    for m in _entries(bundle, "MedicationRequest"):
        if m.get("status") != "active":
            continue
        if not _coding_matches(m.get("medicationCodeableConcept"), med["system"], med["code"]):
            continue
        authored = m.get("authoredOn")
        if authored and date.fromisoformat(authored[:10]) >= cutoff:
            return True
    return False


def _cond_active(cond: dict[str, Any], bundle: dict[str, Any],
                 *, onset_cutoff: date | None) -> bool:
    for c in _entries(bundle, "Condition"):
        if not _coding_matches(c.get("code"), cond["system"], cond["code"]):
            continue
        clin = c.get("clinicalStatus") or {}
        if not any(coding.get("code") == "active" for coding in clin.get("coding", [])):
            continue
        if onset_cutoff is not None:
            onset = c.get("onsetDateTime")
            if not onset or date.fromisoformat(onset[:10]) < onset_cutoff:
                continue
        return True
    return False


def _enc_count(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    cutoff = date.fromisoformat(cutoff_date(rule))
    n = 0
    for e in _entries(bundle, "Encounter"):
        if (e.get("class") or {}).get("code") != rule["enc_class"]:
            continue
        start = e.get("period", {}).get("start")
        if start and date.fromisoformat(start[:10]) >= cutoff:
            n += 1
    return n >= rule["min_count"]


def _combined(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    return (_obs_threshold(rule, bundle)
            and _cond_active(rule["cond"], bundle, onset_cutoff=None)
            and not _cond_active(rule["excl_cond"], bundle, onset_cutoff=None))


def _recency_excl(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    if not _obs_recency(rule, bundle):
        return False
    if rule.get("exclude", True):
        return not _cond_active(rule["excl_cond"], bundle, onset_cutoff=None)
    return True


def _cond_active_rule(rule: dict[str, Any], bundle: dict[str, Any]) -> bool:
    cutoff = (date.fromisoformat(cutoff_date(rule))
              if rule.get("lookback_days") is not None else None)
    return _cond_active(rule["cond"], bundle, onset_cutoff=cutoff)


_EVAL_BY_KIND = {
    "obs_threshold": _obs_threshold,
    "obs_recency": _obs_recency,
    "obs_latest": _obs_latest,
    "med_active": _med_active,
    "cond_active": _cond_active_rule,
    "enc_count": _enc_count,
    "combined": _combined,
    "recency_excl": _recency_excl,
}


def recompute(rule: dict[str, Any], patients: list[dict[str, Any]]) -> dict[str, bool]:
    """Independent recomputation: {patient_id: Eligible} for each FHIR bundle."""
    if rule.get("ref_date", REF_DATE) != REF_DATE:
        raise ValueError("lane rules must use the frozen reference date %s" % REF_DATE)
    out: dict[str, bool] = {}
    for bundle in patients:
        out[_patient_id(bundle)] = _EVAL_BY_KIND[rule["kind"]](rule, bundle)
    return out
