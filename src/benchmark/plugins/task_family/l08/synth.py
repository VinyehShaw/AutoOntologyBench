"""L08 synthetic FHIR R4 panel generator (deterministic, no real PHI).

Each instance carries a panel of synthetic patient bundles. Every bundle is a
FHIR R4 collection Bundle: one Patient resource (synthetic id, no name, no
address — nothing resembling real patient data) plus synthetic clinical
resources (Observation / Condition / MedicationRequest / Encounter) with
standard vocabulary codings.

Generation is stratified and boundary-aware: for each patient the generator
first draws an "evidence bias" (whether this patient is intended to carry
rule-satisfying evidence, ~50-60% by family), then samples concrete values
around the rule threshold (including exact-boundary values), dates around the
lookback cutoff (including the cutoff day itself), and statuses split between
active and non-active. Stratification only shapes the DATA distribution so
panels are discriminating; the answer is never forced — it is always computed
by rule evaluation (engine + independent recomputation), and degenerate
panels are honestly rejected by the builder (TRIVIAL / NON_SEMANTIC) and
re-sampled.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from benchmark.plugins.task_family.l08.families import (
    COND_POOL, ENC_POOL, MED_POOL, OBS_POOL,
)
from benchmark.plugins.verifier.l08.elm import REF_DATE

_GENDERS = ["male", "female", "unknown"]
_NON_ACTIVE_COND = ["resolved", "recurrence"]
_NON_ACTIVE_MED = ["completed", "on-hold", "cancelled"]

#: per-family probability that a patient carries rule-satisfying evidence
_EVIDENCE_BIAS = {
    "obs_threshold": 0.55,
    "obs_recency": 0.60,
    "obs_latest": 0.55,
    "med_active": 0.55,
    "cond_active": 0.55,
    "enc_count": 0.55,
    "combined": 0.75,      # per conjunct (obs / cond / no-excl)
    "recency_excl": 0.65,  # obs-recency evidence; exclusion independent (~25%)
}


def _obs_resource(oid: str, system: str, code: str, value: float, unit: str,
                  day: str, pid: str) -> dict[str, Any]:
    return {
        "resourceType": "Observation", "id": oid, "status": "final",
        "code": {"coding": [{"system": system, "code": code}]},
        "valueQuantity": {"value": value, "unit": unit},
        "effectiveDateTime": day,
        "subject": {"reference": "Patient/%s" % pid},
    }


def _cond_resource(cid: str, system: str, code: str, status: str, onset: str,
                   pid: str) -> dict[str, Any]:
    return {
        "resourceType": "Condition", "id": cid,
        "clinicalStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": status}]},
        "code": {"coding": [{"system": system, "code": code}]},
        "onsetDateTime": onset,
        "subject": {"reference": "Patient/%s" % pid},
    }


def _med_resource(mid: str, system: str, code: str, status: str, day: str,
                  pid: str) -> dict[str, Any]:
    return {
        "resourceType": "MedicationRequest", "id": mid, "status": status,
        "medicationCodeableConcept": {"coding": [{"system": system, "code": code}]},
        "authoredOn": day,
        "subject": {"reference": "Patient/%s" % pid},
    }


def _enc_resource(eid: str, klass: str, day: str, pid: str) -> dict[str, Any]:
    return {
        "resourceType": "Encounter", "id": eid, "status": "finished",
        "class": {"system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                  "code": klass},
        "period": {"start": day},
        "subject": {"reference": "Patient/%s" % pid},
    }


def _grid_step(threshold: float) -> float:
    return max(1.0, round(abs(threshold) * 0.06))


def _passing_offset(rng: random.Random, threshold: float) -> float:
    """Value at or above the threshold (>=-style ops)."""
    s = _grid_step(threshold)
    return rng.choice([0.0, 0.0, s, s, 2 * s, 3 * s])


def _failing_offset(rng: random.Random, threshold: float) -> float:
    """Value strictly below the threshold."""
    s = _grid_step(threshold)
    return rng.choice([-s, -s, -2 * s, -3 * s])


def _in_window_day(rng: random.Random, lookback: int) -> str:
    ref = date.fromisoformat(REF_DATE)
    back = rng.choice([0, 1, 2, lookback // 3, lookback // 2, lookback - 1, lookback])
    return (ref - timedelta(days=int(back))).isoformat()


def _out_window_day(rng: random.Random, lookback: int) -> str:
    ref = date.fromisoformat(REF_DATE)
    back = rng.choice([lookback + 1, lookback + 30, 2 * lookback, 3 * lookback])
    return (ref - timedelta(days=int(back))).isoformat()


def _any_day(rng: random.Random, lookback: int) -> str:
    return _in_window_day(rng, lookback) if rng.random() < 0.6 else _out_window_day(rng, lookback)


def generate_panel(rule: dict[str, Any], n_patients: int, seed: int) -> list[dict[str, Any]]:
    """Deterministic stratified synthetic panel for a rule (module docstring)."""
    rng = random.Random(seed)
    kind = rule["kind"]
    ref = date.fromisoformat(REF_DATE)
    lookback = int(rule.get("lookback_days", 365))
    bias = _EVIDENCE_BIAS.get(kind, 0.55)
    bundles: list[dict[str, Any]] = []
    for i in range(n_patients):
        pid = "synthetic-pat-%04d" % i
        has_evidence = rng.random() < bias
        resources: list[dict[str, Any]] = [{
            "resourceType": "Patient", "id": pid,
            "gender": rng.choice(_GENDERS),
            "birthDate": "%04d-%02d-%02d" % (1950 + rng.randint(0, 55),
                                             rng.randint(1, 12), rng.randint(1, 28)),
        }]
        seq = 0

        def next_id(prefix: str) -> str:
            nonlocal seq
            seq += 1
            return "%s-%s-%02d" % (prefix, pid, seq)

        threshold = float(rule.get("threshold", 0))
        obs = rule.get("obs")

        # --- observations ---
        if obs is not None:
            if kind == "obs_threshold":
                # patient qualifies iff ANY target obs value passes
                if has_evidence:
                    values = [_passing_offset(rng, threshold)] + [
                        rng.choice([_passing_offset(rng, threshold),
                                    _failing_offset(rng, threshold)])
                        for _ in range(rng.randint(0, 2))]
                else:
                    values = [_failing_offset(rng, threshold)
                              for _ in range(rng.randint(1, 3))]
                for v in values:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + v, obs["unit"], _any_day(rng, 365), pid))
            elif kind in ("obs_recency", "recency_excl"):
                # qualifying evidence = value passes AND date in window
                if has_evidence:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _passing_offset(rng, threshold), obs["unit"],
                        _in_window_day(rng, lookback), pid))
                for _ in range(rng.randint(0, 2)):
                    passing = rng.random() < 0.5
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + (_passing_offset(rng, threshold) if passing
                                     else _failing_offset(rng, threshold)),
                        obs["unit"], _any_day(rng, lookback), pid))
            elif kind == "obs_latest":
                # only the LATEST target obs decides; earlier ones are noise
                n_obs = rng.randint(2, 4)
                days = sorted({_any_day(rng, 400) for _ in range(n_obs)})
                latest_day = days[-1]
                latest_passing = has_evidence
                for d in days:
                    passing = latest_passing if d == latest_day else rng.random() < 0.5
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + (_passing_offset(rng, threshold) if passing
                                     else _failing_offset(rng, threshold)),
                        obs["unit"], d, pid))
            elif kind == "combined":
                if has_evidence:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _passing_offset(rng, threshold), obs["unit"],
                        _any_day(rng, 365), pid))
                if rng.random() < 0.3:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _failing_offset(rng, threshold), obs["unit"],
                        _any_day(rng, 365), pid))
            # distractor observations (other LOINC codes)
            for _ in range(rng.randint(1, 2)):
                other = rng.choice([o for o in OBS_POOL if o["code"] != obs["code"]])
                resources.append(_obs_resource(
                    next_id("obs"), other["system"], other["code"],
                    float(rng.choice(other["thresholds"])), other["unit"],
                    _any_day(rng, 365), pid))

        # --- conditions ---
        for cond_key in ("cond", "excl_cond"):
            cond = rule.get(cond_key)
            if cond is None:
                continue
            if kind == "cond_active" and cond_key == "cond":
                # patient qualifies iff target cond active (+ onset in window)
                if has_evidence:
                    if rule.get("lookback_days") is not None:
                        onset = _in_window_day(rng, lookback)
                    else:
                        # no-lookback rules: mix recent and old onsets so the
                        # onset-window counterfactual stays discriminating
                        onset = rng.choice([_in_window_day(rng, 90),
                                            _out_window_day(rng, 365)])
                    resources.append(_cond_resource(next_id("cond"), cond["system"],
                                                    cond["code"], "active", onset, pid))
                for _ in range(rng.randint(0, 1)):
                    status = rng.choice(["active"] + _NON_ACTIVE_COND)
                    onset = _any_day(rng, 400)
                    if rule.get("lookback_days") and status == "active":
                        onset = rng.choice([_in_window_day(rng, lookback),
                                            _out_window_day(rng, lookback)])
                    resources.append(_cond_resource(next_id("cond"), cond["system"],
                                                    cond["code"], status, onset, pid))
            elif cond_key == "excl_cond" and kind in ("combined", "recency_excl"):
                # exclusion: independent of the evidence bias (~25% active)
                if rng.random() < 0.5:
                    status = "active" if rng.random() < 0.5 else rng.choice(_NON_ACTIVE_COND)
                    resources.append(_cond_resource(next_id("cond"), cond["system"],
                                                    cond["code"], status,
                                                    _any_day(rng, 400), pid))
            elif cond_key == "cond" and kind == "combined":
                if rng.random() < 0.75:
                    resources.append(_cond_resource(next_id("cond"), cond["system"],
                                                    cond["code"], "active",
                                                    _any_day(rng, 400), pid))
                elif rng.random() < 0.4:
                    resources.append(_cond_resource(next_id("cond"), cond["system"],
                                                    cond["code"],
                                                    rng.choice(_NON_ACTIVE_COND),
                                                    _any_day(rng, 400), pid))
        # distractor condition (unrelated code)
        if rule.get("cond") or rule.get("excl_cond"):
            used = {rule.get("cond", {}).get("code"), rule.get("excl_cond", {}).get("code")}
            other = rng.choice([c for c in COND_POOL if c["code"] not in used])
            resources.append(_cond_resource(
                next_id("cond"), other["system"], other["code"],
                rng.choice(["active"] + _NON_ACTIVE_COND),
                (ref - timedelta(days=rng.choice([60, 300]))).isoformat(), pid))

        # --- medications ---
        if rule.get("med"):
            med = rule["med"]
            if has_evidence:
                resources.append(_med_resource(next_id("med"), med["system"], med["code"],
                                               "active", _in_window_day(rng, lookback), pid))
            for _ in range(rng.randint(0, 2)):
                status = rng.choice(["active"] + _NON_ACTIVE_MED)
                day = (_in_window_day(rng, lookback) if status == "active"
                       else _any_day(rng, lookback))
                resources.append(_med_resource(next_id("med"), med["system"], med["code"],
                                               status, day, pid))
            other = rng.choice([m for m in MED_POOL if m["code"] != med["code"]])
            resources.append(_med_resource(next_id("med"), other["system"], other["code"],
                                           "active", _any_day(rng, 180), pid))

        # --- encounters ---
        if kind == "enc_count":
            min_count = int(rule["min_count"])
            if has_evidence:
                for _ in range(min_count + rng.randint(0, 2)):
                    resources.append(_enc_resource(next_id("enc"), rule["enc_class"],
                                                   _in_window_day(rng, lookback), pid))
            for _ in range(rng.randint(0, 4)):
                klass = rule["enc_class"] if rng.random() < 0.5 else rng.choice(ENC_POOL)
                resources.append(_enc_resource(next_id("enc"), klass,
                                               _any_day(rng, lookback), pid))
        elif rng.random() < 0.5:
            for _ in range(rng.randint(0, 2)):
                resources.append(_enc_resource(next_id("enc"), rng.choice(ENC_POOL),
                                               _any_day(rng, 365), pid))

        bundles.append({"resourceType": "Bundle", "type": "collection",
                        "id": "panel-%s" % pid,
                        "entry": [{"resource": r} for r in resources]})
    return bundles
