"""L08 20K-expansion synthetic FHIR R4 panel generator (deterministic, no PHI).

Scenario generator for the ten NEW rule kinds. Every bundle is a FHIR R4
collection Bundle: one Patient resource (synthetic id prefix
``synthetic-p20k-``, no name, no address — nothing resembling real patient
data) plus synthetic clinical resources with standard vocabulary codings.

New scenario shapes beyond the legacy generator:
- all-pass vs mixed-fail target-observation sequences (obs_all)
- controlled in-window passing counts around min_count, incl. the exact
  boundary count and out-of-window passing noise (obs_count / repair)
- fail->pass and pass->fail ordered pairs, same-day non-pairs, and
  pass->pass / fail->fail decoys (obs_trend)
- 0/1/2 target-medication coverage with active/completed/on-hold statuses
  and out-of-window authoredOn (med_pair)
- resolved conditions with abatementDateTime at/around the cutoff, active
  and recurrence decoys (cond_resolved) — a FHIR element the legacy
  generator never emitted
- encounter-free windows with other-class and out-of-window decoys (enc_gap)
- birthDate stratified around the age cutoff incl. the exact cutoff day
  (age_cond) — the legacy generator never stratified demographics
- values inside/at-bounds/outside closed bands (obs_range)
- independent per-conjunct stratification for the 3-mechanism rule (combo3)

Stratification only shapes the DATA distribution so panels are
discriminating; the answer is never forced — it is always computed by rule
evaluation (engine + independent recomputation), and degenerate panels are
honestly rejected by the builder (TRIVIAL / NON_SEMANTIC) and re-sampled.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from benchmark.plugins.task_family.l08.families_20k import (
    COND_POOL_20K, ENC_POOL_20K, MED_POOL_20K, OBS_POOL_20K,
)
from benchmark.plugins.verifier.l08.elm_20k import REF_DATE_20K, birth_cutoff_date

_GENDERS = ["male", "female", "unknown"]
_NON_ACTIVE_COND = ["resolved", "recurrence"]
_NON_ACTIVE_MED = ["completed", "on-hold", "cancelled"]

#: per-family probability that a patient carries rule-satisfying evidence
_EVIDENCE_BIAS_20K = {
    "obs_all": 0.50,
    "obs_count": 0.55,
    "obs_count_repair": 0.55,
    "obs_trend": 0.55,
    "med_pair": 0.45,     # both meds needed -> lower base rate
    "cond_resolved": 0.55,
    "enc_gap": 0.50,      # gap is the qualifying side
    "age_cond": 0.75,     # per conjunct (old enough / active cond)
    "obs_range": 0.55,
    "combo3": 0.70,       # per conjunct (obs / med / no-excl)
}

_PID_PREFIX = "synthetic-p20k"


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
                   pid: str, abatement: str | None = None) -> dict[str, Any]:
    c = {
        "resourceType": "Condition", "id": cid,
        "clinicalStatus": {"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
            "code": status}]},
        "code": {"coding": [{"system": system, "code": code}]},
        "onsetDateTime": onset,
        "subject": {"reference": "Patient/%s" % pid},
    }
    if abatement is not None:
        c["abatementDateTime"] = abatement
    return c


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
    s = _grid_step(threshold)
    return rng.choice([0.0, 0.0, s, s, 2 * s, 3 * s])


def _failing_offset(rng: random.Random, threshold: float) -> float:
    s = _grid_step(threshold)
    return rng.choice([-s, -s, -2 * s, -3 * s])


def _in_window_day(rng: random.Random, lookback: int) -> str:
    ref = date.fromisoformat(REF_DATE_20K)
    back = rng.choice([0, 1, 2, lookback // 3, lookback // 2, lookback - 1, lookback])
    return (ref - timedelta(days=int(back))).isoformat()


def _out_window_day(rng: random.Random, lookback: int) -> str:
    ref = date.fromisoformat(REF_DATE_20K)
    back = rng.choice([lookback + 1, lookback + 30, 2 * lookback, 3 * lookback])
    return (ref - timedelta(days=int(back))).isoformat()


def _any_day(rng: random.Random, lookback: int) -> str:
    return _in_window_day(rng, lookback) if rng.random() < 0.6 else _out_window_day(rng, lookback)


def _distinct_days(rng: random.Random, lookback: int, k: int) -> list[str]:
    """k distinct day strings (distinct calendar days, newest-first ordering
    is NOT guaranteed; caller sorts)."""
    days: set[str] = set()
    guard = 0
    while len(days) < k and guard < 50:
        guard += 1
        days.add(_any_day(rng, max(lookback, 400)))
    while len(days) < k:  # deterministic fallback spread
        days.add((date.fromisoformat(REF_DATE_20K)
                  - timedelta(days=10 * (len(days) + 1))).isoformat())
    return list(days)


def generate_panel_20k(rule: dict[str, Any], n_patients: int,
                       seed: int) -> list[dict[str, Any]]:
    """Deterministic stratified synthetic panel for a 20K rule."""
    rng = random.Random(seed)
    kind = rule["kind"]
    lookback = int(rule.get("lookback_days", 365))
    bias = _EVIDENCE_BIAS_20K.get(kind, 0.55)
    bundles: list[dict[str, Any]] = []
    for i in range(n_patients):
        pid = "%s-%04d" % (_PID_PREFIX, i)
        has_evidence = rng.random() < bias
        threshold = float(rule.get("threshold", 0))
        obs = rule.get("obs")
        # demographic stratification for age_cond; random otherwise
        if kind == "age_cond":
            cutoff = date.fromisoformat(birth_cutoff_date(rule))
            if has_evidence:
                birth = (cutoff - timedelta(days=rng.choice([0, 30, 400, 2000]))).isoformat()
            else:
                birth = (cutoff + timedelta(days=rng.choice([1, 30, 400, 2000]))).isoformat()
        else:
            birth = "%04d-%02d-%02d" % (1950 + rng.randint(0, 55),
                                        rng.randint(1, 12), rng.randint(1, 28))
        resources: list[dict[str, Any]] = [{
            "resourceType": "Patient", "id": pid,
            "gender": rng.choice(_GENDERS),
            "birthDate": birth,
        }]
        seq = 0

        def next_id(prefix: str) -> str:
            nonlocal seq
            seq += 1
            return "%s-%s-%02d" % (prefix, pid, seq)

        # ------------------------------------------------------- observations
        if obs is not None:
            if kind == "obs_all":
                if has_evidence:
                    values = [_passing_offset(rng, threshold)] + [
                        _passing_offset(rng, threshold)
                        for _ in range(rng.randint(0, 2))]
                else:
                    values = ([_failing_offset(rng, threshold)]
                              + [rng.choice([_passing_offset(rng, threshold),
                                             _failing_offset(rng, threshold)])
                                 for _ in range(rng.randint(0, 2))])
                for v in values:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + v, obs["unit"], _any_day(rng, 365), pid))
            elif kind in ("obs_count", "obs_count_repair"):
                min_count = int(rule["min_count"])
                if has_evidence:
                    k = min_count + rng.choice([0, 0, 1, 2])  # incl. exact boundary
                else:
                    k = rng.randint(0, max(0, min_count - 1))
                for _ in range(k):
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _passing_offset(rng, threshold), obs["unit"],
                        _in_window_day(rng, lookback), pid))
                # noise: out-of-window passing + in-window failing
                for _ in range(rng.randint(0, 2)):
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _passing_offset(rng, threshold), obs["unit"],
                        _out_window_day(rng, lookback), pid))
                for _ in range(rng.randint(0, 2)):
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _failing_offset(rng, threshold), obs["unit"],
                        _in_window_day(rng, lookback), pid))
            elif kind == "obs_trend":
                improve = rule.get("direction", "improve") == "improve"
                days = sorted(_distinct_days(rng, lookback, rng.randint(2, 4)))
                if has_evidence and len(days) >= 2:
                    # guaranteed crossing pair: failing side strictly before passing
                    first_day, later_day = days[0], days[-1]
                    first_off = _failing_offset(rng, threshold) if improve else _passing_offset(rng, threshold)
                    later_off = _passing_offset(rng, threshold) if improve else _failing_offset(rng, threshold)
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + first_off, obs["unit"], first_day, pid))
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + later_off, obs["unit"], later_day, pid))
                    for d in days[1:-1]:
                        resources.append(_obs_resource(
                            next_id("obs"), obs["system"], obs["code"],
                            threshold + rng.choice([_passing_offset(rng, threshold),
                                                    _failing_offset(rng, threshold)]),
                            obs["unit"], d, pid))
                else:
                    # decoys: same-side sequences, same-day pairs, reversed crossings
                    side = rng.choice(["pass", "fail"])
                    for d in days:
                        off = (_passing_offset(rng, threshold) if side == "pass"
                               else _failing_offset(rng, threshold))
                        resources.append(_obs_resource(
                            next_id("obs"), obs["system"], obs["code"],
                            threshold + off, obs["unit"], d, pid))
                    if rng.random() < 0.5 and days:  # same-day pair (never a crossing)
                        off_p, off_f = _passing_offset(rng, threshold), _failing_offset(rng, threshold)
                        if improve:
                            pair = [(off_f, days[0]), (off_p, days[0])]
                        else:
                            pair = [(off_p, days[0]), (off_f, days[0])]
                        for off, d in pair:
                            resources.append(_obs_resource(
                                next_id("obs"), obs["system"], obs["code"],
                                threshold + off, obs["unit"], d, pid))
            elif kind == "obs_range":
                lo, hi = float(rule["lo"]), float(rule["hi"])
                mid = (lo + hi) / 2.0
                s = _grid_step(mid)
                if has_evidence:
                    values = [rng.choice([lo, hi, mid, lo + s, hi - s])]
                else:
                    values = [rng.choice([lo - s, hi + s, lo - 2 * s, hi + 2 * s])]
                for v in values:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        v, obs["unit"], _any_day(rng, 365), pid))
                for _ in range(rng.randint(0, 2)):
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        rng.choice([lo, hi, mid, lo - s, hi + s]), obs["unit"],
                        _any_day(rng, 365), pid))
            elif kind == "combo3":
                if has_evidence:
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + _passing_offset(rng, threshold), obs["unit"],
                        _in_window_day(rng, lookback), pid))
                for _ in range(rng.randint(0, 2)):
                    resources.append(_obs_resource(
                        next_id("obs"), obs["system"], obs["code"],
                        threshold + rng.choice([_passing_offset(rng, threshold),
                                                _failing_offset(rng, threshold)]),
                        obs["unit"], _any_day(rng, lookback), pid))
            # distractor observations (other codes)
            for _ in range(rng.randint(1, 2)):
                other = rng.choice([o for o in OBS_POOL_20K if o["code"] != obs["code"]])
                resources.append(_obs_resource(
                    next_id("obs"), other["system"], other["code"],
                    float(rng.choice(other["thresholds"])), other["unit"],
                    _any_day(rng, 365), pid))

        # --------------------------------------------------------- conditions
        cond_keys: list[str] = []
        if rule.get("cond"):
            cond_keys.append("cond")
        if rule.get("excl_cond"):
            cond_keys.append("excl_cond")
        for cond_key in cond_keys:
            cond = rule[cond_key]
            if kind == "cond_resolved" and cond_key == "cond":
                if has_evidence:
                    onset = _out_window_day(rng, lookback)
                    abate = _in_window_day(rng, lookback)  # incl. cutoff day
                    resources.append(_cond_resource(
                        next_id("cond"), cond["system"], cond["code"], "resolved",
                        onset, pid, abatement=abate))
                for _ in range(rng.randint(0, 2)):
                    shape = rng.random()
                    if shape < 0.4:  # resolved but abated out of window
                        resources.append(_cond_resource(
                            next_id("cond"), cond["system"], cond["code"], "resolved",
                            _out_window_day(rng, lookback), pid,
                            abatement=_out_window_day(rng, lookback)))
                    elif shape < 0.7:  # active decoy
                        resources.append(_cond_resource(
                            next_id("cond"), cond["system"], cond["code"], "active",
                            _any_day(rng, 400), pid))
                    else:  # recurrence decoy (not resolved)
                        resources.append(_cond_resource(
                            next_id("cond"), cond["system"], cond["code"], "recurrence",
                            _any_day(rng, 400), pid,
                            abatement=_in_window_day(rng, lookback)))
            elif kind == "age_cond" and cond_key == "cond":
                if has_evidence:
                    resources.append(_cond_resource(
                        next_id("cond"), cond["system"], cond["code"], "active",
                        _any_day(rng, 400), pid))
                for _ in range(rng.randint(0, 1)):
                    resources.append(_cond_resource(
                        next_id("cond"), cond["system"], cond["code"],
                        rng.choice(_NON_ACTIVE_COND), _any_day(rng, 400), pid))
            elif cond_key == "excl_cond" and kind == "combo3":
                # exclusion: independent of the evidence bias (~30% active)
                if rng.random() < 0.5:
                    status = "active" if rng.random() < 0.6 else rng.choice(_NON_ACTIVE_COND)
                    resources.append(_cond_resource(
                        next_id("cond"), cond["system"], cond["code"], status,
                        _any_day(rng, 400), pid))
        # distractor condition (unrelated code)
        if rule.get("cond") or rule.get("excl_cond"):
            used = {rule.get("cond", {}).get("code"), rule.get("excl_cond", {}).get("code")}
            other = rng.choice([c for c in COND_POOL_20K if c["code"] not in used])
            resources.append(_cond_resource(
                next_id("cond"), other["system"], other["code"],
                rng.choice(["active"] + _NON_ACTIVE_COND),
                (date.fromisoformat(REF_DATE_20K)
                 - timedelta(days=rng.choice([60, 300]))).isoformat(), pid))

        # -------------------------------------------------------- medications
        if kind == "med_pair":
            meds = [rule["med_a"], rule["med_b"]]
            for j, med in enumerate(meds):
                covered = has_evidence or rng.random() < 0.5  # independent per med
                if covered:
                    resources.append(_med_resource(
                        next_id("med"), med["system"], med["code"], "active",
                        _in_window_day(rng, lookback), pid))
                for _ in range(rng.randint(0, 1)):
                    status = rng.choice(["active"] + _NON_ACTIVE_MED)
                    day = (_in_window_day(rng, lookback) if status == "active"
                           else _any_day(rng, lookback))
                    resources.append(_med_resource(
                        next_id("med"), med["system"], med["code"], status, day, pid))
            other = rng.choice([m for m in MED_POOL_20K
                                if m["code"] not in {rule["med_a"]["code"],
                                                     rule["med_b"]["code"]}])
            resources.append(_med_resource(
                next_id("med"), other["system"], other["code"], "active",
                _any_day(rng, 180), pid))
        elif kind == "combo3" and rule.get("med"):
            med = rule["med"]
            if has_evidence or rng.random() < 0.5:
                resources.append(_med_resource(
                    next_id("med"), med["system"], med["code"], "active",
                    _in_window_day(rng, lookback), pid))
            for _ in range(rng.randint(0, 2)):
                status = rng.choice(["active"] + _NON_ACTIVE_MED)
                day = (_in_window_day(rng, lookback) if status == "active"
                       else _any_day(rng, lookback))
                resources.append(_med_resource(
                    next_id("med"), med["system"], med["code"], status, day, pid))
            other = rng.choice([m for m in MED_POOL_20K if m["code"] != med["code"]])
            resources.append(_med_resource(
                next_id("med"), other["system"], other["code"], "active",
                _any_day(rng, 180), pid))

        # -------------------------------------------------------- encounters
        if kind == "enc_gap":
            if not has_evidence:
                # in-window target-class encounter (kills the gap)
                resources.append(_enc_resource(
                    next_id("enc"), rule["enc_class"],
                    _in_window_day(rng, lookback), pid))
            for _ in range(rng.randint(0, 3)):
                klass = (rule["enc_class"] if rng.random() < 0.4
                         else rng.choice([c for c in ENC_POOL_20K
                                          if c != rule["enc_class"]]))
                day = (_out_window_day(rng, lookback) if klass == rule["enc_class"]
                       else _any_day(rng, lookback))
                resources.append(_enc_resource(next_id("enc"), klass, day, pid))
        elif rng.random() < 0.5:
            for _ in range(rng.randint(0, 2)):
                resources.append(_enc_resource(
                    next_id("enc"), rng.choice(ENC_POOL_20K),
                    _any_day(rng, 365), pid))

        bundles.append({"resourceType": "Bundle", "type": "collection",
                        "id": "panel-%s" % pid,
                        "entry": [{"resource": r} for r in resources]})
    return bundles
