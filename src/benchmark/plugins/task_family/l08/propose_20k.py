"""L08 20K-expansion task-family propose + ground (new dataset/seed/ids).

Pipeline position: pinned source snapshot (same cql-clinical snapshot as the
legacy lane — the language + engine are the pinned source; rule content is
lane-synthetic) -> propose (this module) -> ground -> native verifier gates
(verifier_20k) -> shard writer (build_20k).

Propose derives, deterministically from the 20K lane seed 20261101:
- 10 NEW families x 5 base goals = 50 NEW base goals; each base goal is a
  clinical parameterization of its family's NEW rule kind (see elm_20k).
- 5 instances per goal = 250 NEW candidate instances; each instance is a
  fresh synthetic FHIR panel (boundary-aware sampling, see synth_20k).

Identity discipline vs the legacy 8K lane:
- dataset_id ``autoontobench-bank-20000-candidate-dev1`` (legacy keeps
  ``autoontobench-bank-8000-rc1``);
- id prefixes ``l08b2-`` (base goals) / ``l08i2-`` (instances) — a distinct
  namespace from the legacy ``l08b-``/``l08i-`` prefixes;
- seed base 20261101 (legacy used 20261027);
- build_20k additionally verifies no id / signature collisions against the
  legacy bank on disk.

Ground assembles the public view (CQL-style rule text over explicit FHIR
element paths + the synthetic patient panel + the question) and the private
oracle payload skeleton. The reference answer is filled by the builder after
the native gates pass; the public view never contains it.

This module deliberately does NOT export ``PLUGIN``/``get_plugin()`` so the
8K plugin discovery (which resolves ``benchmark.plugins.task_family.l08`` to
the legacy ``propose.PLUGIN``) is unaffected.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Iterator

from benchmark.plugins.task_family.l08.families_20k import (
    ENC_POOL_20K, LOOKBACKS_20K, MED_POOL_20K, MIN_AGES_20K, OBS_POOL_20K,
    PANEL_SIZE_20K, BY_ID_20K, COND_POOL_20K, FamilySpec20k,
)
from benchmark.plugins.task_family.l08.synth_20k import generate_panel_20k
from benchmark.plugins.verifier.l08.elm_20k import (
    REF_DATE_20K, build_elm_20k, intent_text_20k, render_cql_20k,
)
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

LANE = "L08"
DATASET_ID_20K = "autoontobench-bank-20000-candidate-dev1"
PROFILE_20K = "cql-fhir-r4-explicit-paths/2"
SEED_20K = 20261101

ALLOWED_TOOLS_APPLICATION = ["fhir_panel.read", "cql_rule.read"]
ALLOWED_TOOLS_REPAIR = ["fhir_panel.read", "cql_rule.read", "rule_params.edit"]
BUDGET_APPLICATION = {"actions": 12, "probes": 2}
BUDGET_REPAIR = {"actions": 16, "probes": 4}

_CONVENTION = (
    "Rule semantics are pinned to explicit FHIR R4 JSON element paths "
    "(choice-typed elements carry exactly one variant in this dataset; FHIR "
    "primitives compare via their value element; date comparisons are "
    "day-precision; no FHIRHelpers implicit conversions)."
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_key(*parts: Any) -> str:
    import json

    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def _base_goal_id(family_id: str, rule: dict[str, Any]) -> str:
    return "l08b2-" + _sha(json_key(family_id, rule))[:16]


def _instance_id(goal_id: str, attempt: int) -> str:
    return "l08i2-" + _sha("%s|%d" % (goal_id, attempt))[:16]


def _derive_goal_rule_20k(fam: FamilySpec20k, rng: random.Random) -> dict[str, Any]:
    """Deterministic clinical parameterization for one NEW base goal."""
    rule: dict[str, Any] = {"kind": fam.kind, "ref_date": REF_DATE_20K}
    if fam.kind in ("obs_all", "obs_count", "obs_trend", "combo3", "obs_count_repair"):
        obs = dict(rng.choice(OBS_POOL_20K))
        rule["obs"] = obs
        rule["op"] = obs["op"]
        rule["threshold"] = rng.choice(obs["thresholds"])
    if fam.kind in ("obs_count", "combo3", "obs_count_repair"):
        rule["lookback_days"] = rng.choice(LOOKBACKS_20K)
    if fam.kind in ("obs_count", "obs_count_repair"):
        rule["min_count"] = rng.randint(2, 3)
    if fam.kind == "obs_trend":
        rule["direction"] = rng.choice(["improve", "worsen"])
    if fam.kind == "med_pair":
        med_a = dict(rng.choice(MED_POOL_20K))
        med_b = dict(rng.choice([m for m in MED_POOL_20K if m["code"] != med_a["code"]]))
        rule["med_a"] = med_a
        rule["med_b"] = med_b
        rule["lookback_days"] = rng.choice(LOOKBACKS_20K)
    if fam.kind == "cond_resolved":
        rule["cond"] = dict(rng.choice(COND_POOL_20K))
        rule["lookback_days"] = rng.choice(LOOKBACKS_20K)
    if fam.kind == "enc_gap":
        rule["enc_class"] = rng.choice(ENC_POOL_20K)
        rule["lookback_days"] = rng.choice(LOOKBACKS_20K)
    if fam.kind == "age_cond":
        rule["cond"] = dict(rng.choice(COND_POOL_20K))
        rule["min_age_years"] = rng.choice(MIN_AGES_20K)
    if fam.kind == "obs_range":
        obs = dict(rng.choice(OBS_POOL_20K))
        rule["obs"] = obs
        lo = float(rng.choice(obs["thresholds"]))
        rule["lo"] = lo
        rule["hi"] = lo + rng.choice([1.0, 2.0]) * max(1.0, round(abs(lo) * 0.08))
    if fam.kind == "combo3":
        rule["med"] = dict(rng.choice(MED_POOL_20K))
        excl = rng.choice([c for c in COND_POOL_20K if c["code"] != rule.get("cond", {}).get("code")])
        rule["excl_cond"] = dict(excl)
        rule["exclude"] = True
    return rule


def _repair_view_20k(rule: dict[str, Any], rng: random.Random,
                     patients: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Broken rule (public) for the NEW repair family + noop flag.

    Mutates 1-2 of {op, threshold, lookback_days, min_count} away from the
    intent parameters with LARGE deltas, then verifies with the independent
    Python path that the broken rule's classification differs from the
    intended one; escalates through a deterministic ladder if not. Returns
    (broken_rule, noop) where noop=True means no discriminating mutation was
    found (the builder rejects such candidates honestly).
    """
    from benchmark.plugins.verifier.l08 import recompute_20k

    intended = recompute_20k.recompute_20k(rule, patients)
    grid = max(1.0, round(abs(float(rule["threshold"])) * 0.06))
    lookback_ladder = [d for d in (30, 90, 180, 365, 730)
                       if d != int(rule["lookback_days"])]
    op_flip = {"<=": "<", "<": "<=", ">=": ">", ">": ">="}[rule["op"]]
    candidates: list[dict[str, Any]] = []
    for _ in range(6):  # random large mutations first
        broken = dict(rule)
        for param in rng.sample(["op", "threshold", "lookback_days", "min_count"],
                                rng.choice([1, 2])):
            if param == "op":
                broken["op"] = op_flip
            elif param == "threshold":
                broken["threshold"] = float(rule["threshold"]) + rng.choice(
                    [2 * grid, -2 * grid, 4 * grid, -4 * grid])
            elif param == "lookback_days":
                broken["lookback_days"] = rng.choice(lookback_ladder)
            elif param == "min_count":
                broken["min_count"] = max(1, int(rule["min_count"]) + rng.choice([1, 2, -1]))
        candidates.append(broken)
    # deterministic escalation ladder (guaranteed-different shapes last)
    for threshold_delta in (2 * grid, -2 * grid, 6 * grid, -6 * grid):
        for lookback in lookback_ladder:
            candidates.append({**rule, "threshold": float(rule["threshold"]) + threshold_delta,
                               "lookback_days": lookback})
    candidates.append({**rule, "op": op_flip, "min_count": int(rule["min_count"]) + 1})
    candidates.append({**rule, "lookback_days": lookback_ladder[-1],
                       "min_count": max(1, int(rule["min_count"]) - 1)})
    for broken in candidates:
        try:
            if recompute_20k.recompute_20k(broken, patients) != intended:
                return broken, False
        except Exception:
            continue
    return dict(rule), True


class L0820kTaskFamily:
    """Task-family propose/ground for the 10 NEW L08 20K families."""

    def __init__(self, seed: int = SEED_20K) -> None:
        self.seed = seed

    def supported_profiles(self) -> list[str]:
        return [PROFILE_20K]

    def propose(self, snapshot: Any, *, family: str, seed: int = SEED_20K,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        fam = BY_ID_20K[family]
        constraints = constraints or {}
        max_goals = int(constraints.get("max_base_goals", fam.base_goals))
        max_attempts = int(constraints.get("max_attempts_per_goal", 8))
        rng = random.Random("%d|%s" % (seed, family))
        seen_goal_ids: set[str] = set()
        goals_emitted = 0
        derivation_tries = 0
        while goals_emitted < max_goals:
            derivation_tries += 1
            if derivation_tries > max_goals * 40:  # safety: parameterization exhausted
                return
            rule = _derive_goal_rule_20k(fam, rng)
            goal_id = _base_goal_id(fam.family_id, rule)
            if goal_id in seen_goal_ids:
                continue  # same parameterization already emitted for this family
            seen_goal_ids.add(goal_id)
            goals_emitted += 1
            for attempt in range(1, max_attempts + 1):
                instance_id = _instance_id(goal_id, attempt)
                panel_seed = int(_sha("%s|%d|panel20k" % (goal_id, attempt))[:12], 16)
                panel = generate_panel_20k(rule, PANEL_SIZE_20K[fam.complexity], panel_seed)
                ast: dict[str, Any] = {"rule": rule, "patients": panel,
                                       "complexity": fam.complexity}
                if fam.regime == "repair":
                    rrng = random.Random("%s|%d|repair20k" % (goal_id, attempt))
                    broken, noop = _repair_view_20k(rule, rrng, panel)
                    ast["broken_rule"] = broken
                    ast["repair_noop"] = noop
                yield CandidateTaskAST(
                    lane_id=LANE, family_id=fam.family_id, base_goal_id=goal_id,
                    instance_id=instance_id, regime=fam.regime,
                    semantic_profile=PROFILE_20K, ast=ast,
                    source_unit_ids=_source_unit_ids_20k(),
                )

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        rule = ast["rule"]
        panel = ast["patients"]
        regime = candidate.regime
        cql_text = render_cql_20k(rule if regime != "repair" else ast["broken_rule"])
        patient_ids = _patient_ids(panel)
        if regime == "repair":
            user_goal = (
                "Repair the clinical rule so it flags exactly the intended patients.\n\n"
                "Clinical intent: %s\n\n"
                "The rule below is the current (broken) version. Adjust its parameters "
                "(op, threshold, lookback_days, min_count) so the rule matches the intent. "
                "Submit {\"op\", \"threshold\", \"lookback_days\", \"min_count\"}.\n\n"
                "Conventions: %s\n\nBroken rule (CQL, explicit FHIR element paths):\n%s\n"
                "Synthetic FHIR R4 panel (%d patients; bundle per patient):\n%s" % (
                    intent_text_20k(rule), _CONVENTION, cql_text, len(panel),
                    _panel_digest_lines(panel)))
            allowed_tools = list(ALLOWED_TOOLS_REPAIR)
            budget = dict(BUDGET_REPAIR)
        else:
            user_goal = (
                "Determine which patients satisfy the clinical rule.\n\n"
                "Conventions: %s\n\nRule (CQL, explicit FHIR element paths):\n%s\n"
                "Synthetic FHIR R4 panel (%d patients; bundle per patient):\n%s\n"
                "Submit the complete set of eligible patient ids as "
                "{\"answer_ids\": [...]}; partial or superset answers are scored "
                "as failure." % (_CONVENTION, cql_text, len(panel),
                                 _panel_digest_lines(panel)))
            allowed_tools = list(ALLOWED_TOOLS_APPLICATION)
            budget = dict(BUDGET_APPLICATION)
        public = {
            "user_goal": user_goal,
            "allowed_tools": allowed_tools,
            "budget": budget,
            "observation_refs": [],  # filled by the builder (public file + evidence refs)
            "rule_kind": rule["kind"],
            "rule_cql": cql_text,
            "patient_ids": patient_ids,
            "semantic_profile": PROFILE_20K,
            "complexity": ast["complexity"],
            "data_provenance": "fully_synthetic_no_phi",
        }
        oracle_handle = "orl08x_" + _sha(json_key(candidate.instance_id, rule))[:16]
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_handle,
                            native_evidence={})


def _patient_ids(panel: list[dict[str, Any]]) -> list[str]:
    out = []
    for bundle in panel:
        for e in bundle.get("entry", []):
            r = e.get("resource", {})
            if r.get("resourceType") == "Patient":
                out.append(r["id"])
    return out


def _panel_digest_lines(panel: list[dict[str, Any]]) -> str:
    """Compact per-patient resource summary (full bundles live in the public file)."""
    lines = []
    for bundle in panel:
        counts: dict[str, int] = {}
        pid = "?"
        for e in bundle.get("entry", []):
            r = e.get("resource", {})
            rt = r.get("resourceType", "?")
            counts[rt] = counts.get(rt, 0) + 1
            if rt == "Patient":
                pid = r["id"]
        lines.append("%s: %s" % (pid, json_key(counts)))
    return "\n".join(lines)


def _source_unit_ids_20k() -> tuple[str, ...]:
    return (
        "cql-clinical/grammar/cql.g4",
        "cql-clinical/schemas/elm/expression.xsd",
        "cql-clinical/npm/cql-execution",
        "cql-clinical/npm/cql-exec-fhir",
    )
