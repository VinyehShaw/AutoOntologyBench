"""L08 task-family plugin: propose candidates and ground them.

Pipeline position: pinned source snapshot -> propose (this module) -> ground
-> native verifier gates (differential engine-vs-python, counterfactual
witness, self-checks) -> shard writer (build.py).

Propose derives, deterministically from the lane seed:
- 8 families x (6 or 7) base goals = 50 base goals; each base goal is a
  clinical parameterization (vocabulary code + operator + threshold +
  lookback + exclusion) of its family's rule kind.
- 5 instances per goal = 250 candidate instances; each instance is a fresh
  synthetic FHIR panel (boundary-aware sampling, see synth.py).

Ground assembles the public view (CQL-style rule text over explicit FHIR
element paths + the synthetic patient panel + the question) and the private
oracle payload skeleton (rule params + ELM). The reference answer is filled
by the builder after the native gates pass; the public view never contains
it (leakage is scanned at build time).
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Iterator

from benchmark.plugins.task_family.l08.families import (
    COND_POOL, ENC_POOL, LOOKBACKS, MED_POOL, OBS_POOL, PANEL_SIZE, BY_ID,
    FAMILY_TABLE, FamilySpec,
)
from benchmark.plugins.task_family.l08.synth import generate_panel
from benchmark.plugins.verifier.l08.elm import REF_DATE, build_elm, render_cql
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

LANE = "L08"
DATASET_ID = "autoontobench-bank-8000-rc1"
PROFILE = "cql-fhir-r4-explicit-paths/1"
SEED = 20261027

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


def _base_goal_id(family_id: str, rule: dict[str, Any]) -> str:
    return "l08b-" + _sha(json_key(family_id, rule))[:16]


def _instance_id(goal_id: str, attempt: int) -> str:
    return "l08i-" + _sha("%s|%d" % (goal_id, attempt))[:16]


def json_key(*parts: Any) -> str:
    import json

    return json.dumps(parts, sort_keys=True, separators=(",", ":"))


def _derive_goal_rule(fam: FamilySpec, rng: random.Random) -> dict[str, Any]:
    """Deterministic clinical parameterization for one base goal."""
    obs = dict(rng.choice(OBS_POOL))
    rule: dict[str, Any] = {"kind": fam.kind, "ref_date": REF_DATE}
    if fam.kind in ("obs_threshold", "obs_recency", "obs_latest", "combined", "recency_excl"):
        rule["obs"] = obs
        rule["op"] = obs["op"]
        rule["threshold"] = rng.choice(obs["thresholds"])
    if fam.kind in ("obs_recency", "recency_excl"):
        rule["lookback_days"] = rng.choice(LOOKBACKS)
    if fam.kind == "med_active":
        rule["med"] = dict(rng.choice(MED_POOL))
        rule["lookback_days"] = rng.choice(LOOKBACKS)
    if fam.kind == "cond_active":
        rule["cond"] = dict(rng.choice(COND_POOL))
        if rng.random() < 0.6:
            rule["lookback_days"] = rng.choice(LOOKBACKS)
    if fam.kind == "enc_count":
        rule["enc_class"] = rng.choice(ENC_POOL)
        rule["min_count"] = rng.randint(2, 4)
        rule["lookback_days"] = rng.choice(LOOKBACKS)
    if fam.kind == "combined":
        rule["cond"] = dict(rng.choice(COND_POOL))
        excl = rng.choice([c for c in COND_POOL if c["code"] != rule["cond"]["code"]])
        rule["excl_cond"] = dict(excl)
    if fam.kind == "recency_excl":
        excl = rng.choice(COND_POOL)
        rule["excl_cond"] = dict(excl)
        rule["exclude"] = True
    return rule


def _repair_view(rule: dict[str, Any], rng: random.Random,
                 patients: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    """Broken rule (public) for the repair family + noop flag.

    Mutates 1-2 of {op, threshold, lookback_days, exclude} away from the
    intent parameters with LARGE deltas, then verifies with the independent
    Python path that the broken rule's classification differs from the
    intended one; escalates through a deterministic ladder if not. Returns
    (broken_rule, noop) where noop=True means no discriminating mutation was
    found (the builder rejects such candidates honestly).
    """
    from benchmark.plugins.verifier.l08 import recompute

    intended = recompute.recompute(rule, patients)
    grid = max(1.0, round(abs(float(rule["threshold"])) * 0.06))
    lookback_ladder = [d for d in (30, 90, 180, 365, 730)
                       if d != int(rule["lookback_days"])]
    op_flip = {"<=": "<", "<": "<=", ">=": ">", ">": ">="}[rule["op"]]
    candidates: list[dict[str, Any]] = []
    for _ in range(6):  # random large mutations first
        broken = dict(rule)
        for param in rng.sample(["op", "threshold", "lookback_days", "exclude"],
                                rng.choice([1, 2])):
            if param == "op":
                broken["op"] = op_flip
            elif param == "threshold":
                broken["threshold"] = float(rule["threshold"]) + rng.choice(
                    [2 * grid, -2 * grid, 4 * grid, -4 * grid])
            elif param == "lookback_days":
                broken["lookback_days"] = rng.choice(lookback_ladder)
            elif param == "exclude":
                broken["exclude"] = not rule["exclude"]
        candidates.append(broken)
    # deterministic escalation ladder (guaranteed-different shapes last)
    for threshold_delta in (2 * grid, -2 * grid, 6 * grid, -6 * grid):
        for lookback in lookback_ladder:
            candidates.append({**rule, "threshold": float(rule["threshold"]) + threshold_delta,
                               "lookback_days": lookback})
    candidates.append({**rule, "op": op_flip, "exclude": not rule.get("exclude", True)})
    candidates.append({**rule, "lookback_days": lookback_ladder[-1],
                       "exclude": not rule.get("exclude", True)})
    for broken in candidates:
        try:
            if recompute.recompute(broken, patients) != intended:
                return broken, False
        except Exception:
            continue
    return dict(rule), True


class L08TaskFamilyPlugin:
    """TaskFamilyPlugin for the 8 L08 families (see families.FAMILY_TABLE)."""

    def __init__(self, seed: int = SEED) -> None:
        self.seed = seed

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    def propose(self, snapshot: Any, *, family: str, seed: int = SEED,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        fam = BY_ID[family]
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
            rule = _derive_goal_rule(fam, rng)
            goal_id = _base_goal_id(fam.family_id, rule)
            if goal_id in seen_goal_ids:
                continue  # same parameterization already emitted for this family
            seen_goal_ids.add(goal_id)
            goals_emitted += 1
            for attempt in range(1, max_attempts + 1):
                instance_id = _instance_id(goal_id, attempt)
                panel_seed = int(_sha("%s|%d|panel" % (goal_id, attempt))[:12], 16)
                panel = generate_panel(rule, PANEL_SIZE[fam.complexity], panel_seed)
                ast: dict[str, Any] = {"rule": rule, "patients": panel,
                                       "complexity": fam.complexity}
                if fam.regime == "repair":
                    rrng = random.Random("%s|%d|repair" % (goal_id, attempt))
                    broken, noop = _repair_view(rule, rrng, panel)
                    ast["broken_rule"] = broken
                    ast["repair_noop"] = noop
                yield CandidateTaskAST(
                    lane_id=LANE, family_id=fam.family_id, base_goal_id=goal_id,
                    instance_id=instance_id, regime=fam.regime,
                    semantic_profile=PROFILE, ast=ast,
                    source_unit_ids=_source_unit_ids(),
                )

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        rule = ast["rule"]
        panel = ast["patients"]
        regime = candidate.regime
        cql_text = render_cql(rule if regime != "repair" else ast["broken_rule"])
        patient_ids = _patient_ids(panel)
        if regime == "repair":
            from benchmark.plugins.verifier.l08.elm import intent_text

            user_goal = (
                "Repair the clinical rule so it flags exactly the intended patients.\n\n"
                "Clinical intent: %s\n\n"
                "The rule below is the current (broken) version. Adjust its parameters "
                "(op, threshold, lookback_days, exclude) so the rule matches the intent. "
                "Submit {\"op\", \"threshold\", \"lookback_days\", \"exclude\"}.\n\n"
                "Conventions: %s\n\nBroken rule (CQL, explicit FHIR element paths):\n%s\n"
                "Synthetic FHIR R4 panel (%d patients; bundle per patient):\n%s" % (
                    intent_text(rule), _CONVENTION, cql_text, len(panel),
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
            "semantic_profile": PROFILE,
            "complexity": ast["complexity"],
            "data_provenance": "fully_synthetic_no_phi",
        }
        oracle_handle = "orl08_" + _sha(json_key(candidate.instance_id, rule))[:16]
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


def _source_unit_ids() -> tuple[str, ...]:
    return (
        "cql-clinical/grammar/cql.g4",
        "cql-clinical/schemas/elm/expression.xsd",
        "cql-clinical/npm/cql-execution",
        "cql-clinical/npm/cql-exec-fhir",
    )


PLUGIN = L08TaskFamilyPlugin()
