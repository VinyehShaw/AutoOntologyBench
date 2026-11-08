"""T5 checks over the public/agent-facing surface (W8 M3).

T5.3  wellformed_typed_input        TaskSpecV5 schema invariants
T5.8  leak_free                     assert_no_private_keys + W1 scanner tables on
                                    the agent-facing surface (projection minus the
                                    internal provenance block, wording, public state)
T5.9  wording_entails_certified_goal  rule-level: named entities ⊆ public state,
                                    certified compliant targets covered, hidden
                                    goal values/paths absent, guard + tool
                                    constraints entailed. Model critique: NOT_RUN.
T5.10 no_generated_answer_exposed   no solution/expected-answer keys or hidden
                                    plan path/value strings in public text
T5.11 action_length_calibrated      solver min_action_count == plan length, within
                                    budget, nontrivial (W2 qc)
T5.12 originality_dedup             canonical_sha over semantic payload incl. the
                                    certified goal core; paraphrases collapse to one
                                    goal, distinct goals stay distinct
T5.13 declared_track_and_split      track enum + split vocabulary
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from autoontobench.contracts.models import (
    TaskSpecV5, assert_no_private_keys, public_projection,
)
from autoontobench.core.domains.gatekeeper import get_path
from autoontobench.safety import leak_rules
from autoontobench.safety.leak_audit import scan_json_keys
from autoontobench.taskgen.check_types import CheckContext, CheckResult, CheckStatus, dep
from autoontobench.taskgen.checks_native import _scan_text_clean
from autoontobench.taskgen.checks_public_linked_data import (
    analytical_count_leak, analytical_hidden_values, check9_linked_data, is_linked_data,
    value_leaks,
)
from autoontobench.taskgen.native_check import load_private_plan

from autoontobench.taskgen.stated_targets import stated_value_checks, wording_states_target

#: T5.12/T5.13 live in checks_public_dedupe (size split); re-exported so existing
#: consumers (`validate.py`, `qualify_run.py`) keep working unchanged.
from autoontobench.taskgen.checks_public_dedupe import (  # noqa: F401
    check_declared_track_and_split,
    check_originality_dedup,
    spec_goal_payload,
)

P = CheckStatus.PASS
F = CheckStatus.FAIL

FORBIDDEN_PUBLIC_KEYS = {"expected_actions", "rule_text", "gold", "gold_ontology",
                         "native_oracle", "private_native_truth"}
#: fields kept out of the agent-facing projection (internal lineage provenance)
_INTERNAL_PROVENANCE = ("source_refs", "source_group_id", "mechanism_group_id",
                        "combination_signature")


def _stated_value_checks(ctx: CheckContext) -> list[Mapping[str, Any]]:
    return stated_value_checks(ctx.blueprint)


def check_wellformed_typed_input(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "wellformed_typed_input"
    if not spec.source_refs:
        return CheckResult(name, F, "source_refs empty")
    if not spec.user_goal.strip():
        return CheckResult(name, F, "user_goal empty")
    if not spec.allowed_tools:
        return CheckResult(name, F, "allowed_tools empty")
    return CheckResult(name, P)


def _initial_objects(spec: TaskSpecV5, ctx: CheckContext) -> Mapping[str, Any]:
    path = Path(spec.initial_state_ref)
    path = path if path.is_absolute() else ctx.repo_root / path
    data = json.loads(path.read_text(encoding="utf-8"))
    objects = data.get("objects")
    return objects if isinstance(objects, Mapping) else {}


def _hidden_goal_terms(
    spec: TaskSpecV5, ctx: CheckContext
) -> tuple[list[str], list[str]]:
    """Discriminating (value, path) pairs the agent must NOT be told: goal
    value_checks whose target differs from the initial state, plus the private
    plan's set/remove paths and set values. W15 stated-targets triples are public
    by design and excluded (both from the goal checks and from plan steps that
    exactly implement a stated triple)."""
    values: list[str] = []
    paths: list[str] = []
    blueprint = ctx.blueprint or {}
    if is_linked_data(spec):
        return analytical_hidden_values(blueprint), []
    stated = {(str(vc.get("object")), str(vc.get("path")), str(vc.get("equals")))
              for vc in _stated_value_checks(ctx)}
    objects = _initial_objects(spec, ctx)
    for vc in (blueprint.get("goal") or {}).get("value_checks", []):
        if (str(vc.get("object")), str(vc.get("path")), str(vc.get("equals"))) in stated:
            continue
        want = str(vc.get("equals"))
        try:
            current = get_path(objects.get(vc["object"], {}), vc["path"])
        except (KeyError, TypeError):
            current = None
        if str(current) != want:
            values.append(want)
            paths.append(str(vc["path"]))
    for action in load_private_plan(blueprint) or []:
        for path, value in (action.get("set") or {}).items():
            if any(p == str(path) and v == str(value) for _, p, v in stated):
                continue
            paths.append(str(path))
            values.append(str(value))
        for path in action.get("remove") or []:
            if any(p == str(path) for _, p, _ in stated):
                continue
            paths.append(str(path))
    return sorted(set(values)), sorted(set(paths))


def _value_leaks(text: str, values: list[str], paths: list[str]) -> list[str]:
    return value_leaks(text, values, paths)


def check_leak_free(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "leak_free"
    try:
        projection = public_projection(spec)
        assert_no_private_keys(projection)
    except ValueError as exc:
        return CheckResult(name, F, str(exc))
    agent_view = {k: v for k, v in projection.items() if k not in _INTERNAL_PROVENANCE}
    key_hits: list[str] = []
    scan_json_keys(agent_view, "$", key_hits)
    if key_hits:
        return CheckResult(name, F, f"forbidden keys in agent-facing view: {key_hits}")
    surfaces = {"agent_projection": json.dumps(agent_view, ensure_ascii=False)}
    if ctx.wording is not None:
        surfaces["wording"] = "\n".join(
            [ctx.wording.user_goal_en, ctx.wording.user_goal_zh,
             ctx.wording.plausible_reason or ""])
    try:
        surfaces["initial_state"] = json.dumps(
            _initial_objects(spec, ctx), ensure_ascii=False)
    except (OSError, ValueError):
        pass  # missing/unreadable initial state is reported by public_docs_parity
    for surface_name, text in surfaces.items():
        hits = _scan_text_clean(text)
        if hits:
            return CheckResult(name, F, f"W1 scanner hits in {surface_name}: {hits}")
    lowered = json.dumps(agent_view, ensure_ascii=False).lower()
    for key in FORBIDDEN_PUBLIC_KEYS:
        if key in lowered:
            return CheckResult(name, F, f"forbidden public text fragment: {key}")
    return CheckResult(name, P, "projection key-guard clean; W1 scanner clean on "
                                "agent-facing surface, wording and public state",
                       evidence={"surfaces": sorted(surfaces)})


def check_no_generated_answer_exposed(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "no_generated_answer_exposed"
    projection = public_projection(spec)
    for banned in ("expected_answer", "solution", "expected_actions"):
        if banned in projection:
            return CheckResult(name, F, f"public key {banned!r} present")
    if ctx.blueprint is None:
        return dep(name, "blueprint record (hidden goal terms)")
    values, paths = _hidden_goal_terms(spec, ctx)
    hits = _value_leaks(spec.user_goal.lower(), values, paths)
    if hits:
        return CheckResult(name, F, f"user_goal exposes hidden solution terms: {hits}")
    if is_linked_data(spec) and analytical_count_leak(spec.user_goal, ctx.blueprint or {}):
        return CheckResult(name, F, "user_goal exposes the certified answer cardinality")
    return CheckResult(name, P, "no answer keys; no hidden plan/goal terms in public text")


def check_wording_entails_certified_goal(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "wording_entails_certified_goal"
    if ctx.wording is None:
        return dep(name, "T4 Mode-A wording candidate for this blueprint")
    if ctx.blueprint is None:
        return dep(name, "blueprint record (certified goal core)")
    if is_linked_data(spec):
        return check9_linked_data(spec, ctx)
    wording = ctx.wording
    text = f"{wording.user_goal_en}\n{wording.user_goal_zh}".lower()
    public_entities = set(_initial_objects(spec, ctx))
    named = {str(e) for e in wording.named_entities_in_public_state}
    invented = sorted(named - public_entities)
    if invented:
        return CheckResult(name, F, f"wording names entities absent from public state: {invented}")
    goal = ctx.blueprint.get("goal") or {}
    uncovered = sorted(set(goal.get("compliant_objects", [])) - named)
    if uncovered:
        return CheckResult(
            name, F, f"wording never tasks the certified compliant targets: {uncovered}")
    stated = _stated_value_checks(ctx)
    if stated:
        raw_text = f"{wording.user_goal_en}\n{wording.user_goal_zh}"
        unstated = [
            "%s %s=%r" % (vc.get("object"), vc.get("path"), vc.get("equals"))
            for vc in stated
            if not (wording_states_target(wording.user_goal_en, vc)
                    and wording_states_target(wording.user_goal_zh, vc))
        ]
        if unstated:
            return CheckResult(
                name, F, "stated-targets wording omits certified goal terms "
                         f"(the wording IS the goal statement, W15): {unstated}")
    values, paths = _hidden_goal_terms(spec, ctx)
    leaks = _value_leaks(text, values, paths)
    if leaks:
        return CheckResult(name, F, f"wording exposes hidden goal terms: {leaks}")
    if "byte-identical" not in text:
        return CheckResult(name, F, "guard-preservation constraint not entailed by wording")
    tool_names = [t for t in spec.allowed_tools if t in text]
    if "only" not in text or not ("listed tool" in text or tool_names):
        return CheckResult(name, F, "tool-restriction constraint not entailed by wording")
    return CheckResult(
        name, P, "entities ⊆ public state; compliant targets covered; hidden values "
                 "absent; guard + tool constraints entailed (rule-level; DashScope "
                 "readability critique NOT_RUN at M3)"
                 + ("; stated-targets mode: wording states every goal triple "
                    "(W15 repair, exposure intended)" if stated else ""),
        evidence={"named_entities": sorted(named),
                  "compliant_objects": goal.get("compliant_objects", []),
                  "goal_mode": "stated_targets" if stated else "compliance_only",
                  "wording_model_critique": "NOT_RUN"})


def check_action_length_calibrated(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "action_length_calibrated"
    blueprint = ctx.blueprint
    if blueprint is None:
        return dep(name, "W2 blueprint record (solver min_action_count, qc)")
    min_actions = blueprint.get("min_action_count")
    if not isinstance(min_actions, int) or min_actions < 1:
        return CheckResult(name, F, f"solver min_action_count invalid: {min_actions!r}")
    plan = load_private_plan(blueprint)
    if plan is None:
        return dep(name, "private solution plan for plan-length cross-check")
    if len(plan) != min_actions:
        return CheckResult(
            name, F, f"plan length {len(plan)} != solver min_action_count {min_actions}")
    if spec.budgets.max_actions < min_actions:
        return CheckResult(
            name, F, f"budget max_actions {spec.budgets.max_actions} < min {min_actions}")
    nontrivial = bool((blueprint.get("qc") or {}).get("nontrivial"))
    if not nontrivial:
        return CheckResult(name, F, "W2 qc.nontrivial is not true")
    return CheckResult(name, P, "plan length == solver min_action_count ≤ budget; "
                                "solver-certified nontrivial",
                       evidence={"min_action_count": min_actions,
                                 "plan_len": len(plan),
                                 "max_actions": spec.budgets.max_actions,
                                 "interacting_units": blueprint.get("interacting_units")})
