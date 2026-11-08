"""Analytical (linked-data/1) branches of the public-surface T5 checks (W14).

Split out of `checks_public.py` to keep both modules under the review ceiling.
Semantics: the certified query target is the only public entity; the certified
expected answer (IDs/rows) and its cardinality are the hidden terms. The wording
must task the target verbatim (EN+ZH), demand exact set semantics, and carry no
reaction-ID tokens (any RHEA id is either answer leak or invention).
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from autoontobench.contracts.models import TaskSpecV5
from autoontobench.taskgen.check_types import CheckContext, CheckResult, CheckStatus
from autoontobench.taskgen.native_check import load_private_plan

F = CheckStatus.FAIL
P = CheckStatus.PASS

_RHEA_ID_TOKEN = re.compile(r"RHEA:\d+")


def value_leaks(text: str, values: list[str], paths: list[str]) -> list[str]:
    """Hidden-term scanner shared by the gk and linked-data checks."""
    hits = [v for v in values if len(v) >= 8 and v.lower() in text]
    for path in paths:
        if path.lower() in text:
            hits.append(f"path:{path}")
            continue
        segment = path.split(".")[-1].split("[")[0]
        for value in values:
            if len(value) < 8 and re.search(
                    re.escape(segment) + r"[^\w]{0,3}" + re.escape(value) + r"\b",
                    text):
                hits.append(f"pair:{segment}={value}")
    return hits


def is_linked_data(spec: TaskSpecV5) -> bool:
    return spec.ontology_contract == "linked-data/1"


def analytical_hidden_values(blueprint: Mapping[str, Any]) -> list[str]:
    plan = load_private_plan(blueprint) or []
    values: list[str] = []
    for action in plan:
        answer = action.get("answer")
        if isinstance(answer, list):
            values.extend(str(v) for v in answer)
    stats = blueprint.get("answer_stats") or {}
    if stats.get("expected_count") is not None:
        values.append(str(stats["expected_count"]))
    return sorted(set(values))


def analytical_count_leak(text: str, blueprint: Mapping[str, Any]) -> bool:
    count = (blueprint.get("answer_stats") or {}).get("expected_count")
    return count is not None and re.search(rf"\b{re.escape(str(count))}\b", text) is not None


def check9_linked_data(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "wording_entails_certified_goal"
    wording = ctx.wording
    assert wording is not None and ctx.blueprint is not None
    query = (ctx.blueprint.get("goal") or {}).get("query") or {}
    target = str(query.get("chebi", ""))
    if not target:
        return CheckResult(name, F, "linked-data/1 blueprint goal carries no query target")
    text_en, text_zh = wording.user_goal_en, wording.user_goal_zh
    named = {str(e) for e in wording.named_entities_in_public_state}
    invented = sorted(named - {target})
    if invented:
        return CheckResult(name, F, f"wording names entities absent from public state: {invented}")
    if target not in text_en or target not in text_zh:
        return CheckResult(name, F, f"certified query target {target} not named verbatim "
                                    "in both EN and ZH wording")
    if "exact" not in text_en.lower() and "complete" not in text_en.lower():
        return CheckResult(name, F, "exact set semantics not entailed by EN wording")
    if _RHEA_ID_TOKEN.search(text_en) or _RHEA_ID_TOKEN.search(text_zh):
        return CheckResult(name, F, "wording contains RHEA reaction IDs (answer content "
                                    "or invented entities)")
    hits = value_leaks(f"{text_en}\n{text_zh}".lower(),
                       analytical_hidden_values(ctx.blueprint), [])
    if hits:
        return CheckResult(name, F, f"wording exposes hidden answer terms: {hits[:5]}")
    return CheckResult(
        name, P, "query target verbatim EN+ZH; exact-set semantics entailed; entities ⊆ "
                 "public state; no reaction IDs or hidden terms (rule-level; DashScope "
                 "readability critique NOT_RUN)",
        evidence={"query_target": target, "named_entities": sorted(named),
                  "wording_model_critique": "NOT_RUN"})


__all__ = ["is_linked_data", "analytical_hidden_values", "analytical_count_leak",
           "check9_linked_data", "value_leaks"]
