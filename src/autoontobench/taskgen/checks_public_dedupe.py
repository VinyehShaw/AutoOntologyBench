"""T5 public-surface checks for goal identity and declarations (split from
checks_public.py to keep both modules reviewable):

T5.12 originality_dedup        canonical_sha over the semantic payload incl. the
                               certified goal core; paraphrases collapse to one
                               goal, distinct goals stay distinct
T5.13 declared_track_and_split track enum + split vocabulary
"""
from __future__ import annotations

from typing import Any, Mapping

from autoontobench.contracts.models import TaskSpecV5, TaskTrack
from autoontobench.taskgen.check_types import CheckContext, CheckResult, CheckStatus
from autoontobench.taskgen.dedupe import goal_sha

P = CheckStatus.PASS
F = CheckStatus.FAIL

VALID_SPLITS = {"dev", "validation", "hidden_source", "hidden_mechanism",
                "hidden_goal", "hidden_entity"}


def spec_goal_payload(
    spec: TaskSpecV5, blueprint: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "domain": spec.domain,
        "track": spec.track.value,
        "mechanism_group_id": spec.mechanism_group_id,
        "source_rule_unit_ids": sorted(r.source_rule_unit_id for r in spec.source_refs),
    }
    goal = (blueprint or {}).get("goal") or {}
    if goal:
        payload["goal_core"] = {
            "compliant_objects": sorted(str(o) for o in goal.get("compliant_objects", [])),
            "value_checks": sorted(
                (str(v.get("object")), str(v.get("path")), str(v.get("equals")))
                for v in goal.get("value_checks", [])),
            "guards": sorted(str(g["object"]) for g in goal.get("guards", [])),
            "forbidden_mutations": sorted(
                str(f["object"]) for f in goal.get("forbidden_mutations", [])),
        }
        if isinstance(goal.get("query"), Mapping):
            payload["goal_core"]["query"] = {
                str(k): str(v) for k, v in sorted(goal["query"].items())}
    return payload


def check_originality_dedup(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "originality_dedup"
    sha = goal_sha(spec_goal_payload(spec, ctx.blueprint))
    base_id = str((ctx.blueprint or {}).get("blueprint_id") or spec.task_id)
    prior = ctx.seen_goal_shas.get(sha)
    if prior is not None and prior != base_id:
        return CheckResult(name, F, f"semantic duplicate of {prior} (sha {sha[:12]})")
    if prior == base_id:
        return CheckResult(name, P, f"paraphrase of {base_id}; collapses to one goal",
                           evidence={"canonical_sha": sha})
    return CheckResult(name, P, "novel canonical goal", evidence={"canonical_sha": sha})


def check_declared_track_and_split(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "declared_track_and_split"
    if not isinstance(spec.track, TaskTrack):
        return CheckResult(name, F, f"unknown track {spec.track!r}")
    if spec.split not in VALID_SPLITS:
        return CheckResult(
            name, F, f"split {spec.split!r} not in {sorted(VALID_SPLITS)}")
    return CheckResult(name, P)
