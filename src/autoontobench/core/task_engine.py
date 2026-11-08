"""Solver-grounded offline Gatekeeper blueprint builder (Review 5B §4 T2). NO LLM.
goal natively verified -> defects derive non-goal initial -> BFS over pinned-OPA-legal
patch actions -> replay-certify. Run: ``python -m autoontobench.core.task_engine``."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, List, Tuple

from autoontobench.contracts.models import (
    Authorship, Provenance, QC, SourceRef, TaskFamily, TaskSpecV5, TaskStatus, TaskTrack,
)
from autoontobench.core.domains.gatekeeper import (
    GatekeeperDomain, del_path, get_path, set_path, snapshot_sha,
)
from autoontobench.core.native_oracle import (
    GATEKEEPER_COMMIT, GatekeeperNativeOracle, sha256_json)
from autoontobench.core.protocols import NativeState

MAX_BFS_DEPTH = 4
REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATES_PATH = REPO_ROOT / "data" / "erule" / "review5" / "software" / "gk_templates.json"


def _templates(path: Path | None = None) -> List[Dict[str, Any]]:
    doc = json.loads((path or TEMPLATES_PATH).read_text(encoding="utf-8"))
    protos = doc["prototypes"]
    out: List[Dict[str, Any]] = []
    for raw in doc["templates"]:
        t = dict(raw)
        t["objects"] = {
            oid: copy.deepcopy(protos[ref[1:]]) if ref.startswith("@") else ref
            for oid, ref in raw["objects"].items()
        }
        t["provenance"] = Provenance(raw["provenance"])
        out.append(t)
    return out


def _candidates(objects: Dict[str, Any], goal: Dict[str, Any],
                fixable: Dict[str, List[str]],
                variants: Dict[str, Dict[str, List[Any]]]) -> List[Dict[str, Any]]:
    cands: List[Dict[str, Any]] = []
    for oid, paths in sorted(fixable.items()):
        diffs = []
        for p in paths:
            try:
                cur = get_path(objects[oid], p)
            except KeyError:
                cur = None
            if cur != get_path(goal[oid], p):
                diffs.append(p)
        if not diffs:
            continue
        for p in diffs:
            cands.append({"op": "patch_object", "object": oid, "set": {p: get_path(goal[oid], p)}})
            for v in (variants.get(oid, {}).get(p) or []):
                cands.append({"op": "patch_object", "object": oid, "set": {p: v}})
        if len(diffs) > 1:
            cands.append({"op": "patch_object", "object": oid,
                          "set": {p: get_path(goal[oid], p) for p in diffs}})
    return list({sha256_json(c): c for c in cands}.values())


def _at_goal(objects: Dict[str, Any], goal: Dict[str, Any],
             fixable: Dict[str, List[str]]) -> bool:
    def match(oid: str, p: str) -> bool:
        try:
            return get_path(objects[oid], p) == get_path(goal[oid], p)
        except KeyError:
            return False

    return all(match(oid, p) for oid, paths in fixable.items() for p in paths)


@dataclass(frozen=True)
class SearchProblem:
    """One blueprint's plan-search instance (groups the BFS inputs, Smell-2 fix)."""

    spec: TaskSpecV5
    initial: Dict[str, Any]
    goal: Dict[str, Any]
    fixable: Dict[str, List[str]]
    variants: Dict[str, Dict[str, List[Any]]]
    max_depth: int = MAX_BFS_DEPTH


def bfs_solve(domain: GatekeeperDomain,
              prob: SearchProblem) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Shortest legal-action plan from initial to goal; ''legal'' = domain.legal_action."""
    start = domain.reset(prob.spec, seed=prob.spec.seed)
    assert snapshot_sha(start) == sha256_json(prob.initial), "BFS start != initial state"
    search = {"nodes_expanded": 0, "denied_candidates": 0, "max_depth": prob.max_depth}
    if _at_goal(prob.initial, prob.goal, prob.fixable):
        return [], search
    queue = deque([(start, [])])
    visited = {snapshot_sha(start)}
    while queue:
        node, plan = queue.popleft()
        if len(plan) >= prob.max_depth:
            continue
        search["nodes_expanded"] += 1
        for action in _candidates(node.payload["objects"], prob.goal, prob.fixable,
                                  prob.variants):
            trial = dict(action, idempotence_key="bfs-probe")
            probe_state = NativeState(
                payload={**copy.deepcopy(node.payload), "applied_keys": []})
            if not domain.legal_action(probe_state, trial).legal:
                search["denied_candidates"] += 1
                continue
            nxt = domain.step(probe_state, trial).state
            sha = snapshot_sha(nxt)
            if sha in visited:
                continue
            new_plan = plan + [action]
            if _at_goal(nxt.payload["objects"], prob.goal, prob.fixable):
                search["solution_depth"] = len(new_plan)
                return new_plan, search
            visited.add(sha)
            queue.append((nxt, new_plan))
    return [], {**search, "exhausted": True}


def _spec(bid: str, tpl: Dict[str, Any], units, out_dir: Path, log_root: Path,
          goal_doc: Dict[str, Any], seed: int) -> TaskSpecV5:
    return TaskSpecV5(
        task_id=bid, domain="software_api", track=TaskTrack.stateful_action,
        task_family=TaskFamily.application,
        source_refs=[SourceRef(source_id="gatekeeper", source_rule_unit_id=u.rule_unit_id,
                               commit=GATEKEEPER_COMMIT, file_sha256=u.src_rego_sha256)
                     for u in units],
        source_group_id="gatekeeper-library@" + GATEKEEPER_COMMIT[:12],
        mechanism_group_id="+".join(sorted(u.rule_unit_id for u in units)),
        combination_signature=sha256_json({"units": sorted(u.rule_unit_id for u in units),
                                           "template": tpl["template_id"]}),
        user_goal=("Bring the listed Kubernetes resources into a policy-compliant state "
                   "while keeping every guarded resource byte-identical."),
        initial_state_ref=str(out_dir / "states" / (bid + ".initial.json")),
        allowed_tools=["list_resources", "inspect_state", "patch_resource", "commit_task"],
        public_evidence_ref=str(out_dir / "states" / (bid + ".initial.json")),
        ontology_contract="native-only/0",
        private_goal_verifier_ref=str(out_dir / "states" / (bid + ".goal.json")),
        private_native_truth_ref=str(log_root / "opa"),
        private_gold_artifact_ref="PENDING_W6", private_control_specs_ref="PENDING_W6",
        task_provenance=tpl["provenance"], task_authorship=Authorship.deterministic_template,
        split="dev",
        independent_native_checks=["native_policy_before", "native_policy_after",
                                   "goal_reached", "unrelated_guard_unchanged"],
        status=TaskStatus.CANDIDATE, seed=seed,
    )


def _replay(domain: GatekeeperDomain, spec: TaskSpecV5,
            plan: List[Dict[str, Any]]) -> Dict[str, Any]:
    state = domain.reset(spec, seed=spec.seed)
    for i, action in enumerate(plan):
        keyed = dict(action, idempotence_key="%s-step-%d" % (spec.task_id, i))
        decision = domain.legal_action(state, keyed)
        if not decision.legal:
            return {"goal_reached": False, "failed_step": i, "detail": decision.detail}
        state = domain.step(state, keyed).state
    result = domain.verify_terminal(spec, state, None)
    return {"goal_reached": result.goal_reached, "evidence_sha256": result.evidence["evidence_sha256"],
            "terminal_state_sha256": result.evidence["terminal_state_sha256"]}


def _interaction(units, plan) -> int:
    """Max # distinct authentic units a single plan action had to satisfy simultaneously."""
    by_kind = Counter(k for u in units for k in u.match_kinds)
    kind_of = {"pod": "Pod", "deployment": "Deployment", "service": "Service"}
    return max((by_kind.get(kind_of.get(a["object"].split("/")[0], ""), 0) for a in plan),
               default=0)


def build_blueprints(out_dir: Path, log_root: Path, seed: int = 137,
                     limit: int | None = None) -> List[Dict[str, Any]]:
    out_dir = Path(out_dir)
    (out_dir / "states").mkdir(parents=True, exist_ok=True)
    (out_dir / "private").mkdir(parents=True, exist_ok=True)
    oracle = GatekeeperNativeOracle(log_root)
    domain = GatekeeperDomain(oracle=oracle, log_root=log_root)
    unit_by_short = {u.unit.split("/")[-1]: u for u in oracle.policy_set}
    records: List[Dict[str, Any]] = []
    with (out_dir / "build_log.jsonl").open("w", encoding="utf-8") as blog:
        for idx, tpl in enumerate(_templates()[: limit or None], 1):
            bid = "gk_blueprint_%02d" % idx
            units = [unit_by_short[u] for u in tpl["units"]]
            goal_objects, initial_objects = tpl["objects"], copy.deepcopy(tpl["objects"])
            for d in tpl["defects"]:
                target = initial_objects[d["object"]]
                if d["op"] == "set":
                    set_path(target, d["path"], d["value"])
                else:
                    del_path(target, d["path"])
            goal_doc = {"compliant_objects": sorted({vc["object"] for vc in tpl["value_checks"]}),
                        "value_checks": tpl["value_checks"], "guards": tpl["guards"],
                        "forbidden_mutations": tpl["forbidden"],
                        "parameter_overrides": tpl["parameter_overrides"]}
            spec = _spec(bid, tpl, units, out_dir, Path(log_root), goal_doc, seed)
            (out_dir / "states" / (bid + ".initial.json")).write_text(
                json.dumps({"objects": initial_objects}, indent=2, sort_keys=True) + "\n")
            (out_dir / "states" / (bid + ".goal.json")).write_text(
                json.dumps(goal_doc, indent=2, sort_keys=True) + "\n")

            def row(status, **kw):
                blog.write(json.dumps({"blueprint": bid, "template": tpl["template_id"],
                                       "status": status, **kw}, sort_keys=True) + "\n")

            if _at_goal(initial_objects, goal_objects, tpl["fixable"]):
                row("REJECTED", reason="already_solved_at_reset")
                continue
            before = {oid: oracle.evaluate_object(obj, context=bid + ":before").compliant
                      for oid, obj in initial_objects.items()}
            if not all(oracle.evaluate_object(obj, context=bid + ":goalbuild").compliant
                       for oid, obj in goal_objects.items()
                       if oid in goal_doc["compliant_objects"]):
                row("REJECTED", reason="goal_state_not_natively_compliant")
                continue
            plan, search = bfs_solve(domain, SearchProblem(
                spec, initial_objects, goal_objects, tpl["fixable"], tpl["variants"]))
            if not plan:
                row("REJECTED", reason="unsolvable_within_depth_%d" % MAX_BFS_DEPTH,
                    search=search)
                continue
            replay = _replay(domain, spec, plan)
            if not replay["goal_reached"]:
                row("REJECTED", reason="replay_goal_not_reached", replay=replay)
                continue
            plan_path = out_dir / "private" / (bid + ".plan.json")
            keyed = [dict(a, idempotence_key="%s-step-%d" % (bid, i)) for i, a in enumerate(plan)]
            plan_path.write_text(json.dumps({"blueprint": bid, "plan": keyed}, indent=2,
                                            sort_keys=True) + "\n")
            record = {
                "blueprint_id": bid, "spec": json.loads(spec.model_dump_json()),
                "goal": goal_doc,
                "initial_state_sha256": sha256_json(initial_objects),
                "goal_state_sha256": sha256_json(goal_objects),
                "private_solution_plan_path": str(plan_path),
                "min_action_count": len(plan),
                "interacting_units": _interaction(units, plan),
                "unit_clause_counts": {u.rule_unit_id: len(u.parameters.get("labels", [])) or 1
                                       for u in units},
                "search": search, "native_before_compliance": before, "replay": replay,
                "qc": QC(native_pass=True, feasible=True, nontrivial=True).model_dump(),
                "status": TaskStatus.TASK_FEASIBLE.value,
                "provenance": {
                    "source": "https://github.com/open-policy-agent/gatekeeper-library",
                    "commit": GATEKEEPER_COMMIT,
                    "units": [{"rule_unit_id": u.rule_unit_id,
                               "src_rego_sha256": u.src_rego_sha256} for u in units],
                    "task_provenance": tpl["provenance"].value},
            }
            (out_dir / (bid + ".json")).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n")
            row("GENERATED", min_action_count=len(plan), search=search,
                replay_sha=replay["terminal_state_sha256"])
            records.append(record)
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Gatekeeper offline blueprints (no LLM)")
    ap.add_argument("--out", default="data/erule/review5/software/blueprints")
    ap.add_argument("--logs", default="data/erule/review5/software/logs")
    ap.add_argument("--seed", type=int, default=137)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    records = build_blueprints(Path(args.out), Path(args.logs), args.seed, args.limit)
    print("generated=%d -> %s" % (len(records), args.out))
    return 0 if records else 1


if __name__ == "__main__":
    raise SystemExit(main())
