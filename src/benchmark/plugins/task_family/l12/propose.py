"""L12 candidate generation (propose) + grounding for the 10 families.

Generation is solver-grounded: every corruption parameter (swap position,
deletion index, foreign insert, obscured label, projection size) is chosen by
scanning with the native engine until the corruption provably breaks the
native check while the reference passes. The verifier re-proves everything
independently; generation-time checks only shape the search.
"""
from __future__ import annotations

import random
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT
from benchmark.plugins.task_family.l12.families import (
    FAMILY_BY_ID,
    GOAL_EXACT_ARTIFACT,
    GOAL_REPAIR_MODEL,
    GOAL_REPAIR_TRACE,
    INSTANCES_PER_GOAL,
    LANE_ID,
    SEMANTIC_PROFILE,
    FamilySpec,
)

SEED_SALT = 20261028


def _ids(family_id: str, base_key: Any, inst_key: Any) -> tuple[str, str]:
    base_goal_id = "bg-" + engine.sha256_json({"f": family_id, "b": base_key})[:20]
    instance_id = "in-" + engine.sha256_json({"b": base_goal_id, "i": inst_key})[:20]
    return base_goal_id, instance_id


def _rng(*parts: Any) -> random.Random:
    return random.Random(engine.sha256_json(list(parts))[:16])


def _projections_for(n_traces: int) -> list[int]:
    out: list[int] = []
    for k in range(1, 6):
        n = max(2, -(-n_traces * k // 5))
        if n not in out and n <= n_traces:
            out.append(n)
    n = 2
    while len(out) < 5 and n <= n_traces:
        if n not in out:
            out.append(n)
        n += 1
    return sorted(out)


def _source_units(rels: list[str]) -> tuple[str, ...]:
    return tuple(engine.unit_id_for(r) for r in rels)


def _unit_meta(rel: str) -> dict[str, Any]:
    return {
        "unit_id": engine.unit_id_for(rel),
        "path": str(engine.PINNED_ROOT / rel),
        "sha256": engine.sha256_file(engine.PINNED_ROOT / rel),
    }


# ---------------------------------------------------------------------------
# trace-repair families (f01/f02/f03) over gold pairs, check = strict TBR
# ---------------------------------------------------------------------------

def _tbr_fits(trace: list[str], net_rel: str) -> bool:
    return RT.replay([trace], net_rel)[0]["trace_is_fit"]


def _breaking_swaps(trace: list[str], net_rel: str) -> list[int]:
    out = []
    for i in range(len(trace) - 1):
        if trace[i] == trace[i + 1]:
            continue
        c = list(trace)
        c[i], c[i + 1] = c[i + 1], c[i]
        if not _tbr_fits(c, net_rel):
            out.append(i)
    return out


def _breaking_deletions(trace: list[str], net_rel: str) -> list[int]:
    out = []
    for i in range(len(trace)):
        c = list(trace)
        c.pop(i)
        if not _tbr_fits(c, net_rel):
            out.append(i)
    return out


def _trace_pick_order(n: int, seed: int) -> list[int]:
    order = list(range(n))
    _rng("trace-order", seed).shuffle(order)
    return order


def _gen_swap(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = engine.GOLD_PAIRS[base["pair"]]
    traces = RT.traces(pair["log"])
    for trace_idx in _trace_pick_order(len(traces), base["trace_seed"] + inst):
        ref = traces[trace_idx]
        breaking = _breaking_swaps(ref, pair["net"])
        if not breaking:
            continue
        pos = breaking[(base["trace_seed"] + inst) % len(breaking)]
        corrupted = list(ref)
        corrupted[pos], corrupted[pos + 1] = corrupted[pos + 1], corrupted[pos]
        return {
            "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
            "trace_idx": trace_idx, "corrupted_trace": corrupted,
            "corruption": {"mode": "swap_adjacent", "position": pos},
            "reference_trace": ref,
            "guards": ["differs_from_initial", "same_multiset", "same_length"],
        }
    return None


def _gen_deletion(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = engine.GOLD_PAIRS[base["pair"]]
    traces = RT.traces(pair["log"])
    for trace_idx in _trace_pick_order(len(traces), base["trace_seed"] + inst):
        ref = traces[trace_idx]
        breaking = _breaking_deletions(ref, pair["net"])
        if not breaking:
            continue
        pos = breaking[(base["trace_seed"] + inst) % len(breaking)]
        corrupted = list(ref)
        removed = corrupted.pop(pos)
        return {
            "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
            "trace_idx": trace_idx, "corrupted_trace": corrupted,
            "corruption": {"mode": "delete_event", "position": pos, "removed": removed},
            "reference_trace": ref,
            "guards": ["differs_from_initial", "longer_than_initial"],
        }
    return None


def _gen_foreign(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = engine.GOLD_PAIRS[base["pair"]]
    traces = RT.traces(pair["log"])
    alphabet = sorted({a for t in traces for a in t})
    for trace_idx in _trace_pick_order(len(traces), base["trace_seed"] + inst):
        ref = traces[trace_idx]
        rng = _rng("foreign", base["trace_seed"], inst, trace_idx)
        if inst % 2 == 0 or len(ref) < 2:
            label = "zzz_foreign_%02d" % inst
            pos = rng.randrange(len(ref) + 1)
            corrupted = list(ref)
            corrupted.insert(pos, label)
            if not _tbr_fits(corrupted, pair["net"]):
                return {
                    "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
                    "trace_idx": trace_idx, "corrupted_trace": corrupted,
                    "corruption": {"mode": "insert_foreign", "position": pos, "label": label},
                    "reference_trace": ref,
                    "guards": ["differs_from_initial", "shorter_than_initial"],
                }
            continue
        # known activity, wrong position: scan deterministically
        positions = list(range(len(ref) + 1))
        rng.shuffle(positions)
        done = False
        for label in alphabet:
            for pos in positions:
                corrupted = list(ref)
                corrupted.insert(pos, label)
                if not _tbr_fits(corrupted, pair["net"]):
                    done = True
                    break
            if done:
                break
        if done:
            return {
                "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
                "trace_idx": trace_idx, "corrupted_trace": corrupted,
                "corruption": {"mode": "insert_misplaced", "position": pos, "label": label},
                "reference_trace": ref,
                "guards": ["differs_from_initial", "shorter_than_initial"],
            }
    return None


# ---------------------------------------------------------------------------
# f04 model relabel
# ---------------------------------------------------------------------------

def _gen_relabel(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = engine.GOLD_PAIRS[base["pair"]]
    traces = RT.traces(pair["log"])
    transitions = RT.visible_transitions(pair["net"])
    tids = sorted(transitions)
    good = RT.replay(traces, pair["net"])
    if not all(r["trace_is_fit"] for r in good):
        return None  # pair lost gold status; honest rejection downstream
    order = list(range(len(tids)))
    _rng("relabel", base["label_seed"], inst).shuffle(order)
    for offset in order:
        pick = tids[offset]
        original = transitions[pick]
        wrong = "zzz_obscured_%02d" % inst
        bad = RT.replay_on_relabeled(traces, pair["net"], {pick: wrong})
        if all(r["trace_is_fit"] for r in bad):
            continue  # obscured label never exercised by the log; not a defect
        return {
            "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
            "corrupted_tids": {pick: wrong},
            "reference_mapping": {pick: original},
            "log_traces": len(traces),
            "alphabet": sorted({a for t in traces for a in t}),
        }
    return None


# ---------------------------------------------------------------------------
# f08 skeleton / f09 declare trace repair
# ---------------------------------------------------------------------------

def _constraint_check(operation: str, full_traces: list[list[str]]) -> Any:
    if operation == "log_skeleton":
        model = RT.skeleton(full_traces)
        return ("skeleton", model)
    return ("declare", RT.declare_model(full_traces))


def _model_fits(operation: str, model: Any, trace: list[str]) -> dict[str, Any]:
    if operation == "log_skeleton":
        return RT.skeleton_check([trace], model)[0]
    return RT.declare_check([trace], model)[0]


def _skeleton_corruptions(ref: list[str], model: dict[str, Any]) -> list[tuple[str, list[str]]]:
    out: list[tuple[str, list[str]]] = []
    never = sorted(model.get("never_together") or set())
    for pair_ab in never:
        a, b = sorted(pair_ab)
        if a in ref and b not in ref:
            c = list(ref)
            c.insert(ref.index(a) + 1, b)
            out.append(("never_together:%s+%s" % (a, b), c))
        if b in ref and a not in ref:
            c = list(ref)
            c.insert(ref.index(b) + 1, a)
            out.append(("never_together:%s+%s" % (a, b), c))
    freq = model.get("activ_freq") or {}
    for act in sorted(freq):
        counts = sorted(int(x) for x in freq[act])
        if not counts:
            continue
        have = ref.count(act)
        if have and have + 1 not in counts:
            c = list(ref)
            c.insert(ref.index(act) + 1, act)
            out.append(("activ_freq:%s" % act, c))
    return out


_DECLARE_CORRUPTIONS = ("del_last", "del_first", "swap_first_two", "dup_first", "reverse")


def _apply_declare_corruption(op: str, ref: list[str]) -> list[str]:
    c = list(ref)
    if op == "del_last" and c:
        c.pop()
    elif op == "del_first" and c:
        c.pop(0)
    elif op == "swap_first_two" and len(c) >= 2:
        c[0], c[1] = c[1], c[0]
    elif op == "dup_first" and c:
        c.insert(0, c[0])
    elif op == "reverse":
        c.reverse()
    return c


def _gen_constraint_repair(
    spec: FamilySpec, base: dict[str, Any], inst: int
) -> dict[str, Any] | None:
    pair = engine.GOLD_PAIRS[base["pair"]]
    traces = RT.traces(pair["log"])
    _, model = _constraint_check(spec.operation, traces)
    for trace_idx in _trace_pick_order(len(traces), base["trace_seed"] + inst):
        ref = traces[trace_idx]
        ref_verdict = _model_fits(spec.operation, model, ref)
        if not ref_verdict["is_fit"]:
            continue
        if spec.operation == "log_skeleton":
            candidates = _skeleton_corruptions(ref, model)
        else:
            candidates = [
                (op, _apply_declare_corruption(op, ref)) for op in _DECLARE_CORRUPTIONS
            ]
        for mode, corrupted in candidates:
            if corrupted == ref:
                continue
            verdict = _model_fits(spec.operation, model, corrupted)
            if verdict["is_fit"]:
                continue
            return {
                "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
                "trace_idx": trace_idx, "corrupted_trace": corrupted,
                "corruption": {"mode": mode},
                "reference_trace": ref,
                "guards": ["differs_from_initial"],
            }
    return None


# ---------------------------------------------------------------------------
# exact-artifact families (f05/f06/f07/f10)
# ---------------------------------------------------------------------------

def _compute_answer(operation: str, log_rel: str, n: int, net_rel: str | None) -> dict[str, Any]:
    traces = RT.traces(log_rel)
    proj = engine.project_first_n(traces, n)
    if operation == "dfg":
        ans = RT.dfg(proj)
        ans["traces_total"] = len(proj)
        return ans
    if operation == "variants":
        return {"variants": RT.variants(proj), "traces_total": len(proj)}
    if operation == "endpoints":
        ans = RT.endpoints(proj)
        ans["traces_total"] = len(proj)
        return ans
    if operation == "nonconformant_trace_set":
        assert net_rel is not None
        verdicts = RT.replay(proj, net_rel)
        bad = [i for i, v in enumerate(verdicts) if not v["trace_is_fit"]]
        return {"nonconformant_trace_indices": bad, "traces_total": len(proj)}
    raise ValueError("unknown operation %s" % operation)


def _nontrivial(operation: str, ans: dict[str, Any]) -> bool:
    if operation == "dfg":
        return len(ans["edges"]) >= 3
    if operation == "variants":
        return len(ans["variants"]) >= 2
    if operation == "endpoints":
        return len(ans["activity_counts"]) >= 2
    if operation == "nonconformant_trace_set":
        return 1 <= len(ans["nonconformant_trace_indices"]) < ans["traces_total"]
    return False


def _nonconformant_projection_ns(total: int, bad_full: list[int]) -> list[int]:
    """5 first-N cutoffs, each capturing a distinct nonempty bad-trace prefix."""
    out: list[int] = []
    for k in range(1, 6):
        idx = min(len(bad_full) - 1, -(-len(bad_full) * k // 5) - 1)
        n = min(total, bad_full[idx] + 1)
        if n >= 2 and n not in out:
            out.append(n)
    n = bad_full[0] + 1
    while len(out) < 5 and n <= total:
        if n not in out:
            out.append(n)
        n += 1
    return sorted(out)


def _gen_exact(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    if spec.operation == "nonconformant_trace_set":
        pair = engine.NOISY_PAIRS[base["noisy_pair"]]
        log_rel, net_rel = pair["log"], pair["net"]
        traces = RT.traces(log_rel)
        verdicts = RT.replay(traces, net_rel)
        bad_full = [i for i, v in enumerate(verdicts) if not v["trace_is_fit"]]
        if not bad_full:
            return None  # noisy log fully fits the model; nothing to isolate
        projections = _nonconformant_projection_ns(len(traces), bad_full)
        n = projections[inst % len(projections)]
        ans = {
            "nonconformant_trace_indices": [i for i in bad_full if i < n],
            "traces_total": n,
        }
        return {
            "log_rel": log_rel,
            "net_rel": net_rel,
            "noisy_pair": base["noisy_pair"],
            "projection": {"rule": "first_n_traces_document_order", "first_n": n},
            "reference_answer": ans,
        }
    log_rel, net_rel = engine.LOGS[base["log"]], None
    total = len(RT.traces(log_rel))
    projections = _projections_for(total)
    answers: dict[int, dict[str, Any]] = {}
    for n in projections:
        answers[n] = _compute_answer(spec.operation, log_rel, n, net_rel)
    chosen = projections[inst % len(projections)]
    ans = answers[chosen]
    # prefer a projection whose answer differs from earlier instances
    earlier = [answers[n] for n in projections[: inst % len(projections)]]
    if any(_answer_key(e) == _answer_key(ans) for e in earlier):
        for delta in range(1, 26):
            for n2 in (chosen - delta, chosen + delta):
                if 2 <= n2 <= total and n2 not in answers:
                    alt = _compute_answer(spec.operation, log_rel, n2, net_rel)
                    if _nontrivial(spec.operation, alt) and all(
                        _answer_key(e) != _answer_key(alt) for e in earlier
                    ):
                        answers[n2] = alt
                        chosen, ans = n2, alt
                        break
            else:
                continue
            break
    return {
        "log_rel": log_rel,
        "projection": {"rule": "first_n_traces_document_order", "first_n": chosen},
        "reference_answer": ans,
    }


def _answer_key(ans: dict[str, Any]) -> str:
    return engine.sha256_json({k: v for k, v in ans.items() if k != "traces_total"})


# ---------------------------------------------------------------------------
# plugin entry points
# ---------------------------------------------------------------------------

_GENERATORS = {
    "l12-f01-trace-order-repair": _gen_swap,
    "l12-f02-trace-event-restore": _gen_deletion,
    "l12-f03-trace-foreign-purge": _gen_foreign,
    "l12-f04-model-transition-relabel": _gen_relabel,
    "l12-f05-dfg-reconstruction": _gen_exact,
    "l12-f06-variant-partition": _gen_exact,
    "l12-f07-endpoint-frequency-audit": _gen_exact,
    "l12-f08-skeleton-conformance-repair": _gen_constraint_repair,
    "l12-f09-declare-conformance-repair": _gen_constraint_repair,
    "l12-f10-nonconformant-trace-isolation": _gen_exact,
}


class Pm4pyTaskFamilies:
    """TaskFamilyPlugin for lane L12."""

    plugin_id = "l12.pm4py-task-families"

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    def propose(
        self,
        snapshot: SourceSnapshot,
        *,
        family: str,
        seed: int,
        constraints: dict[str, Any],
    ) -> Iterator[CandidateTaskAST]:
        spec = FAMILY_BY_ID[family]
        gen = _GENERATORS[family]
        per_goal = int(constraints.get("instances_per_goal", INSTANCES_PER_GOAL))
        for base in spec.bases:
            base_key = dict(base)
            for inst in range(per_goal):
                payload = gen(spec, base, inst)
                if payload is None:
                    continue
                base_goal_id, instance_id = _ids(family, base_key, inst)
                ast = {
                    "goal_kind": spec.goal_kind,
                    "operation": spec.operation,
                    "regime": spec.regime,
                    "generator": "l12-propose/%s" % SEMANTIC_PROFILE.split("/")[-1],
                    **payload,
                }
                rels = [payload["log_rel"]]
                if payload.get("net_rel"):
                    rels.append(payload["net_rel"])
                yield CandidateTaskAST(
                    lane_id=LANE_ID,
                    family_id=family,
                    base_goal_id=base_goal_id,
                    instance_id=instance_id,
                    regime=spec.regime,
                    semantic_profile=SEMANTIC_PROFILE,
                    ast=ast,
                    source_unit_ids=_source_units(rels),
                )

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        spec = FAMILY_BY_ID[candidate.family_id]
        task_id = "l12-%s" % candidate.instance_id
        oracle_ref = "l12-oracle-%s" % engine.sha256_json({"task": task_id})[:16]
        public = {
            "user_goal": _user_goal(spec, ast),
            "task_material": _task_material(spec, ast),
            "allowed_tools": _allowed_tools(spec),
            "budget": {"actions": 5, "probes": 16},
            "answer_contract": _answer_contract(spec),
        }
        return GroundedTask(
            ast=candidate, public=public, oracle_ref=oracle_ref, native_evidence={}
        )


def _user_goal(spec: FamilySpec, ast: dict[str, Any]) -> str:
    if spec.goal_kind == GOAL_REPAIR_TRACE:
        return (
            "%s Pair '%s' (log %s, net %s); corrupted trace index %d in that log. "
            "Submit the repaired activity sequence."
            % (spec.intent, ast["pair_id"], ast["log_rel"], ast["net_rel"], ast["trace_idx"])
        )
    if spec.goal_kind == GOAL_REPAIR_MODEL:
        return (
            "%s Pair '%s': net %s has %d obscured transition label(s); the full log %s "
            "(%d traces) must replay perfectly. Submit the corrected transition-id->label mapping."
            % (
                spec.intent, ast["pair_id"], ast["net_rel"],
                len(ast["corrupted_tids"]), ast["log_rel"], ast["log_traces"],
            )
        )
    return (
        "%s Log %s, projection: first %d traces in document order. Submit the exact answer artifact."
        % (spec.intent, ast["log_rel"], ast["projection"]["first_n"])
    )


def _task_material(spec: FamilySpec, ast: dict[str, Any]) -> dict[str, Any]:
    if spec.goal_kind == GOAL_REPAIR_TRACE:
        return {
            "log": _unit_meta(ast["log_rel"]),
            "net": _unit_meta(ast["net_rel"]),
            "trace_index_in_log": ast["trace_idx"],
            "corrupted_trace": ast["corrupted_trace"],
            "corruption_mode": ast["corruption"]["mode"],
            "native_check": spec.operation,
        }
    if spec.goal_kind == GOAL_REPAIR_MODEL:
        return {
            "log": _unit_meta(ast["log_rel"]),
            "net": _unit_meta(ast["net_rel"]),
            "obscured_transitions": dict(ast["corrupted_tids"]),
            "log_alphabet": ast["alphabet"],
            "native_check": spec.operation,
        }
    material = {
        "log": _unit_meta(ast["log_rel"]),
        "projection": ast["projection"],
        "operation": spec.operation,
    }
    if ast.get("net_rel"):
        material["net"] = _unit_meta(ast["net_rel"])
    return material


def _allowed_tools(spec: FamilySpec) -> list[str]:
    if spec.goal_kind == GOAL_REPAIR_TRACE:
        return ["trace_edit", "pm4py_conformance_probe"]
    if spec.goal_kind == GOAL_REPAIR_MODEL:
        return ["model_edit", "pm4py_conformance_probe"]
    return ["log_audit", "pm4py_conformance_probe"]


def _answer_contract(spec: FamilySpec) -> dict[str, Any]:
    if spec.goal_kind == GOAL_REPAIR_TRACE:
        return {"kind": "trace", "submit": {"trace": ["<activity-name>"]}}
    if spec.goal_kind == GOAL_REPAIR_MODEL:
        return {"kind": "mapping", "submit": {"mapping": {"<transition-id>": "<label>"}}}
    return {"kind": "exact_artifact", "submit": {"answer": "<operation-specific artifact>"}}
