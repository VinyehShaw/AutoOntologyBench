"""L12 20K expansion: candidate generation (propose) + grounding.

Solver-grounded reverse generation, same discipline as the legacy lane: every
corruption parameter is found by scanning with the native engine until the
corruption provably breaks the native check while the reference passes; the
verifier re-proves everything independently.

Instance picking uses **viable streams with offsets**: for each base goal the
generator enumerates the viable (trace, corruption) / (projection, answer)
combinations in a deterministic order (cached per source), and instance
``inst`` of base ``slice s`` takes stream offset ``s * instances_per_goal +
inst``. Distinct offsets therefore yield distinct task content by
construction; residual collisions (across families sharing a net, or vs the
legacy 8K lane) are still caught by the pipeline's collision gates.
"""
from __future__ import annotations

import random
from typing import Any, Callable, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.task_family.l12 import engine
from benchmark.plugins.task_family.l12.engine import RUNTIME as RT
from benchmark.plugins.task_family.l12.families import (
    GOAL_REPAIR_MODEL,
    GOAL_REPAIR_TRACE,
    FamilySpec,
)
from benchmark.plugins.task_family.l12.propose import (
    _DECLARE_CORRUPTIONS,
    _apply_declare_corruption,
    _constraint_check,
    _model_fits,
    _nonconformant_projection_ns,
    _skeleton_corruptions,
    _tbr_fits,
    _trace_pick_order,
)
from benchmark.plugins.task_family.l12.expand20k import compute
from benchmark.plugins.task_family.l12.expand20k.compute import answer_key
from benchmark.plugins.task_family.l12.expand20k.families import INSTANCES_PER_GOAL
from benchmark.plugins.task_family.l12.expand20k.registry import (
    FP_CONF_PAIRS,
    GENERATOR_VERSION,
    NEW_GOLD_PAIRS,
    NEW_LOGS,
    NEW_NOISY_PAIRS,
    PROJECTION_RULE,
    SEMANTIC_PROFILE,
    projection_ladder,
)

LANE_ID = "L12"
SEED_SALT = 20261101

#: bounded corruption-scan budgets (deterministic; keeps 10k-trace logs cheap)
SCAN_LIMIT = 8        # stop after collecting this many breaking corruptions
SCAN_MAX_PROBES = 120  # stop after probing this many candidate positions
STREAM_CACHE_LIMIT = 30  # max cached viable combos per stream (topup reaches ~26)
RELABEL_STREAM_LIMIT = 36  # dual-relabel: 4 slices x (5 + topup) offsets

#: gold pairs grouped by net: corruption streams are NET-scoped so tasks that
#: share a net can never collide on corrupted-trace content across logs.
_NET_PAIRS: dict[str, list[str]] = {}
for _k, _v in NEW_GOLD_PAIRS.items():
    _NET_PAIRS.setdefault(_v["net"], []).append(_k)
NET_KEYS: tuple[str, ...] = tuple(sorted(_NET_PAIRS))


def _ids(family_id: str, base_key: Any, inst_key: Any) -> tuple[str, str]:
    base_goal_id = "bg-" + engine.sha256_json(
        {"f": family_id, "b": base_key, "d": "20k"})[:20]
    instance_id = "in-" + engine.sha256_json(
        {"b": base_goal_id, "i": inst_key})[:20]
    return base_goal_id, instance_id


def _rng(*parts: Any) -> random.Random:
    return random.Random(engine.sha256_json(list(parts))[:16])


def _source_units(rels: list[str]) -> tuple[str, ...]:
    return tuple(engine.unit_id_for(r) for r in rels)


def _unit_meta(rel: str) -> dict[str, Any]:
    return {
        "unit_id": engine.unit_id_for(rel),
        "path": str(engine.PINNED_ROOT / rel),
        "sha256": engine.sha256_file(engine.PINNED_ROOT / rel),
    }


# ---------------------------------------------------------------------------
# bounded breaking-corruption scanners over the new gold pairs
# ---------------------------------------------------------------------------

def _breaking_distant_swaps(trace: list[str], net_rel: str) -> list[tuple[int, int]]:
    """Swaps with distance >= 3: never equals an adjacent swap (legacy f01),
    a 2- or 3-element segment reversal (f12), or a single relocation (f13)."""
    out: list[tuple[int, int]] = []
    probes = 0
    for i in range(len(trace)):
        for j in range(i + 3, len(trace)):
            if trace[i] == trace[j]:
                continue
            probes += 1
            if probes > SCAN_MAX_PROBES:
                return out
            c = list(trace)
            c[i], c[j] = c[j], c[i]
            if not _tbr_fits(c, net_rel):
                out.append((i, j))
                if len(out) >= SCAN_LIMIT:
                    return out
    return out


def _breaking_reversals(trace: list[str], net_rel: str) -> list[tuple[int, int]]:
    """Reversals of segments with >= 3 elements: a 2-element reversal would be
    an adjacent swap (legacy f01) and a 3-element reversal equals a
    distance-2 swap, so both are excluded by construction."""
    out: list[tuple[int, int]] = []
    probes = 0
    for i in range(len(trace)):
        for j in range(i + 3, len(trace) + 1):
            probes += 1
            if probes > SCAN_MAX_PROBES:
                return out
            c = trace[:i] + list(reversed(trace[i:j])) + trace[j:]
            if c != trace and not _tbr_fits(c, net_rel):
                out.append((i, j))
                if len(out) >= SCAN_LIMIT:
                    return out
    return out


def _breaking_moves(trace: list[str], net_rel: str) -> list[int]:
    out: list[int] = []
    for i in range(len(trace)):
        c = trace[:i] + trace[i + 1:] + [trace[i]]
        if not _tbr_fits(c, net_rel):
            out.append(i)
            if len(out) >= SCAN_LIMIT:
                return out
    return out


def _breaking_duplicates(trace: list[str], net_rel: str) -> list[tuple[int, int]]:
    """(label_pos, insert_pos): duplicate trace[label_pos] at insert_pos."""
    out: list[tuple[int, int]] = []
    probes = 0
    for k in range(len(trace)):
        for p in range(len(trace) + 1):
            probes += 1
            if probes > SCAN_MAX_PROBES:
                return out
            c = list(trace)
            c.insert(p, trace[k])
            if not _tbr_fits(c, net_rel):
                out.append((k, p))
                if len(out) >= SCAN_LIMIT:
                    return out
    return out


def _breaking_double_deletions(trace: list[str], net_rel: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    probes = 0
    for i in range(len(trace)):
        for j in range(i + 1, len(trace)):
            probes += 1
            if probes > SCAN_MAX_PROBES:
                return out
            c = trace[:i] + trace[i + 1:j] + trace[j + 1:]
            if not _tbr_fits(c, net_rel):
                out.append((i, j))
                if len(out) >= SCAN_LIMIT:
                    return out
    return out


# ---------------------------------------------------------------------------
# viable streams (deterministic order, cached per source)
# ---------------------------------------------------------------------------

_STREAMS: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _corruption_stream(
    family_id: str, net_key: str, scanner: Callable, apply: Callable
) -> list[dict[str, Any]]:
    """Viable (trace, corruption) combos for one NET, deterministic order.

    Net-scoped: iterates the gold pairs sharing the net in registry order and
    dedupes by corrupted-trace content, so tasks sharing a net can never
    collide on content across logs (rt100/rt10000 share roadtraffic.pnml).
    """
    cache_key = (family_id, net_key)
    if cache_key in _STREAMS:
        return _STREAMS[cache_key]
    out: list[dict[str, Any]] = []
    seen_content: set[tuple[str, ...]] = set()
    pick_seed = SEED_SALT + sum(ord(c) * (i + 1) for i, c in enumerate(family_id)) \
        + sum(ord(c) * (i + 1) for i, c in enumerate(net_key))
    for pair_key in _NET_PAIRS[net_key]:
        if len(out) >= STREAM_CACHE_LIMIT:
            break
        pair = NEW_GOLD_PAIRS[pair_key]
        traces = RT.traces(pair["log"])
        for trace_idx in _trace_pick_order(len(traces), pick_seed):
            if len(out) >= STREAM_CACHE_LIMIT:
                break
            ref = traces[trace_idx]
            for param in scanner(ref, pair["net"]):
                corrupted = apply(ref, param)
                if corrupted == ref:
                    continue
                ck = tuple(corrupted)
                if ck in seen_content:
                    continue
                seen_content.add(ck)
                out.append({
                    "pair_id": pair_key, "log_rel": pair["log"],
                    "net_rel": pair["net"], "trace_idx": trace_idx,
                    "reference_trace": ref,
                    "corrupted_trace": corrupted, "param": param,
                })
                if len(out) >= STREAM_CACHE_LIMIT:
                    break
    _STREAMS[cache_key] = out
    return out


def _apply_distant_swap(ref: list[str], param: tuple[int, int]) -> list[str]:
    i, j = param
    c = list(ref)
    c[i], c[j] = c[j], c[i]
    return c


def _apply_reversal(ref: list[str], param: tuple[int, int]) -> list[str]:
    i, j = param
    return ref[:i] + list(reversed(ref[i:j])) + ref[j:]


def _apply_move(ref: list[str], param: int) -> list[str]:
    return ref[:param] + ref[param + 1:] + [ref[param]]


def _apply_duplicate(ref: list[str], param: tuple[int, int]) -> list[str]:
    k, p = param
    c = list(ref)
    c.insert(p, ref[k])
    return c


def _apply_double_deletion(ref: list[str], param: tuple[int, int]) -> list[str]:
    i, j = param
    return ref[:i] + ref[i + 1:j] + ref[j + 1:]


def _gen_streamed_corruption(
    spec: FamilySpec, base: dict[str, Any], inst: int,
    scanner: Callable, apply: Callable, mode_of: Callable, guards: list[str],
) -> dict[str, Any] | None:
    stream = _corruption_stream(spec.family_id, base["net"], scanner, apply)
    offset = base["slice"] * INSTANCES_PER_GOAL + inst
    if offset >= len(stream):
        return None
    item = stream[offset]
    corruption = mode_of(item["param"])
    return {
        "pair_id": item["pair_id"], "log_rel": item["log_rel"],
        "net_rel": item["net_rel"],
        "trace_idx": item["trace_idx"],
        "corrupted_trace": item["corrupted_trace"],
        "corruption": corruption,
        "reference_trace": item["reference_trace"],
        "guards": guards,
    }


def _gen_distant_swap(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    return _gen_streamed_corruption(
        spec, base, inst, _breaking_distant_swaps, _apply_distant_swap,
        lambda p: {"mode": "swap_distant", "positions": [p[0], p[1]]},
        ["differs_from_initial", "same_multiset", "same_length"],
    )


def _gen_reversal(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    return _gen_streamed_corruption(
        spec, base, inst, _breaking_reversals, _apply_reversal,
        lambda p: {"mode": "reverse_segment", "segment": [p[0], p[1]]},
        ["differs_from_initial", "same_multiset", "same_length"],
    )


def _gen_move(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    return _gen_streamed_corruption(
        spec, base, inst, _breaking_moves, _apply_move,
        lambda p: {"mode": "move_to_end", "position": p},
        ["differs_from_initial", "same_multiset", "same_length"],
    )


def _gen_duplicate(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    return _gen_streamed_corruption(
        spec, base, inst, _breaking_duplicates, _apply_duplicate,
        lambda p: {"mode": "duplicate_event", "label_position": p[0],
                   "insert_position": p[1]},
        ["differs_from_initial", "shorter_than_initial"],
    )


def _gen_double_deletion(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    return _gen_streamed_corruption(
        spec, base, inst, _breaking_double_deletions, _apply_double_deletion,
        lambda p: {"mode": "delete_two_events", "positions": [p[0], p[1]]},
        ["differs_from_initial", "longer_than_initial"],
    )


# ---------------------------------------------------------------------------
# f16 dual relabel (viable transition-pair stream, cached per pair)
# ---------------------------------------------------------------------------

_RELABEL_STREAMS: dict[str, list[tuple[str, str, str, str]]] = {}


def _relabel_stream(pair_key: str) -> list[tuple[str, str, str, str]]:
    """Viable (t1, t2, label1, label2) combos, deterministic shuffled order."""
    if pair_key in _RELABEL_STREAMS:
        return _RELABEL_STREAMS[pair_key]
    pair = NEW_GOLD_PAIRS[pair_key]
    traces = RT.traces(pair["log"])
    transitions = RT.visible_transitions(pair["net"])
    alphabet = {a for t in traces for a in t}
    exercised = [t for t in sorted(transitions) if transitions[t] in alphabet]
    if len(exercised) < 2:
        _RELABEL_STREAMS[pair_key] = []
        return []
    _rng("dualrelabel-20k", pair_key).shuffle(exercised)
    out: list[tuple[str, str, str, str]] = []
    for x in range(len(exercised)):
        if len(out) >= RELABEL_STREAM_LIMIT:
            break
        for y in range(x + 1, len(exercised)):
            if len(out) >= RELABEL_STREAM_LIMIT:
                break
            t1, t2 = exercised[x], exercised[y]
            wrong = {t1: "zzz_obscured_a", t2: "zzz_obscured_b"}
            bad = RT.replay_on_relabeled(traces, pair["net"], wrong)
            if all(r["trace_is_fit"] for r in bad):
                continue  # not a defect (e.g. duplicate labels cover each other)
            out.append((t1, t2, transitions[t1], transitions[t2]))
    _RELABEL_STREAMS[pair_key] = out
    return out


def _gen_dual_relabel(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = NEW_GOLD_PAIRS[base["pair"]]
    stream = _relabel_stream(base["pair"])
    offset = base["slice"] * INSTANCES_PER_GOAL + inst
    if offset >= len(stream):
        return None
    t1, t2, l1, l2 = stream[offset]
    traces = RT.traces(pair["log"])
    alphabet = sorted({a for t in traces for a in t})
    return {
        "pair_id": base["pair"], "log_rel": pair["log"], "net_rel": pair["net"],
        "corrupted_tids": {
            t1: "zzz_obscured_%02d_a" % (offset % 100),
            t2: "zzz_obscured_%02d_b" % (offset % 100),
        },
        "reference_mapping": {t1: l1, t2: l2},
        "log_traces": len(traces),
        "alphabet": alphabet,
    }


# ---------------------------------------------------------------------------
# f17/f18 constraint repair over log-only bases (viable stream per log)
# ---------------------------------------------------------------------------

_CONSTRAINT_STREAMS: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _constraint_stream(family_id: str, operation: str, log_key: str) -> list[dict[str, Any]]:
    cache_key = (family_id, log_key)
    if cache_key in _CONSTRAINT_STREAMS:
        return _CONSTRAINT_STREAMS[cache_key]
    log_rel = NEW_LOGS[log_key]
    traces = RT.traces(log_rel)
    _, model = _constraint_check(operation, traces)
    pick_seed = SEED_SALT + sum(ord(c) * (i + 1) for i, c in enumerate(family_id))
    out: list[dict[str, Any]] = []
    seen_content: set[tuple[str, ...]] = set()
    for trace_idx in _trace_pick_order(len(traces), pick_seed):
        if len(out) >= STREAM_CACHE_LIMIT:
            break
        ref = traces[trace_idx]
        if tuple(ref) in seen_content:
            continue  # same variant already streamed
        if not _model_fits(operation, model, ref)["is_fit"]:
            continue
        if operation == "log_skeleton":
            candidates = _skeleton_corruptions(ref, model)
        else:
            candidates = [
                (op, _apply_declare_corruption(op, ref)) for op in _DECLARE_CORRUPTIONS
            ]
        for mode, corrupted in candidates:
            if corrupted == ref:
                continue
            if _model_fits(operation, model, corrupted)["is_fit"]:
                continue
            ck = tuple(corrupted)
            if ck in seen_content:
                continue
            seen_content.add(ck)
            out.append({
                "trace_idx": trace_idx, "reference_trace": ref,
                "corrupted_trace": corrupted, "mode": mode,
            })
            if len(out) >= STREAM_CACHE_LIMIT:
                break
    _CONSTRAINT_STREAMS[cache_key] = out
    return out


def _gen_constraint_repair(
    spec: FamilySpec, base: dict[str, Any], inst: int
) -> dict[str, Any] | None:
    log_rel = NEW_LOGS[base["log"]]
    stream = _constraint_stream(spec.family_id, spec.operation, base["log"])
    if inst >= len(stream):
        return None
    item = stream[inst]
    return {
        "log_key": base["log"], "log_rel": log_rel,
        "trace_idx": item["trace_idx"],
        "corrupted_trace": item["corrupted_trace"],
        "corruption": {"mode": item["mode"]},
        "reference_trace": item["reference_trace"],
        "guards": ["differs_from_initial"],
    }


# ---------------------------------------------------------------------------
# exact-artifact families: distinct-answer projection streams
# ---------------------------------------------------------------------------

def _distinct_projection_stream(
    operation: str, log_rel: str, net_rel: str | None = None
) -> list[dict[str, Any]]:
    """(projection, answer) pairs with pairwise-distinct answers, ladder-first.

    Ladder cutoffs come first (complexity-tiered); when more distinct answers
    are needed than the ladder provides, a bounded +-delta extension scans
    neighbouring cutoffs. Bounded to STREAM_CACHE_LIMIT entries.
    """
    total = len(RT.traces(log_rel))
    ladder = projection_ladder(total)
    extra: list[int] = []
    for delta in (1, 2, 3, 5, 8, 13, 21):
        for n0 in ladder:
            for n in (n0 - delta, n0 + delta):
                if 2 <= n <= total and n not in ladder and n not in extra:
                    extra.append(n)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for n in ladder + extra:
        if len(out) >= STREAM_CACHE_LIMIT:
            break
        ans = compute.compute_answer(operation, log_rel, n, net_rel)
        if not compute.nontrivial(operation, ans):
            continue
        k = answer_key(ans)
        if k in seen:
            continue
        seen.add(k)
        out.append({"first_n": n, "reference_answer": ans})
    return out


def _gen_exact_log(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    log_rel = NEW_LOGS[base["log"]]
    stream = _distinct_projection_stream(spec.operation, log_rel)
    if inst >= len(stream):
        return None
    item = stream[inst]
    return {
        "log_rel": log_rel,
        "projection": {"rule": PROJECTION_RULE, "first_n": item["first_n"]},
        "reference_answer": item["reference_answer"],
    }


def _gen_footprints_audit(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    return _gen_exact_log(spec, base, inst)


def _gen_nonconformant(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = NEW_NOISY_PAIRS[base["noisy_pair"]]
    log_rel, net_rel = pair["log"], pair["net"]
    traces = RT.traces(log_rel)
    verdicts = RT.replay(traces, net_rel)
    bad_full = [i for i, v in enumerate(verdicts) if not v["trace_is_fit"]]
    if not bad_full:
        return None  # noisy log fully fits the model; nothing to isolate
    projections = _nonconformant_projection_ns(len(traces), bad_full)
    cand: list[int] = list(projections)
    for delta in (1, 2, 3, 5, 8):
        for n0 in projections:
            for n in (n0 - delta, n0 + delta):
                if 2 <= n <= len(traces) and n not in cand:
                    cand.append(n)
    seen: set[str] = set()
    stream: list[int] = []
    for n in cand:
        if len(stream) >= STREAM_CACHE_LIMIT:
            break
        ans = {
            "nonconformant_trace_indices": [i for i in bad_full if i < n],
            "traces_total": n,
        }
        k = answer_key(ans)
        if k in seen:
            continue
        seen.add(k)
        stream.append(n)
    if inst >= len(stream):
        return None
    n = stream[inst]
    return {
        "log_rel": log_rel,
        "net_rel": net_rel,
        "noisy_pair": base["noisy_pair"],
        "projection": {"rule": PROJECTION_RULE, "first_n": n},
        "reference_answer": {
            "nonconformant_trace_indices": [i for i in bad_full if i < n],
            "traces_total": n,
        },
    }


def _gen_fp_conf_isolation(spec: FamilySpec, base: dict[str, Any], inst: int) -> dict[str, Any] | None:
    pair = FP_CONF_PAIRS[base["noisy_pair"]]
    log_rel, net_rel = pair["log"], pair["net"]
    traces = RT.traces(log_rel)
    ladder = projection_ladder(len(traces))
    extra: list[int] = []
    for delta in (1, 2, 3, 5, 8, 13, 21):
        for n0 in ladder:
            for n in (n0 - delta, n0 + delta):
                if 2 <= n <= len(traces) and n not in ladder and n not in extra:
                    extra.append(n)
    seen: set[str] = set()
    stream: list[dict[str, Any]] = []
    for n in ladder + extra:
        if len(stream) >= STREAM_CACHE_LIMIT:
            break
        ans = compute.compute_answer("footprints_conf", log_rel, n, net_rel)
        if not compute.nontrivial("footprints_conf", ans):
            continue
        k = answer_key(ans)
        if k in seen:
            continue
        seen.add(k)
        stream.append({"first_n": n, "reference_answer": ans})
    if inst >= len(stream):
        return None
    item = stream[inst]
    return {
        "log_rel": log_rel,
        "net_rel": net_rel,
        "noisy_pair": base["noisy_pair"],
        "projection": {"rule": PROJECTION_RULE, "first_n": item["first_n"]},
        "reference_answer": item["reference_answer"],
    }


_GENERATORS_20K = {
    "l12-f11-distant-swap-repair": _gen_distant_swap,
    "l12-f12-segment-reversal-repair": _gen_reversal,
    "l12-f13-event-relocation-repair": _gen_move,
    "l12-f14-duplicate-event-purge": _gen_duplicate,
    "l12-f15-double-event-restore": _gen_double_deletion,
    "l12-f16-dual-transition-relabel": _gen_dual_relabel,
    "l12-f17-skeleton-repair-newlogs": _gen_constraint_repair,
    "l12-f18-declare-repair-newlogs": _gen_constraint_repair,
    "l12-f19-dfg-audit-newlogs": _gen_exact_log,
    "l12-f20-variant-audit-newlogs": _gen_exact_log,
    "l12-f21-endpoint-audit-newlogs": _gen_exact_log,
    "l12-f22-nonconformant-isolation-newpairs": _gen_nonconformant,
    "l12-f23-footprints-audit": _gen_footprints_audit,
    "l12-f24-footprints-conf-isolation": _gen_fp_conf_isolation,
}


# ---------------------------------------------------------------------------
# plugin entry points
# ---------------------------------------------------------------------------

class Pm4pyTaskFamilies20k:
    """TaskFamilyPlugin for the L12 20K expansion (new units, new mechanisms)."""

    plugin_id = "l12.pm4py-task-families-20k"

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
        from benchmark.plugins.task_family.l12.expand20k.families import FAMILY_BY_ID_20K

        spec = FAMILY_BY_ID_20K[family]
        gen = _GENERATORS_20K[family]
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
                    "generator": GENERATOR_VERSION,
                    "seed": SEED_SALT,
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
        from benchmark.plugins.task_family.l12.expand20k.families import FAMILY_BY_ID_20K

        ast = candidate.ast
        spec = FAMILY_BY_ID_20K[candidate.family_id]
        task_id = "l12-%s" % candidate.instance_id
        oracle_ref = "l12-oracle-%s" % engine.sha256_json(
            {"task": task_id, "dataset": "20k"})[:16]
        rels = [ast["log_rel"]]
        if ast.get("net_rel"):
            rels.append(ast["net_rel"])
        public = {
            "user_goal": _user_goal(spec, ast),
            "task_material": _task_material(spec, ast),
            "allowed_tools": _allowed_tools(spec),
            "budget": {"actions": 5, "probes": 16},
            "answer_contract": _answer_contract(spec),
            "observation_refs": list(_source_units(rels)),
        }
        return GroundedTask(
            ast=candidate, public=public, oracle_ref=oracle_ref, native_evidence={}
        )


def _user_goal(spec: FamilySpec, ast: dict[str, Any]) -> str:
    if spec.goal_kind == GOAL_REPAIR_TRACE:
        if ast.get("pair_id"):
            return (
                "%s Pair '%s' (log %s, net %s); corrupted trace index %d in that "
                "log. Submit the repaired activity sequence."
                % (spec.intent, ast["pair_id"], ast["log_rel"], ast["net_rel"],
                   ast["trace_idx"])
            )
        return (
            "%s Log %s; corrupted trace index %d in that log. Submit the "
            "repaired activity sequence."
            % (spec.intent, ast["log_rel"], ast["trace_idx"])
        )
    if spec.goal_kind == GOAL_REPAIR_MODEL:
        return (
            "%s Pair '%s': net %s has %d obscured transition label(s); the full "
            "log %s (%d traces) must replay perfectly. Submit the corrected "
            "transition-id->label mapping."
            % (spec.intent, ast["pair_id"], ast["net_rel"],
               len(ast["corrupted_tids"]), ast["log_rel"], ast["log_traces"])
        )
    return (
        "%s Log %s, projection: first %d traces in document order. Submit the "
        "exact answer artifact."
        % (spec.intent, ast["log_rel"], ast["projection"]["first_n"])
    )


def _task_material(spec: FamilySpec, ast: dict[str, Any]) -> dict[str, Any]:
    if spec.goal_kind == GOAL_REPAIR_TRACE:
        material = {
            "log": _unit_meta(ast["log_rel"]),
            "trace_index_in_log": ast["trace_idx"],
            "corrupted_trace": ast["corrupted_trace"],
            "corruption_mode": ast["corruption"]["mode"],
            "native_check": spec.operation,
        }
        if ast.get("net_rel"):
            material["net"] = _unit_meta(ast["net_rel"])
        return material
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
    if ast.get("noisy_pair"):
        material["noisy_pair"] = ast["noisy_pair"]
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
