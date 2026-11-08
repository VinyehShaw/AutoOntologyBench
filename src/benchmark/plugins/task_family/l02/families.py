"""L02 task families: 12 structural families over the pinned K8s VAP/CEL shard.

Solver-grounded reverse generation (build order §4.1), mirroring the L01
discipline under the CEL native profile ``cel-expression-subset/1``:

  compliant synthetic K8s-shaped object
  -> mechanism mutator induces a natively-proven DENY initial state
     (pinned cel-go evaluates a verbatim upstream validation expression false)
  -> reference repair reverts to a natively-proven ALLOW terminal state
  -> guards pin ungoverned identity fields

All acceptance decisions come from the pinned cel-go helper (engine.py);
mutators are proposals only. Tasks are complete user goals — repair a manifest
set so every governing CEL validation passes, tune the ReplicaLimit params to
the minimal admitting value, probe a hidden policy's decision boundary with
native evidence, produce an exact decision/message/match map — never a single
ALLOW/DENY guess.

Scope honesty: dyn-typed object/params/request only (no namespaceObject, no
variables, no authorizer). This lane does not claim full apiserver admission.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.task_family.l02 import engine

LANE_ID = "L02"
SEMANTIC_PROFILE = engine.PROFILE  # cel-expression-subset/1
GENERATOR_VERSION = "l02-families/1.0.0"

REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"
REGIME_APPLICATION = "application"

#: Policy handles are the public names for hidden policy configs (probe/answer
#: keys). Stable ordering keeps answers deterministic.
HANDLES = ("pol-a", "pol-b", "pol-c")


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    goal_kind: str  # repair_compliance | param_tuning | boundary_probe |
    #                 exact_decision_set | message_expression | match_gate
    mechanisms: tuple[str, ...]
    intent: str
    low: int
    medium: int
    high: int

    @property
    def total(self) -> int:
        return self.low + self.medium + self.high


#: 12 structural families. Distinctness = CEL mechanism semantics x goal shape
#: x object/request cardinality x param configuration; not unit renames.
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("l02-f01-static-replica-limit-repair", REGIME_REPAIR, "repair_compliance",
               ("static-replica-limit",),
               "Bring deployment replica counts under the static ceiling enforced by the admission policy.",
               10, 20, 20),
    FamilySpec("l02-f02-param-replica-limit-repair", REGIME_REPAIR, "repair_compliance",
               ("param-replica-limit",),
               "Bring deployment replica counts under the ceiling carried by the bound ReplicaLimit parameter object.",
               10, 20, 20),
    FamilySpec("l02-f03-param-tuning-minimal-admission", REGIME_ORGANIZATION, "param_tuning",
               ("param-replica-limit",),
               "Adjust the ReplicaLimit parameter object to the smallest maxReplicas value that admits every listed workload.",
               10, 20, 20),
    FamilySpec("l02-f04-boundary-evidence-probe", REGIME_APPLICATION, "boundary_probe",
               ("static-replica-limit", "param-replica-limit", "replica-envelope"),
               "Determine the maximum admissible replicas value for each listed hidden policy and prove it with native evaluations at the boundary and one step beyond.",
               10, 20, 20),
    FamilySpec("l02-f05-exact-decision-set", REGIME_APPLICATION, "exact_decision_set",
               ("static-replica-limit", "demo-name-namespace", "env-label-gate"),
               "Produce the exact per-object admission decision set (ALLOW/DENY per policy) for the given fleet under the hidden policy set.",
               10, 20, 20),
    FamilySpec("l02-f06-message-expression-reconstruction", REGIME_APPLICATION, "message_expression",
               ("param-replica-limit", "audit-annotation"),
               "Compute the exact violation/audit message strings each policy's messageExpression produces for the given objects.",
               10, 20, 20),
    FamilySpec("l02-f07-match-condition-request-gate", REGIME_APPLICATION, "match_gate",
               ("demo-name-namespace",),
               "Determine for each admission request whether the policy's matchConditions select it (all conditions true) and the per-condition values.",
               10, 20, 20),
    FamilySpec("l02-f08-env-label-gate-repair", REGIME_REPAIR, "repair_compliance",
               ("env-label-gate",),
               "Bring workload labels in line with the environment-label gate enforced by the policy.",
               10, 20, 20),
    FamilySpec("l02-f09-demo-name-namespace-repair", REGIME_REPAIR, "repair_compliance",
               ("demo-name-namespace",),
               "Qualify demo-named workloads so the name/namespace validation passes (rename or place them in the demo namespace).",
               10, 20, 20),
    FamilySpec("l02-f10-multi-policy-conjunction-repair", REGIME_REPAIR, "repair_compliance",
               ("static-replica-limit", "env-label-gate", "demo-name-namespace"),
               "Repair a fleet that violates several CEL validations at once (replicas, environment label, demo name) until every rule passes.",
               10, 20, 20),
    FamilySpec("l02-f11-replica-envelope-repair", REGIME_REPAIR, "repair_compliance",
               ("replica-envelope",),
               "Adjust the object's replicas into the [minReplicas, maxReplicas] envelope the CRD validation rule enforces.",
               10, 20, 20),
    FamilySpec("l02-f12-ttl-expiry-consistency-repair", REGIME_REPAIR, "repair_compliance",
               ("ttl-expiry",),
               "Make each lease record consistent so expiry is present and lies beyond created + ttl.",
               10, 20, 20),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}


def _family_targets(family: FamilySpec) -> list[tuple[str, int]]:
    return [("low", family.low), ("medium", family.medium), ("high", family.high)]


def _objects_per_complexity(complexity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}[complexity]


class KubernetesCelTaskFamilies:
    """TaskFamilyPlugin: propose() + ground() for the 12 L02 families."""

    plugin_id = "l02.kubernetes-cel-families"

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    # -- propose ---------------------------------------------------------------
    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int, constraints: dict[str, Any]
    ) -> Iterator[CandidateTaskAST]:
        del snapshot  # policy material is pinned by engine.PINNED_ROOT
        spec = FAMILY_BY_ID[family]
        quota_scale = float(constraints.get("quota_scale", 1.0))
        instances_per_goal = int(constraints.get("instances_per_goal", 5))
        for complexity, n_instances in _family_targets(spec):
            n_instances = max(1, round(n_instances * quota_scale))
            produced = 0
            attempt = 0
            while produced < n_instances and attempt < n_instances * 8:
                attempt += 1
                cand_seed = seed * 1_000_003 + attempt * 977 + produced
                candidate = _build_candidate(spec, complexity, cand_seed,
                                             produced % instances_per_goal)
                if candidate is not None:
                    produced += 1
                    yield candidate

    # -- ground ----------------------------------------------------------------
    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        public = {
            "user_goal": _user_goal(candidate),
            "objects": {oid: _public_view(o) for oid, o in (ast.get("objects") or {}).items()},
            "allowed_tools": _allowed_tools(ast["goal_kind"]),
            "budget": {"probes": ast["evidence_budget"]["probes"],
                       "actions": ast["evidence_budget"]["actions"]},
            "policy_intent": FAMILY_BY_ID[candidate.family_id].intent,
        }
        if ast["goal_kind"] in ("boundary_probe", "exact_decision_set"):
            public["policy_handles"] = [p["handle"] for p in ast["policy_configs"]]
        if ast["goal_kind"] == "param_tuning":
            public["current_params"] = dict(ast["current_params"])
        if ast["goal_kind"] == "message_expression":
            public["params"] = dict(ast["policy_configs"][0]["params"])
        if ast["goal_kind"] == "match_gate":
            public["requests"] = {rid: dict(r) for rid, r in ast["requests"].items()}
            public["match_conditions"] = [
                {"name": n, "expression": e}
                for n, e in ast["policy_configs"][0]["match_conditions"]
            ]
        oracle_ref = "l02-oracle-" + engine.sha256_json({
            "instance_id": candidate.instance_id, "ast": ast["fingerprint"]
        })[:16]
        return GroundedTask(
            ast=candidate,
            public=public,
            oracle_ref=oracle_ref,
            native_evidence={},
        )


# ---------------------------------------------------------------------------
# Candidate construction
# ---------------------------------------------------------------------------

def _allowed_tools(goal_kind: str) -> list[str]:
    if goal_kind == "param_tuning":
        return ["params_edit", "cel_policy_probe"]
    if goal_kind in ("boundary_probe", "exact_decision_set", "message_expression", "match_gate"):
        return ["cel_policy_probe"]
    return ["manifest_edit", "cel_policy_probe"]


def _user_goal(candidate: CandidateTaskAST) -> str:
    spec = FAMILY_BY_ID[candidate.family_id]
    ast = candidate.ast
    names = sorted((ast.get("objects") or {}).keys())
    kind = ast["goal_kind"]
    if kind == "repair_compliance":
        return "%s Objects in scope: %s. Edit the manifests so every governing CEL validation passes." % (
            spec.intent, ", ".join(names))
    if kind == "param_tuning":
        return "%s Objects in scope: %s. Reply with the adjusted parameter object." % (
            spec.intent, ", ".join(names))
    if kind == "boundary_probe":
        handles = [p["handle"] for p in ast["policy_configs"]]
        return "%s Policies in scope: %s. Reply with the boundary per policy and the native proof." % (
            spec.intent, ", ".join(handles))
    if kind == "exact_decision_set":
        handles = [p["handle"] for p in ast["policy_configs"]]
        return ("%s Objects in scope: %s. Policies in scope: %s. Reply with the complete mapping "
                "<object> -> <policy> -> ALLOW/DENY."
                % (spec.intent, ", ".join(names), ", ".join(handles)))
    if kind == "message_expression":
        return ("%s Objects in scope: %s. Reply with the exact message string per object."
                % (spec.intent, ", ".join(names)))
    if kind == "match_gate":
        reqs = sorted((ast.get("requests") or {}).keys())
        return ("%s Requests in scope: %s. Reply with the applies verdict and per-condition values per request."
                % (spec.intent, ", ".join(reqs)))
    raise ValueError("unknown goal_kind %r" % kind)


def _public_view(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if not k.startswith("__")}


def _policy_config(policy_id: str, params: dict[str, Any], handle: str) -> dict[str, Any]:
    unit = engine.load_policies()[policy_id]
    return {
        "handle": handle,
        "policy_id": unit.policy_id,
        "source_path": unit.source_path,
        "validations": list(unit.validations),
        "message_expressions": list(unit.message_expressions),
        "match_conditions": [[n, e] for n, e in unit.match_conditions],
        "params": dict(params),
    }


def _mk_deployments(n: int, rng: random.Random, *, replicas: int,
                    labels: dict[str, str] | None = None) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        name = "%s-%d" % (engine.NAME_POOL[(i + rng.randint(0, 3)) % len(engine.NAME_POOL)],
                          rng.randint(100, 999))
        out.append(engine.deployment(name, "default", replicas, labels))
    return out


def _build_candidate(
    spec: FamilySpec, complexity: str, seed: int, variant: int
) -> CandidateTaskAST | None:
    rng = random.Random(seed)
    kind = spec.goal_kind
    if kind == "repair_compliance":
        return _build_repair(spec, complexity, seed, variant, rng)
    if kind == "param_tuning":
        return _build_param_tuning(spec, complexity, seed, variant, rng)
    if kind == "boundary_probe":
        return _build_boundary_probe(spec, complexity, seed, variant, rng)
    if kind == "exact_decision_set":
        return _build_decision_set(spec, complexity, seed, variant, rng)
    if kind == "message_expression":
        return _build_message_expression(spec, complexity, seed, variant, rng)
    if kind == "match_gate":
        return _build_match_gate(spec, complexity, seed, variant, rng)
    raise ValueError("unknown goal_kind %r" % kind)


def _base_object(mechanism: str, rng: random.Random, idx: int) -> dict[str, Any]:
    name = "%s-%d" % (engine.NAME_POOL[(idx + rng.randint(0, 3)) % len(engine.NAME_POOL)],
                      rng.randint(100, 999))
    if mechanism == "replica-envelope":
        lo = rng.randint(1, 3)
        hi = lo + rng.randint(3, 12)
        return engine.replica_schedule(name, lo, rng.randint(lo, hi), hi)
    if mechanism == "ttl-expiry":
        created = 1_700_000_000 + rng.randint(0, 50_000_000)
        ttl = rng.randint(300, 86_400)
        return engine.ttl_object(name, created, ttl, created + ttl + rng.randint(1, 3600))
    labels: dict[str, str] = {
        "app": name,
        "example.com/environment": rng.choice(["test", "staging", "prod"]),
    }
    replicas = rng.randint(1, 5) if mechanism == "static-replica-limit" else rng.randint(1, 3)
    return engine.deployment(name, "default", replicas, labels)


def _params_for(mechanism: str, rng: random.Random) -> dict[str, Any]:
    if mechanism == "param-replica-limit":
        return {"maxReplicas": rng.choice([2, 3, 4, 5, 8, 10, 20, 50, 100])}
    return {}


def _build_repair(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_objects = _objects_per_complexity(complexity)
    mechanisms = list(spec.mechanisms)
    if spec.family_id == "l02-f10-multi-policy-conjunction-repair":
        # Rule interaction: violate 2 or 3 of the family's mechanisms per object.
        k = 2 if complexity == "low" else 3
        mechanisms = sorted(rng.sample(list(spec.mechanisms), k=k))
    params: dict[str, Any] = {}
    for mech in mechanisms:
        params.update(_params_for(mech, rng))
    policy_configs = []
    source_refs: set[str] = set()
    for handle, mech in zip(HANDLES, mechanisms):
        policy_id = engine.MECHANISMS[mech][0]
        policy_configs.append(_policy_config(policy_id, params, handle))
        source_refs.add(engine.load_policies()[policy_id].source_unit_ref)
    objects: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    for idx in range(n_objects):
        base = _base_object(mechanisms[0], rng, idx + variant)
        violated = base
        repaired_ok = True
        for mech in mechanisms:
            _pid, violate, repair = engine.MECHANISMS[mech]
            proposal = violate(violated, params, rng)
            if proposal is None:
                repaired_ok = False
                break
            violated = proposal
        if not repaired_ok:
            return None
        repaired = violated
        for mech in mechanisms:
            _pid, _v, repair = engine.MECHANISMS[mech]
            proposal = repair(repaired, params, rng)
            if proposal is None:
                return None
            repaired = proposal
        oid = "%s-%d" % (str(base.get("kind", "obj")).lower(), idx)
        objects[oid] = violated
        reference[oid] = repaired
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": objects,
        "reference_terminal": reference,
        "guards": _guards_for(spec, objects),
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {
            "probes": 4 + 2 * len(policy_configs) * n_objects,
            "actions": 4 + 3 * n_objects,
        },
    }
    return _finalize(spec, complexity, ast, seed, variant, source_refs)


def _guards_for(spec: FamilySpec, objects: dict[str, Any]) -> list[dict[str, Any]]:
    """Identity fields the family's mechanisms do not govern."""
    mechs = set(spec.mechanisms)
    guards = []
    for oid in objects:
        paths = ["kind"]
        if "demo-name-namespace" not in mechs:
            paths += ["metadata.name", "metadata.namespace"]
        guards.append({"object": oid, "paths": sorted(set(paths))})
    return guards


def _build_param_tuning(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_objects = _objects_per_complexity(complexity)
    current_params = {"maxReplicas": rng.choice([1, 2, 3])}
    objects: dict[str, Any] = {}
    for idx in range(n_objects):
        obj = _mk_deployments(1, rng, replicas=rng.randint(6, 60))[0]
        oid = "deployment-%d" % idx
        objects[oid] = obj
    needed = max(int(o["spec"]["replicas"]) for o in objects.values())
    reference_params = {"maxReplicas": needed}
    policy_configs = [_policy_config("vap-param-replica-limit", current_params, HANDLES[0])]
    source_refs = {engine.load_policies()["vap-param-replica-limit"].source_unit_ref}
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": objects,
        "current_params": current_params,
        "reference_params": reference_params,
        "guards": [{"object": oid, "paths": ["kind", "metadata.name", "metadata.namespace"]}
                   for oid in objects],
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {"probes": 4 + 2 * n_objects, "actions": 2},
    }
    return _finalize(spec, complexity, ast, seed, variant, source_refs)


def _build_boundary_probe(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_policies = {"low": 1, "medium": 2, "high": 3}[complexity]
    mechs = list(spec.mechanisms)
    rng.shuffle(mechs)
    chosen = mechs[:n_policies]
    policy_configs = []
    probes: dict[str, dict[str, Any]] = {}
    boundaries: dict[str, int] = {}
    source_refs: set[str] = set()
    for handle, mech in zip(HANDLES, chosen):
        policy_id = engine.MECHANISMS[mech][0]
        params = _params_for(mech, rng)
        if mech == "static-replica-limit":
            probe_obj = engine.deployment("probe-%d" % rng.randint(100, 999), "default", 1)
            boundary = 5
        elif mech == "param-replica-limit":
            probe_obj = engine.deployment("probe-%d" % rng.randint(100, 999), "default", 1)
            boundary = int(params["maxReplicas"])
        else:  # replica-envelope
            lo = rng.randint(1, 3)
            boundary = lo + rng.randint(3, 12)
            probe_obj = engine.replica_schedule(
                "probe-%d" % rng.randint(100, 999), lo, lo, boundary)
        policy_configs.append(_policy_config(policy_id, params, handle))
        probes[handle] = probe_obj
        boundaries[handle] = boundary
        source_refs.add(engine.load_policies()[policy_id].source_unit_ref)
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": {},
        "probe_objects": probes,
        "reference_answer": {"boundaries": boundaries},
        "guards": [],
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {"probes": 8 * n_policies, "actions": 1},
    }
    return _finalize(spec, complexity, ast, seed, variant, source_refs)


def _build_decision_set(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_objects = {"low": 2, "medium": 3, "high": 4}[complexity]
    policy_configs = []
    source_refs: set[str] = set()
    for handle, mech in zip(HANDLES, spec.mechanisms):
        policy_id = engine.MECHANISMS[mech][0]
        policy_configs.append(_policy_config(policy_id, {}, handle))
        source_refs.add(engine.load_policies()[policy_id].source_unit_ref)
    units = [engine.load_policies()[p["policy_id"]] for p in policy_configs]
    objects: dict[str, Any] = {}
    for attempt in range(8):
        fleet: dict[str, Any] = {}
        for idx in range(n_objects):
            name = "%s-%d" % (engine.NAME_POOL[(idx + attempt) % len(engine.NAME_POOL)],
                              rng.randint(100, 999))
            labels: dict[str, str] = {"app": name}
            if rng.random() < 0.6:
                labels["example.com/environment"] = rng.choice(["test", "staging", "prod"])
            if rng.random() < 0.4:
                name = "demo-%s" % name
            ns = "demo" if (name.startswith("demo") and rng.random() < 0.5) else "default"
            replicas = rng.choice([1, 2, 3, 4, 5, 6, 8, 12])
            obj = engine.deployment(name, ns, replicas, labels)
            oid = "deployment-%d" % idx
            fleet[oid] = obj
        decisions: list[str] = []
        for obj in fleet.values():
            for unit in units:
                rec = engine.policy_decision(
                    unit, {"object": _public_view(obj), "params": {}},
                    input_id="l02ds-%d" % attempt)
                decisions.append(rec["decision"])
        if decisions and len(set(decisions)) >= 2:
            objects = fleet
            break
    if not objects:
        return None
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": objects,
        "guards": _guards_for(spec, objects),
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {"probes": len(policy_configs) * n_objects + 2, "actions": 1},
    }
    return _finalize(spec, complexity, ast, seed, variant, source_refs)


def _build_message_expression(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_objects = 1 if complexity in ("low", "medium") else 2
    params = {"maxReplicas": rng.randint(2, 100)}
    objects: dict[str, Any] = {}
    for idx in range(n_objects):
        # replicas > params ceiling: the param policy DENYs, so its message is
        # the genuine violation message (the audit valueExpression fires either way).
        replicas = int(params["maxReplicas"]) + rng.randint(1, 50)
        objects["deployment-%d" % idx] = _mk_deployments(1, rng, replicas=replicas)[0]
    policy_configs = [
        _policy_config("vap-param-replica-limit", params, HANDLES[0]),
        _policy_config("vap-audit-annotation", {}, HANDLES[1]),
    ]
    source_refs = {
        engine.load_policies()["vap-param-replica-limit"].source_unit_ref,
        engine.load_policies()["vap-audit-annotation"].source_unit_ref,
    }
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": objects,
        "guards": [],
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {"probes": 2 * n_objects + 2, "actions": 1},
    }
    return _finalize(spec, complexity, ast, seed, variant, source_refs)


def _build_match_gate(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_requests = {"low": 2, "medium": 3, "high": 4}[complexity]
    policy_id = engine.MECHANISMS["demo-name-namespace"][0]
    unit = engine.load_policies()[policy_id]
    policy_configs = [_policy_config(policy_id, {}, HANDLES[0])]
    source_refs = {unit.source_unit_ref}
    request_pool = [
        {"resource": {"group": "coordination.k8s.io", "resource": "leases"},
         "userInfo": {"groups": ["system:authenticated"]}},
        {"resource": {"group": "rbac.authorization.k8s.io", "resource": "roles"},
         "userInfo": {"groups": ["system:authenticated"]}},
        {"resource": {"group": "apps", "resource": "deployments"},
         "userInfo": {"groups": ["system:nodes", "system:authenticated"]}},
        {"resource": {"group": "apps", "resource": "deployments"},
         "userInfo": {"groups": ["system:serviceaccounts"]}},
        {"resource": {"group": "", "resource": "configmaps"},
         "userInfo": {"groups": ["system:authenticated"]}},
    ]
    requests: dict[str, Any] = {}
    applies_seen: set[bool] = set()
    order = list(request_pool)
    rng.shuffle(order)
    for idx in range(n_requests):
        req = order[idx % len(order)]
        requests["req-%d" % idx] = req
    for rid, req in requests.items():
        verdicts = []
        for _name, expr in unit.match_conditions:
            rec = engine.eval_bool(expr, {"object": {}, "params": {}, "request": req},
                                   input_id="l02mg-%s" % rid)
            verdicts.append(bool(rec["value"]))
        applies_seen.add(all(verdicts))
    if len(applies_seen) < 2:
        # extend with pool members until both outcomes are present
        for extra in order:
            if len(requests) >= len(request_pool):
                break
            rid = "req-%d" % len(requests)
            if rid in requests:
                continue
            requests[rid] = extra
            verdicts = []
            for _name, expr in unit.match_conditions:
                rec = engine.eval_bool(expr, {"object": {}, "params": {}, "request": extra},
                                       input_id="l02mg-%s" % rid)
                verdicts.append(bool(rec["value"]))
            applies_seen.add(all(verdicts))
            if len(applies_seen) >= 2:
                break
    if len(applies_seen) < 2:
        return None
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": {},
        "requests": requests,
        "guards": [],
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {"probes": len(unit.match_conditions) * len(requests) + 2,
                            "actions": 1},
    }
    return _finalize(spec, complexity, ast, seed, variant, source_refs)


def _finalize(
    spec: FamilySpec, complexity: str, ast: dict[str, Any], seed: int, variant: int,
    source_refs: set[str],
) -> CandidateTaskAST:
    fingerprint = engine.sha256_json({
        "family": spec.family_id,
        "policies": [(p["policy_id"], engine.sha256_json(p["params"])) for p in ast["policy_configs"]],
        "complexity": complexity,
        "objects": {k: engine.sha256_json(_public_view(v))
                    for k, v in (ast.get("objects") or {}).items()},
        "requests": {k: engine.sha256_json(v) for k, v in (ast.get("requests") or {}).items()},
    })
    ast["fingerprint"] = fingerprint
    base_goal, instance = _ids(spec, complexity, ast, seed, variant)
    return CandidateTaskAST(
        lane_id=LANE_ID,
        family_id=spec.family_id,
        base_goal_id=base_goal,
        instance_id=instance,
        regime=spec.regime,
        semantic_profile=SEMANTIC_PROFILE,
        ast=ast,
        source_unit_ids=tuple(sorted(source_refs)),
    )


def _ids(spec: FamilySpec, complexity: str, ast: dict[str, Any], seed: int,
         variant: int) -> tuple[str, str]:
    """(base_goal_id, instance_id): goal = semantics sans concrete entity values."""
    base_payload = {
        "family": spec.family_id,
        "goal_kind": ast["goal_kind"],
        "complexity": complexity,
        "policies": [(p["policy_id"], engine.sha256_json(p["params"]),
                      engine.sha256_json(p["validations"])) for p in ast["policy_configs"]],
        "object_kinds": sorted({o.get("kind", "?")
                                for o in (ast.get("objects") or {}).values()}),
        "n_objects": len(ast.get("objects") or {}),
        "n_requests": len(ast.get("requests") or {}),
        "variant": variant,
    }
    base_goal = "bg-" + engine.sha256_json(base_payload)[:20]
    instance = "in-" + engine.sha256_json({
        "base_goal": base_goal,
        "objects": {k: engine.sha256_json(_public_view(v))
                    for k, v in (ast.get("objects") or {}).items()},
        "requests": {k: engine.sha256_json(v) for k, v in (ast.get("requests") or {}).items()},
        "seed": seed,
    })[:20]
    return base_goal, instance


def list_families() -> list[str]:
    return [f.family_id for f in FAMILIES]
