"""L01 task families: 20 structural families over the pinned Gatekeeper library.

Solver-grounded reverse generation (build order §4.1):
  compliant base fixture (upstream ``example_allowed*.yaml``)
  -> mechanism mutator induces a natively-proven DENY initial state
  -> reference repair reverts to a natively-proven ALLOW terminal state
  -> guards pin ungoverned identity fields
All acceptance decisions come from pinned ``opa eval`` (engine.py); mutators are
proposals only. Tasks are complete user goals (repair a manifest set so every
governing constraint passes / produce the exact per-object decision set under
the hidden constraint set), never a single ALLOW/DENY guess.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.task_family.l01 import engine

LANE_ID = "L01"
SEMANTIC_PROFILE = "gatekeeper-rego-offline/1"
GENERATOR_VERSION = "l01-families/1.0.0"

REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"
REGIME_APPLICATION = "application"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    goal_kind: str  # repair_compliance | exact_decision_set | organize_fleet
    mechanisms: tuple[str, ...]
    intent: str
    low: int
    medium: int
    high: int

    @property
    def total(self) -> int:
        return self.low + self.medium + self.high


#: 20 structural families. Distinctness = mechanism semantics x goal shape x
#: object cardinality x cross-object interaction; not unit renames.
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec("l01-f01-registry-allowlist-repair", REGIME_REPAIR, "repair_compliance",
               ("allowedrepos",),
               "Move all workload container images onto the approved registry prefixes so the admission policy passes.",
               6, 28, 22),
    FamilySpec("l01-f02-registry-blocklist-repair", REGIME_REPAIR, "repair_compliance",
               ("disallowedrepos",),
               "Migrate workloads off the blocked image registries while keeping them deployable.",
               6, 28, 22),
    FamilySpec("l01-f03-image-tag-policy-repair", REGIME_REPAIR, "repair_compliance",
               ("disallowedtags",),
               "Replace banned image tags (e.g. mutable 'latest'-style tags) with compliant pinned tags.",
               6, 28, 22),
    FamilySpec("l01-f04-image-digest-migration", REGIME_REPAIR, "repair_compliance",
               ("imagedigests",),
               "Rewrite container images to digest-pinned form (name@sha256:...) so every image is immutable.",
               6, 28, 22),
    FamilySpec("l01-f05-label-taxonomy-repair", REGIME_REPAIR, "repair_compliance",
               ("requiredlabels",),
               "Bring workload/namespace labels in line with the governed taxonomy (required keys and allowed values).",
               6, 28, 22),
    FamilySpec("l01-f06-annotation-governance-repair", REGIME_REPAIR, "repair_compliance",
               ("requiredannotations",),
               "Add the mandated ownership/runbook annotations with values that satisfy the annotation policy.",
               6, 28, 22),
    FamilySpec("l01-f07-replica-envelope-repair", REGIME_REPAIR, "repair_compliance",
               ("replicalimits",),
               "Adjust workload replica counts into the permitted envelope for the deployment policy.",
               6, 28, 22),
    FamilySpec("l01-f08-probe-coverage-repair", REGIME_REPAIR, "repair_compliance",
               ("requiredprobes",),
               "Ensure every container declares the required health probes (readiness/liveness as governed).",
               6, 28, 22),
    FamilySpec("l01-f09-sa-token-hardening", REGIME_REPAIR, "repair_compliance",
               ("automount-sa-token",),
               "Disable automounting of service account tokens on the governed workloads.",
               6, 28, 22),
    FamilySpec("l01-f10-service-exposure-hardening", REGIME_REPAIR, "repair_compliance",
               ("block-nodeport", "block-loadbalancer"),
               "Re-expose services through cluster-internal types only (no NodePort, no LoadBalancer).",
               5, 26, 23),
    FamilySpec("l01-f11-external-ip-allowlist", REGIME_REPAIR, "repair_compliance",
               ("externalip",),
               "Constrain Service externalIPs to the sanctioned address list (drop unsanctioned addresses).",
               6, 28, 22),
    FamilySpec("l01-f12-wildcard-ingress-removal", REGIME_REPAIR, "repair_compliance",
               ("block-wildcard-ingress",),
               "Replace wildcard/empty ingress hosts with concrete hostnames.",
               6, 28, 22),
    FamilySpec("l01-f13-https-only-ingress", REGIME_REPAIR, "repair_compliance",
               ("httpsonly",),
               "Make ingresses HTTPS-only: TLS configured and the allow-http=false annotation set.",
               6, 28, 22),
    FamilySpec("l01-f14-ephemeral-storage-limit", REGIME_REPAIR, "repair_compliance",
               ("ephemeralstoragelimit",),
               "Set container ephemeral-storage limits within the governed bound on every container.",
               6, 28, 22),
    FamilySpec("l01-f15-resource-ratio-compliance", REGIME_REPAIR, "repair_compliance",
               ("containerresourceratios",),
               "Rebalance container CPU requests/limits so limit/request ratios stay within policy.",
               6, 28, 22),
    FamilySpec("l01-f16-resource-bounds-compliance", REGIME_REPAIR, "repair_compliance",
               ("containerlimits", "containerrequests"),
               "Set container resource requests and limits inside the permitted CPU/memory bounds.",
               5, 26, 23),
    FamilySpec("l01-f17-storage-class-governance", REGIME_REPAIR, "repair_compliance",
               ("storageclass",),
               "Bring persistent volume claims onto sanctioned storage classes.",
               6, 28, 22),
    FamilySpec("l01-f18-unique-ingress-host-organization", REGIME_ORGANIZATION, "organize_fleet",
               ("uniqueingresshost",),
               "Reorganize ingress host assignments across the fleet so no two ingresses share a host.",
               4, 24, 26),
    FamilySpec("l01-f19-multi-policy-fleet-repair", REGIME_REPAIR, "repair_compliance",
               ("allowedrepos", "disallowedtags", "requiredlabels"),
               "Repair a fleet that violates several policies at once (registry, tag, label) until every constraint passes.",
               0, 26, 28),
    FamilySpec("l01-f20-exact-decision-set", REGIME_APPLICATION, "exact_decision_set",
               ("allowedrepos", "replicalimits", "requiredprobes"),
               "Produce the exact per-object admission decision set (ALLOW/DENY per policy) for the given fleet under the hidden constraint set.",
               6, 26, 22),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

#: Preferred sample constraint per mechanism (params shape drives mutators).
PREFERRED_SAMPLES: dict[str, list[str]] = {
    "allowedrepos": ["samples/repo-must-be-openpolicyagent"],
    "disallowedrepos": ["samples/disallowed-repos"],
    "disallowedtags": ["samples/disallowed-tags"],
    "imagedigests": ["samples/container-image-must-have-digest"],
    "requiredlabels": ["samples/all-must-have-owner", "samples/verify-label-key-only"],
    "requiredannotations": ["samples/all-must-have-certain-set-of-annotations"],
    "replicalimits": ["samples/replicalimits"],
    "requiredprobes": ["samples/must-have-probes"],
    "automount-sa-token": ["samples/automount-serviceaccount-token"],
    "block-nodeport": ["samples/block-node-port"],
    "block-loadbalancer": ["samples/block-load-balancer"],
    "externalip": ["samples/external-ip"],
    "block-wildcard-ingress": ["samples/block-wildcard-ingress"],
    "httpsonly": ["samples/ingress-https-only"],
    "ephemeralstoragelimit": ["samples/ephemeral-storage-limit"],
    "containerresourceratios": ["samples/container-must-meet-ratios"],
    "containerlimits": ["samples/container-must-have-limits"],
    "containerrequests": ["samples/container-must-have-requests"],
    "storageclass": ["samples/storageclass-allowlist", "samples/storageclass"],
    "uniqueingresshost": ["samples/unique-ingress-host"],
}

_INTENT_ENTITY = {
    "low": "the workload",
    "medium": "the listed workloads",
    "high": "the entire fleet",
}


def _family_targets(family: FamilySpec) -> list[tuple[str, int]]:
    """(complexity, n_instances) plan honoring the structural 20/40/40 mix."""
    return [("low", family.low), ("medium", family.medium), ("high", family.high)]


def _objects_per_complexity(complexity: str) -> int:
    return {"low": 1, "medium": 2, "high": 3}[complexity]


class GatekeeperTaskFamilies:
    """TaskFamilyPlugin: propose() + ground() for the 20 L01 families."""

    plugin_id = "l01.gatekeeper-families"

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    # -- propose ---------------------------------------------------------------
    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int, constraints: dict[str, Any]
    ) -> Iterator[CandidateTaskAST]:
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
                candidate = _build_candidate(snapshot, spec, complexity, cand_seed,
                                             produced % instances_per_goal)
                if candidate is not None:
                    produced += 1
                    yield candidate

    # -- ground ----------------------------------------------------------------
    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = candidate.ast
        public = {
            "user_goal": _user_goal(candidate),
            "objects": {oid: _public_view(o) for oid, o in ast["objects"].items()},
            "allowed_tools": ["manifest_edit", "opa_policy_probe"],
            "budget": {"probes": ast["evidence_budget"]["probes"],
                       "actions": ast["evidence_budget"]["actions"]},
            "policy_intent": FAMILY_BY_ID[candidate.family_id].intent,
        }
        oracle_ref = "l01-oracle-" + engine.sha256_json({
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

def _user_goal(candidate: CandidateTaskAST) -> str:
    spec = FAMILY_BY_ID[candidate.family_id]
    ast = candidate.ast
    names = sorted(ast["objects"].keys())
    if ast["goal_kind"] == "exact_decision_set":
        return (
            "%s Objects in scope: %s. Reply with the complete mapping "
            "<object> -> <policy> -> ALLOW/DENY."
            % (spec.intent, ", ".join(names))
        )
    return "%s Objects in scope: %s. Edit the manifests so the goal holds." % (
        spec.intent, ", ".join(names))


def _public_view(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if not k.startswith("__")}


def _pick_sample(mechanism: str, rng: random.Random) -> str | None:
    for sample in PREFERRED_SAMPLES.get(mechanism, []):
        try:
            engine.load_unit(engine.MECHANISMS[mechanism][0], sample)
            return sample
        except Exception:
            continue
    for sample in engine.sample_dirs(engine.MECHANISMS[mechanism][0]):
        try:
            engine.load_unit(engine.MECHANISMS[mechanism][0], sample)
            return sample
        except Exception:
            continue
    return None


def _compliant_bases(mechanism: str, sample: str) -> list[dict[str, Any]]:
    """Upstream case objects of the governed kind (fixtures, not yet filtered)."""
    unit = engine.MECHANISMS[mechanism][0]
    loaded = engine.load_unit(unit, sample)
    objs = []
    for doc in engine.load_case_objects(unit, sample):
        if doc.get("kind") in loaded.match_kinds:
            objs.append(doc)
    return objs


def _build_candidate(
    snapshot: SourceSnapshot,
    spec: FamilySpec,
    complexity: str,
    seed: int,
    variant: int,
) -> CandidateTaskAST | None:
    rng = random.Random(seed)
    n_objects = _objects_per_complexity(complexity)
    if spec.goal_kind == "exact_decision_set":
        return _build_decision_set(spec, complexity, seed, variant, rng)
    if spec.goal_kind == "organize_fleet":
        return _build_organize(spec, complexity, seed, variant, rng)
    return _build_repair(spec, complexity, seed, variant, rng, n_objects)


def _mk_objects(base: dict[str, Any], n: int, rng: random.Random,
                name_pool: list[str]) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        obj = copy.deepcopy(base)
        obj.pop("__fixture__", None)
        meta = obj.setdefault("metadata", {})
        meta["name"] = "%s-%s" % (name_pool[i % len(name_pool)], rng.randint(100, 999))
        meta.setdefault("namespace", "default")
        out.append(obj)
    return out


_NAME_POOL = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]


def _build_repair(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random,
    n_objects: int,
) -> CandidateTaskAST | None:
    chosen: dict[str, str] = {}
    policy_units = []
    unit_payloads = []
    source_refs: set[str] = set()
    for mech in spec.mechanisms:
        sample = _pick_sample(mech, rng)
        if sample is None:
            return None
        chosen[mech] = sample
        unit = engine.MECHANISMS[mech][0]
        loaded = engine.load_unit(unit, sample)
        policy_units.append(loaded)
        unit_payloads.append({
            "unit": unit, "sample": sample, "rule_unit_id": loaded.rule_unit_id,
            "parameters": loaded.parameters, "mechanism": mech,
        })
        source_refs.add("gatekeeper-library/src/%s/src.rego" % unit)
        source_refs.add("gatekeeper-library/library/%s/template.yaml" % unit)
        source_refs.add("gatekeeper-library/library/%s/%s/constraint.yaml" % (unit, sample))
    if len(spec.mechanisms) > 1:
        merged, merged_ref = _merge_multi_policy(spec, chosen, rng, n_objects)
        if merged is None or merged_ref is None:
            return None
        objects, reference_terminal = merged, merged_ref
    else:
        mech = spec.mechanisms[0]
        loaded = policy_units[0]
        bases = _compliant_bases(mech, chosen[mech])
        if not bases:
            return None
        base = copy.deepcopy(bases[(seed + variant) % len(bases)])
        violate = engine.MECHANISMS[mech][1]
        repair = engine.MECHANISMS[mech][2]
        objects = {}
        reference_terminal = {}
        objs = _mk_objects(base, n_objects, rng, _NAME_POOL)
        for idx, obj in enumerate(objs):
            oid = "%s-%d" % (obj.get("kind", "obj").lower(), idx)
            violated = violate(obj, loaded.parameters, rng)
            if violated is None:
                return None
            repaired = repair(violated, loaded.parameters, rng)
            if repaired is None:
                return None
            objects[oid] = violated
            reference_terminal[oid] = repaired
    fingerprint = engine.sha256_json({
        "family": spec.family_id,
        "units": [(p["unit"], p["sample"]) for p in unit_payloads],
        "complexity": complexity,
        "objects": {k: engine.sha256_json(_public_view(v)) for k, v in objects.items()},
    })
    cluster_inventory: list[dict[str, Any]] = []
    for p in unit_payloads:
        cluster_inventory.extend(engine.load_inventory_fixtures(p["unit"], p["sample"]))
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_units": unit_payloads,
        "objects": objects,
        "reference_terminal": reference_terminal,
        "guards": _guards_for(objects),
        "complexity": complexity,
        "fingerprint": fingerprint,
        "evidence_budget": {
            "probes": 4 + 2 * len(policy_units),
            "actions": 4 + 3 * len(objects),
        },
    }
    if cluster_inventory:
        ast["cluster_inventory"] = cluster_inventory
    ids = _ids(spec, complexity, ast, seed, variant)
    return CandidateTaskAST(
        lane_id=LANE_ID,
        family_id=spec.family_id,
        base_goal_id=ids[0],
        instance_id=ids[1],
        regime=spec.regime,
        semantic_profile=SEMANTIC_PROFILE,
        ast=ast,
        source_unit_ids=tuple(sorted(source_refs)),
    )


def _merge_multi_policy(
    spec: FamilySpec, chosen: dict[str, str], rng: random.Random, n_objects: int
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """One object set violated under every mechanism; reference repairs all."""
    per_mech: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for mech in spec.mechanisms:
        sample = chosen[mech]
        unit = engine.MECHANISMS[mech][0]
        loaded = engine.load_unit(unit, sample)
        bases = _compliant_bases(mech, sample)
        if not bases:
            return None, None
        base = copy.deepcopy(bases[0])
        violate = engine.MECHANISMS[mech][1]
        repair = engine.MECHANISMS[mech][2]
        objs = _mk_objects(base, n_objects, rng, _NAME_POOL)
        cur_objects: dict[str, Any] = {}
        for idx, obj in enumerate(objs):
            oid = "%s-%d" % (obj.get("kind", "obj").lower(), idx)
            violated = violate(obj, loaded.parameters, rng)
            if violated is None:
                return None, None
            cur_objects[oid] = violated
        per_mech.append((cur_objects, {}))
    objects = per_mech[0][0]
    for cur_objects, _cur_ref in per_mech[1:]:
        for oid in objects:
            if oid in cur_objects:
                objects[oid] = _deep_overlay(objects[oid], cur_objects[oid])
    reference: dict[str, Any] = {}
    for oid, violated in objects.items():
        repaired: dict[str, Any] = violated
        for mech in spec.mechanisms:
            loaded = engine.load_unit(engine.MECHANISMS[mech][0], chosen[mech])
            repaired = engine.MECHANISMS[mech][2](repaired, loaded.parameters, rng)
            if repaired is None:
                return None, None
        reference[oid] = repaired
    return objects, reference


def _deep_overlay(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(a)
    for key, val in b.items():
        if key == "__fixture__":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_overlay(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _guards_for(objects: dict[str, Any]) -> list[dict[str, Any]]:
    guards = []
    for oid in objects:
        guards.append({"object": oid, "paths": ["metadata.name", "metadata.namespace", "kind"]})
    return guards


def _build_decision_set(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    n_objects = {"low": 2, "medium": 3, "high": 4}[complexity]
    policy_units = []
    unit_payloads = []
    source_refs: set[str] = set()
    objects: dict[str, Any] = {}
    chosen: dict[str, str] = {}
    for mech in spec.mechanisms:
        sample = _pick_sample(mech, rng)
        if sample is None:
            return None
        chosen[mech] = sample
        unit = engine.MECHANISMS[mech][0]
        loaded = engine.load_unit(unit, sample)
        policy_units.append(loaded)
        unit_payloads.append({
            "unit": unit, "sample": sample, "rule_unit_id": loaded.rule_unit_id,
            "parameters": loaded.parameters, "mechanism": mech,
        })
        source_refs.add("gatekeeper-library/src/%s/src.rego" % unit)
        source_refs.add("gatekeeper-library/library/%s/template.yaml" % unit)
        source_refs.add("gatekeeper-library/library/%s/%s/constraint.yaml" % (unit, sample))
    base_kind_objs: list[tuple[str, dict[str, Any]]] = []
    for mech in spec.mechanisms:
        for doc in _compliant_bases(mech, chosen[mech]):
            base_kind_objs.append((mech, doc))
    if not base_kind_objs:
        return None
    loaded_by_mech = {m: engine.load_unit(engine.MECHANISMS[m][0], chosen[m]) for m in spec.mechanisms}

    def _fleet(attempt: int) -> dict[str, Any]:
        fleet: dict[str, Any] = {}
        for idx in range(n_objects):
            mech, base0 = base_kind_objs[(seed + variant + idx + attempt) % len(base_kind_objs)]
            base = copy.deepcopy(base0)
            violate = engine.MECHANISMS[mech][1]
            obj = _mk_objects(base, 1, rng, _NAME_POOL)[0]
            if (idx + attempt) % 2 == 1:
                violated = violate(obj, loaded_by_mech[mech].parameters, rng)
                if violated is not None:
                    obj = violated
            oid = "%s-%d" % (obj.get("kind", "obj").lower(), idx)
            objects_key = oid
            fleet[objects_key] = obj
        return fleet

    # Nontrivial by native construction: the fleet must not be all-ALLOW or
    # all-DENY under its governing units; bounded retries reshuffle fixtures.
    objects = {}
    for attempt in range(6):
        fleet = _fleet(attempt)
        decisions: list[str] = []
        for oid, obj in fleet.items():
            for mech in spec.mechanisms:
                loaded = loaded_by_mech[mech]
                if obj.get("kind") in loaded.match_kinds:
                    rec = engine.opa_eval_object(loaded, obj, context="l01ds-%d" % attempt)
                    decisions.append(rec["decision"])
        if decisions and len(set(decisions)) >= 2:
            objects = fleet
            break
    if not objects:
        return None
    fingerprint = engine.sha256_json({
        "family": spec.family_id,
        "units": [(p["unit"], p["sample"]) for p in unit_payloads],
        "complexity": complexity,
        "objects": {k: engine.sha256_json(_public_view(v)) for k, v in objects.items()},
    })
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_units": unit_payloads,
        "objects": objects,
        "guards": _guards_for(objects),
        "complexity": complexity,
        "fingerprint": fingerprint,
        "evidence_budget": {"probes": 2 * len(policy_units) * len(objects), "actions": 1},
    }
    ids = _ids(spec, complexity, ast, seed, variant)
    return CandidateTaskAST(
        lane_id=LANE_ID,
        family_id=spec.family_id,
        base_goal_id=ids[0],
        instance_id=ids[1],
        regime=spec.regime,
        semantic_profile=SEMANTIC_PROFILE,
        ast=ast,
        source_unit_ids=tuple(sorted(source_refs)),
    )


def _build_organize(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random
) -> CandidateTaskAST | None:
    """uniqueingresshost: fleet reorganization with inventory-backed semantics."""
    mech = spec.mechanisms[0]
    sample = _pick_sample(mech, rng)
    if sample is None:
        return None
    unit = engine.MECHANISMS[mech][0]
    loaded = engine.load_unit(unit, sample)
    bases = _compliant_bases(mech, sample)
    if not bases:
        return None
    n_objects = {"low": 2, "medium": 3, "high": 4}[complexity]
    base = copy.deepcopy(bases[0])
    objects: dict[str, Any] = {}
    reference_terminal: dict[str, Any] = {}
    shared_host = "shared-%d.example.com" % rng.randint(1, 999)
    for idx in range(n_objects):
        obj = _mk_objects(base, 1, rng, ["web", "api", "shop", "blog"])[0]
        meta = obj.setdefault("metadata", {})
        meta["name"] = "ing-%s-%d" % (_NAME_POOL[(seed + idx) % len(_NAME_POOL)], rng.randint(10, 99))
        rules = obj.setdefault("spec", {}).setdefault("rules", [{"host": "", "http": {"paths": []}}])
        for r_idx, rule in enumerate(rules):
            rule["host"] = shared_host if r_idx == 0 else "shared-alt-%d.example.com" % (
                seed % 1000)
        oid = "ingress-%d" % idx
        objects[oid] = obj
        repaired = copy.deepcopy(obj)
        for r_idx, rule in enumerate(repaired["spec"]["rules"]):
            rule["host"] = "%s-r%d.fleet-%d.example.com" % (
                meta["name"].replace("_", "-"), r_idx, seed % 1000)
        reference_terminal[oid] = repaired
    fingerprint = engine.sha256_json({
        "family": spec.family_id,
        "units": [(unit, sample)],
        "complexity": complexity,
        "objects": {k: engine.sha256_json(_public_view(v)) for k, v in objects.items()},
    })
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_units": [{
            "unit": unit, "sample": sample, "rule_unit_id": loaded.rule_unit_id,
            "parameters": loaded.parameters, "mechanism": mech,
        }],
        "objects": objects,
        "reference_terminal": reference_terminal,
        "guards": _guards_for(objects),
        "complexity": complexity,
        "inventory_required": True,
        "fingerprint": fingerprint,
        "evidence_budget": {"probes": 2 * n_objects, "actions": 2 * n_objects},
    }
    ids = _ids(spec, complexity, ast, seed, variant)
    return CandidateTaskAST(
        lane_id=LANE_ID,
        family_id=spec.family_id,
        base_goal_id=ids[0],
        instance_id=ids[1],
        regime=spec.regime,
        semantic_profile=SEMANTIC_PROFILE,
        ast=ast,
        source_unit_ids=(
            "gatekeeper-library/src/%s/src.rego" % unit,
            "gatekeeper-library/library/%s/template.yaml" % unit,
            "gatekeeper-library/library/%s/%s/constraint.yaml" % (unit, sample),
        ),
    )


def _ids(spec: FamilySpec, complexity: str, ast: dict[str, Any], seed: int,
         variant: int) -> tuple[str, str]:
    """(base_goal_id, instance_id): goal = semantics sans concrete entity values."""
    base_payload = {
        "family": spec.family_id,
        "goal_kind": ast["goal_kind"],
        "complexity": complexity,
        "units": [(p["unit"], p["sample"]) for p in ast["policy_units"]],
        "params": [engine.sha256_json(p["parameters"]) for p in ast["policy_units"]],
        "object_kinds": sorted({o.get("kind", "?") for o in ast["objects"].values()}),
        "n_objects": len(ast["objects"]),
        "variant": variant,
    }
    base_goal = "bg-" + engine.sha256_json(base_payload)[:20]
    instance = "in-" + engine.sha256_json({
        "base_goal": base_goal, "objects": {
            k: engine.sha256_json(_public_view(v)) for k, v in ast["objects"].items()
        },
        "seed": seed,
    })[:20]
    return base_goal, instance


def list_families() -> list[str]:
    return [f.family_id for f in FAMILIES]
