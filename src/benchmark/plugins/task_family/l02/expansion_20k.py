"""L02 20K expansion: eight new CEL mechanisms from the pinned VAP/CEL shard.

The legacy 12 L02 families use six mechanisms (static/param replica limit,
environment label, demo name/namespace, replica envelope, TTL expiry). This
module adds eight *different* mechanisms already present verbatim in the pinned
shard but never wired into the legacy lane:

* seven CRD-reference validation expressions from
  ``content/en/docs/reference/using-api/cel.md``
  (``doc-health-prefix``, ``doc-name-prefix``, ``doc-xor-lists``,
  ``doc-set-disjoint``, ``doc-primary-cluster``, ``doc-state-counts``,
  ``doc-widgets-exists``);
* the exempt-label validation fragment of
  ``content/en/examples/access/image-matches-namespace-environment.policy.yaml``
  (``vap-image-env-exempt-fragment``).

Every expression is parsed and compile-checked by ``engine.load_policies()`` at
load time under the pinned ``cel-expression-subset/1`` profile. Task objects are
new synthetic CRD-shaped scenarios; the rules are the external knowledge.

The legacy builder is reused for finalisation/ids, and the class below only
overrides ``propose`` so new mechanisms get new object bases. Expansion quota:
100 new instances (6 families x 12 + 2 families x 14).
"""
from __future__ import annotations

import copy
import os
import random
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import CandidateTaskAST, SourceSnapshot
from benchmark.plugins.task_family.l02 import engine
from benchmark.plugins.task_family.l02 import families as fam
from benchmark.plugins.task_family.l02.families import FamilySpec

LANE_ID = "L02"
SEMANTIC_PROFILE = fam.SEMANTIC_PROFILE
GENERATOR_VERSION = "l02-families-20k/1.0.0"
HANDLE = fam.HANDLES[0]


def _public(obj: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in obj.items() if not k.startswith("__")}


# ---------------------------------------------------------------------------
# Object bases (compliant by construction) for the new mechanisms
# ---------------------------------------------------------------------------

def _health_record(name: str, rng: random.Random) -> dict[str, Any]:
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "HealthRecord",
        "metadata": {"name": name, "namespace": "default"},
        "health": "ok-" + rng.choice(["ready", "healthy", "live"]),
    }


def _named_resource(name: str, rng: random.Random) -> dict[str, Any]:
    nm = "srv-%s-%d" % (engine.NAME_POOL[rng.randrange(len(engine.NAME_POOL))],
                        rng.randint(100, 999))
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "NamedResource",
        "metadata": {"name": nm, "namespace": "default"},
        "prefix": "srv-",
    }


def _xor_bucket(name: str, rng: random.Random) -> dict[str, Any]:
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "XorBucket",
        "metadata": {"name": name, "namespace": "default"},
        "list1": [],
        "list2": [rng.randint(1, 9) for _ in range(rng.randint(1, 4))],
    }


def _set_pair(name: str, rng: random.Random) -> dict[str, Any]:
    set1 = [rng.randint(1, 20) for _ in range(rng.randint(1, 4))]
    pool = [x for x in range(21, 40) if x not in set1]
    rng.shuffle(pool)
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "SetPair",
        "metadata": {"name": name, "namespace": "default"},
        "set1": set1,
        "set2": pool[:rng.randint(1, 4)],
    }


def _cluster_set(name: str, rng: random.Random) -> dict[str, Any]:
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "ClusterSet",
        "metadata": {"name": name, "namespace": "default"},
        "clusters": [{"name": "c1"}, {"name": "c2"}, {"name": "c3"}],
        "primary": "c2",
    }


def _state_count(name: str, rng: random.Random) -> dict[str, Any]:
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "StateCount",
        "metadata": {"name": name, "namespace": "default"},
        "stateCounts": {"Available": rng.randint(1, 9),
                        "Progressing": rng.randint(0, 4)},
    }


def _widget_list(name: str, rng: random.Random) -> dict[str, Any]:
    return {
        "apiVersion": "stable.example.com/v1",
        "kind": "WidgetList",
        "metadata": {"name": name, "namespace": "default"},
        "widgets": [
            {"key": "x", "foo": rng.randint(0, 9)},
            {"key": "z", "foo": rng.randint(0, 30)},
        ],
    }


def _exempt_deployment(name: str, rng: random.Random) -> dict[str, Any]:
    return engine.deployment(name, "default", rng.randint(1, 5),
                             {"app": name, "exempt": "true"})


BASE_BUILDERS = {
    "health-prefix": _health_record,
    "name-prefix": _named_resource,
    "xor-lists": _xor_bucket,
    "set-disjoint": _set_pair,
    "primary-cluster": _cluster_set,
    "state-counts": _state_count,
    "widgets-exists": _widget_list,
    "exempt-label": _exempt_deployment,
}


def _base_object(mech: str, rng: random.Random, idx: int) -> dict[str, Any]:
    name = "%s-%d" % (engine.NAME_POOL[(idx + rng.randint(0, 3)) % len(engine.NAME_POOL)],
                      rng.randint(100, 999))
    return BASE_BUILDERS[mech](name, rng)


# ---------------------------------------------------------------------------
# Mutators: (violate, repair) proposals over governed fields.
# ---------------------------------------------------------------------------

def violate_health_prefix(obj, params, rng):
    new = copy.deepcopy(obj)
    new["health"] = "degraded"
    return new


def repair_health_prefix(obj, params, rng):
    new = copy.deepcopy(obj)
    new["health"] = "ok-" + rng.choice(["ready", "healthy"])
    return new


def violate_name_prefix(obj, params, rng):
    new = copy.deepcopy(obj)
    new["prefix"] = "zzz-"
    return new


def repair_name_prefix(obj, params, rng):
    new = copy.deepcopy(obj)
    name = new["metadata"]["name"]
    new["prefix"] = name.split("-", 1)[0] + "-" if "-" in name else name[:-1]
    return new


def violate_xor_lists(obj, params, rng):
    new = copy.deepcopy(obj)
    new["list1"] = [rng.randint(1, 9)]
    if not new.get("list2"):
        new["list2"] = [rng.randint(1, 9)]
    return new


def repair_xor_lists(obj, params, rng):
    new = copy.deepcopy(obj)
    new["list1"] = []
    return new


def violate_set_disjoint(obj, params, rng):
    new = copy.deepcopy(obj)
    new["set2"] = list(new.get("set2") or []) + [new["set1"][0]]
    return new


def repair_set_disjoint(obj, params, rng):
    new = copy.deepcopy(obj)
    set1 = set(new.get("set1") or [])
    new["set2"] = [x for x in (new.get("set2") or []) if x not in set1]
    if not new["set2"]:
        new["set2"] = [max(set1) + 1 if set1 else 1]
    return new


def violate_primary_cluster(obj, params, rng):
    new = copy.deepcopy(obj)
    new["primary"] = "missing"
    return new


def repair_primary_cluster(obj, params, rng):
    new = copy.deepcopy(obj)
    new["primary"] = new["clusters"][0]["name"]
    return new


def violate_state_counts(obj, params, rng):
    new = copy.deepcopy(obj)
    new["stateCounts"].pop("Available", None)
    return new


def repair_state_counts(obj, params, rng):
    new = copy.deepcopy(obj)
    new["stateCounts"]["Available"] = rng.randint(1, 9)
    return new


def violate_widgets_exist(obj, params, rng):
    new = copy.deepcopy(obj)
    for widget in new.get("widgets") or []:
        if widget.get("key") == "x":
            widget["foo"] = 10 + rng.randint(0, 5)
    return new


def repair_widgets_exist(obj, params, rng):
    new = copy.deepcopy(obj)
    if not any(w.get("key") == "x" for w in new.get("widgets") or []):
        new.setdefault("widgets", []).append({"key": "x", "foo": 0})
    for widget in new["widgets"]:
        if widget.get("key") == "x":
            widget["foo"] = rng.randint(0, 9)
    return new


def violate_exempt_label(obj, params, rng):
    new = copy.deepcopy(obj)
    new["metadata"].setdefault("labels", {})["exempt"] = "false"
    return new


def repair_exempt_label(obj, params, rng):
    new = copy.deepcopy(obj)
    new["metadata"].setdefault("labels", {})["exempt"] = "true"
    return new


#: mechanism key -> (policy_id, violate, repair)
NEW_MECHANISMS = {
    "health-prefix": ("doc-health-prefix", violate_health_prefix, repair_health_prefix),
    "name-prefix": ("doc-name-prefix", violate_name_prefix, repair_name_prefix),
    "xor-lists": ("doc-xor-lists", violate_xor_lists, repair_xor_lists),
    "set-disjoint": ("doc-set-disjoint", violate_set_disjoint, repair_set_disjoint),
    "primary-cluster": ("doc-primary-cluster", violate_primary_cluster, repair_primary_cluster),
    "state-counts": ("doc-state-counts", violate_state_counts, repair_state_counts),
    "widgets-exists": ("doc-widgets-exists", violate_widgets_exist, repair_widgets_exist),
    "exempt-label": ("vap-image-env-exempt-fragment", violate_exempt_label, repair_exempt_label),
}

#: Eight new families; low/medium/high instance counts sum to 100.
NEW_FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l02-f13-health-prefix-repair", fam.REGIME_REPAIR, "repair_compliance",
        ("health-prefix",),
        "Make each health record carry a health string starting with the required prefix.",
        3, 5, 4),
    FamilySpec(
        "l02-f14-name-prefix-repair", fam.REGIME_REPAIR, "repair_compliance",
        ("name-prefix",),
        "Align each resource's declared prefix with its actual metadata.name.",
        3, 5, 4),
    FamilySpec(
        "l02-f15-xor-lists-consistency", fam.REGIME_REPAIR, "repair_compliance",
        ("xor-lists",),
        "Restore the exclusive-or invariant so exactly one of the two lists is empty.",
        3, 5, 4),
    FamilySpec(
        "l02-f16-set-disjoint-repair", fam.REGIME_REPAIR, "repair_compliance",
        ("set-disjoint",),
        "Make the two element sets disjoint again.",
        3, 5, 4),
    FamilySpec(
        "l02-f17-primary-cluster-uniqueness", fam.REGIME_REPAIR, "repair_compliance",
        ("primary-cluster",),
        "Point each object at exactly one existing cluster named by its primary field.",
        3, 5, 4),
    FamilySpec(
        "l02-f18-state-counts-repair", fam.REGIME_REPAIR, "repair_compliance",
        ("state-counts",),
        "Restore the required Available entry in each state-count record.",
        3, 5, 4),
    FamilySpec(
        "l02-f19-widgets-existence", fam.REGIME_REPAIR, "repair_compliance",
        ("widgets-exists",),
        "Ensure each widget list contains a key=x widget with foo below the threshold.",
        4, 6, 4),
    FamilySpec(
        "l02-f20-exempt-label-gate", fam.REGIME_REPAIR, "repair_compliance",
        ("exempt-label",),
        "Set the exempt label the policy requires so the workload is admitted.",
        4, 6, 4),
)

NEW_FAMILY_IDS: tuple[str, ...] = tuple(f.family_id for f in NEW_FAMILIES)


def register() -> None:
    """Idempotently install the new mechanisms/families into the lane registries."""
    for key, spec in NEW_MECHANISMS.items():
        engine.MECHANISMS.setdefault(key, spec)
    for spec in NEW_FAMILIES:
        if spec.family_id not in fam.FAMILY_BY_ID:
            fam.FAMILIES = fam.FAMILIES + (spec,)
            fam.FAMILY_BY_ID[spec.family_id] = spec
    for dirpath, _dirnames, _filenames in os.walk(engine.PINNED_ROOT):
        pass
    engine.load_policies()


def expected_total() -> int:
    return sum(f.total for f in NEW_FAMILIES)


def list_families() -> list[str]:
    return list(NEW_FAMILY_IDS)


def _build_candidate_20k(
    spec: FamilySpec, complexity: str, seed: int, variant: int, rng: random.Random,
) -> CandidateTaskAST | None:
    mech = spec.mechanisms[0]
    policy_id, violate, repair = engine.MECHANISMS[mech]
    policy_configs = [fam._policy_config(policy_id, {}, HANDLE)]
    source_refs = {engine.load_policies()[policy_id].source_unit_ref}
    n_objects = fam._objects_per_complexity(complexity)
    objects: dict[str, Any] = {}
    reference: dict[str, Any] = {}
    for idx in range(n_objects):
        base = _base_object(mech, rng, idx + variant)
        violated = violate(base, {}, rng)
        if violated is None:
            return None
        repaired = repair(violated, {}, rng)
        if repaired is None:
            return None
        oid = "%s-%d" % (str(base.get("kind", "obj")).lower(), idx)
        objects[oid] = violated
        reference[oid] = repaired
    ast = {
        "goal_kind": spec.goal_kind,
        "policy_configs": policy_configs,
        "objects": objects,
        "reference_terminal": reference,
        "guards": fam._guards_for(spec, objects),
        "complexity": complexity,
        "fingerprint": "",
        "evidence_budget": {
            "probes": 4 + 2 * len(policy_configs) * n_objects,
            "actions": 4 + 3 * n_objects,
        },
    }
    return fam._finalize(spec, complexity, ast, seed, variant, source_refs)


class KubernetesCelTaskFamilies20K(fam.KubernetesCelTaskFamilies):
    """Legacy family plugin with the expansion candidate builder wired in."""

    plugin_id = "l02.kubernetes-cel-families-20k"

    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int,
        constraints: dict[str, Any],
    ) -> Iterator[CandidateTaskAST]:
        del snapshot
        spec = fam.FAMILY_BY_ID[family]
        quota_scale = float(constraints.get("quota_scale", 1.0))
        instances_per_goal = int(constraints.get("instances_per_goal", 5))
        for complexity, n_instances in fam._family_targets(spec):
            n_instances = max(1, round(n_instances * quota_scale))
            produced = 0
            attempt = 0
            while produced < n_instances and attempt < n_instances * 8:
                attempt += 1
                cand_seed = seed * 1_000_003 + attempt * 977 + produced
                candidate = _build_candidate_20k(
                    spec, complexity, cand_seed, produced % instances_per_goal,
                    random.Random(cand_seed))
                if candidate is not None:
                    produced += 1
                    yield candidate