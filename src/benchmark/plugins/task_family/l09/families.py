"""L09 task families: 6 structural families over the pinned WHO SMART-ANC snapshot.

Solver-grounded reverse generation (build order §4.1), mirroring the L01 lane
pattern: a pristine upstream artifact (PlanDefinition / ValueSet / CQL library)
is the reference terminal state; a mechanism mutator corrupts 1-3 semantic slots
to produce the initial state; the pinned deterministic checks in ``engine.py``
decide acceptance — mutators only propose edits.

Native scope honesty: no task requires CQL *evaluation* (Java absent). Every
check is a structural/terminological recomputation over the pinned artifacts
(profile ``smart-anc-fhir-structural/1``). Families cover the lane's mandated
real problems: structured guideline branches (f01, f06), referral / plan
constraints (f05), terminology membership and grouper composition (f03, f04),
and decision-support dependency wiring (f02).

Complexity tiers are structural: low = 1 corrupted slot, medium = 2, high = 3.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST,
    GroundedTask,
    SourceSnapshot,
)
from benchmark.plugins.task_family.l09 import engine

LANE_ID = "L09"
SEMANTIC_PROFILE = engine.PROFILE
GENERATOR_VERSION = "l09-families/1.0.0"

REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"

FMT_JSON = "fhir-json"
FMT_CQL = "cql-text"

INSTANCES_PER_GOAL = 5
MARGIN_PER_GOAL = 2  # extra candidates per base goal; admission capped at 5


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    mechanism: str  # native check name
    intent: str
    # (low_bg, medium_bg, high_bg) base-goal counts; 5 instances per base goal
    low_bg: int
    medium_bg: int
    high_bg: int

    @property
    def base_goals(self) -> int:
        return self.low_bg + self.medium_bg + self.high_bg

    @property
    def total(self) -> int:
        return self.base_goals * INSTANCES_PER_GOAL


#: 6 structural families. Distinctness = artifact kind x slot semantics x native
#: check x graph/terminology interaction; not artifact renames. Base-goal tiers
#: give the lane exactly 40 base goals / 200 instances at complexity 40/80/80.
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l09-f01-decision-condition-identifier-repair", REGIME_REPAIR,
        "check_condition_identifiers",
        "Restore the decision-table branch conditions of the guideline PlanDefinition so every applicability condition identifier resolves to an expression defined in the linked decision-support library.",
        1, 3, 3,
    ),
    FamilySpec(
        "l09-f02-cql-include-dependency-repair", REGIME_REPAIR,
        "check_cql_includes",
        "Repair the CQL library's include directives so every referenced dependency library resolves against the pinned guideline library set with the declared version.",
        1, 3, 3,
    ),
    FamilySpec(
        "l09-f03-valueset-concept-membership-repair", REGIME_REPAIR,
        "check_valueset_expansion",
        "Repair the answer-list ValueSet so the expansion recomputed from its compose rule matches the published expansion and every listed code exists in the guideline code system.",
        2, 3, 3,
    ),
    FamilySpec(
        "l09-f04-valueset-grouper-membership-organization", REGIME_ORGANIZATION,
        "check_valueset_expansion",
        "Organize the grouper ValueSet's member list so its composed expansion exactly matches the published expansion of the guideline value set.",
        1, 2, 3,
    ),
    FamilySpec(
        "l09-f05-care-plan-chain-repair", REGIME_REPAIR,
        "check_plan_references",
        "Repair the ANC care plan so every action reference (activity / sub-plan) resolves and the visit sequence links form one acyclic referral-to-scheduling chain as published.",
        2, 2, 2,
    ),
    FamilySpec(
        "l09-f06-cql-branch-valueset-reference-repair", REGIME_REPAIR,
        "check_cql_valueset_refs",
        "Repair the decision-table CQL branch expressions so every valueset reference resolves to a declaration in the included concepts library.",
        1, 3, 2,
    ),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

IDENTITY_PATHS = ("resourceType", "url", "name", "id", "version", "status", "title")


# ---------------------------------------------------------------------------
# slot extraction (shared by eligibility and grounding; single source of truth)
# ---------------------------------------------------------------------------

def slots_f01(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"path": s["path"], "value": s["value"], "kind": "condition_identifier"}
            for s in engine.condition_slots(doc)]


def slots_f02(text: str) -> list[dict[str, Any]]:
    out = []
    for inc in engine.parse_includes(text):
        line = inc["line"]
        out.append({
            "find": line,
            "value": line,
            "library": inc["library"],
            "version": inc["version"],
            "kind": "cql_include",
        })
    return out


def slots_f03(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for i, inc in enumerate((doc.get("compose") or {}).get("include") or []):
        for j, c in enumerate(inc.get("concept") or []):
            if c.get("code"):
                out.append({
                    "path": "compose.include.%d.concept.%d.code" % (i, j),
                    "value": c["code"],
                    "kind": "concept_code",
                })
    return out


def slots_f04(doc: dict[str, Any]) -> list[dict[str, Any]]:
    kb = engine.load_kb()
    out = []
    for i, inc in enumerate((doc.get("compose") or {}).get("include") or []):
        for j, m in enumerate(inc.get("valueSet") or []):
            if kb.valueset_recorded.get(m):
                out.append({
                    "path": "compose.include.%d.valueSet.%d" % (i, j),
                    "value": m,
                    "kind": "member_canonical",
                })
    return out


def slots_f05(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path, act in engine.iter_actions(doc):
        if act.get("definitionCanonical"):
            out.append({
                "path": path + ".definitionCanonical",
                "value": act["definitionCanonical"],
                "kind": "definition_canonical",
            })
        for j, rela in enumerate(act.get("relatedAction") or []):
            out.append({
                "path": "%s.relatedAction.%d.actionId" % (path, j),
                "value": rela.get("actionId"),
                "kind": "related_action",
            })
    return out


def slots_f06(text: str) -> list[dict[str, Any]]:
    kb = engine.load_kb()
    incs = engine.parse_includes(text)
    alias_to_lib = {i["alias"] or i["library"]: i["library"] for i in incs}
    out = []
    for m in engine._ALIASREF_RE.finditer(text):
        alias, refname = m.group(1), m.group(2)
        lib = alias_to_lib.get(alias)
        if lib is None or not lib.startswith("ANC"):
            continue
        if refname not in kb.cql_declarations.get(lib, frozenset()):
            continue
        out.append({
            "find": m.group(0),
            "value": refname,
            "alias": alias,
            "library": lib,
            "kind": "aliased_valueset_ref",
        })
    return out


# ---------------------------------------------------------------------------
# eligibility (compose-stage filtering; the verifier re-checks everything)
# ---------------------------------------------------------------------------

def _load_json_units(prefix: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for _canon, rel in sorted(engine.load_kb().canonical_to_path.items()):
        if rel.startswith(prefix):
            out.append((rel, engine.load_artifact(rel)))
    return out


def _load_cql_units() -> list[tuple[str, str]]:
    root = engine.ANC_ROOT / "input" / "cql"
    return [
        ("input/cql/%s" % p.name, p.read_text(encoding="utf-8"))
        for p in sorted(root.glob("*.cql"))
    ]


def eligible_f01() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kb()
    for rel, doc in _load_json_units("input/resources/plandefinition/"):
        if "plandefinition-ANCDT" not in rel:
            continue
        slots = slots_f01(doc)
        if len(slots) < 3 or engine.check_condition_identifiers(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f02() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kb()
    for rel, text in _load_cql_units():
        if "/ANC" not in rel:
            continue
        slots = slots_f02(text)
        if len(slots) < 3 or engine.check_cql_includes(text, kb):
            continue
        out.append({"path": rel, "format": FMT_CQL, "text": text, "slots": slots})
    return out


def eligible_f03() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kb()
    for rel, doc in _load_json_units("input/vocabulary/valueset/"):
        compose = doc.get("compose") or {}
        incs = compose.get("include") or []
        if compose.get("exclude") or not incs:
            continue
        if not all(
            i.get("system") == engine.CUSTOM_CS and "concept" in i and "filter" not in i
            for i in incs
        ):
            continue
        if not kb.valueset_recorded.get(doc.get("url")):
            continue
        slots = slots_f03(doc)
        if len(slots) < 3 or engine.check_valueset_expansion(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f04() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kb()
    for rel, doc in _load_json_units("input/vocabulary/valueset/"):
        compose = doc.get("compose") or {}
        incs = compose.get("include") or []
        if compose.get("exclude") or not incs:
            continue
        if not all(set(i.keys()) == {"valueSet"} for i in incs):
            continue
        if not kb.valueset_recorded.get(doc.get("url")):
            continue
        slots = slots_f04(doc)
        if len(slots) < 3 or engine.check_valueset_expansion(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f05() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kb()
    for rel, doc in _load_json_units("input/resources/plandefinition/manual/"):
        slots = slots_f05(doc)
        if len(slots) < 3 or engine.check_plan_references(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f06() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kb()
    for rel, text in _load_cql_units():
        if "ANCDT" not in rel:
            continue
        if engine.check_cql_valueset_refs(text, kb):
            continue
        slots = slots_f06(text)
        if len(slots) < 3:
            continue
        out.append({"path": rel, "format": FMT_CQL, "text": text, "slots": slots})
    return out


# ---------------------------------------------------------------------------
# distractor generation (wrong values that provably violate the native check)
# ---------------------------------------------------------------------------

def _suffix(rng: random.Random) -> str:
    return hashlib.sha256(str(rng.random()).encode()).hexdigest()[:6]


def _distractors_f01(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kb()
    own_libs = set(entry["doc"].get("library") or [])
    own_stmts = set().union(
        *(kb.library_elm_statements.get(c, frozenset()) for c in own_libs)
    ) if own_libs else set()
    foreign = sorted({
        s for canon, stmts in kb.library_elm_statements.items()
        if canon not in own_libs
        for s in stmts
        if s != slot["value"] and s not in own_stmts
    })
    rng.shuffle(foreign)
    return (foreign + ["Should Unresolvable-%s" % _suffix(rng)])[:5]


def _distractors_f02(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kb()
    name, declared = slot["library"], slot.get("version")
    pinned = kb.cql_library_versions.get(name)
    line = slot["find"]
    out = [line.replace(name, "%sUnresolvable%s" % (name, _suffix(rng)), 1)]
    if declared and pinned and declared != pinned:
        out.append(line.replace("'%s'" % declared, "'999.0.0'", 1))
    return out


def _distractors_f03(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kb()
    in_vs = {s["value"] for s in entry["slots"]}
    pool = sorted(kb.codesystem_codes - in_vs)
    rng.shuffle(pool)
    return (pool + ["ANC.ZZZ.%s" % _suffix(rng).upper()])[:4]


def _distractors_f04(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kb()
    target = kb.valueset_recorded.get(slot["value"], frozenset())
    members_in_vs = {s["value"] for s in entry["slots"]}
    cands = [
        canon for canon, rel in sorted(kb.canonical_to_path.items())
        if rel.startswith("input/vocabulary/valueset/")
        and canon not in members_in_vs
        and kb.valueset_recorded.get(canon, frozenset()) not in (frozenset(), target)
    ]
    rng.shuffle(cands)
    return (cands + ["http://smart.who.int/anc/ValueSet/anc-zz-%s" % _suffix(rng)])[:4]


def _distractors_f05(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    val = slot["value"] or ""
    if slot["path"].endswith("definitionCanonical"):
        base, _, tail = val.rpartition("/")
        return ["%s/%sZZ%s" % (base, tail, _suffix(rng))]
    return ["zz-no-such-action-%s" % _suffix(rng)]


def _distractors_f06(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    return ['%s."%s Unresolvable-%s"' % (slot["alias"], slot["value"], _suffix(rng))]


DISTRACTORS: dict[str, Callable[..., list[str]]] = {
    "l09-f01-decision-condition-identifier-repair": _distractors_f01,
    "l09-f02-cql-include-dependency-repair": _distractors_f02,
    "l09-f03-valueset-concept-membership-repair": _distractors_f03,
    "l09-f04-valueset-grouper-membership-organization": _distractors_f04,
    "l09-f05-care-plan-chain-repair": _distractors_f05,
    "l09-f06-cql-branch-valueset-reference-repair": _distractors_f06,
}

ELIGIBLE: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "l09-f01-decision-condition-identifier-repair": eligible_f01,
    "l09-f02-cql-include-dependency-repair": eligible_f02,
    "l09-f03-valueset-concept-membership-repair": eligible_f03,
    "l09-f04-valueset-grouper-membership-organization": eligible_f04,
    "l09-f05-care-plan-chain-repair": eligible_f05,
    "l09-f06-cql-branch-valueset-reference-repair": eligible_f06,
}


def slot_key(slot: dict[str, Any]) -> str:
    return slot.get("path") or slot.get("find") or ""


# ---------------------------------------------------------------------------
# plugin
# ---------------------------------------------------------------------------

class SmartAncTaskFamilies:
    """TaskFamilyPlugin for lane L09."""

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

    # -- propose ---------------------------------------------------------------
    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int, constraints: dict[str, Any]
    ) -> Iterator[CandidateTaskAST]:
        spec = FAMILY_BY_ID[family]
        pool = constraints.get("pool")
        if pool is None:
            pool = ELIGIBLE[family]()
        if not pool:
            return
        distract = DISTRACTORS[family]
        bg_cursor = 0
        for n_slots, n_goals in ((1, spec.low_bg), (2, spec.medium_bg), (3, spec.high_bg)):
            for _ in range(n_goals):
                entry = pool[bg_cursor % len(pool)]
                bg_cursor += 1
                slots_all = entry["slots"]
                picker = random.Random("%s:%s:%d" % (seed, family, bg_cursor))
                slot_idx = sorted(picker.sample(range(len(slots_all)), min(n_slots, len(slots_all))))
                chosen = [slots_all[i] for i in slot_idx]
                base_goal_key = json.dumps(
                    [family, entry["path"], [slot_key(s) for s in chosen]], sort_keys=True)
                base_goal_id = "bg-" + hashlib.sha256(base_goal_key.encode()).hexdigest()[:20]
                for inst_i in range(INSTANCES_PER_GOAL + MARGIN_PER_GOAL):
                    irng = random.Random("%s:%s:%d" % (seed, base_goal_id, inst_i))
                    corrupted = []
                    for s in chosen:
                        wrongs = distract(entry, s, irng)
                        corrupted.append({**s, "wrong_value": wrongs[inst_i % len(wrongs)]})
                    instance_key = json.dumps(
                        [base_goal_id, [(slot_key(s), s["wrong_value"]) for s in corrupted]],
                        sort_keys=True)
                    instance_id = "in-" + hashlib.sha256(instance_key.encode()).hexdigest()[:20]
                    yield CandidateTaskAST(
                        lane_id=LANE_ID,
                        family_id=family,
                        base_goal_id=base_goal_id,
                        instance_id=instance_id,
                        regime=spec.regime,
                        semantic_profile=SEMANTIC_PROFILE,
                        ast={
                            "goal_kind": "repair_artifact",
                            "mechanism": spec.mechanism,
                            "artifact_path": entry["path"],
                            "artifact_format": entry["format"],
                            "slots": corrupted,
                            "complexity": {1: "low", 2: "medium", 3: "high"}[len(chosen)],
                            "intent": spec.intent,
                            "generator": GENERATOR_VERSION,
                        },
                        source_unit_ids=("smart-anc/%s" % entry["path"],),
                    )

    # -- ground ----------------------------------------------------------------
    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        ast = dict(candidate.ast)
        path = ast["artifact_path"]
        fmt = ast["artifact_format"]
        corrupted_keys = {slot_key(s) for s in ast["slots"]}
        if fmt == FMT_CQL:
            reference_text = engine.ANC_ROOT.joinpath(path).read_text(encoding="utf-8")
            initial_text = reference_text
            for slot in ast["slots"]:
                if slot["find"] not in initial_text:
                    raise ValueError("corruption anchor missing: %s" % slot["find"])
                initial_text = initial_text.replace(slot["find"], slot["wrong_value"], 1)
            reference: Any = {"format": fmt, "text": reference_text}
            initial: Any = {"format": fmt, "text": initial_text}
            artifact_id = path.rsplit("/", 1)[-1]
            all_slots = (
                slots_f02(reference_text) if ast["mechanism"] == "check_cql_includes"
                else slots_f06(reference_text)
            )
            guards = [
                {"must_contain_count": s["find"], "count": reference_text.count(s["find"])}
                for s in all_slots
                if slot_key(s) not in corrupted_keys
            ]
        else:
            doc = engine.load_artifact(path)
            reference = {"format": fmt, "content": doc}
            initial = {"format": fmt, "content": engine.apply_corruptions(doc, ast["slots"])}
            artifact_id = doc.get("name") or doc.get("id") or path.rsplit("/", 1)[-1]
            slot_fn = {
                "check_condition_identifiers": slots_f01,
                "check_valueset_expansion": (
                    slots_f03 if "f03" in candidate.family_id else slots_f04
                ),
                "check_plan_references": slots_f05,
            }[ast["mechanism"]]
            all_slots = slot_fn(doc)
            guards = [
                {"path": p, "value": engine.get_path(doc, p)}
                for p in IDENTITY_PATHS
                if engine.get_path(doc, p) is not None
            ]
            guards += [
                {"path": s["path"], "value": s["value"]}
                for s in all_slots
                if slot_key(s) not in corrupted_keys
            ]
        oracle_ref = "l09-oracle-%s" % hashlib.sha256(
            json.dumps([candidate.instance_id, ast["slots"]], sort_keys=True,
                       default=str).encode()
        ).hexdigest()[:16]
        n_slots = len(ast["slots"])
        public = {
            "user_goal": (
                "%s Artifact under edit: %s (pinned WHO SMART-ANC snapshot "
                "commit %s). %d semantic slot(s) are broken. Edit the artifact "
                "so the native structural checks pass against the pinned snapshot."
                % (ast["intent"], artifact_id, engine.ANC_COMMIT[:12], n_slots)
            ),
            "artifact_id": artifact_id,
            "artifact_format": fmt,
            "artifact": initial,
            "observation_refs": list(_observation_refs(ast)),
            "allowed_tools": ["artifact_edit", "structure_probe"],
            "budget": {"probes": 6, "actions": 6 + 3 * n_slots},
            "complexity": ast["complexity"],
        }
        return GroundedTask(
            ast=candidate,
            public=public,
            oracle_ref=oracle_ref,
            native_evidence={},
        )


def _observation_refs(ast: dict[str, Any]) -> tuple[str, ...]:
    """Agent-visible pointers into the pinned knowledge base (the units whose
    semantics decide the task). The artifact's own pristine snapshot path is
    deliberately excluded: the corrupted copy is public material, the pristine
    file would be the answer."""
    refs: set[str] = set()
    mech = ast["mechanism"]
    if mech == "check_condition_identifiers":
        refs.add("smart-anc/input/resources/library/")
    elif mech in ("check_cql_includes", "check_cql_valueset_refs"):
        refs.add("smart-anc/input/cql/")
    elif mech == "check_valueset_expansion":
        refs.add("smart-anc/input/vocabulary/")
    elif mech == "check_plan_references":
        refs.add("smart-anc/input/resources/activitydefinition/")
        refs.add("smart-anc/input/resources/plandefinition/")
    return tuple(sorted(refs))
