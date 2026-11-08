"""L09x task families: 8 new structural families for the 20K expansion (E09).

All families are **new mechanisms beyond the legacy L09 lane** (f01..f06):
new artifact kinds under edit (Questionnaire, Measure, ConceptMap, Library
JSON, CQL valueset *declarations*) and new slot semantics on a known kind
(ValueSet display alignment). None is a rename or parameter-only variant of a
legacy family; the distinctness argument per family is recorded in the lane
REPORT. The semantic profile stays ``smart-anc-fhir-structural/1`` (no CQL
evaluation; Java is absent on this host).

Solver-grounded reverse generation, mirroring the legacy lane: pristine pinned
artifact = reference terminal; 1-3 semantic slots corrupted (complexity
low/medium/high) with distractors constructed to provably violate the family's
native check; guards pin identity fields + all non-corrupted slots; the oracle
holds slots + guards + reference sha256. ``verify_terminal`` = family check
passes ∧ every corrupted slot restored exactly ∧ all guards hold.

Base-goal plan (120 base goals x 5 instances = 600 new tasks; complexity
120 low / 240 medium / 240 high):

=========== == ====== =====================================================
family      bg regime  mechanism
=========== == ====== =====================================================
f07         12 repair  questionnaire answerValueSet resolution
f08         12 repair  measure population/stratifier criteria -> ELM
f09         18 repair  ConceptMap custom-source membership + display
f10         18 repair  Library codeFilter.valueSet resolution
f11         18 repair  ValueSet concept display alignment
f12          9 repair  Library anc relatedArtifact resolution
f13         15 organ.  Library relatedArtifacts cover CQL includes
f14         18 repair  CQL valueset declaration resolution
=========== == ====== =====================================================
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
from benchmark.plugins.task_family.l09x import engine
from benchmark.plugins.task_family.l09 import engine as legacy

LANE_ID = "L09"
SEMANTIC_PROFILE = engine.PROFILE
GENERATOR_VERSION = "l09x-families/1.0.0"

REGIME_REPAIR = "repair"
REGIME_ORGANIZATION = "organization"

FMT_JSON = "fhir-json"
FMT_CQL = "cql-text"

INSTANCES_PER_GOAL = 5
MARGIN_PER_GOAL = 2

ANC_BASE = "http://smart.who.int/anc"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    regime: str
    mechanism: str
    intent: str
    low_bg: int
    medium_bg: int
    high_bg: int

    @property
    def base_goals(self) -> int:
        return self.low_bg + self.medium_bg + self.high_bg

    @property
    def total(self) -> int:
        return self.base_goals * INSTANCES_PER_GOAL


FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "l09-f07-questionnaire-answer-valueset-repair", REGIME_REPAIR,
        "check_questionnaire_valueset_bindings",
        "Repair the client-facing Questionnaire so every item answer value set "
        "resolves to a value set published in the pinned guideline snapshot.",
        2, 5, 5,
    ),
    FamilySpec(
        "l09-f08-measure-criteria-identifier-repair", REGIME_REPAIR,
        "check_measure_criteria",
        "Repair the coverage Measure so every population and stratifier "
        "criteria identifier resolves to an expression defined in the linked "
        "indicator library.",
        2, 5, 5,
    ),
    FamilySpec(
        "l09-f09-conceptmap-source-membership-repair", REGIME_REPAIR,
        "check_conceptmap_source_membership",
        "Repair the terminology ConceptMap so every mapped source concept "
        "exists in the guideline custom code system with the recorded display.",
        4, 7, 7,
    ),
    FamilySpec(
        "l09-f10-library-datarequirement-valueset-repair", REGIME_REPAIR,
        "check_library_datarequirements",
        "Repair the decision-support Library so every data requirement code "
        "filter points at a value set published in the pinned snapshot.",
        4, 7, 7,
    ),
    FamilySpec(
        "l09-f11-valueset-display-consistency-repair", REGIME_REPAIR,
        "check_valueset_displays",
        "Repair the answer-list ValueSet so every concept display matches the "
        "recorded display of that code in the guideline code system.",
        4, 7, 7,
    ),
    FamilySpec(
        "l09-f12-library-related-artifact-wiring-repair", REGIME_REPAIR,
        "check_library_related_artifacts",
        "Repair the decision-support Library so every guideline-scoped "
        "related-artifact dependency resolves to a pinned artifact.",
        1, 4, 4,
    ),
    FamilySpec(
        "l09-f13-library-include-coverage-organization", REGIME_ORGANIZATION,
        "check_library_include_coverage",
        "Organize the decision-support Library's dependency list so every "
        "library included by its companion CQL source is covered by a "
        "related-artifact dependency.",
        3, 6, 6,
    ),
    FamilySpec(
        "l09-f14-cql-valueset-declaration-repair", REGIME_REPAIR,
        "check_cql_valueset_declarations",
        "Repair the concepts CQL library so every valueset declaration points "
        "at a value set published in the pinned guideline snapshot.",
        4, 7, 7,
    ),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}

IDENTITY_PATHS = ("resourceType", "url", "name", "id", "version", "status", "title")


# ---------------------------------------------------------------------------
# slot extraction (single source of truth for eligibility + grounding)
# ---------------------------------------------------------------------------

def slots_f07(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": p + ".answerValueSet", "value": it.get("answerValueSet"),
         "kind": "answer_valueset"}
        for p, it in engine.iter_questionnaire_items(doc)
        if it.get("answerValueSet")
    ]


def slots_f08(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": s["path"], "value": s["value"], "kind": "criteria_identifier"}
        for s in engine._measure_criteria_slots(doc)
    ]


def slots_f09(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": s["path"], "value": s["value"], "kind": "source_concept_code",
         "display_path": s["display_path"], "display": s["display"]}
        for s in engine._conceptmap_element_slots(doc)
    ]


def slots_f10(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": s["path"], "value": s["value"], "kind": "codefilter_valueset"}
        for s in engine._codefilter_slots(doc)
    ]


def slots_f11(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": s["path"], "value": s["value"], "kind": "concept_display",
         "code_path": s["code_path"], "code": s["code"]}
        for s in engine._display_slots(doc)
    ]


def slots_f12(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": s["path"], "value": s["value"], "kind": "related_artifact"}
        for s in engine._related_artifact_slots(doc)
    ]


def slots_f13(doc: dict[str, Any]) -> list[dict[str, Any]]:
    return slots_f12(doc)


def slots_f14(text: str) -> list[dict[str, Any]]:
    return [
        {"find": d["line"], "value": d["url"], "name": d["name"],
         "kind": "valueset_declaration"}
        for d in engine.parse_valueset_declarations(text)
    ]


# ---------------------------------------------------------------------------
# eligibility (compose-stage filtering; the verifier re-checks everything)
# ---------------------------------------------------------------------------

def _load_json_units(prefix: str) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for _canon, rel in sorted(engine.load_kbx().canonical_to_path.items()):
        if rel.startswith(prefix):
            out.append((rel, engine.load_artifact(rel)))
    return out


def _load_cql_units() -> list[tuple[str, str]]:
    root = engine.ANC_ROOT / "input" / "cql"
    return [
        ("input/cql/%s" % p.name, p.read_text(encoding="utf-8"))
        for p in sorted(root.glob("*.cql"))
    ]


def eligible_f07() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, doc in _load_json_units("input/resources/questionnaire/"):
        slots = slots_f07(doc)
        if len(slots) < 3 or engine.check_questionnaire_valueset_bindings(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f08() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, doc in _load_json_units("input/resources/measure/"):
        slots = slots_f08(doc)
        if len(slots) < 3 or engine.check_measure_criteria(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f09() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, doc in _load_json_units("input/vocabulary/conceptmap/"):
        slots = slots_f09(doc)
        if len(slots) < 3 or engine.check_conceptmap_source_membership(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f10() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, doc in _load_json_units("input/resources/library/"):
        slots = slots_f10(doc)
        if len(slots) < 3 or engine.check_library_datarequirements(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f11() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
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
        slots = slots_f11(doc)
        if len(slots) < 3 or engine.check_valueset_displays(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f12() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, doc in _load_json_units("input/resources/library/"):
        slots = slots_f12(doc)
        if len(slots) < 3 or engine.check_library_related_artifacts(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f13() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, doc in _load_json_units("input/resources/library/"):
        name = doc.get("name")
        if not name or not kb.cql_includes.get(name):
            continue
        slots = slots_f13(doc)
        bases = [s["value"].split("|")[0] for s in slots]
        if len(slots) < 3 or len(set(bases)) != len(bases):
            continue
        if engine.check_library_include_coverage(doc, kb):
            continue
        out.append({"path": rel, "format": FMT_JSON, "doc": doc, "slots": slots})
    return out


def eligible_f14() -> list[dict[str, Any]]:
    out = []
    kb = engine.load_kbx()
    for rel, text in _load_cql_units():
        slots = slots_f14(text)
        if len(slots) < 3 or engine.check_cql_valueset_declarations(text, kb):
            continue
        out.append({"path": rel, "format": FMT_CQL, "text": text, "slots": slots})
    return out


# ---------------------------------------------------------------------------
# distractor generation (wrong values that provably violate the native check)
# ---------------------------------------------------------------------------

def _suffix(rng: random.Random) -> str:
    return hashlib.sha256(str(rng.random()).encode()).hexdigest()[:6]


def _non_valueset_canonicals(kb: engine.AncKbX, exclude: set[str], rng: random.Random) -> list[str]:
    cands = [
        canon for canon, kind in sorted(kb.canonical_kinds.items())
        if kind != "ValueSet" and canon not in exclude
    ]
    rng.shuffle(cands)
    return cands[:2]


def _distractors_f07(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
    in_q = {s["value"] for s in entry["slots"]}
    out = ["%s/ValueSet/anc-zz-%s" % (ANC_BASE, _suffix(rng))]
    out += _non_valueset_canonicals(kb, in_q, rng)
    return out


def _distractors_f08(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
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
    return (foreign + ["Unresolvable Criteria-%s" % _suffix(rng)])[:5]


def _distractors_f09(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
    in_map = {s["value"] for s in entry["slots"]}
    element_display = slot.get("display")
    pool = sorted(
        code for code in kb.codesystem_codes - in_map
        if kb.codesystem_display.get(code) != element_display
    )
    rng.shuffle(pool)
    return (pool + ["ANC.ZZZ.%s" % _suffix(rng).upper()])[:4]


def _distractors_f10(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
    in_lib = {s["value"] for s in entry["slots"]}
    out = ["%s/ValueSet/anc-zz-%s" % (ANC_BASE, _suffix(rng))]
    out += _non_valueset_canonicals(kb, in_lib, rng)
    return out


def _distractors_f11(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
    correct = slot["value"]
    in_vs = {s["value"] for s in entry["slots"]}
    used_codes = {s["code"] for s in entry["slots"]}
    pool = sorted({
        d for code, d in kb.codesystem_display.items()
        if d and d != correct and d not in in_vs and code not in used_codes
    })
    rng.shuffle(pool)
    return (pool + ["Unrecorded display %s" % _suffix(rng)])[:4]


def _distractors_f12(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    val = slot["value"] or ""
    base, _, _ = val.rpartition("/")
    return ["%s/anc-zz-%s" % (base, _suffix(rng))]


def _distractors_f13(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
    name = entry["doc"].get("name")
    includes = kb.cql_includes.get(name, frozenset())
    covered = {s["value"].split("|")[0] for s in entry["slots"]}
    target_canons = {
        kb.include_canonical_by_name.get(i) for i in includes
    } - {None}
    cands = [
        canon for canon in sorted(kb.canonical_to_path)
        if canon not in target_canons and canon not in covered
    ]
    rng.shuffle(cands)
    return (cands + ["%s/ValueSet/anc-zz-%s" % (ANC_BASE, _suffix(rng))])[:4]


def _distractors_f14(entry: dict[str, Any], slot: dict[str, Any], rng: random.Random) -> list[str]:
    kb = engine.load_kbx()
    line = slot["find"]
    out = [line.replace("'%s'" % slot["value"], "'%s/ValueSet/anc-zz-%s'" % (ANC_BASE, _suffix(rng)), 1)]
    for canon in _non_valueset_canonicals(kb, {slot["value"]}, rng):
        out.append(line.replace("'%s'" % slot["value"], "'%s'" % canon, 1))
    return out


DISTRACTORS: dict[str, Callable[..., list[str]]] = {
    "l09-f07-questionnaire-answer-valueset-repair": _distractors_f07,
    "l09-f08-measure-criteria-identifier-repair": _distractors_f08,
    "l09-f09-conceptmap-source-membership-repair": _distractors_f09,
    "l09-f10-library-datarequirement-valueset-repair": _distractors_f10,
    "l09-f11-valueset-display-consistency-repair": _distractors_f11,
    "l09-f12-library-related-artifact-wiring-repair": _distractors_f12,
    "l09-f13-library-include-coverage-organization": _distractors_f13,
    "l09-f14-cql-valueset-declaration-repair": _distractors_f14,
}

ELIGIBLE: dict[str, Callable[[], list[dict[str, Any]]]] = {
    "l09-f07-questionnaire-answer-valueset-repair": eligible_f07,
    "l09-f08-measure-criteria-identifier-repair": eligible_f08,
    "l09-f09-conceptmap-source-membership-repair": eligible_f09,
    "l09-f10-library-datarequirement-valueset-repair": eligible_f10,
    "l09-f11-valueset-display-consistency-repair": eligible_f11,
    "l09-f12-library-related-artifact-wiring-repair": eligible_f12,
    "l09-f13-library-include-coverage-organization": eligible_f13,
    "l09-f14-cql-valueset-declaration-repair": eligible_f14,
}


def slot_key(slot: dict[str, Any]) -> str:
    return slot.get("path") or slot.get("find") or ""


# ---------------------------------------------------------------------------
# plugin
# ---------------------------------------------------------------------------

class SmartAncExpansionTaskFamilies:
    """TaskFamilyPlugin for the L09 20K-expansion families (l09x)."""

    def supported_profiles(self) -> list[str]:
        return [SEMANTIC_PROFILE]

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
        used_subsets: set[tuple[str, tuple[str, ...]]] = set()
        bg_cursor = 0
        for n_slots, n_goals in ((1, spec.low_bg), (2, spec.medium_bg), (3, spec.high_bg)):
            for _ in range(n_goals):
                entry, chosen = self._pick_distinct_subset(
                    pool, n_slots, bg_cursor, used_subsets,
                    "%s:%s" % (seed, family))
                bg_cursor += 1
                if entry is None:
                    continue
                used_subsets.add((entry["path"], tuple(slot_key(s) for s in chosen)))
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

    @staticmethod
    def _pick_distinct_subset(
        pool: list[dict[str, Any]],
        n_slots: int,
        bg_cursor: int,
        used_subsets: set[tuple[str, tuple[str, ...]]],
        seed_prefix: str,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Round-robin entry with duplicate-subset retry: small-slot artifacts
        exhaust their distinct subsets in a tier, so advance to the next pool
        entry instead of emitting a duplicate base goal."""
        for attempt in range(len(pool) * 2):
            entry = pool[(bg_cursor + attempt) % len(pool)]
            slots_all = entry["slots"]
            picker = random.Random("%s:%d:%d" % (seed_prefix, bg_cursor, attempt))
            slot_idx = sorted(
                picker.sample(range(len(slots_all)), min(n_slots, len(slots_all))))
            chosen = [slots_all[i] for i in slot_idx]
            key = (entry["path"], tuple(slot_key(s) for s in chosen))
            if key not in used_subsets:
                return entry, chosen
        return None, []

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
            all_slots = slots_f14(reference_text)
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
                "check_questionnaire_valueset_bindings": slots_f07,
                "check_measure_criteria": slots_f08,
                "check_conceptmap_source_membership": slots_f09,
                "check_library_datarequirements": slots_f10,
                "check_valueset_displays": slots_f11,
                "check_library_related_artifacts": slots_f12,
                "check_library_include_coverage": slots_f13,
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
        oracle_ref = "l09x-oracle-%s" % hashlib.sha256(
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
    """Agent-visible KB pointers; the artifact's own pristine path excluded."""
    refs: set[str] = set()
    mech = ast["mechanism"]
    if mech == "check_questionnaire_valueset_bindings":
        refs.add("smart-anc/input/vocabulary/valueset/")
    elif mech == "check_measure_criteria":
        refs.add("smart-anc/input/resources/library/")
    elif mech == "check_conceptmap_source_membership":
        refs.add("smart-anc/input/vocabulary/codesystem/")
    elif mech == "check_library_datarequirements":
        refs.add("smart-anc/input/vocabulary/valueset/")
    elif mech == "check_valueset_displays":
        refs.add("smart-anc/input/vocabulary/codesystem/")
    elif mech == "check_library_related_artifacts" or mech == "check_library_include_coverage":
        refs.add("smart-anc/input/resources/")
        refs.add("smart-anc/input/vocabulary/")
        refs.add("smart-anc/input/profiles/")
        if mech == "check_library_include_coverage":
            refs.add("smart-anc/input/cql/")
    elif mech == "check_library_cql_valueset_coverage":
        refs.add("smart-anc/input/cql/")
    elif mech == "check_cql_valueset_declarations":
        refs.add("smart-anc/input/vocabulary/valueset/")
    return tuple(sorted(refs))
