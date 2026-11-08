"""L09x native engine: 20K-expansion structural checks over the pinned WHO SMART-ANC snapshot.

This module extends the legacy L09 lane (``benchmark/plugins/task_family/l09``)
for the 20K expansion work unit E09. It **never modifies** the legacy engine,
legacy staging (``bank/staging/l09``) or the 8K build; it imports the legacy
KB loader read-only and adds the new knowledge-base views + checks needed by
the eight new task families (f07..f14).

Native scope honesty (unchanged from the legacy lane): the declared semantic
profile stays ``smart-anc-fhir-structural/1`` — every check is a deterministic
pure-Python recomputation over the pinned snapshot. **No CQL evaluation
runtime exists on this host (Java absent); no task requires one and none
claims one.**

New checks (each returns a list of violation dicts; empty list = pass):

- ``check_questionnaire_valueset_bindings``  every Questionnaire item
  ``answerValueSet`` must resolve to a pinned ValueSet resource.
- ``check_measure_criteria``                 Measure ``library`` canonicals
  must resolve to pinned Libraries carrying ELM; every group population /
  stratifier ``text/cql-identifier`` criteria must resolve to an ELM
  ``statements.def[].name`` of a linked Library.
- ``check_conceptmap_source_membership``     for ConceptMap groups whose
  ``source`` is the custom CodeSystem, every element ``code`` must exist in
  the pinned CodeSystem and its ``display`` must equal the recorded display.
- ``check_library_datarequirements``         every Library
  ``dataRequirement.codeFilter.valueSet`` must resolve to a pinned ValueSet.
- ``check_valueset_displays``                for ValueSet compose includes
  over the custom CodeSystem, every concept ``code`` must exist in the
  CodeSystem and its ``display`` must equal the recorded display.
- ``check_library_related_artifacts``        every anc-scoped
  ``relatedArtifact.resource`` canonical of a Library must resolve to a
  pinned artifact.
- ``check_library_include_coverage``         every ``include`` target of a
  Library's companion CQL library (matched by name) must resolve to a known
  library canonical (Library JSON url, or — for CQL-only libraries such as
  FHIRHelpers — the canonical recorded in the snapshot's related-artifact
  dependencies) and be covered by an anc-scoped ``relatedArtifact``
  dependency of the Library JSON.
- ``check_cql_valueset_declarations``        every CQL ``valueset "X": 'url'``
  declaration must resolve to a pinned ValueSet resource.

Knowledge-side counterfactuals (applied to *views*; the pinned snapshot is
never mutated) reuse the legacy perturbation kinds; ``drop_canonical``
additionally removes the canonical from the library-canonical index so
include-coverage verdicts flip.
"""
from __future__ import annotations

import dataclasses
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l09 import engine as legacy

REPO_ROOT = legacy.REPO_ROOT
ANC_ROOT = legacy.ANC_ROOT
ANC_COMMIT = legacy.ANC_COMMIT
CUSTOM_CS = legacy.CUSTOM_CS
ENGINE = "python-structural"
ENGINE_VERSION = "2.0.0"  # l09x extension checks (20K expansion)
PROFILE = legacy.PROFILE  # smart-anc-fhir-structural/1 — unchanged, honest

sha256_json = legacy.sha256_json
get_path = legacy.get_path
set_path = legacy.set_path
load_artifact = legacy.load_artifact

_VSDECL_LINE_RE = re.compile(
    r"valueset\s+\"([^\"]+)\"\s*:\s*'([^']+)'"
)


@dataclass(frozen=True)
class AncKbX(legacy.AncKB):
    """Extended KB view: legacy fields plus the views the new checks need.

    Inheritance keeps every legacy perturbation kind working on the extended
    view.
    """

    canonical_kinds: dict[str, str] = dataclasses.field(default_factory=dict)
    codesystem_display: dict[str, str] = dataclasses.field(default_factory=dict)
    cql_includes: dict[str, frozenset] = dataclasses.field(default_factory=dict)
    include_canonical_by_name: dict[str, str] = dataclasses.field(default_factory=dict)

    def with_perturbation_x(self, p: "KbPerturbationX") -> "AncKbX":
        if p.drop_canonical:
            base = super().with_perturbation(p)
            dropped = p.drop_canonical
            return dataclasses.replace(
                base,
                canonical_kinds={k: v for k, v in self.canonical_kinds.items()
                                 if k != dropped},
                include_canonical_by_name={
                    k: v for k, v in self.include_canonical_by_name.items()
                    if v != dropped
                },
            )
        # other legacy perturbation kinds: dataclasses.replace preserves the
        # AncKbX subclass (and therefore the extended views) through the
        # legacy logic.
        return super().with_perturbation(p)


@dataclass(frozen=True)
class KbPerturbationX(legacy.KbPerturbation):
    pass


_KBX_CACHE: AncKbX | None = None


def load_kbx() -> AncKbX:
    global _KBX_CACHE
    if _KBX_CACHE is not None:
        return _KBX_CACHE
    base = legacy.load_kb()
    canonical_kinds: dict[str, str] = {}
    codesystem_display: dict[str, str] = {}
    library_canonical_by_name: dict[str, str] = {}
    related_artifact_bases: set[str] = set()
    for scope in ("input/resources", "input/vocabulary", "input/profiles"):
        for path in sorted((ANC_ROOT / scope).rglob("*.json")):
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            url = doc.get("url")
            rt = doc.get("resourceType")
            if isinstance(url, str) and rt:
                canonical_kinds[url] = rt
            if rt == "CodeSystem" and url == CUSTOM_CS:
                for c in doc.get("concept") or []:
                    if c.get("code"):
                        codesystem_display[c["code"]] = c.get("display")
            if rt == "Library" and isinstance(url, str) and doc.get("name"):
                library_canonical_by_name[doc["name"]] = url
            if rt == "Library":
                for ra in doc.get("relatedArtifact") or []:
                    res = ra.get("resource")
                    if isinstance(res, str) and res.startswith("http://smart.who.int/anc/"):
                        related_artifact_bases.add(res.split("|")[0])
    cql_includes: dict[str, frozenset] = {}
    include_names: set[str] = set()
    for path in sorted((ANC_ROOT / "input/cql").glob("*.cql")):
        text = path.read_text(encoding="utf-8")
        m = legacy._LIB_RE.search(text)
        if not m:
            continue
        incs = frozenset(i["library"] for i in legacy.parse_includes(text))
        cql_includes[m.group(1)] = incs
        include_names |= set(incs)
    include_canonical_by_name: dict[str, str] = {}
    for name in sorted(include_names):
        if name in library_canonical_by_name:
            include_canonical_by_name[name] = library_canonical_by_name[name]
            continue
        cands = {b for b in related_artifact_bases if b.endswith("/Library/%s" % name)}
        if len(cands) == 1:
            include_canonical_by_name[name] = cands.pop()
    _KBX_CACHE = AncKbX(
        canonical_to_path=base.canonical_to_path,
        cql_library_versions=base.cql_library_versions,
        cql_declarations=base.cql_declarations,
        valueset_recorded=base.valueset_recorded,
        codesystem_codes=base.codesystem_codes,
        library_elm_statements=base.library_elm_statements,
        canonical_kinds=canonical_kinds,
        codesystem_display=codesystem_display,
        cql_includes=cql_includes,
        include_canonical_by_name=include_canonical_by_name,
    )
    return _KBX_CACHE


def is_valueset(kb: AncKbX, canonical: str) -> bool:
    return kb.canonical_kinds.get(canonical.split("|")[0]) == "ValueSet"


# ---------------------------------------------------------------------------
# Questionnaire
# ---------------------------------------------------------------------------

def iter_questionnaire_items(
    doc: dict[str, Any], prefix: str = "item"
) -> list[tuple[str, dict[str, Any]]]:
    """Depth-first (json_path, item) list for every questionnaire item."""
    out: list[tuple[str, dict[str, Any]]] = []
    for i, it in enumerate(doc.get(prefix) or []):
        path = "%s.%d" % (prefix, i)
        out.append((path, it))
        out.extend(iter_questionnaire_items(it, path + ".item"))
    return out


def check_questionnaire_valueset_bindings(
    doc: dict[str, Any], kb: AncKbX
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for path, item in iter_questionnaire_items(doc):
        avs = item.get("answerValueSet")
        if not isinstance(avs, str) or not avs:
            continue
        base = avs.split("|")[0]
        if base not in kb.canonical_to_path:
            violations.append({
                "kind": "unresolved_answer_valueset", "path": path + ".answerValueSet",
                "value": avs,
            })
        elif not is_valueset(kb, avs):
            violations.append({
                "kind": "non_valueset_answer_valueset", "path": path + ".answerValueSet",
                "value": avs, "resolved_kind": kb.canonical_kinds.get(base),
            })
    return violations


# ---------------------------------------------------------------------------
# Measure
# ---------------------------------------------------------------------------

def _measure_criteria_slots(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gi, g in enumerate(doc.get("group") or []):
        for pi, pop in enumerate(g.get("population") or []):
            ex = (pop.get("criteria") or {})
            if ex.get("language") == "text/cql-identifier" and ex.get("expression"):
                out.append({
                    "path": "group.%d.population.%d.criteria.expression" % (gi, pi),
                    "value": ex["expression"],
                    "population": pop.get("id"),
                })
        st = g.get("stratifier")
        stratifiers = st if isinstance(st, list) else ([st] if st else [])
        for si, s in enumerate(stratifiers):
            crit = s.get("criteria")
            crits = crit if isinstance(crit, list) else ([crit] if crit else [])
            for ci, c in enumerate(crits):
                if c and c.get("language") == "text/cql-identifier" and c.get("expression"):
                    if isinstance(crit, list):
                        path = "group.%d.stratifier.%d.criteria.%d.expression" % (gi, si, ci)
                    else:
                        path = "group.%d.stratifier.%d.criteria.expression" % (gi, si)
                    out.append({
                        "path": path,
                        "value": c["expression"],
                        "stratifier": s.get("id"),
                    })
    return out


def check_measure_criteria(doc: dict[str, Any], kb: AncKbX) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    stmts: set[str] = set()
    for canon in doc.get("library") or []:
        if canon not in kb.library_elm_statements:
            violations.append({"kind": "unresolved_measure_library", "canonical": canon})
        else:
            stmts |= set(kb.library_elm_statements[canon])
    for slot in _measure_criteria_slots(doc):
        if slot["value"] not in stmts:
            violations.append({
                "kind": "unresolved_criteria_identifier",
                "path": slot["path"],
                "identifier": slot["value"],
            })
    return violations


# ---------------------------------------------------------------------------
# ConceptMap
# ---------------------------------------------------------------------------

def _conceptmap_element_slots(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for gi, g in enumerate(doc.get("group") or []):
        if g.get("source") != CUSTOM_CS:
            continue
        for ei, el in enumerate(g.get("element") or []):
            if el.get("code"):
                out.append({
                    "path": "group.%d.element.%d.code" % (gi, ei),
                    "value": el["code"],
                    "display_path": "group.%d.element.%d.display" % (gi, ei),
                    "display": el.get("display"),
                })
    return out


def check_conceptmap_source_membership(
    doc: dict[str, Any], kb: AncKbX
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for slot in _conceptmap_element_slots(doc):
        code = slot["value"]
        if code not in kb.codesystem_codes:
            violations.append({
                "kind": "unknown_source_code", "path": slot["path"], "code": code,
            })
            continue
        recorded = kb.codesystem_display.get(code)
        if slot["display"] is not None and recorded is not None and slot["display"] != recorded:
            violations.append({
                "kind": "source_display_mismatch", "path": slot["display_path"],
                "code": code, "declared": slot["display"], "recorded": recorded,
            })
    return violations


# ---------------------------------------------------------------------------
# Library dataRequirements / relatedArtifact / CQL coverage
# ---------------------------------------------------------------------------

def _codefilter_slots(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for di, dr in enumerate(doc.get("dataRequirement") or []):
        for fi, cf in enumerate(dr.get("codeFilter") or []):
            if cf.get("valueSet"):
                out.append({
                    "path": "dataRequirement.%d.codeFilter.%d.valueSet" % (di, fi),
                    "value": cf["valueSet"],
                })
    return out


def check_library_datarequirements(
    doc: dict[str, Any], kb: AncKbX
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for slot in _codefilter_slots(doc):
        vs = slot["value"]
        base = vs.split("|")[0]
        if base not in kb.canonical_to_path:
            violations.append({
                "kind": "unresolved_codefilter_valueset", "path": slot["path"], "value": vs,
            })
        elif not is_valueset(kb, vs):
            violations.append({
                "kind": "non_valueset_codefilter", "path": slot["path"], "value": vs,
                "resolved_kind": kb.canonical_kinds.get(base),
            })
    return violations


def _related_artifact_slots(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ri, ra in enumerate(doc.get("relatedArtifact") or []):
        res = ra.get("resource")
        if isinstance(res, str) and res.startswith("http://smart.who.int/anc/"):
            out.append({
                "path": "relatedArtifact.%d.resource" % ri,
                "value": res,
                "type": ra.get("type"),
            })
    return out


def check_library_related_artifacts(
    doc: dict[str, Any], kb: AncKbX
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for slot in _related_artifact_slots(doc):
        if slot["value"].split("|")[0] not in kb.canonical_to_path:
            violations.append({
                "kind": "unresolved_related_artifact", "path": slot["path"],
                "value": slot["value"],
            })
    return violations


def check_library_include_coverage(doc: dict[str, Any], kb: AncKbX) -> list[dict[str, Any]]:
    """Cross-artifact DAK consistency: every ``include`` target of the
    companion CQL library (matched by Library ``name``) must resolve to a
    known library canonical and be covered by an anc-scoped
    ``relatedArtifact`` dependency of the Library JSON."""
    violations: list[dict[str, Any]] = []
    name = doc.get("name")
    includes = kb.cql_includes.get(name, frozenset()) if name else frozenset()
    covered = {s["value"].split("|")[0] for s in _related_artifact_slots(doc)}
    for inc in sorted(includes):
        canon = kb.include_canonical_by_name.get(inc)
        if canon is None:
            violations.append({
                "kind": "unknown_include_library", "library": name, "include": inc,
            })
        elif canon not in covered:
            violations.append({
                "kind": "uncovered_include", "library": name, "include": inc,
                "canonical": canon,
            })
    return violations


# ---------------------------------------------------------------------------
# ValueSet display consistency
# ---------------------------------------------------------------------------

def _display_slots(doc: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, inc in enumerate((doc.get("compose") or {}).get("include") or []):
        if inc.get("system") != CUSTOM_CS:
            continue
        for j, c in enumerate(inc.get("concept") or []):
            if c.get("code"):
                out.append({
                    "path": "compose.include.%d.concept.%d.display" % (i, j),
                    "value": c.get("display"),
                    "code_path": "compose.include.%d.concept.%d.code" % (i, j),
                    "code": c["code"],
                })
    return out


def check_valueset_displays(doc: dict[str, Any], kb: AncKbX) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for slot in _display_slots(doc):
        code = slot["code"]
        if code not in kb.codesystem_codes:
            violations.append({
                "kind": "unknown_codesystem_code", "path": slot["code_path"], "code": code,
            })
            continue
        recorded = kb.codesystem_display.get(code)
        if slot["value"] is not None and recorded is not None and slot["value"] != recorded:
            violations.append({
                "kind": "display_mismatch", "path": slot["path"], "code": code,
                "declared": slot["value"], "recorded": recorded,
            })
    return violations


# ---------------------------------------------------------------------------
# CQL valueset declarations
# ---------------------------------------------------------------------------

def parse_valueset_declarations(text: str) -> list[dict[str, Any]]:
    out = []
    for m in _VSDECL_LINE_RE.finditer(text):
        out.append({
            "name": m.group(1),
            "url": m.group(2),
            "line": m.group(0),
            "span": list(m.span()),
        })
    return out


def check_cql_valueset_declarations(text: str, kb: AncKbX) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for decl in parse_valueset_declarations(text):
        url = decl["url"]
        base = url.split("|")[0]
        if base not in kb.canonical_to_path:
            violations.append({
                "kind": "unresolved_valueset_declaration", "name": decl["name"], "url": url,
            })
        elif not is_valueset(kb, url):
            violations.append({
                "kind": "non_valueset_declaration", "name": decl["name"], "url": url,
                "resolved_kind": kb.canonical_kinds.get(base),
            })
    return violations


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def apply_corruptions(artifact: Any, slots: list[dict[str, Any]]) -> Any:
    return legacy.apply_corruptions(artifact, slots)


def evidence_record(
    check_name: str,
    artifact: Any,
    violations: list[dict[str, Any]],
    *,
    context: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check": check_name,
        "engine": ENGINE,
        "engine_version": ENGINE_VERSION,
        "profile": PROFILE,
        "source_commit": ANC_COMMIT,
        "artifact_sha256": sha256_json(artifact),
        "violation_count": len(violations),
        "violations": violations[:20],
        "context": context,
        **(extra or {}),
    }
