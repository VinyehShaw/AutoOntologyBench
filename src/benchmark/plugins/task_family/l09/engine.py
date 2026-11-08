"""L09 native engine: deterministic structural checks over the pinned WHO SMART-ANC snapshot.

Native scope (declared semantic profile ``smart-anc-fhir-structural/1``): every
check is recomputed in pure Python from the pinned artifacts — no CQL evaluation
(the official cql-evaluator is Java-based and Java is absent on this host):

- ``check_condition_identifiers``  PlanDefinition ``condition`` (kind
  ``applicability``, language ``text/cql-identifier``) identifiers must resolve
  to ELM ``statements.def[].name`` of the Library the PlanDefinition links.
- ``check_cql_includes``           CQL ``include`` statements must resolve to
  pinned CQL libraries; a declared version must match the target library's
  ``library ... version 'v'`` header.
- ``check_cql_valueset_refs``      aliased ``Alias."Name"`` references must
  resolve to a ``valueset``/``define`` declaration inside the included library.
- ``check_valueset_expansion``     the expansion recomputed from ``compose``
  (recursive grouper includes + explicit concept lists) must equal the
  publisher-recorded ``expansion.contains`` set; custom-CodeSystem concept
  codes must exist in the pinned CodeSystem.
- ``check_plan_references``        PlanDefinition ``definitionCanonical`` and
  ``relatedAction.actionId`` must resolve; the relatedAction graph must be
  acyclic, and every action reachable from a root action.

Each check returns a list of violation dicts; empty list = pass. Knowledge-side
perturbation (removing an ELM statement, a CQL declaration, a CodeSystem code, a
canonical target) is applied to *views* of the KB so witnesses never mutate the
pinned snapshot.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[4]
ANC_ROOT = (
    REPO_ROOT / "external_knowledge" / "pinned" / "smart-anc" / "commit-301a6b8b9043"
)
ANC_COMMIT = "301a6b8b904361315ca38b28ee9fbd133f9d0e94"
ENGINE = "python-structural"
ENGINE_VERSION = "1.0.0"
PROFILE = "smart-anc-fhir-structural/1"
CUSTOM_CS = "http://smart.who.int/anc/CodeSystem/anc-custom-codes"

_INC_RE = re.compile(
    r"^include\s+(\S+?)(?:\s+version\s+'([^']+)')?(?:\s+called\s+(\S+))?\s*$", re.M
)
_LIB_RE = re.compile(r"^library\s+(\S+?)(?:\s+version\s+'([^']+)')?\s*$", re.M)
_VSDECL_RE = re.compile(r'valueset\s+"([^"]+)"\s*:\s*\'([^\']+)\'')
_DEFINE_RE = re.compile(r'define\s+(?:fluent\s+)?"([^"]+)"\s*:', re.M)
_ALIASREF_RE = re.compile(r'\b([A-Za-z]\w*)\."([^"]+)"')


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get_path(obj: Any, path: str) -> Any:
    """Dotted path with numeric segments for arrays; ``None`` when absent."""
    cur = obj
    for seg in path.split("."):
        if isinstance(cur, list):
            if not seg.isdigit() or int(seg) >= len(cur):
                return None
            cur = cur[int(seg)]
        elif isinstance(cur, dict):
            if seg not in cur:
                return None
            cur = cur[seg]
        else:
            return None
    return cur


def set_path(obj: Any, path: str, value: Any) -> None:
    cur = obj
    segs = path.split(".")
    for seg in segs[:-1]:
        cur = cur[int(seg)] if isinstance(cur, list) else cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


@dataclass(frozen=True)
class KbPerturbation:
    """Knowledge-side counterfactual view spec (applied to a KB copy)."""

    drop_elm_statement: tuple[str, str] | None = None  # (library canonical, statement)
    drop_cql_include_target: str | None = None  # library name removed from index
    drop_cql_declaration: tuple[str, str] | None = None  # (library name, decl name)
    drop_codesystem_code: str | None = None
    drop_member_expansion: tuple[str, frozenset] | None = None  # (vs url, codes)
    drop_canonical: str | None = None  # canonical removed from index


@dataclass(frozen=True)
class AncKB:
    """Knowledge-base view over the pinned snapshot (immutable)."""

    canonical_to_path: dict[str, str]
    cql_library_versions: dict[str, str | None]  # library name -> declared version
    cql_declarations: dict[str, frozenset]  # library name -> valueset names + define names
    valueset_recorded: dict[str, frozenset]  # vs canonical -> {(system, code)}
    codesystem_codes: frozenset
    library_elm_statements: dict[str, frozenset]  # library canonical -> statement names

    def with_perturbation(self, p: KbPerturbation) -> "AncKB":
        if p.drop_elm_statement:
            canon, stmt = p.drop_elm_statement
            stmts = set(self.library_elm_statements.get(canon, frozenset()))
            stmts.discard(stmt)
            return dataclass_replace(
                self, library_elm_statements={**self.library_elm_statements, canon: frozenset(stmts)}
            )
        if p.drop_cql_include_target:
            name = p.drop_cql_include_target
            return dataclass_replace(
                self,
                cql_library_versions={k: v for k, v in self.cql_library_versions.items() if k != name},
            )
        if p.drop_cql_declaration:
            lib, decl = p.drop_cql_declaration
            decls = set(self.cql_declarations.get(lib, frozenset()))
            decls.discard(decl)
            return dataclass_replace(
                self, cql_declarations={**self.cql_declarations, lib: frozenset(decls)}
            )
        if p.drop_codesystem_code:
            codes = set(self.codesystem_codes)
            codes.discard(p.drop_codesystem_code)
            return dataclass_replace(self, codesystem_codes=frozenset(codes))
        if p.drop_member_expansion:
            url, codes = p.drop_member_expansion
            cur = set(self.valueset_recorded.get(url, frozenset()))
            cur -= set(codes)
            return dataclass_replace(
                self, valueset_recorded={**self.valueset_recorded, url: frozenset(cur)}
            )
        if p.drop_canonical:
            return dataclass_replace(
                self,
                canonical_to_path={k: v for k, v in self.canonical_to_path.items() if k != p.drop_canonical},
            )
        return self


def dataclass_replace(obj: Any, **kw: Any) -> Any:
    return dataclasses.replace(obj, **kw)


def load_artifact(rel_path: str) -> Any:
    return json.loads((ANC_ROOT / rel_path).read_text(encoding="utf-8"))


_KB_CACHE: AncKB | None = None


def load_kb() -> AncKB:
    global _KB_CACHE
    if _KB_CACHE is not None:
        return _KB_CACHE
    canonical_to_path: dict[str, str] = {}
    valueset_recorded: dict[str, frozenset] = {}
    codes: set[str] = set()
    elm_statements: dict[str, frozenset] = {}
    cql_versions: dict[str, str | None] = {}
    cql_decls: dict[str, frozenset] = {}

    for scope in ("input/resources", "input/vocabulary", "input/profiles"):
        for path in sorted((ANC_ROOT / scope).rglob("*.json")):
            rel = str(path.relative_to(ANC_ROOT))
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            url = doc.get("url")
            if isinstance(url, str):
                canonical_to_path[url] = rel
            if doc.get("resourceType") == "ValueSet" and url:
                rec = (doc.get("expansion") or {}).get("contains") or []
                valueset_recorded[url] = frozenset(
                    (c.get("system"), c.get("code")) for c in rec if c.get("code")
                )
            if doc.get("resourceType") == "CodeSystem" and url == CUSTOM_CS:
                codes = {c["code"] for c in doc.get("concept", []) if c.get("code")}
            if doc.get("resourceType") == "Library" and url:
                for content in doc.get("content", []):
                    if content.get("contentType") == "application/elm+json" and content.get("data"):
                        elm = json.loads(base64.b64decode(content["data"]))
                        stmts = {
                            s["name"]
                            for s in (elm["library"].get("statements") or {}).get("def", [])
                            if s.get("name")
                        }
                        elm_statements[url] = frozenset(stmts)
                        break

    for path in sorted((ANC_ROOT / "input/cql").glob("*.cql")):
        text = path.read_text(encoding="utf-8")
        m = _LIB_RE.search(text)
        if not m:
            continue
        name, version = m.group(1), m.group(2)
        cql_versions[name] = version
        decls = set(dict(_VSDECL_RE.findall(text)).keys())
        decls |= set(_DEFINE_RE.findall(text))
        cql_decls[name] = frozenset(decls)

    _KB_CACHE = AncKB(
        canonical_to_path=canonical_to_path,
        cql_library_versions=cql_versions,
        cql_declarations=cql_decls,
        valueset_recorded=valueset_recorded,
        codesystem_codes=frozenset(codes),
        library_elm_statements=elm_statements,
    )
    return _KB_CACHE


# ---------------------------------------------------------------------------
# PlanDefinition helpers
# ---------------------------------------------------------------------------

def iter_actions(pd: dict[str, Any], prefix: str = "action") -> Iterable[tuple[str, dict[str, Any]]]:
    """Yield (json_path, action) for every action, depth-first."""
    for i, act in enumerate(pd.get("action") or []):
        path = "%s.%d" % (prefix, i)
        yield path, act
        yield from iter_actions(act, path + ".action")


def condition_slots(pd: dict[str, Any]) -> list[dict[str, Any]]:
    """Every applicability cql-identifier condition as a corruptible slot."""
    out: list[dict[str, Any]] = []
    for path, act in iter_actions(pd):
        for j, cond in enumerate(act.get("condition") or []):
            expr = cond.get("expression") or {}
            if cond.get("kind") == "applicability" and expr.get("language") == "text/cql-identifier":
                out.append({
                    "path": "%s.condition.%d.expression.expression" % (path, j),
                    "value": expr.get("expression"),
                    "action_title": act.get("title"),
                })
    return out


def check_condition_identifiers(pd: dict[str, Any], kb: AncKB) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    libs = pd.get("library") or []
    stmts: set[str] = set()
    for canon in libs:
        if canon not in kb.library_elm_statements:
            violations.append({"kind": "unresolved_library", "canonical": canon})
        else:
            stmts |= set(kb.library_elm_statements[canon])
    for slot in condition_slots(pd):
        if slot["value"] not in stmts:
            violations.append({
                "kind": "unresolved_condition_identifier",
                "path": slot["path"],
                "identifier": slot["value"],
            })
    return violations


def check_plan_references(pd: dict[str, Any], kb: AncKB) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    actions = dict(iter_actions(pd))
    ids = {a.get("id"): p for p, a in actions.items() if a.get("id")}
    edges: dict[str, str] = {}
    for path, act in actions.items():
        canon = act.get("definitionCanonical")
        if canon and canon.split("|")[0] not in kb.canonical_to_path:
            violations.append({
                "kind": "unresolved_definition_canonical",
                "path": path + ".definitionCanonical",
                "canonical": canon,
            })
        for j, rel in enumerate(act.get("relatedAction") or []):
            target = rel.get("actionId")
            if target not in ids:
                violations.append({
                    "kind": "dangling_related_action",
                    "path": "%s.relatedAction.%d.actionId" % (path, j),
                    "target": target,
                })
            elif act.get("id"):
                edges[act["id"]] = target
    # cycle detection over before/after links
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {i: WHITE for i in edges}
    def dfs(n: str) -> bool:
        color[n] = GRAY
        nxt = edges.get(n)
        if nxt is not None:
            if color.get(nxt) == GRAY:
                return True
            if color.get(nxt) == WHITE and dfs(nxt):
                return True
        color[n] = BLACK
        return False
    for n in list(edges):
        if color.get(n) == WHITE and dfs(n):
            violations.append({"kind": "related_action_cycle", "at": n})
            break
    # reachability: actions with no incoming link are roots; all id-actions reachable
    if edges:
        targets = set(edges.values())
        roots = [i for i in ids if i not in targets]
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(t for s, t in edges.items() if s == n)
        for i in ids:
            if i not in seen and i not in roots and ids[i]:
                violations.append({"kind": "unreachable_action", "action_id": i})
    return violations


# ---------------------------------------------------------------------------
# CQL helpers
# ---------------------------------------------------------------------------

def parse_includes(cql_text: str) -> list[dict[str, Any]]:
    out = []
    for m in _INC_RE.finditer(cql_text):
        out.append({
            "library": m.group(1),
            "version": m.group(2),
            "alias": m.group(3),
            "line": m.group(0),
            "span": list(m.span()),
        })
    return out


def check_cql_includes(cql_text: str, kb: AncKB) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for inc in parse_includes(cql_text):
        name = inc["library"]
        if name not in kb.cql_library_versions:
            violations.append({"kind": "unresolved_include", "library": name})
            continue
        pinned_version = kb.cql_library_versions[name]
        if inc["version"] and pinned_version and inc["version"] != pinned_version:
            violations.append({
                "kind": "include_version_mismatch",
                "library": name,
                "declared": inc["version"],
                "pinned": pinned_version,
            })
    return violations


def check_cql_valueset_refs(cql_text: str, kb: AncKB) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    incs = parse_includes(cql_text)
    alias_to_lib = {i["alias"] or i["library"]: i["library"] for i in incs}
    for alias, refname in _ALIASREF_RE.findall(cql_text):
        lib = alias_to_lib.get(alias)
        if lib is None:
            continue  # unresolved alias is check_cql_includes' job
        decls = kb.cql_declarations.get(lib, frozenset())
        if refname not in decls:
            violations.append({
                "kind": "unresolved_aliased_reference",
                "alias": alias,
                "library": lib,
                "reference": refname,
            })
    return violations


# ---------------------------------------------------------------------------
# ValueSet helpers
# ---------------------------------------------------------------------------

def recompute_expansion(vs: dict[str, Any], kb: AncKB) -> frozenset:
    """Deterministic expansion of the *target* ValueSet from its ``compose``.

    Explicit ``concept`` lists contribute their (system, code) pairs; grouper
    ``valueSet`` member references contribute the member's publisher-recorded
    expansion from the pinned snapshot (members of eligible groupers always have
    one). Filter/system-only includes contribute nothing (such artifacts are
    excluded from eligibility at proposal time). Because member expansions are
    read from the KB view, a knowledge-side perturbation of a member's recorded
    expansion flips the grouper check.
    """
    out: set = set()
    for inc in (vs.get("compose") or {}).get("include") or []:
        if "concept" in inc:
            for c in inc.get("concept") or []:
                if c.get("code"):
                    out.add((inc.get("system"), c["code"]))
        for member_url in inc.get("valueSet") or []:
            out |= set(kb.valueset_recorded.get(member_url, frozenset()))
    return frozenset(out)


def check_valueset_expansion(vs: dict[str, Any], kb: AncKB) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    url = vs.get("url")
    recomputed = recompute_expansion(vs, kb)
    recorded = kb.valueset_recorded.get(url, frozenset())
    for system, code in sorted(set(recomputed) - set(recorded)):
        violations.append({"kind": "unexpected_concept", "system": system, "code": code})
    for system, code in sorted(set(recorded) - set(recomputed)):
        violations.append({"kind": "missing_concept", "system": system, "code": code})
    for inc in (vs.get("compose") or {}).get("include") or []:
        if inc.get("system") == CUSTOM_CS:
            for c in inc.get("concept") or []:
                if c.get("code") and c["code"] not in kb.codesystem_codes:
                    violations.append({
                        "kind": "unknown_codesystem_code",
                        "system": CUSTOM_CS,
                        "code": c["code"],
                    })
    return violations


# ---------------------------------------------------------------------------
# corruption + witness helpers
# ---------------------------------------------------------------------------

def apply_corruptions(artifact: Any, slots: list[dict[str, Any]]) -> Any:
    """Return a copy of ``artifact`` with each slot's path set to wrong_value."""
    doc = copy.deepcopy(artifact)
    for slot in slots:
        set_path(doc, slot["path"], slot["wrong_value"])
    return doc


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
