"""L04 task families: 27 structural families over pinned Rhea 142 + ChEBI 256.

Every family maps a ``params`` dict to an exact answer set computed over the
streaming KB (fast path) and independently re-computed by the verifier plugin
via rdflib SPARQL over the pinned RDF graphs (+ owlrl closure only for the one
family whose declared profile supports it: ``rdf-sparql+owlrl-rdfs/1``).

Family axes (FINAL-BUILD-ORDER-v2 §1 L04: 跨源关联 / 组合反应条件 / 参与物映射修复):
  org_*   organization    — cross-source joins (Rhea RDF <-> ChEBI OWL / TSV maps)
  app_*   application     — combined reaction-condition exact-set queries
  rep_*   repair          — corrupted artifact repair verified by downstream answers
  con_*   construction    — knowledge construction under a budgeted probe interface
  ont_*   application/org — ChEBI subclass-closure tasks (cross-source when joined)

Instance state axes: ``evidence_bundle_variant`` (org/app/ont: tsv_view vs
rdf_view public bundles), ``corruption_seed`` (rep), ``evidence_sample_seed``
(con). Base goal identity = sha(family, canonical params); instance identity =
sha(base_goal, state descriptor). Paraphrase-only multiplication is never used.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from benchmark.plugins.source.l04.kb import L04KB

STABLE_SEED = 20261027

#: Instances shipped per base goal (state-axis variation only; never paraphrase
#: multiplication). 25 families x (14..15 goals) x 5 instances = 360 goals /
#: 1800 instances for the aob-8000-rc1 L04 quota.
INSTANCES_PER_GOAL = 5

#: Tools the agent may call in every L04 task (contract §1 public.allowed_tools).
ALLOWED_TOOLS = ["inspect_schema", "inspect_state", "read_public_doc",
                 "probe_native", "edit_artifact", "validate_artifact",
                 "check_action_with_ontology", "submit_final"]

PROFILE_LINKED = "linked-data/1"
PROFILE_SPARQL = "rdf-sparql/1"
PROFILE_CLOSURE = "rdf-sparql+closure/1"
PROFILE_OWLRL = "rdf-sparql+owlrl-rdfs/1"

RH = "http://rdf.rhea-db.org/"
CHEBI_IRI = "http://purl.obolibrary.org/obo/CHEBI_"
GO_IRI = "http://purl.obolibrary.org/obo/GO_"
EC_IRI = "http://purl.uniprot.org/enzyme/"


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def base_goal_id(family_id: str, params: dict[str, Any]) -> str:
    return sha_text(f"l04|{family_id}|{canonical_params(params)}")[:32]


def even_spread(candidates: list[Any], size_of: Callable[[Any], int], n: int) -> list[Any]:
    ordered = sorted(candidates, key=lambda c: (size_of(c), canonical_params(c)
                                                if isinstance(c, dict) else str(c)))
    if n >= len(ordered):
        return ordered
    return [ordered[round(i * (len(ordered) - 1) / (n - 1))] for i in range(n)]


# ---------------------------------------------------------------------------
# answer helpers over the KB (fast path)
# ---------------------------------------------------------------------------

_C2M: dict[int, dict[str, frozenset[str]]] = {}
_ROWCOUNT: dict[int, dict[str, int]] = {}
_ANC: dict[int, dict[str, frozenset[str]]] = {}
_RXN2COMPOUNDS: dict[int, dict[str, frozenset[str]]] = {}
_RXN2CHEBI: dict[int, dict[str, frozenset[str]]] = {}


def _rxn2compounds(kb: L04KB) -> dict[str, frozenset[str]]:
    key = id(kb)
    if key not in _RXN2COMPOUNDS:
        m: dict[str, set[str]] = {}
        for r, _s, _p, c, _ch in kb.participants:
            m.setdefault(r, set()).add(c)
        _RXN2COMPOUNDS[key] = {k: frozenset(v) for k, v in m.items()}
    return _RXN2COMPOUNDS[key]


def _rxn2chebi(kb: L04KB) -> dict[str, frozenset[str]]:
    key = id(kb)
    if key not in _RXN2CHEBI:
        m: dict[str, set[str]] = {}
        for r, _s, _p, _c, ch in kb.participants:
            m.setdefault(r, set()).add(ch)
        _RXN2CHEBI[key] = {k: frozenset(v) for k, v in m.items()}
    return _RXN2CHEBI[key]


def _c2m(kb: L04KB) -> dict[str, frozenset[str]]:
    key = id(kb)
    if key not in _C2M:
        masters = _master_cache(kb)
        m: dict[str, set[str]] = {}
        for r, _s, _p, _c, ch in kb.participants:
            if r in masters:
                m.setdefault(ch, set()).add(r)
        _C2M[key] = {k: frozenset(v) for k, v in m.items()}
    return _C2M[key]


def _rowcount(kb: L04KB) -> dict[str, int]:
    key = id(kb)
    if key not in _ROWCOUNT:
        rc: dict[str, int] = {}
        for r, _s, _p, _c, _ch in kb.participants:
            rc[r] = rc.get(r, 0) + 1
        _ROWCOUNT[key] = rc
    return _ROWCOUNT[key]


def _ancestors(kb: L04KB, chebi: str) -> frozenset[str]:
    key = id(kb)
    cache = _ANC.setdefault(key, {})
    if chebi not in cache:
        cache[chebi] = kb.chebi_ancestors(chebi)
    return cache[chebi]


def masters_involving(kb: L04KB, masters: frozenset[str], chebi: str) -> frozenset[str]:
    return _c2m(kb).get(chebi, frozenset())


def edge_cuttable(kb: L04KB, anchor: str) -> bool:
    """True iff removing one direct rdfs:subClassOf edge of ``anchor`` changes
    its transitive ancestor closure (ChEBI multiple inheritance makes most
    single-edge cuts no-ops; the counterfactual witness needs a real cut)."""
    parents = kb.chebi_parents.get(anchor, ())
    if not parents:
        return False
    full = _ancestors(kb, anchor)
    for p in parents:
        reduced: set[str] = set()
        for q in parents:
            if q == p:
                continue
            reduced.add(q)
            reduced |= set(_ancestors(kb, q))
        if full - reduced:
            return True
    return False


def participant_rows(kb: L04KB, rhea_ids: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted({
        "|".join([r, s, p, c, ch])
        for r, s, p, c, ch in kb.participants if r in rhea_ids
    }))


def _sorted_ids(ids: frozenset[str]) -> tuple[str, ...]:
    return tuple(sorted(ids, key=lambda r: int(r.split(":")[1])))


@dataclass(frozen=True)
class Family:
    family_id: str
    regime: str
    profile: str
    complexity: str
    description: str
    state_axis: str
    instances_per_goal: int
    max_base_goals: int
    min_answer: int
    max_answer: int
    propose: Callable[[L04KB, random.Random], Iterator[dict[str, Any]]]
    answer: Callable[[L04KB, dict[str, Any], frozenset[str]], tuple[str, ...]]
    goal_text: Callable[[dict[str, Any], L04KB], str]
    witness_kind: str


def state_descriptor(fam: Family, base_seed: int, inst_idx: int) -> dict[str, Any]:
    """Instance identity on the family's state axis. ``evidence_bundle_variant``
    cycles tsv/rdf serializations and (from idx 2) adds a seeded distractor
    reaction set; the other axes vary the corruption/evidence seed."""
    if fam.state_axis == "evidence_bundle_variant":
        if inst_idx == 0:
            return {"axis": fam.state_axis, "variant": "tsv_view"}
        if inst_idx == 1:
            return {"axis": fam.state_axis, "variant": "rdf_view"}
        variant = "tsv_view" if inst_idx % 2 == 0 else "rdf_view"
        return {"axis": fam.state_axis, "variant": variant,
                "distractor_seed": (base_seed * 7919 + inst_idx * 104729)
                                   & 0xFFFFFFFF}
    return {"axis": fam.state_axis,
            "seed": (base_seed * 131_071 + inst_idx * 1_009) & 0xFFFFFFFF}


def source_unit_ids(params: dict[str, Any]) -> tuple[str, ...]:
    """Source-unit anchors named by the task params (never answer members)."""
    ids: list[str] = []
    for key in ("chebi", "chebi_a", "chebi_b", "ancestor", "ph73"):
        if key in params:
            ids.append(f"chebi_entity/{params[key]}")
    for c in params.get("chebi_set", []):
        ids.append(f"chebi_entity/{c}")
    if "master" in params:
        ids.append(f"rhea_reaction/{params['master']}")
    if "rhea" in params:
        ids.append(f"rhea_reaction/{params['rhea']}")
    if "go" in params:
        ids.append(f"rhea_go_link/{params['go']}")
    if "ec" in params:
        ids.append(f"rhea_xref/EC:{params['ec']}")
    if "ec_class" in params:
        ids.append(f"rhea_xref/EC_class:{params['ec_class']}")
    if "db" in params:
        ids.append(f"rhea_xref/{params['db']}")
    if "status" in params:
        ids.append(f"rhea_reaction/status:{params['status']}")
    if "side" in params:
        ids.append(f"rhea_participant/side:{params['side']}")
    return tuple(sorted(set(ids)))


# ---------------------------------------------------------------------------
# parameter pools
# ---------------------------------------------------------------------------

def _pool_chebi_by_size(kb: L04KB, masters: frozenset[str],
                        lo: int, hi: int) -> list[str]:
    return [c for c, v in _c2m(kb).items() if lo <= len(v) < hi]


def _rows_count(kb: L04KB, masters: frozenset[str], chebi: str) -> int:
    rc = _rowcount(kb)
    return sum(rc.get(r, 0) for r in masters_involving(kb, masters, chebi))


# ---------------------------------------------------------------------------
# family definitions
# ---------------------------------------------------------------------------

def _mk_org_reactions_involving() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = _pool_chebi_by_size(kb, _master_cache(kb), 20, 500)
        for c in even_spread(pool, lambda c: len(masters_involving(kb, _master_cache(kb), c)), 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        return _sorted_ids(masters_involving(kb, masters, params["chebi"]))
    def goal(params, kb):
        name = kb.chebi_labels.get(params["chebi"], "")
        return (f"Assemble the published Rhea-to-ChEBI participant cross-references and "
                f"report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers (RHEA:*) having at least one participant cross-referenced "
                f"to ChEBI entity {params['chebi']} ({name}). Identity must come from "
                f"published cross-reference IDs, never from names.")
    return Family("org_reactions_involving_chebi", "organization", PROFILE_LINKED,
                  "low", "master reactions involving a ChEBI entity (W14 type 1)",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 20, 500,
                  propose, answer, goal, "swap_accession")


def _mk_org_participant_rows() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = [c for c in _pool_chebi_by_size(kb, masters, 5, 130)
                if 20 <= _rows_count(kb, masters, c) < 500]
        for c in even_spread(pool, lambda c: _rows_count(kb, masters, c), 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        return participant_rows(kb, masters_involving(kb, masters, params["chebi"]))
    def goal(params, kb):
        return (f"Assemble the published Rhea-to-ChEBI participant cross-references and "
                f"report the exact, complete, sorted set of participant metadata rows "
                f"(RHEA:* | side | participant IRI | compound IRI | CHEBI:*) for every "
                f"master Rhea reaction with at least one participant cross-referenced "
                f"to ChEBI entity {params['chebi']}.")
    return Family("org_participant_rows_involving_chebi", "organization", PROFILE_LINKED,
                  "medium", "two-stage participant-row join (W14 type 2)",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 20, 500,
                  propose, answer, goal, "swap_accession")


def _mk_org_conjunction() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        rxn2ch = kb.reaction_to_chebis
        pool = _pool_chebi_by_size(kb, masters, 30, 400)
        rng.shuffle(pool)
        seen = set()
        out: list[dict[str, Any]] = []
        for a in pool:
            partners: dict[str, int] = {}
            for r in masters_involving(kb, masters, a):
                for b in rxn2ch.get(r, ()):
                    if b != a:
                        partners[b] = partners.get(b, 0) + 1
            for b, n in partners.items():
                if 5 <= n < 200 and (b, a) not in seen and b in pool:
                    seen.add((a, b))
                    out.append({"chebi_a": a, "chebi_b": b})
        for p in even_spread(out, lambda p: p["_n"] if "_n" in p else 5, 60):
            yield p
    def answer(kb, params, masters):
        return _sorted_ids(masters_involving(kb, masters, params["chebi_a"])
                           & masters_involving(kb, masters, params["chebi_b"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers that involve BOTH ChEBI entity {params['chebi_a']} AND "
                f"ChEBI entity {params['chebi_b']} as participants, using only "
                f"published cross-reference IDs.")
    return Family("org_reactions_chebi_conjunction", "organization", PROFILE_SPARQL,
                  "medium", "two-entity conjunctive participant join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 5, 200,
                  propose, answer, goal, "swap_accession")


def _mk_org_difference() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        rxn2ch = kb.reaction_to_chebis
        pool = _pool_chebi_by_size(kb, masters, 40, 500)
        rng.shuffle(pool)
        out = []
        seen = set()
        for a in pool[:600]:
            partners: dict[str, int] = {}
            for r in masters_involving(kb, masters, a):
                for b in rxn2ch.get(r, ()):
                    if b != a:
                        partners[b] = partners.get(b, 0) + 1
            for b, inter in partners.items():
                if inter >= 3 and (b, a) not in seen:
                    diff = len(masters_involving(kb, masters, a)) - inter
                    if 10 <= diff < 300:
                        seen.add((a, b))
                        out.append({"chebi_a": a, "chebi_b": b})
        for p in even_spread(out, lambda p: 10, 60):
            yield p
    def answer(kb, params, masters):
        return _sorted_ids(masters_involving(kb, masters, params["chebi_a"])
                           - masters_involving(kb, masters, params["chebi_b"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers that involve ChEBI entity {params['chebi_a']} but do NOT "
                f"involve ChEBI entity {params['chebi_b']}, using only published "
                f"cross-reference IDs.")
    return Family("org_reactions_chebi_difference", "organization", PROFILE_SPARQL,
                  "medium", "two-entity difference join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 300,
                  propose, answer, goal, "swap_accession")


def _mk_org_ancestor() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        uni = frozenset(kb.chebi_to_reactions)
        counts: dict[str, int] = {}
        descendants_cache: dict[str, frozenset[str]] = {}
        for c in uni:
            for a in _ancestors(kb, c):
                counts[a] = counts.get(a, 0) + 1
        pool = [a for a, n in counts.items()
                if 8 <= n <= 200 and edge_cuttable(kb, a)]
        def rxn_count(a: str) -> int:
            desc = descendants_cache.setdefault(
                a, frozenset(c for c in uni if a in _ancestors(kb, c)))
            return len(set().union(*(masters_involving(kb, masters, c) for c in desc)))
        pool = [a for a in pool if 20 <= rxn_count(a) <= 800]
        for a in even_spread(pool, rxn_count, 60):
            yield {"ancestor": a}
    def answer(kb, params, masters):
        uni = frozenset(kb.chebi_to_reactions)
        a = params["ancestor"]
        desc = [c for c in uni if a in _ancestors(kb, c)]
        out: set[str] = set()
        for c in desc:
            out |= masters_involving(kb, masters, c)
        return _sorted_ids(frozenset(out))
    def goal(params, kb):
        name = kb.chebi_labels.get(params["ancestor"], "")
        return (f"Using the pinned ChEBI ontology subclass hierarchy and the published "
                f"Rhea-to-ChEBI cross-references, report the exact, complete, sorted "
                f"set of master Rhea reaction identifiers involving ANY ChEBI entity "
                f"that is a (transitive) subclass of {params['ancestor']} ({name}).")
    return Family("org_reactions_chebi_ancestor", "organization", PROFILE_CLOSURE,
                  "high", "ontology-closure cross-source join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 20, 800,
                  propose, answer, goal, "remove_subclass_edge")


def _mk_org_go_terms() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        go_by_master = kb.go_by_master
        pool = []
        for c in _pool_chebi_by_size(kb, masters, 10, 800):
            gos = set()
            for r in masters_involving(kb, masters, c):
                gos |= go_by_master.get(r, frozenset())
            if 2 <= len(gos) <= 60:
                pool.append(c)
        for c in even_spread(pool, lambda c: 2, 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        gos: set[str] = set()
        for r in masters_involving(kb, masters, params["chebi"]):
            gos |= kb.go_by_master.get(r, frozenset())
        return tuple(sorted(gos))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of Gene Ontology terms "
                f"(GO:*) cross-referenced from master Rhea reactions that involve "
                f"ChEBI entity {params['chebi']}, joining the pinned Rhea GO mapping "
                f"with the published participant cross-references.")
    return Family("org_go_terms_of_reactions_involving_chebi", "organization",
                  PROFILE_SPARQL, "medium", "Rhea->GO cross-source join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 60,
                  propose, answer, goal, "swap_accession")


def _mk_org_go_and_chebi() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        go2m = {g: ms for g, ms in kb.go_to_masters.items() if 5 <= len(ms) <= 400}
        r2ch = _rxn2chebi(kb)
        out = []
        for g, ms in sorted(go2m.items()):
            counts: dict[str, int] = {}
            for m in ms:
                for ch in r2ch.get(m, ()):
                    counts[ch] = counts.get(ch, 0) + 1
            for ch, n in counts.items():
                if 3 <= n <= 100:
                    out.append({"go": g, "chebi": ch, "_n": n})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        return _sorted_ids(frozenset(kb.go_to_masters.get(params["go"], ()))
                           & masters_involving(kb, masters, params["chebi"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers that are cross-referenced to GO term {params['go']} AND "
                f"involve ChEBI entity {params['chebi']} as a participant.")
    return Family("org_reactions_by_go_term_and_chebi", "organization", PROFILE_SPARQL,
                  "medium", "GO-annotation + participant combined join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 100,
                  propose, answer, goal, "swap_accession")


def _mk_org_copartners() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        rxn2ch = kb.reaction_to_chebis
        pool = []
        for c in _pool_chebi_by_size(kb, masters, 5, 400):
            partners: set[str] = set()
            for r in masters_involving(kb, masters, c):
                partners |= set(rxn2ch.get(r, ()))
            partners.discard(c)
            if 10 <= len(partners) <= 300:
                pool.append(c)
        for c in even_spread(pool, lambda c: 10, 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        rxn2ch = kb.reaction_to_chebis
        partners: set[str] = set()
        for r in masters_involving(kb, masters, params["chebi"]):
            partners |= set(rxn2ch.get(r, ()))
        partners.discard(params["chebi"])
        return tuple(sorted(partners, key=lambda c: int(c.split(":")[1])))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of ChEBI entity identifiers "
                f"(CHEBI:*) that co-participate with ChEBI entity {params['chebi']} "
                f"in at least one master Rhea reaction (excluding "
                f"{params['chebi']} itself).")
    return Family("org_chebi_copartners", "organization", PROFILE_SPARQL,
                  "medium", "co-participation entity set",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 300,
                  propose, answer, goal, "swap_accession")


def _mk_org_ec_class() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        ec_by_master = kb.ec_by_master
        out = []
        for ec_class in ("1", "2", "3", "4", "5", "6", "7"):
            masters_in_class = {m for m, ecs in ec_by_master.items()
                                if any(e.startswith(ec_class + ".") for e in ecs)}
            chs: dict[str, int] = {}
            for m in masters_in_class:
                for ch in kb.reaction_to_chebis.get(m, frozenset()):
                    chs[ch] = chs.get(ch, 0) + 1
            for ch, n in chs.items():
                if 3 <= n <= 100:
                    out.append({"ec_class": ec_class, "chebi": ch, "_n": n})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        ec_class = params["ec_class"] + "."
        masters_in_class = {m for m, ecs in kb.ec_by_master.items()
                            if any(e.startswith(ec_class) for e in ecs)}
        return _sorted_ids(frozenset(masters_in_class)
                           & masters_involving(kb, masters, params["chebi"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers cross-referenced to EC class {params['ec_class']}.- "
                f"(top-level enzyme class) AND involving ChEBI entity "
                f"{params['chebi']} as a participant.")
    return Family("org_reactions_by_ec_class_and_chebi", "organization", PROFILE_SPARQL,
                  "medium", "EC-class + participant combined join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 100,
                  propose, answer, goal, "swap_accession")


def _mk_org_xrefs() -> Family:
    DBS = ("KEGG_REACTION", "METACYC", "ECOCYC", "REACTOME")
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        xref_by_master = kb.xrefs_by_master
        out = []
        for c in _pool_chebi_by_size(kb, masters, 10, 500):
            ms = masters_involving(kb, masters, c)
            for db in DBS:
                hits = sorted({x for m in ms for x in xref_by_master.get(m, ())
                               if x.startswith(db + ":")})
                if 5 <= len(hits) <= 200:
                    out.append({"chebi": c, "db": db, "_n": len(hits)})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        ms = masters_involving(kb, masters, params["chebi"])
        db = params["db"] + ":"
        return tuple(sorted({x for m in ms for x in kb.xrefs_by_master.get(m, ())
                             if x.startswith(db)}))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of {params['db']} "
                f"cross-reference identifiers (as \"{params['db']}:<id>\") attached to "
                f"master Rhea reactions involving ChEBI entity {params['chebi']}, "
                f"using the pinned Rhea cross-reference mapping.")
    return Family("org_xrefs_of_reactions_involving_chebi", "organization",
                  PROFILE_SPARQL, "medium", "reaction-set -> xref-DB projection",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 5, 200,
                  propose, answer, goal, "swap_accession")


def _mk_app_children() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        rows = [d for d in kb.directions
                if d.get("RHEA_ID_LR") and d.get("RHEA_ID_RL") and d.get("RHEA_ID_BI")]
        for d in even_spread(rows, lambda d: 1, 60):
            yield {"master": "RHEA:" + d["RHEA_ID_MASTER"]}
    def answer(kb, params, masters):
        m = params["master"].split(":")[1]
        row = next(d for d in kb.directions if d["RHEA_ID_MASTER"] == m)
        out = []
        for code, col in (("LR", "RHEA_ID_LR"), ("RL", "RHEA_ID_RL"), ("BI", "RHEA_ID_BI")):
            if row.get(col):
                out.append(f"{code}=RHEA:{row[col]}")
        return tuple(sorted(out))
    def goal(params, kb):
        return (f"Report the exact, complete set of directional and bidirectional "
                f"child reactions of master reaction {params['master']} as "
                f"\"CODE=RHEA:<id>\" assignments (CODE in LR/RL/BI). The pinned "
                f"master and child reaction equations are provided in the evidence "
                f"bundle; in Rhea's writing convention the LR (forward) child's "
                f"equation keeps the master equation's left-hand side, the RL "
                f"(reverse) child swaps the sides, and the bidirectional child "
                f"uses the <=> marker.")
    return Family("app_directional_children_of_master", "application", PROFILE_SPARQL,
                  "low", "master -> directional children expansion",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 3,
                  propose, answer, goal, "swap_direction_children")


def _mk_app_transport() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in _pool_chebi_by_size(kb, masters, 2, 5000):
            n = sum(1 for r in masters_involving(kb, masters, c)
                    if kb.reactions[r]["is_transport"])
            if 2 <= n <= 100:
                pool.append((c, n))
        for c, _n in even_spread(pool, lambda t: t[1], 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        return _sorted_ids(frozenset(
            r for r in masters_involving(kb, masters, params["chebi"])
            if kb.reactions[r]["is_transport"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers flagged as transport reactions (rh:isTransport=true in "
                f"the pinned Rhea RDF) that involve ChEBI entity {params['chebi']}.")
    return Family("app_transport_reactions_involving_chebi", "application",
                  PROFILE_SPARQL, "medium", "transport-flag condition join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 100,
                  propose, answer, goal, "swap_accession")


def _mk_app_unbalanced() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in _pool_chebi_by_size(kb, masters, 2, 5000):
            n = sum(1 for r in masters_involving(kb, masters, c)
                    if not kb.reactions[r]["is_balanced"])
            if 1 <= n <= 60:
                pool.append((c, n))
        for c, _n in even_spread(pool, lambda t: t[1], 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        return _sorted_ids(frozenset(
            r for r in masters_involving(kb, masters, params["chebi"])
            if not kb.reactions[r]["is_balanced"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers NOT flagged as chemically balanced "
                f"(rh:isChemicallyBalanced=false in the pinned Rhea RDF) that involve "
                f"ChEBI entity {params['chebi']}.")
    return Family("app_unbalanced_reactions_involving_chebi", "application",
                  PROFILE_SPARQL, "medium", "chemical-balance condition join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 1, 60,
                  propose, answer, goal, "swap_accession")


def _mk_app_equations_by_ec() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        ec_by_master = kb.ec_by_master
        ec_counts: dict[str, int] = {}
        for m in masters:
            for e in ec_by_master.get(m, ()):
                ec_counts[e] = ec_counts.get(e, 0) + 1
        pool = [e for e, n in ec_counts.items() if 3 <= n <= 12]
        for e in even_spread(pool, lambda e: ec_counts[e], 60):
            yield {"ec": e}
    def answer(kb, params, masters):
        out = []
        for m in sorted(masters, key=lambda r: int(r.split(":")[1])):
            if params["ec"] in kb.ec_by_master.get(m, ()):
                eq = kb.reactions[m]["equation"]
                out.append(f"{m} | {eq}")
        return tuple(sorted(out))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of canonical Rhea equations "
                f"(as \"RHEA:<id> | <equation>\" rows) for all master reactions "
                f"cross-referenced to EC number {params['ec']} in the pinned Rhea RDF.")
    return Family("app_reaction_equations_by_ec", "application", PROFILE_SPARQL,
                  "low", "EC-xref -> exact equation rows",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 12,
                  propose, answer, goal, "remove_xref")


def _mk_app_status() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        out = []
        for status in ("Preliminary", "Obsolete"):
            pool = []
            for c in _pool_chebi_by_size(kb, masters, 2, 5000):
                n = sum(1 for r in masters_involving(kb, masters, c)
                        if kb.reactions[r]["status"] == status)
                if 1 <= n <= 50:
                    pool.append((c, n))
            for c, n in even_spread(pool, lambda t: t[1], 30):
                out.append({"status": status, "chebi": c})
        rng.shuffle(out)
        yield from out[:60]
    def answer(kb, params, masters):
        return _sorted_ids(frozenset(
            r for r in masters_involving(kb, masters, params["chebi"])
            if kb.reactions[r]["status"] == params["status"]))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers with curation status {params['status']} (rh:status in "
                f"the pinned Rhea RDF) that involve ChEBI entity {params['chebi']}.")
    return Family("app_reactions_by_status_and_chebi", "application", PROFILE_SPARQL,
                  "medium", "curation-status condition join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 1, 50,
                  propose, answer, goal, "swap_accession")


def _mk_app_side_participants() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        by_reaction: dict[str, dict[str, list]] = {}
        for r, s, p, c, ch in kb.participants:
            if r in masters:
                by_reaction.setdefault(r, {}).setdefault(s, []).append((p, c, ch))
        out = []
        for r, sides in by_reaction.items():
            for s, rows in sides_iter(rows_by_side=by_reaction[r]):
                if 2 <= len(rows) <= 8:
                    out.append({"rhea": r, "side": s, "_n": len(rows)})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        return tuple(sorted(
            f"{p}|{c}|{ch}"
            for r, s, p, c, ch in kb.participants
            if r == params["rhea"] and s == params["side"]))
    def goal(params, kb):
        side = "left (substrate)" if params["side"] == "L" else "right (product)"
        return (f"Report the exact, complete, sorted set of participant rows "
                f"(participant IRI | compound IRI | CHEBI:*) on the {side} side of "
                f"master Rhea reaction {params['rhea']} per the pinned Rhea RDF.")
    return Family("app_side_participants_of_reaction", "application", PROFILE_SPARQL,
                  "low", "single-side participant rows",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 8,
                  propose, answer, goal, "flip_side")


def sides_iter(rows_by_side: dict[str, list]) -> Iterator[tuple[str, list]]:
    for s in ("L", "R"):
        if s in rows_by_side:
            yield s, rows_by_side[s]


def _mk_rep_crosswalk(k: int, family_id: str, complexity: str) -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = _pool_chebi_by_size(kb, masters, 25, 400)
        for c in even_spread(pool, lambda c: len(masters_involving(kb, _master_cache(kb), c)), 40):
            yield {"chebi": c}
    def answer(kb, params, masters):
        # reference answer of the governed downstream query under the REPAIRED artifact
        return _sorted_ids(masters_involving(kb, masters, params["chebi"]))
    def goal(params, kb):
        return (f"The Rhea-compound-to-ChEBI crosswalk artifact for the reactions in "
                f"scope contains {k} corrupted entr{'y' if k == 1 else 'ies'} "
                f"(a compound IRI mapped to a wrong ChEBI accession). Using the "
                f"published Rhea/ChEBI cross-reference evidence, repair the artifact "
                f"and report the exact, complete, sorted set of master Rhea reaction "
                f"identifiers involving ChEBI entity {params['chebi']} computed "
                f"through your repaired crosswalk.")
    return Family(family_id, "repair", PROFILE_LINKED, complexity,
                  f"crosswalk artifact repair ({k} corrupted entries)",
                  "corruption_seed", INSTANCES_PER_GOAL, 40, 20, 400,
                  propose, answer, goal, "crosswalk_repair")


def _mk_rep_directions() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        rows = [d for d in kb.directions
                if d.get("RHEA_ID_LR") and d.get("RHEA_ID_RL") and d.get("RHEA_ID_BI")]
        for d in even_spread(rows, lambda d: 1, 40):
            yield {"master": "RHEA:" + d["RHEA_ID_MASTER"]}
    def answer(kb, params, masters):
        m = params["master"].split(":")[1]
        row = next(d for d in kb.directions if d["RHEA_ID_MASTER"] == m)
        out = []
        for code, col in (("LR", "RHEA_ID_LR"), ("RL", "RHEA_ID_RL"), ("BI", "RHEA_ID_BI")):
            if row.get(col):
                out.append(f"{code}=RHEA:{row[col]}")
        return tuple(sorted(out))
    def goal(params, kb):
        return (f"The direction-assignment table artifact has the LR/RL child "
                f"reactions of master {params['master']} swapped. The pinned master "
                f"and child reaction equations are provided as evidence (Rhea "
                f"convention: the LR child's equation keeps the master equation's "
                f"left-hand side, the RL child swaps the sides, and the "
                f"bidirectional child uses the <=> marker). Repair the assignment "
                f"and report the exact \"CODE=RHEA:<id>\" set for master "
                f"{params['master']}.")
    return Family("rep_direction_assignment_repair", "repair", PROFILE_SPARQL,
                  "medium", "LR/RL direction assignment repair",
                  "corruption_seed", INSTANCES_PER_GOAL, 40, 2, 3,
                  propose, answer, goal, "swap_direction_children")


def _mk_rep_side_flip() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        by_reaction: dict[str, dict[str, int]] = {}
        for r, s, _p, _c, _ch in kb.participants:
            if r in masters:
                by_reaction.setdefault(r, {}).setdefault(s, 0)
                by_reaction[r][s] += 1
        pool = [r for r, sd in by_reaction.items()
                if 2 <= sd.get("L", 0) <= 8 and 2 <= sd.get("R", 0) <= 8]
        for r in even_spread(pool, lambda r: 1, 40):
            yield {"rhea": r}
    def answer(kb, params, masters):
        return tuple(sorted(
            f"{s}|{p}|{c}|{ch}"
            for r, s, p, c, ch in kb.participants if r == params["rhea"]))
    def goal(params, kb):
        return (f"The participant side-assignment artifact for master reaction "
                f"{params['rhea']} has one participant moved to the wrong side of the "
                f"equation. The pinned canonical equation is provided as evidence. "
                f"Repair the side assignment and report the exact, complete, sorted "
                f"set of \"side|participant IRI|compound IRI|CHEBI:*\" rows for "
                f"{params['rhea']}.")
    return Family("rep_participant_side_flip_repair", "repair", PROFILE_SPARQL,
                  "medium", "participant side-assignment repair",
                  "corruption_seed", INSTANCES_PER_GOAL, 40, 4, 16,
                  propose, answer, goal, "flip_side")


def _mk_rep_ph73() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        # source entities X whose pH-7.3 major-microspecies target is a
        # Rhea-referenced ChEBI entity (X itself need not be reaction-referenced)
        masters = _master_cache(kb)
        c2m = _c2m(kb)
        # the repair bundle needs >=3 source entities sharing the pH-7.3 target
        tgt_sources: dict[str, int] = {}
        for row in kb.ph73:
            if row["CHEBI"] != row["CHEBI_PH7_3"]:
                tgt_sources[row["CHEBI_PH7_3"]] = \
                    tgt_sources.get(row["CHEBI_PH7_3"], 0) + 1
        out = []
        for row in kb.ph73:
            src, tgt = row["CHEBI"], row["CHEBI_PH7_3"]
            if src == tgt or tgt_sources.get(tgt, 0) < 3:
                continue
            mapped = "CHEBI:" + tgt
            n = len(c2m.get(mapped, ()))
            if 5 <= n <= 300:
                out.append({"chebi": "CHEBI:" + src, "ph73": mapped, "_n": n})
        for p in even_spread(out, lambda p: p["_n"], 40):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        return _sorted_ids(masters_involving(kb, masters, params["ph73"]))
    def goal(params, kb):
        return (f"The ChEBI pH-7.3 microspecies mapping artifact contains corrupted "
                f"target accessions. Using the pinned chebi_pH7_3 mapping evidence, "
                f"repair the artifact and report the exact, complete, sorted set of "
                f"master Rhea reaction identifiers involving the major microspecies "
                f"at pH 7.3 of ChEBI entity {params['chebi']}.")
    return Family("rep_ph73_mapping_repair", "repair", PROFILE_SPARQL, "high",
                  "pH-7.3 microspecies mapping repair",
                  "corruption_seed", INSTANCES_PER_GOAL, 40, 5, 300,
                  propose, answer, goal, "corrupt_ph73")


def _mk_ont_ancestor_closure() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        uni = sorted(kb.chebi_to_reactions)
        pool = [c for c in uni
                if 3 <= len(_ancestors(kb, c)) <= 40 and edge_cuttable(kb, c)]
        for c in even_spread(pool, lambda c: len(_ancestors(kb, c)), 60):
            yield {"chebi": c}
    def answer(kb, params, masters):
        return tuple(sorted(_ancestors(kb, params["chebi"]),
                            key=lambda c: int(c.split(":")[1])))
    def goal(params, kb):
        name = kb.chebi_labels.get(params["chebi"], "")
        return (f"Using only the pinned ChEBI ontology (chebi_lite release 256) "
                f"subclass hierarchy, report the exact, complete, sorted set of "
                f"ancestor ChEBI class identifiers of {params['chebi']} ({name}) under "
                f"transitive rdfs:subClassOf closure (excluding itself).")
    return Family("ont_chebi_ancestor_closure", "application", PROFILE_CLOSURE,
                  "medium", "ChEBI transitive ancestor closure",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "remove_subclass_edge")


def _mk_ont_common_ancestors() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        rxn2ch = kb.reaction_to_chebis
        out = []
        for r in sorted(masters, key=lambda x: int(x.split(":")[1])):
            chs = sorted(rxn2ch.get(r, ()))
            if len(chs) < 2:
                continue
            if not any(edge_cuttable(kb, c) for c in chs):
                continue
            common: set[str] | None = None
            for c in chs:
                anc = set(_ancestors(kb, c))
                common = anc if common is None else common & anc
            common = (common or set())
            if 2 <= len(common) <= 30:
                out.append({"rhea": r, "_n": len(common)})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        chs = sorted(kb.reaction_to_chebis.get(params["rhea"], ()))
        common: set[str] | None = None
        for c in chs:
            anc = set(_ancestors(kb, c))
            common = anc if common is None else common & anc
        return tuple(sorted(common or set(), key=lambda c: int(c.split(":")[1])))
    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of ChEBI class identifiers "
                f"that are common ancestors (transitive rdfs:subClassOf) of ALL "
                f"ChEBI-cross-referenced participants of master Rhea reaction "
                f"{params['rhea']}, using the pinned ChEBI ontology.")
    return Family("ont_chebi_common_ancestors_of_reaction", "application",
                  PROFILE_CLOSURE, "high", "participant common-ancestor intersection",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 30,
                  propose, answer, goal, "remove_subclass_edge")


def _mk_ont_descendants_in_rhea() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        uni = frozenset(kb.chebi_to_reactions)
        counts: dict[str, int] = {}
        for c in uni:
            for a in _ancestors(kb, c):
                counts[a] = counts.get(a, 0) + 1
        pool = [a for a, n in counts.items()
                if 5 <= n <= 200 and edge_cuttable(kb, a)]
        for a in even_spread(pool, lambda a: counts[a], 60):
            yield {"ancestor": a}
    def answer(kb, params, masters):
        uni = frozenset(kb.chebi_to_reactions)
        a = params["ancestor"]
        return tuple(sorted((c for c in uni if a in _ancestors(kb, c)),
                            key=lambda c: int(c.split(":")[1])))
    def goal(params, kb):
        name = kb.chebi_labels.get(params["ancestor"], "")
        return (f"Report the exact, complete, sorted set of ChEBI entity identifiers "
                f"that are (transitive) subclasses of {params['ancestor']} ({name}) "
                f"in the pinned ChEBI ontology AND are cross-referenced by at least "
                f"one Rhea reaction participant in the pinned Rhea release.")
    return Family("ont_chebi_descendants_in_rhea", "organization", PROFILE_CLOSURE,
                  "high", "ontology closure ∩ cross-source universe",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 5, 200,
                  propose, answer, goal, "remove_subclass_edge")


def _mk_ont_dual_engine() -> Family:
    # Same answers as ont_chebi_descendants_in_rhea but the declared profile
    # requires owlrl RDFS materialization; the verifier computes the answer BOTH
    # via SPARQL property paths and via owlrl closure and requires agreement.
    base = _mk_ont_descendants_in_rhea()
    return Family("ont_chebi_closure_dual_engine", "organization", PROFILE_OWLRL,
                  "high", "dual-engine (property-path == owlrl) closure join",
                  base.state_axis, base.instances_per_goal, 40,
                  base.min_answer, base.max_answer,
                  base.propose, base.answer, base.goal_text, "remove_subclass_edge")


def _mk_con_crosswalk_probes() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        rxn2ch = kb.reaction_to_chebis
        pool = _pool_chebi_by_size(kb, masters, 15, 200)
        rng.shuffle(pool)
        out = []
        for a in pool[:300]:
            partners: dict[str, int] = {}
            for r in masters_involving(kb, masters, a):
                for b in rxn2ch.get(r, ()):
                    if b != a:
                        partners[b] = partners.get(b, 0) + 1
            top = sorted(partners, key=partners.get, reverse=True)[:2]
            if len(top) < 2:
                continue
            triple = tuple(sorted([a] + top))
            r2c = _rxn2compounds(kb)
            compounds = set()
            for c in triple:
                for r in masters_involving(kb, masters, c):
                    compounds |= set(r2c.get(r, ()))
            union_n = len(set().union(*(
                masters_involving(kb, masters, c) for c in triple)))
            if 30 <= union_n <= 400 and 20 <= len(compounds) <= 600:
                out.append({"chebi_set": list(triple), "_n": union_n})
        for p in even_spread(out, lambda p: p["_n"], 40):
            yield {k: v for k, v in p.items() if k != "_n"}
    def answer(kb, params, masters):
        out: set[str] = set()
        for c in params["chebi_set"]:
            out |= masters_involving(kb, masters, c)
        return _sorted_ids(frozenset(out))
    def goal(params, kb):
        ents = ", ".join(params["chebi_set"])
        return (f"Only the reaction-structure facts (reaction sides and participant "
                f"compound IRIs, without ChEBI identity links) are public, plus a "
                f"budgeted probe tool that reveals the published ChEBI accession of "
                f"one compound per probe within the stated budget. Construct the "
                f"Rhea-compound-to-ChEBI crosswalk entries you need, then report the "
                f"exact, complete, sorted set of master Rhea reaction identifiers "
                f"involving ANY of the ChEBI entities {{{ents}}}.")
    return Family("con_crosswalk_via_budgeted_probes", "construction", PROFILE_LINKED,
                  "high", "budgeted-probe crosswalk construction",
                  "evidence_sample_seed", INSTANCES_PER_GOAL, 40, 30, 400,
                  propose, answer, goal, "swap_accession")


def _mk_con_profile_assembly() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        rows = [d for d in kb.directions
                if d.get("RHEA_ID_LR") and d.get("RHEA_ID_RL") and d.get("RHEA_ID_BI")]
        rc = _rowcount(kb)
        pool = [d for d in rows
                if 3 <= rc.get("RHEA:" + d["RHEA_ID_MASTER"], 0) <= 10]
        for d in even_spread(pool, lambda d: 1, 40):
            yield {"master": "RHEA:" + d["RHEA_ID_MASTER"]}
    def answer(kb, params, masters):
        m = params["master"]
        rows = []
        d = next(x for x in kb.directions if x["RHEA_ID_MASTER"] == m.split(":")[1])
        for code, col in (("LR", "RHEA_ID_LR"), ("RL", "RHEA_ID_RL"), ("BI", "RHEA_ID_BI")):
            if d.get(col):
                rows.append(f"child|{code}|RHEA:{d[col]}")
        for x in sorted(kb.xrefs_by_master.get(m, ())):
            rows.append(f"xref|{x}")
        for g in sorted(kb.go_by_master.get(m, ())):
            rows.append(f"go|{g}")
        for r, s, p, c, ch in kb.participants:
            if r == m:
                rows.append(f"participant|{s}|{p}|{c}|{ch}")
        return tuple(sorted(rows))
    def goal(params, kb):
        return (f"Only the canonical equation of master reaction {params['master']} is "
                f"public, plus a budgeted probe tool returning the full pinned record "
                f"(children / xrefs / GO terms / participants) of one reaction per "
                f"probe within the stated budget. Assemble and report the exact, "
                f"complete, sorted reaction-profile rows (child|CODE|RHEA:*, "
                f"xref|DB:id, go|GO:*, participant|side|IRI|compound|CHEBI:*) for "
                f"{params['master']}.")
    return Family("con_reaction_profile_assembly", "construction", PROFILE_SPARQL,
                  "high", "budgeted-probe reaction profile assembly",
                  "evidence_sample_seed", INSTANCES_PER_GOAL, 40, 6, 30,
                  propose, answer, goal, "swap_direction_children")


_MASTER_CACHE: dict[int, frozenset[str]] = {}


def _master_cache(kb: L04KB) -> frozenset[str]:
    # Master = reaction with rdfs:subClassOf rh:Reaction in the pinned RDF
    # (kb.master_ids). The directions TSV is a strict subset projection
    # (296 RDF masters have no direction row); the native verifier filters
    # masters by the RDF class, so the fast path must use the same set.
    key = id(kb)
    if key not in _MASTER_CACHE:
        _MASTER_CACHE[key] = kb.master_ids
    return _MASTER_CACHE[key]


# Ship set = 25 of 27 implemented families (L04 checkpoint-0 decision, per
# FINAL doc §1 "substantive difference" rule):
#   - ont_chebi_closure_dual_engine dropped: identical goal structure to
#     ont_chebi_descendants_in_rhea (only the verifier path differed);
#   - rep_crosswalk_single_flip dropped: k=1 difficulty knob of
#     rep_crosswalk_multi_swap, same mechanism.
# The constructors stay in this file; the families are excluded from FAMILIES.
FAMILIES: tuple[Family, ...] = (
    _mk_org_reactions_involving(),
    _mk_org_participant_rows(),
    _mk_org_conjunction(),
    _mk_org_difference(),
    _mk_org_ancestor(),
    _mk_org_go_terms(),
    _mk_org_go_and_chebi(),
    _mk_org_copartners(),
    _mk_org_ec_class(),
    _mk_org_xrefs(),
    _mk_app_children(),
    _mk_app_transport(),
    _mk_app_unbalanced(),
    _mk_app_equations_by_ec(),
    _mk_app_status(),
    _mk_app_side_participants(),
    _mk_rep_crosswalk(3, "rep_crosswalk_multi_swap", "high"),
    _mk_rep_directions(),
    _mk_rep_side_flip(),
    _mk_rep_ph73(),
    _mk_ont_ancestor_closure(),
    _mk_ont_common_ancestors(),
    _mk_ont_descendants_in_rhea(),
    _mk_con_crosswalk_probes(),
    _mk_con_profile_assembly(),
)

FAMILY_BY_ID: dict[str, Family] = {f.family_id: f for f in FAMILIES}

assert len(FAMILIES) == len(FAMILY_BY_ID) == 25, "family ids must be unique"
assert all(f.instances_per_goal == INSTANCES_PER_GOAL for f in FAMILIES)
