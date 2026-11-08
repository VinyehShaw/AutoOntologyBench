"""L04 NativeVerifierPlugin: rdflib SPARQL reference verification + witnesses.

Reference answers are recomputed natively against the pinned graphs (G_rhea /
G_chebi parsed directly from the pinned files; G_derived clearly namespaced).
The fast-path (KB) answer and the SPARQL answer must be exactly equal or the
candidate is rejected. ``ont_chebi_closure_dual_engine`` additionally requires
owlrl RDFS-materialization to agree with SPARQL property paths.

Evidence rows carry: family, params, query text + sha256, graph manifest
(graph ids + content hashes + engine version), answer sha256/count, verdict —
every verdict is independently recomputable from the pinned bytes.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import rdflib

from benchmark.plugins.source.l04.kb import L04KB
from benchmark.plugins.verifier.l04 import witness as W
from benchmark.plugins.verifier.l04.graphs import (
    load_chebi_graph, load_derived_graph, load_rhea_graph,
)
from benchmark.plugins.verifier.l04.sparql import (
    build_query, build_set_queries, normalize_rows,
)

_ENGINE = f"rdflib {rdflib.__version__}"

_COMBINED_CACHE: dict[frozenset[str], Any] = {}


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def combined_graph(pin_root, graph_set: str):
    """Union graph for the requested set, parsed per process once.

    Parses the pinned files directly into one Graph (no graph-to-graph copy).
    """
    names = frozenset(graph_set.split("+"))
    if names in _COMBINED_CACHE:
        return _COMBINED_CACHE[names]
    g = rdflib.Graph()
    if "rhea" in names:
        src = load_rhea_graph(pin_root)
        for t in src:
            g.add(t)
    if "chebi" in names:
        src = load_chebi_graph(pin_root)
        for t in src:
            g.add(t)
    if "derived" in names:
        src = load_derived_graph(pin_root)
        for t in src:
            g.add(t)
    _COMBINED_CACHE[names] = g
    return g


_STRUCT_CACHE: dict[str, tuple[dict[str, tuple[str, ...]], dict[str, frozenset[str]]]] = {}


def chebi_structures_rdflib(pin_root) -> tuple[dict[str, tuple[str, ...]],
                                               dict[str, frozenset[str]]]:
    """(parents, ancestor-closure index) computed by BFS over the rdflib-parsed
    pinned ChEBI graph — the independent second path (the KB fast path parses
    the same bytes with stdlib ElementTree). Once per worker process."""
    key = str(pin_root)
    if key in _STRUCT_CACHE:
        return _STRUCT_CACHE[key]
    g = load_chebi_graph(pin_root)
    RDFS_SC = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    parents: dict[str, list[str]] = {}
    for s, o in g.subject_objects(RDFS_SC):
        ss, oo = str(s), str(o)
        if ss.startswith("http://purl.obolibrary.org/obo/CHEBI_") and \
           oo.startswith("http://purl.obolibrary.org/obo/CHEBI_"):
            parents.setdefault("CHEBI:" + ss.rsplit("_", 1)[-1], []).append(
                "CHEBI:" + oo.rsplit("_", 1)[-1])
    parents_t = {k: tuple(sorted(v)) for k, v in parents.items()}
    index: dict[str, frozenset[str]] = {}

    def ancestors_of(e: str) -> frozenset[str]:
        if e in index:
            return index[e]
        seen: set[str] = set()
        stack = list(parents_t.get(e, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(parents_t.get(node, ()))
        index[e] = frozenset(seen)
        return index[e]

    for e in parents_t:
        ancestors_of(e)
    _STRUCT_CACHE[key] = (parents_t, index)
    return _STRUCT_CACHE[key]


def _common_ancestors_query(classes: list[str]) -> str:
    lines = "\n".join(
        f"  chebi:{c.split(':')[1]} rdfs:subClassOf+ ?anc ." for c in classes)
    return (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>\n"
        "SELECT DISTINCT ?anc WHERE {\n" + lines + "\n"
        "  FILTER(STRSTARTS(STR(?anc), \"http://purl.obolibrary.org/obo/CHEBI_\"))\n}")


CLOSURE_FAMILIES = {
    "org_reactions_chebi_ancestor", "ont_chebi_ancestor_closure",
    "ont_chebi_common_ancestors_of_reaction", "ont_chebi_descendants_in_rhea",
    "ont_chebi_closure_dual_engine",
}


def _slice_graph(edges: list[tuple[str, str]]):
    g = rdflib.Graph()
    RDFS_SC = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    for child, parent in edges:
        g.add((rdflib.URIRef("http://purl.obolibrary.org/obo/CHEBI_"
                             + child.split(":")[1]),
               RDFS_SC,
               rdflib.URIRef("http://purl.obolibrary.org/obo/CHEBI_"
                             + parent.split(":")[1])))
    return g


def _verify_closure(pin_root, family_id: str, params: dict[str, Any],
                    expected: tuple[str, ...], kb: L04KB) -> dict[str, Any]:
    """Dual-path closure verification: rdflib-parsed pinned ChEBI (path B)
    recomputes the closure; an exact slice SPARQL run supplies the recorded
    SPARQL evidence. Any disagreement with the ElementTree-derived fast path
    (path A) rejects the candidate."""
    parents_b, index_b = chebi_structures_rdflib(pin_root)
    universe = sorted(kb.chebi_to_reactions)
    dual_path_agree = True
    notes: dict[str, Any] = {}

    if family_id == "org_reactions_chebi_ancestor":
        anchor = params["ancestor"]
        desc_b = sorted(c for c in universe if anchor in index_b.get(c, ()))
        desc_a = sorted(c for c in universe if anchor in kb.chebi_ancestors(c))
        dual_path_agree = desc_a == desc_b
        values = " ".join("chebi:" + c.split(":")[1] for c in desc_b)
        query = (
            "PREFIX rh: <http://rdf.rhea-db.org/>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>\n"
            "SELECT DISTINCT ?acc WHERE {\n"
            f"  VALUES ?class {{ {values} }}\n"
            "  ?c rh:chebi ?class . ?p rh:compound ?c . ?s rh:contains ?p .\n"
            "  ?rxn rh:side ?s ; rh:accession ?acc .\n"
            "  FILTER EXISTS { ?rxn rdfs:subClassOf rh:Reaction }\n}")
        g = combined_graph(pin_root, "rhea")
        got = tuple(sorted({str(r[0]) for r in g.query(query)}))
        notes["descendants_in_universe"] = len(desc_b)
        evidence_match = set(got) == set(expected)

    elif family_id == "ont_chebi_ancestor_closure":
        anchor = params["chebi"]
        anc_b = sorted(index_b.get(anchor, ()), key=lambda c: int(c.split(":")[1]))
        anc_a = sorted(kb.chebi_ancestors(anchor), key=lambda c: int(c.split(":")[1]))
        dual_path_agree = anc_a == anc_b
        chain_nodes = {anchor} | set(anc_b)
        edges = [(e, p) for e in chain_nodes for p in parents_b.get(e, ())
                 if p in chain_nodes]
        slice_g = _slice_graph(edges)
        query = _ancestor_probe(anchor)
        got = tuple(sorted("CHEBI:" + str(r[0]).rsplit("_", 1)[-1]
                           for r in slice_g.query(query)))
        got = tuple(sorted(got, key=lambda c: int(c.split(":")[1])))
        evidence_match = got == tuple(anc_b) == tuple(anc_a)
        notes["slice_triples"] = len(slice_g)

    elif family_id == "ont_chebi_common_ancestors_of_reaction":
        classes = sorted(kb.reaction_to_chebis.get(params["rhea"], ()))
        inter_b: set[str] | None = None
        for c in classes:
            anc = set(index_b.get(c, ()))
            inter_b = anc if inter_b is None else inter_b & anc
        got_b = sorted(inter_b or set(), key=lambda c: int(c.split(":")[1]))
        dual_path_agree = got_b == list(expected)
        chain_nodes = set(classes) | set(got_b)
        for c in classes:
            chain_nodes |= set(index_b.get(c, ()))
        edges = [(e, p) for e in sorted(chain_nodes) for p in parents_b.get(e, ())
                 if p in chain_nodes]
        slice_g = _slice_graph(edges)
        query = _common_ancestors_query(classes)
        got = tuple(sorted(("CHEBI:" + str(r[0]).rsplit("_", 1)[-1]
                            for r in slice_g.query(query)),
                           key=lambda c: int(c.split(":")[1])))
        evidence_match = got == tuple(got_b)
        notes["participant_classes"] = len(classes)
        notes["slice_triples"] = len(slice_g)

    else:  # ont_chebi_descendants_in_rhea / ont_chebi_closure_dual_engine
        anchor = params["ancestor"]
        desc_b = sorted((c for c in universe if anchor in index_b.get(c, ())),
                        key=lambda c: int(c.split(":")[1]))
        desc_a = sorted((c for c in universe if anchor in kb.chebi_ancestors(c)),
                        key=lambda c: int(c.split(":")[1]))
        dual_path_agree = desc_a == desc_b
        chain_nodes = {anchor} | set(desc_b)
        for c in desc_b:
            chain_nodes |= set(index_b.get(c, ()))
        edges = [(e, p) for e in sorted(chain_nodes) for p in parents_b.get(e, ())
                 if p in chain_nodes]
        slice_g = _slice_graph(edges)
        values = " ".join("chebi:" + c.split(":")[1] for c in universe)
        query = (
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>\n"
            "SELECT DISTINCT ?class WHERE {\n"
            f"  ?class rdfs:subClassOf+ chebi:{anchor.split(':')[1]} .\n"
            f"  VALUES ?class {{ {values} }}\n}}")
        got = tuple(sorted(("CHEBI:" + str(r[0]).rsplit("_", 1)[-1]
                            for r in slice_g.query(query)),
                           key=lambda c: int(c.split(":")[1])))
        evidence_match = got == tuple(desc_b)
        notes["slice_triples"] = len(slice_g)

    ok = dual_path_agree and evidence_match
    return {
        "verdict": "PASS" if ok else "FAIL",
        "matches_fast_path": evidence_match,
        "dual_path_closure_agree": dual_path_agree,
        "graph_set": "chebi(+rhea)",
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "query_text": query,
        "expected_sha256": _sha(sorted(set(expected))),
        "engine": _ENGINE,
        **notes,
    }


def _ancestor_probe(anchor: str) -> str:
    return (
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
        "PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>\n"
        "SELECT DISTINCT ?anc WHERE {\n"
        f"  chebi:{anchor.split(':')[1]} rdfs:subClassOf+ ?anc .\n"
        "  FILTER(STRSTARTS(STR(?anc), \"http://purl.obolibrary.org/obo/CHEBI_\"))\n}")


def _graph_meta_snapshot() -> dict[str, Any]:
    from benchmark.plugins.verifier.l04.graphs import _GRAPH_META
    return {gid: dict(meta) for gid, meta in _GRAPH_META.items()}


def _graph_manifest_sha() -> str:
    return _sha(_graph_meta_snapshot())


def verify_reference(pin_root, family_id: str, params: dict[str, Any],
                     expected: tuple[str, ...], kb: L04KB) -> dict[str, Any]:
    """Independent SPARQL recomputation of the fast-path answer."""
    if family_id in CLOSURE_FAMILIES:
        ev = _verify_closure(pin_root, family_id, params, expected, kb)
        ev["graph_manifest_sha"] = _graph_manifest_sha()
        return ev
    if family_id in ("org_reactions_chebi_conjunction",
                     "org_reactions_chebi_difference"):
        graph_set, queries, op = build_set_queries(family_id, params)
        g = combined_graph(pin_root, graph_set)
        sets = [set(normalize_rows("acc", g.query(q))) for q in queries]
        got = sets[0] & sets[1] if op == "intersection" else sets[0] - sets[1]
        match = got == set(expected)
        return {
            "verdict": "PASS" if match else "FAIL",
            "matches_fast_path": match,
            "graph_set": graph_set,
            "graph_triples": len(g),
            "graphs": _graph_meta_snapshot(),
            "graph_manifest_sha": _graph_manifest_sha(),
            "set_operation": op,
            "query_sha256": _sha(queries),
            "query_text": "\n---\n".join(queries),
            "operand_counts": [len(s) for s in sets],
            "answer_sha256": _sha(sorted(got)),
            "answer_count": len(got),
            "expected_sha256": _sha(sorted(set(expected))),
            "engine": _ENGINE,
        }
    graph_set, query, norm = build_query(family_id, params)
    if graph_set.startswith("artifact+"):
        # repair families verify against gold-artifact graph merged with G_rhea;
        # reference path here uses the GOLD mapping (correct by construction).
        graph_set = graph_set.split("+", 1)[1]
    g = combined_graph(pin_root, graph_set)
    results = g.query(query)
    got = normalize_rows(norm, results)
    match = set(got) == set(expected)
    return {
        "verdict": "PASS" if match else "FAIL",
        "matches_fast_path": match,
        "graph_set": graph_set,
        "graph_triples": len(g),
        "graphs": _graph_meta_snapshot(),
        "graph_manifest_sha": _graph_manifest_sha(),
        "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
        "query_text": query,
        "answer_sha256": _sha(sorted(set(got))),
        "answer_count": len(set(got)),
        "expected_sha256": _sha(sorted(set(expected))),
        "engine": _ENGINE,
    }


def owlrl_dual_check(pin_root, params: dict[str, Any],
                     expected: tuple[str, ...], kb: L04KB) -> dict[str, Any]:
    """owlrl RDFS materialization must agree with the property-path answer.

    Runs on the task slice: all subclass edges among (Rhea-universe descendants
    of the anchor) union (their ancestor chains). Transitivity over this slice
    is exact: every path contributing to the closure is edge-complete inside
    the slice by construction.
    """
    import owlrl
    anchor = params["ancestor"]
    universe = sorted(kb.chebi_to_reactions)
    descendants = [c for c in universe if anchor in kb.chebi_ancestors(c)]
    nodes = set(descendants) | {anchor}
    for c in descendants:
        nodes |= set(kb.chebi_ancestors(c))
    g = rdflib.Graph()
    RDFS_SC = rdflib.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    for e in nodes:
        for parent in kb.chebi_parents.get(e, ()):
            if parent in nodes:
                g.add((rdflib.URIRef("http://purl.obolibrary.org/obo/CHEBI_"
                                     + e.split(":")[1]),
                       RDFS_SC,
                       rdflib.URIRef("http://purl.obolibrary.org/obo/CHEBI_"
                                     + parent.split(":")[1])))
    n_before = len(g)
    owlrl.DeductiveClosure(owlrl.RDFS_Semantics).expand(g)
    anchor_ref = rdflib.URIRef(
        "http://purl.obolibrary.org/obo/CHEBI_" + anchor.split(":")[1])
    inferred = {
        "CHEBI:" + str(s).rsplit("_", 1)[-1]
        for s in g.subjects(RDFS_SC, anchor_ref)
    }
    inferred_desc = sorted(c for c in inferred if c in set(universe))
    match = inferred_desc == sorted(set(expected))
    return {
        "verdict": "PASS" if match else "FAIL",
        "engine": f"owlrl {owlrl.__version__ if hasattr(owlrl, '__version__') else '7.6.2'} (RDFS_Semantics)",
        "slice_triples_before": n_before,
        "slice_triples_after": len(g),
        "inferred_descendants_in_universe": len(inferred_desc),
        "matches_property_path": match,
    }


_MASTER_PARTICIPANT_COUNT: dict[int, dict[str, int]] = {}


def _master_participant_count(kb: L04KB) -> dict[str, int]:
    """ChEBI -> number of master-reaction participant rows, cached per KB."""
    key = id(kb)
    if key not in _MASTER_PARTICIPANT_COUNT:
        masters = kb.master_ids
        counts: dict[str, int] = {}
        for r, _s, _p, _c, ch in kb.participants:
            if r in masters:
                counts[ch] = counts.get(ch, 0) + 1
        _MASTER_PARTICIPANT_COUNT[key] = counts
    return _MASTER_PARTICIPANT_COUNT[key]


def _stable_seed(family_id: str, seed: int) -> int:
    fam = int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:4], "big")
    return ((seed * 1_000_003) ^ fam) & 0xFFFFFFFF


def certify_witness(kb: L04KB, family_id: str, witness_kind: str,
                    params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Dispatch the counterfactual witness for one candidate instance."""
    import random
    rng = random.Random(_stable_seed(family_id, seed))
    universe = sorted(kb.chebi_to_reactions)

    def pick_control(exclude: set[str], lo: int = 5, hi: int = 400) -> str | None:
        counts = _master_participant_count(kb)
        cands = [c for c in universe
                 if c not in exclude and lo <= counts.get(c, 0) <= hi]
        return rng.choice(cands) if cands else None

    if witness_kind == "swap_accession":
        target = params.get("chebi") or params.get("chebi_a") or (
            sorted(params["chebi_set"])[0] if "chebi_set" in params else None)
        if target is None:
            return {"ok": False, "reason": "no target entity in params"}
        decoy = params.get("chebi_b") or pick_control({target})
        control = pick_control({target, decoy} if decoy else {target})
        if decoy is None or control is None:
            return {"ok": False, "reason": "no decoy/control available"}
        return W.witness_swap_accession(kb, target, decoy, control)

    if witness_kind == "remove_subclass_edge":
        anchor = params.get("chebi") or params.get("ancestor")
        if anchor is None and "rhea" in params:
            # common-ancestors family: anchor on a participant class with a
            # real single-edge cut (prefiltered at propose time)
            from benchmark.plugins.task_family.l04.families import edge_cuttable
            for c in sorted(kb.reaction_to_chebis.get(params["rhea"], ())):
                if edge_cuttable(kb, c):
                    anchor = c
                    break
        if anchor is None:
            return {"ok": False, "reason": "no anchor entity in params"}
        for _ in range(8):
            control = rng.choice(universe)
            if control != anchor and kb.chebi_ancestors(control):
                w = W.witness_remove_subclass_edge(kb, anchor, control)
                if w["ok"]:
                    return w
        return {"ok": False, "reason": "no working subclass-edge cut found"}

    if witness_kind == "flip_side":
        return W.witness_flip_side(kb, params["rhea"])

    if witness_kind == "swap_direction_children":
        return W.witness_swap_direction_children(kb, params["master"])

    if witness_kind == "corrupt_ph73":
        for _ in range(6):
            decoy = pick_control({params["chebi"], params["ph73"]})
            control = pick_control({params["chebi"], params["ph73"], decoy})
            if decoy is None or control is None:
                return {"ok": False, "reason": "no decoy for ph73 witness"}
            w = W.witness_corrupt_ph73(kb, params["chebi"], params["ph73"],
                                       decoy, control)
            if w.get("ok"):
                return w
        return {"ok": False, "reason": "no decoy flips the governed probe"}

    if witness_kind == "remove_xref":
        return W.witness_remove_xref(kb, params["ec"])

    if witness_kind == "crosswalk_repair":
        # the repair witness IS the swap_accession counterfactual: corrupted
        # crosswalk (target -> decoy) must flip the governed downstream answer
        # while a control entity's answer stays fixed.
        target = params["chebi"]
        decoy = pick_control({target})
        control = pick_control({target, decoy} if decoy else {target})
        if decoy is None or control is None:
            return {"ok": False, "reason": "no decoy/control available"}
        return W.witness_swap_accession(kb, target, decoy, control)

    return {"ok": False, "reason": f"unknown witness kind {witness_kind}"}


# ---------------------------------------------------------------------------
# §3.2 NativeVerifierPlugin facade
# ---------------------------------------------------------------------------

from pathlib import Path

from benchmark.schema.plugin_protocols import (
    GroundedTask, NativeVerdict, ProofOrCounterexample, SemanticWitness,
)

DEFAULT_PIN_ROOT = (Path(__file__).resolve().parents[4]
                    / "external_knowledge" / "pinned" / "rhea-chebi"
                    / "rhea-142+chebi-256")


def _family_params(task) -> tuple[str, dict[str, Any], tuple[str, ...], int]:
    ast = task.ast if isinstance(task, GroundedTask) else task
    fam_id = ast.family_id.split(".", 1)[-1]
    payload = ast.ast
    return (fam_id, dict(payload["params"]),
            tuple(payload.get("answer", ())), int(payload.get("seed", 0)))


class L04NativeVerifier:
    """FINAL-BUILD-ORDER-v2 §3.2 facade over the L04 rdflib SPARQL stack.

    ``prove_exists`` recomputes the reference answer from the pinned graphs and
    requires exact agreement with the fast-path answer carried by the AST;
    ``verify_terminal`` is the exact-set terminal check; and
    ``certify_semantic_relevance`` reruns the governed counterfactual witness.
    """

    lane_id = "L04"

    def __init__(self, pin_root: str | Path | None = None):
        self.pin_root = Path(pin_root) if pin_root else DEFAULT_PIN_ROOT
        self._kb: L04KB | None = None

    @property
    def kb(self) -> L04KB:
        if self._kb is None:
            from benchmark.plugins.source.l04.kb import load_kb
            self._kb = load_kb(str(self.pin_root))
        return self._kb

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        from benchmark.plugins.task_family.l04.families import FAMILY_BY_ID
        fam_id, params, expected, _seed = _family_params(grounded_task)
        fam = FAMILY_BY_ID.get(fam_id)
        if fam is None:
            return ProofOrCounterexample(False, reason=f"unknown family {fam_id}")
        if not expected:
            from benchmark.plugins.task_family.l04.families import _master_cache
            expected = tuple(sorted(set(fam.answer(self.kb, params,
                                                   _master_cache(self.kb)))))
        ev = verify_reference(self.pin_root, fam_id, params, expected, self.kb)
        ok = ev.get("verdict") == "PASS" and len(set(expected)) > 0
        evidence = {k: v for k, v in ev.items() if k != "query_text"}
        evidence["answer_count"] = len(set(expected))
        return ProofOrCounterexample(ok, evidence=evidence,
                                     reason="" if ok else ev.get("reason", ""))

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any
                        ) -> NativeVerdict:
        fam_id, _params, expected, _seed = _family_params(task)
        answer = submitted_state_or_answer
        if isinstance(answer, dict):
            answer = answer.get("answer")
        if not (isinstance(answer, (list, tuple))
                and all(isinstance(x, str) for x in answer)):
            return NativeVerdict(False, {"reason": "submitted answer must be list[str]"})
        got, want = set(answer), set(expected)
        return NativeVerdict(got == want, {
            "family": fam_id,
            "expected_count": len(want), "submitted_count": len(got),
            "missing_sample": sorted(want - got)[:5],
            "extra_sample": sorted(got - want)[:5],
            "expected_sha256": _sha(sorted(want)),
            "submitted_sha256": _sha(sorted(got)),
            "engine": _ENGINE,
        })

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        from benchmark.plugins.task_family.l04.families import FAMILY_BY_ID
        fam_id, params, _expected, seed = _family_params(task)
        fam = FAMILY_BY_ID.get(fam_id)
        if fam is None:
            return SemanticWitness(False, {"reason": f"unknown family {fam_id}"})
        wit = certify_witness(self.kb, fam_id, fam.witness_kind, params, seed)
        return SemanticWitness(bool(wit.get("ok")), witness=wit)
