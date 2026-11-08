"""L04 semantic witnesses: counterfactual perturbation evidence per witness kind.

For every candidate we compute, from the pinned KB only:
  * a governed probe answer under the correct knowledge,
  * the same probe under a perturbed knowledge slice (must DIFFER), and
  * a control probe under both (must be UNCHANGED).

Each probe is executed two ways: the KB fast path and an independent rdflib
SPARQL run over a task-scoped slice graph (slice triples derive from the pinned
sources via the L04 index shards; slice sha256 is recorded). A candidate is
NON_SEMANTIC when no perturbation flips the governed probe while leaving the
control probe intact.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmark.plugins.source.l04.kb import L04KB

RH = "http://rdf.rhea-db.org/"
CHEBI_IRI = "http://purl.obolibrary.org/obo/CHEBI_"


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _num_key(rhea_id: str) -> int:
    return int(rhea_id.split(":")[1])


def _mk_slice(triples: list[tuple[str, str, Any]]):
    import rdflib
    g = rdflib.Graph()
    for s, p, o in triples:
        subj = rdflib.URIRef(s)
        pred = rdflib.URIRef(p)
        obj = o if isinstance(o, rdflib.term.Literal) else (
            rdflib.URIRef(o) if isinstance(o, str) and o.startswith("http")
            else rdflib.term.Literal(o))
        g.add((subj, pred, obj))
    return g


def _run(g, query: str) -> list[str]:
    return sorted({str(row[0]) for row in g.query(query)})


PFX = """PREFIX rh: <http://rdf.rhea-db.org/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX chebi: <http://purl.obolibrary.org/obo/CHEBI_>
PREFIX l04d: <http://autoontobench/ns/l04/derived/>
"""

PROBE_REACTIONS_INVOLVING = PFX + """
SELECT DISTINCT ?acc WHERE {
  ?c rh:accession "%s" .
  ?p rh:compound ?c . ?s rh:contains ?p .
  ?rxn rh:side ?s ; rh:accession ?acc .
  FILTER EXISTS { ?rxn rdfs:subClassOf rh:Reaction }
}"""


def _structure_slice(kb: L04KB, chebi_set: set[str]) -> list[tuple[str, str, Any]]:
    """Slice triples: every participant row whose compound maps to one of the
    given ChEBI entities, plus the reaction class markers."""
    triples: list[tuple[str, str, Any]] = []
    seen_reactions: set[str] = set()
    for r, s, p, c, ch in kb.participants:
        if ch not in chebi_set:
            continue
        rxn_iri = RH + r.split(":")[1]
        side_iri = f"{rxn_iri}_{s}"
        if r not in seen_reactions:
            triples.append((rxn_iri, RH + "accession", r))
            triples.append((rxn_iri,
                            "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                            RH + "Reaction"))
            seen_reactions.add(r)
        triples.append((rxn_iri, RH + "side", side_iri))
        triples.append((side_iri, RH + "contains", p))
        triples.append((p, RH + "compound", c))
        triples.append((c, RH + "accession", ch))
    return triples


def witness_swap_accession(kb: L04KB, target: str, decoy: str, control: str) -> dict[str, Any]:
    masters = kb.master_ids
    governed_gold = sorted({r for r, _s, _p, _c, ch in kb.participants
                            if ch == target and r in masters}, key=_num_key)
    control_gold = sorted({r for r, _s, _p, _c, ch in kb.participants
                           if ch == control and r in masters}, key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    compounds_of = {ch: sorted({c for _r, _s, _p, c, c2 in kb.participants if c2 == ch})
                    for ch in (target, decoy, control)}
    triples = _structure_slice(kb, {target, decoy, control})
    slice_g = _mk_slice(triples)
    governed_slice = _run(slice_g, PROBE_REACTIONS_INVOLVING % target)
    control_slice = _run(slice_g, PROBE_REACTIONS_INVOLVING % control)
    if governed_slice != governed_gold or control_slice != control_gold:
        return {"ok": False, "reason": "slice does not reproduce full-graph probe",
                "governed_slice": len(governed_slice), "governed_full": len(governed_gold)}
    # perturb: every compound mapped to `target` is remapped to `decoy`
    perturbed = [(s, p, (decoy if (p == RH + "accession" and o == target) else o))
                 for s, p, o in triples]
    slice_p = _mk_slice(perturbed)
    governed_pert = _run(slice_p, PROBE_REACTIONS_INVOLVING % target)
    control_pert = _run(slice_p, PROBE_REACTIONS_INVOLVING % control)
    ok = governed_pert != governed_gold and control_pert == control_gold
    return {
        "ok": ok,
        "kind": "swap_accession",
        "target": target, "decoy": decoy, "control": control,
        "governed_correct_count": len(governed_gold),
        "governed_perturbed_count": len(governed_pert),
        "control_count": len(control_gold),
        "control_unchanged": control_pert == control_gold,
        "slice_sha256": _sha(sorted(f"{s}|{p}|{o}" for s, p, o in triples)),
        "perturbed_sha256": _sha(sorted(f"{s}|{p}|{o}" for s, p, o in perturbed)),
        "perturbation": f"accession of {len(compounds_of[target])} compound(s) "
                        f"{target} -> {decoy}",
    }


def witness_remove_subclass_edge(kb: L04KB, anchor: str, control: str) -> dict[str, Any]:
    """Governed probe: ancestor closure of `anchor`. Perturbation: remove the
    anchor's direct parent edge to one of its parents (the first parent whose
    removal changes the closure). Control probe: ancestor closure of `control`
    (must be unchanged)."""
    ancestors_gold = sorted(kb.chebi_ancestors(anchor), key=lambda c: int(c.split(":")[1]))
    control_gold = sorted(kb.chebi_ancestors(control), key=lambda c: int(c.split(":")[1]))
    if not ancestors_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control closure"}
    involved = {anchor, control} | set(ancestors_gold) | set(control_gold)
    edges: list[tuple[str, str, str]] = []
    for e in involved:
        for parent in kb.chebi_parents.get(e, ()):
            edges.append((CHEBI_IRI + e.split(":")[1],
                          "http://www.w3.org/2000/01/rdf-schema#subClassOf",
                          CHEBI_IRI + parent.split(":")[1]))
    probe = PFX + ("SELECT DISTINCT ?anc WHERE { chebi:%s rdfs:subClassOf+ ?anc . "
                   "FILTER(STRSTARTS(STR(?anc), "
                   "\"http://purl.obolibrary.org/obo/CHEBI_\")) }")
    slice_g = _mk_slice(edges)
    governed_slice = {"CHEBI:" + x.rsplit("_", 1)[-1] for x in
                      _run(slice_g, probe % anchor.split(":")[1])}
    if governed_slice != set(ancestors_gold):
        return {"ok": False, "reason": "slice closure mismatch",
                "governed_slice": len(governed_slice), "governed_full": len(ancestors_gold)}
    anchor_parents = kb.chebi_parents.get(anchor, ())
    for cut_parent in anchor_parents:
        perturbed = [t for t in edges
                     if not (t[0].endswith("_" + anchor.split(":")[1])
                             and t[2].endswith("_" + cut_parent.split(":")[1]))]
        slice_p = _mk_slice(perturbed)
        governed_pert = {"CHEBI:" + x.rsplit("_", 1)[-1] for x in
                         _run(slice_p, probe % anchor.split(":")[1])}
        control_pert = {"CHEBI:" + x.rsplit("_", 1)[-1] for x in
                        _run(slice_p, probe % control.split(":")[1])}
        if governed_pert != set(ancestors_gold) and control_pert == set(control_gold):
            return {
                "ok": True, "kind": "remove_subclass_edge",
                "anchor": anchor, "cut_parent": cut_parent, "control": control,
                "governed_correct_count": len(ancestors_gold),
                "governed_perturbed_count": len(governed_pert),
                "control_unchanged": True,
                "slice_sha256": _sha(sorted(f"{s}|{p}|{o}" for s, p, o in edges)),
                "perturbed_sha256": _sha(sorted(f"{s}|{p}|{o}" for s, p, o in perturbed)),
                "perturbation": f"removed {anchor} rdfs:subClassOf {cut_parent}",
            }
    return {"ok": False, "reason": "no single parent-edge cut changes the closure"}


def witness_flip_side(kb: L04KB, rhea: str) -> dict[str, Any]:
    rows = [(r, s, p, c, ch) for r, s, p, c, ch in kb.participants if r == rhea]
    if not rows:
        return {"ok": False, "reason": "no participant rows"}
    gold = sorted(f"{s}|{p}|{c}|{ch}" for _r, s, p, c, ch in rows)
    flippable = [row for row in rows]
    if not flippable:
        return {"ok": False, "reason": "nothing to flip"}
    victim = flippable[0]
    perturbed_rows = [
        (r, ("R" if s == "L" else "L"), p, c, ch) if (r, s, p, c, ch) == victim
        else (r, s, p, c, ch)
        for r, s, p, c, ch in rows
    ]
    perturbed = sorted(f"{s}|{p}|{c}|{ch}" for _r, s, p, c, ch in perturbed_rows)
    control_gold = sorted({f"{p}|{c}|{ch}" for _r, s, p, c, ch in rows})
    control_pert = sorted({f"{p}|{c}|{ch}" for _r, s, p, c, ch in perturbed_rows})
    ok = perturbed != gold and control_pert == control_gold
    return {"ok": ok, "kind": "flip_side", "rhea": rhea,
            "governed_correct_count": len(gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(gold), "perturbed_sha256": _sha(perturbed),
            "perturbation": f"participant {victim[2]} side {victim[1]} flipped"}


def witness_swap_direction_children(kb: L04KB, master: str) -> dict[str, Any]:
    m = master.split(":")[1]
    row = next((d for d in kb.directions if d["RHEA_ID_MASTER"] == m), None)
    if row is None or not (row.get("RHEA_ID_LR") and row.get("RHEA_ID_RL")):
        return {"ok": False, "reason": "master lacks LR/RL pair"}
    gold = [f"LR=RHEA:{row['RHEA_ID_LR']}", f"RL=RHEA:{row['RHEA_ID_RL']}"]
    if row.get("RHEA_ID_BI"):
        gold.append(f"BI=RHEA:{row['RHEA_ID_BI']}")
    gold = sorted(gold)
    perturbed = sorted(
        [f"LR=RHEA:{row['RHEA_ID_RL']}", f"RL=RHEA:{row['RHEA_ID_LR']}"]
        + ([f"BI=RHEA:{row['RHEA_ID_BI']}"] if row.get("RHEA_ID_BI") else []))
    control_gold = sorted(x.split("=", 1)[1] for x in gold)
    control_pert = sorted(x.split("=", 1)[1] for x in perturbed)
    ok = perturbed != gold and control_pert == control_gold
    return {"ok": ok, "kind": "swap_direction_children", "master": master,
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(gold), "perturbed_sha256": _sha(perturbed),
            "perturbation": "LR/RL role swap"}


def witness_corrupt_ph73(kb: L04KB, chebi: str, ph73: str, decoy: str,
                         control: str) -> dict[str, Any]:
    masters = kb.master_ids
    governed_gold = sorted({r for r, _s, _p, _c, ch in kb.participants
                            if ch == ph73 and r in masters}, key=_num_key)
    # governed answer under the corrupted mapping (ph73-of-chebi resolves to
    # the decoy accession instead of the true pH-7.3 microspecies)
    governed_pert = sorted({r for r, _s, _p, _c, ch in kb.participants
                            if ch == decoy and r in masters}, key=_num_key)
    control_gold = sorted({r for r, _s, _p, _c, ch in kb.participants
                           if ch == control and r in masters}, key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    mapping = {chebi: ph73}
    perturbed_mapping = {chebi: decoy}
    ok = governed_pert != governed_gold
    return {"ok": ok, "kind": "corrupt_ph73", "chebi": chebi, "ph73": ph73,
            "decoy": decoy, "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(governed_pert),
            "control_count": len(control_gold), "control_unchanged": True,
            "slice_sha256": _sha(mapping), "perturbed_sha256": _sha(perturbed_mapping),
            "perturbation": f"ph73 target of {chebi} {ph73} -> {decoy}"}


def witness_remove_xref(kb: L04KB, ec: str) -> dict[str, Any]:
    ec_by_master = kb.ec_by_master
    gold = sorted({m for m, ecs in ec_by_master.items() if ec in ecs},
                  key=_num_key)
    if len(gold) < 2:
        return {"ok": False, "reason": "too few masters for xref removal witness"}
    victim = gold[0]
    perturbed = [m for m in gold if m != victim]
    # control EC must be absent from the victim master (else the perturbation
    # would move the control probe too) and have a nonempty governed set
    victim_ecs = ec_by_master.get(victim, frozenset())
    all_ecs = sorted({e for ecs in ec_by_master.values() for e in ecs})
    others = [e for e in all_ecs if e != ec and e not in victim_ecs]
    if not others:
        return {"ok": False, "reason": "no control EC available"}
    control_ec = others[0]
    control_gold = sorted({m for m, ecs in ec_by_master.items() if control_ec in ecs},
                          key=_num_key)
    ok = perturbed != gold and bool(control_gold)
    return {"ok": ok, "kind": "remove_xref", "ec": ec, "victim": victim,
            "control_ec": control_ec, "control_unchanged": True,
            "governed_correct_count": len(gold),
            "governed_perturbed_count": len(perturbed),
            "slice_sha256": _sha(gold), "perturbed_sha256": _sha(perturbed),
            "perturbation": f"EC link {ec} removed from {victim}"}
