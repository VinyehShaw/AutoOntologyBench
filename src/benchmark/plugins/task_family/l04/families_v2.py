"""L04 v2 task families: 18 NEW mechanisms over pinned Rhea 142 + ChEBI 256
(20K expansion, dataset autoontobench-bank-20000-candidate-dev1).

Every family here is mechanistically disjoint from the 25 legacy families in
``families.py`` (see REPORT.md mechanism matrix). No legacy family is renamed
or re-parameterized: each v2 family computes an answer under a predicate or
artifact kind that no legacy family uses:

  org_masters_isomer_formula_class     molecular-formula equivalence-class join
  org_masters_by_charge_and_chebi      formal-charge predicate join
  org_masters_by_citation_band_and_chebi  citation-count band predicate join
  org_masters_isa_neighborhood         child-level reaction is_a graph walk
  org_masters_by_name_token            published-name lexical predicate join
  org_masters_by_smiles_substring      SMILES lexical (substructure proxy) join
  org_masters_involving_ph73_microspecies  pH-7.3 mapping-resolution join (org)
  org_masters_closure_conjunction      ChEBI-closure-level conjunctive join
  org_masters_multi_ec                 multi-EC cardinality (>=2) predicate
  org_masters_no_ec                    EC-absence (negation) predicate
  org_masters_go_mediated              GO-term-mediated master neighborhood
  org_opposite_side_copartners         side-aware (opposite-only) co-participation
  app_masters_chebi_both_sides         side-crossing self-join (L AND R)
  app_masters_by_participant_cardinality  participant-count band predicate
  con_ancestor_closure_via_probes      ontology closure construction under
                                       budgeted parent probes (construction)
  rep_go_link_table_repair             GO-link table artifact repair
  rep_xref_table_repair                cross-reference table artifact repair
  rep_status_table_repair              curation-status table artifact repair

State axes, instance identity, wording-leak rules and the dual-path SPARQL
verification contract are identical to the legacy lane (families.py /
builder.py docstrings). Instance identity = sha(base_goal, state descriptor);
base goal identity = sha("l04", family, canonical params). The builder
additionally excludes every legacy (8K) canonical-params signature, so no v2
base goal can collide with a legacy base goal even across families.
"""
from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Iterator

from benchmark.plugins.source.l04.kb import L04KB
from benchmark.plugins.task_family.l04.families import (
    INSTANCES_PER_GOAL, PROFILE_CLOSURE, PROFILE_LINKED, PROFILE_SPARQL,
    Family, _ancestors, _c2m, _master_cache, _rowcount, _sorted_ids,
    canonical_params, edge_cuttable, even_spread, masters_involving,
)

#: Bands for the citation-count predicate (citation cardinality of a master).
CITATION_BANDS = ("0-0", "1-1", "2-3", "4-9")
#: Bands for the participant-cardinality predicate (participant rows/master).
CARDINALITY_BANDS = ("2-2", "3-3", "4-5", "6-20")
#: Non-zero formal charges with sufficient coverage (0 is the majority case).
CHARGE_VALUES = ("-4", "-3", "-2", "-1", "1", "2")


def parse_band(band: str) -> tuple[int, int]:
    lo, hi = band.split("-")
    return int(lo), int(hi)


# ---------------------------------------------------------------------------
# v2 sidecar maps (compound formula/charge, relationships, citation counts).
# Cached per KB; loaded from the same pinned index shards the legacy lane uses.
# ---------------------------------------------------------------------------

_COMP_MAPS: dict[int, tuple[dict[str, str], dict[str, str], dict[str, str]]] = {}
_NEIGHBORHOOD: dict[int, dict[str, frozenset[str]]] = {}
_PH73_MAP: dict[int, dict[str, str]] = {}
_BYR_SIDE: dict[int, dict[str, dict[str, frozenset[str]]]] = {}


def _compound_maps(kb: L04KB) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """(compound -> formula, compound -> charge, compound -> chebi) from the
    pinned ``rhea_compound`` index shard."""
    key = id(kb)
    if key not in _COMP_MAPS:
        path = Path(kb.root) / "indexes" / "rhea_compound.jsonl.gz"
        formula: dict[str, str] = {}
        charge: dict[str, str] = {}
        chebi: dict[str, str] = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                formula[row["compound"]] = row.get("formula", "")
                charge[row["compound"]] = row.get("charge", "")
                chebi[row["compound"]] = row["chebi"]
        _COMP_MAPS[key] = (formula, charge, chebi)
    return _COMP_MAPS[key]


def _ph73_map(kb: L04KB) -> dict[str, str]:
    """chebi -> pH-7.3 major microspecies (both sides CHEBI:-prefixed)."""
    key = id(kb)
    if key not in _PH73_MAP:
        out: dict[str, str] = {}
        for row in kb.ph73:
            src = row["CHEBI"]
            if not src.startswith("CHEBI:"):
                src = "CHEBI:" + src
            tgt = row["CHEBI_PH7_3"]
            if not tgt.startswith("CHEBI:"):
                tgt = "CHEBI:" + tgt
            out[src] = tgt
        _PH73_MAP[key] = out
    return _PH73_MAP[key]


def _byr_side(kb: L04KB) -> dict[str, dict[str, frozenset[str]]]:
    """master -> {"L": chebi set, "R": chebi set} over master participant rows."""
    key = id(kb)
    if key not in _BYR_SIDE:
        masters = _master_cache(kb)
        m: dict[str, dict[str, set[str]]] = {}
        for r, s, _p, _c, ch in kb.participants:
            if r in masters:
                m.setdefault(r, {}).setdefault(s, set()).add(ch)
        _BYR_SIDE[key] = {
            r: {s: frozenset(v) for s, v in d.items()} for r, d in m.items()}
    return _BYR_SIDE[key]


def _master_neighborhood(kb: L04KB) -> dict[str, frozenset[str]]:
    """Master -> related masters via one child-level is_a edge (either
    direction) from the pinned rhea-relationships TSV.

    Rhea serializes the same edges in the RDF as reaction-level
    rdfs:subClassOf links between the child reaction IRIs; the SPARQL
    reference path reads the RDF, so fast path (TSV) and native path (RDF)
    are independent serializations of one published mapping.
    """
    key = id(kb)
    if key not in _NEIGHBORHOOD:
        masters = _master_cache(kb)
        child2master: dict[str, str] = {}
        for m in masters:
            u = kb.reactions[m]
            for x in u["children_lr_rl"] + u["children_bi"]:
                child2master[x] = m
        mn: dict[str, set[str]] = {}
        for row in kb.relationships:
            ma = child2master.get(row["FROM_REACTION_ID"])
            mb = child2master.get(row["TO_REACTION_ID"])
            if ma and mb and ma != mb:
                mn.setdefault(ma, set()).add(mb)
                mn.setdefault(mb, set()).add(ma)
        _NEIGHBORHOOD[key] = {k: frozenset(v) for k, v in mn.items()}
    return _NEIGHBORHOOD[key]


def _smiles_map(kb: L04KB) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in kb.smiles.items():
        out[k if k.startswith("CHEBI:") else "CHEBI:" + k] = v
    return out


def _cit_count(kb: L04KB, master: str) -> int:
    return len(kb.reactions[master].get("citations") or ())


def _masters_with_formula(kb: L04KB, formula: str) -> frozenset[str]:
    fmap, _charge, _ch = _compound_maps(kb)
    masters = _master_cache(kb)
    comps = {c for c, f in fmap.items() if f == formula}
    out: set[str] = set()
    for r, _s, _p, c, _ch in kb.participants:
        if c in comps and r in masters:
            out.add(r)
    return frozenset(out)


def _masters_with_charge_and_chebi(kb: L04KB, chebi: str,
                                   charge: str) -> frozenset[str]:
    _fmap, charge_of, _ch = _compound_maps(kb)
    masters = _master_cache(kb)
    rows_by_rxn: dict[str, list[tuple[str, str]]] = {}
    for r, _s, _p, c, ch in kb.participants:
        if r in masters:
            rows_by_rxn.setdefault(r, []).append((c, ch))
    out: set[str] = set()
    for r in masters_involving(kb, masters, chebi):
        rows = rows_by_rxn.get(r, [])
        if any(charge_of.get(c, "") == charge for c, _ch2 in rows):
            out.add(r)
    return frozenset(out)


def _masters_isa_neighborhood(kb: L04KB, chebi: str) -> frozenset[str]:
    masters = _master_cache(kb)
    mn = _master_neighborhood(kb)
    own = set(masters_involving(kb, masters, chebi))
    out: set[str] = set()
    for m in own:
        out |= set(mn.get(m, ()))
    out -= own
    return frozenset(out)


def _masters_go_mediated(kb: L04KB, chebi: str) -> frozenset[str]:
    masters = _master_cache(kb)
    gos: set[str] = set()
    for m in masters_involving(kb, masters, chebi):
        gos |= set(kb.go_by_master.get(m, ()))
    out: set[str] = set()
    for g in gos:
        out |= set(kb.go_to_masters.get(g, ()))
    out &= set(masters)
    return frozenset(out)


def source_unit_ids_v2(params: dict[str, Any]) -> tuple[str, ...]:
    """v2 source-unit anchors (params never name answer members)."""
    ids: list[str] = []
    for key in ("chebi", "ancestor_a", "ancestor_b"):
        if key in params:
            ids.append(f"chebi_entity/{params[key]}")
    if "formula" in params:
        ids.append(f"rhea_compound/formula:{params['formula']}")
    if "charge" in params:
        ids.append(f"rhea_compound/charge:{params['charge']}")
    if "token" in params:
        ids.append(f"chebi_name/token:{params['token']}")
    if "smiles_sub" in params:
        ids.append("rhea_smiles/substring")
    if "band" in params:
        ids.append(f"rhea_reaction/cardinality_band:{params['band']}")
    if "db" in params:
        ids.append(f"rhea_xref/{params['db']}")
    if "status" in params:
        ids.append(f"rhea_reaction/status:{params['status']}")
    return tuple(sorted(set(ids)))


# ---------------------------------------------------------------------------
# family definitions
# ---------------------------------------------------------------------------

def _mk_org_formula_class() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        fmap, _c, _ch = _compound_maps(kb)
        f2c: dict[str, set[str]] = {}
        for comp, f in fmap.items():
            if f:
                f2c.setdefault(f, set()).add(comp)
        comp2masters: dict[str, set[str]] = {}
        masters = _master_cache(kb)
        for r, _s, _p, c, _ch in kb.participants:
            if r in masters:
                comp2masters.setdefault(c, set()).add(r)
        pool = []
        for f, comps in f2c.items():
            if len(comps) < 2:
                continue  # isomer CLASS: formula must be shared by >=2 compounds
            ms: set[str] = set()
            for cpd in comps:
                ms |= comp2masters.get(cpd, set())
            if 20 <= len(ms) <= 400:
                pool.append({"formula": f, "_n": len(ms)})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        return _sorted_ids(_masters_with_formula(kb, params["formula"]))

    def goal(params, kb):
        return (f"Using the pinned Rhea compound records, report the exact, "
                f"complete, sorted set of master Rhea reaction identifiers "
                f"(RHEA:*) having at least one participant whose compound has "
                f"molecular formula \"{params['formula']}\" (rh:formula). The "
                f"formula class groups several distinct compounds; identity "
                f"must come from the published compound records, never from "
                f"names.")
    return Family("org_masters_isomer_formula_class", "organization",
                  PROFILE_SPARQL, "medium",
                  "molecular-formula equivalence-class join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 20, 400,
                  propose, answer, goal, "blank_formula_class")


def _mk_org_charge_chebi() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        _fmap, charge_of, _ch = _compound_maps(kb)
        rows_by_rxn: dict[str, list[tuple[str, str]]] = {}
        for r, _s, _p, c, ch in kb.participants:
            if r in masters:
                rows_by_rxn.setdefault(r, []).append((c, ch))
        # count masters per (chebi, charge-of-any-participant-compound)
        cnt: dict[tuple[str, str], int] = {}
        for r, rows in rows_by_rxn.items():
            chs = {ch for _c, ch in rows}
            qs = {charge_of.get(c, "") for c, _ch in rows}
            for ch in chs:
                for q in qs:
                    if q in CHARGE_VALUES:
                        cnt[(ch, q)] = cnt.get((ch, q), 0) + 1
        pool = [{"chebi": ch, "charge": q, "_n": n}
                for (ch, q), n in cnt.items() if 10 <= n <= 200]
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        return _sorted_ids(_masters_with_charge_and_chebi(
            kb, params["chebi"], params["charge"]))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers that involve ChEBI entity "
                f"{params['chebi']} as a participant AND have at least one "
                f"participant whose compound carries formal charge "
                f"{params['charge']} (rh:charge in the pinned Rhea RDF).")
    return Family("org_masters_by_charge_and_chebi", "organization",
                  PROFILE_SPARQL, "medium",
                  "formal-charge predicate join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 200,
                  propose, answer, goal, "swap_accession")


def _mk_org_citation_band() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            for band in CITATION_BANDS:
                lo, hi = parse_band(band)
                n = sum(1 for m in masters_involving(kb, masters, c)
                        if lo <= _cit_count(kb, m) <= hi)
                if 15 <= n <= 300:
                    pool.append({"chebi": c, "band": band, "_n": n})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        lo, hi = parse_band(params["band"])
        return _sorted_ids(frozenset(
            m for m in masters_involving(kb, masters, params["chebi"])
            if lo <= _cit_count(kb, m) <= hi))

    def goal(params, kb):
        lo, hi = parse_band(params["band"])
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers that involve ChEBI entity "
                f"{params['chebi']} and whose pinned citation count "
                f"(number of rh:citation PubMed links) lies in the band "
                f"[{lo}, {hi}] (inclusive).")
    return Family("org_masters_by_citation_band_and_chebi", "organization",
                  PROFILE_SPARQL, "medium",
                  "citation-count band predicate join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 15, 300,
                  propose, answer, goal, "push_citations_out_of_band")


def _mk_org_isa_neighborhood() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = [{"chebi": c, "_n": len(_masters_isa_neighborhood(kb, c))}
                for c in sorted(_c2m(kb))]
        pool = [p for p in pool if 10 <= p["_n"] <= 300]
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        return _sorted_ids(_masters_isa_neighborhood(kb, params["chebi"]))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers related to the reactions involving "
                f"ChEBI entity {params['chebi']} via the pinned reaction "
                f"is_a hierarchy: a master M is in the answer iff some "
                f"directional/bidirectional child of M is is_a-linked (one "
                f"edge, either direction, per rhea-relationships / the "
                f"reaction-level rdfs:subClassOf links) to a child of a "
                f"master reaction involving {params['chebi']}. Exclude "
                f"master reactions that themselves involve {params['chebi']}.")
    return Family("org_masters_isa_neighborhood", "organization",
                  PROFILE_SPARQL, "high",
                  "child-level reaction is_a graph walk",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 300,
                  propose, answer, goal, "strip_isa_links")


def _mk_org_name_token() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        c2m = _c2m(kb)
        # candidate tokens: lowercase alpha words (len 4..10) appearing in a
        # published name; matching semantics are plain case-insensitive
        # SUBSTRING (aligned with SPARQL CONTAINS(LCASE(?name), tok)). A
        # whole-word occurrence prefilter bounds the per-token full scan.
        names_lc = {ch: kb.names.get(ch, "").lower() for ch in c2m}
        word_hits: dict[str, int] = {}
        for ch, nm in names_lc.items():
            for w in set(re.split(r"[^a-z]+", nm)):
                if 4 <= len(w) <= 10:
                    word_hits[w] = word_hits.get(w, 0) + 1
        cands = sorted(w for w, n in word_hits.items() if n <= 80)
        pool = []
        for tok in cands:
            ents = {ch for ch, nm in names_lc.items() if tok in nm}
            if 5 <= len(ents) <= 80:
                ms: set[str] = set()
                for e in ents:
                    ms |= set(masters_involving(kb, masters, e))
                if 20 <= len(ms) <= 400:
                    pool.append({"token": tok, "_n": len(ms)})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        tok = params["token"]
        out: set[str] = set()
        for e in _c2m(kb):
            if tok in kb.names.get(e, "").lower():
                out |= set(masters_involving(kb, masters, e))
        return _sorted_ids(frozenset(out))

    def goal(params, kb):
        return (f"Using the pinned Rhea-published ChEBI name list "
                f"(chebiId_name.tsv), report the exact, complete, sorted set "
                f"of master Rhea reaction identifiers involving ANY ChEBI "
                f"entity whose published name contains the substring "
                f"\"{params['token']}\" (case-insensitive).")
    return Family("org_masters_by_name_token", "organization", PROFILE_SPARQL,
                  "medium", "published-name lexical predicate join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 20, 400,
                  propose, answer, goal, "rename_name_token")


def _mk_org_smiles_substring() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        c2m = _c2m(kb)
        sm = {k: v for k, v in _smiles_map(kb).items() if k in c2m}
        cand: dict[str, int] = {}
        for k, v in sm.items():
            atoms = re.findall(r"\[[^\]]+\]|Cl|Br|[A-Z][a-z]?|[cnops]|\(|\)"
                               r"|=|#|-|\+|\\|/|[0-9]", v)
            toks: set[str] = set()
            for n in (2, 3, 4, 5):
                for i in range(len(atoms) - n + 1):
                    t = "".join(atoms[i:i + n])
                    if len(t) >= 3 and any(ch.isalpha() for ch in t):
                        toks.add(t)
            for t in toks:
                cand[t] = cand.get(t, 0) + 1
        pool = []
        for t, n_ent in cand.items():
            if 8 <= n_ent <= 100:
                ents = {k for k, v in sm.items() if t in v}
                ms: set[str] = set()
                for e in ents:
                    ms |= set(masters_involving(kb, masters, e))
                if 25 <= len(ms) <= 400:
                    pool.append({"smiles_sub": t, "_n": len(ms)})
        rng.shuffle(pool)
        # spread over matched-master counts, then diversify by first character
        pool = even_spread(pool, lambda p: p["_n"], 240)
        picked: list[dict[str, Any]] = []
        seen_sig: set[str] = set()
        for p in sorted(pool, key=lambda p: (p["smiles_sub"][:2], p["_n"])):
            sig = p["smiles_sub"][:3]
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            picked.append(p)
            if len(picked) >= 60:
                break
        for p in picked:
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        sub = params["smiles_sub"]
        c2m = _c2m(kb)
        sm = _smiles_map(kb)
        out: set[str] = set()
        for e, v in sm.items():
            if e in c2m and sub in v:
                out |= set(masters_involving(kb, masters, e))
        return _sorted_ids(frozenset(out))

    def goal(params, kb):
        return (f"Using the pinned Rhea SMILES mapping (rhea-chebi-smiles.tsv), "
                f"report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers involving ANY ChEBI entity whose "
                f"published SMILES string contains the exact substring "
                f"\"{params['smiles_sub']}\" (case-sensitive).")
    return Family("org_masters_by_smiles_substring", "organization",
                  PROFILE_SPARQL, "high", "SMILES lexical substructure join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 25, 400,
                  propose, answer, goal, "edit_smiles")


def _mk_org_ph73_join() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        ph = _ph73_map(kb)
        c2m = _c2m(kb)
        pool = []
        for src, tgt in sorted(ph.items()):
            if src == tgt:
                continue  # identity mappings carry no resolution step
            n = len(masters_involving(kb, masters, tgt))
            if 15 <= n <= 400:
                pool.append({"chebi": src, "_n": n})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        tgt = _ph73_map(kb)[params["chebi"]]
        return _sorted_ids(masters_involving(kb, masters, tgt))

    def goal(params, kb):
        name = kb.chebi_labels.get(params["chebi"], "")
        return (f"Using the pinned ChEBI pH-7.3 major-microspecies mapping "
                f"(chebi_pH7_3_mapping.tsv), resolve the major microspecies "
                f"at pH 7.3 of ChEBI entity {params['chebi']} ({name}) and "
                f"report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers involving that microspecies entity "
                f"as a participant.")
    return Family("org_masters_involving_ph73_microspecies", "organization",
                  PROFILE_SPARQL, "medium",
                  "pH-7.3 mapping-resolution join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 15, 400,
                  propose, answer, goal, "corrupt_ph73")


def _mk_org_closure_conjunction() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        uni = sorted(kb.chebi_to_reactions)
        counts: dict[str, int] = {}
        for c in uni:
            for a in _ancestors(kb, c):
                counts[a] = counts.get(a, 0) + 1
        pool_anc = [a for a, n in counts.items() if 3 <= n <= 60]
        desc_cache: dict[str, frozenset[str]] = {}

        def desc(a: str) -> frozenset[str]:
            if a not in desc_cache:
                desc_cache[a] = frozenset(
                    c for c in uni if a in _ancestors(kb, c))
            return desc_cache[a]

        mset_cache: dict[str, frozenset[str]] = {}

        def mset(a: str) -> frozenset[str]:
            if a not in mset_cache:
                out: set[str] = set()
                for c in desc(a):
                    out |= set(masters_involving(kb, masters, c))
                mset_cache[a] = frozenset(out)
            return mset_cache[a]

        rng.shuffle(pool_anc)
        pool_anc = pool_anc[:500]  # bound the O(n^2) pair scan
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for i, a in enumerate(pool_anc):
            for b in pool_anc[i + 1:]:
                if (b, a) in seen:
                    continue
                inter = mset(a) & mset(b)
                if 8 <= len(inter) <= 200 and len(mset(a) | mset(b)) <= 1500:
                    seen.add((a, b))
                    out.append({"ancestor_a": a, "ancestor_b": b,
                                "_n": len(inter)})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        uni = sorted(kb.chebi_to_reactions)
        out: set[str] | None = None
        for key in ("ancestor_a", "ancestor_b"):
            a = params[key]
            ms: set[str] = set()
            for c in uni:
                if a in _ancestors(kb, c):
                    ms |= set(masters_involving(kb, masters, c))
            out = ms if out is None else out & ms
        return _sorted_ids(frozenset(out or set()))

    def goal(params, kb):
        na = kb.chebi_labels.get(params["ancestor_a"], "")
        nb = kb.chebi_labels.get(params["ancestor_b"], "")
        return (f"Using the pinned ChEBI subclass hierarchy and the published "
                f"Rhea-to-ChEBI cross-references, report the exact, complete, "
                f"sorted set of master Rhea reaction identifiers that involve "
                f"at least one ChEBI entity that is a (transitive) subclass "
                f"of {params['ancestor_a']} ({na}) AND at least one ChEBI "
                f"entity that is a (transitive) subclass of "
                f"{params['ancestor_b']} ({nb}) as participants.")
    return Family("org_masters_closure_conjunction", "organization",
                  PROFILE_CLOSURE, "high",
                  "ChEBI-closure-level conjunctive join",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 8, 200,
                  propose, answer, goal, "detach_descendants")


def _mk_org_multi_ec() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            n = sum(1 for m in masters_involving(kb, masters, c)
                    if len(kb.ec_by_master.get(m, ())) >= 2)
            if 3 <= n <= 300:
                pool.append({"chebi": c, "_n": n})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        return _sorted_ids(frozenset(
            m for m in masters_involving(kb, masters, params["chebi"])
            if len(kb.ec_by_master.get(m, ())) >= 2))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers that involve ChEBI entity "
                f"{params['chebi']} and are cross-referenced to TWO OR MORE "
                f"distinct EC numbers (rh:ec) in the pinned Rhea release "
                f"(multi-function enzyme reactions).")
    return Family("org_masters_multi_ec", "organization", PROFILE_SPARQL,
                  "medium", "multi-EC cardinality (>=2) predicate",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 300,
                  propose, answer, goal, "strip_extra_ec")


def _mk_org_no_ec() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            n = sum(1 for m in masters_involving(kb, masters, c)
                    if not kb.ec_by_master.get(m))
            if 10 <= n <= 300:
                pool.append({"chebi": c, "_n": n})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        return _sorted_ids(frozenset(
            m for m in masters_involving(kb, masters, params["chebi"])
            if not kb.ec_by_master.get(m)))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers that involve ChEBI entity "
                f"{params['chebi']} and have NO EC number cross-reference "
                f"(no rh:ec link) in the pinned Rhea release.")
    return Family("org_masters_no_ec", "organization", PROFILE_SPARQL,
                  "medium", "EC-absence (negation) predicate",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 300,
                  propose, answer, goal, "add_ec_to_governed")


def _mk_org_go_mediated() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            n = len(_masters_go_mediated(kb, c))
            if 20 <= n <= 600:
                pool.append({"chebi": c, "_n": n})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        return _sorted_ids(_masters_go_mediated(kb, params["chebi"]))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers that share at least one Gene Ontology "
                f"term (GO:*, via the pinned Rhea GO mapping) with at least "
                f"one master reaction involving ChEBI entity "
                f"{params['chebi']}. A master qualifies whenever its GO "
                f"annotation set intersects the GO annotation set of the "
                f"reactions involving {params['chebi']}.")
    return Family("org_masters_go_mediated", "organization", PROFILE_SPARQL,
                  "high", "GO-term-mediated master neighborhood",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 20, 600,
                  propose, answer, goal, "strip_go_of_governed")


def _mk_org_opp_copartners() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        byr = _byr_side(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            partners: set[str] = set()
            for r in masters_involving(kb, masters, c):
                d = byr.get(r, {})
                if c in d.get("L", ()):
                    partners |= set(d.get("R", ()))
                if c in d.get("R", ()):
                    partners |= set(d.get("L", ()))
            partners.discard(c)
            if 10 <= len(partners) <= 300:
                pool.append({"chebi": c, "_n": len(partners)})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        byr = _byr_side(kb)
        partners: set[str] = set()
        for r in masters_involving(kb, masters, params["chebi"]):
            d = byr.get(r, {})
            if params["chebi"] in d.get("L", ()):
                partners |= set(d.get("R", ()))
            if params["chebi"] in d.get("R", ()):
                partners |= set(d.get("L", ()))
        partners.discard(params["chebi"])
        return tuple(sorted(partners, key=lambda c: int(c.split(":")[1])))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of ChEBI entity "
                f"identifiers (CHEBI:*) that appear on the OPPOSITE side of "
                f"the equation from ChEBI entity {params['chebi']} in at "
                f"least one master Rhea reaction (i.e., the entity is on the "
                f"right when {params['chebi']} is on the left, or vice "
                f"versa; same-side co-occurrence does not count). Exclude "
                f"{params['chebi']} itself.")
    return Family("org_opposite_side_copartners", "organization",
                  PROFILE_SPARQL, "medium",
                  "side-aware (opposite-only) co-participation",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 300,
                  propose, answer, goal, "swap_accession")


def _mk_app_both_sides() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        byr = _byr_side(kb)
        cnt: dict[str, int] = {}
        for r, d in byr.items():
            for ch in d.get("L", frozenset()) & d.get("R", frozenset()):
                cnt[ch] = cnt.get(ch, 0) + 1
        pool = [{"chebi": c, "_n": n} for c, n in cnt.items() if 5 <= n <= 400]
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        byr = _byr_side(kb)
        return _sorted_ids(frozenset(
            r for r in masters_involving(kb, masters, params["chebi"])
            if params["chebi"] in byr.get(r, {}).get("L", frozenset())
            and params["chebi"] in byr.get(r, {}).get("R", frozenset())))

    def goal(params, kb):
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers in which ChEBI entity "
                f"{params['chebi']} appears on BOTH sides of the equation "
                f"(at least one left-side and at least one right-side "
                f"participant row) per the pinned Rhea RDF.")
    return Family("app_masters_chebi_both_sides", "application",
                  PROFILE_SPARQL, "medium", "side-crossing self-join (L AND R)",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 5, 400,
                  propose, answer, goal, "swap_accession")


def _mk_app_cardinality() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        rc = _rowcount(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            for band in CARDINALITY_BANDS:
                lo, hi = parse_band(band)
                n = sum(1 for m in masters_involving(kb, masters, c)
                        if lo <= rc.get(m, 0) <= hi)
                if 10 <= n <= 200:
                    pool.append({"chebi": c, "band": band, "_n": n})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        lo, hi = parse_band(params["band"])
        rc = _rowcount(kb)
        return _sorted_ids(frozenset(
            m for m in masters_involving(kb, masters, params["chebi"])
            if lo <= rc.get(m, 0) <= hi))

    def goal(params, kb):
        lo, hi = parse_band(params["band"])
        return (f"Report the exact, complete, sorted set of master Rhea "
                f"reaction identifiers that involve ChEBI entity "
                f"{params['chebi']} and whose total participant-row count "
                f"(left-side plus right-side rh:contains rows in the pinned "
                f"Rhea RDF) lies in the band [{lo}, {hi}] (inclusive).")
    return Family("app_masters_by_participant_cardinality", "application",
                  PROFILE_SPARQL, "medium", "participant-count band predicate",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 200,
                  propose, answer, goal, "strip_participants_of_governed")


def _mk_con_closure_probes() -> Family:
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
        return (f"Only the label of ChEBI entity {params['chebi']} ({name}) "
                f"is public, plus a budgeted probe tool that reveals the "
                f"direct rdfs:subClassOf parents of one ChEBI class per "
                f"probe within the stated budget. Construct the transitive "
                f"ancestor closure yourself and report the exact, complete, "
                f"sorted set of ancestor ChEBI class identifiers of "
                f"{params['chebi']} in the pinned ChEBI ontology (excluding "
                f"{params['chebi']} itself).")
    return Family("con_ancestor_closure_via_probes", "construction",
                  PROFILE_CLOSURE, "high",
                  "ontology closure construction under budgeted parent probes",
                  "evidence_sample_seed", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "remove_subclass_edge")


def _mk_rep_go_link() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            ms = masters_involving(kb, masters, c)
            if not (20 <= len(ms) <= 500):
                continue
            gos: set[str] = set()
            for m in ms:
                gos |= set(kb.go_by_master.get(m, ()))
            # need >=2 GO links each uniquely contributed by one master so a
            # 2-entry corruption provably moves the governed answer
            contrib: dict[str, int] = {}
            for m in ms:
                for g in kb.go_by_master.get(m, ()):
                    contrib[g] = contrib.get(g, 0) + 1
            unique_links = sum(1 for g, n in contrib.items() if n == 1)
            if 3 <= len(gos) <= 60 and unique_links >= 2:
                pool.append({"chebi": c, "_n": len(gos)})
        for p in even_spread(pool, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        gos: set[str] = set()
        for m in masters_involving(kb, masters, params["chebi"]):
            gos |= set(kb.go_by_master.get(m, ()))
        return tuple(sorted(gos))

    def goal(params, kb):
        return (f"The Rhea-to-GO link table artifact for the reactions in "
                f"scope contains 2 corrupted entries (a reaction mapped to a "
                f"wrong GO term). Using the published participant "
                f"cross-references and the budgeted probe tool (which "
                f"reveals the published GO terms of one reaction per "
                f"probe), repair the artifact and report the exact, "
                f"complete, sorted set of GO terms (GO:*) cross-referenced "
                f"from master Rhea reactions involving ChEBI entity "
                f"{params['chebi']} computed through your repaired table.")
    return Family("rep_go_link_table_repair", "repair", PROFILE_SPARQL,
                  "high", "GO-link table artifact repair (2 corrupted links)",
                  "corruption_seed", INSTANCES_PER_GOAL, 60, 3, 60,
                  propose, answer, goal, "remove_go_link_governed")


def _mk_rep_xref_table() -> Family:
    DBS = ("KEGG_REACTION", "METACYC", "ECOCYC", "REACTOME")

    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        out = []
        for c in sorted(_c2m(kb)):
            ms = masters_involving(kb, masters, c)
            if not (10 <= len(ms) <= 500):
                continue
            for db in DBS:
                hits: dict[str, int] = {}
                for m in ms:
                    for x in kb.xrefs_by_master.get(m, ()):
                        if x.startswith(db + ":"):
                            hits[x] = hits.get(x, 0) + 1
                unique = sum(1 for x, n in hits.items() if n == 1)
                if 5 <= len(hits) <= 200 and unique >= 3:
                    out.append({"chebi": c, "db": db, "_n": len(hits)})
        for p in even_spread(out, lambda p: p["_n"], 60):
            yield {k: v for k, v in p.items() if k != "_n"}

    def answer(kb, params, masters):
        ms = masters_involving(kb, masters, params["chebi"])
        db = params["db"] + ":"
        return tuple(sorted({x for m in ms for x in kb.xrefs_by_master.get(m, ())
                             if x.startswith(db)}))

    def goal(params, kb):
        return (f"The Rhea cross-reference table artifact for the reactions "
                f"in scope contains 3 corrupted entries (a cross-reference "
                f"written under a wrong database prefix). Using the "
                f"published participant cross-references and the budgeted "
                f"probe tool (which reveals the published cross-reference "
                f"list of one reaction per probe), repair the artifact and "
                f"report the exact, complete, sorted set of {params['db']} "
                f"cross-reference identifiers (as \"{params['db']}:<id>\") "
                f"attached to master Rhea reactions involving ChEBI entity "
                f"{params['chebi']} computed through your repaired table.")
    return Family("rep_xref_table_repair", "repair", PROFILE_SPARQL,
                  "high", "cross-reference table artifact repair (3 entries)",
                  "corruption_seed", INSTANCES_PER_GOAL, 60, 5, 200,
                  propose, answer, goal, "swap_xref_db_prefix")


def _mk_rep_status_table() -> Family:
    def propose(kb: L04KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        masters = _master_cache(kb)
        out = []
        for status in ("Preliminary", "Obsolete"):
            for c in sorted(_c2m(kb)):
                ms = masters_involving(kb, masters, c)
                with_s = [m for m in ms
                          if kb.reactions[m]["status"] == status]
                without_s = [m for m in ms
                             if kb.reactions[m]["status"] != status]
                # k=2: one false-negative flip (S -> other) + one
                # false-positive flip (other -> S) both move the answer
                if 3 <= len(with_s) <= 50 and len(without_s) >= 1:
                    out.append({"status": status, "chebi": c,
                                "_n": len(with_s)})
        rng.shuffle(out)
        yield from out[:60]

    def answer(kb, params, masters):
        return _sorted_ids(frozenset(
            m for m in masters_involving(kb, masters, params["chebi"])
            if kb.reactions[m]["status"] == params["status"]))

    def goal(params, kb):
        return (f"The curation-status table artifact for the reactions in "
                f"scope contains 2 corrupted entries (reactions assigned a "
                f"wrong curation status). Using the published participant "
                f"cross-references and the budgeted probe tool (which "
                f"reveals the published curation status of one reaction "
                f"per probe), repair the artifact and report the exact, "
                f"complete, sorted set of master Rhea reaction identifiers "
                f"with curation status {params['status']} that involve "
                f"ChEBI entity {params['chebi']} computed through your "
                f"repaired table.")
    return Family("rep_status_table_repair", "repair", PROFILE_SPARQL,
                  "medium", "curation-status table artifact repair (2 entries)",
                  "corruption_seed", INSTANCES_PER_GOAL, 60, 3, 50,
                  propose, answer, goal, "flip_status")


FAMILIES_V2: tuple[Family, ...] = (
    _mk_org_formula_class(),
    _mk_org_charge_chebi(),
    _mk_org_citation_band(),
    _mk_org_isa_neighborhood(),
    _mk_org_name_token(),
    _mk_org_smiles_substring(),
    _mk_org_ph73_join(),
    _mk_org_closure_conjunction(),
    _mk_org_multi_ec(),
    _mk_org_no_ec(),
    _mk_org_go_mediated(),
    _mk_org_opp_copartners(),
    _mk_app_both_sides(),
    _mk_app_cardinality(),
    _mk_con_closure_probes(),
    _mk_rep_go_link(),
    _mk_rep_xref_table(),
    _mk_rep_status_table(),
)

FAMILY_V2_BY_ID: dict[str, Family] = {f.family_id: f for f in FAMILIES_V2}

assert len(FAMILIES_V2) == len(FAMILY_V2_BY_ID) == 18, "family ids must be unique"
assert all(f.instances_per_goal == INSTANCES_PER_GOAL for f in FAMILIES_V2)
assert not (set(FAMILY_V2_BY_ID)
            & {f"org_reactions_involving_chebi", "org_participant_rows_involving_chebi",
               "org_reactions_chebi_conjunction", "org_reactions_chebi_difference",
               "org_reactions_chebi_ancestor", "org_go_terms_of_reactions_involving_chebi",
               "org_reactions_by_go_term_and_chebi", "org_chebi_copartners",
               "org_reactions_by_ec_class_and_chebi", "org_xrefs_of_reactions_involving_chebi",
               "app_directional_children_of_master", "app_transport_reactions_involving_chebi",
               "app_unbalanced_reactions_involving_chebi", "app_reaction_equations_by_ec",
               "app_reactions_by_status_and_chebi", "app_side_participants_of_reaction",
               "rep_crosswalk_multi_swap", "rep_direction_assignment_repair",
               "rep_participant_side_flip_repair", "rep_ph73_mapping_repair",
               "ont_chebi_ancestor_closure", "ont_chebi_common_ancestors_of_reaction",
               "ont_chebi_descendants_in_rhea", "con_crosswalk_via_budgeted_probes",
               "con_reaction_profile_assembly"}), "v2 family ids must not reuse legacy ids"
