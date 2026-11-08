"""L16 pools: shared pair-count indexes for family proposal.

Computed once per KB (cached by object id); families sample from these pools.
All counting is deterministic (sorted iteration, no RNG).
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.source.l16.kb import L16KB

_POOLS: dict[int, dict[str, Any]] = {}

EXPERIMENTAL_GO_CODES = frozenset(
    {"EXP", "IDA", "IPI", "IMP", "IGI", "IEP", "HTP", "HDA", "HMP", "HGI",
     "HEP"})

FEATURE_BOUNDARY_TYPES = (
    "SIGNAL", "TRANSIT", "PROPEP", "CHAIN", "PEPTIDE", "TRANSMEM", "DOMAIN",
    "REPEAT", "MOTIF", "DNA_BIND", "ZN_FING", "CA_BIND", "BINDING", "ACT_SITE",
    "METAL", "REGION", "TOPO_DOM",
)

XREF_DBS = ("PDB", "EMBL", "RefSeq", "Ensembl", "MIM", "Reactome", "GeneID",
            "KEGG", "InterPro", "Pfam", "CCDS", "DrugBank")

GOOD_ECOS = ("ECO:0000269", "ECO:0000250", "ECO:0000255", "ECO:0000305",
             "ECO:0000244", "ECO:0000312")


def pools(kb: L16KB) -> dict[str, Any]:
    key = id(kb)
    if key in _POOLS:
        return _POOLS[key]
    p: dict[str, Any] = {}

    # (taxon, keyword) counts, mid-size taxa only
    tax_kw: dict[tuple[int, str], int] = {}
    for taxon, accs in kb.taxon_members.items():
        if not (30 <= len(accs) <= 60000):
            continue
        for acc in accs:
            for kw in kb.keywords_by_acc.get(acc, ()):
                tax_kw[(taxon, kw)] = tax_kw.get((taxon, kw), 0) + 1
    p["taxon_keyword"] = sorted(tax_kw.items())

    # (taxon, pe) counts
    tax_pe: dict[tuple[int, int], int] = {}
    for taxon, accs in kb.taxon_members.items():
        if len(accs) < 30:
            continue
        for acc in accs:
            pe = kb.entries[acc]["pe_level"]
            tax_pe[(taxon, pe)] = tax_pe.get((taxon, pe), 0) + 1
    p["taxon_pe"] = sorted(tax_pe.items())

    # (taxon, eco) counts over function comments
    tax_eco: dict[tuple[int, str], int] = {}
    for taxon, accs in kb.taxon_members.items():
        if not (30 <= len(accs) <= 60000):
            continue
        for acc in accs:
            for eco in kb.function_ecos.get(acc, ()):
                if eco in GOOD_ECOS:
                    tax_eco[(taxon, eco)] = tax_eco.get((taxon, eco), 0) + 1
    p["taxon_eco"] = sorted(tax_eco.items())

    # (taxon, go) counts (inverted from the per-accession go view)
    tax_go: dict[tuple[int, str], int] = {}
    for acc, gos in kb.go_by_acc.items():
        t = kb.entries[acc]["taxon_id"]
        if not (30 <= len(kb.taxon_members.get(t, ())) <= 60000):
            continue
        for g in {g for g, _a, _e in gos}:
            tax_go[(t, g)] = tax_go.get((t, g), 0) + 1
    p["taxon_go"] = sorted(tax_go.items())

    # (taxon, subcellular term) counts (inverted from subcell_members)
    tax_loc: dict[tuple[int, str], int] = {}
    for term, accs in kb.subcell_members.items():
        for acc in accs:
            t = kb.entries[acc]["taxon_id"]
            if 30 <= len(kb.taxon_members.get(t, ())) <= 60000:
                tax_loc[(t, term)] = tax_loc.get((t, term), 0) + 1
    p["taxon_location"] = sorted(tax_loc.items())

    # ec prefix counts (global)
    ec_prefix: dict[str, int] = {}
    for acc, ecs in kb.ec_by_acc.items():
        prefixes: set[str] = set()
        for e in ecs:
            parts = e.split(".")
            for depth in (2, 3):
                if len(parts) > depth:
                    prefixes.add(".".join(parts[:depth]))
        for pre in prefixes:
            ec_prefix[pre] = ec_prefix.get(pre, 0) + 1
    p["ec_prefix"] = sorted(ec_prefix.items())

    # (taxon, ec-prefix + rhea) pair counts for the construction family
    tax_pairs: dict[int, set[tuple[str, str]]] = {}
    for acc, pairs in kb.cat_pairs_by_acc.items():
        t = kb.entries[acc]["taxon_id"]
        tax_pairs.setdefault(t, set()).update(pairs)
    p["taxon_ec_rhea_pairs"] = sorted(
        (t, sorted(ps)) for t, ps in tax_pairs.items())

    # (accession, db) xref counts
    acc_db: dict[tuple[str, str], int] = {}
    for (acc, dbn), ids in kb.xrefs_by_acc_db.items():
        if dbn in XREF_DBS:
            acc_db[(acc, dbn)] = len(ids)
    p["acc_db_xrefs"] = sorted(acc_db.items())

    _POOLS[key] = p
    return p
