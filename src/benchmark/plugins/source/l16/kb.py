"""L16 KB access layer: in-memory derived views over a versioned SQLite store.

Nothing here touches the network. Two stores exist (DAT path A, XML path B);
``load_kb`` defaults to the DAT store (oracle/fast path) and the verifier's
reference path opens the XML store explicitly by filename.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DAT_STORE = "store/l16_uniprot_dat.sqlite"
XML_STORE = "store/l16_uniprot_xml.sqlite"


@dataclass(frozen=True)
class L16KB:
    """Derived views over one pinned L16 store (DAT or XML)."""

    root: Path
    store_name: str
    meta: dict[str, str]
    entries: dict[str, dict[str, Any]]               # accession -> core fields
    taxon_members: dict[int, frozenset[str]]
    gene_members: dict[str, frozenset[str]]          # primary gene -> acc
    keyword_members: dict[str, frozenset[str]]
    ec_members: dict[str, frozenset[str]]            # exact EC -> acc
    go_members: dict[str, frozenset[str]]            # go_id -> acc
    go_evidence: dict[tuple[str, str], frozenset[str]]  # (go_id, acc) -> codes
    rhea_members: dict[str, frozenset[str]]          # rhea_id -> acc
    subcell_members: dict[str, frozenset[str]]       # term -> acc
    features_by_acc: dict[str, tuple[tuple[str, str, str], ...]]
    variants_by_acc: dict[str, tuple[tuple[str, str, str, str, str], ...]]
    isoforms_by_acc: dict[str, tuple[tuple[str, str], ...]]
    xrefs_by_acc_db: dict[tuple[str, str], tuple[str, ...]]
    xrefs_by_acc: dict[str, dict[str, tuple[str, ...]]]  # acc -> db -> ids
    acc_by_xref: dict[tuple[str, str], frozenset[str]]
    pubmed_members: dict[str, frozenset[str]]
    proteome_members: dict[str, frozenset[str]]
    disease_members: dict[str, frozenset[str]]       # disease name -> acc
    function_ecos: dict[str, frozenset[str]]         # acc -> ECO codes
    keywords_by_acc: dict[str, frozenset[str]]
    ec_by_acc: dict[str, frozenset[str]]
    rhea_by_acc: dict[str, frozenset[str]]
    catalytic_ec_members: dict[str, frozenset[str]]  # catalytic EC -> acc
    cat_pairs_by_acc: dict[str, frozenset[tuple[str, str]]]  # acc -> (ec, rhea)
    genes_all_by_acc: dict[str, tuple[str, ...]]
    synonyms_by_acc: dict[str, tuple[str, ...]]
    go_by_acc: dict[str, tuple[tuple[str, str, str], ...]]  # acc -> (go,aspect,evidence)
    subcell_by_acc: dict[str, tuple[tuple[str, str], ...]]  # acc -> (term, ecos_json)
    diseases_by_acc: dict[str, tuple[tuple[str, str, str], ...]]
    pubmeds_by_acc: dict[str, tuple[str, ...]]
    proteomes_by_acc: dict[str, tuple[str, ...]]
    catalytic_by_acc: dict[str, tuple[tuple[str, ...], ...]]  # acc -> (rhea_ids sorted, ecs sorted)

    def entry(self, acc: str) -> dict[str, Any]:
        return self.entries[acc]


def load_kb(root: str | Path, *, store: str = DAT_STORE) -> L16KB:
    root = Path(root)
    cache = root / "store" / (Path(store).stem + ".kb.pkl")
    store_path = root / store
    if cache.is_file() and store_path.is_file():
        st_s, st_c = store_path.stat(), cache.stat()
        if st_c.st_mtime < st_s.st_mtime:
            cache.unlink()  # store rebuilt after caching: stale
        else:
            import pickle
            try:
                kb = pickle.loads(cache.read_bytes())
                if kb.meta.get("source_sha256"):
                    return kb
            except Exception:
                pass
    kb = _load_kb_fresh(root, store=store)
    try:
        import pickle
        cache.write_bytes(pickle.dumps(kb, protocol=4))
    except OSError:
        pass
    return kb


def _load_kb_fresh(root: str | Path, *, store: str = DAT_STORE) -> L16KB:
    root = Path(root)
    db = sqlite3.connect(f"file:{root / store}?mode=ro", uri=True)
    try:
        meta = dict(db.execute("SELECT key, value FROM meta").fetchall())

        entries: dict[str, dict[str, Any]] = {}
        taxon_m: dict[int, set[str]] = {}
        gene_m: dict[str, set[str]] = {}
        genes_all_m: dict[str, tuple[str, ...]] = {}
        syn_m: dict[str, tuple[str, ...]] = {}
        for (acc, name, pe, taxon, gene, prot, seqlen, org, gall, gsyn) in db.execute(
                "SELECT accession, entry_name, pe_level, taxon_id, "
                "gene_primary, protein_name, sequence_length, "
                "organism_scientific, genes_all, gene_synonyms FROM entries"):
            entries[acc] = {
                "accession": acc, "entry_name": name, "pe_level": pe,
                "taxon_id": taxon, "gene_primary": gene,
                "protein_name": prot, "sequence_length": seqlen,
                "organism_scientific": org,
            }
            taxon_m.setdefault(taxon, set()).add(acc)
            if gene:
                gene_m.setdefault(gene, set()).add(acc)
            genes_all_m[acc] = tuple(json.loads(gall))
            syn_m[acc] = tuple(json.loads(gsyn))

        def members(sql: str, key_first: bool) -> dict[Any, frozenset[str]]:
            m: dict[Any, set[str]] = {}
            for a, b in db.execute(sql):
                k, v = (b, a) if key_first else (a, b)
                m.setdefault(k, set()).add(v)
            return {k: frozenset(v) for k, v in m.items()}

        keyword_members = members("SELECT accession, keyword FROM keywords", True)
        ec_members = members("SELECT accession, ec FROM ec", True)
        go_members = members("SELECT accession, go_id FROM go_refs", True)
        rhea_members = members("SELECT accession, rhea_id FROM rhea", True)
        subcell_members = members("SELECT accession, term FROM subcellular", True)
        pubmed_members = members("SELECT accession, pubmed_id FROM pubmeds", True)
        proteome_members = members(
            "SELECT accession, id FROM xrefs WHERE db='Proteomes'", True)
        disease_members = members("SELECT accession, name FROM diseases", True)

        go_evidence: dict[tuple[str, str], set[str]] = {}
        for acc, go_id, ev in db.execute(
                "SELECT accession, go_id, evidence FROM go_refs"):
            code = ev.split(":")[0] if ev else ""
            go_evidence.setdefault((go_id, acc), set()).add(code)
        go_evidence = {k: frozenset(v) for k, v in go_evidence.items()}

        features_by_acc: dict[str, list[tuple[str, str, str]]] = {}
        for acc, ft, b, e in db.execute(
                "SELECT accession, ftype, begin, end FROM features"):
            features_by_acc.setdefault(acc, []).append((ft, b, e))
        variants_by_acc: dict[str, list[tuple[str, str, str, str, str]]] = {}
        for acc, b, e, o, v, vid in db.execute(
                "SELECT accession, begin, end, original, variation, var_id "
                "FROM variants"):
            variants_by_acc.setdefault(acc, []).append((b, e, o, v, vid))
        isoforms_by_acc: dict[str, list[tuple[str, str]]] = {}
        for acc, iid, nm in db.execute(
                "SELECT accession, iso_id, name FROM isoforms"):
            isoforms_by_acc.setdefault(acc, []).append((iid, nm))

        xrefs_by_acc_db: dict[tuple[str, str], list[str]] = {}
        acc_by_xref: dict[tuple[str, str], set[str]] = {}
        xrefs_by_acc: dict[str, dict[str, list[str]]] = {}
        for acc, dbn, rid in db.execute("SELECT accession, db, id FROM xrefs"):
            xrefs_by_acc_db.setdefault((acc, dbn), []).append(rid)
            xrefs_by_acc.setdefault(acc, {}).setdefault(dbn, []).append(rid)
            acc_by_xref.setdefault((dbn, rid), set()).add(acc)

        function_ecos: dict[str, set[str]] = {}
        for acc, ecos in db.execute(
                "SELECT accession, ecos FROM comments_function"):
            function_ecos.setdefault(acc, set()).update(json.loads(ecos))

        keywords_by_acc: dict[str, set[str]] = {}
        for acc, kw in db.execute("SELECT accession, keyword FROM keywords"):
            keywords_by_acc.setdefault(acc, set()).add(kw)
        ec_by_acc: dict[str, set[str]] = {}
        for acc, ecn in db.execute("SELECT accession, ec FROM ec"):
            ec_by_acc.setdefault(acc, set()).add(ecn)
        rhea_by_acc: dict[str, set[str]] = {}
        for acc, rid in db.execute("SELECT accession, rhea_id FROM rhea"):
            rhea_by_acc.setdefault(acc, set()).add(rid)
        catalytic_ec_m: dict[str, set[str]] = {}
        for acc, ecn in db.execute("SELECT accession, ec FROM catalytic_ec"):
            catalytic_ec_m.setdefault(ecn, set()).add(acc)
        cat_pairs: dict[str, set[tuple[str, str]]] = {}
        for acc, ecn, rid in db.execute(
                "SELECT accession, ec, rhea_id FROM catalytic_pairs"):
            cat_pairs.setdefault(acc, set()).add((ecn, rid))

        go_by_acc: dict[str, list[tuple[str, str, str]]] = {}
        for acc, go_id, aspect, ev in db.execute(
                "SELECT accession, go_id, aspect, evidence FROM go_refs"):
            go_by_acc.setdefault(acc, []).append((go_id, aspect, ev))
        subcell_by_acc: dict[str, list[tuple[str, str]]] = {}
        for acc, term, ecos in db.execute(
                "SELECT accession, term, ecos FROM subcellular"):
            subcell_by_acc.setdefault(acc, []).append((term, ecos))
        diseases_by_acc: dict[str, list[tuple[str, str, str]]] = {}
        for acc, nm, acr, mim in db.execute(
                "SELECT accession, name, acronym, mim FROM diseases"):
            diseases_by_acc.setdefault(acc, []).append((nm, acr, mim))
        pubmeds_by_acc: dict[str, list[str]] = {}
        for acc, pmid in db.execute(
                "SELECT accession, pubmed_id FROM pubmeds"):
            pubmeds_by_acc.setdefault(acc, []).append(pmid)
        proteomes_by_acc: dict[str, list[str]] = {}
        for acc, rid in db.execute(
                "SELECT accession, id FROM xrefs WHERE db='Proteomes'"):
            proteomes_by_acc.setdefault(acc, []).append(rid)
        catalytic_by_acc: dict[str, list[tuple[str, ...]]] = {}
        tmp_cat: dict[str, list[tuple[str, str]]] = {}
        for acc, rid in db.execute(
                "SELECT accession, rhea_id FROM catalytic_rhea"):
            tmp_cat.setdefault(acc, []).append((rid, ""))
        for acc, ecn in db.execute(
                "SELECT accession, ec FROM catalytic_ec"):
            tmp_cat.setdefault(acc, []).append(("", ecn))
        for acc, items in tmp_cat.items():
            rheas = tuple(sorted({r for r, _ in items if r}))
            ecs = tuple(sorted({e for _, e in items if e}))
            catalytic_by_acc[acc] = (rheas, ecs)
    finally:
        db.close()

    return L16KB(
        root=root,
        store_name=store,
        meta=meta,
        entries=entries,
        taxon_members={k: frozenset(v) for k, v in taxon_m.items()},
        gene_members={k: frozenset(v) for k, v in gene_m.items()},
        keyword_members=keyword_members,
        ec_members=ec_members,
        go_members=go_members,
        go_evidence=go_evidence,
        rhea_members=rhea_members,
        subcell_members=subcell_members,
        features_by_acc={k: tuple(sorted(v)) for k, v in features_by_acc.items()},
        variants_by_acc={k: tuple(sorted(v)) for k, v in variants_by_acc.items()},
        isoforms_by_acc={k: tuple(sorted(v)) for k, v in isoforms_by_acc.items()},
        xrefs_by_acc_db={k: tuple(sorted(set(v)))
                         for k, v in xrefs_by_acc_db.items()},
        xrefs_by_acc={a: {d: tuple(sorted(set(ids))) for d, ids in m.items()}
                      for a, m in xrefs_by_acc.items()},
        acc_by_xref={k: frozenset(v) for k, v in acc_by_xref.items()},
        pubmed_members=pubmed_members,
        proteome_members=proteome_members,
        disease_members=disease_members,
        function_ecos={k: frozenset(v) for k, v in function_ecos.items()},
        keywords_by_acc={k: frozenset(v) for k, v in keywords_by_acc.items()},
        ec_by_acc={k: frozenset(v) for k, v in ec_by_acc.items()},
        rhea_by_acc={k: frozenset(v) for k, v in rhea_by_acc.items()},
        catalytic_ec_members={k: frozenset(v) for k, v in catalytic_ec_m.items()},
        cat_pairs_by_acc={k: frozenset(v) for k, v in cat_pairs.items()},
        genes_all_by_acc=genes_all_m,
        synonyms_by_acc=syn_m,
        go_by_acc={k: tuple(sorted(set(v))) for k, v in go_by_acc.items()},
        subcell_by_acc={k: tuple(sorted(set(v)))
                        for k, v in subcell_by_acc.items()},
        diseases_by_acc={k: tuple(sorted(set(v)))
                         for k, v in diseases_by_acc.items()},
        pubmeds_by_acc={k: tuple(sorted(set(v)))
                        for k, v in pubmeds_by_acc.items()},
        proteomes_by_acc={k: tuple(sorted(set(v)))
                          for k, v in proteomes_by_acc.items()},
        catalytic_by_acc=catalytic_by_acc,
    )
