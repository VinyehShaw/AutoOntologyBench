"""L16 source census: real counts over the pinned DAT-derived store.

Writes a JSON stats file; the human-facing SOURCE_CENSUS.md is rendered from
it by the lane report step (no fabricated counts — every number here is a SQL
query result over the pinned store).

CLI:
  .venv/bin/python -m benchmark.plugins.source.l16.census \
      --pin-root external_knowledge/pinned/uniprot/uniprot-2026_03-sprot \
      --out <pin-root>/store/census_stats.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path


def _q(db: sqlite3.Connection, sql: str, args: tuple = ()):
    return db.execute(sql, args).fetchall()


def run(pin_root: Path, out_path: Path | None) -> dict:
    t0 = time.time()
    db = sqlite3.connect(
        f"file:{Path(pin_root) / 'store' / 'l16_uniprot_dat.sqlite'}?mode=ro",
        uri=True)
    try:
        stats: dict = {}
        stats["meta"] = dict(_q(db, "SELECT key, value FROM meta"))
        stats["entries"] = _q(db, "SELECT COUNT(*) FROM entries")[0][0]
        stats["reviewed"] = _q(db, "SELECT COUNT(*) FROM entries WHERE reviewed=1")[0][0]
        stats["unreviewed"] = _q(db, "SELECT COUNT(*) FROM entries WHERE reviewed=0")[0][0]
        stats["distinct_primary_genes"] = _q(
            db, "SELECT COUNT(DISTINCT gene_primary) FROM entries "
                "WHERE gene_primary != ''")[0][0]
        stats["distinct_taxa"] = _q(
            db, "SELECT COUNT(DISTINCT taxon_id) FROM entries")[0][0]
        stats["pe_distribution"] = {str(r[0]): r[1] for r in _q(
            db, "SELECT pe_level, COUNT(*) FROM entries GROUP BY pe_level")}
        stats["top_taxa"] = [{"taxon_id": r[0], "organism": r[1], "entries": r[2]}
                             for r in _q(db, "SELECT taxon_id, "
                                             "organism_scientific, COUNT(*) "
                                             "FROM entries GROUP BY taxon_id "
                                             "ORDER BY COUNT(*) DESC LIMIT 15")]
        stats["function_comments"] = _q(
            db, "SELECT COUNT(*) FROM comments_function")[0][0]
        stats["entries_with_function"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM comments_function")[0][0]
        stats["function_eco_distribution"] = {r[0]: r[1] for r in _q(db, """
            SELECT je.value, COUNT(*) FROM comments_function c,
            json_each(c.ecos) je GROUP BY je.value ORDER BY COUNT(*) DESC
            LIMIT 15""")}
        stats["entries_with_ec"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM ec")[0][0]
        stats["distinct_ec"] = _q(db, "SELECT COUNT(DISTINCT ec) FROM ec")[0][0]
        stats["go_crossrefs"] = _q(db, "SELECT COUNT(*) FROM go_refs")[0][0]
        stats["entries_with_go"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM go_refs")[0][0]
        stats["distinct_go_terms"] = _q(
            db, "SELECT COUNT(DISTINCT go_id) FROM go_refs")[0][0]
        stats["go_evidence_distribution"] = {r[0]: r[1] for r in _q(
            db, "SELECT evidence, COUNT(*) FROM go_refs GROUP BY evidence "
                "ORDER BY COUNT(*) DESC LIMIT 15")}
        stats["rhea_crossrefs"] = _q(db, "SELECT COUNT(*) FROM rhea")[0][0]
        stats["distinct_rhea"] = _q(
            db, "SELECT COUNT(DISTINCT rhea_id) FROM rhea")[0][0]
        stats["entries_with_rhea"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM rhea")[0][0]
        stats["features_total"] = _q(db, "SELECT COUNT(*) FROM features")[0][0]
        stats["feature_type_distribution"] = {r[0]: r[1] for r in _q(
            db, "SELECT ftype, COUNT(*) FROM features GROUP BY ftype "
                "ORDER BY COUNT(*) DESC")}
        stats["variants_total"] = _q(db, "SELECT COUNT(*) FROM variants")[0][0]
        stats["variants_with_id"] = _q(
            db, "SELECT COUNT(*) FROM variants WHERE var_id != ''")[0][0]
        stats["entries_with_variants"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM variants")[0][0]
        stats["isoforms_total"] = _q(db, "SELECT COUNT(*) FROM isoforms")[0][0]
        stats["entries_with_isoforms"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM isoforms")[0][0]
        stats["disease_annotations"] = _q(
            db, "SELECT COUNT(*) FROM diseases")[0][0]
        stats["distinct_diseases"] = _q(
            db, "SELECT COUNT(DISTINCT name) FROM diseases")[0][0]
        stats["entries_with_disease"] = _q(
            db, "SELECT COUNT(DISTINCT accession) FROM diseases")[0][0]
        stats["subcellular_annotations"] = _q(
            db, "SELECT COUNT(*) FROM subcellular")[0][0]
        stats["distinct_subcellular_terms"] = _q(
            db, "SELECT COUNT(DISTINCT term) FROM subcellular")[0][0]
        stats["pubmed_links"] = _q(db, "SELECT COUNT(*) FROM pubmeds")[0][0]
        stats["distinct_pubmed"] = _q(
            db, "SELECT COUNT(DISTINCT pubmed_id) FROM pubmeds")[0][0]
        stats["proteome_memberships"] = _q(
            db, "SELECT COUNT(*) FROM xrefs WHERE db='Proteomes'")[0][0]
        stats["distinct_proteomes"] = _q(
            db, "SELECT COUNT(DISTINCT id) FROM xrefs WHERE db='Proteomes'")[0][0]
        stats["xref_db_distribution"] = {r[0]: r[1] for r in _q(
            db, "SELECT db, COUNT(*) FROM xrefs GROUP BY db "
                "ORDER BY COUNT(*) DESC LIMIT 25")}
        stats["wall_seconds"] = round(time.time() - t0, 1)
    finally:
        db.close()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    return stats


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="L16 source census")
    ap.add_argument("--pin-root", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv[1:] if argv and argv[0].endswith(".py") else argv)
    stats = run(Path(args.pin_root), Path(args.out) if args.out else None)
    print(json.dumps(stats, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
