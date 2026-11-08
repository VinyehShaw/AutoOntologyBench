"""L16 versioned local store: SQLite build from canonical record streams.

Two stores are built from the same pinned snapshot by two independent parsers:
  * ``l16_uniprot_dat.sqlite`` — path A (flat file) — oracle/fast path
  * ``l16_uniprot_xml.sqlite`` — path B (XML) — independent reference path

Both carry a ``meta`` table with the release id, source file sha256 and parser
identity so every downstream answer is attributable to pinned bytes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Iterator

SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE entries(
  accession TEXT PRIMARY KEY,
  entry_name TEXT NOT NULL,
  reviewed INTEGER NOT NULL,
  entry_version INTEGER,
  sequence_version INTEGER,
  sequence_length INTEGER,
  sequence_crc64 TEXT,
  protein_name TEXT,
  gene_primary TEXT,
  genes_all TEXT,
  gene_synonyms TEXT,
  organism_scientific TEXT,
  organism_common TEXT,
  taxon_id INTEGER,
  lineage TEXT,
  pe_level INTEGER
);
CREATE TABLE keywords(accession TEXT NOT NULL, keyword TEXT NOT NULL);
CREATE TABLE ec(accession TEXT NOT NULL, ec TEXT NOT NULL);
CREATE TABLE go_refs(accession TEXT NOT NULL, go_id TEXT NOT NULL,
                     aspect TEXT, term TEXT, evidence TEXT);
CREATE TABLE rhea(accession TEXT NOT NULL, rhea_id TEXT NOT NULL);
CREATE TABLE xrefs(accession TEXT NOT NULL, db TEXT NOT NULL, id TEXT NOT NULL);
CREATE TABLE features(accession TEXT NOT NULL, ftype TEXT NOT NULL,
                      begin TEXT, end TEXT);
CREATE TABLE variants(accession TEXT NOT NULL, begin TEXT, end TEXT,
                      original TEXT, variation TEXT, var_id TEXT);
CREATE TABLE comments_function(accession TEXT NOT NULL, text TEXT,
                               ecos TEXT, pubmeds TEXT);
CREATE TABLE subcellular(accession TEXT NOT NULL, scope TEXT, term TEXT,
                         ecos TEXT);
CREATE TABLE diseases(accession TEXT NOT NULL, name TEXT, acronym TEXT,
                      mim TEXT);
CREATE TABLE isoforms(accession TEXT NOT NULL, iso_id TEXT, name TEXT,
                      synonyms TEXT);
CREATE TABLE pubmeds(accession TEXT NOT NULL, pubmed_id TEXT);
CREATE TABLE catalytic_rhea(accession TEXT NOT NULL, rhea_id TEXT NOT NULL);
CREATE TABLE catalytic_ec(accession TEXT NOT NULL, ec TEXT NOT NULL);
CREATE TABLE catalytic_pairs(accession TEXT NOT NULL, ec TEXT NOT NULL,
                             rhea_id TEXT NOT NULL);
"""

INDEXES = """
CREATE INDEX idx_entries_taxon ON entries(taxon_id);
CREATE INDEX idx_entries_pe ON entries(pe_level);
CREATE INDEX idx_entries_gene ON entries(gene_primary);
CREATE INDEX idx_keywords_acc ON keywords(accession);
CREATE INDEX idx_keywords_kw ON keywords(keyword);
CREATE INDEX idx_ec_acc ON ec(accession);
CREATE INDEX idx_ec_ec ON ec(ec);
CREATE INDEX idx_go_acc ON go_refs(accession);
CREATE INDEX idx_go_id ON go_refs(go_id);
CREATE INDEX idx_go_ev ON go_refs(evidence);
CREATE INDEX idx_rhea_acc ON rhea(accession);
CREATE INDEX idx_rhea_id ON rhea(rhea_id);
CREATE INDEX idx_xrefs_acc ON xrefs(accession);
CREATE INDEX idx_xrefs_dbid ON xrefs(db, id);
CREATE INDEX idx_features_acc ON features(accession);
CREATE INDEX idx_features_type ON features(ftype);
CREATE INDEX idx_variants_acc ON variants(accession);
CREATE INDEX idx_subcell_acc ON subcellular(accession);
CREATE INDEX idx_subcell_term ON subcellular(term);
CREATE INDEX idx_diseases_acc ON diseases(accession);
CREATE INDEX idx_diseases_name ON diseases(name);
CREATE INDEX idx_isoforms_acc ON isoforms(accession);
CREATE INDEX idx_pubmeds_acc ON pubmeds(accession);
CREATE INDEX idx_pubmeds_id ON pubmeds(pubmed_id);
CREATE INDEX idx_catrhea_acc ON catalytic_rhea(accession);
CREATE INDEX idx_catrhea_id ON catalytic_rhea(rhea_id);
CREATE INDEX idx_catec_acc ON catalytic_ec(accession);
CREATE INDEX idx_catpairs_acc ON catalytic_pairs(accession);
"""

#: Post-load derived table (indexed ECO lookup for the evidence family).
DERIVE = """
CREATE TABLE IF NOT EXISTS function_ecos AS
  SELECT DISTINCT c.accession AS accession, je.value AS eco
  FROM comments_function c, json_each(c.ecos) je;
CREATE INDEX IF NOT EXISTS idx_feco_eco ON function_ecos(eco);
CREATE INDEX IF NOT EXISTS idx_feco_acc ON function_ecos(accession);
"""

_BATCH = 8192


def _j(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def build_store(records: Iterator[dict[str, Any]], out_path: Path, *,
                meta: dict[str, str], progress_every: int = 50000) -> dict[str, Any]:
    t0 = time.time()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    db = sqlite3.connect(str(out_path))
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.executescript(SCHEMA)
    count = 0
    with db:
        db.executemany("INSERT INTO meta(key,value) VALUES (?,?)",
                       sorted(meta.items()))
        buf: dict[str, list[tuple]] = {k: [] for k in (
            "entries", "keywords", "ec", "go_refs", "rhea", "xrefs", "features",
            "variants", "comments_function", "subcellular", "diseases",
            "isoforms", "pubmeds", "catalytic_rhea", "catalytic_ec",
            "catalytic_pairs")}

        def flush() -> None:
            db.executemany(
                "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                buf["entries"])
            db.executemany("INSERT INTO keywords VALUES (?,?)", buf["keywords"])
            db.executemany("INSERT INTO ec VALUES (?,?)", buf["ec"])
            db.executemany("INSERT INTO go_refs VALUES (?,?,?,?,?)", buf["go_refs"])
            db.executemany("INSERT INTO rhea VALUES (?,?)", buf["rhea"])
            db.executemany("INSERT INTO xrefs VALUES (?,?,?)", buf["xrefs"])
            db.executemany("INSERT INTO features VALUES (?,?,?,?)", buf["features"])
            db.executemany("INSERT INTO variants VALUES (?,?,?,?,?,?)", buf["variants"])
            db.executemany("INSERT INTO comments_function VALUES (?,?,?,?)",
                           buf["comments_function"])
            db.executemany("INSERT INTO subcellular VALUES (?,?,?,?)",
                           buf["subcellular"])
            db.executemany("INSERT INTO diseases VALUES (?,?,?,?)", buf["diseases"])
            db.executemany("INSERT INTO isoforms VALUES (?,?,?,?)", buf["isoforms"])
            db.executemany("INSERT INTO pubmeds VALUES (?,?)", buf["pubmeds"])
            db.executemany("INSERT INTO catalytic_rhea VALUES (?,?)",
                           buf["catalytic_rhea"])
            db.executemany("INSERT INTO catalytic_ec VALUES (?,?)",
                           buf["catalytic_ec"])
            db.executemany("INSERT INTO catalytic_pairs VALUES (?,?,?)",
                           buf["catalytic_pairs"])
            for v in buf.values():
                v.clear()

        for rec in records:
            a = rec["accession"]
            buf["entries"].append((
                a, rec["entry_name"], 1 if rec["reviewed"] else 0,
                rec["entry_version"], rec["sequence_version"],
                rec["sequence_length"], rec["sequence_crc64"],
                rec["protein_name"], rec["gene_primary"], _j(rec["genes_all"]),
                _j(rec["gene_synonyms"]), rec["organism_scientific"],
                rec["organism_common"], rec["taxon_id"], _j(rec["lineage"]),
                rec["pe_level"]))
            buf["keywords"] += [(a, k) for k in rec["keywords"]]
            buf["ec"] += [(a, e) for e in rec["ec_numbers"]]
            buf["go_refs"] += [(a, g["id"], g["aspect"], g["term"], g["evidence"])
                               for g in rec["go_refs"]]
            buf["rhea"] += [(a, r) for r in rec["rhea_refs"]]
            buf["xrefs"] += [(a, x["db"], x["id"]) for x in rec["xrefs"]]
            buf["features"] += [(a, f["type"], f["begin"], f["end"])
                                for f in rec["features"]]
            buf["variants"] += [(a, v["begin"], v["end"], v["original"],
                                 v["variation"], v["var_id"])
                                for v in rec["variants"]]
            buf["comments_function"] += [
                (a, c["text"], _j(c["ecos"]), _j(c["pubmeds"]))
                for c in rec["comments_function"]]
            buf["subcellular"] += [(a, s["scope"], s["term"], _j(s["ecos"]))
                                   for s in rec["subcellular"]]
            buf["diseases"] += [(a, d["name"], d["acronym"], d["mim"])
                                for d in rec["diseases"]]
            buf["isoforms"] += [(a, i["iso_id"], i["name"], _j(i["synonyms"]))
                                for i in rec["isoforms"]]
            buf["pubmeds"] += [(a, p) for p in rec["pubmed_ids"]]
            for cat in rec["catalytic_activities"]:
                buf["catalytic_rhea"] += [(a, r) for r in cat["rhea_ids"]]
                buf["catalytic_ec"] += [(a, e) for e in cat["ec"]]
                buf["catalytic_pairs"] += [
                    (a, e, r) for e in cat["ec"] for r in cat["rhea_ids"]]
            count += 1
            if count % _BATCH == 0:
                flush()
            if progress_every and count % progress_every == 0:
                print(f"  store {out_path.name}: {count} entries "
                      f"({time.time()-t0:.0f}s)", flush=True)
        flush()
    db.executescript(INDEXES)
    db.executescript(DERIVE)
    db.execute("ANALYZE")  # planner stats; without these joins pick wrong index
    db.execute("INSERT OR REPLACE INTO meta(key,value) VALUES (?,?)",
               ("entry_count", str(count)))
    db.commit()
    db.close()
    return {"store": str(out_path), "entries": count,
            "wall_seconds": round(time.time() - t0, 1)}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="L16 store builder")
    ap.add_argument("--parser", choices=["dat", "xml"], required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--release-id", required=True)
    ap.add_argument("--source-sha256", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv[1:] if argv and Path(argv[0]).suffix == ".py"
                         else argv)
    if args.parser == "dat":
        from benchmark.plugins.source.l16.uniprot_dat import iter_dat as it
        parser_id = "uniprot_dat.py/1.0"
    else:
        from benchmark.plugins.source.l16.uniprot_xml import iter_xml as it
        parser_id = "uniprot_xml.py/1.0"
    meta = {"release_id": args.release_id, "parser": parser_id,
            "source_file": Path(args.source).name,
            "source_sha256": args.source_sha256,
            "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    stats = build_store(it(Path(args.source), limit=args.limit),
                        Path(args.out), meta=meta)
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv))
