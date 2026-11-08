"""L15 source census: real counts from the pinned snapshot + built store.

Writes ``expansion/lanes/L15/SOURCE_CENSUS.md`` (via --out) with a JSON sidecar.
Counts are recomputed from the pinned files / store every run; never edited by
hand.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

from source_registry.fetchers.l15_reactome import (
    PINNED_ROOT, RELEASE_ID, PINNED_FILES, read_sha256sums,
)
from benchmark.plugins.source.l15.plugin import STORE_DB, STORE_MANIFEST


def _count_lines(path: Path) -> int:
    n = 0
    with path.open("rb") as fh:
        for _ in fh:
            n += 1
    return n


def census() -> dict:
    out: dict = {"release_id": RELEASE_ID, "generated_utc":
                 time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    sums_file = PINNED_ROOT / "SHA256SUMS.txt"
    sums = read_sha256sums(sums_file) if sums_file.is_file() else {}
    files = {}
    for name in PINNED_FILES:
        p = PINNED_ROOT / name
        files[name] = {
            "present": p.is_file() and p.stat().st_size > 0,
            "bytes": p.stat().st_size if p.is_file() else 0,
            "sha256": sums.get(name, ""),
        }
        if files[name]["present"] and name.endswith(".txt"):
            files[name]["lines"] = _count_lines(p)
    out["files"] = files
    if STORE_DB.is_file():
        con = sqlite3.connect(str(STORE_DB))
        c: dict[str, int] = {}
        for table, label in (
            ("m_pathways", "pathways_total"),
            ("m_pathway_edges", "pathway_hierarchy_edges"),
            ("m_summation", "pathway_summations"),
            ("m_reaction_pmids", "reaction_pmid_rows"),
            ("m_xref_events", "xref_event_rows_total"),
            ("m_protein_roles", "protein_role_rows"),
            ("m_complex_pathway", "complex_pathway_rows"),
            ("m_entity_lit", "entity_lit_rows"),
            ("b_events", "biopax_events"),
            ("b_pe", "biopax_physical_entities"),
            ("b_reaction_sides", "biopax_reaction_sides"),
            ("b_controls", "biopax_controls"),
            ("b_complex_components", "biopax_complex_components"),
            ("b_compartments", "biopax_compartments"),
            ("b_xrefs", "biopax_xrefs"),
            ("b_modifications", "biopax_modifications"),
        ):
            try:
                c[label] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error:
                c[label] = 0
        c["pathways_human"] = con.execute(
            "SELECT COUNT(*) FROM m_pathways WHERE species='Homo sapiens'").fetchone()[0]
        c["species_total"] = con.execute(
            "SELECT COUNT(DISTINCT species) FROM m_pathways").fetchone()[0]
        c["reactions_human_biopax"] = con.execute(
            "SELECT COUNT(*) FROM b_events WHERE class != 'Pathway'").fetchone()[0] \
            if _t(con, "b_events") else 0
        c["pathways_human_biopax"] = con.execute(
            "SELECT COUNT(*) FROM b_events WHERE class='Pathway'").fetchone()[0] \
            if _t(con, "b_events") else 0
        c["distinct_compartments"] = con.execute(
            "SELECT COUNT(DISTINCT compartment) FROM b_compartments").fetchone()[0] \
            if _t(con, "b_compartments") else 0
        c["uniprot_accessions_human"] = con.execute(
            "SELECT COUNT(DISTINCT accession) FROM m_xref_events "
            "WHERE db='UniProt' AND species='Homo sapiens'").fetchone()[0] \
            if _t(con, "m_xref_events") else 0
        out["store_counts"] = c
        if STORE_MANIFEST.is_file():
            out["store_manifest"] = json.loads(STORE_MANIFEST.read_text())
    return out


def _t(con: sqlite3.Connection, table: str) -> bool:
    return bool(con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchall())


def render_md(c: dict) -> str:
    lines = [
        "# L15 Reactome — Source Census (release 98)",
        "",
        f"- release: {c['release_id']} (ContentService /data/database/version at acquisition)",
        f"- generated_utc: {c['generated_utc']}",
        "- pin root: `external_knowledge/pinned/reactome/release-98/`",
        "- license: Reactome database data CC0 (images/icons CC BY 4.0 excluded; data-only lane)",
        "",
        "## Pinned files",
        "",
        "| file | bytes | sha256 (prefix) | present |",
        "|---|---:|---|---|",
    ]
    for name, f in sorted(c["files"].items()):
        sha = (f["sha256"] or "")[:16]
        lines.append(f"| {name} | {f['bytes']} | {sha} | {f['present']} |")
    sc = c.get("store_counts", {})
    if sc:
        lines += ["", "## Parsed store counts (reactome.db)", "",
                  "| entity/relation | count |", "|---|---:|---|"]
        for k in sorted(sc):
            lines.append(f"| {k} | {sc[k]} |")
    checks = (c.get("store_manifest") or {}).get("checks", {})
    if checks:
        lines += ["", "## Independent-parser cross-checks", "",
                  "```json", json.dumps(checks, indent=1, sort_keys=True)[:2000], "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    c = census()
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[4] / "data" / "erule" / "review5" /
        "build8000" / "expansion" / "lanes" / "L15")
    out.mkdir(parents=True, exist_ok=True)
    (out / "SOURCE_CENSUS.md").write_text(render_md(c), encoding="utf-8")
    (out / "source_census.json").write_text(
        json.dumps(c, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out), "counts": c.get("store_counts", {})},
                     sort_keys=True)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
