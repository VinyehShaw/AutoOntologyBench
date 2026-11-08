"""L15 store builder: pinned Reactome release-98 files -> SQLite truth store.

Three parse paths, deliberately independent:
- path M: official mapping TSVs (stdlib csv) -> ``m_*`` tables;
- path B: BioPAX per-pathway OWL files -> ``b_*`` tables (streaming XML
  extractor; rdflib re-parse of a sample cross-checks the extractor);
- path S: human SBML (libsbml, optional) -> ``s_*`` spot-check tables.

Every table row carries no fabricated data: loaders copy fields verbatim from
pinned bytes. The manifest records per-input sha256 so the store build is a
deterministic function of the pinned snapshot.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator
from xml.etree.ElementTree import iterparse

BP_NS = "http://www.biopax.org/release/biopax-level3.owl#"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

STABLE_RE = re.compile(r"R-[A-Z]{3}-\d+")
REACTOME_URI_RE = re.compile(r"[#/](R-[A-Z]{3}-\d+(?:\.\d+)?)(?:\b|$)")

# ---------------------------------------------------------------------------
# generic helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _has_table(con: sqlite3.Connection, table: str) -> bool:
    return bool(con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,)).fetchall())


def _rows(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            yield line.split("\t")


def _exec_many(con: sqlite3.Connection, sql: str, rows: Iterator[tuple]) -> int:
    cur = con.cursor()
    n = 0
    for r in rows:
        cur.execute(sql, r)
        n += 1
    return n


# ---------------------------------------------------------------------------
# path M: mapping TSV loaders (column counts verified against release-98 bytes)
# ---------------------------------------------------------------------------


#: (table, ddl, source file, row->tuple). Column layouts verified against the
#: pinned release-98 bytes (see census + acquisition receipts).
M_SPECS: tuple[tuple[str, str, str, Any], ...] = (
    ("m_pathways",
     "CREATE TABLE m_pathways (id TEXT PRIMARY KEY, name TEXT, species TEXT)",
     "ReactomePathways.txt",
     lambda r: tuple(r[:3]) if len(r) >= 3 else None),
    ("m_pathway_edges",
     "CREATE TABLE m_pathway_edges (parent TEXT, child TEXT)",
     "ReactomePathwaysRelation.txt",
     lambda r: (r[0], r[1]) if len(r) >= 2 else None),
    ("m_summation",
     "CREATE TABLE m_summation (id TEXT PRIMARY KEY, text TEXT)",
     "pathway2summation.txt",
     lambda r: (r[0], r[1]) if len(r) >= 2 else None),
    ("m_stable_ids",
     "CREATE TABLE m_stable_ids (id TEXT PRIMARY KEY, rest TEXT)",
     "reactome_stable_ids.txt",
     lambda r: (r[0], "\t".join(r[1:])) if r else None),
    ("m_reaction_exporter_raw",
     "CREATE TABLE m_reaction_exporter_raw (pathway_id TEXT, reaction_id TEXT,"
     " reaction_name TEXT, accession TEXT, roles TEXT)",
     "reactome_reaction_exporter.txt",
     lambda r: tuple(r[:5]) if len(r) >= 5 and r[0] != "pathway_id" else None),
    ("m_reaction_pmids",
     "CREATE TABLE m_reaction_pmids (reaction_id TEXT, pmid TEXT)",
     "ReactionPMIDS.txt",
     lambda r: (r[0], r[1]) if len(r) >= 2 else None),
    ("m_complex_pathway",
     "CREATE TABLE m_complex_pathway (complex_id TEXT, pathway_id TEXT,"
     " top_level_pathway TEXT)",
     "Complex_2_Pathway_human.txt",
     lambda r: tuple(r[:3]) if len(r) >= 3 and r[0] != "complex" else None),
    ("m_complex_pmids",
     "CREATE TABLE m_complex_pmids (complex_id TEXT, pmid TEXT)",
     "ComplexParticipantsPubMedIdentifiers_human.txt",
     lambda r: None),  # exploded below (pipe-separated PMIDs)
    ("m_entity_lit",
     "CREATE TABLE m_entity_lit (accession TEXT, entity_id TEXT, pmid TEXT)",
     "UniProt_Entity_LiteratureReferences.txt",
     lambda r: None),  # exploded below (bracketed PMID list)
    ("m_protein_roles",
     "CREATE TABLE m_protein_roles (accession TEXT, role TEXT, reaction_id TEXT)",
     "ProteinRoleReaction.txt",
     lambda r: (r[0], r[1], r[2]) if len(r) >= 3 else None),
    ("m_disease_pathways",
     "CREATE TABLE m_disease_pathways (id TEXT, name TEXT)",
     "HumanDiseasePathways.txt",
     lambda r: (r[0], r[1] if len(r) > 1 else "") if r else None),
)

M_INDEXES = (
    "CREATE INDEX idx_m_edges_child ON m_pathway_edges(child)",
    "CREATE INDEX idx_m_edges_parent ON m_pathway_edges(parent)",
    "CREATE INDEX idx_m_pmids_reaction ON m_reaction_pmids(reaction_id)",
    "CREATE INDEX idx_m_xref_event ON m_xref_events(event_id)",
    "CREATE INDEX idx_m_xref_acc ON m_xref_events(db, accession)",
    "CREATE INDEX idx_m_xref_species ON m_xref_events(species)",
    "CREATE INDEX idx_m_xref_name ON m_xref_events(event_name)",
    "CREATE INDEX idx_m_complex_pw ON m_complex_pathway(pathway_id)",
    "CREATE INDEX idx_m_entity_lit ON m_entity_lit(accession)",
    "CREATE INDEX idx_m_roles_reaction ON m_protein_roles(reaction_id)",
    "CREATE INDEX idx_m_roles_acc ON m_protein_roles(accession)",
)


def load_mapping_tables(con: sqlite3.Connection, root: Path,
                        skip_missing: bool = False) -> dict[str, int]:
    def have(name: str) -> bool:
        ok = (root / name).is_file() and (root / name).stat().st_size > 0
        if not ok and not skip_missing:
            raise FileNotFoundError(f"mapping file missing: {name}")
        return ok

    counts: dict[str, int] = {}
    for table, ddl, fname, row_map in M_SPECS:
        if not have(fname):
            continue
        con.execute(ddl)
        n = 0
        for r in _rows(root / fname):
            t = row_map(r)
            if t is None:
                continue
            marks = ",".join("?" * len(t))
            con.execute(f"INSERT OR REPLACE INTO {table} VALUES ({marks})", t)
            n += 1
        counts[table] = n

    con.execute("""CREATE TABLE IF NOT EXISTS m_xref_events
                   (db TEXT, accession TEXT, event_id TEXT, event_name TEXT,
                    evidence TEXT, species TEXT)""")
    for db, fname in (("UniProt", "UniProt2Reactome_All_Levels.txt"),
                      ("ChEBI", "ChEBI2Reactome_All_Levels.txt"),
                      ("NCBI", "NCBI2Reactome_All_Levels.txt")):
        if not have(fname):
            continue
        n = 0
        for r in _rows(root / fname):
            if len(r) < 6:
                continue
            con.execute("INSERT INTO m_xref_events VALUES (?,?,?,?,?,?)",
                        (db, r[0], r[1], r[3], r[4], r[5]))
            n += 1
        counts[f"m_xref_events_{db}"] = n

    # Complex participants file: header row; cols identifier, name,
    # participants (pipe-sep db:accession), participatingComplex, PMIDs (|).
    cx_path = root / "ComplexParticipantsPubMedIdentifiers_human.txt"
    if have("ComplexParticipantsPubMedIdentifiers_human.txt"):
        con.execute("""CREATE TABLE IF NOT EXISTS m_complex_parts
                       (identifier TEXT, participant TEXT, name TEXT)""")
        n_parts = n_pmids = 0
        for r in _rows(cx_path):
            if len(r) < 5 or r[0] == "identifier":
                continue
            ident = r[0]
            for part in r[2].split("|"):
                if part:
                    con.execute("INSERT INTO m_complex_parts VALUES (?,?,?)",
                                (ident, part, r[1]))
                    n_parts += 1
            for pmid in r[4].split("|"):
                if pmid:
                    con.execute("INSERT INTO m_complex_pmids VALUES (?,?)",
                                (ident, pmid))
                    n_pmids += 1
        counts["m_complex_parts"] = n_parts
        counts["m_complex_pmids"] = n_pmids

    el_path = root / "UniProt_Entity_LiteratureReferences.txt"
    if have("UniProt_Entity_LiteratureReferences.txt"):
        n_lit = 0
        for r in _rows(el_path):
            if len(r) < 4 or r[0] == "Identifier":
                continue
            for pmid in r[3].strip().strip("[]").split(","):
                pmid = pmid.strip()
                if pmid:
                    con.execute("INSERT INTO m_entity_lit VALUES (?,?,?)",
                                (r[0], r[1], pmid))
                    n_lit += 1
        counts["m_entity_lit"] = n_lit

    con.execute("""CREATE TABLE IF NOT EXISTS m_reaction_roles
                   (reaction_id TEXT, accession TEXT, role TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS m_reaction_pathway
                   (pathway_id TEXT, reaction_id TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS m_reaction_names
                   (reaction_id TEXT PRIMARY KEY, name TEXT)""")
    try:
        n_roles = 0
        for r in con.execute(
                "SELECT pathway_id, reaction_id, reaction_name, accession, roles"
                " FROM m_reaction_exporter_raw"):
            con.execute("INSERT OR REPLACE INTO m_reaction_names VALUES (?,?)",
                        (r[1], r[2]))
            con.execute("INSERT INTO m_reaction_pathway VALUES (?,?)", (r[0], r[1]))
            roles = r[4].strip().strip("[]")
            for role in (x.strip() for x in roles.split(",") if x.strip()):
                con.execute("INSERT INTO m_reaction_roles VALUES (?,?,?)",
                            (r[1], r[3], role))
                n_roles += 1
        counts["m_reaction_roles"] = n_roles
        counts["m_reaction_pathway"] = con.execute(
            "SELECT COUNT(*) FROM m_reaction_pathway").fetchone()[0]
        con.execute("CREATE INDEX IF NOT EXISTS idx_m_rr_reaction ON m_reaction_roles(reaction_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_m_rr_acc ON m_reaction_roles(accession)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_m_rp_path ON m_reaction_pathway(pathway_id)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_m_rp_reaction ON m_reaction_pathway(reaction_id)")
    except sqlite3.Error:
        pass

    for sql in M_INDEXES:
        try:
            con.execute(sql)
        except sqlite3.Error:
            pass
    return counts


# ---------------------------------------------------------------------------
# path B tables
# ---------------------------------------------------------------------------


def _create_b_tables(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE b_events (id TEXT, class TEXT, name TEXT, uri TEXT, file TEXT,
                           PRIMARY KEY (id, file));
    CREATE TABLE b_pathway_components (pathway TEXT, component TEXT, file TEXT);
    CREATE TABLE b_reaction_sides (reaction TEXT, side TEXT, pe_uri TEXT, file TEXT);
    CREATE TABLE b_controls (reaction TEXT, control_class TEXT, control_type TEXT,
                             controller_pe TEXT, file TEXT);
    CREATE TABLE b_complex_components (complex_uri TEXT, component_uri TEXT, file TEXT);
    CREATE TABLE b_compartments (pe_uri TEXT, compartment TEXT, file TEXT);
    CREATE TABLE b_pe (uri TEXT, class TEXT, display_name TEXT, file TEXT,
                       PRIMARY KEY (uri, file));
    CREATE TABLE b_xrefs (pe_uri TEXT, db TEXT, accession TEXT, file TEXT);
    CREATE TABLE b_modifications (pe_uri TEXT, feature TEXT, file TEXT);
    CREATE TABLE b_organisms (file TEXT PRIMARY KEY, organism TEXT);
    CREATE INDEX idx_b_events_id ON b_events(id);
    CREATE INDEX idx_b_pc_path ON b_pathway_components(pathway);
    CREATE INDEX idx_b_sides ON b_reaction_sides(reaction, side);
    CREATE INDEX idx_b_controls ON b_controls(reaction);
    CREATE INDEX idx_b_complex ON b_complex_components(complex_uri);
    CREATE INDEX idx_b_comp ON b_compartments(pe_uri);
    CREATE INDEX idx_b_xrefs ON b_xrefs(pe_uri);
    CREATE INDEX idx_b_mods ON b_modifications(pe_uri);
    """)


def load_biopax_zip(con: sqlite3.Connection, source: Path,
                    *, only_species: str | None = "Homo_sapiens",
                    progress_every: int = 200) -> dict[str, int]:
    """Parse BioPAX .owl files into b_* tables. ``source`` may be biopax.zip
    or a directory of verified member .owl files (member-level acquisition)."""
    counts: dict[str, int] = {"files": 0, "events": 0, "pe": 0}
    if source.is_dir():
        items: list[tuple[str, bytes]] = []
        names = sorted(p for p in source.glob("*.owl"))
        if only_species:
            names = [p for p in names if only_species.lower() in p.name.lower()]
        for p in names:
            items.append((p.name, p.read_bytes()))
        _load_biopax_items(con, items, counts, progress_every)
        return counts
    with zipfile.ZipFile(source) as zf:
        names = [n for n in zf.namelist()
                 if n.endswith(".owl") and not n.startswith("__MACOSX")]
        if only_species:
            names = [n for n in names if only_species.lower() in n.lower()]
        items = [(name.rsplit("/", 1)[-1], zf.read(name)) for name in names]
    _load_biopax_items(con, items, counts, progress_every)
    return counts


def _load_biopax_items(con: sqlite3.Connection, items: list[tuple[str, bytes]],
                       counts: dict[str, int], progress_every: int) -> None:
    for base, raw in items:
        try:
            nodes = parse_biopax_nodes(raw)
            facts = facts_from_nodes(nodes)
        except Exception:
            counts["parse_errors"] = counts.get("parse_errors", 0) + 1
            continue
        con.executemany("INSERT OR REPLACE INTO b_events VALUES (?,?,?,?,?)",
                        [(s, c, n, u, base) for s, c, n, u in facts.events])
        con.executemany("INSERT INTO b_pathway_components VALUES (?,?,?)",
                        [(p, c, base) for p, c in facts.pathway_components])
        con.executemany("INSERT INTO b_reaction_sides VALUES (?,?,?,?)",
                        [(r, s, p, base) for r, s, p in facts.reaction_sides])
        con.executemany("INSERT INTO b_controls VALUES (?,?,?,?,?)",
                        [(r, c, t, p, base) for r, c, t, p in facts.controls])
        con.executemany("INSERT INTO b_complex_components VALUES (?,?,?)",
                        [(c, p, base) for c, p in facts.complex_components])
        con.executemany("INSERT INTO b_compartments VALUES (?,?,?)",
                        [(p, t, base) for p, t in facts.compartments])
        con.executemany("INSERT OR REPLACE INTO b_pe VALUES (?,?,?,?)",
                        [(u, c, n, base) for u, c, n in facts.pe])
        con.executemany("INSERT INTO b_xrefs VALUES (?,?,?,?)",
                        [(p, d, a, base) for p, d, a in facts.xrefs])
        con.executemany("INSERT INTO b_modifications VALUES (?,?,?)",
                        [(p, t, base) for p, t in facts.modifications])
        if facts.organism:
            con.execute("INSERT OR REPLACE INTO b_organisms VALUES (?,?)",
                        (base, facts.organism))
        counts["files"] += 1
        counts["events"] += len(facts.events)
        counts["pe"] += len(facts.pe)
        if progress_every and counts["files"] % progress_every == 0:
            con.commit()


# ---------------------------------------------------------------------------
# path S: SBML spot cross-check (reaction sides), optional
# ---------------------------------------------------------------------------


def load_sbml_spot(con: sqlite3.Connection, tgz_path: Path,
                   *, max_models: int = 400) -> dict[str, int]:
    try:
        from benchmark.plugins.source.l07.sbml_struct import parse_sbml_bytes
    except Exception:
        return {"sbml_models": 0, "skipped": 1}
    counts = {"sbml_models": 0, "sbml_reactions": 0}
    con.executescript("""
    CREATE TABLE IF NOT EXISTS s_reaction_sides (model TEXT, reaction TEXT,
                                                 side TEXT, species TEXT);
    CREATE INDEX IF NOT EXISTS idx_s_sides ON s_reaction_sides(reaction, side);
    """)
    done = 0
    with tarfile.open(tgz_path) as tf:
        for member in tf:
            if not (member.name.endswith(".xml") or member.name.endswith(".sbml")) or done >= max_models:
                continue
            raw = tf.extractfile(member)
            if raw is None:
                continue
            try:
                struct = parse_sbml_bytes(raw.read())
            except Exception:
                continue
            for rxn in struct.reactions:
                for sp, _stoich in rxn.reactants:
                    con.execute("INSERT INTO s_reaction_sides VALUES (?,?,?,?)",
                                (member.name, rxn.rid, "left", sp))
                for sp, _stoich in rxn.products:
                    con.execute("INSERT INTO s_reaction_sides VALUES (?,?,?,?)",
                                (member.name, rxn.rid, "right", sp))
            done += 1
            counts["sbml_models"] = done
    counts["sbml_reactions"] = con.execute(
        "SELECT COUNT(*) FROM s_reaction_sides").fetchone()[0]
    return counts


EVENT_CLASSES = {
    "Pathway", "Reaction", "BlackboxEvent", "Interaction", "ComplexAssembly",
    "Transport", "BiochemicalReaction", "Degradation", "BiochemicalPathwayStep",
    "TransportWithBiochemicalReaction",
}
CONTROL_CLASSES = {"Catalysis", "Control", "Regulation", "TemplateReactionRegulation",
                   "MolecularInteraction"}
PE_CLASSES = {
    "Protein", "Complex", "SmallMolecule", "Rna", "Dna", "PhysicalEntity",
    "EntitySet", "GenomeEncodedEntity", "CandidateSet", "DefinedSet",
    "OpenSet", "SequenceEntity", "SimplePhysicalEntity",
}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _sid(text: str) -> str:
    m = REACTOME_URI_RE.search(text)
    return m.group(1) if m else ""


class BioNode:
    __slots__ = ("uri", "cls", "props")

    def __init__(self, uri: str, cls: str) -> None:
        self.uri = uri
        self.cls = cls
        self.props: dict[str, list[str]] = {}

    def add(self, pred: str, val: str) -> None:
        self.props.setdefault(pred, []).append(val)


def parse_biopax_nodes(raw: bytes) -> dict[str, BioNode]:
    """Parse one BioPAX OWL file into {uri: BioNode} with generic props.

    Streaming iterparse; every rdf:about / rdf:nodeID element becomes a node;
    child elements append (predicate, resource-or-text) pairs. No schema
    assumptions beyond the BioPAX3/RDF layer, so the extractor survives
    Reactome URI-convention details.
    """
    import io
    nodes: dict[str, BioNode] = {}
    stack: list[str] = []
    for event, elem in iterparse(io.BytesIO(raw), events=("start", "end")):
        if event == "start":
            about = (elem.attrib.get(f"{{{RDF_NS}}}about")
                     or elem.attrib.get(f"{{{RDF_NS}}}nodeID"))
            if about:
                node = nodes.get(about)
                if node is None:
                    node = BioNode(about, _local(elem.tag))
                    nodes[about] = node
                stack.append(about)
            continue
        tag = _local(elem.tag)
        if stack and elem.attrib.get(f"{{{RDF_NS}}}about", "") == stack[-1] and tag == nodes[stack[-1]].cls:
            stack.pop()
            elem.clear()
            continue
        if not stack:
            elem.clear()
            continue
        parent = nodes[stack[-1]]
        res = elem.attrib.get(f"{{{RDF_NS}}}resource", "")
        text = (elem.text or "").strip()
        if res:
            parent.add(tag, "@" + res)
        elif text:
            parent.add(tag, text)
        elem.clear()
    return nodes


class BiopaxFileFacts:
    """Relation rows from one BioPAX file. Tuple arities:
    events=(sid, class, name, uri); pathway_components=(pathway_sid, component_sid);
    reaction_sides=(reaction_sid, side, pe_uri); controls=(reaction_sid, control_class,
    control_type, controller_pe_uri); complex_components=(complex_uri, component_uri);
    compartments=(pe_uri, term); pe=(uri, class, display_name);
    xrefs=(pe_uri, db, accession); modifications=(pe_uri, feature_term)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, str, str]] = []
        self.pathway_components: list[tuple[str, str]] = []
        self.reaction_sides: list[tuple[str, str, str]] = []
        self.controls: list[tuple[str, str, str, str]] = []
        self.complex_components: list[tuple[str, str]] = []
        self.compartments: list[tuple[str, str]] = []
        self.pe: list[tuple[str, str, str]] = []
        self.xrefs: list[tuple[str, str, str]] = []
        self.modifications: list[tuple[str, str]] = []
        self.organism: str = ""


def facts_from_nodes(nodes: dict[str, BioNode]) -> BiopaxFileFacts:
    f = BiopaxFileFacts()
    uri2sid: dict[str, str] = {}
    for uri, node in nodes.items():
        sid = _sid(uri)
        if sid:
            uri2sid[uri] = sid
    for uri, node in nodes.items():
        cls = node.cls
        if cls in EVENT_CLASSES:
            sid = uri2sid.get(uri) or _sid(node.props.get("displayName", [""])[0]) or uri
            name = (node.props.get("displayName") or node.props.get("name") or [""])[0]
            f.events.append((sid, cls, name, uri))
            for comp in node.props.get("pathwayComponent", []):
                if comp.startswith("@"):
                    csid = uri2sid.get(comp[1:]) or _sid(comp[1:]) or comp[1:]
                    f.pathway_components.append((sid, csid))
        if cls in EVENT_CLASSES or cls in PE_CLASSES:
            for side in ("left", "right"):
                for ref in node.props.get(side, []):
                    if ref.startswith("@"):
                        f.reaction_sides.append((uri2sid.get(uri) or _sid(uri) or uri, side, ref[1:]))
        if cls in CONTROL_CLASSES:
            ctype = (node.props.get("controlType") or [""])[0]
            controlled = [r[1:] for r in node.props.get("controlled", []) if r.startswith("@")]
            controllers = [r[1:] for r in node.props.get("controller", []) if r.startswith("@")]
            for c in controlled:
                csid = uri2sid.get(c) or _sid(c) or c
                for ctl in controllers:
                    f.controls.append((csid, cls, ctype, ctl))
        if cls in PE_CLASSES:
            name = (node.props.get("displayName") or node.props.get("name") or [""])[0]
            f.pe.append((uri, cls, name))
            for comp in node.props.get("component", []):
                if comp.startswith("@"):
                    f.complex_components.append((uri, comp[1:]))
            for loc in node.props.get("cellularLocation", []):
                if loc.startswith("@"):
                    vnode = nodes.get(loc[1:])
                    term = ""
                    if vnode is not None:
                        term = (vnode.props.get("term") or vnode.props.get("name") or [""])[0]
                    f.compartments.append((uri, term or loc[1].rsplit("/", 1)[-1]))
            for er in node.props.get("entityReference", []):
                if not er.startswith("@"):
                    continue
                ref = nodes.get(er[1:])
                if ref is None:
                    continue
                for x in ref.props.get("xref", []):
                    if not x.startswith("@"):
                        continue
                    xn = nodes.get(x[1:])
                    if xn is None:
                        continue
                    db = (xn.props.get("db") or [""])[0]
                    acc = (xn.props.get("id") or [""])[0]
                    if db and acc:
                        f.xrefs.append((uri, db, acc))
            for feat in node.props.get("feature", []) + node.props.get("modification", []):
                if not feat.startswith("@"):
                    continue
                fn = nodes.get(feat[1:])
                if fn is None:
                    continue
                term = ""
                for mt in fn.props.get("modificationType", []):
                    if mt.startswith("@"):
                        mn = nodes.get(mt[1:])
                        if mn is not None:
                            term = (mn.props.get("term") or mn.props.get("name") or [""])[0]
                if not term:
                    term = fn.cls
                f.modifications.append((uri, term))
        if cls == "BioSource":
            nm = (node.props.get("name") or [""])[0]
            if nm:
                f.organism = nm
    return f


def rdflib_member_check(members_dir: Path,
                        *, max_uncompressed_mb: int = 64) -> dict[str, Any]:
    """Independent second parser: rdflib re-parse of the smallest pinned BioPAX
    member, cross-checked against the streaming extractor on identical bytes."""
    import rdflib
    from rdflib.namespace import RDF
    files = sorted(members_dir.glob("*.owl"), key=lambda p: p.stat().st_size)
    files = [f for f in files
             if f.stat().st_size <= max_uncompressed_mb * (1 << 20)]
    if not files:
        return {"ok": False, "reason": "no_small_member"}
    path = files[0]
    raw = path.read_bytes()
    fast = facts_from_nodes(parse_biopax_nodes(raw))
    g = rdflib.Graph()
    g.parse(data=raw, format="xml")
    bp = rdflib.Namespace(BP_NS)
    checks = {
        "pathway": (sum(1 for _s, c, _n, _u in fast.events if c == "Pathway"),
                    len(list(g.subjects(RDF.type, bp.Pathway)))),
        "biochemical_reaction": (
            sum(1 for _s, c, _n, _u in fast.events if c == "BiochemicalReaction"),
            len(list(g.subjects(RDF.type, bp.BiochemicalReaction)))),
        "complex": (sum(1 for _u, c, _n in fast.pe if c == "Complex"),
                    len(list(g.subjects(RDF.type, bp.Complex)))),
        "protein": (sum(1 for _u, c, _n in fast.pe if c == "Protein"),
                    len(list(g.subjects(RDF.type, bp.Protein)))),
    }
    agree = all(a == b for a, b in checks.values())
    return {"ok": agree, "member": path.name, "checks": checks}


def build_store(pinned_root: Path, out_dir: Path, *, partial: bool = False) -> dict[str, Any]:
    """Build the SQLite truth store from the pinned snapshot. Deterministic:
    same pinned bytes -> same tables (row order is insertion order)."""
    from source_registry.fetchers.l15_reactome import RELEASE_ID, read_sha256sums
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sums_path = pinned_root / "SHA256SUMS.txt"
    sums = read_sha256sums(sums_path) if sums_path.is_file() else {}
    if sums and not partial:
        missing = [n for n in sums if not (pinned_root / n).is_file()]
        if missing:
            raise FileNotFoundError(f"pinned files missing at build: {missing}")
        for name, want in sums.items():
            got = _sha256_file(pinned_root / name)
            if got != want:
                raise ValueError(f"sha256 mismatch: {name}: {got[:16]} != {want[:16]}")
    if not sums:
        sums = {n: _sha256_file(pinned_root / n) for n in
                (p.name for p in pinned_root.iterdir() if p.is_file()
                 and p.suffix in (".txt", ".zip", ".tgz"))}
    db_path = out_dir / "reactome.db"
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    manifest: dict[str, Any] = {
        "release_id": RELEASE_ID, "built_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs_sha256": sums, "counts": {}, "checks": {}}
    try:
        con.execute("PRAGMA journal_mode=OFF")
        con.execute("PRAGMA synchronous=OFF")
        manifest["counts"].update(load_mapping_tables(con, pinned_root, skip_missing=partial))
        con.commit()
        biopax_zip = pinned_root / "biopax.zip"
        members_dir = pinned_root / "biopax_members"
        biopax_src: Path | None = None
        if members_dir.is_dir() and any(members_dir.glob("*.owl")):
            biopax_src = members_dir
        elif biopax_zip.is_file():
            biopax_src = biopax_zip
        if biopax_src is not None:
            _create_b_tables(con)
            b_counts = load_biopax_zip(con, biopax_src, only_species=None)
            con.commit()
            manifest["counts"].update({f"b_{k}": v for k, v in b_counts.items()})
            manifest["counts"]["b_source"] = str(biopax_src)
            if members_dir.is_dir():
                manifest["checks"]["rdflib_member"] = rdflib_member_check(members_dir)
        sbml_tgz = pinned_root / "homo_sapiens.3.1.sbml.tgz"
        if sbml_tgz.is_file():
            manifest["counts"].update(load_sbml_spot(con, sbml_tgz))
        con.commit()
        # Derived fallback: when the published UniProt All_Levels mapping is
        # absent from the pin, derive accession -> events(all levels) from
        # BioPAX (xref -> PE -> reactions -> direct pathways -> ancestors).
        # Marked derived; f10 cross-checks against ContentService samples.
        has_uniprot_map = con.execute(
            "SELECT COUNT(*) FROM m_xref_events WHERE db='UniProt'").fetchone()[0] \
            if _has_table(con, "m_xref_events") else 0
        if _has_table(con, "b_events") and not has_uniprot_map:
            con.execute("""CREATE TABLE d_uniprot_events
                           (accession TEXT, event_id TEXT)""")
            con.execute("CREATE INDEX idx_due_acc ON d_uniprot_events(accession)")
            child2parents: dict[str, set[str]] = {}
            for (p,) in con.execute("SELECT DISTINCT pathway FROM b_pathway_components"):
                child2parents.setdefault(p, set())
            for pw, comp in con.execute(
                    "SELECT pathway, component FROM b_pathway_components"):
                child2parents.setdefault(comp, set()).add(pw)
            def ancestors(e: str) -> set[str]:
                seen: set[str] = set()
                stack = [e]
                while stack:
                    cur = stack.pop()
                    for p in child2parents.get(cur, ()):
                        if p not in seen:
                            seen.add(p)
                            stack.append(p)
                return seen
            n_d = 0
            for (acc, pe) in con.execute(
                    "SELECT DISTINCT accession, pe_uri FROM b_xrefs"
                    " WHERE db LIKE 'UniProt%'"):
                evs: set[str] = set()
                for (rxn,) in con.execute(
                        "SELECT DISTINCT reaction FROM b_reaction_sides WHERE pe_uri=?",
                        (pe,)):
                    evs.add(rxn)
                    evs |= ancestors(rxn)
                for (rxn,) in con.execute(
                        "SELECT DISTINCT reaction FROM b_controls WHERE controller_pe=?",
                        (pe,)):
                    evs.add(rxn)
                    evs |= ancestors(rxn)
                for e in sorted(evs):
                    con.execute("INSERT INTO d_uniprot_events VALUES (?,?)", (acc, e))
                    n_d += 1
            manifest["counts"]["d_uniprot_events"] = n_d
            con.commit()
    finally:
        con.close()
    manifest["wall_seconds"] = round(time.time() - t0, 1)
    manifest["db_sha256"] = _sha256_file(db_path)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
