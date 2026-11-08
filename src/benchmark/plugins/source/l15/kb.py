"""L15 KB: typed query layer over the pinned Reactome store (reactome.db).

Both the task proposers and the native verifier query through this layer; the
verifier additionally re-derives answers through the *other* parse path where
the relation exists in both (BioPAX ``b_*`` vs mapping ``m_*`` tables).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class L15KB:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self.con = sqlite3.connect(self.db_path)
        self.con.row_factory = sqlite3.Row
        self._has_b: bool | None = None

    def close(self) -> None:
        self.con.close()

    def _q(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        return self.con.execute(sql, args).fetchall()

    def has_biopax(self) -> bool:
        if self._has_b is None:
            self._has_b = bool(self.con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='b_events'"
            ).fetchall())
        return self._has_b

    # -- mapping path (M) ------------------------------------------------------

    def m_pathways(self, species: str | None = None) -> list[dict[str, Any]]:
        if species:
            return [dict(r) for r in self._q(
                "SELECT * FROM m_pathways WHERE species=?", (species,))]
        return [dict(r) for r in self._q("SELECT * FROM m_pathways")]

    def m_pathway_name(self, pid: str) -> str:
        r = self._q("SELECT name FROM m_pathways WHERE id=?", (pid,))
        return r[0]["name"] if r else ""

    def m_children(self, parent: str) -> list[str]:
        return sorted(r["child"] for r in self._q(
            "SELECT child FROM m_pathway_edges WHERE parent=?", (parent,)))

    def m_parents(self, child: str) -> list[str]:
        return sorted(r["parent"] for r in self._q(
            "SELECT parent FROM m_pathway_edges WHERE child=?", (child,)))

    def m_all_parents(self) -> list[str]:
        return sorted(r["parent"] for r in self._q(
            "SELECT DISTINCT parent FROM m_pathway_edges"))

    def m_event_pmids(self, event_id: str) -> list[str]:
        return sorted({str(r["pmid"]) for r in self._q(
            "SELECT pmid FROM m_reaction_pmids WHERE reaction_id=?", (event_id,))})

    def m_uniprot_events(self, accession: str) -> list[str]:
        return sorted({r["event_id"] for r in self._q(
            "SELECT event_id FROM m_xref_events WHERE db='UniProt' AND accession=?",
            (accession,))})

    def m_xref_events_of(self, db: str, accession: str) -> list[str]:
        return sorted({r["event_id"] for r in self._q(
            "SELECT event_id FROM m_xref_events WHERE db=? AND accession=?",
            (db, accession))})

    def m_event_xrefs(self, db: str, event_id: str) -> list[str]:
        return sorted({r["accession"] for r in self._q(
            "SELECT accession FROM m_xref_events WHERE db=? AND event_id=?",
            (db, event_id))})

    def m_event_species(self, event_id: str) -> list[str]:
        return sorted({r["species"] for r in self._q(
            "SELECT DISTINCT species FROM m_xref_events WHERE event_id=?", (event_id,))})

    def m_pathway_species(self, pathway_id: str) -> str:
        r = self._q("SELECT species FROM m_pathways WHERE id=?", (pathway_id,))
        return r[0]["species"] if r else ""

    def m_accessions(self, db: str) -> list[str]:
        return sorted({r["accession"] for r in self._q(
            "SELECT DISTINCT accession FROM m_xref_events WHERE db=?", (db,))})

    # -- BioPAX path (B) --------------------------------------------------------

    def b_events(self, cls: str | None = None) -> list[dict[str, Any]]:
        if cls:
            return [dict(r) for r in self._q(
                "SELECT * FROM b_events WHERE class=?", (cls,))]
        return [dict(r) for r in self._q("SELECT * FROM b_events")]

    def b_event_name(self, sid: str) -> str:
        r = self._q("SELECT name FROM b_events WHERE id=?", (sid,))
        return r[0]["name"] if r else ""

    def b_pathway_components(self, pathway: str) -> list[str]:
        return sorted(r["component"] for r in self._q(
            "SELECT component FROM b_pathway_components WHERE pathway=?", (pathway,)))

    def b_reaction_participants(self, reaction: str, side: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q(
            """SELECT s.pe_uri, p.display_name, p.class FROM b_reaction_sides s
               JOIN b_pe p ON p.uri = s.pe_uri
               WHERE s.reaction=? AND s.side=? ORDER BY s.pe_uri""",
            (reaction, side))]

    def b_reaction_controls(self, reaction: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q(
            """SELECT c.control_class, c.control_type, c.controller_pe,
                      p.display_name
               FROM b_controls c LEFT JOIN b_pe p ON p.uri = c.controller_pe
               WHERE c.reaction=? ORDER BY c.controller_pe""",
            (reaction,))]

    def b_complex_components(self, complex_uri: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q(
            """SELECT c.component_uri, p.display_name, p.class
               FROM b_complex_components c LEFT JOIN b_pe p ON p.uri = c.component_uri
               WHERE c.complex_uri=? ORDER BY c.component_uri""",
            (complex_uri,))]

    def b_compartment(self, pe_uri: str) -> str:
        r = self._q("SELECT compartment FROM b_compartments WHERE pe_uri=?", (pe_uri,))
        return r[0]["compartment"] if r else ""

    def b_pe_modifications(self, pe_uri: str) -> list[str]:
        return sorted(r["feature"] for r in self._q(
            "SELECT feature FROM b_modifications WHERE pe_uri=?", (pe_uri,)))

    def b_pe_xrefs(self, pe_uri: str) -> list[dict[str, Any]]:
        return [dict(r) for r in self._q(
            "SELECT db, accession FROM b_xrefs WHERE pe_uri=? ORDER BY db, accession",
            (pe_uri,))]

    def b_pe(self, uri: str) -> dict[str, Any] | None:
        r = self._q("SELECT * FROM b_pe WHERE uri=?", (uri,))
        return dict(r[0]) if r else None
