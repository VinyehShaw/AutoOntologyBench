"""L15 grounding: CandidateTaskAST -> GroundedTask (public bundle + oracle ref).

Public bundles carry evidence slices from the pinned store sufficient to derive
the answer by correct semantic processing, plus distractor material. The answer
string itself is never serialized into the bundle (leakage is checked at
qualification time against the reference).
"""
from __future__ import annotations

import json
from random import Random
from typing import Any

from benchmark.plugins.source.l15.kb import L15KB
from benchmark.plugins.task_family.l15 import goals as G
from benchmark.plugins.task_family.l15.families import BY_ID

ALLOWED_TOOLS = ["inspect_schema", "inspect_state", "read_public_doc",
                 "probe_native", "edit_artifact", "validate_artifact",
                 "check_action_with_ontology", "submit_final"]

SCHEMA_DOC = (
    "Reactome release-98 evidence bundle. Entities: PhysicalEntity (PE; "
    "identity = uri_tail, e.g. the trailing id of its BioPAX URI; display "
    "names are non-unique and never identify), Event (Pathway or "
    "ReactionLikeEvent; identity = stable id R-HSA-*). Relations: pathway "
    "hierarchy edges (parent->child), pathway components, reaction sides "
    "(left=input, right=output), controls (Catalysis vs Regulation with "
    "controlType ACTIVATION/INHIBITION), complex components (one level), "
    "compartments (same molecule in different compartments = different PEs), "
    "modification features (modified forms are distinct PEs), external "
    "accession mappings (evidence codes such as IEA mark inferred "
    "annotations), and literature PMIDs.")


def _rows(kb: L15KB, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in kb._q(sql, args)]


def _pe_record(kb: L15KB, uri: str) -> dict[str, Any]:
    pe = kb.b_pe(uri) or {}
    return {"pe_uri": G._tail(uri), "name": pe.get("display_name", ""),
            "class": pe.get("class", ""), "compartment": kb.b_compartment(uri)}


def _goal_text(kind: str, ast: dict[str, Any], kb: L15KB) -> str:
    if kind == "pathway_children":
        name = kb.m_pathway_name(ast["parent"]) or ast["parent"]
        return (f"Reconstruct the exact set of direct children (sub-pathways and "
                f"directly contained events) of Reactome pathway {ast['parent']} "
                f"({name!r}) from the hierarchy edge evidence. Report sorted stable "
                f"ids. Children of any other pathway must not be included.")
    if kind == "pathway_events":
        name = kb.b_event_name(ast["pathway"])
        return (f"Report the exact set of ReactionLikeEvent stable ids directly "
                f"contained in pathway {ast['pathway']} ({name!r}): components that "
                f"are events, excluding any event reachable only through a nested "
                f"sub-pathway.")
    if kind == "reaction_inputs":
        return (f"Assemble the exact set of input PhysicalEntities (pe_uri tails) "
                f"of reaction {ast['reaction']} — entities on the left side only. "
                f"Catalysts, regulators and outputs are excluded; a complex or set "
                f"counts as one entity (do not substitute its components).")
    if kind == "reaction_outputs":
        return (f"Assemble the exact set of output PhysicalEntities (pe_uri tails) "
                f"of reaction {ast['reaction']} — entities on the right side only.")
    if kind == "direction_repair":
        return (f"The reaction record for {ast['reaction']} in this bundle has one "
                f"participant on the wrong side (its input/output sides were "
                f"corrupted). Using the independent protein-role evidence rows, "
                f"repair the record: report the corrected left/right assignment as "
                f"sorted pe_uri tail lists.")
    if kind == "reaction_catalyst":
        return (f"Report the exact set of catalyst PhysicalEntities (pe_uri tails) "
                f"of reaction {ast['reaction']}: controllers of Catalysis controls "
                f"only — regulators (Activation/Inhibition) are not catalysts.")
    if kind == "reaction_regulation":
        return (f"Organize the regulation of reaction {ast['reaction']}: report the "
                f"exact sorted set of (regulator pe_uri tail | controlType) pairs "
                f"for non-catalysis controls. ACTIVATION and INHIBITION are "
                f"distinct and must not be merged or flipped.")
    if kind == "complex_members":
        return (f"Report the exact one-level component set (pe_uri tails) of "
                f"complex {ast['complex']}. Do not flatten nested complexes; "
                f"components are entities, not their reference molecules.")
    if kind == "compartment_repair":
        return (f"The compartment assignments of the physical entities in the "
                f"bundle were corrupted. Using the cross-file evidence rows (the "
                f"same entities correctly compartment-annotated elsewhere), "
                f"restore each entity's true compartment. Report a complete "
                f"pe_uri-tail -> compartment map.")
    if kind == "uniprot_events":
        return (f"Report the exact sorted set of Reactome event stable ids that "
                f"UniProt accession {ast['accession']} maps to in Homo sapiens at "
                f"all levels (reactions and their parent pathways). Identity comes "
                f"from stable ids only, never from names.")
    if kind == "modified_forms":
        return (f"Within the pathway slice, report the exact set of distinct "
                f"modified PhysicalEntity forms (pe_uri tails) whose reference "
                f"protein is {ast['accession']}: only entities carrying at least "
                f"one modification feature. The unmodified form must not be "
                f"included, and distinct modifications are distinct entities.")
    if kind == "xref_consistency":
        return (f"Report the exact sorted set of {ast['db']} accessions mapped to "
                f"Reactome event {ast['event']} in the {ast['db']}2Reactome "
                f"mapping. Rows for other databases or other events must not leak "
                f"in.")
    if kind == "species_provenance":
        return (f"For the event named {ast['event_name']!r}, report the exact "
                f"sorted set of species in which a Reactome event with exactly "
                f"this name exists in the all-levels mapping. Inferred (IEA) "
                f"annotations count as their annotated species; do not drop or "
                f"reinterpret them.")
    if kind == "reaction_pmids":
        return (f"Report the exact sorted set of PubMed identifiers cited as "
                f"literature evidence for reaction {ast['reaction']}. PMIDs of "
                f"same-named reactions in other species must not be included.")
    if kind == "multihop_chain":
        return (f"Report the exact sorted set of pathway stable ids that contain "
                f"(directly or through nested sub-pathways) at least one reaction "
                f"catalysed by UniProt accession {ast['accession']} "
                f"(catalystActivity role). Reactions where the accession is "
                f"merely a substrate do not count.")
    if kind == "complex_pathways":
        return (f"Report the exact sorted set of pathway stable ids in which the "
                f"complex {ast['complex']} participates (official complex-to-"
                f"pathway mapping). The complex is a unit: do not list pathways "
                f"of its individual components.")
    if kind == "entity_pmids":
        return (f"Report the exact sorted set of PubMed identifiers cited for the "
                f"Reactome physical entity linked to UniProt accession "
                f"{ast['accession']} (entity-level literature references).")
    if kind == "disease_subset":
        return ("From the candidate pathway list in the bundle, report the exact "
                "sorted subset that Reactome flags as disease pathways "
                "(HumanDiseasePathways). Identity by stable id only.")
    if kind == "top_level_subset":
        return ("From the candidate pathway list in the bundle, report the exact "
                "sorted subset that are top-level pathways (no parent in the "
                "Reactome pathway hierarchy).")
    if kind == "reaction_pathways":
        return (f"Report the exact sorted set of pathway stable ids in which "
                f"reaction {ast['reaction']} is a direct member. Ancestor "
                f"pathways reachable only through the hierarchy must not be "
                f"included; identity by stable id only.")
    if kind == "coreaction_pairs":
        return (f"Report the exact sorted set of UniProt accessions that appear "
                f"as reaction inputs in reactions where {ast['accession']} has "
                f"the catalyst role. Co-substrates in reactions where it is "
                f"merely an input do not count.")
    raise ValueError(f"unknown kind {kind!r}")


def _bundle(kind: str, ast: dict[str, Any], kb: L15KB, rng: Random) -> dict[str, Any]:
    b: dict[str, Any] = {"schema_doc": SCHEMA_DOC, "kind": kind}
    if kind == "pathway_children":
        parent = ast["parent"]
        edges = _rows(kb, "SELECT parent, child FROM m_pathway_edges WHERE parent=?",
                      (parent,))
        for gp in kb.m_parents(parent)[:2]:
            edges += _rows(kb,
                           "SELECT parent, child FROM m_pathway_edges WHERE parent=?",
                           (gp,))
        others = _rows(kb,
                       """SELECT parent, child FROM m_pathway_edges
                          WHERE parent != ? AND parent LIKE 'R-HSA-%'
                          ORDER BY child LIMIT 12""", (parent,))
        rng.shuffle(others)
        edges += others[:6]
        b["edges"] = sorted({(e["parent"], e["child"]) for e in edges},
                            key=lambda t: (t[0], t[1]))
        b["edges"] = [{"parent": p, "child": c} for p, c in b["edges"]]
        b["names"] = {pid: kb.m_pathway_name(pid)
                      for e in b["edges"] for pid in (e["parent"], e["child"])}
        b["parent"] = parent
    elif kind == "pathway_events":
        p = ast["pathway"]
        comps = kb.b_pathway_components(p)
        subs = G.pathway_subpathways(kb, p)
        b["pathway"] = p
        b["components"] = comps
        b["subpathways"] = subs
        b["subpathway_events"] = {s: G.pathway_reactions(kb, s) for s in subs[:6]}
    elif kind in ("reaction_inputs", "reaction_outputs"):
        sides = G.reaction_sides(kb, ast["reaction"])
        ctrls = G.reaction_controls(kb, ast["reaction"])
        parts: list[dict[str, Any]] = []
        for side, uris in (("left", sides["left"]), ("right", sides["right"])):
            for u in uris:
                rec = _pe_record(kb, u)
                rec["side"] = side
                parts.append(rec)
        for c in ctrls:
            if c.get("controller_pe"):
                rec = _pe_record(kb, c["controller_pe"])
                rec["side"] = "controller:" + (c["control_class"] or "")
                parts.append(rec)
        sibs = _rows(kb,
                     """SELECT DISTINCT pe_uri FROM b_reaction_sides
                        WHERE reaction != ? LIMIT 8""", (ast["reaction"],))
        for s in sibs[:4]:
            rec = _pe_record(kb, s["pe_uri"])
            rec["side"] = "not_a_participant"
            parts.append(rec)
        b["reaction"] = ast["reaction"]
        b["reaction_name"] = kb.b_event_name(ast["reaction"])
        b["participants"] = sorted(parts, key=lambda r: (r["pe_uri"], r["side"]))
    elif kind == "direction_repair":
        sides = G.reaction_sides(kb, ast["reaction"])
        left = [u for u in sides["left"]]
        right = [u for u in sides["right"]]
        swapped = ast["swapped_pe"]
        corrupt_left = [G._tail(u) for u in left if G._tail(u) != swapped]
        corrupt_right = [G._tail(u) for u in right] + (
            [swapped] if G._tail(swapped) in [G._tail(u) for u in left] else [])
        roles = _rows(kb,
                      "SELECT accession, role FROM m_reaction_roles WHERE reaction_id=?",
                      (ast["reaction"],))
        pe_xrefs = {G._tail(u): [x["accession"] for x in kb.b_pe_xrefs(u)]
                    for u in left + right}
        b["reaction"] = ast["reaction"]
        b["corrupted_record"] = {"left": sorted(corrupt_left),
                                 "right": sorted(corrupt_right)}
        b["role_evidence"] = roles
        b["pe_accessions"] = pe_xrefs
    elif kind == "reaction_catalyst":
        b["reaction"] = ast["reaction"]
        b["reaction_name"] = kb.b_event_name(ast["reaction"])
        ctrls = G.reaction_controls(kb, ast["reaction"])
        b["controls"] = [{"control_class": c["control_class"],
                          "control_type": c["control_type"],
                          "controller": G._tail(c["controller_pe"] or ""),
                          "controller_name": c.get("display_name", "")}
                         for c in ctrls if c.get("controller_pe")]
        sides = G.reaction_sides(kb, ast["reaction"])
        b["participants"] = [G._tail(u) for u in sides["left"] + sides["right"]]
    elif kind == "reaction_regulation":
        b["reaction"] = ast["reaction"]
        b["reaction_name"] = kb.b_event_name(ast["reaction"])
        ctrls = G.reaction_controls(kb, ast["reaction"])
        b["controls"] = [{"control_class": c["control_class"],
                          "control_type": c["control_type"],
                          "controller": G._tail(c["controller_pe"] or "")}
                         for c in ctrls if c.get("controller_pe")]
    elif kind == "complex_members":
        rows = kb.b_complex_components(ast["complex"])
        b["complex"] = ast["complex"]
        b["complex_name"] = (kb.b_pe(ast["complex"]) or {}).get("display_name", "")
        b["components_raw"] = [_pe_record(kb, r["component_uri"]) for r in rows]
        nested: dict[str, Any] = {}
        for r in rows:
            sub = kb.b_complex_components(r["component_uri"])
            if sub:
                nested[G._tail(r["component_uri"])] = [
                    _pe_record(kb, s["component_uri"]) for s in sub]
        b["nested_components"] = nested
    elif kind == "compartment_repair":
        pes = ast["pe_uris"]
        corrupted = G.corrupt_f09(kb, ast, rng)
        evidence: dict[str, str] = {}
        for u in pes:
            rows = _rows(kb,
                         "SELECT compartment, file FROM b_compartments WHERE pe_uri=?",
                         (u,))
            for r in rows:
                if r["file"] != ast["file"]:
                    evidence[G._tail(u)] = r["compartment"]
                    break
        b["file"] = ast["file"]
        b["corrupted_assignment"] = corrupted
        b["cross_file_evidence"] = evidence
        b["entities"] = [_pe_record(kb, u) for u in pes]
    elif kind == "uniprot_events":
        if G._f10_source(kb) == "m":
            rows = _rows(kb,
                         """SELECT event_id, event_name, species, evidence FROM m_xref_events
                            WHERE db='UniProt' AND accession=?""",
                         (ast["accession"],))
            b["mapping_rows"] = rows
            b["source"] = "UniProt2Reactome_All_Levels.txt (pinned)"
        else:
            pes = _rows(kb,
                        """SELECT DISTINCT pe_uri FROM b_xrefs
                           WHERE accession=? AND db LIKE 'UniProt%'""",
                        (ast["accession"],))
            pe_blocks: list[dict[str, Any]] = []
            edge_set: set[tuple[str, str]] = set()
            for pe in pes:
                uri = pe["pe_uri"]
                rxns = sorted({r["reaction"] for r in _rows(
                    kb, "SELECT DISTINCT reaction FROM b_reaction_sides WHERE pe_uri=?",
                    (uri,))} | {r["reaction"] for r in _rows(
                        kb, "SELECT DISTINCT reaction FROM b_controls WHERE controller_pe=?",
                        (uri,))})
                block: dict[str, Any] = {"pe_uri": G._tail(uri),
                                         "name": (kb.b_pe(uri) or {}).get("display_name", ""),
                                         "reactions": rxns}
                pe_blocks.append(block)
                seen: set[str] = set()
                frontier = list(rxns)
                while frontier:
                    cur = frontier.pop()
                    if cur in seen:
                        continue
                    seen.add(cur)
                    for row in _rows(kb,
                                     "SELECT DISTINCT pathway, component FROM b_pathway_components WHERE component=?",
                                     (cur,)):
                        edge_set.add((row["pathway"], row["component"]))
                        frontier.append(row["pathway"])
            b["pe_blocks"] = pe_blocks
            b["hierarchy_edges"] = [{"parent": p, "child": c} for p, c in sorted(edge_set)]
            b["derivation_rule"] = ("events = reactions the accession's PEs take part "
                                    "in (any side or as controller) UNION all ancestor "
                                    "pathways reachable via hierarchy edges")
            b["source"] = "pinned BioPAX (derived view; see REPORT.md §2)"
        b["accession"] = ast["accession"]
    elif kind == "modified_forms":
        rows = _rows(kb,
                     """SELECT DISTINCT x.pe_uri AS pe FROM b_xrefs x
                        WHERE x.accession=? AND x.db LIKE 'UniProt%' AND x.file=?""",
                     (ast["accession"], ast["file"]))
        pes = []
        for r in rows:
            rec = _pe_record(kb, r["pe"])
            rec["features"] = kb.b_pe_modifications(r["pe"])
            pes.append(rec)
        b["accession"] = ast["accession"]
        b["file"] = ast["file"]
        b["entities"] = sorted(pes, key=lambda r: r["pe_uri"])
    elif kind == "xref_consistency":
        rows = _rows(kb,
                     "SELECT db, accession FROM m_xref_events WHERE event_id=?",
                     (ast["event"],))
        decoys = _rows(kb,
                       """SELECT db, accession, event_id FROM m_xref_events
                          WHERE db=? AND event_id != ? LIMIT 6""",
                       (ast["db"], ast["event"]))
        b["event"] = ast["event"]
        b["db"] = ast["db"]
        b["rows_for_event_all_dbs"] = rows
        b["decoy_rows_other_events"] = decoys
    elif kind == "species_provenance":
        rows = _rows(kb,
                     """SELECT DISTINCT species, evidence FROM m_xref_events
                        WHERE db=? AND event_name=?""",
                     (G._f13_db(kb), ast["event_name"]))
        b["event_name"] = ast["event_name"]
        b["rows"] = rows
    elif kind == "reaction_pmids":
        rows = _rows(kb,
                     "SELECT reaction_id, pmid FROM m_reaction_pmids WHERE reaction_id=?",
                     (ast["reaction"],))
        name_row = _rows(kb, "SELECT event_name FROM m_xref_events WHERE event_id=? LIMIT 1",
                         (ast["reaction"],))
        decoys: list[dict[str, Any]] = []
        if name_row:
            decoys = _rows(kb,
                           """SELECT p.reaction_id, p.pmid FROM m_reaction_pmids p
                              JOIN m_xref_events x ON x.event_id=p.reaction_id
                              WHERE x.event_name=? AND p.reaction_id != ? LIMIT 6""",
                           (name_row[0]["event_name"], ast["reaction"]))
        b["reaction"] = ast["reaction"]
        b["rows"] = rows
        b["decoy_rows_same_name"] = decoys
    elif kind == "multihop_chain":
        roles = _rows(kb,
                      "SELECT reaction_id, role FROM m_reaction_roles WHERE accession=?",
                      (ast["accession"],))
        rxns = sorted({r["reaction_id"] for r in roles})
        edges: set[tuple[str, str]] = set()
        frontier = list(rxns)
        seen: set[str] = set()
        while frontier:
            cur = frontier.pop()
            if cur in seen:
                continue
            seen.add(cur)
            for p in kb.m_parents(cur):
                edges.add((p, cur))
                frontier.append(p)
            for row in _rows(kb,
                             "SELECT DISTINCT pathway_id FROM m_reaction_pathway WHERE reaction_id=?",
                             (cur,)):
                edges.add((row["pathway_id"], cur))
                frontier.append(row["pathway_id"])
        b["accession"] = ast["accession"]
        b["role_rows"] = roles
        b["hierarchy_edges"] = [{"parent": p, "child": c} for p, c in sorted(edges)]
    elif kind == "complex_pathways":
        b["complex"] = ast["complex"]
        b["rows"] = _rows(kb,
                          "SELECT pathway_id, top_level_pathway FROM m_complex_pathway WHERE complex_id=?",
                          (ast["complex"],))
        others = _rows(kb,
                       "SELECT complex_id, pathway_id FROM m_complex_pathway WHERE complex_id != ? LIMIT 8",
                       (ast["complex"],))
        b["decoy_rows_other_complexes"] = others
    elif kind == "entity_pmids":
        b["accession"] = ast["accession"]
        b["rows"] = _rows(kb,
                          "SELECT entity_id, pmid FROM m_entity_lit WHERE accession=?",
                          (ast["accession"],))
        b["decoy_rows_other_accessions"] = _rows(
            kb, "SELECT accession, pmid FROM m_entity_lit WHERE accession != ? LIMIT 8",
            (ast["accession"],))
    elif kind == "disease_subset":
        names = {r["id"]: kb.m_pathway_name(r["id"]) for r in
                 ({"id": c} for c in ast["candidates"])}
        b["candidates"] = [{"id": c, "name": names.get(c, "")}
                           for c in ast["candidates"]]
        b["disease_rows_sample"] = _rows(
            kb, "SELECT id FROM m_disease_pathways LIMIT 40")
    elif kind == "top_level_subset":
        names = {c: kb.m_pathway_name(c) for c in ast["candidates"]}
        b["candidates"] = [{"id": c, "name": names.get(c, "")}
                           for c in ast["candidates"]]
        edge_rows: list[dict[str, str]] = []
        for c in ast["candidates"]:
            edge_rows += [{"parent": p, "child": c} for p in kb.m_parents(c)]
            edge_rows += [{"parent": c, "child": ch} for ch in kb.m_children(c)[:3]]
        b["hierarchy_rows"] = edge_rows
    elif kind == "reaction_pathways":
        b["reaction"] = ast["reaction"]
        name_rows = _rows(kb, "SELECT name FROM m_reaction_names WHERE reaction_id=?",
                          (ast["reaction"],))
        b["reaction_name"] = name_rows[0]["name"] if name_rows else ""
        b["membership_rows"] = _rows(
            kb, "SELECT pathway_id FROM m_reaction_pathway WHERE reaction_id=?",
            (ast["reaction"],))
        edges: set[tuple[str, str]] = set()
        for r in b["membership_rows"]:
            for anc in G._ancestors_of(kb, r["pathway_id"]):
                edges.add((anc, r["pathway_id"]))
        b["ancestor_edges_of_member_pathways"] = [
            {"parent": p, "child": c} for p, c in sorted(edges)]
        b["decoy_rows_other_reactions"] = _rows(
            kb,
            "SELECT reaction_id, pathway_id FROM m_reaction_pathway WHERE reaction_id != ? LIMIT 10",
            (ast["reaction"],))
    elif kind == "coreaction_pairs":
        b["accession"] = ast["accession"]
        b["catalyst_rows"] = _rows(
            kb, "SELECT reaction_id FROM m_reaction_roles WHERE accession=? AND role='catalyst'",
            (ast["accession"],))
        rxns = [r["reaction_id"] for r in b["catalyst_rows"]]
        inputs: list[dict[str, str]] = []
        for r in rxns[:40]:
            inputs += _rows(kb,
                            "SELECT accession FROM m_reaction_roles WHERE reaction_id=? AND role='input'",
                            (r,))
        b["input_rows_for_catalyzed_reactions"] = inputs
        b["decoy_input_rows_other_reactions"] = _rows(
            kb,
            "SELECT reaction_id, accession FROM m_reaction_roles WHERE role='input' LIMIT 12")
    return b


def ground_candidate(kb: L15KB, cand_ast: dict[str, Any], family_id: str,
                     base_goal_id: str, instance_id: str, inst_idx: int,
                     seed: int) -> dict[str, Any]:
    spec = BY_ID[family_id]
    kind = cand_ast["kind"]
    rng = Random(f"ground|{family_id}|{base_goal_id}|{inst_idx}|{seed}")
    bundle = _bundle(kind, cand_ast, kb, rng)
    user_goal = _goal_text(kind, cand_ast, kb)
    public = {
        "user_goal": user_goal,
        "allowed_tools": list(ALLOWED_TOOLS),
        "budget": {"actions": 8, "probes": 0},
        "evidence_bundle": bundle,
    }
    return public
