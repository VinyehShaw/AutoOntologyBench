"""L15 goal machinery: per-kind proposal, reference answer, mutation,
counterfactual and public-bundle builders over the pinned Reactome store.

Conventions:
- Physical entity identity = the BioPAX URI tail (``pe_uri``), stable within a
  release; display names are shown in bundles for readability only.
- Answers are exact sorted sets (or maps) computed from the store; mutations
  must flip the verifier verdict; counterfactuals demonstrate the answer is
  semantics-sensitive (a relation/direction/compartment change changes it).
"""
from __future__ import annotations

from random import Random
from typing import Any, Callable

from benchmark.plugins.source.l15.kb import L15KB

HUMAN = "Homo sapiens"


def _tails(uris: list[str]) -> list[str]:
    return sorted({u.rsplit("/", 1)[-1].split("#")[-1] for u in uris})


def _tail(uri: str) -> str:
    return uri.rsplit("/", 1)[-1].split("#")[-1]


# ---------------------------------------------------------------------------
# universe queries
# ---------------------------------------------------------------------------


def human_pathways(kb: L15KB) -> list[str]:
    rows = kb._q("SELECT DISTINCT id FROM b_events WHERE class='Pathway'")
    return sorted(r["id"] for r in rows if str(r["id"]).startswith("R-HSA"))


def pathway_reactions(kb: L15KB, pathway: str) -> list[str]:
    rows = kb._q(
        """SELECT DISTINCT c.component AS id FROM b_pathway_components c
           JOIN b_events e ON e.id = c.component AND e.file = c.file
           WHERE c.pathway=? AND e.class != 'Pathway'""", (pathway,))
    return sorted(r["id"] for r in rows)


def pathway_subpathways(kb: L15KB, pathway: str) -> list[str]:
    rows = kb._q(
        """SELECT DISTINCT c.component AS id FROM b_pathway_components c
           JOIN b_events e ON e.id = c.component AND e.file = c.file
           WHERE c.pathway=? AND e.class = 'Pathway'""", (pathway,))
    return sorted(r["id"] for r in rows)


def reactions_with_sides(kb: L15KB) -> list[str]:
    rows = kb._q(
        """SELECT reaction FROM b_reaction_sides GROUP BY reaction
           HAVING SUM(side='left') > 0 AND SUM(side='right') > 0""")
    return sorted(r["reaction"] for r in rows)


def reaction_sides(kb: L15KB, reaction: str) -> dict[str, list[str]]:
    out = {"left": [], "right": []}
    for r in kb._q("SELECT side, pe_uri FROM b_reaction_sides WHERE reaction=?",
                   (reaction,)):
        out[r["side"]].append(r["pe_uri"])
    return {k: sorted(set(v)) for k, v in out.items()}


def reaction_controls(kb: L15KB, reaction: str) -> list[dict[str, Any]]:
    return kb.b_reaction_controls(reaction)


def complexes(kb: L15KB, min_members: int = 2) -> list[str]:
    rows = kb._q(
        """SELECT complex_uri, COUNT(DISTINCT component_uri) AS n
           FROM b_complex_components GROUP BY complex_uri HAVING n>=?""",
        (min_members,))
    return sorted(r["complex_uri"] for r in rows)


def pe_label(kb: L15KB, uri: str) -> str:
    pe = kb.b_pe(uri)
    name = pe["display_name"] if pe else ""
    comp = kb.b_compartment(uri)
    return f"{name} [{comp}]" if comp else name


# ---------------------------------------------------------------------------
# kind registry
# ---------------------------------------------------------------------------


def propose_f01(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f01", lambda k: k._q(
        """SELECT parent, COUNT(DISTINCT child) AS n FROM m_pathway_edges
           WHERE parent LIKE 'R-HSA-%' GROUP BY parent
           HAVING n BETWEEN 2 AND 30"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"parent": pick["parent"]}


def answer_f01(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": kb.m_children(ast["parent"])}


def agree_f01(kb: L15KB, ast: dict[str, Any]) -> bool:
    b = set(kb.b_pathway_components(ast["parent"]))
    m = set(kb.m_children(ast["parent"]))
    return bool(b) and b == m


def mutate_f01(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = list(ref["answer_ids"])
    parents = kb.m_parents(ast["parent"]) or kb.m_all_parents()
    foreign: list[str] = []
    for gp in parents[:4]:
        foreign += [c for c in kb.m_children(gp) if c not in ans]
    swap = sorted(foreign)[0] if foreign else "R-HSA-1"
    return {"answer_ids": sorted(ans[1:] + [swap]) if len(ans) > 1
            else sorted(ans + [swap])}


def counter_f01(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    parents = kb.m_parents(ast["parent"])
    for gp in parents[:3]:
        alt = kb.m_children(gp)
        if alt != ref["answer_ids"]:
            return {"counterfactual": f"parent_swapped:{gp}",
                    "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f02(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f02", lambda k: k._q(
        """SELECT c.pathway AS p, COUNT(DISTINCT c.component) AS n
           FROM b_pathway_components c JOIN b_events e
             ON e.id = c.component AND e.file = c.file
           WHERE e.class != 'Pathway' AND c.pathway LIKE 'R-HSA-%'
           GROUP BY c.pathway HAVING n BETWEEN 3 AND 40"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"pathway": pick["p"]}


def answer_f02(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": pathway_reactions(kb, ast["pathway"])}


def agree_f02(kb: L15KB, ast: dict[str, Any]) -> bool:
    m_rows = kb._q(
        "SELECT DISTINCT reaction_id FROM m_reaction_pathway WHERE pathway_id=?",
        (ast["pathway"],))
    m = sorted(r["reaction_id"] for r in m_rows)
    b = pathway_reactions(kb, ast["pathway"])
    if not m:
        return False
    return sorted(set(m) & set(b)) == sorted(b)


def mutate_f02(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = list(ref["answer_ids"])
    subs = pathway_subpathways(kb, ast["pathway"])
    nested: list[str] = []
    for s in subs[:3]:
        nested += pathway_reactions(kb, s)
    nested = [n for n in sorted(nested) if n not in ans]
    if nested:
        return {"answer_ids": sorted(ans + [nested[0]])}
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans + ["R-HSA-1"]}


def counter_f02(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    subs = pathway_subpathways(kb, ast["pathway"])
    for s in subs[:3]:
        nested = pathway_reactions(kb, s)
        if nested:
            alt = sorted(set(ref["answer_ids"]) | set(nested))
            if alt != ref["answer_ids"]:
                return {"counterfactual": f"include_nested_subpathway:{s}",
                        "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    alt = sorted(set(ref["answer_ids"][1:]))
    if alt != ref["answer_ids"]:
        return {"counterfactual": "drop_direct_event", "answer_diff": 1}
    return None


def _propose_reaction(kb: L15KB, rng: Random, min_side: int = 1) -> dict[str, Any] | None:
    rows = _universe(kb, "reaction_sides", lambda k: k._q(
        """SELECT reaction, SUM(side='left') AS nl, SUM(side='right') AS nr
           FROM b_reaction_sides GROUP BY reaction
           HAVING nl>=? AND nr>=1""", (min_side,)))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"reaction": pick["reaction"]}


def answer_f03(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": _tails(reaction_sides(kb, ast["reaction"])["left"])}


def answer_f04(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": _tails(reaction_sides(kb, ast["reaction"])["right"])}


def _roles_agree(kb: L15KB, ast: dict[str, Any], m_role: str,
                 b_uris: list[str]) -> tuple[bool, dict[str, Any]]:
    """Agreement between BioPAX side/control PEs and ProteinRoleReaction rows:
    the accession sets of *protein* PEs must match the mapping rows."""
    mapped = {"catalystActivity": "catalyst"}.get(m_role, m_role)
    rows = kb._q(
        "SELECT DISTINCT accession FROM m_reaction_roles WHERE reaction_id=? AND role=?",
        (ast["reaction"], mapped))
    m_accs = sorted(r["accession"] for r in rows)
    b_accs: set[str] = set()
    for uri in b_uris:
        for x in kb.b_pe_xrefs(uri):
            if x["db"].lower().startswith("uniprot"):
                b_accs.add(x["accession"])
    return (sorted(b_accs) == m_accs), {"m": m_accs, "b": sorted(b_accs)}


def agree_f03(kb: L15KB, ast: dict[str, Any]) -> bool:
    sides = reaction_sides(kb, ast["reaction"])
    ok, _d = _roles_agree(kb, ast, "input", sides["left"])
    return ok


def agree_f04(kb: L15KB, ast: dict[str, Any]) -> bool:
    sides = reaction_sides(kb, ast["reaction"])
    ok, _d = _roles_agree(kb, ast, "output", sides["right"])
    return ok


def _mutate_side(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any],
                 other_side: str) -> dict[str, Any]:
    ans = list(ref["answer_ids"])
    other = _tails(reaction_sides(kb, ast["reaction"])[other_side])
    foreign = [o for o in other if o not in ans]
    if foreign:
        return {"answer_ids": sorted(ans + [foreign[0]])}
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans}


def mutate_f03(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    sides = reaction_sides(kb, ast["reaction"])
    ctrls = [c["controller_pe"] for c in reaction_controls(kb, ast["reaction"])
             if c.get("controller_pe")]
    foreign = [u for u in _tails([c for c in ctrls]) if u not in ref["answer_ids"]]
    if foreign:
        return {"answer_ids": sorted(ref["answer_ids"] + [foreign[0]])}
    return _mutate_side(kb, ast, ref, "right")


def mutate_f04(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    return _mutate_side(kb, ast, ref, "left")


def counter_f03(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    alt = answer_f04(kb, ast)["answer_ids"]
    if alt != ref["answer_ids"]:
        return {"counterfactual": "direction_flipped",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def counter_f04(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    alt = answer_f03(kb, ast)["answer_ids"]
    if alt != ref["answer_ids"]:
        return {"counterfactual": "direction_flipped",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f05(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f05", lambda k: k._q(
        """SELECT reaction, SUM(side='left') AS nl, SUM(side='right') AS nr
           FROM b_reaction_sides GROUP BY reaction HAVING nl>=1 AND nr>=1"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    r = pick["reaction"]
    sides = reaction_sides(kb, r)
    left, right = sides["left"], sides["right"]
    if not left or not right:
        return None
    swapped = left[rng.randrange(len(left))]
    return {"reaction": r, "swapped_pe": _tail(swapped)}


def answer_f05(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    sides = reaction_sides(kb, ast["reaction"])
    return {"assignment": {
        "left": _tails(sides["left"]), "right": _tails(sides["right"])}}


def corrupt_f05(ast: dict[str, Any]) -> dict[str, Any]:
    """The corrupted public record: swapped_pe moved to the wrong side."""
    return {"reaction": ast["reaction"], "swapped_pe": ast["swapped_pe"]}


def mutate_f05(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    left = list(ref["assignment"]["left"])
    right = list(ref["assignment"]["right"])
    swapped = ast["swapped_pe"]
    if swapped in left:
        left.remove(swapped)
        right.append(swapped)
    elif swapped in right:
        right.remove(swapped)
        left.append(swapped)
    return {"assignment": {"left": sorted(left), "right": sorted(right)}}


def counter_f05(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    mut = mutate_f05(kb, ast, ref)
    if mut != ref:
        return {"counterfactual": "side_swap_changes_assignment",
                "answer_diff": 2}
    return None


def _reactions_with_control(kb: L15KB, cls: str | None = None) -> list[str]:
    if cls:
        rows = kb._q(
            "SELECT DISTINCT reaction FROM b_controls WHERE control_class=?",
            (cls,))
    else:
        rows = kb._q("SELECT DISTINCT reaction FROM b_controls")
    return sorted(r["reaction"] for r in rows)


def propose_f06(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    cands = _universe(kb, "f06", lambda k: _reactions_with_control(k, "Catalysis"))
    if not cands:
        return None
    return {"reaction": cands[rng.randrange(len(cands))]}


def answer_f06(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = [c for c in reaction_controls(kb, ast["reaction"])
            if c["control_class"] == "Catalysis" and c.get("controller_pe")]
    return {"answer_ids": _tails([c["controller_pe"] for c in rows])}


def agree_f06(kb: L15KB, ast: dict[str, Any]) -> bool:
    rows = [c for c in reaction_controls(kb, ast["reaction"])
            if c["control_class"] == "Catalysis" and c.get("controller_pe")]
    ok, _d = _roles_agree(kb, ast, "catalystActivity",
                          [c["controller_pe"] for c in rows])
    return ok


def mutate_f06(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    regs = [c["controller_pe"] for c in reaction_controls(kb, ast["reaction"])
            if c["control_class"] != "Catalysis" and c.get("controller_pe")]
    foreign = [u for u in _tails(regs) if u not in ref["answer_ids"]]
    if foreign:
        return {"answer_ids": sorted(ref["answer_ids"] + [foreign[0]])}
    ans = ref["answer_ids"]
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans}


def counter_f06(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    allc = [c["controller_pe"] for c in reaction_controls(kb, ast["reaction"])
            if c.get("controller_pe")]
    if not allc:
        return None
    alt = _tails(allc)
    if alt != ref["answer_ids"]:
        return {"counterfactual": "catalysis_vs_regulation_merged",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f07(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f07", lambda k: k._q(
        """SELECT DISTINCT reaction FROM b_controls
           WHERE control_class != 'Catalysis' AND control_type != ''"""))
    rows = list(rows)
    if not cands:
        return None
    return {"reaction": cands[rng.randrange(len(cands))]}


def answer_f07(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = [c for c in reaction_controls(kb, ast["reaction"])
            if c["control_class"] != "Catalysis" and c.get("controller_pe")]
    return {"pairs": sorted({_tail(c["controller_pe"]) + "|" + c["control_type"]
                             for c in rows})}


def agree_f07(kb: L15KB, ast: dict[str, Any]) -> bool:
    rows = [c for c in reaction_controls(kb, ast["reaction"])
            if c["control_class"] != "Catalysis" and c.get("controller_pe")]
    if not rows:
        return False
    b_pairs: set[tuple[str, str]] = set()
    for c in rows:
        accs = {x["accession"] for x in kb.b_pe_xrefs(c["controller_pe"])
                if x["db"].lower().startswith("uniprot")}
        for a in accs:
            b_pairs.add((a, c["control_type"]))
    m_pairs: set[tuple[str, str]] = set()
    for r in kb._q(
            "SELECT accession, role FROM m_reaction_roles WHERE reaction_id=?",
            (ast["reaction"],)):
        if r["role"] == "positive":
            m_pairs.add((r["accession"], "ACTIVATION"))
        elif r["role"] == "negative":
            m_pairs.add((r["accession"], "INHIBITION"))
    return bool(m_pairs) and b_pairs == m_pairs


def mutate_f07(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    pairs = list(ref["pairs"])
    flipped = []
    for p in pairs:
        pe, _, t = p.rpartition("|")
        nt = {"ACTIVATION": "INHIBITION", "INHIBITION": "ACTIVATION"}.get(t, "INHIBITION")
        flipped.append(pe + "|" + nt)
    return {"pairs": sorted(flipped)}


def counter_f07(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    mut = mutate_f07(kb, ast, ref)
    if mut["pairs"] != ref["pairs"]:
        return {"counterfactual": "control_type_flipped",
                "answer_diff": len(set(mut["pairs"]) ^ set(ref["pairs"]))}
    return None


def propose_f08(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    cands = _universe(kb, "f08", lambda k: complexes(k, 2))
    if not cands:
        return None
    return {"complex": cands[rng.randrange(len(cands))]}


def answer_f08(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb.b_complex_components(ast["complex"])
    return {"answer_ids": _tails([r["component_uri"] for r in rows])}


def mutate_f08(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    rows = kb.b_complex_components(ast["complex"])
    nested: list[str] = []
    for r in rows:
        nested += [c["component_uri"] for c in kb.b_complex_components(r["component_uri"])]
    foreign = [u for u in _tails(nested) if u not in ref["answer_ids"]]
    if foreign:
        return {"answer_ids": sorted(ref["answer_ids"] + [foreign[0]])}
    ans = ref["answer_ids"]
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans}


def counter_f08(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb.b_complex_components(ast["complex"])
    flat: set[str] = set(ref["answer_ids"])
    for r in rows:
        flat |= set(_tails([c["component_uri"]
                            for c in kb.b_complex_components(r["component_uri"])]))
    alt = sorted(flat)
    if alt != ref["answer_ids"]:
        return {"counterfactual": "components_flattened_recursively",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def _pe_with_compartment(kb: L15KB) -> list[str]:
    rows = kb._q(
        """SELECT pe_uri, COUNT(DISTINCT compartment) AS n FROM b_compartments
           GROUP BY pe_uri""")
    return sorted(r["pe_uri"] for r in rows)


def propose_f09(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f09", lambda k: k._q(
        """SELECT file, COUNT(DISTINCT pe_uri) AS n FROM b_compartments
           GROUP BY file HAVING n BETWEEN 4 AND 12"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    pes = [r["pe_uri"] for r in kb._q(
        "SELECT DISTINCT pe_uri FROM b_compartments WHERE file=?", (pick["file"],))]
    return {"file": pick["file"], "pe_uris": sorted(pes)[:12]}


def answer_f09(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"assignment": {_tail(u): kb.b_compartment(u) for u in ast["pe_uris"]}}


def corrupt_f09(kb: L15KB, ast: dict[str, Any], rng: Random) -> dict[str, Any]:
    ref = answer_f09(kb, ast)["assignment"]
    comps = sorted(set(ref.values()))
    if len(comps) < 2:
        comps = comps + ["cytosol"]
    wrong = {k: (comps[(comps.index(v) + 1) % len(comps)]) for k, v in ref.items()}
    return wrong


def mutate_f09(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    wrong = dict(ref["assignment"])
    keys = sorted(wrong)
    k0 = keys[0]
    others = sorted({v for k, v in wrong.items() if v != wrong[k0]})
    wrong[k0] = others[0] if others else "nucleoplasm"
    return {"assignment": wrong}


def counter_f09(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    merged = {k: "cytosol" for k in ref["assignment"]}
    if merged != ref["assignment"]:
        return {"counterfactual": "compartments_merged_to_cytosol",
                "answer_diff": sum(1 for k in ref["assignment"]
                                   if merged[k] != ref["assignment"][k])}
    return None


def _f10_source(kb: L15KB) -> str:
    """'m' when the published UniProt All_Levels mapping is pinned, else 'd'
    (BioPAX-derived; see store.py d_uniprot_events, marked derived)."""
    try:
        n = kb._q("SELECT COUNT(*) AS c FROM m_xref_events WHERE db='UniProt'")[0]["c"]
        if n:
            return "m"
    except Exception:
        pass
    try:
        if kb._q("SELECT COUNT(*) AS n FROM d_uniprot_events")[0]["n"]:
            return "d"
    except Exception:
        pass
    return "m"


def propose_f10(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    if _f10_source(kb) == "m":
        rows = _universe(kb, "f10m", lambda k: k._q(
            """SELECT accession, COUNT(DISTINCT event_id) AS n FROM m_xref_events
               WHERE db='UniProt' AND species='Homo sapiens'
               GROUP BY accession HAVING n BETWEEN 2 AND 25"""))
    else:
        rows = _universe(kb, "f10d", lambda k: k._q(
            """SELECT accession, COUNT(DISTINCT event_id) AS n FROM d_uniprot_events
               GROUP BY accession HAVING n BETWEEN 2 AND 25"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"accession": pick["accession"]}


def answer_f10(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    if _f10_source(kb) == "m":
        rows = kb._q(
            """SELECT DISTINCT event_id FROM m_xref_events
               WHERE db='UniProt' AND accession=? AND species='Homo sapiens'""",
            (ast["accession"],))
    else:
        rows = kb._q(
            "SELECT DISTINCT event_id FROM d_uniprot_events WHERE accession=?",
            (ast["accession"],))
    return {"answer_ids": sorted(r["event_id"] for r in rows)}


def mutate_f10(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = list(ref["answer_ids"])
    if _f10_source(kb) == "m":
        rows = kb._q(
            """SELECT DISTINCT event_id FROM m_xref_events
               WHERE db='UniProt' AND accession=? AND species!='Homo sapiens'""",
            (ast["accession"],))
        foreign = [r["event_id"] for r in rows if r["event_id"] not in ans]
    else:
        rows = kb._q(
            """SELECT DISTINCT event_id FROM d_uniprot_events
               WHERE accession != ?""", (ast["accession"],))
        foreign = [r["event_id"] for r in rows if r["event_id"] not in ans]
    if foreign:
        return {"answer_ids": sorted(ans + [sorted(foreign)[0]])}
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans}


def counter_f10(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    if _f10_source(kb) == "m":
        ans = set(ref["answer_ids"])
        top_level = {e for e in ans
                     if kb._q("SELECT 1 FROM m_pathways WHERE id=?", (e,))
                     and not kb._q("SELECT 1 FROM m_pathway_edges WHERE child=?",
                                   (e,))}
        if top_level and top_level != ans:
            alt = sorted(ans - top_level)
            return {"counterfactual": "top_level_pathways_excluded",
                    "answer_diff": len(top_level)}
        allsp = kb.m_uniprot_events(ast["accession"])
        if allsp != ref["answer_ids"]:
            return {"counterfactual": "species_filter_removed",
                    "answer_diff": len(set(allsp) ^ ans)}
        return None
    rows = kb._q(
        "SELECT DISTINCT event_id FROM d_uniprot_events WHERE accession=?",
        (ast["accession"],))
    direct = sorted(r["event_id"] for r in rows
                    if not kb._q("SELECT 1 FROM b_pathway_components WHERE pathway=? LIMIT 1",
                                 (r["event_id"],)))
    if direct and direct != ref["answer_ids"]:
        return {"counterfactual": "ancestor_pathways_excluded",
                "answer_diff": len(set(direct) ^ set(ref["answer_ids"]))}
    return None


def propose_f11(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f11", lambda k: k._q(
        """SELECT x.accession AS acc, m.file AS file, COUNT(DISTINCT m.pe_uri) AS n
           FROM b_modifications m
           JOIN b_xrefs x ON x.pe_uri = m.pe_uri
           WHERE x.db LIKE 'UniProt%'
           GROUP BY x.accession, m.file HAVING n BETWEEN 2 AND 15"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"accession": pick["acc"], "file": pick["file"]}


def answer_f11(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        """SELECT DISTINCT m.pe_uri AS pe FROM b_modifications m
           JOIN b_xrefs x ON x.pe_uri = m.pe_uri
           WHERE x.accession=? AND x.db LIKE 'UniProt%' AND m.file=?""",
        (ast["accession"], ast["file"]))
    return {"answer_ids": _tails([r["pe"] for r in rows])}


def mutate_f11(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        """SELECT DISTINCT pe_uri AS pe FROM b_xrefs
           WHERE accession=? AND db LIKE 'UniProt%'""",
        (ast["accession"],))
    allpe = _tails([r["pe"] for r in rows])
    foreign = [p for p in allpe if p not in ref["answer_ids"]]
    if foreign:
        return {"answer_ids": sorted(ref["answer_ids"] + [foreign[0]])}
    ans = ref["answer_ids"]
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans}


def counter_f11(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        """SELECT DISTINCT pe_uri AS pe FROM b_xrefs
           WHERE accession=? AND db LIKE 'UniProt%'""",
        (ast["accession"],))
    alt = _tails([r["pe"] for r in rows])
    if alt != ref["answer_ids"]:
        return {"counterfactual": "modified_and_unmodified_merged",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f12(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f12", lambda k: k._q(
        """SELECT event_id, db, COUNT(DISTINCT accession) AS n FROM m_xref_events
           WHERE event_id LIKE 'R-HSA-%' AND db IN ('ChEBI','NCBI')
           GROUP BY event_id, db HAVING n BETWEEN 2 AND 25"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"event": pick["event_id"], "db": pick["db"]}


def answer_f12(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": kb.m_event_xrefs(ast["db"], ast["event"])}


def _norm_acc(db: str, acc: str) -> str:
    a = acc.strip()
    if db == "ChEBI":
        return a[6:] if a.upper().startswith("CHEBI:") else a
    if db == "NCBI":
        for pre in ("NCBI Gene:", "NCBI_Gene:", "Entrez:", "GeneID:"):
            if a.startswith(pre):
                return a[len(pre):]
    return a


def agree_f12(kb: L15KB, ast: dict[str, Any]) -> bool:
    sides = kb._q(
        "SELECT DISTINCT pe_uri FROM b_reaction_sides WHERE reaction=?",
        (ast["event"],))
    if not sides:
        return True
    db_prefix = {"ChEBI": "CHEBI", "Ensembl": "ENSEMBL", "NCBI": "NCBI"}.get(ast["db"], ast["db"])
    b_accs: set[str] = set()
    for r in sides:
        for x in kb.b_pe_xrefs(r["pe_uri"]):
            if x["db"].upper().startswith(db_prefix.upper()[:4]):
                b_accs.add(_norm_acc(ast["db"], x["accession"]))
    m_accs = {_norm_acc(ast["db"], a) for a in answer_f12(kb, ast)["answer_ids"]}
    return bool(b_accs) and b_accs == m_accs


def mutate_f12(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    rows = kb._q(
        """SELECT DISTINCT accession FROM m_xref_events
           WHERE db=? AND event_id != ? LIMIT 500""",
        (ast["db"], ast["event"]))
    foreign = sorted(r["accession"] for r in rows if r["accession"] not in ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f12(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    others = {"ChEBI": "NCBI", "NCBI": "ChEBI"}.get(ast["db"], "NCBI")
    alt = kb.m_event_xrefs(others, ast["event"])
    if alt and alt != ref["answer_ids"]:
        return {"counterfactual": f"db_swapped:{others}",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def _f13_db(kb: L15KB) -> str:
    for db in ("UniProt", "ChEBI"):
        try:
            if kb._q("SELECT COUNT(*) AS n FROM m_xref_events WHERE db=?", (db,))[0]["n"]:
                return db
        except Exception:
            continue
    return "UniProt"


def propose_f13(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    db = _f13_db(kb)
    rows = _universe(kb, "f13:" + db, lambda k: k._q(
        """SELECT event_name, COUNT(DISTINCT species) AS n FROM m_xref_events
           WHERE db=? AND event_name != ''
           GROUP BY event_name HAVING n BETWEEN 2 AND 20
           AND SUM(species='Homo sapiens') > 0""", (db,)))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"event_name": pick["event_name"]}


def answer_f13(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        """SELECT DISTINCT species FROM m_xref_events
           WHERE db=? AND event_name=?""",
        (_f13_db(kb), ast["event_name"]))
    return {"answer_ids": sorted(r["species"] for r in rows)}


def mutate_f13(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    all_species = _universe(kb, "species_univ", lambda k: [
        {"species": r["species"]} for r in k._q("SELECT DISTINCT species FROM m_pathways")])
    foreign = sorted(r["species"] for r in all_species if r["species"] not in ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f13(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        """SELECT DISTINCT species FROM m_xref_events
           WHERE db=? AND event_name=? AND evidence!='IEA'""",
        (_f13_db(kb), ast["event_name"]))
    alt = sorted(r["species"] for r in rows)
    if alt and alt != ref["answer_ids"]:
        return {"counterfactual": "iea_inferred_species_excluded",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f14(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f14", lambda k: k._q(
        """SELECT reaction_id, COUNT(DISTINCT pmid) AS n FROM m_reaction_pmids
           WHERE reaction_id LIKE 'R-HSA-%'
           GROUP BY reaction_id HAVING n BETWEEN 1 AND 15"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"reaction": pick["reaction_id"]}


def answer_f14(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": kb.m_event_pmids(ast["reaction"])}


def mutate_f14(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = list(ref["answer_ids"])
    rows = kb._q(
        """SELECT DISTINCT pmid FROM m_reaction_pmids WHERE reaction_id != ?
           AND pmid NOT IN (SELECT pmid FROM m_reaction_pmids WHERE reaction_id=?)
           LIMIT 3""",
        (ast["reaction"], ast["reaction"]))
    foreign = [str(r["pmid"]) for r in rows]
    if foreign:
        return {"answer_ids": sorted(ans + [foreign[0]])}
    return {"answer_ids": sorted(ans[1:])} if len(ans) > 1 else {"answer_ids": ans + ["1"]}


def counter_f14(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    name_row = kb._q("SELECT name FROM m_reaction_names WHERE reaction_id=? LIMIT 1",
                     (ast["reaction"],))
    if name_row and name_row[0]["name"]:
        rows = kb._q(
            """SELECT DISTINCT reaction_id FROM m_reaction_names
               WHERE name=? AND reaction_id != ? LIMIT 3""",
            (name_row[0]["name"], ast["reaction"]))
        alt_pmids: set[str] = set()
        for r in rows[:3]:
            alt_pmids |= set(kb.m_event_pmids(r["reaction_id"]))
        if alt_pmids:
            alt = sorted(set(ref["answer_ids"]) | alt_pmids)
            if alt != ref["answer_ids"]:
                return {"counterfactual": "same_name_other_reaction_pmids_merged",
                        "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    pws = [r["pathway_id"] for r in kb._q(
        "SELECT DISTINCT pathway_id FROM m_reaction_pathway WHERE reaction_id=?",
        (ast["reaction"],))]
    for p in pws[:2]:
        sibs = [r["reaction_id"] for r in kb._q(
            "SELECT DISTINCT reaction_id FROM m_reaction_pathway WHERE pathway_id=?",
            (p,))]
        for sib in sibs[:4]:
            if sib == ast["reaction"]:
                continue
            alt = sorted(set(ref["answer_ids"]) | set(kb.m_event_pmids(sib)))
            if alt != ref["answer_ids"]:
                return {"counterfactual": f"sibling_reaction_pmids_merged:{sib}",
                        "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f15(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f15", lambda k: k._q(
        """SELECT accession, COUNT(DISTINCT reaction_id) AS n FROM m_reaction_roles
           WHERE role='catalyst' AND reaction_id LIKE 'R-HSA-%'
           GROUP BY accession HAVING n BETWEEN 1 AND 8"""))
    if not rows:
        return None
    pick = rows[rng.randrange(len(rows))]
    return {"accession": pick["accession"]}


_CLOSURE: dict[int, dict[str, set[str]]] = {}


def _pathway_closure(kb: L15KB) -> dict[str, set[str]]:
    key = id(kb)
    if key not in _CLOSURE:
        parents: dict[str, set[str]] = {}
        for r in kb._q("SELECT parent, child FROM m_pathway_edges"):
            parents.setdefault(r["child"], set()).add(r["parent"])
        _CLOSURE[key] = parents
    return _CLOSURE[key]


def _ancestors_of(kb: L15KB, event: str) -> set[str]:
    parents = _pathway_closure(kb)
    seen: set[str] = set()
    stack = [event]
    while stack:
        cur = stack.pop()
        for p in parents.get(cur, ()):
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return seen


def _reaction_pathways(kb: L15KB, reaction: str) -> set[str]:
    return {r["pathway_id"] for r in kb._q(
        "SELECT DISTINCT pathway_id FROM m_reaction_pathway WHERE reaction_id=?",
        (reaction,))}


def answer_f15(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        """SELECT DISTINCT reaction_id FROM m_reaction_roles
           WHERE accession=? AND role='catalyst'""",
        (ast["accession"],))
    out: set[str] = set()
    for r in rows:
        direct = _reaction_pathways(kb, r["reaction_id"])
        out |= direct
        for p in direct:
            out |= _ancestors_of(kb, p)
    return {"answer_ids": sorted(out)}


def agree_f15(kb: L15KB, ast: dict[str, Any]) -> bool:
    rows = kb._q(
        """SELECT DISTINCT reaction_id FROM m_reaction_roles
           WHERE accession=? AND role='catalyst' AND reaction_id LIKE 'R-HSA-%'""",
        (ast["accession"],))
    for r in rows:
        cat = [c for c in reaction_controls(kb, r["reaction_id"])
               if c["control_class"] == "Catalysis" and c.get("controller_pe")]
        b_accs: set[str] = set()
        for c in cat:
            for x in kb.b_pe_xrefs(c["controller_pe"]):
                if x["db"].lower().startswith("uniprot"):
                    b_accs.add(x["accession"])
        if b_accs and ast["accession"] not in b_accs:
            return False
    return True


def mutate_f15(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    rows = kb._q(
        """SELECT DISTINCT accession FROM m_reaction_roles
           WHERE role='input' AND accession != ? LIMIT 20""",
        (ast["accession"],))
    for r in rows:
        alt = answer_f15(kb, {"accession": r["accession"]})["answer_ids"]
        foreign = [a for a in alt if a not in ans]
        if foreign:
            return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f15(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        """SELECT DISTINCT reaction_id FROM m_reaction_roles
           WHERE accession=? AND role='input'""",
        (ast["accession"],))
    out: set[str] = set()
    for r in rows:
        out |= _ancestors_of(kb, r["reaction_id"])
    alt = sorted(out)
    if alt != ref["answer_ids"]:
        return {"counterfactual": "role_changed_to_input",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None



_UNIVERSE: dict = {}


def _universe(kb: L15KB, key: str, builder) -> list:
    if key not in _UNIVERSE:
        _UNIVERSE[key] = builder(kb)
    return _UNIVERSE[key]

class KindOps:
    def __init__(self, propose, answer, mutate, counter, agree=None):
        self.propose = propose
        self.answer = answer
        self.mutate = mutate
        self.counter = counter
        self.agree = agree


KINDS: dict[str, KindOps] = {
    "pathway_children": KindOps(propose_f01, answer_f01, mutate_f01, counter_f01, agree_f01),
    "pathway_events": KindOps(propose_f02, answer_f02, mutate_f02, counter_f02, agree_f02),
    "reaction_inputs": KindOps(_propose_reaction, answer_f03, mutate_f03, counter_f03, agree_f03),
    "reaction_outputs": KindOps(_propose_reaction, answer_f04, mutate_f04, counter_f04, agree_f04),
    "direction_repair": KindOps(propose_f05, answer_f05, mutate_f05, counter_f05),
    "reaction_catalyst": KindOps(propose_f06, answer_f06, mutate_f06, counter_f06, agree_f06),
    "reaction_regulation": KindOps(propose_f07, answer_f07, mutate_f07, counter_f07, agree_f07),
    "complex_members": KindOps(propose_f08, answer_f08, mutate_f08, counter_f08),
    "compartment_repair": KindOps(propose_f09, answer_f09, mutate_f09, counter_f09),
    "uniprot_events": KindOps(propose_f10, answer_f10, mutate_f10, counter_f10),
    "modified_forms": KindOps(propose_f11, answer_f11, mutate_f11, counter_f11),
    "xref_consistency": KindOps(propose_f12, answer_f12, mutate_f12, counter_f12, agree_f12),
    "species_provenance": KindOps(propose_f13, answer_f13, mutate_f13, counter_f13),
    "reaction_pmids": KindOps(propose_f14, answer_f14, mutate_f14, counter_f14),
    "multihop_chain": KindOps(propose_f15, answer_f15, mutate_f15, counter_f15, agree_f15),
}


def propose_f16(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f16", lambda k: k._q(
        """SELECT complex_id, COUNT(DISTINCT pathway_id) AS n FROM m_complex_pathway
           GROUP BY complex_id HAVING n BETWEEN 1 AND 15"""))
    if not rows:
        return None
    return {"complex": rows[rng.randrange(len(rows))]["complex_id"]}


def answer_f16(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        "SELECT DISTINCT pathway_id FROM m_complex_pathway WHERE complex_id=?",
        (ast["complex"],))
    return {"answer_ids": sorted(r["pathway_id"] for r in rows)}


def mutate_f16(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    rows = kb._q(
        "SELECT DISTINCT pathway_id FROM m_complex_pathway WHERE complex_id != ? LIMIT 300",
        (ast["complex"],))
    foreign = sorted(r["pathway_id"] for r in rows if r["pathway_id"] not in ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f16(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        "SELECT DISTINCT complex_id FROM m_complex_pathway WHERE complex_id != ? LIMIT 20",
        (ast["complex"],))
    for r in rows:
        alt = answer_f16(kb, {"complex": r["complex_id"]})["answer_ids"]
        if alt and alt != ref["answer_ids"]:
            return {"counterfactual": f"complex_id_swapped:{r['complex_id']}",
                    "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f17(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f17", lambda k: k._q(
        """SELECT accession, COUNT(DISTINCT pmid) AS n FROM m_entity_lit
           GROUP BY accession HAVING n BETWEEN 1 AND 20"""))
    if not rows:
        return None
    return {"accession": rows[rng.randrange(len(rows))]["accession"]}


def answer_f17(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        "SELECT DISTINCT pmid FROM m_entity_lit WHERE accession=?",
        (ast["accession"],))
    return {"answer_ids": sorted(str(r["pmid"]) for r in rows)}


def mutate_f17(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    rows = kb._q(
        "SELECT DISTINCT pmid FROM m_entity_lit WHERE accession != ? LIMIT 200",
        (ast["accession"],))
    foreign = sorted(str(r["pmid"]) for r in rows if str(r["pmid"]) not in ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f17(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        "SELECT DISTINCT accession FROM m_entity_lit WHERE accession != ? LIMIT 5",
        (ast["accession"],))
    for r in rows:
        alt = answer_f17(kb, {"accession": r["accession"]})["answer_ids"]
        if alt and alt != ref["answer_ids"]:
            return {"counterfactual": f"accession_swapped:{r['accession']}",
                    "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def _pathway_universe(kb: L15KB) -> list[str]:
    return _universe(kb, "human_pathways", lambda k: sorted(
        r["id"] for r in k._q("SELECT id FROM m_pathways WHERE species='Homo sapiens'")))


def propose_f18(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    disease = _universe(kb, "f18_disease", lambda k: sorted(
        r["id"] for r in k._q("SELECT id FROM m_disease_pathways")))
    if not disease:
        return None
    uni = _pathway_universe(kb)
    n_disease = rng.randint(1, 6)
    picks = rng.sample(disease, min(n_disease, len(disease)))
    others = rng.sample(uni, 12 - len(picks))
    cands = sorted(set(picks) | set(others))
    if not picks:
        return None
    return {"candidates": cands}


def answer_f18(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    disease = _universe(kb, "f18_disease", lambda k: set(
        r["id"] for r in k._q("SELECT id FROM m_disease_pathways")))
    return {"answer_ids": sorted(c for c in ast["candidates"] if c in disease)}


def mutate_f18(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    non_ans = [c for c in ast["candidates"] if c not in ans]
    if non_ans:
        return {"answer_ids": sorted(ans | {non_ans[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f18(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    complement = sorted(set(ast["candidates"]) - set(ref["answer_ids"]))
    if complement and complement != ref["answer_ids"]:
        return {"counterfactual": "disease_flag_inverted",
                "answer_diff": len(set(complement) ^ set(ref["answer_ids"]))}
    return None


def propose_f19(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    uni = _pathway_universe(kb)
    parents_closure = _pathway_closure(kb)
    tops = [p for p in uni if p not in parents_closure]
    if not tops:
        return None
    n_top = rng.randint(1, 5)
    picks = rng.sample(tops, min(n_top, len(tops)))
    non_tops = [p for p in uni if p in parents_closure]
    others = rng.sample(non_tops, min(12 - len(picks), len(non_tops)))
    cands = sorted(set(picks) | set(others))
    if not picks:
        return None
    return {"candidates": cands}


def answer_f19(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    parents_closure = _pathway_closure(kb)
    return {"answer_ids": sorted(c for c in ast["candidates"] if c not in parents_closure)}


def mutate_f19(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    non_ans = [c for c in ast["candidates"] if c not in ans]
    if non_ans:
        return {"answer_ids": sorted(ans | {non_ans[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f19(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    complement = sorted(set(ast["candidates"]) - set(ref["answer_ids"]))
    if complement and complement != ref["answer_ids"]:
        return {"counterfactual": "top_level_definition_inverted",
                "answer_diff": len(set(complement) ^ set(ref["answer_ids"]))}
    return None


def propose_f20(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f20", lambda k: k._q(
        """SELECT name, COUNT(DISTINCT reaction_id) AS n FROM m_reaction_names
           WHERE name != '' GROUP BY name HAVING n BETWEEN 1 AND 8"""))
    if not rows:
        return None
    return {"name": rows[rng.randrange(len(rows))]["name"]}


def answer_f20(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q("SELECT reaction_id FROM m_reaction_names WHERE name=?",
                 (ast["name"],))
    return {"answer_ids": sorted(r["reaction_id"] for r in rows)}


def mutate_f20(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    variant = ast["name"].strip().lower()
    rows = kb._q(
        "SELECT reaction_id FROM m_reaction_names WHERE LOWER(TRIM(name))=? AND name != ? LIMIT 5",
        (variant, ast["name"]))
    foreign = sorted(r["reaction_id"] for r in rows if r["reaction_id"] not in ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    if len(ans) > 1:
        return {"answer_ids": sorted(ans)[1:]}
    return {"answer_ids": sorted(ans)}


def counter_f20(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        "SELECT DISTINCT reaction_id FROM m_reaction_names WHERE name LIKE '%' || ? || '%'",
        (ast["name"],))
    alt = sorted({r["reaction_id"] for r in rows})
    if alt != ref["answer_ids"]:
        return {"counterfactual": "substring_superset_match",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    variant = ast["name"].strip().lower()
    rows = kb._q(
        "SELECT DISTINCT reaction_id FROM m_reaction_names WHERE LOWER(TRIM(name))=?",
        (variant,))
    alt = sorted({r["reaction_id"] for r in rows})
    if alt != ref["answer_ids"]:
        return {"counterfactual": "case_whitespace_insensitive_match",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


def propose_f21(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f21", lambda k: k._q(
        """SELECT c.accession AS acc, COUNT(DISTINCT i.accession) AS n
           FROM m_reaction_roles c
           JOIN m_reaction_roles i ON i.reaction_id = c.reaction_id AND i.role='input'
           WHERE c.role='catalyst' AND c.accession != i.accession
           GROUP BY c.accession HAVING n BETWEEN 1 AND 20"""))
    if not rows:
        return None
    return {"accession": rows[rng.randrange(len(rows))]["acc"]}


def answer_f21(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    rows = kb._q(
        """SELECT DISTINCT i.accession AS acc FROM m_reaction_roles c
           JOIN m_reaction_roles i ON i.reaction_id = c.reaction_id AND i.role='input'
           WHERE c.role='catalyst' AND c.accession=? AND i.accession != ?""",
        (ast["accession"], ast["accession"]))
    return {"answer_ids": sorted(r["acc"] for r in rows)}


def mutate_f21(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    rows = kb._q(
        """SELECT DISTINCT i.accession AS acc FROM m_reaction_roles c
           JOIN m_reaction_roles i ON i.reaction_id = c.reaction_id AND i.role='input'
           WHERE c.role='input' AND c.accession=? LIMIT 300""",
        (ast["accession"],))
    foreign = sorted(r["acc"] for r in rows if r["acc"] not in ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f21(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    rows = kb._q(
        """SELECT DISTINCT i.accession AS acc FROM m_reaction_roles c
           JOIN m_reaction_roles i ON i.reaction_id = c.reaction_id AND i.role='input'
           WHERE c.role='input' AND c.accession=?""",
        (ast["accession"],))
    alt = sorted({r["acc"] for r in rows} - {ast["accession"]})
    if alt != ref["answer_ids"]:
        return {"counterfactual": "role_pair_swapped:input_input",
                "answer_diff": len(set(alt) ^ set(ref["answer_ids"]))}
    return None


KINDS.update({
    "complex_pathways": KindOps(propose_f16, answer_f16, mutate_f16, counter_f16),
    "entity_pmids": KindOps(propose_f17, answer_f17, mutate_f17, counter_f17),
    "disease_subset": KindOps(propose_f18, answer_f18, mutate_f18, counter_f18),
    "top_level_subset": KindOps(propose_f19, answer_f19, mutate_f19, counter_f19),
    "name_to_ids": KindOps(propose_f20, answer_f20, mutate_f20, counter_f20),
    "coreaction_pairs": KindOps(propose_f21, answer_f21, mutate_f21, counter_f21),
})


def propose_f20b(kb: L15KB, rng: Random) -> dict[str, Any] | None:
    rows = _universe(kb, "f20b", lambda k: k._q(
        """SELECT reaction_id, COUNT(DISTINCT pathway_id) AS n FROM m_reaction_pathway
           WHERE reaction_id LIKE 'R-HSA-%'
           GROUP BY reaction_id HAVING n BETWEEN 2 AND 8"""))
    if not rows:
        return None
    return {"reaction": rows[rng.randrange(len(rows))]["reaction_id"]}


def answer_f20b(kb: L15KB, ast: dict[str, Any]) -> dict[str, Any]:
    return {"answer_ids": sorted(_reaction_pathways(kb, ast["reaction"]))}


def mutate_f20b(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
    ans = set(ref["answer_ids"])
    ancestors: set[str] = set()
    for p in ans:
        ancestors |= _ancestors_of(kb, p)
    foreign = sorted(a for a in ancestors - ans)
    if foreign:
        return {"answer_ids": sorted(ans | {foreign[0]})}
    return {"answer_ids": sorted(ans)[1:]} if len(ans) > 1 else {"answer_ids": sorted(ans)}


def counter_f20b(kb: L15KB, ast: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any] | None:
    ans = set(ref["answer_ids"])
    with_anc = set(ans)
    for p in ans:
        with_anc |= _ancestors_of(kb, p)
    alt = sorted(with_anc)
    if alt != ref["answer_ids"]:
        return {"counterfactual": "ancestor_pathways_included",
                "answer_diff": len(with_anc - ans)}
    return None


KINDS["reaction_pathways"] = KindOps(propose_f20b, answer_f20b, mutate_f20b, counter_f20b)
