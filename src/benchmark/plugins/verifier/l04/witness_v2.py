"""L04 v2 semantic witnesses: counterfactual perturbation evidence for the 18
new families (20K expansion).

Same contract as the legacy witness module (witness.py): a governed probe
answer under the correct knowledge, the same probe under a perturbed
knowledge slice (must DIFFER), and a control probe under both (must be
UNCHANGED). Following the legacy KB-only kinds (flip_side,
swap_direction_children, corrupt_ph73, remove_xref), the v2 kinds are
computed from the pinned KB with recorded gold/perturbed sha256 slices.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmark.plugins.source.l04.kb import L04KB
from benchmark.plugins.task_family.l04 import families_v2 as F2
from benchmark.plugins.task_family.l04.families import (
    _ancestors, _c2m, _master_cache, _rowcount, masters_involving,
)


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _num_key(rhea_id: str) -> int:
    return int(rhea_id.split(":")[1])


def _involving(kb: L04KB, chebi: str) -> list[str]:
    masters = kb.master_ids
    return sorted({r for r, _s, _p, _c, ch in kb.participants
                   if ch == chebi and r in masters}, key=_num_key)


def _pick_control_entity(kb: L04KB, rng, exclude: set[str],
                         lo: int = 5, hi: int = 400) -> str | None:
    universe = sorted(kb.chebi_to_reactions)
    counts: dict[str, int] = {}
    masters = kb.master_ids
    for r, _s, _p, _c, ch in kb.participants:
        if r in masters:
            counts[ch] = counts.get(ch, 0) + 1
    cands = [c for c in universe
             if c not in exclude and lo <= counts.get(c, 0) <= hi]
    return rng.choice(cands) if cands else None


_NO_EC_POOL: dict[int, list[str]] = {}
_STATUS_POOL: dict[int, dict[str, list[str]]] = {}
_CIT_POOL: dict[int, dict[str, list[str]]] = {}


def _citation_control_pool(kb: L04KB, band: str) -> list[str]:
    """Entities whose citation-band answer for ``band`` is non-empty (rare
    for band 0-0: most entities have only cited masters)."""
    key = id(kb)
    if key not in _CIT_POOL:
        from benchmark.plugins.task_family.l04.families_v2 import (
            CITATION_BANDS, parse_band)
        masters = _master_cache(kb)
        cit = {m: len(kb.reactions[m].get("citations") or ()) for m in masters}
        pools: dict[str, list[str]] = {b: [] for b in CITATION_BANDS}
        for c in sorted(_c2m(kb)):
            counts = [cit.get(m, 0)
                      for m in masters_involving(kb, masters, c)]
            for b in CITATION_BANDS:
                lo, hi = parse_band(b)
                if any(lo <= n <= hi for n in counts):
                    pools[b].append(c)
        _CIT_POOL[key] = pools
    return _CIT_POOL[key].get(band, [])


def _no_ec_control_pool(kb: L04KB) -> list[str]:
    """Entities with a nonempty no-EC answer (control probes for the EC-absence
    family must themselves have no-EC masters to discriminate)."""
    key = id(kb)
    if key not in _NO_EC_POOL:
        masters = _master_cache(kb)
        pool = []
        for c in sorted(_c2m(kb)):
            n = sum(1 for m in masters_involving(kb, masters, c)
                    if not kb.ec_by_master.get(m))
            if 3 <= n <= 400:
                pool.append(c)
        _NO_EC_POOL[key] = pool
    return _NO_EC_POOL[key]


def _status_control_pool(kb: L04KB, status: str) -> list[str]:
    """Entities with at least one master carrying ``status`` (rare statuses
    like Preliminary/Obsolete make a random control almost always empty)."""
    key = id(kb)
    if key not in _STATUS_POOL:
        masters = _master_cache(kb)
        pools: dict[str, list[str]] = {}
        for c in sorted(_c2m(kb)):
            for st in ("Preliminary", "Obsolete"):
                n = sum(1 for m in masters_involving(kb, masters, c)
                        if kb.reactions[m]["status"] == st)
                if 1 <= n <= 60:
                    pools.setdefault(st, []).append(c)
        _STATUS_POOL[key] = pools
    return _STATUS_POOL[key].get(status, [])


def witness_blank_formula_class(kb: L04KB, formula: str,
                                control_formula: str) -> dict[str, Any]:
    """Governed: masters involving compounds with rh:formula == ``formula``.
    Perturbation: rewrite the formula of every compound in the class to a
    sentinel (class erased -> governed answer becomes empty). Control: the
    analogous answer of an untouched formula class (must be unchanged)."""
    governed_gold = sorted(F2._masters_with_formula(kb, formula), key=_num_key)
    control_gold = sorted(F2._masters_with_formula(kb, control_formula),
                          key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    fmap, _c, _ch = F2._compound_maps(kb)
    perturbed_map = {c: ("Z" if f == formula else f)
                     for c, f in fmap.items()}
    masters = kb.master_ids
    # recompute governed under the perturbed formula map
    comps_still = {c for c, f in perturbed_map.items() if f == formula}
    pert: set[str] = set()
    for r, _s, _p, c, _ch in kb.participants:
        if c in comps_still and r in masters:
            pert.add(r)
    control_comps = {c for c, f in perturbed_map.items() if f == control_formula}
    control_pert: set[str] = set()
    for r, _s, _p, c, _ch in kb.participants:
        if c in control_comps and r in masters:
            control_pert.add(r)
    ok = not pert and control_pert == set(control_gold)
    return {"ok": ok, "kind": "blank_formula_class", "formula": formula,
            "control_formula": control_formula,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == set(control_gold),
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert) + ["|", *sorted(control_pert)]),
            "perturbation": f"formula of all compounds in class {formula} erased"}


def witness_push_citations_out_of_band(kb: L04KB, chebi: str, band: str,
                                       control: str) -> dict[str, Any]:
    """Governed: masters involving ``chebi`` with citation count in ``band``.
    Perturbation: citation counts of all masters involving ``chebi`` are set
    to hi+1 (above the band) -> governed answer becomes empty. Control: the
    same band query for an untouched entity (unchanged)."""
    lo, hi = F2.parse_band(band)
    masters = _master_cache(kb)
    cit = {m: len(kb.reactions[m].get("citations") or ())
           for m in masters}
    own = set(masters_involving(kb, masters, chebi))
    governed_gold = sorted(m for m in own if lo <= cit.get(m, 0) <= hi)
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted(m for m in ctrl if lo <= cit.get(m, 0) <= hi)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    pert_cit = dict(cit)
    for m in own:
        pert_cit[m] = hi + 1
    pert = [m for m in own if lo <= pert_cit.get(m, 0) <= hi]
    control_pert = sorted(m for m in ctrl if lo <= pert_cit.get(m, 0) <= hi)
    ok = not pert and control_pert == control_gold
    return {"ok": ok, "kind": "push_citations_out_of_band", "chebi": chebi,
            "band": band, "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(pert + ["|", *control_pert]),
            "perturbation": f"citation counts of {len(own)} masters pushed above band"}


def witness_strip_isa_links(kb: L04KB, chebi: str,
                            control: str) -> dict[str, Any]:
    """Governed: is_a neighborhood answer of ``chebi``. Perturbation: remove
    every child-level is_a edge touching a child of any master involving
    ``chebi`` -> the neighborhood becomes empty. Control: neighborhood of an
    untouched entity (unchanged)."""
    masters = _master_cache(kb)
    mn = F2._master_neighborhood(kb)
    own = set(masters_involving(kb, masters, chebi))
    governed_gold = sorted(F2._masters_isa_neighborhood(kb, chebi), key=_num_key)
    control_gold = sorted(F2._masters_isa_neighborhood(kb, control), key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    own_children: set[str] = set()
    for m in own:
        u = kb.reactions[m]
        own_children |= set(u["children_lr_rl"]) | set(u["children_bi"])
    # perturb: drop relationship rows touching a child of an own master
    kept = [row for row in kb.relationships
            if row["FROM_REACTION_ID"] not in own_children
            and row["TO_REACTION_ID"] not in own_children]
    child2master: dict[str, str] = {}
    for m in masters:
        u = kb.reactions[m]
        for x in u["children_lr_rl"] + u["children_bi"]:
            child2master[x] = m
    mn_p: dict[str, set[str]] = {}
    for row in kept:
        ma = child2master.get(row["FROM_REACTION_ID"])
        mb = child2master.get(row["TO_REACTION_ID"])
        if ma and mb and ma != mb:
            mn_p.setdefault(ma, set()).add(mb)
            mn_p.setdefault(mb, set()).add(ma)

    def neigh(ch: str) -> set[str]:
        ms = set(masters_involving(kb, masters, ch))
        out: set[str] = set()
        for m in ms:
            out |= mn_p.get(m, set())
        return out - ms

    pert = neigh(chebi)
    control_pert = neigh(control)
    ok = not pert and control_pert == set(control_gold)
    return {"ok": ok, "kind": "strip_isa_links", "chebi": chebi,
            "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert) + ["|", *sorted(control_pert)]),
            "perturbation": f"is_a edges touching children of {len(own)} masters removed"}


def witness_rename_token(kb: L04KB, token: str,
                         control_token: str) -> dict[str, Any]:
    """Governed: masters involving any entity whose published name contains
    ``token`` (case-insensitive substring). Perturbation: the token is erased
    from every matching published name -> governed answer becomes empty.
    Control: the answer of another token whose matched entities are untouched
    (must be unchanged)."""
    masters = _master_cache(kb)
    c2m = _c2m(kb)

    def answer_of(tok: str, names: dict[str, str]) -> set[str]:
        out: set[str] = set()
        for e in c2m:
            if tok in names.get(e, "").lower():
                out |= set(masters_involving(kb, masters, e))
        return out

    governed_gold = sorted(answer_of(token, kb.names), key=_num_key)
    control_gold = sorted(answer_of(control_token, kb.names), key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    pert_names = {e: (nm.lower().replace(token, "zzz") if token in nm.lower()
                      else nm)
                  for e, nm in kb.names.items()}
    pert = answer_of(token, pert_names)
    control_pert = answer_of(control_token, pert_names)
    ok = not pert and control_pert == set(control_gold)
    return {"ok": ok, "kind": "rename_name_token", "token": token,
            "control_token": control_token,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == set(control_gold),
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert) + ["|", *sorted(control_pert)]),
            "perturbation": f"token '{token}' erased from all matching published names"}


def witness_edit_smiles(kb: L04KB, sub: str, control_sub: str) -> dict[str, Any]:
    """Governed: masters involving any entity whose SMILES contains the
    substring ``sub``. Perturbation: the substring is blanked in every
    matching SMILES -> governed answer becomes empty. Control: the answer of
    another substring over untouched SMILES (unchanged)."""
    masters = _master_cache(kb)
    c2m = _c2m(kb)
    sm = F2._smiles_map(kb)

    def answer_of(s: str, smiles: dict[str, str]) -> set[str]:
        out: set[str] = set()
        for e, v in smiles.items():
            if e in c2m and s in v:
                out |= set(masters_involving(kb, masters, e))
        return out

    governed_gold = sorted(answer_of(sub, sm), key=_num_key)
    control_gold = sorted(answer_of(control_sub, sm), key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    pert_sm = {e: (v.replace(sub, "###") if sub in v else v)
               for e, v in sm.items()}
    pert = answer_of(sub, pert_sm)
    control_pert = answer_of(control_sub, pert_sm)
    ok = not pert and control_pert == set(control_gold)
    return {"ok": ok, "kind": "edit_smiles", "smiles_sub": sub,
            "control_sub": control_sub,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == set(control_gold),
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert) + ["|", *sorted(control_pert)]),
            "perturbation": f"substring '{sub}' blanked in all matching SMILES"}


def witness_detach_descendants(kb: L04KB, anchor_a: str, anchor_b: str,
                               control_anchor: str) -> dict[str, Any]:
    """Governed: closure-conjunction master answer of (anchor_a, anchor_b).
    Perturbation: every universe descendant of anchor_a loses all direct
    parent edges -> nothing remains a descendant of anchor_a -> conjunction
    empty. Control: the transitive closure of an untouched anchor
    (unchanged)."""
    masters = _master_cache(kb)
    uni = sorted(kb.chebi_to_reactions)

    def conj(a: str, b: str, parents: dict[str, tuple[str, ...]]) -> set[str]:
        def anc_of(c: str) -> set[str]:
            seen: set[str] = set()
            stack = list(parents.get(c, ()))
            while stack:
                n = stack.pop()
                if n in seen:
                    continue
                seen.add(n)
                stack.extend(parents.get(n, ()))
            return seen
        out: set[str] | None = None
        for anchor in (a, b):
            ms: set[str] = set()
            for c in uni:
                if anchor in anc_of(c):
                    ms |= set(masters_involving(kb, masters, c))
            out = ms if out is None else out & ms
        return out or set()

    def closure_of(c: str, parents: dict[str, tuple[str, ...]]) -> set[str]:
        seen: set[str] = set()
        stack = list(parents.get(c, ()))
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(parents.get(n, ()))
        return seen

    desc_a = {c for c in uni if anchor_a in _ancestors(kb, c)}
    governed_gold = sorted(conj(anchor_a, anchor_b, kb.chebi_parents),
                           key=_num_key)
    control_gold = sorted(closure_of(control_anchor, kb.chebi_parents))
    if not governed_gold or not control_gold or not desc_a:
        return {"ok": False, "reason": "empty governed, control or descendant set"}
    perturbed_parents = {e: (() if e in desc_a else ps)
                         for e, ps in kb.chebi_parents.items()}
    pert = conj(anchor_a, anchor_b, perturbed_parents)
    control_pert = closure_of(control_anchor, perturbed_parents)
    ok = not pert and control_pert == set(control_gold)
    return {"ok": ok, "kind": "detach_descendants", "ancestor_a": anchor_a,
            "ancestor_b": anchor_b, "control_anchor": control_anchor,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == set(control_gold),
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert) + ["|", *sorted(control_pert)]),
            "perturbation": f"all parent edges of {len(desc_a)} descendants of "
                            f"{anchor_a} removed"}


def witness_strip_extra_ec(kb: L04KB, chebi: str,
                           control: str) -> dict[str, Any]:
    """Governed: masters involving ``chebi`` with >=2 EC links. Perturbation:
    every master involving ``chebi`` keeps at most one EC link -> governed
    answer becomes empty. Control: multi-EC answer of an untouched entity
    (unchanged)."""
    masters = _master_cache(kb)
    own = set(masters_involving(kb, masters, chebi))
    governed_gold = sorted(m for m in own
                           if len(kb.ec_by_master.get(m, ())) >= 2)
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted(m for m in ctrl
                          if len(kb.ec_by_master.get(m, ())) >= 2)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    pert_ec = {m: (tuple(sorted(kb.ec_by_master.get(m, ()))[:1]) if m in own
                   else kb.ec_by_master.get(m, ()))
               for m in masters}
    pert = [m for m in own if len(pert_ec.get(m, ())) >= 2]
    control_pert = sorted(m for m in ctrl if len(pert_ec.get(m, ())) >= 2)
    ok = not pert and control_pert == control_gold
    return {"ok": ok, "kind": "strip_extra_ec", "chebi": chebi,
            "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(pert + ["|", *control_pert]),
            "perturbation": f"extra EC links of {len(own)} masters stripped to <=1"}


def witness_add_ec_to_governed(kb: L04KB, chebi: str,
                               control: str) -> dict[str, Any]:
    """Governed: masters involving ``chebi`` with NO EC link. Perturbation:
    every master involving ``chebi`` gains a synthetic EC link -> the no-EC
    answer becomes empty. Control: no-EC answer of an untouched entity
    (unchanged)."""
    masters = _master_cache(kb)
    own = set(masters_involving(kb, masters, chebi))
    governed_gold = sorted(m for m in own if not kb.ec_by_master.get(m))
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted(m for m in ctrl if not kb.ec_by_master.get(m))
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}

    def has_ec(m: str, perturbed: bool) -> bool:
        if perturbed and m in own:
            return True
        return bool(kb.ec_by_master.get(m))

    pert = [m for m in own if not has_ec(m, True)]
    control_pert = sorted(m for m in ctrl if not has_ec(m, True))
    ok = not pert and control_pert == control_gold
    return {"ok": ok, "kind": "add_ec_to_governed", "chebi": chebi,
            "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(pert + ["|", *control_pert]),
            "perturbation": f"synthetic EC link added to {len(own)} masters"}


def witness_strip_go_of_governed(kb: L04KB, chebi: str,
                                 control: str) -> dict[str, Any]:
    """Governed: GO-mediated neighborhood answer of ``chebi``. Perturbation:
    every GO link of masters involving ``chebi`` is removed -> no shared GO
    terms remain -> governed answer becomes empty. Control: the GO-mediated
    answer of an untouched entity whose neighborhood does not depend on the
    stripped links (unchanged)."""
    masters = _master_cache(kb)
    governed_gold = sorted(F2._masters_go_mediated(kb, chebi), key=_num_key)
    control_gold = sorted(F2._masters_go_mediated(kb, control), key=_num_key)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    own = set(masters_involving(kb, masters, chebi))
    pert_go_by_master = {m: (frozenset() if m in own else kb.go_by_master.get(m, frozenset()))
                         for m in masters}

    def mediated(ch: str) -> set[str]:
        gos: set[str] = set()
        for m in masters_involving(kb, masters, ch):
            gos |= set(pert_go_by_master.get(m, ()))
        out: set[str] = set()
        for m in masters:
            if pert_go_by_master.get(m, frozenset()) & gos:
                out.add(m)
        return out

    pert = mediated(chebi)
    control_pert = mediated(control)
    ok = not pert and control_pert == set(control_gold)
    return {"ok": ok, "kind": "strip_go_of_governed", "chebi": chebi,
            "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == set(control_gold),
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert) + ["|", *sorted(control_pert)]),
            "perturbation": f"GO links of {len(own)} masters involving {chebi} removed"}


def witness_strip_participants_of_governed(kb: L04KB, chebi: str, band: str,
                                           control: str) -> dict[str, Any]:
    """Governed: masters involving ``chebi`` with participant-row count in
    ``band``. Perturbation: every participant row of masters involving
    ``chebi`` is removed (count 0, below every shipped band's lo>=2) ->
    governed answer becomes empty. Control: the band answer of an untouched
    entity (unchanged)."""
    lo, hi = F2.parse_band(band)
    masters = _master_cache(kb)
    rc = _rowcount(kb)
    own = set(masters_involving(kb, masters, chebi))
    governed_gold = sorted(m for m in own if lo <= rc.get(m, 0) <= hi)
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted(m for m in ctrl if lo <= rc.get(m, 0) <= hi)
    if not governed_gold or not control_gold:
        return {"ok": False, "reason": "empty governed or control probe"}
    pert_rc = dict(rc)
    for m in own:
        pert_rc[m] = 0
    pert = [m for m in own if lo <= pert_rc.get(m, 0) <= hi]
    control_pert = sorted(m for m in ctrl if lo <= pert_rc.get(m, 0) <= hi)
    ok = not pert and control_pert == control_gold
    return {"ok": ok, "kind": "strip_participants_of_governed", "chebi": chebi,
            "band": band, "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(pert + ["|", *control_pert]),
            "perturbation": f"participant rows of {len(own)} masters removed"}


def witness_remove_go_link_governed(kb: L04KB, chebi: str,
                                    control: str) -> dict[str, Any]:
    """Repair witness for rep_go_link_table_repair. Governed: GO terms of
    masters involving ``chebi``. Perturbation: the GO links uniquely
    contributed by a single scope master (the corruption carrier kind) are
    dropped -> governed answer loses exactly those terms. Control: GO answer
    of another entity recomputed under the same link removal (must be
    unchanged — a control sharing a dropped link's master is rejected)."""
    masters = _master_cache(kb)
    own = set(masters_involving(kb, masters, chebi))
    contrib: dict[str, int] = {}
    for m in own:
        for g in kb.go_by_master.get(m, ()):
            contrib[g] = contrib.get(g, 0) + 1
    governed_gold = sorted(contrib)
    if not governed_gold:
        return {"ok": False, "reason": "empty governed probe"}
    dropped_links = {(m, g) for m in own for g in kb.go_by_master.get(m, ())
                     if contrib[g] == 1}
    dropped = {g for _m, g in dropped_links}
    pert = [g for g in governed_gold if g not in dropped]
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted({g for m in ctrl for g in kb.go_by_master.get(m, ())})
    control_pert = sorted({g for m in ctrl for g in kb.go_by_master.get(m, ())
                           if (m, g) not in dropped_links})
    ok = (bool(dropped) and pert != governed_gold and bool(control_gold)
          and control_pert == control_gold)
    return {"ok": ok, "kind": "remove_go_link_governed", "chebi": chebi,
            "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(pert + ["|", *control_pert]),
            "perturbation": f"{len(dropped)} uniquely-contributed GO links dropped"}


def witness_swap_xref_db_prefix(kb: L04KB, chebi: str, db: str,
                                control: str) -> dict[str, Any]:
    """Repair witness for rep_xref_table_repair. Governed: ``db`` xrefs of
    masters involving ``chebi``. Perturbation: the uniquely-contributed
    ``db`` xrefs are rewritten under a different database prefix -> the
    governed answer loses exactly those xrefs. Control: ``db`` xrefs of an
    untouched entity (unchanged)."""
    masters = _master_cache(kb)
    own = set(masters_involving(kb, masters, chebi))
    hits: dict[str, int] = {}
    for m in own:
        for x in kb.xrefs_by_master.get(m, ()):
            if x.startswith(db + ":"):
                hits[x] = hits.get(x, 0) + 1
    governed_gold = sorted(hits)
    if not governed_gold:
        return {"ok": False, "reason": "empty governed probe"}
    moved_links = {(m, x) for m in own for x in kb.xrefs_by_master.get(m, ())
                   if x.startswith(db + ":") and hits[x] == 1}
    moved = {x for _m, x in moved_links}
    pert = [x for x in governed_gold if x not in moved]
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted({x for m in ctrl for x in kb.xrefs_by_master.get(m, ())
                           if x.startswith(db + ":")})
    control_pert = sorted({x for m in ctrl for x in kb.xrefs_by_master.get(m, ())
                           if x.startswith(db + ":")
                           and (m, x) not in moved_links})
    ok = (bool(moved) and pert != governed_gold and bool(control_gold)
          and control_pert == control_gold)
    return {"ok": ok, "kind": "swap_xref_db_prefix", "chebi": chebi, "db": db,
            "control": control,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(pert + ["|", *control_pert]),
            "perturbation": f"{len(moved)} uniquely-contributed {db} xrefs reprefixed"}


def witness_flip_status(kb: L04KB, chebi: str, status: str,
                        control: str) -> dict[str, Any]:
    """Repair witness for rep_status_table_repair. Governed: masters
    involving ``chebi`` with curation status ``status``. Perturbation: one
    governed master is flipped to a wrong status and one non-governed master
    is flipped into ``status`` -> the governed answer changes. Control: the
    status answer of an untouched entity (unchanged)."""
    masters = _master_cache(kb)
    own = set(masters_involving(kb, masters, chebi))
    governed_gold = sorted(m for m in own
                           if kb.reactions[m]["status"] == status)
    others = sorted(m for m in own
                    if kb.reactions[m]["status"] != status)
    ctrl = set(masters_involving(kb, masters, control))
    control_gold = sorted(m for m in ctrl
                          if kb.reactions[m]["status"] == status)
    if not governed_gold or not others or not control_gold:
        return {"ok": False, "reason": "empty governed/other/control probe"}
    out_m, in_m = governed_gold[0], others[0]
    pert_set = (set(governed_gold) - {out_m}) | {in_m}
    flips = {out_m: "FLIPPED_OUT", in_m: status}

    def status_of(m: str, perturbed: bool) -> str:
        if perturbed and m in flips:
            return flips[m]
        return kb.reactions[m]["status"]

    control_pert = sorted(m for m in ctrl
                          if status_of(m, True) == status)
    ok = pert_set != set(governed_gold) and control_pert == control_gold
    return {"ok": ok, "kind": "flip_status", "chebi": chebi, "status": status,
            "control": control, "flip_out": out_m, "flip_in": in_m,
            "governed_correct_count": len(governed_gold),
            "governed_perturbed_count": len(pert_set),
            "control_count": len(control_gold),
            "control_unchanged": control_pert == control_gold,
            "slice_sha256": _sha(governed_gold + ["|", *control_gold]),
            "perturbed_sha256": _sha(sorted(pert_set) + ["|", *control_pert]),
            "perturbation": f"status of {out_m} flipped out, {in_m} flipped in"}


# ---------------------------------------------------------------------------
# dispatcher (v2 kinds + reuse of the legacy kinds)
# ---------------------------------------------------------------------------

def _stable_seed(family_id: str, seed: int) -> int:
    fam = int.from_bytes(hashlib.sha256(family_id.encode()).digest()[:4], "big")
    return ((seed * 1_000_003) ^ fam) & 0xFFFFFFFF


def certify_witness_v2(kb: L04KB, family_id: str, witness_kind: str,
                       params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Dispatch the counterfactual witness for one v2 candidate instance."""
    import random
    rng = random.Random(_stable_seed(family_id, seed))

    if witness_kind == "blank_formula_class":
        fmap, _c, _ch = F2._compound_maps(kb)
        others = sorted({f for f in fmap.values()
                         if f and f != params["formula"]})
        if not others:
            return {"ok": False, "reason": "no control formula"}
        for _ in range(8):
            control = rng.choice(others)
            if F2._masters_with_formula(kb, control):
                return witness_blank_formula_class(kb, params["formula"], control)
        return {"ok": False, "reason": "no control formula with reactions"}

    if witness_kind == "push_citations_out_of_band":
        pool = [c for c in _citation_control_pool(kb, params["band"])
                if c != params["chebi"]]
        for _ in range(8):
            if not pool:
                break
            control = rng.choice(pool)
            w = witness_push_citations_out_of_band(kb, params["chebi"],
                                                   params["band"], control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working citation-band witness"}

    if witness_kind == "strip_isa_links":
        for _ in range(8):
            control = _pick_control_entity(kb, rng, {params["chebi"]})
            if control is None:
                break
            w = witness_strip_isa_links(kb, params["chebi"], control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working is_a-strip witness"}

    if witness_kind == "rename_name_token":
        toks = sorted({w for e in _c2m(kb)
                       for w in __import__("re").split(
                           r"[^a-z]+", kb.names.get(e, "").lower())
                       if 4 <= len(w) <= 10 and w != params["token"]})
        for _ in range(8):
            control = rng.choice(toks)
            w = witness_rename_token(kb, params["token"], control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working name-token witness"}

    if witness_kind == "edit_smiles":
        subs = ["OP(=O)(O)", "C(=O)S", "c1ccccc1", "NC=O", "[nH]", "C(=O)O",
                "CC(=O)", "NCC", "O=P"]
        tried = 0
        for control in subs:
            if control == params["smiles_sub"]:
                continue
            w = witness_edit_smiles(kb, params["smiles_sub"], control)
            tried += 1
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working SMILES witness"}

    if witness_kind == "detach_descendants":
        uni = sorted(kb.chebi_to_reactions)
        anc_counts: dict[str, int] = {}
        for c in uni:
            for a in _ancestors(kb, c):
                anc_counts[a] = anc_counts.get(a, 0) + 1
        pool = [a for a, n in anc_counts.items()
                if 3 <= n <= 60 and a not in (params["ancestor_a"],
                                              params["ancestor_b"])]
        for _ in range(12):
            if not pool:
                break
            control = rng.choice(pool)
            w = witness_detach_descendants(kb, params["ancestor_a"],
                                           params["ancestor_b"], control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working detach witness"}

    if witness_kind in ("strip_extra_ec", "add_ec_to_governed",
                        "strip_go_of_governed",
                        "remove_go_link_governed", "flip_status"):
        if witness_kind == "add_ec_to_governed":
            pool = [c for c in _no_ec_control_pool(kb)
                    if c != params["chebi"]]
            pick = lambda: rng.choice(pool) if pool else None
        elif witness_kind == "flip_status":
            pool = [c for c in _status_control_pool(kb, params["status"])
                    if c != params["chebi"]]
            pick = lambda: rng.choice(pool) if pool else None
        else:
            pick = lambda: _pick_control_entity(kb, rng, {params["chebi"]})
        fn = {"strip_extra_ec": lambda c: witness_strip_extra_ec(kb, params["chebi"], c),
              "add_ec_to_governed": lambda c: witness_add_ec_to_governed(kb, params["chebi"], c),
              "strip_go_of_governed": lambda c: witness_strip_go_of_governed(kb, params["chebi"], c),
              "remove_go_link_governed": lambda c: witness_remove_go_link_governed(kb, params["chebi"], c),
              "flip_status": lambda c: witness_flip_status(kb, params["chebi"], params["status"], c),
              }[witness_kind]
        for _ in range(8):
            control = pick()
            if control is None:
                break
            w = fn(control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": f"no working {witness_kind} witness"}

    if witness_kind == "strip_participants_of_governed":
        for _ in range(8):
            control = _pick_control_entity(kb, rng, {params["chebi"]})
            if control is None:
                break
            w = witness_strip_participants_of_governed(
                kb, params["chebi"], params["band"], control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working cardinality witness"}

    if witness_kind == "swap_xref_db_prefix":
        for _ in range(8):
            control = _pick_control_entity(kb, rng, {params["chebi"]})
            if control is None:
                break
            w = witness_swap_xref_db_prefix(kb, params["chebi"],
                                            params["db"], control)
            if w["ok"]:
                return w
        return {"ok": False, "reason": "no working xref-prefix witness"}

    if witness_kind == "corrupt_ph73":
        # same mechanism as the legacy pH-7.3 witness, anchored on the
        # mapping source entity and its resolved microspecies target
        from benchmark.plugins.verifier.l04 import witness as W
        target = F2._ph73_map(kb)[params["chebi"]]
        for _ in range(6):
            decoy = _pick_control_entity(kb, rng, {params["chebi"], target})
            control = _pick_control_entity(
                kb, rng, {params["chebi"], target, decoy or ""})
            if decoy is None or control is None:
                break
            w = W.witness_corrupt_ph73(kb, params["chebi"], target,
                                       decoy, control)
            if w.get("ok"):
                return w
        return {"ok": False, "reason": "no decoy flips the governed probe"}

    # legacy kinds reused verbatim
    from benchmark.plugins.verifier.l04 import plugin as legacy_plugin
    return legacy_plugin.certify_witness(kb, family_id, witness_kind,
                                         params, seed)
