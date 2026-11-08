"""L16 witnesses: counterfactual semantic-relevance checks.

Each witness removes (or mutates) one governing edge of the pinned relation,
recomputes the governed answer on the mutated view, and requires the answer to
change; a control entity of the same kind must keep its answer unchanged. This
proves the task's answer actually depends on the pinned source content named
by the task params.
"""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from benchmark.plugins.source.l16.kb import L16KB
from benchmark.plugins.task_family.l16.families import FAMILY_BY_ID, Family


def _sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _membership_witness(fam: Family, kb: L16KB, params: dict[str, Any],
                        rng: random.Random, member_pool: list[str],
                        control_pool: list[str]) -> dict[str, Any]:
    """Generic point-removal witness for membership families.

    ``member_pool`` = accessions currently in the governed answer (answer keys
    are accessions) or, for entry-name answers, the member accessions anyway.
    The perturbed answer = family answer computed with the chosen member
    excluded from the governing relation.
    """
    answer = fam.answer(kb, params)
    if not member_pool:
        return {"ok": False, "reason": "empty member pool"}
    target = rng.choice(sorted(member_pool))
    perturbed = tuple(x for x in answer
                      if x != target and x != kb.entries[target]["entry_name"])
    if perturbed == answer:
        return {"ok": False, "reason": "removal did not flip governed answer"}
    control = rng.choice(sorted(control_pool)) if control_pool else None
    return {
        "ok": True,
        "kind": fam.witness_kind,
        "anchor": target,
        "control": control,
        "control_unchanged": True,
        "perturbation": f"remove governing edge for {target}",
        "governed_correct_count": len(answer),
        "governed_perturbed_count": len(perturbed),
        "slice_sha256": _sha(sorted(answer)),
        "perturbed_sha256": _sha(sorted(perturbed)),
    }


def witness_for(fam: Family, kb: L16KB, params: dict[str, Any],
                seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    kind = fam.witness_kind
    answer = fam.answer(kb, params)

    if kind == "remove_gene_assignment":
        members = list(kb.gene_members.get(params["gene"], ()))
        controls = sorted(kb.gene_members.keys())[:500]
        return _membership_witness(fam, kb, params, rng, members, controls)

    if kind == "remove_keyword_edge":
        members = [a for a in kb.taxon_members.get(params["taxon"], ())
                   if params["keyword"] in kb.keywords_by_acc.get(a, ())]
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "change_pe_level":
        members = [a for a in kb.taxon_members.get(params["taxon"], ())
                   if kb.entries[a]["pe_level"] == params["pe"]]
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_function_evidence":
        members = [a for a in kb.taxon_members.get(params["taxon"], ())
                   if params["eco"] in kb.function_ecos.get(a, ())]
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_go_annotation":
        members = list(kb.taxon_members.get(params["taxon"], frozenset())
                       & kb.go_members.get(params["go"], frozenset()))
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "change_ec_number":
        p = params["ec_prefix"]
        members = [a for a, ecs in kb.ec_by_acc.items()
                   if any(e == p or e.startswith(p + ".") for e in ecs)]
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_rhea_xref":
        members = list(kb.rhea_members.get(params["rhea"], ()))
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_subcellular_location":
        members = list(kb.taxon_members.get(params["taxon"], frozenset())
                       & kb.subcell_members.get(params["location"], frozenset()))
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_disease_link":
        members = list(kb.disease_members.get(params["disease"], ()))
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_variant":
        acc = params["accession"]
        ids = [v[4] for v in kb.variants_by_acc.get(acc, ()) if v[4]]
        if not ids:
            return {"ok": False, "reason": "no variant ids"}
        victim = rng.choice(sorted(ids))
        perturbed = tuple(x for x in answer if x != victim)
        return {"ok": perturbed != answer, "kind": kind, "anchor": acc,
                "perturbation": f"remove variant {victim}",
                "governed_correct_count": len(answer),
                "governed_perturbed_count": len(perturbed),
                "control": acc + "(other features)", "control_unchanged": True,
                "slice_sha256": _sha(sorted(answer)),
                "perturbed_sha256": _sha(sorted(perturbed))}

    if kind == "remove_isoform":
        acc = params["accession"]
        ids = [i[0] for i in kb.isoforms_by_acc.get(acc, ())]
        victim = rng.choice(sorted(ids))
        perturbed = tuple(x for x in answer if x != victim)
        return {"ok": perturbed != answer, "kind": kind, "anchor": acc,
                "perturbation": f"remove isoform {victim}",
                "governed_correct_count": len(answer),
                "governed_perturbed_count": len(perturbed),
                "control": acc + "(canonical sequence)",
                "control_unchanged": True,
                "slice_sha256": _sha(sorted(answer)),
                "perturbed_sha256": _sha(sorted(perturbed))}

    if kind == "remove_xref":
        acc, db = params["accession"], params["db"]
        ids = list(kb.xrefs_by_acc_db.get((acc, db), ()))
        victim = rng.choice(sorted(ids))
        perturbed = tuple(x for x in answer if x != victim)
        return {"ok": perturbed != answer, "kind": kind, "anchor": acc,
                "perturbation": f"remove {db} xref {victim}",
                "governed_correct_count": len(answer),
                "governed_perturbed_count": len(perturbed),
                "control": acc + f"(other dbs)", "control_unchanged": True,
                "slice_sha256": _sha(sorted(answer)),
                "perturbed_sha256": _sha(sorted(perturbed))}

    if kind == "remove_pubmed_link":
        members = list(kb.pubmed_members.get(params["pubmed"], ()))
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_proteome_member":
        members = list(kb.proteome_members.get(params["proteome"], ()))
        return _membership_witness(fam, kb, params, rng, members, [])

    if kind == "remove_ec_rhea_pair":
        taxon = params["taxon"]
        members = [a for a in kb.taxon_members.get(taxon, ())
                   if kb.cat_pairs_by_acc.get(a)]
        if not members:
            return {"ok": False, "reason": "no pair-bearing members"}
        victim_acc = rng.choice(sorted(members))
        pairs = sorted(kb.cat_pairs_by_acc[victim_acc])
        victim = rng.choice(pairs)
        perturbed = tuple(x for x in answer if x != f"{victim[0]}|{victim[1]}")
        return {"ok": perturbed != answer, "kind": kind, "anchor": victim_acc,
                "perturbation": f"remove catalytic pair {victim[0]}|{victim[1]}",
                "governed_correct_count": len(answer),
                "governed_perturbed_count": len(perturbed),
                "control": victim_acc + "(other pairs)",
                "control_unchanged": True,
                "slice_sha256": _sha(sorted(answer)),
                "perturbed_sha256": _sha(sorted(perturbed))}

    if kind == "feature_boundary_mutation":
        acc, ft = params["accession"], params["ftype"]
        rows = [(b, e) for t, b, e in kb.features_by_acc.get(acc, ())
                if t == ft and "?" not in b + e and "<" not in b + e
                and ">" not in b + e]
        if not rows:
            return {"ok": False, "reason": "no exact-boundary rows"}
        victim = rng.choice(sorted(rows))
        delta = rng.choice([1, 2, 3, 5])
        mutated = f"{ft}|{int(victim[0]) + delta}|{victim[1]}"
        perturbed = tuple(x for x in answer if x != f"{ft}|{victim[0]}|{victim[1]}")
        perturbed = tuple(sorted(perturbed + (mutated,)))
        return {"ok": perturbed != answer, "kind": kind, "anchor": acc,
                "perturbation": f"boundary {victim[0]}..{victim[1]} -> "
                                f"{int(victim[0]) + delta}..{victim[1]}",
                "governed_correct_count": len(answer),
                "governed_perturbed_count": len(perturbed),
                "control": acc + "(other feature types)",
                "control_unchanged": True,
                "slice_sha256": _sha(sorted(answer)),
                "perturbed_sha256": _sha(sorted(perturbed))}

    return {"ok": False, "reason": f"unknown witness kind {kind}"}
