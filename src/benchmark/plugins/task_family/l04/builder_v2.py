"""L04 v2 candidate -> qualified shard builder (20K expansion).

Builds 1650 NEW Rhea+ChEBI tasks (330 base goals x 5 state-axis instances)
from the 18 new-mechanism families in ``families_v2.py`` into
``bank/staging_20k/l04`` under dataset_id
``autoontobench-bank-20000-candidate-dev1``, seed 20261101.

Pipeline identical in structure to the legacy 8K builder (builder.py):
extract -> compose -> construct_solution -> sample_start_state ->
native_verify -> semantic_check -> canonicalize/dedup -> package into
immutable 250-record shards. Differences vs the legacy lane:

  * families: the 18 v2 families only (legacy 25 are never resampled);
  * legacy collision guard: every candidate whose canonical-params signature
    equals a legacy (bank/staging/l04, 8K) shipped OR candidate-row signature
    is rejected with code LEGACY_COLLISION;
  * dataset id / seed / output root per the 20K expansion config;
  * incremental shard flush every 250 records + STATUS checkpoint.

CLI:
  .venv/bin/python -m benchmark.plugins.task_family.l04.builder_v2 \
      --out bank/staging_20k/l04 --target-instances 1650 --workers 48
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l04.kb import L04KB, load_kb
from benchmark.plugins.task_family.l04.builder import (
    Candidate, _remove_paths, _rows_by_rxn, family_goal_quotas,
    sha_json, sha_text, split_of, utcnow, wording_leak_check, bundle_leak_scan,
    write_shard, RHEA_REF, CHEBI_REF, MAX_INSTANCE_ATTEMPTS,
)
from benchmark.plugins.task_family.l04.families import (
    ALLOWED_TOOLS, INSTANCES_PER_GOAL, Family, _ancestors, _c2m,
    _master_cache, _rowcount, base_goal_id, canonical_params,
    masters_involving, state_descriptor,
)
from benchmark.plugins.task_family.l04 import families_v2 as F2
from benchmark.plugins.task_family.l04.families_v2 import (
    FAMILIES_V2, FAMILY_V2_BY_ID, source_unit_ids_v2,
)
from benchmark.plugins.verifier.l04.domain import L04Domain
from benchmark.plugins.verifier.l04 import sparql_v2
from benchmark.plugins.verifier.l04 import witness_v2

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "aob-task/5.0"
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
LANE = "L04"
SOURCE_GROUP = "rhea@release-142+chebi@release-256"
SNAPSHOT_ID = "rhea-chebi/rhea-142+chebi-256"
PIN_ROOT_REL = "external_knowledge/pinned/rhea-chebi/rhea-142+chebi-256"
LEGACY_ROOT_REL = "bank/staging/l04"
SUPPORTED_PROFILES = sorted({f.profile for f in FAMILIES_V2})
SHARD_SIZE = 250
STATUS_MARKER = "<!-- L04-20K-PRODUCTION-STATE -->"
DEFAULT_SEED = 20261101


class StatusWriterV2:
    """Restart-safe production state writer (v2 marker namespace)."""

    def __init__(self, path: Path | None):
        self.path = path
        self.flush_lines: list[str] = []

    def update(self, phase: str, counts: dict[str, Any], next_step: str,
               blocker: str = "none", log_line: str | None = None) -> None:
        if log_line:
            self.flush_lines.append(log_line)
        if self.path is None:
            return
        prior = ""
        if self.path.is_file():
            prior = self.path.read_text(encoding="utf-8")
        head = prior.split(STATUS_MARKER)[0].rstrip() if STATUS_MARKER in prior \
            else prior.rstrip()
        lines = [STATUS_MARKER, "",
                 "## Production state (auto-updated by builder_v2; overwritten "
                 "each flush)", "",
                 f"- updated_utc: {utcnow()}",
                 f"- phase: {phase}"]
        lines += [f"- {k}: {v}" for k, v in counts.items()]
        if self.flush_lines:
            lines += ["", "### shard flush log"] + [f"- {x}" for x in self.flush_lines]
        lines += ["", f"- next: {next_step}", f"- blocker: {blocker}", ""]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(head + "\n\n" + "\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# legacy collision guard
# ---------------------------------------------------------------------------

def load_legacy_signatures(repo_root: Path) -> dict[str, set[str]]:
    """All identity material of the legacy 8K L04 lane: task ids, instance
    ids, base-goal ids and canonical-params signatures from both the shipped
    qualified shards and the candidate rows (rejected candidates included, so
    a v2 goal cannot shadow even an unshipped legacy candidate)."""
    root = repo_root / LEGACY_ROOT_REL
    out: dict[str, set[str]] = {"task_id": set(), "instance_id": set(),
                                "base_goal_id": set(), "params_sig": set()}
    for shard in sorted((root / "qualified").glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out["task_id"].add(r["task_id"])
            out["instance_id"].add(r["instance_id"])
            out["base_goal_id"].add(r["base_goal_id"])
            out["params_sig"].add(r["combination_signature"])
    for shard in sorted((root / "candidates").glob("shard-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out["base_goal_id"].add(row["base_goal_id"])
            if "params" in row:
                out["params_sig"].add(sha_text(canonical_params(row["params"])))
    return out


# ---------------------------------------------------------------------------
# v2 public evidence bundles
# ---------------------------------------------------------------------------

def _scope_reactions_v2(kb: L04KB, family: Family,
                        params: dict[str, Any]) -> frozenset[str]:
    """Masters whose rows are needed in the public slice for this v2 task."""
    masters = _master_cache(kb)
    c2m = _c2m(kb)
    fid = family.family_id
    ids: set[str] = set()
    if "chebi" in params and fid != "org_masters_involving_ph73_microspecies":
        ids |= set(c2m.get(params["chebi"], ()))
    if fid == "org_masters_involving_ph73_microspecies":
        ids |= set(c2m.get(F2._ph73_map(kb)[params["chebi"]], ()))
        ids |= set(c2m.get(params["chebi"], ()))
    if fid == "org_masters_isomer_formula_class":
        ids |= set(F2._masters_with_formula(kb, params["formula"]))
    if fid == "org_masters_by_name_token":
        tok = params["token"]
        for e in c2m:
            if tok in kb.names.get(e, "").lower():
                ids |= set(c2m.get(e, ()))
    if fid == "org_masters_by_smiles_substring":
        sm = F2._smiles_map(kb)
        for e in c2m:
            if e in sm and params["smiles_sub"] in sm[e]:
                ids |= set(c2m.get(e, ()))
    if fid == "org_masters_isa_neighborhood":
        own = set(c2m.get(params["chebi"], ()))
        mn = F2._master_neighborhood(kb)
        nb: set[str] = set()
        for m in own:
            nb |= set(mn.get(m, ()))
        ids |= own | nb
    if fid == "org_masters_go_mediated":
        ids |= set(F2._masters_go_mediated(kb, params["chebi"]))
    if fid == "org_masters_closure_conjunction":
        uni = sorted(kb.chebi_to_reactions)
        for key in ("ancestor_a", "ancestor_b"):
            a = params[key]
            for c in uni:
                if a in _ancestors(kb, c):
                    ids |= set(c2m.get(c, ()))
    return frozenset(ids)


def _scope_chebi_entities(kb: L04KB, family: Family,
                          params: dict[str, Any]) -> set[str]:
    """ChEBI entities whose auxiliary evidence (names/SMILES/ph73/edges)
    belongs in the bundle: every entity participating in scope rows, plus
    the closure cone of closure-family participants, plus the query entity."""
    scope = _scope_reactions_v2(kb, family, params)
    scope_rows = {r for r in scope}
    ents: set[str] = set()
    for r, _s, _p, _c, ch in kb.participants:
        if r in scope_rows:
            ents.add(ch)
    for key in ("chebi", "ancestor_a", "ancestor_b"):
        if key in params:
            ents.add(params[key])
    if family.family_id in ("org_masters_closure_conjunction",):
        out = set(ents)
        for e in ents:
            out |= set(_ancestors(kb, e))
        return out
    if family.family_id == "con_ancestor_closure_via_probes":
        return {params["chebi"]}
    return ents


def build_public_bundle_v2(kb: L04KB, family: Family, params: dict[str, Any],
                           variant: str, out_dir: Path,
                           distractor_seed: int | None = None) -> list[Path]:
    """Agent-visible evidence bundle for v2 org/app families.

    Same contract as the legacy bundle writer: tsv_view emits JSONL row
    files; rdf_view emits the Turtle slice of the participant cross-reference
    core. v2 families additionally emit the auxiliary evidence file(s) their
    mechanism requires (formulas, charges, citations, is_a links, names,
    SMILES, pH-7.3 rows, subclass edges). Distractor reactions (seeded) are
    mixed into the core rows exactly as in the legacy lane.
    """
    masters = _master_cache(kb)
    scope = _scope_reactions_v2(kb, family, params)
    rows_by_rxn = _rows_by_rxn(kb)
    fid = family.family_id
    rows = [r for rxn in sorted(scope) for r in rows_by_rxn.get(rxn, [])]
    if distractor_seed is not None:
        rng = random.Random(distractor_seed)
        extra = rng.sample(sorted(masters - scope),
                           k=min(40, len(masters - scope)))
        rows += [r for rxn in sorted(extra) for r in rows_by_rxn.get(rxn, [])]
        scope = scope | frozenset(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(name: str, records: list[dict[str, Any]]) -> None:
        path = out_dir / name
        text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                       for r in records)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    part_records = [{"rhea_id": r, "side": s, "participant": p,
                     "compound": c, "chebi": ch}
                    for r, s, p, c, ch in rows]
    meta = []
    for r in sorted(scope, key=lambda x: int(x.split(":")[1])):
        u = kb.reactions[r]
        rec = {"rhea_id": r, "equation": u["equation"], "status": u["status"],
               "is_transport": u["is_transport"],
               "is_chemically_balanced": u["is_balanced"],
               "children_lr_rl": u["children_lr_rl"],
               "children_bi": u["children_bi"]}
        if u["ec"]:
            rec["ec"] = u["ec"]
        meta.append(rec)

    if variant == "tsv_view":
        emit("rhea_participants.jsonl", part_records)
        emit("rhea_reactions.jsonl", meta)
    elif variant == "rdf_view":
        lines = ["@prefix rh: <http://rdf.rhea-db.org/> .",
                 "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                 "@prefix chebi: <http://purl.obolibrary.org/obo/CHEBI_> .", ""]
        for r, s, p, c, ch in rows:
            num = r.split(":")[1]
            lines.append(f"<http://rdf.rhea-db.org/{num}> rh:accession \"{r}\" ;")
            lines.append("  rdfs:subClassOf rh:Reaction ;")
            lines.append(f"  rh:side <http://rdf.rhea-db.org/{num}_{s}> .")
            lines.append(f"<http://rdf.rhea-db.org/{num}_{s}> rh:contains <{p}> .")
            lines.append(f"<{p}> rh:compound <{c}> .")
            lines.append(f"<{c}> rh:accession \"{ch}\" .")
        path = out_dir / "rhea_slice.ttl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    else:
        raise ValueError(variant)

    scope_compounds = sorted({c for _r, _s, _p, c, _ch in rows})
    scope_entities = _scope_chebi_entities(kb, family, params)
    if distractor_seed is not None:
        # distractor entities ride along in the core rows; their auxiliary
        # rows are included as well so the bundle stays self-consistent
        scope_entities |= {ch for _r, _s, _p, _c, ch in rows}
    fmap, charge_of, _comp2ch = F2._compound_maps(kb)

    if fid == "org_masters_isomer_formula_class":
        emit("compound_formulas.jsonl",
             [{"compound": c, "formula": fmap.get(c, "")}
              for c in scope_compounds])
    if fid == "org_masters_by_charge_and_chebi":
        emit("compound_charges.jsonl",
             [{"compound": c, "charge": charge_of.get(c, "")}
              for c in scope_compounds])
    if fid == "org_masters_by_citation_band_and_chebi":
        emit("rhea_citations.jsonl",
             [{"rhea_id": m, "citations": kb.reactions[m]["citations"]}
              for m in sorted(scope, key=lambda x: int(x.split(":")[1]))])
    if fid == "org_masters_isa_neighborhood":
        children: set[str] = set()
        for m in scope:
            u = kb.reactions[m]
            children |= set(u["children_lr_rl"]) | set(u["children_bi"])
        emit("rhea_isa_links.jsonl",
             [{"from": row["FROM_REACTION_ID"], "to": row["TO_REACTION_ID"]}
              for row in kb.relationships
              if row["FROM_REACTION_ID"] in children
              or row["TO_REACTION_ID"] in children])
    if fid == "org_masters_by_name_token":
        emit("chebi_names.jsonl",
             [{"chebi": e, "name": kb.names.get(e, "")}
              for e in sorted(scope_entities)])
    if fid == "org_masters_by_smiles_substring":
        sm = F2._smiles_map(kb)
        emit("chebi_smiles.jsonl",
             [{"chebi": e, "smiles": sm.get(e, "")}
              for e in sorted(scope_entities) if e in sm])
    if fid == "org_masters_involving_ph73_microspecies":
        ph = F2._ph73_map(kb)
        rows_out = [{"chebi": e, "ph73": ph[e]} for e in sorted(scope_entities)
                    if e in ph]
        emit("chebi_ph73.jsonl", rows_out)
    if fid == "org_masters_closure_conjunction":
        edges = sorted({(e, a) for e in scope_entities
                        for a in _ancestors(kb, e)})
        emit("chebi_subclass_edges.jsonl",
             [{"chebi": c, "ancestor": a} for c, a in edges])
    if fid == "org_masters_go_mediated":
        go_rows = [{"rhea_id": m, "go": g}
                   for m in sorted(scope) for g in
                   sorted(kb.go_by_master.get(m, ()))]
        if go_rows:
            emit("rhea_go_links.jsonl", go_rows)
    return written


def build_repair_bundle_v2(kb: L04KB, family: Family, params: dict[str, Any],
                           seed: int, out_dir: Path
                           ) -> tuple[list[Path], dict[str, Any]]:
    """Corrupted artifact + public evidence for the v2 repair families."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    masters = _master_cache(kb)
    rows_by_rxn = _rows_by_rxn(kb)
    written: list[Path] = []

    def emit(name: str, payload: Any) -> None:
        path = out_dir / name
        text = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                if name.endswith(".json") else
                "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                        for r in payload))
        path.write_text(text + ("" if text.endswith("\n") else "\n"),
                        encoding="utf-8")
        written.append(path)

    fid = family.family_id
    scope_rxns = masters_involving(kb, masters, params["chebi"])
    emit("rhea_participants.jsonl",
         [{"rhea_id": r, "side": s, "participant": p, "compound": c,
           "chebi": ch}
          for rxn in sorted(scope_rxns) for r, s, p, c, ch in
          rows_by_rxn.get(rxn, [])])

    if fid == "rep_go_link_table_repair":
        gold = {m: sorted(kb.go_by_master.get(m, ())) for m in sorted(scope_rxns)}
        gold = {m: g for m, g in gold.items()}
        contrib: dict[str, list[str]] = {}
        for m, gos in gold.items():
            for g in gos:
                contrib.setdefault(g, []).append(m)
        unique = [(m, g) for g, ms_ in contrib.items() if len(ms_) == 1
                  for m in ms_]
        if len(unique) < 2:
            return written, {"ok": False,
                             "reason": "scope lacks uniquely-contributed GO links"}
        defects = rng.sample(unique, 2)
        all_go = sorted({g for gos in gold.values() for g in gos})
        decoys = sorted(set(kb.go_to_masters) - set(all_go))
        if len(decoys) < 2:
            return written, {"ok": False, "reason": "no decoy GO terms"}
        corrupt = {m: list(gos) for m, gos in gold.items()}
        defect_rows = []
        for m, g in defects:
            d = decoys[rng.randrange(len(decoys))]
            corrupt[m] = sorted((set(corrupt[m]) - {g}) | {d})
            defect_rows.append({"rhea_id": m, "true": g, "corrupted_to": d})
        emit("go_link_artifact.json", {
            "profile": "rdf-sparql/1",
            "links": [{"rhea_id": m, "go": g} for m in sorted(corrupt)
                      for g in corrupt[m]],
            "note": "artifact under repair; some entries may be corrupted"})
        emit("probe_spec.json", {
            "tool": "probe_native",
            "arity": "one reaction id per probe",
            "returns": "published GO terms of the reaction",
            "budget": 4})
        return written, {"ok": True, "kind": "go_link_table", "k": 2,
                         "defects": defect_rows, "probe_budget": 4}

    if fid == "rep_xref_table_repair":
        db = params["db"]
        gold = {m: sorted(kb.xrefs_by_master.get(m, ()))
                for m in sorted(scope_rxns)}
        contrib: dict[str, list[str]] = {}
        for m, xs in gold.items():
            for x in xs:
                if x.startswith(db + ":"):
                    contrib.setdefault(x, []).append(m)
        unique = [(m, x) for x, ms_ in contrib.items() if len(ms_) == 1
                  for m in ms_]
        if len(unique) < 3:
            return written, {"ok": False,
                             "reason": "scope lacks uniquely-contributed xrefs"}
        defects = rng.sample(unique, 3)
        other_dbs = [d for d in ("KEGG_REACTION", "METACYC", "ECOCYC",
                                 "REACTOME", "MACIE") if d != db]
        corrupt = {m: list(xs) for m, xs in gold.items()}
        defect_rows = []
        for m, x in defects:
            wrong_db = other_dbs[rng.randrange(len(other_dbs))]
            corrupt[m] = sorted((set(corrupt[m]) - {x})
                                | {wrong_db + ":" + x.split(":", 1)[1]})
            defect_rows.append({"rhea_id": m, "true": x,
                                "corrupted_to": wrong_db + ":"
                                + x.split(":", 1)[1]})
        emit("xref_artifact.json", {
            "profile": "rdf-sparql/1",
            "xrefs": [{"rhea_id": m, "xref": x} for m in sorted(corrupt)
                      for x in corrupt[m]],
            "note": "artifact under repair; some entries may carry a wrong "
                    "database prefix"})
        emit("probe_spec.json", {
            "tool": "probe_native",
            "arity": "one reaction id per probe",
            "returns": "published cross-reference list of the reaction",
            "budget": 5})
        return written, {"ok": True, "kind": "xref_table", "k": 3,
                         "defects": defect_rows, "probe_budget": 5}

    if fid == "rep_status_table_repair":
        status = params["status"]
        with_s = [m for m in sorted(scope_rxns)
                  if kb.reactions[m]["status"] == status]
        without_s = [m for m in sorted(scope_rxns)
                     if kb.reactions[m]["status"] != status]
        if not with_s or not without_s:
            return written, {"ok": False,
                             "reason": "scope lacks both status classes"}
        out_m = rng.choice(with_s)
        in_m = rng.choice(without_s)
        statuses = ("Approved", "Preliminary", "Obsolete")
        wrong_for_out = rng.choice([s for s in statuses if s != status])
        corrupt = {m: kb.reactions[m]["status"] for m in sorted(scope_rxns)}
        corrupt[out_m] = wrong_for_out
        corrupt[in_m] = status
        emit("status_artifact.json", {
            "profile": "rdf-sparql/1",
            "status": [{"rhea_id": m, "status": corrupt[m]}
                       for m in sorted(corrupt)],
            "note": "artifact under repair; some entries may carry a wrong "
                    "curation status"})
        emit("probe_spec.json", {
            "tool": "probe_native",
            "arity": "one reaction id per probe",
            "returns": "published curation status of the reaction",
            "budget": 4})
        return written, {"ok": True, "kind": "status_table", "k": 2,
                         "defects": [
                             {"rhea_id": out_m, "true": status,
                              "corrupted_to": wrong_for_out},
                             {"rhea_id": in_m,
                              "true": kb.reactions[in_m]["status"],
                              "corrupted_to": status}],
                         "probe_budget": 4}

    return written, {"ok": False, "reason": f"no v2 repair bundle for {fid}"}


def build_construction_bundle_v2(kb: L04KB, family: Family,
                                 params: dict[str, Any], seed: int,
                                 out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(name: str, payload: Any) -> None:
        path = out_dir / name
        text = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                if name.endswith(".json") else
                "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                        for r in payload))
        path.write_text(text + ("" if text.endswith("\n") else "\n"),
                        encoding="utf-8")
        written.append(path)

    if family.family_id == "con_ancestor_closure_via_probes":
        anc = kb.chebi_ancestors(params["chebi"])
        budget = len(anc) + 1  # BFS visits the entity + every ancestor once
        emit("chebi_entity.json", {
            "chebi": params["chebi"],
            "label": kb.chebi_labels.get(params["chebi"], ""),
            "note": "subclass parents are NOT public; use probe_native"})
        emit("probe_spec.json", {
            "tool": "probe_native",
            "arity": "one ChEBI class id per probe",
            "returns": "direct rdfs:subClassOf parents of the class in the "
                       "pinned ChEBI ontology",
            "budget": budget})
        return written, {"ok": True, "kind": "con_closure",
                         "probe_budget": budget}
    return written, {"ok": False,
                     "reason": f"no v2 construction bundle for {family.family_id}"}


# ---------------------------------------------------------------------------
# v2 sampling (with legacy-signature exclusion)
# ---------------------------------------------------------------------------

def sample_base_goals_v2(kb: L04KB, family: Family, n_goals: int, seed: int,
                         legacy_sigs: set[str]
                         ) -> tuple[list[Candidate], list[dict[str, Any]]]:
    rng = random.Random(seed)
    masters = _master_cache(kb)
    out: list[Candidate] = []
    rejects: list[dict[str, Any]] = []
    seen_params: set[str] = set()
    for params in family.propose(kb, rng):
        key = canonical_params(params)
        if key in seen_params:
            continue
        seen_params.add(key)
        if sha_text(key) in legacy_sigs:
            rejects.append({"family": family.family_id, "params": params,
                            "code": "LEGACY_COLLISION",
                            "reason": "canonical-params signature already in "
                                      "the legacy 8K lane"})
            continue
        try:
            answer = family.answer(kb, params, masters)
        except Exception as exc:
            rejects.append({"family": family.family_id, "params": params,
                            "code": "UNSOLVABLE",
                            "reason": f"reference_error: {exc}"})
            continue
        n = len(set(answer))
        if n == 0:
            rejects.append({"family": family.family_id, "params": params,
                            "code": "UNSOLVABLE", "reason": "empty answer"})
            continue
        if not (family.min_answer <= n <= family.max_answer):
            rejects.append({"family": family.family_id, "params": params,
                            "code": "TRIVIAL",
                            "reason": f"answer size {n} outside "
                                      f"[{family.min_answer},{family.max_answer}]"})
            continue
        bid = base_goal_id(family.family_id, params)
        out.append(Candidate(
            family=family, params=params, base_goal_id=bid,
            answer=tuple(sorted(set(answer))),
            mechanism_group_id=f"l04/{family.family_id}/{bid[:12]}",
            user_goal=family.goal_text(params, kb),
            provenance=("source_exact" if family.family_id.startswith("app_")
                        else "source_derived_composition"),
            seed=seed,
        ))
        if len(out) >= n_goals:
            break
    return out, rejects


# ---------------------------------------------------------------------------
# SPARQL reference verification (parallel workers; lazy per-process graphs)
# ---------------------------------------------------------------------------

def _sparql_worker_v2(job: dict[str, Any]) -> dict[str, Any]:
    pin_root = Path(job["pin_root"])
    kb = load_kb(str(pin_root))
    try:
        ev = sparql_v2.verify_reference_v2(
            pin_root, job["family"], job["params"], tuple(job["expected"]), kb)
    except Exception as exc:
        return {"base_goal_id": job["base_goal_id"], "verdict": "ERROR",
                "reason": f"{type(exc).__name__}: {exc}"[:400]}
    ev["base_goal_id"] = job["base_goal_id"]
    return ev


def run_sparql_verification_v2(pin_root: Path, candidates: list[Candidate],
                               workers: int) -> dict[str, dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [{"pin_root": str(pin_root), "family": c.family.family_id,
             "params": c.params, "expected": list(c.answer),
             "base_goal_id": c.base_goal_id} for c in candidates]
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_sparql_worker_v2, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            ev = fut.result()
            results[ev["base_goal_id"]] = ev
            if (i + 1) % 50 == 0:
                print(f"  sparql verify {i + 1}/{len(jobs)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# selection (cheap gates + quota + surplus donation), v2 family registry
# ---------------------------------------------------------------------------

def select_ship_goals_v2(candidates: list[Candidate],
                         sparql_evidence: dict[str, dict[str, Any]],
                         quotas: dict[str, int], goals_total: int,
                         family_goal_cap: int, stats: dict[str, Any],
                         legacy_base_goals: set[str]
                         ) -> tuple[list[Candidate], list[Candidate]]:
    rejections: list[dict[str, Any]] = []

    def reject(c: Candidate, code: str, reason: str) -> None:
        rejections.append({"family": c.family.family_id, "params": c.params,
                           "base_goal_id": c.base_goal_id, "code": code,
                           "reason": reason})
        stats["rejections"][code] = stats["rejections"].get(code, 0) + 1

    ship: list[Candidate] = []
    surplus: list[Candidate] = []
    per_family_ship: dict[str, int] = {f.family_id: 0 for f in FAMILIES_V2}
    seen_base_goals: set[str] = set()
    fam_index = {f.family_id: i for i, f in enumerate(FAMILIES_V2)}

    for c in candidates:
        fam = c.family
        if c.base_goal_id in legacy_base_goals:
            reject(c, "LEGACY_COLLISION",
                   "base-goal id already shipped in the legacy 8K lane")
            continue
        if c.base_goal_id in seen_base_goals:
            reject(c, "DUPLICATE", "base goal already selected")
            continue
        if fam.profile not in SUPPORTED_PROFILES:
            reject(c, "PROFILE_UNSUPPORTED", fam.profile)
            continue
        ev = sparql_evidence.get(c.base_goal_id, {})
        if ev.get("verdict") != "PASS":
            reject(c, "UNSOLVABLE",
                   "sparql reference disagreement: "
                   + (ev.get("reason") or
                      f"expected_sha {ev.get('expected_sha256')} != "
                      f"answer_sha {ev.get('answer_sha256')}"))
            continue
        leak = wording_leak_check(c.user_goal, c.answer, c.params)
        if leak:
            reject(c, "LEAKAGE", leak)
            continue
        c.reference_evidence = ev
        seen_base_goals.add(c.base_goal_id)
        if per_family_ship[fam.family_id] < quotas[fam.family_id]:
            ship.append(c)
            per_family_ship[fam.family_id] += 1
        else:
            surplus.append(c)

    shortfall = goals_total - len(ship)
    donated = 0
    if shortfall > 0:
        by_fam: dict[str, list[Candidate]] = {f.family_id: []
                                              for f in FAMILIES_V2}
        for c in surplus:
            by_fam[c.family.family_id].append(c)
        picked: list[Candidate] = []
        while shortfall > 0:
            progressed = False
            for fam in FAMILIES_V2:
                if shortfall <= 0:
                    break
                fid = fam.family_id
                if per_family_ship[fid] >= family_goal_cap:
                    continue
                if by_fam[fid]:
                    c = by_fam[fid].pop(0)
                    picked.append(c)
                    per_family_ship[fid] += 1
                    shortfall -= 1
                    donated += 1
                    progressed = True
            if not progressed:
                break
        ship.extend(picked)
        ship.sort(key=lambda c: (fam_index[c.family.family_id], c.seq))
        surplus = [c for fam in FAMILIES_V2 for c in by_fam[fam.family_id]]
    stats["surplus_donated_goals"] = donated
    stats["goal_shortfall_after_donation"] = goals_total - len(ship)
    stats["rejections"].setdefault("_selection_rows", rejections)
    return ship, surplus


# ---------------------------------------------------------------------------
# goal-atomic instance materialization (v2 bundles + v2 witnesses)
# ---------------------------------------------------------------------------

def reference_plan_v2(fam: Family, meta: dict[str, Any]) -> dict[str, Any]:
    if fam.state_axis == "corruption_seed":
        return {"steps": ["inspect corrupted artifact",
                          "align with public evidence", "probe suspect entries",
                          "repair defect entries", "submit repaired artifact",
                          "submit answer"],
                "probe_budget": meta.get("probe_budget", 0), "min_actions": 2}
    if fam.state_axis == "evidence_sample_seed":
        return {"steps": ["probe direct parents within budget",
                          "assemble transitive closure", "submit answer"],
                "probe_budget": meta.get("probe_budget", 0), "min_actions": 2}
    return {"steps": ["align provided source views", "compute exact answer set",
                      "submit answer"],
            "probe_budget": 0, "min_actions": 1}


def materialize_goal_v2(c: Candidate, kb: L04KB, out_root: Path,
                        domain: L04Domain, seen_instances: set[str],
                        stats: dict[str, Any],
                        start_inst_idx: int = 0,
                        n_instances: int = INSTANCES_PER_GOAL,
                        ) -> list[dict[str, Any]] | None:
    fam = c.family
    made: list[dict[str, Any]] = []
    goal_paths: list[Path] = []
    inst_rejections: list[dict[str, Any]] = []
    inst_idx = start_inst_idx
    attempts = 0
    bundle_failure: str | None = None
    while len(made) < n_instances and attempts < MAX_INSTANCE_ATTEMPTS:
        state_desc = state_descriptor(fam, c.seed, inst_idx)
        wit_seed = (c.seed * 131_071 + inst_idx * 1_009) & 0xFFFFFFFF
        inst_idx += 1
        attempts += 1
        instance_id = sha_text(
            f"{c.base_goal_id}|{json.dumps(state_desc, sort_keys=True)}")[:32]
        if instance_id in seen_instances:
            continue
        task_id = f"l04-{c.base_goal_id[:12]}-{instance_id[:8]}"
        public_dir = out_root / "public" / task_id

        meta: dict[str, Any] = {"ok": True}
        if fam.state_axis == "corruption_seed":
            bundle_files, meta = build_repair_bundle_v2(
                kb, fam, c.params, state_desc["seed"], public_dir)
        elif fam.state_axis == "evidence_sample_seed":
            bundle_files, meta = build_construction_bundle_v2(
                kb, fam, c.params, state_desc["seed"], public_dir)
        else:
            bundle_files = build_public_bundle_v2(
                kb, fam, c.params, state_desc["variant"], public_dir,
                distractor_seed=state_desc.get("distractor_seed"))
        goal_paths.append(public_dir)
        if not meta.get("ok"):
            bundle_failure = f"state build: {meta.get('reason')}"
            break
        leak = bundle_leak_scan(bundle_files, c.answer, fam)
        if leak:
            bundle_failure = leak
            break

        wit = witness_v2.certify_witness_v2(kb, fam.family_id,
                                            fam.witness_kind, c.params,
                                            wit_seed)
        if not wit.get("ok"):
            _remove_paths([public_dir])
            inst_rejections.append({
                "family": fam.family_id, "params": c.params,
                "base_goal_id": c.base_goal_id, "code": "NON_SEMANTIC",
                "reason": f"witness failed: {wit.get('reason', '')}"})
            continue

        ev = c.reference_evidence
        init_state = {
            "task_id": task_id, "family": fam.family_id, "params": c.params,
            "query": {"family": fam.family_id, "params": c.params},
            "bundle_files": [str(p.relative_to(out_root))
                             for p in bundle_files],
            "state": state_desc,
            "bundle_sha256": sha_text("".join(
                sha_text(p.read_text()) for p in bundle_files)),
        }
        state_path = public_dir / "initial_state.json"
        state_path.write_text(json.dumps(init_state, indent=2, sort_keys=True)
                              + "\n", encoding="utf-8")

        query_obj = {"family": fam.family_id, "params": c.params}
        st = domain.reset(task_id, query_obj)
        reset_result = domain.verify_terminal(
            st, c.answer, reference_engine=ev.get("engine", "rdflib"),
            evidence_graph_sha256=ev.get("graph_manifest_sha", ""))
        st2, _fb = domain.step(
            st, {"type": "submit_answer", "answer": list(c.answer)})
        term_result = domain.verify_terminal(
            st2, c.answer, reference_engine=ev.get("engine", "rdflib"),
            evidence_graph_sha256=ev.get("graph_manifest_sha", ""))
        if reset_result.goal_reached or not term_result.goal_reached:
            bundle_failure = "domain replay failed to discriminate"
            break

        seen_instances.add(instance_id)
        oracle_handle = "l04-oracle-" + sha_text(task_id)[:16]
        oracle = {
            "task_id": task_id,
            "family": fam.family_id,
            "params": c.params,
            "expected_answer": list(c.answer),
            "expected_answer_sha256": sha_json(sorted(set(c.answer))),
            "reference_plan": reference_plan_v2(fam, meta),
            "witness": wit,
            "corruption": meta,
            "sparql_reference": ev,
        }
        oracle_path = out_root / "oracle" / f"{oracle_handle}.json"
        oracle_path.parent.mkdir(parents=True, exist_ok=True)
        oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True,
                                          ensure_ascii=False) + "\n",
                               encoding="utf-8")
        goal_paths.append(oracle_path)
        made.append(task_record_v2(
            c=c, task_id=task_id, instance_id=instance_id,
            state_desc=state_desc, public_dir=public_dir,
            state_path=state_path, oracle_handle=oracle_handle,
            out_root=out_root, meta=meta, wit=wit,
            reset_ok=not reset_result.goal_reached,
            terminal_ok=term_result.goal_reached))

    stats["rejections"].setdefault("_instance_rows", []).extend(inst_rejections)
    for r in inst_rejections:
        stats["rejections"][r["code"]] = stats["rejections"].get(r["code"], 0) + 1
    if len(made) < n_instances:
        _remove_paths(goal_paths)
        reason = bundle_failure or (
            f"only {len(made)}/{n_instances} instances materialized "
            f"in {attempts} attempts")
        stats["rejections"]["_dropped_goal_rows"].append({
            "family": fam.family_id, "params": c.params,
            "base_goal_id": c.base_goal_id, "code": "UNSOLVABLE",
            "reason": reason})
        stats["rejections"]["UNSOLVABLE"] = stats["rejections"].get("UNSOLVABLE", 0) + 1
        return None
    return made


def task_record_v2(c: Candidate, task_id: str, instance_id: str,
                   state_desc: dict[str, Any], public_dir: Path,
                   state_path: Path, oracle_handle: str, out_root: Path,
                   meta: dict[str, Any], wit: dict[str, Any],
                   reset_ok: bool, terminal_ok: bool) -> dict[str, Any]:
    fam = c.family
    ev = c.reference_evidence
    rel = lambda p: str(p.relative_to(out_root))
    bundle_files = sorted(
        str(p.relative_to(out_root))
        for p in public_dir.iterdir() if p.is_file())
    oracle_rel = f"oracle/{oracle_handle}.json"
    reverify_cmd = (".venv/bin/python -m "
                    "benchmark.plugins.verifier.l04.reverify_v2 "
                    f"--pin-root {PIN_ROOT_REL} "
                    f"--oracle bank/staging_20k/l04/{oracle_rel}")
    return {
        "schema_version": SCHEMA,
        "task_id": task_id,
        "dataset_id": DATASET_ID,
        "lane_id": LANE,
        "domain": "biochemical_linked_data",
        "track": "analytical",
        "task_family": fam.regime,
        "task_regime": fam.regime,
        "family_id": f"l04.{fam.family_id}",
        "base_goal_id": c.base_goal_id,
        "instance_id": instance_id,
        "origin_mechanism_group": c.mechanism_group_id,
        "source_group_id": SOURCE_GROUP,
        "source_refs": [RHEA_REF, CHEBI_REF],
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_unit_ids": list(source_unit_ids_v2(c.params)),
        "combination_signature": sha_text(canonical_params(c.params)),
        "mechanism_group_id": c.mechanism_group_id,
        "complexity": fam.complexity,
        "public": {
            "user_goal": c.user_goal,
            "allowed_tools": list(ALLOWED_TOOLS),
            "budget": {"actions": 4, "probes": meta.get("probe_budget", 0)},
            "observation_refs": bundle_files,
        },
        "initial_state_ref": rel(state_path),
        "public_evidence_ref": rel(public_dir),
        "ontology_contract": fam.profile,
        "semantic_profile": fam.profile,
        "oracle_handle": oracle_handle,
        "private_goal_verifier_ref": oracle_rel,
        "private_native_truth_ref": PIN_ROOT_REL,
        "private_gold_artifact_ref": oracle_rel,
        "private_control_specs_ref": oracle_rel,
        "task_provenance": c.provenance,
        "authorship": "deterministic_template",
        "split": split_of(c.base_goal_id),
        "seed": c.seed,
        "independent_native_checks": [
            "reference_sparql_matches_fast_path", "reset_not_solved",
            "terminal_solved", "witness_governed_flips",
            "witness_control_unchanged", "no_answer_leak"],
        "native_evidence": {
            "stage": "reference_sparql_matches_fast_path",
            "verdict": ev.get("verdict"),
            "engine": ev.get("engine"),
            "graph_set": ev.get("graph_set"),
            "graph_manifest_sha": ev.get("graph_manifest_sha"),
            "query_sha256": ev.get("query_sha256"),
            "answer_sha256": sha_json(sorted(set(c.answer))),
            "commands": [reverify_cmd],
            "witness": {"kind": fam.witness_kind, "ok": True,
                        "witness_sha256": sha_json(wit)},
            "domain_replay": {"reset_goal_reached": not reset_ok,
                              "terminal_goal_reached": terminal_ok},
        },
        "status": "QUALIFIED",
        "qualification": {
            "native_verified": True, "solvable": True, "nontrivial": True,
            "semantic_relevance": True, "leakage_clean": True,
            "release_eligible": True},
        "state_axis": state_desc,
        "answer_size": len(c.answer),
        "x_l04": {
            "witness_kind": fam.witness_kind,
            "witness_ok": True,
            "sparql_graph_set": ev.get("graph_set"),
            "sparql_query_sha256": ev.get("query_sha256"),
            "answer_sha256": sha_json(sorted(set(c.answer))),
        },
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def build_v2(pin_root: Path, out_root: Path, target_instances: int,
             workers: int, seed: int = DEFAULT_SEED,
             status_path: Path | None = None) -> dict[str, Any]:
    t0 = time.time()
    kb = load_kb(str(pin_root))
    status = StatusWriterV2(status_path)
    goals_total = target_instances // INSTANCES_PER_GOAL
    quotas = family_goal_quotas(FAMILIES_V2, goals_total)
    family_goal_cap = max(24, int(target_instances * 0.08)) // INSTANCES_PER_GOAL
    stats: dict[str, Any] = {
        "qualified": 0,
        "rejections": {"_dropped_goal_rows": []},
        "families": {},
        "started_utc": utcnow(),
    }
    legacy = load_legacy_signatures(REPO_ROOT)
    status.update("sampling", {"target_instances": target_instances,
                               "goals_total": goals_total,
                               "families": len(FAMILIES_V2),
                               "legacy_signatures": len(legacy["params_sig"])},
                  "per-family base-goal sampling (legacy collision guard on)")

    candidates: list[Candidate] = []
    sample_rejections: list[dict[str, Any]] = []
    seq = 0
    for i, fam in enumerate(FAMILIES_V2):
        quota = quotas[fam.family_id]
        n_sample = min(fam.max_base_goals, max(quota + 10, int(quota * 2.2) + 1))
        cands, rejects = sample_base_goals_v2(kb, fam, n_sample,
                                              seed + i * 977,
                                              legacy["params_sig"])
        for c in cands:
            c.seq = seq
            seq += 1
        candidates.extend(cands)
        sample_rejections.extend(rejects)
        for r in rejects:
            stats["rejections"][r["code"]] = stats["rejections"].get(r["code"], 0) + 1
        stats["families"][fam.family_id] = {"candidates": len(cands),
                                            "quota_goals": quota}
        print(f"[sample {i+1}/{len(FAMILIES_V2)}] {fam.family_id} "
              f"candidates={len(cands)} rejects={len(rejects)}", flush=True)

    status.update("sparql_verify",
                  {"candidates": len(candidates), "workers": workers},
                  f"rdflib SPARQL reference verification of {len(candidates)} "
                  "candidates",
                  log_line=f"{utcnow()} sampled {len(candidates)} candidates "
                           f"({len(sample_rejections)} sampling rejections)")
    sparql_evidence = run_sparql_verification_v2(pin_root, candidates, workers)
    passed = sum(1 for ev in sparql_evidence.values()
                 if ev.get("verdict") == "PASS")
    print(f"sparql reference: {passed}/{len(candidates)} PASS", flush=True)

    ship, surplus = select_ship_goals_v2(candidates, sparql_evidence, quotas,
                                         goals_total, family_goal_cap, stats,
                                         legacy["base_goal_id"])
    status.update("materialize",
                  {"candidates": len(candidates), "sparql_pass": passed,
                   "ship_goals": len(ship), "surplus_goals": len(surplus)},
                  "goal-atomic materialization with 250-record shard flushes",
                  log_line=f"{utcnow()} selection: ship {len(ship)}/{goals_total} "
                           f"goals (sparql pass {passed}/{len(candidates)}, "
                           f"donated {stats['surplus_donated_goals']})")

    domain = L04Domain()
    seen_instances: set[str] = set()
    buffer: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    shipped_goal_ids: set[str] = set()
    shipped_per_family: dict[str, int] = {f.family_id: 0 for f in FAMILIES_V2}
    surplus_iter = list(surplus)
    plan: list[Candidate] = list(ship)
    idx = 0
    while idx < len(plan):
        c = plan[idx]
        idx += 1
        recs = materialize_goal_v2(c, kb, out_root, domain, seen_instances,
                                   stats)
        if recs is None:
            replacement = None
            while surplus_iter:
                alt = surplus_iter.pop(0)
                if alt.base_goal_id in shipped_goal_ids:
                    continue
                if shipped_per_family[alt.family.family_id] >= family_goal_cap:
                    continue
                recs_alt = materialize_goal_v2(alt, kb, out_root, domain,
                                               seen_instances, stats)
                if recs_alt is not None:
                    replacement = (alt, recs_alt)
                    break
            if replacement is None:
                continue
            alt, recs = replacement
            c = alt
        shipped_goal_ids.add(c.base_goal_id)
        shipped_per_family[c.family.family_id] += 1
        buffer.extend(recs)
        stats["qualified"] += len(recs)
        while len(buffer) >= SHARD_SIZE:
            rows, buffer = buffer[:SHARD_SIZE], buffer[SHARD_SIZE:]
            m = write_shard(out_root, "qualified", len(shard_manifests), rows)
            shard_manifests.append(m)
            status.update(
                "materialize",
                {"qualified_instances": stats["qualified"],
                 "goals_shipped": len(shipped_goal_ids),
                 "shards_written": len(shard_manifests)},
                "continue materialization",
                log_line=f"{utcnow()} {m['shard_id']}: +{m['records']} "
                         f"(cum {stats['qualified']} qualified / "
                         f"{len(shipped_goal_ids)} goals)")
        if idx % 25 == 0:
            print(f"  materialize {idx}/{len(plan)} goals, "
                  f"qualified={stats['qualified']}", flush=True)
    if buffer:
        m = write_shard(out_root, "qualified", len(shard_manifests), buffer)
        shard_manifests.append(m)
        buffer = []

    selection_rows = stats["rejections"].pop("_selection_rows", [])
    instance_rows = stats["rejections"].pop("_instance_rows", [])
    dropped_goal_rows = stats["rejections"].pop("_dropped_goal_rows", [])
    rejection_by_goal: dict[str, dict[str, Any]] = {}
    for r in sample_rejections + selection_rows + instance_rows + dropped_goal_rows:
        bg = r.get("base_goal_id")
        if bg and bg not in rejection_by_goal:
            rejection_by_goal[bg] = r
    candidate_rows = []
    for c in candidates:
        fid = f"l04.{c.family.family_id}"
        ev = sparql_evidence.get(c.base_goal_id, {})
        rej = rejection_by_goal.get(c.base_goal_id)
        shipped = c.base_goal_id in shipped_goal_ids
        candidate_rows.append({
            "schema_version": SCHEMA,
            "dataset_id": DATASET_ID,
            "lane_id": LANE,
            "family_id": fid,
            "base_goal_id": c.base_goal_id,
            "complexity": c.family.complexity,
            "params": c.params,
            "answer_size": len(c.answer),
            "answer_sha256": sha_json(sorted(set(c.answer))),
            "native_verdict": ev.get("verdict", "ERROR"),
            "native_evidence": {k: v for k, v in ev.items()
                                if k != "query_text"},
            "status": "QUALIFIED" if shipped else "REJECTED",
            "rejection": ({"code": rej["code"], "reason": rej["reason"]}
                          if (rej and not shipped) else None),
        })
    cand_manifests = []
    for i in range(0, len(candidate_rows), SHARD_SIZE):
        cand_manifests.append(write_shard(
            out_root, "candidates", i // SHARD_SIZE,
            candidate_rows[i:i + SHARD_SIZE]))

    all_rejections = (sample_rejections + selection_rows + instance_rows
                      + dropped_goal_rows)
    rej_dir = out_root / "rejections"
    rej_dir.mkdir(parents=True, exist_ok=True)
    rej_path = rej_dir / "rejections.jsonl"
    with rej_path.open("w", encoding="utf-8") as fh:
        for row in all_rejections:
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    (rej_dir / "manifest.json").write_text(json.dumps({
        "lane": LANE, "records": len(all_rejections),
        "sha256": sha_text(rej_path.read_text(encoding="utf-8")),
        "path": str(rej_path.resolve()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats.update({
        "wall_seconds": round(time.time() - t0, 2),
        "base_goals": len(shipped_goal_ids),
        "unique_families": len({f"l04.{c.family.family_id}"
                                for c in candidates
                                if c.base_goal_id in shipped_goal_ids}),
        "candidates": len(candidate_rows),
        "candidates_native_pass": passed,
    })
    per_family_qualified_inst: dict[str, int] = {
        f"l04.{f.family_id}": 0 for f in FAMILIES_V2}
    for c in candidates:
        if c.base_goal_id in shipped_goal_ids:
            per_family_qualified_inst[f"l04.{c.family.family_id}"] += \
                INSTANCES_PER_GOAL
    from source_registry.fetchers.l04_rhea_chebi import LICENSE_VERDICTS
    build_report = {
        "lane": LANE,
        "dataset_id": DATASET_ID,
        "seed": seed,
        "expansion": "20K candidate dev1 (new families only; legacy 8K lane "
                     "at bank/staging/l04 untouched)",
        "legacy_guard": {
            "legacy_root": LEGACY_ROOT_REL,
            "legacy_task_ids": len(legacy["task_id"]),
            "legacy_instance_ids": len(legacy["instance_id"]),
            "legacy_base_goal_ids": len(legacy["base_goal_id"]),
            "legacy_param_signatures": len(legacy["params_sig"]),
        },
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_refs": [RHEA_REF, CHEBI_REF],
        "target_qualified_instances": target_instances,
        "instances_per_goal": INSTANCES_PER_GOAL,
        "started_at": stats["started_utc"],
        "wall_seconds": stats["wall_seconds"],
        "base_goals": stats["base_goals"],
        "families": stats["unique_families"],
        "candidates": stats["candidates"],
        "candidates_native_pass": passed,
        "qualified": stats["qualified"],
        "surplus_donated_goals": stats["surplus_donated_goals"],
        "goal_shortfall_after_donation": stats["goal_shortfall_after_donation"],
        "rejections_by_reason": dict(sorted(
            (k, v) for k, v in stats["rejections"].items())),
        "per_family": {
            f"l04.{f.family_id}": {
                "quota_goals": quotas[f.family_id],
                "candidates": stats["families"][f.family_id]["candidates"],
                "qualified_goals": sum(
                    1 for c in candidates
                    if c.family is f and c.base_goal_id in shipped_goal_ids),
                "qualified_instances": per_family_qualified_inst[
                    f"l04.{f.family_id}"],
            }
            for f in FAMILIES_V2
        },
        "shards": [m["shard_id"] for m in shard_manifests],
        "shard_manifests": shard_manifests + cand_manifests,
        "license_verdicts": LICENSE_VERDICTS,
    }
    (out_root / "build_report.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    status.update(
        "done",
        {"qualified_instances": stats["qualified"],
         "base_goals": stats["base_goals"],
         "families": stats["unique_families"],
         "shards": len(shard_manifests),
         "wall_seconds": stats["wall_seconds"]},
        "final verification (counts/fields, reverify samples) + REPORT.md",
        log_line=f"{utcnow()} build complete: {stats['qualified']} qualified / "
                 f"{stats['base_goals']} goals / {stats['unique_families']} "
                 "families")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="L04 v2 Rhea+ChEBI task builder (20K expansion)")
    ap.add_argument("--pin-root", default=str(
        REPO_ROOT / "external_knowledge/pinned/rhea-chebi/rhea-142+chebi-256"))
    ap.add_argument("--out", default=str(REPO_ROOT / "bank/staging_20k/l04"))
    ap.add_argument("--target-instances", type=int, default=1650)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--status-file", default=None)
    ap.add_argument("--smoke", type=int, default=None,
                    help="per-family goal cap for a smoke run (no shards "
                         "smaller than one goal are flushed)")
    args = ap.parse_args(argv)
    status_file = args.status_file or str(Path(args.out) / "STATUS.md")
    stats = build_v2(Path(args.pin_root), Path(args.out),
                     args.target_instances, args.workers, args.seed,
                     Path(status_file))
    print(json.dumps({k: v for k, v in stats.items() if k != "rejections"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
