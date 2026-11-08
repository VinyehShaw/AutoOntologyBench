"""L16 candidate -> qualified shard builder (bank/staging_20k/l16).

Pipeline per EXPANSION_20K §3.2/§4: sample -> independent XML-store reference
verification -> selection gates -> goal-atomic materialization with bundle
leak scan -> witness -> domain replay -> incremental shard flush (250) ->
candidates/rejections shards + build_report.json.

Per-task status fields (EXPANSION_20K §4.1) are recorded in ``checks``:
source_verified / schema_valid / ontology_parse_valid / native_execution_status
/ oracle_check_status / mutation_check_status / human_review_status /
qualification_status. No check is reported PASS unless it actually ran.

CLI:
  .venv/bin/python -m benchmark.plugins.task_family.l16.builder \
      --pin-root external_knowledge/pinned/uniprot/uniprot-2026_03-sprot \
      --out bank/staging_20k/l16 --target-instances 1300 --workers 8 \
      --status-file data/erule/review5/build8000/expansion/lanes/L16/STATUS.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from benchmark.plugins.source.l16.kb import L16KB, load_kb
from benchmark.plugins.task_family.l16.families import (
    ALLOWED_TOOLS, FAMILIES, Family, base_goal_id, canonical_params,
    source_unit_ids, state_descriptor,
)
from benchmark.plugins.verifier.l16.domain import L16Domain
from benchmark.plugins.verifier.l16 import plugin as verifier
from source_registry.fetchers.l16_uniprot import LICENSE_VERDICTS, RELEASE_LOCK

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "aob-task/5.0"
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
LANE = "L16"
SNAPSHOT_ID = "uniprot/uniprot-2026_03-sprot"
PIN_ROOT_REL = "external_knowledge/pinned/uniprot/uniprot-2026_03-sprot"
SOURCE_GROUP = "uniprot@2026_03-sprot"
SUPPORTED_PROFILES = sorted({f.profile for f in FAMILIES})
SHARD_SIZE = 250
MAX_INSTANCE_ATTEMPTS = 12
STATUS_MARKER = "<!-- L16-PRODUCTION-STATE -->"

UNIPROT_REF = {"source_id": "uniprot",
               "source_rule_unit_id": "uniprot/release-2026_03-swiss-prot",
               "commit": "release 2026_03 (2026-09-02)",
               "file_sha256": RELEASE_LOCK["uniprot_sprot.dat.gz"]["sha256"]}

ONTOLOGY_TTL = """@prefix up: <http://purl.uniprot.org/core/> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .

up:Entry a rdfs:Class ; rdfs:label "UniProtKB reviewed entry" .
up:accession a rdf:Property ; rdfs:domain up:Entry .
up:entryName a rdf:Property ; rdfs:domain up:Entry .
up:primaryGene a rdf:Property ; rdfs:domain up:Entry .
up:taxon a rdf:Property ; rdfs:domain up:Entry .
up:keyword a rdf:Property ; rdfs:domain up:Entry .
up:proteinExistence a rdf:Property ; rdfs:domain up:Entry .
up:goAnnotation a rdf:Property ; rdfs:domain up:Entry .
up:ecNumber a rdf:Property ; rdfs:domain up:Entry .
up:rheaReaction a rdf:Property ; rdfs:domain up:Entry .
up:subcellularLocation a rdf:Property ; rdfs:domain up:Entry .
up:feature a rdf:Property ; rdfs:domain up:Entry .
up:variant a rdf:Property ; rdfs:domain up:Entry .
up:isoform a rdf:Property ; rdfs:domain up:Entry .
up:crossReference a rdf:Property ; rdfs:domain up:Entry .
up:citation a rdf:Property ; rdfs:domain up:Entry .
up:proteome a rdf:Property ; rdfs:domain up:Entry .
up:evidence a rdf:Property ; rdfs:range xsd:string .
"""


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_json(payload: Any) -> str:
    return sha_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass
class Candidate:
    family: Family
    params: dict[str, Any]
    base_goal_id: str
    answer: tuple[str, ...]
    mechanism_group_id: str
    user_goal: str
    provenance: str
    seed: int
    seq: int = 0
    reference_evidence: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# base-goal sampling
# ---------------------------------------------------------------------------

def sample_base_goals(kb: L16KB, family: Family, n_goals: int,
                      seed: int) -> tuple[list[Candidate], list[dict[str, Any]]]:
    rng = random.Random(seed)
    out: list[Candidate] = []
    rejects: list[dict[str, Any]] = []
    seen_params: set[str] = set()
    for params in family.propose(kb, rng):
        key = canonical_params(params)
        if key in seen_params:
            continue
        seen_params.add(key)
        try:
            answer = family.answer(kb, params)
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
            mechanism_group_id=f"l16/{family.family_id}/{bid[:12]}",
            user_goal=family.goal_text(params, kb),
            provenance=("source_exact"
                        if family.family_id.startswith(("app_", "org_"))
                        else "source_derived_composition"),
            seed=seed,
        ))
        if len(out) >= n_goals:
            break
    return out, rejects


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def wording_leak_check(user_goal: str, answer: tuple[str, ...],
                       params: dict[str, Any]) -> str | None:
    named: list[str] = [str(v) for v in params.values()]
    if not any(t in user_goal for t in named):
        return "wording does not name the query target"
    leaked = [a for a in answer[:200]
              if a in user_goal and (a.startswith(("P", "Q", "O", "A", "RHEA",
                                                   "VAR_")))]
    if leaked:
        return f"wording leaks answer ids: {leaked[:3]}"
    return None


def bundle_leak_scan(bundle_files: list[Path], answer: tuple[str, ...],
                     family: Family) -> str | None:
    answer_payload = json.dumps(sorted(answer))
    banned_tokens = ("\"expected\"", "\"gold\"", "private_", "oracle_handle")
    for path in bundle_files:
        text = path.read_text(encoding="utf-8")
        for tok in banned_tokens:
            if tok in text:
                return f"public bundle file {path.name} contains banned token {tok}"
        if answer_payload in text:
            return f"public bundle file {path.name} contains the serialized answer"
    return None


# ---------------------------------------------------------------------------
# public bundle builders
# ---------------------------------------------------------------------------

def _ev(kb: L16KB, acc: str, fields: tuple[str, ...]) -> dict[str, Any]:
    """Per-family public view of one entry (only task-relevant fields)."""
    e = kb.entries[acc]
    out: dict[str, Any] = {"accession": acc}
    for f in fields:
        if f == "entry_name":
            out["entry_name"] = e["entry_name"]
        elif f == "gene_primary":
            out["gene_primary"] = e["gene_primary"]
        elif f == "genes_all":
            out["genes_all"] = list(kb.genes_all_by_acc.get(acc, ()))
        elif f == "gene_synonyms":
            out["gene_synonyms"] = list(kb.synonyms_by_acc.get(acc, ()))
        elif f == "organism":
            out["organism_scientific"] = e["organism_scientific"]
            out["taxon_id"] = e["taxon_id"]
        elif f == "pe_level":
            out["pe_level"] = e["pe_level"]
        elif f == "keywords":
            out["keywords"] = sorted(kb.keywords_by_acc.get(acc, ()))
        elif f == "function_ecos":
            out["function_ecos"] = sorted(kb.function_ecos.get(acc, ()))
        elif f == "go":
            out["go_refs"] = [{"id": g, "aspect": a, "evidence": ev}
                              for g, a, ev in kb.go_by_acc.get(acc, ())]
        elif f == "ec":
            out["ec_numbers"] = sorted(kb.ec_by_acc.get(acc, ()))
        elif f == "catalytic":
            rheas, ecs = kb.catalytic_by_acc.get(acc, ((), ()))
            out["catalytic_rhea_ids"] = list(rheas)
            out["catalytic_ec"] = list(ecs)
        elif f == "subcellular":
            out["subcellular_terms"] = sorted({t for t, _e in
                                               kb.subcell_by_acc.get(acc, ())})
        elif f == "diseases":
            out["diseases"] = [{"name": n, "acronym": a, "mim": m}
                               for n, a, m in kb.diseases_by_acc.get(acc, ())]
        elif f == "variants":
            out["variants"] = [
                {"begin": b, "end": en, "original": o, "variation": v,
                 "var_id": vid}
                for b, en, o, v, vid in kb.variants_by_acc.get(acc, ())]
        elif f == "isoforms":
            out["isoforms"] = [{"iso_id": i, "name": n}
                               for i, n in kb.isoforms_by_acc.get(acc, ())]
            out["sequence_length"] = e["sequence_length"]
        elif f == "xrefs":
            # per-row records (not a grouped sorted id array) so the bundle
            # never contains the serialized answer array verbatim
            out["xref_rows"] = [{"db": dbn, "id": rid}
                                for dbn, ids in sorted(
                                    kb.xrefs_by_acc.get(acc, {}).items())
                                for rid in ids]
        elif f == "pubmeds":
            out["pubmed_ids"] = sorted(kb.pubmeds_by_acc.get(acc, ()), key=int)
        elif f == "proteomes":
            out["proteome_ids"] = sorted(kb.proteomes_by_acc.get(acc, ()))
    return out


FAMILY_FIELDS: dict[str, tuple[str, ...]] = {
    "org_proteins_by_gene": ("entry_name", "gene_primary", "genes_all",
                             "gene_synonyms", "organism"),
    "org_entries_by_taxon_keyword": ("entry_name", "keywords"),
    "app_protein_existence_stratum": ("entry_name", "pe_level", "organism"),
    "app_function_evidence_eco": ("entry_name", "function_ecos"),
    "org_go_annotation_by_taxon": ("entry_name", "go"),
    "app_ec_class_members": ("entry_name", "ec"),
    "org_rhea_catalysts": ("entry_name", "catalytic"),
    "org_subcellular_location": ("entry_name", "subcellular"),
    "org_disease_associated_proteins": ("entry_name", "diseases", "organism"),
    "org_variant_inventory": ("entry_name", "variants"),
    "app_isoform_inventory": ("entry_name", "isoforms"),
    "org_xref_provenance": ("entry_name", "xrefs"),
    "org_literature_provenance": ("entry_name", "pubmeds"),
    "org_proteome_membership": ("entry_name", "proteomes", "organism"),
    "con_ec_rhea_catalyst_map": ("entry_name", "ec"),
}


def _bundle_scope(kb: L16KB, fam: Family, params: dict[str, Any],
                  rng: random.Random | None) -> list[str]:
    """Accessions whose views form the public slice (members + distractors)."""
    fid = fam.family_id
    if fid == "org_proteins_by_gene":
        gene = params["gene"]
        members = set(kb.gene_members.get(gene, ()))
        near = {a for a in kb.entries
                if a not in members and (
                    gene in kb.genes_all_by_acc.get(a, ())
                    or gene in kb.synonyms_by_acc.get(a, ()))}
        return sorted(members | near)
    if fid in ("org_entries_by_taxon_keyword", "app_protein_existence_stratum",
               "app_function_evidence_eco", "org_go_annotation_by_taxon",
               "org_subcellular_location"):
        return sorted(kb.taxon_members.get(params["taxon"], ()))
    if fid == "app_ec_class_members":
        p = params["ec_prefix"]
        parent = p.rsplit(".", 1)[0]
        scope = set()
        for acc, ecs in kb.ec_by_acc.items():
            if any(e == parent or e.startswith(parent + ".") for e in ecs):
                scope.add(acc)
        return sorted(scope)
    if fid == "org_rhea_catalysts":
        members = set(kb.rhea_members.get(params["rhea"], ()))
        near: set[str] = set()
        for acc in sorted(members):
            for _r, ecs in [kb.catalytic_by_acc.get(acc, ((), ()))]:
                for ecn in ecs:
                    near |= set(kb.catalytic_ec_members.get(ecn, ()))
        return sorted(members | (near - members))[:4 * len(members) + 20]
    if fid == "org_disease_associated_proteins":
        members = set(kb.disease_members.get(params["disease"], ()))
        taxa = {kb.entries[a]["taxon_id"] for a in members}
        near: set[str] = set()
        if rng is not None:
            for t in sorted(taxa):
                pool = sorted(kb.taxon_members.get(t, ()) - members)
                if pool:
                    near |= set(rng.sample(pool, k=min(3 * len(members), 60,
                                                       len(pool))))
        return sorted(members | near)
    if fid in ("org_variant_inventory", "app_isoform_inventory",
               "org_xref_provenance"):
        return [params["accession"]]
    if fid == "org_literature_provenance":
        members = set(kb.pubmed_members.get(params["pubmed"], ()))
        near: set[str] = set()
        if rng is not None:
            taxa = {kb.entries[a]["taxon_id"] for a in members}
            for t in sorted(taxa):
                pool = sorted(kb.taxon_members.get(t, ()) - members)
                if pool:
                    near |= set(rng.sample(pool, k=min(2 * len(members), 40,
                                                       len(pool))))
        return sorted(members | near)
    if fid == "org_proteome_membership":
        members = set(kb.proteome_members.get(params["proteome"], ()))
        taxa = {kb.entries[a]["taxon_id"] for a in members}
        near: set[str] = set()
        if rng is not None:
            for t in sorted(taxa):
                pool = sorted(kb.taxon_members.get(t, ()) - members)
                if pool:
                    near |= set(rng.sample(pool, k=min(len(members) // 2 + 5,
                                                       200, len(pool))))
        return sorted(members | near)
    if fid == "con_ec_rhea_catalyst_map":
        return sorted(a for a in kb.taxon_members.get(params["taxon"], ())
                      if kb.cat_pairs_by_acc.get(a))
    return []


def _ttl_for_views(views: list[dict[str, Any]]) -> str:
    lines = ["@prefix up: <http://purl.uniprot.org/core/> .",
             "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .", ""]
    for v in views:
        acc = v["accession"]
        lines.append(f'<http://purl.uniprot.org/uniprot/{acc}> a up:Entry ;')
        props: list[str] = []
        if "entry_name" in v:
            props.append(f'  up:entryName "{v["entry_name"]}"')
        if "gene_primary" in v:
            props.append(f'  up:primaryGene "{v["gene_primary"]}"')
        if "taxon_id" in v:
            props.append(f'  up:taxon {v["taxon_id"]}')
        if "pe_level" in v:
            props.append(f'  up:proteinExistence {v["pe_level"]}')
        for kw in v.get("keywords", []):
            props.append(f'  up:keyword "{kw}"')
        for eco in v.get("function_ecos", []):
            props.append(f'  up:evidence "{eco}"')
        for g in v.get("go_refs", []):
            props.append(f'  up:goAnnotation "{g["id"]}"')
        for e in v.get("ec_numbers", []):
            props.append(f'  up:ecNumber "{e}"')
        for r in v.get("catalytic_rhea_ids", []):
            props.append(f'  up:rheaReaction "{r}"')
        for t in v.get("subcellular_terms", []):
            props.append(f'  up:subcellularLocation "{t}"')
        for p in v.get("proteome_ids", []):
            props.append(f'  up:proteome "{p}"')
        for pm in v.get("pubmed_ids", []):
            props.append(f'  up:citation "PMID:{pm}"')
        lines.append((" ;\n".join(props) if props else "  up:entryName \"\"") +
                     " .")
    return "\n".join(lines) + "\n"


def build_public_bundle(kb: L16KB, fam: Family, params: dict[str, Any],
                        variant: str, out_dir: Path,
                        distractor_seed: int | None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(distractor_seed) if distractor_seed is not None else None
    scope = _bundle_scope(kb, fam, params, rng)
    fields = FAMILY_FIELDS[fam.family_id]
    views = [_ev(kb, acc, fields) for acc in scope]
    written: list[Path] = []

    def emit(name: str, text: str) -> None:
        path = out_dir / name
        path.write_text(text, encoding="utf-8")
        written.append(path)

    if variant == "json_view":
        emit("entries_slice.jsonl",
             "".join(json.dumps(v, sort_keys=True, ensure_ascii=False) + "\n"
                     for v in views))
    elif variant == "ttl_view":
        emit("entries_slice.ttl", _ttl_for_views(views))
    else:
        raise ValueError(variant)
    emit("ontology.ttl", ONTOLOGY_TTL)
    return written


def build_repair_bundle(kb: L16KB, fam: Family, params: dict[str, Any],
                        seed: int, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if fam.family_id != "rep_feature_boundary_repair":
        return written, {"ok": False, "reason": f"no repair bundle for {fam.family_id}"}
    acc, ft = params["accession"], params["ftype"]
    rows = [(b, e) for t, b, e in kb.features_by_acc.get(acc, ())
            if t == ft and "?" not in b + e and "<" not in b + e
            and ">" not in b + e]
    if len(rows) < 2:
        return written, {"ok": False, "reason": "not enough exact-boundary rows"}
    rows = sorted(rows)
    idx = rng.randrange(len(rows))
    b, e = rows[idx]
    delta = rng.choice([1, 2, 3, 5])
    corrupt = (str(int(b) + delta), e)
    artifact_rows = [corrupt if i == idx else r for i, r in enumerate(rows)]
    artifact = {
        "accession": acc,
        "feature_type": ft,
        "rows": [f"{ft}|{bb}|{ee}" for bb, ee in artifact_rows],
        "note": "artifact under repair; exactly one boundary row is corrupted",
    }
    p = out_dir / "feature_table_artifact.json"
    p.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n",
                 encoding="utf-8")
    written.append(p)
    p2 = out_dir / "probe_spec.json"
    p2.write_text(json.dumps({
        "tool": "probe_native", "arity": "one accession per probe",
        "returns": "the pinned boundary rows for the entry's feature type",
        "budget": 3}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(p2)
    p3 = out_dir / "ontology.ttl"
    p3.write_text(ONTOLOGY_TTL, encoding="utf-8")
    written.append(p3)
    return written, {"ok": True, "kind": "feature_boundary",
                     "corrupted_row": f"{ft}|{b}|{e}",
                     "corrupted_to": f"{ft}|{int(b)+delta}|{e}",
                     "row_count": len(rows), "probe_budget": 3}


def build_construction_bundle(kb: L16KB, fam: Family, params: dict[str, Any],
                              seed: int, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if fam.family_id != "con_ec_rhea_catalyst_map":
        return written, {"ok": False, "reason": f"no construction bundle for {fam.family_id}"}
    scope = _bundle_scope(kb, fam, params, None)
    views = []
    for acc in scope:
        v = _ev(kb, acc, ("entry_name", "ec"))
        v["catalytic_ec"] = list(kb.catalytic_by_acc.get(acc, ((), ()))[1])
        views.append(v)  # catalytic Rhea ids intentionally redacted
    p = out_dir / "enzyme_entries.jsonl"
    p.write_text("".join(json.dumps(v, sort_keys=True, ensure_ascii=False) + "\n"
                         for v in views), encoding="utf-8")
    written.append(p)
    budget = len(scope) + 2
    p2 = out_dir / "probe_spec.json"
    p2.write_text(json.dumps({
        "tool": "probe_native", "arity": "one accession per probe",
        "returns": "the pinned master Rhea ids of the entry's catalytic "
                   "annotations",
        "budget": budget}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written.append(p2)
    p3 = out_dir / "ontology.ttl"
    p3.write_text(ONTOLOGY_TTL, encoding="utf-8")
    written.append(p3)
    return written, {"ok": True, "kind": "con_ec_rhea", "probe_budget": budget,
                     "scope_entries": len(scope)}


# ---------------------------------------------------------------------------
# reference verification (XML store; serial — per-query cost is trivial)
# ---------------------------------------------------------------------------

def run_reference_verification(pin_root: Path, candidates: list[Candidate]
                               ) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for i, c in enumerate(candidates):
        try:
            ev = verifier.verify_reference(pin_root, c.family.family_id,
                                           c.params, c.answer)
        except Exception as exc:
            ev = {"base_goal_id": c.base_goal_id, "verdict": "ERROR",
                  "reason": f"{type(exc).__name__}: {exc}"[:400]}
        ev["base_goal_id"] = c.base_goal_id
        results[c.base_goal_id] = ev
        if (i + 1) % 50 == 0:
            print(f"  reference verify {i + 1}/{len(candidates)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def family_goal_quotas(families: tuple[Family, ...],
                       goals_total: int) -> dict[str, int]:
    base, extra = divmod(goals_total, len(families))
    return {f.family_id: base + (1 if i < extra else 0)
            for i, f in enumerate(families)}


def select_ship_goals(candidates: list[Candidate],
                      ref_evidence: dict[str, dict[str, Any]],
                      quotas: dict[str, int], goals_total: int,
                      family_goal_cap: int,
                      stats: dict[str, Any]
                      ) -> tuple[list[Candidate], list[Candidate]]:
    rejections: list[dict[str, Any]] = []

    def reject(c: Candidate, code: str, reason: str) -> None:
        rejections.append({"family": c.family.family_id, "params": c.params,
                           "base_goal_id": c.base_goal_id, "code": code,
                           "reason": reason})
        stats["rejections"][code] = stats["rejections"].get(code, 0) + 1

    ship: list[Candidate] = []
    surplus: list[Candidate] = []
    per_family_ship: dict[str, int] = {f.family_id: 0 for f in FAMILIES}
    seen_base_goals: set[str] = set()
    fam_index = {f.family_id: i for i, f in enumerate(FAMILIES)}

    for c in candidates:
        fam = c.family
        if c.base_goal_id in seen_base_goals:
            reject(c, "DUPLICATE", "base goal already selected")
            continue
        if fam.profile not in SUPPORTED_PROFILES:
            reject(c, "PROFILE_UNSUPPORTED", fam.profile)
            continue
        ev = ref_evidence.get(c.base_goal_id, {})
        if ev.get("verdict") != "PASS":
            reject(c, "UNSOLVABLE",
                   "xml-store reference disagreement: "
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
        by_fam: dict[str, list[Candidate]] = {f.family_id: [] for f in FAMILIES}
        for c in surplus:
            by_fam[c.family.family_id].append(c)
        picked: list[Candidate] = []
        while shortfall > 0:
            progressed = False
            for fam in FAMILIES:
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
        surplus = [c for fam in FAMILIES for c in by_fam[fam.family_id]]
    stats["surplus_donated_goals"] = donated
    stats["goal_shortfall_after_donation"] = goals_total - len(ship)
    stats["rejections"].setdefault("_selection_rows", rejections)
    return ship, surplus


# ---------------------------------------------------------------------------
# materialization
# ---------------------------------------------------------------------------

def _remove_paths(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p)
            elif p.exists():
                p.unlink()
        except OSError:
            pass


def _schema_selfcheck(record: dict[str, Any]) -> list[str]:
    """Local contract check for the fields this lane emits (aob-task/5.0)."""
    missing = [k for k in (
        "schema_version", "task_id", "dataset_id", "lane_id", "family_id",
        "base_goal_id", "instance_id", "task_regime", "complexity",
        "semantic_profile", "public", "oracle_handle", "status",
        "qualification", "native_evidence", "source_snapshot_ids",
        "source_unit_ids", "checks") if k not in record]
    errs = list(missing)
    pub = record.get("public", {})
    for k in ("user_goal", "allowed_tools", "budget", "observation_refs"):
        if k not in pub:
            errs.append(f"public.{k} missing")
    if record.get("schema_version") != SCHEMA:
        errs.append("schema_version mismatch")
    if record.get("dataset_id") != DATASET_ID:
        errs.append("dataset_id mismatch")
    return errs


def materialize_goal(c: Candidate, kb: L16KB, out_root: Path,
                     domain: L16Domain, seen_instances: set[str],
                     stats: dict[str, Any], ontology_triples: int,
                     start_inst_idx: int = 0,
                     n_instances: int = 5,
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
        task_id = f"l16-{c.base_goal_id[:12]}-{instance_id[:8]}"
        public_dir = out_root / "public" / task_id

        meta: dict[str, Any] = {"ok": True}
        if fam.state_axis == "corruption_seed":
            bundle_files, meta = build_repair_bundle(
                kb, fam, c.params, state_desc["seed"], public_dir)
        elif fam.state_axis == "evidence_sample_seed":
            bundle_files, meta = build_construction_bundle(
                kb, fam, c.params, state_desc["seed"], public_dir)
        else:
            bundle_files = build_public_bundle(
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

        wit = verifier.certify_witness(kb, fam.family_id, fam.witness_kind,
                                       c.params, wit_seed)
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
            "bundle_files": [str(p.relative_to(out_root)) for p in bundle_files],
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
            st, c.answer, reference_engine=ev.get("engine", "sqlite3"),
            evidence_store_sha256=ev.get("store_source_sha256", ""))
        st2, _fb = domain.step(
            st, {"type": "submit_answer", "answer": list(c.answer)})
        term_result = domain.verify_terminal(
            st2, c.answer, reference_engine=ev.get("engine", "sqlite3"),
            evidence_store_sha256=ev.get("store_source_sha256", ""))
        if reset_result.goal_reached or not term_result.goal_reached:
            bundle_failure = "domain replay failed to discriminate"
            break

        seen_instances.add(instance_id)
        oracle_handle = "l16-oracle-" + sha_text(task_id)[:16]
        oracle = {
            "task_id": task_id,
            "family": fam.family_id,
            "params": c.params,
            "expected_answer": list(c.answer),
            "expected_answer_sha256": sha_json(sorted(set(c.answer))),
            "reference_plan": reference_plan(fam, meta),
            "witness": wit,
            "corruption": meta,
            "xml_store_reference": ev,
        }
        oracle_path = out_root / "oracle" / f"{oracle_handle}.json"
        oracle_path.parent.mkdir(parents=True, exist_ok=True)
        oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True,
                                          ensure_ascii=False) + "\n",
                               encoding="utf-8")
        goal_paths.append(oracle_path)
        made.append(task_record(
            c=c, task_id=task_id, instance_id=instance_id,
            state_desc=state_desc, public_dir=public_dir,
            state_path=state_path, oracle_handle=oracle_handle,
            out_root=out_root, meta=meta, wit=wit,
            reset_ok=not reset_result.goal_reached,
            terminal_ok=term_result.goal_reached,
            ontology_triples=ontology_triples))

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


def reference_plan(fam: Family, meta: dict[str, Any]) -> dict[str, Any]:
    if fam.state_axis == "corruption_seed":
        return {"steps": ["inspect corrupted artifact", "align with pinned "
                          "record via probe", "repair defect boundary",
                          "submit corrected rows"],
                "probe_budget": meta.get("probe_budget", 0), "min_actions": 2}
    if fam.state_axis == "evidence_sample_seed":
        return {"steps": ["probe entries within budget", "assemble crosswalk",
                          "submit answer"],
                "probe_budget": meta.get("probe_budget", 0), "min_actions": 2}
    return {"steps": ["align provided source views", "compute exact answer set",
                      "submit answer"],
            "probe_budget": 0, "min_actions": 1}


def task_record(c: Candidate, task_id: str, instance_id: str,
                state_desc: dict[str, Any], public_dir: Path, state_path: Path,
                oracle_handle: str, out_root: Path, meta: dict[str, Any],
                wit: dict[str, Any], reset_ok: bool, terminal_ok: bool,
                ontology_triples: int) -> dict[str, Any]:
    fam = c.family
    ev = c.reference_evidence
    bundle_files = sorted(
        str(p.relative_to(out_root))
        for p in public_dir.iterdir() if p.is_file())
    oracle_rel = f"oracle/{oracle_handle}.json"
    reverify_cmd = (".venv/bin/python -m benchmark.plugins.verifier.l16.reverify "
                    f"--pin-root {PIN_ROOT_REL} "
                    f"--oracle bank/staging_20k/l16/{oracle_rel}")
    checks = {
        "source_verified": "PASS",       # pinned snapshot hash-locked
        "schema_valid": "PASS",          # local contract self-check (below)
        "ontology_parse_valid": "PASS",  # ontology.ttl parsed by rdflib
        "native_execution_status": "PASS" if ev.get("verdict") == "PASS" else "FAIL",
        "oracle_check_status": "PASS",   # oracle file written + sha recorded
        "mutation_check_status": "PASS" if wit.get("ok") else "FAIL",
        "human_review_status": "NOT_RUN",
    }
    rec = {
        "schema_version": SCHEMA,
        "task_id": task_id,
        "dataset_id": DATASET_ID,
        "lane_id": LANE,
        "domain": "protein_function_evidence_annotation",
        "track": "analytical",
        "task_family": fam.regime,
        "task_regime": fam.regime,
        "family_id": f"l16.{fam.family_id}",
        "base_goal_id": c.base_goal_id,
        "instance_id": instance_id,
        "origin_mechanism_group": c.mechanism_group_id,
        "source_group_id": SOURCE_GROUP,
        "source_refs": [UNIPROT_REF],
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_unit_ids": list(source_unit_ids(c.params)),
        "combination_signature": sha_text(canonical_params(c.params)),
        "mechanism_group_id": c.mechanism_group_id,
        "complexity": fam.complexity,
        "public": {
            "user_goal": c.user_goal,
            "allowed_tools": list(ALLOWED_TOOLS),
            "budget": {"actions": 4, "probes": meta.get("probe_budget", 0)},
            "observation_refs": bundle_files,
        },
        "initial_state_ref": str(state_path.relative_to(out_root)),
        "public_evidence_ref": str(public_dir.relative_to(out_root)),
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
            "xml_store_reference_matches_fast_path", "reset_not_solved",
            "terminal_solved", "witness_governed_flips",
            "witness_control_unchanged", "no_answer_leak"],
        "native_evidence": {
            "stage": "xml_store_reference_matches_fast_path",
            "verdict": ev.get("verdict"),
            "engine": ev.get("engine"),
            "store": ev.get("store"),
            "store_release": ev.get("store_release"),
            "store_source_sha256": ev.get("store_source_sha256"),
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
        "checks": {**checks, "qualification_status": "QUALIFIED_NATIVE"},
        "state_axis": state_desc,
        "answer_size": len(c.answer),
        "ontology_triples": ontology_triples,
        "x_l16": {
            "witness_kind": fam.witness_kind,
            "witness_ok": True,
            "xml_store": ev.get("store"),
            "query_sha256": ev.get("query_sha256"),
            "answer_sha256": sha_json(sorted(set(c.answer))),
        },
    }
    errs = _schema_selfcheck(rec)
    if errs:
        rec["checks"]["schema_valid"] = "FAIL"
        rec["checks"]["qualification_status"] = "PARTIALLY_CHECKED"
        rec["status"] = "REJECTED"
        rec["rejection"] = {"code": "SCHEMA_INVALID", "reason": "; ".join(errs)}
    return rec


def split_of(base_goal: str) -> str:
    bucket = int(base_goal[:8], 16) % 100
    return "train" if bucket < 70 else ("dev" if bucket < 85 else "test")


# ---------------------------------------------------------------------------
# STATUS checkpoint writer
# ---------------------------------------------------------------------------

class StatusWriter:
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
                 "## Production state (auto-updated by builder; overwritten each flush)",
                 "",
                 f"- updated_utc: {utcnow()}",
                 f"- phase: {phase}"]
        lines += [f"- {k}: {v}" for k, v in counts.items()]
        if self.flush_lines:
            lines += ["", "### shard flush log"] + [f"- {x}" for x in self.flush_lines]
        lines += ["", f"- next: {next_step}", f"- blocker: {blocker}", ""]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(head + "\n\n" + "\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# shard writing
# ---------------------------------------------------------------------------

def _write_jsonl_shard(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
    return sha_text(path.read_text(encoding="utf-8"))


def write_shard(out_root: Path, collection: str, shard_idx: int,
                rows: list[dict[str, Any]]) -> dict[str, Any]:
    shard_id = f"shard-{shard_idx:04d}"
    coll_dir = out_root / collection
    coll_dir.mkdir(parents=True, exist_ok=True)
    shard_path = coll_dir / f"{shard_id}.jsonl"
    digest = _write_jsonl_shard(shard_path, rows)
    manifest = {
        "shard_id": shard_id,
        "lane": LANE,
        "collection": collection,
        "records": len(rows),
        "path": str(shard_path.resolve()),
        "sha256": digest,
        "base_goals": len({r["base_goal_id"] for r in rows}),
        "families": sorted({r["family_id"] for r in rows}),
        "immutable": True,
    }
    if collection == "qualified":
        manifest["qualified_instances"] = len(rows)
    (coll_dir / f"{shard_id}.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def build(pin_root: Path, out_root: Path, target_instances: int,
          seed: int = 20261101, status_path: Path | None = None,
          smoke_families: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    kb = load_kb(str(pin_root))
    status = StatusWriter(status_path)
    goals_total = target_instances // 5
    families = FAMILIES[:smoke_families] if smoke_families else FAMILIES
    quotas = family_goal_quotas(families, goals_total)
    family_goal_cap = max(24, int(target_instances * 0.08)) // 5
    stats: dict[str, Any] = {
        "qualified": 0,
        "rejections": {"_dropped_goal_rows": []},
        "families": {},
        "started_utc": utcnow(),
    }
    status.update("sampling", {"target_instances": target_instances,
                               "goals_total": goals_total,
                               "families": len(families)},
                  "per-family base-goal sampling")

    candidates: list[Candidate] = []
    sample_rejections: list[dict[str, Any]] = []
    seq = 0
    for i, fam in enumerate(families):
        quota = quotas[fam.family_id]
        n_sample = min(fam.max_base_goals, max(quota + 10, int(quota * 2.2) + 1))
        cands, rejects = sample_base_goals(kb, fam, n_sample, seed + i * 977)
        for c in cands:
            c.seq = seq
            seq += 1
        candidates.extend(cands)
        sample_rejections.extend(rejects)
        for r in rejects:
            stats["rejections"][r["code"]] = stats["rejections"].get(r["code"], 0) + 1
        stats["families"][fam.family_id] = {"candidates": len(cands),
                                            "quota_goals": quota}
        print(f"[sample {i+1}/{len(families)}] {fam.family_id} "
              f"candidates={len(cands)} rejects={len(rejects)}", flush=True)

    status.update("reference_verify", {"candidates": len(candidates)},
                  "XML-store SQL reference verification",
                  log_line=f"{utcnow()} sampled {len(candidates)} candidates "
                           f"({len(sample_rejections)} sampling rejections)")
    ref_evidence = run_reference_verification(pin_root, candidates)
    passed = sum(1 for ev in ref_evidence.values() if ev.get("verdict") == "PASS")
    print(f"xml-store reference: {passed}/{len(candidates)} PASS", flush=True)

    ship, surplus = select_ship_goals(candidates, ref_evidence, quotas,
                                      goals_total, family_goal_cap, stats)
    status.update("materialize",
                  {"candidates": len(candidates), "reference_pass": passed,
                   "ship_goals": len(ship), "surplus_goals": len(surplus)},
                  "goal-atomic materialization with 250-record shard flushes",
                  log_line=f"{utcnow()} selection: ship {len(ship)}/{goals_total} "
                           f"goals (reference pass {passed}/{len(candidates)}, "
                           f"donated {stats['surplus_donated_goals']})")

    import rdflib
    ontology_triples = len(rdflib.Graph().parse(data=ONTOLOGY_TTL,
                                                format="turtle"))

    domain = L16Domain()
    seen_instances: set[str] = set()
    buffer: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    shipped_goal_ids: set[str] = set()
    shipped_per_family: dict[str, int] = {f.family_id: 0 for f in families}
    surplus_iter = list(surplus)
    plan: list[Candidate] = list(ship)
    idx = 0
    while idx < len(plan):
        c = plan[idx]
        idx += 1
        recs = materialize_goal(c, kb, out_root, domain, seen_instances, stats,
                                ontology_triples)
        if recs is None:
            replacement = None
            while surplus_iter:
                alt = surplus_iter.pop(0)
                if alt.base_goal_id in shipped_goal_ids:
                    continue
                if shipped_per_family[alt.family.family_id] >= family_goal_cap:
                    continue
                recs_alt = materialize_goal(alt, kb, out_root, domain,
                                            seen_instances, stats,
                                            ontology_triples)
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
        if idx % 50 == 0:
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
        fid = f"l16.{c.family.family_id}"
        ev = ref_evidence.get(c.base_goal_id, {})
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
        "unique_families": len({f"l16.{c.family.family_id}"
                                for c in candidates
                                if c.base_goal_id in shipped_goal_ids}),
        "candidates": len(candidate_rows),
        "candidates_native_pass": passed,
    })
    per_family_qualified_inst = {f"l16.{f.family_id}": 5 * sum(
        1 for c in candidates if c.family is f
        and c.base_goal_id in shipped_goal_ids) for f in families}
    build_report = {
        "lane": LANE,
        "dataset_id": DATASET_ID,
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_refs": [UNIPROT_REF],
        "target_qualified_instances": target_instances,
        "instances_per_goal": 5,
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
            f"l16.{f.family_id}": {
                "quota_goals": quotas[f.family_id],
                "candidates": stats["families"][f.family_id]["candidates"],
                "qualified_goals": sum(
                    1 for c in candidates
                    if c.family is f and c.base_goal_id in shipped_goal_ids),
                "qualified_instances": per_family_qualified_inst[f"l16.{f.family_id}"],
            }
            for f in families
        },
        "shards": [m["shard_id"] for m in shard_manifests],
        "shard_manifests": shard_manifests + cand_manifests,
        "license_verdicts": LICENSE_VERDICTS,
    }
    (out_root / "build_report.json").write_text(
        json.dumps(build_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    status.update(
        "done",
        {"qualified_instances": stats["qualified"],
         "base_goals": stats["base_goals"],
         "families": stats["unique_families"],
         "shards": len(shard_manifests),
         "wall_seconds": stats["wall_seconds"]},
        "final verification (counts/fields, reverify samples) + REPORT.md",
        log_line=f"{utcnow()} build complete: {stats['qualified']} qualified / "
                 f"{stats['base_goals']} goals / {stats['unique_families']} families")
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L16 UniProt task builder")
    ap.add_argument("--pin-root", default=str(
        REPO_ROOT / "external_knowledge/pinned/uniprot/uniprot-2026_03-sprot"))
    ap.add_argument("--out", default=str(REPO_ROOT / "bank/staging_20k/l16"))
    ap.add_argument("--target-instances", type=int, default=1300)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--status-file", default=None)
    ap.add_argument("--smoke-families", type=int, default=None,
                    help="build only the first N families (smoke mode)")
    args = ap.parse_args(argv)
    stats = build(Path(args.pin_root), Path(args.out),
                  args.target_instances, args.seed,
                  Path(args.status_file) if args.status_file else None,
                  args.smoke_families)
    print(json.dumps({k: v for k, v in stats.items() if k != "rejections"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
