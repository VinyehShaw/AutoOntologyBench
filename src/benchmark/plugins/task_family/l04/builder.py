"""L04 candidate -> qualified shard builder.

Pipeline per FINAL-BUILD-ORDER-v2 §4.1: extract -> compose -> construct_solution
-> sample_start_state -> native_verify -> semantic_check -> canonicalize/dedup
-> package into immutable shards.

Production shape (aob-8000-rc1 L04 quota): 25 families x (14..15 base goals)
x 5 state-axis instances = 360 base goals / 1800 QUALIFIED instances.
Goals per family: ``family_goal_quotas`` distributes the goal budget evenly
(first ``G % 25`` families in declaration order get one extra goal).

Orchestration:
  1. sample_base_goals per family (oversampled vs quota; fast-path answer gates)
  2. run_sparql_verification over all candidates (process pool; rdflib over the
     pinned graphs is the independent reference path)
  3. selection gates (SPARQL PASS, wording leak) -> ship quota per family,
     rest to the surplus pool; global shortfall is donated round-robin from
     the surplus pool (family instance cap 8% of target enforced)
  4. materialize ship goals goal-atomically (5 instances each, bounded retries
     on the instance state axis); a goal that cannot reach 5 instances is
     dropped (its partial files are removed) and backfilled from the surplus
     pool, so shipped counts stay exact
  5. incremental shard flush every 250 qualified records + STATUS checkpoint

Gates (cheap -> expensive), rejection codes:
  UNSOLVABLE / TRIVIAL / PROFILE_UNSUPPORTED / LEAKAGE / DUPLICATE /
  NON_SEMANTIC / LICENSE_HOLD.

CLI:
  .venv/bin/python -m benchmark.plugins.task_family.l04.builder \
      --pin-root external_knowledge/pinned/rhea-chebi/rhea-142+chebi-256 \
      --out bank/staging/l04 --target-instances 1800 --workers 64 \
      --status-file data/erule/review5/build8000/lanes/L04/STATUS.md
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
from typing import Any, Iterator

from benchmark.plugins.source.l04.kb import L04KB, load_kb
from benchmark.plugins.task_family.l04.families import (
    ALLOWED_TOOLS, FAMILIES, INSTANCES_PER_GOAL, Family, _ancestors,
    base_goal_id, canonical_params, masters_involving, source_unit_ids,
    state_descriptor, _c2m, _master_cache,
)
from benchmark.plugins.verifier.l04.domain import L04Domain
from benchmark.plugins.verifier.l04 import plugin as verifier
from source_registry.fetchers.l04_rhea_chebi import LICENSE_VERDICTS, RELEASE_LOCK

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "aob-task/5.0"
DATASET_ID = "autoontobench-bank-8000-rc1"
LANE = "L04"
SOURCE_GROUP = "rhea@release-142+chebi@release-256"
SNAPSHOT_ID = "rhea-chebi/rhea-142+chebi-256"
PIN_ROOT_REL = "external_knowledge/pinned/rhea-chebi/rhea-142+chebi-256"
SUPPORTED_PROFILES = sorted({f.profile for f in FAMILIES})
SHARD_SIZE = 250
MAX_INSTANCE_ATTEMPTS = 12
STATUS_MARKER = "<!-- L04-PRODUCTION-STATE -->"

RHEA_REF = {"source_id": "rhea", "source_rule_unit_id": "rhea/release-142",
            "commit": "release 142 (2026-09-02)",
            "file_sha256": RELEASE_LOCK["rhea/rdf/rhea.rdf.gz"]["sha256"]}
CHEBI_REF = {"source_id": "chebi", "source_rule_unit_id": "chebi/release-256",
             "commit": "release 256 (2026-10-06)",
             "file_sha256": RELEASE_LOCK["chebi/chebi_lite.owl.gz"]["sha256"]}

DROPPED_FAMILIES = {
    "ont_chebi_closure_dual_engine":
        "same goal structure as ont_chebi_descendants_in_rhea "
        "(verifier-path-only difference); dropped per FINAL doc §1 "
        "substantive-difference rule",
    "rep_crosswalk_single_flip":
        "k=1 difficulty knob of rep_crosswalk_multi_swap (same mechanism)",
}


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha_json(payload: Any) -> str:
    return sha_text(json.dumps(payload, sort_keys=True, ensure_ascii=False))


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# candidate container
# ---------------------------------------------------------------------------

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

def sample_base_goals(kb: L04KB, family: Family, n_goals: int,
                      seed: int) -> tuple[list[Candidate], list[dict[str, Any]]]:
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
        try:
            answer = family.answer(kb, params, masters)
        except Exception as exc:  # reference cannot be computed
            rejects.append({"family": family.family_id, "params": params,
                            "code": "UNSOLVABLE", "reason": f"reference_error: {exc}"})
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
            family=family, params=params, base_goal_id=bid, answer=tuple(sorted(set(answer))),
            mechanism_group_id=f"l04/{family.family_id}/{bid[:12]}",
            user_goal=family.goal_text(params, kb),
            provenance=("source_exact" if family.family_id.startswith(("app_", "ont_"))
                        else "source_derived_composition"),
            seed=seed,
        ))
        if len(out) >= n_goals:
            break
    return out, rejects


# ---------------------------------------------------------------------------
# leakage / wording gates
# ---------------------------------------------------------------------------

def wording_leak_check(user_goal: str, answer: tuple[str, ...],
                       params: dict[str, Any]) -> str | None:
    """The wording must name its query target but never leak answer members."""
    named: list[str] = []
    for k, v in params.items():
        if isinstance(v, str):
            named.append(v)
        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            named.extend(v)
    if not any(t in user_goal for t in named):
        return "wording does not name the query target"
    leaked = [a for a in answer[:200] if a.split("|")[0].split(" = ")[0] in user_goal
              and (a.startswith("RHEA:") or a.startswith("CHEBI:"))]
    if leaked:
        return f"wording leaks answer ids: {leaked[:3]}"
    return None


def bundle_leak_scan(bundle_files: list[Path], answer: tuple[str, ...],
                     family: Family) -> str | None:
    """Public bundles must not contain the serialized answer or gold markers.

    The linked-data/1 ChEBI-identity ban applies to evidence bundles
    (``evidence_bundle_variant``): those tasks must not hand the agent the
    published identity links. Repair artifacts (``corruption_seed``) carry the
    compound->ChEBI mapping AS the object under repair, so ChEBI accessions in
    the artifact are the corruption carrier, not leaked evidence; the
    serialized-answer and gold-marker bans still apply to every bundle.
    """
    answer_payload = json.dumps(sorted(answer))
    banned_tokens = ("\"expected\"", "\"gold\"", "private_", "oracle_handle")
    for path in bundle_files:
        text = path.read_text(encoding="utf-8")
        for tok in banned_tokens:
            if tok in text:
                return f"public bundle file {path.name} contains banned token {tok}"
        if answer_payload in text:
            return f"public bundle file {path.name} contains the serialized answer"
        if (family.profile == "linked-data/1"
                and family.state_axis == "evidence_bundle_variant"
                and "CHEBI:" in text):
            return f"linked-data public file {path.name} leaks ChEBI identity links"
    return None


# ---------------------------------------------------------------------------
# instance state builders
# ---------------------------------------------------------------------------

_ROWS_BY_RXN: dict[int, dict[str, list[tuple[str, str, str, str, str]]]] = {}


def _rows_by_rxn(kb: L04KB) -> dict[str, list[tuple[str, str, str, str, str]]]:
    key = id(kb)
    if key not in _ROWS_BY_RXN:
        m: dict[str, list[tuple[str, str, str, str, str]]] = {}
        for row in kb.participants:
            m.setdefault(row[0], []).append(row)
        _ROWS_BY_RXN[key] = m
    return _ROWS_BY_RXN[key]


def _scope_reactions(kb: L04KB, family: Family, params: dict[str, Any],
                     masters: frozenset[str]) -> frozenset[str]:
    """Reactions whose rows are needed in the public slice for this task."""
    c2m = _c2m(kb)
    ids: set[str] = set()
    if "chebi" in params:
        ids |= set(c2m.get(params["chebi"], ()))
    if "chebi_a" in params:
        ids |= set(c2m.get(params["chebi_a"], ()))
    if "chebi_b" in params:
        ids |= set(c2m.get(params["chebi_b"], ()))
    if "chebi_set" in params:
        for c in params["chebi_set"]:
            ids |= set(c2m.get(c, ()))
    if "ancestor" in params:
        uni = sorted(kb.chebi_to_reactions)
        for c in uni:
            if params["ancestor"] in _ancestors(kb, c):
                ids |= set(c2m.get(c, ()))
    if "master" in params:
        ids.add(params["master"])
        # direction tasks need the child equations in scope: LR/RL roles are
        # recovered from the pinned master+child equation texts.
        mrec = kb.reactions.get(params["master"])
        if mrec:
            ids |= {f"RHEA:{c}" for c in mrec["children_lr_rl"]}
            ids |= {f"RHEA:{c}" for c in mrec["children_bi"]}
    if "rhea" in params:
        ids.add(params["rhea"])
    if "go" in params:
        ids |= set(kb.go_to_masters.get(params["go"], ()))
    if "ec" in params:
        ids |= {m for m, ecs in kb.ec_by_master.items() if params["ec"] in ecs}
    if "ec_class" in params:
        ids |= {m for m, ecs in kb.ec_by_master.items()
                if any(e.startswith(params["ec_class"] + ".") for e in ecs)}
    return frozenset(ids)


def build_public_bundle(kb: L04KB, family: Family, params: dict[str, Any],
                        variant: str, out_dir: Path,
                        distractor_seed: int | None = None) -> list[Path]:
    """Write the agent-visible evidence bundle. Returns written file paths.

    variant ``tsv_view``: JSONL row files (one row per record).
    variant ``rdf_view``: Turtle serialization of the same slice.
    linked-data/1 profiles never include ChEBI identity links (W14 rule).
    """
    masters = _master_cache(kb)
    scope = _scope_reactions(kb, family, params, masters)
    linked = family.profile == "linked-data/1"
    rows_by_rxn = _rows_by_rxn(kb)
    rows = [r for rxn in sorted(scope) for r in rows_by_rxn.get(rxn, [])]
    if distractor_seed is not None:
        rng = random.Random(distractor_seed)
        extra = rng.sample(sorted(masters - scope), k=min(40, len(masters - scope)))
        rows += [r for rxn in sorted(extra) for r in rows_by_rxn.get(rxn, [])]
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def emit(name: str, records: list[dict[str, Any]]) -> None:
        path = out_dir / name
        text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                       for r in records)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    if variant == "tsv_view":
        part_records = []
        for r, s, p, c, ch in rows:
            rec = {"rhea_id": r, "side": s, "participant": p, "compound": c}
            if not linked:
                rec["chebi"] = ch
            part_records.append(rec)
        emit("rhea_participants.jsonl", part_records)
        meta = []
        for r in sorted(scope, key=lambda x: int(x.split(":")[1])):
            u = kb.reactions[r]
            rec = {"rhea_id": r, "equation": u["equation"], "status": u["status"],
                   "is_transport": u["is_transport"],
                   "is_chemically_balanced": u["is_balanced"],
                   "children_lr_rl": u["children_lr_rl"], "children_bi": u["children_bi"]}
            if u["ec"]:
                rec["ec"] = u["ec"]
            meta.append(rec)
        emit("rhea_reactions.jsonl", meta)
        if not linked:
            edges = sorted({(ch, a) for _r, _s, _p, _c, ch in rows
                            for a in _ancestors(kb, ch)})
            if edges:
                emit("chebi_subclass_edges.jsonl",
                     [{"chebi": c, "parent": a} for c, a in edges])
            xref_rows = [{"rhea_id": m, "xref": x}
                         for m in sorted(scope) for x in
                         sorted(kb.xrefs_by_master.get(m, ()))]
            if xref_rows:
                emit("rhea_xrefs.jsonl", xref_rows)
            go_rows = [{"rhea_id": m, "go": g}
                       for m in sorted(scope) for g in
                       sorted(kb.go_by_master.get(m, ()))]
            if go_rows:
                emit("rhea_go_links.jsonl", go_rows)
    elif variant == "rdf_view":
        lines = ["@prefix rh: <http://rdf.rhea-db.org/> .",
                 "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
                 "@prefix chebi: <http://purl.obolibrary.org/obo/CHEBI_> .", ""]
        for r, s, p, c, ch in rows:
            num = r.split(":")[1]
            lines.append(f"<{RH_TTL}{num}> rh:accession \"{r}\" ;")
            lines.append(f"  rdfs:subClassOf rh:Reaction ;")
            lines.append(f"  rh:side <{RH_TTL}{num}_{s}> .")
            lines.append(f"<{RH_TTL}{num}_{s}> rh:contains <{p}> .")
            lines.append(f"<{p}> rh:compound <{c}> .")
            if not linked:
                lines.append(f"<{c}> rh:accession \"{ch}\" .")
        path = out_dir / "rhea_slice.ttl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    else:
        raise ValueError(variant)
    return written


RH_TTL = "http://rdf.rhea-db.org/"


def build_repair_bundle(kb: L04KB, family: Family, params: dict[str, Any],
                        seed: int, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    """Corrupted artifact + public evidence. Returns (files, corruption_record)."""
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    masters = _master_cache(kb)
    rows_by_rxn = _rows_by_rxn(kb)
    corruption: dict[str, Any] = {}
    written: list[Path] = []

    def emit(name: str, payload: Any) -> None:
        path = out_dir / name
        text = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
                if name.endswith(".json") else
                "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                        for r in payload))
        path.write_text(text + ("" if text.endswith("\n") else "\n"), encoding="utf-8")
        written.append(path)

    if family.family_id.startswith("rep_crosswalk"):
        k = 1 if family.family_id.endswith("single_flip") else 3
        target = params["chebi"]
        scope_rxns = masters_involving(kb, masters, target)
        gold_map = {}
        for rxn in scope_rxns:
            for _r, _s, _p, c, ch in rows_by_rxn.get(rxn, []):
                gold_map[c] = ch
        # One ChEBI entity maps to exactly one compound IRI in Rhea, so the k
        # corrupted entries are: 1 critical entry (the target's compound, which
        # moves the governed answer) + (k-1) decoy entries (other compounds in
        # scope, which the agent must still detect to repair the artifact).
        critical_pool = sorted(c for c, ch in gold_map.items() if ch == target)
        other_pool = sorted(c for c, ch in gold_map.items() if ch != target)
        if not critical_pool or len(other_pool) < k - 1:
            return written, {"ok": False,
                             "reason": "crosswalk scope lacks defect entries"}
        accessions = sorted(set(gold_map.values()))
        chosen = [rng.choice(critical_pool)] + rng.sample(other_pool, k - 1)
        corrupt = dict(gold_map)
        defects = []
        for i, comp in enumerate(chosen):
            true_val = gold_map[comp]
            candidates_pool = [a for a in accessions if a != true_val]
            decoy = candidates_pool[rng.randrange(len(candidates_pool))]
            corrupt[comp] = decoy
            defects.append({"compound": comp, "true": true_val,
                            "corrupted_to": decoy,
                            "affects_governed_answer": i == 0})
        emit("crosswalk_artifact.json", {
            "profile": "linked-data/1",
            "mapping": {c: ch for c, ch in sorted(corrupt.items())},
            "note": "artifact under repair; some entries may be corrupted"})
        structure = [{"rhea_id": r, "side": s, "participant": p, "compound": c}
                     for rxn in sorted(scope_rxns)
                     for r, s, p, c, _ch in rows_by_rxn.get(rxn, [])]
        emit("rhea_participants.jsonl", structure)
        emit("probe_spec.json", {
            "tool": "probe_native", "arity": "one compound IRI per probe",
            "returns": "published ChEBI accession of the compound",
            "budget": k + 2})
        corruption = {"ok": True, "kind": "crosswalk", "k": k,
                      "defects": defects, "gold_map_size": len(gold_map),
                      "probe_budget": k + 2}

    elif family.family_id == "rep_direction_assignment_repair":
        m = params["master"].split(":")[1]
        row = next(d for d in kb.directions if d["RHEA_ID_MASTER"] == m)
        gold = {"LR": "RHEA:" + row["RHEA_ID_LR"], "RL": "RHEA:" + row["RHEA_ID_RL"]}
        if row.get("RHEA_ID_BI"):
            gold["BI"] = "RHEA:" + row["RHEA_ID_BI"]
        corrupt = dict(gold)
        corrupt["LR"], corrupt["RL"] = gold["RL"], gold["LR"]
        emit("direction_artifact.json", {
            "master": params["master"], "assignments": corrupt,
            "note": "artifact under repair; LR/RL assignments may be swapped"})
        evidence = [{"rhea_id": params["master"], "role": "master",
                     "equation": kb.reactions[params["master"]]["equation"]}]
        for code, child in sorted(gold.items()):
            u = kb.reactions[child]
            evidence.append({"rhea_id": child, "role": "child",
                             "equation": u["equation"]})
        emit("children_equations.jsonl", evidence)
        corruption = {"ok": True, "kind": "direction_swap", "gold": gold}

    elif family.family_id == "rep_participant_side_flip_repair":
        rhea = params["rhea"]
        rows = [(s, p, c, ch) for _r, s, p, c, ch in rows_by_rxn.get(rhea, [])]
        idx = rng.randrange(len(rows))
        s, p, c, ch = rows[idx]
        flipped = "R" if s == "L" else "L"
        corrupt_rows = [
            {"side": (flipped if i == idx else s0), "participant": p0,
             "compound": c0, "chebi": ch0}
            for i, (s0, p0, c0, ch0) in enumerate(rows)]
        emit("participant_sides_artifact.json", {
            "rhea": rhea, "rows": corrupt_rows,
            "note": "artifact under repair; one participant may be on the wrong side"})
        emit("reaction_evidence.json", {
            "rhea": rhea, "equation": kb.reactions[rhea]["equation"]})
        corruption = {"ok": True, "kind": "side_flip", "flipped_participant": p,
                      "true_side": s, "corrupted_side": flipped}

    elif family.family_id == "rep_ph73_mapping_repair":
        target = params["chebi"]
        scope = sorted({target, params["ph73"]} |
                       {("CHEBI:" + row["CHEBI"]
                         if not row["CHEBI"].startswith("CHEBI:")
                         else row["CHEBI"])
                        for row in kb.ph73
                        if row["CHEBI_PH7_3"] == params["ph73"].split(":")[1]})[:50]
        gold_map = {("CHEBI:" + row["CHEBI"] if not row["CHEBI"].startswith("CHEBI:")
                     else row["CHEBI"]): "CHEBI:" + row["CHEBI_PH7_3"]
                    for row in kb.ph73
                    if ("CHEBI:" + row["CHEBI"] if not row["CHEBI"].startswith("CHEBI:")
                        else row["CHEBI"]) in set(scope)}
        keys = sorted(gold_map)
        if len(keys) < 3:
            return written, {"ok": False, "reason": "ph73 scope too small"}
        corrupt = dict(gold_map)
        victims = rng.sample(keys, 2)
        for v in victims:
            others = [k2 for k2 in keys if k2 != v]
            corrupt[v] = gold_map[others[rng.randrange(len(others))]]
        emit("ph73_artifact.json", {
            "mapping": {k2: v2 for k2, v2 in sorted(corrupt.items())},
            "note": "artifact under repair; up to 2 targets may be corrupted"})
        emit("ph73_reference_extract.jsonl",
             [{"chebi": k2, "ph73": v2} for k2, v2 in sorted(gold_map.items())])
        corruption = {"ok": True, "kind": "ph73", "victims": victims,
                      "gold": {v: gold_map[v] for v in victims}}
    else:
        corruption = {"ok": False, "reason": f"no repair bundle for {family.family_id}"}
    return written, corruption


def build_construction_bundle(kb: L04KB, family: Family, params: dict[str, Any],
                              seed: int, out_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    masters = _master_cache(kb)
    rows_by_rxn = _rows_by_rxn(kb)

    def emit(name: str, payload: Any) -> None:
        path = out_dir / name
        if name.endswith(".json"):
            text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        else:
            text = "".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
                           for r in payload)
        path.write_text(text, encoding="utf-8")
        written.append(path)

    if family.family_id == "con_crosswalk_via_budgeted_probes":
        scope = _scope_reactions(kb, family, params, masters)
        rng = random.Random(seed)
        distractors = rng.sample(sorted(masters - scope),
                                 k=min(60, len(masters - scope)))
        rows = [r for rxn in sorted(scope | frozenset(distractors))
                for r in rows_by_rxn.get(rxn, [])]
        structure = [{"rhea_id": r, "side": s, "participant": p, "compound": c}
                     for r, s, p, c, _ch in rows]
        emit("rhea_participants.jsonl", structure)
        compounds = sorted({c for rxn in scope
                            for _r, _s, _p, c, _ch in rows_by_rxn.get(rxn, [])})
        budget = len(compounds)
        emit("probe_spec.json", {
            "tool": "probe_native", "arity": "one compound IRI per probe",
            "returns": "published ChEBI accession of the compound",
            "budget": budget})
        return written, {"ok": True, "kind": "con_crosswalk", "probe_budget": budget,
                         "scope_compounds": len(compounds),
                         "distractor_reactions": len(distractors)}
    if family.family_id == "con_reaction_profile_assembly":
        m = params["master"]
        emit("reaction_evidence.json", {
            "rhea": m, "equation": kb.reactions[m]["equation"]})
        emit("probe_spec.json", {
            "tool": "probe_native",
            "arity": "one (rhea_id, aspect) pair per probe; aspect in "
                     "{children, xrefs, go, participants}",
            "returns": "the pinned record slice for that reaction and aspect",
            "budget": 4})
        return written, {"ok": True, "kind": "con_profile", "probe_budget": 4}
    return written, {"ok": False, "reason": f"no construction bundle for {family.family_id}"}


# ---------------------------------------------------------------------------
# SPARQL reference verification (parallel workers; lazy per-process graphs)
# ---------------------------------------------------------------------------

def _sparql_worker(job: dict[str, Any]) -> dict[str, Any]:
    pin_root, family_id, params, expected = (
        Path(job["pin_root"]), job["family"], job["params"], tuple(job["expected"]))
    kb = load_kb(str(pin_root))
    try:
        ev = verifier.verify_reference(pin_root, family_id, params, expected, kb)
    except Exception as exc:
        return {"base_goal_id": job["base_goal_id"], "verdict": "ERROR",
                "reason": f"{type(exc).__name__}: {exc}"[:400]}
    ev["base_goal_id"] = job["base_goal_id"]
    return ev


def run_sparql_verification(pin_root: Path, candidates: list[Candidate],
                            workers: int) -> dict[str, dict[str, Any]]:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [{"pin_root": str(pin_root), "family": c.family.family_id,
             "params": c.params, "expected": list(c.answer),
             "base_goal_id": c.base_goal_id} for c in candidates]
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_sparql_worker, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs)):
            ev = fut.result()
            results[ev["base_goal_id"]] = ev
            if (i + 1) % 50 == 0:
                print(f"  sparql verify {i + 1}/{len(jobs)}", flush=True)
    return results


# ---------------------------------------------------------------------------
# selection: cheap gates + quota + surplus donation
# ---------------------------------------------------------------------------

def family_goal_quotas(families: tuple[Family, ...],
                       goals_total: int) -> dict[str, int]:
    base, extra = divmod(goals_total, len(families))
    return {f.family_id: base + (1 if i < extra else 0)
            for i, f in enumerate(families)}


def select_ship_goals(candidates: list[Candidate],
                      sparql_evidence: dict[str, dict[str, Any]],
                      quotas: dict[str, int], goals_total: int,
                      family_goal_cap: int,
                      stats: dict[str, Any]
                      ) -> tuple[list[Candidate], list[Candidate]]:
    """Apply SPARQL + wording gates, take each family's quota in sampling
    order, then donate from the surplus pool (family declaration order,
    round-robin) until the global goal budget is met."""
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

    for c in candidates:  # candidates are already in (family, sampling) order
        fam = c.family
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
# goal-atomic instance materialization
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


def materialize_goal(c: Candidate, kb: L04KB, out_root: Path,
                     domain: L04Domain, seen_instances: set[str],
                     stats: dict[str, Any],
                     start_inst_idx: int = 0,
                     n_instances: int = INSTANCES_PER_GOAL,
                     ) -> list[dict[str, Any]] | None:
    """Build ``n_instances`` instance states of one verified goal. Goal-atomic:
    returns None (and removes partial files) when the goal cannot reach the
    requested instance count.

    ``start_inst_idx`` lets the top-up path generate additional deterministic
    state-axis variants (indices 5, 6, ...) of an already shipped goal without
    touching the first five public bundles."""
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

        # domain replay: reset -> not solved; submit reference -> solved
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
            "reference_plan": reference_plan(fam, c, meta),
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
        made.append(task_record(
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


def reference_plan(fam: Family, c: Candidate, meta: dict[str, Any]) -> dict[str, Any]:
    if fam.state_axis == "corruption_seed":
        return {"steps": ["inspect corrupted artifact", "align with public evidence",
                          "repair defect entries", "submit repaired artifact",
                          "submit answer"],
                "probe_budget": meta.get("probe_budget", 0),
                "min_actions": 2}
    if fam.state_axis == "evidence_sample_seed":
        return {"steps": ["probe required compounds/records within budget",
                          "assemble constructed artifact", "submit answer"],
                "probe_budget": meta.get("probe_budget", 0),
                "min_actions": 2}
    return {"steps": ["align provided source views", "compute exact answer set",
                      "submit answer"],
            "probe_budget": 0, "min_actions": 1}


def task_record(c: Candidate, task_id: str, instance_id: str,
                state_desc: dict[str, Any], public_dir: Path, state_path: Path,
                oracle_handle: str, out_root: Path, meta: dict[str, Any],
                wit: dict[str, Any], reset_ok: bool,
                terminal_ok: bool) -> dict[str, Any]:
    fam = c.family
    ev = c.reference_evidence
    rel = lambda p: str(p.relative_to(out_root))
    bundle_files = sorted(
        str(p.relative_to(out_root))
        for p in public_dir.iterdir() if p.is_file())
    oracle_rel = f"oracle/{oracle_handle}.json"
    reverify_cmd = (".venv/bin/python -m benchmark.plugins.verifier.l04.reverify "
                    f"--pin-root {PIN_ROOT_REL} "
                    f"--oracle bank/staging/l04/{oracle_rel}")
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


def split_of(base_goal: str) -> str:
    bucket = int(base_goal[:8], 16) % 100
    return "train" if bucket < 70 else ("dev" if bucket < 85 else "test")


# ---------------------------------------------------------------------------
# STATUS checkpoint writer (restart-safe production state)
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

def build(pin_root: Path, out_root: Path, target_instances: int, workers: int,
          seed: int = 20261027, status_path: Path | None = None) -> dict[str, Any]:
    t0 = time.time()
    kb = load_kb(str(pin_root))
    status = StatusWriter(status_path)
    goals_total = target_instances // INSTANCES_PER_GOAL
    quotas = family_goal_quotas(FAMILIES, goals_total)
    family_goal_cap = max(24, int(target_instances * 0.08)) // INSTANCES_PER_GOAL
    stats: dict[str, Any] = {
        "qualified": 0,
        "rejections": {"_dropped_goal_rows": []},
        "families": {},
        "started_utc": utcnow(),
    }
    status.update("sampling", {"target_instances": target_instances,
                               "goals_total": goals_total,
                               "families": len(FAMILIES)},
                  "per-family base-goal sampling")

    # 1. sample candidates (oversampled vs quota; sampling-order = deterministic)
    candidates: list[Candidate] = []
    sample_rejections: list[dict[str, Any]] = []
    seq = 0
    for i, fam in enumerate(FAMILIES):
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
        print(f"[sample {i+1}/{len(FAMILIES)}] {fam.family_id} "
              f"candidates={len(cands)} rejects={len(rejects)}", flush=True)

    # 2. independent SPARQL reference verification
    status.update("sparql_verify",
                  {"candidates": len(candidates), "workers": workers},
                  f"rdflib SPARQL reference verification of {len(candidates)} candidates",
                  log_line=f"{utcnow()} sampled {len(candidates)} candidates "
                           f"({len(sample_rejections)} sampling rejections)")
    sparql_evidence = run_sparql_verification(pin_root, candidates, workers)
    passed = sum(1 for ev in sparql_evidence.values() if ev.get("verdict") == "PASS")
    print(f"sparql reference: {passed}/{len(candidates)} PASS", flush=True)

    # 3. selection
    ship, surplus = select_ship_goals(candidates, sparql_evidence, quotas,
                                      goals_total, family_goal_cap, stats)
    status.update("materialize",
                  {"candidates": len(candidates), "sparql_pass": passed,
                   "ship_goals": len(ship), "surplus_goals": len(surplus)},
                  "goal-atomic materialization with 250-record shard flushes",
                  log_line=f"{utcnow()} selection: ship {len(ship)}/{goals_total} goals "
                           f"(sparql pass {passed}/{len(candidates)}, "
                           f"donated {stats['surplus_donated_goals']})")

    # 4. materialize (goal-atomic, backfill from surplus) + incremental flush
    domain = L04Domain()
    seen_instances: set[str] = set()
    buffer: list[dict[str, Any]] = []
    shard_manifests: list[dict[str, Any]] = []
    shipped_goal_ids: set[str] = set()
    shipped_per_family: dict[str, int] = {f.family_id: 0 for f in FAMILIES}
    surplus_iter = list(surplus)
    plan: list[Candidate] = list(ship)
    idx = 0
    while idx < len(plan):
        c = plan[idx]
        idx += 1
        recs = materialize_goal(c, kb, out_root, domain, seen_instances, stats)
        if recs is None:
            replacement = None
            while surplus_iter:
                alt = surplus_iter.pop(0)
                if alt.base_goal_id in shipped_goal_ids:
                    continue
                if shipped_per_family[alt.family.family_id] >= family_goal_cap:
                    continue
                recs_alt = materialize_goal(alt, kb, out_root, domain,
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
        if idx % 50 == 0:
            print(f"  materialize {idx}/{len(plan)} goals, "
                  f"qualified={stats['qualified']}", flush=True)
    if buffer:
        m = write_shard(out_root, "qualified", len(shard_manifests), buffer)
        shard_manifests.append(m)
        buffer = []

    # 5. candidate rows (status reflects what actually shipped) + rejections
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
            "native_evidence": {k: v for k, v in ev.items() if k != "query_text"},
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
    per_family_qualified_inst: dict[str, int] = {f"l04.{f.family_id}": 0
                                                 for f in FAMILIES}
    for c in candidates:
        if c.base_goal_id in shipped_goal_ids:
            per_family_qualified_inst[f"l04.{c.family.family_id}"] += INSTANCES_PER_GOAL
    build_report = {
        "lane": LANE,
        "dataset_id": DATASET_ID,
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
                "qualified_instances": per_family_qualified_inst[f"l04.{f.family_id}"],
            }
            for f in FAMILIES
        },
        "dropped_families": DROPPED_FAMILIES,
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
        "final verification (counts/fields, reverify samples, preflight) + REPORT.md",
        log_line=f"{utcnow()} build complete: {stats['qualified']} qualified / "
                 f"{stats['base_goals']} goals / {stats['unique_families']} families")
    return stats


# ---------------------------------------------------------------------------
# top-up: extra deterministic state-axis instances on already shipped goals
# ---------------------------------------------------------------------------

def _load_existing_goal_groups(out_root: Path
                               ) -> tuple[list[dict[str, Any]],
                                          dict[str, list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    for p in sorted((out_root / "qualified").glob("shard-*.jsonl")):
        rows.extend(json.loads(line) for line in
                    p.read_text(encoding="utf-8").splitlines() if line.strip())
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["base_goal_id"], []).append(r)
    return rows, groups


def _candidate_from_shipped_goal(kb: L04KB, fam: Family,
                                 rows: list[dict[str, Any]],
                                 oracle: dict[str, Any]) -> Candidate:
    """Rebuild the builder's Candidate container from an already shipped goal.

    Wording, seed and provenance are copied verbatim from the shipped record;
    params/answer/reference evidence come from that goal's private oracle. No
    wording is regenerated, so a top-up can never silently paraphrase a goal."""
    rep = sorted(rows, key=lambda r: r["task_id"])[0]
    return Candidate(
        family=fam,
        params=oracle["params"],
        base_goal_id=rep["base_goal_id"],
        answer=tuple(sorted(set(oracle["expected_answer"]))),
        mechanism_group_id=rep["mechanism_group_id"],
        user_goal=rep["public"]["user_goal"],
        provenance=rep["task_provenance"],
        seed=rep["seed"],
        reference_evidence=oracle["sparql_reference"],
    )


def top_up(out_root: Path, pin_root: Path, extra_instances: int,
           workers: int = 24) -> dict[str, Any]:
    """Add ``extra_instances`` state-axis variants to existing base goals.

    Each round gives one extra instance to the next goal of every family, so
    the 50-instance top-up is 25 families x 2 goals x 1 extra variant. The new
    instance index starts after the goal's existing instance count (5), which
    makes the state descriptor and the public bundle deterministic and distinct
    from the original five. Base goals, families and wording are unchanged."""
    t0 = time.time()
    kb = load_kb(str(pin_root))
    rows, groups = _load_existing_goal_groups(out_root)
    if not groups:
        raise RuntimeError(f"no shipped goals under {out_root}")
    existing_instances = {r["instance_id"] for r in rows}
    fam_goals: dict[str, list[str]] = {f.family_id: [] for f in FAMILIES}
    for bid, grp in groups.items():
        fid = grp[0]["family_id"]
        if fid.startswith("l04."):
            fid = fid[4:]
        if fid in fam_goals:
            fam_goals[fid].append(bid)
    for fid in fam_goals:
        fam_goals[fid].sort()

    stats: dict[str, Any] = {"rejections": {"_dropped_goal_rows": []}}
    domain = L04Domain()
    made: list[dict[str, Any]] = []
    extended: list[dict[str, Any]] = []
    failed_goals: list[str] = []
    ptr = {f.family_id: 0 for f in FAMILIES}
    while len(made) < extra_instances:
        progressed = False
        for fam in FAMILIES:
            if len(made) >= extra_instances:
                break
            goals = fam_goals[fam.family_id]
            while ptr[fam.family_id] < len(goals):
                bid = goals[ptr[fam.family_id]]
                ptr[fam.family_id] += 1
                grp = groups[bid]
                rep = sorted(grp, key=lambda r: r["task_id"])[0]
                oracle = json.loads(
                    (out_root / rep["private_goal_verifier_ref"]).read_text(
                        encoding="utf-8"))
                c = _candidate_from_shipped_goal(kb, fam, grp, oracle)
                start = len(grp)
                recs = materialize_goal(
                    c, kb, out_root, domain, existing_instances, stats,
                    start_inst_idx=start, n_instances=1)
                if recs:
                    made.extend(recs)
                    extended.append({
                        "base_goal_id": bid,
                        "family_id": f"l04.{fam.family_id}",
                        "start_inst_idx": start,
                        "added": len(recs),
                        "task_ids": [r["task_id"] for r in recs],
                    })
                    progressed = True
                    break
                failed_goals.append(bid)
        if not progressed:
            break

    if not made:
        raise RuntimeError("top-up produced no instances")
    qdir = out_root / "qualified"
    next_idx = max((int(p.stem.split("-")[1])
                    for p in qdir.glob("shard-*.jsonl")), default=-1) + 1
    manifest = write_shard(out_root, "qualified", next_idx, made)

    report_path = out_root / "build_report.json"
    rep = json.loads(report_path.read_text(encoding="utf-8"))
    wall = round(time.time() - t0, 2)
    rep["original_target_qualified_instances"] = rep.get(
        "target_qualified_instances")
    rep["target_qualified_instances"] = len(rows) + len(made)
    rep["qualified"] = len(rows) + len(made)
    rep["base_goals"] = len(groups)
    rep["families"] = len({r["family_id"] for r in rows + made})
    rep["wall_seconds_total"] = round(
        float(rep.get("wall_seconds", 0)) + wall, 2)
    rep.setdefault("shards", []).append(manifest["shard_id"])
    rep.setdefault("shard_manifests", []).append(manifest)
    per_family_added: dict[str, int] = {}
    for r in made:
        per_family_added[r["family_id"]] = per_family_added.get(
            r["family_id"], 0) + 1
    for fid, n in sorted(per_family_added.items()):
        pf = rep.setdefault("per_family", {}).setdefault(fid, {})
        pf["qualified_instances"] = pf.get("qualified_instances", 0) + n
        pf["top_up_instances"] = pf.get("top_up_instances", 0) + n
    topup_rejections = {k: v for k, v in stats["rejections"].items()
                        if not k.startswith("_")}
    rep["top_up"] = {
        "method": ("extra deterministic instance/state-axis variant on "
                   "already shipped base goals; no new goals, no wording "
                   "rewrites, no entity swaps"),
        "added_instances": len(made),
        "extended_goals": len(extended),
        "failed_goals": failed_goals,
        "state_start_index": sorted({e["start_inst_idx"] for e in extended}),
        "wall_seconds": wall,
        "shard": manifest["shard_id"],
        "extensions": extended,
        "rejections_by_reason": dict(sorted(topup_rejections.items())),
    }
    if topup_rejections:
        rep["top_up_rejections_by_reason"] = topup_rejections
    report_path.write_text(
        json.dumps(rep, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "qualified_before": len(rows),
        "qualified_after": len(rows) + len(made),
        "added_instances": len(made),
        "extended_goals": len(extended),
        "base_goals": len(groups),
        "families": rep["families"],
        "shard": manifest["shard_id"],
        "wall_seconds": wall,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L04 Rhea+ChEBI task builder")
    ap.add_argument("--pin-root", default=str(
        REPO_ROOT / "external_knowledge/pinned/rhea-chebi/rhea-142+chebi-256"))
    ap.add_argument("--out", default=str(REPO_ROOT / "bank/staging/l04"))
    ap.add_argument("--target-instances", type=int, default=1800)
    ap.add_argument("--top-up", type=int, default=None,
                    help="add N extra deterministic state-axis instances to "
                         "already shipped base goals (does not resample goals)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--status-file", default=None)
    args = ap.parse_args(argv)
    if args.top_up:
        stats = top_up(Path(args.out), Path(args.pin_root), args.top_up,
                       args.workers)
    else:
        stats = build(Path(args.pin_root), Path(args.out),
                      args.target_instances, args.workers, args.seed,
                      Path(args.status_file) if args.status_file else None)
    print(json.dumps({k: v for k, v in stats.items() if k != "rejections"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
