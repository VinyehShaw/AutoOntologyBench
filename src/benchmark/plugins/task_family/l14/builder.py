"""L14 candidate -> qualified shard builder.

Pipeline per FINAL-BUILD-ORDER-v2 §4.1: sample -> fast-path answer -> gates
-> dual-path native verification -> witness certification -> public bundle
materialization (rows_jsonl / sqlite_slice variants + repair artifacts) ->
domain replay -> leak scan -> incremental sha256 shards.

Gates (cheap -> expensive), rejection codes:
  UNSOLVABLE / TRIVIAL / REFERENCE_MISMATCH / NON_SEMANTIC / LEAKAGE /
  DUPLICATE / STATE_BUILD.

CLI:
  .venv/bin/python -m benchmark.plugins.task_family.l14.builder \
      --pin-root external_knowledge/pinned/chembl/chembl-37 \
      --out bank/staging_20k/l14 --target-instances 1500 --workers 8 \
      --status-file data/erule/review5/build8000/expansion/lanes/L14/STATUS.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.source.l14.kb import L14KB, load_kb
from benchmark.plugins.task_family.l14.families import (
    ALLOWED_TOOLS, FAMILIES, INSTANCES_PER_GOAL, Family,
    base_goal_id, canonical_params, sha_text, source_unit_ids,
    state_descriptor,
)
from benchmark.plugins.verifier.l14.domain import L14Domain
from benchmark.plugins.verifier.l14 import plugin as verifier
from source_registry.fetchers.l14_chembl import (
    LICENSE_VERDICTS, RELEASE_LOCK, SQLITE_TARBALL,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
SCHEMA = "aob-task/5.0"
DATASET_ID = "autoontobench-bank-20000-candidate-dev1"
LANE = "L14"
SOURCE_GROUP = "chembl@release-37"
SNAPSHOT_ID = "chembl/chembl-37"
PIN_ROOT_REL = "external_knowledge/pinned/chembl/chembl-37"
SUPPORTED_PROFILES = sorted({f.profile for f in FAMILIES})
SHARD_SIZE = 250
MAX_INSTANCE_ATTEMPTS = 8
STATUS_MARKER = "<!-- L14-PRODUCTION-STATE -->"

CHEMBL_REF = {"source_id": "chembl",
              "source_rule_unit_id": "chembl/release-37",
              "commit": "release 37 (2026-05-01)",
              "file_sha256": RELEASE_LOCK[SQLITE_TARBALL]["sha256"]}

DATASET_DOC = """# ChEMBL 37 evidence slice (pinned snapshot)

Rows below are exact copies from the pinned ChEMBL 37 SQLite release
(DOI 10.6019/CHEMBL.database.37). Measurement semantics follow the ChEMBL
standardization rules:

- `standard_type` is the standardized measurement type; IC50, Ki, EC50, Kd
  and Potency are DISTINCT measures and must never be mixed in one set.
- `standard_relation` is part of the fact: '=' is an exact value, while
  '<', '>', '<=', '>=' are bounds and must never be read as exact values.
- `standard_value` is expressed in `standard_units`. Exact unit scales:
  1 uM = 1000 nM; 1 nM = 1000 pM; 1 mM = 1,000,000 nM.
- assay_type 'B' = binding assay; confidence_score >= 8 marks the
  highest-confidence assay-to-target assignments.
- molecule identity is by ChEMBL ID; molecule_hierarchy maps salt/form
  members to their registered parent molecule.

ChEMBL data is from http://www.ebi.ac.uk/chembl - the version of ChEMBL is
chembl_37 (CC BY-SA 3.0; cite Mendez et al. 2019, DOI 10.1093/nar/gky1075).
"""


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
    witness: dict[str, Any] = field(default_factory=dict)


def family_goal_quotas(families: tuple[Family, ...], goals_total: int
                       ) -> dict[str, int]:
    base = goals_total // len(families)
    rem = goals_total % len(families)
    return {f.family_id: base + (1 if i < rem else 0)
            for i, f in enumerate(families)}


# ---------------------------------------------------------------------------
# candidate sampling
# ---------------------------------------------------------------------------

def sample_base_goals(kb: L14KB, family: Family, n_goals: int,
                      seed: int) -> tuple[list[Candidate], list[dict]]:
    rng = random.Random(seed)
    out: list[Candidate] = []
    rejects: list[dict] = []
    seen: set[str] = set()
    for params in family.propose(kb, rng):
        key = canonical_params(params)
        if key in seen:
            continue
        seen.add(key)
        try:
            answer = family.answer(kb, params)
        except Exception as exc:
            rejects.append({"family": family.family_id, "params": params,
                            "code": "UNSOLVABLE",
                            "reason": f"reference_error: {exc}"})
            continue
        answer = tuple(sorted(set(answer)))
        n = len(answer)
        if n == 0:
            rejects.append({"family": family.family_id, "params": params,
                            "code": "UNSOLVABLE", "reason": "empty answer"})
            continue
        if not (family.min_answer <= n <= family.max_answer):
            rejects.append({"family": family.family_id, "params": params,
                            "code": "TRIVIAL",
                            "reason": f"answer size {n} outside "
                                      f"[{family.min_answer},"
                                      f"{family.max_answer}]"})
            continue
        bid = base_goal_id(family.family_id, params)
        out.append(Candidate(
            family=family, params=params, base_goal_id=bid, answer=answer,
            mechanism_group_id=f"l14/{family.family_id}/{bid[:12]}",
            user_goal=family.goal_text(params, kb),
            provenance="source_derived_composition",
            seed=seed,
        ))
        if len(out) >= n_goals:
            break
    return out, rejects


# ---------------------------------------------------------------------------
# reference verification (process pool): clean-parser recompute + witness
# ---------------------------------------------------------------------------

def _verify_one(args) -> tuple[str, dict[str, Any], dict[str, Any]]:
    pin_root, family_id, params, answer, witness_kind, seed, bid = args
    kb = load_kb(pin_root)
    ev = verifier.verify_reference(pin_root, family_id, params,
                                   tuple(answer), kb)
    wit = verifier.certify_witness(kb, family_id, witness_kind, params,
                                   tuple(answer), seed)
    return bid, ev, wit


def run_reference_verification(pin_root: Path, candidates: list[Candidate],
                               workers: int) -> None:
    jobs = [(str(pin_root), c.family.family_id, c.params, list(c.answer),
             c.family.witness_kind, c.seed, c.base_goal_id)
            for c in candidates]
    if workers <= 1:
        for j in jobs:
            bid, ev, wit = _verify_one(j)
            _assign(candidates, bid, ev, wit)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for bid, ev, wit in pool.map(_verify_one, jobs, chunksize=4):
            _assign(candidates, bid, ev, wit)


def _assign(candidates: list[Candidate], bid: str, ev: dict, wit: dict) -> None:
    for c in candidates:
        if c.base_goal_id == bid:
            c.reference_evidence = ev
            c.witness = wit
            return


# ---------------------------------------------------------------------------
# public bundle construction
# ---------------------------------------------------------------------------

def _q(con, sql, args=()):
    cur = con.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _q_in(con, sql_tmpl: str, col: str, ids: list[Any],
          extra_args: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Run ``sql_tmpl`` with a {col} IN (...) placeholder, chunked below
    SQLite's variable limit. sql_tmpl must contain exactly one '{marks}'."""
    out: list[dict[str, Any]] = []
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        marks = ",".join("?" * len(chunk))
        out.extend(_q(con, sql_tmpl.format(marks=marks),
                      tuple(chunk) + tuple(extra_args)))
    return out


ACTIVITY_COLS = ("act.activity_id, act.assay_id, act.molregno, "
                 "act.standard_type, act.standard_relation, "
                 "act.standard_value, act.standard_units, "
                 "act.type AS published_type, act.relation AS "
                 "published_relation, act.value AS published_value, "
                 "act.units AS published_units")
ASSAY_COLS = ("assay_id, chembl_id, assay_type, assay_organism, tid, "
              "doc_id, confidence_score, description")
MOLECULE_COLS = "molregno, chembl_id, pref_name, molecule_type, max_phase"
TARGET_COLS = ("tid, chembl_id, pref_name, target_type, organism, tax_id")
DOC_COLS = "doc_id, chembl_id, journal, year, doi, pubmed_id"


def scope_rows(kb: L14KB, family_id: str,
               params: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """All source rows the agent needs to derive the answer (no answer list,
    no gold markers)."""
    con = kb.connection()
    try:
        return _scope_rows(con, family_id, params)
    finally:
        con.close()


def _scope_rows(con, family_id: str,
                params: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}

    def molecules_for_activities():
        mols = sorted({r["molregno"] for r in out.get("activities", [])
                       if r.get("molregno") is not None})
        if mols:
            out["molecules"] = _q_in(
                con, f"SELECT {MOLECULE_COLS} FROM molecule_dictionary "
                     "WHERE molregno IN ({marks})", "molregno", mols)
            out["molecule_hierarchy"] = _q_in(
                con, "SELECT molregno, parent_molregno, active_molregno "
                     "FROM molecule_hierarchy WHERE molregno IN ({marks})",
                "molregno", mols)

    if family_id == "org_parent_salt_members":
        p = params["parent_chembl_id"]
        out["molecules"] = _q(
            con, f"SELECT {MOLECULE_COLS} FROM molecule_dictionary WHERE "
                 "molregno IN (SELECT molregno FROM molecule_hierarchy WHERE "
                 "parent_molregno = (SELECT molregno FROM "
                 "molecule_dictionary WHERE chembl_id = ?)) "
                 "OR chembl_id = ?", (p, p))
        out["molecule_hierarchy"] = _q(
            con, "SELECT molregno, parent_molregno, active_molregno FROM "
                 "molecule_hierarchy WHERE parent_molregno = (SELECT molregno "
                 "FROM molecule_dictionary WHERE chembl_id = ?)", (p,))
        return out

    if family_id == "app_molecule_assays_of_type":
        m = params["molecule_chembl_id"]
        out["activities"] = _q(
            con, f"SELECT {ACTIVITY_COLS} FROM activities act WHERE "
                 "act.molregno = (SELECT molregno FROM molecule_dictionary "
                 "WHERE chembl_id = ?)", (m,))
        aids = sorted({r["assay_id"] for r in out["activities"]})
        if aids:
            out["assays"] = _q_in(con, f"SELECT {ASSAY_COLS} FROM assays "
                                       "WHERE assay_id IN ({marks})",
                                  "assay_id", aids)
        molecules_for_activities()
        return out

    if family_id in ("app_target_binding_set", "org_assay_context_organism",
                     "app_multihop_target_journal"):
        t = params["target_chembl_id"]
        out["targets"] = _q(
            con, f"SELECT {TARGET_COLS} FROM target_dictionary WHERE "
                 "chembl_id = ?", (t,))
        out["assays"] = _q(
            con, f"SELECT {ASSAY_COLS} FROM assays WHERE tid = (SELECT tid "
                 "FROM target_dictionary WHERE chembl_id = ?)", (t,))
        aids = sorted({r["assay_id"] for r in out["assays"]})
        if aids:
            out["activities"] = _q_in(
                con, f"SELECT {ACTIVITY_COLS} FROM activities act WHERE "
                     "act.assay_id IN ({marks})", "assay_id", aids)
            if family_id == "app_multihop_target_journal":
                dids = sorted({r["doc_id"] for r in out["assays"]
                               if r.get("doc_id") is not None})
                if dids:
                    out["docs"] = _q_in(
                        con, f"SELECT {DOC_COLS} FROM docs WHERE doc_id IN "
                             "({marks})", "doc_id", dids)
        molecules_for_activities()
        return out

    if family_id in ("app_measurement_exact_set", "rep_relation_flip_repair",
                     "rep_standardization_table",
                     "app_unit_normalized_potency"):
        a = params["assay_chembl_id"]
        out["assays"] = _q(
            con, f"SELECT {ASSAY_COLS} FROM assays WHERE chembl_id = ?", (a,))
        out["activities"] = _q(
            con, f"SELECT {ACTIVITY_COLS} FROM activities act WHERE "
                 "act.assay_id = (SELECT assay_id FROM assays WHERE "
                 "chembl_id = ?)", (a,))
        tids = sorted({r["tid"] for r in out["assays"]
                       if r.get("tid") is not None})
        if tids:
            marks = ",".join("?" * len(tids))
            out["targets"] = _q(
                con, f"SELECT {TARGET_COLS} FROM target_dictionary WHERE "
                     f"tid IN ({marks})", tuple(tids))
        molecules_for_activities()
        return out

    if family_id == "org_target_class_members":
        out["protein_classification"] = _q(
            con, "SELECT protein_class_id, parent_id, pref_name, short_name, "
                 "class_level FROM protein_classification")
        out["component_class"] = _q(
            con, "SELECT component_id, protein_class_id FROM component_class")
        out["target_components"] = _q(
            con, "SELECT tid, component_id FROM target_components")
        out["targets"] = _q(
            con, f"SELECT {TARGET_COLS} FROM target_dictionary WHERE "
                 "target_type = 'SINGLE PROTEIN' AND organism = "
                 "'Homo sapiens'")
        return out

    if family_id == "org_journal_year_provenance":
        j, y = params["journal"], params["year"]
        out["docs"] = _q(con, f"SELECT {DOC_COLS} FROM docs WHERE journal = ? "
                              "AND year = ?", (j, y))
        dids = sorted({r["doc_id"] for r in out["docs"]})
        if dids:
            out["assays"] = _q_in(con, f"SELECT {ASSAY_COLS} FROM assays WHERE "
                                       "doc_id IN ({marks})", "doc_id", dids)
            aids = sorted({r["assay_id"] for r in out["assays"]})
            if aids:
                out["activities"] = _q_in(
                    con, f"SELECT {ACTIVITY_COLS} FROM activities act WHERE "
                         "act.assay_id IN ({marks})", "assay_id", aids)
        molecules_for_activities()
        return out

    if family_id == "app_mechanism_action_type":
        t = params["target_chembl_id"]
        out["targets"] = _q(con, f"SELECT {TARGET_COLS} FROM "
                                 "target_dictionary WHERE chembl_id = ?", (t,))
        out["drug_mechanism"] = _q(
            con, "SELECT mec_id, molregno, tid, mechanism_of_action, "
                 "action_type, direct_interaction FROM drug_mechanism WHERE "
                 "tid = (SELECT tid FROM target_dictionary WHERE "
                 "chembl_id = ?)", (t,))
        mols = sorted({r["molregno"] for r in out["drug_mechanism"]})
        if mols:
            out["molecule_hierarchy"] = _q_in(
                con, "SELECT molregno, parent_molregno, active_molregno FROM "
                     "molecule_hierarchy WHERE molregno IN ({marks})",
                "molregno", mols)
            parents = sorted({r["parent_molregno"] for r in
                              out["molecule_hierarchy"]
                              if r.get("parent_molregno") is not None}
                             | set(mols))
            out["molecules"] = _q_in(
                con, f"SELECT {MOLECULE_COLS} FROM molecule_dictionary WHERE "
                     "molregno IN ({marks})", "molregno", parents)
        return out

    if family_id == "app_indication_mesh":
        m = params["mesh_heading"]
        out["drug_indication"] = _q(
            con, "SELECT drugind_id, molregno, mesh_heading, efo_id, "
                 "efo_term, max_phase_for_ind FROM drug_indication WHERE "
                 "mesh_heading = ?", (m,))
        mols = sorted({r["molregno"] for r in out["drug_indication"]})
        if mols:
            out["molecule_hierarchy"] = _q_in(
                con, "SELECT molregno, parent_molregno, active_molregno FROM "
                     "molecule_hierarchy WHERE molregno IN ({marks})",
                "molregno", mols)
            parents = sorted({r["parent_molregno"] for r in
                              out["molecule_hierarchy"]
                              if r.get("parent_molregno") is not None}
                             | set(mols))
            out["molecules"] = _q_in(
                con, f"SELECT {MOLECULE_COLS} FROM molecule_dictionary WHERE "
                     "molregno IN ({marks})", "molregno", parents)
        return out

    if family_id == "org_uniprot_target_map":
        acc = params["accession"]
        out["component_sequences"] = _q(
            con, "SELECT component_id, accession, description, organism "
                 "FROM component_sequences WHERE accession = ?", (acc,))
        comps = sorted({r["component_id"] for r in out["component_sequences"]})
        if comps:
            out["target_components"] = _q_in(
                con, "SELECT tid, component_id FROM target_components WHERE "
                     "component_id IN ({marks})", "component_id", comps)
            tids = sorted({r["tid"] for r in out["target_components"]})
            if tids:
                out["targets"] = _q_in(
                    con, f"SELECT {TARGET_COLS} FROM target_dictionary WHERE "
                         "tid IN ({marks})", "tid", tids)
        return out

    raise ValueError(f"no scope builder for {family_id}")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n"
        for r in rows), encoding="utf-8")


def build_bundle(kb: L14KB, fam: Family, params: dict[str, Any],
                 variant: str, out_dir: Path,
                 distractor_seed: int | None = None,
                 corruption_seed: int | None = None
                 ) -> tuple[list[Path], dict[str, Any]]:
    """Write the agent-visible evidence bundle. Returns (files, meta).

    variants: rows_jsonl | sqlite_slice. Repair families also get the
    corrupted artifact + expected-corrections record in meta. The unit
    normalization family redacts standard_value/standard_units from the
    activity rows: the agent must derive nM potencies from the published
    rows via the documented exact scale rules.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scope = scope_rows(kb, fam.family_id, params)
    if fam.family_id == "app_unit_normalized_potency":
        for r in scope.get("activities", []):
            r.pop("standard_value", None)
            r.pop("standard_units", None)
    meta: dict[str, Any] = {"ok": True}

    if distractor_seed is not None and "activities" in scope:
        # seeded distractor rows: activities of an unrelated target
        rng = random.Random(distractor_seed)
        con = kb.connection()
        try:
            dt = _q(con, "SELECT tid FROM target_dictionary ORDER BY tid "
                         "LIMIT 4000")
            pick = dt[rng.randrange(len(dt))]["tid"]
            d_assays = _q(con, f"SELECT {ASSAY_COLS} FROM assays WHERE "
                               "tid = ? ORDER BY assay_id LIMIT 12", (pick,))
            d_aids = [a["assay_id"] for a in d_assays]
            if d_aids:
                marks = ",".join("?" * len(d_aids))
                d_acts = _q(con, f"SELECT {ACTIVITY_COLS} FROM activities act "
                                 f"WHERE act.assay_id IN ({marks}) "
                                 "ORDER BY act.activity_id LIMIT 60",
                            tuple(d_aids))
                scope.setdefault("assays", []).extend(d_assays)
                scope["activities"].extend(d_acts)
                d_mols = sorted({r["molregno"] for r in d_acts
                                 if r.get("molregno") is not None})
                if d_mols:
                    marks = ",".join("?" * len(d_mols))
                    scope.setdefault("molecules", []).extend(_q(
                        con, f"SELECT {MOLECULE_COLS} FROM "
                             f"molecule_dictionary WHERE molregno IN "
                             f"({marks})", tuple(d_mols)))
                meta["distractor_tid"] = pick
        finally:
            con.close()

    corruption: dict[str, Any] = {}
    if fam.regime == "repair":
        corruption = _build_corruption(scope, fam.family_id, params,
                                       corruption_seed or 0)
        meta["corruption"] = {k: (len(v) if isinstance(v, dict) else v)
                              for k, v in corruption.items()}
        meta["expected_artifact"] = corruption["expected"]
        meta["probe_budget"] = 0

    written: list[Path] = []
    doc = out_dir / "DATASET.md"
    doc.write_text(DATASET_DOC, encoding="utf-8")
    written.append(doc)
    if variant == "rows_jsonl":
        for name, rows in sorted(scope.items()):
            path = out_dir / f"{name}.jsonl"
            _write_jsonl(path, rows)
            written.append(path)
    elif variant == "sqlite_slice":
        dbp = out_dir / "evidence_slice.sqlite"
        _write_sqlite_slice(dbp, scope)
        written.append(dbp)
    else:
        raise ValueError(variant)
    if fam.regime == "repair":
        art = out_dir / "standardization_artifact.json"
        art.write_text(json.dumps(corruption["artifact"], indent=2,
                                  sort_keys=True, ensure_ascii=False) + "\n",
                       encoding="utf-8")
        written.append(art)
    return written, meta


def _write_sqlite_slice(path: Path, scope: dict[str, list[dict[str, Any]]]
                        ) -> None:
    if path.exists():
        path.unlink()
    con = sqlite3.connect(str(path))
    try:
        for name, rows in sorted(scope.items()):
            if not rows:
                continue
            cols = sorted({k for r in rows for k in r})
            defs = ", ".join(f'"{c}" TEXT' for c in cols)
            con.execute(f'CREATE TABLE "{name}" ({defs})')
            marks = ",".join("?" * len(cols))
            con.executemany(
                f'INSERT INTO "{name}" ({", ".join(chr(34) + c + chr(34) for c in cols)}) '
                f"VALUES ({marks})",
                [[json.dumps(r.get(c), ensure_ascii=False)
                  if isinstance(r.get(c), (dict, list)) else r.get(c)
                  for c in cols] for r in rows])
        con.commit()
    finally:
        con.close()


def _build_corruption(scope: dict[str, list[dict[str, Any]]], family_id: str,
                      params: dict[str, Any], seed: int) -> dict[str, Any]:
    """Corrupt the standardization artifact against published evidence."""
    rng = random.Random(seed)
    acts = [r for r in scope.get("activities", [])
            if r.get("standard_type") == params["standard_type"]
            and r.get("standard_value") is not None
            and r.get("standard_units") == "nM"]
    artifact_rows: dict[str, dict[str, Any]] = {}
    expected: dict[str, dict[str, Any]] = {}
    corruptible = []
    for r in acts:
        key = str(r["activity_id"])
        row = {"activity_id": r["activity_id"],
               "published_type": r.get("published_type"),
               "published_relation": r.get("published_relation"),
               "published_value": r.get("published_value"),
               "published_units": r.get("published_units"),
               "standard_type": r["standard_type"],
               "standard_relation": r.get("standard_relation"),
               "standard_value": r.get("standard_value"),
               "standard_units": r.get("standard_units")}
        artifact_rows[key] = dict(row)
        corruptible.append((key, r, row))
    n_corrupt = min(len(corruptible), max(3, len(corruptible) // 4))
    picks = rng.sample(corruptible, n_corrupt) if corruptible else []
    for key, r, row in picks:
        mode = rng.randrange(3)
        if family_id == "rep_relation_flip_repair":
            mode = 0
        if mode == 0 and r.get("standard_relation") in ("<", ">"):
            expected[key] = {"standard_relation": r["standard_relation"]}
            row["standard_relation"] = "="
        elif mode == 0:
            expected[key] = {"standard_relation": r["standard_relation"]}
            row["standard_relation"] = ("<" if r.get("standard_relation")
                                        == "=" else "=")
        elif mode == 1:
            expected[key] = {"standard_value": r["standard_value"]}
            row["standard_value"] = (r["standard_value"] or 0) * 1000.0
            row["standard_units"] = "uM"
            expected[key]["standard_units"] = "nM"
        else:
            expected[key] = {"standard_type": r["standard_type"]}
            others = [t for t in ("IC50", "Ki", "EC50", "Kd", "Potency")
                      if t != r["standard_type"]]
            row["standard_type"] = others[rng.randrange(len(others))]
    return {"artifact": {"rows": artifact_rows},
            "expected": {"corrected_rows": expected},
            "n_corrupted": len(expected)}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

BANNED_TOKENS = ('"expected"', '"gold"', "private_", "oracle_handle")


def wording_leak_check(user_goal: str, answer: tuple[str, ...],
                       params: dict[str, Any]) -> str | None:
    named = [str(v) for v in params.values() if isinstance(v, str)]
    if not any(t in user_goal for t in named):
        return "wording does not name the query target"
    if len(answer) <= 200:
        payload = json.dumps(sorted(answer))
        if payload in user_goal:
            return "wording leaks the serialized answer"
    return None


def bundle_leak_scan(bundle_files: list[Path], answer: tuple[str, ...]
                     ) -> str | None:
    payload = json.dumps(sorted(answer))
    for path in bundle_files:
        if path.suffix == ".sqlite":
            continue  # binary; answer-list ban enforced on jsonl/json/md
        text = path.read_text(encoding="utf-8")
        for tok in BANNED_TOKENS:
            if tok in text:
                return f"public bundle file {path.name} contains banned " \
                       f"token {tok}"
        if payload in text:
            return f"public bundle file {path.name} contains the " \
                   f"serialized answer"
    return None


def bundle_parse_check(bundle_files: list[Path]) -> str | None:
    """Actually parse every public bundle file: JSONL lines, JSON docs and
    the sqlite_slice must all be machine-readable. Backs the
    ontology_parse_valid status (never asserted without execution)."""
    for path in bundle_files:
        try:
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as fh:
                    for line in fh:
                        if line.strip():
                            json.loads(line)
            elif path.suffix == ".json":
                json.loads(path.read_text(encoding="utf-8"))
            elif path.suffix == ".sqlite":
                con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
                tables = [r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'")]
                if not tables:
                    return f"{path.name}: sqlite slice has no tables"
                for t in tables:
                    con.execute(f'SELECT count(*) FROM "{t}"').fetchone()
                con.close()
            elif path.suffix == ".md":
                path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"{path.name}: parse failed: {exc}"
    return None


REQUIRED_RECORD_FIELDS = (
    "schema_version", "task_id", "dataset_id", "lane_id", "family_id",
    "base_goal_id", "instance_id", "task_regime", "complexity",
    "semantic_profile", "public", "oracle_handle", "status",
    "qualification", "native_evidence", "source_snapshot_ids",
    "source_unit_ids", "check_statuses",
)


def record_schema_check(rec: dict[str, Any]) -> str | None:
    """Field presence/type check backing the schema_valid status."""
    for k in REQUIRED_RECORD_FIELDS:
        if k not in rec:
            return f"missing field {k}"
    if rec["schema_version"] != SCHEMA:
        return f"schema_version {rec['schema_version']} != {SCHEMA}"
    if rec["dataset_id"] != DATASET_ID:
        return f"dataset_id {rec['dataset_id']} != {DATASET_ID}"
    pub = rec["public"]
    for k in ("user_goal", "allowed_tools", "budget", "observation_refs"):
        if k not in pub:
            return f"public.{k} missing"
    if not isinstance(pub["user_goal"], str) or len(pub["user_goal"]) < 40:
        return "user_goal too short"
    if not pub["observation_refs"]:
        return "empty observation_refs"
    return None


# ---------------------------------------------------------------------------
# instance materialization
# ---------------------------------------------------------------------------

def materialize_goal(c: Candidate, kb: L14KB, out_root: Path,
                     domain: L14Domain, seen_instances: set[str],
                     stats: dict[str, Any]
                     ) -> list[dict[str, Any]] | None:
    fam = c.family
    made: list[dict[str, Any]] = []
    n_instances = fam.instances_per_goal
    goal_paths: list[Path] = []
    inst_rejections: list[dict[str, Any]] = []
    inst_idx = 0
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
        task_id = f"l14-{c.base_goal_id[:12]}-{instance_id[:8]}"
        public_dir = out_root / "public" / task_id

        try:
            bundle_files, meta = build_bundle(
                kb, fam, c.params, state_desc["variant"], public_dir,
                distractor_seed=state_desc.get("distractor_seed"),
                corruption_seed=wit_seed)
        except Exception as exc:
            bundle_failure = f"state build: {exc}"
            break
        goal_paths.append(public_dir)
        if not meta.get("ok"):
            bundle_failure = f"state build: {meta.get('reason')}"
            break
        parse_err = bundle_parse_check(bundle_files)
        if parse_err:
            bundle_failure = f"bundle parse: {parse_err}"
            break
        leak = bundle_leak_scan(bundle_files, c.answer)
        if leak:
            bundle_failure = leak
            break

        ev = c.reference_evidence
        init_state = {
            "task_id": task_id, "family": fam.family_id, "params": c.params,
            "query": {"family": fam.family_id, "params": c.params},
            "bundle_files": [str(p.relative_to(out_root))
                             for p in bundle_files],
            "state": state_desc,
            "bundle_sha256": sha_text("".join(
                sha_text(p.read_bytes().hex()) for p in bundle_files)),
        }
        state_path = public_dir / "initial_state.json"
        state_path.write_text(json.dumps(init_state, indent=2,
                                         sort_keys=True) + "\n",
                              encoding="utf-8")

        query_obj = {"family": fam.family_id, "params": c.params}
        expected_artifact = meta.get("expected_artifact")
        st = domain.reset(task_id, query_obj)
        reset_result = domain.verify_terminal(
            st, c.answer, reference_engine=ev.get("engine", "sqlite3"),
            evidence_db_sha256=ev.get("db_manifest_sha", ""),
            expected_artifact=expected_artifact)
        st2, _fb = domain.step(
            st, {"type": "submit_answer", "answer": list(c.answer)})
        if fam.regime == "repair" and expected_artifact is not None:
            fixed_rows = {k: dict(v) for k, v in
                          (expected_artifact.get("corrected_rows") or {})
                          .items()}
            art_payload = {"rows": {}}
            for key in list(fixed_rows):
                art_payload["rows"][key] = fixed_rows[key]
            st2, _ = domain.step(
                domain.reset(task_id, query_obj),
                {"type": "submit_artifact", "artifact": art_payload})
            st2, _ = domain.step(
                st2, {"type": "submit_answer", "answer": list(c.answer)})
        term_result = domain.verify_terminal(
            st2, c.answer, reference_engine=ev.get("engine", "sqlite3"),
            evidence_db_sha256=ev.get("db_manifest_sha", ""),
            expected_artifact=expected_artifact)
        if reset_result.goal_reached or not term_result.goal_reached:
            bundle_failure = "domain replay failed to discriminate"
            break

        seen_instances.add(instance_id)
        oracle_handle = "l14-oracle-" + sha_text(task_id)[:16]
        oracle = {
            "task_id": task_id,
            "family": fam.family_id,
            "params": c.params,
            "expected_answer": list(c.answer),
            "expected_answer_sha256": sha_json(sorted(set(c.answer))),
            "reference_plan": reference_plan(fam, meta),
            "witness": c.witness,
            "reference": ev,
            "repair": ({"expected_artifact": expected_artifact}
                       if expected_artifact else None),
            "corruption": meta.get("corruption"),
        }
        oracle_path = out_root / "oracle" / f"{oracle_handle}.json"
        oracle_path.parent.mkdir(parents=True, exist_ok=True)
        oracle_path.write_text(json.dumps(oracle, indent=2, sort_keys=True,
                                          ensure_ascii=False) + "\n",
                               encoding="utf-8")
        goal_paths.append(oracle_path)
        rec = task_record(
            c=c, task_id=task_id, instance_id=instance_id,
            state_desc=state_desc, public_dir=public_dir,
            state_path=state_path, oracle_handle=oracle_handle,
            out_root=out_root, meta=meta,
            reset_ok=not reset_result.goal_reached,
            terminal_ok=term_result.goal_reached)
        schema_err = record_schema_check(rec)
        if schema_err:
            _remove_paths([public_dir, oracle_path])
            inst_rejections.append({
                "family": fam.family_id, "params": c.params,
                "base_goal_id": c.base_goal_id, "code": "SCHEMA",
                "reason": schema_err})
            continue
        made.append(rec)

    stats["rejections"].setdefault("_instance_rows", []).extend(
        inst_rejections)
    for r in inst_rejections:
        stats["rejections"][r["code"]] = stats["rejections"].get(
            r["code"], 0) + 1
    if len(made) < n_instances:
        _remove_paths(goal_paths)
        reason = bundle_failure or (
            f"only {len(made)}/{n_instances} instances materialized "
            f"in {attempts} attempts")
        stats["rejections"]["_dropped_goal_rows"].append({
            "family": fam.family_id, "params": c.params,
            "base_goal_id": c.base_goal_id, "code": "STATE_BUILD",
            "reason": reason})
        stats["rejections"]["STATE_BUILD"] = stats["rejections"].get(
            "STATE_BUILD", 0) + 1
        return None
    return made


def _remove_paths(paths: list[Path]) -> None:
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        elif p.exists():
            p.unlink(missing_ok=True)


def reference_plan(fam: Family, meta: dict[str, Any]) -> dict[str, Any]:
    if fam.regime == "repair":
        return {"steps": ["inspect corrupted standardization artifact",
                          "align with published evidence rows",
                          "repair type/qualifier/unit entries",
                          "submit repaired artifact", "submit answer"],
                "probe_budget": meta.get("probe_budget", 0),
                "min_actions": 2}
    return {"steps": ["align provided source views",
                      "compute exact answer set", "submit answer"],
            "probe_budget": 0, "min_actions": 1}


def task_record(c: Candidate, task_id: str, instance_id: str,
                state_desc: dict[str, Any], public_dir: Path,
                state_path: Path, oracle_handle: str, out_root: Path,
                meta: dict[str, Any], reset_ok: bool,
                terminal_ok: bool) -> dict[str, Any]:
    fam = c.family
    ev = c.reference_evidence
    rel = lambda p: str(p.relative_to(out_root))
    bundle_files = sorted(
        str(p.relative_to(out_root))
        for p in public_dir.iterdir() if p.is_file())
    oracle_rel = f"oracle/{oracle_handle}.json"
    reverify_cmd = (
        ".venv/bin/python -m benchmark.plugins.verifier.l14.reverify "
        f"--pin-root {PIN_ROOT_REL} "
        f"--oracle bank/staging_20k/l14/{oracle_rel}")
    return {
        "schema_version": SCHEMA,
        "task_id": task_id,
        "dataset_id": DATASET_ID,
        "lane_id": LANE,
        "domain": "drug_discovery_bioactivity",
        "track": "analytical",
        "task_family": fam.regime,
        "task_regime": fam.regime,
        "family_id": f"l14.{fam.family_id}",
        "base_goal_id": c.base_goal_id,
        "instance_id": instance_id,
        "origin_mechanism_group": c.mechanism_group_id,
        "source_group_id": SOURCE_GROUP,
        "source_refs": [CHEMBL_REF],
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_unit_ids": list(source_unit_ids(c.params)),
        "combination_signature": sha_text(canonical_params(c.params)),
        "mechanism_group_id": c.mechanism_group_id,
        "complexity": fam.complexity,
        "public": {
            "user_goal": c.user_goal,
            "allowed_tools": list(ALLOWED_TOOLS),
            "budget": {"actions": 4,
                       "probes": meta.get("probe_budget", 0)},
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
            "reference_clean_parser_matches_fast_sql", "reset_not_solved",
            "terminal_solved", "witness_governed_flips",
            "witness_control_unchanged", "no_answer_leak"],
        "check_statuses": {
            "source_verified": "PASS",
            "schema_valid": "PASS",
            "ontology_parse_valid": "PASS",
            "native_execution_status": "PASS",
            "oracle_check_status": "PASS",
            "mutation_check_status": "PASS",
            "human_review_status": "NOT_RUN",
        },
        "native_evidence": {
            "stage": "reference_clean_parser_matches_fast_sql",
            "verdict": ev.get("verdict"),
            "engine": ev.get("engine"),
            "db_manifest_sha": ev.get("db_manifest_sha"),
            "answer_sha256": sha_json(sorted(set(c.answer))),
            "commands": [reverify_cmd],
            "witness": {"kind": fam.witness_kind, "ok": True,
                        "witness_sha256": sha_json(c.witness)},
            "domain_replay": {"reset_goal_reached": not reset_ok,
                              "terminal_goal_reached": terminal_ok},
        },
        "status": "QUALIFIED",
        "qualification": {
            "native_verified": True, "solvable": True, "nontrivial": True,
            "semantic_relevance": True, "leakage_clean": True,
            "release_eligible": True},
        "qualification_status": "QUALIFIED_NATIVE",
        "state_axis": state_desc,
        "answer_size": len(c.answer),
        "x_l14": {
            "witness_kind": fam.witness_kind,
            "witness_ok": True,
            "db_manifest_sha": ev.get("db_manifest_sha"),
            "answer_sha256": sha_json(sorted(set(c.answer))),
        },
    }


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
        head = prior.split(STATUS_MARKER)[0].rstrip() \
            if STATUS_MARKER in prior else prior.rstrip()
        lines = [STATUS_MARKER, "",
                 "## Production state (auto-updated by builder; "
                 "overwritten each flush)", "",
                 f"- updated_utc: {utcnow()}",
                 f"- phase: {phase}"]
        lines += [f"- {k}: {v}" for k, v in counts.items()]
        if self.flush_lines:
            lines += ["", "### shard flush log"] + \
                     [f"- {x}" for x in self.flush_lines]
        lines += ["", f"- next: {next_step}", f"- blocker: {blocker}", ""]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(head + "\n\n" + "\n".join(lines),
                             encoding="utf-8")


# ---------------------------------------------------------------------------
# shard writing
# ---------------------------------------------------------------------------

def _write_jsonl_shard(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False)
                     + "\n")
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
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def select_ship_goals(candidates: list[Candidate], quotas: dict[str, int],
                      goals_total: int, family_goal_cap: int,
                      stats: dict[str, Any]
                      ) -> tuple[list[Candidate], list[Candidate]]:
    ship: list[Candidate] = []
    surplus: list[Candidate] = []
    per_family: dict[str, int] = {f.family_id: 0 for f in FAMILIES}
    stats["surplus_donated_goals"] = 0
    stats["goal_shortfall_after_donation"] = 0
    for c in candidates:
        ev, wit = c.reference_evidence, c.witness
        if ev.get("verdict") != "PASS":
            stats["rejections"]["_selection_rows"].append({
                "family": c.family.family_id, "params": c.params,
                "base_goal_id": c.base_goal_id, "code": "REFERENCE_MISMATCH",
                "reason": f"clean-parser verdict {ev.get('verdict')}: "
                          f"{ev.get('reason', '')}"})
            stats["rejections"]["REFERENCE_MISMATCH"] = \
                stats["rejections"].get("REFERENCE_MISMATCH", 0) + 1
            continue
        if not wit.get("ok"):
            stats["rejections"]["_selection_rows"].append({
                "family": c.family.family_id, "params": c.params,
                "base_goal_id": c.base_goal_id, "code": "NON_SEMANTIC",
                "reason": f"witness: {wit.get('reason', '')}"})
            stats["rejections"]["NON_SEMANTIC"] = \
                stats["rejections"].get("NON_SEMANTIC", 0) + 1
            continue
        leak = wording_leak_check(c.user_goal, c.answer, c.params)
        if leak:
            stats["rejections"]["_selection_rows"].append({
                "family": c.family.family_id, "params": c.params,
                "base_goal_id": c.base_goal_id, "code": "LEAKAGE",
                "reason": leak})
            stats["rejections"]["LEAKAGE"] = \
                stats["rejections"].get("LEAKAGE", 0) + 1
            continue
        if per_family[c.family.family_id] < quotas[c.family.family_id]:
            ship.append(c)
            per_family[c.family.family_id] += 1
        else:
            surplus.append(c)
    ship_goals = len(ship)
    if ship_goals < goals_total:
        shortfall = goals_total - ship_goals
        donated = 0
        keep_surplus: list[Candidate] = []
        for c in surplus:
            if donated < shortfall and per_family[c.family.family_id] < \
                    family_goal_cap:
                ship.append(c)
                per_family[c.family.family_id] += 1
                donated += 1
            else:
                keep_surplus.append(c)
        surplus = keep_surplus
        stats["surplus_donated_goals"] = donated
        stats["goal_shortfall_after_donation"] = shortfall - donated
    return ship, surplus


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def build(pin_root: Path, out_root: Path, target_instances: int,
          workers: int, seed: int = 20261101,
          status_path: Path | None = None,
          smoke_n: int | None = None) -> dict[str, Any]:
    t0 = time.time()
    kb = load_kb(str(pin_root))
    kb.pools()  # warm pool cache once in the parent process
    status = StatusWriter(status_path)
    goals_total = target_instances // INSTANCES_PER_GOAL
    quotas = family_goal_quotas(FAMILIES, goals_total)
    family_goal_cap = max(24, int(target_instances * 0.08)) // \
        INSTANCES_PER_GOAL
    stats: dict[str, Any] = {
        "qualified": 0,
        "rejections": {"_dropped_goal_rows": [], "_selection_rows": []},
        "families": {},
        "started_utc": utcnow(),
    }
    status.update("sampling",
                  {"target_instances": target_instances,
                   "goals_total": goals_total,
                   "families": len(FAMILIES)},
                  "per-family base-goal sampling")

    candidates: list[Candidate] = []
    sample_rejections: list[dict] = []
    seq = 0
    for i, fam in enumerate(FAMILIES):
        quota = quotas[fam.family_id]
        if smoke_n is not None:
            n_sample = min(fam.max_base_goals,
                           max(2, smoke_n // len(FAMILIES) + 1))
        else:
            n_sample = min(fam.max_base_goals,
                           max(quota + 10, int(quota * 2.2) + 1))
        cands, rejects = sample_base_goals(kb, fam, n_sample, seed + i * 977)
        for c in cands:
            c.seq = seq
            seq += 1
        candidates.extend(cands)
        sample_rejections.extend(rejects)
        for r in rejects:
            stats["rejections"][r["code"]] = stats["rejections"].get(
                r["code"], 0) + 1
        stats["families"][fam.family_id] = {"candidates": len(cands),
                                            "quota_goals": quota}
        print(f"[sample {i + 1}/{len(FAMILIES)}] {fam.family_id} "
              f"candidates={len(cands)} rejects={len(rejects)}", flush=True)

    status.update("reference_verify",
                  {"candidates": len(candidates), "workers": workers},
                  "clean-parser reference verification + witness",
                  log_line=f"{utcnow()} sampled {len(candidates)} candidates "
                           f"({len(sample_rejections)} sampling rejections)")
    run_reference_verification(pin_root, candidates, workers)
    passed = sum(1 for c in candidates
                 if c.reference_evidence.get("verdict") == "PASS")
    wit_ok = sum(1 for c in candidates if c.witness.get("ok"))
    print(f"reference: {passed}/{len(candidates)} PASS; witness ok "
          f"{wit_ok}/{len(candidates)}", flush=True)

    ship, surplus = select_ship_goals(candidates, quotas, goals_total,
                                      family_goal_cap, stats)
    status.update("materialize",
                  {"candidates": len(candidates), "reference_pass": passed,
                   "witness_ok": wit_ok, "ship_goals": len(ship),
                   "surplus_goals": len(surplus)},
                  "goal-atomic materialization with 250-record shard flushes",
                  log_line=f"{utcnow()} selection: ship {len(ship)}"
                           f"/{goals_total} goals (reference pass "
                           f"{passed}/{len(candidates)}, witness ok "
                           f"{wit_ok}, donated "
                           f"{stats['surplus_donated_goals']})")

    domain = L14Domain()
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
        recs = materialize_goal(c, kb, out_root, domain, seen_instances,
                                stats)
        if recs is None:
            replacement = None
            while surplus_iter:
                alt = surplus_iter.pop(0)
                if alt.base_goal_id in shipped_goal_ids:
                    continue
                if shipped_per_family[alt.family.family_id] >= \
                        family_goal_cap:
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
            m = write_shard(out_root, "qualified", len(shard_manifests),
                            rows)
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
    for r in (sample_rejections + selection_rows + instance_rows
              + dropped_goal_rows):
        bg = r.get("base_goal_id")
        if bg and bg not in rejection_by_goal:
            rejection_by_goal[bg] = r
    candidate_rows = []
    for c in candidates:
        fid = f"l14.{c.family.family_id}"
        ev = c.reference_evidence
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
            "native_evidence": {k: v for k, v in ev.items()},
            "witness_ok": c.witness.get("ok", False),
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
            fh.write(json.dumps(row, sort_keys=True, ensure_ascii=False)
                     + "\n")
    (rej_dir / "manifest.json").write_text(json.dumps({
        "lane": LANE, "records": len(all_rejections),
        "sha256": sha_text(rej_path.read_text(encoding="utf-8")),
        "path": str(rej_path.resolve()),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stats.update({
        "wall_seconds": round(time.time() - t0, 2),
        "base_goals": len(shipped_goal_ids),
        "unique_families": len({f"l14.{c.family.family_id}"
                                for c in candidates
                                if c.base_goal_id in shipped_goal_ids}),
        "candidates": len(candidate_rows),
        "candidates_native_pass": passed,
        "candidates_witness_ok": wit_ok,
    })
    per_family_qualified_inst: dict[str, int] = {
        f"l14.{f.family_id}": 0 for f in FAMILIES}
    for c in candidates:
        if c.base_goal_id in shipped_goal_ids:
            per_family_qualified_inst[f"l14.{c.family.family_id}"] += \
                INSTANCES_PER_GOAL
    build_report = {
        "lane": LANE,
        "dataset_id": DATASET_ID,
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_refs": [CHEMBL_REF],
        "target_qualified_instances": target_instances,
        "instances_per_goal": INSTANCES_PER_GOAL,
        "started_at": stats["started_utc"],
        "wall_seconds": stats["wall_seconds"],
        "base_goals": stats["base_goals"],
        "families": stats["unique_families"],
        "candidates": stats["candidates"],
        "candidates_native_pass": passed,
        "candidates_witness_ok": wit_ok,
        "qualified": stats["qualified"],
        "surplus_donated_goals": stats["surplus_donated_goals"],
        "goal_shortfall_after_donation":
            stats["goal_shortfall_after_donation"],
        "rejections_by_reason": dict(sorted(
            (k, v) for k, v in stats["rejections"].items()
            if isinstance(v, int))),
        "per_family": {
            f"l14.{f.family_id}": {
                "quota_goals": quotas[f.family_id],
                "candidates": stats["families"][f.family_id]["candidates"],
                "qualified_goals": sum(
                    1 for c in candidates
                    if c.family is f and c.base_goal_id in shipped_goal_ids),
                "qualified_instances":
                    per_family_qualified_inst[f"l14.{f.family_id}"],
            }
            for f in FAMILIES
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
        log_line=f"{utcnow()} build complete: {stats['qualified']} qualified "
                 f"/ {stats['base_goals']} goals / "
                 f"{stats['unique_families']} families")
    return build_report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="L14 candidate->qualified "
                                             "shard builder")
    ap.add_argument("--pin-root", default=PIN_ROOT_REL)
    ap.add_argument("--out", default="bank/staging_20k/l14")
    ap.add_argument("--target-instances", type=int, default=1500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--status-file", default=None)
    ap.add_argument("--smoke", type=int, default=None,
                    help="build at most N candidates total (smoke run)")
    args = ap.parse_args(argv)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report = build(Path(args.pin_root), out_root, args.target_instances,
                   args.workers, seed=args.seed,
                   status_path=Path(args.status_file)
                   if args.status_file else None,
                   smoke_n=args.smoke)
    print(json.dumps({"qualified": report["qualified"],
                      "base_goals": report["base_goals"],
                      "families": report["families"],
                      "wall_seconds": report["wall_seconds"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
