"""L14 task families: 14 substantive families over pinned ChEMBL 37 SQLite.

Every family maps a ``params`` dict to an exact answer set of ChEMBL IDs
computed by a focused SQL query over the pinned DB (fast path) and
independently re-computed by the verifier plugin: (a) a clean-parser path
that loads scoped raw rows with trivial SELECTs and applies the predicate in
pure Python (no join/filter logic in SQL), and (b) for mutation sensitivity,
the same predicate re-run on a perturbed in-memory SQLite slice.

Family axes (audit plan v2 §L14; 12+ required families — coverage map in
REPORT.md):
  org_*   organization    — hierarchy/classification/provenance organization
  app_*   application     — exact-set queries with measurement semantics
  rep_*   repair          — corrupted standardization/qualifier artifact repair

Measurement rules enforced by construction:
  * standard_type / standard_relation / standard_units always participate;
    IC50, Ki, EC50, Kd, Potency are never mixed in one answer set;
  * qualifier families gate on standard_relation='=' exactly;
  * unit family converts only exact-scale units (nM/uM/pM/mM).

Instance state axis: ``bundle_variant`` (rows_jsonl vs sqlite_slice public
evidence; from idx 2 a seeded distractor row set is added).
Base goal identity = sha(family, canonical params); instance identity =
sha(base_goal, state descriptor). Paraphrase-only multiplication is never used.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from benchmark.plugins.source.l14.kb import L14KB

STABLE_SEED = 20261101

#: Instances shipped per base goal (state-axis variation only).
INSTANCES_PER_GOAL = 3

#: Tools the agent may call in every L14 task (contract §1 public.allowed_tools).
ALLOWED_TOOLS = ["inspect_schema", "inspect_state", "read_public_doc",
                 "probe_native", "edit_artifact", "validate_artifact",
                 "check_action_with_ontology", "submit_final"]

PROFILE_RELATIONAL = "relational-sql/1"
PROFILE_MEASUREMENT = "measurement-semantics/1"
PROFILE_REPAIR = "artifact-repair/1"

UNIT_SCALE_TO_NM = {"nM": 1.0, "uM": 1_000.0, "pM": 0.001, "mM": 1_000_000.0}
PUBLISHED_UNIT_SCALE_TO_NM = {
    "nM": 1.0, "nmol/L": 1.0,
    "uM": 1_000.0, "µM": 1_000.0, "um": 1_000.0, "umol/L": 1_000.0,
    "microM": 1_000.0,
    "pM": 0.001, "pmol/L": 0.001,
    "mM": 1_000_000.0, "mmol/L": 1_000_000.0,
    "M": 1_000_000_000.0, "mol/L": 1_000_000_000.0,
}
MEAS_TYPES = ("IC50", "Ki", "EC50", "Kd", "Potency")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def base_goal_id(family_id: str, params: dict[str, Any]) -> str:
    return sha_text(f"l14|{family_id}|{canonical_params(params)}")[:32]


@dataclass(frozen=True)
class Family:
    family_id: str
    regime: str
    profile: str
    complexity: str
    description: str
    state_axis: str
    instances_per_goal: int
    max_base_goals: int
    min_answer: int
    max_answer: int
    propose: Callable[[L14KB, random.Random], Iterator[dict[str, Any]]]
    answer: Callable[[L14KB, dict[str, Any]], tuple[str, ...]]
    goal_text: Callable[[dict[str, Any], L14KB], str]
    witness_kind: str


def state_descriptor(fam: Family, base_seed: int, inst_idx: int) -> dict[str, Any]:
    """Instance identity on the bundle-variant axis (representation change,
    never paraphrase multiplication)."""
    if fam.state_axis == "bundle_variant":
        if inst_idx == 0:
            return {"axis": "bundle_variant", "variant": "rows_jsonl"}
        if inst_idx == 1:
            return {"axis": "bundle_variant", "variant": "sqlite_slice"}
        variant = "rows_jsonl" if inst_idx % 2 == 0 else "sqlite_slice"
        return {"axis": "bundle_variant", "variant": variant,
                "distractor_seed": (base_seed * 7919 + inst_idx * 104729)
                                   & 0xFFFFFFFF}
    return {"axis": fam.state_axis,
            "seed": (base_seed * 131_071 + inst_idx * 1_009) & 0xFFFFFFFF}


def source_unit_ids(params: dict[str, Any]) -> tuple[str, ...]:
    """Source-unit anchors named by the task params (never answer members)."""
    ids: list[str] = []
    for key, kind in (("parent_chembl_id", "chembl_molecule"),
                      ("molecule_chembl_id", "chembl_molecule"),
                      ("target_chembl_id", "chembl_target"),
                      ("assay_chembl_id", "chembl_assay"),
                      ("accession", "chembl_component")):
        if key in params:
            ids.append(f"{kind}/{params[key]}")
    if "protein_class_id" in params:
        ids.append(f"chembl_protein_class/{params['protein_class_id']}")
    if "journal" in params:
        ids.append(f"chembl_document/journal:{params['journal']}")
    if "mesh_heading" in params:
        ids.append(f"chembl_indication/mesh:{params['mesh_heading']}")
    if "standard_type" in params:
        ids.append(f"chembl_activity/standard_type:{params['standard_type']}")
    if "action_type" in params:
        ids.append(f"chembl_mechanism/action_type:{params['action_type']}")
    return tuple(sorted(set(ids)))


# ---------------------------------------------------------------------------
# shared SQL fragments + row loaders (fast path)
# ---------------------------------------------------------------------------

def _one(con, sql: str, args: tuple[Any, ...]):
    row = con.execute(sql, args).fetchone()
    return row[0] if row else None


def _molregno(con, chembl_id: str) -> int | None:
    return _one(con, "SELECT molregno FROM molecule_dictionary "
                     "WHERE chembl_id = ?", (chembl_id,))


def _tid(con, chembl_id: str) -> int | None:
    return _one(con, "SELECT tid FROM target_dictionary WHERE chembl_id = ?",
                (chembl_id,))


def _assay_id(con, chembl_id: str) -> int | None:
    return _one(con, "SELECT assay_id FROM assays WHERE chembl_id = ?",
                (chembl_id,))


def _sorted(rows) -> tuple[str, ...]:
    return tuple(sorted({str(r[0]) for r in rows}))


def _pick_threshold(con, sql_values: str, args: tuple[Any, ...],
                    lo: int, hi: int, rng: random.Random) -> float | None:
    """Pick a value threshold so the count of distinct molecules at or below
    it lands in [lo, hi]. Deterministic given the value distribution; the rng
    only chooses between equally-valid decile candidates."""
    values = sorted(float(r[0]) for r in con.execute(sql_values, args))
    if len(values) < lo:
        return None
    candidates: list[float] = []
    for q in (0.15, 0.25, 0.4, 0.55, 0.6, 0.75, 0.9):
        thr = values[min(len(values) - 1, int(q * len(values)))]
        n = sum(1 for v in values if v <= thr)
        if lo <= n <= hi:
            candidates.append(thr)
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]


# ---------------------------------------------------------------------------
# family definitions
# ---------------------------------------------------------------------------

def _mk_org_parent_salt_members() -> Family:
    fid = "org_parent_salt_members"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = list(kb.pools()["parents_with_children"])
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM molecule_dictionary "
                                "WHERE molregno = ?", (row["parent_molregno"],))
                if cid:
                    yield {"parent_chembl_id": cid}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT md.chembl_id FROM molecule_hierarchy h "
                "JOIN molecule_dictionary md ON md.molregno = h.molregno "
                "WHERE h.parent_molregno = (SELECT molregno FROM "
                "  molecule_dictionary WHERE chembl_id = ?) "
                "AND md.chembl_id != ?", (params["parent_chembl_id"],
                                          params["parent_chembl_id"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Assemble the ChEMBL molecule-hierarchy membership and report "
                "the exact, complete, sorted set of ChEMBL molecule IDs whose "
                "registered parent molecule is {parent_chembl_id} (salt and "
                "alternate-form members; exclude the parent itself). Identity "
                "must come from the molecule_hierarchy parent links, never "
                "from names."
                ).format(**params)

    return Family(fid, "organization", PROFILE_RELATIONAL, "low",
                  "Molecule identity: parent/salt membership via "
                  "molecule_hierarchy", "bundle_variant", INSTANCES_PER_GOAL,
                  400, 3, 60, propose, answer, goal_text,
                  "swap_parent_link")


def _mk_app_molecule_assays_of_type() -> Family:
    fid = "app_molecule_assays_of_type"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["molecule_assay_type"]
                if 6 <= r["n_assays"] <= 60]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM molecule_dictionary "
                                "WHERE molregno = ?", (row["molregno"],))
                if cid:
                    yield {"molecule_chembl_id": cid,
                           "standard_type": row["standard_type"]}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT a.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "WHERE act.molregno = (SELECT molregno FROM "
                "  molecule_dictionary WHERE chembl_id = ?) "
                "AND act.standard_type = ? AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value IS NOT NULL",
                (params["molecule_chembl_id"], params["standard_type"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Reconstruct the molecule-assay-activity linkage and report "
                "the exact, complete, sorted set of ChEMBL assay IDs in which "
                "molecule {molecule_chembl_id} has a '{standard_type}' "
                "activity measured with standard_relation '=' in standard "
                "units nM. Other measurement types (IC50/Ki/EC50/Kd/Potency) "
                "and other qualifiers (<, >, <=, >=) must not be mixed in."
                ).format(**params)

    return Family(fid, "application", PROFILE_MEASUREMENT, "medium",
                  "Molecule-Assay-Activity relation reconstruction with "
                  "measurement semantics", "bundle_variant",
                  INSTANCES_PER_GOAL, 400, 6, 60, propose, answer, goal_text,
                  "reassign_assay_rows")


def _assay_count(con, tid: int) -> int:
    return _one(con, "SELECT count(*) FROM assays WHERE tid = ?",
                (tid,)) or 0


def _mk_app_target_binding_set() -> Family:
    fid = "app_target_binding_set"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["target_measurement"]
                if 30 <= r["n_molecules"] <= 20000]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM target_dictionary "
                                "WHERE tid = ?", (row["tid"],))
                if not cid:
                    continue
                # bundle-size guard: cap total assays on the target so the
                # public evidence slice stays agent-consumable
                if not (8 <= _assay_count(con, row["tid"]) <= 1200):
                    continue
                thr = _pick_threshold(
                    con,
                    "SELECT DISTINCT act.standard_value FROM activities act "
                    "JOIN assays a ON a.assay_id = act.assay_id "
                    "WHERE a.tid = ? AND a.assay_type = 'B' "
                    "AND a.confidence_score >= 8 AND act.standard_type = ? "
                    "AND act.standard_relation = '=' "
                    "AND act.standard_units = 'nM' "
                    "AND act.standard_value IS NOT NULL "
                    "AND act.molregno IS NOT NULL",
                    (row["tid"], row["standard_type"]), 8, 40, rng)
                if thr is not None:
                    yield {"target_chembl_id": cid,
                           "standard_type": row["standard_type"],
                           "threshold": thr}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN target_dictionary td ON td.tid = a.tid "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE td.chembl_id = ? AND a.assay_type = 'B' "
                "AND a.confidence_score >= 8 AND act.standard_type = ? "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value <= ?",
                (params["target_chembl_id"], params["standard_type"],
                 params["threshold"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Report the exact, complete, sorted set of ChEMBL molecule IDs "
                "with a '{standard_type}' binding measurement against target "
                "{target_chembl_id} at or below {threshold} nM, counting only "
                "binding assays (assay_type 'B') with confidence_score >= 8, "
                "standard_relation '=' and standard_units 'nM'. Do not mix "
                "measurement types or qualifiers."
                ).format(**params)

    return Family(fid, "application", PROFILE_MEASUREMENT, "medium",
                  "Activity-Target binding exact set with threshold",
                  "bundle_variant", INSTANCES_PER_GOAL, 500, 8, 40,
                  propose, answer, goal_text, "rescale_value")


def _mk_org_assay_context_organism() -> Family:
    fid = "org_assay_context_organism"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["target_organism_assays"]
                if 8 <= r["n_assays"] <= 300]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM target_dictionary "
                                "WHERE tid = ?", (row["tid"],))
                if not cid:
                    continue
                if _assay_count(con, row["tid"]) > 3000:
                    continue
                yield {"target_chembl_id": cid,
                       "assay_organism": row["assay_organism"]}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT a.chembl_id FROM assays a "
                "JOIN target_dictionary td ON td.tid = a.tid "
                "WHERE td.chembl_id = ? AND a.assay_type = 'B' "
                "AND a.assay_organism = ?",
                (params["target_chembl_id"], params["assay_organism"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Organize the assay context and report the exact, complete, "
                "sorted set of ChEMBL assay IDs of binding assays "
                "(assay_type 'B') run on target {target_chembl_id} in "
                "organism '{assay_organism}'. Assay organism is part of the "
                "assay semantics: assays of the same target in other "
                "organisms must be excluded."
                ).format(**params)

    return Family(fid, "organization", PROFILE_RELATIONAL, "low",
                  "Assay context organization (type x organism x target)",
                  "bundle_variant", INSTANCES_PER_GOAL, 400, 8, 80,
                  propose, answer, goal_text, "reassign_target")


def _mk_app_measurement_exact_set() -> Family:
    fid = "app_measurement_exact_set"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["assay_measurement"]
                if 15 <= r["n_molecules"] <= 150]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM assays "
                                "WHERE assay_id = ?", (row["assay_id"],))
                if cid:
                    yield {"assay_chembl_id": cid,
                           "standard_type": row["standard_type"]}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE a.chembl_id = ? AND act.standard_type = ? "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value IS NOT NULL",
                (params["assay_chembl_id"], params["standard_type"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Report the exact, complete, sorted set of ChEMBL molecule IDs "
                "that have a '{standard_type}' measurement in assay "
                "{assay_chembl_id} with standard_relation '=' and standard "
                "units nM. Rows with other relations ('<', '>', '<=', '>=') "
                "or other measurement types are different facts and must be "
                "excluded."
                ).format(**params)

    return Family(fid, "application", PROFILE_MEASUREMENT, "low",
                  "Measurement semantics exact set within one assay",
                  "bundle_variant", INSTANCES_PER_GOAL, 500, 15, 150,
                  propose, answer, goal_text, "flip_relation")


def _mk_app_unit_normalized_potency() -> Family:
    """Unit normalization: the bundle ships PUBLISHED (value, units) rows only
    (standard_value/standard_units redacted for this family); the agent must
    apply the exact molar scale rules to recover nM potencies. Oracle truth is
    the pinned standardized columns; a sampling-time gate requires the
    published-conversion answer to equal the standardized answer exactly."""
    fid = "app_unit_normalized_potency"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = list(kb.pools()["assay_pubunits_mix"])
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM assays "
                                "WHERE assay_id = ?", (row["assay_id"],))
                if not cid:
                    continue
                thr = _pick_threshold(
                    con,
                    "SELECT act.standard_value FROM activities act "
                    "WHERE act.assay_id = ? AND act.standard_type = ? "
                    "AND act.standard_relation = '=' "
                    "AND act.standard_units = 'nM' "
                    "AND act.standard_value IS NOT NULL",
                    (row["assay_id"], row["standard_type"]), 6, 40, rng)
                if thr is None:
                    continue
                if not _conversion_consistent(con, row["assay_id"],
                                              row["standard_type"], thr):
                    continue
                yield {"assay_chembl_id": cid,
                       "standard_type": row["standard_type"],
                       "threshold": thr}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE a.chembl_id = ? AND act.standard_type = ? "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value <= ?",
                (params["assay_chembl_id"], params["standard_type"],
                 params["threshold"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("The evidence bundle for assay {assay_chembl_id} carries the "
                "PUBLISHED measurement rows (published type, relation, value "
                "and units in mixed molar units such as uM, nM, pM, mM). "
                "Normalize every value to nM with the exact scale rules "
                "(1 uM = 1000 nM; 1 nM = 1000 pM; 1 mM = 1,000,000 nM and "
                "the equivalent mol/L spellings), keep only exact-equality "
                "('{standard_type}', relation '=') rows, and report the "
                "exact, complete, sorted set of ChEMBL molecule IDs at or "
                "below {threshold} nM."
                ).format(**params)

    return Family(fid, "application", PROFILE_MEASUREMENT, "high",
                  "Unit normalization from published units to nM "
                  "(exact-scale molar units only)", "bundle_variant",
                  INSTANCES_PER_GOAL, 300, 6, 40, propose, answer, goal_text,
                  "flip_relation")


def _conversion_consistent(con, assay_id: int, standard_type: str,
                           thr: float) -> bool:
    """Gate: the published->nM conversion answer must equal the standardized
    answer for this (assay, type, threshold)."""
    std = {r[0] for r in con.execute(
        "SELECT DISTINCT act.molregno FROM activities act "
        "WHERE act.assay_id = ? AND act.standard_type = ? "
        "AND act.standard_relation = '=' AND act.standard_units = 'nM' "
        "AND act.standard_value <= ?", (assay_id, standard_type, thr))}
    pub = set()
    for mol, val, units, rel in con.execute(
            "SELECT act.molregno, act.value, act.units, act.relation "
            "FROM activities act WHERE act.assay_id = ? "
            "AND act.standard_type = ? AND act.value IS NOT NULL "
            "AND act.units IS NOT NULL", (assay_id, standard_type)):
        f = PUBLISHED_UNIT_SCALE_TO_NM.get(units)
        if f is None or rel_ok(rel) is False:
            continue
        if rel_ok(rel) and float(val) * f <= thr:
            pub.add(mol)
    return std == pub


def rel_ok(rel) -> bool:
    return rel == "="


def _mk_rep_relation_flip_repair() -> Family:
    fid = "rep_relation_flip_repair"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["assay_measurement"]
                if 15 <= r["n_molecules"] <= 120]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM assays "
                                "WHERE assay_id = ?", (row["assay_id"],))
                if not cid:
                    continue
                mixed = con.execute(
                    "SELECT count(*) FROM activities act JOIN assays a "
                    "ON a.assay_id = act.assay_id WHERE a.chembl_id = ? "
                    "AND act.standard_type = ? AND act.standard_units = 'nM' "
                    "AND act.standard_relation IN ('<','>') "
                    "AND act.standard_value IS NOT NULL",
                    (cid, row["standard_type"])).fetchone()[0]
                if not (4 <= mixed <= 40):
                    continue
                thr = _pick_threshold(
                    con,
                    "SELECT act.standard_value FROM activities act "
                    "JOIN assays a ON a.assay_id = act.assay_id "
                    "WHERE a.chembl_id = ? AND act.standard_type = ? "
                    "AND act.standard_relation = '=' "
                    "AND act.standard_units = 'nM' "
                    "AND act.standard_value IS NOT NULL",
                    (cid, row["standard_type"]), 6, 30, rng)
                if thr is not None:
                    yield {"assay_chembl_id": cid,
                           "standard_type": row["standard_type"],
                           "threshold": thr}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE a.chembl_id = ? AND act.standard_type = ? "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value <= ?",
                (params["assay_chembl_id"], params["standard_type"],
                 params["threshold"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("The standardization artifact for assay {assay_chembl_id} "
                "('{standard_type}', nM) has had some measurement qualifiers "
                "rewritten: rows whose true standard_relation is '<' or '>' "
                "were flattened to '='. Using the published (type, relation, "
                "value, units) evidence rows, repair the artifact so every "
                "row carries its true standard_relation, then report the "
                "exact, complete, sorted set of ChEMBL molecule IDs whose "
                "repaired relation is '=' with value at or below {threshold} "
                "nM."
                ).format(**params)

    return Family(fid, "repair", PROFILE_REPAIR, "high",
                  "Measurement qualifier repair ('<','>' must not read as "
                  "'=')", "bundle_variant", INSTANCES_PER_GOAL, 300, 6, 30,
                  propose, answer, goal_text, "flip_relation")


def _mk_rep_standardization_table() -> Family:
    fid = "rep_standardization_table"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["assay_measurement"]
                if 15 <= r["n_molecules"] <= 120
                and r["standard_type"] in ("IC50", "Ki", "EC50")]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM assays "
                                "WHERE assay_id = ?", (row["assay_id"],))
                if not cid:
                    continue
                variants = con.execute(
                    "SELECT count(DISTINCT act.type || '|' || "
                    "COALESCE(act.units,'')) FROM activities act "
                    "JOIN assays a ON a.assay_id = act.assay_id "
                    "WHERE a.chembl_id = ? AND act.standard_type = ? "
                    "AND act.type IS NOT NULL",
                    (cid, row["standard_type"])).fetchone()[0]
                if variants < 2:
                    continue
                thr = _pick_threshold(
                    con,
                    "SELECT act.standard_value FROM activities act "
                    "JOIN assays a ON a.assay_id = act.assay_id "
                    "WHERE a.chembl_id = ? AND act.standard_type = ? "
                    "AND act.standard_relation = '=' "
                    "AND act.standard_units = 'nM' "
                    "AND act.standard_value IS NOT NULL",
                    (cid, row["standard_type"]), 6, 30, rng)
                if thr is not None:
                    yield {"assay_chembl_id": cid,
                           "standard_type": row["standard_type"],
                           "threshold": thr}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE a.chembl_id = ? AND act.standard_type = ? "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value <= ?",
                (params["assay_chembl_id"], params["standard_type"],
                 params["threshold"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("The standardization table for assay {assay_chembl_id} maps "
                "published (type, relation, value, units) rows to standard "
                "form, but a seeded corruption rewrote some standard_type "
                "labels, dropped some qualifiers, and mis-scaled some unit "
                "conversions. Repair the table so that standard_type, "
                "standard_relation, standard_value and standard_units match "
                "the published evidence and the exact unit scale rules, then "
                "report the exact, complete, sorted set of ChEMBL molecule "
                "IDs with repaired '{standard_type}' '=' values at or below "
                "{threshold} nM."
                ).format(**params)

    return Family(fid, "repair", PROFILE_REPAIR, "high",
                  "Standardization-table repair (type/qualifier/unit)",
                  "bundle_variant", INSTANCES_PER_GOAL, 300, 6, 30,
                  propose, answer, goal_text, "rescale_units")


def _mk_org_target_class_members() -> Family:
    fid = "org_target_class_members"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["protein_class_targets"]
                if 6 <= r["n_targets"] <= 120]
        rng.shuffle(pool)
        for row in pool:
            yield {"protein_class_id": row["protein_class_id"],
                   "class_name": row["pref_name"],
                   "class_level": row["class_level"]}

    _subtree_cte = (
        "WITH RECURSIVE sub(id) AS ("
        "  SELECT ? "
        "  UNION ALL "
        "  SELECT pc.protein_class_id FROM protein_classification pc "
        "  JOIN sub s ON pc.parent_id = s.id"
        ") ")

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                _subtree_cte +
                "SELECT DISTINCT td.chembl_id FROM target_dictionary td "
                "JOIN target_components tc ON tc.tid = td.tid "
                "JOIN component_class cc ON cc.component_id = tc.component_id "
                "WHERE td.target_type = 'SINGLE PROTEIN' "
                "AND td.organism = 'Homo sapiens' "
                "AND cc.protein_class_id IN (SELECT id FROM sub)",
                (params["protein_class_id"],)))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Using the ChEMBL protein-family classification, report the "
                "exact, complete, sorted set of ChEMBL target IDs of human "
                "SINGLE PROTEIN targets classified at or below class "
                "'{class_name}' (class id {protein_class_id}, level "
                "{class_level}). Membership includes the whole subclass "
                "subtree; targets of other types or organisms are excluded."
                ).format(**params)

    return Family(fid, "organization", PROFILE_RELATIONAL, "high",
                  "Target classification organization (protein family "
                  "subtree)", "bundle_variant", INSTANCES_PER_GOAL, 300,
                  6, 120, propose, answer, goal_text, "relabel_class")


def _mk_org_journal_year_provenance() -> Family:
    fid = "org_journal_year_provenance"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["journal_year_activity"]
                if 25 <= r["n_molecules"] <= 150]
        rng.shuffle(pool)
        for row in pool:
            yield {"journal": row["journal"], "year": row["year"]}

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN docs d ON d.doc_id = a.doc_id "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE d.journal = ? AND d.year = ? "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_type IN ('IC50','Ki','EC50') "
                "AND act.standard_value IS NOT NULL",
                (params["journal"], params["year"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Reconstruct the document provenance chain "
                "(activity -> assay -> document) and report the exact, "
                "complete, sorted set of ChEMBL molecule IDs whose IC50, Ki "
                "or EC50 activities (standard_relation '=', nM) were reported "
                "in assays documented in journal '{journal}' in year {year}. "
                "The document link is part of the fact: measurements from "
                "other venues or years must be excluded."
                ).format(**params)

    return Family(fid, "organization", PROFILE_RELATIONAL, "medium",
                  "Document provenance chain (activity-assay-document)",
                  "bundle_variant", INSTANCES_PER_GOAL, 400, 25, 150,
                  propose, answer, goal_text, "relink_document")


def _mk_app_mechanism_action_type() -> Family:
    fid = "app_mechanism_action_type"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["mechanism_targets"]
                if 4 <= r["n_parents"] <= 100]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM target_dictionary "
                                "WHERE tid = ?", (row["tid"],))
                if cid:
                    yield {"target_chembl_id": cid,
                           "action_type": row["action_type"]}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT pmd.chembl_id FROM drug_mechanism m "
                "JOIN target_dictionary td ON td.tid = m.tid "
                "JOIN molecule_hierarchy h ON h.molregno = m.molregno "
                "JOIN molecule_dictionary pmd "
                "  ON pmd.molregno = h.parent_molregno "
                "WHERE td.chembl_id = ? AND m.action_type = ?",
                (params["target_chembl_id"], params["action_type"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Report the exact, complete, sorted set of ChEMBL parent "
                "molecule IDs whose drug_mechanism records map them to "
                "target {target_chembl_id} with action_type '{action_type}'. "
                "Report at parent-molecule level (salt forms collapse to "
                "their registered parent). Other action types on the same "
                "target are different facts and must be excluded."
                ).format(**params)

    return Family(fid, "application", PROFILE_RELATIONAL, "medium",
                  "Drug-mechanism of action mapping at parent level",
                  "bundle_variant", INSTANCES_PER_GOAL, 300, 4, 100,
                  propose, answer, goal_text, "retype_action")


def _mk_app_indication_mesh() -> Family:
    fid = "app_indication_mesh"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["indication_mesh"]
                if 4 <= r["n_parents"] <= 100]
        rng.shuffle(pool)
        for row in pool:
            yield {"mesh_heading": row["mesh_heading"]}

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT pmd.chembl_id FROM drug_indication d "
                "JOIN molecule_hierarchy h ON h.molregno = d.molregno "
                "JOIN molecule_dictionary pmd "
                "  ON pmd.molregno = h.parent_molregno "
                "WHERE d.mesh_heading = ? AND d.max_phase_for_ind >= 2",
                (params["mesh_heading"],)))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Report the exact, complete, sorted set of ChEMBL parent "
                "molecule IDs that carry a drug_indication record with MeSH "
                "heading '{mesh_heading}' at max_phase_for_ind >= 2. Report "
                "at parent-molecule level. An indication association is not "
                "a proven causal claim; only the recorded indication rows "
                "with the requested phase bound count."
                ).format(**params)

    return Family(fid, "application", PROFILE_RELATIONAL, "medium",
                  "Drug-indication relation with phase bound",
                  "bundle_variant", INSTANCES_PER_GOAL, 300, 4, 100,
                  propose, answer, goal_text, "rekey_indication")


def _mk_org_uniprot_target_map() -> Family:
    fid = "org_uniprot_target_map"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = list(kb.pools()["uniprot_targets"])
        rng.shuffle(pool)
        for row in pool:
            yield {"accession": row["accession"]}

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT td.chembl_id FROM component_sequences cs "
                "JOIN target_components tc "
                "  ON tc.component_id = cs.component_id "
                "JOIN target_dictionary td ON td.tid = tc.tid "
                "WHERE cs.accession = ? "
                "AND td.target_type = 'SINGLE PROTEIN' "
                "AND td.organism = 'Homo sapiens'",
                (params["accession"],)))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Resolve the cross-resource mapping and report the exact, "
                "complete, sorted set of ChEMBL target IDs of human SINGLE "
                "PROTEIN targets that have a target component with UniProt "
                "accession {accession}. Identity must come from the "
                "component_sequences accession link, never from names."
                ).format(**params)

    return Family(fid, "organization", PROFILE_RELATIONAL, "low",
                  "Cross-resource mapping: UniProt accession -> ChEMBL "
                  "target", "bundle_variant", INSTANCES_PER_GOAL, 500,
                  1, 6, propose, answer, goal_text, "rewire_component")


def _mk_app_multihop_target_journal() -> Family:
    fid = "app_multihop_target_journal"

    def propose(kb: L14KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool = [r for r in kb.pools()["target_journal_multihop"]
                if 8 <= r["n_molecules"] <= 150]
        rng.shuffle(pool)
        con = kb.connection()
        try:
            for row in pool:
                cid = _one(con, "SELECT chembl_id FROM target_dictionary "
                                "WHERE tid = ?", (row["tid"],))
                if not cid:
                    continue
                if _assay_count(con, row["tid"]) > 2000:
                    continue
                yield {"target_chembl_id": cid, "journal": row["journal"]}
        finally:
            con.close()

    def answer(kb: L14KB, params: dict[str, Any]) -> tuple[str, ...]:
        con = kb.connection()
        try:
            return _sorted(con.execute(
                "SELECT DISTINCT md.chembl_id FROM activities act "
                "JOIN assays a ON a.assay_id = act.assay_id "
                "JOIN docs d ON d.doc_id = a.doc_id "
                "JOIN target_dictionary td ON td.tid = a.tid "
                "JOIN molecule_dictionary md ON md.molregno = act.molregno "
                "WHERE td.chembl_id = ? AND d.journal = ? "
                "AND a.assay_type = 'B' AND a.confidence_score >= 8 "
                "AND act.standard_type = 'IC50' "
                "AND act.standard_relation = '=' "
                "AND act.standard_units = 'nM' "
                "AND act.standard_value IS NOT NULL",
                (params["target_chembl_id"], params["journal"])))
        finally:
            con.close()

    def goal_text(params: dict[str, Any], kb: L14KB) -> str:
        return ("Follow the molecule-activity-assay-target-document chain and "
                "report the exact, complete, sorted set of ChEMBL molecule "
                "IDs that have an IC50 measurement (standard_relation '=', "
                "nM) in a binding assay (assay_type 'B', confidence_score "
                ">= 8) on target {target_chembl_id} documented in journal "
                "'{journal}'. Every hop constrains the answer; dropping the "
                "journal or the target constraint is a different query."
                ).format(**params)

    return Family(fid, "application", PROFILE_MEASUREMENT, "high",
                  "Multi-hop competency query over molecule-target-assay-"
                  "document", "bundle_variant", INSTANCES_PER_GOAL, 400,
                  8, 150, propose, answer, goal_text, "relink_document")


FAMILIES: tuple[Family, ...] = (
    _mk_org_parent_salt_members(),
    _mk_app_molecule_assays_of_type(),
    _mk_app_target_binding_set(),
    _mk_org_assay_context_organism(),
    _mk_app_measurement_exact_set(),
    _mk_app_unit_normalized_potency(),
    _mk_rep_relation_flip_repair(),
    _mk_rep_standardization_table(),
    _mk_org_target_class_members(),
    _mk_org_journal_year_provenance(),
    _mk_app_mechanism_action_type(),
    _mk_app_indication_mesh(),
    _mk_org_uniprot_target_map(),
    _mk_app_multihop_target_journal(),
)

FAMILY_BY_ID: dict[str, Family] = {f.family_id: f for f in FAMILIES}
