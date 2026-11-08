"""L05 shard builder: candidates -> native gates -> unified aob-task/5.0 shards.

Gate chain per candidate (cheap -> expensive):
  1. ground (public view + oracle payload shaped)
  2. prove_exists           — differential index-vs-native agreement + solvable
                               + nontrivial gates        (reject: UNSOLVABLE /
                               TRIVIAL / PROFILE_UNSUPPORTED)
  3. certify_semantic_relevance — counterfactual witness (reject: NON_SEMANTIC)
  4. verifier self-check  — verify_terminal(reference) MUST pass and a seeded
                               wrong answer MUST fail
  5. leakage scan         — serialized answer must not appear in public bytes
                               (reject: LEAKAGE); no private_* keys in public
  6. dedup                — instance_id unique; base goals deduplicated

Each shipped record follows the flat ``aob-task/5.0`` contract (plan §1):
top-level lane/family/base-goal/instance/task ids, ``task_regime``,
``complexity``, ``semantic_profile``, ``public{user_goal, allowed_tools,
budget, observation_refs, ...}``, ``oracle_handle``, ``status``, six
``qualification`` flags, ``native_evidence`` with a reproducible re-check
command, and ``source_snapshot_ids``/``source_unit_ids``. Gold answers live only
in ``oracle/<instance>.json``; the record carries hashes/counts.

Output per shard: ``candidates.jsonl`` (all, with verdicts),
``qualified.jsonl``, ``public/<instance>.json``, ``oracle/<instance>.json``,
``manifest.json`` (sha256). Lane aggregate: ``build_report.json`` and
``rejections/rejections.jsonl``. Shards are immutable once written; resume
continues after the last manifest and skips already-shipped instances/goals.

CLI: ``python -m benchmark.plugins.task_family.l05.build --out bank/staging/l05``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from rdkit import __version__ as RDKIT_VERSION

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.task_family.l05.families import BY_ID, FAMILY_TABLE, FamilySpec
from benchmark.plugins.task_family.l05.propose import (
    DATASET_ID, L05TaskFamilyPlugin, PROFILE,
)
from benchmark.plugins.verifier.l05.recheck import answer_sha, recheck_command
from benchmark.plugins.verifier.l05.query_eval import QueryError
from benchmark.plugins.verifier.l05.verifier import RDKitNativeVerifier
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l05_rdkit import RELEASE_ID

LANE = "L05"
SNAPSHOT_ID = f"rdkit-molecules/{RELEASE_ID}"
INDEX_REL = f"external_knowledge/indexes/rdkit-molecules/{RELEASE_ID}"
BANK_REL = "bank/staging/l05"
QUALIFICATION_KEYS = ("native_verified", "solvable", "nontrivial",
                      "semantic_relevance", "leakage_clean", "release_eligible")


def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_sha(files: list[Path], root: Path) -> str:
    import hashlib
    lines = [f"{p.relative_to(root).as_posix()}:{_sha256_file(p)}" for p in sorted(files)]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _index_manifest_sha(index_dir: Path) -> str:
    manifest = Path(index_dir) / "manifest.json"
    return _sha256_file(manifest) if manifest.is_file() else ""


def _evidence_record(proof_ev: dict[str, Any], witness: dict[str, Any],
                     ref_sub: dict[str, Any], pos: dict[str, Any],
                     neg: dict[str, Any], *, index_sha: str,
                     instance_id: str, bank_rel: str = BANK_REL) -> dict[str, Any]:
    reference_sha = answer_sha(ref_sub)
    witness_ev: dict[str, Any] = {
        "verdict": "pass",
        "counterfactual": witness.get("counterfactual"),
        "answer_diff": witness.get("answer_diff"),
    }
    if "mutated_query" in witness:
        witness_ev["mutated_query_sha256"] = canonical_sha(witness["mutated_query"])
    prove_ev: dict[str, Any] = {
        "verdict": "pass",
        "differential_agree": proof_ev.get("differential_agree"),
        "answer_size": proof_ev.get("answer_size"),
        "universe_size": proof_ev.get("universe_size"),
        "reference_answer_sha256": reference_sha,
        "rdkit_version": proof_ev.get("rdkit_version"),
    }
    if "interaction_margins" in proof_ev:
        prove_ev["interaction_margins"] = proof_ev["interaction_margins"]
    if "broken_parses" in proof_ev:
        prove_ev["broken_parses"] = proof_ev["broken_parses"]
        prove_ev["broken_diff"] = proof_ev.get("broken_diff")
    terminal: dict[str, Any] = {
        "verdict": "pass",
        "positive_pass": bool(pos.get("valid")),
        "negative_rejected": not bool(neg.get("valid")),
        "positive": {k: pos.get(k) for k in ("expected", "received") if k in pos},
        "negative": {k: neg.get(k) for k in ("expected", "received") if k in neg},
    }
    if "per_label" in pos:
        terminal["labels"] = sorted(pos["per_label"])
    return {
        "engine": "rdkit",
        "engine_version": RDKIT_VERSION,
        "semantic_profile": PROFILE,
        "index_manifest_sha256": index_sha,
        "prove_exists": prove_ev,
        "semantic_witness": witness_ev,
        "verify_terminal": terminal,
        "commands": [recheck_command(instance_id, index_rel=INDEX_REL,
                                     bank_rel=bank_rel)],
        "verdict": "pass",
    }


def _record(grounded: GroundedTask, spec: FamilySpec, *, proof_ev: dict[str, Any],
            witness: dict[str, Any], ref_sub: dict[str, Any], pos: dict[str, Any],
            neg: dict[str, Any], index_sha: str, seed: int,
            dataset_id: str = DATASET_ID, profile: str = PROFILE,
            bank_rel: str = BANK_REL) -> dict[str, Any]:
    cand = grounded.ast
    public = dict(grounded.public)
    public["observation_refs"] = [
        f"public/{cand.instance_id}.json",
        f"{INDEX_REL}/membership.json",
        f"{INDEX_REL}/hierarchy.json",
    ]
    return {
        "schema_version": "aob-task/5.0",
        "dataset_id": dataset_id,
        "lane_id": cand.lane_id,
        "family_id": cand.family_id,
        "base_goal_id": cand.base_goal_id,
        "instance_id": cand.instance_id,
        "task_id": f"{LANE}-{cand.instance_id}",
        "task_regime": spec.regime,
        "complexity": cand.ast.get("complexity", spec.complexity),
        "semantic_profile": profile,
        "source_snapshot_ids": [SNAPSHOT_ID],
        "source_unit_ids": list(cand.source_unit_ids),
        "origin_mechanism_group": spec.family_id,
        "public": public,
        "oracle_handle": grounded.oracle_ref,
        "status": "QUALIFIED",
        "qualification": {k: True for k in QUALIFICATION_KEYS},
        "native_evidence": _evidence_record(
            proof_ev, witness, ref_sub, pos, neg,
            index_sha=index_sha, instance_id=cand.instance_id, bank_rel=bank_rel),
        "authorship": "deterministic_template",
        "provenance": "source_derived_composition",
        "seed": seed,
    }


def _leakage_scan(public_blob: str, reference: Any, kind: str,
                  golden_smarts: str | None = None) -> bool:
    needles: list[str] = []
    if isinstance(reference, dict) and "alignment" in reference:
        needles += [json.dumps(sorted(v)) for v in reference["alignment"].values()]
    elif isinstance(reference, dict) and "answer_ids" in reference:
        needles.append(json.dumps(sorted(reference["answer_ids"])))
    if kind == "repair" and golden_smarts:
        needles.append(golden_smarts)
    return not any(n and n in public_blob for n in needles)


def build_shards(index_dir: Path, out_root: Path, *, seed: int = 20261027,
                 shard_size: int = 250, resume: bool = True,
                 families: list[str] | None = None,
                 max_base_goals: dict[str, int] | None = None,
                 family_table: tuple[FamilySpec, ...] = FAMILY_TABLE,
                 dataset_id: str = DATASET_ID, profile: str = PROFILE,
                 bank_rel: str = BANK_REL,
                 qualified_target: int | None = None) -> dict[str, Any]:
    t_start = time.time()
    plugin = L05TaskFamilyPlugin(index_dir, seed=seed)
    verifier = RDKitNativeVerifier(index_dir)
    index_sha = _index_manifest_sha(index_dir)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "rejections").mkdir(exist_ok=True)
    stats: dict[str, Any] = {"candidates": 0, "qualified": 0, "rejections": {},
                             "families": {}, "shards": []}
    seen_instances, seen_goals = (_prior_ids(out_root) if resume else (set(), set()))
    shard_rows: list[dict[str, Any]] = []
    shard_no = _next_shard_no(out_root) if resume else 0
    def flush_shard() -> None:
        nonlocal shard_no, shard_rows
        if not shard_rows:
            return
        shard_dir = out_root / f"shard_{shard_no:04d}"
        manifest = _write_shard(shard_dir, shard_rows, seed=seed, profile=profile)
        stats["shards"].append(manifest)
        shard_no += 1
        shard_rows = []

    stopped = False
    for fam in family_table:
        if families and fam.family_id not in families:
            continue
        if stopped:
            break
        fam_stats = {"candidates": 0, "qualified": 0, "base_goals": 0,
                     "rejections": {}}
        constraints = {"max_base_goals": (max_base_goals or {}).get(
            fam.family_id, fam.base_goals)}
        for cand in plugin.propose(None, family=fam.family_id, seed=seed,
                                   constraints=constraints):
            if cand.base_goal_id in seen_goals or cand.instance_id in seen_instances:
                continue
            fam_stats["candidates"] += 1
            stats["candidates"] += 1
            row = _qualify_one(plugin, verifier, cand, fam, seed, index_sha,
                               dataset_id=dataset_id, profile=profile,
                               bank_rel=bank_rel)
            if row["qualified"]:
                seen_instances.add(cand.instance_id)
                fam_stats["qualified"] += 1
                stats["qualified"] += 1
            else:
                code = row["rejection"].split(":", 1)[0]
                fam_stats["rejections"][code] = fam_stats["rejections"].get(code, 0) + 1
                stats["rejections"][code] = stats["rejections"].get(code, 0) + 1
            shard_rows.append(row)
            if len(shard_rows) >= shard_size:
                flush_shard()
            if qualified_target is not None and stats["qualified"] >= qualified_target:
                stopped = True
                break
        stats["families"][fam.family_id] = fam_stats
    flush_shard()
    stats["wall_seconds"] = round(time.time() - t_start, 1)
    stats["qualified_per_hour"] = round(
        stats["qualified"] / max(stats["wall_seconds"] / 3600, 1e-9), 1)
    report = write_build_report(out_root, seed=seed, index_dir=Path(index_dir),
                                wall_seconds=stats["wall_seconds"],
                                dataset_id=dataset_id, profile=profile)
    stats["build_report"] = report
    return stats


def _prior_ids(out_root: Path) -> tuple[set[str], set[str]]:
    instances: set[str] = set()
    goals: set[str] = set()
    for qpath in sorted(Path(out_root).glob("shard_*/qualified.jsonl")):
        for line in qpath.open(encoding="utf-8"):
            record = json.loads(line)
            instances.add(record["instance_id"])
            goals.add(record["base_goal_id"])
    return instances, goals


def _next_shard_no(out_root: Path) -> int:
    existing = sorted(p.name for p in Path(out_root).glob("shard_*")
                      if (p / "manifest.json").exists())
    return (int(existing[-1].split("_")[1]) + 1) if existing else 0


def _qualify_one(plugin: L05TaskFamilyPlugin, verifier: RDKitNativeVerifier,
                 cand: CandidateTaskAST, fam: FamilySpec, seed: int,
                 index_sha: str, *, dataset_id: str = DATASET_ID,
                 profile: str = PROFILE, bank_rel: str = BANK_REL) -> dict[str, Any]:
    row: dict[str, Any] = {"instance_id": cand.instance_id, "family_id": fam.family_id,
                           "base_goal_id": cand.base_goal_id, "qualified": False,
                           "rejection": "", "record": None}
    try:
        grounded = plugin.ground(cand)
        proof = verifier.prove_exists(grounded)
        row["proof"] = proof.evidence
        if not proof.exists:
            row["rejection"] = proof.reason
            return row
        witness = verifier.certify_semantic_relevance(grounded)
        row["witness"] = witness.witness
        if not witness.ok:
            row["rejection"] = "NON_SEMANTIC:no_counterfactual"
            return row
        ref_sub = verifier.reference_submission(cand.ast)
        pos = verifier.verify_terminal(grounded, ref_sub)
        neg = verifier.verify_terminal(grounded, verifier.wrong_submission(cand.ast))
        row["self_check"] = {"positive": pos.valid, "negative_must_fail": not neg.valid}
        if not pos.valid or neg.valid:
            row["rejection"] = "PROFILE_UNSUPPORTED:verifier_selfcheck"
            return row
        public_for_scan = dict(grounded.public)
        golden = None
        kind = cand.ast["kind"]
        if kind == "repair":
            golden = cand.ast["golden_smarts"]
            public_for_scan.pop("broken_smarts", None)
        public_blob = json.dumps(public_for_scan, sort_keys=True)
        if kind == "align":
            ref_map = {k: sorted(v) for k, v in ref_sub["alignment"].items()}
            table = grounded.public["misaligned_table"]
            if any(sorted(table.get(lbl, [])) == ref_map[lbl] for lbl in ref_map):
                row["rejection"] = "LEAKAGE:alignment_row_correct_in_public"
                return row
        elif not _leakage_scan(public_blob, ref_sub, kind, golden):
            row["rejection"] = "LEAKAGE:answer_in_public"
            return row
        if "private_" in public_blob:
            row["rejection"] = "LEAKAGE:private_key_in_public"
            return row
        row["grounded"] = grounded
        row["ref_sub"] = ref_sub
        row["pos"] = pos.detail
        row["neg"] = neg.detail
        row["record"] = _record(
            grounded, fam, proof_ev=proof.evidence, witness=witness.witness,
            ref_sub=ref_sub, pos=pos.detail, neg=neg.detail,
            index_sha=index_sha, seed=seed, dataset_id=dataset_id,
            profile=profile, bank_rel=bank_rel)
        row["qualified"] = True
        return row
    except QueryError as exc:
        row["rejection"] = f"PROFILE_UNSUPPORTED:{exc}"
        return row
    except Exception as exc:  # one bad candidate must not abort the shard run
        row["rejection"] = f"INTERNAL_ERROR:{type(exc).__name__}:{str(exc)[:160]}"
        return row


def _write_shard(shard_dir: Path, rows: list[dict[str, Any]], *, seed: int,
                 profile: str = PROFILE) -> dict[str, Any]:
    shard_dir.mkdir(parents=True, exist_ok=False)
    (shard_dir / "public").mkdir()
    (shard_dir / "oracle").mkdir()
    qualified = 0
    families: set[str] = set()
    with (shard_dir / "candidates.jsonl").open("w", encoding="utf-8") as cand_fh, \
         (shard_dir / "qualified.jsonl").open("w", encoding="utf-8") as qual_fh:
        for row in rows:
            cand_fh.write(json.dumps({
                "instance_id": row["instance_id"], "family_id": row["family_id"],
                "base_goal_id": row["base_goal_id"], "qualified": row["qualified"],
                "rejection": row["rejection"],
            }, sort_keys=True) + "\n")
            if not row["qualified"]:
                continue
            qualified += 1
            families.add(row["family_id"])
            grounded: GroundedTask = row["grounded"]
            record = row["record"]
            oracle_path = shard_dir / "oracle" / f"{row['instance_id']}.json"
            oracle_path.write_text(json.dumps({
                "oracle_handle": grounded.oracle_ref,
                "ast": grounded.ast.ast,
                "reference_submission": row["ref_sub"],
                "proof": row["proof"], "witness": row["witness"],
                "self_check": row["self_check"],
            }, indent=1, sort_keys=True) + "\n")
            (shard_dir / "public" / f"{row['instance_id']}.json").write_text(
                json.dumps({"task_id": record["task_id"], **record["public"]},
                           indent=1, sort_keys=True) + "\n")
            qual_fh.write(json.dumps(record, sort_keys=True) + "\n")
    public_files = sorted((shard_dir / "public").glob("*.json"))
    oracle_files = sorted((shard_dir / "oracle").glob("*.json"))
    files = {}
    for name in ("candidates.jsonl", "qualified.jsonl"):
        p = shard_dir / name
        files[name] = {"sha256": _sha256_file(p), "bytes": p.stat().st_size}
    manifest = {
        "lane": LANE, "shard": shard_dir.name, "seed": seed,
        "source_snapshot_ids": [SNAPSHOT_ID], "semantic_profile": profile,
        "candidates": len(rows), "qualified": qualified,
        "rejected": len(rows) - qualified,
        "base_goals": len({r["base_goal_id"] for r in rows if r["qualified"]}),
        "families": sorted(families),
        "files": files,
        "public_files": len(public_files),
        "oracle_files": len(oracle_files),
        "public_tree_sha256": _tree_sha(public_files, shard_dir / "public"),
        "oracle_tree_sha256": _tree_sha(oracle_files, shard_dir / "oracle"),
    }
    (shard_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def write_build_report(out_root: Path, *, seed: int, index_dir: Path,
                       wall_seconds: float | None = None,
                       dataset_id: str = DATASET_ID, profile: str = PROFILE,
                       quota: dict[str, Any] | None = None) -> dict[str, Any]:
    out_root = Path(out_root)
    shards: list[dict[str, Any]] = []
    fam: dict[str, dict[str, Any]] = {}
    qualified = candidates = 0
    base_goals: set[str] = set()
    families: set[str] = set()
    rejections: dict[str, int] = {}
    rejection_rows: list[dict[str, Any]] = []
    seeds: set[int] = set()
    for mpath in sorted(out_root.glob("shard_*/manifest.json")):
        m = json.loads(mpath.read_text())
        shards.append(m)
        candidates += m["candidates"]
        qualified += m["qualified"]
        seeds.add(m.get("seed"))
        for line in (mpath.parent / "qualified.jsonl").open(encoding="utf-8"):
            record = json.loads(line)
            base_goals.add(record["base_goal_id"])
            families.add(record["family_id"])
            fstat = fam.setdefault(record["family_id"], {
                "qualified": 0, "base_goals": set()})
            fstat["qualified"] += 1
            fstat["base_goals"].add(record["base_goal_id"])
        for line in (mpath.parent / "candidates.jsonl").open(encoding="utf-8"):
            row = json.loads(line)
            fstat = fam.setdefault(row["family_id"], {
                "qualified": 0, "base_goals": set(), "candidates": 0,
                "rejections": {}})
            fstat.setdefault("candidates", 0)
            fstat["candidates"] += 1
            if not row["qualified"]:
                code = (row["rejection"] or "UNKNOWN").split(":", 1)[0]
                rejections[code] = rejections.get(code, 0) + 1
                fstat.setdefault("rejections", {})
                fstat["rejections"][code] = fstat["rejections"].get(code, 0) + 1
                rejection_rows.append({
                    "instance_id": row["instance_id"],
                    "family_id": row["family_id"],
                    "base_goal_id": row["base_goal_id"],
                    "rejection": row["rejection"],
                })
    family_table = {
        fid: {"candidates": st.get("candidates", 0), "qualified": st["qualified"],
              "base_goals": len(st["base_goals"]),
              "rejections": st.get("rejections", {})}
        for fid, st in sorted(fam.items())
    }
    report = {
        "lane": LANE,
        "dataset_id": dataset_id,
        "semantic_profile": profile,
        "source_snapshot_ids": [SNAPSHOT_ID],
        "index_manifest_sha256": _index_manifest_sha(index_dir),
        "native_engine": {"engine": "rdkit", "version": RDKIT_VERSION},
        "quota": quota or {"qualified_instances": 950, "base_goals": 190,
                           "task_families": 16},
        "seeds": sorted(s for s in seeds if s is not None),
        "qualified": qualified,
        "base_goals": len(base_goals),
        "families": len(families),
        "candidates": candidates,
        "rejected_total": sum(rejections.values()),
        "rejections": dict(sorted(rejections.items())),
        "family_table": family_table,
        "shards": [{"shard": m["shard"], "path": str((out_root / m['shard']).resolve()),
                    "candidates": m["candidates"], "qualified": m["qualified"],
                    "rejected": m["rejected"], "base_goals": m["base_goals"],
                    "files": m["files"],
                    "manifest_sha256": _sha256_file(
                        out_root / m["shard"] / "manifest.json")}
                   for m in shards],
        "wall_seconds": wall_seconds,
        "wall_seconds_scope": "latest build_shards invocation only",
    }
    (out_root / "build_report.json").write_text(
        json.dumps(report, indent=1, sort_keys=True) + "\n")
    rejection_path = out_root / "rejections" / "rejections.jsonl"
    rejection_path.parent.mkdir(exist_ok=True)
    with rejection_path.open("w", encoding="utf-8") as fh:
        for row in sorted(rejection_rows, key=lambda r: r["instance_id"]):
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L05 qualified shards")
    ap.add_argument("--index", default=str(
        Path("external_knowledge/indexes/rdkit-molecules") / RELEASE_ID))
    ap.add_argument("--out", default=BANK_REL)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--shard-size", type=int, default=250)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    stats = build_shards(Path(args.index), Path(args.out), seed=args.seed,
                         shard_size=args.shard_size, resume=not args.no_resume)
    print(json.dumps({k: stats[k] for k in ("candidates", "qualified",
                                            "rejections", "wall_seconds")},
                     indent=1, sort_keys=True))
    return 0 if stats["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())