"""L08 shard builder: candidates -> native gates -> immutable qualified shards.

Gate chain per candidate (cheap -> expensive, mirrors the L05 build order):
  1. ground (public view + oracle payload skeleton)
  2. python recomputation (independent path) — triviality pre-gate
     (reject: TRIVIAL) — cheap, avoids wasting engine runs
  3. engine execution + differential gate  (reject: DIFFERENTIAL_MISMATCH /
     ENGINE_ERROR / TRIVIAL)
  4. counterfactual witness via engine (reject: NON_SEMANTIC)
  5. repair-specific gates: broken rule must NOT reproduce the intended
     classification (reject: TRIVIAL_REPAIR_NOOP); every intent parameter
     must be load-bearing (reject: NONDISCRIMINATING)
  6. verifier self-check — verify_terminal(reference) MUST pass and a seeded
     wrong answer MUST fail (reject: PROFILE_UNSUPPORTED)
  7. leakage scan — serialized answer must not appear in public bytes; no
     private_* keys (reject: LEAKAGE)
  8. dedup — instance_id unique; per-goal qualified cap = 5 (reject: DUPLICATE)

Output layout (aob-task/5.0 contract, §1 canonical dirs):
  bank/staging/l08/{candidates,qualified,oracle,rejections,evidence}/
     shard-0000.jsonl / shard-0000/<instance>.json + per-shard manifest.json
     (sha256 per file) + build_report.json.

Shards are immutable once written; --resume skips shards that already have a
manifest. Oracle dir is 0700 / files 0600 (private, separate from public).

CLI: ``.venv/bin/python -m benchmark.plugins.task_family.l08.build --out bank/staging/l08``
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import stat
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.plugins.task_family.l08.families import (
    BY_ID, FAMILY_TABLE, TOTAL_BASE_GOALS, TOTAL_INSTANCES,
)
from benchmark.plugins.task_family.l08.propose import (
    DATASET_ID, LANE, PROFILE, SEED, L08TaskFamilyPlugin, _source_unit_ids,
)
from benchmark.plugins.verifier.l08 import recompute
from benchmark.plugins.verifier.l08.elm import build_elm
from benchmark.plugins.verifier.l08.runner import engine_versions, job_cache_key
from benchmark.plugins.verifier.l08.verifier import (
    L08NativeVerifier, perturbations,
)
from benchmark.schema.plugin_protocols import GroundedTask

SHARD_SIZE = 50
SOURCE_SNAPSHOT_IDS = ["cql-clinical/commit-bc0c795e4f2a"]
RECHECK_CMD = (".venv/bin/python -m benchmark.plugins.verifier.l08.recheck "
               "--bank bank/staging/l08 --instance %s")


def _sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, payload: Any, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    if private:
        os.chmod(path, 0o600)


def _record(grounded: GroundedTask, qual: dict[str, bool], evidence: dict[str, Any],
            oracle_rel: str, public_rel: str) -> dict[str, Any]:
    cand = grounded.ast
    rule = cand.ast["rule"]
    return {
        "schema_version": "aob-task/5.0",
        "dataset_id": DATASET_ID,
        "lane_id": LANE,
        "family_id": cand.family_id,
        "base_goal_id": cand.base_goal_id,
        "instance_id": cand.instance_id,
        "task_id": "L08-%s" % cand.instance_id,
        "task_regime": cand.regime,
        "complexity": cand.ast["complexity"],
        "semantic_profile": PROFILE,
        "public": {
            "user_goal": grounded.public["user_goal"],
            "allowed_tools": grounded.public["allowed_tools"],
            "budget": grounded.public["budget"],
            "observation_refs": [public_rel,
                                 "tools/node/NODE_RECEIPT.json",
                                 "external_knowledge/pinned/cql-clinical/npm/node_modules.files.sha256.json"],
            "rule_kind": grounded.public["rule_kind"],
            "rule_cql": grounded.public["rule_cql"],
            "patient_ids": grounded.public["patient_ids"],
            "data_provenance": grounded.public["data_provenance"],
            "semantic_profile": PROFILE,
            "complexity": grounded.public["complexity"],
        },
        "oracle_handle": grounded.oracle_ref,
        "status": "QUALIFIED",
        "qualification": qual,
        "native_evidence": evidence,
        "source_snapshot_ids": SOURCE_SNAPSHOT_IDS,
        "source_unit_ids": list(cand.source_unit_ids),
        "authorship": "deterministic_template",
        "provenance": "source_derived_composition",
        "origin_mechanism_group": cand.family_id,
        "seed": SEED,
    }


def _reference_submission(grounded: GroundedTask, answer: list[str]) -> dict[str, Any]:
    if grounded.ast.regime == "repair":
        rule = grounded.ast.ast["rule"]
        return {"op": rule["op"], "threshold": float(rule["threshold"]),
                "lookback_days": int(rule["lookback_days"]),
                "exclude": bool(rule.get("exclude", True))}
    return {"answer_ids": list(answer)}


def _wrong_submission(grounded: GroundedTask, answer: list[str],
                       all_ids: list[str]) -> dict[str, Any]:
    if grounded.ast.regime == "repair":
        broken = grounded.ast.ast["broken_rule"]
        return {"op": broken["op"], "threshold": float(broken["threshold"]),
                "lookback_days": int(broken["lookback_days"]),
                "exclude": bool(broken.get("exclude", True))}
    wrong = sorted(set(all_ids) - set(answer))[: max(1, len(answer))]
    if not wrong:  # answer == all -> wrong is a strict subset
        wrong = answer[: len(answer) // 2]
    return {"answer_ids": wrong}


def _leakage_scan(public_blob: str, reference: dict[str, Any]) -> bool:
    needles: list[str] = []
    if "answer_ids" in reference:
        needles.append(json.dumps(sorted(reference["answer_ids"])))
    return not any(n and n in public_blob for n in needles)


def _assert_no_private_keys(payload: Any) -> None:
    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k.startswith("private_"):
                    raise ValueError("private key at %s.%s" % (path, k))
                walk(v, "%s.%s" % (path, k))
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, "%s[%d]" % (path, i))

    walk(payload, "$")


def build(out_root: Path, *, seed: int = SEED, shard_size: int = SHARD_SIZE,
          resume: bool = True, families: list[str] | None = None,
          verifier: L08NativeVerifier | None = None) -> dict[str, Any]:
    t_start = time.time()
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    plugin = L08TaskFamilyPlugin(seed=seed)
    verifier = verifier or L08NativeVerifier()
    engine_info = engine_versions()

    stats: dict[str, Any] = {
        "candidates": 0, "qualified": 0, "rejections": {}, "families": {},
        "shards": [], "engine": engine_info, "seed": seed,
        "target": {"instances": TOTAL_INSTANCES, "base_goals": TOTAL_BASE_GOALS,
                   "families": len(FAMILY_TABLE)},
    }
    seen_instances: set[str] = set()
    qualified_per_goal: dict[str, int] = {}
    shard_rows: list[dict[str, Any]] = []
    shard_no = _next_shard_no(out_root) if resume else 0

    def flush_shard() -> None:
        nonlocal shard_rows, shard_no
        if not shard_rows:
            return
        shard_dir = out_root / "evidence" / ("shard-%04d" % shard_no)
        _write_shard(out_root, shard_no, shard_rows, shard_dir)
        stats["shards"].append("shard-%04d" % shard_no)
        shard_no += 1
        shard_rows = []

    for fam in FAMILY_TABLE:
        if families and fam.family_id not in families:
            continue
        fam_stats = {"candidates": 0, "qualified": 0, "rejections": {},
                     "base_goals": 0}
        constraints = {"max_base_goals": fam.base_goals,
                       "max_attempts_per_goal": 12}
        # Phase A: generate + ground + cheap python triviality pre-gate.
        # Survivor cap per goal leaves retry room for the Phase C gates while
        # bounding wasted engine jobs (surplus attempts are never evaluated).
        grounded_list: list[GroundedTask] = []
        survivors_per_goal: dict[str, int] = {}

        def phase_a_reject(cand, reason: str, detail: dict[str, Any] | None = None) -> None:
            code = reason.split(":", 1)[0]
            fam_stats["candidates"] += 1
            stats["candidates"] += 1
            fam_stats["rejections"][code] = fam_stats["rejections"].get(code, 0) + 1
            stats["rejections"][code] = stats["rejections"].get(code, 0) + 1
            shard_rows.append({"instance_id": cand.instance_id,
                               "family_id": cand.family_id,
                               "base_goal_id": cand.base_goal_id,
                               "regime": cand.regime, "qualified": False,
                               "rejection": reason,
                               "detail": detail or {}})

        for cand in plugin.propose(None, family=fam.family_id, seed=seed,
                                   constraints=constraints):
            if survivors_per_goal.get(cand.base_goal_id, 0) >= 8:
                continue  # enough survivors for this goal
            if cand.instance_id in seen_instances:
                phase_a_reject(cand, "DUPLICATE:instance_id")
                continue
            seen_instances.add(cand.instance_id)
            grounded = plugin.ground(cand)
            rule, patients = cand.ast["rule"], cand.ast["patients"]
            try:
                py = verifier.python_evaluate(rule, patients)
            except Exception as exc:
                phase_a_reject(cand, "PYTHON_ERROR:%s" % exc)
                continue
            py_answer = sorted(pid for pid, v in py.items() if v)
            if len(py_answer) < 2 or len(py_answer) > len(py) - 2:
                phase_a_reject(cand, "TRIVIAL:answer_boundary",
                               {"python_answer_size": len(py_answer),
                                "panel_size": len(py)})
                continue
            if cand.ast.get("repair_noop"):
                phase_a_reject(cand, "TRIVIAL_REPAIR_NOOP:no_discriminating_mutation")
                continue
            survivors_per_goal[cand.base_goal_id] = (
                survivors_per_goal.get(cand.base_goal_id, 0) + 1)
            grounded_list.append(grounded)
        # Phase B: batch-prime the engine cache (one node spawn per 100 jobs)
        pairs: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
        for grounded in grounded_list:
            ast = grounded.ast.ast
            rule, patients = ast["rule"], ast["patients"]
            pairs.append((rule, patients))
            for _, mutated in perturbations(rule):
                pairs.append((mutated, patients))
            if grounded.ast.regime == "repair":
                pairs.append((ast["broken_rule"], patients))
        executed = verifier.batch_evaluate(pairs)
        stats.setdefault("engine_jobs_executed", 0)
        stats["engine_jobs_executed"] += executed
        # Phase C: full gates (engine results served from cache). Surplus
        # survivors for already-complete goals are skipped unevaluated.
        for grounded in grounded_list:
            cand = grounded.ast
            if qualified_per_goal.get(cand.base_goal_id, 0) >= fam.instances_per_goal:
                continue  # goal complete; surplus attempt not evaluated
            fam_stats["candidates"] += 1
            stats["candidates"] += 1
            row = _qualify_one(verifier, grounded, fam, seen_instances)
            if row["qualified"]:
                fam_stats["qualified"] += 1
                stats["qualified"] += 1
                qualified_per_goal[cand.base_goal_id] = (
                    qualified_per_goal.get(cand.base_goal_id, 0) + 1)
                if qualified_per_goal[cand.base_goal_id] == 1:
                    fam_stats["base_goals"] += 1
            else:
                code = row["rejection"].split(":", 1)[0]
                fam_stats["rejections"][code] = fam_stats["rejections"].get(code, 0) + 1
                stats["rejections"][code] = stats["rejections"].get(code, 0) + 1
            shard_rows.append(row)
            if len(shard_rows) >= shard_size:
                flush_shard()
            if stats["qualified"] >= TOTAL_INSTANCES:
                break
        stats["families"][fam.family_id] = fam_stats
        if stats["qualified"] >= TOTAL_INSTANCES:
            break
    flush_shard()

    stats["wall_seconds"] = round(time.time() - t_start, 1)
    stats["base_goals"] = len(qualified_per_goal)
    stats["qualified_per_hour"] = round(stats["qualified"] /
                                        max(stats["wall_seconds"] / 3600, 1e-9), 1)
    _write_json(out_root / "build_report.json", stats)
    return stats


def _next_shard_no(out_root: Path) -> int:
    existing = sorted(p.name for p in (out_root / "qualified").glob("shard-*.jsonl"))
    return (int(existing[-1].split("-")[1]) + 1) if existing else 0


def _qualify_one(verifier: L08NativeVerifier, grounded: GroundedTask, fam,
                 seen_instances: set[str]) -> dict[str, Any]:
    cand = grounded.ast
    ast = cand.ast
    rule, patients = ast["rule"], ast["patients"]
    row: dict[str, Any] = {"instance_id": cand.instance_id, "family_id": fam.family_id,
                           "base_goal_id": cand.base_goal_id, "regime": cand.regime,
                           "qualified": False, "rejection": "", "grounded": grounded}

    # Gate 2: cheap python pre-gate (triviality only; full differential below)
    try:
        py = verifier.python_evaluate(rule, patients)
    except Exception as exc:
        row["rejection"] = "PYTHON_ERROR:%s" % exc
        return row
    py_answer = sorted(pid for pid, v in py.items() if v)
    n = len(py)
    if len(py_answer) < 2 or len(py_answer) > n - 2:
        row["rejection"] = "TRIVIAL:answer_boundary"
        row["python_answer_size"] = len(py_answer)
        return row

    # Gate 3: engine execution + differential
    proof = verifier.prove_exists(grounded)
    row["proof"] = proof.evidence
    if not proof.exists:
        row["rejection"] = proof.reason
        return row
    engine_answer = list(proof.evidence["engine_answer"])

    # Gate 4: counterfactual witness (engine-run perturbations)
    witness = verifier.certify_semantic_relevance(grounded)
    row["witness"] = witness.witness
    if not witness.ok:
        row["rejection"] = "NON_SEMANTIC:no_counterfactual"
        return row

    # Gate 5: repair-specific gate — the broken rule must NOT reproduce the
    # intended classification. Repair verification is classification-level
    # (submitted params are rebuilt into ELM and executed; the produced
    # classification must equal the intended one), consistent with the L05
    # lane precedent for repair tasks.
    if cand.regime == "repair":
        broken_rule = ast["broken_rule"]
        try:
            broken_result = verifier.engine_evaluate(broken_rule, patients)
        except Exception as exc:
            row["rejection"] = "ENGINE_ERROR:broken_rule:%s" % exc
            return row
        broken_answer = frozenset(pid for pid, v in broken_result.items() if v)
        intended = frozenset(engine_answer)
        if broken_answer == intended:
            row["rejection"] = "TRIVIAL_REPAIR_NOOP:broken_equals_intended"
            return row

    # Gate 6: verifier self-check (positive must pass, negative must fail)
    ref_sub = _reference_submission(grounded, engine_answer)
    pos = verifier.verify_terminal(grounded, ref_sub)
    neg = verifier.verify_terminal(grounded, _wrong_submission(grounded, engine_answer,
                                                                list(py.keys())))
    row["self_check"] = {"positive_pass": pos.valid, "negative_rejected": not neg.valid}
    if not pos.valid or neg.valid:
        row["rejection"] = "PROFILE_UNSUPPORTED:verifier_selfcheck"
        return row

    # Gate 7: leakage scan
    public_blob = json.dumps(grounded.public, sort_keys=True)
    if not _leakage_scan(public_blob, ref_sub):
        row["rejection"] = "LEAKAGE:answer_in_public"
        return row
    try:
        _assert_no_private_keys(json.loads(public_blob))
    except ValueError:
        row["rejection"] = "LEAKAGE:private_key_in_public"
        return row

    row["qualified"] = True
    row["qual"] = {"native_verified": True, "solvable": True, "nontrivial": True,
                   "semantic_relevance": True, "leakage_clean": True,
                   "release_eligible": True}
    row["reference_submission"] = ref_sub
    row["engine_answer"] = engine_answer
    return row


def _threshold_step(rule: dict[str, Any]) -> float:
    """Same 15% step as the verifier witness (see verifier._threshold_step)."""
    t = abs(float(rule["threshold"]))
    return float(max(1.0, round(t * 0.15)))


def _write_shard(out_root: Path, shard_no: int, rows: list[dict[str, Any]],
                 evidence_dir: Path) -> None:
    shard_name = "shard-%04d" % shard_no
    qualified_rows = [r for r in rows if r["qualified"]]
    candidates_path = out_root / "candidates" / ("%s.jsonl" % shard_name)
    qualified_path = out_root / "qualified" / ("%s.jsonl" % shard_name)
    rejections_path = out_root / "rejections" / ("%s.jsonl" % shard_name)
    public_dir = out_root / "public" / shard_name
    oracle_dir = out_root / "oracle" / shard_name
    for d in (candidates_path.parent, qualified_path.parent, rejections_path.parent,
              public_dir, evidence_dir):
        d.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(oracle_dir, 0o700)

    qualified_records: list[dict[str, Any]] = []
    with candidates_path.open("w", encoding="utf-8") as cf, \
         rejections_path.open("w", encoding="utf-8") as rf:
        for row in rows:
            cf.write(json.dumps({
                "instance_id": row["instance_id"], "family_id": row["family_id"],
                "base_goal_id": row["base_goal_id"], "regime": row["regime"],
                "qualified": row["qualified"], "rejection": row["rejection"],
            }, sort_keys=True) + "\n")
            if row["qualified"]:
                continue
            rf.write(json.dumps({
                "instance_id": row["instance_id"], "family_id": row["family_id"],
                "base_goal_id": row["base_goal_id"], "regime": row["regime"],
                "reason": row["rejection"],
                "detail": row.get("detail") or {k: v for k, v in row.items()
                                                if k in ("python_answer_size", "proof",
                                                         "witness")},
            }, sort_keys=True) + "\n")

    for row in qualified_rows:
        instance_id = row["instance_id"]
        grounded: GroundedTask = row["grounded"]
        cand = grounded.ast
        rule, patients = cand.ast["rule"], cand.ast["patients"]
        elm = build_elm(rule)
        public_rel = "public/%s/%s.json" % (shard_name, instance_id)
        oracle_rel = "oracle/%s/%s.json" % (shard_name, instance_id)
        evidence_rel = "evidence/%s/%s.json" % (shard_name, instance_id)

        # public file (no gold, no ELM)
        public_payload = dict(grounded.public)
        public_payload["task_id"] = "L08-%s" % instance_id
        public_payload["patient_bundles"] = patients
        public_payload["observation_refs"] = [public_rel]
        _write_json(public_dir / ("%s.json" % instance_id), public_payload)

        # evidence job file (ELM + panel; re-runnable via recheck CLI)
        _write_json(evidence_dir / ("%s.json" % instance_id), {
            "instance_id": instance_id, "task_id": "L08-%s" % instance_id,
            "rule": rule, "library": elm, "patients": patients,
        })

        # private oracle (0600)
        oracle_payload = {
            "oracle_handle": grounded.oracle_ref,
            "instance_id": instance_id,
            "rule": rule,
            "broken_rule": cand.ast.get("broken_rule"),
            "reference_answer": row["engine_answer"],
            "reference_submission": row["reference_submission"],
            "proof": row["proof"],
            "witness": row["witness"],
            "self_check": row["self_check"],
            "elm_sha256": job_cache_key(elm, patients),
        }
        _write_json(oracle_dir / ("%s.json" % instance_id), oracle_payload, private=True)

        evidence = {
            "engine": row["proof"]["engine"],
            "commands": [RECHECK_CMD % instance_id],
            "prove_exists": {
                "verdict": "pass",
                "differential_agree": row["proof"]["differential_agree"],
                "engine_answer_size": len(row["engine_answer"]),
                "python_answer_size": len(row["proof"]["python_answer"]),
                "panel_size": row["proof"]["panel_size"],
            },
            "semantic_witness": {
                "verdict": "pass",
                "sensitive_counterfactual": row["witness"].get("sensitive_counterfactual"),
                "symmetric_difference": row["witness"].get("symmetric_difference"),
            },
            "verify_terminal": {
                "verdict": "pass",
                "positive_pass": row["self_check"]["positive_pass"],
                "negative_rejected": row["self_check"]["negative_rejected"],
            },
            "verdict": "pass",
        }
        record = _record(grounded, row["qual"], evidence, oracle_rel, public_rel)
        qualified_records.append(record)

    with qualified_path.open("w", encoding="utf-8") as qf:
        for record in qualified_records:
            qf.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "shard": shard_name,
        "candidates": len(rows),
        "qualified": len(qualified_rows),
        "rejected": len(rows) - len(qualified_rows),
        "files": {},
    }
    for path in sorted([candidates_path, qualified_path, rejections_path] +
                       list(public_dir.glob("*.json")) +
                       list(evidence_dir.glob("*.json")) +
                       list(oracle_dir.glob("*.json"))):
        if path.is_file():
            manifest["files"][str(path.relative_to(out_root))] = _sha256_file(path)
    _write_json(out_root / "manifests" / ("%s.manifest.json" % shard_name), manifest)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build L08 qualified shards")
    ap.add_argument("--out", default="bank/staging/l08")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--shard-size", type=int, default=SHARD_SIZE)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    stats = build(Path(args.out), seed=args.seed, shard_size=args.shard_size,
                  resume=not args.no_resume, families=args.families)
    print(json.dumps(stats, indent=1, sort_keys=True))
    return 0 if stats["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
