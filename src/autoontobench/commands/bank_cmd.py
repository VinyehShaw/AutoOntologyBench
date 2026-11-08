"""``python -m autoontobench bank`` — build/preflight/status/audit/package +
BYOK generate/validate (S10). Handlers stay thin: queue/cache/shard/registry
do the work. Exit codes: 0 ok, 1 failed checks, 2 usage/config error.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from autoontobench.factory import cli_support as cs
from autoontobench.factory.build import (
    StageFailure, StageResult, TelemetryWriter, collect_metrics, run_worker)
from autoontobench.factory.cache import ContentAddressedCache
from autoontobench.factory.queue import STAGES, BuildQueue
from autoontobench.factory.registry import (
    audit_plugins, discover_plugins, load_lock, load_plugin_file, load_sources,
    plugin_covers_lane)
from autoontobench.factory.shard import ShardStore

REPO = cs.REPO_ROOT
QUEUE_PATH = REPO / "bank" / "queue.db"
GEN_VERSION = "aob-factory/0.1.0"


def _load_build_config(args: Namespace) -> cs.LoadedConfig:
    return cs.load_yaml_strict(cs.resolve_repo_path(args.config), cs.BANK_BUILD_CONFIG_SPEC)


def cmd_preflight(args: Namespace) -> int:
    try:
        cfg = _load_build_config(args)
    except cs.ConfigError as exc:
        cs.eprint(f"[bank preflight] {exc}")
        return 2
    checks: list[tuple[str, bool, str]] = []
    data = cfg.data
    quotas = cs.resolve_repo_path(data["target"]["quotas_file"])
    checks.append(("config parses", True, str(cfg.path)))
    checks.append(("quotas file exists", quotas.is_file(), str(quotas)))
    if quotas.is_file():
        import yaml

        lanes = yaml.safe_load(quotas.read_text(encoding="utf-8"))
        total = sum(l["qualified_instances"] for l in lanes["lanes"])
        checks.append(("lane quotas sum to target", total == data["target"]["qualified_instances"],
                       f"{total}/{data['target']['qualified_instances']}"))
    lock_path = cs.resolve_repo_path(data["source_lock_file"])
    try:
        lock = load_lock(lock_path)
        checks.append(("source lock valid", True, f"{len(lock.entries)} pinned"))
    except cs.ConfigError as exc:
        checks.append(("source lock valid", False, str(exc)))
    try:
        with BuildQueue(QUEUE_PATH) as q:
            mode = q._db.execute("PRAGMA journal_mode").fetchone()[0]
        checks.append(("queue writable (WAL)", mode == "wal", str(QUEUE_PATH)))
    except Exception as exc:
        checks.append(("queue writable (WAL)", False, str(exc)))
    for tool in ("opa", "gator"):
        p = REPO / "tools" / "bin" / tool
        checks.append((f"tool {tool}", p.is_file(), str(p)))
    for d in ("external_knowledge/objects", "external_knowledge/pinned", "bank/staging"):
        path = REPO / d
        path.mkdir(parents=True, exist_ok=True)
        checks.append((f"dir writable {d}", os.access(path, os.W_OK), str(path)))
    audit = audit_plugins(REPO / "benchmark" / "plugins")
    checks.append(("lane plugins discovered", True,
                   ", ".join(f"{k}={len(v)} ({','.join(sorted(v)) or '-'})"
                             for k, v in audit.plugins.items())))
    failed = 0
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<28} {detail}")
        failed += 0 if ok else 1
    for issue in audit.issues:
        print(f"WARN  plugin {issue.kind}/{issue.lane}: {issue.reason} [{issue.path}]")
    print(f"preflight: {len(checks) - failed}/{len(checks)} checks passed, "
          f"{len(audit.issues)} plugin issue(s)")
    return 1 if failed else 0


def cmd_build(args: Namespace) -> int:
    cfg = _load_build_config(args)
    data = cfg.data
    build_id = data["bank_id"]
    lock = load_lock(cs.resolve_repo_path(data["source_lock_file"]))
    if getattr(args, "lane", None):
        return _build_lane(args, cfg, lock)
    sources = load_sources(REPO / "source_registry" / "sources.yaml")
    plugins = discover_plugins(REPO / "benchmark" / "plugins")
    telemetry = TelemetryWriter(cs.resolve_repo_path(data["outputs"]["telemetry"]),
                                cs.resolve_repo_path(data["outputs"]["failures"]))
    quota = int(data["target"]["qualified_instances"])
    if args.dry_run:
        for sid, spec in sorted(sources.items()):
            state = "pinned" if sid in lock.entries else "UNPINNED"
            fam = ("family-plugin" if plugin_covers_lane(plugins["task_family"], spec.lane)
                   else "no-family-plugin")
            print(f"DRY-RUN lane {spec.lane} source {sid}: {state}, {fam}")
        print(f"DRY-RUN build {build_id}: {len(sources)} lanes, quota {quota}")
        return 0
    stop = threading.Event()
    with BuildQueue(QUEUE_PATH) as queue:
        queue.requeue_stale()
        lock_hash = lock.lock_hash or "unpinned"
        for sid, spec in sorted(sources.items()):
            has_family = plugin_covers_lane(plugins["task_family"], spec.lane)
            item_id, created = queue.enqueue(
                source_lock_hash=lock_hash, lane=spec.lane, family=f"{spec.lane}-bootstrap",
                base_goal="bootstrap", seed=int(data["build"]["stable_seed"]),
                generator_version=GEN_VERSION)
            if not created:
                continue
            reason = None
            if not has_family:
                reason = "PLUGIN_MISSING"
            elif sid not in lock.entries:
                reason = "SOURCE_UNPINNED"
            if reason is not None:
                claimed = queue.claim_id(item_id, "build-bootstrap")
                if claimed:
                    queue.fail(claimed.id, error=f"lane {spec.lane}: {reason.lower()}",
                               rejection_reason=reason)
        telemetry.tick(queue, build_id=build_id, quota=quota)
        print(f"[bank build] {build_id}: queued lanes; "
              f"counts={queue.counts()} (lanes with landed plugins proceed via workers)")
    return 0


def _build_lane(args: Namespace, cfg: cs.LoadedConfig, lock: Any) -> int:
    """``bank build --lane L``: dispatch the lane's real production entrypoint
    (R2). One queue work item per (lane, seed) dispatch, stage handlers around
    the production call, idempotent re-runs, PLUGIN_MISSING dead-letters for
    unknown lanes. ``--out`` redirects the lane's STAGING into a scratch dir —
    the lane's real staging is never touched."""
    from autoontobench.factory.lane_dispatch import (
        bind_entrypoint_kwargs, dispatch_lane_build, find_lane_entrypoint,
        lane_missing_detail)

    data = cfg.data
    lane = str(args.lane).strip()
    seed = int(args.seed) if getattr(args, "seed", None) is not None else int(
        data["build"]["stable_seed"])
    out_dir = cs.resolve_repo_path(args.out) if getattr(args, "out", None) else None
    plugins_root = REPO / "benchmark" / "plugins"
    if args.dry_run:
        entrypoint, issues = find_lane_entrypoint(plugins_root, lane)
        if entrypoint is None:
            print(f"DRY-RUN lane {lane.upper()}: PLUGIN_MISSING: "
                  f"{lane_missing_detail(plugins_root, lane, issues)}")
            return 1
        try:
            bound = bind_entrypoint_kwargs(
                entrypoint, limit=args.limit, seed=seed,
                resume=True if args.resume else None, out_dir=out_dir)
        except cs.ConfigError as exc:
            cs.eprint(f"[bank build] lane {lane}: {exc}")
            return 2
        print(f"DRY-RUN lane {entrypoint.lane.upper()} dispatch via {entrypoint.origin}")
        print(f"DRY-RUN entrypoint_path={entrypoint.path}")
        print(f"DRY-RUN kwargs={{{', '.join(f'{k}={v!r}' for k, v in sorted(bound.kwargs.items()))}}}"
              f" staging_redirect={bound.staging_redirect}")
        if bound.notes:
            for note in bound.notes:
                print(f"DRY-RUN note: {note}")
        print(f"DRY-RUN out={out_dir or '<lane STAGING>'} seed={seed} "
              f"limit={args.limit} resume={bool(args.resume)}")
        return 0
    with BuildQueue(QUEUE_PATH) as queue:
        queue.requeue_stale()
        outcome = dispatch_lane_build(
            queue, plugins_root=plugins_root, lane=lane,
            lock_hash=lock.lock_hash or "unpinned", seed=seed, limit=args.limit,
            out_dir=out_dir, resume=bool(args.resume), generator_version=GEN_VERSION,
            build_id=data["bank_id"])
    print(f"[bank build] lane {outcome.lane}: {outcome.status} (item #{outcome.item_id}, "
          f"invoked={outcome.invoked})")
    print(f"[bank build] {outcome.detail}")
    if outcome.rejection_reason:
        print(f"[bank build] rejection_reason={outcome.rejection_reason}")
    if outcome.out_dir is not None:
        print(f"[bank build] out={outcome.out_dir}")
    return 0 if outcome.status in ("dispatched", "already-done") else 1


def cmd_status(args: Namespace) -> int:
    if not QUEUE_PATH.is_file():
        cs.eprint(f"[bank status] no queue at {QUEUE_PATH}; run bank build first")
        return 1
    with BuildQueue(QUEUE_PATH) as queue:
        snap = collect_metrics(queue, build_id=args.build_id)
        print(json.dumps(snap, indent=2, ensure_ascii=False))
        if args.by_lane:
            print(f"{'lane':<6} {'pending':>8} {'running':>8} {'done':>6} {'dead':>6}")
            for row in queue.by_lane():
                print(f"{row['lane']:<6} {row.get('pending', 0):>8} {row.get('running', 0):>8} "
                      f"{row.get('done', 0):>6} {row.get('dead', 0):>6}")
    return 0


def cmd_audit(args: Namespace) -> int:
    build_root = REPO / "bank" / "builds" / args.build_id
    problems: list[str] = []
    public_files = sorted(build_root.rglob("*.jsonl")) if build_root.is_dir() else []
    from autoontobench.contracts.models import assert_no_private_keys

    checked = 0
    for path in public_files:
        if "private" in path.parts:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                assert_no_private_keys(json.loads(line))
            except ValueError as exc:
                problems.append(f"{path}: {exc}")
            checked += 1
    store = ShardStore(REPO / "bank" / "staging")
    for meta in store.committed():
        sp = Path(meta["path"])
        if not sp.is_file():
            problems.append(f"shard missing: {sp}")
    print(f"[bank audit] build={args.build_id} public_rows={checked} "
          f"shards={len(store.committed())} problems={len(problems)}")
    for p in problems:
        print(f"  FAIL {p}")
    if args.strict and problems:
        return 1
    return 0


def cmd_package(args: Namespace) -> int:
    from autoontobench.factory.gates.license_matrix import LicenseMatrix
    from autoontobench.factory.release.package import (
        PackagingError, package_private, package_public, preview_public)

    release = args.release
    src = Path(args.input) if getattr(args, "input", None) else (
        REPO / "bank" / "builds" / args.build_id)
    if not src.is_dir():
        cs.eprint(f"[bank package] nothing to package at {src}")
        return 1
    name = args.build_id or src.name
    manifest_path = src / "task_manifest.json"
    if args.build_id is None and manifest_path.is_file():
        name = json.loads(manifest_path.read_text(encoding="utf-8")).get("bank_id", name)
    releases_root = REPO / "bank" / "releases"

    if release == "private-local":
        dest = releases_root / release / name
        dest.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for path in sorted(src.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            hashes[str(rel)] = cs.sha256_hex(path.read_bytes())
        cs.write_json(dest / "PACKAGE_MANIFEST.json", {
            "release": release, "source": str(src), "files": hashes,
            "packaged_at": cs.utc_now_iso()})
        print(f"[bank package] {release} <- {src} files={len(hashes)} -> {dest}")
        return 0

    matrix_path = Path(args.license_matrix) if getattr(args, "license_matrix", None) else (
        REPO / "source_registry" / "licenses" / "source_redistribution_matrix.csv")
    try:
        if release == "public":
            matrix = LicenseMatrix.load(
                matrix_csv=matrix_path if matrix_path.is_file() else None,
                sources_yaml=REPO / "source_registry" / "sources.yaml")
            if getattr(args, "dry_run", False):
                preview = preview_public(src, license_matrix=matrix)
                print(f"[bank package] DRY-RUN public build={name} "
                      f"total={preview.total} eligible={preview.eligible} "
                      f"excluded={preview.excluded}")
                if preview.held_sources:
                    held = ", ".join(f"{k}={v}" for k, v in
                                     list(preview.held_sources.items())[:20])
                    print(f"  held sources: {held}")
                print(f"  matrix={matrix_path if matrix_path.is_file() else '(live matrix absent -> sources.yaml fallback)'}")
                return 0
            result = package_public(
                src, releases_root, build_id=name, license_matrix=matrix,
                attribution_dir=REPO / "bank" / "release_obligations",
                matrix_csv=matrix_path if matrix_path.is_file() else None)
            print(f"[bank package] public <- {src} files={result.file_count} "
                  f"root_sha256={result.root_sha256[:16]}… -> {result.root}")
            return 0
        result = package_private(src, releases_root, build_id=name)
        print(f"[bank package] private <- {src} files={result.file_count} "
              f"root_sha256={result.root_sha256[:16]}… -> {result.root}")
        return 0
    except PackagingError as exc:
        cs.eprint(f"[bank package] blocked: {exc}")
        return 1


# -------------------------------------------------------------- BYOK generate

def cmd_generate(args: Namespace) -> int:
    cfg = cs.load_yaml_strict(cs.resolve_repo_path(args.config), cs.BYOK_BANK_CONFIG_SPEC)
    data = cfg.data
    out_dir = cs.resolve_repo_path(args.output or data["output"]["public_dir"])
    src_cfg = cs.load_yaml_strict((cfg.path.parent / data["source_config"]).resolve(),
                                  cs.BYOK_SOURCE_CONFIG_SPEC)
    sdata = src_cfg.data
    plugins = {kind: load_plugin_file((src_cfg.path.parent / sdata["plugins"][kind]).resolve(),
                                      kind=kind)
               for kind in ("source", "task_family", "verifier")}
    if args.dry_run:
        print(f"DRY-RUN generate bank={data['bank_id']} source={sdata['source_id']} "
              f"candidates={data['generation']['candidate_count']} out={out_dir}")
        return 0
    return _generate_real(data, sdata, plugins, out_dir, resume=bool(args.resume))


def _generate_real(data: dict[str, Any], sdata: dict[str, Any],
                   plugins: dict[str, Any], out_dir: Path, *, resume: bool) -> int:
    from benchmark.schema.plugin_protocols import (
        CandidateTaskAST, GroundedTask, SourceRelease, SourceSnapshot)

    cache = ContentAddressedCache(REPO / "external_knowledge")
    source = plugins["source"].plugin
    family_plugin = plugins["task_family"].plugin
    verifier = plugins["verifier"].plugin
    pinned = sdata["pinned"]
    release = SourceRelease(release_id=pinned["commit_or_release"][:12],
                            version=pinned["upstream_version"],
                            urls=tuple(pinned["urls"]))
    descriptor = source.describe()
    receipt = source.fetch(release, resume=resume, cache_dir=cache.pinned)
    report = source.verify_snapshot(receipt)
    if not report.ok:
        cs.eprint(f"[bank generate] snapshot verification failed: {report.missing}")
        return 1
    snapshot = SourceSnapshot(source_id=sdata["source_id"], release_id=release.release_id,
                              root=Path(receipt.cache_dir), descriptor=descriptor)
    gen = data["generation"]
    produced: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    lock_hash = cs.canonical_json_sha(sdata)
    with BuildQueue(QUEUE_PATH) as queue:
        for family in gen["families"]:
            constraints = {"count": gen["candidate_count"], "regime": gen["regime"],
                           "semantic_profile": gen["semantic_profile"]}
            for cand in family_plugin.propose(snapshot, family=family,
                                              seed=int(gen["seed"]), constraints=constraints):
                item_id, _ = queue.enqueue(
                    source_lock_hash=lock_hash, lane=f"BYOK:{sdata['source_id']}",
                    family=cand.family_id, base_goal=cand.base_goal_id,
                    seed=int(cand.ast.get("seed", 0)), generator_version=GEN_VERSION)
                grounded = family_plugin.ground(cand)
                proof = verifier.prove_exists(grounded)
                if not proof.exists:
                    queue.fail(item_id, error=proof.reason, rejection_reason="UNSOLVABLE")
                    rejected.append({"candidate": cand.instance_id, "reason": proof.reason})
                    continue
                witness = verifier.certify_semantic_relevance(grounded)
                if not witness.ok:
                    queue.fail(item_id, error="no semantic witness",
                               rejection_reason="NON_SEMANTIC")
                    rejected.append({"candidate": cand.instance_id, "reason": "NON_SEMANTIC"})
                    continue
                digest = cs.canonical_json_sha(cand.ast)
                if digest in seen:
                    queue.fail(item_id, error="duplicate", rejection_reason="DUPLICATE")
                    continue
                seen.add(digest)
                for _stage in STAGES[:-1]:
                    claimed = queue.claim_id(item_id, "byok-generate")
                    if claimed is None:
                        break
                    queue.advance(claimed.id, verdict="ok", duration_s=0.0,
                                  payload_hash=digest)
                produced.append({"task_id": f"{data['bank_id']}-{cand.instance_id}",
                                 "instance_id": cand.instance_id,
                                 "plugin_paths": {k: str(v.path) for k, v in plugins.items()},
                                 "family_id": cand.family_id, "base_goal_id": cand.base_goal_id,
                                 "regime": cand.regime, "semantic_profile": cand.semantic_profile,
                                 "public": grounded.public, "ast_sha256": digest,
                                 "ast": cand.ast, "oracle_ref": grounded.oracle_ref,
                                 "native_evidence": grounded.native_evidence,
                                 "status": "QUALIFIED"})
    _write_user_bank(out_dir, data, sdata, produced, rejected)
    print(f"[bank generate] bank={data['bank_id']} qualified={len(produced)} "
          f"rejected={len(rejected)} out={out_dir}")
    return 0 if produced else 1


def _write_user_bank(out_dir: Path, data: dict[str, Any], sdata: dict[str, Any],
                     produced: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    public_dir = out_dir / "public"
    private_dir = out_dir / "private_oracle"
    public_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    private_dir.chmod(0o700)
    with (public_dir / "tasks.jsonl").open("w", encoding="utf-8") as fh:
        for row in produced:
            fh.write(json.dumps({k: row[k] for k in
                                 ("task_id", "instance_id", "family_id", "base_goal_id",
                                  "regime", "semantic_profile", "public", "ast_sha256",
                                  "status")}, ensure_ascii=False, sort_keys=True) + "\n")
    with (private_dir / "oracle.json").open("w", encoding="utf-8") as fh:
        json.dump({row["task_id"]: {"ast": row["ast"], "oracle_ref": row["oracle_ref"],
                                    "native_evidence": row["native_evidence"]}
                   for row in produced}, fh, indent=1)
    plugin_paths = produced[0]["plugin_paths"] if produced else dict(sdata["plugins"])
    cs.write_json(out_dir / "task_manifest.json", {
        "bank_id": data["bank_id"], "owner": data["owner"], "license": data["license"],
        "source_id": sdata["source_id"], "source_provenance": sdata["pinned"],
        "plugins": plugin_paths, "qualified": len(produced), "rejected": rejected,
        "task_hashes": {r["task_id"]: r["ast_sha256"] for r in produced},
        "built_at": cs.utc_now_iso()})


def cmd_validate(args: Namespace) -> int:
    in_dir = cs.resolve_repo_path(args.input)
    manifest_path = in_dir / "task_manifest.json"
    if not manifest_path.is_file():
        cs.eprint(f"[bank validate] no task_manifest.json in {in_dir}")
        return 2
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tasks = [json.loads(line) for line in
             (in_dir / "public" / "tasks.jsonl").read_text(encoding="utf-8").splitlines()
             if line.strip()]
    problems: list[str] = []
    if len(tasks) != int(manifest["qualified"]):
        problems.append(f"tasks.jsonl has {len(tasks)} rows, manifest says {manifest['qualified']}")
    for row in tasks:
        if manifest["task_hashes"].get(row["task_id"]) != row["ast_sha256"]:
            problems.append(f"{row['task_id']}: hash mismatch vs manifest")
    if not args.native_only:
        oracle = json.loads((in_dir / "private_oracle" / "oracle.json").read_text(encoding="utf-8"))
        if set(oracle) != {r["task_id"] for r in tasks}:
            problems.append("private oracle ids != public task ids")
    plugin_path = Path(manifest["plugins"]["verifier"])
    verifier_ref = load_plugin_file(plugin_path if plugin_path.is_absolute()
                                    else REPO / plugin_path, kind="verifier")
    from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask

    oracle = json.loads((in_dir / "private_oracle" / "oracle.json").read_text(encoding="utf-8"))
    revalidated = 0
    for row in tasks:
        record = oracle[row["task_id"]]
        ast = CandidateTaskAST(lane_id="BYOK", family_id=row["family_id"],
                               base_goal_id=row["base_goal_id"],
                               instance_id=row["instance_id"], regime=row["regime"],
                               semantic_profile=row["semantic_profile"], ast=record["ast"],
                               source_unit_ids=tuple(record["ast"].get("source_unit_ids", ())))
        grounded = GroundedTask(ast=ast, public=row["public"],
                                oracle_ref=record["oracle_ref"],
                                native_evidence=record["native_evidence"])
        proof = verifier_ref.plugin.prove_exists(grounded)
        if not proof.exists:
            problems.append(f"{row['task_id']}: native re-proof failed: {proof.reason}")
        else:
            revalidated += 1
    print(f"[bank validate] input={in_dir} tasks={len(tasks)} revalidated={revalidated} "
          f"problems={len(problems)} native_only={bool(args.native_only)}")
    for p in problems:
        print(f"  FAIL {p}")
    return 1 if problems else 0


def run(args: Namespace) -> int:
    handler = {"preflight": cmd_preflight, "build": cmd_build, "status": cmd_status,
               "audit": cmd_audit, "package": cmd_package, "generate": cmd_generate,
               "validate": cmd_validate}.get(args.bank_command)
    if handler is None:
        cs.eprint("[bank] subcommand required: preflight|build|status|audit|package|generate|validate")
        return 2
    try:
        return handler(args)
    except cs.ConfigError as exc:
        cs.eprint(f"[bank] {exc}")
        return 2
    except cs.FactoryError as exc:
        cs.eprint(f"[bank] {exc}")
        return 1
