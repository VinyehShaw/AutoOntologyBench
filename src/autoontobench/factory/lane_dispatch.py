"""Generic lane-build dispatch (R2): ``bank build --lane <L>`` actually runs
the lane's production entrypoint through the persistent queue.

Convention (no lane list hardcoded): a lane is dispatchable when
``benchmark/plugins/task_family/<lane>/`` exposes a module-level production
callable named one of :data:`ENTRYPOINT_NAMES` (``run_build`` /
``build_shards`` / ``build`` / ``run`` / ...). The callable is invoked with
keyword arguments bound by signature inspection:

* ``--limit N``   -> first accepted of ``max_per_family``/``limit``/...
* ``--seed S``    -> ``seed`` (default: the config's ``build.stable_seed``)
* ``--resume``    -> ``resume=True`` (only if the callable takes it)
* ``--out DIR``   -> an out-ish parameter (``out_root``/``out_dir``/...) when
  the callable has one; otherwise the module-level ``STAGING`` constant is
  redirected for the duration of the call (the lane pipelines resolve
  ``STAGING`` at call time), so dispatch never has to edit lane code.

Scratch containment (``--out``): besides the entrypoint's ``STAGING``, every
module-level UPPERCASE ``Path`` constant anywhere in the imported lane
closure that points under the lane's real staging
(``<repo>/bank/staging/<lane>/...``) is redirected to the mirror under
``--out`` for the duration of the production call — lane verifiers keep
their own constants (e.g. ``ORACLE_DIR``) that would otherwise escape a
scratch dispatch and write into the lane's real staging.

Invocation goes through the queue with stage handlers, mirroring the BYOK
``_generate_real`` seam: one work item per (lane, seed) dispatch, claimed by
id (never lane-scoped claims, so bootstrap items are never stolen), the real
production call happens at the ``candidate_ast`` stage, remaining stages are
factory bookkeeping. Unknown lanes are never silently ignored: no entrypoint
means a queue item failed with ``PLUGIN_MISSING`` naming lane/kind/path.
"""
from __future__ import annotations

import inspect
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from autoontobench.factory.build import StageFailure, StageResult
from autoontobench.factory.cli_support import ConfigError, utc_now_iso, write_json
from autoontobench.factory.plugin_discovery import (
    _ImportContext, _import_context, _import_module)
from autoontobench.factory.queue import STAGES, BuildQueue

#: Production entrypoint names in priority order (first match across the
#: lane's modules, modules scanned in sorted order).
ENTRYPOINT_NAMES: tuple[str, ...] = (
    "run_build", "build_shards", "build", "run", "build_lane", "produce")

#: Parameter names the dispatcher may bind ``--out`` to, in priority order.
OUT_PARAM_NAMES: tuple[str, ...] = (
    "out_root", "out_dir", "output_dir", "staging_root", "staging", "out")

#: Parameter names the dispatcher may bind ``--limit`` to, in priority order.
LIMIT_PARAM_NAMES: tuple[str, ...] = (
    "max_per_family", "limit", "max_candidates", "max_total", "max_instances")

#: Queue family/base_goal tags for a lane production dispatch (idempotency
#: key = lock_hash|lane|family|base_goal|seed|generator_version).
DISPATCH_FAMILY = "{lane}-production"
DISPATCH_BASE_GOAL = "lane-build"

#: Stage at which the production entrypoint is invoked.
PRODUCTION_STAGE = "candidate_ast"

#: Stage at which the dispatch target is validated/created.
BIND_STAGE = "source_snapshot_ready"

_DISPATCH_REPORT = "dispatch_report.json"


@dataclass(frozen=True)
class LaneEntrypoint:
    lane: str  # lowercase dir name, e.g. "l13"
    module: ModuleType
    func: Callable[..., Any]
    path: Path
    origin: str  # e.g. "benchmark.plugins.task_family.l13.pipeline:run_build"


@dataclass(frozen=True)
class BoundCall:
    kwargs: dict[str, Any] = field(default_factory=dict)
    staging_redirect: Path | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class DispatchOutcome:
    lane: str
    status: str  # dispatched | already-done | in-progress | failed | plugin-missing
    item_id: int
    invoked: bool
    detail: str
    out_dir: Path | None = None
    stats: dict[str, Any] | None = None
    rejection_reason: str | None = None
    entrypoint: str | None = None


# ------------------------------------------------------------- entrypoint find

def lane_missing_detail(plugins_root: Path, lane: str,
                        issues: list[str] | None = None) -> str:
    """Explicit PLUGIN_MISSING text naming lane/kind/path. Shared by the CLI
    dry-run and the real dispatch so both report identically."""
    lane_l = lane.lower()
    detail = (f"task_family/{lane_l} [{plugins_root / 'task_family' / lane_l}]: "
              f"no production entrypoint (convention: {'/'.join(ENTRYPOINT_NAMES[:4])})")
    if issues:
        detail += "; " + "; ".join(issues)
    return detail


def find_lane_entrypoint(plugins_root: Path, lane: str) -> tuple[LaneEntrypoint | None, list[str]]:
    """Locate the lane's production entrypoint by convention. Returns
    (entrypoint, import_issues); entrypoint is None when the lane dir is
    missing, empty, or exposes no production callable."""
    lane_dir = plugins_root / "task_family" / lane.lower()
    if not lane_dir.is_dir():
        return None, []
    ctx = _import_context(plugins_root)
    kind_prefix = f"{ctx.prefix}.task_family" if ctx.prefix else None
    kind_ctx = _ImportContext(prefix=kind_prefix, anchor=ctx.anchor)
    issues: list[str] = []
    modules: list[tuple[ModuleType, Path]] = []
    targets: list[tuple[str | None, Path]] = []
    init_path = lane_dir / "__init__.py"
    if init_path.is_file() and init_path.stat().st_size > 0:
        targets.append((f"{kind_prefix}.{lane_dir.name}" if kind_prefix else None, init_path))
    targets.extend(
        (f"{kind_prefix}.{lane_dir.name}.{p.stem}" if kind_prefix else None, p)
        for p in sorted(lane_dir.glob("*.py"))
        if p.name != "__init__.py" and not p.name.startswith("_"))
    for dotted, path in targets:
        try:
            modules.append((_import_module(dotted, path, kind_ctx), path))
        except Exception as exc:  # per-lane isolation, same policy as discovery
            issues.append(f"import failed: {path.name}: {type(exc).__name__}: {exc}")
    for name in ENTRYPOINT_NAMES:  # name priority, then module order
        for module, path in modules:
            func = vars(module).get(name)
            if callable(func):
                return LaneEntrypoint(lane=lane_dir.name, module=module, func=func,
                                      path=path, origin=f"{module.__name__}:{name}"), issues
    return None, issues


# ------------------------------------------------------------------ kwarg bind

def _bindable(sig: inspect.Signature, name: str) -> bool:
    p = sig.parameters.get(name)
    return p is not None and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                        inspect.Parameter.KEYWORD_ONLY)


def bind_entrypoint_kwargs(entrypoint: LaneEntrypoint, *, limit: int | None = None,
                           seed: int | None = None, resume: bool | None = None,
                           out_dir: Path | str | None = None) -> BoundCall:
    """Bind CLI intent to the entrypoint's signature. Raises ConfigError when
    ``--out`` is requested but the entrypoint supports no redirection, or when
    the entrypoint requires parameters the dispatcher cannot infer."""
    sig = inspect.signature(entrypoint.func)
    kwargs: dict[str, Any] = {}
    notes: list[str] = []
    if limit is not None:
        for name in LIMIT_PARAM_NAMES:
            if _bindable(sig, name):
                kwargs[name] = limit
                break
        else:
            notes.append(f"--limit {limit} not bindable: {entrypoint.origin} accepts "
                         f"none of {', '.join(LIMIT_PARAM_NAMES)}")
    if seed is not None and _bindable(sig, "seed"):
        kwargs["seed"] = int(seed)
    if resume is not None and _bindable(sig, "resume"):
        kwargs["resume"] = bool(resume)
    redirect: Path | None = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        for name in OUT_PARAM_NAMES:
            if _bindable(sig, name):
                kwargs[name] = out_dir
                break
        else:
            staging = getattr(entrypoint.module, "STAGING", None)
            if isinstance(staging, Path):
                redirect = out_dir  # lane pipelines resolve STAGING at call time
            else:
                raise ConfigError(
                    f"lane {entrypoint.lane}: entrypoint {entrypoint.origin} supports no "
                    f"--out redirection (no parameter in {', '.join(OUT_PARAM_NAMES)} and "
                    f"no module-level STAGING constant)")
    missing = [name for name, p in sig.parameters.items()
               if p.default is inspect.Parameter.empty
               and p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.KEYWORD_ONLY)
               and name not in kwargs]
    if missing:
        raise ConfigError(
            f"lane {entrypoint.lane}: entrypoint {entrypoint.origin} requires parameters "
            f"the dispatcher cannot infer: {missing} (lane needs a dispatch facade)")
    return BoundCall(kwargs=kwargs, staging_redirect=redirect, notes=tuple(notes))


# ------------------------------------------------------------------- dispatch

def _staging_redirects(entrypoint: LaneEntrypoint, out_dir: Path,
                       plugins_root: Path) -> list[tuple[ModuleType, str, Path, Path]]:
    """(module, attr, old, new) for every module-level UPPERCASE Path constant
    under the lane's real staging dir, across all imported modules. Lane
    verifiers define their own constants (e.g. ``ORACLE_DIR``) resolved from
    ``__file__`` at import time; without this they escape a scratch dispatch."""
    staging_root = plugins_root.parent.parent / "bank" / "staging"
    lane = entrypoint.lane.lower()
    found: list[tuple[ModuleType, str, Path, Path]] = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name, val in list(vars(module).items()):
            if not isinstance(val, Path) or not name.isupper():
                continue
            try:
                rel = val.relative_to(staging_root)
            except ValueError:
                continue
            parts = rel.parts
            if not parts or parts[0].lower() != lane:
                continue
            target = out_dir if len(parts) == 1 else out_dir.joinpath(*parts[1:])
            if target != val:
                found.append((module, name, val, target))
    return found


_FLAT_MISSING = object()


def _failure_reason(exc: Exception) -> str:
    if isinstance(exc, StageFailure):
        return exc.reason
    if isinstance(exc, FileNotFoundError):
        return "SOURCE_UNPINNED"
    if isinstance(exc, (ImportError, AttributeError)):
        return "PLUGIN_MISSING"
    return "WORKER_ERROR"


def _stage_handlers(entrypoint: LaneEntrypoint, bound: BoundCall,
                    out_dir: Path | None, plugins_root: Path,
                    sink: dict[str, Any]) -> dict[str, Callable]:
    def bind_stage(item) -> StageResult:
        target = out_dir if out_dir is not None else getattr(
            entrypoint.module, "STAGING", None)
        if target is not None:
            Path(target).mkdir(parents=True, exist_ok=True)
        return StageResult(verdict="ok", payload_ref=str(target) if target else None)

    def production_stage(item) -> StageResult:
        saved = (_staging_redirects(entrypoint, out_dir, plugins_root)
                 if out_dir is not None else [])
        flat_old: Any = _FLAT_MISSING
        if (out_dir is not None and bound.staging_redirect is not None
                and not any(m is entrypoint.module and n == "STAGING"
                            for m, n, _o, _t in saved)):
            flat_old = getattr(entrypoint.module, "STAGING", _FLAT_MISSING)
            entrypoint.module.STAGING = bound.staging_redirect
        for module, name, _old, new in saved:
            setattr(module, name, new)
        sink["path_redirects"] = {
            f"{m.__name__}.{n}": str(t) for m, n, _o, t in saved}
        started = time.time()
        try:
            stats = entrypoint.func(**bound.kwargs)
        finally:
            for module, name, old, _new in saved:
                setattr(module, name, old)
            if flat_old is not _FLAT_MISSING:
                entrypoint.module.STAGING = flat_old
        sink["stats"] = stats if isinstance(stats, dict) else {"returned": repr(stats)}
        sink["duration_s"] = round(time.time() - started, 3)
        return StageResult(verdict="ok", payload_ref=str(out_dir) if out_dir else None)

    def bookkeeping(item) -> StageResult:
        return StageResult(verdict="ok")

    return {BIND_STAGE: bind_stage, PRODUCTION_STAGE: production_stage}


def dispatch_lane_build(queue: BuildQueue, *, plugins_root: Path, lane: str,
                        lock_hash: str, seed: int, limit: int | None = None,
                        out_dir: Path | str | None = None, resume: bool = False,
                        generator_version: str, build_id: str = "",
                        worker: str = "lane-dispatch",
                        max_attempts: int | None = None) -> DispatchOutcome:
    """Enqueue + run one lane production dispatch through the queue. Idempotent:
    a DONE item is not re-invoked unless ``resume``; DEAD/RUNNING items are
    reported (retry with ``resume``). Unknown lanes dead-letter with
    PLUGIN_MISSING naming lane/kind/path. ``max_attempts`` overrides the
    queue default for this dispatch's item (tests / fast-fail runs)."""
    lane_l = lane.lower()
    lane_u = lane.upper()
    out_dir = Path(out_dir) if out_dir is not None else None
    entrypoint, issues = find_lane_entrypoint(plugins_root, lane_l)
    if entrypoint is None:
        detail = lane_missing_detail(plugins_root, lane_l, issues)
        item_id, _ = queue.enqueue(
            source_lock_hash=lock_hash, lane=lane_u,
            family=DISPATCH_FAMILY.format(lane=lane_u), base_goal=DISPATCH_BASE_GOAL,
            seed=seed, generator_version=generator_version, max_attempts=max_attempts)
        claimed = queue.claim_id(item_id, worker)
        if claimed is not None:
            queue.fail(claimed.id, error=detail, rejection_reason="PLUGIN_MISSING")
        return DispatchOutcome(lane=lane_u, status="plugin-missing", item_id=item_id,
                               invoked=False, detail=detail,
                               rejection_reason="PLUGIN_MISSING")
    bound = bind_entrypoint_kwargs(entrypoint, limit=limit, seed=seed,
                                   resume=resume or None, out_dir=out_dir)
    item_id, created = queue.enqueue(
        source_lock_hash=lock_hash, lane=lane_u,
        family=DISPATCH_FAMILY.format(lane=lane_u), base_goal=DISPATCH_BASE_GOAL,
        seed=seed, generator_version=generator_version, max_attempts=max_attempts)
    if not created:
        existing = queue.get(item_id)
        if existing is not None:
            if existing.status == "DONE" and not resume:
                return DispatchOutcome(
                    lane=lane_u, status="already-done", item_id=item_id, invoked=False,
                    detail=(f"item #{item_id} already DONE (use --resume to re-run); "
                            f"payload_ref={existing.payload_ref}"),
                    out_dir=out_dir, entrypoint=entrypoint.origin)
            if existing.status == "RUNNING" and not resume:
                return DispatchOutcome(
                    lane=lane_u, status="in-progress", item_id=item_id, invoked=False,
                    detail=f"item #{item_id} is RUNNING (another dispatch); use --resume to force",
                    out_dir=out_dir, entrypoint=entrypoint.origin)
            if existing.status == "DEAD" and not resume:
                return DispatchOutcome(
                    lane=lane_u, status="failed", item_id=item_id, invoked=False,
                    detail=(f"item #{item_id} DEAD from a previous dispatch "
                            f"(rejection_reason={existing.status and _dead_reason(queue, item_id)}); "
                            f"use --resume to retry"),
                    out_dir=out_dir, rejection_reason=_dead_reason(queue, item_id),
                    entrypoint=entrypoint.origin)
            if resume:
                queue.reset_item(item_id)
    sink: dict[str, Any] = {}
    handlers = _stage_handlers(entrypoint, bound, out_dir, plugins_root, sink)
    invoked_at = utc_now_iso()
    for stage in STAGES:
        claimed = queue.claim_id(item_id, worker)
        if claimed is None:
            return DispatchOutcome(
                lane=lane_u, status="in-progress", item_id=item_id, invoked=False,
                detail=f"item #{item_id} left PENDING mid-dispatch (concurrent claim); "
                       f"re-run or use --resume",
                out_dir=out_dir, entrypoint=entrypoint.origin)
        handler = handlers.get(stage, lambda item: StageResult(verdict="ok"))
        try:
            result = handler(claimed)
        except StageFailure as exc:
            queue.fail(claimed.id, error=str(exc), rejection_reason=exc.reason,
                       retry_after_s=exc.retry_after_s)
            return DispatchOutcome(lane=lane_u, status="failed", item_id=item_id,
                                   invoked=stage == PRODUCTION_STAGE,
                                   detail=f"stage {stage}: {exc}", out_dir=out_dir,
                                   rejection_reason=exc.reason,
                                   entrypoint=entrypoint.origin)
        except Exception as exc:
            reason = _failure_reason(exc)
            queue.fail(claimed.id, error=f"{type(exc).__name__}: {exc}",
                       rejection_reason=reason)
            return DispatchOutcome(
                lane=lane_u, status="failed", item_id=item_id,
                invoked=stage == PRODUCTION_STAGE,
                detail=f"stage {stage}: {type(exc).__name__}: {exc}",
                out_dir=out_dir, rejection_reason=reason, entrypoint=entrypoint.origin)
        queue.advance(claimed.id, verdict=result.verdict,
                      duration_s=result.duration_s if hasattr(result, "duration_s") else 0.0,
                      payload_hash=result.payload_hash, payload_ref=result.payload_ref,
                      checkpoint_ref=result.checkpoint_ref, cost_usd=result.cost_usd,
                      tokens_in=result.tokens_in, tokens_out=result.tokens_out)
    report_dir = out_dir if out_dir is not None else getattr(
        entrypoint.module, "STAGING", None)
    if isinstance(report_dir, Path):
        write_json(report_dir / _DISPATCH_REPORT, {
            "schema": "aob-lane-dispatch/1",
            "lane": lane_u, "build_id": build_id,
            "entrypoint": entrypoint.origin,
            "entrypoint_path": str(entrypoint.path),
            "kwargs": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in bound.kwargs.items()},
            "staging_redirect": str(bound.staging_redirect) if bound.staging_redirect else None,
            "path_redirects": sink.get("path_redirects", {}),
            "notes": list(bound.notes),
            "limit": limit, "seed": seed, "resume": resume,
            "queue_item_id": item_id, "invoked_at": invoked_at,
            "duration_s": sink.get("duration_s"),
            "lane_stats": sink.get("stats"),
            "out_dir": str(report_dir),
        })
    return DispatchOutcome(
        lane=lane_u, status="dispatched", item_id=item_id, invoked=True,
        detail=(f"item #{item_id} DONE via {entrypoint.origin}; "
                f"stats={_stats_summary(sink.get('stats'))}"),
        out_dir=out_dir, stats=sink.get("stats"), entrypoint=entrypoint.origin)


def _dead_reason(queue: BuildQueue, item_id: int) -> str | None:
    with queue._lock:
        row = queue._db.execute(
            "SELECT rejection_reason FROM work_items WHERE id=?", (item_id,)).fetchone()
    return row["rejection_reason"] if row else None


def _stats_summary(stats: dict[str, Any] | None) -> str:
    if not isinstance(stats, dict):
        return "n/a"
    keys = ("qualified", "candidates", "base_goals", "rejected", "wall_seconds")
    picked = {k: stats[k] for k in keys if k in stats}
    return repr(picked) if picked else repr({k: stats[k] for k in list(stats)[:6]})
