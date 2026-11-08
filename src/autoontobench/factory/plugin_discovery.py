"""Generic lane plugin discovery + structural validation for the Task Factory.

Lane plugins live under ``benchmark/plugins/{source,task_family,verifier}/``
either as single-file drop-ins (``<name>.py`` exposing ``PLUGIN`` or
``get_plugin()``; the BYOK shape produced by ``plugin scaffold``) or as
per-lane packages (``<lane>/`` whose modules expose ``PLUGIN`` /
``get_plugin()`` or define a class covering the kind's interface surface).
Discovery is generic: no lane list is hardcoded, one lane's broken import
never blinds the others (per-lane isolation), and every gap is reported as a
``PluginIssue`` instead of a silent zero.

Interface surfaces follow the governing build order §3.2 (see
``PLUGIN_SURFACES``); the source kind additionally accepts the locked
``ExternalKnowledgeSource`` protocol shape from
``benchmark/schema/plugin_protocols.py``. Validation is structural (callable
attribute coverage): lane plugins requiring constructor args (e.g. a verifier
bound to a pinned index) are referenced by class and bound at dispatch time.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from autoontobench.factory.cli_support import ConfigError

PLUGIN_KINDS: tuple[str, ...] = ("source", "task_family", "verifier")

#: Acceptable callable surfaces per plugin kind (governing doc §3.2). A plugin
#: is valid when it covers ANY one surface completely. The source kind accepts
#: the lane §3.2 shape (pin_and_index/enumerate_units) or the locked
#: ExternalKnowledgeSource protocol shape (describe/fetch/inventory/...).
PLUGIN_SURFACES: dict[str, tuple[tuple[str, ...], ...]] = {
    "source": (
        ("pin_and_index", "enumerate_units"),
        ("describe", "discover_releases", "fetch", "verify_snapshot", "inventory"),
    ),
    "task_family": (("supported_profiles", "propose", "ground"),),
    "verifier": (("prove_exists", "verify_terminal", "certify_semantic_relevance"),),
}


@dataclass(frozen=True)
class PluginRef:
    kind: str
    name: str
    path: Path
    plugin: object
    origin: str = ""  # resolution trail, e.g. "class benchmark.plugins...L04SourcePlugin"


@dataclass(frozen=True)
class PluginIssue:
    """A lane/drop-in that was inspected but yielded no valid plugin."""

    kind: str
    lane: str
    path: Path
    reason: str


@dataclass(frozen=True)
class PluginAudit:
    plugins: dict[str, dict[str, PluginRef]]
    issues: tuple[PluginIssue, ...]


def missing_surface_methods(plugin: object, kind: str) -> tuple[str, ...]:
    """Method gaps vs the kind's closest surface; empty iff some surface is
    fully covered by callable attributes."""
    best: tuple[str, ...] | None = None
    for surface in PLUGIN_SURFACES[kind]:
        gaps = tuple(m for m in surface if not callable(getattr(plugin, m, None)))
        if not gaps:
            return ()
        if best is None or len(gaps) < len(best):
            best = gaps
    return best or ()


def surface_summary(kind: str) -> str:
    return " or ".join("+".join(surface) for surface in PLUGIN_SURFACES[kind])


def plugin_covers_lane(plugins: Mapping[str, PluginRef], lane: str) -> bool:
    """Registry lanes are uppercase ("L01"), plugin dirs lowercase ("l01");
    match case-insensitively (substring keeps names like "l01.gatekeeper")."""
    probe = lane.lower()
    return any(probe in name.lower() for name in plugins)


# ------------------------------------------------------------------ loading

def _load_module_from_file(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"plugin not importable: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class _ImportContext:
    """How plugins_root is importable: dotted package prefix + sys.path anchor,
    or both None for unpackaged trees (file-location fallback)."""

    prefix: str | None
    anchor: Path | None


def _import_context(plugins_root: Path) -> _ImportContext:
    parts: list[str] = []
    node = plugins_root
    while (node / "__init__.py").is_file():
        parts.append(node.name)
        node = node.parent
    if not parts:
        return _ImportContext(prefix=None, anchor=None)
    return _ImportContext(prefix=".".join(reversed(parts)), anchor=node)


def _import_module(dotted: str | None, path: Path, ctx: _ImportContext) -> ModuleType:
    if dotted is not None and ctx.anchor is not None:
        anchor_str = str(ctx.anchor)
        if anchor_str not in sys.path:
            sys.path.insert(0, anchor_str)
        return importlib.import_module(dotted)
    return _load_module_from_file(f"aob_plugin_{path.stem}", path)


def load_plugin_file(path: Path, *, kind: str) -> PluginRef:
    """Import a plugin module from an arbitrary file (BYOK). The module exposes
    a module-level ``PLUGIN`` object or a ``get_plugin()`` factory."""
    if not path.is_file():
        raise ConfigError(f"plugin not found: {path}")
    module = _load_module_from_file(f"aob_plugin_{kind}_{path.stem}", path)
    plugin = getattr(module, "PLUGIN", None)
    if plugin is None and callable(getattr(module, "get_plugin", None)):
        plugin = module.get_plugin()
    if plugin is None:
        raise ConfigError(f"{path}: expose PLUGIN or get_plugin()")
    return PluginRef(kind=kind, name=path.stem, path=path, plugin=plugin,
                     origin=f"{path.name}:PLUGIN/get_plugin()")


# --------------------------------------------------------------- resolution

def _explicit_export(module: ModuleType) -> object | None:
    plugin = getattr(module, "PLUGIN", None)
    if plugin is not None:
        return plugin
    factory = getattr(module, "get_plugin", None)
    if callable(factory):
        return factory()
    return None


def _scan_plugin_class(module: ModuleType, *, kind: str) -> type | None:
    """First class defined in this module covering the kind's surface
    (sorted attribute order keeps the pick deterministic)."""
    for attr in sorted(vars(module)):
        cls = vars(module)[attr]
        if (isinstance(cls, type) and cls.__module__ == module.__name__
                and not missing_surface_methods(cls, kind)):
            return cls
    return None


def _materialize(cls: type) -> object:
    """Zero-arg constructible plugins are instantiated; plugins requiring
    constructor args (e.g. a verifier bound to a pinned index) are referenced
    by class — the factory binds lane args at dispatch time."""
    try:
        return cls()
    except Exception:  # zero-arg probe only: any constructor need -> class ref
        return cls


def _discover_lane(kind: str, lane_dir: Path,
                   ctx: _ImportContext) -> tuple[PluginRef | None, list[PluginIssue]]:
    lane = lane_dir.name
    issues: list[PluginIssue] = []
    imported: list[ModuleType] = []
    targets: list[tuple[str | None, Path]] = []
    init_path = lane_dir / "__init__.py"
    if init_path.is_file():
        targets.append((f"{ctx.prefix}.{lane}" if ctx.prefix else None, init_path))
    module_files = [p for p in sorted(lane_dir.glob("*.py"))
                    if p.name != "__init__.py" and not p.name.startswith("_")]
    targets.extend((f"{ctx.prefix}.{lane}.{p.stem}" if ctx.prefix else None, p)
                   for p in module_files)
    for dotted, path in targets:
        try:
            module = _import_module(dotted, path, ctx)
        except Exception as exc:  # per-lane isolation: one broken import never blinds others
            issues.append(PluginIssue(kind, lane, path,
                                      f"import failed: {type(exc).__name__}: {exc}"))
            continue
        imported.append(module)
        try:
            explicit = _explicit_export(module)
        except Exception as exc:
            issues.append(PluginIssue(kind, lane, path,
                                      f"get_plugin() raised: {type(exc).__name__}: {exc}"))
            continue
        if explicit is None:
            continue
        gaps = missing_surface_methods(explicit, kind)
        if gaps:
            issues.append(PluginIssue(kind, lane, path,
                                      f"exported plugin misses methods: {', '.join(gaps)}"))
            return None, issues
        return (PluginRef(kind, lane, lane_dir, explicit,
                          origin=f"{module.__name__}:PLUGIN/get_plugin()"), issues)
    for module in imported:
        cls = _scan_plugin_class(module, kind=kind)
        if cls is not None:
            return (PluginRef(kind, lane, lane_dir, _materialize(cls),
                              origin=f"class {cls.__module__}.{cls.__name__}"), issues)
    if module_files and not issues:
        issues.append(PluginIssue(
            kind, lane, lane_dir,
            f"no PLUGIN/get_plugin() and no class covering {surface_summary(kind)}"))
    return None, issues


def audit_plugins(plugins_root: Path) -> PluginAudit:
    """Discover + validate every plugin under plugins_root. Returns valid
    plugins keyed by kind then lane/drop-in name, plus per-lane issues for
    code that exists but does not expose a conformant entrypoint."""
    out: dict[str, dict[str, PluginRef]] = {kind: {} for kind in PLUGIN_KINDS}
    issues: list[PluginIssue] = []
    ctx = _import_context(plugins_root)
    for kind in out:
        kind_dir = plugins_root / kind
        if not kind_dir.is_dir():
            continue
        kind_prefix = f"{ctx.prefix}.{kind}" if ctx.prefix else None
        kind_ctx = _ImportContext(prefix=kind_prefix, anchor=ctx.anchor)
        for path in sorted(kind_dir.glob("*.py")):  # single-file drop-ins (BYOK shape)
            if path.name.startswith("_"):
                continue
            dotted = f"{kind_prefix}.{path.stem}" if kind_prefix else None
            try:
                module = _import_module(dotted, path, ctx)
            except Exception as exc:
                issues.append(PluginIssue(kind, path.stem, path,
                                          f"import failed: {type(exc).__name__}: {exc}"))
                continue
            try:
                plugin = _explicit_export(module)
            except Exception as exc:
                issues.append(PluginIssue(kind, path.stem, path,
                                          f"get_plugin() raised: {type(exc).__name__}: {exc}"))
                continue
            if plugin is None:
                cls = _scan_plugin_class(module, kind=kind)
                plugin = _materialize(cls) if cls is not None else None
            if plugin is None:
                issues.append(PluginIssue(
                    kind, path.stem, path,
                    f"no PLUGIN/get_plugin() and no class covering {surface_summary(kind)}"))
                continue
            gaps = missing_surface_methods(plugin, kind)
            if gaps:
                issues.append(PluginIssue(kind, path.stem, path,
                                          f"plugin misses methods: {', '.join(gaps)}"))
                continue
            out[kind][path.stem] = PluginRef(kind, path.stem, path, plugin,
                                             origin=module.__name__)
        lane_dirs = sorted(p for p in kind_dir.iterdir()
                           if p.is_dir() and not p.name.startswith(("_", ".")))
        for lane_dir in lane_dirs:
            ref, lane_issues = _discover_lane(kind, lane_dir, kind_ctx)
            issues.extend(lane_issues)
            if ref is not None:
                out[kind][ref.name] = ref
    return PluginAudit(plugins=out, issues=tuple(issues))


def discover_plugins(plugins_root: Path) -> dict[str, dict[str, PluginRef]]:
    """Valid plugins under benchmark/plugins/{source,task_family,verifier}/.
    Missing dirs and not-yet-landed lanes are fine; gaps surface via
    ``audit_plugins`` issues instead of silent zeros."""
    return audit_plugins(plugins_root).plugins
