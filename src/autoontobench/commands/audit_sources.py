"""audit-sources: merge ``sources/registry.d/*.yaml`` and print a per-source status table.

Uses W5's registry loader (``autoontobench.taskgen.registry``); shards are merged
deterministically (filename-sorted) and malformed shards are reported, never
silently dropped. With ``--native-test``, each source's domain adapter module is
probed for a ``native_smoke()`` entrypoint which is run when exposed; domain
modules that do not expose one are reported as ``no-smoke-entrypoint`` instead of
being silently skipped.
"""
from __future__ import annotations

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Tuple

import yaml

from autoontobench.taskgen.registry import (
    RegistryError,
    SourceEntry,
    SourceRegistry,
    load_registry,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = "sources/registry.d"

#: registry source_id -> domain adapter module under autoontobench.core.domains
DOMAIN_MODULES = {
    "gatekeeper": "gatekeeper",
    "cel_kubernetes_vap": "cel",
    "camunda_dmn": "dmn",
    "rhea": "rhea_chebi",
    "chebi": "rhea_chebi",
    "rdkit": "rdkit_smarts",
    "go": "go",
    "cql_examples": "cql_who",
    "who_smart_anc": "cql_who",
    "biomodels": "biomodels",
    "behavior_1k_bddl": "bddl",
}


def _load_single_file(path: Path) -> SourceRegistry:
    """Tolerant single-file variant of load_registry (plan sanctioned registry.yaml)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return SourceRegistry(registry_dir=str(path))
    raw_entries: Any = data.get("sources") if isinstance(data, dict) else None
    if not isinstance(raw_entries, list):
        raw_entries = [data]
    merged: dict[str, SourceEntry] = {}
    for raw in raw_entries:
        source_id = raw.get("source_id") if isinstance(raw, dict) else None
        if not source_id or not isinstance(source_id, str):
            raise RegistryError(f"{path}: entry missing 'source_id'")
        merged[source_id] = SourceEntry(source_id=source_id, shard=path.name, payload=raw)
    return SourceRegistry(
        entries=merged, registry_dir=str(path), shards_loaded=(path.name,)
    )


def _pinned_ref(payload: dict[str, Any]) -> str:
    ref = payload.get("pinned_ref") or payload.get("commit") or "-"
    return " ".join(str(ref).split())


def _smoke_entrypoint(source_id: str) -> Tuple[Callable[[], int] | None, str]:
    """Probe the domain adapter for a native_smoke() entrypoint (Convention: () -> exit code)."""
    module_name = DOMAIN_MODULES.get(source_id)
    if module_name is None:
        return None, "no-domain-module"
    try:
        module = importlib.import_module(f"autoontobench.core.domains.{module_name}")
    except Exception as exc:  # boundary probe: an unimportable adapter is audit data
        return None, f"import-error:{type(exc).__name__}"
    fn = getattr(module, "native_smoke", None)
    return (fn, "available") if callable(fn) else (None, "no-smoke-entrypoint")


def _row(source_id: str, status: str, pinned: str, smoke: str | None) -> str:
    base = f"{source_id:<22} {status:<28} {pinned}"
    return base if smoke is None else f"{base}  smoke={smoke}"


def run(args: Namespace) -> int:
    registry_path = Path(args.registry or DEFAULT_REGISTRY)
    if not registry_path.is_absolute():
        registry_path = REPO_ROOT / registry_path
    try:
        registry = (
            load_registry(registry_path)
            if registry_path.is_dir()
            else _load_single_file(registry_path)
        )
    except RegistryError as exc:
        sys.stderr.write(f"[audit-sources] registry error: {exc}\n")
        return 2

    rows: list[dict[str, Any]] = []
    for source_id in registry.source_ids():
        entry = registry.entries[source_id]
        status = " ".join(str(entry.payload.get("status", "UNKNOWN")).split())
        pinned = _pinned_ref(entry.payload)
        smoke_note: str | None = None
        if args.native_test:
            fn, smoke_note = _smoke_entrypoint(source_id)
            if fn is not None:
                smoke_note = f"exit={fn()}"
        rows.append(
            {"source_id": source_id, "status": status, "pinned_ref": pinned,
             "shard": entry.shard, "smoke": smoke_note}
        )

    print(f"registry: {registry.registry_dir}")
    print(f"shards_loaded={len(registry.shards_loaded)} broken={len(registry.broken_shards)}")
    header = f"{'source_id':<22} {'status':<28} pinned_ref"
    print(header + ("  smoke" if args.native_test else ""))
    for row in rows:
        print(_row(row["source_id"], row["status"], row["pinned_ref"], row["smoke"]))
    for broken in registry.broken_shards:
        print(f"BROKEN_SHARD {broken.shard}: {broken.error}", file=sys.stderr)

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        report = {
            "registry": str(registry_path),
            "shards_loaded": list(registry.shards_loaded),
            "broken_shards": [{"shard": b.shard, "error": b.error}
                              for b in registry.broken_shards],
            "sources": rows,
        }
        (out_dir / "source_audit.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return 0
