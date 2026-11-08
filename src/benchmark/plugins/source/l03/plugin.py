"""L03 source plugin: pin/index the two pinned Camunda repos, enumerate DMN units.

``build_index`` parses every ``*.dmn`` / ``*.dmn11.xml`` in both pinned releases
with the ``aob-dmn-static`` runtime, FEEL-checks every rule entry, extracts the
DRG (required-decision graph, with cycle detection), and writes a combined
index under ``external_knowledge/indexes/camunda-dmn-examples/combined/``:

- ``tables.jsonl``   — one fully-parsed decision table per line (usable units)
- ``chains.jsonl``   — one DRG chain per line (files with >=2 usable decisions
                       linked by informationRequirements, acyclic)
- ``rejects.jsonl``  — per-file/per-table honest exclusion reasons
- ``index.json``     — counts, hit-policy distribution, timings, input digests

Demo tables (camunda-bpm-examples) and engine-test tables (camunda-bpm-platform)
carry a ``constituent_role`` field so reporting keeps the two apart (build
order L03 constraint: demo vs real external tables are reported separately).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterator

from benchmark.plugins.verifier.l03.dmn_model import (
    DMN_SUFFIXES,
    DmnParseError,
    DmnUnsupportedError,
    parse_definitions,
    table_digest,
)
from benchmark.plugins.verifier.l03.feel import (
    FeelUnsupportedError,
    eval_output_entry,
    match_unary,
)
from benchmark.schema.plugin_protocols import IndexManifest, SourceSnapshot, SourceUnit
from source_registry.fetchers import l03_camunda_dmn as fetcher

LANE_ID = "L03"
SOURCE_ID = fetcher.SOURCE_ID
INDEX_DIR = fetcher.INDEX_BASE / "combined"


class L03SourcePlugin:
    """SourcePlugin for lane L03 (Camunda DMN decision tables)."""

    plugin_id = "l03.camunda-dmn"

    def pin_and_index(self, source_config: dict[str, Any]) -> list[SourceSnapshot]:
        snapshots = []
        for release in fetcher.CamundaDmnExamplesSource().discover_releases():
            root = fetcher.pinned_root(release.release_id)
            if not root.is_dir():
                if not source_config.get("fetch_if_missing"):
                    raise FileNotFoundError(
                        "pinned snapshot missing: %s (run source fetch first)" % root
                    )
                fetcher.CamundaDmnExamplesSource().fetch(
                    release, resume=True, cache_dir=fetcher.PINNED_BASE / "_cache"
                )
            snapshots.append(fetcher.load_snapshot(release.release_id))
        return snapshots

    def enumerate_units(self, snapshot: SourceSnapshot | None = None) -> Iterator[SourceUnit]:
        snapshots = [snapshot] if snapshot is not None else fetcher.load_snapshots()
        for snap in snapshots:
            yield from fetcher.CamundaDmnExamplesSource().inventory(snap)

    # -- index build ------------------------------------------------------------

    def build_index(self, out_dir: Path = INDEX_DIR) -> IndexManifest:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        tables: list[dict[str, Any]] = []
        rejects: list[dict[str, Any]] = []
        for release in fetcher.CamundaDmnExamplesSource().discover_releases():
            root = fetcher.pinned_root(release.release_id)
            role = _role_of(release.release_id)
            for path in sorted(root.rglob("*")):
                if not (path.is_file() and path.name.endswith(DMN_SUFFIXES)):
                    continue
                rel = str(path.relative_to(root))
                unit_id = "%s/%s/%s" % (SOURCE_ID, release.release_id, rel)
                try:
                    defs = parse_definitions(path)
                except DmnParseError as exc:
                    rejects.append({"unit": unit_id, "reason": "parse_error", "detail": str(exc)[:200]})
                    continue
                except DmnUnsupportedError as exc:
                    rejects.append({"unit": unit_id, "reason": "profile_unsupported", "detail": str(exc)[:200]})
                    continue
                file_decisions: dict[str, dict[str, Any]] = {}
                for dec_id, dec in defs.decisions.items():
                    if dec.table is None:
                        continue
                    ok, reason = _table_profile_check(dec.table)
                    if not ok:
                        rejects.append({
                            "unit": unit_id, "decision": dec_id,
                            "reason": "profile_unsupported", "detail": reason,
                        })
                        continue
                    row = {
                        "unit_id": unit_id,
                        "release_id": release.release_id,
                        "constituent_role": role,
                        "path": rel,
                        "decision_id": dec_id,
                        "decision_name": dec.name,
                        "table_digest": table_digest(dec.table),
                        "hit_policy": dec.table.hit_policy,
                        "aggregation": dec.table.aggregation,
                        "inputs": [
                            {"expression": c.expression, "type_ref": c.type_ref,
                             "label": c.label}
                            for c in dec.table.inputs
                        ],
                        "outputs": [
                            {"name": o.name, "type_ref": o.type_ref,
                             "values": list(o.values)}
                            for o in dec.table.outputs
                        ],
                        "rules": [
                            {"id": r.id, "input_entries": list(r.input_entries),
                             "output_entries": list(r.output_entries),
                             "description": r.description}
                            for r in dec.table.rules
                        ],
                        "required_decision_ids": list(dec.required_decision_ids),
                    }
                    tables.append(row)
                    file_decisions[dec_id] = row
        tables_path = out_dir / "tables.jsonl"
        with tables_path.open("w", encoding="utf-8") as fh:
            for row in sorted(tables, key=lambda r: (r["unit_id"], r["decision_id"])):
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        rejects_path = out_dir / "rejects.jsonl"
        with rejects_path.open("w", encoding="utf-8") as fh:
            for row in rejects:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
        hit_dist: dict[str, int] = {}
        for row in tables:
            key = row["hit_policy"] + ("+" + row["aggregation"] if row["aggregation"] else "")
            hit_dist[key] = hit_dist.get(key, 0) + 1
        chains = _usable_chains(tables, rejects)
        with (out_dir / "chains.jsonl").open("w", encoding="utf-8") as fh:
            for chain in chains:
                fh.write(json.dumps(chain, sort_keys=True) + "\n")
        manifest = {
            "source_id": SOURCE_ID,
            "release_ids": [r.release_id for r in
                            fetcher.CamundaDmnExamplesSource().discover_releases()],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "engine": fetcher.NATIVE_ENGINE,
            "tables_usable": len(tables),
            "tables_demo": sum(1 for r in tables if r["constituent_role"] == "demo"),
            "tables_engine_test": sum(
                1 for r in tables if r["constituent_role"] == "engine-test-tables"
            ),
            "chains_usable": len(chains),
            "rejects": len(rejects),
            "hit_policy_distribution": hit_dist,
            "wall_seconds": round(time.time() - t0, 2),
        }
        (out_dir / "index.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        return IndexManifest(
            source_id=SOURCE_ID,
            release_id="combined",
            out_dir=str(out_dir),
            unit_count=len(tables),
            kind_counts={"decision_table": len(tables), "chain": len(chains)},
        )

    @staticmethod
    def load_index(out_dir: Path = INDEX_DIR) -> dict[str, Any]:
        out_dir = Path(out_dir)
        with (out_dir / "tables.jsonl").open(encoding="utf-8") as fh:
            tables = [json.loads(line) for line in fh]
        with (out_dir / "chains.jsonl").open(encoding="utf-8") as fh:
            chains = [json.loads(line) for line in fh]
        return {
            "manifest": json.loads((out_dir / "index.json").read_text()),
            "tables": tables,
            "chains": chains,
        }


def _role_of(release_id: str) -> str:
    for c in fetcher.CONSTITUENTS:
        if c["release_id"] == release_id:
            return c["role"]
    return "unknown"


def _table_profile_check(table: Any) -> tuple[bool, str]:
    """True when every rule entry is inside the supported FEEL subset."""
    for rule in table.rules:
        for clause, entry in zip(table.inputs, rule.input_entries):
            try:
                # Probe with a neutral value; unsupported syntax raises.
                match_unary(entry, None, clause.type_ref)
            except FeelUnsupportedError as exc:
                return False, "inputEntry %r: %s" % (entry, str(exc)[:120])
        for entry in rule.output_entries:
            try:
                eval_output_entry(entry, {})
            except FeelUnsupportedError as exc:
                return False, "outputEntry %r: %s" % (entry, str(exc)[:200])
    return True, ""


def _usable_chains(tables: list[dict[str, Any]],
                   rejects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group usable tables of one file into DRG chains (acyclic only)."""
    by_unit: dict[str, list[dict[str, Any]]] = {}
    for row in tables:
        by_unit.setdefault(row["unit_id"], []).append(row)
    chains: list[dict[str, Any]] = []
    for unit_id, rows in sorted(by_unit.items()):
        if len(rows) < 2:
            continue
        ids = {r["decision_id"] for r in rows}
        edges = {
            r["decision_id"]: [d for d in r["required_decision_ids"] if d in ids]
            for r in rows
        }
        if not any(edges.values()):
            continue
        order = _toposort(edges)
        if order is None:
            rejects.append({
                "unit": unit_id,
                "reason": "profile_unsupported",
                "detail": "cyclic required-decision graph",
            })
            continue
        terminals = [d for d in order if not any(d in deps for deps in edges.values())]
        chains.append({
            "unit_id": unit_id,
            "release_id": rows[0]["release_id"],
            "constituent_role": rows[0]["constituent_role"],
            "path": rows[0]["path"],
            "decision_ids": order,
            "terminal_decision_ids": terminals,
            "edges": edges,
        })
    return chains


def _toposort(edges: dict[str, list[str]]) -> list[str] | None:
    """Dependency-first order (dependencies before dependents); None if cyclic."""
    order: list[str] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(node: str) -> bool:
        if node in permanent:
            return True
        if node in temporary:
            return False
        temporary.add(node)
        for dep in edges.get(node, []):
            if not visit(dep):
                return False
        temporary.discard(node)
        permanent.add(node)
        order.append(node)
        return True

    for node in sorted(edges):
        if not visit(node):
            return None
    return order


__all__ = ["L03SourcePlugin", "INDEX_DIR"]
