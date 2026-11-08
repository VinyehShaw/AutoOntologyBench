"""L04 KB access layer: lazy, memory-bounded views over the built index shards.

Only small derived maps are materialized; the participant table (165k rows) is
loaded once as tuples and shared. Nothing here touches the network.
"""
from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Iterator

INDEX_SUBDIR = "indexes"


def _read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


@dataclass(frozen=True)
class L04KB:
    """In-memory derived views over the pinned L04 index shards."""

    root: Path
    participants: tuple[tuple[str, str, str, str, str], ...]
    reactions: dict[str, dict[str, Any]]
    xrefs: tuple[dict[str, str], ...]
    go_links: tuple[dict[str, str], ...]
    directions: tuple[dict[str, str], ...]
    relationships: tuple[dict[str, str], ...]
    smiles: dict[str, str]
    names: dict[str, str]
    ph73: tuple[dict[str, str], ...]
    chebi_parents: dict[str, tuple[str, ...]]
    chebi_labels: dict[str, str]
    chebi_deprecated: frozenset[str]
    index_manifest: dict[str, Any] = field(default_factory=dict)

    # -- derived views ------------------------------------------------------
    @cached_property
    def chebi_to_reactions(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for rhea_id, _side, _p, _c, chebi in self.participants:
            out.setdefault(chebi, set()).add(rhea_id)
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def reaction_to_chebis(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for rhea_id, _side, _p, _c, chebi in self.participants:
            out.setdefault(rhea_id, set()).add(chebi)
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def master_ids(self) -> frozenset[str]:
        return frozenset(r for r, u in self.reactions.items() if u["kind"] == "master")

    @cached_property
    def go_by_master(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for row in self.go_links:
            out.setdefault("RHEA:" + row["MASTER_ID"], set()).add(row["ID"])
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def master_to_go(self) -> dict[str, frozenset[str]]:
        return self.go_by_master

    @cached_property
    def go_to_masters(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for row in self.go_links:
            out.setdefault(row["ID"], set()).add("RHEA:" + row["MASTER_ID"])
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def xrefs_by_master(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for row in self.xrefs:
            out.setdefault("RHEA:" + row["MASTER_ID"], set()).add(
                f'{row["DB"]}:{row["ID"]}')
        return {k: frozenset(v) for k, v in out.items()}

    @cached_property
    def ec_by_master(self) -> dict[str, frozenset[str]]:
        out: dict[str, set[str]] = {}
        for row in self.xrefs:
            if row["DB"] == "EC":
                out.setdefault("RHEA:" + row["MASTER_ID"], set()).add(row["ID"])
        return {k: frozenset(v) for k, v in out.items()}

    def chebi_ancestors(self, chebi: str) -> frozenset[str]:
        """Transitive rdfs:subClassOf closure over the pinned chebi_lite slice."""
        seen: set[str] = set()
        stack = list(self.chebi_parents.get(chebi, ()))
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(self.chebi_parents.get(node, ()))
        return frozenset(seen)

    def chebi_descendants_in_universe(self, ancestor: str,
                                      universe: frozenset[str]) -> frozenset[str]:
        return frozenset(c for c in universe
                         if ancestor in self.chebi_ancestors(c))


@lru_cache(maxsize=2)
def load_kb(pin_root: str) -> L04KB:
    root = Path(pin_root)
    idx = root / INDEX_SUBDIR

    def shard(kind: str) -> Path:
        return idx / f"{kind}.jsonl.gz"

    participants = tuple(
        (r["rhea_id"], r["side"], r["participant"], r["compound"], r["chebi"])
        for r in _read_jsonl_gz(shard("rhea_participant"))
    )
    reactions = {r["rhea_id"]: r for r in _read_jsonl_gz(shard("rhea_reaction"))}
    xrefs = tuple(_read_jsonl_gz(shard("rhea_xref")))
    go_links = tuple(_read_jsonl_gz(shard("rhea_go_link")))
    directions = tuple(_read_jsonl_gz(shard("rhea_direction_row")))
    relationships = tuple(_read_jsonl_gz(shard("rhea_relationship")))
    smiles = {r["CHEBI"]: r["SMILES"] for r in _read_jsonl_gz(shard("rhea_chebi_smiles"))}
    names = {r["CHEBI_ID"]: r["NAME"] for r in _read_jsonl_gz(shard("chebi_name_row"))}
    ph73 = tuple(_read_jsonl_gz(shard("chebi_ph73_mapping")))
    parents: dict[str, tuple[str, ...]] = {}
    labels: dict[str, str] = {}
    deprecated: set[str] = set()
    for e in _read_jsonl_gz(shard("chebi_entity")):
        parents[e["chebi"]] = tuple(e["parents"])
        labels[e["chebi"]] = e["label"]
        if e["deprecated"]:
            deprecated.add(e["chebi"])
    manifest = json.loads((idx / "index_manifest.json").read_text())
    return L04KB(
        root=root, participants=participants, reactions=reactions, xrefs=xrefs,
        go_links=go_links, directions=directions, relationships=relationships,
        smiles=smiles, names=names, ph73=ph73, chebi_parents=parents,
        chebi_labels=labels, chebi_deprecated=frozenset(deprecated),
        index_manifest=manifest,
    )
