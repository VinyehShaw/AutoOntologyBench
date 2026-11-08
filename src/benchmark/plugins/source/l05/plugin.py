"""L05 source plugin: pin/index the RDKit hierarchy + ChEBI structures, enumerate units.

``build_index`` is the heavy stage: it parses the pinned SDF once, dedups by
InChIKey, then computes every hierarchy group's match count on every molecule
with a process pool (canonical-SMILES interchange, deterministic). Outputs
under ``external_knowledge/indexes/rdkit-molecules/<release_id>/``:

- ``hierarchy.json``      — parsed + cross-checked group tree (with digests)
- ``molecules.jsonl``     — one MoleculeRecord per line (deduped corpus)
- ``membership.json``     — group label -> sorted [chebi_id] (match_count >= 1)
- ``match_counts.json``   — group label -> {chebi_id: n_matches} (nonzero only)
- ``manifest.json``       — versions, input sha256s, counts, timings
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from rdkit import Chem, __version__ as RDKIT_VERSION

from benchmark.schema.plugin_protocols import (
    FetchReceipt, IndexManifest, SourceSnapshot, SourceUnit,
)
from benchmark.plugins.source.l05.molecules import (
    MoleculeRecord, dedup_records, iter_molecule_records,
)
from benchmark.plugins.source.l05.smarts import (
    FGroup, cross_check_with_rdkit, parse_flat, parse_hierarchy,
)
from source_registry.fetchers.l05_rdkit import RELEASE_ID, SOURCE_ID, RDKitMoleculesSource

_PATTERNS: dict[str, str] = {}


def _init_patterns(patterns: dict[str, str]) -> None:
    global _PATTERNS
    _PATTERNS = patterns


def _match_chunk(chunk: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Return {label: {chebi_id: n_matches}} for [(chebi_id, canonical_smiles)]."""
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    pats = {label: Chem.MolFromSmarts(s) for label, s in _PATTERNS.items()}
    out: dict[str, dict[str, int]] = {}
    for chebi_id, smiles in chunk:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for label, pat in pats.items():
            n = len(mol.GetSubstructMatches(pat, maxMatches=1000))
            if n:
                out.setdefault(label, {})[chebi_id] = n
    return out


class L05SourcePlugin:
    source_id = SOURCE_ID

    def __init__(self, release_root: Path | None = None) -> None:
        src = RDKitMoleculesSource()
        self.release_root = Path(release_root) if release_root else src.release_dir

    @property
    def hierarchy_path(self) -> Path:
        return self.release_root / "rdkit" / "Data" / "Functional_Group_Hierarchy.txt"

    @property
    def flat_path(self) -> Path:
        return self.release_root / "rdkit" / "Data" / "FunctionalGroups.txt"

    @property
    def sdf_path(self) -> Path:
        return self.release_root / "chebi" / "SDF" / "chebi_3_stars.sdf.gz"

    def load_hierarchy(self) -> tuple[FGroup, ...]:
        text = self.hierarchy_path.read_text(encoding="utf-8")
        groups = parse_hierarchy(text)
        cross_check_with_rdkit(text, groups)
        flat = [FGroup(label=f"flat:{g.label}", smarts=g.smarts, name=g.name,
                       depth=0, parent_label=None, rxn_smarts=None)
                for g in parse_flat(self.flat_path.read_text(encoding="utf-8"))]
        return tuple(groups) + tuple(flat)

    def load_molecules(self) -> tuple[list[MoleculeRecord], dict[str, int]]:
        return dedup_records(iter_molecule_records(self.sdf_path))

    def enumerate_units(self, snapshot: SourceSnapshot | None = None) -> Iterator[SourceUnit]:
        for g in self.load_hierarchy():
            yield SourceUnit(unit_id=f"fg:{g.label}", kind="smarts_group",
                             payload_ref=str(self.hierarchy_path), digest=g.digest,
                             metadata={"label": g.label, "smarts": g.smarts,
                                       "depth": g.depth, "parent": g.parent_label})
        mols, stats = self.load_molecules()
        for m in mols:
            yield SourceUnit(unit_id=f"mol:{m.chebi_id}", kind="molecule",
                             payload_ref=str(self.sdf_path), digest=m.digest,
                             metadata={"name": m.name, "smiles": m.canonical_smiles,
                                       "inchikey": m.inchikey, "heavy_atoms": m.heavy_atoms})

    @property
    def index_dir(self) -> Path:
        return Path("external_knowledge/indexes") / SOURCE_ID / RELEASE_ID

    def pin_and_index(self, out_dir: Path | None = None, *,
                      verify: bool = True) -> IndexManifest:
        """§3.2 factory surface: verify pinned bytes in place, then index them."""
        src = RDKitMoleculesSource()
        receipt_path = (src.ek_root / "fetch_receipts" / SOURCE_ID
                        / f"{RELEASE_ID}.json")
        if verify and receipt_path.is_file():
            receipt = FetchReceipt(
                **json.loads(receipt_path.read_text())["receipt"])
            report = src.verify_snapshot(receipt)
            if not report.ok:
                raise RuntimeError(
                    f"pinned snapshot failed verification: {report.missing}")
        target = Path(out_dir) if out_dir else self.index_dir
        manifest_path = target / "manifest.json"
        if manifest_path.is_file():
            m = json.loads(manifest_path.read_text())
            return IndexManifest(
                source_id=SOURCE_ID, release_id=RELEASE_ID, out_dir=str(target),
                unit_count=m["hierarchy_groups"] + m["molecules_total"],
                kind_counts={"smarts_group": m["hierarchy_groups"],
                             "molecule": m["molecules_total"]})
        return self.build_index(target)

    def build_index(self, out_dir: Path, *, workers: int | None = None,
                    chunk_size: int = 500) -> IndexManifest:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        groups = self.load_hierarchy()
        (out_dir / "hierarchy.json").write_text(json.dumps(
            {"rdkit_version": RDKIT_VERSION, "source": str(self.hierarchy_path),
             "groups": [asdict(g) | {"digest": g.digest} for g in groups]},
            indent=1, sort_keys=True) + "\n")
        t1 = time.time()
        mols, stats = self.load_molecules()
        with (out_dir / "molecules.jsonl").open("w", encoding="utf-8") as fh:
            for m in mols:
                fh.write(json.dumps(asdict(m), sort_keys=True) + "\n")
        t2 = time.time()
        patterns = {g.label: g.smarts for g in groups}
        pairs = [(m.chebi_id, m.canonical_smiles) for m in mols]
        chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
        merged: dict[str, dict[str, int]] = {}
        workers = workers or max(1, min(32, (os.cpu_count() or 4) // 2))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_patterns,
                                 initargs=(patterns,)) as pool:
            for partial in pool.map(_match_chunk, chunks):
                for label, hits in partial.items():
                    merged.setdefault(label, {}).update(hits)
        t3 = time.time()
        membership = {label: sorted(hits) for label, hits in sorted(merged.items())}
        (out_dir / "membership.json").write_text(
            json.dumps(membership, sort_keys=True) + "\n")
        (out_dir / "match_counts.json").write_text(
            json.dumps({k: dict(sorted(v.items())) for k, v in sorted(merged.items())},
                       sort_keys=True) + "\n")
        manifest = {
            "source_id": self.source_id, "release_id": RELEASE_ID,
            "rdkit_version": RDKIT_VERSION,
            "hierarchy_groups": len(groups),
            "molecules_total": len(mols),
            "molecule_stats": stats,
            "group_nonempty": sum(1 for v in membership.values() if v),
            "timings_s": {"hierarchy": round(t1 - t0, 2), "molecules": round(t2 - t1, 2),
                          "match": round(t3 - t2, 2)},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        return IndexManifest(
            source_id=self.source_id, release_id=RELEASE_ID, out_dir=str(out_dir),
            unit_count=len(groups) + len(mols),
            kind_counts={"smarts_group": len(groups), "molecule": len(mols)})

    # convenience loader for downstream plugins
    @staticmethod
    def load_index(out_dir: Path) -> dict[str, Any]:
        out_dir = Path(out_dir)
        with (out_dir / "molecules.jsonl").open(encoding="utf-8") as fh:
            mols = [json.loads(line) for line in fh]
        return {
            "manifest": json.loads((out_dir / "manifest.json").read_text()),
            "hierarchy": json.loads((out_dir / "hierarchy.json").read_text())["groups"],
            "molecules": mols,
            "membership": json.loads((out_dir / "membership.json").read_text()),
            "match_counts": json.loads((out_dir / "match_counts.json").read_text()),
        }
