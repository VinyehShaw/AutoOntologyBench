"""L05 source plugin: stream the pinned ChEBI 3-star SDF into typed molecule units.

Parse-don't-validate at the boundary: every molecule crosses exactly once here,
becoming an immutable :class:`MoleculeRecord` with a natively recomputed
canonical SMILES (the SDF-supplied ``smiles`` property is only a cross-check,
never the truth). Deduplication is by InChIKey (fallback: canonical SMILES);
duplicates are counted, never silently merged into identity claims.
"""
from __future__ import annotations

import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ChEBI SDF property names have changed spelling across releases; probe both.
_ID_KEYS = ("chebi_id", "ChEBI ID", "CHEBI_ID")
_NAME_KEYS = ("chebi_name", "ChEBI Name", "CHEBI_NAME")
_SMILES_KEYS = ("smiles", "SMILES")
_IK_KEYS = ("inchikey", "InChIKey", "inchi_key")


@dataclass(frozen=True)
class MoleculeRecord:
    chebi_id: str            # e.g. "CHEBI:15377"
    name: str
    canonical_smiles: str    # recomputed natively from the mol block
    inchikey: str
    heavy_atoms: int
    sdf_smiles_agree: bool   # cross-check vs the SDF-supplied SMILES field

    @property
    def digest(self) -> str:
        blob = f"{self.chebi_id}|{self.canonical_smiles}".encode()
        return hashlib.sha256(blob).hexdigest()


def _first_prop(mol: Chem.Mol, keys: tuple[str, ...]) -> str:
    props = set(mol.GetPropNames())
    for k in keys:
        if k in props:
            return mol.GetProp(k).strip()
    return ""


def iter_molecule_records(sdf_gz: Path) -> Iterator[MoleculeRecord | str]:
    """Yield MoleculeRecord for parseable entries; yield "SKIP:<reason>" else."""
    with gzip.open(sdf_gz, "rb") as fh:
        suppl = Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=True)
        for mol in suppl:
            if mol is None:
                yield "SKIP:unparseable_molblock"
                continue
            chebi_id = _first_prop(mol, _ID_KEYS)
            if not chebi_id.startswith("CHEBI:"):
                yield "SKIP:missing_chebi_id"
                continue
            canonical = Chem.MolToSmiles(mol, canonical=True)
            if not canonical:
                yield "SKIP:empty_canonical_smiles"
                continue
            sdf_smiles = _first_prop(mol, _SMILES_KEYS)
            agree = False
            if sdf_smiles:
                ref = Chem.MolFromSmiles(sdf_smiles)
                agree = ref is not None and Chem.MolToSmiles(ref, canonical=True) == canonical
            yield MoleculeRecord(
                chebi_id=chebi_id,
                name=_first_prop(mol, _NAME_KEYS),
                canonical_smiles=canonical,
                inchikey=_first_prop(mol, _IK_KEYS),
                heavy_atoms=mol.GetNumHeavyAtoms(),
                sdf_smiles_agree=agree,
            )


def dedup_records(records: Iterator[MoleculeRecord | str]) -> tuple[list[MoleculeRecord], dict[str, int]]:
    """Dedup by InChIKey (fallback canonical SMILES); keep first occurrence."""
    stats = {"skipped": 0, "duplicate": 0, "sdf_smiles_disagree": 0}
    seen: set[str] = set()
    out: list[MoleculeRecord] = []
    for item in records:
        if isinstance(item, str):
            stats["skipped"] += 1
            continue
        key = item.inchikey or item.canonical_smiles
        if key in seen:
            stats["duplicate"] += 1
            continue
        seen.add(key)
        if not item.sdf_smiles_agree:
            stats["sdf_smiles_disagree"] += 1
        out.append(item)
    return out, stats
