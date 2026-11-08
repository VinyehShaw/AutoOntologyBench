"""L05 molecule descriptor features (20K expansion, new molecule mechanisms).

The legacy L05 families only use functional-group membership, substructure match
counts and heavy-atom counts. The 20K expansion adds a second molecule axis:
topological/composition descriptors computed per molecule. These are deliberately
cheap integer features so a band query is answerable both from a precomputed
feature cache (``eval_index``) and directly from the RDKit ``Mol`` object
(``eval_native``) for the differential gate.

Only RDKit-native descriptor functions are used (pinned engine version is part
of the native evidence); no external model or lookup table is involved.
"""
from __future__ import annotations

from typing import Any

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

ELEMENT_KEYS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")

#: Public descriptor keys -> short human description (used in task wording).
PROP_LABELS: dict[str, str] = {
    "rings": "total ring count",
    "aromatic_rings": "aromatic ring count",
    "aliphatic_rings": "aliphatic ring count",
    "saturated_rings": "saturated ring count",
    "rotatable_bonds": "rotatable bond count",
    "hbd": "hydrogen-bond donor count",
    "hba": "hydrogen-bond acceptor count",
    "hetero_atoms": "heteroatom count",
    "formal_charge": "net formal charge",
    "stereo_centers": "stereo-center count",
    "heavy_atoms": "heavy-atom count",
    "mw": "molecular weight (Da, rounded)",
    "fraction_csp3": "fraction of sp3 carbons (percent, rounded)",
    **{f"elem_{e}": f"number of {e} atoms" for e in ELEMENT_KEYS},
}


def molecule_features(mol: Chem.Mol) -> dict[str, int]:
    """Compute the L05 expansion feature vector for one parsed molecule."""
    counts = {f"elem_{e}": 0 for e in ELEMENT_KEYS}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        key = f"elem_{sym}"
        if key in counts:
            counts[key] += 1
    feats: dict[str, int] = {
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(mol),
        "saturated_rings": rdMolDescriptors.CalcNumSaturatedRings(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "hetero_atoms": rdMolDescriptors.CalcNumHeteroatoms(mol),
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "stereo_centers": int(rdMolDescriptors.CalcNumAtomStereoCenters(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "mw": int(round(Descriptors.MolWt(mol))),
        "fraction_csp3": int(round(rdMolDescriptors.CalcFractionCSP3(mol) * 100)),
    }
    feats.update(counts)
    return feats


def features_from_smiles(smiles: str) -> dict[str, int] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return molecule_features(mol)