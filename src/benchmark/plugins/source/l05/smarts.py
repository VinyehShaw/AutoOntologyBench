"""L05 source plugin: parse the pinned RDKit functional-group SMARTS hierarchy.

Two parsers, one cross-check (Build-order §4.3 independence rule):
- ``parse_hierarchy`` is this lane's own strict tab/dotted-label parser.
- ``cross_check_with_rdkit`` rebuilds the same hierarchy through RDKit's own
  ``rdkit.Chem.FunctionalGroups.BuildFuncGroupHierarchy`` and requires
  label/smarts/parent agreement on every node.

Every group SMARTS must compile via ``Chem.MolFromSmarts`` at parse time
(fail-closed: a group that the pinned native engine cannot parse is a source
defect, never silently dropped).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import FunctionalGroups as _FG

_TAB_SPLIT = re.compile(r"\t+")


@dataclass(frozen=True)
class FGroup:
    """One node of the RDKit functional-group hierarchy."""

    label: str          # dotted path, e.g. "Amine.Primary.Aromatic"
    smarts: str         # query SMARTS (first atom is the distinguishing feature)
    name: str           # human-readable name column
    depth: int          # 0 = top-level group
    parent_label: str | None
    rxn_smarts: str | None  # optional removal reaction (not used for matching)

    @property
    def digest(self) -> str:
        blob = f"{self.label}\t{self.smarts}\t{self.name}".encode()
        return hashlib.sha256(blob).hexdigest()


class HierarchyParseError(ValueError):
    pass


def parse_hierarchy(text: str) -> tuple[FGroup, ...]:
    """Strictly parse Functional_Group_Hierarchy.txt content (fail-closed)."""
    groups: list[FGroup] = []
    seen: dict[str, FGroup] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//")[0].strip()
        if not line:
            continue
        parts = _TAB_SPLIT.split(line.strip())
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) < 3:
            raise HierarchyParseError(f"line {line_no}: <3 tab-separated fields: {raw!r}")
        label, smarts, name = parts[0], parts[1], parts[2]
        if label in seen:
            raise HierarchyParseError(f"line {line_no}: duplicate label {label!r}")
        chain = label.split(".")
        parent_label = ".".join(chain[:-1]) or None
        if parent_label is not None and parent_label not in seen:
            raise HierarchyParseError(
                f"line {line_no}: parent {parent_label!r} of {label!r} not defined earlier")
        if Chem.MolFromSmarts(smarts) is None:
            raise HierarchyParseError(f"line {line_no}: unparsable SMARTS {smarts!r}")
        node = FGroup(label=label, smarts=smarts, name=name, depth=len(chain) - 1,
                      parent_label=parent_label,
                      rxn_smarts=parts[3] if len(parts) > 3 else None)
        seen[label] = node
        groups.append(node)
    if not groups:
        raise HierarchyParseError("empty hierarchy")
    return tuple(groups)


def cross_check_with_rdkit(text: str, groups: tuple[FGroup, ...]) -> None:
    """Rebuild via RDKit's own parser; require full structural agreement."""
    ref_nodes = _FG.BuildFuncGroupHierarchy(data=text, force=True)
    flat: dict[str, object] = {}

    def walk(node: object) -> None:
        flat[node.label] = node  # type: ignore[attr-defined]
        for ch in node.children:  # type: ignore[attr-defined]
            walk(ch)

    for top in ref_nodes:
        walk(top)
    ours = {g.label: g for g in groups}
    if set(ours) != set(flat):
        missing = set(flat) - set(ours)
        extra = set(ours) - set(flat)
        raise HierarchyParseError(
            f"label set mismatch vs rdkit parser: missing={sorted(missing)} extra={sorted(extra)}")
    for label, g in ours.items():
        ref = flat[label]
        if ref.smarts != g.smarts:  # type: ignore[attr-defined]
            raise HierarchyParseError(f"{label}: smarts mismatch vs rdkit parser")
        ref_parent_ref = ref.parent  # type: ignore[attr-defined]
        ref_parent_node = ref_parent_ref() if ref_parent_ref is not None else None
        ref_parent = ref_parent_node.label if ref_parent_node is not None else None
        if ref_parent != g.parent_label:
            raise HierarchyParseError(f"{label}: parent mismatch vs rdkit parser")


def parse_flat(text: str) -> tuple[FGroup, ...]:
    """Parse the flat ``FunctionalGroups.txt`` list (label/TAB SMARTS/TAB note).

    Labels in this file are SMILES-fragment-like (e.g. ``-C(=O)O``) and have
    no hierarchy; ``parent_label`` is always None. Returned labels are
    prefixed ``flat:`` at call sites to avoid collisions with hierarchy labels.
    """
    groups: list[FGroup] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.split("//")[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in _TAB_SPLIT.split(line.strip()) if p.strip()]
        if len(parts) < 2:
            raise HierarchyParseError(f"flat line {line_no}: <2 fields: {raw!r}")
        label, smarts = parts[0], parts[1]
        name = parts[2] if len(parts) > 2 else label
        if label in seen:
            raise HierarchyParseError(f"flat line {line_no}: duplicate label {label!r}")
        if Chem.MolFromSmarts(smarts) is None:
            raise HierarchyParseError(f"flat line {line_no}: unparsable SMARTS {smarts!r}")
        seen.add(label)
        groups.append(FGroup(label=label, smarts=smarts, name=name, depth=0,
                             parent_label=None, rxn_smarts=None))
    return tuple(groups)


def descendants(groups: tuple[FGroup, ...], label: str) -> tuple[str, ...]:
    """All labels strictly below ``label`` in the hierarchy (transitive)."""
    children: dict[str | None, list[str]] = {}
    for g in groups:
        children.setdefault(g.parent_label, []).append(g.label)
    out: list[str] = []
    stack = list(children.get(label, ()))
    while stack:
        cur = stack.pop()
        out.append(cur)
        stack.extend(children.get(cur, ()))
    return tuple(sorted(out))


def top_level(groups: tuple[FGroup, ...]) -> tuple[FGroup, ...]:
    return tuple(g for g in groups if g.parent_label is None)
