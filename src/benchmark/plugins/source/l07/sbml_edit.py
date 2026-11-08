"""L07 SBML edit operations: typed, libsbml-checked mutations of a pinned model.

Every edit returns new SBML *bytes*; the original pinned file is never touched.
Each edit re-parses the result with libsbml and refuses to return a document
with parse errors, so downstream solver calls always see well-formed SBML.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import libsbml


class SbmlEditError(ValueError):
    pass


@dataclass(frozen=True)
class ParameterEdit:
    parameter_id: str
    new_value: float
    kind: str = "parameter"  # "parameter" | "species_initial" | "kinetic_law"
    species_id: str = ""
    reaction_id: str = ""
    formula: str = ""


def _roundtrip(doc: libsbml.SBMLDocument) -> bytes:
    raw = libsbml.writeSBMLToString(doc).encode("utf-8")
    check = libsbml.readSBMLFromString(raw.decode("utf-8"))
    for i in range(check.getNumErrors()):
        e = check.getError(i)
        if e.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            raise SbmlEditError(f"edit produced invalid SBML: {e.getShortMessage()[:120]}")
    return raw


def apply_edit(sbml_bytes: bytes, edit: ParameterEdit) -> bytes:
    doc = libsbml.readSBMLFromString(sbml_bytes.decode("utf-8"))
    m = doc.getModel()
    if m is None:
        raise SbmlEditError("no model element")
    if edit.kind == "parameter":
        p = m.getParameter(edit.parameter_id)
        if p is None:
            raise SbmlEditError(f"unknown parameter {edit.parameter_id!r}")
        p.setValue(float(edit.new_value))
    elif edit.kind == "species_initial":
        s = m.getSpecies(edit.species_id)
        if s is None:
            raise SbmlEditError(f"unknown species {edit.species_id!r}")
        if s.isSetInitialAmount():
            s.setInitialAmount(float(edit.new_value))
        elif s.isSetInitialConcentration():
            s.setInitialConcentration(float(edit.new_value))
        else:
            s.setInitialConcentration(float(edit.new_value))
    elif edit.kind == "kinetic_law":
        r = m.getReaction(edit.reaction_id)
        if r is None:
            raise SbmlEditError(f"unknown reaction {edit.reaction_id!r}")
        kl = r.getKineticLaw()
        if kl is None:
            raise SbmlEditError(f"reaction {edit.reaction_id!r} has no kinetic law")
        ast = libsbml.parseL3Formula(edit.formula)
        if ast is None:
            raise SbmlEditError("submitted formula does not parse as L3 infix")
        kl.setMath(ast)
    else:
        raise SbmlEditError(f"unknown edit kind {edit.kind!r}")
    return _roundtrip(doc)


def all_reaction_mentions(sbml_bytes: bytes) -> dict[str, list[str]]:
    """Sorted species ids referenced by every reaction's kinetic law, parsed once."""
    doc = libsbml.readSBMLFromString(sbml_bytes.decode("utf-8"))
    m = doc.getModel()
    if m is None:
        raise SbmlEditError("no model element")
    species_ids = {s.getId() for s in m.getListOfSpecies()}
    out: dict[str, list[str]] = {}
    for r in m.getListOfReactions():
        kl = r.getKineticLaw()
        if kl is None or not kl.isSetMath():
            continue
        formula = libsbml.formulaToL3String(kl.getMath()) or ""
        names: list[str] = []

        def walk(node: Any) -> None:
            if node is None:
                return
            if node.getType() == libsbml.AST_NAME:
                name = node.getName()
                if name in species_ids and name not in names:
                    names.append(name)
            for i in range(node.getNumChildren()):
                walk(node.getChild(i))

        walk(kl.getMath())
        if not names:
            names = sorted(s for s in species_ids if s in formula)
        out[r.getId()] = sorted(names)
    return out


def species_formula_mentions(sbml_bytes: bytes, reaction_id: str) -> list[str]:
    """Sorted species ids referenced by a reaction's kinetic-law formula."""
    doc = libsbml.readSBMLFromString(sbml_bytes.decode("utf-8"))
    m = doc.getModel()
    if m is None:
        raise SbmlEditError("no model element")
    r = m.getReaction(reaction_id)
    if r is None or r.getKineticLaw() is None or not r.getKineticLaw().isSetMath():
        return []
    species_ids = {s.getId() for s in m.getListOfSpecies()}
    formula = libsbml.formulaToL3String(r.getKineticLaw().getMath()) or ""
    out: list[str] = []

    def walk(node: Any) -> None:
        if node is None:
            return
        if node.getType() == libsbml.AST_NAME:
            name = node.getName()
            if name in species_ids and name not in out:
                out.append(name)
        for i in range(node.getNumChildren()):
            walk(node.getChild(i))

    walk(r.getKineticLaw().getMath())
    if not out:
        # fall back to substring scan over the L3 rendering
        out = sorted(s for s in species_ids if s in formula)
    return sorted(out)
