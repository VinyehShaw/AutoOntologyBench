"""L07 libsbml structure extraction: typed summary of a pinned SBML model.

Pure libsbml parse (no solver). The summary is the shared vocabulary the
task-family plugin proposes over and the native verifier recomputes from the
pinned bytes; nothing here is trusted blindly downstream.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import libsbml

LIBSBML_VERSION = libsbml.getLibSBMLDottedVersion()


class SbmlStructError(ValueError):
    """Model is not usable for L07 tasks (parse errors or missing structure)."""


@dataclass(frozen=True)
class SpeciesInfo:
    sid: str
    name: str
    compartment: str
    initial_amount: float | None
    initial_concentration: float | None
    constant: bool
    boundary: bool
    has_only_substance_units: bool


@dataclass(frozen=True)
class ReactionInfo:
    rid: str
    reversible: bool
    reactants: tuple[tuple[str, float], ...]
    products: tuple[tuple[str, float], ...]
    modifiers: tuple[str, ...]
    kinetic_formula: str | None
    local_parameters: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ModelStruct:
    model_id: str
    name: str
    level: int
    version: int
    sha256: str
    species: tuple[SpeciesInfo, ...]
    reactions: tuple[ReactionInfo, ...]
    parameters: tuple[tuple[str, float, bool], ...]  # (pid, value, constant)
    compartments: tuple[tuple[str, float], ...]
    n_rules: int
    n_events: int
    n_function_definitions: int
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    @property
    def floating_species(self) -> list[str]:
        return [s.sid for s in self.species if not s.constant and not s.boundary]

    @property
    def boundary_species(self) -> list[str]:
        return [s.sid for s in self.species if s.boundary]

    @property
    def settable_parameters(self) -> list[str]:
        return [pid for pid, _, const in self.parameters if not const]

    def parameter_value(self, pid: str) -> float:
        for p, v, _ in self.parameters:
            if p == pid:
                return v
        raise KeyError(pid)

    def species_info(self, sid: str) -> SpeciesInfo:
        for s in self.species:
            if s.sid == sid:
                return s
        raise KeyError(sid)

    def reaction_info(self, rid: str) -> ReactionInfo:
        for r in self.reactions:
            if r.rid == rid:
                return r
        raise KeyError(rid)

    def to_json(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "level": self.level,
            "version": self.version,
            "sha256": self.sha256,
            "species": [
                {"sid": s.sid, "name": s.name, "compartment": s.compartment,
                 "initial_amount": s.initial_amount,
                 "initial_concentration": s.initial_concentration,
                 "constant": s.constant, "boundary": s.boundary,
                 "has_only_substance_units": s.has_only_substance_units}
                for s in self.species
            ],
            "reactions": [
                {"rid": r.rid, "reversible": r.reversible,
                 "reactants": [list(x) for x in r.reactants],
                 "products": [list(x) for x in r.products],
                 "modifiers": list(r.modifiers),
                 "kinetic_formula": r.kinetic_formula,
                 "local_parameters": [list(x) for x in r.local_parameters]}
                for r in self.reactions
            ],
            "parameters": [{"pid": p, "value": v, "constant": c}
                           for p, v, c in self.parameters],
            "compartments": [{"cid": c, "size": z} for c, z in self.compartments],
            "n_rules": self.n_rules,
            "n_events": self.n_events,
            "n_function_definitions": self.n_function_definitions,
            "floating_species": self.floating_species,
            "boundary_species": self.boundary_species,
            "settable_parameters": self.settable_parameters,
            "diagnostics": list(self.diagnostics),
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> "ModelStruct":
        return ModelStruct(
            model_id=d["model_id"], name=d["name"], level=d["level"],
            version=d["version"], sha256=d["sha256"],
            species=tuple(SpeciesInfo(
                sid=s["sid"], name=s["name"], compartment=s["compartment"],
                initial_amount=s["initial_amount"],
                initial_concentration=s["initial_concentration"],
                constant=s["constant"], boundary=s["boundary"],
                has_only_substance_units=s["has_only_substance_units"],
            ) for s in d["species"]),
            reactions=tuple(ReactionInfo(
                rid=r["rid"], reversible=r["reversible"],
                reactants=tuple((x[0], float(x[1])) for x in r["reactants"]),
                products=tuple((p[0], float(p[1])) for p in r["products"]),
                modifiers=tuple(r["modifiers"]),
                kinetic_formula=r["kinetic_formula"],
                local_parameters=tuple((lp[0], float(lp[1]))
                                       for lp in r["local_parameters"]),
            ) for r in d["reactions"]),
            parameters=tuple((p["pid"], float(p["value"]), bool(p["constant"]))
                             for p in d["parameters"]),
            compartments=tuple((c["cid"], float(c["size"])) for c in d["compartments"]),
            n_rules=d["n_rules"], n_events=d["n_events"],
            n_function_definitions=d["n_function_definitions"],
            diagnostics=tuple(d.get("diagnostics", ())),
        )


def parse_sbml_bytes(raw: bytes) -> ModelStruct:
    doc = libsbml.readSBMLFromString(raw.decode("utf-8"))
    errors: list[str] = []
    fatal = 0
    for i in range(doc.getNumErrors()):
        e = doc.getError(i)
        if e.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
            fatal += 1
            errors.append(f"{e.getErrorId()}:{e.getShortMessage()[:120]}")
    m = doc.getModel()
    if m is None:
        raise SbmlStructError(f"no model element ({fatal} errors)")
    species: list[SpeciesInfo] = []
    for s in m.getListOfSpecies():
        amt = s.getInitialAmount() if s.isSetInitialAmount() else None
        conc = s.getInitialConcentration() if s.isSetInitialConcentration() else None
        species.append(SpeciesInfo(
            sid=s.getId(), name=s.getName() or "", compartment=s.getCompartment() or "",
            initial_amount=amt, initial_concentration=conc,
            constant=bool(s.getConstant()), boundary=bool(s.getBoundaryCondition()),
            has_only_substance_units=bool(s.getHasOnlySubstanceUnits()),
        ))
    reactions: list[ReactionInfo] = []
    for r in m.getListOfReactions():
        kl = r.getKineticLaw()
        formula = None
        local: list[tuple[str, float]] = []
        if kl is not None:
            if kl.isSetMath():
                formula = libsbml.formulaToL3String(kl.getMath())
            for lp in kl.getListOfLocalParameters():
                local.append((lp.getId(), lp.getValue()))
            for lp in kl.getListOfParameters():
                local.append((lp.getId(), lp.getValue()))
        reactions.append(ReactionInfo(
            rid=r.getId(), reversible=bool(r.getReversible()),
            reactants=tuple((sr.getSpecies(), float(sr.getStoichiometry()))
                            for sr in r.getListOfReactants()),
            products=tuple((sr.getSpecies(), float(sr.getStoichiometry()))
                           for sr in r.getListOfProducts()),
            modifiers=tuple(sr.getSpecies() for sr in r.getListOfModifiers()),
            kinetic_formula=formula,
            local_parameters=tuple(local),
        ))
    params = tuple((p.getId(), float(p.getValue()), bool(p.getConstant()))
                   for p in m.getListOfParameters())
    comps = tuple((c.getId(), float(c.getSize()) if c.isSetSize() else float("nan"))
                  for c in m.getListOfCompartments())
    return ModelStruct(
        model_id=m.getId() or "", name=m.getName() or "",
        level=int(doc.getLevel()), version=int(doc.getVersion()),
        sha256=hashlib.sha256(raw).hexdigest(),
        species=tuple(species), reactions=tuple(reactions), parameters=params,
        compartments=comps, n_rules=int(m.getNumRules()), n_events=int(m.getNumEvents()),
        n_function_definitions=int(m.getNumFunctionDefinitions()),
        diagnostics=tuple(errors),
    )


def parse_sbml_file(path: Path) -> ModelStruct:
    return parse_sbml_bytes(Path(path).read_bytes())
