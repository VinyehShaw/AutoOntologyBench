"""L08 20K-expansion clinical vocabulary + family table (10 NEW families).

All clinical content is SYNTHETIC (same lane constraint as legacy): standard
vocabulary codes (LOINC / SNOMED / RxNorm / HL7 v3 ActCode /
condition-clinical) are used with display names for readability; patient
panels are generated deterministically from seeds (see synth_20k.py). No real
patient data, no real PHI.

The pools EXTEND the legacy vocabulary with codes the legacy lane never used
(plus a subset of legacy codes re-used in genuinely new rule structures);
display names are illustrative — rule text and data always use the same code,
so task semantics never depend on real-world code correctness.

The 10 families below are new mechanisms (see elm_20k module docstring); none
is a re-parameterization of a legacy family.
"""
from __future__ import annotations

from typing import Any

LOINC = "http://loinc.org"
LOINC_NAME = "LOINC"
SNOMED = "http://snomed.info/sct"
SNOMED_NAME = "SNOMED"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
RXNORM_NAME = "RXNORM"
COND_CLINICAL = "http://terminology.hl7.org/CodeSystem/condition-clinical"
V3_ACTCODE = "http://terminology.hl7.org/CodeSystem/v3-ActCode"

#: (code, display, unit, plausible thresholds, default op) — 3 NEW LOINC codes
#: (718-7, 2823-3, 6690-2) + 5 legacy codes re-used inside NEW rule structures.
OBS_POOL_20K: list[dict[str, Any]] = [
    {"code": "718-7", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Hemoglobin [Mass/volume] in Blood", "unit": "g/dL",
     "thresholds": [12.0, 13.5], "op": ">="},
    {"code": "2823-3", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Potassium [Moles/volume] in Serum or Plasma", "unit": "mmol/L",
     "thresholds": [5.0, 5.5], "op": ">="},
    {"code": "6690-2", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Leukocytes [#/volume] in Blood by Automated count", "unit": "10*3/uL",
     "thresholds": [10.0, 11.0], "op": ">="},
    {"code": "8462-4", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Diastolic blood pressure", "unit": "mm[Hg]",
     "thresholds": [80, 85, 90], "op": ">="},
    {"code": "4548-4", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Hemoglobin A1c/Hemoglobin.total in Blood", "unit": "%",
     "thresholds": [5.7, 6.5, 7.0], "op": ">="},
    {"code": "2160-0", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Creatinine [Mass/volume] in Serum or Plasma", "unit": "mg/dL",
     "thresholds": [1.2, 1.5, 2.0], "op": ">="},
    {"code": "1751-7", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Albumin [Mass/volume] in Serum or Plasma", "unit": "g/dL",
     "thresholds": [3.5, 3.0], "op": "<="},
    {"code": "2339-0", "system": LOINC, "cs_name": LOINC_NAME,
     "display": "Glucose [Mass/volume] in Blood", "unit": "mg/dL",
     "thresholds": [100, 110, 126], "op": ">="},
]

#: 4 NEW SNOMED conditions + 2 legacy codes re-used in NEW structures.
COND_POOL_20K: list[dict[str, Any]] = [
    {"code": "84114007", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Heart failure"},
    {"code": "709044004", "system": SNOMED, "cs_name": SNOMED_NAME,
     "display": "Chronic kidney disease"},
    {"code": "35489007", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Depressive disorder"},
    {"code": "313217001", "system": SNOMED, "cs_name": SNOMED_NAME,
     "display": "Chronic pain"},
    {"code": "195967001", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Asthma"},
    {"code": "44054006", "system": SNOMED, "cs_name": SNOMED_NAME,
     "display": "Diabetes mellitus type 2"},
]

#: 3 NEW RxNorm medications (legacy used metformin/atorvastatin/lisinopril).
MED_POOL_20K: list[dict[str, Any]] = [
    {"code": "8664", "system": RXNORM, "cs_name": RXNORM_NAME, "display": "metoprolol"},
    {"code": "11289", "system": RXNORM, "cs_name": RXNORM_NAME, "display": "warfarin"},
    {"code": "36567", "system": RXNORM, "cs_name": RXNORM_NAME, "display": "simvastatin"},
]

#: legacy used AMB/EMER/IMP; HH (home health) is new for the 20K expansion.
ENC_POOL_20K: list[str] = ["AMB", "EMER", "IMP", "HH"]

#: legacy used 90/180/365; the 20K windows differ so scenario calendars differ.
LOOKBACKS_20K: list[int] = [60, 120, 270, 540]

#: plausible age cutoffs for the demographic family (years at reference date)
MIN_AGES_20K: list[int] = [50, 60, 65, 70, 75]

PANEL_SIZE_20K = {"low": 8, "medium": 10, "high": 12}


class FamilySpec20k:
    def __init__(self, family_id: str, kind: str, regime: str, complexity: str,
                 base_goals: int, instances_per_goal: int = 5) -> None:
        self.family_id = family_id
        self.kind = kind
        self.regime = regime
        self.complexity = complexity
        self.base_goals = base_goals
        self.instances_per_goal = instances_per_goal


FAMILY_TABLE_20K: list[FamilySpec20k] = [
    FamilySpec20k("l08-f09-observation-all-values", "obs_all", "application", "medium", 5),
    FamilySpec20k("l08-f10-observation-count-window", "obs_count", "application", "medium", 5),
    FamilySpec20k("l08-f11-observation-trend-crossing", "obs_trend", "application", "high", 5),
    FamilySpec20k("l08-f12-medication-pair-active", "med_pair", "application", "medium", 5),
    FamilySpec20k("l08-f13-condition-resolved-window", "cond_resolved", "application", "low", 5),
    FamilySpec20k("l08-f14-encounter-care-gap", "enc_gap", "application", "medium", 5),
    FamilySpec20k("l08-f15-demographic-condition", "age_cond", "application", "high", 5),
    FamilySpec20k("l08-f16-observation-range-band", "obs_range", "application", "low", 5),
    FamilySpec20k("l08-f17-multi-mechanism-eligibility", "combo3", "application", "high", 5),
    FamilySpec20k("l08-f18-rule-parameter-repair-v2", "obs_count_repair", "repair", "high", 5),
]

BY_ID_20K = {f.family_id: f for f in FAMILY_TABLE_20K}

TOTAL_BASE_GOALS_20K = sum(f.base_goals for f in FAMILY_TABLE_20K)  # 50
TOTAL_INSTANCES_20K = sum(f.base_goals * f.instances_per_goal for f in FAMILY_TABLE_20K)  # 250
