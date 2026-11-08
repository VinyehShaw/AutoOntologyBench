"""L08 clinical vocabulary + family table (8 families, 50 base goals).

All clinical content is SYNTHETIC: standard vocabulary codes (LOINC / SNOMED /
RxNorm / HL7 v3 ActCode) are used with display names for readability; patient
panels are generated deterministically from seeds (see synth.py). No real
patient data, no real PHI (lane constraint).

Codes are drawn from widely-cited standard-vocabulary values; display names
are illustrative for the synthetic panels (the rule text and the data always
use the same code, so task semantics never depend on real-world code
correctness — documented in the lane REPORT).
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

#: (code, display, unit, plausible thresholds, default op)
OBS_POOL: list[dict[str, Any]] = [
    {"code": "8462-4", "system": LOINC, "cs_name": LOINC_NAME, "display": "Diastolic blood pressure", "unit": "mm[Hg]",
     "thresholds": [80, 85, 90, 95], "op": ">="},
    {"code": "8480-6", "system": LOINC, "cs_name": LOINC_NAME, "display": "Systolic blood pressure", "unit": "mm[Hg]",
     "thresholds": [130, 140, 150], "op": ">="},
    {"code": "2339-0", "system": LOINC, "cs_name": LOINC_NAME, "display": "Glucose [Mass/volume] in Blood", "unit": "mg/dL",
     "thresholds": [100, 110, 126], "op": ">="},
    {"code": "4548-4", "system": LOINC, "cs_name": LOINC_NAME, "display": "Hemoglobin A1c/Hemoglobin.total in Blood", "unit": "%",
     "thresholds": [5.7, 6.5, 7.0], "op": ">="},
    {"code": "2093-3", "system": LOINC, "cs_name": LOINC_NAME, "display": "Cholesterol [Mass/volume] in Serum or Plasma", "unit": "mg/dL",
     "thresholds": [180, 200, 240], "op": ">="},
    {"code": "2160-0", "system": LOINC, "cs_name": LOINC_NAME, "display": "Creatinine [Mass/volume] in Serum or Plasma", "unit": "mg/dL",
     "thresholds": [1.2, 1.5, 2.0], "op": ">="},
    {"code": "29463-7", "system": LOINC, "cs_name": LOINC_NAME, "display": "Body Mass Index (BMI) [Ratio]",
     "unit": "kg/m2", "thresholds": [25, 30], "op": ">="},
    {"code": "1751-7", "system": LOINC, "cs_name": LOINC_NAME, "display": "Albumin [Mass/volume] in Serum or Plasma", "unit": "g/dL",
     "thresholds": [3.5, 3.0], "op": "<="},
]

COND_POOL: list[dict[str, Any]] = [
    {"code": "195967001", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Asthma"},
    {"code": "44054006", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Diabetes mellitus type 2"},
    {"code": "38341003", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Hypertensive disorder"},
    {"code": "49436004", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Atrial fibrillation"},
    {"code": "13645005", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Chronic obstructive pulmonary disease"},
    {"code": "230690007", "system": SNOMED, "cs_name": SNOMED_NAME, "display": "Cerebrovascular accident"},
]

MED_POOL: list[dict[str, Any]] = [
    {"code": "6809", "system": RXNORM, "cs_name": RXNORM_NAME, "display": "metformin"},
    {"code": "83367", "system": RXNORM, "cs_name": RXNORM_NAME, "display": "atorvastatin"},
    {"code": "284642", "system": RXNORM, "cs_name": RXNORM_NAME, "display": "lisinopril"},
]

ENC_POOL: list[str] = ["AMB", "EMER", "IMP"]

LOOKBACKS: list[int] = [90, 180, 365]

PANEL_SIZE = {"low": 8, "medium": 10, "high": 12}


class FamilySpec:
    def __init__(self, family_id: str, kind: str, regime: str, complexity: str,
                 base_goals: int, instances_per_goal: int = 5) -> None:
        self.family_id = family_id
        self.kind = kind
        self.regime = regime
        self.complexity = complexity
        self.base_goals = base_goals
        self.instances_per_goal = instances_per_goal


FAMILY_TABLE: list[FamilySpec] = [
    FamilySpec("l08-f01-observation-threshold", "obs_threshold", "application", "low", 7),
    FamilySpec("l08-f02-observation-recency", "obs_recency", "application", "medium", 7),
    FamilySpec("l08-f03-observation-latest-value", "obs_latest", "application", "high", 6),
    FamilySpec("l08-f04-medication-active-window", "med_active", "application", "medium", 6),
    FamilySpec("l08-f05-condition-active-status", "cond_active", "application", "low", 6),
    FamilySpec("l08-f06-encounter-frequency", "enc_count", "application", "medium", 6),
    FamilySpec("l08-f07-combined-eligibility", "combined", "application", "high", 6),
    FamilySpec("l08-f08-rule-parameter-repair", "recency_excl", "repair", "high", 6),
]

BY_ID = {f.family_id: f for f in FAMILY_TABLE}

TOTAL_BASE_GOALS = sum(f.base_goals for f in FAMILY_TABLE)  # 50
TOTAL_INSTANCES = sum(f.base_goals * f.instances_per_goal for f in FAMILY_TABLE)  # 250
