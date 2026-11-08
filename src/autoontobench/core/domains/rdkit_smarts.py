"""RDKit executable-SMARTS domain adapter — Review 5B workstream W3 (scientific sources).

Profile: ``smarts-executable/1`` (analytical track). An authored artifact stores SMARTS
patterns; a FIXED matcher primitive (``Chem.MolFromSmiles`` / ``Chem.MolFromSmarts`` /
``HasSubstructMatch``) executes exactly what was submitted — the reference pattern is
never substituted for the submitted one. The reference answer is computed by the SAME
primitive over the PINNED source patterns and pinned molecules (RDKit
``references/sources/rdkit`` at commit 4a0ae3ac…, BSD-3-Clause), so it is independent
of any submitted artifact.

Native truth runs in the disjoint ``rdkit`` conda env (rdkit 2026.09.1) via the pinned
script ``sources/rdkit/native_match.py``; this module never imports rdkit.

Screening objectives are constructed source-derived screening objectives: conjunctive
structural conditions plus an exclusion over a fixed molecule panel. No clinical
efficacy, reaction, or material-property prediction.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import GoalResult, TaskSpecV5
from autoontobench.core.protocols import (
    NativeDecision,
    NativeState,
    NativeStateAndFeedback,
    PublicObservation,
)

PROFILE = "smarts-executable/1"
RDKIT_ENV = "rdkit"
NATIVE_MATCHER = Path("sources/rdkit/native_match.py")
FIXTURE = Path("data/erule/review5/scientific/fixtures/rdkit_screening_nci200.json")

# Verbatim rows from the pinned RDKit data files (never edited; source_rule_unit ids).
REFERENCE_PATTERNS: dict[str, dict[str, str]] = {
    "carboxylic_acid": {
        "smarts": "*-C(=O)[O;D1]",
        "source": "references/sources/rdkit/Data/FunctionalGroups.txt row '-C(=O)O'",
    },
    "alcohol_aliphatic": {
        "smarts": "[O;H1;$(O-!@[C;!$(C=!@[O,N,S])])]",
        "source": "references/sources/rdkit/Data/Functional_Group_Hierarchy.txt row 'Alcohol.Aliphatic'",
    },
    "nitro": {
        "smarts": "*-[N;D3](=[O;D1])[O;D1]",
        "source": "references/sources/rdkit/Data/FunctionalGroups.txt row '-NO2'",
    },
}

# Smoke composition: 2 structural conditions + 1 exclusion (all clauses non-vacuous
# over the pinned NCI-200 fixture: |A∩B|=5, exclusion removes 1, answer size 4).
SMOKE_COMPOSITION: dict[str, list[str]] = {
    "include": ["carboxylic_acid", "alcohol_aliphatic"],
    "exclude": ["nitro"],
}


class RDKitRunnerError(RuntimeError):
    """The pinned rdkit-env matcher failed or returned malformed output."""


class MalformedArtifact(ValueError):
    """Submitted artifact/answer does not match the smarts-executable/1 schema."""


@dataclass(frozen=True)
class Molecule:
    mol_id: str
    smiles: str
    synthetic: bool


@dataclass(frozen=True)
class MatchResult:
    matches: dict[str, tuple[str, ...]]  # pattern name -> sorted molecule ids
    invalid_smiles: tuple[str, ...]
    invalid_smarts: tuple[str, ...]
    rdkit_version: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_fixture(path: Path = FIXTURE) -> tuple[Molecule, ...]:
    payload = json.loads(path.read_text())
    return tuple(
        Molecule(m["id"], m["smiles"], bool(m.get("synthetic", False)))
        for m in payload["molecules"]
    )


def run_native_matcher(
    molecules: tuple[Molecule, ...], patterns: dict[str, str]
) -> MatchResult:
    """Execute the FIXED matcher primitive in the pinned rdkit conda env (subprocess)."""
    request = {
        "molecules": [{"id": m.mol_id, "smiles": m.smiles} for m in molecules],
        "patterns": [{"name": name, "smarts": smarts} for name, smarts in patterns.items()],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="rdkit_req_", delete=False
    ) as fh:
        json.dump(request, fh)
        request_path = fh.name
    proc = subprocess.run(
        ["conda", "run", "-n", RDKIT_ENV, "python", str(NATIVE_MATCHER),
         "--request", request_path],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RDKitRunnerError(f"native matcher rc={proc.returncode}: {proc.stderr[-400:]}")
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise RDKitRunnerError(f"unparseable matcher output: {proc.stdout[-400:]!r}") from exc
    return MatchResult(
        matches={k: tuple(v) for k, v in payload["matches"].items()},
        invalid_smiles=tuple(payload["invalid_smiles"]),
        invalid_smarts=tuple(payload["invalid_smarts"]),
        rdkit_version=payload["rdkit_version"],
    )


def compose(matches: dict[str, tuple[str, ...]], composition: dict[str, list[str]]) -> tuple[str, ...]:
    """set composition: (intersection of include) minus (union of exclude)."""
    include = [set(matches[name]) for name in composition["include"]]
    exclude = [set(matches[name]) for name in composition["exclude"]]
    base = set.intersection(*include) if include else set()
    blocked = set.union(*exclude) if exclude else set()
    return tuple(sorted(base - blocked))


def reference_answer(
    molecules: tuple[Molecule, ...], composition: dict[str, list[str]]
) -> tuple[MatchResult, tuple[str, ...]]:
    """Native truth: PINNED source patterns through the fixed primitive."""
    patterns = {}
    for name in composition["include"] + composition["exclude"]:
        if name not in REFERENCE_PATTERNS:
            raise MalformedArtifact(f"no pinned reference pattern named {name!r}")
        patterns[name] = REFERENCE_PATTERNS[name]["smarts"]
    result = run_native_matcher(molecules, patterns)
    return result, compose(result.matches, composition)


def consumer_screening(
    molecules: tuple[Molecule, ...], artifact_bytes: bytes, composition: dict[str, list[str]]
) -> tuple[MatchResult, tuple[str, ...]]:
    """Execute the SUBMITTED artifact's patterns through the same fixed primitive."""
    payload = json.loads(artifact_bytes.decode())
    if payload.get("profile") != PROFILE or not isinstance(payload.get("patterns"), list):
        raise MalformedArtifact("artifact must be {profile: smarts-executable/1, patterns: [...]}")
    patterns = {p["name"]: p["smarts"] for p in payload["patterns"]}
    result = run_native_matcher(molecules, patterns)
    if result.invalid_smarts:
        raise MalformedArtifact(f"unparseable SMARTS in artifact: {result.invalid_smarts}")
    return result, compose(result.matches, composition)


class RDKitSmartsDomain:
    """Analytical NativeDomain: the only mutating action is ``submit_answer``."""

    def __init__(self, molecules: tuple[Molecule, ...]):
        self._molecules = molecules

    def reset(self, spec: TaskSpecV5, seed: int) -> NativeState:
        instance = json.loads(Path(spec.initial_state_ref).read_text())
        return NativeState(
            payload={"task_id": spec.task_id, "composition": instance["composition"],
                     "submitted_answer": None, "terminal": False}
        )

    def observe(self, state: NativeState, rights: set[str]) -> PublicObservation:
        return PublicObservation(payload={
            "task_id": state.payload["task_id"],
            "composition": state.payload["composition"],
            "molecule_count": len(self._molecules),
            "terminal": state.payload["terminal"],
        })

    def legal_action(self, state: NativeState, action: dict[str, Any]) -> NativeDecision:
        if action.get("type") != "submit_answer":
            return NativeDecision(legal=None, detail={"note": "unknown action type"})
        answer = action.get("answer")
        ok = isinstance(answer, list) and all(isinstance(x, str) for x in answer)
        return NativeDecision(legal=ok, detail={} if ok else {"reason": "answer must be list[str]"})

    def step(self, state: NativeState, action: dict[str, Any]) -> NativeStateAndFeedback:
        if state.payload["terminal"]:
            return NativeStateAndFeedback(state, {"error": "already terminal"}, 0.0)
        decision = self.legal_action(state, action)
        if decision.legal is not True:
            return NativeStateAndFeedback(state, {"rejected": decision.detail}, 0.0)
        new = NativeState(payload={**state.payload,
                                   "submitted_answer": sorted(set(action["answer"])),
                                   "terminal": True})
        return NativeStateAndFeedback(new, {"accepted": True}, 1.0)

    def verify_terminal(
        self, spec: TaskSpecV5, state: NativeState, submitted_answer: object
    ) -> GoalResult:
        native, expected_ids = reference_answer(self._molecules, state.payload["composition"])
        raw = state.payload["submitted_answer"] if submitted_answer is None else submitted_answer
        if not (isinstance(raw, list) and all(isinstance(x, str) for x in raw)):
            raise MalformedArtifact("submitted answer must be a list of molecule id strings")
        got = set(raw)
        expected = set(expected_ids)
        false_negatives = sorted(expected - got)  # missed true members
        false_positives = sorted(got - expected)  # wrongly included
        return GoalResult(
            goal_reached=got == expected,
            evidence={
                "expected": sorted(expected),
                "expected_count": len(expected),
                "submitted_count": len(got),
                "false_negatives": false_negatives,
                "false_positives": false_positives,
                "invalid_smiles": list(native.invalid_smiles),
                "invalid_smarts": list(native.invalid_smarts),
                "rdkit_version": native.rdkit_version,
                "expected_sha256": _sha256_bytes(json.dumps(sorted(expected)).encode()),
                "submitted_sha256": _sha256_bytes(json.dumps(sorted(got)).encode()),
                "reference_engine": f"rdkit {native.rdkit_version} via sources/rdkit/native_match.py",
            },
        )
