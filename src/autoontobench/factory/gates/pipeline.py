"""Qualification gate pipeline (owner: S11).

Each gate is a pure callable over (task row, evidence, context) reading REAL
on-disk artifacts from the build directory layout (FINAL-BUILD-ORDER-v2 §7):
`bank/builds/<build_id>/{candidates,qualified_tasks,task_manifest,
native_evidence_index,rejections}.jsonl`. Missing files are honest empties,
never fabricated verdicts.

Ladder (§6.2): CANDIDATE -> NATIVE_VALIDATED -> SOLVABLE -> NONTRIVIAL ->
SEMANTICALLY_RELEVANT -> LEAK_FREE -> QUALIFIED -> RELEASE_ELIGIBLE.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from autoontobench.factory.gates.dedup import DedupDecision
from autoontobench.factory.gates.license_matrix import LicenseMatrix
from autoontobench.factory.gates.reasons import (
    GATE_LADDER,
    Gate,
    GateVerdict,
    Reason,
    TaskDecision,
)

STATUS_BY_GATE: dict[Gate, str] = {
    Gate.SCHEMA: "CANDIDATE",
    Gate.NATIVE_VERIFIED: "NATIVE_VALIDATED",
    Gate.SOLVABLE: "SOLVABLE",
    Gate.NONTRIVIAL: "NONTRIVIAL",
    Gate.SEMANTIC_RELEVANCE: "SEMANTICALLY_RELEVANT",
    Gate.LEAKAGE_CLEAN: "LEAK_FREE",
    Gate.DEDUP: "QUALIFIED",
    Gate.RELEASE_ELIGIBLE: "RELEASE_ELIGIBLE",
}

REQUIRED_ROW_KEYS: tuple[str, ...] = ("task_id", "lane_id")


# ---------------------------------------------------------------- evidence io

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Tolerant JSONL reader: missing file -> [], bad line -> skipped w/ marker."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            rows.append({"_malformed": True, "_raw_prefix": line[:80]})
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


@dataclass(frozen=True)
class BuildArtifacts:
    """Real on-disk build inputs; empty lists when the files do not exist yet."""

    build_dir: Path
    candidates: tuple[dict[str, Any], ...] = ()
    qualified: tuple[dict[str, Any], ...] = ()
    task_manifest: tuple[dict[str, Any], ...] = ()
    evidence_index: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejections: tuple[dict[str, Any], ...] = ()


def load_build(build_dir: Path) -> BuildArtifacts:
    evidence: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(build_dir / "native_evidence_index.jsonl"):
        tid = row.get("task_id")
        if isinstance(tid, str):
            evidence[tid] = row
    return BuildArtifacts(
        build_dir=build_dir,
        candidates=tuple(load_jsonl(build_dir / "candidates.jsonl")),
        qualified=tuple(load_jsonl(build_dir / "qualified_tasks.jsonl")),
        task_manifest=tuple(load_jsonl(build_dir / "task_manifest.jsonl")),
        evidence_index=evidence,
        rejections=tuple(load_jsonl(build_dir / "rejections.jsonl")),
    )


def _row_field(row: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = row.get(name)
        if value is not None:
            return value
    return default


def row_source_ids(row: dict[str, Any]) -> list[str]:
    out: list[str] = []
    refs = row.get("source_refs") or []
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and isinstance(ref.get("source_id"), str):
                out.append(ref["source_id"])
    snaps = row.get("source_snapshot_ids") or []
    if isinstance(snaps, list):
        for snap in snaps:
            if isinstance(snap, str):
                out.append(snap.split("@")[0])
    return sorted(set(out))


# ---------------------------------------------------------------- gates

def gate_schema(row: dict[str, Any]) -> GateVerdict:
    if row.get("_malformed"):
        return GateVerdict(Gate.SCHEMA, False, Reason.SCHEMA_INVALID,
                           detail={"raw_prefix": row.get("_raw_prefix", "")})
    missing = [k for k in REQUIRED_ROW_KEYS if not row.get(k)]
    if missing:
        return GateVerdict(Gate.SCHEMA, False, Reason.SCHEMA_INVALID,
                           detail={"missing": missing})
    return GateVerdict(Gate.SCHEMA, True, Reason.OK)


def gate_native_verified(row: dict[str, Any], evidence: dict[str, Any] | None) -> GateVerdict:
    qual = row.get("qualification") or {}
    ev_ok = bool(evidence and evidence.get("ok") is True)
    ref = (evidence or {}).get("evidence_ref") or (evidence or {}).get("verifier_sha")
    if ev_ok or qual.get("native_verified") is True:
        refs = (f"native_evidence_index:{ref}",) if ref else ()
        return GateVerdict(Gate.NATIVE_VERIFIED, True, Reason.OK, evidence_refs=refs)
    reason = Reason.PROOF_MISSING if evidence is None else Reason.PROFILE_UNSUPPORTED
    return GateVerdict(Gate.NATIVE_VERIFIED, False, reason,
                       detail={"evidence_present": evidence is not None})


def gate_solvable(row: dict[str, Any], evidence: dict[str, Any] | None) -> GateVerdict:
    ev = evidence or {}
    proof = ev.get("proof") or {}
    if proof.get("exists") is True or ev.get("solvable") is True:
        return GateVerdict(Gate.SOLVABLE, True, Reason.OK)
    return GateVerdict(Gate.SOLVABLE, False, Reason.UNSOLVABLE)


def gate_nontrivial(row: dict[str, Any], evidence: dict[str, Any] | None) -> GateVerdict:
    ev = evidence or {}
    if ev.get("nontrivial") is not None:
        ok = ev["nontrivial"] is True
        return GateVerdict(Gate.NONTRIVIAL, ok, Reason.OK if ok else Reason.TRIVIAL)
    steps = ev.get("solution_steps", ev.get("min_actions"))
    start_sha = ev.get("initial_state_sha")
    goal_sha = ev.get("goal_state_sha")
    if isinstance(steps, int) and steps >= 1 and start_sha and goal_sha and start_sha != goal_sha:
        return GateVerdict(Gate.NONTRIVIAL, True, Reason.OK,
                           detail={"solution_steps": steps})
    return GateVerdict(Gate.NONTRIVIAL, False, Reason.TRIVIAL,
                       detail={"solution_steps": steps,
                               "distinct_start_goal": bool(start_sha and start_sha != goal_sha)})


def gate_semantic_relevance(row: dict[str, Any], evidence: dict[str, Any] | None) -> GateVerdict:
    ev = evidence or {}
    witness = ev.get("semantic_witness") or {}
    if witness.get("ok") is True or ev.get("semantic_relevance") is True:
        return GateVerdict(Gate.SEMANTIC_RELEVANCE, True, Reason.OK)
    return GateVerdict(Gate.SEMANTIC_RELEVANCE, False, Reason.NON_SEMANTIC)


def gate_leakage_clean(
    row: dict[str, Any],
    *,
    public_root: Path | None = None,
    private_root: Path | None = None,
) -> GateVerdict:
    """Leak gate: private-key guard on the row + leak_audit on the public dir."""
    from autoontobench.contracts.models import assert_no_private_keys

    public_payload = {k: v for k, v in row.items() if k != "private_oracle"}
    try:
        assert_no_private_keys(public_payload)
    except ValueError as exc:
        return GateVerdict(Gate.LEAKAGE_CLEAN, False, Reason.LEAKAGE, detail={"error": str(exc)})
    if public_root is None or not public_root.exists():
        return GateVerdict(Gate.LEAKAGE_CLEAN, True, Reason.OK,
                           detail={"note": "no public dir on disk; key-scan only"})
    from autoontobench.safety.leak_audit import scan_public_root

    findings = scan_public_root(public_root, private_root)
    leaked = [f for f in findings if f.verdict == "leaked"]
    if leaked:
        return GateVerdict(
            Gate.LEAKAGE_CLEAN, False, Reason.LEAKAGE,
            evidence_refs=tuple(f.path for f in leaked[:5]),
            detail={"leaked_findings": len(leaked)},
        )
    return GateVerdict(Gate.LEAKAGE_CLEAN, True, Reason.OK)


def gate_dedup(decision: DedupDecision | None) -> GateVerdict:
    if decision is None or not decision.is_duplicate:
        return GateVerdict(Gate.DEDUP, True, Reason.OK)
    return GateVerdict(Gate.DEDUP, False, Reason.DUPLICATE,
                       detail={"duplicate_of": decision.duplicate_of, "level": decision.level})


def gate_release_eligible(row: dict[str, Any], matrix: LicenseMatrix) -> GateVerdict:
    source_ids = row_source_ids(row)
    if not source_ids:
        return GateVerdict(Gate.RELEASE_ELIGIBLE, False, Reason.SOURCE_MISSING)
    ok, verdicts = matrix.gate_task(source_ids)
    if ok:
        return GateVerdict(Gate.RELEASE_ELIGIBLE, True, Reason.OK)
    held = [v.source_id for v in verdicts if not v.shippable]
    return GateVerdict(Gate.RELEASE_ELIGIBLE, False, Reason.LICENSE_HOLD,
                       detail={"held_sources": held})


# ---------------------------------------------------------------- pipeline

def run_pipeline(
    rows: Iterable[dict[str, Any]],
    *,
    evidence_index: dict[str, dict[str, Any]],
    dedup_decisions: dict[str, DedupDecision] | None = None,
    license_matrix: LicenseMatrix | None = None,
    public_root: Path | None = None,
    private_root: Path | None = None,
) -> list[TaskDecision]:
    """Run the full ladder per row, short-circuiting on the first failure."""
    decisions: list[TaskDecision] = []
    for row in rows:
        task_id = str(row.get("task_id", "<missing>"))
        lane_id = str(row.get("lane_id", "<unknown>"))
        evidence = evidence_index.get(task_id)
        verdicts: list[GateVerdict] = []
        status = "REJECTED"
        reject = Reason.OK
        for gate in GATE_LADDER:
            if gate is Gate.SCHEMA:
                verdict = gate_schema(row)
            elif gate is Gate.NATIVE_VERIFIED:
                verdict = gate_native_verified(row, evidence)
            elif gate is Gate.SOLVABLE:
                verdict = gate_solvable(row, evidence)
            elif gate is Gate.NONTRIVIAL:
                verdict = gate_nontrivial(row, evidence)
            elif gate is Gate.SEMANTIC_RELEVANCE:
                verdict = gate_semantic_relevance(row, evidence)
            elif gate is Gate.LEAKAGE_CLEAN:
                verdict = gate_leakage_clean(row, public_root=public_root,
                                             private_root=private_root)
            elif gate is Gate.DEDUP:
                verdict = gate_dedup((dedup_decisions or {}).get(task_id))
            elif gate is Gate.RELEASE_ELIGIBLE:
                if license_matrix is None:
                    verdict = GateVerdict(gate, True, Reason.OK,
                                          detail={"note": "no license matrix loaded"})
                else:
                    verdict = gate_release_eligible(row, license_matrix)
            else:  # pragma: no cover - GATE_LADDER is exhaustive
                raise AssertionError(f"unhandled gate {gate}")
            verdicts.append(verdict)
            if not verdict.passed:
                reject = verdict.reason
                break
            status = STATUS_BY_GATE[gate]
        decisions.append(
            TaskDecision(task_id=task_id, lane_id=lane_id, status=status,
                         verdicts=tuple(verdicts), reject_reason=reject)
        )
    return decisions
