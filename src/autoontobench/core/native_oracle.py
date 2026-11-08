"""Native OPA oracle for the Gatekeeper software domain (Review 5B §4 T0, §11.1).

Independent terminal ground truth for the Gatekeeper NativeDomain. Wraps the
certified pinned pipeline (``native_runners/opa_runner.py`` + the four closure
units from ``references/sources/gatekeeper-library`` @ a8d4799d…) with:

* a typed ``PolicySet`` — the authentic rule units + their sample-constraint
  applicability (match.kinds / namespaces, re-parsed from the pinned
  constraint.yaml via ``rule_ingestion.yaml_subset``; never re-implemented);
* deterministic memoization keyed by input-document sha256 (a cache hit is NOT
  a native call; it is recorded as ``cached: true`` in the ledger);
* an append-only JSONL ledger — every native subprocess call lands one row with
  unit, decision, exit_code, input sha256 and the opa_runner log path.

This module never imports or calls ``OntologyTool`` / ``ontology_consumer``
(HARD RULE: native truth stays outside any submitted-ontology consumer).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from native_runners.opa_runner import run_native_input
from rule_ingestion.gatekeeper_adapter import admission_input, parse_unit
from rule_ingestion.yaml_subset import parse_yaml_subset

REPO_ROOT = Path(__file__).resolve().parents[2]
GATEKEEPER_REPO = REPO_ROOT / "references" / "sources" / "gatekeeper-library"
GATEKEEPER_COMMIT = "a8d4799d8dafd47ec5fafe945ef277d329e2cceb"
OPA_BIN = "/usr/local/bin/opa"
OPA_VERSION = "0.70.0"

#: The four closure units and the sample constraint that pins their parameters
#: (audit/GATEKEEPER_CLOSURE.md §4). (unit, sample_dir)
GATEKEEPER_SLICE: Tuple[Tuple[str, str], ...] = (
    ("general/requiredlabels", "samples/verify-label-key-only"),
    ("general/replicalimits", "samples/replicalimits"),
    ("general/allowedrepos", "samples/repo-must-be-openpolicyagent"),
    ("general/block-nodeport-services", "samples/block-node-port"),
)


def sha256_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyUnit:
    """One authentic Gatekeeper rule unit + sample-constraint applicability."""

    rule_unit_id: str
    unit: str
    package: str
    parameters: Dict[str, Any]
    match_kinds: Tuple[str, ...]
    match_namespaces: Tuple[str, ...]
    src_rego_sha256: str
    raw: Any  # rule_ingestion.gatekeeper_adapter.GatekeeperRuleUnit

    def governs(self, obj: Dict[str, Any]) -> bool:
        if obj.get("kind") not in self.match_kinds:
            return False
        if self.match_namespaces:
            ns = (obj.get("metadata") or {}).get("namespace", "default")
            return ns in self.match_namespaces
        return True


@dataclass(frozen=True)
class UnitVerdict:
    rule_unit_id: str
    decision: str  # "ALLOW" | "DENY"
    violation_count: int
    input_sha256: str
    log_path: str
    cached: bool


@dataclass(frozen=True)
class ObjectVerdict:
    compliant: bool
    verdicts: Tuple[UnitVerdict, ...]

    @property
    def evidence_sha256(self) -> str:
        return sha256_json(
            [
                {
                    "unit": v.rule_unit_id,
                    "decision": v.decision,
                    "input_sha256": v.input_sha256,
                }
                for v in self.verdicts
            ]
        )


def _match_namespaces(repo: Path, unit: str, sample_dir: str) -> Tuple[str, ...]:
    doc = parse_yaml_subset(
        (repo / "library" / unit / sample_dir / "constraint.yaml").read_text(encoding="utf-8")
    )
    spec = doc.get("spec") or {}
    match = spec.get("match") or {}
    nss = match.get("namespaces") or []
    return tuple(str(ns) for ns in nss)


def load_policy_set(
    repo_root: Path | None = None,
    slice_spec: Tuple[Tuple[str, str], ...] = GATEKEEPER_SLICE,
) -> Tuple[PolicyUnit, ...]:
    """Parse the pinned slice into typed policy units (fail-closed)."""
    repo = Path(repo_root) if repo_root else GATEKEEPER_REPO
    units: List[PolicyUnit] = []
    for unit_name, sample_dir in slice_spec:
        raw = parse_unit(repo, unit_name, sample_dir)
        if raw.source_commit != GATEKEEPER_COMMIT:
            raise ValueError(
                "gatekeeper-library pin drift: %s != %s" % (raw.source_commit, GATEKEEPER_COMMIT)
            )
        units.append(
            PolicyUnit(
                rule_unit_id=raw.rule_unit_id,
                unit=raw.unit,
                package=raw.package,
                parameters=raw.parameters,
                match_kinds=raw.match_kinds,
                match_namespaces=_match_namespaces(repo, unit_name, sample_dir),
                src_rego_sha256=raw.src_rego_sha256,
                raw=raw,
            )
        )
    return tuple(units)


class GatekeeperNativeOracle:
    """Pinned-OPA object evaluator with memoization and an append-only ledger."""

    def __init__(
        self,
        log_root: Path,
        policy_set: Tuple[PolicyUnit, ...] | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else GATEKEEPER_REPO
        self.policy_set = policy_set if policy_set is not None else load_policy_set(self.repo_root)
        log_root = Path(log_root).resolve()
        self.log_dir = log_root / "opa"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = log_root / "oracle_ledger.jsonl"
        self._cache: Dict[Tuple[str, str], UnitVerdict] = {}
        self._calls: List[int] = [0]

    def with_overrides(
        self, overrides: Dict[str, Dict[str, Any]]
    ) -> "GatekeeperNativeOracle":
        """Episode view with per-unit parameter overrides (source_parameterized tasks).

        Shares the memo cache and ledger: the cache key already contains the full
        input document sha (parameters included), so no cross-contamination.
        """
        if not overrides:
            return self
        new_set = []
        for unit in self.policy_set:
            if unit.rule_unit_id in overrides:
                merged = dict(unit.parameters)
                merged.update(overrides[unit.rule_unit_id])
                new_set.append(replace(unit, parameters=merged,
                                       raw=replace(unit.raw, parameters=merged)))
            else:
                new_set.append(unit)
        view = GatekeeperNativeOracle.__new__(GatekeeperNativeOracle)
        view.repo_root = self.repo_root
        view.policy_set = tuple(new_set)
        view.log_dir = self.log_dir
        view.ledger_path = self.ledger_path
        view._cache = self._cache
        view._calls = self._calls
        return view

    def _eval_unit(self, unit: PolicyUnit, obj: Dict[str, Any], context: str) -> UnitVerdict:
        input_doc = admission_input(unit.raw, obj)
        input_sha = sha256_json(input_doc)
        key = (unit.rule_unit_id, input_sha)
        if key in self._cache:
            hit = self._cache[key]
            self._ledger(unit, hit.decision, hit.violation_count, input_sha, hit.log_path,
                        cached=True, context=context)
            return UnitVerdict(unit.rule_unit_id, hit.decision, hit.violation_count,
                               input_sha, hit.log_path, cached=True)
        outcome = run_native_input(
            unit.raw,
            input_doc,
            input_id="%s__%s" % (context, input_sha[:12]),
            repo_root=self.repo_root,
            log_dir=self.log_dir,
            opa_bin=OPA_BIN,
        )
        self._calls[0] += 1
        verdict = UnitVerdict(
            unit.rule_unit_id, outcome.decision, outcome.violation_count,
            outcome.input_sha256, str(outcome.log_path), cached=False,
        )
        self._cache[key] = verdict
        self._ledger(unit, outcome.decision, outcome.violation_count, outcome.input_sha256,
                     str(outcome.log_path), cached=False, context=context,
                     exit_code=outcome.exit_code, command=outcome.command)
        return verdict

    def evaluate_object(
        self, obj: Dict[str, Any], context: str = "eval"
    ) -> ObjectVerdict:
        """Natively evaluate one k8s-like object under every applicable unit."""
        verdicts = tuple(
            self._eval_unit(unit, obj, context)
            for unit in self.policy_set
            if unit.governs(obj)
        )
        return ObjectVerdict(
            compliant=all(v.decision == "ALLOW" for v in verdicts), verdicts=verdicts
        )

    def _ledger(self, unit: PolicyUnit, decision: str, violations: int, input_sha: str,
                log_path: str, *, cached: bool, context: str,
                exit_code: int | None = None, command: Any = None) -> None:
        row = {
            "oracle": "opa",
            "opa_version": OPA_VERSION,
            "source_commit": GATEKEEPER_COMMIT,
            "rule_unit_id": unit.rule_unit_id,
            "src_rego_sha256": unit.src_rego_sha256,
            "decision": decision,
            "violation_count": violations,
            "input_sha256": input_sha,
            "cached": cached,
            "context": context,
            "log_path": log_path,
        }
        if exit_code is not None:
            row["exit_code"] = exit_code
            row["command_sha256"] = hashlib.sha256(
                " ".join(command).encode("utf-8")
            ).hexdigest()
        with self.ledger_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
