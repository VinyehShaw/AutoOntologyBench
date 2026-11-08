"""Stable plugin contracts for the AutoOntologyBench task factory.

Interface boundary from FINAL-BUILD-ORDER-v2 sections 2.1 and 3.2.
Implementations live in ``source_registry/fetchers/`` and
``benchmark/plugins/{source,task_family,verifier}/``. The frozen task record
contract (``aob-task/5.0``) remains in ``autoontobench/contracts/models.py``;
these types are additive and must not mutate existing bank artifacts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass(frozen=True)
class SourceDescriptor:
    source_id: str
    official_url: str
    download_url_or_api: str
    source_type: str
    upstream_version: str
    commit_or_release: str
    retrieved_at: str
    sha256: str
    native_engine: str
    license_url: str
    redistribution_policy: str
    estimated_bytes: int | None = None
    is_complete_snapshot: bool = False
    coverage_fraction_if_known: float | None = None
    dependencies: tuple[str, ...] = ()
    attribution: str = ""
    usage_constraints: str = ""


@dataclass(frozen=True)
class SourceRelease:
    release_id: str
    version: str
    urls: tuple[str, ...]
    published_at: str | None = None
    notes: str = ""


@dataclass(frozen=True)
class FetchReceipt:
    source_id: str
    release_id: str
    status: str
    bytes_downloaded: int
    file_count: int
    cache_dir: str
    sha256_manifest: str
    fetched_at: str
    resumed: bool = False
    missing_assets: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class VerificationReport:
    ok: bool
    checks: dict[str, Any] = field(default_factory=dict)
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    release_id: str
    root: Path
    descriptor: SourceDescriptor


@dataclass(frozen=True)
class SourceUnit:
    unit_id: str
    kind: str
    payload_ref: str
    digest: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IndexManifest:
    source_id: str
    release_id: str
    out_dir: str
    unit_count: int
    kind_counts: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class ExternalKnowledgeSource(Protocol):
    source_id: str

    def describe(self) -> SourceDescriptor: ...

    def discover_releases(self) -> list[SourceRelease]: ...

    def fetch(self, release: SourceRelease, *, resume: bool, cache_dir: Path) -> FetchReceipt: ...

    def verify_snapshot(self, receipt: FetchReceipt) -> VerificationReport: ...

    def inventory(self, snapshot: SourceSnapshot) -> Iterator[SourceUnit]: ...

    def index(self, snapshot: SourceSnapshot, *, out_dir: Path) -> IndexManifest: ...


@dataclass(frozen=True)
class CandidateTaskAST:
    lane_id: str
    family_id: str
    base_goal_id: str
    instance_id: str
    regime: str
    semantic_profile: str
    ast: dict[str, Any]
    source_unit_ids: tuple[str, ...]


@dataclass(frozen=True)
class GroundedTask:
    ast: CandidateTaskAST
    public: dict[str, Any]
    oracle_ref: str
    native_evidence: dict[str, Any]


@dataclass(frozen=True)
class ProofOrCounterexample:
    exists: bool
    evidence: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class NativeVerdict:
    valid: bool
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SemanticWitness:
    ok: bool
    witness: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TaskFamilyPlugin(Protocol):
    def supported_profiles(self) -> list[str]: ...

    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int, constraints: dict[str, Any]
    ) -> Iterator[CandidateTaskAST]: ...

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask: ...


@runtime_checkable
class NativeVerifierPlugin(Protocol):
    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample: ...

    def verify_terminal(self, task: GroundedTask, submitted_state_or_answer: Any) -> NativeVerdict: ...

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness: ...