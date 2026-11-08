"""aggregate: load and validate append-only ``runs/**/researcher_trials.jsonl`` rows.

Implements the strict input contract for P8 paper export
(Implementation §6 E5 + §11.6, Shared Contract §F):

- every line must parse as a ``RunRowV5``;
- ``SIM:`` content anywhere in a raw line is a hard failure in strict mode
  (simulated data must never enter a submission export);
- unknown keys (metric names outside the frozen RunRowV5 field set) are a hard
  failure in strict mode;
- placeholder strings (TBD/TODO/PLACEHOLDER/FIXME/XXX) are a hard failure;
- ``*_sha`` fields must be well-formed hashes (``[sha256:]<64 lowercase hex>``);
  a malformed hash is a source-hash mismatch and fails strict mode;
- SCORED rows must carry the fields the scoring contract requires
  (task_success, native_terminal_verifier_sha, native_terminal_evidence_sha);
- duplicate run_ids across files fail strict mode (append-only log violated).

Non-strict mode collects every finding into the diagnostics channel and keeps
going; strict mode raises :class:`StrictExportError` on the first class of
finding (all findings of that class are reported together).

Claimability (honest default state): rows whose provider is ``mock`` or whose
``model_exact_id`` is a scripted/mock id are *dev-loop evidence*, never paper
claims. They are aggregated into the ``non_claimable`` channel with an explicit
reason and never contribute to a claim cell.

W17 additions (Codex alignment wave):

- **Superseded/scratch exclusion**: any path with a segment starting with
  ``_`` (e.g. ``pilot_agents/_superseded_attempt1_*/``) is skipped. The
  preregistration instrument-fault rule excludes those attempts from the
  official trials log; they remain on disk for audit but are not export input.
- **Experiment scopes**: the first relative path segment is the experiment
  log root (``pilot_agents``, ``preflight_p1``, ...). Append-only integrity is
  enforced *within* a scope, and it protects paper claims: a duplicate
  ``run_id`` inside one scope is a strict failure whenever ANY of the
  contested rows is claimable (the identity of a potential claim row is
  ambiguous). Duplicates confined to non-claimable dev-loop rows (e.g. the
  regenerated ``pilot_controls_dev`` fixture tree, whose README declares it
  is not the immutable trial log) degrade to a ``duplicate_run_id_non_claimable``
  warning. The same ``run_id`` appearing in *different* scopes is a
  cross-generation collision (e.g. the W11 v1-supply pilot and the W16
  v2-supply preflight share ``dev--<task>--<arm>--seed137`` ids): both rows
  are kept, tagged with their ``experiment_scope``, and a
  ``cross_scope_run_id_collision`` warning is recorded — never silently merged
  or dropped.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from autoontobench.contracts.models import RunRowV5

TRIALS_GLOB = "researcher_trials.jsonl"

# Strict-mode rejection patterns (Implementation §8.11 export negative cases).
SIM_MARKER = "SIM:"
PLACEHOLDER_STRINGS = frozenset({"TBD", "TODO", "PLACEHOLDER", "FIXME", "XXX"})
SHA_RE = re.compile(r"^(sha256:)?[0-9a-f]{64}$")
RUN_ROW_FIELDS = frozenset(RunRowV5.model_fields)

# Providers / models that mark a row as dev-loop evidence, never a paper claim.
NON_CLAIMABLE_PROVIDERS = frozenset({"mock"})
NON_CLAIMABLE_MODEL_PREFIXES = ("mock-", "scripted-")
FIXTURE_DOMAIN_MARKERS = ("fixture", "mock_")


class StrictExportError(Exception):
    """Raised in strict mode when the input violates the export contract."""

    def __init__(self, kind: str, findings: List[str]) -> None:
        self.kind = kind
        self.findings = findings
        super().__init__(f"strict export rejected input ({kind}): " + "; ".join(findings[:5]))


@dataclass(frozen=True)
class ClassifiedRow:
    """A parsed run row plus its claimability verdict."""

    row: RunRowV5
    source_path: str  # path relative to the input root (deterministic output)
    claimable: bool
    non_claimable_reason: Optional[str]
    experiment_scope: str = "."  # first relative path segment (experiment log root)

    @property
    def run_ref(self) -> str:
        """Scope-qualified run reference: ``<scope>/<run_id>``."""
        return self.row.run_id if self.experiment_scope == "." else f"{self.experiment_scope}/{self.row.run_id}"


@dataclass
class LoadReport:
    """Result of loading every trials file under an input root."""

    rows: List[ClassifiedRow] = field(default_factory=list)
    files: List[Dict[str, Any]] = field(default_factory=list)  # per-file evidence entries
    warnings: List[str] = field(default_factory=list)
    input_digest: str = ""  # sha256 over sorted (path, sha256) of inputs
    skipped_files: List[Dict[str, str]] = field(default_factory=list)  # superseded/scratch exclusions

    @property
    def claimable_rows(self) -> List[ClassifiedRow]:
        return [r for r in self.rows if r.claimable]

    @property
    def non_claimable_rows(self) -> List[ClassifiedRow]:
        return [r for r in self.rows if not r.claimable]


def _fail(strict: bool, kind: str, findings: List[str], warnings: List[str]) -> None:
    if not findings:
        return
    if strict:
        raise StrictExportError(kind, findings)
    warnings.extend(f"{kind}: {f}" for f in findings)


def _classify(row: RunRowV5) -> Optional[str]:
    """Return the non-claimable reason, or None if the row is claimable."""
    provider = (row.provider or "").strip().lower()
    model = (row.model_exact_id or "").strip().lower()
    domain = (row.domain or "").strip().lower()
    if provider in NON_CLAIMABLE_PROVIDERS:
        return "MOCK_PROVIDER"
    if any(model.startswith(p) for p in NON_CLAIMABLE_MODEL_PREFIXES):
        return "SCRIPTED_MODEL"
    if any(marker in domain for marker in FIXTURE_DOMAIN_MARKERS):
        return "FIXTURE_DOMAIN"
    return None


def _validate_line(
    raw: str, where: str, strict: bool, warnings: List[str]
) -> Optional[RunRowV5]:
    """Validate one raw JSONL line; returns None when unusable (non-strict only)."""
    if SIM_MARKER in raw:
        _fail(strict, "sim_content", [f"{where}: contains '{SIM_MARKER}'"], warnings)
        return None
    try:
        payload: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(strict, "parse_error", [f"{where}: {exc}"], warnings)
        return None
    if not isinstance(payload, dict):
        _fail(strict, "parse_error", [f"{where}: line is not a JSON object"], warnings)
        return None

    unknown = sorted(k for k in payload if k not in RUN_ROW_FIELDS)
    if unknown:
        _fail(strict, "unknown_metric", [f"{where}: unknown fields {unknown}"], warnings)

    placeholders = sorted(
        k for k, v in payload.items() if isinstance(v, str) and v.strip().upper() in PLACEHOLDER_STRINGS
    )
    if placeholders:
        _fail(
            strict,
            "placeholder_value",
            [f"{where}: placeholder strings in fields {placeholders}"],
            warnings,
        )

    bad_hashes = sorted(
        k
        for k, v in payload.items()
        if k.endswith("_sha") and v is not None and not SHA_RE.match(str(v))
    )
    if bad_hashes:
        _fail(
            strict,
            "source_hash_mismatch",
            [f"{where}: malformed hash fields {bad_hashes}"],
            warnings,
        )

    try:
        row = RunRowV5.model_validate(payload)
    except ValueError as exc:
        _fail(strict, "schema_violation", [f"{where}: {exc}"], warnings)
        return None

    if row.status == "SCORED":
        missing = [
            name
            for name in ("task_success", "native_terminal_verifier_sha", "native_terminal_evidence_sha")
            if getattr(row, name) is None
        ]
        if missing:
            _fail(
                strict,
                "scored_row_incomplete",
                [f"{where}: SCORED row {row.run_id} missing {missing}"],
                warnings,
            )
    return row


def load_runs(input_dir: Path, strict: bool = True) -> LoadReport:
    """Load every ``researcher_trials.jsonl`` under ``input_dir`` (recursive)."""
    report = LoadReport()
    if not input_dir.is_dir():
        raise StrictExportError("input_missing", [f"input dir not found: {input_dir}"])

    trials_paths = sorted(input_dir.rglob(TRIALS_GLOB))
    seen_run_ids: Dict[str, str] = {}  # (scope, run_id) -> where  (append-only per scope)
    contested: Dict[tuple, List[str]] = {}  # (scope, run_id) -> [where, ...]
    run_id_scopes: Dict[str, set] = {}  # run_id -> {scope, ...} (collision detection)
    row_claimable: Dict[tuple, bool] = {}
    duplicate_findings: List[str] = []
    digest = hashlib.sha256()

    for path in trials_paths:
        rel = path.relative_to(input_dir).as_posix()
        if any(part.startswith("_") for part in rel.split("/")):
            report.skipped_files.append(
                {"path": rel, "reason": "superseded/scratch path segment ('_' prefix); "
                                       "excluded per preregistration instrument-fault rule"}
            )
            continue
        scope = rel.split("/", 1)[0] if "/" in rel else "."
        blob = path.read_bytes()
        file_sha = hashlib.sha256(blob).hexdigest()
        digest.update(rel.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\x00")
        entry: Dict[str, Any] = {
            "path": rel,
            "sha256": file_sha,
            "rows": 0,
            "claimable_rows": 0,
            "non_claimable_rows": 0,
        }
        for lineno, raw in enumerate(blob.decode("utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            where = f"{rel}:{lineno}"
            row = _validate_line(raw, where, strict, report.warnings)
            if row is None:
                continue
            reason = _classify(row)
            scoped_id = (scope, row.run_id)
            prior = seen_run_ids.get(scoped_id)
            if prior is not None:
                contested.setdefault(scoped_id, [prior]).append(where)
            else:
                seen_run_ids[scoped_id] = where
            row_claimable.setdefault(scoped_id, False)
            row_claimable[scoped_id] = row_claimable[scoped_id] or reason is None
            run_id_scopes.setdefault(row.run_id, set()).add(scope)
            report.rows.append(
                ClassifiedRow(row=row, source_path=rel, claimable=reason is None,
                              non_claimable_reason=reason, experiment_scope=scope)
            )
            entry["rows"] += 1
            if reason is None:
                entry["claimable_rows"] += 1
            else:
                entry["non_claimable_rows"] += 1
        report.files.append(entry)

    for scoped_id in sorted(contested):
        wheres = sorted(contested[scoped_id])
        if row_claimable.get(scoped_id):
            duplicate_findings.append(
                f"run_id {scoped_id[1]} (scope {scoped_id[0]}) appears at {wheres} "
                "and at least one contested row is claimable"
            )
        else:
            report.warnings.append(
                f"duplicate_run_id_non_claimable: run_id {scoped_id[1]} (scope {scoped_id[0]}) "
                f"appears at {wheres}; all contested rows are non-claimable dev-loop rows"
            )
    _fail(strict, "duplicate_run_id", duplicate_findings, report.warnings)
    for rid in sorted(run_id_scopes):
        scopes = sorted(run_id_scopes[rid])
        if len(scopes) > 1:
            report.warnings.append(
                f"cross_scope_run_id_collision: run_id {rid} appears in experiment scopes "
                f"{scopes}; rows kept and scope-tagged (distinct experiment generations "
                f"may share the dev--<task>--<arm>--seed<seed> id namespace)"
            )
    report.input_digest = digest.hexdigest()
    return report
