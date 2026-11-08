"""aux_inputs: typed loading of the W17 auxiliary evidence the exporter joins on.

The run log (``runs/**/researcher_trials.jsonl``) is the primary input; the
Codex-alignment views additionally join on *real* repository evidence:

- qualified task supplies (``benchmarks/qualified/tasks*.jsonl`` + manifests),
- the task-factory funnel and source execution matrix CSVs,
- blueprint difficulty records (``min_action_count`` / ``interacting_units``),
- the W11 pilot per-run failure analysis,
- intervention / Φ-causality / identifiability instrument evidence.

Every file is hashed; ``AuxData.digest`` covers exactly the files that were
actually read, so a missing input degrades the dependent view to PENDING with a
named reason instead of fabricating content. All parsing is deterministic.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Deterministic supply labels for the known qualified-task files. Fallback: stem.
SUPPLY_LABELS: Dict[str, str] = {
    "tasks": "gatekeeper_v1_w8m3",
    "tasks_v2": "gatekeeper_w15_v2",
    "tasks_rhea": "rhea_chebi_w14",
}


@dataclass(frozen=True)
class ExportAux:
    """Paths of auxiliary evidence inputs. Absent paths => dependent view PENDING."""

    qualified_tasks: Tuple[Path, ...] = ()
    qualified_manifests: Tuple[Path, ...] = ()
    blueprint_dirs: Tuple[Tuple[str, Path], ...] = ()  # (supply label, dir of *.json)
    funnel_csv: Optional[Path] = None
    source_matrix_csv: Optional[Path] = None
    pilot_analysis: Optional[Path] = None
    intervention_files: Tuple[Path, ...] = ()
    phi_demonstration: Optional[Path] = None
    gold_o_validation: Optional[Path] = None
    discovery_certificate: Optional[Path] = None


@dataclass(frozen=True)
class SupplyData:
    """One qualified task supply: parsed task rows + its qualification manifest."""

    label: str
    tasks_path: str
    tasks_sha256: str
    manifest_path: Optional[str]
    manifest_sha256: Optional[str]
    rows: Tuple[Dict[str, Any], ...]  # slim per-row dicts (never the full spec)
    manifest_counts: Dict[str, Any]
    manifest_checks_summary: Dict[str, Any]
    gate_block: Dict[str, Any]


@dataclass
class AuxData:
    """Parsed auxiliary evidence plus provenance."""

    files: List[Dict[str, str]] = field(default_factory=list)  # {role, path, sha256}
    digest: str = ""
    supplies: List[SupplyData] = field(default_factory=list)
    blueprints: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # blueprint_id -> info
    blueprint_conflicts: List[str] = field(default_factory=list)
    funnel_rows: List[Dict[str, str]] = field(default_factory=list)
    source_matrix_rows: List[Dict[str, str]] = field(default_factory=list)
    pilot_analysis: Optional[Dict[str, Any]] = None
    interventions: List[Dict[str, Any]] = field(default_factory=list)  # {path, sha256, data}
    phi_demonstration: Optional[Dict[str, Any]] = None
    gold_o_validation: Optional[Dict[str, Any]] = None
    discovery_cert_rows: List[Dict[str, Any]] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)  # roles requested but absent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _slim_task_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    spec = raw.get("spec") or {}
    qual = raw.get("qualification") or {}
    return {
        "task_id": raw.get("task_id"),
        "base_goal_id": raw.get("base_goal_id"),
        "canonical_sha": raw.get("canonical_sha"),
        "decision": qual.get("decision"),
        "source_group_id": spec.get("source_group_id"),
        "mechanism_group_id": spec.get("mechanism_group_id"),
        "domain": spec.get("domain"),
        "track": spec.get("track"),
        "task_family": spec.get("task_family"),
        "task_regime": spec.get("task_regime"),
        "ontology_contract": spec.get("ontology_contract"),
        "split": spec.get("split"),
        "task_provenance": spec.get("task_provenance"),
    }


def load_aux(aux: ExportAux) -> AuxData:
    """Read every existing aux input; record provenance; never fabricate."""
    out = AuxData()
    digest = hashlib.sha256()

    def record(role: str, path: Path) -> str:
        sha = _sha256(path)
        out.files.append({"role": role, "path": str(path), "sha256": sha})
        digest.update(role.encode()); digest.update(b"\x00")
        digest.update(str(path).encode()); digest.update(b"\x00")
        digest.update(sha.encode()); digest.update(b"\x00")
        return sha

    manifests_by_stem: Dict[str, Tuple[Path, str]] = {}
    for mpath in aux.qualified_manifests:
        if mpath.is_file():
            manifests_by_stem[mpath.stem] = (mpath, record("qualified_manifest", mpath))
        else:
            out.missing.append(f"qualified_manifest:{mpath}")
    for tpath in aux.qualified_tasks:
        if not tpath.is_file():
            out.missing.append(f"qualified_tasks:{tpath}")
            continue
        tsha = record("qualified_tasks", tpath)
        stem = tpath.stem
        label = SUPPLY_LABELS.get(stem, stem)
        manifest_path, manifest_sha = manifests_by_stem.get(stem.replace("tasks", "manifest", 1), (None, None))
        manifest: Dict[str, Any] = _read_json(manifest_path) if manifest_path else {}
        rows = tuple(
            _slim_task_row(json.loads(line))
            for line in tpath.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        gate = {k: v for k, v in manifest.items() if k.startswith("g") and isinstance(v, dict)}
        out.supplies.append(SupplyData(
            label=label, tasks_path=str(tpath), tasks_sha256=tsha,
            manifest_path=str(manifest_path) if manifest_path else None,
            manifest_sha256=manifest_sha, rows=rows,
            manifest_counts=manifest.get("counts") or {},
            manifest_checks_summary=manifest.get("checks_summary") or {},
            gate_block=gate,
        ))

    for label, bdir in aux.blueprint_dirs:
        if not bdir.is_dir():
            out.missing.append(f"blueprint_dir:{label}:{bdir}")
            continue
        for bpath in sorted(bdir.glob("*.json")):
            try:
                bp = _read_json(bpath)
            except json.JSONDecodeError:
                continue
            bid = bp.get("blueprint_id")
            if not bid:
                continue
            record(f"blueprint:{label}", bpath)
            diff = bp.get("difficulty") or {}
            info = {
                "min_action_count": bp.get("min_action_count", diff.get("min_action_count")),
                "interacting_units": bp.get("interacting_units", diff.get("interacting_units")),
                "goal_mode": bp.get("goal_mode"),
                "supply": label,
            }
            prior = out.blueprints.get(bid)
            if prior and (prior["min_action_count"] != info["min_action_count"]):
                out.blueprint_conflicts.append(
                    f"{bid}: min_action_count {prior['min_action_count']} ({prior['supply']}) "
                    f"vs {info['min_action_count']} ({info['supply']})"
                )
            elif prior is None or info["supply"] > prior["supply"]:
                out.blueprints[bid] = info  # deterministic: latest supply label wins ties

    if aux.funnel_csv is not None:
        if aux.funnel_csv.is_file():
            record("task_factory_funnel", aux.funnel_csv)
            with aux.funnel_csv.open(newline="", encoding="utf-8") as fh:
                out.funnel_rows = list(csv.DictReader(fh))
        else:
            out.missing.append(f"funnel_csv:{aux.funnel_csv}")
    if aux.source_matrix_csv is not None:
        if aux.source_matrix_csv.is_file():
            record("source_execution_matrix", aux.source_matrix_csv)
            with aux.source_matrix_csv.open(newline="", encoding="utf-8") as fh:
                out.source_matrix_rows = list(csv.DictReader(fh))
        else:
            out.missing.append(f"source_matrix_csv:{aux.source_matrix_csv}")
    if aux.pilot_analysis is not None:
        if aux.pilot_analysis.is_file():
            record("pilot_analysis", aux.pilot_analysis)
            out.pilot_analysis = _read_json(aux.pilot_analysis)
        else:
            out.missing.append(f"pilot_analysis:{aux.pilot_analysis}")
    for ipath in aux.intervention_files:
        if ipath.is_file():
            sha = record("intervention_evidence", ipath)
            out.interventions.append({"path": str(ipath), "sha256": sha, "data": _read_json(ipath)})
        else:
            out.missing.append(f"intervention:{ipath}")
    if aux.phi_demonstration is not None:
        if aux.phi_demonstration.is_file():
            record("phi_demonstration", aux.phi_demonstration)
            out.phi_demonstration = _read_json(aux.phi_demonstration)
        else:
            out.missing.append(f"phi_demonstration:{aux.phi_demonstration}")
    if aux.gold_o_validation is not None:
        if aux.gold_o_validation.is_file():
            record("gold_o_validation", aux.gold_o_validation)
            out.gold_o_validation = _read_json(aux.gold_o_validation)
        else:
            out.missing.append(f"gold_o_validation:{aux.gold_o_validation}")
    if aux.discovery_certificate is not None:
        if aux.discovery_certificate.is_file():
            record("discovery_certificate", aux.discovery_certificate)
            out.discovery_cert_rows = [
                json.loads(line)
                for line in aux.discovery_certificate.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            out.missing.append(f"discovery_certificate:{aux.discovery_certificate}")

    out.digest = digest.hexdigest()
    return out


def default_aux(repo_root: Path) -> ExportAux:
    """Repo-standard aux layout (used by the CLI; tests pass explicit fixtures)."""
    bq = repo_root / "benchmarks" / "qualified"
    ae = repo_root / "data" / "erule" / "review5" / "agent_eval"
    return ExportAux(
        qualified_tasks=tuple(p for p in (
            bq / "tasks.jsonl", bq / "tasks_v2.jsonl", bq / "tasks_rhea.jsonl") if p.is_file()),
        qualified_manifests=tuple(p for p in (
            bq / "manifest.json", bq / "manifest_v2.json", bq / "manifest_rhea.json") if p.is_file()),
        blueprint_dirs=tuple((label, d) for label, d in (
            ("gk_v1", repo_root / "benchmarks" / "candidates" / "gk_pilot"),
            ("gk_w15_v2", repo_root / "data" / "erule" / "review5" / "software" / "blueprints_v2"),
            ("rhea_w14", repo_root / "data" / "erule" / "review5" / "scientific" / "blueprints"),
        ) if d.is_dir()),
        funnel_csv=repo_root / "audit" / "review5" / "TASK_FACTORY_FUNNEL.csv",
        source_matrix_csv=repo_root / "audit" / "review5" / "SOURCE_EXECUTION_MATRIX.csv",
        pilot_analysis=ae / "pilot_analysis.json",
        intervention_files=tuple(p for p in (
            repo_root / "data" / "erule" / "review5" / "evaluation" / "interventions" / "gk_blueprint_07_inloop_bridge.json",
            repo_root / "data" / "erule" / "review5" / "evaluation" / "interventions" / "gk_blueprint_07_o_attempt.json",
        ) if p.is_file()),
        phi_demonstration=repo_root / "data" / "erule" / "review5" / "scientific" / "logs" / "phi_demonstration.json",
        gold_o_validation=ae / "gold_o" / "validation.json",
        discovery_certificate=repo_root / "data" / "erule" / "identifiability_certificate.jsonl",
    )
