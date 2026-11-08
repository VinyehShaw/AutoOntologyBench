"""BYOK task-family plugin (example 1): Gatekeeper policy-repair tasks.

Family `gatekeeper-policy-repair`: given a ConstraintTemplate's embedded Rego,
a constraint's parameters, and a violating resource manifest, produce the exact
violation set and a repaired manifest. Deterministic enumeration over the
snapshot's policy units; seed controls the starting offset. Grounding keeps the
expected violations and the allowed example in the private oracle only.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator

import yaml

from autoontobench.factory import cli_support as cs
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask, SourceSnapshot

FAMILY = "gatekeeper-policy-repair"
PROFILE = "rego-opa/0.70"


class GatekeeperPolicyFamily:
    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    def propose(self, snapshot: SourceSnapshot, *, family: str, seed: int,
                constraints: dict[str, Any]) -> Iterator[CandidateTaskAST]:
        count = int(constraints.get("count", 10))
        inner = _inner_root(Path(snapshot.root))
        triples = _candidate_triples(inner)
        offset = seed % len(triples) if triples else 0
        produced = 0
        for i, (policy, sample, variant) in enumerate(triples):
            if produced >= count:
                return
            if i < offset:
                continue
            constraint = sample / "constraint.yaml"
            allowed = sample / "example_allowed.yaml"
            instance = f"{policy.name}-{sample.name}-{variant.stem.removeprefix('example_').removesuffix('.yaml')}"
            yield CandidateTaskAST(
                lane_id="BYOK", family_id=FAMILY, base_goal_id=f"{policy.name}-{sample.name}",
                instance_id=instance,
                regime=str(constraints.get("regime", "repair")),
                semantic_profile=PROFILE,
                ast={"policy": policy.name, "sample": sample.name,
                     "variant": variant.name,
                     "template_sha256": cs.sha256_hex((policy / 'template.yaml').read_bytes()),
                     "constraint": str(constraint.relative_to(inner)),
                     "disallowed": str(variant.relative_to(inner)),
                     "allowed": str(allowed.relative_to(inner)),
                     "snapshot_root": str(snapshot.root), "seed": seed},
                source_unit_ids=(f"gatekeeper-library-repin-1946bc28/{policy.name}",))
            produced += 1

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        root = _inner_root(Path(candidate.ast["snapshot_root"]))
        template = _load_template(root / "library" / "general" / candidate.ast["policy"]
                                  / "template.yaml")
        assert template is not None
        params = (yaml.safe_load((root / candidate.ast["constraint"]).read_text())
                  .get("spec", {}).get("parameters") or {})
        disallowed = yaml.safe_load((root / candidate.ast["disallowed"]).read_text())
        allowed = yaml.safe_load((root / candidate.ast["allowed"]).read_text())
        regos = list(template["regos"])
        if any("data.lib." in r for r in regos):
            regos += _shared_libs(root)
        public = {
            "user_goal": (
                f"The Gatekeeper policy '{candidate.ast['policy']}' (Rego package "
                f"{template['package']}) is enforced with the given parameters. "
                "The attached Kubernetes resource manifest violates it. "
                "1) List the exact violation messages the policy returns for this "
                "resource. 2) Produce a repaired manifest that passes the policy."),
            "resource_manifest": disallowed,
            "constraint_parameters": params,
            "policy_rego": regos,
            "allowed_tools": ["opa-eval"],
            "budget": {"probes": 4, "actions": 8},
        }
        return GroundedTask(
            ast=candidate, public=public,
            oracle_ref=f"oracle://byok/{candidate.instance_id}",
            native_evidence={"package": template["package"], "regos": regos,
                             "parameters": params, "disallowed": disallowed,
                             "allowed": allowed})


def _shared_libs(inner: Path) -> list[str]:
    lib_dir = inner / "src" / "rego"
    if not lib_dir.is_dir():
        return []
    return [p.read_text(encoding="utf-8") for p in sorted(lib_dir.glob("lib_*/lib_*.rego"))]


def _inner_root(root: Path) -> Path:
    """The GitHub tarball extracts under gatekeeper-library-<sha>/; resolve it."""
    for child in root.iterdir():
        if (child / "library" / "general").is_dir():
            return child
    return root


def _load_template(path: Path) -> dict[str, Any] | None:
    """Both gatekeeper-library template shapes: legacy ``targets[].rego`` and
    current ``targets[].code[engine=Rego].source.rego``. A template may embed
    several Rego blocks (policy + shared ``lib.*`` packages); all are returned,
    main package = the block defining ``violation`` rules."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    regos: list[str] = []
    for target in data.get("spec", {}).get("targets", []) or []:
        if target.get("rego"):
            regos.append(str(target["rego"]))
        for code in target.get("code", []) or []:
            if code.get("engine") == "Rego" and code.get("source", {}).get("rego"):
                regos.append(str(code["source"]["rego"]))
    main_pkg = None
    for rego in regos:
        if "violation" in rego:
            match = re.search(r"^package\s+(\S+)", rego, re.M)
            if match:
                main_pkg = match.group(1)
                break
    return {"regos": regos, "package": main_pkg} if main_pkg else None


def _candidate_triples(inner: Path) -> list[tuple[Path, Path, Path]]:
    """(policy_dir, sample_dir, disallowed_variant) with parseable Rego and a
    matching example_allowed.yaml. Deterministic order."""
    triples: list[tuple[Path, Path, Path]] = []
    lib = inner / "library" / "general"
    if not lib.is_dir():
        return triples
    for policy in sorted(p for p in lib.iterdir() if p.is_dir()):
        template = policy / "template.yaml"
        samples = policy / "samples"
        if not template.is_file() or not samples.is_dir() or _load_template(template) is None:
            continue
        for sample in sorted(s for s in samples.iterdir() if s.is_dir()):
            if not (sample / "constraint.yaml").is_file() or not (sample / "example_allowed.yaml").is_file():
                continue
            for variant in sorted(sample.glob("*.yaml")):
                if "disallowed" in variant.name:
                    triples.append((policy, sample, variant))
    return triples


PLUGIN = GatekeeperPolicyFamily()
