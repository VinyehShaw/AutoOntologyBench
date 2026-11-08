"""L15 task-family plugin: propose CandidateTaskASTs over the pinned store."""
from __future__ import annotations

from pathlib import Path
from random import Random
from typing import Any, Iterator

from autoontobench.contracts.models import canonical_sha
from benchmark.plugins.source.l15.kb import L15KB
import os
from benchmark.plugins.source.l15.plugin import STORE_DB
from benchmark.plugins.task_family.l15 import goals as G
from benchmark.plugins.task_family.l15.families import BY_ID, LANE, PROFILE, FamilySpec
from benchmark.schema.plugin_protocols import CandidateTaskAST

MAX_PROPOSE_ATTEMPTS = 400


class L15TaskFamilyPlugin:
    """TaskFamilyPlugin for the 15 L15 families (see families.FAMILY_TABLE)."""

    def __init__(self, db_path: str | Path | None = None, *, seed: int) -> None:
        self.kb = L15KB(db_path or os.environ.get("L15_STORE_DB") or STORE_DB)
        self.seed = seed

    def supported_profiles(self) -> list[str]:
        return [PROFILE]

    def propose(self, snapshot: Any = None, *, family: str, seed: int | None = None,
                constraints: dict[str, Any] | None = None) -> Iterator[CandidateTaskAST]:
        spec = BY_ID[family]
        base_seed = self.seed if seed is None else seed
        max_goals = (constraints or {}).get("max_base_goals", spec.base_goals)
        ops = G.KINDS[spec.kind]
        seen_params: set[str] = set()
        goals_done = 0
        for attempt in range(MAX_PROPOSE_ATTEMPTS):
            if goals_done >= max_goals:
                break
            rng = Random(f"{family}|base|{attempt}|{base_seed}")
            try:
                params = ops.propose(self.kb, rng)
            except Exception:
                continue
            if params is None:
                continue
            pkey = canonical_sha({"family": family, "params": params})
            if pkey in seen_params:
                continue
            seen_params.add(pkey)
            base_goal_id = "l15b-" + pkey[:12]
            instances = []
            for inst_seq in range(spec.instances_per_goal):
                inst_rng = Random(f"{family}|inst|{pkey}|{inst_seq}|{base_seed}")
                variant = "records" if inst_seq % 2 == 0 else "table"
                ast = dict(params)
                ast["kind"] = spec.kind
                ast["complexity"] = spec.complexity
                ast["evidence_view"] = variant
                ast["distractor_seed"] = inst_rng.randrange(1 << 30)
                instance_id = "l15i-" + canonical_sha(
                    {"base": base_goal_id, "seq": inst_seq})[:16]
                instances.append(CandidateTaskAST(
                    lane_id=LANE, family_id=family, base_goal_id=base_goal_id,
                    instance_id=instance_id, regime=spec.regime,
                    semantic_profile=PROFILE, ast=ast,
                    source_unit_ids=_unit_ids(spec.kind, params)))
            yield from instances
            goals_done += 1

    def ground(self, candidate: CandidateTaskAST) -> Any:
        from benchmark.plugins.task_family.l15.ground import ground_candidate
        from benchmark.schema.plugin_protocols import GroundedTask
        public = ground_candidate(self.kb, candidate.ast, candidate.family_id,
                                  candidate.base_goal_id, candidate.instance_id,
                                  0, self.seed)
        oracle_ref = "orl15_" + canonical_sha(
            {"instance": candidate.instance_id})[:16]
        return GroundedTask(ast=candidate, public=public, oracle_ref=oracle_ref,
                            native_evidence={})


def _unit_ids(kind: str, params: dict[str, Any]) -> tuple[str, ...]:
    out: list[str] = []
    for key in ("parent", "pathway", "reaction", "complex", "event", "accession",
                "file", "event_name"):
        if key in params:
            out.append(f"{key}:{params[key]}")
    return tuple(out)
