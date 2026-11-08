"""L04 TaskFamilyPlugin facade (FINAL-BUILD-ORDER-v2 §3.2).

Thin adapter over the 25 shipped families in ``families.py``: ``propose``
streams CandidateTaskASTs (params + fast-path reference answer + instance
state) and ``ground`` attaches the public goal text and independent SPARQL
native evidence. Production-scale batch building lives in ``builder.py``;
this facade is the factory-facing single-task path.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from benchmark.schema.plugin_protocols import (
    CandidateTaskAST, GroundedTask, SourceSnapshot,
)
from benchmark.plugins.source.l04.kb import load_kb
from benchmark.plugins.task_family.l04.families import (
    ALLOWED_TOOLS, FAMILIES, FAMILY_BY_ID, INSTANCES_PER_GOAL,
    base_goal_id, canonical_params, sha_text, source_unit_ids,
    state_descriptor, _master_cache,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PIN_ROOT = (REPO_ROOT / "external_knowledge" / "pinned" / "rhea-chebi"
                    / "rhea-142+chebi-256")
LANE_ID = "L04"


class L04TaskFamilies:
    """§3.2 task_family surface for the pinned Rhea 142 + ChEBI 256 pair."""

    lane_id = LANE_ID

    def __init__(self, pin_root: str | Path | None = None):
        self.pin_root = Path(pin_root) if pin_root else DEFAULT_PIN_ROOT

    def supported_profiles(self) -> list[str]:
        return sorted({f.profile for f in FAMILIES})

    def propose(
        self, snapshot: SourceSnapshot, *, family: str, seed: int,
        constraints: dict[str, Any],
    ) -> Iterator[CandidateTaskAST]:
        fam = FAMILY_BY_ID[family.split(".", 1)[-1]]
        kb = load_kb(str(snapshot.root))
        masters = _master_cache(kb)
        rng = random.Random(seed)
        n_goals = int(constraints.get("n_goals", fam.max_base_goals))
        seen: set[str] = set()
        emitted = 0
        for params in fam.propose(kb, rng):
            key = canonical_params(params)
            if key in seen:
                continue
            seen.add(key)
            try:
                answer = tuple(sorted(set(fam.answer(kb, params, masters))))
            except Exception:
                continue
            if not (fam.min_answer <= len(answer) <= fam.max_answer):
                continue
            bid = base_goal_id(fam.family_id, params)
            for inst_idx in range(INSTANCES_PER_GOAL):
                state = state_descriptor(fam, seed, inst_idx)
                instance_id = sha_text(
                    f"{bid}|{json.dumps(state, sort_keys=True)}")[:32]
                yield CandidateTaskAST(
                    lane_id=LANE_ID,
                    family_id=f"l04.{fam.family_id}",
                    base_goal_id=bid,
                    instance_id=instance_id,
                    regime=fam.regime,
                    semantic_profile=fam.profile,
                    ast={
                        "params": params,
                        "answer": list(answer),
                        "state": state,
                        "complexity": fam.complexity,
                        "witness_kind": fam.witness_kind,
                        "seed": seed,
                        "generator": "l04-families/1.0",
                    },
                    source_unit_ids=source_unit_ids(params),
                )
            emitted += 1
            if emitted >= n_goals:
                return

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        from benchmark.plugins.verifier.l04 import plugin as verifier
        fam = FAMILY_BY_ID[candidate.family_id.split(".", 1)[-1]]
        params = dict(candidate.ast["params"])
        answer = tuple(candidate.ast["answer"])
        kb = load_kb(str(self.pin_root))
        ev = verifier.verify_reference(self.pin_root, fam.family_id, params,
                                       answer, kb)
        public = {
            "user_goal": fam.goal_text(params, kb),
            "allowed_tools": list(ALLOWED_TOOLS),
            "budget": {"actions": 4, "probes": 0},
            "observation_refs": [],
        }
        oracle_ref = "l04-oracle-" + sha_text(candidate.instance_id)[:16]
        return GroundedTask(
            ast=candidate,
            public=public,
            oracle_ref=oracle_ref,
            native_evidence={
                "engine": ev.get("engine"),
                "verdict": ev.get("verdict"),
                "query_sha256": ev.get("query_sha256"),
                "graph_manifest_sha": ev.get("graph_manifest_sha"),
                "answer_sha256": ev.get("answer_sha256")
                                 or ev.get("expected_sha256"),
            },
        )


PLUGIN = L04TaskFamilies()
