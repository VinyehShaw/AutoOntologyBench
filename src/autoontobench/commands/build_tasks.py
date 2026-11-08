"""build-tasks: offline solver-grounded blueprint generation (Review 5B §7 P2).

Wired: ``--source gatekeeper`` via W2's ``core/task_engine.build_blueprints``
(goal natively verified -> defects derive non-goal initial -> BFS over
pinned-OPA-legal patch actions -> replay-certify). Deterministic; no network, no
LLM — the engine has no network code path, so ``--offline`` is a recorded
declaration, not a behavioral switch. Other sources exit 2 naming their owner
workstream.

Oracle/step logs are routed to ``data/erule/review5/integration/logs/`` (W7-owned)
so sibling workstreams' log trees stay untouched.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_LOGS = REPO_ROOT / "data" / "erule" / "review5" / "integration" / "logs"
DEFAULT_SEED = 137

#: owner workstream per unwired source id (registry or matrix spelling)
SOURCE_OWNERS = {
    "rhea": "W3", "chebi": "W3", "rdkit": "W3", "go": "W3",
    "cql": "W4", "cql_examples": "W4", "smart_anc": "W4", "who_smart_anc": "W4",
    "biomodels": "W4", "behavior": "W4", "behavior_1k_bddl": "W4",
    "cel": "W2", "cel_kubernetes_vap": "W2", "camunda": "W2", "camunda_dmn": "W2",
}


def run(args: Namespace) -> int:
    source = args.source
    if not source:
        sys.stderr.write("[build-tasks] --source is required (wired: gatekeeper)\n")
        return 2
    if source != "gatekeeper":
        owner = SOURCE_OWNERS.get(source, "W2/W3/W4")
        sys.stderr.write(
            f"[build-tasks] source '{source}' not wired into the CLI yet. Owner: {owner}.\n"
        )
        return 2
    if not args.out:
        sys.stderr.write("[build-tasks] --out DIR is required\n")
        return 2

    from autoontobench.core.task_engine import build_blueprints  # deferred: pulls the OPA chain

    out_dir = Path(args.out)
    log_root = INTEGRATION_LOGS / "gatekeeper"
    log_root.mkdir(parents=True, exist_ok=True)
    seed = args.seed if args.seed is not None else DEFAULT_SEED
    records = build_blueprints(out_dir, log_root, seed=seed, limit=args.limit)
    print(
        f"source=gatekeeper generated={len(records)} seed={seed} "
        f"offline={bool(args.offline)} out={out_dir}"
    )
    for record in records:
        print(
            f"  {record['blueprint_id']}: min_actions={record['min_action_count']} "
            f"interacting_units={record['interacting_units']} "
            f"replay_goal={record['replay']['goal_reached']}"
        )
    return 0 if records else 1
