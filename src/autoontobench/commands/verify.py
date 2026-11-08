"""verify: replay-verify every task blueprint in ``--input`` through the native domain.

Gatekeeper blueprints (W2): ``reset`` -> replay the private solution plan (each
step must be accepted by the independent ``legal_action`` = pinned OPA) ->
``verify_terminal`` must report ``goal_reached``. ``--strict`` fails fast on the
first mismatch; without it every blueprint is attempted and the exit code is 1
if any failed. No network, no LLM.

Oracle/step logs are routed to ``data/erule/review5/integration/logs/verify/``
(W7-owned) so sibling workstreams' log trees stay untouched.
"""
from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

from autoontobench.contracts.models import TaskSpecV5
from autoontobench.core.domains.gatekeeper import GatekeeperDomain
from autoontobench.core.protocols import NativeState

REPO_ROOT = Path(__file__).resolve().parents[2]
INTEGRATION_LOGS = REPO_ROOT / "data" / "erule" / "review5" / "integration" / "logs"
BLUEPRINT_GLOB = "gk_blueprint_*.json"


def _replay_plan(
    domain: GatekeeperDomain, spec: TaskSpecV5, plan: List[Dict[str, Any]]
) -> Dict[str, Any]:
    state: NativeState = domain.reset(spec, seed=spec.seed)
    for i, action in enumerate(plan):
        decision = domain.legal_action(state, action)
        if not decision.legal:
            return {"ok": False, "stage": f"legal_action[{i}]",
                    "detail": {"action": action, "decision": decision.detail}}
        state = domain.step(state, action).state
    result = domain.verify_terminal(spec, state, None)
    if not result.goal_reached:
        failed_checks = [c for c in result.evidence.get("checks", []) if not c.get("pass")]
        return {"ok": False, "stage": "verify_terminal",
                "detail": {"failed_checks": failed_checks,
                           "terminal_state_sha256": result.evidence.get("terminal_state_sha256")}}
    return {"ok": True, "steps": len(plan),
            "terminal_state_sha256": result.evidence["terminal_state_sha256"],
            "evidence_sha256": result.evidence["evidence_sha256"]}


def _verify_one(
    domain: GatekeeperDomain, record_path: Path, input_dir: Path
) -> Tuple[str, Dict[str, Any]]:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
        spec = TaskSpecV5.model_validate(record["spec"])
        bid = record.get("blueprint_id", record_path.stem)
        plan_path = Path(record["private_solution_plan_path"])
        if not plan_path.is_file():
            plan_path = input_dir / "private" / f"{bid}.plan.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))["plan"]
    except (OSError, KeyError, ValueError) as exc:
        return record_path.stem, {"ok": False, "stage": "load",
                                  "detail": {"error": f"{type(exc).__name__}: {exc}"}}
    return bid, _replay_plan(domain, spec, plan)


def run(args: Namespace) -> int:
    if not args.input:
        sys.stderr.write("[verify] --input DIR is required\n")
        return 2
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.stderr.write(f"[verify] input dir not found: {input_dir}\n")
        return 2
    record_paths = sorted(input_dir.glob(BLUEPRINT_GLOB))
    if not record_paths:
        sys.stderr.write(f"[verify] no {BLUEPRINT_GLOB} under {input_dir}\n")
        return 1

    domain = GatekeeperDomain(log_root=INTEGRATION_LOGS / "verify")
    failures: List[Tuple[str, Dict[str, Any]]] = []
    verified = 0
    for record_path in record_paths:
        bid, outcome = _verify_one(domain, record_path, input_dir)
        if outcome["ok"]:
            verified += 1
            print(
                f"OK {bid} steps={outcome['steps']} "
                f"terminal={outcome['terminal_state_sha256'][:12]} "
                f"evidence={outcome['evidence_sha256'][:12]}"
            )
        else:
            failures.append((bid, outcome))
            detail = json.dumps(outcome["detail"], sort_keys=True)[:400]
            print(f"FAIL {bid} stage={outcome['stage']} detail={detail}", file=sys.stderr)
            if args.strict:
                break
    print(f"verified={verified} failed={len(failures)} total={len(record_paths)}")
    return 0 if not failures else 1
