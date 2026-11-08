"""Reproduce one L07 20K-candidate native-evidence check from an oracle payload.

Same contract as the legacy ``verify_cli``: recompute the reference answer from
pinned SBML bytes (never from the stored reference), check the pinned model
sha256, compare fresh vs stored reference tolerantly, and require the native
verifier to accept the stored reference. Exit 0 iff all checks pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmark.plugins.source.l07.simulate import PROTOCOL, load_roadrunner
from benchmark.plugins.task_family.l07.families_20k import BY_ID_20K, LANE, PROFILE
from benchmark.plugins.task_family.l07.verify_cli import _close
from benchmark.plugins.verifier.l07.verifier_20k import SbmlNativeVerifier20K
from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask
from source_registry.fetchers.l07_biomodels import INDEX_ROOT


def _grounded(payload: dict[str, Any]) -> GroundedTask:
    ast = payload["ast"]
    family = ast.get("family_id") or payload.get("family_id") or ""
    spec = BY_ID_20K.get(family)
    cand = CandidateTaskAST(
        lane_id=LANE,
        family_id=family,
        base_goal_id=ast.get("base_goal_id", "reconstructed"),
        instance_id=payload["instance_id"],
        regime=spec.regime if spec else "application",
        semantic_profile=PROFILE,
        ast=ast,
        source_unit_ids=(),
    )
    return GroundedTask(ast=cand, public={}, oracle_ref=payload["oracle_handle"],
                        native_evidence={})


def verify_oracle(oracle_path: Path, index_dir: Path) -> dict[str, Any]:
    payload = json.loads(Path(oracle_path).read_text(encoding="utf-8"))
    verifier = SbmlNativeVerifier20K(index_dir)
    ast = payload["ast"]
    checks: dict[str, Any] = {}
    ok = True

    model_id = ast["model_id"]
    recorded_sha = payload.get("model_sha256")
    fresh_sha = verifier.sim.struct(model_id).sha256
    checks["model_sha256_match"] = fresh_sha == recorded_sha
    ok &= checks["model_sha256_match"]

    fresh = verifier.reference_submission(ast)
    checks["reference_recomputed_matches_stored"] = _close(
        fresh, payload["reference_submission"])
    ok &= checks["reference_recomputed_matches_stored"]

    verdict = verifier.verify_terminal(_grounded(payload),
                                       payload["reference_submission"])
    checks["native_verifier_accepts_stored_reference"] = bool(verdict.valid)
    ok &= bool(verdict.valid)

    return {
        "ok": ok,
        "oracle": str(oracle_path),
        "task_id": payload.get("task_id"),
        "instance_id": payload.get("instance_id"),
        "family_id": payload.get("family_id"),
        "kind": ast.get("kind"),
        "solver": {
            "engine": "libsbml+libroadrunner",
            "libsbml_version": payload.get("solver", {}).get("libsbml_version"),
            "roadrunner_version": load_roadrunner().__version__,
            "protocol": dict(PROTOCOL),
        },
        "checks": checks,
        "verdict": "PASS" if ok else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute one L07 20K oracle verdict")
    ap.add_argument("--oracle", required=True, help="path to an oracle/*.json payload")
    ap.add_argument("--index", default=str(INDEX_ROOT), help="screening index dir")
    args = ap.parse_args(argv)
    result = verify_oracle(Path(args.oracle), Path(args.index))
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())