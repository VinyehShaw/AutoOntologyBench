"""L09x NativeVerifierPlugin: deterministic structural verification of the
SMART-ANC 20K-expansion tasks (families f07..f14).

Mirrors the legacy L09 verifier's contract but writes only to
``bank/staging_20k/l09/`` — the legacy 8K staging/build is never touched.
All checks are recomputed in pure Python from the pinned snapshot
(``l09x.engine.AncKbX``); no CQL evaluation runtime exists on this host
(Java absent) and no task depends on one. Independence: the verifier
re-derives every verdict from the pinned artifacts; generator output is
never trusted as gold.

- ``prove_exists``: family check on the initial (corrupted) artifact must
  violate; on the pristine reference artifact must pass. Evidence persisted
  under ``bank/staging_20k/l09/evidence/<instance_id>/``.
- ``verify_terminal``: submitted artifact passes iff (a) family check passes,
  (b) all guards hold, (c) every corrupted slot carries the reference value.
- ``certify_semantic_relevance``: knowledge-side counterfactual — perturb the
  KB view (drop the referenced ValueSet canonical / ELM statement /
  CodeSystem code / related-artifact or include-target canonical) and prove
  the reference artifact then violates the same check.

CLI (independent reproduction, recorded in every qualified record's evidence):

    .venv/bin/python -m benchmark.plugins.verifier.l09x.structural_verifier \
        reproduce --evidence-dir bank/staging_20k/l09/evidence/<instance_id>

Exit 0 iff the initial artifact violates and the reference artifact passes.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from benchmark.schema.plugin_protocols import (
    GroundedTask,
    NativeVerdict,
    ProofOrCounterexample,
    SemanticWitness,
)
from benchmark.plugins.task_family.l09x import engine
from benchmark.plugins.task_family.l09x.engine import AncKbX, KbPerturbationX

REPO_ROOT = Path(__file__).resolve().parents[4]
LANE_ROOT = REPO_ROOT / "bank" / "staging_20k" / "l09"
ORACLE_DIR = LANE_ROOT / "oracle"
EVIDENCE_DIR = LANE_ROOT / "evidence"


def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _artifact_payload_doc(payload: dict[str, Any]) -> Any:
    art = payload.get("artifact") or {}
    if art.get("format") == "cql-text":
        return art.get("text") or ""
    return art.get("content") or {}


def run_check(mechanism: str, artifact: Any, kb: AncKbX) -> list[dict[str, Any]]:
    if mechanism == "check_questionnaire_valueset_bindings":
        return engine.check_questionnaire_valueset_bindings(artifact, kb)
    if mechanism == "check_measure_criteria":
        return engine.check_measure_criteria(artifact, kb)
    if mechanism == "check_conceptmap_source_membership":
        return engine.check_conceptmap_source_membership(artifact, kb)
    if mechanism == "check_library_datarequirements":
        return engine.check_library_datarequirements(artifact, kb)
    if mechanism == "check_valueset_displays":
        return engine.check_valueset_displays(artifact, kb)
    if mechanism == "check_library_related_artifacts":
        return engine.check_library_related_artifacts(artifact, kb)
    if mechanism == "check_library_include_coverage":
        return engine.check_library_include_coverage(artifact, kb)
    if mechanism == "check_cql_valueset_declarations":
        return engine.check_cql_valueset_declarations(artifact, kb)
    raise ValueError("unknown mechanism %s" % mechanism)


def reference_artifact(ast: dict[str, Any]) -> Any:
    path = ast["artifact_path"]
    if ast["artifact_format"] == "cql-text":
        return engine.ANC_ROOT.joinpath(path).read_text(encoding="utf-8")
    return engine.load_artifact(path)


def _slot_holds(submitted: Any, slot: dict[str, Any]) -> bool:
    if "path" in slot and slot["path"]:
        return engine.get_path(submitted, slot["path"]) == slot["value"]
    find = slot["find"]
    expected_count = slot.get("count", 1)
    return submitted.count(find) == expected_count and slot["wrong_value"] not in submitted


def _guard_holds(submitted: Any, guard: dict[str, Any]) -> bool:
    if "path" in guard:
        return engine.get_path(submitted, guard["path"]) == guard["value"]
    needle = guard["must_contain_count"]
    return submitted.count(needle) == guard["count"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _perturbation_for(ast: dict[str, Any], reference: Any, kb: AncKbX) -> KbPerturbationX | None:
    mech = ast["mechanism"]
    if mech == "check_questionnaire_valueset_bindings":
        for _path, item in engine.iter_questionnaire_items(reference):
            avs = item.get("answerValueSet")
            if avs:
                return KbPerturbationX(drop_canonical=avs.split("|")[0])
    elif mech == "check_measure_criteria":
        libs = reference.get("library") or []
        for slot in engine._measure_criteria_slots(reference):
            for canon in libs:
                if slot["value"] in kb.library_elm_statements.get(canon, frozenset()):
                    return KbPerturbationX(drop_elm_statement=(canon, slot["value"]))
    elif mech == "check_conceptmap_source_membership":
        for slot in engine._conceptmap_element_slots(reference):
            return KbPerturbationX(drop_codesystem_code=slot["value"])
    elif mech == "check_library_datarequirements":
        for slot in engine._codefilter_slots(reference):
            return KbPerturbationX(drop_canonical=slot["value"].split("|")[0])
    elif mech == "check_valueset_displays":
        for slot in engine._display_slots(reference):
            return KbPerturbationX(drop_codesystem_code=slot["code"])
    elif mech == "check_library_related_artifacts":
        for slot in engine._related_artifact_slots(reference):
            return KbPerturbationX(drop_canonical=slot["value"].split("|")[0])
    elif mech == "check_library_include_coverage":
        name = reference.get("name")
        includes = kb.cql_includes.get(name, frozenset())
        covered = {s["value"].split("|")[0] for s in engine._related_artifact_slots(reference)}
        for inc in sorted(includes):
            canon = kb.include_canonical_by_name.get(inc)
            if canon and canon in covered:
                return KbPerturbationX(drop_canonical=canon)
    elif mech == "check_cql_valueset_declarations":
        for decl in engine.parse_valueset_declarations(reference):
            return KbPerturbationX(drop_canonical=decl["url"].split("|")[0])
    return None


class SmartAncExpansionStructuralVerifier:
    """NativeVerifierPlugin for the L09 20K-expansion families."""

    plugin_id = "l09x.structural-verifier"

    def prove_exists(self, grounded_task: GroundedTask) -> ProofOrCounterexample:
        ast = grounded_task.ast.ast
        mech = ast["mechanism"]
        kb = engine.load_kbx()
        initial = _artifact_payload_doc(grounded_task.public)
        reference = reference_artifact(ast)
        ctx = grounded_task.ast.instance_id

        init_violations = run_check(mech, initial, kb)
        ref_violations = run_check(mech, reference, kb)

        evidence_dir = EVIDENCE_DIR / ctx
        evidence_dir.mkdir(parents=True, exist_ok=True)
        ext = "txt" if ast["artifact_format"] == "cql-text" else "json"
        (evidence_dir / ("initial.%s" % ext)).write_text(
            initial if isinstance(initial, str)
            else json.dumps(initial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        (evidence_dir / ("reference.%s" % ext)).write_text(
            reference if isinstance(reference, str)
            else json.dumps(reference, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        _write_json_atomic(evidence_dir / "meta.json", {
            "instance_id": ctx,
            "mechanism": mech,
            "artifact_format": ast["artifact_format"],
            "artifact_path": ast["artifact_path"],
            "source_commit": engine.ANC_COMMIT,
            "profile": engine.PROFILE,
            "engine_version": engine.ENGINE_VERSION,
            "dataset_id": "autoontobench-bank-20000-candidate-dev1",
        })

        reproduce = (
            ".venv/bin/python -m benchmark.plugins.verifier.l09x.structural_verifier "
            "reproduce --evidence-dir %s" % evidence_dir
        )
        evidence: dict[str, Any] = {
            "goal_kind": ast["goal_kind"],
            "mechanism": mech,
            "reproduce_command": reproduce,
            "evals": [
                engine.evidence_record(mech, initial, init_violations,
                                       context="%s:init" % ctx),
                engine.evidence_record(mech, reference, ref_violations,
                                       context="%s:ref" % ctx),
            ],
        }
        if not init_violations:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="TRIVIAL: initial artifact already passes %s" % mech)
        if ref_violations:
            return ProofOrCounterexample(
                exists=False, evidence=evidence,
                reason="UNSOLVABLE: reference artifact violates %s: %s"
                       % (mech, json.dumps(ref_violations)[:300]))
        evidence["reference_artifact_sha256"] = engine.sha256_json(reference)
        return ProofOrCounterexample(exists=True, evidence=evidence)

    def verify_terminal(
        self, task: GroundedTask, submitted_state_or_answer: Any
    ) -> NativeVerdict:
        ast = task.ast.ast
        oracle = self._oracle(task)
        if oracle is None:
            return NativeVerdict(valid=False, detail={"reason": "oracle_missing"})
        submitted = _artifact_payload_doc(submitted_state_or_answer or {})
        if not submitted:
            return NativeVerdict(valid=False, detail={"reason": "empty_submission"})
        kb = engine.load_kbx()
        checks: list[dict[str, Any]] = []
        violations = run_check(ast["mechanism"], submitted, kb)
        checks.append({
            "check": ast["mechanism"],
            "pass": not violations,
            "violation_count": len(violations),
            "violations": violations[:20],
            "artifact_sha256": engine.sha256_json(submitted),
        })
        ok = not violations
        for slot in oracle["slots"]:
            held = _slot_holds(submitted, slot)
            checks.append({"check": "slot_exact", "slot": slot.get("path") or slot.get("find"),
                           "pass": held})
            ok = ok and held
        for guard in oracle["guards"]:
            held = _guard_holds(submitted, guard)
            checks.append({"check": "guard",
                           "guard": guard.get("path") or guard.get("must_contain_count", "")[:40],
                           "pass": held})
            ok = ok and held
        return NativeVerdict(valid=ok, detail={"checks": checks})

    def certify_semantic_relevance(self, task: GroundedTask) -> SemanticWitness:
        ast = task.ast.ast
        kb = engine.load_kbx()
        reference = reference_artifact(ast)
        ctx = task.ast.instance_id
        correct_violations = run_check(ast["mechanism"], reference, kb)
        if correct_violations:
            return SemanticWitness(ok=False, witness={
                "reason": "reference artifact fails check on unperturbed KB",
                "violations": correct_violations[:5],
            })
        perturb = _perturbation_for(ast, reference, kb)
        if perturb is None:
            return SemanticWitness(ok=False, witness={"reason": "no perturbation available"})
        perturbed_kb = kb.with_perturbation_x(perturb)
        wrong_violations = run_check(ast["mechanism"], reference, perturbed_kb)
        if not wrong_violations:
            return SemanticWitness(ok=False, witness={
                "reason": "perturbation did not flip the check",
                "perturbation": {k: _jsonable(v) for k, v in perturb.__dict__.items() if v},
            })
        return SemanticWitness(ok=True, witness={
            "counterfactuals": [{
                "mode": "knowledge_perturbation",
                "perturbation": {k: _jsonable(v) for k, v in perturb.__dict__.items() if v},
                "correct_violation_count": len(correct_violations),
                "wrong_violation_count": len(wrong_violations),
                "wrong_violations": wrong_violations[:5],
                "correct_evidence": engine.evidence_record(
                    ast["mechanism"], reference, correct_violations,
                    context="%s:sem-correct" % ctx),
            }],
        })

    def write_oracle(self, grounded_task: GroundedTask, extra: dict[str, Any]) -> Path:
        ast = grounded_task.ast.ast
        guards = _guards_for(grounded_task)
        payload = {
            "oracle_ref": grounded_task.oracle_ref,
            "mechanism": ast["mechanism"],
            "artifact_path": ast["artifact_path"],
            "artifact_format": ast["artifact_format"],
            "slots": [
                {k: v for k, v in s.items() if k in ("path", "find", "value", "wrong_value", "kind", "count")}
                for s in _slots_with_counts(grounded_task)
            ],
            "guards": guards,
            "reference_artifact_sha256": extra.get("reference_artifact_sha256"),
            "engine": engine.ENGINE,
            "engine_version": engine.ENGINE_VERSION,
            "semantic_profile": engine.PROFILE,
            "source_commit": engine.ANC_COMMIT,
        }
        path = ORACLE_DIR / ("%s.json" % grounded_task.oracle_ref)
        _write_json_atomic(path, payload)
        return path

    def _oracle(self, task: GroundedTask) -> dict[str, Any] | None:
        path = ORACLE_DIR / ("%s.json" % task.oracle_ref)
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))


def _slots_with_counts(grounded_task: GroundedTask) -> list[dict[str, Any]]:
    ast = grounded_task.ast.ast
    slots = [dict(s) for s in ast["slots"]]
    if ast["artifact_format"] == "cql-text":
        reference = reference_artifact(ast)
        for s in slots:
            s["count"] = reference.count(s["find"])
    return slots


def _guards_for(grounded_task: GroundedTask) -> list[dict[str, Any]]:
    from benchmark.plugins.task_family.l09x import families

    ast = grounded_task.ast.ast
    reference = reference_artifact(ast)
    corrupted = {s.get("path") or s.get("find") for s in ast["slots"]}
    if ast["artifact_format"] == "cql-text":
        slot_fn = families.slots_f14
        return [
            {"must_contain_count": s["find"], "count": reference.count(s["find"])}
            for s in slot_fn(reference)
            if s["find"] not in corrupted
        ]
    slot_fn = {
        "check_questionnaire_valueset_bindings": families.slots_f07,
        "check_measure_criteria": families.slots_f08,
        "check_conceptmap_source_membership": families.slots_f09,
        "check_library_datarequirements": families.slots_f10,
        "check_valueset_displays": families.slots_f11,
        "check_library_related_artifacts": families.slots_f12,
        "check_library_include_coverage": families.slots_f13,
    }[ast["mechanism"]]
    guards = [
        {"path": p, "value": engine.get_path(reference, p)}
        for p in families.IDENTITY_PATHS
        if engine.get_path(reference, p) is not None
    ]
    guards += [
        {"path": s["path"], "value": s["value"]}
        for s in slot_fn(reference)
        if s["path"] not in corrupted
    ]
    return guards


# ---------------------------------------------------------------------------
# independent reproduction CLI
# ---------------------------------------------------------------------------

def reproduce(evidence_dir: Path) -> dict[str, Any]:
    """Recompute the prove_exists verdicts from persisted evidence artifacts.

    The audit evaluates the instance's declared mechanism (meta.json): a lane
    task is scoped to one native check; artifacts may carry upstream quirks
    under unrelated checks, which are outside the task's contract."""
    init = next(evidence_dir.glob("initial.*"))
    ref = next(evidence_dir.glob("reference.*"))
    initial: Any = init.read_text(encoding="utf-8")
    reference: Any = ref.read_text(encoding="utf-8")
    if init.suffix == ".json":
        initial = json.loads(initial)
        reference = json.loads(reference)
    meta_path = evidence_dir / "meta.json"
    if meta_path.is_file():
        mechanisms = [json.loads(meta_path.read_text(encoding="utf-8"))["mechanism"]]
    else:
        raise SystemExit("meta.json missing in %s" % evidence_dir)
    kb = engine.load_kbx()
    out = {}
    for name, art in (("initial", initial), ("reference", reference)):
        out[name] = {
            "artifact_sha256": engine.sha256_json(art),
            "violations_by_check": {
                mech: len(run_check(mech, art, kb)) for mech in mechanisms
            },
        }
    out["mechanisms_audited"] = mechanisms
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="L09x verifier reproduction CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)
    rep = sub.add_parser("reproduce")
    rep.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "reproduce":
        result = reproduce(args.evidence_dir)
        print(json.dumps(result, indent=2, sort_keys=True))
        init_bad = any(result["initial"]["violations_by_check"].values())
        ref_bad = any(result["reference"]["violations_by_check"].values())
        return 0 if (init_bad and not ref_bad) else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
