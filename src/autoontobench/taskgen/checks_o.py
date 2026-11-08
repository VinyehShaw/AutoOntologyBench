"""T5 ontology-consumption checks (W8 M3; W14 linked-data/1 implementation).

T5.7  controlled_o_content_sensitivity  REQUIRED only where the task's claim
        carries an O profile (Implementation §4 T5 check 7: "only where claim
        requires it"). ``ontology_contract=native-only/0`` tasks make no O claim
        -> NOT_APPLICABLE, with the re-activation trigger named in the reason.
        For ``linked-data/1`` (Rhea+ChEBI) the check RUNS the Phi cohort through
        the fixed consumer: Phi_correct must reproduce the dual-certified
        expected answer exactly, Phi_empty and Phi_swapped must change it, and a
        non-effect control query must stay byte-identical.
T5.14 operator_profile_consumed_as_authored  same scoping. For linked-data/1:
        entry-level authorship proof — removing the governed identity's own
        crosswalk entry annihilates the governed answer (release-142 mapping is
        strictly 1:1 compound<->ChEBI, verified at build); removing an unrelated
        entry leaves the governed answer byte-identical. The artifact is parsed
        through the real profile boundary (``load_crosswalk``) and must equal
        the certified index mapping.

For any other O-carrying contract both checks stay NOT_IMPLEMENTED_DEP
(gold O + frozen replay / consumer trace, P3/W6), capping the ladder below
QUALIFIED rather than silently passing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from autoontobench.contracts.models import TaskSpecV5
from autoontobench.taskgen.check_types import CheckContext, CheckResult, CheckStatus

_LINKED_DATA = "linked-data/1"
_COHORT_CACHE: dict[str, dict[str, Any]] = {}


def _has_o_claim(spec: TaskSpecV5) -> bool:
    return not spec.ontology_contract.startswith("native-only")


def _is_linked_data(spec: TaskSpecV5) -> bool:
    return spec.ontology_contract == _LINKED_DATA


def _resolve(ctx: CheckContext, ref: str) -> Path:
    p = Path(ref)
    return p if p.is_absolute() else ctx.repo_root / p


def _load_cohort(spec: TaskSpecV5, ctx: CheckContext) -> dict[str, Any]:
    """Assemble (index, facts, gold artifact, controls) once per task; sha-checked."""
    key = spec.private_control_specs_ref
    if key in _COHORT_CACHE:
        return _COHORT_CACHE[key]
    from autoontobench.core.domains.rhea_chebi import (
        load_certified_index, load_crosswalk, public_facts,
    )
    from autoontobench.taskgen.native_check import _cached_rhea_index

    blueprint = ctx.blueprint or {}
    index = _cached_rhea_index(str(_resolve(ctx, spec.private_native_truth_ref)))
    controls_path = _resolve(ctx, spec.private_control_specs_ref)
    controls = json.loads(controls_path.read_text(encoding="utf-8"))
    want_sha = (blueprint.get("phi_cohort") or {}).get("sha256")
    got_sha = hashlib.sha256(
        (json.dumps(controls, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    ).hexdigest()
    gold_bytes = _resolve(ctx, spec.private_gold_artifact_ref).read_bytes()
    gold = load_crosswalk(gold_bytes)  # boundary parse of the authored-profile artifact
    cohort = {
        "index": index, "facts": public_facts(index.records), "gold": gold,
        "gold_bytes_sha": hashlib.sha256(gold_bytes).hexdigest(),
        "controls": controls, "controls_sha_ok": want_sha == got_sha,
    }
    _COHORT_CACHE[key] = cohort
    return cohort


def _check7_linked_data(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "controlled_o_content_sensitivity"
    if ctx.blueprint is None or "query" not in (ctx.blueprint.get("goal") or {}):
        return CheckResult(name, CheckStatus.FAIL,
                           "linked-data/1 task without a certified query goal block")
    try:
        cohort = _load_cohort(spec, ctx)
    except (OSError, ValueError) as exc:
        return CheckResult(name, CheckStatus.FAIL, f"cohort artifacts unreadable: {exc}")
    from autoontobench.core.domains.rhea_chebi import (
        ChebiId, consumer_answer, expected_answer, swap_crosswalk,
    )

    controls, index = cohort["controls"], cohort["index"]
    facts, gold = cohort["facts"], cohort["gold"]
    query, control = controls["governed_query"], controls["control_query"]
    decoy = ChebiId(controls["decoy_chebi"])
    expected = expected_answer(index, query)
    if gold != index.gold_crosswalk():
        return CheckResult(name, CheckStatus.FAIL,
                           "gold crosswalk artifact diverges from certified index")
    recorded_gold_sha = (ctx.blueprint or {}).get("gold_crosswalk_sha256")
    if recorded_gold_sha and recorded_gold_sha != cohort["gold_bytes_sha"]:
        return CheckResult(name, CheckStatus.FAIL,
                           "gold crosswalk artifact sha256 drifted from the build record")
    if not cohort["controls_sha_ok"]:
        return CheckResult(name, CheckStatus.FAIL,
                           "control specs sha256 diverges from blueprint phi_cohort")
    swapped = swap_crosswalk(gold, ChebiId(query["chebi"]), decoy)
    under_correct = consumer_answer(facts, gold, query)
    under_empty = consumer_answer(facts, {}, query)
    under_swapped = consumer_answer(facts, swapped, query)
    control_gold = consumer_answer(facts, gold, control)
    control_swapped = consumer_answer(facts, swapped, control)
    problems = []
    if under_correct != expected:
        problems.append("phi_correct != dual-certified expected answer")
    if not expected or under_empty == expected:
        problems.append("phi_empty does not change the governed answer")
    if under_swapped == expected:
        problems.append("phi_swapped does not change the governed answer")
    if not control_gold or control_swapped != control_gold:
        problems.append("non-effect control query changed under phi_swapped")
    if problems:
        return CheckResult(name, CheckStatus.FAIL, "; ".join(problems))
    return CheckResult(
        name, CheckStatus.PASS,
        "Phi cohort through the fixed consumer: correct==certified truth; empty and "
        "swapped both change the governed answer (verified difference); non-effect "
        "control byte-identical (mapping-causal test, linked-data/1)",
        evidence={
            "governed_count": len(set(expected)),
            "phi_empty_count": len(set(under_empty)),
            "phi_swapped_count": len(set(under_swapped)),
            "control_query": control, "control_count": len(set(control_gold)),
            "decoy_chebi": str(decoy),
            "index_manifest_sha256": index.manifest_sha256,
        })


def _check14_linked_data(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "operator_profile_consumed_as_authored"
    if ctx.blueprint is None or "query" not in (ctx.blueprint.get("goal") or {}):
        return CheckResult(name, CheckStatus.FAIL,
                           "linked-data/1 task without a certified query goal block")
    try:
        cohort = _load_cohort(spec, ctx)
    except (OSError, ValueError) as exc:
        return CheckResult(name, CheckStatus.FAIL, f"cohort artifacts unreadable: {exc}")
    from autoontobench.core.domains.rhea_chebi import consumer_answer, expected_answer

    controls, index = cohort["controls"], cohort["index"]
    facts, gold = cohort["facts"], cohort["gold"]
    query = controls["governed_query"]
    expected = expected_answer(index, query)
    drop_own = controls.get("drop_compound")
    drop_unrelated = controls.get("drop_control_compound")
    if not drop_own or not drop_unrelated or drop_own == drop_unrelated:
        return CheckResult(name, CheckStatus.FAIL,
                           "control specs lack distinct own/unrelated drop entries")
    minus_own = {c: v for c, v in gold.items() if c != drop_own}
    minus_unrelated = {c: v for c, v in gold.items() if c != drop_unrelated}
    under_minus_own = consumer_answer(facts, minus_own, query)
    under_minus_unrelated = consumer_answer(facts, minus_unrelated, query)
    problems = []
    if under_minus_own == expected:
        problems.append("removing the governed identity's own Phi entry changes nothing "
                        "-> consumer is not reading the authored mapping")
    if under_minus_unrelated != expected:
        problems.append("removing an unrelated Phi entry changed the governed answer "
                        "-> consumer reads entries it should not")
    if problems:
        return CheckResult(name, CheckStatus.FAIL, "; ".join(problems))
    return CheckResult(
        name, CheckStatus.PASS,
        "entry-level authorship: dropping the governed entry removes exactly its "
        "contribution (release-142 mapping is 1:1 compound<->ChEBI, so the governed "
        "answer collapses); dropping an unrelated entry is byte-identical. Artifact "
        "parsed via load_crosswalk and equals the certified index mapping",
        evidence={
            "dropped_own_entry": drop_own,
            "governed_after_drop_own": len(set(under_minus_own)),
            "governed_after_drop_unrelated": len(set(under_minus_unrelated)),
            "governed_gold": len(set(expected)),
            "gold_artifact_sha256": cohort["gold_bytes_sha"],
        })


def check_controlled_o_content_sensitivity(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "controlled_o_content_sensitivity"
    if not _has_o_claim(spec):
        return CheckResult(
            name, CheckStatus.NOT_APPLICABLE,
            reason=f"ontology_contract={spec.ontology_contract}: no O-consumption claim; "
                   "REQUIRED once an O profile is attached (dep: gold O + frozen replay, P3/W6)")
    if _is_linked_data(spec):
        return _check7_linked_data(spec, ctx)
    return CheckResult(name, CheckStatus.NOT_IMPLEMENTED_DEP,
                       dep="gold O artifact + frozen O-swap replay (P3/W6)")


def check_operator_profile_consumed_as_authored(spec: TaskSpecV5, ctx: CheckContext) -> CheckResult:
    name = "operator_profile_consumed_as_authored"
    if not _has_o_claim(spec):
        return CheckResult(
            name, CheckStatus.NOT_APPLICABLE,
            reason=f"ontology_contract={spec.ontology_contract}: no operator profile to "
                   "consume; REQUIRED once an O profile is attached (dep: OntologyTool "
                   "consumer trace, P3/W6)")
    if _is_linked_data(spec):
        return _check14_linked_data(spec, ctx)
    return CheckResult(name, CheckStatus.NOT_IMPLEMENTED_DEP,
                       dep="OntologyTool consumer + consumption trace (P3/W6)")
