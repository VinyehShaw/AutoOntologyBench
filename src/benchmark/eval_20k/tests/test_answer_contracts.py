"""answer_contracts tests: per-lane prompt contract hints and the post-parse
alias normaliser. All offline — no API calls, no oracle mounts.
"""
from __future__ import annotations

from benchmark.eval_20k import model_runner
from benchmark.eval_20k.answer_contracts import contract_hint, normalize_answer
from benchmark.eval_20k.model_runner import build_messages

from .test_model_runner import make_task_row


def _task(lane: str, family: str = "", regime: str = "application",
          public: dict | None = None):
    row = make_task_row(lane, 1)
    row["family_id"] = family or ("%s.family" % lane.lower())
    row["task_regime"] = regime
    if public:
        row["public"].update(public)
    from benchmark.eval_20k.tasks import validate_task_record
    return validate_task_record(row, line_no=1)


# -- contract hints ------------------------------------------------------------

def test_every_lane_has_a_contract_hint():
    for i in range(1, 17):
        lane = "L%02d" % i
        hint = contract_hint(_task(lane))
        assert isinstance(hint, str) and len(hint) > 40, lane


def test_exact_set_lanes_say_bare_array_not_answer_ids():
    for lane in ("L04", "L14", "L16"):
        hint = contract_hint(_task(lane))
        assert "bare JSON array of strings" in hint, lane
        assert '"answer_ids"' in hint  # named as the forbidden wrapper
        assert "NOT an object" in hint


def test_l01_hint_splits_decisions_vs_objects_by_family():
    decision = contract_hint(_task("L01", family="l01-f20-exact-decision-set"))
    assert '"decisions"' in decision and '"objects"' not in decision
    repair = contract_hint(_task("L01", family="l01-f01-registry-allowlist-repair",
                                 regime="repair"))
    assert '"objects"' in repair and '"decisions"' not in repair


def test_l09_hint_depends_on_artifact_format():
    cql = contract_hint(_task("L09", public={"artifact_format": "cql-text"}))
    assert '"format": "cql-text"' in cql and '"text"' in cql
    fhir = contract_hint(_task("L09", public={"artifact_format": "fhir-json"}))
    assert '"artifact"' in fhir and '"content"' in fhir
    assert "edits" in fhir  # the known wrong shape is called out


def test_l11_hint_is_primitive_plan_of_lists():
    hint = contract_hint(_task("L11"))
    assert '"primitive_plan"' in hint
    assert '[["GRASP"' in hint or '"GRASP"' in hint
    assert "Do NOT send a" in hint and "bare list of strings" in hint


def test_l13_hint_wrapper_key_by_family():
    body = contract_hint(_task("L13", family="l13-x02-additional-property-body-repair"))
    assert '"body"' in body
    param = contract_hint(_task("L13", family="l13-f01-required-parameter-repair"))
    assert '"parameters"' in param
    status = contract_hint(_task("L13", family="l13-x04-response-status-selection"))
    assert '"status_codes"' in status
    # version_migration resolves via the public coordinate mode
    mig_params = contract_hint(_task(
        "L13", family="l13-f05-api-version-migration",
        public={"coordinate": {"mode": "params"}}))
    assert '"parameters"' in mig_params
    mig_resp = contract_hint(_task(
        "L13", family="l13-x08-response-version-migration",
        public={"coordinate": {"mode": "response"}}))
    assert '"instance"' in mig_resp


def test_l02_hint_by_family():
    assert '"params"' in contract_hint(
        _task("L02", family="l02-f03-param-tuning-minimal-admission"))
    assert '"boundaries"' in contract_hint(
        _task("L02", family="l02-f04-boundary-evidence-probe"))
    assert '"decisions"' in contract_hint(
        _task("L02", family="l02-f05-exact-decision-set"))
    assert '"messages"' in contract_hint(
        _task("L02", family="l02-f06-message-expression-reconstruction"))
    assert '"match"' in contract_hint(
        _task("L02", family="l02-f07-match-condition-request-gate"))
    assert '"objects"' in contract_hint(
        _task("L02", family="l02-f11-replica-envelope-repair", regime="repair"))


def test_l03_l07_l10_family_shapes():
    assert '"table"' in contract_hint(_task("L03", family="l03-f11-output-entry-repair"))
    assert '"inconsistent_case_id"' in contract_hint(
        _task("L03", family="l03-f10-suite-conformance-audit"))
    assert '"necessarily_matching"' in contract_hint(
        _task("L03", family="l03-f14-partial-input-rule-audit"))
    assert '"ranking"' in contract_hint(_task("L07", family="l07x-f02-peak-time-ranking"))
    assert '"accepted_values"' in contract_hint(
        _task("L07", family="l07x-f04-init-threshold-search"))
    assert '"repaired_formula"' in contract_hint(
        _task("L07", family="l07-f07-kinetic-law-repair", regime="repair"))


def test_l05_l06_l08_l15_hints():
    assert '"repaired_smarts"' in contract_hint(
        _task("L05", family="l05.smarts_repair_syntax", regime="repair"))
    assert '"answer_ids"' in contract_hint(_task("L05", family="l05.fg_profile_exact"))
    assert '"answer_genes"' in contract_hint(
        _task("L06", family="l06.org_term_annotated_genes"))
    assert '"parent"' in contract_hint(
        _task("L06", family="l06.repair_isa_parent_swap", regime="repair"))
    assert '"alignment"' in contract_hint(
        _task("L06", family="l06.org_align_gene_aspect_annotations"))
    assert '"min_count"' in contract_hint(_task("L08", regime="repair"))
    assert '"answer_ids"' in contract_hint(_task("L08"))
    assert '"atoms"' in contract_hint(_task("L10", family="l10-f06-obligation-atom-query"))
    assert '"fragment"' in contract_hint(
        _task("L10", family="l10-f01-goal-predicate-repair", regime="repair"))
    assert '"true_conjuncts"' in contract_hint(
        _task("L10", family="l10-f07-conjunct-truth-audit"))
    assert '"answer_ids"' in contract_hint(_task("L15", family="l15-f20-pathway-membership"))


def test_l12_hint_uses_public_answer_contract():
    hint = contract_hint(_task(
        "L12", public={"answer_contract": {"kind": "trace",
                                           "submit": {"trace": ["<a>"]}}}))
    assert '"trace"' in hint


def test_build_messages_embeds_lane_contract():
    task = _task("L04")
    user = build_messages(task)[1]["content"]
    assert "Answer payload contract for lane L04" in user
    assert "bare JSON array of strings" in user
    assert '"answer"' in user  # outer submission envelope unchanged


def test_default_temperature_is_zero():
    args = model_runner.build_parser().parse_args(["--out", "x.jsonl"])
    assert args.temperature == 0.0


# -- alias normalisation --------------------------------------------------------

def _norm(lane, answer, family="", public=None, regime="application"):
    return normalize_answer(_task(lane, family=family, public=public,
                                  regime=regime), answer)


def test_l04_l16_answer_ids_alias_unwraps_to_bare_list():
    for lane in ("L04", "L16"):
        got, note = _norm(lane, {"answer_ids": ["A", "B"]})
        assert got == ["A", "B"] and note, lane
        got, note = _norm(lane, {"answer_ids": []})
        assert got == [] and note, lane
        # already-native shapes pass through untouched
        for native in (["A"], {"answer": ["A"]}, {"answer": []}):
            got, note = _norm(lane, native)
            assert got == native and note is None, (lane, native)


def test_l14_answer_ids_and_answer_aliases_unwrap():
    got, note = _norm("L14", {"answer_ids": ["CHEMBL25"]})
    assert got == ["CHEMBL25"] and note
    got, note = _norm("L14", {"answer": ["CHEMBL25"]})
    assert got == ["CHEMBL25"] and note


def test_exact_set_alias_needs_exactly_one_key():
    # a second key means it is not a pure alias: leave for the verifier
    got, note = _norm("L04", {"answer_ids": ["A"], "confidence": 0.9})
    assert got == {"answer_ids": ["A"], "confidence": 0.9} and note is None
    # non-string members are not a valid alias payload
    got, note = _norm("L16", {"answer_ids": [1, 2]})
    assert got == {"answer_ids": [1, 2]} and note is None


def test_l11_bare_plans_wrap_into_primitive_plan():
    got, note = _norm("L11", ["GRASP pop_1", "PLACE_INSIDE pop_1 bucket_1"])
    assert got == {"primitive_plan": [["GRASP", "pop_1"],
                                      ["PLACE_INSIDE", "pop_1", "bucket_1"]]}
    assert note
    got, note = _norm("L11", [["OPEN", "ashcan.n.01_1"]])
    assert got == {"primitive_plan": [["OPEN", "ashcan.n.01_1"]]} and note
    # unknown primitives / non-plan garbage stay untouched
    got, note = _norm("L11", ["FLY to the moon"])
    assert got == ["FLY to the moon"] and note is None
    got, note = _norm("L11", {"primitive_plan": [["GRASP", "x"]]})
    assert note is None


def test_l13_raw_payloads_wrap_under_family_key():
    got, note = _norm("L13", {"name": "x", "keep": 1},
                      family="l13-x02-additional-property-body-repair")
    assert got == {"body": {"name": "x", "keep": 1}} and note
    got, note = _norm("L13", {"body": {"name": "x"}},
                      family="l13-x02-additional-property-body-repair")
    assert note is None  # already wrapped
    got, note = _norm("L13", ["200", "404"],
                      family="l13-x04-response-status-selection")
    assert got == {"status_codes": ["200", "404"]} and note
    got, note = _norm("L13", "application/json",
                      family="l13-x06-response-header-construction")
    assert got == {"header_value": "application/json"} and note
    got, note = _norm("L13", {"limit": 5},
                      family="l13-f01-required-parameter-repair")
    assert got == {"parameters": {"limit": 5}} and note


def test_wrong_lane_schemas_are_never_rescued():
    # the v3 L01 failure: an L05-style answer must NOT be rewritten
    got, note = _norm("L01", {"repaired_smarts": "[#6]-[#7]"})
    assert got == {"repaired_smarts": "[#6]-[#7]"} and note is None
    # L09 edit lists cannot be turned into a full artifact
    got, note = _norm("L09", {"edits": [{"path": "a.b", "value": 1}]})
    assert got == {"edits": [{"path": "a.b", "value": 1}]} and note is None
    # L05/L08 etc. have no aliases at all
    got, note = _norm("L05", {"ids": ["CHEBI:1"]})
    assert got == {"ids": ["CHEBI:1"]} and note is None


def test_row_from_result_applies_normalisation():
    from autoontobench.models.dashscope_types import CallResult, Usage
    import json as _json
    task = _task("L04")
    res = CallResult(custom_id=task.task_id, model_exact_id="m",
                     prompt_sha="x", usage=Usage(input=1, output=1),
                     latency_s=0.01, attempts=1,
                     content=_json.dumps({"task_id": task.task_id,
                                          "answer": {"answer_ids": ["RHEA:1"]}}))
    row = model_runner.row_from_result(task, res, "m")
    assert row["answer"] == ["RHEA:1"]
    assert row["notes"].startswith("normalized:")


def test_row_from_result_keeps_parse_errors_untouched():
    from autoontobench.models.dashscope_types import CallResult
    task = _task("L04")
    res = CallResult(custom_id=task.task_id, model_exact_id="m",
                     prompt_sha="x", latency_s=0.01, attempts=1,
                     content="not json")
    row = model_runner.row_from_result(task, res, "m")
    assert row["answer"]["_error"] == "invalid_json"
    assert row["notes"] == "invalid_json"
