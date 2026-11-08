"""Lane-specific answer contracts for the 20K model runner.

The 20K harness scores each ``answer`` payload with the lane's own native
verifier (``benchmark.plugins.verifier.lXX`` / ``.../task_family/lXX_exp``);
every lane accepts exactly one JSON shape per task variant, and a wrong
*wrapper* scores zero even when the content is right (the v3 zero-TSR cause:
models answered ``{"answer_ids": [...]}`` to lanes that read a bare list, or
``{"edits": [...]}`` to a lane that reads ``{"artifact": ...}``).

This module is derived from the native verifiers' ``verify_terminal`` entry
points (the exact ``submitted.get(<key>)`` reads) and provides:

* :func:`contract_hint` — the per-task answer contract injected into
  :func:`benchmark.eval_20k.model_runner.build_messages`, with a small valid
  JSON example per accepted variant;
* :func:`normalize_answer` — a post-parse repair of *harmless wrapper
  aliases* only (a dict whose single value is the payload under a different
  key, a bare list where the verifier wraps one).  It never invents content
  and never rescues a wrong-lane schema (e.g. an L05-style
  ``{"repaired_smarts": ...}`` sent to L01 stays a failure).

Only the public task record (lane/family/regime/public fields) is consulted —
the same information the model sees; no oracle material is read here.
"""
from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Per-family contract tables (family_id -> answer-shape key).
#
# Derived from the lane family registries
# (benchmark/plugins/task_family/lXX/families*.py) and the matching
# ``verify_terminal`` key reads.  Every table covers every family of that
# lane present in release_v3/tasks/qualified_tasks_test.jsonl.
# ---------------------------------------------------------------------------

#: L02 (CEL admission policies): goal_kind -> (key, note).  f13+ are all
#: repair_compliance (see task_family/l02/expansion_20k.py).
_L02_BY_FAMILY: dict[str, str] = {
    "l02-f03-param-tuning-minimal-admission": "params",
    "l02-f04-boundary-evidence-probe": "boundaries",
    "l02-f05-exact-decision-set": "decisions",
    "l02-f06-message-expression-reconstruction": "messages",
    "l02-f07-match-condition-request-gate": "match",
}

#: L03 (DMN): goal_kind -> submit keys (legacy plugin.py + plugin_20k.py).
_L03_BY_FAMILY: dict[str, str] = {
    "l03-f01-exact-decision-output": "output",
    "l03-f02-rule-match-audit": "matched_rule_ids",
    "l03-f03-rule-entry-repair": "table",
    "l03-f04-hit-policy-repair": "table",
    "l03-f05-cross-table-chain-eval": "output",
    "l03-f06-cross-table-binding-repair": "tables",
    "l03-f07-collect-aggregation-result": "output",
    "l03-f08-hidden-rule-input-synthesis": "input",
    "l03-f09-tck-scenario-output": "output",
    "l03-f10-suite-conformance-audit": "inconsistent_case_id",
    "l03-f11-output-entry-repair": "table",
    "l03-f12-aggregation-repair": "table",
    "l03-f13-rule-deletion-repair": "table",
    "l03-f14-partial-input-rule-audit": "partial_audit",
    "l03-f15-chain-root-synthesis": "input",
    "l03-f16-ordered-policy-output": "output",
}

#: L05 (RDKit): family prefix -> shape key.
_L05_ALIGN_FAMILIES = frozenset({"l05.org_align_group_molecule_sets"})

#: L06 (GO): repair families -> fix-payload shape key; the two gene-answer
#: families; the alignment family.  All other L06 families read answer_ids.
_L06_REPAIR_SHAPES: dict[str, str] = {
    "l06.repair_isa_parent_swap": "term_parent",
    "l06.repair_missing_isa_edge": "term_parent",
    "l06.repair_orphan_reconnect": "term_parents",
    "l06.repair_relation_type_confusion": "subject_object_predicate",
    "l06.repair_partof_edge_deletion": "subject_object",
    "l06.repair_spurious_isa_edge": "term_remove_parent",
    "l06.repair_regulation_polarity_swap": "subject_object_predicate",
    "l06.repair_spurious_partof_edge": "subject_remove_object",
}
_L06_GENE_FAMILIES = frozenset(
    {"l06.org_term_annotated_genes", "l06.org_exp_genes_for_term"})
_L06_ALIGN_FAMILIES = frozenset({"l06.org_align_gene_aspect_annotations"})

#: L07 (SBML/roadrunner): family -> shape key.
_L07_BY_FAMILY: dict[str, str] = {
    "l07-f01-threshold-set-query": "answer_ids",
    "l07-f02-final-ranking-query": "ranking",
    "l07-f03-trajectory-shape-organization": "assignment",
    "l07-f04-parameter-threshold-search": "accepted_values",
    "l07-f05-parameter-value-repair": "parameter_value",
    "l07-f06-initial-value-repair": "initial_value",
    "l07-f07-kinetic-law-repair": "repaired_formula",
    "l07-f08-network-role-query": "answer_ids",
    "l07-f09-species-role-organization": "assignment",
    "l07-f10-steady-ordering-construction": "accepted_values",
    "l07x-f01-retention-set": "answer_ids",
    "l07x-f02-peak-time-ranking": "ranking",
    "l07x-f03-pulse-set": "answer_ids",
    "l07x-f04-init-threshold-search": "accepted_values",
    "l07x-f05-init-window-search": "accepted_values",
    "l07x-f06-reaction-degree-query": "answer_ids",
}

#: L10 (BDDL): family -> shape key (legacy bddl_verifier + 20K expansion).
_L10_BY_FAMILY: dict[str, str] = {
    "l10-f01-goal-predicate-repair": "fragment",
    "l10-f02-goal-arity-repair": "fragment",
    "l10-f03-goal-binding-repair": "fragment",
    "l10-f04-init-relation-repair": "fragment",
    "l10-f05-quantifier-category-repair": "fragment",
    "l10-f06-obligation-atom-query": "atoms",
    "l10-f07-conjunct-truth-audit": "true_conjuncts",
    "l10-f08-init-relation-organization": "groups",
    "l10-f09-objects-declaration-audit": "mismatched",
    "l10-f10-goal-construction": "goal_text",
    "l10-f11-minimal-state-construction": "atoms",
    "l10-f12-cross-activity-category-join": "mapping",
    "l10-f13-multi-defect-structure-audit": "findings",
    "l10-f14-quantifier-inventory-audit": "quantifiers",
    "l10-f15-conjunct-shape-census": "shapes",
    "l10-f16-conjunct-reference-closure": "closure",
    "l10-f17-goal-polarity-audit": "negative_leaves",
    "l10-f18-discriminating-state-construction": "atoms",
    "l10-f19-weakening-goal-construction": "goal_text",
    "l10-f20-init-contradiction-audit": "contradicted",
    "l10-f21-init-duplicate-audit": "duplicated",
    "l10-f22-cross-activity-predicate-join": "mapping",
    "l10-f23-room-occupancy-organization": "rooms",
    "l10-f24-obligation-provenance-partition": "provenance",
    "l10-f25-predicate-usage-matrix": "usage",
}

#: L13 (OpenAPI): family -> submission-wrapper key the native verifier reads.
#: ``version_migration`` families resolve via the public coordinate mode.
_L13_BY_FAMILY: dict[str, str] = {
    "l13-f01-required-parameter-repair": "parameters",
    "l13-f02-parameter-type-repair": "parameters",
    "l13-f03-request-body-repair": "body",
    "l13-f04-response-instance-construction": "instance",
    "l13-f05-api-version-migration": "version_migration",
    "l13-f06-cross-api-schema-mapping": "instance",
    "l13-f07-enum-constraint-repair": "parameters",
    "l13-f08-operation-selection": "operation_keys",
    "l13-f09-security-requirement-construction": "alternatives",
    "l13-x01-pattern-parameter-repair": "parameters",
    "l13-x02-additional-property-body-repair": "body",
    "l13-x03-nested-required-body-repair": "body",
    "l13-x04-response-status-selection": "status_codes",
    "l13-x06-response-header-construction": "header_value",
    "l13-x07-union-response-construction": "instance",
    "l13-x08-response-version-migration": "version_migration",
    "l13-x09-array-parameter-repair": "parameters",
    "l13-x11-compound-parameter-repair": "parameters",
    "l13-x12-array-constrained-body-repair": "body",
}

#: L13 wrapper keys whose payload is a JSON object (raw dicts may be wrapped).
_L13_OBJECT_KEYS = frozenset({"parameters", "body", "instance"})

#: L11 symbolic primitives (lane contract; mirrors adapters/l11.py).
_L11_PRIMITIVES = (
    "GRASP", "PLACE_INSIDE", "PLACE_ON_TOP", "OPEN", "CLOSE",
    "TOGGLE_ON", "TOGGLE_OFF",
)

# ---------------------------------------------------------------------------
# Hint text fragments (shape key -> instruction with a small valid example).
# ---------------------------------------------------------------------------

_L03_SHAPES: dict[str, str] = {
    "output": '{"output": <exact decision output value>} — e.g. '
              '{"output": "Dish: Spareribs"} or {"output": 42.5}',
    "matched_rule_ids": '{"matched_rule_ids": ["<rule-id>", ...]} — the ids of '
                        'every rule that matches the given input case',
    "table": '{"table": {<the full repaired DMN decision table payload>}} — '
             'same structure as the table shown in the materials, with the '
             'broken cell(s) fixed',
    "tables": '{"tables": {"<decision-id>": {<full repaired table payload>}, '
              '...}} — every decision table in the chain, repaired',
    "input": '{"input": {"<variable>": <value>, ...}} — a typed input '
             'assignment',
    "inconsistent_case_id": '{"inconsistent_case_id": "<case-id>"}',
    "partial_audit": '{"necessarily_matching": ["<rule-id>", ...], '
                     '"possibly_matching": ["<rule-id>", ...]}',
}

_L10_SHAPES: dict[str, str] = {
    "fragment": '{"fragment": "<single BDDL token>"} — one token only: no '
                'whitespace, no parentheses (e.g. the corrected predicate, '
                'argument, or category name)',
    "atoms": '{"atoms": ["(pred obj ...)", ...]} — a sorted list of unique '
             'ground BDDL atom strings',
    "true_conjuncts": '{"true_conjuncts": [0, 3, ...]} — a sorted list of '
                      '0-based conjunct indices',
    "groups": '{"groups": {"<predicate>": ["(atom ...)", ...]}} — every init '
              'atom grouped under its predicate',
    "mismatched": '{"mismatched": ["<instance>", ...]} — a sorted list of '
                  'object instances',
    "goal_text": '{"goal_text": "(and ...)"} — one BDDL goal expression '
                 'string',
    "mapping": '{"mapping": {"<category-or-predicate>": ["<instance>", ...]}}',
    "findings": '{"findings": [["<section>", "<code>"], ...]} — a sorted list '
                'of [section, code] pairs',
    "quantifiers": '{"quantifiers": [["<head>", "<variable>", '
                   '"<iterable-category>"], ...]} — in textual order',
    "shapes": '{"shapes": {"0": "<shape-tag>", ...}} — conjunct index (as a '
              'string) -> shape tag',
    "closure": '{"closure": {"0": ["<instance>", ...], ...}} — conjunct index '
               '(as a string) -> sorted referenced instances',
    "negative_leaves": '{"negative_leaves": [1, 4, ...]} — a sorted list of '
                       'leaf indices',
    "contradicted": '{"contradicted": ["(atom ...)", ...]} — sorted atom '
                    'strings',
    "duplicated": '{"duplicated": ["(atom ...)", ...]} — sorted atom strings',
    "rooms": '{"rooms": {"<room>": ["<instance>", ...]}} — sorted instances '
             'per room',
    "provenance": '{"provenance": {"(atom ...)": [0, 2, ...]}} — atom -> '
                  'sorted generating-conjunct indices',
    "usage": '{"usage": {"<predicate>": ["init", "goal"]}} — sorted section '
             'list per predicate',
}

_L07_SHAPES: dict[str, str] = {
    "answer_ids": '{"answer_ids": ["<species-or-reaction-id>", ...]} — the '
                  'exact set, as strings',
    "ranking": '{"ranking": ["<id>", ...]} — an ORDERED list (rank 1 first), '
               'no ties',
    "assignment": '{"assignment": {"<species-id>": "<label>", ...}} — one '
                  'entry per universe member',
    "accepted_values": '{"accepted_values": [<float>, ...]} — the exact set '
                       'of candidate values that pass',
    "parameter_value": '{"parameter_value": <number>}',
    "initial_value": '{"initial_value": <number>}',
    "repaired_formula": '{"repaired_formula": "<kinetic-law formula string>"}',
}

_L06_REPAIR_HINTS: dict[str, str] = {
    "term_parent": '{"term": "GO:0000000", "parent": "GO:0000001"} — the '
                   'focus term and its correct is_a parent',
    "term_parents": '{"term": "GO:0000000", "parents": ["GO:0000001", ...]} — '
                    'the focus term and all its restored parents',
    "subject_object_predicate": '{"subject": "GO:xxx", "object": "GO:yyy", '
                                '"predicate": "<correct relation>"}',
    "subject_object": '{"subject": "GO:xxx", "object": "GO:yyy"} — identify '
                      'the edge endpoints to repair',
    "term_remove_parent": '{"term": "GO:xxx", "remove_parent": "GO:yyy"} — '
                          'the spurious is_a edge to remove',
    "subject_remove_object": '{"subject": "GO:xxx", "remove_object": '
                             '"GO:yyy"} — the spurious edge to remove',
}


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _hint_l01(family: str) -> str:
    if family == "l01-f20-exact-decision-set":
        return ('For this exact-decision-set task the answer payload is:\n'
                '{"decisions": {"<object-id>": {"<policy-unit-id>": '
                '"ALLOW"|"DENY", ...}, ...}}\n'
                'One entry per object in scope, covering every policy unit.\n'
                'Example: {"decisions": {"deployment-0": {"pol-a": "ALLOW", '
                '"pol-b": "DENY"}}}')
    return (
        'This is a manifest edit (repair/organization) task. The answer '
        'payload is the FULL set of corrected manifests:\n'
        '{"objects": {"<object-id>": {<the complete manifest object after '
        'your edits>}, ...}}\n'
        'One entry per object in scope; each value is the whole Kubernetes '
        'manifest (all fields, not just the changed ones). Guarded fields '
        'must stay identical to the input.\n'
        'Example: {"objects": {"ingress-0": {"apiVersion": '
        '"networking.k8s.io/v1", "kind": "Ingress", "metadata": {...}, '
        '"spec": {...}}}}')


def _hint_l02(family: str) -> str:
    shape = _L02_BY_FAMILY.get(family)
    if shape == "params":
        return ('The answer payload is:\n'
                '{"params": {"maxReplicas": <integer>}} — the minimally '
                'admitting parameter object.')
    if shape == "boundaries":
        return ('The answer payload is:\n'
                '{"boundaries": {"<policy-handle>": <integer>, ...}} — the '
                'maximum admissible replicas value per policy in scope.\n'
                'Example: {"boundaries": {"pol-a": 3}}')
    if shape == "decisions":
        return ('The answer payload is:\n'
                '{"decisions": {"<object-id>": {"<policy-handle>": '
                '"ALLOW"|"DENY", ...}, ...}} — the complete '
                'object -> policy -> decision mapping.\n'
                'Example: {"decisions": {"deployment-0": {"pol-a": "ALLOW", '
                '"pol-b": "DENY"}}}')
    if shape == "messages":
        return ('The answer payload is:\n'
                '{"messages": {"<object-id>": {"<policy-handle>": '
                '"<exact message string>", ...}, ...}}')
    if shape == "match":
        return ('The answer payload is:\n'
                '{"match": {"<request-id>": {"applies": true|false, '
                '"conditions": {"<condition-name>": true|false, ...}}, ...}}')
    return (
        'This is a manifest edit (repair) task. The answer payload is the '
        'FULL set of corrected manifests:\n'
        '{"objects": {"<object-id>": {<the complete manifest object after '
        'your edits>}, ...}}\n'
        'One entry per object in scope; each value is the whole object with '
        'every governing CEL validation passing.')


def _hint_l03(family: str) -> str:
    shape = _L03_BY_FAMILY.get(family, "output")
    return ('The answer payload for this DMN task is:\n'
            + _L03_SHAPES[shape])


def _hint_l05(family: str, regime: str) -> str:
    if family in _L05_ALIGN_FAMILIES:
        return ('The answer payload is:\n'
                '{"alignment": {"<group-label>": ["CHEBI:<id>", ...], ...}} — '
                'for every label, the exact sorted set of molecule ids.')
    if regime == "repair" or ".smarts_repair" in family:
        return ('The answer payload is:\n'
                '{"repaired_smarts": "<one corrected SMARTS pattern '
                'string>"}\n'
                'Example: {"repaired_smarts": "[#6]-[#7](-[#6])-[#8]"}')
    return ('The answer payload is:\n'
            '{"answer_ids": ["CHEBI:<id>", ...]} — the exact set of molecule '
            'ids satisfying the structural condition (no extras, none '
            'missing).\n'
            'Example: {"answer_ids": ["CHEBI:15377", "CHEBI:16134"]}')


def _hint_l06(family: str) -> str:
    if family in _L06_GENE_FAMILIES:
        return ('The answer payload is:\n'
                '{"answer_genes": ["<gene-id>", ...]} — the exact set of '
                'candidate genes annotated to the term(s).')
    if family in _L06_ALIGN_FAMILIES:
        return ('The answer payload is:\n'
                '{"alignment": {"<gene>": ["GO:<id>", ...], ...}} — for every '
                'gene, the exact sorted set of GO term ids.')
    shape = _L06_REPAIR_SHAPES.get(family)
    if shape is not None:
        return ('This is a GO graph repair task; the answer payload is the '
                'minimal fix descriptor:\n' + _L06_REPAIR_HINTS[shape])
    return ('The answer payload is:\n'
            '{"answer_ids": ["GO:<id>", ...]} — the exact set of GO term ids '
            '(or regulation gene ids) answering the closure query.\n'
            'Example: {"answer_ids": ["GO:0008150", "GO:0050896"]}')


def _hint_l07(family: str) -> str:
    shape = _L07_BY_FAMILY.get(family, "answer_ids")
    return ('The answer payload for this SBML simulation task is:\n'
            + _L07_SHAPES[shape])


def _hint_l08(regime: str) -> str:
    if regime == "repair":
        return ('This is a rule-parameter repair task. The answer payload '
                'carries exactly the four rule parameters:\n'
                '{"op": ">="|">"|"<="|"<", "threshold": <number>, '
                '"lookback_days": <integer>, "min_count": <integer>=1+}\n'
                'Example: {"op": ">=", "threshold": 2.0, "lookback_days": '
                '365, "min_count": 1}')
    return ('The answer payload is:\n'
            '{"answer_ids": ["<patient-id>", ...]} — the exact set of '
            'patients satisfying the clinical rule (no extras, none '
            'missing).\n'
            'Example: {"answer_ids": ["patient-001", "patient-042"]}')


def _hint_l09(public: dict[str, Any]) -> str:
    if public.get("artifact_format") == "cql-text":
        return (
            'The answer payload carries the FULL repaired CQL artifact text:\n'
            '{"artifact": {"format": "cql-text", "text": "<the complete '
            'repaired CQL source>"}}\n'
            'Send the entire artifact with the broken slots fixed — not a '
            'diff, not an edit list, not a single field.')
    return (
        'The answer payload carries the FULL repaired JSON artifact:\n'
        '{"artifact": {"content": {<the complete repaired artifact as JSON '
        'object>}}}\n'
        'Copy the artifact shown in the materials and fix the broken '
        'slot(s) in place — every other field must stay byte-identical. '
        'Do NOT send a diff/edit list ({"edits": [...]} is scored as an '
        'empty submission), and do not send a SMARTS pattern (wrong lane).')


def _hint_l10(family: str) -> str:
    shape = _L10_BY_FAMILY.get(family)
    if shape is None:
        return ('The answer payload is a single JSON object whose key is the '
                'one named by the user goal.')
    return ('The answer payload for this BDDL task is:\n' + _L10_SHAPES[shape])


def _hint_l11() -> str:
    return (
        'The answer payload is an action program object:\n'
        '{"primitive_plan": [["<PRIMITIVE>", "<object>", ...], ...]}\n'
        'Each step is a JSON LIST whose first element is one of '
        'GRASP / PLACE_INSIDE / PLACE_ON_TOP / OPEN / CLOSE / TOGGLE_ON / '
        'TOGGLE_OFF and whose remaining elements are BDDL object instance '
        'names from the scene (e.g. can__of__soda.n.01_1). Do NOT send a '
        'bare list of strings.\n'
        'Example: {"primitive_plan": [["GRASP", "can__of__soda.n.01_1"], '
        '["PLACE_INSIDE", "ashcan.n.01_1"]]}')


def _hint_l12(public: dict[str, Any]) -> str:
    contract = public.get("answer_contract")
    if isinstance(contract, dict) and isinstance(contract.get("submit"), dict):
        return ('The answer payload must follow the task\'s own '
                'answer_contract exactly:\n' + _json(contract["submit"]))
    return ('The answer payload follows the task\'s answer_contract: '
            '{"trace": ["<activity>", ...]} for a trace repair, '
            '{"mapping": {"<transition-id>": "<label>"}} for a model '
            'relabel, {"answer": <artifact>} for an exact-artifact query.')


def _l13_wrapper_key(family: str, public: dict[str, Any]) -> str | None:
    key = _L13_BY_FAMILY.get(family)
    if key == "version_migration":
        # Mirrors the native verifiers: keyed on the coordinate mode.
        # ``params`` -> parameters (base verifier), ``response`` -> instance
        # (ExpansionVerifier), everything else -> body (base verifier).
        mode = (public.get("coordinate") or {}).get("mode")
        if mode == "params":
            return "parameters"
        if mode == "response":
            return "instance"
        return "body"
    return key


def _hint_l13(family: str, public: dict[str, Any]) -> str:
    key = _l13_wrapper_key(family, public)
    examples = {
        "parameters": '{"parameters": {"<param-name>": <value>, ...}} — the '
                      'complete request parameters object satisfying the '
                      'parameter contract',
        "body": '{"body": {<the complete JSON request body>}} — the full '
                'body object valid against the requestBody schema',
        "instance": '{"instance": {<the complete JSON instance>}} — a JSON '
                    'value valid against the target schema',
        "operation_keys": '{"operation_keys": ["<METHOD path>", ...]} — e.g. '
                          '"GET /pets"',
        "alternatives": '{"alternatives": [{"<scheme-name>": [<scope>, ...]}, '
                        '...]} — the security requirement alternatives',
        "status_codes": '{"status_codes": ["<code>", ...]} — e.g. '
                        '["200", "404"]',
        "media_types": '{"media_types": ["<media-type>", ...]}',
        "header_value": '{"header_value": "<the exact header value string>"}',
    }
    if key is None:
        return ('The answer payload is a single JSON object whose wrapper '
                'key is the one named by the user goal (parameters / body / '
                'instance / ...).')
    return ('The answer payload is a JSON object with exactly this wrapper '
            'key:\n' + examples[key] + '\nDo NOT send the raw payload without '
            'the wrapper key.')


def _hint_l15(family: str) -> str:
    if family == "l15-f07-regulation":
        return ('The answer payload is:\n'
                '{"pairs": ["<pe>|<control_type>", ...]} — sorted '
                'controller-pe/control-type pairs.')
    return ('The answer payload is:\n'
            '{"answer_ids": ["<Reactome stable id or accession>", ...]} — '
            'the exact sorted set.\n'
            'Example: {"answer_ids": ["R-HSA-199420", "R-HSA-5653895"]}')


def contract_hint(task: Any) -> str | None:
    """Return the lane-specific answer-contract block for ``task``.

    ``task`` is a :class:`~benchmark.eval_20k.tasks.TaskRecord` (only its
    public fields are read).  Returns ``None`` only for an unknown lane.
    """
    lane = task.lane_id.upper()
    family = task.family_id
    regime = task.task_regime
    public = task.public or {}
    if lane == "L01":
        return _hint_l01(family)
    if lane == "L02":
        return _hint_l02(family)
    if lane == "L03":
        return _hint_l03(family)
    if lane == "L04":
        return ('The answer payload is a bare JSON array of strings — the '
                'exact, complete, sorted set of identifiers:\n'
                '["RHEA:10000", "RHEA:10001", ...]\n'
                'NOT an object: do not wrap it as {"answer_ids": [...]} or '
                '{"answer": [...]}. An empty set is [].')
    if lane == "L05":
        return _hint_l05(family, regime)
    if lane == "L06":
        return _hint_l06(family)
    if lane == "L07":
        return _hint_l07(family)
    if lane == "L08":
        return _hint_l08(regime)
    if lane == "L09":
        return _hint_l09(public)
    if lane == "L10":
        return _hint_l10(family)
    if lane == "L11":
        return _hint_l11()
    if lane == "L12":
        return _hint_l12(public)
    if lane == "L13":
        return _hint_l13(family, public)
    if lane == "L14":
        return ('The answer payload is a bare JSON array of strings — the '
                'exact, complete, sorted set of ChEMBL identifiers:\n'
                '["CHEMBL25", "CHEMBL941", ...]\n'
                'NOT an object: do not wrap it as {"answer_ids": [...]} or '
                '{"rows": ...}. An empty set is [].')
    if lane == "L15":
        return _hint_l15(family)
    if lane == "L16":
        return ('The answer payload is a bare JSON array of strings — the '
                'exact, complete, sorted set of UniProt accessions or ids:\n'
                '["P12345", "Q67890", ...]\n'
                'NOT an object: do not wrap it as {"answer_ids": [...]} or '
                '{"answer": [...]}. An empty set is [].')
    return None


# ---------------------------------------------------------------------------
# Post-parse alias normalisation.
# ---------------------------------------------------------------------------

def _as_str_list(value: Any) -> list[str] | None:
    if (isinstance(value, list)
            and all(isinstance(x, str) for x in value)):
        return list(value)
    return None


def _unwrap_single_key(answer: Any, keys: tuple[str, ...]
                       ) -> list[str] | None:
    """Return the list value when ``answer`` is exactly one alias key wrapping
    a list of strings; otherwise None."""
    if not isinstance(answer, dict) or len(answer) != 1:
        return None
    key = next(iter(answer))
    if key not in keys:
        return None
    return _as_str_list(answer[key])


def _normalize_l11(answer: Any) -> tuple[Any, str | None]:
    """Accept a bare plan (list) where the verifier wants {"primitive_plan": ...}.

    Unambiguous cases only: a list of step lists, or a list of
    "<PRIMITIVE> <arg> ..." strings whose head token is a known primitive.
    """
    if isinstance(answer, dict) and "primitive_plan" in answer:
        return answer, None
    steps: list[list[str]] = []
    if isinstance(answer, list) and answer:
        if all(isinstance(s, (list, tuple)) and s
               and all(isinstance(x, str) for x in s)
               and s[0] in _L11_PRIMITIVES for s in answer):
            steps = [list(s) for s in answer]
        elif all(isinstance(s, str) and s.split()
                 and s.split()[0] in _L11_PRIMITIVES for s in answer):
            steps = [s.split() for s in answer]
        else:
            return answer, None
        return {"primitive_plan": steps}, "l11_wrapped_bare_plan"
    return answer, None


def _is_l13_wrapper(answer: dict[str, Any], key: str) -> bool:
    return len(answer) == 1 and key in answer


def _normalize_l13(family: str, public: dict[str, Any], answer: Any
                   ) -> tuple[Any, str | None]:
    """Wrap a raw payload under the family-implied wrapper key.

    Mirrors the adapter-side l13_answer.normalize_answer but keyed off the
    public family/coordinate instead of the oracle payload.
    """
    key = _l13_wrapper_key(family, public)
    if key is None:
        return answer, None
    if isinstance(answer, dict):
        if _is_l13_wrapper(answer, key):
            return answer, None
        if key in _L13_OBJECT_KEYS:
            return {key: answer}, "l13_wrapped_raw_object"
        return answer, None
    if isinstance(answer, list):
        return {key: answer}, "l13_wrapped_bare_list"
    if key == "header_value" and isinstance(answer, str):
        return {key: answer}, "l13_wrapped_bare_string"
    return answer, None


def normalize_answer(task: Any, answer: Any) -> tuple[Any, str | None]:
    """Repair harmless wrapper aliases; return ``(answer, note)``.

    ``note`` is ``None`` when the answer is already in (or beyond repair
    into) the native shape, else a short provenance tag.  Only these
    lane-native aliases are normalised — anything else is returned verbatim
    so the lane verifier scores it as-is:

    * L04/L14/L16: ``{"answer_ids": [str, ...]}`` (or L14 ``{"answer":
      [...]}``) -> the bare list those verifiers iterate/compare;
    * L11: a bare plan (list of step lists, or of ``"PRIMITIVE arg"``
      strings) -> ``{"primitive_plan": [[...], ...]}``;
    * L13: a raw ``parameters``/``body``/``instance`` object, any bare list,
      or a bare header string -> wrapped under the family's key (an existing
      one-key wrapper is left untouched).
    """
    lane = task.lane_id.upper()
    if lane in ("L04", "L14", "L16"):
        keys = ("answer_ids", "answer") if lane == "L14" else ("answer_ids",)
        ids = _unwrap_single_key(answer, keys)
        if ids is not None:
            return ids, "%s_unwrapped_%s" % (lane.lower(), "alias_list")
        return answer, None
    if lane == "L11":
        return _normalize_l11(answer)
    if lane == "L13":
        return _normalize_l13(task.family_id, task.public or {}, answer)
    return answer, None


__all__ = ["contract_hint", "normalize_answer"]
