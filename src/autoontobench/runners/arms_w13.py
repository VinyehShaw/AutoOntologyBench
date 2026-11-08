"""The W13 G2 arm registry (owner: W13): all seven arms as data.

Arms are thin configs (``arm_config.py`` rule); this module is the single
registry the W13 driver and tests consume. W6 shipped ``direct`` /
``ontology`` / ``oracle_assisted`` / ``rag`` / ``code_dsl`` stub configs; W13
completes the set with the run-time wiring behind them:

- ``gold`` (GOLD_ONTOLOGY_UPPER, oracle-assisted): the task's gold-O composed
  from the W13 per-unit bundles of its governed units
  (``autoontobench.runners.gold_o``), PRE-INSTALLED and READ-ONLY. The gold-O
  is EXTRA information — NOT a fair autonomous baseline (marked in the parity
  note, which the harness records in every run header).
- ``corrupt`` / ``repaired``: mechanistic controls over the certified
  per-unit variants (``gold_o/validation.json``: corrupt differs from gold on
  >= 1 governed action per unit; repaired is decision-identical to gold);
  pre-installed, read-only. The model-facing note is intentionally the same
  neutral text as gold (the corruption is an instrument property, recorded in
  the arm id / parity note / run metadata, never announced to the model).
- ``code_dsl``: the agent authors ``program.json`` (gk-dsl/1 restricted
  constraint language; matched primitive operators eq/prefix/in/range/count
  over the SAME serialized facts), evaluated by the fixed interpreter under
  ``autoontobench.safety.sandbox`` via ``DualProfileConsumer``.
- ``rag``: DIRECT tool surface; the mounted root is the W13 retrieval corpus
  (``rag_corpus``); parity policy documented in ``agent_eval/REPORT.md``.
"""
from __future__ import annotations

from autoontobench.core.policy_hooks import FileEditPolicy
from autoontobench.runners import direct as direct_arm
from autoontobench.runners import ontology as ontology_arm
from autoontobench.runners import rag as rag_arm
from autoontobench.runners.arm_config import ArmConfig, make_arm
from autoontobench.runners.dsl_worker import DSL_PROFILE

GOLD = make_arm(
    "gold",
    representation="shacl-rule/1",
    initial_artifact="gold",
    artifact_editable=False,
    parity_note=(
        "GOLD oracle-assisted UPPER REFERENCE: pre-installed gold-O composed from the "
        "task's governed units; EXTRA information, NOT a fair autonomous baseline"
    ),
)
CORRUPT = make_arm(
    "corrupt",
    representation="shacl-rule/1",
    initial_artifact="corrupt",
    artifact_editable=False,
    parity_note=(
        "mechanistic control: pre-installed corrupt gold-O (one governed semantic "
        "mutation per unit, certified in gold_o/validation.json); read-only"
    ),
)
REPAIRED = make_arm(
    "repaired",
    representation="shacl-rule/1",
    initial_artifact="repaired",
    artifact_editable=False,
    parity_note=(
        "mechanistic control: pre-installed repaired gold-O (decision-identical to "
        "gold, certified in gold_o/validation.json); read-only"
    ),
)
CODE_DSL = make_arm(
    "code_dsl",
    representation=DSL_PROFILE,
    initial_artifact="dsl_stub",
    artifact_editable=True,
    parity_note=(
        "CODE_DSL_MATCHED: agent authors program.json (gk-dsl/1; matched primitive "
        "operators eq/prefix/in/range/count over the same serialized facts); fixed "
        "sandboxed interpreter (runners/dsl_worker.py)"
    ),
)

#: The full W13 arm registry.
ARMS_W13: dict[str, ArmConfig] = {
    "direct": direct_arm.ARM,
    "ontology": ontology_arm.ARM,
    "gold": GOLD,
    "corrupt": CORRUPT,
    "repaired": REPAIRED,
    "code_dsl": CODE_DSL,
    "rag": rag_arm.ARM,
}

#: The code_dsl arm's editable file set (the DSL program only).
DSL_FILE_POLICY = FileEditPolicy(allowed_names=frozenset({"program.json"}))

#: Model-facing construction notes (real runs only; mock runs are scripted).
#: gold/corrupt/repaired intentionally share one neutral text (see module docstring).
PREINSTALLED_ARTIFACT_NOTE = (
    "# Artifact arm (shacl-rule/1, PRE-INSTALLED read-only artifact)\n"
    "A policy artifact is PRE-INSTALLED as your working artifact and is READ-ONLY "
    "(edit_artifact is not available in this arm). Consult it via "
    "check_action_with_ontology (its OPINION on an action — never native truth), "
    "validate_artifact, and seal it (seal_artifact) before submit_final. It encodes "
    "admission clauses over the serialized vocabulary of TOOL_GUIDE.md §6 (base "
    "urn:aob:gk#; action fields gk:object / gk:set_<normalized path>; state records "
    "carry type/name/namespace/label_<k>/image_<i>/replicas/service_type)."
)
CODE_DSL_NOTE = (
    "# Artifact arm (code-dsl/1, SELF_BUILT constraint program)\n"
    "You own a working constraint program that starts EMPTY (every check returns "
    "UNKNOWN). Author it with edit_artifact as the single file program.json: "
    '{"schema": "gk-dsl/1", "rules": [{"name": str, "match": {"action_kind": str, '
    '"object_type": str}, "require": [CONDITION]}]}. CONDITION = {"op": "eq"|"prefix"|'
    '"in"|"range"|"count", "field": FIELD, ...} over the serialized fields of '
    "TOOL_GUIDE.md §6; range takes min/max, count takes min, in takes values, "
    "eq/prefix take value. A condition reads the action's set-override when present, "
    "else the object's current state field; an absent effective value FAILS the "
    "condition. A rule COVERS an action when action_kind fits and APPLIES when "
    "object_type also fits; any failed require in an applied rule means DENY; no "
    "covering rule means UNKNOWN. validate_artifact / check_action_with_ontology / "
    "seal_artifact work as in the ontology arm."
)
ARM_NOTES: dict[str, str] = {
    "gold": PREINSTALLED_ARTIFACT_NOTE,
    "corrupt": PREINSTALLED_ARTIFACT_NOTE,
    "repaired": PREINSTALLED_ARTIFACT_NOTE,
    "code_dsl": CODE_DSL_NOTE,
}
