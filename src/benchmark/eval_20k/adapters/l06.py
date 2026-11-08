"""20K harness adapter for lane L06 (Gene Ontology closure / repair).

The lane ships two native verifiers over the *same* pinned GO index:

* ``benchmark.plugins.verifier.l06.verifier.L06NativeVerifier`` -- the legacy
  8K families (``isa_ancestors`` ... ``repair_relation_type_confusion``);
* ``benchmark.expansion.l06.verify20k.L06ExpansionVerifier`` -- the 20K
  expansion families (``posregulators`` ... ``repair_spurious_partof_edge``)
  plus the GAF-evidence kinds (``term_annotated_genes`` / ``exp_term_genes``).

Both verifiers consume the same oracle payload AST (``payload["ast"]``) and
recompute every verdict from the pinned index; this adapter only resolves the
index directory and dispatches ``verify_terminal`` to the verifier generation
that owns the task's ``ast.kind``.  No lane semantics are re-implemented here.

The index is large (~34 MB compressed: annotations/closures/graph/manifest
plus terms), so support probing only checks that every file the verifiers
load is present; the index is loaded lazily on the first evaluation and is
shared by both verifier generations.

Resolution order for the index directory:

1. ``--lane-kw L06.index_dir=<dir>``
2. ``$AOB_L06_INDEX``
3. the newest ``external_knowledge/indexes/gene-ontology/<release>`` in-repo.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported

REPO_ROOT = Path(__file__).resolve().parents[3]
INDEX_ROOT = REPO_ROOT / "external_knowledge" / "indexes" / "gene-ontology"

#: Files ``L06SourcePlugin.load_index`` and the 20K evidence index read.
REQUIRED_INDEX_FILES: tuple[str, ...] = (
    "manifest.json",
    "graph.json.gz",
    "closures.json.gz",
    "gene_terms.json.gz",
    "depth.json",
    "terms.jsonl",
    "annotations.jsonl.gz",
)

#: ``ast.kind`` values owned by the 20K-expansion verifier.  The legacy kinds
#: are disjoint (the legacy verifier's ``answer_query``/repair tables define
#: them), so membership is an unambiguous dispatch key.
EXPANSION_KINDS: frozenset[str] = frozenset({
    "posregulators",
    "negregulators",
    "generic_only_regulators",
    "dual_regulators",
    "partof_isa_overlap",
    "reg_subject_descendants",
    "partof_depth_band",
    "exp_annotation_closure",
    "evidence_discriminated",
    "term_annotated_genes",
    "exp_term_genes",
    "repair_partof_edge_deletion",
    "repair_spurious_isa_edge",
    "repair_regulation_polarity_swap",
    "repair_spurious_partof_edge",
})


def _default_index_dir() -> Path | None:
    if not INDEX_ROOT.is_dir():
        return None
    candidates = sorted(p for p in INDEX_ROOT.iterdir() if p.is_dir())
    return candidates[-1] if candidates else None


def _resolve_index_dir(lane_kw: dict[str, str]) -> Path | None:
    value = (lane_kw.get("index_dir")
             or os.environ.get("AOB_L06_INDEX"))
    if value:
        return Path(value)
    return _default_index_dir()


class _L06Dispatch:
    """Lazily build and dispatch to the legacy / 20K native verifiers.

    Construction is deliberately cheap: the pinned index is loaded on the
    first ``verify_terminal`` and shared by both verifiers.  Each verifier
    generation is built only when a task of its kind is actually scored.
    """

    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self._index: dict[str, Any] | None = None
        self._legacy: Any = None
        self._expansion: Any = None

    # -- lazy native verifiers ------------------------------------------------
    def _load_index(self) -> dict[str, Any]:
        if self._index is None:
            from benchmark.plugins.source.l06.plugin import L06SourcePlugin

            self._index = L06SourcePlugin.load_index(self.index_dir)
        return self._index

    def _legacy_verifier(self) -> Any:
        if self._legacy is None:
            from benchmark.plugins.verifier.l06.verifier import L06NativeVerifier

            self._legacy = L06NativeVerifier(self.index_dir,
                                            index=self._load_index())
        return self._legacy

    def _expansion_verifier(self) -> Any:
        if self._expansion is None:
            from benchmark.expansion.l06.verify20k import L06ExpansionVerifier

            self._expansion = L06ExpansionVerifier(self.index_dir,
                                                   index=self._load_index())
        return self._expansion

    def verifier_for(self, ast: dict[str, Any]) -> Any:
        kind = ast.get("kind")
        return (self._expansion_verifier() if kind in EXPANSION_KINDS
                else self._legacy_verifier())

    def verify_terminal(self, task: Any, submitted: Any) -> Any:
        return self.verifier_for(task.ast.ast).verify_terminal(task, submitted)


class Adapter(LaneAdapter):
    lane_id = "L06"

    # -- construction ---------------------------------------------------------

    def build_verifier(self) -> Any:
        try:
            # Import-only probes keep support honest without loading 34 MB.
            from benchmark.plugins.verifier.l06.verifier import (  # noqa: F401
                L06NativeVerifier,
            )
            from benchmark.expansion.l06.verify20k import (  # noqa: F401
                L06ExpansionVerifier,
            )
        except ImportError as exc:
            raise LaneNotSupported(
                "L06 native verifier not importable: %s" % exc) from exc

        index = _resolve_index_dir(self.lane_kw)
        if index is None:
            raise LaneNotSupported(
                "lane L06 requires index_dir (--lane-kw L06.index_dir=... "
                "or AOB_L06_INDEX); no in-repo gene-ontology index found "
                "under %s" % INDEX_ROOT)
        if not index.is_dir():
            raise LaneNotSupported(
                "lane L06 index dir does not exist: %s" % index)
        missing = [name for name in REQUIRED_INDEX_FILES
                   if not (index / name).is_file()]
        if missing:
            raise LaneNotSupported(
                "lane L06 index dir %s is incomplete; missing: %s"
                % (index, ", ".join(missing)))
        return _L06Dispatch(index)

    def verifier_id(self, verifier: Any) -> str:
        if isinstance(verifier, _L06Dispatch):
            return ("benchmark.plugins.verifier.l06.verifier.L06NativeVerifier"
                    "|benchmark.expansion.l06.verify20k.L06ExpansionVerifier")
        return super().verifier_id(verifier)

    # -- reference / negative helpers (self-checks) ---------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ref = oracle_payload.get("reference_submission")
        if ref is None:
            raise LaneNotSupported(
                "L06 oracle payload exposes no reference_submission")
        return ref

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """Drop one member from the oracle's own reference (negative witness).

        Works for both the answer-set submissions (``answer_ids`` /
        ``answer_genes``) and the single-edge repair submissions (removing a
        required identity field makes the verifier reject the fix).
        """
        ref = self.reference_answer(oracle_payload)
        if not isinstance(ref, dict):
            raise LaneNotSupported("L06 reference submission is not an object")
        wrong = dict(ref)
        for key in ("answer_ids", "answer_genes", "parents"):
            values = wrong.get(key)
            if isinstance(values, list) and values:
                wrong[key] = values[:-1]
                return wrong
        if "alignment" in wrong and isinstance(wrong["alignment"], dict):
            alignment = {g: list(v) for g, v in wrong["alignment"].items()}
            gene = sorted(alignment)[0]
            if alignment[gene]:
                alignment[gene] = alignment[gene][:-1]
            else:
                alignment[gene] = ["GO:0000001"]
            wrong["alignment"] = alignment
            return wrong
        for key in ("parent", "predicate", "remove_parent", "remove_object",
                    "object"):
            if key in wrong:
                del wrong[key]
                return wrong
        raise LaneNotSupported(
            "L06 reference submission cannot be mutated into a negative: %r"
            % sorted(ref))


__all__ = ["Adapter", "EXPANSION_KINDS"]