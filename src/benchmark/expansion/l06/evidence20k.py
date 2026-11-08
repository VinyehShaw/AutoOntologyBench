"""L06 20K-expansion GAF evidence index.

Streams the pinned GOA annotation rows
(``external_knowledge/indexes/gene-ontology/<release>/annotations.jsonl.gz``,
itself derived from the sha256-pinned ``goa_human.gaf.gz``) exactly once and
builds the evidence structures the new annotation-evidence families need:

  gene -> {"exp": set, "other": set, "all": set}   direct P-aspect terms
      split by evidence class (experimental vs not)
  term -> propagated gene sets (all / experimental), where propagation is
      the declared-profile closure (is_a + part_of ancestors) of each
      direct annotation term

The index is rebuilt from the pinned source by BOTH the proposer (universe
construction) and the verifier (answer recomputation); stored ASTs never
carry evidence data, so no stored evidence is ever trusted.
"""
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

from benchmark.expansion.l06.families20k import EXPERIMENTAL_EVIDENCE_CODES

EXP_CODES = frozenset(EXPERIMENTAL_EVIDENCE_CODES)


def load_evidence(index_dir: Path,
                  closures: dict[str, Any]) -> dict[str, Any]:
    """Build the evidence index. ``closures`` is the legacy closure index
    (isa_anc / partof_anc) from the same pinned release."""
    isa_anc = closures["isa_anc"]
    partof_anc = closures["partof_anc"]
    gene_ev: dict[str, dict[str, set[str]]] = {}
    term_genes: dict[str, set[str]] = {}
    term_genes_exp: dict[str, set[str]] = {}
    # propagated-only reverse sets: genes whose direct annotation lies
    # STRICTLY below the term (isa or part_of closure) — drawing candidate
    # panels from these guarantees the answer depends on the fragment's
    # propagation edges (semantic non-degeneracy), never on direct lookup.
    term_genes_prop: dict[str, set[str]] = {}
    term_genes_exp_prop: dict[str, set[str]] = {}
    ann_path = Path(index_dir) / "annotations.jsonl.gz"
    with gzip.open(ann_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            row = __import__("json").loads(line)
            if row["aspect"] != "P":
                continue
            g = row["symbol"]
            t = row["go_id"]
            is_exp = row["evidence"] in EXP_CODES
            by = gene_ev.setdefault(g, {"exp": set(), "other": set(),
                                        "all": set()})
            by["exp" if is_exp else "other"].add(t)
            by["all"].add(t)
            for a in (t, *isa_anc.get(t, ()), *partof_anc.get(t, ())):
                term_genes.setdefault(a, set()).add(g)
                if is_exp:
                    term_genes_exp.setdefault(a, set()).add(g)
            for a in (*isa_anc.get(t, ()), *partof_anc.get(t, ())):
                term_genes_prop.setdefault(a, set()).add(g)
                if is_exp:
                    term_genes_exp_prop.setdefault(a, set()).add(g)
    return {"gene_ev": gene_ev, "term_genes": term_genes,
            "term_genes_exp": term_genes_exp,
            "term_genes_prop": term_genes_prop,
            "term_genes_exp_prop": term_genes_exp_prop}
