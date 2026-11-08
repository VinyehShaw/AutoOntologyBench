"""L16 task families: 16 substantive families over pinned UniProt 2026_03 Swiss-Prot.

Every family maps a ``params`` dict to an exact answer set computed over the
DAT-derived local store (fast path) and independently recomputed by the
verifier plugin from the XML-derived store (independent parser + SQL engine).
The two parsers were cross-checked entry-for-entry on a fixed sample
(``crosscheck_receipt.json``).

Family axes (EXPANSION_20K v2 L16: identity / reviewed-axis / function+evidence
/ GO / EC / Rhea / subcellular / feature boundaries / variants / isoforms /
cross-references / literature / taxonomy-proteome / construction):

  org_*   organization    — exact-set alignment over pinned record relations
  app_*   application     — stratified/confidence-aware exact-set queries
  rep_*   repair          — corrupted artifact repair verified by pinned truth
  con_*   construction    — knowledge construction under a budgeted probe interface

Instance state axes: ``evidence_bundle_variant`` (org/app/con: json_view vs
ttl_view serialization of the public slice, + seeded distractors from idx 2),
``corruption_seed`` (rep). Base goal identity = sha(family, canonical params);
instance identity = sha(base_goal, state descriptor). Paraphrase-only
multiplication is never used.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Iterator

from benchmark.plugins.source.l16.kb import L16KB
from benchmark.plugins.task_family.l16.pools import (
    EXPERIMENTAL_GO_CODES, FEATURE_BOUNDARY_TYPES, XREF_DBS, pools,
)

STABLE_SEED = 20261101  # 20K seed base (EXPANSION_20K §2)

INSTANCES_PER_GOAL = 5

ALLOWED_TOOLS = ["inspect_schema", "inspect_state", "read_public_doc",
                 "probe_native", "edit_artifact", "validate_artifact",
                 "check_action_with_ontology", "submit_final"]

PROFILE_LINKED = "linked-data/1"
PROFILE_SPARQL = "rdf-sparql/1"

GO_EVIDENCE_NOTE = (
    "GO evidence codes distinguish experimental (EXP/IDA/IPI/IMP/IGI/IEP/HTP/"
    "HDA/HMP/HGI/HEP) from computational (IEA/ISS/ISO/IBA/...) assertions; "
    "treat cross-references as annotation provenance, not causal proof.")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def canonical_params(params: dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def base_goal_id(family_id: str, params: dict[str, Any]) -> str:
    return sha_text(f"l16|{family_id}|{canonical_params(params)}")[:32]


@dataclass(frozen=True)
class Family:
    family_id: str
    regime: str
    profile: str
    complexity: str
    description: str
    state_axis: str
    instances_per_goal: int
    max_base_goals: int
    min_answer: int
    max_answer: int
    propose: Callable[[L16KB, random.Random], Iterator[dict[str, Any]]]
    answer: Callable[[L16KB, dict[str, Any]], tuple[str, ...]]
    goal_text: Callable[[dict[str, Any], L16KB], str]
    witness_kind: str


def state_descriptor(fam: Family, base_seed: int, inst_idx: int) -> dict[str, Any]:
    """Instance identity on the family's state axis."""
    if fam.state_axis == "evidence_bundle_variant":
        if inst_idx == 0:
            return {"axis": fam.state_axis, "variant": "json_view"}
        if inst_idx == 1:
            return {"axis": fam.state_axis, "variant": "ttl_view"}
        variant = "json_view" if inst_idx % 2 == 0 else "ttl_view"
        return {"axis": fam.state_axis, "variant": variant,
                "distractor_seed": (base_seed * 7919 + inst_idx * 104729)
                                   & 0xFFFFFFFF}
    return {"axis": fam.state_axis,
            "seed": (base_seed * 131_071 + inst_idx * 1_009) & 0xFFFFFFFF}


def source_unit_ids(params: dict[str, Any]) -> tuple[str, ...]:
    """Source-unit anchors named by the task params (never answer members)."""
    ids: list[str] = []
    if "accession" in params:
        ids.append(f"uniprot_entry/{params['accession']}")
    for key in ("gene", "keyword", "location", "disease", "rhea", "go",
                "pubmed", "proteome", "ec_prefix", "ftype", "db"):
        if key in params:
            ids.append(f"uniprot_facet/{key}:{params[key]}")
    if "taxon" in params:
        ids.append(f"uniprot_taxon/{params['taxon']}")
    return tuple(sorted(set(ids)))


TAXON_SCOPE_CAP = 3000  # bundle-size guard for taxon-scoped families


def _taxon_ok(kb: L16KB, taxon: int) -> bool:
    return len(kb.taxon_members.get(taxon, ())) <= TAXON_SCOPE_CAP


def _org(kb: L16KB, taxon: int) -> str:
    for acc in sorted(kb.taxon_members.get(taxon, ())):
        return kb.entries[acc]["organism_scientific"]
    return f"taxon {taxon}"


def _sorted_accs(accs) -> tuple[str, ...]:
    return tuple(sorted(accs))


# ---------------------------------------------------------------------------
# family definitions
# ---------------------------------------------------------------------------

def _mk_org_proteins_by_gene() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for gene, accs in sorted(kb.gene_members.items()):
            if 3 <= len(accs) <= 50:
                yield {"gene": gene}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(kb.gene_members.get(params["gene"], ()))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        n = len(kb.gene_members.get(params["gene"], ()))
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, organize "
                f"the protein identity mapping: list every reviewed entry whose "
                f"primary gene name is exactly '{params['gene']}' (gene "
                f"synonyms and ORF/locus names must not match). The release "
                f"contains a small number ({n}) of such entries; return the "
                f"sorted set of their primary accessions.")

    return Family("org_proteins_by_gene", "organization", PROFILE_LINKED, "low",
                  "protein accession identity by primary gene name",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 50,
                  propose, answer, goal, "remove_gene_assignment")


def _mk_org_entries_by_taxon_keyword() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for (taxon, kw), n in pools(kb)["taxon_keyword"]:
            if not _taxon_ok(kb, taxon):
                continue
            if 3 <= n <= 40:
                yield {"taxon": taxon, "keyword": kw}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        accs = kb.taxon_members.get(params["taxon"], frozenset()) \
            & kb.keyword_members.get(params["keyword"], frozenset())
        return tuple(sorted(kb.entries[a]["entry_name"] for a in accs))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"Within the pinned UniProtKB/Swiss-Prot 2026_03 entries for "
                f"{_org(kb, params['taxon'])} (NCBI taxon {params['taxon']}), "
                f"list the entry names of all reviewed entries carrying the "
                f"UniProt keyword '{params['keyword']}'. Return the sorted set "
                f"of entry names.")

    return Family("org_entries_by_taxon_keyword", "organization",
                  PROFILE_LINKED, "low",
                  "entry-name organization by taxonomy x keyword",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "remove_keyword_edge")


def _mk_app_protein_existence_stratum() -> Family:
    PE_LABEL = {1: "evidence at protein level", 2: "evidence at transcript "
                "level", 3: "inferred from homology", 4: "predicted",
                5: "uncertain"}

    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for (taxon, pe), n in pools(kb)["taxon_pe"]:
            if not _taxon_ok(kb, taxon):
                continue
            if pe in (2, 3, 4, 5) and 3 <= n <= 40:
                yield {"taxon": taxon, "pe": pe}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(a for a in kb.taxon_members.get(params["taxon"], ())
                            if kb.entries[a]["pe_level"] == params["pe"])

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"All entries in this lane's pinned UniProtKB/Swiss-Prot "
                f"2026_03 snapshot are reviewed (TrEMBL is excluded), so the "
                f"annotation-confidence axis inside the reviewed set is the "
                f"protein-existence (PE) level. For {_org(kb, params['taxon'])}"
                f" (taxon {params['taxon']}), list the reviewed entries at PE "
                f"level {params['pe']} ({PE_LABEL[params['pe']]}). Return the "
                f"sorted set of primary accessions.")

    return Family("app_protein_existence_stratum", "application",
                  PROFILE_SPARQL, "low",
                  "reviewed-set stratification by protein-existence level "
                  "(Swiss-Prot-only realization of the reviewed/unreviewed "
                  "axis; TrEMBL excluded by pin)",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "change_pe_level")


def _mk_app_function_evidence_eco() -> Family:
    ECO_LABEL = {
        "ECO:0000269": "experimental evidence",
        "ECO:0000250": "sequence-similarity inference",
        "ECO:0000255": "rule-based (HAMAP/RuleBase) inference",
        "ECO:0000305": "curator inference",
        "ECO:0000244": "combined computational + literature assertion",
        "ECO:0000312": "imported from an external database record",
    }

    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for (taxon, eco), n in pools(kb)["taxon_eco"]:
            if not _taxon_ok(kb, taxon):
                continue
            if 3 <= n <= 40:
                yield {"taxon": taxon, "eco": eco}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(
            a for a in kb.taxon_members.get(params["taxon"], ())
            if params["eco"] in kb.function_ecos.get(a, ()))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"For {_org(kb, params['taxon'])} (taxon {params['taxon']}) in "
                f"the pinned UniProtKB/Swiss-Prot 2026_03 release, list the "
                f"reviewed entries whose FUNCTION annotation carries evidence "
                f"code {params['eco']} ({ECO_LABEL[params['eco']]}). This is an "
                f"annotation-provenance query: do not relabel computational "
                f"inference as experimental proof. Return the sorted set of "
                f"primary accessions.")

    return Family("app_function_evidence_eco", "application", PROFILE_SPARQL,
                  "medium", "function annotation by ECO evidence code",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "remove_function_evidence")


def _mk_org_go_annotation_by_taxon() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for (taxon, go_id), n in pools(kb)["taxon_go"]:
            if not _taxon_ok(kb, taxon):
                continue
            if 3 <= n <= 40:
                yield {"taxon": taxon, "go": go_id}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        accs = kb.taxon_members.get(params["taxon"], frozenset()) \
            & kb.go_members.get(params["go"], frozenset())
        return _sorted_accs(accs)

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, list the "
                f"reviewed entries of {_org(kb, params['taxon'])} (taxon "
                f"{params['taxon']}) whose GO cross-references include "
                f"{params['go']}. {GO_EVIDENCE_NOTE} Return the sorted set of "
                f"primary accessions.")

    return Family("org_go_annotation_by_taxon", "organization", PROFILE_LINKED,
                  "low", "GO cross-reference membership by taxon",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "remove_go_annotation")


def _mk_app_ec_class_members() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for prefix, n in pools(kb)["ec_prefix"]:
            if 4 <= n <= 60:
                yield {"ec_prefix": prefix}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        p = params["ec_prefix"]
        out: set[str] = set()
        for acc, ecs in kb.ec_by_acc.items():
            if any(e == p or e.startswith(p + ".") for e in ecs):
                out.add(acc)
        return _sorted_accs(out)

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, organize "
                f"the enzyme classification: list every reviewed entry whose "
                f"protein-name EC numbers include an EC identifier in class "
                f"{params['ec_prefix']} (i.e. equal to {params['ec_prefix']} "
                f"itself or one of its descendants like "
                f"{params['ec_prefix']}.1; sibling classes do not count). "
                f"Return the sorted set of primary accessions.")

    return Family("app_ec_class_members", "application", PROFILE_SPARQL, "low",
                  "EC class membership over protein-name EC numbers",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 4, 60,
                  propose, answer, goal, "change_ec_number")


def _mk_org_rhea_catalysts() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for rhea_id, accs in sorted(kb.rhea_members.items()):
            if 2 <= len(accs) <= 40:
                yield {"rhea": rhea_id}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(kb.rhea_members.get(params["rhea"], ()))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, list the "
                f"reviewed enzyme entries whose CATALYTIC ACTIVITY annotation "
                f"cross-references the Rhea master reaction "
                f"{params['rhea']} (directional physiological-direction child "
                f"reactions do not count). This lane shares Rhea identifiers "
                f"with L04 but the truth here is UniProt-side annotation, "
                f"computed from the pinned UniProt files only. Return the "
                f"sorted set of primary accessions.")

    return Family("org_rhea_catalysts", "organization", PROFILE_LINKED, "low",
                  "Rhea master-reaction cross-reference membership",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 40,
                  propose, answer, goal, "remove_rhea_xref")


def _mk_org_subcellular_location() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for (taxon, term), n in pools(kb)["taxon_location"]:
            if not _taxon_ok(kb, taxon):
                continue
            if 3 <= n <= 40:
                yield {"taxon": taxon, "location": term}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        accs = kb.taxon_members.get(params["taxon"], frozenset()) \
            & kb.subcell_members.get(params["location"], frozenset())
        return _sorted_accs(accs)

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, list the "
                f"reviewed entries of {_org(kb, params['taxon'])} (taxon "
                f"{params['taxon']}) annotated with subcellular location "
                f"'{params['location']}'. Location terms are normalized to the "
                f"UniProt controlled vocabulary (topology/orientation "
                f"qualifiers are not part of the term). Return the sorted set "
                f"of primary accessions.")

    return Family("org_subcellular_location", "organization", PROFILE_LINKED,
                  "low", "subcellular-location membership by taxon",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 40,
                  propose, answer, goal, "remove_subcellular_location")


def _mk_rep_feature_boundary_repair() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        pool: list[tuple[str, str, int]] = []
        for acc, feats in kb.features_by_acc.items():
            by_type: dict[str, int] = {}
            for ft, b, e in feats:
                if ft in FEATURE_BOUNDARY_TYPES and "?" not in b + e \
                        and "<" not in b + e and ">" not in b + e:
                    by_type[ft] = by_type.get(ft, 0) + 1
            for ft, n in sorted(by_type.items()):
                if 2 <= n <= 10:
                    pool.append((acc, ft, n))
        for acc, ft, n in sorted(pool):
            yield {"accession": acc, "ftype": ft}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        ft = params["ftype"]
        rows = [(b, e) for t, b, e in kb.features_by_acc.get(
            params["accession"], ()) if t == ft
            and "?" not in b + e and "<" not in b + e and ">" not in b + e]
        return tuple(sorted(f"{ft}|{b}|{e}" for b, e in rows))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        e = kb.entries[params["accession"]]
        return (f"The feature-table artifact for reviewed entry "
                f"{params['accession']} ({e['entry_name']}, "
                f"{e['organism_scientific']}) contains exactly one corrupted "
                f"sequence boundary among its {params['ftype']} features with "
                f"exact boundaries (features with uncertain <?> boundaries are "
                f"out of scope). Identify the corrupted row against the pinned "
                f"UniProtKB/Swiss-Prot 2026_03 record (probe_native returns "
                f"the true boundary rows for this entry), repair the artifact, "
                f"and submit the complete corrected set of "
                f"'{params['ftype']}|begin|end' rows.")

    return Family("rep_feature_boundary_repair", "repair", PROFILE_LINKED,
                  "medium", "sequence-feature boundary repair on pinned truth",
                  "corruption_seed", INSTANCES_PER_GOAL, 60, 2, 10,
                  propose, answer, goal, "feature_boundary_mutation")


def _mk_org_disease_associated_proteins() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for name, accs in sorted(kb.disease_members.items()):
            if 2 <= len(accs) <= 30:
                yield {"disease": name}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(kb.disease_members.get(params["disease"], ()))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, list the "
                f"reviewed entries with a structured DISEASE annotation naming "
                f"'{params['disease']}'. This is an annotation-membership "
                f"query over curated disease labels, not a causal claim. "
                f"Return the sorted set of primary accessions.")

    return Family("org_disease_associated_proteins", "organization",
                  PROFILE_LINKED, "low", "structured disease-annotation "
                  "membership", "evidence_bundle_variant", INSTANCES_PER_GOAL,
                  60, 2, 30, propose, answer, goal, "remove_disease_link")


def _mk_org_variant_inventory() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for acc, variants in sorted(kb.variants_by_acc.items()):
            ids = [v[4] for v in variants if v[4]]
            if 3 <= len(ids) <= 30:
                yield {"accession": acc}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return tuple(sorted(v[4] for v in kb.variants_by_acc.get(
            params["accession"], ()) if v[4]))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        e = kb.entries[params["accession"]]
        return (f"Organize the curated sequence-variant inventory of reviewed "
                f"entry {params['accession']} ({e['entry_name']}, "
                f"{e['organism_scientific']}) in the pinned UniProtKB/"
                f"Swiss-Prot 2026_03 release: list every VARIANT feature "
                f"identifier (VAR_...) on the canonical sequence. Return the "
                f"sorted set of variant ids.")

    return Family("org_variant_inventory", "organization", PROFILE_LINKED,
                  "low", "curated variant-id inventory of one entry",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 3, 30,
                  propose, answer, goal, "remove_variant")


def _mk_app_isoform_inventory() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for acc, isos in sorted(kb.isoforms_by_acc.items()):
            if 2 <= len(isos) <= 15:
                yield {"accession": acc}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return tuple(sorted(i[0] for i in kb.isoforms_by_acc.get(
            params["accession"], ())))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        e = kb.entries[params["accession"]]
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, "
                f"reconstruct the isoform inventory of reviewed entry "
                f"{params['accession']} ({e['entry_name']}): list every "
                f"isoform identifier annotated under ALTERNATIVE PRODUCTS, "
                f"including the canonical isoform id when it is annotated "
                f"(isoform ids follow the '<accession>-<n>' convention). "
                f"Return the sorted set of isoform ids.")

    return Family("app_isoform_inventory", "application", PROFILE_SPARQL, "low",
                  "isoform / canonical-coordinate inventory of one entry",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 15,
                  propose, answer, goal, "remove_isoform")


def _mk_org_xref_provenance() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for (acc, db), n in pools(kb)["acc_db_xrefs"]:
            if 2 <= n <= 25:
                yield {"accession": acc, "db": db}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return tuple(sorted(kb.xrefs_by_acc_db.get(
            (params["accession"], params["db"]), ())))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        e = kb.entries[params["accession"]]
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, "
                f"reconstruct the cross-reference provenance of reviewed entry "
                f"{params['accession']} ({e['entry_name']}): list every "
                f"{params['db']} identifier cross-referenced by this entry. "
                f"Return the sorted set of {params['db']} ids.")

    return Family("org_xref_provenance", "organization", PROFILE_LINKED, "low",
                  "entry cross-reference id inventory by database",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 25,
                  propose, answer, goal, "remove_xref")


def _mk_org_literature_provenance() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for pmid, accs in sorted(kb.pubmed_members.items()):
            if 2 <= len(accs) <= 30:
                yield {"pubmed": pmid}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(kb.pubmed_members.get(params["pubmed"], ()))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, "
                f"reconstruct the literature provenance: list every reviewed "
                f"entry whose reference list cites PubMed {params['pubmed']}. "
                f"Return the sorted set of primary accessions.")

    return Family("org_literature_provenance", "organization", PROFILE_LINKED,
                  "low", "PubMed citation -> entry provenance",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 2, 30,
                  propose, answer, goal, "remove_pubmed_link")


def _mk_org_proteome_membership() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for upid, accs in sorted(kb.proteome_members.items()):
            if 10 <= len(accs) <= 300:
                yield {"proteome": upid}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        return _sorted_accs(kb.proteome_members.get(params["proteome"], ()))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        members = kb.proteome_members.get(params["proteome"], frozenset())
        taxa = sorted({kb.entries[a]["taxon_id"] for a in members})
        org = _org(kb, taxa[0]) if taxa else "unknown organism"
        return (f"In the pinned UniProtKB/Swiss-Prot 2026_03 release, "
                f"reconstruct the proteome membership: list every reviewed "
                f"entry cross-referenced to Proteomes id {params['proteome']} "
                f"({org}). The evidence slice mixes in same-organism entries "
                f"that are NOT in this proteome; only exact {params['proteome']} "
                f"membership counts. Return the sorted set of primary "
                f"accessions.")

    return Family("org_proteome_membership", "organization", PROFILE_LINKED,
                  "medium", "proteome membership (taxonomy/proteome axis)",
                  "evidence_bundle_variant", INSTANCES_PER_GOAL, 60, 10, 300,
                  propose, answer, goal, "remove_proteome_member")


def _mk_con_ec_rhea_catalyst_map() -> Family:
    def propose(kb: L16KB, rng: random.Random) -> Iterator[dict[str, Any]]:
        for taxon, pairs in pools(kb)["taxon_ec_rhea_pairs"]:
            if not _taxon_ok(kb, taxon):
                continue
            if 4 <= len(pairs) <= 30:
                yield {"taxon": taxon}

    def answer(kb: L16KB, params: dict[str, Any]) -> tuple[str, ...]:
        out: set[str] = set()
        for acc in kb.taxon_members.get(params["taxon"], ()):
            for ec, rhea in kb.cat_pairs_by_acc.get(acc, ()):
                out.add(f"{ec}|{rhea}")
        return tuple(sorted(out))

    def goal(params: dict[str, Any], kb: L16KB) -> str:
        return (f"Construct the enzyme-catalysis crosswalk for "
                f"{_org(kb, params['taxon'])} (taxon {params['taxon']}) in the "
                f"pinned UniProtKB/Swiss-Prot 2026_03 release: for every "
                f"reviewed entry that has BOTH a catalytic-activity EC number "
                f"and a master Rhea cross-reference, pair each catalytic EC "
                f"with each master Rhea id of the same reaction annotation. "
                f"The public slice lists the entries and their EC numbers but "
                f"redacts Rhea ids; probe_native reveals one entry's Rhea ids "
                f"per probe within budget. Return the sorted set of "
                f"'EC|RHEA' pairs.")

    return Family("con_ec_rhea_catalyst_map", "construction", PROFILE_LINKED,
                  "medium", "budgeted EC<->Rhea crosswalk construction",
                  "evidence_sample_seed", INSTANCES_PER_GOAL, 60, 4, 30,
                  propose, answer, goal, "remove_ec_rhea_pair")


FAMILIES: tuple[Family, ...] = (
    _mk_org_proteins_by_gene(),
    _mk_org_entries_by_taxon_keyword(),
    _mk_app_protein_existence_stratum(),
    _mk_app_function_evidence_eco(),
    _mk_org_go_annotation_by_taxon(),
    _mk_app_ec_class_members(),
    _mk_org_rhea_catalysts(),
    _mk_org_subcellular_location(),
    _mk_rep_feature_boundary_repair(),
    _mk_org_disease_associated_proteins(),
    _mk_org_variant_inventory(),
    _mk_app_isoform_inventory(),
    _mk_org_xref_provenance(),
    _mk_org_literature_provenance(),
    _mk_org_proteome_membership(),
    _mk_con_ec_rhea_catalyst_map(),
)

FAMILY_BY_ID = {f.family_id: f for f in FAMILIES}
