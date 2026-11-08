# ATTRIBUTION — AutoOntologyBench 20K public test release

Build: `aob-20000-candidate-dev1`. Decisions follow the source redistribution matrix
(live matrix + the 20K addendum for ChEMBL / Reactome / UniProt / dmn-tck),
with PI confirmation of academic permission for the bundled raw databases.
Per-source details: see the `SUMMARY.md` in each subdirectory here.

| Source | Lane(s) | Tasks | License (pinned evidence) | Decision | Obligation |
|---|---|---:|---|---|---|
| gatekeeper-library | L01 | 1200 | Apache-2.0 (repo-wide, incl. policy template YAML) | REDISTRIBUTE_OK | Include LICENSE + NOTICE (copied verbatim from the pinned commit). |
| kubernetes-vap-cel | L02 | 700 | CC-BY-4.0 (kubernetes/website docs shard) | REDISTRIBUTE_OK | Credit Kubernetes authors, link the license, indicate changes. |
| camunda-dmn-examples | L03 | 700 | Apache-2.0 (both pinned commits; educational-example profile) | REDISTRIBUTE_OK | Include LICENSE + NOTICE per pinned commit. |
| dmn-tck | L03 | 300 | Apache-2.0 (dmn-tck conformance-test fixtures) | REDISTRIBUTE_OK | Retain LICENSE (Apache-2.0). Fixtures are conformance tests, not enterprise production content. |
| rhea-chebi | L04 | 3500 | Rhea CC-BY-4.0 + ChEBI CC-BY-4.0 (both pinned) | REDISTRIBUTE_OK | Reproduce the Rhea copyright statement; ChEBI attribution. Licenses ship with the data. |
| rdkit-molecules | L05 | 1300 | BSD-3-Clause (RDKit repo+Data) + CC-BY-4.0 (ChEBI rel255 SDF) | REDISTRIBUTE_OK | BSD notice for RDKit; ChEBI CC-BY-4.0 attribution for the SDF. |
| gene-ontology | L06 | 2300 | CC-BY-4.0 (embedded in go-basic.obo; GAF license asserted CC-BY-4.0 per EBI GOA terms, not separately pinned) | REDISTRIBUTE_OK | CC-BY-4.0 attribution to the Gene Ontology Consortium. GAF license confirmation is an open item flagged to the PI. |
| biomodels | L07 | 950 | Per-model CC0/CC-BY (not machine-readable); EBI terms of use | FETCH_RECIPE_ONLY | Fetch recipe + sha256 pins + per-model attribution; model XML content is not redistributed in this tree (raw databases are bundled separately under release_20k/source_data per PI academic-permission confirmation). |
| cql-clinical | L08 | 500 | Apache-2.0 (repo + vendored npm engines cql-execution 3.3.2 / cql-exec-fhir 2.2.0) | REDISTRIBUTE_OK | Include LICENSE. Tasks use repo grammar/schemas + synthetic patients only (no real patient data). |
| smart-anc | L09 | 800 | CC0-1.0 (pinned LICENSE.md; registry expected CC-BY — corrected to CC0) | REDISTRIBUTE_OK | Attribution recommended (credit WHO), not legally required under CC0. |
| behavior-1k-bddl | L10 | 900 | MIT (Stanford Vision and Learning Group) | REDISTRIBUTE_OK | Include MIT notice. OmniGibson scene/object assets are separate and NOT bundled (symbolic-only lane). |
| behavior-1k-omnigibson | L11 | 250 | Asset licenses require acceptance (NVIDIA Omniverse EULA + BEHAVIOR-1K data license; PI has accepted both) | METADATA_ONLY | No asset content in this tree. Re-verification requires the caller to accept the EULA (OMNI_KIT_ACCEPT_EULA=YES); the tool refuses to run otherwise. |
| pm4py-process | L12 | 1150 | AGPL-3.0 (pinned full text; sources.yaml said GPL-3.0-verify — corrected by pinned text) | FETCH_RECIPE_ONLY | Fetch recipe + sha256 + attribution; XES/PNML event-log content is copyleft and is not redistributed in this tree (raw databases are bundled separately under release_20k/source_data per PI academic-permission confirmation). |
| apis-guru | L13 | 1150 | Repo-level CC0 1.0 + fair-use README; individual OpenAPI specs carry per-spec upstream licenses | METADATA_ONLY | Index + hashes only; per-spec license obligations pass to the user. |
| chembl | L14 | 1500 | CC BY-SA 3.0 (ChEMBL 37 pinned LICENSE + REQUIRED.ATTRIBUTION) | FETCH_RECIPE_ONLY (matrix) — raw DB bundled separately under release_20k/source_data per PI academic-permission confirmation | Attribution: Mendez et al. 2019, DOI 10.1093/nar/gky1075; preserve ChEMBL IDs and the release number; ShareAlike terms apply to derivative adaptations of the database. |
| reactome | L15 | 1500 | CC0 (database data); images/icons CC BY 4.0 excluded | REDISTRIBUTE_OK (20K addendum; data files only) | Attribution retained in receipts; images/icons are excluded from redistribution. |
| uniprot | L16 | 1300 | CC BY 4.0 (UniProt README LICENSE section; Swiss-Prot only — TrEMBL not used) | REDISTRIBUTE_OK (20K addendum) | Retain UniProt Consortium attribution + release identifier. |

## Key citations

- **ChEMBL**: Mendez D, Gaulton A, Bento AP, et al. *ChEMBL: towards direct
  deposition of bioassay data.* Nucleic Acids Res. 2019;47(D1):D930-D940.
  DOI: 10.1093/nar/gky1075 (CC BY-SA 3.0; preserve ChEMBL IDs + release number).
- **UniProt**: UniProt Consortium. *UniProt: the Universal Protein
  Knowledgebase.* (CC BY 4.0; release 2026_03, Swiss-Prot section).
- **Reactome**: reactome/release-98 database data (CC0; images/icons excluded).
- **Gene Ontology**: Gene Ontology Consortium (CC-BY-4.0).
- **Rhea / ChEBI**: Rhea CC-BY-4.0; ChEBI CC-BY-4.0.
- **RDKit**: BSD-3-Clause.
- **BEHAVIOR-1K BDDL**: MIT, Stanford Vision and Learning Group.

## Open items (flagged to the PI, recorded honestly)

- L06 GAF license is asserted CC-BY-4.0 per EBI GOA terms but has no pinned
  standalone license file.
- L09 smart-anc: registry expected CC-BY-4.0/WHO terms; the pinned LICENSE.md is
  CC0-1.0. WHO is credited regardless.
- L11 OmniGibson assets: redistribution requires EULA/data-license acceptance
  (PI has accepted); this tree ships task metadata only, no assets.
