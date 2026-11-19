# noqa: SIZE_OK — declarative per-source/per-world license registry (a pure
# data table; splitting it would scatter one auditable legal mapping).
"""License/attribution registry for the Core-track benchmark release.

Static, hand-audited data keyed by ``world_id``: which upstream sources a world
derives from, which license/attribution files ship verbatim in
``LICENSES/<source_id>/`` (each with its repo-relative origin), and how every
license reference in the world's ``metadata.json::license`` string resolves
(shipped into the release, or explicitly marked not shipped with a reason).

Nothing here is fetched or copied at runtime beyond reading the declared
origin files; the registry is the single place that maps a world to its
upstream license evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from .errors import ReleaseError


@dataclass(frozen=True, slots=True)
class BundleFile:
    """One file shipped under ``LICENSES/<source_id>/`` in the release."""

    dest: str  # path relative to LICENSES/<source_id>/
    origin: str  # repo-relative path read verbatim at assembly time
    note: str  # one-line provenance note for the per-source SUMMARY.md


@dataclass(frozen=True, slots=True)
class SourceLicense:
    """One upstream source's license posture and shipped evidence files."""

    source_id: str
    license_label: str
    attribution: str
    obligation: str
    files: tuple[BundleFile, ...]
    notes: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Reference:
    """Resolution of one license pointer from a world's metadata."""

    pointer: str  # the reference as it appears in metadata/provenance
    status: Literal["shipped", "not_shipped"]
    paths: tuple[str, ...] = ()  # release-relative paths (status == "shipped")
    detail: str = ""  # note (shipped) or reason (not_shipped)


@dataclass(frozen=True, slots=True)
class WorldLicense:
    """Per-world license wiring: sources used + declared reference resolutions."""

    world_id: str
    sources: tuple[str, ...]
    references: tuple[Reference, ...] = ()
    snapshot_pointer: str = "source_snapshot/"
    snapshot_note: str = (
        "License/attribution texts copied verbatim from the pinned source "
        "snapshot; snapshot hash manifests (SNAPSHOT.json) and fetch receipts "
        "recording build-machine absolute paths remain in the repository."
    )


_V3: Final = "release_v3/LICENSES"
_REGISTRY: Final = "sources/registry.d"


def _snap(world: str) -> str:
    return f"pilot_worlds/{world}_scaleup/source_snapshot"


def _receipt(name: str) -> BundleFile:
    return BundleFile(
        dest=f"registry_receipts/{name}.yaml",
        origin=f"{_REGISTRY}/{name}.yaml",
        note=f"verbatim copy of the source-registry receipt {_REGISTRY}/{name}.yaml",
    )


SOURCES: Final[dict[str, SourceLicense]] = {
    "apis-guru": SourceLicense(
        source_id="apis-guru",
        license_label="repo-level CC0-1.0 + per-spec upstream licenses",
        attribution="APIs.guru (APIs-guru/openapi-directory on GitHub) and the per-spec publishers as indexed.",
        obligation=(
            "Metadata-only posture: the release ships the pinned unit index hashes, "
            "no OpenAPI spec content; per-spec upstream license obligations pass to "
            "the user."
        ),
        files=(
            BundleFile("LICENSE", f"{_V3}/apis-guru/LICENSE", "CC0 1.0 legal code of the pinned openapi-directory repo"),
            BundleFile("APIS_GURU_LICENSE.txt", f"{_snap('world_L13')}/APIS_GURU_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
        ),
        notes=(
            "No sources/registry.d/apis-guru.yaml exists in this repository state "
            "(recorded in the world license field and the snapshot declaration).",
        ),
        citations=("apis-guru openapi-directory, pinned commit f04b8d0bcd39.",),
    ),
    "behavior-1k-bddl": SourceLicense(
        source_id="behavior-1k-bddl",
        license_label="MIT",
        attribution="Stanford Vision and Learning Lab, 2021.",
        obligation="Retain the MIT notice.",
        files=(
            BundleFile("LICENSE", f"{_V3}/behavior-1k-bddl/LICENSE", "MIT license text of the pinned BEHAVIOR-1K bddl repository"),
            BundleFile("BEHAVIOR_LICENSE.txt", f"{_snap('world_L10')}/BEHAVIOR_LICENSE.txt", "license statement copied verbatim from the world_L10 pinned source snapshot"),
            BundleFile("BDDL_LICENSE.txt", f"{_snap('world_L11')}/BDDL_LICENSE.txt", "license statement copied verbatim from the world_L11 pinned source snapshot"),
            _receipt("behavior"),
        ),
        citations=(
            "BEHAVIOR-1K bddl3 activity definitions and knowledge tables, tag "
            "v3.9.3-post2 (commit a8247a8cc1633fe1ca0cc66aa07243d46c64f155).",
        ),
    ),
    "behavior-1k-omnigibson": SourceLicense(
        source_id="behavior-1k-omnigibson",
        license_label="assets non-redistributable (NVIDIA Omniverse EULA + BEHAVIOR-1K data license)",
        attribution="Stanford Vision and Learning Lab (OmniGibson code, MIT); BEHAVIOR-1K scene/object assets are license-gated.",
        obligation=(
            "No asset content ships in this release; re-verification requires the "
            "caller to accept the upstream licenses (the harness refuses to run "
            "otherwise)."
        ),
        files=(
            BundleFile("OMNIGIBSON_ASSETS_LICENSE.txt", f"{_snap('world_L11')}/OMNIGIBSON_ASSETS_LICENSE.txt", "asset-license statement copied verbatim from the world_L11 pinned source snapshot"),
        ),
        notes=(
            "The OmniGibson/Isaac Sim embodied layer is not part of this release; "
            "the L11 pack's embodied certification leg is recorded BLOCKED "
            "(ModuleNotFoundError: omnigibson) in the pack's "
            "verification/native_certification.json.",
        ),
    ),
    "biomodels": SourceLicense(
        source_id="biomodels",
        license_label="per-model CC0-1.0 / no in-file license block (build-time scan)",
        attribution="BioModels (EMBL-EBI) and the per-model authors as indexed.",
        obligation=(
            "The pack embeds no SBML bytes; per-model license scan results and "
            "attribution are recorded in BIOMODELS_LICENSE.txt."
        ),
        files=(
            BundleFile("BIOMODELS_LICENSE.txt", f"{_snap('world_L07')}/BIOMODELS_LICENSE.txt", "per-model dc:license scan copied verbatim from the pinned source snapshot"),
            _receipt("biomodels"),
        ),
        notes=(
            "Per-model licenses are not machine-readable upstream; the build-time "
            "scan over the 102 pinned models is shipped verbatim.",
        ),
        citations=("BioModels api-20261028 slice, 102 models per the spec's universe rule.",),
    ),
    "camunda-dmn-examples": SourceLicense(
        source_id="camunda-dmn-examples",
        license_label="Apache-2.0",
        attribution="Camunda. Copyright 2013-2025 Camunda Services GmbH.",
        obligation="Retain the Apache-2.0 LICENSE and NOTICE files per pinned commit.",
        files=(
            BundleFile("LICENSE.commit-6c7f4c4adb4b", f"{_V3}/camunda-dmn-examples/LICENSE.commit-6c7f4c4adb4b", "Apache-2.0 text of the pinned examples commit"),
            BundleFile("LICENSE.commit-ee4826e5e76c", f"{_V3}/camunda-dmn-examples/LICENSE.commit-ee4826e5e76c", "Apache-2.0 text of the pinned engine-test commit (camunda-bpm-platform)"),
            BundleFile("NOTICE.commit-6c7f4c4adb4b", f"{_V3}/camunda-dmn-examples/NOTICE.commit-6c7f4c4adb4b", "NOTICE of the pinned examples commit"),
            BundleFile("NOTICE.commit-ee4826e5e76c", f"{_V3}/camunda-dmn-examples/NOTICE.commit-ee4826e5e76c", "NOTICE of the pinned engine-test commit"),
            BundleFile("DMN_LICENSES.txt", f"{_snap('world_L03')}/DMN_LICENSES.txt", "license statement copied verbatim from the pinned source snapshot"),
            _receipt("camunda"),
        ),
        notes=(
            "The second pinned tarball is camunda-bpm-platform engine-test tables "
            "filed under the camunda-dmn-examples source id; same Apache-2.0 license.",
        ),
        citations=(
            "camunda-dmn-examples pinned commits 6c7f4c4adb4b (demo) and "
            "ee4826e5e76c (engine-test).",
        ),
    ),
    "chembl": SourceLicense(
        source_id="chembl",
        license_label="CC BY-SA 3.0",
        attribution="ChEMBL database (EMBL-EBI), DOI 10.6019/CHEMBL.database.37.",
        obligation=(
            "Attribution per REQUIRED.ATTRIBUTION; preserve ChEMBL IDs and the "
            "release number; share-alike terms apply to derivative adaptations of "
            "the database."
        ),
        files=(
            BundleFile("LICENSE", f"{_V3}/chembl/LICENSE", "CC BY-SA 3.0 legal code of the pinned ChEMBL 37 snapshot"),
            BundleFile("REQUIRED.ATTRIBUTION", f"{_V3}/chembl/REQUIRED.ATTRIBUTION", "required attribution text of the pinned ChEMBL 37 snapshot"),
            BundleFile("CHEMBL_LICENSE.txt", f"{_snap('world_L14')}/CHEMBL_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
        ),
        citations=(
            "Mendez D, Gaulton A, Bento AP, et al. ChEMBL: towards direct "
            "deposition of bioassay data. Nucleic Acids Res. "
            "2019;47(D1):D930-D940. DOI: 10.1093/nar/gky1075.",
        ),
    ),
    "cql-clinical": SourceLicense(
        source_id="cql-clinical",
        license_label="Apache-2.0; vendored Node.js runtime MIT",
        attribution="CQL is published by HL7; reference implementation by the CQFramework project.",
        obligation="Retain the Apache-2.0 LICENSE.",
        files=(
            BundleFile("LICENSE", f"{_V3}/cql-clinical/LICENSE", "Apache-2.0 text of the pinned clinical_quality_language commit"),
            BundleFile("CQL_LICENSE.txt", f"{_snap('world_L08')}/CQL_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            _receipt("cql"),
        ),
        notes=("Panels and rule libraries are fully synthetic; no PHI.",),
        citations=(
            "clinical_quality_language pinned commit "
            "bc0c795e4f2a26e8ec3488efd3287e37a86983f8.",
        ),
    ),
    "dmn-tck": SourceLicense(
        source_id="dmn-tck",
        license_label="Apache-2.0",
        attribution="OMG DMN Technology Compatibility Kit (dmn-tck/tck).",
        obligation="Retain the Apache-2.0 license text (LICENSE-ASL-2.0.txt).",
        files=(
            BundleFile("LICENSE-ASL-2.0.txt", f"{_V3}/dmn-tck/LICENSE-ASL-2.0.txt", "Apache-2.0 text of the pinned dmn-tck commit"),
        ),
        notes=(
            "The on-disk fetch receipt's license field reads 'MIT' — stale, "
            "recorded, receipt left untouched per the isolation contract (see "
            "camunda-dmn-examples/DMN_LICENSES.txt).",
        ),
        citations=("dmn-tck conformance-test fixtures, pinned commit 1e349176e5ae.",),
    ),
    "gatekeeper-library": SourceLicense(
        source_id="gatekeeper-library",
        license_label="Apache-2.0",
        attribution="Gatekeeper library. Copyright 2018-2020 The Gatekeeper Authors.",
        obligation="Retain the Apache-2.0 LICENSE and NOTICE with any redistribution of the pinned rule units.",
        files=(
            BundleFile("LICENSE", f"{_snap('world_L01')}/gatekeeper-library/LICENSE", "Apache-2.0 text copied verbatim from the pinned source snapshot"),
            BundleFile("NOTICE", f"{_V3}/gatekeeper-library/NOTICE", "NOTICE of the pinned gatekeeper-library commit"),
            _receipt("gatekeeper"),
        ),
        citations=(
            "Open Policy Agent Gatekeeper policy library, pinned commit "
            "a8d4799d8dafd47ec5fafe945ef277d329e2cceb.",
        ),
    ),
    "gene-ontology": SourceLicense(
        source_id="gene-ontology",
        license_label="CC-BY-4.0",
        attribution="The Gene Ontology Consortium; EMBL-EBI GOA.",
        obligation="CC BY 4.0 attribution to the Gene Ontology Consortium.",
        files=(
            BundleFile("GO_LICENSE.txt", f"{_snap('world_L06')}/GO_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            _receipt("go"),
        ),
        notes=(
            "The GOA human annotation file (goa_human.gaf.238.gz) carries no "
            "standalone pinned license file; CC BY 4.0 is asserted per EBI GOA "
            "standard terms and recorded in the registry receipt — open item "
            "flagged to the PI.",
        ),
        citations=("go-basic.obo go-2026-08-05; GOA human annotations goa238.",),
    ),
    "kubernetes-vap-cel": SourceLicense(
        source_id="kubernetes-vap-cel",
        license_label="CC-BY-4.0 (documentation), Apache-2.0 (code examples)",
        attribution="The Kubernetes Authors.",
        obligation="Credit the Kubernetes authors, link the CC BY 4.0 license, and indicate changes.",
        files=(
            BundleFile("LICENSE", f"{_V3}/kubernetes-vap-cel/LICENSE", "CC BY 4.0 legal code of the pinned kubernetes/website shard"),
            BundleFile("SNAPSHOT_LICENSE.txt", f"{_snap('world_L02')}/LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            _receipt("cel"),
        ),
        citations=(
            "kubernetes/website validating-admission-policy docs shard, pinned "
            "commit ed70e311737a341574ed5758b7af24e3bba232cb.",
        ),
    ),
    "pm4py-process": SourceLicense(
        source_id="pm4py-process",
        license_label="AGPL-3.0-or-later",
        attribution="Process Intelligence Solutions GmbH (pm4py authors).",
        obligation=(
            "Fetch-recipe posture: the full AGPL-3.0 license text ships here; "
            "upstream event-log content is referenced by hash, not bundled."
        ),
        files=(
            BundleFile("LICENSE", f"{_V3}/pm4py-process/LICENSE", "AGPL-3.0 text of the pinned pm4py tag"),
            BundleFile("PM4PY_LICENSE.txt", f"{_snap('world_L12')}/PM4PY_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
        ),
        notes=(
            "No sources/registry.d receipt exists for pm4py-process in this "
            "repository state (recorded in the world license field).",
            "Whether derived task data triggers AGPL obligations is an open legal "
            "question flagged for PI/legal review; no pm4py code ships in this "
            "release.",
        ),
        citations=("pm4py tag 2.7.23.11 test artifacts.",),
    ),
    "reactome": SourceLicense(
        source_id="reactome",
        license_label="CC0-1.0 (database data)",
        attribution="Reactome (reactome.org).",
        obligation=(
            "CC0 data; pathway illustrations/icons/art are CC BY 4.0 and are NOT "
            "part of this release (data-only lane)."
        ),
        files=(
            BundleFile("REACTOME_LICENSE.txt", f"{_snap('world_L15')}/REACTOME_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            BundleFile("fetch_receipt.json", f"{_snap('world_L15')}/fetch_receipt.json", "pinned fetch evidence recording the license verdict (byte-identical to external_knowledge/fetch_receipts/reactome/receipt-release-98.json)"),
        ),
        notes=(
            "fetch_receipt.json is the only fetch receipt bundled in this "
            "release: it contains no build-machine absolute paths.",
        ),
        citations=("Reactome release-98.",),
    ),
    "rhea-chebi": SourceLicense(
        source_id="rhea-chebi",
        license_label="Rhea CC-BY-4.0; ChEBI CC-BY-4.0",
        attribution="Rhea (SIB Swiss Institute of Bioinformatics); ChEBI (European Bioinformatics Institute, EMBL-EBI).",
        obligation=(
            "Reproduce the Rhea copyright statement with each copy "
            "(rhea_LICENSE.txt); provide ChEBI attribution and the release "
            "identifier. ChEBI is CC BY 4.0; no ShareAlike obligation is asserted."
        ),
        files=(
            BundleFile("rhea_LICENSE.txt", f"{_snap('world_L04')}/rhea_LICENSE.txt", "Rhea license/copyright statement copied verbatim from the pinned source snapshot"),
            BundleFile("chebi_LICENSE", f"{_snap('world_L04')}/chebi_LICENSE", "pinned ChEBI license text copied verbatim from the source snapshot"),
            BundleFile("chebi_README", f"{_snap('world_L04')}/chebi_README", "pinned ChEBI README copied verbatim from the source snapshot"),
            _receipt("rhea"),
            _receipt("chebi"),
        ),
        notes=(
            "ChEBI's current license is CC BY 4.0. The pinned ChEBI README "
            "contains a stale ShareAlike line, but the pinned chebi_LICENSE, "
            "the ChEBI GitHub LICENSE, EMBL-EBI's current training terms, and "
            "the 2025 ChEBI paper all state CC BY 4.0. The release follows the "
            "pinned LICENSE text; attribution to ChEBI and release 256 is "
            "required.",
        ),
        citations=("Rhea release 142; ChEBI release 256.",),
    ),
    "rdkit-molecules": SourceLicense(
        source_id="rdkit-molecules",
        license_label="RDKit BSD-3-Clause; ChEBI CC-BY-4.0",
        attribution="Rational Discovery LLC, Greg Landrum, Julie Penzotti and others (RDKit); ChEBI (EMBL-EBI).",
        obligation="Retain the BSD-3-Clause notice for RDKit content; CC BY 4.0 attribution for ChEBI structures.",
        files=(
            BundleFile("RDKIT_LICENSE.txt", f"{_snap('world_L05')}/RDKIT_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            BundleFile("CHEBI_LICENSE.txt", f"{_snap('world_L05')}/CHEBI_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            BundleFile("rdkit_license.txt", f"{_V3}/rdkit-molecules/rdkit_license.txt", "RDKit BSD-3-Clause license text"),
            BundleFile("chebi_SDF_LICENSE", f"{_V3}/rdkit-molecules/chebi_SDF_LICENSE", "CC BY 4.0 legal code of the pinned ChEBI rel255 SDF"),
            _receipt("rdkit"),
            _receipt("chebi"),
        ),
        citations=(
            "RDKit Release_2026_09_1 functional-group hierarchy; ChEBI release "
            "255 3-star structures.",
        ),
    ),
    "smart-anc": SourceLicense(
        source_id="smart-anc",
        license_label="CC0-1.0",
        attribution="World Health Organization (WHO).",
        obligation="CC0 requires no attribution; WHO is credited regardless.",
        files=(
            BundleFile("LICENSE.md", f"{_V3}/smart-anc/LICENSE.md", "CC0 1.0 legal code of the pinned smart-anc commit"),
            BundleFile("SMART_ANC_LICENSE.txt", f"{_snap('world_L09')}/SMART_ANC_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
            _receipt("smart_anc"),
        ),
        notes=(
            "The registry expected CC-BY-4.0/WHO terms while the pinned "
            "LICENSE.md is CC0-1.0; the pinned text governs (recorded mismatch).",
        ),
        citations=(
            "WHO SMART-ANC implementation guide, pinned commit "
            "301a6b8b904361315ca38b28ee9fbd133f9d0e94.",
        ),
    ),
    "uniprot": SourceLicense(
        source_id="uniprot",
        license_label="CC-BY-4.0",
        attribution="UniProt Consortium (EMBL-EBI, SIB, PIR).",
        obligation="Retain the UniProt Consortium attribution and the release identifier (2026_03, Swiss-Prot section).",
        files=(
            BundleFile("README", f"{_V3}/uniprot/README", "pinned snapshot README carrying the LICENSE section"),
            BundleFile("UNIPROT_LICENSE.txt", f"{_snap('world_L16')}/UNIPROT_LICENSE.txt", "license statement copied verbatim from the pinned source snapshot"),
        ),
        citations=("UniProtKB/Swiss-Prot release 2026_03.",),
    ),
}


def _reg_receipt_ref(name: str, source_id: str) -> Reference:
    return Reference(
        pointer=f"{_REGISTRY}/{name}.yaml",
        status="shipped",
        paths=(f"LICENSES/{source_id}/registry_receipts/{name}.yaml",),
        detail="verbatim copy of the source-registry receipt ships in the bundle",
    )


_FETCH_RECEIPT_REASON: Final = (
    "the fetch receipt is repository-internal build evidence (it records "
    "build-machine absolute paths); its license verdict is restated verbatim "
    "in the bundled snapshot license text"
)

WORLDS: Final[dict[str, WorldLicense]] = {
    "world_L01": WorldLicense(
        world_id="world_L01",
        sources=("gatekeeper-library",),
        references=(_reg_receipt_ref("gatekeeper", "gatekeeper-library"),),
    ),
    "world_L02": WorldLicense(
        world_id="world_L02",
        sources=("kubernetes-vap-cel",),
        references=(_reg_receipt_ref("cel", "kubernetes-vap-cel"),),
    ),
    "world_L03": WorldLicense(
        world_id="world_L03",
        sources=("camunda-dmn-examples", "dmn-tck"),
        references=(_reg_receipt_ref("camunda", "camunda-dmn-examples"),),
    ),
    "world_L04": WorldLicense(
        world_id="world_L04",
        sources=("rhea-chebi",),
        references=(
            _reg_receipt_ref("rhea", "rhea-chebi"),
            _reg_receipt_ref("chebi", "rhea-chebi"),
        ),
    ),
    "world_L05": WorldLicense(
        world_id="world_L05",
        sources=("rdkit-molecules",),
        references=(
            _reg_receipt_ref("rdkit", "rdkit-molecules"),
            _reg_receipt_ref("chebi", "rdkit-molecules"),
        ),
    ),
    "world_L06": WorldLicense(
        world_id="world_L06",
        sources=("gene-ontology",),
        references=(_reg_receipt_ref("go", "gene-ontology"),),
    ),
    "world_L07": WorldLicense(
        world_id="world_L07",
        sources=("biomodels",),
        references=(_reg_receipt_ref("biomodels", "biomodels"),),
        snapshot_pointer="source_snapshot/BIOMODELS_LICENSE.txt",
    ),
    "world_L08": WorldLicense(
        world_id="world_L08",
        sources=("cql-clinical",),
        references=(_reg_receipt_ref("cql", "cql-clinical"),),
    ),
    "world_L09": WorldLicense(
        world_id="world_L09",
        sources=("smart-anc",),
        references=(_reg_receipt_ref("smart_anc", "smart-anc"),),
    ),
    "world_L10": WorldLicense(
        world_id="world_L10",
        sources=("behavior-1k-bddl",),
        references=(_reg_receipt_ref("behavior", "behavior-1k-bddl"),),
    ),
    "world_L11": WorldLicense(
        world_id="world_L11",
        sources=("behavior-1k-bddl", "behavior-1k-omnigibson"),
        references=(
            _reg_receipt_ref("behavior", "behavior-1k-bddl"),
            Reference(
                pointer="BEHAVIOR-1K OmniGibson scene/object assets",
                status="not_shipped",
                detail=(
                    "license-gated (behavior.stanford.edu/agreement) plus NVIDIA "
                    "Isaac Sim EULA; never embedded — referenced by snapshot id "
                    "and hashes only (see "
                    "LICENSES/behavior-1k-omnigibson/OMNIGIBSON_ASSETS_LICENSE.txt)"
                ),
            ),
        ),
    ),
    "world_L12": WorldLicense(
        world_id="world_L12",
        sources=("pm4py-process",),
        references=(
            Reference(
                pointer="the fetch receipt",
                status="not_shipped",
                detail=_FETCH_RECEIPT_REASON + " (LICENSES/pm4py-process/PM4PY_LICENSE.txt)",
            ),
            Reference(
                pointer="sources/registry.d receipt for pm4py-process",
                status="not_shipped",
                detail=(
                    "no sources/registry.d receipt exists for pm4py-process in "
                    "this repository state (recorded in the world license field)"
                ),
            ),
        ),
    ),
    "world_L13": WorldLicense(
        world_id="world_L13",
        sources=("apis-guru",),
        references=(
            Reference(
                pointer="the fetch receipt",
                status="not_shipped",
                detail=_FETCH_RECEIPT_REASON + " (LICENSES/apis-guru/APIS_GURU_LICENSE.txt)",
            ),
            Reference(
                pointer="sources/registry.d/apis-guru.yaml",
                status="not_shipped",
                detail=(
                    "no sources/registry.d/apis-guru.yaml exists in this "
                    "repository state (recorded in the world license field and "
                    "the snapshot declaration)"
                ),
            ),
            Reference(
                pointer="per-spec licenses as indexed",
                status="not_shipped",
                detail=(
                    "the release ships the pinned unit index hashes only (no "
                    "OpenAPI spec content); per-spec upstream license "
                    "obligations pass to the user"
                ),
            ),
        ),
    ),
    "world_L14": WorldLicense(
        world_id="world_L14",
        sources=("chembl",),
        references=(
            Reference(
                pointer="the pinned snapshot LICENSE/REQUIRED.ATTRIBUTION",
                status="shipped",
                paths=("LICENSES/chembl/LICENSE", "LICENSES/chembl/REQUIRED.ATTRIBUTION"),
                detail="verbatim copies of the pinned ChEMBL 37 snapshot files",
            ),
        ),
    ),
    "world_L15": WorldLicense(
        world_id="world_L15",
        sources=("reactome",),
        references=(
            Reference(
                pointer="external_knowledge/fetch_receipts/reactome/receipt-release-98.json",
                status="shipped",
                paths=("LICENSES/reactome/fetch_receipt.json",),
                detail=(
                    "byte-identical copy; records "
                    "receipt-release-98.json::license_verdict verbatim"
                ),
            ),
        ),
    ),
    "world_L16": WorldLicense(
        world_id="world_L16",
        sources=("uniprot",),
        references=(
            Reference(
                pointer="the pinned snapshot README LICENSE section",
                status="shipped",
                paths=("LICENSES/uniprot/README",),
                detail="verbatim copy of the pinned snapshot README (LICENSE section included)",
            ),
        ),
    ),
}

CODE_LICENSE_ORIGIN: Final = "release_candidates/AutoOntologyBench/LICENSE"


def resolve_world(world_id: str) -> WorldLicense:
    """Return the world's license wiring or raise ``license_source_missing``."""
    world = WORLDS.get(world_id)
    if world is None:
        raise ReleaseError(
            "license_source_missing",
            f"no license attribution registry entry for {world_id}; "
            "add one to scripts/core_release/license_registry.py",
        )
    return world


__all__ = [
    "CODE_LICENSE_ORIGIN",
    "SOURCES",
    "WORLDS",
    "BundleFile",
    "Reference",
    "SourceLicense",
    "WorldLicense",
    "resolve_world",
]
