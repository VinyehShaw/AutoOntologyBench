"""License and attribution layer for the Core-track benchmark release.

The assembler ships a self-contained license bundle with every release:

- ``LICENSE`` — Apache-2.0 (the project code license, copied verbatim).
- ``LICENSE-DATA.md`` — CC BY 4.0 grant for the release data.
- ``NOTICE`` — project + third-party attribution notices.
- ``LICENSES/`` — per-source bundle: verbatim upstream license/attribution
  texts, registry-receipt copies, and a generated ``SUMMARY.md`` per source,
  plus a generated ``ATTRIBUTION.md`` master table.

The mapping from a world to its upstream evidence lives in
``license_registry.py``; document text lives in ``license_render.py`` and the
``templates/`` directory. Everything is deterministic (sorted iteration, no
timestamps), so rebuilds stay byte-identical.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from .errors import ReleaseError
from .license_registry import (
    CODE_LICENSE_ORIGIN,
    SOURCES,
    Reference,
    SourceLicense,
    WorldLicense,
    resolve_world,
)
from .license_render import render_attribution, render_notice, render_summary
from .validate import ValidatedWorld

LICENSE_ATTRIBUTION_SCHEMA: Final = "aob-core-license-attribution/1.0"
CC_BY_4_0_LEGALCODE: Final = "LICENSES/legalcode/CC-BY-4.0.txt"

_TEMPLATE_DIR: Final = Path(__file__).with_name("templates")

_REQUIRED_TOP_LEVEL: Final = (
    "LICENSE",
    "LICENSE-DATA.md",
    "NOTICE",
    "LICENSES/README.md",
    "LICENSES/ATTRIBUTION.md",
    CC_BY_4_0_LEGALCODE,
)


def _sources_for(world_ids: list[str]) -> dict[str, SourceLicense]:
    source_ids: set[str] = set()
    for world_id in world_ids:
        source_ids.update(resolve_world(world_id).sources)
    return {source_id: SOURCES[source_id] for source_id in sorted(source_ids)}


def _snapshot_reference(world: WorldLicense) -> Reference:
    """Resolve the world's ``source_snapshot/`` pointer to bundled files."""
    prefix = f"pilot_worlds/{world.world_id}_scaleup/source_snapshot/"
    paths = sorted(
        f"LICENSES/{source_id}/{bundle_file.dest}"
        for source_id in world.sources
        for bundle_file in SOURCES[source_id].files
        if bundle_file.origin.startswith(prefix)
    )
    return Reference(
        pointer=world.snapshot_pointer,
        status="shipped",
        paths=tuple(paths),
        detail=world.snapshot_note,
    )


def _reference_block(reference: Reference) -> dict[str, object]:
    block: dict[str, object] = {"pointer": reference.pointer, "status": reference.status}
    if reference.status == "shipped":
        block["paths"] = sorted(reference.paths)
        block["note"] = reference.detail
    else:
        block["reason"] = reference.detail
    return block


def world_license_attribution(world_id: str) -> dict[str, object]:
    """Metadata block recording how every license reference of a world resolves."""
    world = resolve_world(world_id)
    bundle_paths = sorted(
        f"LICENSES/{source_id}/{bundle_file.dest}"
        for source_id in world.sources
        for bundle_file in SOURCES[source_id].files
    )
    bundle_paths.extend(
        f"LICENSES/{source_id}/SUMMARY.md" for source_id in sorted(world.sources)
    )
    references = [_snapshot_reference(world), *world.references]
    return {
        "schema": LICENSE_ATTRIBUTION_SCHEMA,
        "project_grant": {
            "code": "Apache-2.0",
            "data": "CC-BY-4.0",
            "files": ["LICENSE", "LICENSE-DATA.md", "NOTICE", "LICENSES/ATTRIBUTION.md"],
            "sign_off": (
                "PENDING PI SIGN-OFF (posture per "
                "reports/w7_license_recommendation.md, DRAFT)"
            ),
        },
        "bundle_paths": sorted(bundle_paths),
        "references": [_reference_block(reference) for reference in references],
    }


def manifest_license_block() -> dict[str, object]:
    """Top-level ``license`` object for MANIFEST.json."""
    return {
        "code": "Apache-2.0",
        "data": "CC-BY-4.0",
        "files": {
            "attribution": "LICENSES/ATTRIBUTION.md",
            "code_license": "LICENSE",
            "data_legalcode": CC_BY_4_0_LEGALCODE,
            "data_license": "LICENSE-DATA.md",
            "notice": "NOTICE",
        },
        "sign_off": "PENDING PI SIGN-OFF",
        "recommendation": "reports/w7_license_recommendation.md (DRAFT)",
        "note": (
            "Upstream source terms in LICENSES/ and NOTICE are not superseded by "
            "the project code/data grants."
        ),
    }


def _copy_bundle_files(
    root: Path, sources: dict[str, SourceLicense], repo_root: Path, written: list[str]
) -> None:
    for source in sources.values():
        source_dir = root / "LICENSES" / source.source_id
        for bundle_file in source.files:
            origin = repo_root / bundle_file.origin
            if not origin.is_file():
                raise ReleaseError(
                    "license_source_missing",
                    f"{bundle_file.origin} not found (needed for "
                    f"LICENSES/{source.source_id}/{bundle_file.dest})",
                )
            target = source_dir / bundle_file.dest
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(origin.read_bytes())
            written.append(f"LICENSES/{source.source_id}/{bundle_file.dest}")


def write_license_layer(
    root: Path, worlds: list[ValidatedWorld], repo_root: Path
) -> list[str]:
    """Write LICENSE/LICENSE-DATA.md/NOTICE/LICENSES/; return the written paths."""
    ordered = sorted(worlds, key=lambda world: world.world_id)
    sources = _sources_for([world.world_id for world in ordered])
    licenses_dir = root / "LICENSES"
    written: list[str] = []
    _copy_bundle_files(root, sources, repo_root, written)
    for source in sources.values():
        source_dir = licenses_dir / source.source_id
        (source_dir / "SUMMARY.md").write_text(
            render_summary(source, ordered), encoding="utf-8"
        )
        written.append(f"LICENSES/{source.source_id}/SUMMARY.md")
    (licenses_dir / "legalcode").mkdir(parents=True, exist_ok=True)
    (licenses_dir / "legalcode" / "CC-BY-4.0.txt").write_bytes(
        (_TEMPLATE_DIR / "CC-BY-4.0.txt").read_bytes()
    )
    (licenses_dir / "README.md").write_text(
        (_TEMPLATE_DIR / "licenses_README.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (licenses_dir / "ATTRIBUTION.md").write_text(
        render_attribution(ordered, sources), encoding="utf-8"
    )
    code_license = repo_root / CODE_LICENSE_ORIGIN
    if not code_license.is_file():
        raise ReleaseError(
            "license_source_missing", f"{CODE_LICENSE_ORIGIN} not found (needed for LICENSE)"
        )
    (root / "LICENSE").write_bytes(code_license.read_bytes())
    (root / "LICENSE-DATA.md").write_text(
        (_TEMPLATE_DIR / "LICENSE-DATA.md").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (root / "NOTICE").write_text(render_notice(sources, ordered), encoding="utf-8")
    written.extend(_REQUIRED_TOP_LEVEL)
    return sorted(written)


__all__ = [
    "CC_BY_4_0_LEGALCODE",
    "LICENSE_ATTRIBUTION_SCHEMA",
    "manifest_license_block",
    "resolve_world",
    "world_license_attribution",
    "write_license_layer",
]
