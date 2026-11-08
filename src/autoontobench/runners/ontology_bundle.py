"""Artifact bundle wire format (owner: W6): ``aob-ontology-bundle/1.0``.

One canonical JSON document ``{"bundle", "profile", "files": {name: text}}``
(``sort_keys``, tight separators). The exact canonical bytes are what
``seal`` hashes. Required files for ``shacl-rule/1``: ``ontology.ttl``,
``shapes.ttl``, ``manifest.json``; optional: ``mappings.json`` (contract
``mappings/1.0`` informational mirror) and ``phi.json`` (``erule-phi/0.1``
Graw name normalization — absent means identity naming, which can only yield
UNKNOWN downstream, never a wrong verdict).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

BUNDLE_SCHEMA = "aob-ontology-bundle/1.0"
REQUIRED_FILES = ("ontology.ttl", "shapes.ttl", "manifest.json")
OPTIONAL_FILES = ("mappings.json", "phi.json")
MAX_BUNDLE_BYTES = 4 * 1024 * 1024


class BundleRejected(ValueError):
    """The bundle envelope failed structural validation (fail-closed)."""

    def __init__(self, reasons: list[str]):
        self.reasons = list(reasons)
        super().__init__("bundle rejected: " + "; ".join(self.reasons))


def build_bundle_bytes(files: Mapping[str, str], profile: str) -> bytes:
    return json.dumps(
        {"bundle": BUNDLE_SCHEMA, "profile": profile, "files": dict(files)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def bundle_content_hash(bundle_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(bundle_bytes).hexdigest()


def parse_bundle(bundle_bytes: bytes) -> dict[str, str]:
    if len(bundle_bytes) > MAX_BUNDLE_BYTES:
        raise BundleRejected(
            ["bundle: %d bytes exceed cap %d" % (len(bundle_bytes), MAX_BUNDLE_BYTES)]
        )
    try:
        doc: Any = json.loads(bundle_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleRejected(["bundle: not a JSON document: %s" % exc]) from exc
    if not isinstance(doc, dict) or doc.get("bundle") != BUNDLE_SCHEMA:
        raise BundleRejected(["bundle: schema must be %r" % BUNDLE_SCHEMA])
    files = doc.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        raise BundleRejected(["bundle: files must be an object of name -> utf8 content"])
    unknown = sorted(set(files) - set(REQUIRED_FILES) - set(OPTIONAL_FILES))
    if unknown:
        raise BundleRejected(["bundle: unexpected files %s (fail-closed)" % unknown])
    missing = [name for name in REQUIRED_FILES if name not in files]
    if missing:
        raise BundleRejected(["bundle: missing required files %s" % missing])
    return dict(files)
