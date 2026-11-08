"""Private oracle mounting for the 20K harness.

The public task file contains *no answers* — only an ``oracle_handle`` and an
``instance_id`` per task. The private oracle payload (task AST, reference
submission, evidence) is mounted separately, e.g.::

    --oracle /path/to/private/aob-20000-candidate-dev1/oracle

or, for local development, directly against the bank staging trees
(``bank/staging_20k`` and ``bank/staging``), which this store searches when
``include_repo_staging`` is enabled.

Resolution is keyed by *filename stem*: a task's oracle payload is the first
readable ``<instance_id>.json`` or ``<oracle_handle>.json`` found under the
mounted roots (per-lane ``oracle/`` directories, shard layouts and flat
layouts are all indexed). Payloads are loaded lazily and cached.

This module only *reads* oracle material; nothing is ever written into the
mounted roots, ``bank/`` or ``release_20k/``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

#: In-repo private staging roots searched when ``include_repo_staging`` is on
#: (read-only; the official flow mounts the private release oracle instead).
REPO_STAGING_ROOTS: tuple[str, ...] = ("bank/staging_20k", "bank/staging")

#: Environment variable with os.pathsep-separated oracle roots (prepended to
#: --oracle roots; lowest precedence is repo staging).
ENV_ORACLE_ROOTS = "AOB_20K_ORACLE"


class OracleNotFound(KeyError):
    """No oracle payload found for a task in any mounted root."""


class OracleStore:
    """Read-only view over one or more mounted oracle roots."""

    def __init__(self, roots: Iterable[str | Path]) -> None:
        self.roots: tuple[Path, ...] = tuple(Path(r) for r in roots)
        for root in self.roots:
            if not root.is_dir():
                raise FileNotFoundError("oracle root not found: %s" % root)
        # per (lane_lower) index: filename stem -> path
        self._index: dict[str, dict[str, Path]] = {}

    # -- index construction -------------------------------------------------

    def _lane_dirs(self, lane_lower: str) -> Iterable[tuple[Path, Path]]:
        """Yield (root, lane_dir) pairs that exist for this lane."""
        for root in self.roots:
            lane_dir = root / lane_lower
            if lane_dir.is_dir():
                yield root, lane_dir

    def _build_lane_index(self, lane_lower: str) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for root, lane_dir in self._lane_dirs(lane_lower):
            candidates: list[Path] = []
            flat = lane_dir / "oracle"
            if flat.is_dir():
                candidates.extend(sorted(flat.glob("*.json")))
            # flat lane layout: <root>/<lane>/<key>.json (fixture mounts)
            candidates.extend(sorted(lane_dir.glob("*.json")))
            # shard layouts: shard_*/oracle, shard-*/oracle, */oracle
            for sub in sorted(lane_dir.iterdir()):
                if not sub.is_dir() or sub.name == "oracle":
                    continue
                odir = sub / "oracle"
                if odir.is_dir():
                    candidates.extend(sorted(odir.glob("*.json")))
            for path in candidates:
                index.setdefault(path.stem, path)
        # root-level layouts (mounted single-lane or merged oracle dirs)
        for root in self.roots:
            odir = root / "oracle"
            if odir.is_dir():
                for path in sorted(odir.glob("*.json")):
                    index.setdefault(path.stem, path)
        return index

    def _lane_index(self, lane_id: str) -> dict[str, Path]:
        lane_lower = lane_id.lower()
        if lane_lower not in self._index:
            self._index[lane_lower] = self._build_lane_index(lane_lower)
        return self._index[lane_lower]

    # -- public API ----------------------------------------------------------

    def locate(self, lane_id: str, instance_id: str,
               oracle_handle: str | None = None) -> Path:
        """Return the path of the oracle payload for a task."""
        index = self._lane_index(lane_id)
        for key in (instance_id, oracle_handle):
            if key and key in index:
                return index[key]
        raise OracleNotFound(
            "no oracle payload for lane=%s instance=%s handle=%s in roots: %s"
            % (lane_id, instance_id, oracle_handle,
               ", ".join(str(r) for r in self.roots)))

    def load(self, lane_id: str, instance_id: str,
             oracle_handle: str | None = None) -> dict[str, Any]:
        """Load the oracle payload JSON for a task."""
        path = self.locate(lane_id, instance_id, oracle_handle)
        with path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ValueError("oracle payload is not a JSON object: %s" % path)
        return payload

    def has(self, lane_id: str, instance_id: str,
            oracle_handle: str | None = None) -> bool:
        try:
            self.locate(lane_id, instance_id, oracle_handle)
            return True
        except OracleNotFound:
            return False


def default_oracle_roots(*, include_repo_staging: bool = True,
                         extra_roots: Iterable[str | Path] = ()) -> list[Path]:
    """Resolve oracle roots: env var, then explicit --oracle roots, then
    (optionally) the repo staging trees. Missing roots are skipped here and
    reported by the CLI."""
    roots: list[Path] = []
    env = os.environ.get(ENV_ORACLE_ROOTS, "")
    for part in env.split(os.pathsep):
        if part.strip():
            roots.append(Path(part.strip()))
    roots.extend(Path(r) for r in extra_roots)
    if include_repo_staging:
        for rel in REPO_STAGING_ROOTS:
            candidate = REPO_ROOT / rel
            if candidate.is_dir():
                roots.append(candidate)
    # de-duplicate, preserve precedence order
    seen: set[Path] = set()
    unique: list[Path] = []
    for root in roots:
        key = root.resolve()
        if key not in seen and root.is_dir():
            seen.add(key)
            unique.append(root)
    return unique
