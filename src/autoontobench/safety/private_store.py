"""Private-root helpers (owner: W1, G0 gate).

The private root (`worlds/private/<world_id>/`) holds everything the public
projection strips: hidden fixtures, native oracle logs, the gold reference
artifact, provenance manifests and the leak baseline. It must never be
readable by the researcher uid — defence in depth via chmod 0700/0600, with
the mount boundary (container isolation) scheduled for M2.
"""
from __future__ import annotations

import os
import stat
from pathlib import Path

DIR_MODE = 0o700
FILE_MODE = 0o600


def private_world_root(private_base: Path, world_id: str) -> Path:
    return private_base / world_id


def seal_permissions(root: Path) -> int:
    """chmod a private tree to owner-only (0700 dirs / 0600 files). Returns count."""
    sealed = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            os.chmod(p, DIR_MODE)
            sealed += 1
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            os.chmod(p, FILE_MODE)
            sealed += 1
    if root.exists() and not root.is_symlink():
        os.chmod(root, DIR_MODE)
        sealed += 1
    return sealed


def is_owner_only(path: Path) -> bool:
    mode = stat.S_IMODE(path.stat().st_mode)
    return mode & 0o077 == 0
