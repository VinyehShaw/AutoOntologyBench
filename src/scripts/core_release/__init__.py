"""Core-track benchmark release assembler (packaging framework, contract v1).

Reads Core-track world packs (the 17-file contract layout), validates them, and
emits a benchmark-only release under ``release_core/``: public task records,
per-world public materials + metadata, and hash manifests. Private oracle
material is never copied.

Entry point: ``src/scripts/assemble_core_release.py``. Contract: the generated
``release_core/CORE_RELEASE_CONTRACT.md``.
"""