"""Store-backed oracle resolution for the L11 native replay verifier.

``benchmark.plugins.verifier.l11.replay_verifier.OmnigibsonReplayVerifier``
resolves its replay oracle by filename inside the frozen pilot directory
(``data/erule/review5/build8000/lanes/L11/artifacts/pilot``), which only
covers the 12 pilot tasks (``oracle_<activity>_<instance_id>.json``). The 20K
build mounts oracle payloads through the harness ``OracleStore`` instead
(``bank/staging_20k/l11/oracle/l11-oracle-<hash>.json`` or a mounted private
oracle root).

This subclass changes *only* oracle resolution: the payload loaded by the
harness is attached to the task dict under :data:`ORACLE_PAYLOAD_KEY` and
returned here. The scoring logic (``verify_terminal``: submitted
``primitive_plan`` equality against the natively replayed oracle program) is
inherited byte-for-byte from the lane's own verifier plugin.

No simulator imports anywhere in this module; safe to import from the plain
repo venv.
"""
from __future__ import annotations

from typing import Any

from benchmark.plugins.verifier.l11.replay_verifier import OmnigibsonReplayVerifier

#: Task-dict key under which the adapter passes the mounted oracle payload.
ORACLE_PAYLOAD_KEY = "_l11_oracle_payload"


class StoreBackedReplayVerifier(OmnigibsonReplayVerifier):
    """OmnigibsonReplayVerifier whose oracle comes from the mounted 20K
    oracle store (via the task dict) instead of the pilot directory."""

    plugin_id = "l11.omnigibson-replay/store-backed-20k"

    def _oracle_for(self, task: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(task, dict):
            return None
        payload = task.get(ORACLE_PAYLOAD_KEY)
        return payload if isinstance(payload, dict) else None


__all__ = ["StoreBackedReplayVerifier", "ORACLE_PAYLOAD_KEY"]
