"""L13 eval adapter: OpenAPI contract truth (APIs.guru pinned specs).

The 20K L13 oracle payloads do **not** carry a top-level ``ast`` (unlike the
L05-style lanes); they carry the re-derivable contract material directly
(``goal_kind``/``unit_id``/``contract``/``reference``/...).  ``extract_ast``
rebuilds exactly the AST the lane's own audit/recheck code uses
(``bank/audits/aob-20000-audit-v3/l13/scripts/audit_v3.py::ast_from_oracle``)
and ``reference_answer`` replays the lane's own
``l13_exp.pipeline.reference_submission`` so the harness never invents an
answer.

Native verifier: ``l13_exp.verifier.ExpansionVerifier`` -- the 20K subclass of
``benchmark/plugins/verifier/l13/openapi_verifier.py::OpenAPIContractVerifier``
that adds the expansion kinds (``status_selection``, ``media_type_selection``,
``header_construction``, response-mode ``version_migration``).  The legacy
8K carryover oracle rows use those same kinds, so the subclass is required for
the lane to be live on all 1,150 rows; it is constructed exactly like the base
class (``ExpansionVerifier(oracle_dir)``).

The native verifier re-reads its own oracle file from ``oracle_dir`` by
``oracle_ref``.  We point it at a private symlink mirror of the read-only
staging oracle trees because the base constructor ``mkdir``/``chmod``s its
directory -- mirroring keeps the bank trees untouched.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

from benchmark.eval_20k.lanes import LaneAdapter, LaneNotSupported, Verdict

from .l13_answer import normalize_answer

#: Verifier exceptions caused by an answer whose shape does not match the
#: submission wrapper (``'str' object has no attribute 'get'`` etc.).  These
#: are task failures, not harness/environment errors, so they are recorded as
#: a scored ``False`` instead of poisoning the run as ``ERROR``.
_SHAPE_ERROR_RE = re.compile(
    r"verify_terminal raised (AttributeError|TypeError|KeyError):")

REPO_ROOT = Path(__file__).resolve().parents[3]

#: oracle trees searched when no explicit ``--lane-kw L13.oracle_dir`` is given.
DEFAULT_ORACLE_DIRS: tuple[str, ...] = (
    "bank/staging_20k/l13/oracle",
    "bank/staging_20k/l13/legacy_carryover/oracle",
    "bank/staging/l13/oracle",
)

#: fields the lane's ``ast_from_oracle`` consumes (see audit_v3.py).
_AST_FIELDS = (
    "goal_kind", "unit_id", "unit_digest", "unit_id_b", "unit_digest_b",
    "coordinate", "contract", "contract_b", "initial", "reference", "guards",
    "constraints", "witness",
)


def _split_roots(value: str) -> list[Path]:
    return [Path(p) for p in value.split(os.pathsep) if p.strip()]


class Adapter(LaneAdapter):
    lane_id = "L13"

    def __init__(self, lane_kw: dict[str, str] | None = None) -> None:
        super().__init__(lane_kw)
        self._mirror_dir: Path | None = None

    # -- resources -------------------------------------------------------------

    def _oracle_dirs(self) -> list[Path]:
        explicit = (self.lane_kw.get("oracle_dir")
                    or os.environ.get("AOB_L13_ORACLE"))
        candidates: list[Path] = []
        if explicit:
            candidates.extend(_split_roots(explicit))
        else:
            candidates.extend(REPO_ROOT / rel for rel in DEFAULT_ORACLE_DIRS)
        dirs: list[Path] = []
        for raw in candidates:
            # accept either an oracle dir or a lane/root dir containing one
            # (a lane root carries both ``oracle/`` and the legacy carryover)
            for cand in (raw / "oracle", raw / "legacy_carryover" / "oracle", raw):
                if cand.is_dir() and any(cand.glob("*.json")):
                    if cand not in dirs:
                        dirs.append(cand)
        return dirs

    def _mirror_oracles(self, dirs: list[Path]) -> Path:
        """Symlink every oracle JSON into a private dir (constructor chmods it)."""
        mirror = Path(tempfile.mkdtemp(prefix="aob_l13_oracle_"))
        atexit.register(shutil.rmtree, mirror, ignore_errors=True)
        for src in dirs:
            for path in sorted(src.glob("*.json")):
                dest = mirror / path.name
                if dest.exists():
                    continue
                try:
                    os.symlink(path.resolve(), dest)
                except OSError:  # cross-device / no-symlink fallback
                    shutil.copy2(path, dest)
        return mirror

    # -- verifier --------------------------------------------------------------

    def build_verifier(self) -> Any:
        try:
            from source_registry.fetchers import l13_apis_guru as fetcher
        except ImportError as exc:  # pragma: no cover - import guard
            raise LaneNotSupported(
                "L13 pinned snapshot module not importable: %s" % exc) from exc
        if not fetcher.PINNED_ROOT.is_dir():
            raise LaneNotSupported(
                "L13 requires the pinned APIs.guru snapshot at %s"
                % fetcher.PINNED_ROOT)
        dirs = self._oracle_dirs()
        if not dirs:
            raise LaneNotSupported(
                "L13 requires the staging oracle tree "
                "(bank/staging_20k/l13/oracle); pass "
                "--lane-kw L13.oracle_dir=<dir> or set AOB_L13_ORACLE")
        try:
            from benchmark.plugins.task_family.l13_exp.verifier import (
                ExpansionVerifier,
            )
            verifier_cls = ExpansionVerifier
        except ImportError:
            try:
                from benchmark.plugins.verifier.l13.openapi_verifier import (
                    OpenAPIContractVerifier as verifier_cls,
                )
            except ImportError as exc:  # pragma: no cover - import guard
                raise LaneNotSupported(
                    "L13 verifier not importable: %s" % exc) from exc
        self._mirror_dir = self._mirror_oracles(dirs)
        try:
            return verifier_cls(self._mirror_dir)
        except Exception as exc:  # native engine unavailable is data, not a crash
            raise LaneNotSupported(
                "L13 verifier construction failed: %s: %s"
                % (type(exc).__name__, exc)) from exc

    # -- scoring ---------------------------------------------------------------

    def evaluate(self, task: Any, prediction: Any, oracle_store: Any) -> Any:
        """Stage the *mounted* oracle file before the native verifier reads it.

        The native verifier reloads its own oracle from ``oracle_dir`` by
        ``oracle_ref``; the harness may be pointed at a private ``--oracle``
        mount rather than the in-repo staging tree, so point the mirror at the
        exact payload ``OracleStore`` resolved for this task.
        """
        try:
            self.verifier
        except LaneNotSupported:
            return super().evaluate(task, prediction, oracle_store)
        try:
            source = Path(oracle_store.locate(
                task.lane_id, task.instance_id, task.oracle_handle))
        except Exception:
            return super().evaluate(task, prediction, oracle_store)
        mirror = self._mirror_dir
        if mirror is not None:
            for key in (task.oracle_handle, task.instance_id):
                if not key:
                    continue
                dest = mirror / ("%s.json" % key)
                try:
                    if dest.is_symlink() or dest.exists():
                        dest.unlink()
                except OSError:
                    pass
                try:
                    os.symlink(source.resolve(), dest)
                except OSError:
                    shutil.copy2(source, dest)
        payload = None
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = None
        note = None
        if isinstance(payload, dict):
            answer, note = normalize_answer(prediction.answer, payload)
            if note is not None:
                prediction = replace(prediction, answer=answer)
        verdict = super().evaluate(task, prediction, oracle_store)
        detail = dict(verdict.detail or {})
        if note is not None:
            detail["answer_normalized"] = note
            verdict = replace(verdict, detail=detail)
        if verdict.status == "ERROR" and _SHAPE_ERROR_RE.search(verdict.reason):
            return Verdict(
                status="SCORED", valid=False,
                detail={"error": verdict.reason,
                        **({"answer_normalized": note} if note else {})},
                reason="answer shape rejected: %s" % verdict.reason,
                verifier_id=verdict.verifier_id,
                oracle_path=verdict.oracle_path,
            )
        return verdict

    # -- AST rebuild -----------------------------------------------------------

    def extract_ast(self, oracle_payload: dict[str, Any]) -> dict[str, Any]:
        ast = oracle_payload.get("ast")
        if isinstance(ast, dict):
            return ast
        missing = [k for k in _AST_FIELDS if k not in oracle_payload
                   and k not in ("unit_id_b", "unit_digest_b", "contract_b",
                                 "initial")]
        if missing:
            raise LaneNotSupported(
                "L13 oracle payload cannot rebuild its task AST; missing: %s"
                % ", ".join(sorted(missing)))
        # exact mirror of audit_v3.ast_from_oracle
        return {
            "goal_kind": oracle_payload["goal_kind"],
            "unit_id": oracle_payload["unit_id"],
            "unit_digest": oracle_payload["unit_digest"],
            "unit_id_b": oracle_payload.get("unit_id_b"),
            "unit_digest_b": oracle_payload.get("unit_digest_b"),
            "coordinate": oracle_payload["coordinate"],
            "contract": oracle_payload["contract"],
            "contract_b": oracle_payload.get("contract_b"),
            "initial": oracle_payload.get("initial"),
            "reference": oracle_payload["reference"],
            "guards": oracle_payload["guards"],
            "constraints": oracle_payload["constraints"],
            "witness": oracle_payload["witness"],
        }

    # -- reference / negative helpers -----------------------------------------

    def reference_answer(self, oracle_payload: dict[str, Any]) -> Any:
        ast = self.extract_ast(oracle_payload)
        try:
            from benchmark.plugins.task_family.l13_exp.pipeline import (
                reference_submission,
            )
        except ImportError as exc:  # pragma: no cover - import guard
            raise LaneNotSupported(
                "L13 reference_submission not importable: %s" % exc) from exc
        return reference_submission(ast)

    def wrong_answer(self, oracle_payload: dict[str, Any]) -> Any:
        """Deterministically invalid submission derived from the oracle.

        Every branch is a *genuine* terminal-check rejection (missing key,
        unknown operation/status, schema-invalid object, or the header
        witness's own ``bad_value``) -- never a harness-invented metric.
        """
        ast = self.extract_ast(oracle_payload)
        kind = ast["goal_kind"]
        ref = self.reference_answer(oracle_payload)
        if kind == "operation_selection":
            return {"operation_keys": ["__aob_no_such_operation__"]}
        if kind in ("status_selection",):
            return {"status_codes": ["__aob_no_such_status__"]}
        if kind == "media_type_selection":
            return {"media_types": ["__aob_no_such_media_type__"]}
        if kind == "security_requirement":
            return {"alternatives": []}
        if kind == "header_construction":
            bad = (oracle_payload.get("witness") or {}).get("bad_value")
            return {"header_value": bad if bad is not None else "__aob_invalid__"}
        # param/body/response/migration/mapping: an empty/foreign object fails
        # the schema or exact-key checks.
        key = next(iter(ref), None)
        if key is None:
            return {}
        return {key: {"__aob_invalid__": True} if isinstance(ref[key], dict)
                else ["__aob_invalid__"]}