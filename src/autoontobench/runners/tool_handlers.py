"""The eleven tool implementations (owner: W6). Mixin for ``ToolSession``;
separated to keep each module reviewable. All state lives on the session.

Hard rules (Review-5B §5): native truth stays outside the ontology tool;
``validate_artifact`` is syntax-only and leak-closed (never reveals gold,
native rules, or hidden witnesses); observations carry no ``private_*`` keys.
``model_observation`` fields: ``observation`` / ``budgets_remaining`` /
``last_feedback`` / ``last_tool_result`` (W12: every executed tool result,
echoed on the next turn) / ``submitted``.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from autoontobench.contracts.models import assert_no_private_keys, public_projection
from autoontobench.core.ontology_consumer import OntologyRejected
from autoontobench.core.policy_hooks import PolicyViolation
from autoontobench.runners.ontology_bundle import (
    build_bundle_bytes,
    bundle_content_hash,
)
from autoontobench.runners.serialization import project_for_ontology

_MAX_DOC_BYTES = 262144
_MAX_VIOLATION_LINES = 10


class ToolHandlers:
    # -- artifact plumbing (shared by edit/validate/check/seal) -----------------

    def _current_o_sha(self) -> str | None:
        if self._forced_artifact_bytes is not None:
            return bundle_content_hash(self._forced_artifact_bytes)
        if self._files and self._ontology_profile is not None:
            return bundle_content_hash(self._prepared_bundle())
        return self._sealed_sha

    def _stamp_manifest(self) -> None:
        """Recompute the erule ``contract_hash`` field of the agent's manifest.

        Disclosed harness-side bookkeeping: the contract hash is provenance
        metadata over the agent's own files, recomputed here so agents do not
        hand-compute hashes after each edit. Rule content is untouched.
        """
        raw = self._files.get("manifest.json")
        if raw is None:
            return
        try:
            manifest = json.loads(raw)
        except json.JSONDecodeError:
            return  # the validator reports the syntax error verbatim
        if not isinstance(manifest, dict):
            return
        from erule.contract.validator import compute_contract_hash

        manifest = dict(manifest)
        manifest["contract_hash"] = compute_contract_hash(
            ontology_bytes=self._files.get("ontology.ttl", "").encode("utf-8"),
            shapes_bytes=self._files.get("shapes.ttl", "").encode("utf-8"),
            mappings_bytes=(
                self._files["mappings.json"].encode("utf-8")
                if "mappings.json" in self._files
                else None
            ),
            manifest=manifest,
        )
        self._files["manifest.json"] = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False
        )

    def _prepared_bundle(self) -> bytes:
        if self._ontology_profile is None:
            raise PolicyViolation("NO_PROFILE", "this arm has no ontology profile")
        self._stamp_manifest()
        return build_bundle_bytes(self._files, self._ontology_profile)

    def _load_working_artifact(self) -> Any:
        if self._ontology is None or self._ontology_profile is None:
            raise PolicyViolation("NO_ONTOLOGY_TOOL", "this arm carries no ontology tool")
        bundle = (
            self._forced_artifact_bytes
            if self._forced_artifact_bytes is not None
            else self._prepared_bundle()
        )
        key = bundle_content_hash(bundle)
        if key not in self._ontology_cache:
            self._ontology_cache[key] = self._ontology.load(bundle, self._ontology_profile)
        return self._ontology_cache[key]

    def _observe_public(self) -> dict[str, Any]:
        observation = self._domain.observe(self._state, self._rights).payload
        assert_no_private_keys(observation)
        return observation

    def model_observation(self) -> dict[str, Any]:
        observation = {
            "observation": self._observe_public(),
            "budgets_remaining": self._meter.remaining(),
            "last_feedback": self._last_feedback,
            "last_tool_result": self._last_tool_result,
            "submitted": self._submitted,
        }
        assert_no_private_keys(observation)
        return observation

    # -- tool implementations -----------------------------------------------------

    def _require_args(self, args: Mapping[str, Any], *names: str) -> None:
        missing = [n for n in names if n not in args]
        if missing:
            raise PolicyViolation("ARGS_MISSING", "missing args %s" % missing)

    def _tool_inspect_schema(self, args: Mapping[str, Any]) -> dict[str, Any]:
        projection = public_projection(self._spec)
        return {
            "task_id": projection["task_id"],
            "domain": projection["domain"],
            "track": projection["track"],
            "task_family": projection["task_family"],
            "ontology_contract": projection["ontology_contract"],
            "allowed_tools": projection["allowed_tools"],
            "budgets": projection["budgets"],
            "user_goal": projection["user_goal"],
        }

    def _tool_inspect_state(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {"observation": self._observe_public()}

    def _tool_read_public_doc(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_args(args, "name")
        if self._public_root is None:
            raise PolicyViolation("NO_PUBLIC_ROOT", "no public document root for this run")
        name = str(args["name"])
        candidate = (self._public_root / name).resolve()
        root = self._public_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise PolicyViolation("PATH_ESCAPE", "document path escapes the public root")
        if "private" in candidate.parts:
            raise PolicyViolation("PATH_ESCAPE", "private paths are not readable")
        if not candidate.is_file():
            raise PolicyViolation("DOC_NOT_FOUND", "no such public document: %r" % name)
        data = candidate.read_bytes()[: _MAX_DOC_BYTES + 1]
        if len(data) > _MAX_DOC_BYTES:
            raise PolicyViolation("DOC_TOO_LARGE", "public document exceeds byte cap")
        return {"name": name, "content": data.decode("utf-8", "replace")}

    def _tool_probe_native(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_args(args, "action")
        action = args["action"]
        if not isinstance(action, Mapping):
            raise PolicyViolation("ACTION_MALFORMED", "probe action must be an object")
        self._allowlist.check(action)
        self._meter.charge("probe")
        decision = self._domain.legal_action(self._state, dict(action))
        return {
            "probe": "native_legality",
            "legal": decision.legal,
            "detail": decision.detail,
            "probes_remaining": self._meter.remaining()["oracle_probes"],
        }

    def _tool_edit_artifact(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if not self._artifact_editable:
            raise PolicyViolation("ARTIFACT_READONLY", "this arm's artifact is not editable")
        if self._sealed_sha is not None:
            raise PolicyViolation("ARTIFACT_SEALED", "artifact is sealed; edits are closed")
        self._require_args(args, "files")
        files = args["files"]
        self._file_policy.check(files)
        self._files.update(files)
        return {"edited": sorted(files), "o_sha": self._current_o_sha()}

    def _tool_validate_artifact(self, args: Mapping[str, Any]) -> dict[str, Any]:
        try:
            self._load_working_artifact()
        except OntologyRejected as exc:
            return {"valid": False, "reasons": exc.reasons[:_MAX_VIOLATION_LINES]}
        return {"valid": True, "reasons": []}

    def _tool_check_action_with_ontology(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_args(args, "action")
        action = args["action"]
        if not isinstance(action, Mapping):
            raise PolicyViolation("ACTION_MALFORMED", "action must be an object")
        artifact = self._load_working_artifact()
        # one-directional bridge (serialization.py): adapted facts reach the O opinion only, never native truth
        facts_state, facts_action, serialization = project_for_ontology(
            self._domain, self._observe_public(), dict(action)
        )
        prediction = self._ontology.query_or_check(  # type: ignore[union-attr]
            artifact, facts_state, {"action": facts_action}
        )
        detail = prediction.detail
        return {
            "prediction": prediction.prediction,
            "status": detail.get("status"),
            "serialization": serialization,
            "o_sha": detail.get("o_sha"),
            "violations": detail.get("violations", [])[:_MAX_VIOLATION_LINES],
            "unknowns": detail.get("unknowns", [])[:_MAX_VIOLATION_LINES],
            "note": "ontology opinion only; native truth is decided by perform_action",
        }

    def _tool_perform_action(self, args: Mapping[str, Any]) -> dict[str, Any]:
        self._require_args(args, "action")
        action = args["action"]
        if not isinstance(action, Mapping):
            raise PolicyViolation("ACTION_MALFORMED", "action must be an object")
        self._meter.charge("action")
        try:
            self._allowlist.check(action)
        except PolicyViolation:
            self.unsafe_action_count += 1
            raise
        decision = self._domain.legal_action(self._state, dict(action))
        if decision.legal is not True:
            if decision.legal is False:
                self.unsafe_action_count += 1
            feedback = {"applied": False, "legal": decision.legal, "detail": decision.detail}
            self._last_feedback = feedback
            return feedback
        outcome = self._domain.step(self._state, dict(action))
        self._state = outcome.state
        feedback = {"applied": True, "legal": True, "detail": outcome.feedback}
        self._last_feedback = feedback
        return feedback

    def _tool_get_feedback(self, args: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "last_feedback": self._last_feedback,
            "budgets_remaining": self._meter.remaining(),
            "unsafe_action_count": self.unsafe_action_count,
            "submitted": self._submitted,
        }

    def _tool_seal_artifact(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if self._sealed_sha is not None:
            raise PolicyViolation("ALREADY_SEALED", "seal_artifact is a single snapshot")
        artifact = self._load_working_artifact()
        self._sealed_sha = self._ontology.seal(artifact)  # type: ignore[union-attr]
        return {"sealed": True, "o_sha": self._sealed_sha}

    def _tool_submit_final(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if self._submitted:
            raise PolicyViolation("ALREADY_SUBMITTED", "submit_final is terminal")
        self._submitted = True
        self._final_answer = args.get("answer")
        assert_no_private_keys(self._final_answer)
        return {"submitted": True}
