"""Shared CLI support: strict YAML config loading, errors, small utilities.

Rule (FINAL-BUILD-ORDER-v2 section 5.4): a config file MUST NOT be read with
silent key ignoring. Every mapping is validated against an explicit key spec;
an unknown key is a hard ``ConfigError`` naming the dotted key path.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class FactoryError(Exception):
    """Base class for factory failures. Exit code 1."""


class ConfigError(FactoryError):
    """Config validation failure. Exit code 2 (usage-style)."""


class FetchError(FactoryError):
    """Download/verification failure."""


class QueueError(FactoryError):
    """Persistent queue failure."""


def utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json_sha(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- strict YAML

#: A key-spec maps key name -> expected type (or tuple of types) or a nested
#: key-spec dict for a mapping value. Lists are typed as ``list`` (elements
#: unchecked beyond ``yaml.safe_load`` guarantees); leaf ``None`` means any.
KeySpec = dict[str, Any]

BANK_BUILD_CONFIG_SPEC: KeySpec = {
    "schema": str,
    "bank_id": str,
    "source_lock_file": str,
    "target": {
        "qualified_instances": int,
        "base_goals": int,
        "families": int,
        "quotas_file": str,
    },
    "build": {
        "agent_evaluation": bool,
        "qualification_native_only": bool,
        "generate_and_verify_streaming": bool,
        "resume": bool,
        "stable_seed": int,
        "maintain_immutable_shards": bool,
        "preserve_existing_runs": bool,
    },
    "parallelism": {
        "download_per_host_initial": int,
        "download_per_host_ramp": list,
        "native_cpu_workers": (str, int),
        "gpu_workers": (str, int),
        "llm_in_flight_per_model_initial": int,
        "llm_in_flight_per_model_ramp": list,
        "llm_online_rpm_tpm": (str, dict),
        "queue_backpressure": bool,
        "worker_shard_candidate_size": int,
        "retries": {"max_attempts": int, "exponential_backoff": bool, "honor_retry_after": bool},
        "failure_dead_letter_queue": bool,
        "checkpoints": {"on_shard_commit": bool},
    },
    "outputs": {"root": str, "telemetry": str, "failures": str},
}

BYOK_SOURCE_CONFIG_SPEC: KeySpec = {
    "schema": str,
    "source_id": str,
    "lane": str,
    "official_url": str,
    "download_url_or_api": str,
    "source_type": str,
    "native_engine": str,
    "license": str,
    "license_url": str,
    "redistribution_policy": str,
    "pinned": {"commit_or_release": str, "urls": list, "sha256": str, "upstream_version": str},
    "plugins": {"source": str, "task_family": str, "verifier": str},
}

BYOK_BANK_CONFIG_SPEC: KeySpec = {
    "schema": str,
    "bank_id": str,
    "owner": str,
    "license": str,
    "source_config": str,
    "generation": {
        "families": list,
        "candidate_count": int,
        "seed": int,
        "regime": str,
        "semantic_profile": str,
    },
    "output": {"public_dir": str, "private_dir": str},
}


def _type_name(spec: Any) -> str:
    if isinstance(spec, dict):
        return "mapping"
    if isinstance(spec, tuple):
        return "|".join(t.__name__ for t in spec)
    return spec.__name__


def validate_keys(node: Any, spec: KeySpec, path: str) -> None:
    """Reject unknown keys, wrong types and missing required keys."""
    if not isinstance(node, dict):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(node).__name__}")
    for key in node:
        if key not in spec:
            known = ", ".join(sorted(spec))
            raise ConfigError(f"{path + '.' if path else ''}{key}: unknown key (known: {known})")
    for key, sub in spec.items():
        dotted = f"{path}.{key}" if path else key
        if key not in node:
            raise ConfigError(f"{dotted}: missing required key")
        value = node[key]
        if isinstance(sub, dict):
            validate_keys(value, sub, dotted)
        elif not isinstance(value, sub):
            raise ConfigError(f"{dotted}: expected {_type_name(sub)}, got {type(value).__name__}")


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    data: dict[str, Any]

    def sha256(self) -> str:
        return canonical_json_sha(self.data)


def load_yaml_strict(path: Path, spec: KeySpec) -> LoadedConfig:
    if not path.is_file():
        raise ConfigError(f"config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: YAML parse error: {exc}") from exc
    validate_keys(data, spec, "")
    return LoadedConfig(path=path, data=data)


def resolve_repo_path(p: str | os.PathLike[str]) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (REPO_ROOT / path)


def eprint(msg: str) -> None:
    sys.stderr.write(msg.rstrip("\n") + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
