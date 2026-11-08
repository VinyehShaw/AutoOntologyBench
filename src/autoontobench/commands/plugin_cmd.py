"""``python -m autoontobench plugin scaffold --name <name>`` (S10).

Generates a minimal but runnable BYOK plugin trio (source / task_family /
verifier) plus a source config and bank config. The generated verifier is a
real deterministic local check (sha256-of-answer equality) so the scaffold
passes ``bank generate`` smoke runs out of the box; users replace the marked
seams with their native engine.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

from autoontobench.factory import cli_support as cs

_SOURCE = '''"""BYOK source plugin: {name}. Fetches one pinned file into the content cache."""
from __future__ import annotations

from benchmark.schema.plugin_protocols import (
    FetchReceipt, SourceDescriptor, SourceRelease, SourceSnapshot, SourceUnit,
    VerificationReport)


class {cls}Source:
    source_id = "{name}"

    def describe(self) -> SourceDescriptor:
        return SourceDescriptor(
            source_id=self.source_id,
            official_url="TODO-official-url",
            download_url_or_api="TODO-download-url",
            source_type="single-file",
            upstream_version="1.0.0",
            commit_or_release="PINNED",
            retrieved_at="",
            sha256="",
            native_engine="local",
            license_url="TODO-license-url",
            redistribution_policy="fetch_recipe_only")

    def discover_releases(self) -> list[SourceRelease]:
        return []

    def fetch(self, release: SourceRelease, *, resume: bool, cache_dir) -> FetchReceipt:
        raise NotImplementedError("wire to autoontobench.factory.cache.download")

    def verify_snapshot(self, receipt: FetchReceipt) -> VerificationReport:
        return VerificationReport(ok=True)

    def inventory(self, snapshot: SourceSnapshot):
        return iter(())


PLUGIN = {cls}Source()
'''

_FAMILY = '''"""BYOK task-family plugin: {name}."""
from __future__ import annotations

from benchmark.schema.plugin_protocols import CandidateTaskAST, GroundedTask


class {cls}Family:
    def supported_profiles(self) -> list[str]:
        return ["local-deterministic/1"]

    def propose(self, snapshot, *, family: str, seed: int, constraints: dict):
        count = int(constraints.get("count", 10))
        for i in range(count):
            answer = f"{{family}}-answer-{{seed}}-{{i}}"
            yield CandidateTaskAST(
                lane_id="BYOK", family_id=family, base_goal_id=f"{{family}}-g{{i}}",
                instance_id=f"{{family}}-{{seed}}-{{i}}",
                regime=str(constraints.get("regime", "application")),
                semantic_profile=str(constraints.get("semantic_profile", "local-deterministic/1")),
                ast={{"goal": "echo", "answer_sha256": _sha(answer), "seed": seed + i}},
                source_unit_ids=())

    def ground(self, candidate: CandidateTaskAST) -> GroundedTask:
        return GroundedTask(
            ast=candidate,
            public={{"user_goal": "Reproduce the pinned local fact for " + candidate.instance_id,
                     "allowed_tools": [], "budget": {{"probes": 1, "actions": 1}}}},
            oracle_ref="oracle:{{}}".format(candidate.instance_id),
            native_evidence={{"answer_sha256": candidate.ast["answer_sha256"]}})


def _sha(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


PLUGIN = {cls}Family()
'''

_VERIFIER = '''"""BYOK native verifier plugin: {name}."""
from __future__ import annotations

import hashlib

from benchmark.schema.plugin_protocols import NativeVerdict, ProofOrCounterexample, SemanticWitness


class {cls}Verifier:
    def prove_exists(self, grounded_task) -> ProofOrCounterexample:
        ok = bool(grounded_task.native_evidence.get("answer_sha256"))
        return ProofOrCounterexample(exists=ok, evidence={{"check": "sha256-present"}},
                                     reason="" if ok else "missing answer hash")

    def verify_terminal(self, task, submitted_state_or_answer) -> NativeVerdict:
        digest = hashlib.sha256(str(submitted_state_or_answer).encode()).hexdigest()
        return NativeVerdict(valid=digest == task.native_evidence.get("answer_sha256"),
                             detail={{"submitted_sha256": digest}})

    def certify_semantic_relevance(self, task) -> SemanticWitness:
        return SemanticWitness(ok=bool(task.public.get("user_goal")),
                               witness={{"kind": "public-goal-present"}})


PLUGIN = {cls}Verifier()
'''

_SOURCE_CFG = """schema: aob-byok-source/1
source_id: {name}
lane: BYOK
official_url: TODO-official-url
download_url_or_api: TODO-download-url
source_type: single-file
native_engine: local
license: TODO
license_url: TODO-license-url
redistribution_policy: fetch_recipe_only
pinned:
  commit_or_release: v1.0.0
  urls: [TODO-download-url]
  sha256: {zeros}
  upstream_version: 1.0.0
plugins:
  source: {name}_source.py
  task_family: {name}_family.py
  verifier: {name}_verifier.py
"""

_BANK_CFG = """schema: aob-byok-bank/1
bank_id: byok-{name}
owner: TODO-owner
license: TODO
source_config: source.yaml
generation:
  families: [default]
  candidate_count: 10
  seed: 1
  regime: application
  semantic_profile: local-deterministic/1
output:
  public_dir: ./{name}_bank
  private_dir: ./{name}_bank/private_oracle
"""


def _class_name(name: str) -> str:
    return "".join(part.capitalize() for part in name.replace("-", "_").split("_") if part) or "Byok"


def cmd_scaffold(args: Namespace) -> int:
    name = args.name
    if not name or not all(c.isalnum() or c in "-_" for c in name):
        cs.eprint(f"[plugin scaffold] invalid name {name!r}")
        return 2
    out = Path(args.out) if args.out else Path.cwd() / f"aob_plugin_{name}"
    if out.exists() and any(out.iterdir()):
        cs.eprint(f"[plugin scaffold] {out} not empty")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    cls = _class_name(name)
    files = {
        f"{name}_source.py": _SOURCE.format(name=name, cls=cls),
        f"{name}_family.py": _FAMILY.format(name=name, cls=cls),
        f"{name}_verifier.py": _VERIFIER.format(name=name, cls=cls),
        "source.yaml": _SOURCE_CFG.format(name=name, zeros="0" * 64),
        "bank.yaml": _BANK_CFG.format(name=name),
        "README.md": (f"# BYOK plugin `{name}`\n\n"
                      "1. Fill source.yaml pinned urls/sha256/commit_or_release and the\n"
                      f"   fetch() seam in {name}_source.py (use "
                      "autoontobench.factory.cache.ContentAddressedCache.download).\n"
                      "2. Replace the family proposer and verifier with your native engine.\n"
                      "3. python -m autoontobench source add --config source.yaml\n"
                      "4. python -m autoontobench bank generate --config bank.yaml --output ./out\n"),
    }
    for rel, content in files.items():
        if content is not None:
            (out / rel).write_text(content, encoding="utf-8")
    print(f"[plugin scaffold] {out}: {sorted(files)}")
    return 0


def run(args: Namespace) -> int:
    if getattr(args, "plugin_command", None) != "scaffold":
        cs.eprint("[plugin] subcommand required: scaffold")
        return 2
    return cmd_scaffold(args)
