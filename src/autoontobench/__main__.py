"""AutoOntologyBench Review-5 CLI (owner: W7 integration, scaffold by orchestrator).

Commands (Implementation §1): audit-sources, qualify, build-tasks, inspect, verify,
run-researchers, run-controls, export-paper.
Common flags: --config --out --limit --seed --split --offline --strict.
Unknown flags fail fast (argparse). `inspect --dry-run` is implemented and strictly
read-only (no network, no artifact writes). `audit-sources`, `build-tasks` and
`verify` are wired to the W5 registry loader and the W2 Gatekeeper native engine
(see autoontobench/commands/). Commands whose owner module is not yet wired exit
with code 2 and a stderr note naming the workstream.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from autoontobench.commands import audit_sources, build_tasks, verify
from autoontobench.commands import bank_cmd, plugin_cmd, source_cmd

COMMANDS = [
    "audit-sources",
    "qualify",
    "build-tasks",
    "inspect",
    "verify",
    "run-researchers",
    "run-controls",
    "export-paper",
]

FACTORY_GROUPS = ("source", "bank", "plugin")


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--split", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--strict", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m autoontobench",
        description="AutoOntologyBench Review-5 CLI (inspect/audit-sources/build-tasks/verify wired; others owned by W5/W6)",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        _common(sp)
        if name == "inspect":
            sp.add_argument("--dry-run", action="store_true")
        if name == "audit-sources":
            sp.add_argument("--registry", type=str, default="sources/registry.d")
            sp.add_argument("--native-test", action="store_true")
        if name == "build-tasks":
            sp.add_argument("--source", type=str, default=None)
        if name == "verify":
            sp.add_argument("--input", type=str, default=None)
        if name == "export-paper":
            sp.add_argument("--input", type=str, default="runs")
            sp.add_argument("--output", type=str, default="results")
    _add_factory_groups(sub)
    return ap


def _add_factory_groups(sub: argparse._SubParsersAction) -> None:
    src = sub.add_parser("source", help="Source Hub: list/fetch/audit/add external sources")
    src_sub = src.add_subparsers(dest="source_command", required=True)
    p = src_sub.add_parser("list", help="list registry sources with lock status")
    p.add_argument("--overlay", action="store_true", help="show BYOK overlay lock")
    p = src_sub.add_parser("fetch", help="fetch sources (pinned, resume-safe)")
    p.add_argument("--all", action="store_true")
    p.add_argument("--source", type=str, default=None)
    p.add_argument("--pinned", action="store_true", help="fetch only lock-pinned revisions")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-sustainable", action="store_true")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overlay", action="store_true")
    p = src_sub.add_parser("audit", help="audit fetch receipts, locks, coverage")
    p.add_argument("--all", action="store_true")
    p.add_argument("--full-index", action="store_true")
    p.add_argument("--out", type=str, default=None)
    p = src_sub.add_parser("add", help="register a BYOK source (validated, no silent ignore)")
    p.add_argument("--config", type=str, required=True)
    src.set_defaults(handler=source_cmd.run)

    bank = sub.add_parser("bank", help="Task bank: preflight/build/status/audit/package/generate/validate")
    bank_sub = bank.add_subparsers(dest="bank_command", required=True)
    for name_ in ("preflight", "build"):
        p = bank_sub.add_parser(name_)
        p.add_argument("--config", type=str, required=True)
        p.add_argument("--dry-run", action="store_true")
        if name_ == "build":
            p.add_argument("--all-lanes", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--max-sustainable", action="store_true")
            p.add_argument("--no-agent-evaluation", action="store_true")
            p.add_argument("--lane", type=str, default=None,
                           help="dispatch one lane's real production entrypoint "
                                "(benchmark/plugins/task_family/<lane>/ run_build/build/run)")
            p.add_argument("--limit", type=int, default=None,
                           help="per-family/total cap passed to the lane entrypoint")
            p.add_argument("--out", type=str, default=None,
                           help="scratch output dir (redirects the lane's STAGING); "
                                "never the lane's real staging")
            p.add_argument("--seed", type=int, default=None,
                           help="build seed (default: config build.stable_seed)")
    p = bank_sub.add_parser("status")
    p.add_argument("--build-id", type=str, required=True)
    p.add_argument("--by-lane", action="store_true")
    p = bank_sub.add_parser("audit")
    p.add_argument("--build-id", type=str, required=True)
    p.add_argument("--strict", action="store_true")
    p = bank_sub.add_parser("package")
    p.add_argument("--build-id", type=str, default=None)
    p.add_argument("--input", type=str, default=None)
    p.add_argument("--release", type=str, required=True,
                   choices=["public", "private", "private-local"])
    p.add_argument("--license-matrix", type=str, default=None,
                   help="override matrix CSV (e.g. the PROPOSED matrix) without "
                        "touching the live source_redistribution_matrix.csv")
    p.add_argument("--dry-run", action="store_true",
                   help="public: report license eligibility counts only, write nothing")
    p = bank_sub.add_parser("generate", help="BYOK: build a user bank from a plugin config")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = bank_sub.add_parser("validate", help="re-run native verification on a bank dir")
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--native-only", action="store_true")
    bank.set_defaults(handler=bank_cmd.run)

    plug = sub.add_parser("plugin", help="Plugin tooling: scaffold")
    plug_sub = plug.add_subparsers(dest="plugin_command", required=True)
    p = plug_sub.add_parser("scaffold", help="generate a runnable BYOK plugin trio")
    p.add_argument("--name", type=str, required=True)
    p.add_argument("--out", type=str, default=None)
    plug.set_defaults(handler=plugin_cmd.run)


# ---------------------------------------------------------------- implemented

def cmd_inspect(args: argparse.Namespace) -> int:
    """Strictly read-only: prints the I/O contract for this repo checkout."""
    root = Path(__file__).resolve().parent.parent
    info = {
        "command": "inspect",
        "dry_run": bool(args.dry_run),
        "config": args.config,
        "repo_root": str(root),
        "package_dir": str(Path(__file__).resolve().parent),
        "io_contract": {
            "inputs": [
                "sources/registry.d/<source_id>.yaml  (pinned source entries)",
                "benchmarks/candidates/**  (unverified task candidates)",
                "configs/*.yaml",
            ],
            "outputs": [
                "benchmarks/qualified/tasks.jsonl",
                "runs/researcher_trials.jsonl (append-only)",
                "results/* (paper-facing, strict mode)",
                "audit/review5/P*_ACCEPTANCE.md",
            ],
        },
        "guarantee": "read-only: no network calls, no artifact mutation, no model spend",
    }
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_export_paper(args: argparse.Namespace) -> int:
    from autoontobench.export.paper import run_cli

    return run_cli(Path(args.input), Path(args.output), strict=bool(args.strict))


HANDLERS: dict[str, object] = {
    "inspect": cmd_inspect,
    "audit-sources": audit_sources.run,
    "build-tasks": build_tasks.run,
    "verify": verify.run,
    "export-paper": cmd_export_paper,
}

# Owner workstreams for not-yet-wired commands (exit 2 until wired):
OWNERS = {
    "qualify": "W5 (Task Factory T5; scripts/qualify_tasks.py)",
    "run-researchers": "W6 (Agent Evaluation; G0-gated for real runs)",
    "run-controls": "W6 (Agent Evaluation; G0-gated for real runs)",
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    factory_handler = getattr(args, "handler", None)
    if callable(factory_handler):
        return int(factory_handler(args))
    handler = HANDLERS.get(args.command)
    if handler is None:
        sys.stderr.write(
            f"[autoontobench] '{args.command}' not wired into the CLI yet. "
            f"Owner: {OWNERS.get(args.command, 'unassigned')}.\n"
        )
        return 2
    return int(handler(args))  # type: ignore[operator]


if __name__ == "__main__":
    raise SystemExit(main())
