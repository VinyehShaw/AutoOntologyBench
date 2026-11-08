"""RAG arm corpus + retriever (owner: W13).

The RAG arm keeps the DIRECT tool surface (``read_public_doc`` IS the
retrieval channel); what differs is the mounted corpus. Three chunk classes,
all public-budget (see the parity policy in ``agent_eval/REPORT.md`` §W13):

1. ``base``   — the two W12 public docs (``ARTIFACT_FORMAT.md``,
   ``TOOL_GUIDE.md``), shared verbatim with every arm's mounted root;
2. ``generic`` — authored constraint-TYPE documentation
   (``agent_eval/rag_corpus/*.md``), derived only from the licenseable
   generic material of the pinned, Apache-2.0 ``gatekeeper-library`` sources
   (what each constraint TYPE checks and its parameter schema). NO parameter
   VALUES, NO gold values (``HYGIENE_BANS`` is enforced by
   ``tests/agent/test_rag_corpus.py``): which prefix/label/range a task
   demands is never stated;
3. ``task``   — the task's OWN public evidence (the initial-state JSON,
   verbatim content of ``public_evidence_ref``), chunked per task.

Retrieval is a deterministic, offline, lexical scorer (term frequency x idf
over whitespace/punctuation tokens; ties broken by chunk name). No
embeddings, no network, no model calls. ``retrieve(query, chunks)`` returns
the ranked chunk list; the same function produces the INDEX.md ranking shown
to the agent, so the documented ranking and the served one cannot drift.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_EVAL_ROOT = REPO_ROOT / "data" / "erule" / "review5" / "agent_eval"
PUBLIC_DOCS_ROOT = AGENT_EVAL_ROOT / "public_docs"
GENERIC_CORPUS_ROOT = AGENT_EVAL_ROOT / "rag_corpus"

TOP_K = 4

#: Substrings that must never appear in AUTHORED (base/generic) chunks —
#: task/policy-instantiation values (gold or otherwise). The per-task state
#: chunk is exempt: it IS the task's public evidence, included verbatim.
HYGIENE_BANS: tuple[str, ...] = (
    "pizza",
    "openpolicyagent",
    "evilcorp",
    "server:v1",
    "gatekeeper:v3",
    "nginx",
)

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable document chunk."""

    name: str  # file name inside the mounted root
    title: str  # one-line summary (INDEX.md shows this)
    origin: str  # "base" | "generic" | "task"
    text: str


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def retrieve(query: str, chunks: Iterable[Chunk], *, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Rank ``chunks`` against ``query`` (deterministic tf-idf, no ties)."""
    chunk_list = sorted(chunks, key=lambda c: c.name)
    query_terms = _tokens(query)
    doc_tokens = {c.name: _tokens(c.title + "\n" + c.text) for c in chunk_list}
    n_docs = max(1, len(chunk_list))
    df: dict[str, int] = {}
    for tokens in doc_tokens.values():
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    ranked: list[dict[str, Any]] = []
    for chunk in chunk_list:
        tokens = doc_tokens[chunk.name]
        if not tokens:
            continue
        score = 0.0
        for term in query_terms:
            tf = tokens.count(term)
            if tf:
                score += tf * (math.log((n_docs + 1) / (df.get(term, 0) + 1)) + 1.0)
        ranked.append(
            {"name": chunk.name, "title": chunk.title, "origin": chunk.origin,
             "score": round(score, 6)}
        )
    ranked.sort(key=lambda r: (-r["score"], r["name"]))
    for rank, entry in enumerate(ranked):
        entry["rank"] = rank + 1
    return ranked[:top_k]


def base_chunks() -> list[Chunk]:
    """The W12 public docs, shared verbatim with every arm."""
    out: list[Chunk] = []
    titles = {
        "ARTIFACT_FORMAT.md": "The aob-ontology-bundle/1.0 artifact wire format (shacl-rule/1 file set, manifest, mappings, phi)",
        "TOOL_GUIDE.md": "The 11-tool surface, action envelope, feedback channels, path syntax, serialized check vocabulary",
    }
    for path in sorted(PUBLIC_DOCS_ROOT.glob("*.md")):
        out.append(Chunk(path.name, titles.get(path.name, path.name), "base",
                         path.read_text(encoding="utf-8")))
    return out


def generic_chunks() -> list[Chunk]:
    """The authored constraint-type docs (``rag_corpus/*.md``; title = first heading)."""
    out: list[Chunk] = []
    for path in sorted(GENERIC_CORPUS_ROOT.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title = next(
            (line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")),
            path.name,
        )
        out.append(Chunk(path.name, title, "generic", text))
    return out


def task_chunk(task_id: str, initial_state: Mapping[str, Any]) -> Chunk:
    """The task's own public evidence (initial state), verbatim as one chunk."""
    text = (
        "# %s — public task state (initial)\n\n"
        "This is the verbatim public evidence for task `%s`: the initial state "
        "of every object you may inspect (also available live via `inspect_state`).\n\n"
        "```json\n%s\n```\n"
    ) % (task_id, task_id, json.dumps(initial_state, indent=2, sort_keys=True))
    return Chunk("task_state.md", "%s public initial state (verbatim evidence)" % task_id,
                 "task", text)


def build_mounted_root(
    spec: Any,
    initial_state: Mapping[str, Any],
    dest: Path,
) -> dict[str, Any]:
    """Materialize the rag arm's mounted public root for one task.

    Writes base + generic + task chunks plus ``INDEX.md`` (the chunk catalog
    with titles and the deterministic ``retrieve`` ranking for the task's
    user goal). Returns the index payload (also recorded in run evidence).
    """
    task_id, user_goal = spec.task_id, spec.user_goal
    dest.mkdir(parents=True, exist_ok=True)
    chunks = base_chunks() + generic_chunks() + [task_chunk(task_id, initial_state)]
    for chunk in chunks:
        (dest / chunk.name).write_text(chunk.text, encoding="utf-8")
    ranked = retrieve(user_goal, chunks)
    lines = [
        "# INDEX.md — retrievable public corpus for this task (RAG arm)",
        "",
        "Chunks are served verbatim via `read_public_doc {\"name\": ...}`. The ranking",
        "below is the deterministic lexical retriever output for your task goal;",
        "every chunk is readable regardless of rank.",
        "",
        "## Ranking for the task goal",
    ]
    for entry in ranked:
        lines.append(
            "%d. `%s` (score %.3f, %s) — %s"
            % (entry["rank"], entry["name"], entry["score"], entry["origin"], entry["title"])
        )
    lines += ["", "## Full catalog", ""]
    for chunk in chunks:
        lines.append("- `%s` (%s) — %s" % (chunk.name, chunk.origin, chunk.title))
    lines.append("")
    (dest / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "task_id": task_id,
        "chunks": [c.name for c in chunks],
        "ranking": ranked,
        "root": str(dest),
    }


def sync_public_docs_into(dest: Path) -> None:
    """Copy the shared base docs (used when assembling non-rag roots in tests)."""
    for path in sorted(PUBLIC_DOCS_ROOT.glob("*.md")):
        shutil.copyfile(path, dest / path.name)
