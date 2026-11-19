#!/usr/bin/env bash
# Download the public AutoOntologyBench task set from HuggingFace and run the one-click
# evaluator against a separately mounted private oracle.
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SRC_ROOT/.." && pwd)"
cd "$REPO_ROOT"

HF_DATASET_REPO="${HF_DATASET_REPO:-Vinyeh/AutoOntologyBench}"
DATA_DIR="${AOB_DATA_DIR:-$REPO_ROOT/../aob-data}"

if command -v hf >/dev/null 2>&1; then
  hf download "$HF_DATASET_REPO" --repo-type dataset --local-dir "$DATA_DIR"
else
  python3 - "$HF_DATASET_REPO" "$DATA_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dest)
PY
fi

TASKS=""
for candidate in \
  "$DATA_DIR/tasks/qualified_tasks_test.jsonl" \
  "$REPO_ROOT/AutoOntologyBench/tasks/qualified_tasks_test.jsonl"; do
  if [ -f "$candidate" ]; then
    TASKS="$candidate"
    break
  fi
done
if [ -z "$TASKS" ]; then
  echo "eval_from_hf: task file not found after download under $DATA_DIR" >&2
  exit 2
fi

if [ -z "${EVAL_ORACLE_ROOTS:-}" ]; then
  echo "eval_from_hf: EVAL_ORACLE_ROOTS is required." >&2
  echo "Point it at the private oracle bundle, e.g.:" >&2
  echo "  export EVAL_ORACLE_ROOTS=$REPO_ROOT/../release_oracle/staging_20k:$REPO_ROOT/../release_oracle/staging" >&2
  exit 3
fi

export EVAL_TASKS="$TASKS"
exec bash "$SRC_ROOT/scripts/eval.sh" "$@"