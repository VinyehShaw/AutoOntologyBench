#!/usr/bin/env bash
# Download the public AutoOntologyBench Core dataset from HuggingFace and run
# the one-click evaluator against a separately mounted private oracle.
#
#   export HF_TOKEN=...
#   export AOB_CORE_ORACLE=/path/to/private/oracle/core_oracle.jsonl
#   bash src/scripts/eval_core_from_hf.sh --provider vllm --model local-model \
#     --base-url http://127.0.0.1:8000/v1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

HF_DATASET_REPO="${AOB_CORE_HF_REPO:-Vinyeh/AutoOntologyBench}"
DATA_DIR="${AOB_CORE_DATA_DIR:-${AOB_DATA_DIR:-$ROOT/../aob-data}}"

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
  "$DATA_DIR/core_worlds/data/core/tasks/tasks.jsonl" \
  "$DATA_DIR/data/core/tasks/tasks.jsonl" \
  "$DATA_DIR/tasks/tasks.jsonl"; do
  if [ -f "$candidate" ]; then
    TASKS="$candidate"
    break
  fi
done
if [ -z "$TASKS" ]; then
  echo "eval_core_from_hf: Core task file not found under $DATA_DIR" >&2
  echo "Set AOB_CORE_HF_REPO=Vinyeh/AutoOntologyBench-Core or AOB_CORE_DATA_DIR if the layout differs." >&2
  exit 2
fi

export EVAL_TASKS="$TASKS"
exec bash "$SCRIPT_DIR/eval_core.sh" "$@"