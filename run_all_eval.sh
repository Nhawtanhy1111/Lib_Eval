#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/minhln14/D/nhatanh_test/LibEvolutionEval-main"
cd "$ROOT"

MODEL="mistral"
BASE_MODEL="mistral"
TASK="eval_examples"
SETTING="results_smoke"

USE_RETRIEVAL=1
RETRIEVAL_TYPE="codesage"
NUM_TO_RETRIEVE=3

LOG_ROOT="$ROOT/run_logs/$SETTING"
mkdir -p "$LOG_ROOT"

build_extra_args() {
  if [ "$USE_RETRIEVAL" -eq 1 ]; then
    echo "--use_retrieval --retrieval_type $RETRIEVAL_TYPE --num_to_retrieve $NUM_TO_RETRIEVE"
  else
    echo ""
  fi
}

get_retrieval_dir() {
  if [ "$USE_RETRIEVAL" -eq 1 ]; then
    echo "${RETRIEVAL_TYPE}_num_ret_${NUM_TO_RETRIEVE}_conf_thresh_0.0"
  else
    echo "no_augmentation"
  fi
}

summarize_folder() {
  local result_dir="$1"

  python - "$result_dir" <<'PY'
import os
import sys
import json
import glob

result_dir = sys.argv[1]
metrics = ["em", "es", "id_em", "id_precision", "id_recall", "id_f1"]

files = sorted(
    fp for fp in glob.glob(os.path.join(result_dir, "*_results.jsonl"))
    if not fp.endswith("_detailed_results.jsonl")
)

if not files:
    print(f"No summary result files in {result_dir}")
    sys.exit(0)

total = 0
sums = {k: 0.0 for k in metrics}

for fp in files:
    with open(fp, "r") as f:
        d = json.load(f)

    n = d.get("total", 0)
    total += n
    for k in metrics:
        sums[k] += d.get(k, 0.0) * n

if total == 0:
    summary = {k: 0.0 for k in metrics}
    summary["total"] = 0
else:
    summary = {k: sums[k] / total for k in metrics}
    summary["total"] = total

out_path = os.path.join(result_dir, "folder_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print(f"Saved: {out_path}")
PY
}

EXTRA_ARGS="$(build_extra_args)"
RETRIEVAL_DIR="$(get_retrieval_dir)"

echo "Running all files per folder..."
echo "Logs: $LOG_ROOT"

for source_dir in "$ROOT/data/eval_examples"/*; do
  [ -d "$source_dir" ] || continue
  source="$(basename "$source_dir")"

  echo "===== SOURCE: $source ====="
  mkdir -p "$LOG_ROOT/$source"

  found_any=0

  for f in "$source_dir"/v_*.jsonl; do
    [ -f "$f" ] || continue
    found_any=1

    version="$(basename "$f" .jsonl)"
    log_file="$LOG_ROOT/$source/${version}.log"

    echo ">>> Running $source / $version"
    {
      echo "===== $(date) ====="
      echo "SOURCE=$source VERSION=$version"

      python evaluate.py \
        --model "$MODEL" \
        --base_model "$BASE_MODEL" \
        --mode generate \
        --task "$TASK" \
        --source "$source" \
        --api_version "$version" \
        --task_version "$version" \
        --setting_desc_variable "$SETTING" \
        $EXTRA_ARGS

      python evaluate.py \
        --model "$MODEL" \
        --base_model "$BASE_MODEL" \
        --mode score \
        --task "$TASK" \
        --source "$source" \
        --api_version "$version" \
        --task_version "$version" \
        --setting_desc_variable "$SETTING" \
        $EXTRA_ARGS
    } > "$log_file" 2>&1

    echo "Saved log to $log_file"
  done

  if [ "$found_any" -eq 0 ]; then
    echo "No valid v_*.jsonl file found for $source"
    continue
  fi

  RESULT_BASE="$ROOT/$SETTING/$RETRIEVAL_DIR/eval_examples/$MODEL/$source"
  summarize_folder "$RESULT_BASE"
done

echo "All-files-per-folder runs finished."