#!/usr/bin/env bash
set -euo pipefail

# Project Root
ROOT="/root/Lib_Val"
cd "$ROOT"

# Basic Configuration
MODEL="mistral"           # Ollama model tag
TASK="eval_examples"
SOURCE="enriched_torch"   # The folder inside data/eval_examples/
SETTING="results_enriched"

# RAG Configuration
USE_RETRIEVAL=1
RETRIEVAL_TYPE="codesage"
NUM_TO_RETRIEVE=3

# Directory where evaluate.py saves results (matches the dynamic naming in your python script)
RETRIEVAL_DIR="${RETRIEVAL_TYPE}_num_ret_${NUM_TO_RETRIEVE}_conf_thresh_0.0"

# Versions to process
VERSIONS=("v_1_6_0" "v_1_10_0" "v_2_0_0")

LOG_ROOT="$ROOT/run_logs/$SETTING"
mkdir -p "$LOG_ROOT"

# --- HELPER: Summarize folder results ---
summarize_folder() {
  local result_dir="$1"
  python3 - "$result_dir" <<'PY'
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

summary = {k: (sums[k] / total if total > 0 else 0.0) for k in metrics}
summary["total"] = total

out_path = os.path.join(result_dir, "folder_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
PY
}

# --- MAIN LOOP ---
echo "Starting Enriched Evaluation Pipeline..."

for V in "${VERSIONS[@]}"; do
    echo "========================================"
    echo ">>> PROCESSING VERSION: $V"
    echo "========================================"
    
    LOG_FILE="$LOG_ROOT/${V}_enriched.log"
    
    {
      echo "--- STARTING GENERATION: $(date) ---"
      # api_version = base version for doc lookup
      # task_version = enriched version for prompt lookup
      python evaluate.py \
          --model "$MODEL" \
          --use_ollama \
          --mode generate \
          --task "$TASK" \
          --source "$SOURCE" \
          --api_version "$V" \
          --task_version "${V}_enriched" \
          --use_retrieval \
          --use_api_description \
          --retrieval_type "$RETRIEVAL_TYPE" \
          --num_to_retrieve "$NUM_TO_RETRIEVE" \
          --setting_desc_variable "$SETTING"

      echo "--- STARTING SCORING: $(date) ---"
      python evaluate.py \
          --model "$MODEL" \
          --mode score \
          --task "$TASK" \
          --source "$SOURCE" \
          --api_version "$V" \
          --task_version "${V}_enriched" \
          --use_retrieval \
          --use_api_description \
          --retrieval_type "$RETRIEVAL_TYPE" \
          --num_to_retrieve "$NUM_TO_RETRIEVE" \
          --setting_desc_variable "$SETTING"
          
    } > "$LOG_FILE" 2>&1
    
    echo "Finished $V. Logs saved to $LOG_FILE"
done

# Run folder summarization for the final output directory
RESULT_BASE_PATH="$ROOT/$SETTING/$RETRIEVAL_DIR/$TASK/$MODEL/$SOURCE"
if [ -d "$RESULT_BASE_PATH" ]; then
    summarize_folder "$RESULT_BASE_PATH"
fi

# --- FINAL SUMMARY TABLE (Paper Style) ---
echo ""
echo "============================================================"
echo "           LIBEVOLUTIONEVAL: ENRICHED SUMMARY              "
echo "============================================================"
printf "| %-12s | %-10s | %-10s | %-10s |\n" "Version" "EM (%)" "ES" "ID F1"
echo "|--------------|------------|------------|------------|"

for V in "${VERSIONS[@]}"; do
    # Locate the specific score file
    FILE_NAME="task_${V}_enriched_doc_${V}_results.jsonl"
    SCORE_FILE="$RESULT_BASE_PATH/$FILE_NAME"
    
    if [ -f "$SCORE_FILE" ]; then
        METRICS=$(python3 -c "
import json
with open('$SCORE_FILE') as f:
    d = json.load(f)
    print(f\"{d.get('em', 0.0):.2f}|{d.get('es', 0.0):.2f}|{d.get('id_f1', 0.0):.2f}\")
")
        EM=$(echo $METRICS | cut -d'|' -f1)
        ES=$(echo $METRICS | cut -d'|' -f2)
        F1=$(echo $METRICS | cut -d'|' -f3)
        printf "| %-12s | %-10s | %-10s | %-10s |\n" "$V" "$EM" "$ES" "$F1"
    else
        printf "| %-12s | %-10s | %-10s | %-10s |\n" "$V" "ERR" "ERR" "ERR"
    fi
done
echo "============================================================"
echo "All evaluation runs finished."