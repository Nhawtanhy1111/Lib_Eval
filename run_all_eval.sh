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

# Versions to process
VERSIONS=("v_1_1_0" "v_1_2_0" "v_1_4_0" "v_1_6_0" "v_1_8_0" "v_1_10_0" "v_2_0_0")

LOG_ROOT="$ROOT/run_logs/$SETTING"
mkdir -p "$LOG_ROOT"

# Ensure 'bc' is installed for average calculation
if ! command -v bc &> /dev/null; then
    echo "Error: 'bc' is not installed. Install it with: sudo apt-get install bc"
    exit 1
fi

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
echo "Starting Enriched Evaluation Pipeline for ${#VERSIONS[@]} versions..."

for V in "${VERSIONS[@]}"; do
    # Check if required data files exist
    TASK_FILE="./data/eval_examples/$SOURCE/${V}_enriched.jsonl"
    DOC_FILE="./data/package_apis/torch/${V}.jsonl"

    if [[ ! -f "$TASK_FILE" || ! -f "$DOC_FILE" ]]; then
        echo "[-] Skipping $V: Missing data files."
        continue
    fi

    echo "========================================"
    echo ">>> PROCESSING VERSION: $V"
    echo "========================================"
    
    LOG_FILE="$LOG_ROOT/${V}_enriched.log"
    
    {
      echo "--- STARTING GENERATION: $(date) ---"
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
          
    } > "$LOG_FILE" 2>&1 || echo "Warning: Error processing $V. Check $LOG_FILE"
    
    echo "Finished $V. Logs saved to $LOG_FILE"
done

# --- FINAL SUMMARY TABLE (Paper Style) ---
echo ""
echo "============================================================"
echo "           LIBEVOLUTIONEVAL: ENRICHED SUMMARY              "
echo "============================================================"
printf "| %-12s | %-10s | %-10s | %-10s |\n" "Version" "EM (%)" "ES" "ID F1"
echo "|--------------|------------|------------|------------|"

TOTAL_EM=0
TOTAL_ES=0
TOTAL_F1=0
VALID_COUNT=0

for V in "${VERSIONS[@]}"; do
    FILE_NAME="task_${V}_enriched_doc_${V}_results.jsonl"
    # Recursive search ensures we find the file regardless of folder names
    SCORE_FILE=$(find "$ROOT/$SETTING" -name "$FILE_NAME" | head -n 1)
    
    if [[ -n "$SCORE_FILE" && -f "$SCORE_FILE" ]]; then
        METRICS=$(python3 -c "
import json
with open('$SCORE_FILE') as f:
    d = json.load(f)
    print(f\"{d.get('em', 0.0):.2f}|{d.get('es', 0.0):.2f}|{d.get('id_f1', 0.0):.2f}\")
")
        EM=$(echo "$METRICS" | cut -d'|' -f1)
        ES=$(echo "$METRICS" | cut -d'|' -f2)
        F1=$(echo "$METRICS" | cut -d'|' -f3)

        printf "| %-12s | %-10s | %-10s | %-10s |\n" "$V" "$EM" "$ES" "$F1"
        
        TOTAL_EM=$(echo "$TOTAL_EM + $EM" | bc)
        TOTAL_ES=$(echo "$TOTAL_ES + $ES" | bc)
        TOTAL_F1=$(echo "$TOTAL_F1 + $F1" | bc)
        VALID_COUNT=$((VALID_COUNT + 1))
    else
        printf "| %-12s | %-10s | %-10s | %-10s |\n" "$V" "MISSING" "MISSING" "MISSING"
    fi
done

if [ "$VALID_COUNT" -gt 0 ]; then
    AVG_EM=$(echo "scale=2; $TOTAL_EM / $VALID_COUNT" | bc)
    AVG_ES=$(echo "scale=2; $TOTAL_ES / $VALID_COUNT" | bc)
    AVG_F1=$(echo "scale=2; $TOTAL_F1 / $VALID_COUNT" | bc)
    
    echo "|--------------|------------|------------|------------|"
    printf "| %-12s | %-10s | %-10s | %-10s |\n" "AVERAGE" "$AVG_EM" "$AVG_ES" "$AVG_F1"
fi

echo "============================================================"
echo "All evaluation runs finished."