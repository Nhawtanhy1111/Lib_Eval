#!/usr/bin/env bash
set -euo pipefail

# Configuration
MODEL="mistral"
SETTING="results_enriched"
VERSIONS=("v_1_6_0" "v_1_10_0" "v_2_0_0")
RETRIEVAL_DIR="codesage_num_ret_3_api_desc_true"

# Ensure logs directory exists
mkdir -p "run_logs/$SETTING"

for V in "${VERSIONS[@]}"; do
    echo "========================================"
    echo ">>> PROCESSING VERSION: $V"
    echo "========================================"

    # 1. GENERATION PHASE
    python evaluate.py \
        --model "$MODEL" \
        --use_ollama \
        --mode generate \
        --task "eval_examples" \
        --source "enriched_torch" \
        --api_version "$V" \
        --task_version "${V}_enriched" \
        --use_retrieval \
        --use_api_description \
        --retrieval_type "codesage" \
        --num_to_retrieve 3 \
        --setting_desc_variable "$SETTING"

    # 2. SCORING PHASE
    python evaluate.py \
        --model "$MODEL" \
        --mode score \
        --task "eval_examples" \
        --source "enriched_torch" \
        --api_version "$V" \
        --task_version "${V}_enriched" \
        --use_retrieval \
        --use_api_description \
        --retrieval_type "codesage" \
        --num_to_retrieve 3 \
        --setting_desc_variable "$SETTING"
done

# --- 3. AGGREGATION BLOCK ---
echo "========================================"
echo "FINAL ENRICHED SUMMARY TABLE"
echo "========================================"
printf "%-15s | %-10s | %-10s | %-10s\n" "Version" "EM (%)" "ES" "ID_F1"
echo "------------------------------------------------------------"

for V in "${VERSIONS[@]}"; do
    SUMMARY_FILE="./$SETTING/$RETRIEVAL_DIR/eval_examples/$MODEL/enriched_torch/task_${V}_enriched_doc_${V}_results.jsonl"
    
    if [ -f "$SUMMARY_FILE" ]; then
        EM=$(python3 -c "import json; print(json.load(open('$SUMMARY_FILE'))['em'])")
        ES=$(python3 -c "import json; print(json.load(open('$SUMMARY_FILE'))['es'])")
        F1=$(python3 -c "import json; print(json.load(open('$SUMMARY_FILE'))['id_f1'])")
        printf "%-15s | %-10s | %-10s | %-10s\n" "$V" "$EM" "$ES" "$F1"
    else
        printf "%-15s | %-10s | %-10s | %-10s\n" "$V" "N/A" "N/A" "N/A"
    fi
done