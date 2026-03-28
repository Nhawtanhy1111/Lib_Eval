#!/bin/bash

DATA_DIR="/media/minhln14/D/nhatanh_test/LibEvolutionEval-main/data/eval_examples/torch_hard_introduced_or_deprecated"
RESULTS_DIR="results"
MODEL="mistral"
BASE_MODEL="mistral"
TASK="eval_examples"
SOURCE="torch_hard_introduced_or_deprecated"

for file in $DATA_DIR/v_*.jsonl; do
    version=$(basename "$file" .jsonl)
    echo "Running generation for $version"
    python evaluate.py --model $MODEL --base_model $BASE_MODEL --mode generate --task $TASK --source $SOURCE --api_version $version --task_version $version --setting_desc_variable $RESULTS_DIR

    echo "Running scoring for $version"
    python evaluate.py --model $MODEL --base_model $BASE_MODEL --mode score --task $TASK --source $SOURCE --api_version $version --task_version $version --setting_desc_variable $RESULTS_DIR
done