#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------
# CONFIG
# -----------------------------------------
# Tuples of version:source
version_source_pairs=(
  "v_2_0_0:torch_direct_api"
  "3_0_3:matplotlib_direct_api"
  # "v_2_2_0:torch_indirect_api"
)

models=("starcoder2-3b")
tasks=("eval_examples")
results_dir="results"
logs_dir="logs"

# GPU slots
gpus=(0 1 2 3 4 5 6 7)
num_gpus=${#gpus[@]}

mkdir -p "${logs_dir}"

# -----------------------------------------
# FLAGS
# -----------------------------------------
debug=""
for arg in "$@"; do
  [[ "$arg" == "--debug" ]] && debug="--debug"
done

# -----------------------------------------
# MAIN LOOP
# -----------------------------------------
job_idx=0
for pair in "${version_source_pairs[@]}"; do
  IFS=":" read -r version source <<<"$pair"

  for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
      gpu_id=${gpus[$(( job_idx % num_gpus ))]}

      input_file="./data/${task}/${source}/${version}.jsonl"
      [[ ! -f "$input_file" ]] && {
        echo "[WARN] Missing input: $input_file"
        job_idx=$((job_idx + 1))
        continue
      }

      gen_log="${logs_dir}/generate_${model}_${version}_${task}_${source}.log"
      score_log="${logs_dir}/score_${model}_${version}_${task}_${source}.log"

      (
        echo "[INFO] GPU ${gpu_id} → Version: ${version}, Source: ${source}, Model: ${model}, Task: ${task}"

        CUDA_VISIBLE_DEVICES="${gpu_id}" python3 evaluate.py \
          --task "${task}" \
          --mode generate \
          --model "${model}" \
          --base_model "${model}" \
          --source "${source}" \
          --api_version "${version}" \
          --task_version "${version}" \
          --setting_desc_variable "${results_dir}" \
          ${debug} \
          >"${gen_log}" 2>&1

        CUDA_VISIBLE_DEVICES="${gpu_id}" python3 evaluate.py \
          --task "${task}" \
          --mode score \
          --model "${model}" \
          --base_model "${model}" \
          --source "${source}" \
          --api_version "${version}" \
          --task_version "${version}" \
          --setting_desc_variable "${results_dir}" \
          ${debug} \
          >"${score_log}" 2>&1

        echo "[DONE] ${model} ${version} ${task} ${source} (GPU ${gpu_id})"
      ) &

      if (( (job_idx + 1) % num_gpus == 0 )); then
        wait
      fi

      job_idx=$((job_idx + 1))
    done
  done
done

wait
echo "[INFO] All jobs finished. Logs in '${logs_dir}/'."
