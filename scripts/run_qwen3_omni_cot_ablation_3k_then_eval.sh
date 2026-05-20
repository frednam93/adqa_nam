#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
FAC=/home/user/ssdmain/dcase-adqa/external/Fun-Audio-Chat
MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct

source /home/user/miniconda3/etc/profile.d/conda.sh
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_train() {
  local config=$1
  conda activate FunAudioChat
  cd "${FAC}"
  llamafactory-cli train "training/configs/${config}"
}

run_eval() {
  local adapter=$1
  local output=$2
  local max_tokens=$3
  conda activate FunAudioChat
  cd "${ROOT}"
  python -m dcase_adqa.eval_qwen3_omni \
    --manifest "${MANIFEST}" \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --max-new-tokens "${max_tokens}" \
    --output "${output}"
}

echo "$(date '+%m%d %H:%M:%S') silent_cot_3k train start"
run_train dcase_adqa_qwen3_omni_qlora_silent_cot_3k.yaml
echo "$(date '+%m%d %H:%M:%S') silent_cot_3k eval start"
run_eval \
  "${ROOT}/outputs/qwen3_omni_qlora_silent_cot_3k" \
  "${ROOT}/outputs/qwen3_omni_silent_cot_3k_dev_full.jsonl" \
  24

echo "$(date '+%m%d %H:%M:%S') explicit_cot_3k train start"
run_train dcase_adqa_qwen3_omni_qlora_cot_3k.yaml
echo "$(date '+%m%d %H:%M:%S') explicit_cot_3k eval start"
run_eval \
  "${ROOT}/outputs/qwen3_omni_qlora_cot_3k" \
  "${ROOT}/outputs/qwen3_omni_cot_3k_dev_full.jsonl" \
  512
