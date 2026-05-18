#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
cd "${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m dcase_adqa.eval_qwen3_omni \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --model /home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct \
  --output "${ROOT}/outputs/qwen3_omni_base_dev_full.jsonl"
