#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
MODEL_DIR=/home/user/ssdmain/models/dcase_adqa/gemma_4_e4b_it
MODEL_ID=google/gemma-4-E4B-it

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm
cd "${ROOT}"

if [[ ! -d "${MODEL_DIR}" || ! -f "${MODEL_DIR}/config.json" ]]; then
  mkdir -p "${MODEL_DIR}"
  huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_DIR}" --local-dir-use-symlinks False
fi

python -m dcase_adqa.eval_gemma4_audio \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --model "${MODEL_DIR}" \
  --output "${ROOT}/outputs/gemma4_e4b_dev_full.jsonl" \
  "$@"
