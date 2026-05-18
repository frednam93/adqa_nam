#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm
cd "${ROOT}"

python -m dcase_adqa.eval_openai_audio_chat \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --model nemotron_3_nano_omni \
  --base-url http://127.0.0.1:${PORT:-8000}/v1 \
  --audio-mode file_url \
  --output "${ROOT}/outputs/nemotron3_nano_omni_dev_full.jsonl" \
  "$@"
