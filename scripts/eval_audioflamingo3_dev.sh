#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm
cd "${ROOT}"

python -m dcase_adqa.eval_audioflamingo3 \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --model /home/user/ssdmain/models/dcase_adqa/audio_flamingo_3_hf \
  --output "${ROOT}/outputs/audioflamingo3_dev_full.jsonl" \
  "$@"
