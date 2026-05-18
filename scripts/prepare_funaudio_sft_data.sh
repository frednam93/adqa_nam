#!/usr/bin/env bash
set -euo pipefail

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat

python -m dcase_adqa.prepare_funaudio_sft \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/train.jsonl \
  --output /home/user/ssdmain/datasets/dcase2026_task5/funaudio_sft/train.jsonl

python -m dcase_adqa.prepare_funaudio_sft \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --output /home/user/ssdmain/datasets/dcase2026_task5/funaudio_sft/dev.jsonl
