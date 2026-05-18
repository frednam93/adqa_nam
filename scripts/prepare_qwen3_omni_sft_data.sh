#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
DATA=/home/user/ssdmain/datasets/dcase2026_task5

python3 -m dcase_adqa.prepare_qwen3_omni_sft \
  --manifest "${DATA}/manifests/train.jsonl" \
  --output "${DATA}/qwen3_omni_sft/train.jsonl"

python3 -m dcase_adqa.prepare_qwen3_omni_sft \
  --manifest "${DATA}/manifests/dev.jsonl" \
  --output "${DATA}/qwen3_omni_sft/dev.jsonl"

echo "Qwen3-Omni SFT data written under ${DATA}/qwen3_omni_sft"
