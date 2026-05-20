#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
DATA=/home/user/ssdmain/datasets/dcase2026_task5
MODE=${QWEN3_COT_MODE:-answer_only}
OUT_NAME=${QWEN3_SFT_OUT_NAME:-qwen3_omni_sft}

python3 -m dcase_adqa.prepare_qwen3_omni_sft \
  --manifest "${DATA}/manifests/train.jsonl" \
  --output "${DATA}/${OUT_NAME}/train.jsonl" \
  --cot-mode "${MODE}"

python3 -m dcase_adqa.prepare_qwen3_omni_sft \
  --manifest "${DATA}/manifests/dev.jsonl" \
  --output "${DATA}/${OUT_NAME}/dev.jsonl" \
  --cot-mode answer_only

echo "Qwen3-Omni SFT data written under ${DATA}/${OUT_NAME} cot_mode=${MODE}"
