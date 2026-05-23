#!/usr/bin/env bash
set -euo pipefail

cd /home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat

MANIFEST_DIR=/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests
DEV_OUT=/home/user/ssdmain/dcase-adqa/outputs/ablations/qwen3_base_dev
TRAIN_OUT=/home/user/ssdmain/dcase-adqa/outputs/ablations/qwen3_base_train_strat80
mkdir -p "$DEV_OUT" "$TRAIN_OUT"

python3 -m dcase_adqa.prepare_ablation_manifests \
  --out-dir "$MANIFEST_DIR" \
  --train-per-category 80

# Dev normal was already evaluated as the official Qwen3 base dev run.
cp -n /home/user/ssdmain/dcase-adqa/outputs/qwen3_omni_base_dev_full.jsonl "$DEV_OUT/normal.jsonl"

python3 -m dcase_adqa.eval_qwen3_omni_ablation_suite \
  --manifest-dir "$MANIFEST_DIR" \
  --split-prefix dev_full \
  --out-dir "$DEV_OUT"

python3 -m dcase_adqa.eval_qwen3_omni_ablation_suite \
  --manifest-dir "$MANIFEST_DIR" \
  --split-prefix train_strat80 \
  --out-dir "$TRAIN_OUT"
