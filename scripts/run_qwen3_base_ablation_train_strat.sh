#!/usr/bin/env bash
set -euo pipefail

cd /home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat

MANIFEST_DIR=/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests
OUT_DIR=/home/user/ssdmain/dcase-adqa/outputs/ablations/qwen3_base_train_strat80

python3 -m dcase_adqa.prepare_ablation_manifests \
  --out-dir "$MANIFEST_DIR" \
  --train-per-category 80

python3 -m dcase_adqa.eval_qwen3_omni_ablation_suite \
  --manifest-dir "$MANIFEST_DIR" \
  --split-prefix train_strat80 \
  --out-dir "$OUT_DIR"
