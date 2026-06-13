#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${DCASE_TASK5_ROOT:?Set DCASE_TASK5_ROOT in .env or the environment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

python -m dcase_adqa.build_full_audio_dependency_sft \
  --train-manifest "$OUTPUT_ROOT/ablation_manifests/train_full_normal.jsonl" \
  --shuffle-manifest "$OUTPUT_ROOT/ablation_manifests/train_full_shuffle_audio_random.jsonl" \
  --ablation-dir "$OUTPUT_ROOT/ablations/qwen3_base_train_full_min" \
  --sft-out-dir "$DCASE_TASK5_ROOT/qwen3_omni_sft_audio_dep_full" \
  --analysis-out-dir "$OUTPUT_ROOT/analysis/audio_dependency_full"
python -m dcase_adqa.write_audio_dependency_buckets \
  --manifest "$OUTPUT_ROOT/ablation_manifests/dev_full_normal.jsonl" \
  --results-dir "$OUTPUT_ROOT/ablations/qwen3_base_dev" \
  --output "$OUTPUT_ROOT/analysis/audio_dependency/dev_audio_dependency_buckets.csv"
python -m dcase_adqa.prepare_final_submission_sft \
  --train-manifest "$DCASE_TASK5_ROOT/manifests/train.jsonl" \
  --dev-manifest "$DCASE_TASK5_ROOT/manifests/dev.jsonl" \
  --train-buckets "$OUTPUT_ROOT/analysis/audio_dependency_full/train_full_audio_dependency_buckets.csv" \
  --dev-buckets "$OUTPUT_ROOT/analysis/audio_dependency/dev_audio_dependency_buckets.csv" \
  --out-dir "$DCASE_TASK5_ROOT/qwen3_omni_sft_final_submission"
