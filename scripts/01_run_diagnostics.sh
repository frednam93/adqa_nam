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
: "${QWEN3_OMNI_MODEL:?Set QWEN3_OMNI_MODEL}"
mkdir -p "$OUTPUT_ROOT/ablations/qwen3_base_train_full_min"
python -m dcase_adqa.prepare_ablation_manifests \
  --train-manifest "$DCASE_TASK5_ROOT/manifests/train.jsonl" \
  --dev-manifest "$DCASE_TASK5_ROOT/manifests/dev.jsonl" \
  --out-dir "$OUTPUT_ROOT/ablation_manifests"
python -m dcase_adqa.eval_qwen3_omni \
  --manifest "$OUTPUT_ROOT/ablation_manifests/train_full_normal.jsonl" \
  --model "$QWEN3_OMNI_MODEL" \
  --output "$OUTPUT_ROOT/ablations/qwen3_base_train_full_min/normal.jsonl" \
  --prompt-mode letter_only
python -m dcase_adqa.eval_qwen3_omni \
  --manifest "$OUTPUT_ROOT/ablation_manifests/train_full_normal.jsonl" \
  --model "$QWEN3_OMNI_MODEL" \
  --output "$OUTPUT_ROOT/ablations/qwen3_base_train_full_min/empty_audio_question.jsonl" \
  --prompt-mode text_only
python -m dcase_adqa.eval_qwen3_omni \
  --manifest "$OUTPUT_ROOT/ablation_manifests/train_full_shuffle_audio_random.jsonl" \
  --model "$QWEN3_OMNI_MODEL" \
  --output "$OUTPUT_ROOT/ablations/qwen3_base_train_full_min/shuffle_audio_random.jsonl" \
  --prompt-mode letter_only

mkdir -p "$OUTPUT_ROOT/ablations/qwen3_base_dev"
python -m dcase_adqa.eval_qwen3_omni \
  --manifest "$OUTPUT_ROOT/ablation_manifests/dev_full_normal.jsonl" \
  --model "$QWEN3_OMNI_MODEL" \
  --output "$OUTPUT_ROOT/ablations/qwen3_base_dev/normal.jsonl" \
  --prompt-mode letter_only
python -m dcase_adqa.eval_qwen3_omni \
  --manifest "$OUTPUT_ROOT/ablation_manifests/dev_full_normal.jsonl" \
  --model "$QWEN3_OMNI_MODEL" \
  --output "$OUTPUT_ROOT/ablations/qwen3_base_dev/empty_audio_question.jsonl" \
  --prompt-mode text_only
python -m dcase_adqa.eval_qwen3_omni \
  --manifest "$OUTPUT_ROOT/ablation_manifests/dev_full_shuffle_audio_random.jsonl" \
  --model "$QWEN3_OMNI_MODEL" \
  --output "$OUTPUT_ROOT/ablations/qwen3_base_dev/shuffle_audio_random.jsonl" \
  --prompt-mode letter_only
