#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/user/ssdmain/dcase-adqa
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct
DEV=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
ADAPTER=${ROOT}/outputs/qwen3_audio_dep_full_strong_empty_unknown5_3k/checkpoint-2000
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"
run_eval() {
  local mode=$1
  local limit=$2
  local tokens=$3
  local suffix=$4
  local out="${ROOT}/outputs/qwen3_audio_dep_full_strong_empty_unknown5_3k_dev_ckpt2000_${mode}${suffix}.jsonl"
  echo "==== $(date '+%m%d %H:%M:%S') eval empty5@2k ${mode}${suffix} start ===="
  python3 -m dcase_adqa.eval_qwen3_omni \
    --manifest "$DEV" \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --prompt-mode "$mode" \
    --max-new-tokens "$tokens" \
    ${limit:+--limit "$limit"} \
    --output "$out"
  echo "==== $(date '+%m%d %H:%M:%S') eval empty5@2k ${mode}${suffix} done ===="
}
run_eval letter_only 20 8 _smoke20
run_eval cot_then_letter 20 96 _smoke20
run_eval letter_only '' 8 ''
run_eval cot_then_letter '' 96 ''
/home/user/.local/bin/codex-notify --title "DCASE empty5 prompt eval done" --message "letter_only and cot_then_letter full dev finished" || true
