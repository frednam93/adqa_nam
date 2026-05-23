#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
FAC=/home/user/ssdmain/dcase-adqa/external/Fun-Audio-Chat
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct
DEV_MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
LOG_DIR=${ROOT}/outputs/logs
mkdir -p "${LOG_DIR}"

source /home/user/miniconda3/etc/profile.d/conda.sh
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

runs=(
  strong_ac
  strong_hard_ac
  non_easy_ac
  non_easy_empty_unknown5
  non_easy_shuffle_unknown5
  non_easy_empty_shuffle_unknown10
)

python3 -m dcase_adqa.prepare_audio_dependency_sft

run_train() {
  local run=$1
  conda activate FunAudioChat
  cd "${FAC}"
  echo "==== $(date '+%m%d %H:%M:%S') train ${run} start ===="
  llamafactory-cli train "training/configs/dcase_adqa_qwen3_audio_dep_${run}_3k.yaml"
}

run_eval_ckpt() {
  local run=$1
  local step=$2
  local adapter="${ROOT}/outputs/qwen3_audio_dep_${run}_3k/checkpoint-${step}"
  local output="${ROOT}/outputs/qwen3_audio_dep_${run}_3k_dev_ckpt${step}.jsonl"
  if [[ ! -d "${adapter}" ]]; then
    echo "skip missing adapter ${adapter}"
    return 0
  fi
  if [[ -s "${output}" ]]; then
    echo "skip existing eval ${output}"
    return 0
  fi
  conda activate FunAudioChat
  cd "${ROOT}"
  echo "==== $(date '+%m%d %H:%M:%S') eval ${run} ckpt${step} start ===="
  python3 -m dcase_adqa.eval_qwen3_omni \
    --manifest "${DEV_MANIFEST}" \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --max-new-tokens 24 \
    --output "${output}"
}

for run in "${runs[@]}"; do
  run_train "${run}"
  for step in 1000 2000 3000; do
    run_eval_ckpt "${run}" "${step}"
  done
  echo "==== $(date '+%m%d %H:%M:%S') ${run} done ===="
  sleep 3
done

python3 - <<'PY'
import json
from pathlib import Path
root=Path('/home/user/ssdmain/dcase-adqa')
runs=['strong_ac','strong_hard_ac','non_easy_ac','non_easy_empty_unknown5','non_easy_shuffle_unknown5','non_easy_empty_shuffle_unknown10']
for run in runs:
    for step in [1000,2000,3000]:
        p=root/f'outputs/qwen3_audio_dep_{run}_3k_dev_ckpt{step}.jsonl'
        if not p.exists():
            continue
        rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        ok=sum(bool(r.get('correct')) for r in rows)
        pb=sum(r.get('prediction_index') == -1 for r in rows)
        print(f'{run} ckpt{step}: {ok}/{len(rows)} acc={ok/max(len(rows),1):.4f} parse_bad={pb}')
PY
