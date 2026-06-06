#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
FAC=${ROOT}/external/Fun-Audio-Chat
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct
EVAL_MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/eval.jsonl
OUT=${ROOT}/outputs/final_submission
A_ADAPTER=${ROOT}/outputs/qwen3_audio_dep_full_strong_empty_unknown5_3k/checkpoint-2000
B_RUN=${OUT}/qwen3_train_dev_strong_empty5_2k
B_ADAPTER=${B_RUN}/checkpoint-2000
C_ADAPTER=${B_RUN}/checkpoint-1000
mkdir -p "${OUT}/preds" "${OUT}/csv"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

notify() {
  /home/user/.local/bin/codex-notify --title "$1" --message "${2:-}" || true
}

run_eval() {
  local name=$1
  local adapter=$2
  local output=${OUT}/preds/${name}.eval.jsonl
  if [[ ! -d "${adapter}" ]]; then
    echo "missing adapter for ${name}: ${adapter}" >&2
    return 1
  fi
  if [[ -s "${output}" ]]; then
    echo "skip existing eval ${output}"
    return 0
  fi
  cd "${ROOT}"
  echo "==== $(date '+%m%d %H:%M:%S') eval ${name} start ===="
  python3 -m dcase_adqa.eval_qwen3_omni \
    --manifest "${EVAL_MANIFEST}" \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --max-new-tokens 24 \
    --output "${output}"
}

make_single_csv() {
  local name=$1
  cd "${ROOT}"
  python3 -m dcase_adqa.make_submission_outputs single \
    --manifest "${EVAL_MANIFEST}" \
    --pred "${OUT}/preds/${name}.eval.jsonl" \
    --output-csv "${OUT}/csv/Nam_IND_task5_${name}.output.csv" \
    --output-jsonl "${OUT}/csv/${name}.parsed.jsonl"
}

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') prepare train+dev final SFT ===="
python3 -m dcase_adqa.prepare_final_submission_sft

if [[ ! -d "${B_ADAPTER}" ]]; then
  cd "${FAC}"
  echo "==== $(date '+%m%d %H:%M:%S') train B/C train_dev_strong_empty5 start ===="
  llamafactory-cli train training/configs/dcase_adqa_qwen3_final_train_dev_strong_empty5_2000.yaml
  notify "DCASE final train done" "B=${B_ADAPTER}; C=${C_ADAPTER}"
else
  echo "skip existing train ${B_ADAPTER}"
fi

echo "==== $(date '+%m%d %H:%M:%S') B train finished; hand off to unified final watcher ===="
exit 0

run_eval A_train_only_empty5_2k "${A_ADAPTER}"
run_eval B_train_dev_empty5_2k "${B_ADAPTER}"
run_eval C_train_dev_empty5_1k "${C_ADAPTER}"

make_single_csv A_train_only_empty5_2k
make_single_csv B_train_dev_empty5_2k
make_single_csv C_train_dev_empty5_1k

cd "${ROOT}"
python3 -m dcase_adqa.make_submission_outputs ensemble \
  --manifest "${EVAL_MANIFEST}" \
  --name A --pred "${OUT}/preds/A_train_only_empty5_2k.eval.jsonl" \
  --name B --pred "${OUT}/preds/B_train_dev_empty5_2k.eval.jsonl" \
  --name C --pred "${OUT}/preds/C_train_dev_empty5_1k.eval.jsonl" \
  --tie-breaker B \
  --output-csv "${OUT}/csv/Nam_IND_task5_ensemble_ABC_Btie.output.csv" \
  --output-jsonl "${OUT}/csv/ensemble_ABC_Btie.parsed.jsonl"

summary=$(python3 - <<'PY_SUM'
import json
from pathlib import Path
root=Path('/home/user/ssdmain/dcase-adqa/outputs/final_submission')
for p in sorted((root/'preds').glob('*.eval.jsonl')):
    rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    bad=sum(r.get('prediction_index') == -1 for r in rows)
    print(f'{p.name}: n={len(rows)} parse_bad={bad}')
ens=root/'csv/ensemble_ABC_Btie.parsed.jsonl'
if ens.exists():
    rows=[json.loads(x) for x in ens.read_text(encoding='utf-8').splitlines() if x.strip()]
    from collections import Counter
    print('ensemble:', dict(Counter(r['reason'] for r in rows)))
PY_SUM
)
echo "${summary}"
notify "DCASE final eval submissions done" "${summary}"
