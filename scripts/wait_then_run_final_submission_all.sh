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
C_RUN=${OUT}/qwen3_train_dev_strong_empty2p5_3k
C_ADAPTER=${C_RUN}/checkpoint-3000
A_PRED=${OUT}/preds/A_train_only_empty5_2k.eval.jsonl
B_PRED=${OUT}/preds/B_train_dev_empty5_2k.eval.jsonl
C_PRED=${OUT}/preds/C_train_dev_empty2p5_3k.eval.jsonl
mkdir -p "${OUT}/preds" "${OUT}/csv" "${OUT}/preds_basejudge" "${OUT}/csv_basejudge"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

notify() { /home/user/.local/bin/codex-notify --title "$1" --message "${2:-}" || true; }

wait_for_b_train() {
  while pgrep -af 'dcase_adqa_qwen3_final_train_dev_strong_empty5_2000.yaml' >/dev/null; do
    echo "$(date '+%m%d %H:%M:%S') waiting for B train to finish..."
    sleep 60
  done
  if [[ ! -d "${B_ADAPTER}" ]]; then
    echo "missing B adapter after training: ${B_ADAPTER}" >&2
    exit 1
  fi
  # Prevent the old queue script from continuing into obsolete C_empty5_1k eval.
  pkill -f 'bash scripts/run_final_eval_submission_queue.sh' 2>/dev/null || true
}

run_eval() {
  local name=$1
  local adapter=$2
  local output=$3
  if [[ ! -d "${adapter}" ]]; then
    echo "missing adapter for ${name}: ${adapter}" >&2
    exit 1
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
  local pred=$2
  local outdir=$3
  cd "${ROOT}"
  python3 -m dcase_adqa.make_submission_outputs single \
    --manifest "${EVAL_MANIFEST}" \
    --pred "${pred}" \
    --output-csv "${outdir}/Nam_IND_task5_${name}.output.csv" \
    --output-jsonl "${outdir}/${name}.parsed.jsonl"
}

make_bc_ensemble() {
  local b_pred=$1
  local c_pred=$2
  local outdir=$3
  local suffix=$4
  cd "${ROOT}"
  python3 -m dcase_adqa.make_submission_outputs ensemble \
    --manifest "${EVAL_MANIFEST}" \
    --name B --pred "${b_pred}" \
    --name C --pred "${c_pred}" \
    --tie-breaker B \
    --output-csv "${outdir}/Nam_IND_task5_ensemble_BC_Btie${suffix}.output.csv" \
    --output-jsonl "${outdir}/ensemble_BC_Btie${suffix}.parsed.jsonl"
}

wait_for_b_train

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') prepare C train+dev empty2p5 SFT ===="
python3 -m dcase_adqa.prepare_final_submission_sft --empty-ratio 0.025 --run-suffix empty2p5 --steps 3000

if [[ ! -d "${C_ADAPTER}" ]]; then
  cd "${FAC}"
  echo "==== $(date '+%m%d %H:%M:%S') train C train_dev_strong_empty2p5 start ===="
  llamafactory-cli train training/configs/dcase_adqa_qwen3_final_train_dev_strong_empty2p5_3000.yaml
  notify "DCASE final C train done" "C=${C_ADAPTER}"
else
  echo "skip existing C train ${C_ADAPTER}"
fi

run_eval A_train_only_empty5_2k "${A_ADAPTER}" "${A_PRED}"
run_eval B_train_dev_empty5_2k "${B_ADAPTER}" "${B_PRED}"
run_eval C_train_dev_empty2p5_3k "${C_ADAPTER}" "${C_PRED}"

make_single_csv A_train_only_empty5_2k "${A_PRED}" "${OUT}/csv"
make_single_csv B_train_dev_empty5_2k "${B_PRED}" "${OUT}/csv"
make_single_csv C_train_dev_empty2p5_3k "${C_PRED}" "${OUT}/csv"
make_bc_ensemble "${B_PRED}" "${C_PRED}" "${OUT}/csv" ""

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') qwen3 base judge parse_bad start ===="
python3 -m dcase_adqa.judge_eval_parsebad \
  --manifest "${EVAL_MANIFEST}" \
  --out-dir "${OUT}/preds_basejudge" \
  --pred "${A_PRED}" \
  --pred "${B_PRED}" \
  --pred "${C_PRED}"

make_single_csv A_train_only_empty5_2k_basejudge "${OUT}/preds_basejudge/A_train_only_empty5_2k.eval.basejudge.jsonl" "${OUT}/csv_basejudge"
make_single_csv B_train_dev_empty5_2k_basejudge "${OUT}/preds_basejudge/B_train_dev_empty5_2k.eval.basejudge.jsonl" "${OUT}/csv_basejudge"
make_single_csv C_train_dev_empty2p5_3k_basejudge "${OUT}/preds_basejudge/C_train_dev_empty2p5_3k.eval.basejudge.jsonl" "${OUT}/csv_basejudge"
make_bc_ensemble \
  "${OUT}/preds_basejudge/B_train_dev_empty5_2k.eval.basejudge.jsonl" \
  "${OUT}/preds_basejudge/C_train_dev_empty2p5_3k.eval.basejudge.jsonl" \
  "${OUT}/csv_basejudge" \
  "_basejudge"

summary=$(python3 - <<'PY_SUM'
import json
from pathlib import Path
from collections import Counter
root=Path('/home/user/ssdmain/dcase-adqa/outputs/final_submission')
for p in [root/'preds/A_train_only_empty5_2k.eval.jsonl', root/'preds/B_train_dev_empty5_2k.eval.jsonl', root/'preds/C_train_dev_empty2p5_3k.eval.jsonl']:
    if p.exists():
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        bad=sum(r.get('prediction_index') == -1 for r in rows)
        print(f'{p.name}: n={len(rows)} parse_bad={bad}')
for p in sorted((root/'preds_basejudge').glob('*.basejudge.jsonl')):
    rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    orig_bad=sum(r.get('prediction_index') == -1 for r in rows)
    judge_bad=sum(r.get('judge_prediction_index') == -1 for r in rows)
    print(f'{p.name}: original_bad={orig_bad} judge_bad={judge_bad}')
for p in [root/'csv/ensemble_BC_Btie.parsed.jsonl', root/'csv_basejudge/ensemble_BC_Btie_basejudge.parsed.jsonl']:
    if p.exists():
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        print(f'{p.name}: '+str(dict(Counter(r['reason'] for r in rows))))
PY_SUM
)
echo "${summary}"
notify "DCASE final submissions done" "${summary}"
