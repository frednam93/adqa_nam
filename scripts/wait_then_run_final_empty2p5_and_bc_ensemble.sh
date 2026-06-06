#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
FAC=${ROOT}/external/Fun-Audio-Chat
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct
EVAL_MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/eval.jsonl
OUT=${ROOT}/outputs/final_submission
B_PRED=${OUT}/preds/B_train_dev_empty5_2k.eval.jsonl
C_RUN=${OUT}/qwen3_train_dev_strong_empty2p5_3k
C_ADAPTER=${C_RUN}/checkpoint-3000
C_PRED=${OUT}/preds/C_train_dev_empty2p5_3k.eval.jsonl
mkdir -p "${OUT}/preds" "${OUT}/csv" "${OUT}/preds_basejudge" "${OUT}/csv_basejudge"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

notify() { /home/user/.local/bin/codex-notify --title "$1" --message "${2:-}" || true; }

while pgrep -af 'dcase_adqa_qwen3_final_train_dev_strong_empty5_2000.yaml|eval_qwen3_omni.*final_submission' >/dev/null; do
  echo "$(date '+%m%d %H:%M:%S') waiting for current final empty5 queue to finish..."
  sleep 60
done

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') prepare train+dev empty2p5 final SFT ===="
python3 -m dcase_adqa.prepare_final_submission_sft --empty-ratio 0.025 --run-suffix empty2p5 --steps 3000

if [[ ! -d "${C_ADAPTER}" ]]; then
  cd "${FAC}"
  echo "==== $(date '+%m%d %H:%M:%S') train C train_dev_strong_empty2p5 start ===="
  llamafactory-cli train training/configs/dcase_adqa_qwen3_final_train_dev_strong_empty2p5_3000.yaml
  notify "DCASE final C train done" "C=${C_ADAPTER}"
else
  echo "skip existing C train ${C_ADAPTER}"
fi

if [[ ! -s "${C_PRED}" ]]; then
  cd "${ROOT}"
  echo "==== $(date '+%m%d %H:%M:%S') eval C empty2p5 start ===="
  python3 -m dcase_adqa.eval_qwen3_omni \
    --manifest "${EVAL_MANIFEST}" \
    --model "${MODEL}" \
    --adapter "${C_ADAPTER}" \
    --max-new-tokens 24 \
    --output "${C_PRED}"
fi

cd "${ROOT}"
python3 -m dcase_adqa.make_submission_outputs single \
  --manifest "${EVAL_MANIFEST}" \
  --pred "${C_PRED}" \
  --output-csv "${OUT}/csv/Fred_SNU_task5_C_train_dev_empty2p5_3k.output.csv" \
  --output-jsonl "${OUT}/csv/C_train_dev_empty2p5_3k.parsed.jsonl"

if [[ -s "${B_PRED}" ]]; then
  python3 -m dcase_adqa.make_submission_outputs ensemble \
    --manifest "${EVAL_MANIFEST}" \
    --name B --pred "${B_PRED}" \
    --name C --pred "${C_PRED}" \
    --tie-breaker B \
    --output-csv "${OUT}/csv/Fred_SNU_task5_ensemble_BC_Btie.output.csv" \
    --output-jsonl "${OUT}/csv/ensemble_BC_Btie.parsed.jsonl"
fi

# Base Qwen3 judge versions. This is still 30B(B)+30B(C)+30B(judge)=90B.
if [[ -s "${B_PRED}" && -s "${C_PRED}" ]]; then
  python3 -m dcase_adqa.judge_eval_parsebad \
    --manifest "${EVAL_MANIFEST}" \
    --out-dir "${OUT}/preds_basejudge" \
    --pred "${B_PRED}" \
    --pred "${C_PRED}"
  for name in B_train_dev_empty5_2k C_train_dev_empty2p5_3k; do
    python3 -m dcase_adqa.make_submission_outputs single \
      --manifest "${EVAL_MANIFEST}" \
      --pred "${OUT}/preds_basejudge/${name}.eval.basejudge.jsonl" \
      --output-csv "${OUT}/csv_basejudge/Fred_SNU_task5_${name}_basejudge.output.csv" \
      --output-jsonl "${OUT}/csv_basejudge/${name}.basejudge.parsed.jsonl"
  done
  python3 -m dcase_adqa.make_submission_outputs ensemble \
    --manifest "${EVAL_MANIFEST}" \
    --name B --pred "${OUT}/preds_basejudge/B_train_dev_empty5_2k.eval.basejudge.jsonl" \
    --name C --pred "${OUT}/preds_basejudge/C_train_dev_empty2p5_3k.eval.basejudge.jsonl" \
    --tie-breaker B \
    --output-csv "${OUT}/csv_basejudge/Fred_SNU_task5_ensemble_BC_Btie_basejudge.output.csv" \
    --output-jsonl "${OUT}/csv_basejudge/ensemble_BC_Btie_basejudge.parsed.jsonl"
fi

summary=$(python3 - <<'PY_SUM'
import json
from pathlib import Path
from collections import Counter
root=Path('/home/user/ssdmain/dcase-adqa/outputs/final_submission')
for p in sorted((root/'preds').glob('[BC]_train_dev_empty*.eval.jsonl')):
    rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    bad=sum(r.get('prediction_index') == -1 for r in rows)
    print(f'{p.name}: n={len(rows)} parse_bad={bad}')
for p in [root/'csv/ensemble_BC_Btie.parsed.jsonl', root/'csv_basejudge/ensemble_BC_Btie_basejudge.parsed.jsonl']:
    if p.exists():
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        print(f'{p.name}: '+str(dict(Counter(r['reason'] for r in rows))))
PY_SUM
)
echo "${summary}"
notify "DCASE final empty2p5/BC submissions done" "${summary}"
