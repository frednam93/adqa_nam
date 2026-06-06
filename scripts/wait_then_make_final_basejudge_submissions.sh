#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
EVAL_MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/eval.jsonl
OUT=${ROOT}/outputs/final_submission
PREDS=(
  ${OUT}/preds/A_train_only_empty5_2k.eval.jsonl
  ${OUT}/preds/B_train_dev_empty5_2k.eval.jsonl
  ${OUT}/preds/C_train_dev_empty5_1k.eval.jsonl
)
mkdir -p "${OUT}/preds_basejudge" "${OUT}/csv_basejudge"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

notify() {
  /home/user/.local/bin/codex-notify --title "$1" --message "${2:-}" || true
}

while true; do
  ready=1
  for pred in "${PREDS[@]}"; do
    [[ -s "${pred}" ]] || ready=0
  done
  [[ "${ready}" == 1 ]] && break
  echo "$(date '+%m%d %H:%M:%S') waiting for A/B/C eval predictions..."
  sleep 60
done

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') qwen3 base judge parse_bad start ===="
python3 -m dcase_adqa.judge_eval_parsebad \
  --manifest "${EVAL_MANIFEST}" \
  --out-dir "${OUT}/preds_basejudge" \
  --pred "${PREDS[0]}" \
  --pred "${PREDS[1]}" \
  --pred "${PREDS[2]}"

for name in A_train_only_empty5_2k B_train_dev_empty5_2k C_train_dev_empty5_1k; do
  python3 -m dcase_adqa.make_submission_outputs single \
    --manifest "${EVAL_MANIFEST}" \
    --pred "${OUT}/preds_basejudge/${name}.eval.basejudge.jsonl" \
    --output-csv "${OUT}/csv_basejudge/Fred_SNU_task5_${name}_basejudge.output.csv" \
    --output-jsonl "${OUT}/csv_basejudge/${name}.basejudge.parsed.jsonl"
done

python3 -m dcase_adqa.make_submission_outputs ensemble \
  --manifest "${EVAL_MANIFEST}" \
  --name A --pred "${OUT}/preds_basejudge/A_train_only_empty5_2k.eval.basejudge.jsonl" \
  --name B --pred "${OUT}/preds_basejudge/B_train_dev_empty5_2k.eval.basejudge.jsonl" \
  --tie-breaker B \
  --output-csv "${OUT}/csv_basejudge/Fred_SNU_task5_ensemble_AB_Btie_basejudge.output.csv" \
  --output-jsonl "${OUT}/csv_basejudge/ensemble_AB_Btie_basejudge.parsed.jsonl"

summary=$(python3 - <<'PY_SUM'
import json
from pathlib import Path
from collections import Counter
root=Path('/home/user/ssdmain/dcase-adqa/outputs/final_submission')
for p in sorted((root/'preds_basejudge').glob('*.basejudge.jsonl')):
    rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
    original_bad=sum(r.get('prediction_index') == -1 for r in rows)
    judge_bad=sum(r.get('judge_prediction_index') == -1 for r in rows)
    print(f'{p.name}: original_bad={original_bad} judge_bad={judge_bad}')
ens=root/'csv_basejudge/ensemble_AB_Btie_basejudge.parsed.jsonl'
if ens.exists():
    rows=[json.loads(x) for x in ens.read_text(encoding='utf-8').splitlines() if x.strip()]
    print('AB ensemble:', dict(Counter(r['reason'] for r in rows)))
PY_SUM
)
echo "${summary}"
notify "DCASE basejudge submissions done" "${summary}"
