#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
FAC=${ROOT}/external/Fun-Audio-Chat
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct
DEV_MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
RUN_LIST=${ROOT}/outputs/empty5_stability_runs.txt
OUT_DIR=${ROOT}/outputs/analysis/rescore_choice_matching/empty5_stability_qwen3_judge
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

notify() {
  local title=$1
  local message=${2:-}
  /home/user/.local/bin/codex-notify --title "${title}" --message "${message}" || true
}

summarize_eval_file() {
  local path=$1
  python3 - "$path" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
ok = sum(bool(r.get('correct')) for r in rows)
bad = sum(r.get('prediction_index') == -1 for r in rows)
print(f'{p.name}: {ok}/{len(rows)} acc={ok/max(len(rows),1):.4f} parse_bad={bad}')
PY_SUMMARY
}

run_train() {
  local run=$1
  conda activate FunAudioChat
  cd "${FAC}"
  echo "==== $(date '+%m%d %H:%M:%S') train ${run} start ===="
  llamafactory-cli train "training/configs/dcase_adqa_qwen3_audio_dep_full_${run}_3k.yaml"
}

run_eval_ckpt() {
  local run=$1
  local step=$2
  local adapter="${ROOT}/outputs/qwen3_audio_dep_full_${run}_3k/checkpoint-${step}"
  local output="${ROOT}/outputs/qwen3_audio_dep_full_${run}_3k_dev_ckpt${step}.jsonl"
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

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') prepare empty5 stability ===="
python3 -m dcase_adqa.prepare_empty5_stability
mapfile -t runs < "${RUN_LIST}"
notify "DCASE empty5 stability start" "runs=${runs[*]}"

for run in "${runs[@]}"; do
  run_train "${run}"
  for step in 1500 2000 2500 3000; do
    run_eval_ckpt "${run}" "${step}"
  done
  summary=$(for step in 1500 2000 2500 3000; do
    p="${ROOT}/outputs/qwen3_audio_dep_full_${run}_3k_dev_ckpt${step}.jsonl"
    [[ -s "${p}" ]] && summarize_eval_file "${p}"
  done)
  notify "DCASE ${run} done" "${summary}"
  echo "==== $(date '+%m%d %H:%M:%S') ${run} done ===="
  sleep 3
done

mkdir -p "${OUT_DIR}"
python3 scripts/rescore_many_qwen3_text_judge.py \
  --manifest "${DEV_MANIFEST}" \
  --pred-dir outputs \
  --out-dir "${OUT_DIR}" \
  --min-parse-bad 20

summary=$(python3 - <<'PY_FINAL'
import json
from pathlib import Path
root=Path('/home/user/ssdmain/dcase-adqa')
runs=[x.strip() for x in (root/'outputs/empty5_stability_runs.txt').read_text(encoding='utf-8').splitlines() if x.strip()]
lines=[]
best=None
for run in runs:
    for step in [1500,2000,2500,3000]:
        p=root/f'outputs/qwen3_audio_dep_full_{run}_3k_dev_ckpt{step}.jsonl'
        if not p.exists():
            continue
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        ok=sum(bool(r.get('correct')) for r in rows)
        pb=sum(r.get('prediction_index') == -1 for r in rows)
        acc=ok/max(len(rows),1)
        line=f'{run} ckpt{step}: {ok}/{len(rows)} acc={acc:.4f} parse_bad={pb}'
        lines.append(line)
        if best is None or acc > best[0]:
            best=(acc,line)
if best:
    print('best strict: '+best[1])
judge_summary=root/'outputs/analysis/rescore_choice_matching/empty5_stability_qwen3_judge/qwen3_judge_parsebad_summary.json'
if judge_summary.exists():
    items=json.loads(judge_summary.read_text(encoding='utf-8'))
    focus=[x for x in items if any(run in x['file'] for run in runs)]
    if focus:
        jbest=max(focus, key=lambda x:x['judge_acc'])
        print(f"best judge: {jbest['file']} judge={jbest['judge_acc']:.4f} strict={jbest['strict_acc']:.4f} delta={jbest['delta']}")
print('\n'.join(lines))
PY_FINAL
)
notify "DCASE empty5 stability finished" "${summary}"
echo "${summary}"
