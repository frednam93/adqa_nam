#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
MODEL=/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct
BASE_ADAPTER=${ROOT}/outputs/qwen3_audio_dep_full_strong_empty_unknown5_3k/checkpoint-2000
DEV_MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
GRPO_MANIFEST=${ROOT}/outputs/ablation_manifests/train_hard_audio_dependent_grpo.jsonl
OUT_BASE=${ROOT}/outputs/grpo_minimal_hard
JUDGE_LINK_DIR=${ROOT}/outputs/grpo_minimal_hard_judge_inputs
JUDGE_OUT=${ROOT}/outputs/analysis/rescore_choice_matching/grpo_minimal_hard_qwen3_judge
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
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
rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
ok = sum(bool(r.get("correct")) for r in rows)
bad = sum(r.get("prediction_index") == -1 for r in rows)
print(f"{p.name}: {ok}/{len(rows)} acc={ok/max(len(rows),1):.4f} parse_bad={bad}")
PY_SUMMARY
}

run_grpo() {
  local run=$1
  shift
  local out=${OUT_BASE}/${run}
  cd "${ROOT}"
  echo "==== $(date '+%m%d %H:%M:%S') ${run} train start ===="
  python3 -m dcase_adqa.train_qwen3_omni_grpo \
    --manifest "${GRPO_MANIFEST}" \
    --model "${MODEL}" \
    --adapter "${BASE_ADAPTER}" \
    --output-dir "${out}" \
    --max-steps 100 \
    --num-generations 4 \
    --learning-rate 1.0e-6 \
    --temperature 1.0 \
    --max-new-tokens 24 \
    --save-steps 100 \
    --log-steps 1 \
    "$@"
}

run_eval() {
  local run=$1
  local adapter=${OUT_BASE}/${run}/checkpoint-100
  local output=${ROOT}/outputs/qwen3_audio_dep_full_hard_grpo_${run}_dev_ckpt100.jsonl
  cd "${ROOT}"
  echo "==== $(date '+%m%d %H:%M:%S') ${run} eval start ===="
  python3 -m dcase_adqa.eval_qwen3_omni \
    --manifest "${DEV_MANIFEST}" \
    --model "${MODEL}" \
    --adapter "${adapter}" \
    --max-new-tokens 24 \
    --output "${output}"
  summarize_eval_file "${output}"
}

cd "${ROOT}"
echo "==== $(date '+%m%d %H:%M:%S') prepare GRPO manifest ===="
python3 -m dcase_adqa.prepare_grpo_manifest --output "${GRPO_MANIFEST}" --buckets hard_audio_dependent_candidate,wrong_normal_but_shuffle_correct

echo "==== $(date '+%m%d %H:%M:%S') GRPO smoke start ===="
python3 -m dcase_adqa.train_qwen3_omni_grpo \
  --manifest "${GRPO_MANIFEST}" \
  --model "${MODEL}" \
  --adapter "${BASE_ADAPTER}" \
  --output-dir "${OUT_BASE}/smoke" \
  --limit 8 \
  --max-steps 3 \
  --num-generations 4 \
  --learning-rate 1.0e-6 \
  --temperature 1.0 \
  --max-new-tokens 24 \
  --save-steps 3 \
  --log-steps 1
notify "DCASE GRPO smoke done" "starting grpo100 and dapo-lite100"

run_grpo grpo100
run_eval grpo100
notify "DCASE GRPO100 done" "$(summarize_eval_file "${ROOT}/outputs/qwen3_audio_dep_full_hard_grpo_grpo100_dev_ckpt100.jsonl")"
sleep 3

run_grpo dapo_lite100 --dapo-lite
run_eval dapo_lite100
notify "DCASE DAPO-lite100 done" "$(summarize_eval_file "${ROOT}/outputs/qwen3_audio_dep_full_hard_grpo_dapo_lite100_dev_ckpt100.jsonl")"
sleep 3

rm -rf "${JUDGE_LINK_DIR}"
mkdir -p "${JUDGE_LINK_DIR}" "${JUDGE_OUT}"
ln -sf "${ROOT}/outputs/qwen3_audio_dep_full_hard_grpo_grpo100_dev_ckpt100.jsonl" "${JUDGE_LINK_DIR}/qwen3_audio_dep_full_hard_grpo_grpo100_dev_ckpt100.jsonl"
ln -sf "${ROOT}/outputs/qwen3_audio_dep_full_hard_grpo_dapo_lite100_dev_ckpt100.jsonl" "${JUDGE_LINK_DIR}/qwen3_audio_dep_full_hard_grpo_dapo_lite100_dev_ckpt100.jsonl"
python3 scripts/rescore_many_qwen3_text_judge.py \
  --manifest "${DEV_MANIFEST}" \
  --pred-dir "${JUDGE_LINK_DIR}" \
  --out-dir "${JUDGE_OUT}" \
  --min-parse-bad 1

summary=$(python3 - <<'PY_FINAL'
import json
from pathlib import Path
root = Path("/home/user/ssdmain/dcase-adqa")
lines = []
for run in ["grpo100", "dapo_lite100"]:
    p = root / f"outputs/qwen3_audio_dep_full_strong_empty5_{run}_dev_ckpt100.jsonl"
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    ok = sum(bool(r.get("correct")) for r in rows)
    bad = sum(r.get("prediction_index") == -1 for r in rows)
    lines.append(f"{run}: strict={ok/len(rows):.4f} ({ok}/{len(rows)}) parse_bad={bad}")
judge = root / "outputs/analysis/rescore_choice_matching/grpo_minimal_hard_qwen3_judge/qwen3_judge_parsebad_summary.json"
if judge.exists():
    for row in json.loads(judge.read_text(encoding="utf-8")):
        lines.append(f"{row['file']}: judge={row['judge_acc']:.4f} strict={row['strict_acc']:.4f} delta={row['delta']} judge_bad={row['judge_bad']}")
print("\n".join(lines))
PY_FINAL
)
notify "DCASE GRPO queue finished" "${summary}"
echo "${summary}"
