#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh

notify() {
  local title=$1
  local message=${2:-}
  PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}" python3 -m dcase_adqa.notify_telegram --title "${title}" --message "${message}" || true
}

summarize_eval_file() {
  local path=$1
  python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path
p = Path(sys.argv[1])
rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
ok = sum(bool(r.get("correct")) for r in rows)
bad = sum(r.get("prediction_index") == -1 for r in rows)
print(f"{p.name}: {ok}/{len(rows)} acc={ok/max(len(rows), 1):.4f} parse_bad={bad}")
PY
}

echo "==== $(date '+%m%d %H:%M:%S') wait rebalanced queue idle ===="
notify "DCASE AF3/Gemma watcher start" "waiting for rebalanced SFT queue to finish"

while pgrep -af 'run_qwen3_rebalanced_sft_queue|dcase_adqa_qwen3_rebalanced|eval_qwen3_omni' | grep -v 'wait_rebalanced_then_eval_af3_gemma4' >/dev/null; do
  sleep 300
done

cd "${ROOT}"
notify "DCASE AF3/Gemma eval start" "rebalanced queue appears idle"

echo "==== $(date '+%m%d %H:%M:%S') AF3 eval start ===="
bash scripts/eval_audioflamingo3_dev.sh --output "${ROOT}/outputs/audioflamingo3_dev_full_rerun.jsonl"
af3_summary=$(summarize_eval_file "${ROOT}/outputs/audioflamingo3_dev_full_rerun.jsonl")
notify "DCASE AF3 eval done" "${af3_summary}"

echo "==== $(date '+%m%d %H:%M:%S') Gemma4 E4B eval start ===="
bash scripts/eval_gemma4_e4b_dev.sh
gemma_summary=$(summarize_eval_file "${ROOT}/outputs/gemma4_e4b_dev_full.jsonl")
notify "DCASE Gemma4 E4B eval done" "${gemma_summary}"

echo "${af3_summary}"
echo "${gemma_summary}"
notify "DCASE AF3/Gemma eval finished" "${af3_summary}
${gemma_summary}"
