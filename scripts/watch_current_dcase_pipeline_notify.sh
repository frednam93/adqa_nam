#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
OUT=${ROOT}/outputs
PYTHONPATH=${ROOT}/src:${PYTHONPATH:-}
export PYTHONPATH

notify() {
  local title=$1
  local message=${2:-}
  python3 -m dcase_adqa.notify_telegram --title "${title}" --message "${message}" || true
}

count_lines() {
  local p=$1
  [[ -f "$p" ]] && wc -l < "$p" || echo 0
}

summarize_jsonl() {
  python3 - "$@" <<'PY_SUMMARY'
import json
import sys
from pathlib import Path
for arg in sys.argv[1:]:
    p=Path(arg)
    if not p.exists():
        continue
    rows=[]
    with p.open(encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    ok=sum(bool(r.get('correct')) for r in rows)
    bad=sum(r.get('prediction_index') == -1 for r in rows)
    print(f'{p.name}: {ok}/{len(rows)} acc={ok/max(len(rows),1):.4f} parse_bad={bad}')
PY_SUMMARY
}

wait_lines() {
  local path=$1
  local n=$2
  local label=$3
  while true; do
    local cur
    cur=$(count_lines "$path")
    if [[ "$cur" -ge "$n" ]]; then
      return 0
    fi
    sleep 60
  done
}

notify "DCASE watcher attached" "current full-train pipeline watcher is active"

normal=${OUT}/ablations/qwen3_base_train_full_min/normal.jsonl
empty=${OUT}/ablations/qwen3_base_train_full_min/empty_audio_question.jsonl
shuffle=${OUT}/ablations/qwen3_base_train_full_min/shuffle_audio_random.jsonl

wait_lines "$shuffle" 19480 shuffle_audio_random
summary=$(summarize_jsonl "$normal" "$empty" "$shuffle")
notify "DCASE train-full 3-way ablation done" "$summary"

bucket=${OUT}/analysis/audio_dependency_full/train_full_audio_dependency_bucket_summary.md
while [[ ! -s "$bucket" ]]; do sleep 60; done
notify "DCASE audio-dependency buckets ready" "$(sed -n '1,40p' "$bucket")"

runs=(
  strong_ac
  strong_hard_ac
  non_easy_ac
  non_easy_empty_unknown5
  non_easy_shuffle_unknown5
  non_easy_empty_shuffle_unknown10
)

for run in "${runs[@]}"; do
  paths=()
  for step in 1000 2000 3000; do
    p=${OUT}/qwen3_audio_dep_full_${run}_3k_dev_ckpt${step}.jsonl
    while [[ $(count_lines "$p") -lt 1607 ]]; do sleep 60; done
    paths+=("$p")
  done
  summary=$(summarize_jsonl "${paths[@]}")
  notify "DCASE ${run} dev evals done" "$summary"
done

summary=$(python3 - <<'PY_BEST'
import json
from pathlib import Path
root=Path('/home/user/ssdmain/dcase-adqa/outputs')
runs=['strong_ac','strong_hard_ac','non_easy_ac','non_easy_empty_unknown5','non_easy_shuffle_unknown5','non_easy_empty_shuffle_unknown10']
best=None
lines=[]
for run in runs:
    for step in [1000,2000,3000]:
        p=root/f'qwen3_audio_dep_full_{run}_3k_dev_ckpt{step}.jsonl'
        if not p.exists():
            continue
        rows=[json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
        ok=sum(bool(r.get('correct')) for r in rows)
        bad=sum(r.get('prediction_index') == -1 for r in rows)
        acc=ok/max(len(rows),1)
        line=f'{run} ckpt{step}: {ok}/{len(rows)} acc={acc:.4f} parse_bad={bad}'
        lines.append(line)
        if best is None or acc > best[0]:
            best=(acc,line)
if best:
    print('best: '+best[1])
print('\n'.join(lines))
PY_BEST
)
notify "DCASE full audio-dependency queue finished" "$summary"
