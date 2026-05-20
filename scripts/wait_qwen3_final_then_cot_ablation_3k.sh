#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
QWEN_OUT=${QWEN_OUT:-${ROOT}/outputs/qwen3_omni_qlora_final_dev_full.jsonl}
expected=$(wc -l < "${MANIFEST}")

echo "$(date '+%m%d %H:%M:%S') waiting for Qwen3 final eval ${QWEN_OUT} (${expected} rows)"
while true; do
  rows=0
  if [[ -f "${QWEN_OUT}" ]]; then
    rows=$(wc -l < "${QWEN_OUT}")
  fi
  echo "$(date '+%m%d %H:%M:%S') qwen3 final rows=${rows}/${expected}"
  if (( rows >= expected )); then
    break
  fi
  sleep 60
done

echo "$(date '+%m%d %H:%M:%S') starting CoT ablation queue"
"${ROOT}/scripts/run_qwen3_omni_cot_ablation_3k_then_eval.sh"
