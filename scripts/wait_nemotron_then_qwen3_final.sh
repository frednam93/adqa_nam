#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
MANIFEST=/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl
NEMO_OUT=${NEMO_OUT:-${ROOT}/outputs/nemotron3_nano_omni_dev_full.jsonl}
QWEN_OUT=${QWEN_OUT:-${ROOT}/outputs/qwen3_omni_qlora_final_dev_full.jsonl}
QWEN_MODEL=${QWEN_MODEL:-/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct}
QWEN_ADAPTER=${QWEN_ADAPTER:-${ROOT}/outputs/qwen3_omni_qlora_sft}

expected=$(wc -l < "${MANIFEST}")
echo "$(date '+%m%d %H:%M:%S') waiting for Nemotron eval ${NEMO_OUT} (${expected} rows)"

while true; do
  done_rows=0
  if [[ -f "${NEMO_OUT}" ]]; then
    done_rows=$(wc -l < "${NEMO_OUT}")
  fi
  echo "$(date '+%m%d %H:%M:%S') nemotron rows=${done_rows}/${expected}"
  if (( done_rows >= expected )); then
    break
  fi
  sleep 60
done

python - <<'PY' "${NEMO_OUT}"
import json
import sys
path = sys.argv[1]
total = correct = 0
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        total += 1
        correct += int(bool(row.get("correct")))
print(f"nemotron accuracy={correct / max(total, 1):.4f} correct={correct} total={total}")
PY

echo "$(date '+%m%d %H:%M:%S') stopping Nemotron vLLM server"
pkill -f "vllm.entrypoints.openai.api_server.*nemotron_3_nano_omni" || true
sleep 10

echo "$(date '+%m%d %H:%M:%S') starting Qwen3 final adapter eval"
cd "${ROOT}"
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m dcase_adqa.eval_qwen3_omni \
  --manifest "${MANIFEST}" \
  --model "${QWEN_MODEL}" \
  --adapter "${QWEN_ADAPTER}" \
  --output "${QWEN_OUT}"
