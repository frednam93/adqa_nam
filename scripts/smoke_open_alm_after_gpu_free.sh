#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
LOG_DIR="${ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}" "${ROOT}/outputs"
cd "${ROOT}"

echo "$(date '+%Y-%m-%d %H:%M:%S') waiting for GPU compute processes to finish"
while nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -q '[0-9]'; do
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader,nounits || true
  sleep 60
done

echo "$(date '+%Y-%m-%d %H:%M:%S') AF3 smoke start"
bash scripts/eval_audioflamingo3_dev.sh \
  --limit 1 \
  --output "${ROOT}/outputs/audioflamingo3_dev_smoke1.jsonl" \
  2>&1 | tee "${LOG_DIR}/audioflamingo3_smoke_$(date +%Y%m%d_%H%M%S).log"

echo "$(date '+%Y-%m-%d %H:%M:%S') Nemotron vLLM smoke start"
PORT="${PORT:-8011}"
PORT="${PORT}" bash scripts/serve_nemotron3_nano_omni_vllm.sh \
  > "${LOG_DIR}/nemotron3_vllm_smoke_server_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
  if python - <<PY
import urllib.request
try:
    urllib.request.urlopen("http://127.0.0.1:${PORT}/v1/models", timeout=2).read()
    raise SystemExit(0)
except Exception:
    raise SystemExit(1)
PY
  then
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Nemotron server exited early; see ${LOG_DIR}/nemotron3_vllm_smoke_server_*.log" >&2
    exit 1
  fi
  sleep 10
done

PORT="${PORT}" bash scripts/eval_nemotron3_nano_omni_dev.sh \
  --limit 1 \
  --audio-mode data_url \
  --output "${ROOT}/outputs/nemotron3_nano_omni_dev_smoke1.jsonl" \
  2>&1 | tee "${LOG_DIR}/nemotron3_smoke_$(date +%Y%m%d_%H%M%S).log"

echo "$(date '+%Y-%m-%d %H:%M:%S') open ALM smoke done"
