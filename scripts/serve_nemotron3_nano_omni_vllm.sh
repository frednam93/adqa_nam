#!/usr/bin/env bash
set -euo pipefail

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm

MODEL=${1:-/home/user/ssdmain/models/dcase_adqa/nemotron_3_nano_omni_nvfp4}
PORT=${PORT:-8000}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name nemotron_3_nano_omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len 32768 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --limit-mm-per-prompt '{"audio": 1}' \
  --gpu-memory-utilization 0.88 \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype fp8
