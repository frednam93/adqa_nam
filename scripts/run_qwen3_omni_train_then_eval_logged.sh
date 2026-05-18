#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
LOG_DIR="${ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/qwen3_omni_train_then_eval_$(date +%Y%m%d_%H%M%S).log"

echo "log=${LOG}"
"${ROOT}/scripts/run_qwen3_omni_train_then_eval.sh" 2>&1 | tee "${LOG}"
