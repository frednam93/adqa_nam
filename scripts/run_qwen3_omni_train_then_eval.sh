#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
"${ROOT}/scripts/eval_qwen3_omni_base_dev.sh"
"${ROOT}/scripts/run_qwen3_omni_qlora_sft.sh"
"${ROOT}/scripts/eval_qwen3_omni_lora_checkpoints.sh"
