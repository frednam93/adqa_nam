#!/usr/bin/env bash
set -euo pipefail

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm

MODEL=${1:-/home/user/ssdmain/models/dcase_adqa/nemotron_3_nano_omni_nvfp4}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.88}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
MOE_BACKEND=${MOE_BACKEND:-cutlass}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-FLASH_ATTN}

# RTX 50-series/Blackwell currently trips FlashInfer JIT paths in this env.
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER=${VLLM_BLOCKSCALE_FP8_GEMM_FLASHINFER:-0}
export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-FlashInferFP8ScaledMMLinearKernel,FlashInferFp8DeepGEMMDynamicBlockScaledKernel,FlashInferFp8BlockScaledMMKernel}

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name nemotron_3_nano_omni \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --allowed-local-media-path / \
  --limit-mm-per-prompt '{"audio": 1}' \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --reasoning-parser nemotron_v3 \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --moe-backend "${MOE_BACKEND}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  --enforce-eager
