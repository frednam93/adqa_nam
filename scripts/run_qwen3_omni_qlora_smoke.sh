#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa/external/Fun-Audio-Chat
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
cd "${ROOT}"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
llamafactory-cli train training/configs/dcase_adqa_qwen3_omni_qlora_smoke.yaml
