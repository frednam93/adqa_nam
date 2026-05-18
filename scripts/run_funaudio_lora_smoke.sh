#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa/external/Fun-Audio-Chat
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate FunAudioChat
cd "${ROOT}"
export AUDIO_PLACEHOLDER="<|audio_bos|><|AUDIO|><|audio_eos|>"
export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1
export PYTHONPATH="${ROOT}/training/plugin:${ROOT}:${PYTHONPATH:-}"
llamafactory-cli train training/configs/dcase_adqa_lora_smoke.yaml
