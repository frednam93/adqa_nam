#!/usr/bin/env bash
set -euo pipefail

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm

ROOT=/home/user/ssdmain/models/dcase_adqa
mkdir -p "${ROOT}"

hf download nvidia/audio-flamingo-3-hf \
  --local-dir "${ROOT}/audio_flamingo_3_hf"

hf download nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
  --local-dir "${ROOT}/nemotron_3_nano_omni_nvfp4"

echo "Downloaded AF3 and Nemotron NVFP4 model files under ${ROOT}"
