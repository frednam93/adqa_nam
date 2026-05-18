#!/usr/bin/env bash
set -euo pipefail

source /home/user/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx dcase_open_alm; then
  conda create -y -n dcase_open_alm python=3.12
fi
conda activate dcase_open_alm

python -m pip install --upgrade pip
python -m pip install -e /home/user/ssdmain/dcase-adqa
python -m pip install \
  "accelerate>=1.12.0" \
  "bitsandbytes>=0.49.0" \
  "huggingface_hub[hf_xet]>=1.0.0" \
  "librosa>=0.10.0" \
  "openai>=2.0.0" \
  "soundfile>=0.12.0" \
  "torch>=2.8.0" \
  "torchvision" \
  "tqdm" \
  "vllm[audio]>=0.20.0"

# AF3 landed after the currently used FunAudioChat transformers build.
python -m pip install --upgrade "git+https://github.com/huggingface/transformers.git"

echo "Open ALM eval env ready: conda activate dcase_open_alm"
