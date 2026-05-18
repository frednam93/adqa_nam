#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-dcase_adqa}"

source /home/user/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" python=3.10
fi

conda activate "${ENV_NAME}"
python -m pip install --upgrade pip
python -m pip install -r /home/user/ssdmain/dcase-adqa/requirements.txt
python -m pip install -e /home/user/ssdmain/dcase-adqa
