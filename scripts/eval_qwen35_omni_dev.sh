#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/user/ssdmain/dcase-adqa
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_open_alm
cd "${ROOT}"

python -m dcase_adqa.eval_qwen35_omni "$@"
