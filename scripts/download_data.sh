#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/home/user/ssdmain/datasets/dcase2026_task5}"
mkdir -p "${DATA_ROOT}"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate hf_asr

python - <<'PY'
from huggingface_hub import snapshot_download

root = "/home/user/ssdmain/datasets/dcase2026_task5"
repos = {
    "train": "Harland/AudioMCQ-StrongAC-GeminiCoT",
    "dev": "Harland/DCASE2026-Task5-DevSet",
}

for name, repo in repos.items():
    print(f"downloading {name}: {repo}", flush=True)
    path = snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=f"{root}/{name}",
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"done {name}: {path}", flush=True)
PY
