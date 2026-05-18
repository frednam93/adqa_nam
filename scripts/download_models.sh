#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="${MODEL_ROOT:-/home/user/ssdmain/models/dcase_adqa}"
mkdir -p "${MODEL_ROOT}"

source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_adqa

python - <<'PY'
from huggingface_hub import snapshot_download

model_root = "/home/user/ssdmain/models/dcase_adqa"
models = {
    "qwen2_audio_7b_instruct": "Qwen/Qwen2-Audio-7B-Instruct",
    "qwen3_omni_30b_a3b_instruct": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
}

for name, repo in models.items():
    print(f"downloading {name}: {repo}", flush=True)
    path = snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir=f"{model_root}/{name}",
        resume_download=True,
    )
    print(f"done {name}: {path}", flush=True)
PY
