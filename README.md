# DCASE 2026 Task 5 ADQA

Minimal setup for Audio-Dependent Question Answering experiments.

## Data

- Train: `Harland/AudioMCQ-StrongAC-GeminiCoT`
- Dev: `Harland/DCASE2026-Task5-DevSet`
- Local root: `/home/user/ssdmain/datasets/dcase2026_task5`

Download both datasets:

```bash
bash scripts/download_data.sh
```

Create compact JSONL manifests:

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate hf_asr
python -m dcase_adqa.prepare_manifests \
  --data-root /home/user/ssdmain/datasets/dcase2026_task5 \
  --out-dir /home/user/ssdmain/datasets/dcase2026_task5/manifests
```

## Baseline Eval

Install model-side dependencies first:

```bash
bash scripts/install_deps.sh
```

Official DCASE 2026 baselines are Qwen3-Omni, Fun-Audio-Chat, MiMo-Audio, Kimi-Audio, and Step-Audio 2 Mini. Fun-Audio-Chat is the first local baseline target because it is an official baseline and should fit in roughly 24 GB VRAM for inference.

Run a small dev subset with Fun-Audio-Chat:

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_adqa
python -m dcase_adqa.eval_funaudiochat \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --model /home/user/ssdmain/models/dcase_adqa/fun_audio_chat_8b \
  --limit 20 \
  --output outputs/funaudiochat_dev20.jsonl
```

Qwen2-Audio is kept only as a local harness sanity check, not as a reported DCASE 2026 baseline:

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh
conda activate dcase_adqa
python -m dcase_adqa.eval_qwen2_audio \
  --manifest /home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl \
  --limit 20 \
  --model Qwen/Qwen2-Audio-7B-Instruct \
  --output outputs/qwen2_audio_dev20.jsonl
```

The eval script uses exact answer matching against the candidate text and stores raw generations for debugging.
