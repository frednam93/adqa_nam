# ADQA Nam

Code and paper artifacts for the DCASE 2026 Task 5 Audio-Dependent Question Answering submission, **Learning from Audio-Dependency Errors: Data Curation Strategies Based on Model Confusion Patterns in Audio Question Answering**.

This repository intentionally does not include the DCASE data or generated predictions. Download datasets and models from their official sources and point the scripts to local paths through environment variables.

Technical report: https://arxiv.org/abs/2606.22276

Public LoRA adapters for the submitted Qwen3-Omni systems are available on Hugging Face:
https://huggingface.co/frednamfred/adqa_nam_qwen3_omni_lora

## Repository Layout

- `src/dcase_adqa/`: manifest preparation, diagnostic evaluation, SFT data construction, Qwen3-Omni inference, judging, and submission CSV utilities.
- `scripts/`: thin end-to-end wrappers for data preparation, diagnostics, SFT data generation, evaluation, and submission packaging.
- `external_overrides/Fun-Audio-Chat/`: minimal patch/template files used with a local Fun-Audio-Chat checkout.
- `paper/`: DCASE technical report LaTeX source.
- `submission/`: metadata and post-processing files for DCASE system submission.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

```bash
DCASE_TASK5_ROOT=/path/to/dcase2026_task5
QWEN3_OMNI_MODEL=Qwen/Qwen3-Omni-30B-A3B-Instruct
QWEN35_JUDGE_MODEL=Qwen/Qwen3-8B
FUN_AUDIO_CHAT_ROOT=/path/to/Fun-Audio-Chat
OUTPUT_ROOT=outputs
```

The scripts source `.env` automatically when present.

## Reproduction Workflow

```bash
scripts/00_prepare_manifests.sh
scripts/01_run_diagnostics.sh
scripts/02_build_sft_data.sh
scripts/03_train_systems.sh
scripts/04_eval_and_package.sh
```

The training wrapper expects a local Fun-Audio-Chat checkout with the Qwen3-Omni multimodal SFT patch applied. It writes dataset registration and config files into that checkout, then launches its training entry point.

## Final Systems

The submitted systems are Qwen3-Omni LoRA variants trained on audio-dependency-filtered data. The strongest training buckets were selected by comparing base-model behavior under normal audio, empty audio, and shuffled-audio diagnostics. Final CSVs are produced from model predictions using exact MCQ choice normalization, optional judge-based parse repair, and ensemble tie-breaking.

## Citation

If this repository or the technical report is useful, cite:

```bibtex
@article{nam2026learning,
  title = {Learning from Audio-Dependency Errors: Data Curation Strategies Based on Model Confusion Patterns in Audio Question Answering},
  author = {Nam, Hyeonuk},
  journal = {arXiv preprint arXiv:2606.22276},
  year = {2026},
  url = {https://arxiv.org/abs/2606.22276}
}
```

## Release Notes

- Do not commit datasets, local checkpoints, or generated predictions.
- Technical report is available on arXiv: https://arxiv.org/abs/2606.22276
- Public adapters are redirected to Hugging Face: https://huggingface.co/frednamfred/adqa_nam_qwen3_omni_lora
- Dataset access should redirect users to the official DCASE/Hugging Face dataset pages.
- The code is released under the repository license; model adapters inherit the restrictions of the base model and challenge data terms.
