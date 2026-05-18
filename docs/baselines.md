# Baseline Notes

## Official Baselines To Reproduce First

- `Qwen/Qwen3-Omni-30B-A3B-Instruct`: official best reported dev baseline, 0.6248 Top-1 accuracy.
- `Fun-Audio-Chat`: official reported dev baseline, 0.5681 Top-1 accuracy.
- `MiMo-Audio`: official reported dev baseline, 0.5457 Top-1 accuracy.
- `Step-Audio 2 Mini`: official reported dev baseline, 0.5053 Top-1 accuracy.
- `Kimi-Audio`: official reported dev baseline, 0.4636 Top-1 accuracy.

## Local Sanity Check Only

- `Qwen/Qwen2-Audio-7B-Instruct` is not a DCASE 2026 reported baseline. It remains useful only for testing manifest loading, audio decoding, prompt formatting, and answer extraction with a standard Hugging Face model.

## First Experiments

1. Reproduce Qwen3-Omni dev subset if it fits locally or via quantized/offloaded inference.
2. Reproduce Fun-Audio-Chat because it is below the 10B lightweight threshold and has official code.
3. Use Qwen2-Audio only if the official models are blocked, and label it clearly as a harness smoke test.
4. Add answer-choice permutation robustness after one official baseline is running.

## Extra Open ALM Targets

- `nvidia/audio-flamingo-3-hf`: highest-priority non-DCASE open ALM target. Requires a newer Transformers build than the FunAudio/Qwen3 env, so use `dcase_open_alm`.
- `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`: newest 30B-A3B open omni target. Use vLLM OpenAI-compatible serving first; direct Python eval is not the preferred path.
- `Qwen3.5-Omni`: technical-report-only for now. Keep as a placeholder until public/local weights exist.

Setup:

```bash
bash scripts/install_open_alm_deps.sh
bash scripts/download_open_alm_models.sh
```

Eval:

```bash
bash scripts/eval_audioflamingo3_dev.sh --limit 20
bash scripts/serve_nemotron3_nano_omni_vllm.sh
bash scripts/eval_nemotron3_nano_omni_dev.sh --limit 20
```
