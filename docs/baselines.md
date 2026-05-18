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
