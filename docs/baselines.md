# Baseline Notes

## Official Baselines To Reproduce First

- `Qwen/Qwen3-Omni-30B-A3B-Instruct`: official best reported dev baseline, 0.6248 Top-1 accuracy.
- `Fun-Audio-Chat`: official reported dev baseline, 0.5681 Top-1 accuracy.
- `MiMo-Audio`: official reported dev baseline, 0.5457 Top-1 accuracy.
- `Step-Audio 2 Mini`: official reported dev baseline, 0.5053 Top-1 accuracy.
- `Kimi-Audio`: official reported dev baseline, 0.4636 Top-1 accuracy.

## Reproduction Notes

- The final systems use Qwen3-Omni LoRA adapters.
- Single-model submissions normalize parse-failed outputs with base Qwen3-Omni.
- The ensemble submission uses Gemma-4-E4B-it as a small text-only normalizer before voting.
- Non-final open ALM probes, RL runs, and queue scripts were removed from the public repository cleanup.

Run the public wrappers from the repository root:

```bash
scripts/00_prepare_manifests.sh
scripts/01_run_diagnostics.sh
scripts/02_build_sft_data.sh
scripts/03_train_systems.sh
scripts/04_eval_and_package.sh
```
