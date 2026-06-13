#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
: "${DCASE_TASK5_ROOT:?Set DCASE_TASK5_ROOT in .env or the environment}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
: "${FUN_AUDIO_CHAT_ROOT:?Set FUN_AUDIO_CHAT_ROOT}"
if [[ ! -d "$FUN_AUDIO_CHAT_ROOT" ]]; then
  echo "FUN_AUDIO_CHAT_ROOT does not exist: $FUN_AUDIO_CHAT_ROOT" >&2
  exit 1
fi
echo "Apply external_overrides/Fun-Audio-Chat/patches/qwen3_omni_dcase.patch to Fun-Audio-Chat before training."
echo "Then run the generated YAML files under $FUN_AUDIO_CHAT_ROOT/training/configs with the Fun-Audio-Chat training launcher."
