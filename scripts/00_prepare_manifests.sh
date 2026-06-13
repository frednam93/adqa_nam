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

python -m dcase_adqa.prepare_manifests \
  --data-root "$DCASE_TASK5_ROOT" \
  --out-dir "$DCASE_TASK5_ROOT/manifests"
