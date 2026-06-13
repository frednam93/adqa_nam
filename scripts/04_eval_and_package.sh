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
: "${QWEN3_OMNI_MODEL:?Set QWEN3_OMNI_MODEL}"
: "${SYSTEM_A_ADAPTER:?Set SYSTEM_A_ADAPTER}"
: "${SYSTEM_B_ADAPTER:?Set SYSTEM_B_ADAPTER}"
: "${SYSTEM_C_ADAPTER:?Set SYSTEM_C_ADAPTER}"
mkdir -p "$OUTPUT_ROOT/final_submission/preds"
for sys in A B C; do
  adapter_var="SYSTEM_${sys}_ADAPTER"
  adapter="${!adapter_var}"
  python -m dcase_adqa.eval_qwen3_omni \
    --manifest "$DCASE_TASK5_ROOT/manifests/eval.jsonl" \
    --model "$QWEN3_OMNI_MODEL" \
    --adapter "$adapter" \
    --output "$OUTPUT_ROOT/final_submission/preds/system_${sys}.jsonl" \
    --prompt-mode letter_only
done
python -m dcase_adqa.make_submission_outputs single \
  --pred "$OUTPUT_ROOT/final_submission/preds/system_A.jsonl" \
  --manifest "$DCASE_TASK5_ROOT/manifests/eval.jsonl" \
  --output-csv "$OUTPUT_ROOT/final_submission/Nam_IND_task5_1/output.csv" \
  --output-jsonl "$OUTPUT_ROOT/final_submission/Nam_IND_task5_1/output.jsonl"
python -m dcase_adqa.make_submission_outputs single \
  --pred "$OUTPUT_ROOT/final_submission/preds/system_B.jsonl" \
  --manifest "$DCASE_TASK5_ROOT/manifests/eval.jsonl" \
  --output-csv "$OUTPUT_ROOT/final_submission/Nam_IND_task5_2/output.csv" \
  --output-jsonl "$OUTPUT_ROOT/final_submission/Nam_IND_task5_2/output.jsonl"
python -m dcase_adqa.make_submission_outputs single \
  --pred "$OUTPUT_ROOT/final_submission/preds/system_C.jsonl" \
  --manifest "$DCASE_TASK5_ROOT/manifests/eval.jsonl" \
  --output-csv "$OUTPUT_ROOT/final_submission/Nam_IND_task5_3/output.csv" \
  --output-jsonl "$OUTPUT_ROOT/final_submission/Nam_IND_task5_3/output.jsonl"
python -m dcase_adqa.make_submission_outputs ensemble \
  --pred "$OUTPUT_ROOT/final_submission/preds/system_B.jsonl" --name B \
  --pred "$OUTPUT_ROOT/final_submission/preds/system_C.jsonl" --name C \
  --manifest "$DCASE_TASK5_ROOT/manifests/eval.jsonl" \
  --output-csv "$OUTPUT_ROOT/final_submission/Nam_IND_task5_4/output.csv" \
  --output-jsonl "$OUTPUT_ROOT/final_submission/Nam_IND_task5_4/output.jsonl" \
  --tie-breaker B
