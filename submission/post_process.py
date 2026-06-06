"""Post-processing summary for DCASE 2026 Task 5 submissions.

The actual implementation used by this repository is in:
  src/dcase_adqa/make_submission_outputs.py
  src/dcase_adqa/judge_eval_parsebad.py

For each prediction JSONL, the parser first accepts exact option-text matches
or obvious option-letter prefixes. For parse failures, selected systems use the
base Qwen3-Omni model as a text-only response normalizer over the candidate
choices. Ensemble systems vote over normalized choice indices and use the
declared tie-breaker when the vote is tied.
"""
