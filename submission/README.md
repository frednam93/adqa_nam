# DCASE 2026 Task 5 Submission Draft

This directory contains draft metadata and package-building helpers for the planned final submissions.

Planned label prefix: `Nam_IND_task5`

Systems:
- `Nam_IND_task5_1`: train-only strong empty-5% SFT checkpoint 2000, base Qwen3 judge post-processing.
- `Nam_IND_task5_2`: train+dev strong empty-5% SFT checkpoint 2000, base Qwen3 judge post-processing.
- `Nam_IND_task5_3`: train+dev strong empty-2.5% SFT checkpoint 3000, base Qwen3 judge post-processing.
- `Nam_IND_task5_4`: ensemble of systems 2 and 3, tie broken by system 2, base Qwen3 judge post-processing.

Before final submission:
- Fill author name, email, and affiliation TODO fields in `*.meta.yaml`.
- Build `paper/main.tex` into `Nam_IND_task5.technical_report.pdf`.
- Run `scripts/build_submission_package.sh` after all final eval CSV files exist.
