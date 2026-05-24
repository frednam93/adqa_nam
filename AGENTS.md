# Repository Guidelines

## Project Structure & Module Organization

This repository contains DCASE ADQA data preparation, evaluation, analysis, and fine-tuning utilities. Core Python modules live in `src/dcase_adqa/`. Long-running launchers and tmux/watch scripts live in `scripts/`. Generated predictions, logs, adapters, and analysis outputs live under `outputs/` and are ignored by git. External model/training repos are under `external/` or `external_overrides/`.

## Build, Test, and Development Commands

- `pip install -e .`: install the local package in editable mode.
- `python3 -m py_compile src/dcase_adqa/*.py`: syntax-check edited Python modules.
- `bash -n scripts/<script>.sh`: validate shell launchers before starting queues.
- `PYTHONPATH=src python3 -m dcase_adqa.notify_telegram --title "test" --message "ok"`: test Telegram notifications.

Most heavy jobs require the intended conda environment, typically `FunAudioChat`, and should be launched in tmux.

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, snake_case functions/variables, and explicit `Path`-based file handling. Keep scripts deterministic and resumable when possible. Do not hard-code local secrets, bot tokens, or chat IDs.

## Testing Guidelines

There is no formal unit-test suite. For any source edit, run `py_compile`; for script edits, run `bash -n`. For long evaluation or fine-tuning queues, run or reuse a representative smoke first when model/data loading behavior changed.

## Agent-Specific Instructions

Before touching long-running jobs, inspect active tmux panes and processes. Do not assume a tmux pane index exists; verify with `tmux list-panes -a`. Do not interrupt training/eval unless the user asks or a process is clearly broken.

For long-running experiments, queued runs, or watchers, use `/home/user/.local/bin/codex-notify --title "..." --message "..."` to send concise Telegram updates when a major stage finishes, a run fails, or final metrics are available. Credentials live in `/home/user/.config/codex/telegram.env`; never print or commit them. If the user asks to add notifications, use the `telegram-notify` skill.

Keep generated outputs under `outputs/` or dataset-specific roots. Do not commit generated predictions, checkpoints, logs, or external repos.
