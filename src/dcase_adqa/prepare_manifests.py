from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset


DATASETS = {
    "train": ("Harland/AudioMCQ-StrongAC-GeminiCoT", "train"),
    "dev": ("Harland/DCASE2026-Task5-DevSet", "dev"),
    "eval": ("Harland/ADQA-Bench", "eval"),
}


def normalize(split: str, row: dict, audio_root: Path) -> dict:
    if split == "train":
        question = row["question"]
        choices = row["choices"]
    else:
        question = row["question_text"]
        choices = row["multi_choice"]

    choices = list(choices)
    answer = row.get("answer")
    item = {
        "id": str(row["id"]),
        "split": split,
        "audio": str(audio_root / row["audio_path"]),
        "question": question,
        "choices": choices,
        "answer": answer,
        "answer_index": choices.index(answer) if answer in choices else -1,
        "source_dataset": row.get("source_dataset"),
        "question_type": row.get("question_type"),
    }
    if row.get("gemini_cot"):
        item["gemini_cot"] = row["gemini_cot"]
    return item


def write_manifest(split: str, repo: str, hf_split: str, data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    local_root = data_root / split
    audio_root = local_root

    ds = load_dataset(str(local_root) if local_root.exists() else repo, split=hf_split)
    out_path = out_dir / f"{split}.jsonl"

    n = 0
    missing = 0
    bad_answer = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            item = normalize(split, row, audio_root)
            if not Path(item["audio"]).exists():
                missing += 1
            if item["answer_index"] < 0:
                bad_answer += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1

    print(f"{split}: wrote={n} missing_audio={missing} bad_answer={bad_answer} path={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/manifests"))
    args = parser.parse_args()

    for split, (repo, hf_split) in DATASETS.items():
        write_manifest(split, repo, hf_split, args.data_root, args.out_dir)


if __name__ == "__main__":
    main()
