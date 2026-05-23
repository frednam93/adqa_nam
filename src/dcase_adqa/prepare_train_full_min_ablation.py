from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from dcase_adqa.analyze_errors import load_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffled_audio_random(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    n = len(rows)
    for i, row in enumerate(rows):
        j = rng.randrange(n - 1)
        if j >= i:
            j += 1
        donor = rows[j]
        item = dict(row)
        item["source_audio"] = row["audio"]
        item["audio"] = donor["audio"]
        item["donor_id"] = donor["id"]
        item["ablation"] = "shuffle_audio_random"
        out.append(item)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/manifests/train.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests"))
    p.add_argument("--seed", type=int, default=20260523)
    args = p.parse_args()
    rows = load_jsonl(args.train_manifest)
    rng = random.Random(args.seed)
    write_jsonl(args.out_dir / "train_full_normal.jsonl", rows)
    write_jsonl(args.out_dir / "train_full_shuffle_audio_random.jsonl", shuffled_audio_random(rows, rng))
    print(f"train_full={len(rows)}")
    print(args.out_dir / "train_full_normal.jsonl")
    print(args.out_dir / "train_full_shuffle_audio_random.jsonl")


if __name__ == "__main__":
    main()
