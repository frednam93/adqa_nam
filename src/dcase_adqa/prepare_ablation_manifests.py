from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from dcase_adqa.analyze_errors import classify, load_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def shuffled_choices(rows: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        choices = list(item["choices"])
        order = list(range(len(choices)))
        rng.shuffle(order)
        item["choices"] = [choices[i] for i in order]
        item["answer_index"] = order.index(row["answer_index"])
        item["answer"] = item["choices"][item["answer_index"]]
        item["choice_order"] = order
        item["ablation"] = "choice_shuffle"
        out.append(item)
    return out


def shuffled_audio(rows: list[dict], rng: random.Random, mode: str) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[classify(row)[0]].append(row)
    all_rows = list(rows)
    out = []
    for row in rows:
        cat = classify(row)[0]
        if mode == "random":
            pool = [r for r in all_rows if r["id"] != row["id"]]
        elif mode == "same_category":
            pool = [r for r in by_cat[cat] if r["id"] != row["id"]]
            if not pool:
                pool = [r for r in all_rows if r["id"] != row["id"]]
        elif mode == "cross_category":
            pool = [r for r in all_rows if classify(r)[0] != cat]
            if not pool:
                pool = [r for r in all_rows if r["id"] != row["id"]]
        else:
            raise ValueError(mode)
        donor = rng.choice(pool)
        item = dict(row)
        item["source_audio"] = row["audio"]
        item["audio"] = donor["audio"]
        item["donor_id"] = donor["id"]
        item["donor_category"] = classify(donor)[0]
        item["category"] = cat
        item["ablation"] = f"shuffle_audio_{mode}"
        out.append(item)
    return out


def stratified_subset(rows: list[dict], per_category: int, rng: random.Random) -> list[dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cat[classify(row)[0]].append(row)
    selected = []
    for cat in sorted(by_cat):
        bucket = list(by_cat[cat])
        rng.shuffle(bucket)
        selected.extend(bucket[:per_category])
    selected.sort(key=lambda x: x["id"])
    return selected


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dev-manifest", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/manifests/dev.jsonl"))
    p.add_argument("--train-manifest", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/manifests/train.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests"))
    p.add_argument("--train-per-category", type=int, default=80)
    p.add_argument("--seed", type=int, default=20260522)
    args = p.parse_args()

    rng = random.Random(args.seed)
    dev = load_jsonl(args.dev_manifest)
    train = stratified_subset(load_jsonl(args.train_manifest), args.train_per_category, rng)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sets = {"dev_full": dev, f"train_strat{args.train_per_category}": train}
    for prefix, rows in sets.items():
        write_jsonl(args.out_dir / f"{prefix}_normal.jsonl", rows)
        write_jsonl(args.out_dir / f"{prefix}_choice_shuffle.jsonl", shuffled_choices(rows, rng))
        for mode in ("random", "same_category", "cross_category"):
            write_jsonl(args.out_dir / f"{prefix}_shuffle_audio_{mode}.jsonl", shuffled_audio(rows, rng, mode))

    print(f"wrote {args.out_dir}")
    print(f"dev={len(dev)} train_stratified={len(train)}")


if __name__ == "__main__":
    main()
