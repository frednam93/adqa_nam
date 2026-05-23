from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from dcase_adqa.prepare_qwen3_omni_sft import convert_item, load_jsonl

UNKNOWN_TARGET = "Cannot be determined from the audio."
UNKNOWN_SYSTEM = (
    "You are an audio question answering assistant. Listen to the audio and answer with only the exact option text. "
    "If the provided audio is missing or does not contain the information needed to answer the question, answer: "
    f"{UNKNOWN_TARGET}"
)


def read_bucket_csv(path: Path) -> dict[str, str]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = row["bucket"]
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_unknown_item(item: dict, audio: str | None) -> dict:
    converted = convert_item(item, "answer_only")
    converted["system"] = UNKNOWN_SYSTEM
    converted["messages"][-1]["content"] = UNKNOWN_TARGET
    if audio is None:
        converted["audios"] = []
        converted["messages"][0]["content"] = converted["messages"][0]["content"].replace("<audio>\n", "")
    else:
        converted["audios"] = [audio]
    return converted


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/manifests/train.jsonl"))
    p.add_argument("--bucket-csv", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/analysis/audio_dependency/train_strat80_audio_dependency_buckets.csv"))
    p.add_argument("--shuffle-random", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests/train_strat80_shuffle_audio_random.jsonl"))
    p.add_argument("--shuffle-cross", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests/train_strat80_shuffle_audio_cross_category.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep"))
    p.add_argument("--seed", type=int, default=20260523)
    args = p.parse_args()

    rng = random.Random(args.seed)
    train_by_id = {row["id"]: row for row in load_jsonl(args.train_manifest)}
    bucket_by_id = read_bucket_csv(args.bucket_csv)
    shuffle_random_by_id = {row["id"]: row for row in load_jsonl(args.shuffle_random)}
    shuffle_cross_by_id = {row["id"]: row for row in load_jsonl(args.shuffle_cross)}

    strong_ids = [sid for sid,b in bucket_by_id.items() if b == "strong_audio_dependent"]
    hard_ids = [sid for sid,b in bucket_by_id.items() if b == "hard_audio_dependent_candidate"]
    leak_ids = [sid for sid,b in bucket_by_id.items() if b == "audio_helped_but_shuffle_leak"]

    specs = {
        "strong_ac": strong_ids,
        "strong_hard_ac": strong_ids + hard_ids,
        "non_easy_ac": strong_ids + hard_ids + leak_ids,
    }

    made = {}
    for name, ids in specs.items():
        positives = [convert_item(train_by_id[sid], "answer_only") for sid in ids if sid in train_by_id]
        write_jsonl(args.out_dir / name / "train.jsonl", positives)
        made[name] = len(positives)

    # Use non_easy_ac as the default candidate for negative-unknown variants.
    best_ids = specs["non_easy_ac"]
    positives = [convert_item(train_by_id[sid], "answer_only") for sid in best_ids if sid in train_by_id]
    n = len(positives)
    n5 = max(1, round(n * 0.05))
    empty_ids = rng.sample(best_ids, min(n5, len(best_ids)))
    shuffle_ids = rng.sample(best_ids, min(n5, len(best_ids)))

    empty_unknown = positives + [make_unknown_item(train_by_id[sid], None) for sid in empty_ids]
    rng.shuffle(empty_unknown)
    write_jsonl(args.out_dir / "non_easy_empty_unknown5" / "train.jsonl", empty_unknown)
    made["non_easy_empty_unknown5"] = len(empty_unknown)

    shuffle_unknown = positives + [
        make_unknown_item(train_by_id[sid], shuffle_random_by_id.get(sid, shuffle_cross_by_id[sid])["audio"])
        for sid in shuffle_ids
    ]
    rng.shuffle(shuffle_unknown)
    write_jsonl(args.out_dir / "non_easy_shuffle_unknown5" / "train.jsonl", shuffle_unknown)
    made["non_easy_shuffle_unknown5"] = len(shuffle_unknown)

    combo_ids_empty = rng.sample(best_ids, min(n5, len(best_ids)))
    combo_ids_shuffle = rng.sample(best_ids, min(n5, len(best_ids)))
    combo_unknown = positives
    combo_unknown += [make_unknown_item(train_by_id[sid], None) for sid in combo_ids_empty]
    combo_unknown += [
        make_unknown_item(train_by_id[sid], shuffle_cross_by_id.get(sid, shuffle_random_by_id[sid])["audio"])
        for sid in combo_ids_shuffle
    ]
    rng.shuffle(combo_unknown)
    write_jsonl(args.out_dir / "non_easy_empty_shuffle_unknown10" / "train.jsonl", combo_unknown)
    made["non_easy_empty_shuffle_unknown10"] = len(combo_unknown)

    summary = args.out_dir / "summary.json"
    summary.write_text(json.dumps(made, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for k,v in made.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
