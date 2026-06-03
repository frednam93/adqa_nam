from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from dcase_adqa.prepare_qwen3_omni_sft import convert_item, load_jsonl

UNKNOWN_TARGET = "Cannot be determined from the audio."
UNKNOWN_SYSTEM = (
    "You are an audio question answering assistant. Listen to the audio and answer with only the exact option text. "
    "If the provided audio is missing or does not contain the information needed to answer the question, answer: "
    f"{UNKNOWN_TARGET}"
)
CORE_CATEGORIES = {
    "speaker_identity",
    "sound_event",
    "temporal_reasoning",
    "speech_paralinguistic",
    "speech_content",
    "music",
    "counting_quantity",
    "acoustic_property",
    "scene_context",
    "other",
}


def read_buckets(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_unknown_item(item: dict) -> dict:
    converted = convert_item(item, "answer_only")
    converted["system"] = UNKNOWN_SYSTEM
    converted["messages"][-1]["content"] = UNKNOWN_TARGET
    converted["audios"] = []
    converted["messages"][0]["content"] = converted["messages"][0]["content"].replace("<audio>\n", "")
    return converted


def sample_ids(rng: random.Random, ids: list[str], n: int) -> list[str]:
    if n <= 0 or not ids:
        return []
    if n <= len(ids):
        return rng.sample(ids, n)
    return [rng.choice(ids) for _ in range(n)]


def bucket_index(rows: list[dict[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        out[row["bucket"]].append(row["id"])
    return out


def category_index(rows: list[dict[str, str]], ids: list[str]) -> dict[str, list[str]]:
    allowed = set(ids)
    out: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        if row["id"] in allowed:
            out[row.get("category") or "other"].append(row["id"])
    return out


def build_category_balanced_ids(rng: random.Random, rows: list[dict[str, str]], base_ids: list[str], target_total: int) -> list[str]:
    by_cat = category_index(rows, base_ids)
    cats = [c for c in CORE_CATEGORIES if by_cat.get(c)]
    if not cats:
        return base_ids[:]
    per_cat = max(1, round(target_total / len(cats)))
    selected: list[str] = []
    for cat in sorted(cats):
        selected.extend(sample_ids(rng, by_cat[cat], per_cat))
    # Trim deterministically after shuffle if overshot.
    rng.shuffle(selected)
    return selected[:target_total]


def build_variant(rng: random.Random, train_by_id: dict[str, dict], positive_ids: list[str], empty_ratio: float) -> list[dict]:
    positives = [convert_item(train_by_id[sid], "answer_only") for sid in positive_ids if sid in train_by_id]
    n_empty = round(len(positives) * empty_ratio)
    unknowns = [make_unknown_item(train_by_id[sid]) for sid in sample_ids(rng, positive_ids, n_empty) if sid in train_by_id]
    rows = positives + unknowns
    rng.shuffle(rows)
    return rows


def summarize(name: str, ids: list[str], empty_ratio: float, rows_by_id: dict[str, dict[str, str]]) -> dict:
    cats = Counter((rows_by_id[sid].get("category") or "other") for sid in ids if sid in rows_by_id)
    buckets = Counter(rows_by_id[sid]["bucket"] for sid in ids if sid in rows_by_id)
    return {
        "name": name,
        "positive_count": len(ids),
        "empty_count": round(len(ids) * empty_ratio),
        "total_count": len(ids) + round(len(ids) * empty_ratio),
        "buckets": dict(buckets.most_common()),
        "categories": dict(cats.most_common()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/manifests/train.jsonl"))
    p.add_argument("--bucket-csv", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/analysis/audio_dependency_full/train_full_audio_dependency_buckets.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_rebalanced"))
    p.add_argument("--easy-ratio", type=float, default=0.30)
    p.add_argument("--empty-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20260603)
    args = p.parse_args()

    rng = random.Random(args.seed)
    train_by_id = {row["id"]: row for row in load_jsonl(args.train_manifest)}
    bucket_rows = read_buckets(args.bucket_csv)
    rows_by_id = {row["id"]: row for row in bucket_rows}
    buckets = bucket_index(bucket_rows)

    strong = buckets.get("strong_audio_dependent", [])
    hard = buckets.get("hard_audio_dependent_candidate", [])
    leak = buckets.get("audio_helped_but_shuffle_leak", [])
    easy = buckets.get("easy_text_prior", [])

    v1_core = strong + hard
    v1_easy = sample_ids(rng, easy, round(len(v1_core) * args.easy_ratio))
    v1 = v1_core + v1_easy

    v2_core = strong + hard + leak
    v2_easy = sample_ids(rng, easy, round(len(v2_core) * args.easy_ratio))
    v2 = v2_core + v2_easy

    v3_pool = v2_core + v2_easy
    v3 = build_category_balanced_ids(rng, bucket_rows, v3_pool, len(v2))

    variants = {
        "rebalanced_v1_strong_hard_easy30_empty5": v1,
        "rebalanced_v2_strong_hard_leak_easy30_empty5": v2,
        "rebalanced_v3_catbal_v2_empty5": v3,
    }

    summary = {}
    for name, ids in variants.items():
        out = build_variant(rng, train_by_id, ids, args.empty_ratio)
        write_jsonl(args.out_dir / name / "train.jsonl", out)
        summary[name] = summarize(name, ids, args.empty_ratio, rows_by_id)
        print(f"{name}: positives={summary[name]['positive_count']} empty={summary[name]['empty_count']} total={summary[name]['total_count']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
