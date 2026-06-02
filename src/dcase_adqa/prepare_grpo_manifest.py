from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcase_adqa.analyze_errors import load_jsonl, norm_bool
from dcase_adqa.build_full_audio_dependency_sft import bucket, write_jsonl
from dcase_adqa.prepare_preference_continuation import ABLATION_DIR, TRAIN_MANIFEST


ROOT = Path("/home/user/ssdmain/dcase-adqa")
DEFAULT_OUTPUT = ROOT / "outputs/ablation_manifests/train_strong_audio_dependent_grpo.jsonl"


def select_items(buckets: set[str]) -> list[dict]:
    train = load_jsonl(TRAIN_MANIFEST)
    res = {
        name: {row["id"]: row for row in load_jsonl(ABLATION_DIR / f"{name}.jsonl")}
        for name in ["normal", "empty_audio_question", "shuffle_audio_random"]
    }
    items = []
    for item in train:
        sid = item["id"]
        label = bucket(
            norm_bool(res["normal"][sid].get("correct", False)),
            norm_bool(res["empty_audio_question"][sid].get("correct", False)),
            norm_bool(res["shuffle_audio_random"][sid].get("correct", False)),
        )
        if label in buckets:
            out = dict(item)
            out["audio_dependency_bucket"] = label
            items.append(out)
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--buckets",
        default="strong_audio_dependent",
        help="Comma-separated audio-dependency buckets to include.",
    )
    args = parser.parse_args()

    buckets = {x.strip() for x in args.buckets.split(",") if x.strip()}
    items = select_items(buckets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, items)
    print(json.dumps({"output": str(args.output), "buckets": sorted(buckets), "items": len(items)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
