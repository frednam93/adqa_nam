from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(rng: random.Random, rows: list[dict], n: int) -> list[dict]:
    if n <= len(rows):
        return rng.sample(rows, n)
    return [rng.choice(rows) for _ in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full/strong_empty_unknown5/train.jsonl"))
    p.add_argument("--catbal", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_rebalanced/rebalanced_v3_catbal_v2_empty5/train.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_additive_catbal"))
    p.add_argument("--ratios", default="0.10,0.20,0.30")
    p.add_argument("--seed", type=int, default=20260605)
    args = p.parse_args()

    base_rows = load_jsonl(args.base)
    catbal_rows = load_jsonl(args.catbal)
    summary = {}
    for ratio_text in [x.strip() for x in args.ratios.split(",") if x.strip()]:
        ratio = float(ratio_text)
        pct = int(round(ratio * 100))
        rng = random.Random(args.seed + pct)
        n_extra = round(len(base_rows) * ratio)
        rows = list(base_rows) + sample_rows(rng, catbal_rows, n_extra)
        rng.shuffle(rows)
        name = f"strong_empty5_catbal{pct}"
        write_jsonl(args.out_dir / name / "train.jsonl", rows)
        summary[name] = {
            "base_count": len(base_rows),
            "catbal_added": n_extra,
            "total_count": len(rows),
            "catbal_ratio_of_base": ratio,
        }
        print(f"{name}: base={len(base_rows)} add={n_extra} total={len(rows)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
