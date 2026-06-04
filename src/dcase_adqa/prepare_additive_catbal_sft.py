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


def draw_mixture(rng: random.Random, base_rows: list[dict], catbal_rows: list[dict], total: int, catbal_ratio: float) -> tuple[list[dict], int]:
    rows = []
    catbal_count = 0
    for _ in range(total):
        if rng.random() < catbal_ratio:
            rows.append(rng.choice(catbal_rows))
            catbal_count += 1
        else:
            rows.append(rng.choice(base_rows))
    rng.shuffle(rows)
    return rows, catbal_count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full/strong_empty_unknown5/train.jsonl"))
    p.add_argument("--catbal", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_rebalanced/rebalanced_v3_catbal_v2_empty5/train.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_additive_catbal"))
    p.add_argument("--ratios", default="0.10,0.20,0.30")
    p.add_argument("--total-samples", type=int, default=24000, help="Pre-sampled rows for 3000 steps at batch size 8.")
    p.add_argument("--seed", type=int, default=20260605)
    args = p.parse_args()

    base_rows = load_jsonl(args.base)
    catbal_rows = load_jsonl(args.catbal)
    summary = {}
    for ratio_text in [x.strip() for x in args.ratios.split(",") if x.strip()]:
        ratio = float(ratio_text)
        pct = int(round(ratio * 100))
        rng = random.Random(args.seed + pct)
        rows, actual_catbal = draw_mixture(rng, base_rows, catbal_rows, args.total_samples, ratio)
        name = f"strong_empty5_catbalmix{pct}"
        write_jsonl(args.out_dir / name / "train.jsonl", rows)
        summary[name] = {
            "base_pool_count": len(base_rows),
            "catbal_pool_count": len(catbal_rows),
            "total_samples": len(rows),
            "target_catbal_ratio": ratio,
            "sampled_catbal_count": actual_catbal,
            "sampled_catbal_ratio": actual_catbal / max(len(rows), 1),
            "sampling": "with_replacement",
        }
        print(f"{name}: total={len(rows)} catbal={actual_catbal} ratio={actual_catbal/max(len(rows),1):.3f}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
