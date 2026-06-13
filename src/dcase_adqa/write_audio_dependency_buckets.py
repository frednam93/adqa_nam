from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from dcase_adqa.analyze_errors import classify, load_jsonl, norm_bool
from dcase_adqa.build_full_audio_dependency_sft import bucket


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    rows = load_jsonl(args.manifest)
    res = {
        name: {row["id"]: row for row in load_jsonl(args.results_dir / f"{name}.jsonl")}
        for name in ["normal", "empty_audio_question", "shuffle_audio_random"]
    }
    out_rows: list[dict] = []
    counts: Counter[str] = Counter()
    for item in rows:
        sid = item["id"]
        name = bucket(
            norm_bool(res["normal"][sid].get("correct", False)),
            norm_bool(res["empty_audio_question"][sid].get("correct", False)),
            norm_bool(res["shuffle_audio_random"][sid].get("correct", False)),
        )
        cat, conf, evidence = classify(item)
        counts[name] += 1
        out_rows.append(
            {
                "id": sid,
                "bucket": name,
                "category": cat,
                "category_confidence": conf,
                "normal": int(norm_bool(res["normal"][sid].get("correct", False))),
                "empty": int(norm_bool(res["empty_audio_question"][sid].get("correct", False))),
                "shuffle_random": int(norm_bool(res["shuffle_audio_random"][sid].get("correct", False))),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "audio": item.get("audio", ""),
                "category_evidence": "; ".join(evidence),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.output} n={len(out_rows)} counts={dict(counts)}")


if __name__ == "__main__":
    main()
