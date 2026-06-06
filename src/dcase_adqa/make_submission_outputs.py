from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(rows)


def clean_prediction(row: dict, item: dict, fallback_index: int = 0) -> tuple[int, str, str]:
    choices = item["choices"]
    idx = row.get("judge_prediction_index", row.get("prediction_index", -1))
    if isinstance(idx, int) and 0 <= idx < len(choices):
        return idx, choices[idx], "parsed"
    pred = str(row.get("judge_prediction", row.get("prediction", ""))).strip()
    for i, choice in enumerate(choices):
        if pred.lower() == choice.lower():
            return i, choice, "exact_text"
    return fallback_index, choices[fallback_index], "fallback"


def single(pred: Path, manifest: Path, output_csv: Path, output_jsonl: Path | None) -> dict:
    preds = {row["id"]: row for row in load_jsonl(pred)}
    items = load_jsonl(manifest)
    csv_rows = []
    json_rows = []
    methods = Counter()
    for item in items:
        row = preds[item["id"]]
        idx, text, method = clean_prediction(row, item)
        methods[method] += 1
        csv_rows.append({"question": item["id"], "answer": text})
        json_rows.append({"id": item["id"], "prediction_index": idx, "prediction": text, "method": method})
    write_csv(output_csv, csv_rows)
    if output_jsonl is not None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in json_rows) + "\n", encoding="utf-8")
    return {"output_csv": str(output_csv), "n": len(csv_rows), "methods": dict(methods)}


def ensemble(preds: list[Path], names: list[str], manifest: Path, output_csv: Path, output_jsonl: Path, tie_breaker: str) -> dict:
    pred_maps = [{row["id"]: row for row in load_jsonl(path)} for path in preds]
    items = load_jsonl(manifest)
    if tie_breaker not in names:
        raise ValueError(f"tie_breaker must be one of {names}")
    tie_idx = names.index(tie_breaker)
    csv_rows = []
    json_rows = []
    stats = Counter()
    for item in items:
        votes = []
        for pred_map in pred_maps:
            idx, text, method = clean_prediction(pred_map[item["id"]], item)
            votes.append((idx, text, method))
        counts = Counter(idx for idx, _, _ in votes if idx >= 0)
        if counts and counts.most_common(1)[0][1] >= 2:
            idx = counts.most_common(1)[0][0]
            reason = "majority"
        else:
            idx = votes[tie_idx][0]
            reason = f"tie_break_{tie_breaker}"
        answer = item["choices"][idx]
        stats[reason] += 1
        csv_rows.append({"question": item["id"], "answer": answer})
        json_rows.append(
            {
                "id": item["id"],
                "prediction_index": idx,
                "prediction": answer,
                "reason": reason,
                "votes": {name: {"index": vote[0], "answer": vote[1], "method": vote[2]} for name, vote in zip(names, votes)},
            }
        )
    write_csv(output_csv, csv_rows)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in json_rows) + "\n", encoding="utf-8")
    return {"output_csv": str(output_csv), "output_jsonl": str(output_jsonl), "n": len(csv_rows), "stats": dict(stats)}


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    p_single = sub.add_parser("single")
    p_single.add_argument("--pred", type=Path, required=True)
    p_single.add_argument("--manifest", type=Path, required=True)
    p_single.add_argument("--output-csv", type=Path, required=True)
    p_single.add_argument("--output-jsonl", type=Path, default=None)

    p_ens = sub.add_parser("ensemble")
    p_ens.add_argument("--pred", type=Path, action="append", required=True)
    p_ens.add_argument("--name", action="append", required=True)
    p_ens.add_argument("--manifest", type=Path, required=True)
    p_ens.add_argument("--output-csv", type=Path, required=True)
    p_ens.add_argument("--output-jsonl", type=Path, required=True)
    p_ens.add_argument("--tie-breaker", default="B")
    args = p.parse_args()

    if args.cmd == "single":
        summary = single(args.pred, args.manifest, args.output_csv, args.output_jsonl)
    else:
        if len(args.pred) != len(args.name):
            raise ValueError("--pred and --name counts must match")
        summary = ensemble(args.pred, args.name, args.manifest, args.output_csv, args.output_jsonl, args.tie_breaker)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
