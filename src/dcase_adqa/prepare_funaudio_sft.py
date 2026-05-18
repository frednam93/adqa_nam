from __future__ import annotations

import argparse
import json
from pathlib import Path


AUDIO_PLACEHOLDER = "<|audio_bos|><|AUDIO|><|audio_eos|>"
SYSTEM_PROMPT = "You are asked to generate text tokens."


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_instruction(item: dict) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(
        f"({labels[i]}) {choice}" for i, choice in enumerate(item["choices"])
    )
    return (
        f"{AUDIO_PLACEHOLDER}\n"
        f"{item['question']} Choose the correct option from the following options:\n"
        f"{options}"
    )


def convert_item(item: dict) -> dict:
    return {
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": build_instruction(item)},
            {"role": "assistant", "content": item["answer"]},
        ],
        "audio": [
            json.dumps(
                {
                    "path": item["audio"],
                    "token": AUDIO_PLACEHOLDER,
                    "text": "",
                },
                ensure_ascii=False,
            )
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.manifest, args.limit)
    with args.output.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(convert_item(item), ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
