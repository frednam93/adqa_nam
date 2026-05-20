from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are an audio question answering assistant. "
    "Listen to the audio and answer with only the exact option text."
)
SILENT_COT_PROMPT = (
    "You are an audio question answering assistant. "
    "Listen to the audio, reason internally, and answer with only the exact option text."
)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_user_prompt(item: dict, cot_mode: str) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(
        f"({labels[i]}) {choice}" for i, choice in enumerate(item["choices"])
    )
    final_instruction = "Answer with only the exact option text."
    if cot_mode == "cot_answer":
        final_instruction = (
            "Use the provided reasoning supervision during training, then give the final "
            "answer as the exact option text."
        )
    return (
        "<audio>\n"
        f"{item['question']}\n"
        "Choose the correct option from the following options:\n"
        f"{options}\n"
        f"{final_instruction}"
    )


def build_assistant_target(item: dict, cot_mode: str) -> str:
    if cot_mode != "cot_answer":
        return item["answer"]
    cot = (item.get("gemini_cot") or "").strip()
    if not cot:
        return item["answer"]
    return f"<think>\n{cot}\n</think>\n{item['answer']}"


def convert_item(item: dict, cot_mode: str) -> dict:
    system_prompt = SILENT_COT_PROMPT if cot_mode == "silent_prompt" else SYSTEM_PROMPT
    return {
        "system": system_prompt,
        "messages": [
            {"role": "user", "content": build_user_prompt(item, cot_mode)},
            {"role": "assistant", "content": build_assistant_target(item, cot_mode)},
        ],
        "audios": [item["audio"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--cot-mode",
        choices=("answer_only", "silent_prompt", "cot_answer"),
        default="answer_only",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.manifest, args.limit)
    with args.output.open("w", encoding="utf-8") as f:
        for item in rows:
            f.write(json.dumps(convert_item(item, args.cot_mode), ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
