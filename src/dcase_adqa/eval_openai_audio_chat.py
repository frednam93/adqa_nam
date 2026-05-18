from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


SYSTEM_PROMPT = (
    "You are an audio question answering assistant. "
    "Listen to the audio and answer with only the exact option text."
)


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_question(item: dict) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options = "\n".join(
        f"({labels[i]}) {choice}" for i, choice in enumerate(item["choices"])
    )
    return (
        f"{item['question']}\n"
        "Choose the correct option from the following options:\n"
        f"{options}\n"
        "Answer with only the exact option text."
    )


def audio_part(path: str, mode: str) -> dict:
    if mode == "file_url":
        return {"type": "audio_url", "audio_url": {"url": f"file://{path}"}}
    if mode == "base64":
        suffix = Path(path).suffix.lstrip(".") or "wav"
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "input_audio", "input_audio": {"data": data, "format": suffix}}
    if mode == "data_url":
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return {"type": "audio_url", "audio_url": {"url": f"data:{mime};base64,{data}"}}
    raise ValueError(f"unknown audio mode: {mode}")


def choose_prediction(generation: str, choices: list[str]) -> tuple[int, str]:
    text = generation.strip()
    lowered = text.lower()
    for i, choice in enumerate(choices):
        if lowered == choice.lower():
            return i, choice
    for i, choice in enumerate(choices):
        if choice.lower() in lowered:
            return i, choice
    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(choices)]):
        candidates = (f"({letter.lower()})", f"{letter.lower()}.", letter.lower())
        if lowered.startswith(candidates):
            return i, choices[i]
    return -1, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="nemotron_3_nano_omni")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audio-mode", choices=("file_url", "base64", "data_url"), default="file_url")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    correct = 0
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        audio_part(item["audio"], args.audio_mode),
                        {"type": "text", "text": build_question(item)},
                    ],
                },
            ]
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            generation = response.choices[0].message.content or ""
            pred_index, pred_text = choose_prediction(generation, item["choices"])
            is_correct = pred_index == item["answer_index"]
            correct += int(is_correct)
            total += 1
            out.write(
                json.dumps(
                    {
                        "id": item["id"],
                        "answer": item["answer"],
                        "answer_index": item["answer_index"],
                        "prediction": pred_text,
                        "prediction_index": pred_index,
                        "correct": is_correct,
                        "raw_generation": generation,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    print(f"accuracy={correct / max(total, 1):.4f} correct={correct} total={total}")


if __name__ == "__main__":
    main()
