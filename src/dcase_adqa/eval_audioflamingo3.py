from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor

try:
    from transformers import AudioFlamingo3ForConditionalGeneration
except ImportError as exc:  # pragma: no cover - this is an environment guard.
    raise SystemExit(
        "Audio Flamingo 3 requires a Transformers build with "
        "AudioFlamingo3ForConditionalGeneration. Install the open-ALM env with "
        "`scripts/install_open_alm_deps.sh` first."
    ) from exc


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
        f"{SYSTEM_PROMPT}\n"
        f"{item['question']}\n"
        "Choose the correct option from the following options:\n"
        f"{options}\n"
        "Answer with only the exact option text."
    )


def build_conversation(item: dict) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_question(item)},
                {"type": "audio", "path": item["audio"]},
            ],
        }
    ]


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


def move_inputs(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        moved[key] = value
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="/home/user/ssdmain/models/dcase_adqa/audio_flamingo_3_hf",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16"), default="bfloat16")
    args = parser.parse_args()

    dtype = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.dtype]
    rows = load_jsonl(args.manifest, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.bfloat16)

    correct = 0
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows):
            conversation = build_conversation(item)
            inputs = processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            inputs = move_inputs(inputs, model_device, model_dtype)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[:, inputs["input_ids"].shape[1] :]
            generation = processor.decode(generated[0], skip_special_tokens=True)

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
