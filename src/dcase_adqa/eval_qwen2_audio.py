from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_messages(item: dict) -> list[dict]:
    choices = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(item["choices"]))
    prompt = (
        "Listen to the audio and answer the multiple-choice question. "
        "Return only the exact answer text, not the letter.\n\n"
        f"Question: {item['question']}\n"
        f"Choices:\n{choices}\n"
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": item["audio"]},
                {"type": "text", "text": prompt},
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
    for i, letter in enumerate("ABCD"[: len(choices)]):
        if lowered.startswith(letter.lower()):
            return i, choices[i]
    return -1, text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2-Audio-7B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    rows = load_jsonl(args.manifest, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows):
            messages = build_messages(item)
            text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            audios = [
                librosa.load(item["audio"], sr=processor.feature_extractor.sampling_rate)[0]
            ]
            inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            generation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

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
