from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor


REPO_ROOT = Path("/home/user/ssdmain/dcase-adqa/external/Fun-Audio-Chat")
sys.path.insert(0, str(REPO_ROOT))

from funaudiochat.register import register_funaudiochat  # noqa: E402
from utils.constant import AUDIO_TEMPLATE  # noqa: E402


SYSTEM_PROMPT = "You are asked to generate text tokens."


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
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
        f"{item['question']} Choose the correct option from the following options:\n"
        f"{options}"
    )


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
    parser.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/fun_audio_chat_8b"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    register_funaudiochat()
    rows = load_jsonl(args.manifest, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model)
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.sp_gen_kwargs.update({"text_greedy": True, "disable_speech": True})
    model.eval()

    correct = 0
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows):
            audio = [librosa.load(item["audio"], sr=16000)[0]]
            conversation = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": AUDIO_TEMPLATE + "\n" + build_instruction(item)},
            ]
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            inputs = processor(
                text=text,
                audio=audio,
                return_tensors="pt",
                return_token_type_ids=False,
            ).to(model.device)

            with torch.inference_mode():
                generate_ids, _ = model.generate(**inputs)
            generate_ids = generate_ids[:, inputs.input_ids.size(1):]
            generation = processor.decode(generate_ids[0], skip_special_tokens=True)

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
