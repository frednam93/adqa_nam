from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from peft import PeftModel
from qwen_omni_utils import process_mm_info
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoProcessor, BitsAndBytesConfig


SYSTEM_PROMPT = (
    "You are an audio question answering assistant. "
    "Listen to the audio and answer with only the exact option text."
)
GENERIC_PROMPT = (
    "Describe the audio in detail. If speech is present, transcribe or summarize the spoken content. "
    "Mention sound events, speakers, music, timing, and acoustic properties when relevant."
)


def move_inputs(inputs, device: torch.device, dtype: torch.dtype):
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            value = value.to(device)
            if value.is_floating_point():
                value = value.to(dtype)
        moved[key] = value
    return moved


def decode_generated_tokens(generated, prompt_len: int):
    if isinstance(generated, tuple):
        generated = generated[0]
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    return generated[:, prompt_len:]


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


def build_user_content(item: dict, prompt_mode: str) -> list[dict]:
    content = []
    if prompt_mode != "text_only":
        content.append({"type": "audio", "audio": item["audio"]})
    if prompt_mode == "audio_only":
        return content
    if prompt_mode == "generic_audio":
        content.append({"type": "text", "text": GENERIC_PROMPT})
    else:
        content.append({"type": "text", "text": build_question(item)})
    return content


def build_conversation(item: dict, prompt_mode: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": build_user_content(item, prompt_mode)},
    ]


def choose_prediction(generation: str, choices: list[str]) -> tuple[int, str]:
    text = generation.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
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
    parser.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--prompt-mode",
        choices=("qa", "text_only", "audio_only", "generic_audio"),
        default="qa",
        help="qa uses audio+question; text_only omits audio; audio_only omits text; generic_audio asks for free-form description.",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    rows = load_jsonl(args.manifest, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.enable_audio_output = False
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    is_thinker_model = False
    if args.adapter is not None:
        if hasattr(model, "thinker"):
            model = model.thinker
            is_thinker_model = True
        model = PeftModel.from_pretrained(model, args.adapter)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()
    model_device = next(model.parameters()).device
    model_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.bfloat16)

    correct = 0
    total = 0
    with args.output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows):
            conversation = build_conversation(item, args.prompt_mode)
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            )
            inputs = move_inputs(inputs, model_device, model_dtype)

            with torch.inference_mode():
                generate_kwargs = {
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                }
                if not is_thinker_model:
                    generate_kwargs.update(
                        {
                            "return_audio": False,
                            "thinker_return_dict_in_generate": True,
                            "use_audio_in_video": False,
                        }
                    )
                generated = model.generate(**inputs, **generate_kwargs)
            generated = decode_generated_tokens(generated, inputs["input_ids"].shape[1])
            generation = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
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
                        "prompt_mode": args.prompt_mode,
                        "audio": item.get("audio", ""),
                        "source_audio": item.get("source_audio", item.get("audio", "")),
                        "ablation": item.get("ablation", "none"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()

    print(f"accuracy={correct / max(total, 1):.4f} correct={correct} total={total}")


if __name__ == "__main__":
    main()
