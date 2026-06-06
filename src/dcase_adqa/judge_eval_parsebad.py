from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoProcessor, BitsAndBytesConfig

from dcase_adqa.make_submission_outputs import load_jsonl


def build_prompt(choices: list[str], response: str) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = "\n".join(f"({labels[i]}) {choice}" for i, choice in enumerate(choices))
    return (
        "You are an answer normalizer for a multiple-choice evaluation.\n"
        "Map the model response to exactly one of the listed choices.\n"
        "Do not answer the original question. Do not use outside knowledge.\n"
        "If the response clearly refers to one choice, output only its letter.\n"
        "If it is ambiguous or does not match any choice, output only X.\n\n"
        f"Choices:\n{opts}\n\n"
        f"Model response:\n{response}\n\n"
        "Normalized answer letter:"
    )


def parse_letter(text: str, n: int) -> int:
    import re

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    match = re.search(rf"(?i)\b([{re.escape(letters)}X])\b", text)
    if not match:
        match = re.search(rf"(?i)([{re.escape(letters)}X])", text[:12])
    if not match:
        return -1
    letter = match.group(1).upper()
    if letter == "X":
        return -1
    return letters.index(letter)


def load_base_judge(model_path: Path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.enable_audio_output = False
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        quantization_config=qconf,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()
    return processor, model, next(model.parameters()).device


def judge_one(processor, model, device, choices: list[str], response: str, max_new_tokens: int) -> tuple[int, str]:
    prompt = build_prompt(choices, response)
    conv = [
        {"role": "system", "content": [{"type": "text", "text": "You normalize model responses to multiple-choice letters."}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=False,
        )
    if isinstance(generated, tuple):
        generated = generated[0]
    if hasattr(generated, "sequences"):
        generated = generated.sequences
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return parse_letter(raw, len(choices)), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pred", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    manifest = {row["id"]: row for row in load_jsonl(args.manifest)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor, model, device = load_base_judge(args.model)

    for pred_path in args.pred:
        rows = load_jsonl(pred_path)
        out_rows = []
        parse_bad = 0
        fixed = 0
        for i, row in enumerate(rows, 1):
            item = manifest[row["id"]]
            judged_idx = row.get("prediction_index", -1)
            judged_raw = ""
            method = "original"
            if judged_idx == -1:
                parse_bad += 1
                judged_idx, judged_raw = judge_one(
                    processor,
                    model,
                    device,
                    item["choices"],
                    row.get("raw_generation") or row.get("prediction") or "",
                    args.max_new_tokens,
                )
                method = "qwen3_base_judge"
                fixed += int(judged_idx >= 0)
            out = dict(row)
            out.update(
                {
                    "judge_method": method,
                    "judge_raw": judged_raw,
                    "judge_prediction_index": judged_idx,
                    "judge_prediction": item["choices"][judged_idx] if judged_idx >= 0 else row.get("prediction", ""),
                }
            )
            out_rows.append(out)
            if i % 100 == 0:
                print(f"{pred_path.name}: judged {i}/{len(rows)} parse_bad={parse_bad} fixed={fixed}", flush=True)
        out_path = args.out_dir / f"{pred_path.stem}.basejudge.jsonl"
        out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in out_rows) + "\n", encoding="utf-8")
        print(f"wrote {out_path} n={len(out_rows)} parse_bad={parse_bad} fixed={fixed}", flush=True)


if __name__ == "__main__":
    main()
