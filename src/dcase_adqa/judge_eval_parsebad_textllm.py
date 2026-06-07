from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from dcase_adqa.make_submission_outputs import load_jsonl


def build_prompt(choices: list[str], response: str) -> list[dict]:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = "\n".join(f"({labels[i]}) {choice}" for i, choice in enumerate(choices))
    user = (
        "Map the model response to exactly one of the listed multiple-choice options.\n"
        "Do not answer the original question. Do not use outside knowledge.\n"
        "Use only the candidate choices and the model response.\n"
        "If the response clearly refers to one choice, output only its letter.\n"
        "If it is ambiguous or does not match any choice, output only X.\n\n"
        f"Choices:\n{opts}\n\n"
        f"Model response:\n{response}\n\n"
        "Normalized answer letter:"
    )
    return [
        {"role": "system", "content": "You normalize model responses to multiple-choice letters."},
        {"role": "user", "content": user},
    ]


def parse_letter(text: str, n: int) -> int:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n]
    text = text.strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    match = re.search(rf"(?i)\b([{re.escape(letters)}X])\b", text)
    if not match:
        match = re.search(rf"(?i)([{re.escape(letters)}X])", text[:16])
    if not match:
        return -1
    letter = match.group(1).upper()
    if letter == "X":
        return -1
    return letters.index(letter)


def load_judge(model_path: Path, load_4bit: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    kwargs = {
        "trust_remote_code": True,
        "device_map": {"": 0},
        "torch_dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs.pop("torch_dtype", None)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()
    return tokenizer, model


def judge_one(tokenizer, model, choices: list[str], response: str, max_new_tokens: int) -> tuple[int, str]:
    messages = build_prompt(choices, response)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = messages[0]["content"] + "\n\n" + messages[1]["content"]
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[:, inputs["input_ids"].shape[1] :]
    raw = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    return parse_letter(raw, len(choices)), raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pred", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/qwen3_8b"))
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--load-4bit", action="store_true", default=True)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    manifest = {row["id"]: row for row in load_jsonl(args.manifest)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_judge(args.model, args.load_4bit)

    summaries = []
    for pred_path in args.pred:
        rows = load_jsonl(pred_path)
        out_rows = []
        parse_bad = fixed = 0
        correct_before = correct_after = None
        has_labels = all("answer_index" in manifest[row["id"]] and manifest[row["id"]].get("answer_index", -1) >= 0 for row in rows)
        if has_labels:
            correct_before = 0
            correct_after = 0
        for i, row in enumerate(rows, 1):
            item = manifest[row["id"]]
            judged_idx = row.get("prediction_index", -1)
            judged_raw = ""
            method = "original"
            if has_labels:
                correct_before += int(row.get("prediction_index", -1) == item["answer_index"])
            if judged_idx == -1:
                parse_bad += 1
                judged_idx, judged_raw = judge_one(
                    tokenizer,
                    model,
                    item["choices"],
                    row.get("raw_generation") or row.get("prediction") or "",
                    args.max_new_tokens,
                )
                method = "textllm_judge"
                fixed += int(judged_idx >= 0)
            if has_labels:
                correct_after += int(judged_idx == item["answer_index"])
            out = dict(row)
            out.update(
                {
                    "judge_method": method,
                    "judge_raw": judged_raw,
                    "judge_prediction_index": judged_idx,
                    "judge_prediction": item["choices"][judged_idx] if judged_idx >= 0 else row.get("prediction", ""),
                }
            )
            if has_labels:
                out["judge_correct"] = judged_idx == item["answer_index"]
            out_rows.append(out)
            if i % 100 == 0:
                print(f"{pred_path.name}: judged {i}/{len(rows)} parse_bad={parse_bad} fixed={fixed}", flush=True)
        out_path = args.out_dir / f"{pred_path.stem}.qwen3_8b_judge.jsonl"
        out_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in out_rows) + "\n", encoding="utf-8")
        summary = {
            "file": pred_path.name,
            "output": str(out_path),
            "n": len(out_rows),
            "parse_bad": parse_bad,
            "fixed": fixed,
            "judge_bad": sum(1 for r in out_rows if r.get("judge_prediction_index") == -1),
        }
        if has_labels:
            summary.update(
                {
                    "strict_correct": correct_before,
                    "strict_acc": correct_before / len(out_rows),
                    "judge_correct": correct_after,
                    "judge_acc": correct_after / len(out_rows),
                    "delta": correct_after - correct_before,
                }
            )
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    summary_path = args.out_dir / "qwen3_8b_judge_summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()
