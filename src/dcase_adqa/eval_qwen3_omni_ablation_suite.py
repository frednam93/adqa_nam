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

from dcase_adqa.eval_qwen3_omni import (
    build_conversation,
    choose_prediction,
    decode_generated_tokens,
    load_jsonl,
    move_inputs,
)


def load_model(model_path: Path, adapter: Path | None):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.enable_audio_output = False
    quantization_config = BitsAndBytesConfig(
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
        quantization_config=quantization_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    is_thinker_model = False
    if adapter is not None:
        if hasattr(model, "thinker"):
            model = model.thinker
            is_thinker_model = True
        model = PeftModel.from_pretrained(model, adapter)
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    model.eval()
    device = next(model.parameters()).device
    dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.bfloat16)
    return processor, model, is_thinker_model, device, dtype


def eval_one(
    *,
    name: str,
    manifest: Path,
    output: Path,
    prompt_mode: str,
    max_new_tokens: int,
    processor,
    model,
    is_thinker_model: bool,
    device: torch.device,
    dtype: torch.dtype,
    limit: int | None,
) -> tuple[int, int]:
    rows = load_jsonl(manifest, limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(tokenizer, "eos_token_id", None)

    correct = 0
    total = 0
    with output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows, desc=name):
            conversation = build_conversation(item, prompt_mode)
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
            inputs = move_inputs(inputs, device, dtype)
            with torch.inference_mode():
                generate_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
                if pad_token_id is not None:
                    generate_kwargs["pad_token_id"] = pad_token_id
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
                        "prompt_mode": prompt_mode,
                        "audio": item.get("audio", ""),
                        "source_audio": item.get("source_audio", item.get("audio", "")),
                        "donor_id": item.get("donor_id", ""),
                        "donor_category": item.get("donor_category", ""),
                        "ablation": item.get("ablation", name),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out.flush()
    return correct, total


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, required=True)
    p.add_argument("--split-prefix", required=True, help="e.g. dev_full or train_strat80")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct"))
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor, model, is_thinker_model, device, dtype = load_model(args.model, args.adapter)

    jobs = [
        ("empty_audio_question", "normal", "text_only", 24),
        ("shuffle_audio_random", "shuffle_audio_random", "qa", 24),
        ("shuffle_audio_same_category", "shuffle_audio_same_category", "qa", 24),
        ("shuffle_audio_cross_category", "shuffle_audio_cross_category", "qa", 24),
        ("choice_shuffle", "choice_shuffle", "qa", 24),
        ("audio_only_no_prompt", "normal", "audio_only", 96),
        ("generic_prompt", "normal", "generic_audio", 128),
    ]
    summary = []
    for name, manifest_suffix, prompt_mode, max_new_tokens in jobs:
        manifest = args.manifest_dir / f"{args.split_prefix}_{manifest_suffix}.jsonl"
        output = args.out_dir / f"{name}.jsonl"
        if output.exists() and output.stat().st_size > 0:
            print(f"skip existing {name}: {output}")
            rows = load_jsonl(output, None)
            correct = sum(bool(r.get("correct")) for r in rows)
            total = len(rows)
        else:
            print(f"==== {name} prompt={prompt_mode} manifest={manifest} ====")
            correct, total = eval_one(
                name=name,
                manifest=manifest,
                output=output,
                prompt_mode=prompt_mode,
                max_new_tokens=max_new_tokens,
                processor=processor,
                model=model,
                is_thinker_model=is_thinker_model,
                device=device,
                dtype=dtype,
                limit=args.limit,
            )
        acc = correct / max(total, 1)
        summary.append({"name": name, "correct": correct, "total": total, "accuracy": acc})
        print(f"{name}: accuracy={acc:.4f} correct={correct} total={total}")

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
