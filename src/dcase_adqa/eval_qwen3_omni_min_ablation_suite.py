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
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    return processor, model, is_thinker_model, device, dtype, pad_token_id


def eval_one(name, manifest, output, prompt_mode, processor, model, is_thinker_model, device, dtype, pad_token_id):
    rows = load_jsonl(manifest, None)
    output.parent.mkdir(parents=True, exist_ok=True)
    correct = total = 0
    with output.open("w", encoding="utf-8") as out:
        for item in tqdm(rows, desc=name):
            conversation = build_conversation(item, prompt_mode)
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=False)
            inputs = move_inputs(inputs, device, dtype)
            with torch.inference_mode():
                kwargs = {"max_new_tokens": 24, "do_sample": False}
                if pad_token_id is not None:
                    kwargs["pad_token_id"] = pad_token_id
                if not is_thinker_model:
                    kwargs.update({"return_audio": False, "thinker_return_dict_in_generate": True, "use_audio_in_video": False})
                generated = model.generate(**inputs, **kwargs)
            generated = decode_generated_tokens(generated, inputs["input_ids"].shape[1])
            generation = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            pred_index, pred_text = choose_prediction(generation, item["choices"])
            is_correct = pred_index == item["answer_index"]
            correct += int(is_correct); total += 1
            out.write(json.dumps({
                "id": item["id"], "answer": item["answer"], "answer_index": item["answer_index"],
                "prediction": pred_text, "prediction_index": pred_index, "correct": is_correct,
                "raw_generation": generation, "prompt_mode": prompt_mode,
                "audio": item.get("audio", ""), "source_audio": item.get("source_audio", item.get("audio", "")),
                "donor_id": item.get("donor_id", ""), "ablation": item.get("ablation", name),
            }, ensure_ascii=False) + "\n")
            out.flush()
    print(f"{name}: accuracy={correct/max(total,1):.4f} correct={correct} total={total}")
    return {"name": name, "correct": correct, "total": total, "accuracy": correct/max(total,1)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-dir", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests"))
    p.add_argument("--out-dir", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablations/qwen3_base_train_full_min"))
    p.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct"))
    p.add_argument("--adapter", type=Path, default=None)
    args = p.parse_args()
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    processor, model, is_thinker_model, device, dtype, pad_token_id = load_model(args.model, args.adapter)
    jobs = [
        ("normal", args.manifest_dir / "train_full_normal.jsonl", "qa"),
        ("empty_audio_question", args.manifest_dir / "train_full_normal.jsonl", "text_only"),
        ("shuffle_audio_random", args.manifest_dir / "train_full_shuffle_audio_random.jsonl", "qa"),
    ]
    summary=[]
    for name, manifest, prompt_mode in jobs:
        output=args.out_dir / f"{name}.jsonl"
        if output.exists() and output.stat().st_size > 0:
            rows=load_jsonl(output,None)
            correct=sum(bool(r.get("correct")) for r in rows)
            summary.append({"name":name,"correct":correct,"total":len(rows),"accuracy":correct/max(len(rows),1)})
            print(f"skip existing {name}: {output}")
            continue
        summary.append(eval_one(name, manifest, output, prompt_mode, processor, model, is_thinker_model, device, dtype, pad_token_id))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
