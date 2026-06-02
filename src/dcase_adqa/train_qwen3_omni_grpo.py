from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from qwen_omni_utils import process_mm_info
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoProcessor, BitsAndBytesConfig

from dcase_adqa.eval_qwen3_omni import (
    SYSTEM_PROMPT,
    build_conversation,
    choose_prediction,
    decode_generated_tokens,
    move_inputs,
)


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def build_assistant_conversation(item: dict, response: str) -> list[dict]:
    convo = build_conversation(item, "qa")
    convo.append({"role": "assistant", "content": [{"type": "text", "text": response}]})
    return convo


def render_inputs(processor, conversation: list[dict], model_device, model_dtype, add_generation_prompt: bool):
    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
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
    return move_inputs(inputs, model_device, model_dtype)


def response_logprob(model, processor, item: dict, response: str, model_device, model_dtype) -> torch.Tensor:
    prompt_inputs = render_inputs(
        processor,
        build_conversation(item, "qa"),
        model_device,
        model_dtype,
        add_generation_prompt=True,
    )
    full_inputs = render_inputs(
        processor,
        build_assistant_conversation(item, response),
        model_device,
        model_dtype,
        add_generation_prompt=False,
    )
    prompt_len = prompt_inputs["input_ids"].shape[1]
    input_ids = full_inputs["input_ids"]
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    if (labels != -100).sum() == 0:
        # Degenerate generations should not contribute a gradient.
        return input_ids.new_tensor(0, dtype=torch.float32).to(model_device)

    outputs = model(**full_inputs, return_dict=True, use_cache=False)
    logits = outputs.logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    valid = shifted_labels != -100
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    token_logps = torch.gather(F.log_softmax(logits, dim=-1), -1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * valid).sum() / valid.sum().clamp_min(1)


def reward_generation(generation: str, item: dict, parse_bad_penalty: float) -> tuple[float, int, str]:
    pred_index, pred_text = choose_prediction(generation, item["choices"])
    if pred_index < 0:
        return parse_bad_penalty, pred_index, pred_text
    return (1.0 if pred_index == item["answer_index"] else 0.0), pred_index, pred_text


def generate_one(
    model,
    processor,
    item: dict,
    model_device,
    model_dtype,
    max_new_tokens: int,
    temperature: float,
    is_thinker_model: bool,
):
    inputs = render_inputs(
        processor,
        build_conversation(item, "qa"),
        model_device,
        model_dtype,
        add_generation_prompt=True,
    )
    with torch.no_grad():
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
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
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct"))
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target", default="q_proj,v_proj")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--parse-bad-penalty", type=float, default=-0.2)
    parser.add_argument("--dapo-lite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--log-steps", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n", encoding="utf-8")

    rows = load_jsonl(args.manifest, args.limit)
    if not rows:
        raise ValueError(f"No rows loaded from {args.manifest}")

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
    if hasattr(model, "thinker"):
        model = model.thinker
        is_thinker_model = True
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[x.strip() for x in args.lora_target.split(",") if x.strip()],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model.train()
    model_device = next(model.parameters()).device
    model_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.bfloat16)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)

    logs_path = args.output_dir / "train_log.jsonl"
    t0 = time.time()
    train_rows = rows[:]
    global_step = 0
    skipped = 0
    pbar = tqdm(total=args.max_steps, desc="grpo")
    with logs_path.open("w", encoding="utf-8") as log_f:
        while global_step < args.max_steps:
            random.shuffle(train_rows)
            for item in train_rows:
                if global_step >= args.max_steps:
                    break
                generations = [
                    generate_one(
                        model,
                        processor,
                        item,
                        model_device,
                        model_dtype,
                        args.max_new_tokens,
                        args.temperature,
                        is_thinker_model,
                    )
                    for _ in range(args.num_generations)
                ]
                rewards = [reward_generation(gen, item, args.parse_bad_penalty)[0] for gen in generations]
                mean_reward = sum(rewards) / len(rewards)
                variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
                std_reward = math.sqrt(variance)
                if args.dapo_lite and std_reward < 1e-8:
                    skipped += 1
                    global_step += 1
                    pbar.update(1)
                    if global_step % args.log_steps == 0:
                        parsed = [reward_generation(gen, item, args.parse_bad_penalty) for gen in generations]
                        rec = {
                            "step": global_step,
                            "id": item.get("id"),
                            "loss": 0.0,
                            "reward_mean": mean_reward,
                            "reward_std": std_reward,
                            "rewards": rewards,
                            "logps": [],
                            "pred_indices": [x[1] for x in parsed],
                            "elapsed_sec": round(time.time() - t0, 3),
                            "skipped": skipped,
                        }
                        log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        log_f.flush()
                        print(
                            f"step={global_step} loss=0.0000 reward={mean_reward:.3f} "
                            f"std={std_reward:.3f} skipped={skipped}"
                        )
                    if global_step % args.save_steps == 0 or global_step == args.max_steps:
                        save_dir = args.output_dir / f"checkpoint-{global_step}"
                        model.save_pretrained(save_dir)
                        processor.save_pretrained(save_dir)
                    continue
                denom = std_reward if std_reward > 1e-8 else 1.0
                advantages = [(r - mean_reward) / denom for r in rewards]

                optimizer.zero_grad(set_to_none=True)
                loss_values = []
                logps = []
                for gen, adv in zip(generations, advantages, strict=True):
                    lp = response_logprob(model, processor, item, gen, model_device, model_dtype)
                    logps.append(float(lp.detach().cpu()))
                    loss_term = -float(adv) * lp / len(generations)
                    loss_values.append(float(loss_term.detach().cpu()))
                    loss_term.backward()
                    del lp, loss_term
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_value = sum(loss_values)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                global_step += 1
                pbar.update(1)
                if global_step % args.log_steps == 0:
                    parsed = [reward_generation(gen, item, args.parse_bad_penalty) for gen in generations]
                    rec = {
                        "step": global_step,
                        "id": item.get("id"),
                        "loss": loss_value,
                        "reward_mean": mean_reward,
                        "reward_std": std_reward,
                        "rewards": rewards,
                        "logps": logps,
                        "pred_indices": [x[1] for x in parsed],
                        "elapsed_sec": round(time.time() - t0, 3),
                        "skipped": skipped,
                    }
                    log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    log_f.flush()
                    print(
                        f"step={global_step} loss={rec['loss']:.4f} reward={mean_reward:.3f} "
                        f"std={std_reward:.3f} skipped={skipped}"
                    )
                if global_step % args.save_steps == 0 or global_step == args.max_steps:
                    save_dir = args.output_dir / f"checkpoint-{global_step}"
                    model.save_pretrained(save_dir)
                    processor.save_pretrained(save_dir)
    pbar.close()
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
