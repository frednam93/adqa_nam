from __future__ import annotations

import json
import random
from pathlib import Path

from dcase_adqa.analyze_errors import load_jsonl, norm_bool
from dcase_adqa.build_full_audio_dependency_sft import bucket, write_jsonl
from dcase_adqa.prepare_qwen3_omni_sft import SYSTEM_PROMPT, build_user_prompt


ROOT = Path("/home/user/ssdmain/dcase-adqa")
FAC = ROOT / "external/Fun-Audio-Chat"
DATA_INFO = FAC / "training/data/dataset_info.json"
CONFIG_DIR = FAC / "training/configs"
BASE_ADAPTER = ROOT / "outputs/qwen3_audio_dep_full_strong_empty_unknown5_3k/checkpoint-2000"
DATA_DIR = Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_preference")
TRAIN_MANIFEST = ROOT / "outputs/ablation_manifests/train_full_normal.jsonl"
SHUFFLE_MANIFEST = ROOT / "outputs/ablation_manifests/train_full_shuffle_audio_random.jsonl"
ABLATION_DIR = ROOT / "outputs/ablations/qwen3_base_train_full_min"
RUN_LIST = ROOT / "outputs/preference_continuation_runs.txt"
SUMMARY = DATA_DIR / "summary.json"
SEED = 20260602


RUNS = {
    "dpo": {"stage": "dpo", "pref_loss": "sigmoid"},
    "orpo": {"stage": "dpo", "pref_loss": "orpo"},
    "simpo": {"stage": "dpo", "pref_loss": "simpo"},
    "kto": {"stage": "kto"},
}


def strong_audio_items() -> list[dict]:
    train = load_jsonl(TRAIN_MANIFEST)
    res = {
        name: {row["id"]: row for row in load_jsonl(ABLATION_DIR / f"{name}.jsonl")}
        for name in ["normal", "empty_audio_question", "shuffle_audio_random"]
    }
    items = []
    for item in train:
        sid = item["id"]
        label = bucket(
            norm_bool(res["normal"][sid].get("correct", False)),
            norm_bool(res["empty_audio_question"][sid].get("correct", False)),
            norm_bool(res["shuffle_audio_random"][sid].get("correct", False)),
        )
        if label == "strong_audio_dependent":
            items.append(item)
    return items


def wrong_choices(item: dict) -> list[str]:
    answer = item["answer"]
    return [choice for choice in item["choices"] if choice != answer]


def conversation(item: dict, include_answer: bool = False, answer: str | None = None) -> list[dict]:
    messages = [{"from": "human", "value": build_user_prompt(item, "answer_only")}]
    if include_answer:
        messages.append({"from": "gpt", "value": answer if answer is not None else item["answer"]})
    return messages


def build_preference_data(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        for rejected in wrong_choices(item):
            rows.append(
                {
                    "system": SYSTEM_PROMPT,
                    "conversations": conversation(item),
                    "chosen": {"from": "gpt", "value": item["answer"]},
                    "rejected": {"from": "gpt", "value": rejected},
                    "audios": [item["audio"]],
                }
            )
    return rows


def build_kto_data(items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        rows.append(
            {
                "system": SYSTEM_PROMPT,
                "conversations": conversation(item, include_answer=True, answer=item["answer"]),
                "audios": [item["audio"]],
                "kto_tag": True,
            }
        )
        for rejected in wrong_choices(item):
            rows.append(
                {
                    "system": SYSTEM_PROMPT,
                    "conversations": conversation(item, include_answer=True, answer=rejected),
                    "audios": [item["audio"]],
                    "kto_tag": False,
                }
            )
    return rows


def write_config(run: str, stage: str, pref_loss: str | None = None, max_steps: int = 500) -> None:
    dataset = f"dcase_adqa_qwen3_pref_{run}"
    output_dir = ROOT / f"outputs/qwen3_audio_dep_full_strong_empty5_pref_{run}_500"
    lines = [
        "### model",
        "model_name_or_path: /home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct",
        f"adapter_name_or_path: {BASE_ADAPTER}",
        "trust_remote_code: true",
        "quantization_bit: 4",
        "quantization_method: bnb",
        "",
        "### method",
        f"stage: {stage}",
        "do_train: true",
        "finetuning_type: lora",
        "lora_rank: 4",
        "lora_alpha: 8",
        "lora_dropout: 0.05",
        "lora_target: q_proj,v_proj",
        "flash_attn: auto",
        "freeze_vision_tower: true",
        "freeze_multi_modal_projector: true",
        "print_param_status: false",
        "upcast_layernorm: true",
        "pref_beta: 0.1",
    ]
    if pref_loss is not None:
        lines.append(f"pref_loss: {pref_loss}")
    lines += [
        "",
        "### dataset",
        f"dataset: {dataset}",
        "dataset_dir: /home/user/ssdmain/dcase-adqa/external/Fun-Audio-Chat/training/data",
        "template: qwen3_omni_nothink",
        "cutoff_len: 2048",
        "overwrite_cache: true",
        "preprocessing_num_workers: 4",
        "dataloader_num_workers: 2",
        "",
        "### output",
        f"output_dir: {output_dir}",
        "logging_steps: 10",
        "save_strategy: steps",
        "save_steps: 500",
        "plot_loss: true",
        "overwrite_output_dir: true",
        "save_only_model: false",
        "report_to: none",
        "",
        "### train",
        "per_device_train_batch_size: 2",
        "gradient_accumulation_steps: 4",
        "learning_rate: 5.0e-6",
        "lr_scheduler_type: cosine",
        "warmup_ratio: 0.03",
        "bf16: true",
        "ddp_timeout: 180000000",
        "disable_gradient_checkpointing: false",
        f"max_steps: {max_steps}",
    ]
    (CONFIG_DIR / f"dcase_adqa_qwen3_pref_{run}_500.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def register_dataset(run: str, file_path: Path, ranking: bool, kto: bool = False) -> None:
    info = json.loads(DATA_INFO.read_text(encoding="utf-8"))
    columns = {"system": "system", "messages": "conversations", "audios": "audios"}
    if ranking:
        columns.update({"chosen": "chosen", "rejected": "rejected"})
    if kto:
        columns["kto_tag"] = "kto_tag"
    entry = {
        "file_name": str(file_path),
        "formatting": "sharegpt",
        "columns": columns,
        "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
    }
    if ranking:
        entry["ranking"] = True
    info[f"dcase_adqa_qwen3_pref_{run}"] = entry
    DATA_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    items = strong_audio_items()
    pref_rows = build_preference_data(items)
    kto_rows = build_kto_data(items)
    rng.shuffle(pref_rows)
    rng.shuffle(kto_rows)

    made = {
        "seed": SEED,
        "base_adapter": str(BASE_ADAPTER),
        "strong_audio_dependent_items": len(items),
        "preference_rows": len(pref_rows),
        "kto_rows": len(kto_rows),
        "runs": RUNS,
    }
    for run, spec in RUNS.items():
        rows = kto_rows if spec["stage"] == "kto" else pref_rows
        path = DATA_DIR / run / "train.jsonl"
        write_jsonl(path, rows)
        register_dataset(run, path, ranking=spec["stage"] == "dpo", kto=spec["stage"] == "kto")
        write_config(run, spec["stage"], spec.get("pref_loss"))
    SUMMARY.write_text(json.dumps(made, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    RUN_LIST.write_text("\n".join(RUNS) + "\n", encoding="utf-8")
    print(json.dumps(made, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
