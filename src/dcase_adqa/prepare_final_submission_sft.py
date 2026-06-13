from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

from dcase_adqa.build_full_audio_dependency_sft import UNKNOWN_SYSTEM, UNKNOWN_TARGET
from dcase_adqa.prepare_qwen3_omni_sft import convert_item, load_jsonl


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


def make_empty_item(item: dict, target: str = UNKNOWN_TARGET) -> dict:
    converted = convert_item(item, "answer_only")
    converted["system"] = UNKNOWN_SYSTEM
    converted["messages"][0]["content"] = converted["messages"][0]["content"].replace("<audio>\n", "")
    converted["messages"][-1]["content"] = target
    converted["audios"] = []
    return converted


def read_bucket_ids(path: Path, bucket_name: str = "strong_audio_dependent") -> list[str]:
    ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["bucket"] == bucket_name:
                ids.append(row["id"])
    return ids


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def register_dataset(data_info: Path, name: str, train_path: Path) -> None:
    data_info.parent.mkdir(parents=True, exist_ok=True)
    info = json.loads(data_info.read_text(encoding="utf-8")) if data_info.exists() else {}
    info[name] = {
        "file_name": str(train_path),
        "formatting": "sharegpt",
        "columns": {"system": "system", "messages": "messages", "audios": "audios"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
    data_info.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_config(base_config: Path, config_dir: Path, dataset_name: str, output_dir: Path, steps: int, save_steps: int) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{dataset_name}_{steps}.yaml"
    lines: list[str] = []
    for line in base_config.read_text(encoding="utf-8").splitlines():
        line = os.path.expandvars(line)
        if line.startswith("dataset:"):
            lines.append(f"dataset: {dataset_name}")
        elif line.startswith("output_dir:"):
            lines.append(f"output_dir: {output_dir}")
        elif line.startswith("num_train_epochs:"):
            continue
        elif line.startswith("save_steps:"):
            lines.append(f"save_steps: {save_steps}")
        elif line.startswith("logging_steps:"):
            lines.append("logging_steps: 10")
        elif line.startswith("overwrite_output_dir:"):
            lines.append("overwrite_output_dir: true")
        else:
            lines.append(line)
    lines.append(f"max_steps: {steps}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def main() -> None:
    p = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    data_root = env_path("DCASE_TASK5_ROOT", "data/dcase2026_task5")
    output_root = env_path("OUTPUT_ROOT", "outputs")
    fun_audio_root = env_path("FUN_AUDIO_CHAT_ROOT", "external/Fun-Audio-Chat")
    p.add_argument("--train-manifest", type=Path, default=data_root / "manifests/train.jsonl")
    p.add_argument("--dev-manifest", type=Path, default=data_root / "manifests/dev.jsonl")
    p.add_argument("--train-buckets", type=Path, default=output_root / "analysis/audio_dependency_full/train_full_audio_dependency_buckets.csv")
    p.add_argument("--dev-buckets", type=Path, default=output_root / "analysis/audio_dependency/dev_audio_dependency_buckets.csv")
    p.add_argument("--out-dir", type=Path, default=data_root / "qwen3_omni_sft_final_submission")
    p.add_argument("--fun-audio-root", type=Path, default=fun_audio_root)
    p.add_argument("--base-config", type=Path, default=repo_root / "external_overrides/Fun-Audio-Chat/training/configs/dcase_adqa_qwen3_omni_qlora.template.yaml")
    p.add_argument("--seed", type=int, default=20260606)
    p.add_argument("--empty-ratio", type=float, default=0.05)
    p.add_argument("--run-suffix", default=None)
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()

    rng = random.Random(args.seed)
    train_rows = {row["id"]: row for row in load_jsonl(args.train_manifest)}
    dev_rows = {row["id"]: row for row in load_jsonl(args.dev_manifest)}
    train_ids = [sid for sid in read_bucket_ids(args.train_buckets) if sid in train_rows]
    dev_ids = [sid for sid in read_bucket_ids(args.dev_buckets) if sid in dev_rows]

    positive_items = [convert_item(train_rows[sid], "answer_only") for sid in train_ids]
    positive_items += [convert_item(dev_rows[sid], "answer_only") for sid in dev_ids]

    all_ids = [("train", sid) for sid in train_ids] + [("dev", sid) for sid in dev_ids]
    n_empty = max(1, round(len(positive_items) * args.empty_ratio))
    empty_ids = rng.sample(all_ids, min(n_empty, len(all_ids)))
    empty_items = []
    for split, sid in empty_ids:
        source = train_rows[sid] if split == "train" else dev_rows[sid]
        empty_items.append(make_empty_item(source, "Cannot be determined from the audio."))

    data = positive_items + empty_items
    rng.shuffle(data)

    suffix = args.run_suffix or f"empty{str(args.empty_ratio * 100).replace('.', 'p').rstrip('0').rstrip('p')}"
    dataset_name = f"dcase_adqa_qwen3_final_train_dev_strong_{suffix}"
    run_name = f"train_dev_strong_{suffix}"
    train_path = args.out_dir / run_name / "train.jsonl"
    write_jsonl(train_path, data)
    data_info = args.fun_audio_root / "training/data/dataset_info.json"
    register_dataset(data_info, dataset_name, train_path)
    run_output_dir = output_root / f"final_submission/qwen3_{run_name}_{args.steps // 1000}k"
    config_path = write_config(
        base_config=args.base_config,
        config_dir=args.fun_audio_root / "training/configs",
        dataset_name=dataset_name,
        output_dir=run_output_dir,
        steps=args.steps,
        save_steps=1000,
    )

    summary = {
        "dataset_name": dataset_name,
        "train_path": str(train_path),
        "config_path": str(config_path),
        "output_dir": str(run_output_dir),
        "train_strong": len(train_ids),
        "dev_strong": len(dev_ids),
        "positives": len(positive_items),
        "empty_ratio": args.empty_ratio,
        "empty_count": len(empty_items),
        "total": len(data),
        "seed": args.seed,
        "steps": args.steps,
    }
    summary_path = args.out_dir / run_name / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
