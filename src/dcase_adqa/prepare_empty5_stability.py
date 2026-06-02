from __future__ import annotations

import json
import random
from pathlib import Path

from dcase_adqa.analyze_errors import load_jsonl, norm_bool
from dcase_adqa.build_full_audio_dependency_sft import bucket, write_jsonl
from dcase_adqa.prepare_qwen3_omni_sft import convert_item

ROOT = Path("/home/user/ssdmain/dcase-adqa")
FAC = ROOT / "external/Fun-Audio-Chat"
DATA_INFO = FAC / "training/data/dataset_info.json"
BASE_CONFIG = FAC / "training/configs/dcase_adqa_qwen3_omni_qlora.yaml"
SFT_DIR = Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full")
TRAIN_MANIFEST = ROOT / "outputs/ablation_manifests/train_full_normal.jsonl"
ABLATION_DIR = ROOT / "outputs/ablations/qwen3_base_train_full_min"
RUN_LIST = ROOT / "outputs/empty5_stability_runs.txt"
SUMMARY = SFT_DIR / "empty5_stability_summary.json"

UNKNOWN_TARGET = "Cannot be determined from the audio."
PARA_TARGETS = [
    "Cannot be determined from the audio.",
    "The answer cannot be determined from the provided audio.",
    "There is not enough information in the audio to answer.",
    "The provided audio does not contain the information needed to answer.",
    "It is impossible to tell from the audio alone.",
]

RUNS = [
    {"name": "strong_empty_unknown5_seed2", "seed": 20260601, "paraphrase": False},
    {"name": "strong_empty_unknown5_seed3", "seed": 20260602, "paraphrase": False},
    {"name": "strong_empty_unknown5_para5", "seed": 20260603, "paraphrase": True},
]


def strong_audio_ids() -> list[str]:
    train = load_jsonl(TRAIN_MANIFEST)
    res = {
        name: {row["id"]: row for row in load_jsonl(ABLATION_DIR / f"{name}.jsonl")}
        for name in ["normal", "empty_audio_question", "shuffle_audio_random"]
    }
    ids = []
    for item in train:
        sid = item["id"]
        b = bucket(
            norm_bool(res["normal"][sid].get("correct", False)),
            norm_bool(res["empty_audio_question"][sid].get("correct", False)),
            norm_bool(res["shuffle_audio_random"][sid].get("correct", False)),
        )
        if b == "strong_audio_dependent":
            ids.append(sid)
    return ids


def make_empty_item(item: dict, target: str) -> dict:
    converted = convert_item(item, "answer_only")
    converted["system"] = (
        "You are an audio question answering assistant. Listen to the audio and answer with only the exact option text. "
        "If the provided audio is missing or does not contain the information needed to answer the question, answer with the requested unknown response."
    )
    converted["messages"][0]["content"] = converted["messages"][0]["content"].replace("<audio>\n", "")
    converted["messages"][-1]["content"] = target
    converted["audios"] = []
    return converted


def build_data() -> dict:
    train = load_jsonl(TRAIN_MANIFEST)
    train_by_id = {row["id"]: row for row in train}
    strong_ids = strong_audio_ids()
    positives = [convert_item(train_by_id[sid], "answer_only") for sid in strong_ids]
    n_empty = max(1, round(len(positives) * 0.05))
    made = {"strong_ac_base": len(positives), "empty_ratio": 0.05, "empty_count": n_empty}
    for spec in RUNS:
        rng = random.Random(spec["seed"])
        empty_ids = rng.sample(strong_ids, min(n_empty, len(strong_ids)))
        empty_items = []
        for idx, sid in enumerate(empty_ids):
            target = PARA_TARGETS[idx % len(PARA_TARGETS)] if spec["paraphrase"] else UNKNOWN_TARGET
            empty_items.append(make_empty_item(train_by_id[sid], target))
        data = positives + empty_items
        rng.shuffle(data)
        write_jsonl(SFT_DIR / spec["name"] / "train.jsonl", data)
        made[spec["name"]] = {
            "seed": spec["seed"],
            "paraphrase": spec["paraphrase"],
            "total": len(data),
            "empty_count": len(empty_items),
        }
    SUMMARY.write_text(json.dumps(made, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return made


def register_runs() -> None:
    info = json.loads(DATA_INFO.read_text(encoding="utf-8"))
    for spec in RUNS:
        run = spec["name"]
        info[f"dcase_adqa_qwen3_audio_dep_full_{run}"] = {
            "file_name": str(SFT_DIR / run / "train.jsonl"),
            "formatting": "sharegpt",
            "columns": {"system": "system", "messages": "messages", "audios": "audios"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    DATA_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base_cfg = BASE_CONFIG.read_text(encoding="utf-8").splitlines()
    for spec in RUNS:
        run = spec["name"]
        lines = []
        for line in base_cfg:
            if line.startswith("dataset:"):
                lines.append(f"dataset: dcase_adqa_qwen3_audio_dep_full_{run}")
            elif line.startswith("output_dir:"):
                lines.append(f"output_dir: {ROOT}/outputs/qwen3_audio_dep_full_{run}_3k")
            elif line.startswith("num_train_epochs:"):
                continue
            elif line.startswith("save_steps:"):
                lines.append("save_steps: 500")
            elif line.startswith("logging_steps:"):
                lines.append("logging_steps: 10")
            else:
                lines.append(line)
        lines.append("max_steps: 3000")
        (FAC / f"training/configs/dcase_adqa_qwen3_audio_dep_full_{run}_3k.yaml").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )
    RUN_LIST.write_text("\n".join(spec["name"] for spec in RUNS) + "\n", encoding="utf-8")


def main() -> None:
    made = build_data()
    register_runs()
    print(json.dumps({"runs": [spec["name"] for spec in RUNS], "summary": str(SUMMARY), "made": made}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
