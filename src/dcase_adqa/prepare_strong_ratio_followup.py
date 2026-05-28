from __future__ import annotations

import json
import random
from pathlib import Path

from dcase_adqa.analyze_errors import load_jsonl, norm_bool
from dcase_adqa.build_full_audio_dependency_sft import bucket, make_unknown_item, write_jsonl
from dcase_adqa.prepare_qwen3_omni_sft import convert_item

ROOT = Path("/home/user/ssdmain/dcase-adqa")
FAC = ROOT / "external/Fun-Audio-Chat"
DATA_INFO = FAC / "training/data/dataset_info.json"
BASE_CONFIG = FAC / "training/configs/dcase_adqa_qwen3_omni_qlora.yaml"
MODEL_OUTPUT_ROOT = ROOT / "outputs"
RUN_LIST = MODEL_OUTPUT_ROOT / "strong_ratio_followup_runs.txt"
SUMMARY = MODEL_OUTPUT_ROOT / "strong_ratio_followup_summary.json"

TRAIN_MANIFEST = ROOT / "outputs/ablation_manifests/train_full_normal.jsonl"
SHUFFLE_MANIFEST = ROOT / "outputs/ablation_manifests/train_full_shuffle_audio_random.jsonl"
ABLATION_DIR = ROOT / "outputs/ablations/qwen3_base_train_full_min"
SFT_DIR = Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full")
SEED = 20260528
DEV_N = 1607


def eval_summary(run: str, step: int) -> dict | None:
    path = MODEL_OUTPUT_ROOT / f"qwen3_audio_dep_full_{run}_3k_dev_ckpt{step}.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != DEV_N:
        return None
    ok = sum(bool(row.get("correct")) for row in rows)
    parse_bad = sum(row.get("prediction_index") == -1 for row in rows)
    return {
        "run": run,
        "step": step,
        "correct": ok,
        "total": len(rows),
        "acc": ok / max(len(rows), 1),
        "parse_bad": parse_bad,
        "path": str(path),
    }


def choose_followup() -> tuple[str, list[tuple[str, int]]]:
    candidates = [
        "strong_empty_unknown5",
        "strong_empty_shuffle_unknown10",
        "strong_ac",
        "strong_shuffle_unknown5",
    ]
    summaries = [
        item
        for run in candidates
        for step in (1000, 2000, 3000)
        if (item := eval_summary(run, step)) is not None
    ]
    if not summaries:
        raise RuntimeError("No complete strong-negative eval files found.")
    best = max(summaries, key=lambda item: (item["acc"], -item["parse_bad"]))
    if best["run"].startswith("strong_empty_shuffle"):
        selected = [("strong_empty_shuffle_unknown20", 20), ("strong_empty_shuffle_unknown30", 30)]
        family = "empty_shuffle"
    else:
        selected = [("strong_empty_unknown10", 10), ("strong_empty_unknown20", 20)]
        family = "empty"
    SUMMARY.write_text(
        json.dumps({"best": best, "all": summaries, "family": family, "selected": selected}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return family, selected


def build_strong_variants(family: str, selected: list[tuple[str, int]]) -> dict[str, int]:
    rng = random.Random(SEED)
    train = load_jsonl(TRAIN_MANIFEST)
    train_by_id = {row["id"]: row for row in train}
    shuffle_by_id = {row["id"]: row for row in load_jsonl(SHUFFLE_MANIFEST)}
    res = {
        name: {row["id"]: row for row in load_jsonl(ABLATION_DIR / f"{name}.jsonl")}
        for name in ["normal", "empty_audio_question", "shuffle_audio_random"]
    }
    strong_ids = []
    for item in train:
        sid = item["id"]
        b = bucket(
            norm_bool(res["normal"][sid].get("correct", False)),
            norm_bool(res["empty_audio_question"][sid].get("correct", False)),
            norm_bool(res["shuffle_audio_random"][sid].get("correct", False)),
        )
        if b == "strong_audio_dependent":
            strong_ids.append(sid)

    positives = [convert_item(train_by_id[sid], "answer_only") for sid in strong_ids]
    made = {"strong_ac_base": len(positives), "family": family}
    for name, ratio in selected:
        data = list(positives)
        if family == "empty_shuffle":
            n_each = max(1, round(len(positives) * (ratio / 2) / 100))
            empty_ids = rng.sample(strong_ids, min(n_each, len(strong_ids)))
            shuffle_ids = rng.sample(strong_ids, min(n_each, len(strong_ids)))
            data += [make_unknown_item(train_by_id[sid], None) for sid in empty_ids]
            data += [make_unknown_item(train_by_id[sid], shuffle_by_id[sid]["audio"]) for sid in shuffle_ids]
        else:
            n = max(1, round(len(positives) * ratio / 100))
            empty_ids = rng.sample(strong_ids, min(n, len(strong_ids)))
            data += [make_unknown_item(train_by_id[sid], None) for sid in empty_ids]
        rng.shuffle(data)
        write_jsonl(SFT_DIR / name / "train.jsonl", data)
        made[name] = len(data)
    (SFT_DIR / "strong_ratio_followup_summary.json").write_text(json.dumps(made, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return made


def register_runs(runs: list[str]) -> None:
    info = json.loads(DATA_INFO.read_text(encoding="utf-8"))
    for run in runs:
        info[f"dcase_adqa_qwen3_audio_dep_full_{run}"] = {
            "file_name": str(SFT_DIR / run / "train.jsonl"),
            "formatting": "sharegpt",
            "columns": {"system": "system", "messages": "messages", "audios": "audios"},
            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
        }
    DATA_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    base_cfg = BASE_CONFIG.read_text(encoding="utf-8").splitlines()
    for run in runs:
        lines = []
        for line in base_cfg:
            if line.startswith("dataset:"):
                lines.append(f"dataset: dcase_adqa_qwen3_audio_dep_full_{run}")
            elif line.startswith("output_dir:"):
                lines.append(f"output_dir: {ROOT}/outputs/qwen3_audio_dep_full_{run}_3k")
            elif line.startswith("num_train_epochs:"):
                continue
            elif line.startswith("save_steps:"):
                lines.append("save_steps: 1000")
            elif line.startswith("logging_steps:"):
                lines.append("logging_steps: 10")
            else:
                lines.append(line)
        lines.append("max_steps: 3000")
        (FAC / f"training/configs/dcase_adqa_qwen3_audio_dep_full_{run}_3k.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    family, selected = choose_followup()
    made = build_strong_variants(family, selected)
    runs = [name for name, _ in selected]
    register_runs(runs)
    RUN_LIST.write_text("\n".join(runs) + "\n", encoding="utf-8")
    print(json.dumps({"family": family, "runs": runs, "sizes": made, "summary": str(SUMMARY)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
