from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from dcase_adqa.analyze_errors import classify, load_jsonl, norm_bool
from dcase_adqa.prepare_qwen3_omni_sft import convert_item

UNKNOWN_TARGET = "Cannot be determined from the audio."
UNKNOWN_SYSTEM = (
    "You are an audio question answering assistant. Listen to the audio and answer with only the exact option text. "
    "If the provided audio is missing or does not contain the information needed to answer the question, answer: "
    f"{UNKNOWN_TARGET}"
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_unknown_item(item: dict, audio: str | None) -> dict:
    converted = convert_item(item, "answer_only")
    converted["system"] = UNKNOWN_SYSTEM
    converted["messages"][-1]["content"] = UNKNOWN_TARGET
    if audio is None:
        converted["audios"] = []
        converted["messages"][0]["content"] = converted["messages"][0]["content"].replace("<audio>\n", "")
    else:
        converted["audios"] = [audio]
    return converted


def bucket(normal: bool, empty: bool, shuffle: bool) -> str:
    if normal and not empty and not shuffle:
        return "strong_audio_dependent"
    if not normal and not empty and not shuffle:
        return "hard_audio_dependent_candidate"
    if normal and empty:
        return "easy_text_prior"
    if normal and not empty and shuffle:
        return "audio_helped_but_shuffle_leak"
    if not normal and empty:
        return "misleading_audio_or_prior_only"
    if not normal and not empty and shuffle:
        return "wrong_normal_but_shuffle_correct"
    return "other"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests/train_full_normal.jsonl"))
    p.add_argument("--shuffle-manifest", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests/train_full_shuffle_audio_random.jsonl"))
    p.add_argument("--ablation-dir", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/ablations/qwen3_base_train_full_min"))
    p.add_argument("--sft-out-dir", type=Path, default=Path("/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full"))
    p.add_argument("--analysis-out-dir", type=Path, default=Path("/home/user/ssdmain/dcase-adqa/outputs/analysis/audio_dependency_full"))
    p.add_argument("--seed", type=int, default=20260523)
    args = p.parse_args()
    rng = random.Random(args.seed)
    train = load_jsonl(args.train_manifest)
    train_by_id = {r["id"]: r for r in train}
    shuffle_by_id = {r["id"]: r for r in load_jsonl(args.shuffle_manifest)}
    res = {name: {r["id"]: r for r in load_jsonl(args.ablation_dir / f"{name}.jsonl")} for name in ["normal", "empty_audio_question", "shuffle_audio_random"]}
    rows=[]; counts=Counter(); cat_counts=Counter()
    ids_by_bucket={}
    for item in train:
        sid=item["id"]
        b=bucket(
            norm_bool(res["normal"][sid].get("correct", False)),
            norm_bool(res["empty_audio_question"][sid].get("correct", False)),
            norm_bool(res["shuffle_audio_random"][sid].get("correct", False)),
        )
        ids_by_bucket.setdefault(b, []).append(sid)
        counts[b]+=1; cat_counts[(b, classify(item)[0])] += 1
        rows.append({"id":sid,"bucket":b,"category":classify(item)[0],"normal":int(norm_bool(res['normal'][sid].get('correct',False))),"empty":int(norm_bool(res['empty_audio_question'][sid].get('correct',False))),"shuffle_random":int(norm_bool(res['shuffle_audio_random'][sid].get('correct',False))),"question":item.get('question',''),"answer":item.get('answer',''),"audio":item.get('audio','')})
    args.analysis_out_dir.mkdir(parents=True, exist_ok=True)
    csv_path=args.analysis_out_dir/'train_full_audio_dependency_buckets.csv'
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    md=['# Train Full Audio Dependency Buckets','','| bucket | count | share |','|---|---:|---:|']
    for b,n in counts.most_common():
        md.append(f'| {b} | {n} | {n/len(train):.3f} |')
    md += ['', '## Top Categories', '', '| bucket | top categories |', '|---|---|']
    for b,n in counts.most_common():
        tops=', '.join(f'{c}:{k}' for (bb,c),k in cat_counts.items() if bb==b)
        tops_sorted=', '.join(f'{c}:{k}' for c,k in Counter({c:k for (bb,c),k in cat_counts.items() if bb==b}).most_common(8))
        md.append(f'| {b} | {tops_sorted} |')
    (args.analysis_out_dir/'train_full_audio_dependency_bucket_summary.md').write_text('\n'.join(md)+'\n', encoding='utf-8')

    strong=ids_by_bucket.get('strong_audio_dependent', [])
    hard=ids_by_bucket.get('hard_audio_dependent_candidate', [])
    leak=ids_by_bucket.get('audio_helped_but_shuffle_leak', [])
    specs={
        'strong_ac': strong,
        'strong_hard_ac': strong+hard,
        'non_easy_ac': strong+hard+leak,
    }
    made={}
    for name, ids in specs.items():
        data=[convert_item(train_by_id[sid], 'answer_only') for sid in ids]
        write_jsonl(args.sft_out_dir/name/'train.jsonl', data)
        made[name]=len(data)
    best_ids=specs['non_easy_ac']
    positives=[convert_item(train_by_id[sid], 'answer_only') for sid in best_ids]
    n5=max(1, round(len(positives)*0.05))
    empty_ids=rng.sample(best_ids, min(n5, len(best_ids)))
    shuffle_ids=rng.sample(best_ids, min(n5, len(best_ids)))
    combo_empty_ids=rng.sample(best_ids, min(n5, len(best_ids)))
    combo_shuffle_ids=rng.sample(best_ids, min(n5, len(best_ids)))
    variants={
        'non_easy_empty_unknown5': positives + [make_unknown_item(train_by_id[sid], None) for sid in empty_ids],
        'non_easy_shuffle_unknown5': positives + [make_unknown_item(train_by_id[sid], shuffle_by_id[sid]['audio']) for sid in shuffle_ids],
        'non_easy_empty_shuffle_unknown10': positives + [make_unknown_item(train_by_id[sid], None) for sid in combo_empty_ids] + [make_unknown_item(train_by_id[sid], shuffle_by_id[sid]['audio']) for sid in combo_shuffle_ids],
    }
    for name,data in variants.items():
        rng.shuffle(data)
        write_jsonl(args.sft_out_dir/name/'train.jsonl', data)
        made[name]=len(data)
    (args.sft_out_dir/'summary.json').write_text(json.dumps(made,indent=2,ensure_ascii=False)+'\n', encoding='utf-8')
    print('bucket_csv', csv_path)
    for k,v in made.items(): print(f'{k}: {v}')


if __name__ == '__main__':
    main()
