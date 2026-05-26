from __future__ import annotations

import json
import random
from pathlib import Path

from dcase_adqa.analyze_errors import load_jsonl, norm_bool
from dcase_adqa.build_full_audio_dependency_sft import bucket, make_unknown_item, write_jsonl
from dcase_adqa.prepare_qwen3_omni_sft import convert_item

TRAIN_MANIFEST = Path('/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests/train_full_normal.jsonl')
SHUFFLE_MANIFEST = Path('/home/user/ssdmain/dcase-adqa/outputs/ablation_manifests/train_full_shuffle_audio_random.jsonl')
ABLATION_DIR = Path('/home/user/ssdmain/dcase-adqa/outputs/ablations/qwen3_base_train_full_min')
SFT_DIR = Path('/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full')
SEED = 20260527


def main() -> None:
    rng = random.Random(SEED)
    train = load_jsonl(TRAIN_MANIFEST)
    train_by_id = {r['id']: r for r in train}
    shuffle_by_id = {r['id']: r for r in load_jsonl(SHUFFLE_MANIFEST)}
    res = {
        name: {r['id']: r for r in load_jsonl(ABLATION_DIR / f'{name}.jsonl')}
        for name in ['normal', 'empty_audio_question', 'shuffle_audio_random']
    }
    strong_ids = []
    for item in train:
        sid = item['id']
        b = bucket(
            norm_bool(res['normal'][sid].get('correct', False)),
            norm_bool(res['empty_audio_question'][sid].get('correct', False)),
            norm_bool(res['shuffle_audio_random'][sid].get('correct', False)),
        )
        if b == 'strong_audio_dependent':
            strong_ids.append(sid)

    positives = [convert_item(train_by_id[sid], 'answer_only') for sid in strong_ids]
    n5 = max(1, round(len(positives) * 0.05))
    empty_ids = rng.sample(strong_ids, min(n5, len(strong_ids)))
    shuffle_ids = rng.sample(strong_ids, min(n5, len(strong_ids)))
    combo_empty_ids = rng.sample(strong_ids, min(n5, len(strong_ids)))
    combo_shuffle_ids = rng.sample(strong_ids, min(n5, len(strong_ids)))

    variants = {
        'strong_empty_unknown5': positives + [make_unknown_item(train_by_id[sid], None) for sid in empty_ids],
        'strong_shuffle_unknown5': positives + [make_unknown_item(train_by_id[sid], shuffle_by_id[sid]['audio']) for sid in shuffle_ids],
        'strong_empty_shuffle_unknown10': positives
        + [make_unknown_item(train_by_id[sid], None) for sid in combo_empty_ids]
        + [make_unknown_item(train_by_id[sid], shuffle_by_id[sid]['audio']) for sid in combo_shuffle_ids],
    }

    made = {'strong_ac_base': len(positives), 'unknown5_count': n5}
    for name, data in variants.items():
        rng.shuffle(data)
        write_jsonl(SFT_DIR / name / 'train.jsonl', data)
        made[name] = len(data)
        print(f'{name}: {len(data)}')

    summary_path = SFT_DIR / 'strong_negative_summary.json'
    summary_path.write_text(json.dumps(made, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print('summary', summary_path)


if __name__ == '__main__':
    main()
