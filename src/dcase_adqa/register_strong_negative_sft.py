from __future__ import annotations

import json
from pathlib import Path

ROOT = Path('/home/user/ssdmain/dcase-adqa')
FAC = ROOT / 'external/Fun-Audio-Chat'
DATA_INFO = FAC / 'training/data/dataset_info.json'
SFT_DIR = Path('/home/user/ssdmain/datasets/dcase2026_task5/qwen3_omni_sft_audio_dep_full')
RUNS = [
    'strong_empty_unknown5',
    'strong_shuffle_unknown5',
    'strong_empty_shuffle_unknown10',
]


def main() -> None:
    info = json.loads(DATA_INFO.read_text())
    for run in RUNS:
        info[f'dcase_adqa_qwen3_audio_dep_full_{run}'] = {
            'file_name': str(SFT_DIR / run / 'train.jsonl'),
            'formatting': 'sharegpt',
            'columns': {'system': 'system', 'messages': 'messages', 'audios': 'audios'},
            'tags': {'role_tag': 'role', 'content_tag': 'content', 'user_tag': 'user', 'assistant_tag': 'assistant'},
        }
    DATA_INFO.write_text(json.dumps(info, indent=2, ensure_ascii=False) + '\n')

    base_cfg = (FAC / 'training/configs/dcase_adqa_qwen3_omni_qlora.yaml').read_text().splitlines()
    for run in RUNS:
        lines = []
        for line in base_cfg:
            if line.startswith('dataset:'):
                lines.append(f'dataset: dcase_adqa_qwen3_audio_dep_full_{run}')
            elif line.startswith('output_dir:'):
                lines.append(f'output_dir: {ROOT}/outputs/qwen3_audio_dep_full_{run}_3k')
            elif line.startswith('num_train_epochs:'):
                continue
            elif line.startswith('save_steps:'):
                lines.append('save_steps: 1000')
            elif line.startswith('logging_steps:'):
                lines.append('logging_steps: 10')
            else:
                lines.append(line)
        lines.append('max_steps: 3000')
        (FAC / f'training/configs/dcase_adqa_qwen3_audio_dep_full_{run}_3k.yaml').write_text('\n'.join(lines) + '\n')
    print('registered', ', '.join(RUNS))


if __name__ == '__main__':
    main()
