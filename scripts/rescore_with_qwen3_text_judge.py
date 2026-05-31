from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoProcessor, BitsAndBytesConfig


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def build_prompt(choices: list[str], response: str) -> str:
    labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    opts = '\n'.join(f'({labels[i]}) {c}' for i, c in enumerate(choices))
    return (
        'You are an answer normalizer for a multiple-choice evaluation.\n'
        'Your only task is to map the model response to one of the listed choices.\n'
        'Do not answer the original question. Do not use outside knowledge.\n'
        'If the response clearly refers to one choice, output only its letter.\n'
        'If it is ambiguous or does not match any choice, output only X.\n\n'
        f'Choices:\n{opts}\n\n'
        f'Model response:\n{response}\n\n'
        'Normalized answer letter:'
    )


def parse_letter(text: str, n: int) -> int:
    letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:n]
    text = text.strip()
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1].strip()
    m = re.search(rf'(?i)\b([{re.escape(letters)}X])\b', text)
    if not m:
        m = re.search(rf'(?i)([{re.escape(letters)}X])', text[:12])
    if not m:
        return -1
    letter = m.group(1).upper()
    if letter == 'X':
        return -1
    return letters.index(letter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--model', type=Path, default=Path('/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--only-parse-bad', action='store_true')
    ap.add_argument('--max-new-tokens', type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    preds = load_jsonl(args.pred)
    manifest = {r['id']: r for r in load_jsonl(args.manifest)}

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    config.enable_audio_output = False
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        args.model,
        config=config,
        trust_remote_code=True,
        quantization_config=qconf,
        dtype=torch.bfloat16,
        device_map={'': 0},
        low_cpu_mem_usage=True,
    )
    if hasattr(model, 'disable_talker'):
        model.disable_talker()
    model.eval()
    device = next(model.parameters()).device

    out_rows = []
    for i, row in enumerate(preds, 1):
        sid = row['id']
        item = manifest[sid]
        use_judge = (not args.only_parse_bad) or row.get('prediction_index') == -1
        judged_raw = ''
        judged_idx = row.get('prediction_index', -1)
        method = 'original'
        if use_judge:
            prompt = build_prompt(item['choices'], row.get('raw_generation') or row.get('prediction') or '')
            conv = [
                {'role': 'system', 'content': [{'type': 'text', 'text': 'You normalize model responses to multiple-choice letters.'}]},
                {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
            ]
            text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inputs = processor(text=text, return_tensors='pt', padding=True)
            inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
            with torch.inference_mode():
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, return_audio=False, thinker_return_dict_in_generate=True, use_audio_in_video=False)
            if isinstance(gen, tuple):
                gen = gen[0]
            if hasattr(gen, 'sequences'):
                gen = gen.sequences
            new_tokens = gen[:, inputs['input_ids'].shape[1]:]
            judged_raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            judged_idx = parse_letter(judged_raw, len(item['choices']))
            method = 'qwen3_judge'
        correct = judged_idx == item['answer_index']
        out = dict(row)
        out.update({
            'judge_method': method,
            'judge_raw': judged_raw,
            'judge_prediction_index': judged_idx,
            'judge_prediction': item['choices'][judged_idx] if judged_idx >= 0 else (row.get('prediction') or ''),
            'judge_correct': correct,
        })
        out_rows.append(out)
        if i % 10 == 0:
            print(f'processed {i}/{len(preds)}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in out_rows) + '\n', encoding='utf-8')
    strict = sum(bool(r.get('correct')) for r in preds)
    judge = sum(bool(r.get('judge_correct')) for r in out_rows)
    bad = sum(r.get('judge_prediction_index') == -1 for r in out_rows)
    print(f'strict={strict}/{len(out_rows)} acc={strict/len(out_rows):.4f}')
    print(f'judge={judge}/{len(out_rows)} acc={judge/len(out_rows):.4f} judge_bad={bad}')


if __name__ == '__main__':
    main()
