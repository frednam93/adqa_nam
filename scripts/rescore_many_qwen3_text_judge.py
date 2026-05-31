from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForTextToWaveform, AutoProcessor, BitsAndBytesConfig

from rescore_with_qwen3_text_judge import build_prompt, load_jsonl, parse_letter


def load_model(model_path: Path):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.enable_audio_output = False
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForTextToWaveform.from_pretrained(
        model_path,
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
    return processor, model, next(model.parameters()).device


def judge_one(processor, model, device, choices: list[str], response: str, max_new_tokens: int) -> tuple[int, str]:
    prompt = build_prompt(choices, response)
    conv = [
        {'role': 'system', 'content': [{'type': 'text', 'text': 'You normalize model responses to multiple-choice letters.'}]},
        {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
    ]
    text = processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, return_tensors='pt', padding=True)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    with torch.inference_mode():
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_audio=False,
            thinker_return_dict_in_generate=True,
            use_audio_in_video=False,
        )
    if isinstance(gen, tuple):
        gen = gen[0]
    if hasattr(gen, 'sequences'):
        gen = gen.sequences
    new_tokens = gen[:, inputs['input_ids'].shape[1]:]
    raw = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return parse_letter(raw, len(choices)), raw


def rescore_file(pred_path: Path, manifest: dict[str, dict], out_dir: Path, processor, model, device, max_new_tokens: int) -> dict:
    preds = load_jsonl(pred_path)
    out_path = out_dir / f'{pred_path.stem}_qwen3_judge_parsebad.jsonl'
    out_rows = []
    for row in preds:
        item = manifest[row['id']]
        judged_idx = row.get('prediction_index', -1)
        judged_raw = ''
        method = 'original'
        if judged_idx == -1:
            judged_idx, judged_raw = judge_one(
                processor,
                model,
                device,
                item['choices'],
                row.get('raw_generation') or row.get('prediction') or '',
                max_new_tokens,
            )
            method = 'qwen3_judge'
        out = dict(row)
        out.update({
            'judge_method': method,
            'judge_raw': judged_raw,
            'judge_prediction_index': judged_idx,
            'judge_prediction': item['choices'][judged_idx] if judged_idx >= 0 else (row.get('prediction') or ''),
            'judge_correct': judged_idx == item['answer_index'],
        })
        out_rows.append(out)

    out_path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in out_rows) + '\n', encoding='utf-8')
    strict = sum(bool(r.get('correct')) for r in preds)
    judged = sum(bool(r.get('judge_correct')) for r in out_rows)
    parse_bad = sum(1 for r in preds if r.get('prediction_index', -2) == -1)
    judge_bad = sum(1 for r in out_rows if r.get('judge_prediction_index') == -1)
    return {
        'file': pred_path.name,
        'output': str(out_path),
        'n': len(out_rows),
        'parse_bad': parse_bad,
        'strict_correct': strict,
        'strict_acc': strict / len(out_rows),
        'judge_correct': judged,
        'judge_acc': judged / len(out_rows),
        'judge_bad': judge_bad,
        'delta': judged - strict,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--pred-dir', type=Path, default=Path('outputs'))
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--min-parse-bad', type=int, default=24)
    ap.add_argument('--model', type=Path, default=Path('/home/user/ssdmain/models/dcase_adqa/qwen3_omni_30b_a3b_instruct'))
    ap.add_argument('--max-new-tokens', type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    manifest = {r['id']: r for r in load_jsonl(args.manifest)}
    candidates = []
    for pred_path in sorted(args.pred_dir.glob('qwen3_audio_dep*_dev*.jsonl')):
        preds = load_jsonl(pred_path)
        if len(preds) < 1000:
            continue
        parse_bad = sum(1 for r in preds if r.get('prediction_index', -2) == -1)
        if parse_bad >= args.min_parse_bad:
            candidates.append((parse_bad, pred_path))
    candidates.sort(reverse=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f'load model; files={len(candidates)} min_parse_bad={args.min_parse_bad}', flush=True)
    processor, model, device = load_model(args.model)
    summaries = []
    for idx, (parse_bad, pred_path) in enumerate(candidates, 1):
        print(f'[{idx}/{len(candidates)}] {pred_path.name} parse_bad={parse_bad}', flush=True)
        summaries.append(rescore_file(pred_path, manifest, args.out_dir, processor, model, device, args.max_new_tokens))
        s = summaries[-1]
        print(f"  strict={s['strict_acc']:.4f} judge={s['judge_acc']:.4f} delta={s['delta']} judge_bad={s['judge_bad']}", flush=True)
    summary_path = args.out_dir / 'qwen3_judge_parsebad_summary.json'
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {summary_path}', flush=True)


if __name__ == '__main__':
    main()
