from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

LABELS = [
    "speaker_identity",
    "speech_content",
    "speech_paralinguistic",
    "music",
    "sound_event",
    "temporal_reasoning",
    "counting_quantity",
    "linguistic_form",
    "scene_context",
    "caption_summary",
    "acoustic_property",
    "other",
]

RULES: list[tuple[str, list[str]]] = [
    ("speaker_identity", [
        "speaker clip", "same individual", "same speaker", "which speaker", "who is speaking", "voice belongs",
        "person speaking", "speaker", "individual", "male voice", "female voice",
    ]),
    ("speech_content", [
        "what does", "what did", "what is said", "what was said", "what word", "which word", "phrase", "sentence",
        "transcript", "says", "said", "utterance", "spoken", "according to the audio",
    ]),
    ("speech_paralinguistic", [
        "emotion", "emotional", "tone", "mood", "attitude", "accent", "laugh", "cry", "cough", "sigh", "vocal", "voice heard",
        "characterizes the vocal", "performance", "pause", "prosody", "intonation",
    ]),
    ("music", [
        "music", "song", "singing", "singer", "instrument", "guitar", "piano", "drum", "melody", "rhythm", "track", "vocal performance",
        "genre", "musical", "orchestra", "instrumental",
    ]),
    ("sound_event", [
        "sound", "noise", "heard", "event", "occurs", "happens", "animal", "dog", "bird", "car", "engine", "door", "water", "rain",
        "applause", "explosion", "crash", "footsteps", "background",
    ]),
    ("temporal_reasoning", [
        "before", "after", "then", "next", "first", "last", "near the end", "at the beginning", "simultaneously", "during",
        "following", "sequence", "order", "transition", "shift",
    ]),
    ("counting_quantity", [
        "how many", "number of", "count", "how often", "times", "multiple", "several", "once", "twice", "three", "four", "five",
    ]),
    ("linguistic_form", [
        "syllable", "syllables", "phoneme", "phonetic", "pronunciation", "rhyme", "word you just heard", "language", "grammar",
    ]),
    ("caption_summary", [
        "main idea", "summary", "summarizes", "best describes", "most accurately captures", "overall", "description of the audio",
    ]),
    ("scene_context", [
        "where", "setting", "scene", "environment", "location", "context", "situation", "activity", "what are they doing",
    ]),
    ("acoustic_property", [
        "loud", "quiet", "volume", "pitch", "high-pitched", "low-pitched", "tempo", "speed", "fast", "slow", "reverber", "clear", "distorted",
    ]),
]

# Priority is used when multiple labels match. It intentionally puts more specific
# reasoning/question-form classes before broad audio-domain classes.
PRIORITY = {
    "speaker_identity": 0,
    "linguistic_form": 1,
    "counting_quantity": 2,
    "temporal_reasoning": 3,
    "speech_content": 4,
    "speech_paralinguistic": 5,
    "music": 6,
    "sound_event": 7,
    "caption_summary": 8,
    "scene_context": 9,
    "acoustic_property": 10,
    "other": 99,
}


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def norm_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() == "true"
    return bool(value)


def text_blob(item: dict) -> str:
    choices = item.get("choices", [])
    if not isinstance(choices, list):
        choices = list(choices)
    return (item.get("question", "") + " " + " ".join(map(str, choices))).lower()


def classify(item: dict) -> tuple[str, str, list[str]]:
    blob = text_blob(item)
    hits = []
    evidence = []
    for label, patterns in RULES:
        matched = [p for p in patterns if p in blob]
        if matched:
            hits.append(label)
            evidence.extend(f"{label}:{p}" for p in matched[:3])
    if not hits:
        return "other", "low", []
    hits = sorted(set(hits), key=lambda x: PRIORITY[x])
    confidence = "high" if len(evidence) >= 2 else "medium"
    return hits[0], confidence, evidence[:8]


def train_distribution(rows: Iterable[dict]) -> Counter:
    c = Counter()
    for row in rows:
        qt = row.get("question_type") or "None"
        src = row.get("source_dataset") or "None"
        c[(src, qt)] += 1
    return c


def pct(n: int, d: int) -> str:
    return f"{n / d:.4f}" if d else "nan"


def outcome(base_ok: bool, tuned_ok: bool) -> str:
    if base_ok and tuned_ok:
        return "both_correct"
    if base_ok and not tuned_ok:
        return "base_only"
    if not base_ok and tuned_ok:
        return "tuned_only"
    return "both_wrong"


def summarize(rows: list[dict], base: dict[str, dict], tuned: dict[str, dict]) -> tuple[list[dict], dict[str, Counter]]:
    per_sample = []
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for item in rows:
        sid = item["id"]
        b = base.get(sid, {})
        t = tuned.get(sid, {})
        b_ok = norm_bool(b.get("correct", False))
        t_ok = norm_bool(t.get("correct", False))
        cat, conf, ev = classify(item)
        out = outcome(b_ok, t_ok)
        by_cat[cat]["n"] += 1
        by_cat[cat]["base_correct"] += int(b_ok)
        by_cat[cat]["tuned_correct"] += int(t_ok)
        by_cat[cat][out] += 1
        by_cat[cat]["base_parse_bad"] += int(b.get("prediction_index", 0) == -1)
        by_cat[cat]["tuned_parse_bad"] += int(t.get("prediction_index", 0) == -1)
        per_sample.append({
            "id": sid,
            "category": cat,
            "category_confidence": conf,
            "category_evidence": "; ".join(ev),
            "outcome": out,
            "base_correct": b_ok,
            "tuned_correct": t_ok,
            "base_prediction": b.get("prediction", ""),
            "tuned_prediction": t.get("prediction", ""),
            "answer": item.get("answer", ""),
            "answer_index": item.get("answer_index", ""),
            "base_prediction_index": b.get("prediction_index", ""),
            "tuned_prediction_index": t.get("prediction_index", ""),
            "question": item.get("question", ""),
            "choices": json.dumps(item.get("choices", []), ensure_ascii=False),
            "audio": item.get("audio", ""),
            "base_raw": b.get("raw_generation", ""),
            "tuned_raw": t.get("raw_generation", ""),
        })
    return per_sample, by_cat


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def representative_samples(rows: list[dict], limit: int) -> list[dict]:
    selected = []
    # Prefer informative disagreement and hard cases first.
    priority_outcomes = ["base_only", "tuned_only", "both_wrong"]
    for cat in LABELS:
        cat_rows = [r for r in rows if r["category"] == cat]
        for out in priority_outcomes:
            take = [r for r in cat_rows if r["outcome"] == out]
            for r in take[:limit]:
                selected.append(r)
    return selected


def write_report(
    path: Path,
    train_counts: Counter,
    dev_counts: Counter,
    by_cat: dict[str, Counter],
    base_name: str,
    tuned_name: str,
    per_sample_path: Path,
    reps_path: Path,
) -> None:
    lines = []
    lines.append("# DCASE ADQA Error Taxonomy")
    lines.append("")
    lines.append("## Inputs")
    lines.append(f"- base=`{base_name}`")
    lines.append(f"- tuned=`{tuned_name}`")
    lines.append(f"- per-sample CSV=`{per_sample_path}`")
    lines.append(f"- representative JSONL=`{reps_path}`")
    lines.append("")
    lines.append("## Train Source / Question-Type Distribution")
    lines.append("| source_dataset | question_type | count |")
    lines.append("|---|---|---:|")
    for (src, qt), n in train_counts.most_common():
        lines.append(f"| {src} | {qt} | {n} |")
    lines.append("")
    lines.append("## Dev Text Taxonomy Distribution")
    lines.append("| category | count | share |")
    lines.append("|---|---:|---:|")
    total_dev = sum(dev_counts.values())
    for cat, n in dev_counts.most_common():
        lines.append(f"| {cat} | {n} | {n / total_dev:.3f} |")
    lines.append("")
    lines.append("## Base vs Tuned By Category")
    lines.append("| category | n | base acc | tuned acc | base only | tuned only | both wrong | base parse_bad | tuned parse_bad |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat, c in sorted(by_cat.items(), key=lambda kv: (-kv[1]["n"], kv[0])):
        n = c["n"]
        lines.append(
            f"| {cat} | {n} | {pct(c['base_correct'], n)} | {pct(c['tuned_correct'], n)} | "
            f"{c['base_only']} | {c['tuned_only']} | {c['both_wrong']} | {c['base_parse_bad']} | {c['tuned_parse_bad']} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("- This taxonomy is text-derived from dev question/choice wording; it is suitable for triage, not final dataset-quality judgment.")
    lines.append("- Human listening is still required for label ambiguity, inaudible evidence, and whether the model failure is perceptual vs reasoning/formatting.")
    lines.append("- `base_only` rows are the highest-priority cases for diagnosing why finetuning degraded Qwen3 base behavior.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--dev-manifest", type=Path, required=True)
    p.add_argument("--base-result", type=Path, required=True)
    p.add_argument("--tuned-result", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--sample-limit", type=int, default=6)
    args = p.parse_args()

    train_rows = load_jsonl(args.train_manifest)
    dev_rows = load_jsonl(args.dev_manifest)
    base_rows = {r["id"]: r for r in load_jsonl(args.base_result)}
    tuned_rows = {r["id"]: r for r in load_jsonl(args.tuned_result)}

    per_sample, by_cat = summarize(dev_rows, base_rows, tuned_rows)
    dev_counts = Counter(r["category"] for r in per_sample)
    train_counts = train_distribution(train_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.out_dir / "qwen3_base_vs_ckpt9000_by_category.csv"
    reps_path = args.out_dir / "qwen3_base_vs_ckpt9000_representatives.jsonl"
    report_path = args.out_dir / "taxonomy_report.md"

    write_csv(per_sample_path, per_sample)
    reps = representative_samples(per_sample, args.sample_limit)
    with reps_path.open("w", encoding="utf-8") as f:
        for row in reps:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(
        report_path,
        train_counts,
        dev_counts,
        by_cat,
        str(args.base_result),
        str(args.tuned_result),
        per_sample_path,
        reps_path,
    )
    print(f"wrote {report_path}")
    print(f"wrote {per_sample_path}")
    print(f"wrote {reps_path}")


if __name__ == "__main__":
    main()
