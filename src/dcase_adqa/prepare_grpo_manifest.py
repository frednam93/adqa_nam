from __future__ import annotations

import argparse
import json
from pathlib import Path

from dcase_adqa.build_full_audio_dependency_sft import write_jsonl
from dcase_adqa.prepare_preference_continuation import strong_audio_items


ROOT = Path("/home/user/ssdmain/dcase-adqa")
DEFAULT_OUTPUT = ROOT / "outputs/ablation_manifests/train_strong_audio_dependent_grpo.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    items = strong_audio_items()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, items)
    print(json.dumps({"output": str(args.output), "items": len(items)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
