from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.parse_args()
    raise SystemExit(
        "Qwen3.5-Omni eval is not runnable yet because no public/local "
        "Qwen3.5-Omni weights are configured. When open weights appear, use the "
        "Qwen3 Omni eval path as the implementation template and set the local "
        "model path explicitly."
    )


if __name__ == "__main__":
    main()
