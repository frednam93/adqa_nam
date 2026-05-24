"""Small Telegram notifier for long DCASE jobs.

Credentials are intentionally read from environment variables only:
TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import urllib.parse
import urllib.request
from pathlib import Path


def load_env_file() -> None:
    env_path = Path(os.environ.get("TELEGRAM_ENV_FILE", "~/.config/codex/telegram.env")).expanduser()
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in os.environ:
            continue
        try:
            parsed = shlex.split(value, posix=True)
            os.environ[key] = parsed[0] if parsed else ""
        except ValueError:
            os.environ[key] = value.strip().strip("'\"")


def send_message(text: str) -> bool:
    load_env_file()
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("telegram notify skipped: TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID not set", file=sys.stderr)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": text[:3900],
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=20) as resp:
        resp.read()
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", default="")
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--title", default="")
    args = parser.parse_args()

    parts = []
    if args.title:
        parts.append(args.title)
    if args.message:
        parts.append(args.message)
    if args.file:
        parts.append(args.file.read_text(encoding="utf-8").strip())
    text = "\n".join(p for p in parts if p)
    if not text:
        raise SystemExit("empty telegram message")
    send_message(text)


if __name__ == "__main__":
    main()
