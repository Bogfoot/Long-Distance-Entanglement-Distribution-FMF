from __future__ import annotations

import datetime
import re
import time

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

UNSAFE_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
UNSAFE_RECORD_ID_CHARS = re.compile(r"[^A-Za-z0-9_.-]")


def make_record_id() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S.%fZ")


def current_utc_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def future_utc_iso(delay_seconds: float) -> str:
    start = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=delay_seconds)
    return start.isoformat()


def sleep_until_utc(iso_timestamp: str | None) -> None:
    if not iso_timestamp:
        return

    target = datetime.datetime.fromisoformat(iso_timestamp)
    if target.tzinfo is None:
        target = target.replace(tzinfo=datetime.UTC)
    target = target.astimezone(datetime.UTC)

    while True:
        remaining = (target - datetime.datetime.now(datetime.UTC)).total_seconds()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def safe_filename_part(value: str | None, fallback: str = "record", max_length: int = 120) -> str:
    if value is None:
        value = fallback

    cleaned = UNSAFE_FILENAME_CHARS.sub("_", str(value))
    cleaned = UNSAFE_RECORD_ID_CHARS.sub("_", cleaned)
    cleaned = cleaned.strip(" .")
    if not cleaned:
        cleaned = fallback

    stem = cleaned.split(".", 1)[0].upper()
    if stem in WINDOWS_RESERVED_NAMES:
        cleaned = f"{cleaned}_"

    return cleaned[:max_length].rstrip(" .") or fallback
