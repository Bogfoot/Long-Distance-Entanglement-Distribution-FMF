from __future__ import annotations

import socket
import sys
import time
from pathlib import Path

from qkd_names import current_utc_iso, future_utc_iso, make_record_id, safe_filename_part, sleep_until_utc
from qkd_network import receive_file, recv_json, send_json


BOB_HOST = "100.104.228.90"
BOB_PORT = 5001

RECORD_SECONDS = 5.0
SCHEDULE_AHEAD_SECONDS = 2.0
RECORD_COUNT = 1 
PAUSE_BETWEEN_RECORDS = 2
RECORD_ID = None

BASE_DIR = Path(__file__).resolve().parent
ALICE_RECORD_DIR = BASE_DIR / "DataExtSync" / "AliceRaw"
BOB_INCOMING_DIR = BASE_DIR / "DataExtSync" / "Incoming"

# ALICE_INCOMING_DIR = BASE_DIR / "Data" / "AliceRaw"
# BOB_INCOMING_DIR = BASE_DIR / "Data" / "Incoming"


try:
    import QuTAG_MC as qt
except ImportError as exc:
    print("ERROR: Failed to import QuTAG_MC:", exc)
    sys.exit(1)


def make_alice_recording_path(seconds: float, record_id: str) -> Path:
    return ALICE_RECORD_DIR / f"alice_{record_id}_exp_{seconds:.1f}s.bin"


def record_alice_bin_file(tt, seconds: float, record_id: str, start_time_utc: str) -> Path:
    ALICE_RECORD_DIR.mkdir(parents=True, exist_ok=True)
    path = make_alice_recording_path(seconds, record_id)

    print(f"[Alice] Armed record_id={record_id}; waiting until {start_time_utc}")
    sleep_until_utc(start_time_utc)

    print(f"[Alice] Recording {seconds:.1f} s to {path}")
    tt.setExposureTime(int(seconds * 1000))
    tt.writeTimestamps(str(path), tt.FILEFORMAT_BINARY)
    time.sleep(seconds)
    tt.writeTimestamps("", tt.FILEFORMAT_NONE)

    if not path.is_file():
        raise FileNotFoundError(f"Alice recording was not created: {path}")
    return path


def collect_pair(
    tt,
    bob_host: str,
    bob_port: int,
    seconds: float,
    schedule_ahead_seconds: float,
    record_id: str | None = None,
) -> tuple[Path, Path]:
    record_id = safe_filename_part(record_id or make_record_id())
    start_time_utc = future_utc_iso(schedule_ahead_seconds)
    timeout = max(60.0, seconds + schedule_ahead_seconds + 30.0)

    command = {
        "command": "RECORD",
        "seconds": seconds,
        "record_id": record_id,
        "alice_time_utc": current_utc_iso(),
        "start_time_utc": start_time_utc,
    }

    with socket.create_connection((bob_host, bob_port), timeout=timeout) as sock:
        print(
            f"[Alice] Requesting Bob recording record_id={record_id}, "
            f"start_time_utc={start_time_utc}"
        )
        send_json(sock, command)

        alice_path = record_alice_bin_file(tt, seconds, record_id, start_time_utc)

        status = recv_json(sock)
        if not status.get("ok"):
            raise RuntimeError(f"Bob recording failed: {status.get('error')}")

        bob_path, header, sha_ok = receive_file(sock, BOB_INCOMING_DIR)
        if not sha_ok:
            raise RuntimeError(f"SHA256 check failed for received Bob file: {bob_path}")

    print(f"[Alice] Saved Alice file: {alice_path}")
    print(f"[Alice] Saved Bob file:   {bob_path} ({header['size']} bytes)")
    return alice_path, bob_path


def main() -> None:
    if RECORD_COUNT < 1:
        raise ValueError("RECORD_COUNT must be at least 1")
    if RECORD_ID is not None and RECORD_COUNT != 1:
        raise ValueError("RECORD_ID can only be used when RECORD_COUNT is 1")

    tt = qt.QuTAG()
    # tt.setExposureTime(1000)
    try:
        for index in range(RECORD_COUNT):
            if RECORD_COUNT > 1:
                print(f"[Alice] Collecting pair {index + 1}/{RECORD_COUNT}")
            collect_pair(
                tt,
                BOB_HOST,
                BOB_PORT,
                RECORD_SECONDS,
                SCHEDULE_AHEAD_SECONDS,
                RECORD_ID,
            )
            if index + 1 < RECORD_COUNT:
                time.sleep(PAUSE_BETWEEN_RECORDS)
    finally:
        try:
            tt.writeTimestamps("", tt.FILEFORMAT_NONE)
            tt.deInitialize()
        except Exception:
            pass


if __name__ == "__main__":
    main()
