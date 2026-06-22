from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qkd_names import (
    current_utc_iso,
    future_utc_iso,
    make_record_id,
    safe_filename_part,
    sleep_until_utc,
)
from qkd_network import receive_file, recv_json, send_json


@dataclass(frozen=True)
class AcquisitionConfig:
    bob_host: str
    bob_port: int
    alice_record_dir: Path
    incoming_dir: Path
    schedule_ahead_seconds: float = 3.0
    minimum_timeout_seconds: float = 60.0
    timeout_margin_seconds: float = 30.0


@dataclass(frozen=True)
class AcquisitionPair:
    record_id: str
    start_time_utc: str
    duration_seconds: float
    alice_path: Path
    bob_path: Path


class AcquisitionError(RuntimeError):
    """Raised when a paired Alice/Bob acquisition cannot be completed."""


def send_bob_command(
    config: AcquisitionConfig,
    command: dict[str, Any],
    *,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    with socket.create_connection(
        (config.bob_host, config.bob_port),
        timeout=timeout_seconds,
    ) as connection:
        send_json(connection, command)
        return recv_json(connection)


def _alice_recording_path(
    config: AcquisitionConfig,
    duration_seconds: float,
    record_id: str,
) -> Path:
    return (
        config.alice_record_dir
        / f"alice_{record_id}_exp_{duration_seconds:.1f}s.bin"
    )


def _record_alice_file(
    tagger,
    config: AcquisitionConfig,
    duration_seconds: float,
    record_id: str,
    start_time_utc: str,
) -> Path:
    config.alice_record_dir.mkdir(parents=True, exist_ok=True)
    path = _alice_recording_path(config, duration_seconds, record_id)

    sleep_until_utc(start_time_utc)

    print(
        f"[Alice] Recording started | record_id={record_id} | "
        f"duration={duration_seconds:.1f} s"
    )
    tagger.setExposureTime(int(duration_seconds * 1000))
    tagger.writeTimestamps(str(path), tagger.FILEFORMAT_BINARY)
    time.sleep(duration_seconds)
    tagger.writeTimestamps("", tagger.FILEFORMAT_NONE)

    if not path.is_file():
        raise AcquisitionError(
            f"Alice recording stage did not create record_id={record_id}: {path}"
        )
    return path


def acquire_pair(
    tagger,
    config: AcquisitionConfig,
    duration_seconds: float,
    *,
    record_id: str | None = None,
) -> AcquisitionPair:
    selected_record_id = safe_filename_part(record_id or make_record_id())
    start_time_utc = future_utc_iso(config.schedule_ahead_seconds)
    timeout_seconds = max(
        config.minimum_timeout_seconds,
        duration_seconds + config.timeout_margin_seconds,
    )
    command = {
        "command": "RECORD",
        "seconds": duration_seconds,
        "record_id": selected_record_id,
        "alice_time_utc": current_utc_iso(),
        "start_time_utc": start_time_utc,
    }

    try:
        connection = socket.create_connection(
            (config.bob_host, config.bob_port),
            timeout=timeout_seconds,
        )
    except OSError as exc:
        raise AcquisitionError(
            "Could not connect to Bob for "
            f"record_id={selected_record_id} at "
            f"{config.bob_host}:{config.bob_port}: {exc}"
        ) from exc

    with connection:
        send_json(connection, command)
        alice_path = _record_alice_file(
            tagger,
            config,
            duration_seconds,
            selected_record_id,
            start_time_utc,
        )

        status = recv_json(connection)
        if not status.get("ok"):
            raise AcquisitionError(
                f"Bob recording failed for record_id={selected_record_id}: "
                f"{status.get('error', 'unknown error')}"
            )

        bob_path, header, checksum_ok = receive_file(
            connection,
            config.incoming_dir,
        )
        if not checksum_ok:
            raise AcquisitionError(
                f"Bob file checksum failed for record_id={selected_record_id}: "
                f"{bob_path}"
            )

        metadata = header.get("metadata", {})
        received_record_id = metadata.get("record_id", selected_record_id)
        if received_record_id != selected_record_id:
            raise AcquisitionError(
                "Bob returned a mismatched recording: "
                f"requested={selected_record_id}, received={received_record_id}"
            )

        print(
            f"[Alice] Recording complete | record_id={selected_record_id} | "
            f"Alice={alice_path.stat().st_size / 1_000_000:.1f} MB | "
            f"Bob={header['size'] / 1_000_000:.1f} MB"
        )
        return AcquisitionPair(
            record_id=selected_record_id,
            start_time_utc=start_time_utc,
            duration_seconds=float(duration_seconds),
            alice_path=alice_path,
            bob_path=bob_path,
        )


def delete_acquisition_files(
    acquisition: AcquisitionPair,
    config: AcquisitionConfig,
) -> None:
    """Best-effort cleanup of one completed optimizer acquisition."""
    try:
        reply = send_bob_command(
            config,
            {
                "command": "DELETE_RECORDING",
                "record_id": acquisition.record_id,
                "filename": acquisition.bob_path.name,
            },
        )
        if not reply.get("ok"):
            raise AcquisitionError(
                reply.get("error", "Bob rejected DELETE_RECORDING")
            )
    except Exception as exc:
        print(
            f"[Alice] Could not delete Bob optimizer recording "
            f"record_id={acquisition.record_id}: {exc}"
        )

    for path in (acquisition.alice_path, acquisition.bob_path):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            print(f"[Alice] Could not delete optimizer recording {path}: {exc}")
