from __future__ import annotations

import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qkd_epc import (
    init_epc,
    set_epc_temperature,
    set_epc_voltages,
    validate_voltages,
)
from qkd_names import (
    current_utc_iso,
    make_record_id,
    safe_filename_part,
    sleep_until_utc,
)
from qkd_network import recv_json, send_file, send_json

try:
    import QuTAG_MC as qt
except ImportError as exc:
    print("ERROR: Failed to import QuTAG_MC:", exc)
    sys.exit(1)


@dataclass(frozen=True)
class BobConfig:
    host: str
    port: int
    record_dir: Path
    default_record_seconds: float = 10.0
    accept_poll_seconds: float = 0.5
    command_timeout_seconds: float = 2.0
    epc_enabled: bool = True
    epc_device_ref: int = "0000872235"
    epc_start_temperature: float = 50.0


BOB_CONFIG = BobConfig(
    host="0.0.0.0",
    port=5001,
    record_dir=Path(r"C:\Users\RKAdmin\Desktop\LongDistanceQKD\BobData"),
)


@dataclass(frozen=True)
class BobRecording:
    record_id: str
    duration_seconds: float
    path: Path
    alice_time_utc: str | None
    start_time_utc: str | None


class BobRecorder:
    def __init__(self, tagger, config: BobConfig) -> None:
        self.tagger = tagger
        self.config = config

    def record(self, command: dict[str, Any]) -> BobRecording:
        duration_seconds = float(
            command.get("seconds", self.config.default_record_seconds)
        )
        if duration_seconds <= 0:
            raise ValueError(
                f"RECORD seconds must be positive, received {duration_seconds}"
            )

        record_id = safe_filename_part(command.get("record_id") or make_record_id())
        start_time_utc = command.get("start_time_utc")
        alice_time_utc = command.get("alice_time_utc")
        path = self._recording_path(duration_seconds, record_id)

        self.config.record_dir.mkdir(parents=True, exist_ok=True)
        if start_time_utc:
            print(f"[Bob] Armed record_id={record_id}; waiting until {start_time_utc}")
            sleep_until_utc(start_time_utc)

        print(
            f"[Bob] Recording {duration_seconds:.1f} s "
            f"with record_id={record_id} to {path}"
        )
        self.tagger.setExposureTime(int(duration_seconds * 1000))
        self.tagger.writeTimestamps(str(path), self.tagger.FILEFORMAT_BINARY)
        try:
            time.sleep(duration_seconds)
        finally:
            self.stop_timestamp_writing()

        if not path.is_file():
            raise FileNotFoundError(
                f"Bob recording stage did not create record_id={record_id}: {path}"
            )

        return BobRecording(
            record_id=record_id,
            duration_seconds=duration_seconds,
            path=path,
            alice_time_utc=alice_time_utc,
            start_time_utc=start_time_utc,
        )

    def stop_timestamp_writing(self) -> None:
        self.tagger.writeTimestamps("", self.tagger.FILEFORMAT_NONE)

    def delete_recording(self, filename: str) -> Path:
        safe_name = Path(filename).name
        if safe_name != filename or not safe_name.startswith("bob_") or not safe_name.endswith(".bin"):
            raise ValueError(f"Invalid Bob recording filename: {filename!r}")

        path = self.config.record_dir / safe_name
        if not path.is_file():
            raise FileNotFoundError(f"Bob recording does not exist: {path}")
        path.unlink()
        print(f"[Bob] Deleted optimizer recording: {path}")
        return path

    def _recording_path(
        self,
        duration_seconds: float,
        record_id: str,
    ) -> Path:
        return (
            self.config.record_dir / f"bob_{record_id}_exp_{duration_seconds:.1f}s.bin"
        )


class BobCommandServer:
    def __init__(
        self,
        config: BobConfig,
        recorder: BobRecorder,
        epc,
    ) -> None:
        self.config = config
        self.recorder = recorder
        self.epc = epc
        self.handlers = {
            "PING": self._handle_ping,
            "TIME_CHECK": self._handle_time_check,
            "STOP": self._handle_stop,
            "SET_VOLTAGES": self._handle_set_voltages,
            "ZERO_VOLTAGES": self._handle_zero_voltages,
            "SET_TEMPERATURE": self._handle_set_temperature,
            "DELETE_RECORDING": self._handle_delete_recording,
            "RECORD": self._handle_record,
        }

    def serve_forever(self) -> None:
        keep_running = True
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.config.host, self.config.port))
            server.listen(1)
            server.settimeout(self.config.accept_poll_seconds)
            print(f"[Bob] Listening on {self.config.host}:{self.config.port}")

            while keep_running:
                try:
                    connection, address = server.accept()
                except socket.timeout:
                    # Polling lets Python process Ctrl+C even when no client
                    # is connected and accept() would otherwise block forever.
                    continue

                with connection:
                    keep_running = self._handle_connection(
                        connection,
                        address,
                    )

    def _handle_connection(
        self,
        connection: socket.socket,
        address,
    ) -> bool:
        connection.settimeout(self.config.command_timeout_seconds)
        try:
            command = recv_json(connection)
        except socket.timeout:
            print(f"[Bob] Timed out waiting for a command from {address}")
            return True
        except Exception as exc:
            print(f"[Bob] Could not read command from {address}: {exc}")
            return True
        finally:
            connection.settimeout(None)

        command_name = str(command.get("command", "")).upper()
        print(f"[Bob] {address} command: {command_name}")
        handler = self.handlers.get(command_name)
        if handler is None:
            send_json(
                connection,
                {
                    "ok": False,
                    "error": f"Unknown command: {command_name}",
                },
            )
            return True

        try:
            return handler(connection, command)
        except Exception as exc:
            self._stop_recording_after_error()
            print(
                f"[Bob] Command {command_name} from {address} failed: {exc}",
                flush=True,
            )
            try:
                send_json(connection, {"ok": False, "error": str(exc)})
            except Exception:
                pass
            return True

    @staticmethod
    def _handle_ping(
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        send_json(connection, {"ok": True, "message": "PONG"})
        return True

    @staticmethod
    def _handle_time_check(
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        send_json(
            connection,
            {
                "ok": True,
                "bob_time_utc": current_utc_iso(),
            },
        )
        return True

    @staticmethod
    def _handle_stop(
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        send_json(connection, {"ok": True, "message": "Bob stopping"})
        return False

    def _handle_set_voltages(
        self,
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        voltages = self._extract_voltages(command)
        set_epc_voltages("Bob", self.epc, voltages)
        send_json(
            connection,
            {
                "ok": True,
                "message": "Bob EPC voltages updated",
            },
        )
        return True

    def _handle_zero_voltages(
        self,
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        set_epc_voltages("Bob", self.epc, [0.0, 0.0, 0.0, 0.0])
        send_json(
            connection,
            {
                "ok": True,
                "message": "Bob EPC voltages set to zero",
            },
        )
        return True

    def _handle_set_temperature(
        self,
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        if "temperature" not in command:
            raise ValueError("SET_TEMPERATURE requires 'temperature'")
        set_epc_temperature(
            "Bob",
            self.epc,
            float(command["temperature"]),
        )
        send_json(
            connection,
            {
                "ok": True,
                "message": "Bob EPC temperature updated",
            },
        )
        return True

    def _handle_delete_recording(
        self,
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        filename = str(command.get("filename", ""))
        if not filename:
            raise ValueError("DELETE_RECORDING requires 'filename'")
        deleted_path = self.recorder.delete_recording(filename)
        send_json(
            connection,
            {
                "ok": True,
                "message": "Bob recording deleted",
                "name": deleted_path.name,
            },
        )
        return True

    def _handle_record(
        self,
        connection: socket.socket,
        command: dict[str, Any],
    ) -> bool:
        recording = self.recorder.record(command)
        send_json(
            connection,
            {
                "ok": True,
                "message": "Recording complete",
                "record_id": recording.record_id,
            },
        )
        send_file(
            connection,
            recording.path,
            metadata={
                "seconds": recording.duration_seconds,
                "source": "bob",
                "record_id": recording.record_id,
                "alice_time_utc": recording.alice_time_utc,
                "start_time_utc": recording.start_time_utc,
                "bob_finished_utc": current_utc_iso(),
            },
        )
        return True

    @staticmethod
    def _extract_voltages(command: dict[str, Any]) -> list[float]:
        for key in ("voltages", "bob_voltages"):
            if key in command:
                return validate_voltages(command[key])
        raise ValueError("SET_VOLTAGES requires four values in 'voltages'")

    def _stop_recording_after_error(self) -> None:
        try:
            self.recorder.stop_timestamp_writing()
        except Exception:
            pass


def initialize_bob_epc(config: BobConfig):
    if not config.epc_enabled:
        return None
    return init_epc(
        "Bob",
        config.epc_device_ref,
        config.epc_start_temperature,
    )


def shutdown_tagger(tagger) -> None:
    try:
        tagger.writeTimestamps("", tagger.FILEFORMAT_NONE)
        tagger.deInitialize()
    except Exception as exc:
        print(f"[Bob] Tagger shutdown warning: {exc}")


def main() -> None:
    tagger = qt.QuTAG()
    epc = initialize_bob_epc(BOB_CONFIG)
    recorder = BobRecorder(tagger, BOB_CONFIG)
    server = BobCommandServer(BOB_CONFIG, recorder, epc)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Bob] Interrupted by user")
    finally:
        shutdown_tagger(tagger)


if __name__ == "__main__":
    main()
