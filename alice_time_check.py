from __future__ import annotations

import datetime
import socket
import statistics
import threading
import time

from qkd_network import recv_json, send_json

BOB_HOST = "100.104.228.90"
BOB_PORT = 5001
ALICE_TIME_SERVER_HOST = "0.0.0.0"
ALICE_TIME_SERVER_PORT = 5002

N_CHECKS = 20
WARMUP_CHECKS = 2
PAUSE_SECONDS = 0.25
KEEP_SERVER_ALIVE_AFTER_CHECKS = True
HIGH_RTT_WARNING_MS = 100.0
HIGH_JITTER_WARNING_MS = 20.0


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def to_seconds(delta: datetime.timedelta) -> float:
    return delta.total_seconds()


def parse_utc(value: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.UTC)
    return dt.astimezone(datetime.UTC)


def serve_alice_time(stop_event: threading.Event) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((ALICE_TIME_SERVER_HOST, ALICE_TIME_SERVER_PORT))
        server.listen(1)
        server.settimeout(0.5)
        print(f"Alice time server listening on {ALICE_TIME_SERVER_HOST}:{ALICE_TIME_SERVER_PORT}")

        while not stop_event.is_set():
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue

            with conn:
                try:
                    command = recv_json(conn)
                    name = str(command.get("command", "")).upper()
                    if name == "TIME_CHECK":
                        send_json(conn, {"ok": True, "alice_time_utc": utc_now().isoformat()})
                    elif name == "STOP":
                        send_json(conn, {"ok": True, "message": "Alice time server stopping"})
                        stop_event.set()
                    else:
                        send_json(conn, {"ok": False, "error": f"Unknown command: {name}"})
                except Exception as exc:
                    print(f"Alice time server error from {addr}: {exc}")


def start_alice_time_server() -> threading.Event:
    stop_event = threading.Event()
    thread = threading.Thread(target=serve_alice_time, args=(stop_event,), daemon=True)
    thread.start()
    time.sleep(0.2)
    return stop_event


def one_bob_check() -> tuple[float, float]:
    t0 = utc_now()
    with socket.create_connection((BOB_HOST, BOB_PORT), timeout=10) as sock:
        send_json(sock, {"command": "TIME_CHECK"})
        reply = recv_json(sock)
    t1 = utc_now()

    if not reply.get("ok"):
        raise RuntimeError(reply)

    bob_time = parse_utc(reply["bob_time_utc"])
    alice_midpoint = t0 + (t1 - t0) / 2
    offset_s = to_seconds(bob_time - alice_midpoint)
    round_trip_s = to_seconds(t1 - t0)
    return offset_s, round_trip_s


def check_bob_clock() -> None:
    print(f"Checking Bob clock at {BOB_HOST}:{BOB_PORT}")
    print("offset_ms = Bob time - Alice time estimate")
    print(f"Discarding first {WARMUP_CHECKS} warmup checks from summary")

    offsets_ms = []
    rtts_ms = []
    kept_offsets_ms = []
    kept_rtts_ms = []

    for index in range(N_CHECKS):
        offset_s, round_trip_s = one_bob_check()
        offset_ms = 1000.0 * offset_s
        rtt_ms = 1000.0 * round_trip_s
        offsets_ms.append(offset_ms)
        rtts_ms.append(rtt_ms)
        keep = index >= WARMUP_CHECKS
        if keep:
            kept_offsets_ms.append(offset_ms)
            kept_rtts_ms.append(rtt_ms)
        marker = "kept" if keep else "warmup"
        print(f"{index + 1:02d}: offset_ms={offset_ms:+.3f}, round_trip_ms={rtt_ms:.3f} [{marker}]")
        time.sleep(PAUSE_SECONDS)

    print()
    print("Summary, warmup discarded:")
    print(f"offset_ms median={statistics.median(kept_offsets_ms):+.3f}")
    print(f"offset_ms min={min(kept_offsets_ms):+.3f}")
    print(f"offset_ms max={max(kept_offsets_ms):+.3f}")
    print(f"offset_ms stdev={statistics.stdev(kept_offsets_ms):.3f}")
    print(f"round_trip_ms median={statistics.median(kept_rtts_ms):.3f}")
    print(f"round_trip_ms min={min(kept_rtts_ms):.3f}")
    print(f"round_trip_ms max={max(kept_rtts_ms):.3f}")
    print(f"round_trip_ms stdev={statistics.stdev(kept_rtts_ms):.3f}")

    if statistics.median(kept_rtts_ms) > HIGH_RTT_WARNING_MS:
        print("WARNING: Median round trip is high; scheduled starts may be coarse.")
    if statistics.stdev(kept_offsets_ms) > HIGH_JITTER_WARNING_MS:
        print("WARNING: Offset estimate is noisy; scheduled starts may be unstable.")


def main() -> None:
    stop_event = start_alice_time_server()

    try:
        check_bob_clock()
        if KEEP_SERVER_ALIVE_AFTER_CHECKS:
            print()
            print("Alice time server is still running for bob_time_check.py.")
            print("Press Ctrl+C here after Bob's check is done.")
            while not stop_event.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping Alice time server")
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
