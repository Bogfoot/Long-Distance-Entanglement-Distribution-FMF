# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:08:53 2026

@author: Adrian
"""
from __future__ import annotations

import hashlib
import json
import socket
import struct
from pathlib import Path

HOST = "0.0.0.0"
PORT = 5001
OUT_DIR = Path(r"C:\Users\LjubljanaLab\Desktop\LongDistanceQKD")
CHUNK = 1024 * 1024

OUT_DIR.mkdir(parents=True, exist_ok=True)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def main() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind((HOST, PORT))
        server.listen(1)
        print(f"Listening on {HOST}:{PORT}")

        conn, addr = server.accept()
        with conn:
            print(f"Connected by {addr}")

            header_len = struct.unpack("!Q", recv_exact(conn, 8))[0]
            header = json.loads(recv_exact(conn, header_len).decode("utf-8"))

            name = Path(header["name"]).name
            size = int(header["size"])
            expected_sha256 = header["sha256"]

            out_path = OUT_DIR / name
            hasher = hashlib.sha256()
            received = 0

            with out_path.open("wb") as f:
                while received < size:
                    need = min(CHUNK, size - received)
                    chunk = conn.recv(need)
                    if not chunk:
                        raise ConnectionError("Connection closed before file completed")
                    f.write(chunk)
                    hasher.update(chunk)
                    received += len(chunk)
                    print(f"\rReceived {received}/{size} bytes", end="", flush=True)

            actual_sha256 = hasher.hexdigest()
            ok = actual_sha256 == expected_sha256
            print()
            print(f"Saved to: {out_path}")
            print(f"SHA256 ok: {ok}")

            conn.sendall(json.dumps({"ok": ok, "path": str(out_path)}).encode("utf-8"))


if __name__ == "__main__":
    main()