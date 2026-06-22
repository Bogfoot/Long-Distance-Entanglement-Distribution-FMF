from __future__ import annotations

import hashlib
import json
import socket
import struct
from pathlib import Path

SHOW_TRANSFER_PROGRESS = False
CHUNK_SIZE = 1024 * 1024


def recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    data = bytearray()
    while len(data) < nbytes:
        chunk = sock.recv(nbytes - len(data))
        if not chunk:
            raise ConnectionError("Connection closed while receiving data")
        data.extend(chunk)
    return bytes(data)


def send_json(sock: socket.socket, payload: dict) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(encoded)))
    sock.sendall(encoded)


def recv_json(sock: socket.socket) -> dict:
    header_len = struct.unpack("!Q", recv_exact(sock, 8))[0]
    return json.loads(recv_exact(sock, header_len).decode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def send_file(sock: socket.socket, path: Path, metadata: dict | None = None) -> None:
    path = Path(path)
    header = {
        "name": path.name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "metadata": metadata or {},
    }
    send_json(sock, header)

    sent = 0
    size = header["size"]
    with path.open("rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            sock.sendall(chunk)
            sent += len(chunk)
            if SHOW_TRANSFER_PROGRESS:
                print(f"\rSent {sent}/{size} bytes", end="", flush=True)
    if SHOW_TRANSFER_PROGRESS:
        print()


def receive_file(sock: socket.socket, out_dir: Path) -> tuple[Path, dict, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    header = recv_json(sock)

    name = Path(header["name"]).name
    size = int(header["size"])
    expected_sha256 = str(header["sha256"])
    out_path = out_dir / name

    hasher = hashlib.sha256()
    received = 0
    with out_path.open("wb") as f:
        while received < size:
            chunk = sock.recv(min(CHUNK_SIZE, size - received))
            if not chunk:
                raise ConnectionError("Connection closed before file completed")
            f.write(chunk)
            hasher.update(chunk)
            received += len(chunk)
            if SHOW_TRANSFER_PROGRESS:
                print(f"\rReceived {received}/{size} bytes", end="", flush=True)
    if SHOW_TRANSFER_PROGRESS:
        print()

    return out_path, header, hasher.hexdigest() == expected_sha256
