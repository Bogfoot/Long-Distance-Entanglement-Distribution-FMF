from __future__ import annotations

import socket
from pathlib import Path

from qkd_network import recv_json, send_file


RECEIVER_HOST = "100.119.115.112" # Adrian home
# RECEIVER_HOST = "100.96.195.118" # SLM PC
# RECEIVER_HOST = "100.104.228.90" # DRNOVO
# RECEIVER_HOST = "100.112.116.8" # NOVO MESTO
RECEIVER_PORT = 5003
ZIP_PATH = Path(__file__).resolve().parent / "DataExtSync.zip"
# ZIP_PATH = Path(__file__).resolve().parent / "Data" / "Incoming" / "bob_20260522T080056.848277Z_exp_10.0s.bin"
# ZIP_PATH = Path("C:\\Users\\LjubljanaLab\\Downloads\\diagnostic-bundle-20260601103540.zip")
# ZIP_PATH = Path("C:\\Users\\LjubljanaLab\\Desktop\\TempScans\\ScanCode\\PythonLibrary.zip")

def main() -> None:
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"File not found: {ZIP_PATH}")

    with socket.create_connection((RECEIVER_HOST, RECEIVER_PORT), timeout=30) as sock:
        print(f"[Alice] Sending {ZIP_PATH} to {RECEIVER_HOST}:{RECEIVER_PORT}")
        send_file(
            sock,
            ZIP_PATH,
            metadata={
                "source": "alice",
                "description": "LongDistanceQKD Data.zip",
            },
        )
        reply = recv_json(sock)
        if not reply.get("ok"):
            raise RuntimeError(f"Receiver error: {reply.get('error')}")
        print(f"[Alice] Receiver saved file: {reply.get('path')}")


if __name__ == "__main__":
    main()
