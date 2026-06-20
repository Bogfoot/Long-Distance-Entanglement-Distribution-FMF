from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from qkd_epc import init_epc, set_epc_voltages


# Run this script separately on each computer with its local EPC serial.
EPC_DEVICE_SERIAL = "0001005125"
EPC_TEMPERATURE_C = 50

# Use clearly different patterns and allow enough time to inspect the optical
# counts after each change. The final pattern is restored before exit.
VOLTAGE_PATTERN_A = [20.0, 20.0, 20.0, 20.0]
VOLTAGE_PATTERN_B = [100.0, 20.0, 20.0, 20.0]
HOLD_SECONDS = 10.0
RESTORE_VOLTAGES = VOLTAGE_PATTERN_A


def find_cli() -> Path:
    candidates = [
        Path(__file__).resolve().parent / "EPC" / "MCP2210CLI.exe",
        Path(__file__).resolve().parent / "AEPC" / "MCP2210CLI.exe",
        Path(__file__).resolve().parent / "MCP2210CLI.exe",
    ]
    path_entry = shutil.which("MCP2210CLI.exe")
    if path_entry:
        candidates.append(Path(path_entry))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "MCP2210CLI.exe was not found in EPC/, AEPC/, beside this script, "
        "or on PATH"
    )


def verify_device_visible(cli_path: Path) -> None:
    result = subprocess.run(
        [str(cli_path), "-devices"],
        check=False,
        capture_output=True,
        text=True,
    )
    output = "\n".join(
        part.strip() for part in (result.stdout, result.stderr) if part.strip()
    )
    print(output or "[EPC test] MCP2210CLI.exe returned no device-list output")
    if result.returncode != 0:
        raise RuntimeError(
            f"MCP2210 device enumeration failed with exit code {result.returncode}"
        )
    if EPC_DEVICE_SERIAL not in output:
        raise RuntimeError(
            f"EPC serial {EPC_DEVICE_SERIAL} was not present in the device list"
        )
    print(f"[EPC test] Found expected MCP2210 serial {EPC_DEVICE_SERIAL}")


def main() -> None:
    cli_path = find_cli()
    verify_device_visible(cli_path)
    os.environ["PATH"] = (
        str(cli_path.parent)
        + os.pathsep
        + os.environ.get("PATH", "")
    )

    epc = init_epc(
        "EPC test",
        EPC_DEVICE_SERIAL,
        EPC_TEMPERATURE_C,
    )
    try:
        print(f"[EPC test] Applying pattern A: {VOLTAGE_PATTERN_A}")
        set_epc_voltages("EPC test", epc, VOLTAGE_PATTERN_A)
        time.sleep(HOLD_SECONDS)

        print(f"[EPC test] Applying pattern B: {VOLTAGE_PATTERN_B}")
        set_epc_voltages("EPC test", epc, VOLTAGE_PATTERN_B)
        time.sleep(HOLD_SECONDS)
    finally:
        print(f"[EPC test] Restoring voltages: {RESTORE_VOLTAGES}")
        set_epc_voltages("EPC test", epc, RESTORE_VOLTAGES)

    print("[EPC test] CLI and SPI commands completed without an error")


if __name__ == "__main__":
    main()
