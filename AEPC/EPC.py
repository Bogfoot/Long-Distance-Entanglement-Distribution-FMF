import math
import os
from pathlib import Path
import shutil
import subprocess
import time


EPC_COMMAND_BACKEND = "os.system"  # "subprocess" or "os.system"
SHOW_MCP2210_OUTPUT = True


class PolarizationDevice:
    """Explicit-address EPC control — using either serial number or index."""

    def __init__(
        self,
        device_ref,
        command_backend: str | None = None,
        show_cli_output: bool | None = None,
    ):
        if isinstance(device_ref, str) and len(device_ref) > 1:
            self.connection_arg = f"-connectS={device_ref}"  # Serial number
        elif device_ref in [0, 1]:
            self.connection_arg = f"-connectI={device_ref}"  # Index
        else:
            raise ValueError(
                "Invalid device reference. Provide serial number or index 0/1."
            )
        self.command_backend = command_backend or EPC_COMMAND_BACKEND
        self.show_cli_output = (
            SHOW_MCP2210_OUTPUT
            if show_cli_output is None
            else bool(show_cli_output)
        )
        if self.command_backend not in {"subprocess", "os.system"}:
            raise ValueError(
                "command_backend must be 'subprocess' or 'os.system'"
            )
        self.cli_path = self._find_cli()

    @staticmethod
    def _find_cli() -> str:
        local_cli = Path(__file__).resolve().with_name("MCP2210CLI.exe")
        if local_cli.is_file():
            return str(local_cli)

        path_cli = shutil.which("MCP2210CLI.exe")
        if path_cli:
            return path_cli

        raise FileNotFoundError(
            "MCP2210CLI.exe was not found beside EPC.py or on PATH"
        )

    def _spi(self, payload: str):
        command = [
            self.cli_path,
            self.connection_arg,
            f"-spitxfer={payload}",
            "-bd=100000",
            "-cs=gp4",
            "-md=1",
        ]
        if self.command_backend == "os.system":
            command_text = f'"{command[0]}" ' + " ".join(command[1:])
            return_code = os.system(command_text)
            if return_code != 0:
                raise RuntimeError(
                    f"MCP2210CLI.exe failed with exit code {return_code}"
                )
            time.sleep(0.001)
            return

        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=not self.show_cli_output,
                text=not self.show_cli_output,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "MCP2210CLI.exe was not found. Add its folder to PATH or "
                "start Alice/Bob from the folder containing the executable."
            ) from exc
        if result.returncode != 0:
            detail = ""
            if not self.show_cli_output:
                detail = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                f"MCP2210CLI.exe failed with exit code {result.returncode}"
                + (f": {detail}" if detail else "")
            )
        time.sleep(0.001)

    def initialize_outputs(self):
        """Configure the temperature DAC and four polarization DAC outputs."""
        self._spi("28,4f")

    def set_voltage(self, channel: str, V_out: float):
        V_out = float(V_out)
        if channel not in {"DAC0", "DAC1", "DAC2", "DAC3"}:
            raise ValueError("channel must be DAC0, DAC1, DAC2, or DAC3")
        if not 0.0 <= V_out <= 130.0:
            raise ValueError("EPC voltage must be between 0 and 130 V")
        dac_code = round(V_out * 4095 / (5.088 * 25.877))
        dac_code_hex = f"{dac_code:03x}"
        dac_num = f"{8 + int(channel[3]):x}"
        payload = f"{dac_num}{dac_code_hex[0]},{dac_code_hex[1:]}"
        self._spi(payload)

    def sweep_voltages(
        self,
        start_voltage=0.0,
        stop_voltage=130.0,
        step_voltage=1.0,
        sleep_seconds=0.1,
        *,
        simultaneous=False,
        restore_voltages=(0.0, 0.0, 0.0, 0.0),
    ):
        """Sweep DAC0..DAC3 and sleep after every voltage update."""
        start_voltage = float(start_voltage)
        stop_voltage = float(stop_voltage)
        step_voltage = float(step_voltage)
        sleep_seconds = float(sleep_seconds)
        restore_voltages = tuple(float(value) for value in restore_voltages)

        if not 0.0 <= start_voltage <= 130.0:
            raise ValueError("start_voltage must be between 0 and 130 V")
        if not 0.0 <= stop_voltage <= 130.0:
            raise ValueError("stop_voltage must be between 0 and 130 V")
        if step_voltage <= 0.0:
            raise ValueError("step_voltage must be positive")
        if sleep_seconds < 0.0:
            raise ValueError("sleep_seconds cannot be negative")
        if len(restore_voltages) != 4:
            raise ValueError("restore_voltages must contain four values")
        if any(not 0.0 <= value <= 130.0 for value in restore_voltages):
            raise ValueError("restore voltages must be between 0 and 130 V")

        direction = 1.0 if stop_voltage >= start_voltage else -1.0
        signed_step = direction * step_voltage
        voltages = []
        voltage = start_voltage
        while (
            voltage <= stop_voltage + 1.0e-9
            if direction > 0
            else voltage >= stop_voltage - 1.0e-9
        ):
            voltages.append(float(voltage))
            voltage += signed_step

        channels = ("DAC0", "DAC1", "DAC2", "DAC3")
        try:
            if simultaneous:
                for voltage in voltages:
                    print(f"[EPC] DAC0..DAC3 = {voltage:.2f} V")
                    for channel in channels:
                        self.set_voltage(channel, voltage)
                    time.sleep(sleep_seconds)
            else:
                for channel in channels:
                    print(f"[EPC] Sweeping {channel}")
                    for voltage in voltages:
                        print(f"[EPC] {channel} = {voltage:.2f} V")
                        self.set_voltage(channel, voltage)
                        time.sleep(sleep_seconds)
                    self.set_voltage(channel, restore_voltages[int(channel[3])])
        finally:
            for channel, voltage in zip(channels, restore_voltages):
                self.set_voltage(channel, voltage)

    def set_temperature(self, T_th: float):
        R_th0 = 10e3
        Beta = 3950.0
        TempK = 273.0 + float(T_th)
        T0 = 298.0
        V_th = (
            5.088
            * (R_th0 * math.exp(Beta * (1 / TempK - 1 / T0)))
            / (10e3 + R_th0 * math.exp(Beta * (1 / TempK - 1 / T0)))
        )
        V_dac = (V_th - 0.632) / 0.584
        dac_code = round(V_dac * 4095 / 5.088)
        dac_code_hex = f"{dac_code:03x}"
        payload = f"E{dac_code_hex[0]},{dac_code_hex[1:]}"
        self._spi(payload)
