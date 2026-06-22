from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

EPC_MAX_VOLTAGE = 130.0
EPC_STEP_DELAY = 0.001


def load_polarization_device_class() -> type:
    try:
        from EPC.EPC import PolarizationDevice

        return PolarizationDevice
    except ImportError:
        from AEPC.EPC import PolarizationDevice

        return PolarizationDevice


def init_epc(
    name: str,
    device_ref: int | str = 0,
    start_temperature: float | None = 50,
) -> Any:
    polarization_device = load_polarization_device_class()
    epc = polarization_device(device_ref)
    initialize_outputs = getattr(epc, "initialize_outputs", None)
    if initialize_outputs is not None:
        initialize_outputs()
    if start_temperature is not None:
        epc.set_temperature(start_temperature)
    print(f"[{name}] EPC initialized")
    return epc


def validate_voltages(values: Sequence[float]) -> list[float]:
    if len(values) != 4:
        raise ValueError(
            f"Expected four EPC voltages, received {len(values)} values"
        )

    voltages = [float(v) for v in values]
    for voltage in voltages:
        if voltage < 0.0 or voltage > EPC_MAX_VOLTAGE:
            raise ValueError(f"Voltage {voltage} outside 0..{EPC_MAX_VOLTAGE} V")
    return voltages


def set_epc_voltages(
    name: str,
    epc: Any,
    voltages: Sequence[float],
) -> list[float]:
    if epc is None:
        raise RuntimeError(f"{name} EPC is not initialized")

    checked = validate_voltages(voltages)
    for index, voltage in enumerate(checked):
        epc.set_voltage(f"DAC{index}", voltage)
        time.sleep(EPC_STEP_DELAY)
    return checked


def set_epc_temperature(name: str, epc: Any, temperature: float) -> None:
    if epc is None:
        raise RuntimeError(f"{name} EPC is not initialized")

    epc.set_temperature(float(temperature))
    print(f"[{name}] EPC temperature set to {temperature}")
