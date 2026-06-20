from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from Alice import (
    ACQUISITION,
    SYNC_PROCESSING,
    apply_correction_voltages,
    initialize_alice_epc,
)
from qkd_acquisition import acquire_pair, delete_acquisition_files
from qkd_epc_correction import (
    RESULT_PAIR_ORDER,
    analyze_phi_plus_coincidences,
)
from qkd_sync import analyze_sync_coincidences

try:
    import QuTAG_MC as qt
except ImportError as exc:
    print("ERROR: Failed to import QuTAG_MC:", exc)
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Data" / "EPC_Sweeps"

MEASUREMENT_SECONDS = 10.0
SETTLE_SECONDS = 1
VOLTAGE_START = 0.0
VOLTAGE_STOP = 130.0
VOLTAGE_STEP = 5.0
# Set to an explicit sequence for repeatability tests, for example:
# (0.0, 130.0) * 5. Leave as None for the regular ascending sweep.
VOLTAGE_SEQUENCE: tuple[float, ...] | None = None
RESTORE_VOLTAGES = [0.0] * 8
DELETE_RAW_FILES = True
LIVE_PLOT = True

# Each target is swept independently while the other seven controls remain
# at zero. Indices 0..3 belong to Alice and 4..7 belong to Bob.
SWEEP_TARGETS = (
    # ("Alice", 0),
    # ("Alice", 1),
    # ("Alice", 2),
    # ("Alice", 3),
    ("Bob", 0),
    ("Bob", 1),
    ("Bob", 2),
    ("Bob", 3),
)
# SWEEP_TARGETS = (
# ("Bob", 3),
#     )

SWEEP_FIELDNAMES = [
    "timestamp",
    "record_id",
    "epc",
    "dac",
    "swept_voltage",
    "voltages",
    "visibility",
    "vis_HV",
    "vis_DA",
    "QBER_total",
    "QBER_HV",
    "QBER_DA",
    "total_coincidences",
    "sync_markers",
    *[f"C_{label}" for label in RESULT_PAIR_ORDER],
    *[f"delay_{label}_ps" for label in RESULT_PAIR_ORDER],
]


def voltage_points():
      if VOLTAGE_SEQUENCE is not None:
          points = [float(v) for v in VOLTAGE_SEQUENCE]
          invalid = [v for v in points if v < 0.0 or v > 130.0]

          if invalid:
              raise ValueError(f"Invalid voltages: {invalid}")

          return points

      points = []
      voltage = float(VOLTAGE_START)

      while voltage <= VOLTAGE_STOP + 1e-9:
          points.append(voltage)
          voltage += VOLTAGE_STEP

      return points


def create_live_plot():
    plt.ion()
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(15, 7),
        sharex=True,
        sharey=True,
    )
    lines = {}
    history = {target: [] for target in SWEEP_TARGETS}

    for row, epc_name in enumerate(("Alice", "Bob")):
        for dac in range(4):
            axis = axes[row, dac]
            target = (epc_name, dac)
            lines[target] = {
                "visibility": axis.plot(
                    [],
                    [],
                    marker="o",
                    markersize=3,
                    label="Mean",
                )[0],
                "HV": axis.plot([], [], label="H/V")[0],
                "DA": axis.plot([], [], label="D/A")[0],
            }
            axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.4)
            axis.set_title(f"{epc_name} DAC{dac}")
            axis.set_xlim(VOLTAGE_START, VOLTAGE_STOP)
            axis.set_ylim(-1.05, 1.05)
            axis.grid(True, alpha=0.25)
            if row == 1:
                axis.set_xlabel("Voltage (V)")
            if dac == 0:
                axis.set_ylabel("Visibility")
            if row == 0 and dac == 3:
                axis.legend(loc="lower right")

    figure.suptitle("EPC voltage sweep visibility")
    figure.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)
    return figure, lines, history


def update_live_plot(figure, lines, history, target, voltage, correction):
    history[target].append(
        (
            voltage,
            correction.visibility,
            correction.basis_visibility["HV"],
            correction.basis_visibility["DA"],
        )
    )
    rows = history[target]
    voltages = [row[0] for row in rows]
    lines[target]["visibility"].set_data(
        voltages,
        [row[1] for row in rows],
    )
    lines[target]["HV"].set_data(voltages, [row[2] for row in rows])
    lines[target]["DA"].set_data(voltages, [row[3] for row in rows])
    figure.canvas.draw_idle()
    figure.canvas.flush_events()
    plt.pause(0.01)


def result_row(
    *,
    record_id: str,
    epc_name: str,
    dac: int,
    swept_voltage: float,
    voltages: list[float],
    correction,
) -> dict[str, object]:
    row: dict[str, object] = {
        "timestamp": time.time(),
        "record_id": record_id,
        "epc": epc_name,
        "dac": dac,
        "swept_voltage": swept_voltage,
        "voltages": json.dumps(voltages),
        "visibility": correction.visibility,
        "vis_HV": correction.basis_visibility["HV"],
        "vis_DA": correction.basis_visibility["DA"],
        "QBER_total": correction.qber_total,
        "QBER_HV": correction.basis_qber["HV"],
        "QBER_DA": correction.basis_qber["DA"],
        "total_coincidences": correction.total_coincidences,
        "sync_markers": int(correction.sync.clock_map.counters.size),
    }
    for label in RESULT_PAIR_ORDER:
        pair = correction.sync.results_by_name[label]
        row[f"C_{label}"] = pair.count
        row[f"delay_{label}_ps"] = pair.best_delay_ps
    return row


def print_result(epc_name: str, dac: int, voltage: float, correction) -> None:
    counts = correction.counts
    delays = correction.delays_ps
    border = "=" * 92
    print(f"\n{border}")
    print(
        f"[Sweep] {epc_name} DAC{dac} = {voltage:.1f} V | "
        f"visibility={100.0 * correction.visibility:.3f}% | "
        f"HV={100.0 * correction.basis_visibility['HV']:.3f}% | "
        f"DA={100.0 * correction.basis_visibility['DA']:.3f}% | "
        f"QBER={100.0 * correction.qber_total:.3f}% | "
        f"total={correction.total_coincidences}"
    )
    print(
        "[Sweep] Coincidences | "
        + "  ".join(f"{label}={counts[label]}" for label in RESULT_PAIR_ORDER)
    )
    print(
        "[Sweep] Delays (ps) | "
        + "  ".join(
            f"{label}={delays[label]:+.0f}" for label in RESULT_PAIR_ORDER
        )
    )
    print(f"{border}\n")


def measure_point(
    tagger,
    alice_epc,
    epc_name: str,
    dac: int,
    voltage: float,
):
    voltages = [0.0] * 8
    voltage_index = dac if epc_name == "Alice" else 4 + dac
    voltages[voltage_index] = voltage
    apply_correction_voltages(alice_epc, voltages)
    time.sleep(SETTLE_SECONDS)

    acquisition = acquire_pair(tagger, ACQUISITION, MEASUREMENT_SECONDS)
    try:
        synchronized = analyze_sync_coincidences(
            acquisition.alice_path,
            acquisition.bob_path,
            SYNC_PROCESSING.coincidence_pairs,
            sync_channel=SYNC_PROCESSING.sync_channel,
            coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
        )
        correction = analyze_phi_plus_coincidences(synchronized)
        return acquisition, voltages, correction
    except Exception:
        if DELETE_RAW_FILES:
            delete_acquisition_files(acquisition, ACQUISITION)
        raise


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"alice_epc_sweep_{timestamp}.csv"
    plot_path = OUTPUT_DIR / f"alice_epc_sweep_{timestamp}.png"
    points = voltage_points()
    total_measurements = len(SWEEP_TARGETS) * len(points)
    estimated_minutes = total_measurements * (
        MEASUREMENT_SECONDS
        + ACQUISITION.schedule_ahead_seconds
        + SETTLE_SECONDS
    ) / 60.0

    print(f"[Sweep] Results: {csv_path}")
    print(
        f"[Sweep] {total_measurements} measurements; acquisition-only "
        f"estimate={estimated_minutes:.1f} min, excluding synchronization"
    )

    figure = lines = history = None
    if LIVE_PLOT:
        figure, lines, history = create_live_plot()

    alice_epc = initialize_alice_epc()
    tagger = qt.QuTAG()

    try:
        apply_correction_voltages(alice_epc, RESTORE_VOLTAGES)
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=SWEEP_FIELDNAMES)
            writer.writeheader()

            measurement_index = 0
            for epc_name, dac in SWEEP_TARGETS:
                for voltage in points:
                    measurement_index += 1
                    print(
                        f"[Sweep] Point {measurement_index}/{total_measurements}: "
                        f"{epc_name} DAC{dac}={voltage:.1f} V"
                    )
                    acquisition, voltages, correction = measure_point(
                        tagger,
                        alice_epc,
                        epc_name,
                        dac,
                        voltage,
                    )
                    try:
                        row = result_row(
                            record_id=acquisition.record_id,
                            epc_name=epc_name,
                            dac=dac,
                            swept_voltage=voltage,
                            voltages=voltages,
                            correction=correction,
                        )
                        writer.writerow(row)
                        handle.flush()
                        print_result(epc_name, dac, voltage, correction)
                        if LIVE_PLOT:
                            update_live_plot(
                                figure,
                                lines,
                                history,
                                (epc_name, dac),
                                voltage,
                                correction,
                            )
                    finally:
                        if DELETE_RAW_FILES:
                            delete_acquisition_files(acquisition, ACQUISITION)
    except KeyboardInterrupt:
        print("\n[Sweep] Interrupted by user")
    finally:
        try:
            apply_correction_voltages(alice_epc, RESTORE_VOLTAGES)
            print(f"[Sweep] Restored voltages to {RESTORE_VOLTAGES}")
        except Exception as exc:
            print(f"[Sweep] WARNING: Could not restore EPC voltages: {exc}")
        try:
            tagger.writeTimestamps("", tagger.FILEFORMAT_NONE)
            tagger.deInitialize()
        except Exception as exc:
            print(f"[Sweep] WARNING: Could not shut down Alice tagger: {exc}")

        if LIVE_PLOT and figure is not None:
            figure.savefig(plot_path, dpi=170)
            print(f"[Sweep] Saved plot: {plot_path}")
            plt.ioff()


if __name__ == "__main__":
    main()
