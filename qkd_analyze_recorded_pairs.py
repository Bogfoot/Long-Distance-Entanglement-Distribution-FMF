from __future__ import annotations

import re
from pathlib import Path

import coincfinder
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Alice import ACQUISITION, SYNC_PROCESSING
from qkd_sync import (
    PS_PER_SECOND,
    align_bob_to_alice,
    build_clock_map,
    collect_coincidences,
    decode_file,
    find_best_delay,
    flatten_channel,
    normalize_pairs,
    trim_to_range,
)


# File selection and plotting are local. Exposure and synchronization settings
# come from Alice.py through ACQUISITION and SYNC_PROCESSING.
EXPOSURE_SECONDS = SYNC_PROCESSING.analysis_exposure_seconds
RECORD_ID: str | None = None  # None selects the latest complete file pair.
INCLUDE_PARTIAL_LAST_EXPOSURE = False
SHOW_PLOT = True
SAVE_PLOT = True

OUTPUT_DIR = Path(__file__).resolve().parent / "Data" / "RecordedExposureAnalysis"
PAIR_ORDER = ("HH", "VV", "DD", "AA", "HV", "VH", "DA", "AD")

ALICE_PATTERN = re.compile(r"^alice_(.+?)_exp_.*\.bin$")
BOB_PATTERN = re.compile(r"^bob_(.+?)_exp_.*\.bin$")


def select_recording() -> tuple[str, Path, Path]:
    alice_files = {
        match.group(1): path
        for path in ACQUISITION.alice_record_dir.glob("alice_*.bin")
        if (match := ALICE_PATTERN.match(path.name))
    }
    bob_files = {
        match.group(1): path
        for path in ACQUISITION.incoming_dir.glob("bob_*.bin")
        if (match := BOB_PATTERN.match(path.name))
    }
    complete = sorted(alice_files.keys() & bob_files.keys())
    if not complete:
        raise FileNotFoundError("No matching Alice/Bob recording found")

    record_id = RECORD_ID or complete[-1]
    return record_id, alice_files[record_id], bob_files[record_id]


def slice_exposure(
    timestamps: np.ndarray,
    start_ps: float,
    end_ps: float,
) -> np.ndarray:
    first = np.searchsorted(timestamps, start_ps, side="left")
    last = np.searchsorted(timestamps, end_ps, side="left")
    return timestamps[first:last]


def visibility(correlated: int, errors: int) -> float:
    total = correlated + errors
    return (correlated - errors) / total if total else 0.0


def analyze_exposure(
    exposure_index: int,
    relative_start_seconds: float,
    duration_seconds: float,
    alice_channels: dict[int, np.ndarray],
    bob_channels: dict[int, np.ndarray],
    pairs,
) -> dict[str, object]:
    pairs_by_name = {pair.name: pair for pair in pairs}
    delay_references = SYNC_PROCESSING.delay_reference_pairs

    # Every exposure gets new delay scans. Alice.py decides whether pairs use
    # their own scan or share a reference-pair scan.
    reference_names = list(
        dict.fromkeys(
            delay_references[pair.name]
            if delay_references is not None
            else pair.name
            for pair in pairs
        )
    )
    scanned_delays = {}
    for name in reference_names:
        pair = pairs_by_name[name]
        scanned_delays[name], _ = find_best_delay(
            alice_channels[pair.alice_channel],
            bob_channels[pair.bob_channel],
            coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
        )

    counts = {}
    delays = {}
    for pair in pairs:
        reference_name = (
            delay_references[pair.name]
            if delay_references is not None
            else pair.name
        )
        delay_ps = scanned_delays[reference_name]
        delays[pair.name] = delay_ps
        counts[pair.name] = int(
            collect_coincidences(
                alice_channels[pair.alice_channel],
                bob_channels[pair.bob_channel],
                delay_ps,
                SYNC_PROCESSING.coincidence_window_ps,
            ).shape[0]
        )

    vis_hv = visibility(
        counts["HH"] + counts["VV"],
        counts["HV"] + counts["VH"],
    )
    vis_da = visibility(
        counts["DD"] + counts["AA"],
        counts["DA"] + counts["AD"],
    )

    row: dict[str, object] = {
        "exposure_index": exposure_index,
        "start_seconds": relative_start_seconds,
        "exposure_seconds": duration_seconds,
        "visibility": (vis_hv + vis_da) / 2.0,
        "vis_HV": vis_hv,
        "vis_DA": vis_da,
        "QBER_total": (1.0 - (vis_hv + vis_da) / 2.0) / 2.0,
    }

    for name in PAIR_ORDER:
        row[f"C_{name}"] = counts[name]
        row[f"cps_{name}"] = counts[name] / duration_seconds
        row[f"delay_{name}_ps"] = delays[name]

    for channel, timestamps in sorted(alice_channels.items()):
        row[f"alice_ch{channel}_singles"] = timestamps.size
        row[f"alice_ch{channel}_cps"] = timestamps.size / duration_seconds

    for channel, timestamps in sorted(bob_channels.items()):
        row[f"bob_ch{channel}_singles"] = timestamps.size
        row[f"bob_ch{channel}_cps"] = timestamps.size / duration_seconds

    print(
        f"Exposure {exposure_index}: t={relative_start_seconds:.1f} s, "
        f"visibility={100 * row['visibility']:.2f}%, "
        f"HV={100 * vis_hv:.2f}%, DA={100 * vis_da:.2f}%"
    )
    print(
        "  Coincidences: "
        + "  ".join(
            f"{name}={counts[name]} ({counts[name] / duration_seconds:.2f}/s)"
            for name in PAIR_ORDER
        )
    )
    print(
        "  Delays: "
        + "  ".join(
            f"{name}={delays[name] / 1000.0:+.3f} ns"
            for name in PAIR_ORDER
        )
    )
    print(
        "  Alice singles: "
        + "  ".join(
            f"ch{channel}={timestamps.size} "
            f"({timestamps.size / duration_seconds:.1f}/s)"
            for channel, timestamps in sorted(alice_channels.items())
        )
    )
    print(
        "  Bob singles: "
        + "  ".join(
            f"ch{channel}={timestamps.size} "
            f"({timestamps.size / duration_seconds:.1f}/s)"
            for channel, timestamps in sorted(bob_channels.items())
        )
    )
    return row


def plot_results(data: pd.DataFrame, output_path: Path) -> None:
    x = data["start_seconds"]
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for name in PAIR_ORDER:
        axes[0].plot(x, data[f"cps_{name}"], marker=".", label=name)
    axes[0].set_ylabel("Coincidences/s")
    axes[0].legend(ncol=8)

    alice_channels = sorted(
        pair[1] for pair in SYNC_PROCESSING.coincidence_pairs
    )
    bob_channels = sorted(
        pair[2] for pair in SYNC_PROCESSING.coincidence_pairs
    )
    for channel in sorted(set(alice_channels)):
        axes[1].plot(
            x,
            data[f"alice_ch{channel}_cps"],
            label=f"Alice ch{channel}",
        )
    for channel in sorted(set(bob_channels)):
        axes[1].plot(
            x,
            data[f"bob_ch{channel}_cps"],
            linestyle="--",
            label=f"Bob ch{channel}",
        )
    axes[1].set_ylabel("Singles/s")
    axes[1].legend(ncol=4)

    for name in PAIR_ORDER:
        axes[2].plot(
            x,
            data[f"delay_{name}_ps"] / 1000.0,
            marker=".",
            label=name,
        )
    axes[2].set_ylabel("Delay (ns)")
    axes[2].legend(ncol=8)

    axes[3].plot(x, data["visibility"], label="Total")
    axes[3].plot(x, data["vis_HV"], label="H/V")
    axes[3].plot(x, data["vis_DA"], label="D/A")
    axes[3].set_ylabel("Visibility")
    axes[3].set_xlabel("Time from synchronized overlap start (s)")
    axes[3].set_ylim(-1.05, 1.05)
    axes[3].legend()

    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()

    if SAVE_PLOT:
        figure.savefig(output_path, dpi=170)
        print(f"Saved plot: {output_path}")
    if SHOW_PLOT:
        plt.show()


def main() -> None:
    record_id, alice_path, bob_path = select_recording()
    pairs = normalize_pairs(SYNC_PROCESSING.coincidence_pairs)

    print(f"Record ID: {record_id}")
    print(f"Alice file: {alice_path}")
    print(f"Bob file:   {bob_path}")
    print("Building one clock map for the complete recording...")

    alice_decode = decode_file(alice_path, SYNC_PROCESSING.sync_channel)
    bob_decode = decode_file(bob_path, SYNC_PROCESSING.sync_channel)
    clock_map = build_clock_map(alice_decode, bob_decode)

    alice_file_channels, _ = coincfinder.read_file_auto(str(alice_path))
    bob_file_channels, _ = coincfinder.read_file_auto(str(bob_path))
    full_alice_channels = {
        channel: trim_to_range(
            flatten_channel(alice_file_channels, channel),
            clock_map.alice_times_ps[0],
            clock_map.alice_times_ps[-1],
        )
        for channel in {pair.alice_channel for pair in pairs}
    }
    full_bob_channels = {
        channel: align_bob_to_alice(
            flatten_channel(bob_file_channels, channel),
            clock_map,
        )
        for channel in {pair.bob_channel for pair in pairs}
    }

    overlap_start = clock_map.alice_times_ps[0]
    overlap_end = clock_map.alice_times_ps[-1]
    exposure_ps = EXPOSURE_SECONDS * PS_PER_SECOND
    rows = []
    exposure_index = 0
    exposure_start = overlap_start

    while exposure_start < overlap_end:
        exposure_end = min(exposure_start + exposure_ps, overlap_end)
        duration_seconds = (exposure_end - exposure_start) / PS_PER_SECOND
        if (
            duration_seconds < EXPOSURE_SECONDS
            and not INCLUDE_PARTIAL_LAST_EXPOSURE
        ):
            break

        exposure_index += 1
        alice_channels = {
            channel: slice_exposure(values, exposure_start, exposure_end)
            for channel, values in full_alice_channels.items()
        }
        bob_channels = {
            channel: slice_exposure(values, exposure_start, exposure_end)
            for channel, values in full_bob_channels.items()
        }
        rows.append(
            analyze_exposure(
                exposure_index,
                (exposure_start - overlap_start) / PS_PER_SECOND,
                duration_seconds,
                alice_channels,
                bob_channels,
                pairs,
            )
        )
        exposure_start = exposure_end

    data = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / f"recorded_exposures_{record_id}.csv"
    output_plot = OUTPUT_DIR / f"recorded_exposures_{record_id}.png"
    data.to_csv(output_csv, index=False)
    print(f"Saved results: {output_csv}")
    plot_results(data, output_plot)


if __name__ == "__main__":
    main()
