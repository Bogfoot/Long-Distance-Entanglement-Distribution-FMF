from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from Alice import ACQUISITION, SYNC_PROCESSING
from qkd_sync import analyze_sync_coincidence_exposures


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


def plot_results(data: pd.DataFrame, output_path: Path) -> None:
    x = data["start_seconds"]
    figure, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    for name in PAIR_ORDER:
        axes[0].plot(x, data[f"cps_{name}"], marker=".", label=name)
    axes[0].set_ylabel("Coincidences/s")
    axes[0].legend(ncol=8)

    alice_channels = sorted(pair[1] for pair in SYNC_PROCESSING.coincidence_pairs)
    bob_channels = sorted(pair[2] for pair in SYNC_PROCESSING.coincidence_pairs)
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


def exposure_row(analysis) -> dict[str, object]:
    results = analysis.results_by_name
    counts = {name: results[name].count for name in PAIR_ORDER}
    duration = analysis.overlap_duration_s
    hv_correlated = counts["HH"] + counts["VV"]
    hv_errors = counts["HV"] + counts["VH"]
    da_correlated = counts["DD"] + counts["AA"]
    da_errors = counts["DA"] + counts["AD"]
    vis_hv = (
        (hv_correlated - hv_errors) / (hv_correlated + hv_errors)
        if hv_correlated + hv_errors
        else 0.0
    )
    vis_da = (
        (da_correlated - da_errors) / (da_correlated + da_errors)
        if da_correlated + da_errors
        else 0.0
    )
    row: dict[str, object] = {
        "exposure_index": analysis.exposure_index,
        "start_seconds": analysis.exposure_start_s,
        "exposure_seconds": duration,
        "visibility": (vis_hv + vis_da) / 2.0,
        "vis_HV": vis_hv,
        "vis_DA": vis_da,
        "QBER_total": (1.0 - (vis_hv + vis_da) / 2.0) / 2.0,
    }
    for name in PAIR_ORDER:
        result = results[name]
        row[f"C_{name}"] = result.count
        row[f"cps_{name}"] = result.count / duration
        row[f"delay_{name}_ps"] = result.best_delay_ps
    for result in analysis.pair_results:
        alice_channel = result.pair.alice_channel
        bob_channel = result.pair.bob_channel
        row[f"alice_ch{alice_channel}_cps"] = result.alice_event_count / duration
        row[f"bob_ch{bob_channel}_cps"] = result.bob_event_count / duration
    return row


def main() -> None:
    record_id, alice_path, bob_path = select_recording()
    print(f"Analyzing record_id={record_id}")
    analyses = analyze_sync_coincidence_exposures(
        alice_path,
        bob_path,
        SYNC_PROCESSING.coincidence_pairs,
        SYNC_PROCESSING.analysis_exposure_seconds,
        sync_channel=SYNC_PROCESSING.sync_channel,
        coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
        delay_reference_pairs=SYNC_PROCESSING.delay_reference_pairs,
        include_partial_last_exposure=INCLUDE_PARTIAL_LAST_EXPOSURE,
    )
    data = pd.DataFrame(exposure_row(analysis) for analysis in analyses)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = OUTPUT_DIR / f"recorded_exposures_{record_id}.csv"
    output_plot = OUTPUT_DIR / f"recorded_exposures_{record_id}.png"
    data.to_csv(output_csv, index=False)
    print(
        f"Processed {len(data)} exposures of "
        f"{SYNC_PROCESSING.analysis_exposure_seconds:g} s"
    )
    print(f"Saved results: {output_csv}")
    plot_results(data, output_plot)


if __name__ == "__main__":
    main()
