from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from qkd_sync import PS_PER_NS, SyncCoincidenceAnalysis


def save_delay_scan_plot(
    analysis: SyncCoincidenceAnalysis,
    output_path: str | Path,
) -> Path:
    results = analysis.pair_results
    columns = 2
    rows = math.ceil(len(results) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(13, 3.2 * rows),
        squeeze=False,
    )

    for axis, result in zip(axes.flat, results):
        scan = result.delay_scan
        if scan is None:
            axis.text(
                0.5,
                0.5,
                "No delay scan data",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            axis.plot(
                scan.delays_ps / PS_PER_NS,
                scan.counts,
                linewidth=1.2,
            )
            scan_peak_ps = float(
                scan.delays_ps[int(np.argmax(scan.counts))]
            )
            if not np.isclose(scan_peak_ps, result.best_delay_ps):
                axis.axvline(
                    scan_peak_ps / PS_PER_NS,
                    color="gray",
                    linestyle=":",
                    linewidth=1,
                    label=f"scan peak={scan_peak_ps / PS_PER_NS:.3f} ns",
                )
            axis.axvline(
                result.best_delay_ps / PS_PER_NS,
                color="red",
                linestyle="--",
                linewidth=1,
                label=f"used={result.best_delay_ps / PS_PER_NS:.3f} ns",
            )
        axis.set_title(
            f"{result.pair.name}: Alice ch{result.pair.alice_channel}, "
            f"Bob ch{result.pair.bob_channel}"
        )
        axis.set_xlabel("Delay applied to Bob (ns)")
        axis.set_ylabel("Coincidences")
        axis.grid(True, alpha=0.25)
        if scan is not None:
            axis.legend(fontsize=8)

    for axis in axes.flat[len(results):]:
        axis.set_visible(False)

    figure.suptitle(
        "Initial fine delay scans "
        f"({analysis.coincidence_window_ps:g} ps coincidence window)",
        fontsize=14,
    )
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=170)
    plt.close(figure)
    return path
