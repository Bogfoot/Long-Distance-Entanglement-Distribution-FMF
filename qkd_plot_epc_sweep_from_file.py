from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SWEEP_DIR = BASE_DIR / "Data" / "EPC_Sweeps"
CSV_FILE = SWEEP_DIR / "alice_epc_sweep_20260620_152203.csv"

SAVE_PLOT = True
SHOW_RAW_POINTS = True
PAIR_LABELS = ("HH", "HV", "VH", "VV", "DD", "DA", "AD", "AA")
PAIR_COLORS = {
    "HH": "#1f77b4",
    "HV": "#9467bd",
    "VH": "#8c564b",
    "VV": "#ff7f0e",
    "DD": "#2ca02c",
    "DA": "#e377c2",
    "AD": "#7f7f7f",
    "AA": "#d62728",
}
COUNT_COLUMNS = [f"C_{label}" for label in PAIR_LABELS]


def read_sweep(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"EPC sweep CSV does not exist: {path}")

    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    required = {"epc", "dac", "swept_voltage", *COUNT_COLUMNS}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(
            f"{path} is missing columns: {', '.join(missing)}"
        )

    data["epc"] = data["epc"].astype("string").str.strip().str.title()
    for column in ("dac", "swept_voltage", *COUNT_COLUMNS):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    valid = (
        data["epc"].isin(("Alice", "Bob"))
        & data["dac"].isin((0, 1, 2, 3))
        & data["swept_voltage"].notna()
        & data[COUNT_COLUMNS].notna().any(axis=1)
    )
    skipped = int((~valid).sum())
    data = data.loc[valid].copy()
    data["dac"] = data["dac"].astype(int)

    if skipped:
        print(f"Skipped {skipped} invalid or incomplete row(s)")
    if data.empty:
        raise ValueError(f"{path} contains no usable sweep measurements")

    print(
        f"Loaded {len(data)} measurements across "
        f"{data[['epc', 'dac']].drop_duplicates().shape[0]} EPC crystals"
    )
    return data


def plot_sweep(data: pd.DataFrame, source_path: Path):
    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
        sharex=True,
        sharey=True,
    )
    figure.canvas.manager.set_window_title("EPC sweep coincidences")

    for row, epc_name in enumerate(("Alice", "Bob")):
        for dac in range(4):
            axis = axes[row, dac]
            selection = data.loc[
                (data["epc"] == epc_name) & (data["dac"] == dac)
            ].sort_values("swept_voltage")

            if selection.empty:
                axis.text(
                    0.5,
                    0.5,
                    "No measurements",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                )
            else:
                means = (
                    selection.groupby("swept_voltage", as_index=False)[
                        COUNT_COLUMNS
                    ]
                    .mean()
                    .sort_values("swept_voltage")
                )

                for label in PAIR_LABELS:
                    count_column = f"C_{label}"
                    color = PAIR_COLORS[label]
                    valid_raw = selection[count_column].notna()

                    if SHOW_RAW_POINTS:
                        axis.scatter(
                            selection.loc[valid_raw, "swept_voltage"],
                            selection.loc[valid_raw, count_column],
                            s=14,
                            color=color,
                            alpha=0.3,
                            edgecolors="none",
                        )

                    valid_mean = means[count_column].notna()
                    axis.plot(
                        means.loc[valid_mean, "swept_voltage"],
                        means.loc[valid_mean, count_column],
                        color=color,
                        linewidth=1.5,
                        marker="o",
                        markersize=3,
                        label=label,
                    )

            axis.set_title(f"{epc_name} crystal {dac} (DAC{dac})")
            axis.grid(True, alpha=0.25)
            if row == 1:
                axis.set_xlabel("Voltage (V)")
            if dac == 0:
                axis.set_ylabel("Coincidences")

    handles = []
    legend_labels = []
    for axis in axes.flat:
        handles, legend_labels = axis.get_legend_handles_labels()
        if handles:
            break
    if handles:
        figure.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=len(PAIR_LABELS),
            frameon=False,
        )

    figure.suptitle(
        f"EPC sweep coincidence counts: {source_path.name}",
        y=0.99,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return figure


def main() -> None:
    csv_path = CSV_FILE.expanduser().resolve()
    print(f"Reading EPC sweep: {csv_path}")
    data = read_sweep(csv_path)
    figure = plot_sweep(data, csv_path)

    if SAVE_PLOT:
        output_path = csv_path.with_name(
            f"{csv_path.stem}_coincidences.png"
        )
        figure.savefig(output_path, dpi=180)
        print(f"Saved coincidence plot: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
