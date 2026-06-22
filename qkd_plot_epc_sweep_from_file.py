from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SWEEP_DIR = BASE_DIR / "Data" / "EPC_Sweeps"
CSV_FILE = SWEEP_DIR / "alice_epc_sweep_20260620_152203.csv"

SHOW_RAW_POINTS = True
SAVE_PLOT = True

PAIRS = {
    "C_HH": ("H-H", "#1f77b4"),
    "C_HV": ("H-V", "#9467bd"),
    "C_VH": ("V-H", "#8c564b"),
    "C_VV": ("V-V", "#ff7f0e"),
    "C_DD": ("D-D", "#2ca02c"),
    "C_DA": ("D-A", "#e377c2"),
    "C_AD": ("A-D", "#7f7f7f"),
    "C_AA": ("A-A", "#d62728"),
}


def main():
    data = pd.read_csv(CSV_FILE)
    pair_columns = list(PAIRS)

    averages = (
        data.groupby(["epc", "dac", "swept_voltage"], as_index=False)[
            pair_columns
        ]
        .mean()
        .sort_values("swept_voltage")
    )

    figure, axes = plt.subplots(
        2,
        4,
        figsize=(16, 8),
        sharex=True,
        sharey=True,
    )

    for row, epc_name in enumerate(("Alice", "Bob")):
        for dac in range(4):
            axis = axes[row, dac]
            raw = data[(data["epc"] == epc_name) & (data["dac"] == dac)]
            mean = averages[
                (averages["epc"] == epc_name)
                & (averages["dac"] == dac)
            ]

            for column, (label, color) in PAIRS.items():
                if SHOW_RAW_POINTS:
                    axis.scatter(
                        raw["swept_voltage"],
                        raw[column],
                        color=color,
                        s=14,
                        alpha=0.3,
                    )

                axis.plot(
                    mean["swept_voltage"],
                    mean[column],
                    color=color,
                    marker="o",
                    markersize=3,
                    linewidth=1.5,
                    label=label,
                )

            axis.set_title(
                f"{epc_name} EPC | Crystal {dac + 1} (DAC{dac})"
            )
            axis.grid(alpha=0.25)

            if row == 1:
                axis.set_xlabel("Applied voltage (V)")
            if dac == 0:
                axis.set_ylabel("Coincidence count")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=8,
        frameon=False,
    )
    figure.suptitle(
        f"Coincidence counts during EPC voltage sweep\n{CSV_FILE.name}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))

    if SAVE_PLOT:
        output_path = CSV_FILE.with_name(
            f"{CSV_FILE.stem}_coincidences.png"
        )
        figure.savefig(output_path, dpi=180)
        print(f"Saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
