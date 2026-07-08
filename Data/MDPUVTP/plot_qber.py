# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:03:02 2026

@author: LjubljanaLab
"""


import matplotlib.pyplot as plt
import pandas as pd

CSV_FILE = "qber_live_log.csv"


def main() -> None:
    df = pd.read_csv(CSV_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    elapsed_s = df["timestamp"] - df["timestamp"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        elapsed_s,
        df["QBER_total"],
        marker="o",
        linewidth=1.5,
        markersize=5,
        label="QBER_total",
    )

    ax.axhline(
        0.10,
        linestyle="--",
        linewidth=1.5,
        label="10% QBER",
    )

    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("QBER")
    ax.set_title("Total QBER")
    ax.set_ylim(bottom=0)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()