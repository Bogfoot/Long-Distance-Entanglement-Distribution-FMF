from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_FILE = Path("CHSH_S_Log_2025_11_19_11_27_28.csv")


def main() -> None:
    df = pd.read_csv(CSV_FILE)
    df = df.sort_values("timestamp").reset_index(drop=True)

    elapsed_s = df["timestamp"] - df["timestamp"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        elapsed_s,
        df["S_value"],
        yerr=df["S_err"],
        fmt="o-",
        capsize=4,
        linewidth=1.5,
        markersize=5,
        label=r"Measured $S$",
    )

    ax.axhline(
        2.0,
        linestyle="--",
        linewidth=1.5,
        label=r"Bell limit $S=2$",
    )

    ax.axhline(
        2.0 * np.sqrt(2.0),
        linestyle=":",
        linewidth=1.5,
        label=r"Tsirelson bound $S=2\sqrt{2}$",
    )

    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel(r"CHSH parameter $S$")
    ax.set_title("CHSH S Value")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()