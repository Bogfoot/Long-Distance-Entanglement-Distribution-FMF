from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CSV_FILE = Path("CHSH_S_Log_2025_11_19_11_27_28.csv")


CORRELATION_COLUMNS = {
    "E_ab": r"$E(a,b)$",
    "E_abp": r"$E(a,b')$",
    "E_apb": r"$E(a',b)$",
    "E_apbp": r"$E(a',b')$",
}

COINCIDENCE_COLUMNS = [
    "C_HH", "C_HV", "C_VH", "C_VV",
    "C_HD", "C_HA", "C_VD", "C_VA",
    "C_DH", "C_DV", "C_AH", "C_AV",
    "C_DD", "C_DA", "C_AD", "C_AA",
]


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    required_columns = [
        "timestamp",
        "S_value",
        "S_err",
        *CORRELATION_COLUMNS.keys(),
        *COINCIDENCE_COLUMNS,
    ]

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.sort_values("timestamp").reset_index(drop=True)

    df["elapsed_s"] = df["timestamp"] - df["timestamp"].iloc[0]

    return df


def plot_s_value(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        df["elapsed_s"],
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
        label=r"Local bound $S=2$",
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


def plot_correlations(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    for column, label in CORRELATION_COLUMNS.items():
        ax.plot(
            df["elapsed_s"],
            df[column],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=label,
        )

    ax.axhline(0.0, linewidth=1)

    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Correlation E")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("CHSH Correlations")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()


def plot_coincidences(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    for column in COINCIDENCE_COLUMNS:
        ax.plot(
            df["elapsed_s"],
            df[column],
            marker="o",
            linewidth=1.2,
            markersize=3,
            label=column.removeprefix("C_"),
        )

    ax.set_xlabel("Elapsed time [s]")
    ax.set_ylabel("Coincidence counts")
    ax.set_title("CHSH Coincidence Counts")
    ax.grid(True)

    ax.legend(
        ncol=4,
        fontsize=9,
    )

    fig.tight_layout()


def main() -> None:
    df = load_data(CSV_FILE)

    print(f"Measurements: {len(df)}")
    print(f"Duration: {df['elapsed_s'].iloc[-1]:.2f} s")
    print(f"Mean S: {df['S_value'].mean():.5f}")
    print(f"Maximum S: {df['S_value'].max():.5f}")
    print(f"Minimum S: {df['S_value'].min():.5f}")

    plot_s_value(df)
    plot_correlations(df)
    plot_coincidences(df)

    plt.show()


if __name__ == "__main__":
    main()