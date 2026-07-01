from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path(__file__).resolve().parent
CSV_FILE: Path | None = BASE_DIR / "Data" / "alice_results.csv"
HISTORY_ROWS = 200  # Set to 0 for all CHSH rows.
USE_RELATIVE_TIME = True
SAVE_PATH: Path | None = None

CHSH_PAIR_LABELS = (
    "HH", "HV", "VH", "VV",
    "HA", "HD", "VA", "VD",
    "DH", "DV", "AH", "AV",
    "DD", "DA", "AD", "AA",
)
EXPECTATION_LABELS = ("E_ab", "E_abp", "E_apb", "E_apbp")

# Current optimizer convention:
# S = E_ab - E_abp + E_apb + E_apbp.
CURRENT_TERM_SIGNS = (1.0, -1.0, 1.0, 1.0)

# Common alternate convention, useful for cross-checks only.
ALT_TERM_SIGNS = (1.0, 1.0, 1.0, -1.0)

# Local analyzer-output swaps. A real detector/output label swap flips every
# expectation involving that local setting, not just one expectation.
OUTPUT_SWAP_FLIPS = {
    "A_HV": ("E_ab", "E_abp"),
    "A_DA": ("E_apb", "E_apbp"),
    "B_HV": ("E_ab", "E_apb"),
    "B_DA": ("E_abp", "E_apbp"),
}


@dataclass(frozen=True)
class Convention:
    name: str
    term_signs: tuple[float, float, float, float] = CURRENT_TERM_SIGNS
    output_swaps: tuple[str, ...] = ()
    extra_flips: tuple[str, ...] = ()


CONVENTIONS = (
    Convention("current labels"),
    Convention("swap Alice H/V", output_swaps=("A_HV",)),
    Convention("swap Alice D/A", output_swaps=("A_DA",)),
    Convention("swap Bob H/V", output_swaps=("B_HV",)),
    Convention("swap Bob D/A", output_swaps=("B_DA",)),
    Convention("swap both D/A", output_swaps=("A_DA", "B_DA")),
    Convention("alternate CHSH signs", term_signs=ALT_TERM_SIGNS),
    Convention(
        "flip E_apbp only (diagnostic)",
        extra_flips=("E_apbp",),
    ),
)

COLORS = (
    "#000000", "#0072b2", "#d55e00", "#009e73",
    "#cc79a7", "#56b4e9", "#e69f00", "#7f7f7f",
)


def main() -> None:
    csv_path = (CSV_FILE or BASE_DIR / "Data" / "alice_results.csv").resolve()
    rows = read_chsh_rows(csv_path, HISTORY_ROWS)
    if not rows:
        raise ValueError(f"{csv_path} has no rows with all CHSH coincidence counts")

    x, x_label, latest_timestamp = x_values(rows)
    raw_expectations = calculate_raw_expectations(rows)
    convention_results = [evaluate_convention(raw_expectations, convention)
                          for convention in CONVENTIONS]

    print_summary(rows, convention_results)
    plot_conventions(
        csv_path,
        x,
        x_label,
        latest_timestamp,
        convention_results,
    )


def read_chsh_rows(path: Path, history: int) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    chsh_rows: list[dict[str, str]] = []
    skipped = 0
    for row in rows:
        parsed = parse_chsh_row(row)
        if parsed is None:
            skipped += 1
            continue
        chsh_rows.append(parsed)

    if history > 0:
        chsh_rows = chsh_rows[-history:]
    if skipped:
        print(f"Skipped {skipped} rows without complete CHSH count data")
    return chsh_rows


def parse_chsh_row(row: dict[str, str]) -> dict[str, str] | None:
    parsed = dict(row)
    for label in CHSH_PAIR_LABELS:
        name = f"C_{label}"
        value = row.get(name, "")
        try:
            parsed[name] = str(float(value))
        except (TypeError, ValueError):
            return None
    return parsed


def has_float(row: dict[str, str], name: str) -> bool:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value))


def float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values: list[float] = []
    for index, row in enumerate(rows, start=1):
        value = row.get(name, "")
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Internal error: filtered CHSH row {index} has invalid "
                f"{name}={value!r}"
            ) from exc
    return np.asarray(values, dtype=np.float64)


def x_values(
    rows: list[dict[str, str]],
) -> tuple[np.ndarray, str, float | None]:
    if not USE_RELATIVE_TIME or not all(has_float(row, "timestamp") for row in rows):
        return (
            np.arange(1, len(rows) + 1, dtype=np.float64),
            "CHSH measurement index",
            None,
        )

    timestamps = float_column(rows, "timestamp")
    latest_timestamp = float(timestamps[-1])
    return (
        timestamps - latest_timestamp,
        "Time before latest CHSH measurement",
        latest_timestamp,
    )


def calculate_raw_expectations(
    rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    counts = {label: float_column(rows, f"C_{label}") for label in CHSH_PAIR_LABELS}
    return {
        "E_ab": correlation(counts["HH"], counts["HV"], counts["VH"], counts["VV"]),
        "E_abp": correlation(counts["HA"], counts["HD"], counts["VA"], counts["VD"]),
        "E_apb": correlation(counts["DH"], counts["DV"], counts["AH"], counts["AV"]),
        "E_apbp": correlation(counts["DD"], counts["DA"], counts["AD"], counts["AA"]),
    }


def correlation(
    pp: np.ndarray,
    pm: np.ndarray,
    mp: np.ndarray,
    mm: np.ndarray,
) -> np.ndarray:
    total = pp + pm + mp + mm
    return np.divide(
        pp + mm - pm - mp,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )


def evaluate_convention(
    raw_expectations: dict[str, np.ndarray],
    convention: Convention,
) -> dict[str, object]:
    signs = {label: 1.0 for label in EXPECTATION_LABELS}
    for swap in convention.output_swaps:
        for label in OUTPUT_SWAP_FLIPS[swap]:
            signs[label] *= -1.0
    for label in convention.extra_flips:
        signs[label] *= -1.0

    expectations = {
        label: signs[label] * raw_expectations[label]
        for label in EXPECTATION_LABELS
    }
    signed_s = sum(
        sign * expectations[label]
        for sign, label in zip(convention.term_signs, EXPECTATION_LABELS)
    )
    return {
        "convention": convention,
        "expectations": expectations,
        "signed_s": signed_s,
        "s_value": np.abs(signed_s),
    }


def print_summary(
    rows: list[dict[str, str]],
    convention_results: list[dict[str, object]],
) -> None:
    print(f"Loaded {len(rows)} CHSH rows")
    print("\nRanked by mean |S|:")
    ranked = sorted(
        convention_results,
        key=lambda item: float(np.nanmean(item["s_value"])),
        reverse=True,
    )
    for item in ranked:
        convention = item["convention"]
        s_value = item["s_value"]
        signed_s = item["signed_s"]
        expectations = item["expectations"]
        last_e = "  ".join(
            f"{label}={expectations[label][-1]:+.3f}"
            for label in EXPECTATION_LABELS
        )
        finite = np.isfinite(s_value)
        if np.any(finite):
            max_index = int(np.nanargmax(s_value))
            max_text = (
                f"max |S|={s_value[max_index]:.3f} "
                f"(signed={signed_s[max_index]:+.3f})"
            )
        else:
            max_text = "max |S|=nan"
        print(
            f"{convention.name:32s} | "
            f"mean |S|={np.nanmean(s_value):.3f} | "
            f"{max_text} | "
            f"last |S|={s_value[-1]:.3f} | "
            f"last signed={signed_s[-1]:+.3f} | {last_e}"
        )


def plot_conventions(
    csv_path: Path,
    x: np.ndarray,
    x_label: str,
    latest_timestamp: float | None,
    convention_results: list[dict[str, object]],
) -> None:
    figure, axes = plt.subplots(
        5,
        1,
        figsize=(14, 12),
        sharex=True,
        gridspec_kw={"height_ratios": (1.5, 1.0, 1.0, 1.0, 1.0)},
    )
    s_axis = axes[0]
    expectation_axes = dict(zip(EXPECTATION_LABELS, axes[1:]))

    for index, item in enumerate(convention_results):
        convention = item["convention"]
        s_value = item["s_value"]
        expectations = item["expectations"]
        color = COLORS[index % len(COLORS)]
        s_axis.plot(x, s_value, marker=".", markersize=5, linewidth=1.2,
                    color=color, label=convention.name)
        for label, axis in expectation_axes.items():
            axis.plot(x, expectations[label], marker=".", markersize=4,
                      linewidth=1.0, color=color, label=convention.name)

    s_axis.axhline(2.0, color="#666666", linestyle=":", linewidth=1,
                   label="2.0 classical")
    s_axis.axhline(2.5, color="#333333", linestyle="--", linewidth=1,
                   label="2.5")
    s_axis.set_ylabel("|S|")
    s_axis.set_ylim(0.0, 3.0)
    s_axis.set_title(f"CHSH convention/sign-swap check: {csv_path.name}")

    for label, axis in expectation_axes.items():
        axis.axhline(0.0, color="#666666", linestyle=":", linewidth=1)
        axis.set_ylabel(label)
        axis.set_ylim(-1.05, 1.05)

    axes[-1].set_xlabel(x_label)
    if USE_RELATIVE_TIME and latest_timestamp is not None:
        axes[-1].xaxis.set_major_formatter(FuncFormatter(format_relative_time))
        top_axis = s_axis.secondary_xaxis("top")
        top_axis.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, position: format_absolute_time(
                    value,
                    position,
                    latest_timestamp,
                )
            )
        )
        top_axis.set_xlabel("Measurement clock time")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", ncol=2, fontsize=8)

    figure.tight_layout()
    if SAVE_PATH is not None:
        output_path = SAVE_PATH.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=160)
        print(f"Saved plot: {output_path}")
    plt.show()


def format_relative_time(value: float, _position: float) -> str:
    sign = "-" if value < -0.5 else ""
    seconds = int(round(abs(value)))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{sign}{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours:
        return f"{sign}{hours}:{minutes:02d}:{seconds:02d}"
    return f"{sign}{minutes}:{seconds:02d}"


def format_absolute_time(
    relative_seconds: float,
    _position: float,
    latest_timestamp: float,
) -> str:
    timestamp = latest_timestamp + relative_seconds
    time_value = dt.datetime.fromtimestamp(timestamp)
    if abs(relative_seconds) >= 86_400:
        return time_value.strftime("%m-%d %H:%M")
    return time_value.strftime("%H:%M:%S")


if __name__ == "__main__":
    main()
