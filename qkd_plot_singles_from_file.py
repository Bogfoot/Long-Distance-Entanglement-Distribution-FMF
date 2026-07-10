from __future__ import annotations

import csv
import datetime as dt
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
)

BASE_DIR = Path(__file__).resolve().parent
CSV_FILE = BASE_DIR / "Data" / "alice_results.csv"
REFRESH_INTERVAL_SECONDS = 2.0
HISTORY_ROWS = 400  # Set to 0 to display all rows.
LIVE_UPDATE = True
USE_CONSTANT_POINT_SPACING = True
SAVE_PATH: Path | None = None
PLOT_RATE = True  # True: events/s, False: events in the analyzed overlap.
DEDUPLICATE_FILES = True

POLARIZATIONS = ("H", "V", "D", "A")
COLORS = {
    "H": "#0072b2",
    "V": "#d55e00",
    "D": "#009e73",
    "A": "#cc79a7",
}


@dataclass(frozen=True)
class SinglesSeries:
    x: np.ndarray
    timestamps: np.ndarray
    alice: dict[str, np.ndarray]
    bob: dict[str, np.ndarray]


def read_singles(path: Path, history_rows: int) -> SinglesSeries:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows = [row for row in rows if _has_singles(row)]
    if DEDUPLICATE_FILES:
        rows = _deduplicate_measurement_files(rows)
    if history_rows > 0:
        rows = rows[-history_rows:]

    timestamps = _float_column(rows, "timestamp")
    durations = _float_column(rows, "overlap_duration_sec")
    if PLOT_RATE:
        durations = np.where(durations > 0.0, durations, np.nan)
    else:
        durations = np.ones_like(durations)

    alice = {
        pol: _side_polarization_values(rows, "alice", pol) / durations
        for pol in POLARIZATIONS
    }
    bob = {
        pol: _side_polarization_values(rows, "bob", pol) / durations
        for pol in POLARIZATIONS
    }

    if USE_CONSTANT_POINT_SPACING:
        x = np.arange(1, len(rows) + 1, dtype=float)
    else:
        x = timestamps

    return SinglesSeries(x=x, timestamps=timestamps, alice=alice, bob=bob)


def _has_singles(row: dict[str, str]) -> bool:
    return any(
        key.startswith(("alice_events_", "bob_events_")) and _is_float(value)
        for key, value in row.items()
    )


def _deduplicate_measurement_files(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_file: OrderedDict[str, dict[str, str]] = OrderedDict()
    for row in rows:
        key = row.get("alice_file") or row.get("timestamp") or str(len(by_file))
        previous = by_file.get(key)
        if previous is None or _event_column_count(row) >= _event_column_count(previous):
            by_file[key] = row
    return list(by_file.values())


def _event_column_count(row: dict[str, str]) -> int:
    return sum(
        1
        for key, value in row.items()
        if key.startswith(("alice_events_", "bob_events_")) and _is_float(value)
    )


def _side_polarization_values(
    rows: list[dict[str, str]],
    side: str,
    polarization: str,
) -> np.ndarray:
    values = []
    prefix = f"{side}_events_"
    for row in rows:
        candidates = []
        for key, text in row.items():
            if not key.startswith(prefix) or not _is_float(text):
                continue
            pair = key.removeprefix(prefix)
            if len(pair) < 2:
                continue
            pair_polarization = pair[0] if side == "alice" else pair[1]
            if pair_polarization == polarization:
                candidates.append(float(text))
        values.append(float(np.median(candidates)) if candidates else np.nan)
    return np.asarray(values, dtype=float)


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return np.asarray([_float_value(row.get(name, "")) for row in rows], dtype=float)


def _float_value(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return np.nan


def _is_float(text: str) -> bool:
    return np.isfinite(_float_value(text))


class SinglesPlot:
    def __init__(self) -> None:
        self.figure, (self.alice_axis, self.bob_axis) = plt.subplots(
            2,
            1,
            figsize=(12, 7),
            sharex=True,
        )
        self.lines: dict[tuple[str, str], object] = {}
        for side, axis in (("Alice", self.alice_axis), ("Bob", self.bob_axis)):
            for pol in POLARIZATIONS:
                (line,) = axis.plot(
                    [],
                    [],
                    marker=".",
                    markersize=4,
                    linewidth=1.2,
                    label=pol,
                    color=COLORS[pol],
                )
                self.lines[(side.lower(), pol)] = line
            axis.set_ylabel(_y_label())
            axis.set_title(f"{side} singles")
            axis.grid(True, alpha=0.25)
            axis.legend(ncol=4, loc="best")
        self.bob_axis.set_xlabel(
            "Measurement" if USE_CONSTANT_POINT_SPACING else "Time"
        )
        if not USE_CONSTANT_POINT_SPACING:
            self.bob_axis.xaxis.set_major_formatter(
                FuncFormatter(_format_timestamp_axis)
            )
        self.figure.tight_layout()

    def update(self, series: SinglesSeries, csv_path: Path) -> None:
        for pol in POLARIZATIONS:
            self.lines[("alice", pol)].set_data(series.x, series.alice[pol])
            self.lines[("bob", pol)].set_data(series.x, series.bob[pol])

        latest_text = _latest_time_text(series.timestamps)
        self.figure.suptitle(f"Singles from {csv_path.name} | latest {latest_text}")
        self.figure.tight_layout(rect=(0, 0, 1, 0.96))
        for axis in (self.alice_axis, self.bob_axis):
            axis.relim()
            axis.autoscale_view()

        if SAVE_PATH is not None:
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            self.figure.savefig(SAVE_PATH, dpi=160)


def _y_label() -> str:
    return "Singles rate (events/s)" if PLOT_RATE else "Singles per measurement"


def _latest_time_text(timestamps: np.ndarray) -> str:
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size == 0:
        return "unknown"
    return dt.datetime.fromtimestamp(float(finite[-1])).strftime("%Y-%m-%d %H:%M:%S")


def _format_timestamp_axis(value: float, _position: float) -> str:
    if not np.isfinite(value):
        return ""
    return dt.datetime.fromtimestamp(float(value)).strftime("%H:%M:%S")


def wait_for_csv(path: Path, interval_seconds: float) -> bool:
    while not path.exists():
        print(f"Waiting for {path} ...")
        plt.pause(interval_seconds)
        if not plt.fignum_exists(plt.gcf().number):
            return False
    return True


def main() -> None:
    csv_path = CSV_FILE.expanduser()
    plot = SinglesPlot()
    plt.ion()
    if not wait_for_csv(csv_path, REFRESH_INTERVAL_SECONDS):
        return

    while True:
        try:
            series = read_singles(csv_path, HISTORY_ROWS)
            plot.update(series, csv_path)
        except Exception as exc:
            print(f"Could not update singles plot: {exc}")

        plt.pause(REFRESH_INTERVAL_SECONDS if LIVE_UPDATE else 0.001)
        if not LIVE_UPDATE:
            break
        if not plt.fignum_exists(plot.figure.number):
            break

    if not LIVE_UPDATE:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
