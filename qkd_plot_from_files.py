from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_FILES = (
    BASE_DIR / "Data" / "alice_results.csv",
    BASE_DIR / "Data" / "qber_iterlog.csv",
)
CSV_FILE: Path | None = None
REFRESH_INTERVAL_SECONDS = 2.0
HISTORY_ROWS = 100  # Set to 0 to display all rows.
LIVE_UPDATE = True
SAVE_PATH: Path | None = None
PAIR_LABELS = ("HH", "VV", "DD", "AA", "HV", "VH", "DA", "AD")
PAIR_COLORS = {
    "HH": "#1f77b4",
    "VV": "#ff7f0e",
    "DD": "#2ca02c",
    "AA": "#d62728",
    "HV": "#9467bd",
    "VH": "#8c564b",
    "DA": "#e377c2",
    "AD": "#7f7f7f",
}


@dataclass(frozen=True)
class MeasurementSeries:
    measurement_index: np.ndarray
    elapsed_minutes: np.ndarray
    counts: dict[str, np.ndarray]
    total_coincidences: np.ndarray
    visibility_hv: np.ndarray
    visibility_da: np.ndarray

    @property
    def size(self) -> int:
        return int(self.measurement_index.size)



def choose_csv_path(requested_path: Path | None) -> Path:
    if requested_path is not None:
        return requested_path.expanduser().resolve()

    for path in DEFAULT_CSV_FILES:
        if path.is_file():
            return path
    return DEFAULT_CSV_FILES[0]


def read_measurements(path: Path, history: int) -> MeasurementSeries:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if history > 0:
        rows = rows[-history:]
    if not rows:
        raise ValueError(f"{path} contains no measurement rows")

    missing = [
        f"C_{label}"
        for label in PAIR_LABELS
        if f"C_{label}" not in rows[0]
    ]
    if missing:
        raise ValueError(
            f"{path} is missing coincidence columns: {', '.join(missing)}"
        )

    counts = {
        label: _float_column(rows, f"C_{label}")
        for label in PAIR_LABELS
    }
    total = _optional_float_column(rows, "total_coincidences")
    if total is None:
        total = np.sum(np.stack([counts[label] for label in PAIR_LABELS]),
      axis=0,
  )

    visibility_hv = _optional_float_column(rows, "vis_HV")
    if visibility_hv is None:
        visibility_hv = _visibility(
            counts["HH"] + counts["VV"],
            counts["HV"] + counts["VH"],
        )

    visibility_da = _optional_float_column(rows, "vis_DA")
    if visibility_da is None:
        visibility_da = _visibility(
            counts["DD"] + counts["AA"],
            counts["DA"] + counts["AD"],
        )

    timestamps = _optional_float_column(rows, "timestamp")
    measurement_index = np.arange(1, len(rows) + 1, dtype=np.int64)
    if timestamps is None or not np.all(np.isfinite(timestamps)):
        elapsed_minutes = measurement_index.astype(np.float64) - 1.0
    else:
        elapsed_minutes = (timestamps - timestamps[0]) / 60.0

    return MeasurementSeries(
        measurement_index=measurement_index,
        elapsed_minutes=elapsed_minutes,
        counts=counts,
        total_coincidences=total,
        visibility_hv=visibility_hv,
        visibility_da=visibility_da,
    )


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values: list[float] = []
    for row_number, row in enumerate(rows, start=2):
        value = row.get(name, "")
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {name!r} value on CSV row {row_number}: {value!r}"
            ) from exc
    return np.asarray(values, dtype=np.float64)


def _optional_float_column(
    rows: list[dict[str, str]],
    name: str,
) -> np.ndarray | None:
    if name not in rows[0]:
        return None
    if any(row.get(name, "") == "" for row in rows):
        return None
    return _float_column(rows, name)


def _visibility(correlated: np.ndarray, errors: np.ndarray) -> np.ndarray:
    total = correlated + errors
    return np.divide(
        correlated - errors,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )


class MeasurementPlot:
    def __init__(self, source_path: Path) -> None:
        self.source_path = source_path
        self.figure, axes = plt.subplots(
            3,
            1,
            figsize=(13, 10),
            sharex=True,
            gridspec_kw={"height_ratios": (2.2, 1.2, 1.4)},
        )
        self.pairs_ax, self.total_ax, self.visibility_ax = axes
        self.figure.canvas.manager.set_window_title("QKD measurement monitor")

    def update(self, series: MeasurementSeries) -> None:
        x = series.elapsed_minutes
        self.pairs_ax.clear()
        self.total_ax.clear()
        self.visibility_ax.clear()

        for label in PAIR_LABELS:
            self.pairs_ax.plot(
                x,
                series.counts[label],
                marker=".",
                markersize=5,
                linewidth=1.3,
                color=PAIR_COLORS[label],
                label=label,
            )

        self.total_ax.plot(
            x,
            series.total_coincidences,
            color="black",
            marker=".",
            markersize=6,
            linewidth=1.8,
            label="Total",
        )
        self.visibility_ax.plot(
            x,
            series.visibility_hv,
            color="#0072b2",
            marker=".",
            linewidth=1.8,
            label="H/V visibility",
        )
        self.visibility_ax.plot(
            x,
            series.visibility_da,
            color="#d55e00",
            marker=".",
            linewidth=1.8,
            label="D/A visibility",
        )

        self.pairs_ax.set_title(
            f"QKD measurements: {self.source_path.name} "
            f"({series.size} displayed rows)"
        )
        self.pairs_ax.set_ylabel("Coincidences")
        self.total_ax.set_ylabel("Total coincidences")
        self.visibility_ax.set_ylabel("Visibility")
        self.visibility_ax.set_xlabel("\"Time\"")
        self.visibility_ax.set_ylim(-1.05, 1.05)
        self.visibility_ax.axhline(
            0.95,
            color="#666666",
            linestyle="--",
            linewidth=1,
            label="0.95 target",
        )

        for axis in (self.pairs_ax, self.total_ax, self.visibility_ax):
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", ncol=4, fontsize=9)

        self.figure.tight_layout()
        self.figure.canvas.draw_idle()


def wait_for_csv(path: Path, interval_seconds: float) -> bool:
    while not path.is_file():
        print(f"Waiting for CSV: {path}", end="\r", flush=True)
        plt.pause(interval_seconds)
        if not plt.fignum_exists(plt.gcf().number):
            return False
    print(f"Reading CSV: {path}          ")
    return True


def main() -> None:
    csv_path = choose_csv_path(CSV_FILE)
    if REFRESH_INTERVAL_SECONDS <= 0:
        raise ValueError("REFRESH_INTERVAL_SECONDS must be positive")
    if HISTORY_ROWS < 0:
        raise ValueError("HISTORY_ROWS cannot be negative")

    plot = MeasurementPlot(csv_path)
    plt.ion()
    plt.show()
    if not wait_for_csv(csv_path, REFRESH_INTERVAL_SECONDS):
        return

    last_signature: tuple[int, int] | None = None
    while plt.fignum_exists(plot.figure.number):
        try:
            stat = csv_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                series = read_measurements(csv_path, HISTORY_ROWS)
                plot.update(series)
                if SAVE_PATH is not None:
                    output_path = SAVE_PATH.expanduser()
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    plot.figure.savefig(output_path, dpi=160)
                last_signature = signature
        except (OSError, ValueError, csv.Error) as exc:
            # A writer may briefly replace the file while extending its schema.
            print(f"Plot refresh skipped: {exc}")

        if not LIVE_UPDATE:
            plt.ioff()
            plt.show()
            return
        plt.pause(REFRESH_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
