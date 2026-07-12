from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

from qkd_zerothird_analysis_common import (
    GOOD_RESULTS_DIR,
    LOCAL_TZ,
    METADATA_CSV,
    QBER_ITERLOG_CSV,
    finite_values,
    float_value,
    format_float,
    format_percent,
    hours_between,
    load_metadata,
    local_time_text,
    median,
    percentile,
    read_csv_rows,
)


OUTPUT_DIR = GOOD_RESULTS_DIR / "Stability"

MONITOR_BACKEND_NAME = "monitor"
BLOCK_GAP_SECONDS = 30 * 60
SAVE_PLOTS = True
SHOW_PLOTS = False


@dataclass
class MonitorRecord:
    row: dict[str, str]
    timestamp_utc: dt.datetime

    @property
    def monitor_name(self) -> str:
        return self.row.get("optimizer_name", "")

    @property
    def phase(self) -> str:
        return self.row.get("phase", "")

    @property
    def metric(self) -> str:
        return self.row.get("objective_metric", "")


def local_plot_time(timestamp_utc):
    return timestamp_utc.astimezone(LOCAL_TZ).replace(tzinfo=None)


def stability_start_utc() -> dt.datetime:
    metadata = load_metadata(METADATA_CSV)
    starts = [item.start_utc for item in metadata if item.mode == "stability_monitor"]
    if not starts:
        raise RuntimeError("Metadata has no stability_monitor phase")
    return min(starts)


def read_monitor_records() -> list[MonitorRecord]:
    start_utc = stability_start_utc()
    records: list[MonitorRecord] = []
    for row in read_csv_rows(QBER_ITERLOG_CSV):
        if row.get("optimizer_backend") != MONITOR_BACKEND_NAME:
            continue
        try:
            timestamp = dt.datetime.fromtimestamp(
                float(row["timestamp"]),
                dt.timezone.utc,
            )
        except (TypeError, ValueError):
            continue
        if timestamp >= start_utc:
            records.append(MonitorRecord(row=row, timestamp_utc=timestamp))
    records.sort(key=lambda record: record.timestamp_utc)
    return records


def split_blocks(records: list[MonitorRecord]) -> list[list[MonitorRecord]]:
    if not records:
        return []

    blocks: list[list[MonitorRecord]] = [[records[0]]]
    for record in records[1:]:
        previous = blocks[-1][-1]
        gap = (record.timestamp_utc - previous.timestamp_utc).total_seconds()
        if (
            record.monitor_name != previous.monitor_name
            or record.phase != previous.phase
            or gap > BLOCK_GAP_SECONDS
        ):
            blocks.append([record])
        else:
            blocks[-1].append(record)
    return blocks


def block_metric_values(block: list[MonitorRecord]) -> tuple[str, list[float]]:
    metric = block[0].metric
    if metric == "chsh_s":
        return "CHSH_S_value", [
            float_value(record.row, "CHSH_S_value") for record in block
        ]
    return "QBER", [float_value(record.row, "QBER") for record in block]


def threshold_crossing_text(block: list[MonitorRecord], column: str, values: list[float]) -> str:
    if column == "QBER":
        for record, value in zip(block, values):
            if value >= 0.50:
                hours = hours_between(block[0].timestamp_utc, record.timestamp_utc)
                return f"QBER >= 50% after {hours:.2f} h"
        return "QBER did not reach 50%"

    for record, value in zip(block, values):
        if value <= 1.8:
            hours = hours_between(block[0].timestamp_utc, record.timestamp_utc)
            return f"S <= 1.8 after {hours:.2f} h"
    return "S did not fall to 1.8"


def summarize_block(index: int, block: list[MonitorRecord]) -> list[str]:
    column, values = block_metric_values(block)
    clean = finite_values(values)
    first = block[0].timestamp_utc
    last = block[-1].timestamp_utc
    duration_h = hours_between(first, last)
    first_value = clean[0] if clean else float("nan")
    last_value = clean[-1] if clean else float("nan")
    drift_per_hour = (
        (last_value - first_value) / duration_h
        if duration_h > 0 and clean
        else float("nan")
    )
    value_text = (
        f"first={format_percent(first_value, 3)} "
        f"latest={format_percent(last_value, 3)} "
        f"median={format_percent(median(clean), 3)} "
        f"p10={format_percent(percentile(clean, 10), 3)} "
        f"p90={format_percent(percentile(clean, 90), 3)} "
        if column == "QBER"
        else f"first={format_float(first_value, 4)} "
        f"latest={format_float(last_value, 4)} "
        f"median={format_float(median(clean), 4)} "
        f"p10={format_float(percentile(clean, 10), 4)} "
        f"p90={format_float(percentile(clean, 90), 4)} "
    )
    drift_text = (
        format_percent(drift_per_hour, 3)
        if column == "QBER"
        else format_float(drift_per_hour, 4)
    )

    return [
        f"Block {index}: {block[0].monitor_name} ({block[0].phase}, {block[0].metric})",
        f"  rows: {len(block)}",
        f"  first: {local_time_text(first)}",
        f"  last: {local_time_text(last)}",
        f"  span_hours: {duration_h:.2f}",
        f"  {column}: {value_text}",
        f"  drift_per_hour: {drift_text}",
        f"  threshold: {threshold_crossing_text(block, column, values)}",
    ]


def write_summary(records: list[MonitorRecord], blocks: list[list[MonitorRecord]]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "stability_summary.txt"
    lines = [
        "ZeroThird Stability Monitor Summary",
        "",
        f"Input CSV: {QBER_ITERLOG_CSV}",
        f"Metadata CSV: {METADATA_CSV}",
        f"Monitor rows after stability start: {len(records)}",
        f"Monitor blocks: {len(blocks)}",
        "",
    ]
    for index, block in enumerate(blocks, start=1):
        lines.extend(summarize_block(index, block))
        lines.append("")
    path.write_text("\n".join(lines))
    return path


def save_or_show(path: Path) -> None:
    plt.tight_layout()
    if SAVE_PLOTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180)
        print(f"Saved plot: {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def plot_blocks(blocks: list[list[MonitorRecord]]) -> Path:
    path = OUTPUT_DIR / "stability_monitor_blocks.png"
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=False)

    for block in blocks:
        column, values = block_metric_values(block)
        t0 = block[0].timestamp_utc
        hours = [hours_between(t0, record.timestamp_utc) for record in block]
        label = block[0].monitor_name
        if column == "QBER":
            axes[0].plot(hours, values, marker=".", linestyle="None", label=label)
        else:
            axes[1].plot(hours, values, marker=".", linestyle="None", label=label)

    axes[0].axhline(0.50, color="#aa3333", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("QBER")
    axes[0].set_xlabel("Hours since block start")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].axhline(1.8, color="#aa3333", linestyle="--", linewidth=0.8)
    axes[1].axhline(2.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("CHSH S")
    axes[1].set_xlabel("Hours since block start")
    axes[1].set_ylim(0.0, 3.0)
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    figure.suptitle("ZeroThird Stability Monitor Blocks")
    save_or_show(path)
    return path


def plot_absolute_timeline(records: list[MonitorRecord]) -> Path:
    path = OUTPUT_DIR / "stability_monitor_timeline.png"
    qber_records = [record for record in records if record.metric != "chsh_s"]
    chsh_records = [record for record in records if record.metric == "chsh_s"]

    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    if qber_records:
        axes[0].plot(
            [local_plot_time(record.timestamp_utc) for record in qber_records],
            [float_value(record.row, "QBER") for record in qber_records],
            marker=".",
            linestyle="None",
        )
    if chsh_records:
        axes[1].plot(
            [local_plot_time(record.timestamp_utc) for record in chsh_records],
            [float_value(record.row, "CHSH_S_value") for record in chsh_records],
            marker=".",
            linestyle="None",
        )

    axes[0].axhline(0.50, color="#aa3333", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("QBER")
    axes[0].grid(True, alpha=0.25)
    axes[1].axhline(1.8, color="#aa3333", linestyle="--", linewidth=0.8)
    axes[1].axhline(2.0, color="#777777", linestyle=":", linewidth=0.8)
    axes[1].set_ylabel("CHSH S")
    axes[1].set_xlabel("Local time")
    axes[1].set_ylim(0.0, 3.0)
    axes[1].grid(True, alpha=0.25)
    figure.suptitle("ZeroThird Stability Monitor Timeline")
    save_or_show(path)
    return path


def main() -> None:
    records = read_monitor_records()
    if not records:
        raise RuntimeError("No monitor rows found after stability-monitor start")
    blocks = split_blocks(records)
    summary_path = write_summary(records, blocks)
    print(f"Saved summary: {summary_path}")
    plot_blocks(blocks)
    plot_absolute_timeline(records)


if __name__ == "__main__":
    main()
