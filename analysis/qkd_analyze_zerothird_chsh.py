from __future__ import annotations

from collections import Counter, defaultdict
import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from qkd_zerothird_analysis_common import (
    CHSH_EXPECTATION_COLUMNS,
    GOOD_RESULTS_DIR,
    LOCAL_TZ,
    METADATA_CSV,
    ZEROTHIRD_CSV,
    filename_kind,
    finite_values,
    float_value,
    format_float,
    hours_between,
    temperature_rows_for_metric,
    ALICE_CHSH_TARGET,
    mean,
    standard_error,
    stdev,
    is_chsh_row,
    load_metadata,
    local_time_text,
    median,
    metadata_for_time,
    percentile,
    read_csv_rows,
    rows_with_metadata,
)


OUTPUT_DIR = GOOD_RESULTS_DIR / "CHSH"

S_LIMITS = (2.0, 2.4)
SAVE_PLOTS = True
SHOW_PLOTS = False


def local_plot_time(timestamp_utc):
    return timestamp_utc.astimezone(LOCAL_TZ).replace(tzinfo=None)


def metadata_label(meta) -> str:
    return meta.window_label if meta is not None else "unknown"


def metadata_mode(meta) -> str:
    return meta.mode if meta is not None else "unknown"


def chsh_records():
    rows = read_csv_rows(ZEROTHIRD_CSV)
    metadata = load_metadata(METADATA_CSV)
    records = [
        (row, timestamp, meta)
        for row, timestamp, meta in rows_with_metadata(rows, metadata)
        if is_chsh_row(row)
    ]
    records.sort(key=lambda item: item[1])
    return records, metadata


def grouped_by_window(records):
    groups = defaultdict(list)
    for row, timestamp, meta in records:
        groups[metadata_label(meta)].append((row, timestamp, meta))
    return dict(groups)


def row_values(records, column: str) -> list[float]:
    return [float_value(row, column) for row, _, _ in records]


def summary_for_records(label: str, records) -> list[str]:
    if not records:
        return [f"{label}: no rows"]

    s_values = finite_values(row_values(records, "CHSH_S_value"))
    e_values = {
        column: finite_values(row_values(records, column))
        for column in CHSH_EXPECTATION_COLUMNS
    }
    counts = finite_values(row_values(records, "total_coincidences"))
    first_time = records[0][1]
    last_time = records[-1][1]
    duration_h = hours_between(first_time, last_time)
    file_kinds = Counter(filename_kind(row) for row, _, _ in records)

    lines = [
        f"{label}:",
        f"  rows: {len(records)}",
        f"  first: {local_time_text(first_time)}",
        f"  last: {local_time_text(last_time)}",
        f"  span_hours: {duration_h:.2f}",
        f"  filename_kinds: {dict(file_kinds)}",
        (
            "  CHSH_S_value: "
            f"best={format_float(max(s_values), 4)} "
            f"median={format_float(median(s_values), 4)} "
            f"p10={format_float(percentile(s_values, 10), 4)} "
            f"p90={format_float(percentile(s_values, 90), 4)} "
            f"latest={format_float(s_values[-1], 4)}"
        ),
        (
            "  total_coincidences: "
            f"median={format_float(median(counts), 1)} "
            f"p10={format_float(percentile(counts, 10), 1)} "
            f"p90={format_float(percentile(counts, 90), 1)}"
        ),
    ]
    for limit in S_LIMITS:
        fraction = (
            sum(value >= limit for value in s_values) / len(s_values)
            if s_values
            else 0.0
        )
        lines.append(f"  fraction_S_above_{limit:g}: {100.0 * fraction:.1f}%")
    for column, values in e_values.items():
        lines.append(
            f"  {column}: median={format_float(median(values), 4)} "
            f"latest={format_float(values[-1], 4)}"
        )
    lines.append(
        subset_stats_text("S > 2", [value for value in s_values if value > 2.0])
    )
    lines.extend(temperature_summary_text("CHSH_S_value", label))
    return lines


def temperature_summary_text(metric_column: str, label: str) -> list[str]:
    rows = temperature_rows_for_metric(metric_column)
    if metric_column == "CHSH_S_value":
        rows = [row for row in rows if row.get("CHSH_S_value", "").strip()]
    if not rows:
        return [f"  temperature: no rows for {label}"]

    def temp_values(column: str) -> list[float]:
        return finite_values([float_value(row, column) for row in rows])

    t_ljubljana = temp_values("T_Ljubljana")
    t_drnovo = temp_values("T_Drnovo")
    t_delta = temp_values("T_delta")
    lines = [
        f"  temperature: rows={len(rows)}",
        (
            "    T_Ljubljana: "
            f"median={format_float(median(t_ljubljana), 2)} °C "
            f"range={format_float(min(t_ljubljana), 2)}..{format_float(max(t_ljubljana), 2)} °C"
        ),
        (
            "    T_Drnovo: "
            f"median={format_float(median(t_drnovo), 2)} °C "
            f"range={format_float(min(t_drnovo), 2)}..{format_float(max(t_drnovo), 2)} °C"
        ),
        (
            "    T_delta: "
            f"median={format_float(median(t_delta), 2)} °C "
            f"range={format_float(min(t_delta), 2)}..{format_float(max(t_delta), 2)} °C"
        ),
    ]
    return lines


def write_summary(records, metadata) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "chsh_summary.txt"
    groups = grouped_by_window(records)
    mode_counts = Counter(metadata_mode(meta) for _, _, meta in records)
    file_kinds = Counter(filename_kind(row) for row, _, _ in records)

    lines = [
        "CHSH S Summary",
        "",
        f"Input CSV: {ZEROTHIRD_CSV}",
        f"Metadata CSV: {METADATA_CSV}",
        f"Valid CHSH rows: {len(records)}",
        f"Mode row counts: {dict(mode_counts)}",
        f"Filename-kind row counts: {dict(file_kinds)}",
        "",
    ]
    lines.extend(summary_for_records("All CHSH rows", records))
    lines.append("")
    lines.append("By coincidence window:")
    for label in sorted(groups):
        lines.extend(summary_for_records(label, groups[label]))
        lines.append("")
    lines.append("Metadata phases:")
    for meta in metadata:
        lines.append(
            f"  {local_time_text(meta.start_utc)} | {meta.window_label} | "
            f"{meta.mode} | {meta.experiment} | {meta.notes}"
        )
    lines.append("")

    path.write_text("\n".join(lines))
    return path


WINDOW_REGION_ALPHA = 0.12
WINDOW_REGION_COLORS = {
    "160 ps, 1 sigma": "#4C78A8",
    "320 ps, 2 sigma": "#F58518",
    "475 ps, 3 sigma": "#54A24B",
    "480 ps, 3 sigma": "#B279A2",
    "500 ps, 3 sigma": "#E45756",
    "750 ps": "#72B7B2",
    "800 ps": "#FF9DA6",
    "unknown": "#BAB0AC",
}


def window_region_color(label: str) -> str:
    return WINDOW_REGION_COLORS.get(label, "#9D755D")


def merged_metadata_regions(metadata, first_time, last_time):
    if not metadata or first_time >= last_time:
        return []

    active = None
    future = []
    for meta in sorted(metadata, key=lambda item: item.start_utc):
        if meta.start_utc <= first_time:
            active = meta
        elif meta.start_utc < last_time:
            future.append(meta)
        else:
            break

    regions = []
    region_start = first_time
    for meta in future:
        if active is not None:
            regions.append((region_start, meta.start_utc, active.window_label))
        region_start = meta.start_utc
        active = meta
    if active is not None:
        regions.append((region_start, last_time, active.window_label))

    merged = []
    for start, end, label in regions:
        if end <= start:
            continue
        if merged and merged[-1][2] == label:
            previous_start, _, previous_label = merged[-1]
            merged[-1] = (previous_start, end, previous_label)
        else:
            merged.append((start, end, label))
    return merged


def add_metadata_regions(axis, regions, annotate: bool = False) -> None:
    for start, end, label in regions:
        axis.axvspan(
            local_plot_time(start),
            local_plot_time(end),
            facecolor=window_region_color(label),
            alpha=WINDOW_REGION_ALPHA,
            edgecolor="none",
            zorder=0,
        )
        if annotate:
            midpoint = start + (end - start) / 2
            axis.text(
                local_plot_time(midpoint),
                0.98,
                label,
                rotation=90,
                va="top",
                ha="center",
                transform=axis.get_xaxis_transform(),
                fontsize=8,
                color="#333333",
            )


def region_legend_handles(regions):
    handles = []
    seen = set()
    for _, _, label in regions:
        if label in seen:
            continue
        seen.add(label)
        handles.append(
            Patch(
                facecolor=window_region_color(label),
                alpha=WINDOW_REGION_ALPHA,
                edgecolor="none",
                label=f"window: {label}",
            )
        )
    return handles


def add_legend_with_regions(axis, regions, loc="best", ncol=2) -> None:
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(region_legend_handles(regions))
    if handles:
        axis.legend(handles=handles, loc=loc, ncol=ncol, fontsize=8)




WINDOW_REGION_ALPHA = 0.12
WINDOW_REGION_COLORS = {
    "160 ps, 1 sigma": "#4C78A8",
    "320 ps, 2 sigma": "#F58518",
    "475 ps, 3 sigma": "#54A24B",
    "480 ps, 3 sigma": "#B279A2",
    "500 ps, 3 sigma": "#E45756",
    "750 ps": "#72B7B2",
    "800 ps": "#FF9DA6",
    "unknown": "#BAB0AC",
}


def window_region_color(label: str) -> str:
    return WINDOW_REGION_COLORS.get(label, "#9D755D")


def merged_metadata_regions(metadata, first_time, last_time):
    if not metadata or first_time >= last_time:
        return []

    active = None
    future = []
    for meta in sorted(metadata, key=lambda item: item.start_utc):
        if meta.start_utc <= first_time:
            active = meta
        elif meta.start_utc < last_time:
            future.append(meta)
        else:
            break

    regions = []
    region_start = first_time
    for meta in future:
        if active is not None:
            regions.append((region_start, meta.start_utc, active.window_label))
        region_start = meta.start_utc
        active = meta
    if active is not None:
        regions.append((region_start, last_time, active.window_label))

    merged = []
    for start, end, label in regions:
        if end <= start:
            continue
        if merged and merged[-1][2] == label:
            previous_start, _, previous_label = merged[-1]
            merged[-1] = (previous_start, end, previous_label)
        else:
            merged.append((start, end, label))
    return merged


def add_metadata_regions(axis, regions, annotate: bool = False) -> None:
    for start, end, label in regions:
        axis.axvspan(
            local_plot_time(start),
            local_plot_time(end),
            facecolor=window_region_color(label),
            alpha=WINDOW_REGION_ALPHA,
            edgecolor="none",
            zorder=0,
        )
        if annotate:
            midpoint = start + (end - start) / 2
            axis.text(
                local_plot_time(midpoint),
                0.98,
                label,
                rotation=90,
                va="top",
                ha="center",
                transform=axis.get_xaxis_transform(),
                fontsize=8,
                color="#333333",
            )


def region_legend_handles(regions):
    handles = []
    seen = set()
    for _, _, label in regions:
        if label in seen:
            continue
        seen.add(label)
        handles.append(
            Patch(
                facecolor=window_region_color(label),
                alpha=WINDOW_REGION_ALPHA,
                edgecolor="none",
                label=f"window: {label}",
            )
        )
    return handles


def add_legend_with_regions(axis, regions, loc="best", ncol=2) -> None:
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(region_legend_handles(regions))
    if handles:
        axis.legend(handles=handles, loc=loc, ncol=ncol, fontsize=8)




def save_or_show(path: Path) -> None:
    plt.tight_layout()
    if SAVE_PLOTS:
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=180)
        print(f"Saved plot: {path}")
    if SHOW_PLOTS:
        plt.show()
    plt.close()


def boxplot_legend_handles() -> list[Line2D | Patch]:
    return [
        Patch(
            facecolor="#d9d9d9",
            edgecolor="#333333",
            label="25th to 75th percentile",
        ),
        Line2D([0], [0], color="#111111", linewidth=1.6, label="median"),
        Line2D([0], [0], color="#111111", linewidth=1.0, label="whiskers = 1.5 x IQR"),
    ]


def reference_line_handles(entries) -> list[Line2D]:
    return [
        Line2D([0], [0], color=color, linestyle=style, linewidth=1.2, label=label)
        for _, label, color, style in entries
    ]


def subset_stats_text(label: str, values: list[float], percent: bool = False) -> str:
    clean = finite_values(values)
    if not clean:
        return f"  {label}: no rows"
    if percent:
        return (
            f"  {label}: n={len(clean)} "
            f"mean={format_float(mean(clean), 4)} "
            f"median={format_float(median(clean), 4)} "
            f"std={format_float(stdev(clean), 4)} "
            f"sem={format_float(standard_error(clean), 4)}"
        )
    return (
        f"  {label}: n={len(clean)} "
        f"mean={format_float(mean(clean), 4)} "
        f"median={format_float(median(clean), 4)} "
        f"std={format_float(stdev(clean), 4)} "
        f"sem={format_float(standard_error(clean), 4)}"
    )


S_LIMITS = (2.0, 2.4)
TSIRELSON = 2.0 * np.sqrt(2.0)

CHSH_REFERENCE_LINES = [
    (2.0, "S = 2", "#1F77B4", "--"),
    (2.4, "S = 2.4", "#FF7F0E", "-."),
    (TSIRELSON, "Tsirelson bound", "#111111", ":"),
]


def plot_qber_time_series(records, metadata) -> Path:
    path = OUTPUT_DIR / "qber_time_series.png"
    times = [local_plot_time(timestamp) for _, timestamp, _ in records]
    qber = row_values(records, "QBER_total")

    first_time = records[0][1]
    last_time = records[-1][1]
    regions = merged_metadata_regions(metadata, first_time, last_time)

    figure, axis = plt.subplots(figsize=(14, 5))
    add_metadata_regions(axis, regions, annotate=True)
    axis.plot(
        times,
        qber,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#D62728",
        label="QBER total",
    )
    axis.axhline(0.50, color="#1F77B4", linestyle="--", linewidth=1.0, label="QBER 50%")
    axis.set_ylabel("QBER")
    axis.set_xlabel("Local time")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    add_legend_with_regions(axis, regions, loc="upper left", ncol=4)
    figure.suptitle("QBER")
    save_or_show(path)
    return path


def plot_visibility_time_series(records, metadata) -> Path:
    path = OUTPUT_DIR / "visibility_time_series.png"
    times = [local_plot_time(timestamp) for _, timestamp, _ in records]
    visibility = row_values(records, "visibility")
    vis_hv = row_values(records, "vis_HV")
    vis_da = row_values(records, "vis_DA")

    first_time = records[0][1]
    last_time = records[-1][1]
    regions = merged_metadata_regions(metadata, first_time, last_time)

    figure, axis = plt.subplots(figsize=(14, 5))
    add_metadata_regions(axis, regions, annotate=True)
    axis.plot(
        times,
        visibility,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#1F77B4",
        label="visibility total",
    )
    axis.plot(
        times,
        vis_hv,
        marker=".",
        linestyle="None",
        markersize=2,
        alpha=0.7,
        color="#2CA02C",
        label="visibility H/V",
    )
    axis.plot(
        times,
        vis_da,
        marker=".",
        linestyle="None",
        markersize=2,
        alpha=0.7,
        color="#FF7F0E",
        label="visibility D/A",
    )
    axis.set_ylabel("Visibility")
    axis.set_xlabel("Local time")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.25)
    add_legend_with_regions(axis, regions, loc="best", ncol=4)
    figure.suptitle("Visibility")
    save_or_show(path)
    return path


def plot_qber_by_window(records) -> Path:
    path = OUTPUT_DIR / "qber_by_window.png"
    groups = grouped_by_window(records)
    labels = sorted(groups)
    qber_data = [
        finite_values(row_values(groups[label], "QBER_total")) for label in labels
    ]

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(
        qber_data,
        tick_labels=labels,
        showfliers=False,
        boxprops={"color": "#1F77B4"},
        medianprops={"color": "#D62728", "linewidth": 1.4},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
    )
    axis.set_ylabel("QBER")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    axis.legend(
        handles=boxplot_legend_handles() + reference_line_handles(CHSH_REFERENCE_LINES),
        loc="upper right",
        fontsize=8,
    )
    axis.set_title("QBER by Coincidence Window")
    axis.text(
        0.01,
        0.98,
        "25th to 75th percentile (IQR). The line is the median, and whiskers extend to 1.5 x IQR.",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#444444",
    )
    save_or_show(path)
    return path


def plot_visibility_by_window(records) -> Path:
    path = OUTPUT_DIR / "visibility_by_window.png"
    groups = grouped_by_window(records)
    labels = sorted(groups)
    visibility_data = [
        finite_values(row_values(groups[label], "visibility")) for label in labels
    ]

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(
        visibility_data,
        tick_labels=labels,
        showfliers=False,
        boxprops={"color": "#1F77B4"},
        medianprops={"color": "#2CA02C", "linewidth": 1.4},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
    )
    axis.set_ylabel("Visibility")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    axis.legend(handles=boxplot_legend_handles(), loc="lower right", fontsize=8)
    axis.set_title("Visibility by Coincidence Window")
    axis.text(
        0.01,
        0.98,
        "25th to 75th percentile (IQR). The line is the median, and whiskers extend to 1.5 x IQR.",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#444444",
    )
    save_or_show(path)
    return path


def plot_qber_stability_monitor(records) -> Path | None:
    stability_records = [
        record for record in records if metadata_mode(record[2]) == "stability_monitor"
    ]
    if not stability_records:
        return None

    path = OUTPUT_DIR / "qber_stability_monitor.png"
    t0 = stability_records[0][1]
    hours = [hours_between(t0, timestamp) for _, timestamp, _ in stability_records]
    qber = row_values(stability_records, "QBER_total")

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        hours,
        qber,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#D62728",
        label="QBER total",
    )
    axis.axhline(0.50, color="#1F77B4", linestyle="--", linewidth=1.0, label="QBER 50%")
    axis.set_xlabel("Hours since stability-monitor phase start")
    axis.set_ylabel("QBER")
    axis.set_ylim(0.0, 1.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    axis.set_title("QBER During Stability Monitor")
    save_or_show(path)
    return path


def plot_visibility_stability_monitor(records) -> Path | None:
    stability_records = [
        record for record in records if metadata_mode(record[2]) == "stability_monitor"
    ]
    if not stability_records:
        return None

    path = OUTPUT_DIR / "visibility_stability_monitor.png"
    t0 = stability_records[0][1]
    hours = [hours_between(t0, timestamp) for _, timestamp, _ in stability_records]
    visibility = row_values(stability_records, "visibility")

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        hours,
        visibility,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#1F77B4",
        label="visibility total",
    )
    axis.set_xlabel("Hours since stability-monitor phase start")
    axis.set_ylabel("Visibility")
    axis.set_ylim(-0.05, 1.05)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    axis.set_title("Visibility During Stability Monitor")
    save_or_show(path)
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


def plot_s_time_series(records, metadata) -> Path:
    path = OUTPUT_DIR / "chsh_s_time_series.png"
    times = [local_plot_time(timestamp) for _, timestamp, _ in records]
    s_values = row_values(records, "CHSH_S_value")
    counts = row_values(records, "total_coincidences")

    first_time = records[0][1]
    last_time = records[-1][1]
    regions = merged_metadata_regions(metadata, first_time, last_time)

    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for index, axis in enumerate(axes):
        add_metadata_regions(axis, regions, annotate=index == 0)

    axes[0].plot(
        times,
        s_values,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#6A3D9A",
        label="CHSH S",
    )
    for value, label, color, style in CHSH_REFERENCE_LINES:
        axes[0].axhline(value, color=color, linestyle=style, linewidth=1.0, label=label)
    axes[0].set_ylabel("CHSH S")
    axes[0].set_ylim(0.0, 3.0)
    axes[0].grid(True, alpha=0.25)
    add_legend_with_regions(axes[0], regions, loc="upper left", ncol=4)

    axes[1].plot(
        times,
        counts,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#111111",
        label="total coincidences",
    )
    axes[1].set_ylabel("Total coincidences")
    axes[1].set_xlabel("Local time")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    figure.suptitle("CHSH S")
    save_or_show(path)
    return path


def plot_by_window(records) -> Path:
    path = OUTPUT_DIR / "chsh_s_by_window.png"
    groups = grouped_by_window(records)
    labels = sorted(groups)
    data = [
        finite_values(row_values(groups[label], "CHSH_S_value")) for label in labels
    ]

    figure, axis = plt.subplots(figsize=(11, 5))
    axis.boxplot(
        data,
        tick_labels=labels,
        showfliers=False,
        boxprops={"color": "#6A3D9A"},
        medianprops={"color": "#D62728", "linewidth": 1.4},
        whiskerprops={"color": "#333333"},
        capprops={"color": "#333333"},
    )
    for value, label, color, style in CHSH_REFERENCE_LINES:
        axis.axhline(value, color=color, linestyle=style, linewidth=1.0, label=label)
    axis.set_ylabel("CHSH S")
    axis.grid(True, alpha=0.25)
    axis.tick_params(axis="x", rotation=25)
    axis.legend(handles=boxplot_legend_handles(), loc="upper right", fontsize=8)
    axis.set_title("CHSH S by Coincidence Window")
    axis.text(
        0.01,
        0.98,
        "25th to 75th percentile (IQR). The line is the median, and whiskers extend to 1.5 x IQR.",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="#444444",
    )
    save_or_show(path)
    return path


def plot_stability_monitor(records) -> Path | None:
    stability_records = [
        record for record in records if metadata_mode(record[2]) == "stability_monitor"
    ]
    if not stability_records:
        return None

    path = OUTPUT_DIR / "chsh_s_stability_monitor.png"
    t0 = stability_records[0][1]
    hours = [hours_between(t0, timestamp) for _, timestamp, _ in stability_records]
    s_values = row_values(stability_records, "CHSH_S_value")

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(
        hours,
        s_values,
        marker=".",
        linestyle="None",
        markersize=3,
        color="#6A3D9A",
        label="CHSH S",
    )
    axis.axhline(2.0, color="#1F77B4", linestyle="--", linewidth=1.0, label="S = 2")
    axis.axhline(
        1.8,
        color="#D62728",
        linestyle="--",
        linewidth=1.0,
        label="S = 1.8 monitor threshold",
    )
    axis.set_xlabel("Hours since stability-monitor phase start")
    axis.set_ylabel("CHSH S")
    axis.set_ylim(0.0, 3.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    top_axis = axis.twiny()
    top_axis.set_xlim(axis.get_xlim())
    tick_hours = axis.get_xticks()
    top_axis.set_xticks(tick_hours)
    top_axis.set_xticklabels(
        [
            (t0 + dt.timedelta(hours=float(value)))
            .astimezone(LOCAL_TZ)
            .strftime("%H:%M")
            for value in tick_hours
        ]
    )
    top_axis.set_xlabel("Local time")
    axis.set_title("CHSH S During Stability Monitor")
    save_or_show(path)
    return path


def main() -> None:
    records, metadata = chsh_records()
    if not records:
        raise RuntimeError("No valid CHSH rows found for ZeroThird")

    summary_path = write_summary(records, metadata)
    print(f"Saved summary: {summary_path}")
    plot_s_time_series(records, metadata)
    plot_by_window(records)
    plot_stability_monitor(records)


if __name__ == "__main__":
    main()
