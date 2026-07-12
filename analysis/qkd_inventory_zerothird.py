from __future__ import annotations

import csv
import datetime as dt
import re
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo

from qkd_zerothird_analysis_common import (
    BASE_DIR,
    GOOD_RESULTS_DIR,
    LOCAL_TZ,
    METADATA_CSV,
    ZEROTHIRD_CSV,
    filename_kind,
    is_chsh_row,
    is_qkd_row,
    load_metadata,
    local_time_text,
    metadata_for_time,
    read_csv_rows,
    row_run_timestamp,
)


SOURCE_SWITCH_LOCAL = dt.datetime(
    2026,
    7,
    9,
    9,
    50,
    41,
    661521,
    tzinfo=ZoneInfo("Europe/Ljubljana"),
)
SOURCE_SWITCH_UTC = SOURCE_SWITCH_LOCAL.astimezone(dt.timezone.utc)

DELAY_SCAN_DIR = BASE_DIR / "Data" / "DelayScans"
OUTPUT_DIR = GOOD_RESULTS_DIR / "ZeroThirdInventory"

WRITE_RUN_CSV = True
WRITE_DELAY_SCAN_CSV = True
WRITE_ROW_MAP_CSV = True

ALICE_FILE_TIMESTAMP_PATTERN = re.compile(
    r"^alice_(?P<prefix>.*?)(?P<timestamp>\d{8}T\d{6}(?:\.\d+)?Z)"
    r"_exp_.*\.bin$"
)
DELAY_SCAN_TIMESTAMP_PATTERN = re.compile(
    r"^initial_delay_scans_(?P<record_id>.+?)_"
    r"(?P<timestamp>\d{8}T\d{6}(?:\.\d+)?Z)\.png$"
)


def parse_compact_utc_timestamp(timestamp_text: str) -> dt.datetime:
    for fmt in ("%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return dt.datetime.strptime(timestamp_text, fmt).replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            pass
    raise ValueError(f"Could not parse UTC timestamp: {timestamp_text}")


def parse_delay_scan_path(path: Path) -> tuple[str, dt.datetime] | None:
    match = DELAY_SCAN_TIMESTAMP_PATTERN.match(path.name)
    if match is None:
        return None
    return (
        match.group("record_id"),
        parse_compact_utc_timestamp(match.group("timestamp")),
    )


def record_id_from_alice_file(alice_file: str) -> str:
    match = ALICE_FILE_TIMESTAMP_PATTERN.match(Path(alice_file).name)
    if match is None:
        return ""
    prefix = match.group("prefix")
    if prefix.endswith("_"):
        return prefix[:-1]
    return prefix


def int_from_row(row: dict[str, str], column: str) -> int | None:
    try:
        return int(row.get(column, ""))
    except (TypeError, ValueError):
        return None


def sorted_ints(values) -> list[int]:
    return sorted(value for value in values if value is not None)


def int_range_text(values) -> str:
    clean = sorted_ints(values)
    if not clean:
        return ""
    if len(clean) == 1:
        return str(clean[0])
    if clean == list(range(clean[0], clean[-1] + 1)):
        return f"{clean[0]}-{clean[-1]}"
    return ";".join(str(value) for value in clean)


def first_int_text(values) -> str:
    clean = sorted_ints(values)
    return str(clean[0]) if clean else ""


def last_int_text(values) -> str:
    clean = sorted_ints(values)
    return str(clean[-1]) if clean else ""


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def zero_third_rows() -> list[tuple[dict[str, str], dt.datetime]]:
    rows: list[tuple[dict[str, str], dt.datetime]] = []
    for row in read_csv_rows(ZEROTHIRD_CSV):
        timestamp = row_run_timestamp(row)
        if timestamp is None:
            continue
        if timestamp >= SOURCE_SWITCH_UTC:
            rows.append((row, timestamp))
    rows.sort(key=lambda item: item[1])
    return rows


def row_map_records(
    rows: list[tuple[dict[str, str], dt.datetime]]
) -> list[dict[str, str]]:
    metadata = load_metadata(METADATA_CSV)
    records = []
    for row, timestamp in rows:
        meta = metadata_for_time(timestamp, metadata)
        alice_file = row.get("alice_file", "")
        records.append(
            {
                "run_timestamp_utc": timestamp.isoformat(),
                "run_timestamp_local": timestamp.astimezone(LOCAL_TZ).isoformat(),
                "record_id": record_id_from_alice_file(alice_file),
                "alice_file": alice_file,
                "filename_kind": filename_kind(row),
                "result_kind": row.get("result_kind", ""),
                "is_qkd_row": str(is_qkd_row(row)),
                "is_chsh_row": str(is_chsh_row(row)),
                "curated_window": meta.window_label if meta is not None else "unknown",
                "curated_mode": meta.mode if meta is not None else "unknown",
                "alice_results_csv": "Data/alice_results.csv",
                "alice_results_data_row": row.get("alice_results_data_row", ""),
                "alice_results_line_number": row.get("alice_results_line_number", ""),
                "good_results_combined_csv": (
                    "Data/GoodResults/alice_results_good_since_20260628.csv"
                ),
                "good_results_combined_data_row": row.get(
                    "good_results_combined_data_row",
                    "",
                ),
                "zerothird_results_csv": relative_path(ZEROTHIRD_CSV),
                "zerothird_results_data_row": row.get("source_results_data_row", ""),
            }
        )
    return [
        {"zerothird_row_map_data_row": str(index), **record}
        for index, record in enumerate(records, start=1)
    ]


def unique_run_records(
    rows: list[tuple[dict[str, str], dt.datetime]]
) -> list[dict[str, str]]:
    metadata = load_metadata(METADATA_CSV)
    by_file: dict[str, dict[str, object]] = {}
    for row, timestamp in rows:
        alice_file = row.get("alice_file", "")
        if not alice_file:
            continue
        meta = metadata_for_time(timestamp, metadata)
        entry = by_file.setdefault(
            alice_file,
            {
                "alice_file": alice_file,
                "record_id": record_id_from_alice_file(alice_file),
                "timestamp": timestamp,
                "rows": 0,
                "qkd_rows": 0,
                "chsh_rows": 0,
                "filename_kind": filename_kind(row),
                "curated_window": meta.window_label if meta is not None else "unknown",
                "curated_mode": meta.mode if meta is not None else "unknown",
                "alice_results_data_rows": [],
                "alice_results_line_numbers": [],
                "good_results_combined_data_rows": [],
                "zerothird_results_data_rows": [],
            },
        )
        entry["rows"] = int(entry["rows"]) + 1
        if is_qkd_row(row):
            entry["qkd_rows"] = int(entry["qkd_rows"]) + 1
        if is_chsh_row(row):
            entry["chsh_rows"] = int(entry["chsh_rows"]) + 1
        entry["alice_results_data_rows"].append(
            int_from_row(row, "alice_results_data_row")
        )
        entry["alice_results_line_numbers"].append(
            int_from_row(row, "alice_results_line_number")
        )
        entry["good_results_combined_data_rows"].append(
            int_from_row(row, "good_results_combined_data_row")
        )
        entry["zerothird_results_data_rows"].append(
            int_from_row(row, "source_results_data_row")
        )

    records = []
    for entry in by_file.values():
        timestamp = entry["timestamp"]
        assert isinstance(timestamp, dt.datetime)
        records.append(
            {
                "run_timestamp_utc": timestamp.isoformat(),
                "run_timestamp_local": timestamp.astimezone(LOCAL_TZ).isoformat(),
                "record_id": str(entry["record_id"]),
                "alice_file": str(entry["alice_file"]),
                "filename_kind": str(entry["filename_kind"]),
                "curated_window": str(entry["curated_window"]),
                "curated_mode": str(entry["curated_mode"]),
                "rows": str(entry["rows"]),
                "qkd_rows": str(entry["qkd_rows"]),
                "chsh_rows": str(entry["chsh_rows"]),
                "alice_results_csv": "Data/alice_results.csv",
                "alice_results_data_rows": int_range_text(
                    entry["alice_results_data_rows"]
                ),
                "alice_results_line_numbers": int_range_text(
                    entry["alice_results_line_numbers"]
                ),
                "first_alice_results_line_number": first_int_text(
                    entry["alice_results_line_numbers"]
                ),
                "last_alice_results_line_number": last_int_text(
                    entry["alice_results_line_numbers"]
                ),
                "good_results_combined_csv": (
                    "Data/GoodResults/alice_results_good_since_20260628.csv"
                ),
                "good_results_combined_data_rows": int_range_text(
                    entry["good_results_combined_data_rows"]
                ),
                "zerothird_results_csv": relative_path(ZEROTHIRD_CSV),
                "zerothird_results_data_rows": int_range_text(
                    entry["zerothird_results_data_rows"]
                ),
                "first_zerothird_results_data_row": first_int_text(
                    entry["zerothird_results_data_rows"]
                ),
                "last_zerothird_results_data_row": last_int_text(
                    entry["zerothird_results_data_rows"]
                ),
            }
        )
    records.sort(key=lambda item: item["run_timestamp_utc"])
    return [
        {"zerothird_unique_run_data_row": str(index), **record}
        for index, record in enumerate(records, start=1)
    ]


def run_lookups(
    run_records: list[dict[str, str]]
) -> tuple[
    dict[tuple[str, str], list[dict[str, str]]],
    dict[str, list[dict[str, str]]],
]:
    by_record_and_time: dict[tuple[str, str], list[dict[str, str]]] = {}
    by_time: dict[str, list[dict[str, str]]] = {}
    for record in run_records:
        timestamp = record["run_timestamp_utc"]
        by_time.setdefault(timestamp, []).append(record)
        key = (record["record_id"], timestamp)
        by_record_and_time.setdefault(key, []).append(record)
    return by_record_and_time, by_time


def zero_third_delay_scans(
    run_records: list[dict[str, str]],
) -> list[dict[str, str]]:
    metadata = load_metadata(METADATA_CSV)
    runs_by_record_and_time, runs_by_time = run_lookups(run_records)
    records: list[dict[str, str]] = []
    for path in sorted(DELAY_SCAN_DIR.glob("initial_delay_scans_*.png")):
        parsed = parse_delay_scan_path(path)
        if parsed is None:
            continue
        record_id, timestamp = parsed
        if timestamp < SOURCE_SWITCH_UTC:
            continue
        meta = metadata_for_time(timestamp, metadata)
        matching_key_type = "record_id_and_timestamp"
        matching_runs = runs_by_record_and_time.get((record_id, timestamp.isoformat()), [])
        if not matching_runs:
            matching_key_type = "timestamp_only"
            matching_runs = runs_by_time.get(timestamp.isoformat(), [])
        if not matching_runs:
            matching_key_type = "none"
        first_match = matching_runs[0] if matching_runs else {}
        records.append(
            {
                "scan_timestamp_utc": timestamp.isoformat(),
                "scan_timestamp_local": timestamp.astimezone(LOCAL_TZ).isoformat(),
                "record_id": record_id,
                "curated_window": meta.window_label if meta is not None else "unknown",
                "curated_mode": meta.mode if meta is not None else "unknown",
                "curated_experiment": (
                    meta.experiment if meta is not None else "unknown"
                ),
                "matching_key_type": matching_key_type,
                "matching_run_count": str(len(matching_runs)),
                "matching_unique_run_data_row": first_match.get(
                    "zerothird_unique_run_data_row",
                    "",
                ),
                "matching_alice_file": first_match.get("alice_file", ""),
                "matching_alice_results_line_numbers": first_match.get(
                    "alice_results_line_numbers",
                    "",
                ),
                "matching_zerothird_results_data_rows": first_match.get(
                    "zerothird_results_data_rows",
                    "",
                ),
                "path": relative_path(path),
            }
        )
    records.sort(key=lambda item: item["scan_timestamp_utc"])
    return records


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def per_day_counts(records: list[dict[str, str]], timestamp_column: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        timestamp = dt.datetime.fromisoformat(record[timestamp_column])
        counts[timestamp.astimezone(LOCAL_TZ).strftime("%Y-%m-%d")] += 1
    return counts


def largest_gaps(
    records: list[dict[str, str]],
    timestamp_column: str,
    label_column: str,
    limit: int = 8,
) -> list[str]:
    if len(records) < 2:
        return []

    parsed = [
        (
            dt.datetime.fromisoformat(record[timestamp_column]).astimezone(
                dt.timezone.utc
            ),
            record[label_column],
        )
        for record in records
    ]
    gaps = []
    for previous, current in zip(parsed, parsed[1:]):
        previous_time, previous_label = previous
        current_time, current_label = current
        gap_hours = (current_time - previous_time).total_seconds() / 3600.0
        gaps.append((gap_hours, previous_time, current_time, previous_label, current_label))

    lines = []
    for gap_hours, previous_time, current_time, previous_label, current_label in sorted(
        gaps,
        reverse=True,
    )[:limit]:
        lines.append(
            f"{gap_hours:.2f} h: {local_time_text(previous_time)} -> "
            f"{local_time_text(current_time)} | {previous_label} -> {current_label}"
        )
    return lines


def append_counter_lines(lines: list[str], title: str, counts: Counter[str]) -> None:
    lines.append(title)
    if not counts:
        lines.append("  none")
        return
    for key, count in sorted(counts.items()):
        lines.append(f"  {key}: {count}")


def write_summary(
    run_records: list[dict[str, str]],
    delay_scan_records: list[dict[str, str]],
) -> Path:
    metadata = load_metadata(METADATA_CSV)
    path = OUTPUT_DIR / "zerothird_inventory_summary.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_filename_kinds = Counter(record["filename_kind"] for record in run_records)
    run_days = per_day_counts(run_records, "run_timestamp_utc")
    delay_scan_record_ids = Counter(
        record["record_id"] for record in delay_scan_records
    )
    delay_scan_windows = Counter(
        record["curated_window"] for record in delay_scan_records
    )
    delay_scan_days = per_day_counts(delay_scan_records, "scan_timestamp_utc")
    delay_scan_match_types = Counter(
        record["matching_key_type"] for record in delay_scan_records
    )

    lines = [
        "ZeroThird Date / File Inventory",
        "",
        f"Source switch local: {SOURCE_SWITCH_LOCAL.isoformat()}",
        f"Source switch UTC: {SOURCE_SWITCH_UTC.isoformat()}",
        f"Input ZeroThird CSV: {ZEROTHIRD_CSV}",
        f"Delay scan directory: {DELAY_SCAN_DIR}",
        f"Metadata CSV: {METADATA_CSV}",
        "",
        f"Unique ZeroThird Alice run files: {len(run_records)}",
    ]
    if run_records:
        first_run = dt.datetime.fromisoformat(run_records[0]["run_timestamp_utc"])
        last_run = dt.datetime.fromisoformat(run_records[-1]["run_timestamp_utc"])
        lines.extend(
            [
                f"First ZeroThird run file: {local_time_text(first_run)}",
                f"Last ZeroThird run file: {local_time_text(last_run)}",
            ]
        )
    lines.append("")
    append_counter_lines(lines, "Unique Alice run files by filename kind:", run_filename_kinds)
    lines.append("")
    append_counter_lines(lines, "Unique Alice run files by local date:", run_days)
    lines.append("")
    lines.append("Largest ZeroThird run-file gaps:")
    lines.extend(
        largest_gaps(run_records, "run_timestamp_utc", "alice_file") or ["  none"]
    )

    lines.extend(
        [
            "",
            f"ZeroThird-dated delay-scan images: {len(delay_scan_records)}",
        ]
    )
    if delay_scan_records:
        first_scan = dt.datetime.fromisoformat(
            delay_scan_records[0]["scan_timestamp_utc"]
        )
        last_scan = dt.datetime.fromisoformat(
            delay_scan_records[-1]["scan_timestamp_utc"]
        )
        matched = sum(
            1 for record in delay_scan_records if record["matching_key_type"] != "none"
        )
        lines.extend(
            [
                f"First ZeroThird-dated delay scan: {local_time_text(first_scan)}",
                f"Last ZeroThird-dated delay scan: {local_time_text(last_scan)}",
                f"Delay scans matched to ZeroThird Alice run files: {matched}",
            ]
        )
    lines.append("")
    append_counter_lines(lines, "Delay scans by record id:", delay_scan_record_ids)
    lines.append("")
    append_counter_lines(lines, "Delay scans by curated metadata window:", delay_scan_windows)
    lines.append("")
    append_counter_lines(lines, "Delay scans by local date:", delay_scan_days)
    lines.append("")
    append_counter_lines(lines, "Delay scan match type:", delay_scan_match_types)
    lines.append("")
    lines.append("Largest ZeroThird delay-scan gaps:")
    lines.extend(
        largest_gaps(delay_scan_records, "scan_timestamp_utc", "path") or ["  none"]
    )
    lines.append("")
    lines.append("Curated ZeroThird metadata phases:")
    for meta in metadata:
        lines.append(
            f"  {local_time_text(meta.start_utc)} | {meta.window_label} | "
            f"{meta.mode} | {meta.experiment} | {meta.notes}"
        )
    lines.append("")
    lines.append("Cross-reference files:")
    lines.append("  zerothird_results_row_map.csv: one row per ZeroThird result row")
    lines.append("  zerothird_unique_alice_runs.csv: one row per unique alice_file")
    lines.append("  zerothird_delay_scan_inventory.csv: one row per ZeroThird-dated delay scan")
    lines.append("")
    lines.append(
        "Note: delay-scan images are filtered by timestamp and cross-referenced to Alice result rows. "
        "The metadata CSV is the curated timeline derived from these image references."
    )
    lines.append("")

    path.write_text("\n".join(lines))
    return path


def main() -> None:
    rows = zero_third_rows()
    row_records = row_map_records(rows)
    run_records = unique_run_records(rows)
    delay_scan_records = zero_third_delay_scans(run_records)

    if WRITE_ROW_MAP_CSV:
        write_csv(OUTPUT_DIR / "zerothird_results_row_map.csv", row_records)
    if WRITE_RUN_CSV:
        write_csv(OUTPUT_DIR / "zerothird_unique_alice_runs.csv", run_records)
    if WRITE_DELAY_SCAN_CSV:
        write_csv(
            OUTPUT_DIR / "zerothird_delay_scan_inventory.csv",
            delay_scan_records,
        )
    summary_path = write_summary(run_records, delay_scan_records)

    print(f"ZeroThird result rows mapped: {len(row_records)}")
    print(f"Unique ZeroThird Alice run files: {len(run_records)}")
    print(f"ZeroThird-dated delay-scan images: {len(delay_scan_records)}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
