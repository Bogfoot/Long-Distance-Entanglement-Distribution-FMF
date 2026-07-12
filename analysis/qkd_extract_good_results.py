from __future__ import annotations

import csv
import datetime as dt
import re
from collections import Counter
from pathlib import Path
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parents[1]
SOURCE_CSV = BASE_DIR / "Data" / "alice_results.csv"
OUTPUT_DIR = BASE_DIR / "Data" / "GoodResults"

# Keep only the modern source-comparison data. Timestamps are taken from the
# alice_file filename, not from row order.
START_UTC = dt.datetime(2026, 6, 28, 0, 0, 0, tzinfo=dt.timezone.utc)
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

AUREA_SOURCE_NAME = "Aurea"
ZEROTHIRD_SOURCE_NAME = "ZeroThird"

COMBINED_OUTPUT_NAME = "alice_results_good_since_20260628.csv"
AUREA_OUTPUT_NAME = "alice_results_good_Aurea_20260628_to_20260709.csv"
ZEROTHIRD_OUTPUT_NAME = "alice_results_good_ZeroThird_since_20260709.csv"
SUMMARY_OUTPUT_NAME = "good_results_summary.txt"

RUN_TIMESTAMP_PATTERN = re.compile(
    r"^alice_.*?(\d{8}T\d{6}(?:\.\d+)?Z)_exp_.*\.bin$"
)


def parse_run_timestamp_from_alice_file(filename: str) -> dt.datetime | None:
    match = RUN_TIMESTAMP_PATTERN.match(Path(filename).name)
    if match is None:
        return None

    timestamp_text = match.group(1)
    for fmt in ("%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return dt.datetime.strptime(timestamp_text, fmt).replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            pass
    return None


def is_float(text: object) -> bool:
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def result_kind(row: dict[str, str]) -> str:
    has_qkd = any(
        is_float(row.get(name))
        for name in ("visibility", "vis_HV", "vis_DA", "QBER_total")
    )
    has_chsh = any(
        is_float(row.get(name))
        for name in ("CHSH_S_value", "CHSH_S_signed")
    )
    if has_qkd and has_chsh:
        return "QKD+CHSH"
    if has_chsh:
        return "CHSH"
    if has_qkd:
        return "QKD"
    return "other"


def source_name_for_timestamp(run_timestamp_utc: dt.datetime) -> str:
    switch_utc = SOURCE_SWITCH_LOCAL.astimezone(dt.timezone.utc)
    if run_timestamp_utc < switch_utc:
        return AUREA_SOURCE_NAME
    return ZEROTHIRD_SOURCE_NAME


def add_extraction_columns(
    row: dict[str, str],
    run_timestamp_utc: dt.datetime,
    alice_results_data_row: int,
) -> dict[str, str]:
    output = dict(row)
    output["run_timestamp_utc"] = run_timestamp_utc.isoformat()
    output["run_timestamp_local"] = run_timestamp_utc.astimezone(
        SOURCE_SWITCH_LOCAL.tzinfo
    ).isoformat()
    output["source_name"] = source_name_for_timestamp(run_timestamp_utc)
    output["result_kind"] = result_kind(row)
    output["alice_results_data_row"] = str(alice_results_data_row)
    output["alice_results_line_number"] = str(alice_results_data_row + 1)
    return output


def read_good_rows(path: Path) -> tuple[list[dict[str, str]], list[str], int]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        source_fieldnames = list(reader.fieldnames or [])
        good_rows: list[dict[str, str]] = []
        skipped_without_file_timestamp = 0

        for alice_results_data_row, row in enumerate(reader, start=1):
            run_timestamp = parse_run_timestamp_from_alice_file(
                row.get("alice_file", "")
            )
            if run_timestamp is None:
                skipped_without_file_timestamp += 1
                continue
            if run_timestamp < START_UTC:
                continue
            good_rows.append(
                add_extraction_columns(row, run_timestamp, alice_results_data_row)
            )

    fieldnames = [
        "run_timestamp_utc",
        "run_timestamp_local",
        "source_name",
        "result_kind",
        "alice_results_data_row",
        "alice_results_line_number",
        "good_results_combined_data_row",
        "source_results_data_row",
        *source_fieldnames,
    ]
    return good_rows, fieldnames, skipped_without_file_timestamp


def add_output_row_numbers(
    combined_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
) -> None:
    for index, row in enumerate(combined_rows, start=1):
        row["good_results_combined_data_row"] = str(index)
    for index, row in enumerate(source_rows, start=1):
        row["source_results_data_row"] = str(index)


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def timestamp_range(rows: list[dict[str, str]]) -> tuple[str, str]:
    if not rows:
        return "n/a", "n/a"
    return rows[0]["run_timestamp_utc"], rows[-1]["run_timestamp_utc"]


def unique_file_count(rows: list[dict[str, str]]) -> int:
    return len({row.get("alice_file", "") for row in rows if row.get("alice_file")})


def source_summary_line(label: str, rows: list[dict[str, str]]) -> str:
    first, last = timestamp_range(rows)
    kinds = Counter(row["result_kind"] for row in rows)
    kinds_text = ", ".join(f"{name}={count}" for name, count in sorted(kinds.items()))
    if not kinds_text:
        kinds_text = "none"
    return (
        f"{label}: rows={len(rows)}, files={unique_file_count(rows)}, "
        f"first={first}, last={last}, result_kinds=({kinds_text})"
    )


def largest_gaps(rows: list[dict[str, str]], min_gap_hours: float = 1.0) -> list[str]:
    if len(rows) < 2:
        return []

    parsed = [
        (
            dt.datetime.fromisoformat(row["run_timestamp_utc"]),
            row.get("alice_file", ""),
        )
        for row in rows
    ]
    gaps: list[tuple[float, dt.datetime, dt.datetime, str, str]] = []
    for previous, current in zip(parsed, parsed[1:]):
        previous_time, previous_file = previous
        current_time, current_file = current
        gap_hours = (current_time - previous_time).total_seconds() / 3600.0
        if gap_hours >= min_gap_hours:
            gaps.append(
                (
                    gap_hours,
                    previous_time,
                    current_time,
                    previous_file,
                    current_file,
                )
            )

    lines = []
    for gap_hours, previous_time, current_time, previous_file, current_file in sorted(
        gaps,
        reverse=True,
    )[:10]:
        lines.append(
            f"{gap_hours:.2f} h: {previous_time.isoformat()} -> "
            f"{current_time.isoformat()} | {previous_file} -> {current_file}"
        )
    return lines


def write_summary(
    path: Path,
    combined_rows: list[dict[str, str]],
    aurea_rows: list[dict[str, str]],
    zerothird_rows: list[dict[str, str]],
    skipped_without_file_timestamp: int,
) -> None:
    switch_utc = SOURCE_SWITCH_LOCAL.astimezone(dt.timezone.utc)
    lines = [
        "Good Results Extraction",
        "",
        f"Source CSV: {SOURCE_CSV}",
        f"Start UTC: {START_UTC.isoformat()}",
        f"Source switch local: {SOURCE_SWITCH_LOCAL.isoformat()}",
        f"Source switch UTC: {switch_utc.isoformat()}",
        f"Skipped rows without parseable alice_file timestamp: {skipped_without_file_timestamp}",
        "",
        source_summary_line("Combined", combined_rows),
        source_summary_line(AUREA_SOURCE_NAME, aurea_rows),
        source_summary_line(ZEROTHIRD_SOURCE_NAME, zerothird_rows),
        "",
        "Largest gaps >= 1 h:",
    ]
    gap_lines = largest_gaps(combined_rows)
    lines.extend(gap_lines if gap_lines else ["none"])
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    rows, fieldnames, skipped_without_file_timestamp = read_good_rows(SOURCE_CSV)
    aurea_rows = [
        row for row in rows if row["source_name"] == AUREA_SOURCE_NAME
    ]
    zerothird_rows = [
        row for row in rows if row["source_name"] == ZEROTHIRD_SOURCE_NAME
    ]
    add_output_row_numbers(rows, aurea_rows)
    add_output_row_numbers(rows, zerothird_rows)

    write_csv(OUTPUT_DIR / COMBINED_OUTPUT_NAME, rows, fieldnames)
    write_csv(OUTPUT_DIR / AUREA_OUTPUT_NAME, aurea_rows, fieldnames)
    write_csv(OUTPUT_DIR / ZEROTHIRD_OUTPUT_NAME, zerothird_rows, fieldnames)
    write_summary(
        OUTPUT_DIR / SUMMARY_OUTPUT_NAME,
        rows,
        aurea_rows,
        zerothird_rows,
        skipped_without_file_timestamp,
    )

    print(f"Wrote {len(rows)} combined rows to {OUTPUT_DIR / COMBINED_OUTPUT_NAME}")
    print(f"Wrote {len(aurea_rows)} Aurea rows to {OUTPUT_DIR / AUREA_OUTPUT_NAME}")
    print(
        f"Wrote {len(zerothird_rows)} ZeroThird rows to "
        f"{OUTPUT_DIR / ZEROTHIRD_OUTPUT_NAME}"
    )
    print(f"Wrote summary to {OUTPUT_DIR / SUMMARY_OUTPUT_NAME}")


if __name__ == "__main__":
    main()
