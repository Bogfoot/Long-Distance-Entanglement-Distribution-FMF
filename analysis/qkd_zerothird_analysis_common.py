from __future__ import annotations

import csv
import datetime as dt
import math
import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo


BASE_DIR = Path(__file__).resolve().parents[1]
GOOD_RESULTS_DIR = BASE_DIR / "Data" / "GoodResults"
ZEROTHIRD_CSV = GOOD_RESULTS_DIR / "alice_results_good_ZeroThird_since_20260709.csv"
METADATA_CSV = GOOD_RESULTS_DIR / "zerothird_run_metadata.csv"
QBER_ITERLOG_CSV = BASE_DIR / "Data" / "qber_iterlog.csv"
TEMP_MERGED_CSV = BASE_DIR / "Data" / "temperature_analysis" / "merged_measurements_temperature.csv"
LOCAL_TZ = ZoneInfo("Europe/Ljubljana")

RUN_TIMESTAMP_PATTERN = re.compile(
    r"^alice_.*?(\d{8}T\d{6}(?:\.\d+)?Z)_exp_.*\.bin$"
)

QKD_METRIC_COLUMNS = ("visibility", "vis_HV", "vis_DA", "QBER_total")
CHSH_PAIR_LABELS = (
    "HH",
    "HV",
    "VH",
    "VV",
    "HA",
    "HD",
    "VA",
    "VD",
    "DH",
    "DV",
    "AH",
    "AV",
    "DD",
    "DA",
    "AD",
    "AA",
)
CHSH_EXPECTATION_COLUMNS = (
    "CHSH_E_ab",
    "CHSH_E_abp",
    "CHSH_E_apb",
    "CHSH_E_apbp",
)

ALICE_VISIBILITY_TARGET = 0.85
ALICE_QBER_TARGET = (1.0 - ALICE_VISIBILITY_TARGET) / 2.0
ALICE_CHSH_TARGET = 2.4


@dataclass(frozen=True)
class RunMetadata:
    start_utc: dt.datetime
    start_local: dt.datetime
    experiment: str
    window_ps: float | None
    sigma: str
    mode: str
    notes: str

    @property
    def window_label(self) -> str:
        if self.window_ps is None:
            return "unknown"
        sigma_text = f", {self.sigma} sigma" if self.sigma else ""
        return f"{self.window_ps:g} ps{sigma_text}"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def temperature_rows_for_metric(metric_column: str) -> list[dict[str, str]]:
    if not TEMP_MERGED_CSV.exists():
        return []
    rows = read_csv_rows(TEMP_MERGED_CSV)
    return [row for row in rows if row.get(metric_column, "").strip()]


def optimizer_phase_markers() -> list[tuple[dt.datetime, str, str, str]]:
    if not QBER_ITERLOG_CSV.exists():
        return []
    rows = read_csv_rows(QBER_ITERLOG_CSV)
    phase_times: dict[str, list[dt.datetime]] = {"qber": [], "chsh": []}
    for row in rows:
        phase = (row.get("phase") or "").strip().lower()
        if phase not in phase_times:
            continue
        timestamp = row.get("timestamp", "").strip()
        if not timestamp:
            continue
        try:
            value = dt.datetime.fromtimestamp(float(timestamp), dt.timezone.utc)
        except ValueError:
            continue
        phase_times[phase].append(value)

    markers: list[tuple[dt.datetime, str, str, str]] = []
    styles = {"qber": ("#9467BD", "--"), "chsh": ("#1F77B4", "-")}
    for phase, pretty in (("qber", "QBER optimization"), ("chsh", "CHSH optimization")):
        times = phase_times[phase]
        if not times:
            continue
        color, start_style = styles[phase]
        markers.append((min(times), f"{pretty} started", color, start_style))
        markers.append((max(times), f"{pretty} ended", color, ":"))
    return markers


def parse_iso_datetime(value: str) -> dt.datetime:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = dt.datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_filename_timestamp(filename: str) -> dt.datetime | None:
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


def row_run_timestamp(row: dict[str, str]) -> dt.datetime | None:
    if row.get("run_timestamp_utc"):
        return parse_iso_datetime(row["run_timestamp_utc"])
    if row.get("alice_file"):
        parsed = parse_filename_timestamp(row["alice_file"])
        if parsed is not None:
            return parsed
    if row.get("timestamp"):
        try:
            return dt.datetime.fromtimestamp(
                float(row["timestamp"]),
                dt.timezone.utc,
            )
        except (TypeError, ValueError):
            return None
    return None


def finite_float(row: dict[str, str], column: str) -> bool:
    try:
        return math.isfinite(float(row.get(column, "")))
    except (TypeError, ValueError):
        return False


def float_value(row: dict[str, str], column: str, default: float = math.nan) -> float:
    try:
        value = float(row.get(column, ""))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def is_qkd_row(row: dict[str, str]) -> bool:
    return any(finite_float(row, column) for column in QKD_METRIC_COLUMNS)


def has_all_chsh_counts(row: dict[str, str]) -> bool:
    return all(finite_float(row, f"C_{label}") for label in CHSH_PAIR_LABELS)


def is_chsh_row(row: dict[str, str]) -> bool:
    return finite_float(row, "CHSH_S_value") or has_all_chsh_counts(row)


def filename_kind(row: dict[str, str]) -> str:
    name = row.get("alice_file", "")
    if "CHSH_S" in name:
        return "CHSH_S filename"
    if "QKD" in name:
        return "QKD filename"
    return "plain timestamp filename"


def load_metadata(path: Path = METADATA_CSV) -> list[RunMetadata]:
    rows = read_csv_rows(path)
    metadata: list[RunMetadata] = []
    for row in rows:
        start_utc = parse_iso_datetime(row["run_start_utc"])
        start_local = parse_iso_datetime(row["run_start_local"]).astimezone(LOCAL_TZ)
        window_ps = None
        if row.get("window_ps"):
            window_ps = float(row["window_ps"])
        metadata.append(
            RunMetadata(
                start_utc=start_utc,
                start_local=start_local,
                experiment=row.get("experiment", ""),
                window_ps=window_ps,
                sigma=row.get("sigma", ""),
                mode=row.get("mode", ""),
                notes=row.get("notes", ""),
            )
        )
    return sorted(metadata, key=lambda item: item.start_utc)


def metadata_for_time(
    timestamp_utc: dt.datetime,
    metadata: list[RunMetadata],
) -> RunMetadata | None:
    selected: RunMetadata | None = None
    for item in metadata:
        if item.start_utc <= timestamp_utc:
            selected = item
        else:
            break
    return selected


def rows_with_metadata(
    rows: list[dict[str, str]],
    metadata: list[RunMetadata],
) -> list[tuple[dict[str, str], dt.datetime, RunMetadata | None]]:
    tagged = []
    for row in rows:
        timestamp = row_run_timestamp(row)
        if timestamp is None:
            continue
        tagged.append((row, timestamp, metadata_for_time(timestamp, metadata)))
    return tagged


def local_time_text(timestamp_utc: dt.datetime) -> str:
    return timestamp_utc.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")


def hours_between(start_utc: dt.datetime, end_utc: dt.datetime) -> float:
    return (end_utc - start_utc).total_seconds() / 3600.0


def finite_values(values: list[float]) -> list[float]:
    return [value for value in values if math.isfinite(value)]


def mean(values: list[float]) -> float:
    clean = finite_values(values)
    if not clean:
        return math.nan
    return math.fsum(clean) / len(clean)


def stdev(values: list[float]) -> float:
    clean = finite_values(values)
    if len(clean) < 2:
        return math.nan
    avg = mean(clean)
    variance = math.fsum((value - avg) ** 2 for value in clean) / (len(clean) - 1)
    return math.sqrt(variance)


def standard_error(values: list[float]) -> float:
    clean = finite_values(values)
    if len(clean) < 2:
        return math.nan
    return stdev(clean) / math.sqrt(len(clean))


def median(values: list[float]) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return math.nan
    midpoint = len(clean) // 2
    if len(clean) % 2:
        return clean[midpoint]
    return 0.5 * (clean[midpoint - 1] + clean[midpoint])


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(finite_values(values))
    if not clean:
        return math.nan
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * pct / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return clean[lower]
    fraction = rank - lower
    return clean[lower] * (1.0 - fraction) + clean[upper] * fraction


def format_float(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_percent(value: float, digits: int = 2) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"
