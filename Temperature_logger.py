from __future__ import annotations

import csv
import datetime as dt
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
ALICE_CSV = BASE_DIR / "Data" / "alice_results.csv"
OUT_DIR = BASE_DIR / "Data" / "temperature_analysis"
TEMPERATURE_LOG = OUT_DIR / "temperature_log.csv"

POLL_INTERVAL_SECONDS = 10.0
MIN_SAMPLE_INTERVAL_SECONDS = 60.0
BACKFILL_ON_START = True
TIMEZONE = "Europe/Ljubljana"

LOCATIONS = {
    "Ljubljana": (46.0569, 14.5058),
    "Drnovo": (45.9566, 15.4894),
}


def read_latest_alice_timestamp(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
    return None if timestamps.empty else float(timestamps.iloc[-1])


def read_alice_time_span(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None
    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
    return None if timestamps.empty else (float(timestamps.min()), float(timestamps.max()))


def fetch_open_meteo_current() -> dict[str, object]:
    rows = {}

    for name, (lat, lon) in LOCATIONS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m",
            "timezone": TIMEZONE,
        }
        url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)

        with urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        rows[f"T_{name}"] = float(data["current"]["temperature_2m"])
        rows["source_datetime"] = data["current"]["time"]

    rows["T_delta"] = rows["T_Ljubljana"] - rows["T_Drnovo"]
    return rows


def fetch_open_meteo_hourly(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    frames = []

    for name, (lat, lon) in LOCATIONS.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "temperature_2m",
            "timezone": TIMEZONE,
        }
        url = "https://archive-api.open-meteo.com/v1/archive?" + urlencode(params)

        with urlopen(url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))

        block = data["hourly"]
        frames.append(pd.DataFrame({
            "source_datetime": pd.to_datetime(block["time"]),
            f"T_{name}": block["temperature_2m"],
        }))

    temp = frames[0]
    for frame in frames[1:]:
        temp = pd.merge(temp, frame, on="source_datetime", how="outer")

    temp = temp.sort_values("source_datetime")
    temp["T_delta"] = temp["T_Ljubljana"] - temp["T_Drnovo"]
    temp["source"] = "OpenMeteo_hourly_backfill"
    return temp


def write_backfill() -> None:
    if not BACKFILL_ON_START:
        return

    if TEMPERATURE_LOG.is_file() and not pd.read_csv(TEMPERATURE_LOG).empty:
        return

    span = read_alice_time_span(ALICE_CSV)
    if span is None:
        return

    start_ts, end_ts = span
    start_date = dt.datetime.fromtimestamp(start_ts).date()
    end_date = min(dt.datetime.fromtimestamp(end_ts).date(), dt.date.today())

    temp = fetch_open_meteo_hourly(start_date, end_date)
    temp["source_timestamp"] = temp["source_datetime"].astype("int64") / 1e9
    temp["logger_timestamp"] = temp["source_timestamp"]
    temp["logger_datetime"] = temp["source_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    temp["alice_timestamp"] = ""
    temp["alice_datetime"] = ""

    columns = [
        "logger_timestamp",
        "logger_datetime",
        "alice_timestamp",
        "alice_datetime",
        "source",
        "source_timestamp",
        "source_datetime",
        "T_Ljubljana",
        "T_Drnovo",
        "T_delta",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp[columns].to_csv(TEMPERATURE_LOG, index=False)
    print(f"Wrote Open-Meteo hourly backfill: {len(temp)} rows")


def append_temperature_row(alice_timestamp: float | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current = fetch_open_meteo_current()
    logger_timestamp = time.time()

    row = {
        "logger_timestamp": logger_timestamp,
        "logger_datetime": dt.datetime.fromtimestamp(logger_timestamp).isoformat(timespec="seconds"),
        "alice_timestamp": alice_timestamp,
        "alice_datetime": (
            dt.datetime.fromtimestamp(alice_timestamp).isoformat(timespec="seconds")
            if alice_timestamp is not None else ""
        ),
        "source": "OpenMeteo_current",
        "source_timestamp": pd.Timestamp(current["source_datetime"]).timestamp(),
        "source_datetime": current["source_datetime"],
        "T_Ljubljana": current["T_Ljubljana"],
        "T_Drnovo": current["T_Drnovo"],
        "T_delta": current["T_delta"],
    }

    write_header = not TEMPERATURE_LOG.is_file()
    with TEMPERATURE_LOG.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(
        f"{row['logger_datetime']} | "
        f"T_Ljubljana={row['T_Ljubljana']} °C | "
        f"T_Drnovo={row['T_Drnovo']} °C"
    )


def watch_alice_results() -> None:
    write_backfill()

    last_signature = None
    last_sample_time = 0.0

    print(f"Watching: {ALICE_CSV}")
    print(f"Writing:  {TEMPERATURE_LOG}")

    while True:
        try:
            if not ALICE_CSV.is_file():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            stat = ALICE_CSV.stat()
            signature = (stat.st_mtime_ns, stat.st_size)

            if (
                signature != last_signature
                and time.time() - last_sample_time >= MIN_SAMPLE_INTERVAL_SECONDS
            ):
                append_temperature_row(read_latest_alice_timestamp(ALICE_CSV))
                last_signature = signature
                last_sample_time = time.time()

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Stopped temperature logger")
            return
        except Exception as exc:
            print(f"Temperature logger skipped one update: {exc}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    watch_alice_results()