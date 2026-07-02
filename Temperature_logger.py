from __future__ import annotations

import csv
import datetime as dt
import json
import re
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
MIN_ARSO_SAMPLE_INTERVAL_SECONDS = 60.0
BACKFILL_WITH_OPEN_METEO = True

TIMEZONE = "Europe/Ljubljana"

ARSO_URL = (
    "https://meteo.arso.gov.si/uploads/probase/www/observ/surface/text/sl/"
    "observationAms_si_latest.html"
)

ARSO_STATIONS = {
    "Ljubljana": "Ljubljana - Vič",
    "Drnovo": "Letališče Cerklje ob Krki",
}

OPEN_METEO_LOCATIONS = {
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
    if timestamps.empty:
        return None

    return float(timestamps.iloc[-1])


def read_alice_time_span(path: Path) -> tuple[float, float] | None:
    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path, usecols=["timestamp"])
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None

    timestamps = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
    if timestamps.empty:
        return None

    return float(timestamps.min()), float(timestamps.max())


def parse_arso_datetime(text: str) -> pd.Timestamp:
    match = re.search(
        r"(\d{2})\.(\d{2})\.(\d{4})\s+(\d{2}):(\d{2})",
        text,
    )
    if match is None:
        return pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)

    day, month, year, hour, minute = map(int, match.groups())
    return pd.Timestamp(dt.datetime(year, month, day, hour, minute))


def fetch_arso_latest() -> dict[str, object]:
    tables = pd.read_html(ARSO_URL)
    if not tables:
        raise RuntimeError("ARSO latest-observation page contained no tables")

    table = tables[0]
    first_col = table.columns[0]

    temperatures: dict[str, float] = {}
    station_names: dict[str, str] = {}
    station_time = pd.Timestamp.now(tz=TIMEZONE).tz_localize(None)

    for _, row in table.iterrows():
        row_text = " ".join(str(v) for v in row.values if pd.notna(v))

        if "CEST" in row_text or "CET" in row_text:
            station_time = parse_arso_datetime(row_text)
            continue

        station_text = str(row[first_col])

        for label, station in ARSO_STATIONS.items():
            if station.lower() not in station_text.lower():
                continue

            values = []
            for value in row.values:
                number = pd.to_numeric(value, errors="coerce")
                if pd.notna(number):
                    values.append(float(number))

            if not values:
                continue

            temperatures[label] = values[0]
            station_names[label] = station

    missing = [name for name in ARSO_STATIONS if name not in temperatures]
    if missing:
        raise RuntimeError(f"Missing ARSO station temperatures: {missing}")

    return {
        "source_timestamp": station_time,
        "T_Ljubljana": temperatures["Ljubljana"],
        "T_Drnovo": temperatures["Drnovo"],
        "station_Ljubljana": station_names["Ljubljana"],
        "station_Drnovo": station_names["Drnovo"],
    }


def append_temperature_row(alice_timestamp: float | None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    arso = fetch_arso_latest()
    logger_timestamp = time.time()

    row = {
        "logger_timestamp": logger_timestamp,
        "logger_datetime": dt.datetime.fromtimestamp(logger_timestamp).isoformat(timespec="seconds"),
        "alice_timestamp": alice_timestamp,
        "alice_datetime": (
            dt.datetime.fromtimestamp(alice_timestamp).isoformat(timespec="seconds")
            if alice_timestamp is not None
            else ""
        ),
        "source": "ARSO_latest",
        "source_timestamp": pd.Timestamp(arso["source_timestamp"]).timestamp(),
        "source_datetime": pd.Timestamp(arso["source_timestamp"]).isoformat(),
        "T_Ljubljana": arso["T_Ljubljana"],
        "T_Drnovo": arso["T_Drnovo"],
        "T_delta": float(arso["T_Ljubljana"]) - float(arso["T_Drnovo"]),
        "station_Ljubljana": arso["station_Ljubljana"],
        "station_Drnovo": arso["station_Drnovo"],
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
        f"T_Drnovo={row['T_Drnovo']} °C | "
        f"alice_timestamp={alice_timestamp}"
    )


def fetch_open_meteo_hourly(
    name: str,
    lat: float,
    lon: float,
    start_date: dt.date,
    end_date: dt.date,
) -> pd.DataFrame:
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

    hourly = data["hourly"]

    return pd.DataFrame({
        "source_datetime": pd.to_datetime(hourly["time"]),
        f"T_{name}": hourly["temperature_2m"],
    })


def prepend_open_meteo_backfill() -> None:
    if not BACKFILL_WITH_OPEN_METEO:
        return

    if TEMPERATURE_LOG.is_file():
        existing = pd.read_csv(TEMPERATURE_LOG)
        if not existing.empty:
            return

    span = read_alice_time_span(ALICE_CSV)
    if span is None:
        return

    start_ts, end_ts = span
    start_date = dt.datetime.fromtimestamp(start_ts).date()
    end_date = min(dt.datetime.fromtimestamp(end_ts).date(), dt.date.today())

    frames = []
    for name, (lat, lon) in OPEN_METEO_LOCATIONS.items():
        frames.append(fetch_open_meteo_hourly(name, lat, lon, start_date, end_date))

    temp = frames[0]
    for frame in frames[1:]:
        temp = pd.merge(temp, frame, on="source_datetime", how="outer")

    temp = temp.sort_values("source_datetime")
    temp["T_delta"] = temp["T_Ljubljana"] - temp["T_Drnovo"]
    temp["source_timestamp"] = temp["source_datetime"].astype("int64") / 1e9
    temp["source"] = "OpenMeteo_hourly_backfill"

    temp["logger_timestamp"] = temp["source_timestamp"]
    temp["logger_datetime"] = temp["source_datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    temp["alice_timestamp"] = ""
    temp["alice_datetime"] = ""
    temp["station_Ljubljana"] = "Open-Meteo coordinate Ljubljana"
    temp["station_Drnovo"] = "Open-Meteo coordinate Drnovo"

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
        "station_Ljubljana",
        "station_Drnovo",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temp[columns].to_csv(TEMPERATURE_LOG, index=False)

    print(f"Prepended Open-Meteo hourly backfill: {len(temp)} rows")


def watch_alice_results() -> None:
    prepend_open_meteo_backfill()

    last_signature: tuple[int, int] | None = None
    last_arso_sample_time = 0.0

    print(f"Watching: {ALICE_CSV}")
    print(f"Writing:  {TEMPERATURE_LOG}")

    while True:
        try:
            if not ALICE_CSV.is_file():
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            stat = ALICE_CSV.stat()
            signature = (stat.st_mtime_ns, stat.st_size)

            enough_time_passed = (
                time.time() - last_arso_sample_time
                >= MIN_ARSO_SAMPLE_INTERVAL_SECONDS
            )

            if signature != last_signature and enough_time_passed:
                alice_timestamp = read_latest_alice_timestamp(ALICE_CSV)
                append_temperature_row(alice_timestamp)

                last_signature = signature
                last_arso_sample_time = time.time()

            time.sleep(POLL_INTERVAL_SECONDS)

        except KeyboardInterrupt:
            print("Stopped temperature logger")
            return
        except Exception as exc:
            print(f"Temperature logger skipped one update: {exc}")
            time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    watch_alice_results()