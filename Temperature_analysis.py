from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


CSV_PATH = Path("Data/alice_results.csv")
OUT_DIR = Path("Data/temperature_analysis")

TIMEZONE = "Europe/Ljubljana"

LOCATIONS = {
    "Ljubljana": (46.0569, 14.5058),
    "Drnovo": (45.9566, 15.4894),
}

METRICS = [
    "visibility",
    "vis_HV",
    "vis_DA",
    "QBER_total",
    "QBER_HV",
    "QBER_DA",
    "CHSH_S_value",
]

MAX_LAG_MINUTES = 6 * 60
LAG_STEP_MINUTES = 5


def fetch_temperature(name: str, lat: float, lon: float, start: dt.date, end: dt.date) -> pd.DataFrame:
    days_back = max(1, (dt.date.today() - start).days + 1)

    params = {
        "latitude": lat,
        "longitude": lon,
        "past_days": min(days_back, 92),
        "forecast_days": 1,
        "hourly": "temperature_2m",
        "timezone": TIMEZONE,
    }

    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)

    with urlopen(url, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    if "hourly" not in data:
        raise RuntimeError(f"Open-Meteo response for {name} has no hourly data: {data}")

    hourly = data["hourly"]

    frame = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        f"T_{name}": hourly["temperature_2m"],
    })

    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) + pd.Timedelta(days=1)

    return frame[(frame["datetime"] >= start_dt) & (frame["datetime"] < end_dt)]


def read_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV needs a timestamp column")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("datetime")

    keep = ["datetime"] + [column for column in METRICS if column in df.columns]
    return df[keep].dropna(how="all", subset=keep[1:])


def merge_temperature(meas: pd.DataFrame) -> pd.DataFrame:
    start = meas["datetime"].min().date() - dt.timedelta(days=1)
    end = min(meas["datetime"].max().date() + dt.timedelta(days=1), dt.date.today())

    temp = None
    for name, (lat, lon) in LOCATIONS.items():
        one = fetch_temperature(name, lat, lon, start, end)
        temp = one if temp is None else pd.merge(temp, one, on="datetime", how="outer")

    temp = temp.sort_values("datetime").set_index("datetime")
    temp = temp.interpolate(method="time")

    meas = meas.set_index("datetime").sort_index()
    merged = pd.merge_asof(
        meas.reset_index(),
        temp.reset_index().sort_values("datetime"),
        on="datetime",
        direction="nearest",
        tolerance=pd.Timedelta("2h"),
    ).set_index("datetime")

    merged["T_delta"] = merged["T_Ljubljana"] - merged["T_Drnovo"]

    for col in ["T_Ljubljana", "T_Drnovo", "T_delta"]:
        seconds = merged.index.to_series().diff().dt.total_seconds()
        merged[f"d{col}_dt_C_per_h"] = merged[col].diff() / seconds * 3600.0

    return merged


def pearson_table(df: pd.DataFrame) -> pd.DataFrame:
    temp_cols = [
        "T_Ljubljana",
        "T_Drnovo",
        "T_delta",
        "dT_Ljubljana_dt_C_per_h",
        "dT_Drnovo_dt_C_per_h",
        "dT_delta_dt_C_per_h",
    ]

    rows = []
    for metric in [m for m in METRICS if m in df.columns]:
        for temp_col in temp_cols:
            valid = df[[metric, temp_col]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < 3:
                continue

            r, p = pearsonr(valid[temp_col], valid[metric])
            rows.append({
                "metric": metric,
                "temperature_variable": temp_col,
                "n": len(valid),
                "pearson_r": r,
                "p_value": p,
            })

    return pd.DataFrame(rows).sort_values(
        by="pearson_r",
        key=lambda s: s.abs(),
        ascending=False,
    )


def cross_correlation(df: pd.DataFrame, metric: str, temp_col: str) -> pd.DataFrame:
    rows = []

    for lag_min in range(-MAX_LAG_MINUTES, MAX_LAG_MINUTES + 1, LAG_STEP_MINUTES):
        shifted = df[temp_col].copy()
        shifted.index = shifted.index + pd.Timedelta(minutes=lag_min)

        joined = pd.merge_asof(
            df[[metric]].dropna().reset_index().sort_values("datetime"),
            shifted.rename(temp_col).dropna().reset_index().sort_values("datetime"),
            on="datetime",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=max(2, LAG_STEP_MINUTES)),
        ).dropna()

        if len(joined) < 3:
            continue

        r, p = pearsonr(joined[temp_col], joined[metric])
        rows.append({
            "metric": metric,
            "temperature_variable": temp_col,
            "lag_minutes": lag_min,
            "n": len(joined),
            "pearson_r": r,
            "p_value": p,
        })

    return pd.DataFrame(rows)


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.show()


def plot_time_series(df: pd.DataFrame) -> None:
    for metric in [m for m in METRICS if m in df.columns]:
        plt.figure(figsize=(12, 6))

        ax1 = plt.gca()
        ax1.plot(df.index, df[metric], marker=".", linestyle="None", label=metric)
        ax1.set_ylabel(metric)
        ax1.grid(True, alpha=0.25)

        ax2 = ax1.twinx()
        ax2.plot(df.index, df["T_Ljubljana"], linewidth=1.2, label="T Ljubljana")
        ax2.plot(df.index, df["T_Drnovo"], linewidth=1.2, label="T Drnovo")
        ax2.set_ylabel("Temperature / °C")

        lines = ax1.get_lines() + ax2.get_lines()
        ax1.legend(lines, [line.get_label() for line in lines], loc="best")

        save_plot(OUT_DIR / f"time_series_{metric}.png")


def plot_scatter(df: pd.DataFrame) -> None:
    temp_cols = ["T_Ljubljana", "T_Drnovo", "T_delta"]

    for metric in [m for m in METRICS if m in df.columns]:
        for temp_col in temp_cols:
            valid = df[[metric, temp_col]].dropna()
            if len(valid) < 3:
                continue

            r, p = pearsonr(valid[temp_col], valid[metric])

            plt.figure(figsize=(7, 5))
            plt.scatter(valid[temp_col], valid[metric], s=18)
            plt.xlabel(temp_col)
            plt.ylabel(metric)
            plt.title(f"{metric} vs {temp_col}: r={r:.3f}, p={p:.3g}")
            plt.grid(True, alpha=0.25)

            save_plot(OUT_DIR / f"scatter_{metric}_vs_{temp_col}.png")


def plot_cross_correlations(df: pd.DataFrame) -> pd.DataFrame:
    all_cc = []

    for metric in [m for m in METRICS if m in df.columns]:
        for temp_col in ["T_Ljubljana", "T_Drnovo", "T_delta"]:
            cc = cross_correlation(df, metric, temp_col)
            if cc.empty:
                continue

            all_cc.append(cc)

            best = cc.iloc[cc["pearson_r"].abs().argmax()]

            plt.figure(figsize=(8, 5))
            plt.plot(cc["lag_minutes"], cc["pearson_r"], marker=".")
            plt.axvline(best["lag_minutes"], linestyle="--", linewidth=1)
            plt.axhline(0.0, linestyle=":", linewidth=1)
            plt.xlabel("Temperature lag / min")
            plt.ylabel("Pearson r")
            plt.title(
                f"{metric} vs {temp_col}; best lag={best['lag_minutes']:.0f} min, "
                f"r={best['pearson_r']:.3f}"
            )
            plt.grid(True, alpha=0.25)

            save_plot(OUT_DIR / f"crosscorr_{metric}_vs_{temp_col}.png")

    if not all_cc:
        return pd.DataFrame()

    return pd.concat(all_cc, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    measurements = read_measurements(CSV_PATH)
    df = merge_temperature(measurements)

    df.to_csv(OUT_DIR / "merged_measurements_temperature.csv")

    pearson = pearson_table(df)
    pearson.to_csv(OUT_DIR / "pearson_correlations.csv", index=False)

    crosscorr = plot_cross_correlations(df)
    if not crosscorr.empty:
        crosscorr.to_csv(OUT_DIR / "cross_correlations.csv", index=False)

        best_lags = crosscorr.loc[
            crosscorr.groupby(["metric", "temperature_variable"])["pearson_r"]
            .apply(lambda s: s.abs().idxmax())
            .values
        ]
        best_lags.to_csv(OUT_DIR / "best_cross_correlation_lags.csv", index=False)

    plot_time_series(df)
    plot_scatter(df)

    print(f"Saved temperature analysis to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()