from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


PLOT_RANGE = (150, 0)

CSV_PATH = Path("Data/alice_results.csv")
OUT_DIR = Path("Data/temperature_analysis")
TEMPERATURE_LOG_PATH = OUT_DIR / "temperature_log.csv"

METRICS = [
    "visibility",
    "vis_HV",
    "vis_DA",
    "QBER_total",
    "QBER_HV",
    "QBER_DA",
    "CHSH_S_value",
]

MAX_LAG_MINUTES = 60
LAG_STEP_MINUTES = 5
TEMPERATURE_MERGE_TOLERANCE = pd.Timedelta("90min")


def apply_plot_range(df: pd.DataFrame) -> pd.DataFrame:
    if PLOT_RANGE is None:
        return df

    older, newer = PLOT_RANGE
    n = len(df)

    stop = n if newer == 0 else max(0, n - newer)
    start = 0 if older == 0 else max(0, n - older)

    return df.iloc[start:stop].copy()


def read_measurements(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        raise ValueError("CSV needs a timestamp column")

    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.sort_values("datetime")

    keep = ["datetime"] + [column for column in METRICS if column in df.columns]
    return df[keep].dropna(how="all", subset=keep[1:])


def read_temperature_log(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing temperature log: {path}")

    temp = pd.read_csv(path)

    required = ["T_Ljubljana", "T_Drnovo"]
    missing = [col for col in required if col not in temp.columns]
    if missing:
        raise ValueError(f"Temperature log is missing columns: {missing}")

    datetime_series = pd.Series(pd.NaT, index=temp.index, dtype="datetime64[ns]")

    if "alice_timestamp" in temp.columns:
        ts = pd.to_numeric(temp["alice_timestamp"], errors="coerce")
        datetime_series = datetime_series.fillna(
            pd.to_datetime(ts, unit="s", errors="coerce")
        )

    if "logger_timestamp" in temp.columns:
        ts = pd.to_numeric(temp["logger_timestamp"], errors="coerce")
        datetime_series = datetime_series.fillna(
            pd.to_datetime(ts, unit="s", errors="coerce")
        )

    if "source_timestamp" in temp.columns:
        ts = pd.to_numeric(temp["source_timestamp"], errors="coerce")
        datetime_series = datetime_series.fillna(
            pd.to_datetime(ts, unit="s", errors="coerce")
        )

    if "source_datetime" in temp.columns:
        datetime_series = datetime_series.fillna(
            pd.to_datetime(temp["source_datetime"], errors="coerce")
        )

    temp["datetime"] = datetime_series
    temp = temp.dropna(subset=["datetime"])

    keep = ["datetime", "T_Ljubljana", "T_Drnovo"]
    temp = temp[keep].copy()

    temp["T_Ljubljana"] = pd.to_numeric(temp["T_Ljubljana"], errors="coerce")
    temp["T_Drnovo"] = pd.to_numeric(temp["T_Drnovo"], errors="coerce")
    temp = temp.dropna(subset=["T_Ljubljana", "T_Drnovo"])

    if temp.empty:
        raise ValueError("Temperature log contains no valid temperature rows after parsing")

    temp = temp.groupby("datetime", as_index=False).mean(numeric_only=True)
    temp["T_delta"] = temp["T_Ljubljana"] - temp["T_Drnovo"]

    return temp.sort_values("datetime")


def merge_temperature(meas: pd.DataFrame) -> pd.DataFrame:
    temp = read_temperature_log(TEMPERATURE_LOG_PATH)

    meas = meas.sort_values("datetime")
    temp = temp.sort_values("datetime")

    merged = pd.merge_asof(
        meas,
        temp,
        on="datetime",
        direction="nearest",
        tolerance=TEMPERATURE_MERGE_TOLERANCE,
    ).set_index("datetime")

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

    if not rows:
        return pd.DataFrame()

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
        
        if joined[temp_col].nunique() < 2 or joined[metric].nunique() < 2:
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
    plt.close()


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
    measurements = apply_plot_range(measurements)
    df = merge_temperature(measurements)
    print(df[["T_Ljubljana", "T_Drnovo", "T_delta"]].describe())
    print("Rows with temperature:", df[["T_Ljubljana", "T_Drnovo"]].dropna().shape[0], "/", len(df))

    df.to_csv(OUT_DIR / "merged_measurements_temperature.csv")

    pearson = pearson_table(df)
    pearson.to_csv(OUT_DIR / "pearson_correlations.csv", index=False)

    crosscorr = plot_cross_correlations(df)
    if not crosscorr.empty:
        crosscorr.to_csv(OUT_DIR / "cross_correlations.csv", index=False)

        valid_crosscorr = crosscorr.replace([np.inf, -np.inf], np.nan).dropna(
                            subset=["pearson_r"]
                            )

        if not valid_crosscorr.empty:
            best_indices = (
                valid_crosscorr.assign(abs_r=valid_crosscorr["pearson_r"].abs())
                .groupby(["metric", "temperature_variable"])["abs_r"]
                .idxmax()
                .dropna()
                .astype(int)
            )
        
            best_lags = valid_crosscorr.loc[best_indices].drop(columns=["abs_r"], errors="ignore")
            best_lags.to_csv(OUT_DIR / "best_cross_correlation_lags.csv", index=False)
    else:
        print("No valid cross-correlation values found.")
    
    plot_time_series(df)
    plot_scatter(df)
    
    print(f"Saved temperature analysis to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()