from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

try:
    from Alice import OPTIMIZER
except BaseException:
    OPTIMIZER = None

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV_FILES = (
    BASE_DIR / "Data" / "alice_results.csv",
    BASE_DIR / "Data" / "qber_iterlog.csv",
)
CSV_FILE: Path | None = None
REFRESH_INTERVAL_SECONDS = 2.0
# PLOT_RANGE selects valid measurement rows by offset from the newest row.
# (1000, 0) matches the old HISTORY_ROWS=1000 behavior.
# (300, 100) plots rows inside the latest 300 while skipping the newest 100.
# Set to None or (0, 0) to display all rows.
PLOT_RANGE: tuple[int | None, int | None] | None = (50, 0)
USE_CONSTANT_POINT_SPACING = True
LIVE_UPDATE = True
SAVE_PATH: Path | None = None
PLOT_MODE = "chsh"  # "both", "visibility", or "CHSH".
PAIR_TREND_WINDOW = 5  # Set to 0 or 1 to disable.


def _objective_family(metric_text: str | None) -> str | None:
    if metric_text is None:
        return None
    metric = metric_text.strip().lower().replace("-", "_")
    if metric in {"chsh", "chsh_s", "s", "s_value"}:
        return "chsh_s"
    if metric in {
        "visibility",
        "total_visibility",
        "vis_hv",
        "hv_visibility",
        "hv",
        "vis_da",
        "da_visibility",
        "da",
    }:
        return "visibility"
    return None


def _optimizer_target_for_metric(metric_family: str, default: float) -> float:
    if OPTIMIZER is None:
        return default
    return default
    primary_family = _objective_family(OPTIMIZER.objective_metric)
    secondary_family = _objective_family(OPTIMIZER.secondary_objective_metric)
    if primary_family == metric_family and OPTIMIZER.objective_target is not None:
        return float(OPTIMIZER.objective_target)
    if (
        secondary_family == metric_family
        and OPTIMIZER.secondary_objective_target is not None
    ):
        return float(OPTIMIZER.secondary_objective_target)
    return default


VISIBILITY_TARGET = _optimizer_target_for_metric("visibility", 1/np.sqrt(2))
CHSH_CLASSICAL_LIMIT = 2.0
CHSH_QUANTUM_LIMIT = 2.0 * np.sqrt(2.0)
ERRORBAR_KWARGS = {"capsize": 2, "elinewidth": 0.8, "capthick": 0.8}
PAIR_LABELS = ("HH", "VV", "DD", "AA", "HV", "VH", "DA", "AD")
CHSH_PAIR_LABELS = (
    "HH", "HV", "VH", "VV",
    "HA", "HD", "VA", "VD",
    "DH", "DV", "AH", "AV",
    "DD", "DA", "AD", "AA",
)
# Keep this label convention identical to analyze_chsh_s_coincidences()
# in qkd_epc_correction.py.
CHSH_EXPECTATION_LABELS = ("E_ab", "E_abp", "E_apb", "E_apbp")
CHSH_EXPECTATION_DISPLAY_LABELS = {
    "E_ab": "E(1,1)",
    "E_abp": "E(1,2)",
    "E_apb": "E(2,1)",
    "E_apbp": "E(2,2)",
}
CHSH_EXPECTATION_COLUMNS = {
    "E_ab": "CHSH_E_ab",
    "E_abp": "CHSH_E_abp",
    "E_apb": "CHSH_E_apb",
    "E_apbp": "CHSH_E_apbp",
}
CHSH_EXPECTATION_COLORS = {
    "E_ab": "#0072b2",
    "E_abp": "#d55e00",
    "E_apb": "#009e73",
    "E_apbp": "#cc79a7",
}
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


def _calculate_chsh_from_count_row(
    row_counts: dict[str, float],
) -> tuple[dict[str, float], float, float]:
    counts: dict[str, float] = {}
    for label in CHSH_PAIR_LABELS:
        value = float(row_counts[label])
        if not np.isfinite(value):
            raise ValueError(f"Invalid CHSH count C_{label}={value!r}")
        counts[label] = value

    def correlation(pp: str, pm: str, mp: str, mm: str) -> float:
        n_pp = counts[pp]
        n_pm = counts[pm]
        n_mp = counts[mp]
        n_mm = counts[mm]
        total = n_pp + n_pm + n_mp + n_mm
        if total <= 0:
            return 0.0
        return float((n_pp + n_mm - n_pm - n_mp) / total)

    e_ab = correlation("HH", "HV", "VH", "VV")
    e_abp = correlation("HD", "HA", "VD", "VA")
    e_apb = correlation("DH", "DV", "AH", "AV")
    e_apbp = correlation("DD", "DA", "AD", "AA")
    signed_s = float(e_ab - e_abp + e_apb + e_apbp)
    return (
        {
            "E_ab": e_ab,
            "E_abp": e_abp,
            "E_apb": e_apb,
            "E_apbp": e_apbp,
        },
        signed_s,
        abs(signed_s),
    )


@dataclass(frozen=True)
class MeasurementSeries:
    latest_timestamp: float
    time_axis_positions: np.ndarray
    time_axis_timestamps: np.ndarray
    measurement_index: np.ndarray
    relative_seconds: np.ndarray
    counts: dict[str, np.ndarray]
    total_coincidences: np.ndarray
    visibility_total: np.ndarray
    visibility_total_error: np.ndarray
    visibility_hv: np.ndarray
    visibility_hv_error: np.ndarray
    visibility_da: np.ndarray
    visibility_da_error: np.ndarray
    qber_total: np.ndarray
    qber_total_error: np.ndarray
    qber_hv: np.ndarray
    qber_hv_error: np.ndarray
    qber_da: np.ndarray
    qber_da_error: np.ndarray
    chsh_relative_seconds: np.ndarray
    chsh_s_value: np.ndarray
    chsh_s_error: np.ndarray
    chsh_from_counts: np.ndarray
    chsh_expectations: dict[str, np.ndarray]
    chsh_expectation_errors: dict[str, np.ndarray]

    @property
    def size(self) -> int:
        return int(self.measurement_index.size)

    @property
    def chsh_size(self) -> int:
        return int(self.chsh_s_value.size)


def choose_csv_path(requested_path: Path | None) -> Path:
    if requested_path is not None:
        return requested_path.expanduser().resolve()

    for path in DEFAULT_CSV_FILES:
        if path.is_file():
            return path
    return DEFAULT_CSV_FILES[0]


def _normalized_plot_mode(mode_text: str | None = None) -> str:
    mode = (PLOT_MODE if mode_text is None else mode_text).strip().lower()
    aliases = {
        "both": "both",
        "all": "both",
        "visibility": "visibility",
        "vis": "visibility",
        "qkd": "visibility",
        "chsh": "chsh",
        "chsh_s": "chsh",
        "s": "chsh",
    }
    if mode not in aliases:
        raise ValueError(
            "PLOT_MODE must be 'both', 'visibility', or 'CHSH'; "
            f"received {mode_text if mode_text is not None else PLOT_MODE!r}"
        )
    return aliases[mode]


def _selected_indices_for_mode(
    mode: str,
    qkd_indices: list[int],
    chsh_indices: list[int],
) -> list[int]:
    if mode == "visibility":
        return qkd_indices
    if mode == "chsh":
        return chsh_indices
    return qkd_indices + chsh_indices


def _normalize_plot_range(
    plot_range: tuple[int | None, int | None] | None,
) -> tuple[int, int] | None:
    if plot_range is None:
        return None
    try:
        older_offset, newer_offset = plot_range
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "PLOT_RANGE must be None or a tuple like (older, newer)"
        ) from exc

    older = 0 if older_offset is None else int(older_offset)
    newer = 0 if newer_offset is None else int(newer_offset)
    if older < 0 or newer < 0:
        raise ValueError("PLOT_RANGE offsets cannot be negative")
    if older > 0 and older <= newer:
        raise ValueError(
            "PLOT_RANGE older offset must be greater than the newer offset; "
            "use None or 0 as older offset for no older limit"
        )
    return older, newer


def _select_plot_range(
    indices: list[int],
    plot_range: tuple[int | None, int | None] | None,
) -> list[int]:
    normalized = _normalize_plot_range(plot_range)
    if normalized is None:
        return indices

    older, newer = normalized
    row_count = len(indices)
    stop = max(0, row_count - newer) if newer else row_count
    start = 0 if older == 0 else max(0, row_count - older)
    if start > stop:
        start = stop
    return indices[start:stop]


def _measurement_x_values(
    row_timestamps: np.ndarray,
    latest_timestamp: float,
) -> np.ndarray:
    if row_timestamps.size == 0:
        return _empty_array()
    if USE_CONSTANT_POINT_SPACING:
        return (
            np.arange(row_timestamps.size, dtype=np.float64)
            - float(row_timestamps.size - 1)
        )
    return row_timestamps - latest_timestamp


def _time_axis_data(
    mode: str,
    pair_positions: np.ndarray,
    pair_timestamps: np.ndarray,
    chsh_positions: np.ndarray,
    chsh_timestamps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "chsh" and chsh_positions.size > 0:
        return chsh_positions, chsh_timestamps
    if mode == "visibility" and pair_positions.size > 0:
        return pair_positions, pair_timestamps
    if pair_positions.size > 0:
        return pair_positions, pair_timestamps
    return chsh_positions, chsh_timestamps


def read_measurements(
    path: Path,
    plot_range: tuple[int | None, int | None] | None,
    plot_mode: str | None = None,
) -> MeasurementSeries:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise ValueError(f"{path} contains no measurement rows")

    mode = _normalized_plot_mode(plot_mode)
    timestamps = _timestamp_column(rows)

    qkd_indices = [
        index
        for index in _indices_with_pair_counts(rows, PAIR_LABELS)
        if _has_qkd_metrics(rows[index])
    ]
    chsh_indices = [
        index
        for index, row in enumerate(rows)
        if _has_chsh_counts(row) or _has_float(row, "CHSH_S_value")
    ]

    qkd_indices = _select_plot_range(qkd_indices, plot_range)
    chsh_indices = _select_plot_range(chsh_indices, plot_range)

    selected_indices = _selected_indices_for_mode(mode, qkd_indices, chsh_indices)
    if not selected_indices:
        raise ValueError(
            f"{path} contains no measurement rows for PLOT_MODE={mode!r}"
        )

    latest_timestamp = float(np.max(timestamps[selected_indices]))
    qkd_rows = [rows[index] for index in qkd_indices]
    chsh_rows = [rows[index] for index in chsh_indices]
    pair_indices = (
        [
            index
            for index in chsh_indices
            if _has_pair_counts(rows[index], PAIR_LABELS)
        ]
        if mode == "chsh"
        else qkd_indices
    )
    pair_rows = [rows[index] for index in pair_indices]

    if pair_rows:
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
            label: _float_column(pair_rows, f"C_{label}")
            for label in PAIR_LABELS
        }
        total = _optional_float_column(pair_rows, "total_coincidences")
        if total is None:
            total = np.sum(
                np.stack([counts[label] for label in PAIR_LABELS]),
                axis=0,
            )
        measurement_index = np.arange(1, len(pair_rows) + 1, dtype=np.int64)
        pair_timestamps = timestamps[pair_indices]
        relative_seconds = _measurement_x_values(pair_timestamps, latest_timestamp)
    else:
        counts = {label: _empty_array() for label in PAIR_LABELS}
        total = _empty_array()
        measurement_index = np.zeros(0, dtype=np.int64)
        pair_timestamps = _empty_array()
        relative_seconds = _empty_array()

    if qkd_rows:
        hv_correlated = _float_column(qkd_rows, "C_HH") + _float_column(
            qkd_rows, "C_VV"
        )
        hv_errors = _float_column(qkd_rows, "C_HV") + _float_column(
            qkd_rows, "C_VH"
        )
        da_correlated = _float_column(qkd_rows, "C_DD") + _float_column(
            qkd_rows, "C_AA"
        )
        da_errors = _float_column(qkd_rows, "C_DA") + _float_column(
            qkd_rows, "C_AD"
        )

        visibility_hv = _optional_float_column(qkd_rows, "vis_HV")
        if visibility_hv is None:
            visibility_hv = _visibility(hv_correlated, hv_errors)
        visibility_hv_error = _visibility_error(hv_correlated, hv_errors)

        visibility_da = _optional_float_column(qkd_rows, "vis_DA")
        if visibility_da is None:
            visibility_da = _visibility(da_correlated, da_errors)
        visibility_da_error = _visibility_error(da_correlated, da_errors)

        visibility_total = _optional_float_column(qkd_rows, "visibility")
        if visibility_total is None:
            visibility_total = (visibility_hv + visibility_da) / 2.0
        visibility_total_error = 0.5 * np.sqrt(
            visibility_hv_error**2 + visibility_da_error**2
        )

        qber_hv = _optional_float_column(qkd_rows, "QBER_HV")
        if qber_hv is None:
            qber_hv = _qber(hv_correlated, hv_errors)
        qber_hv_error = _qber_error(hv_correlated, hv_errors)

        qber_da = _optional_float_column(qkd_rows, "QBER_DA")
        if qber_da is None:
            qber_da = _qber(da_correlated, da_errors)
        qber_da_error = _qber_error(da_correlated, da_errors)

        total_correlated = hv_correlated + da_correlated
        total_errors = hv_errors + da_errors
        qber_total = _optional_float_column(qkd_rows, "QBER_total")
        if qber_total is None:
            qber_total = _qber(total_correlated, total_errors)
        qber_total_error = _qber_error(total_correlated, total_errors)
    else:
        visibility_hv = _empty_array()
        visibility_hv_error = _empty_array()
        visibility_da = _empty_array()
        visibility_da_error = _empty_array()
        visibility_total = _empty_array()
        visibility_total_error = _empty_array()
        qber_hv = _empty_array()
        qber_hv_error = _empty_array()
        qber_da = _empty_array()
        qber_da_error = _empty_array()
        qber_total = _empty_array()
        qber_total_error = _empty_array()

    if chsh_rows:
        (
            chsh_s_value,
            chsh_s_error,
            chsh_from_counts,
            chsh_expectations,
            chsh_expectation_errors,
        ) = _chsh_series(chsh_rows)
        chsh_timestamps = timestamps[chsh_indices]
        chsh_relative_seconds = _measurement_x_values(
            chsh_timestamps,
            latest_timestamp,
        )
        valid_chsh = np.isfinite(chsh_s_value)
        chsh_s_value = chsh_s_value[valid_chsh]
        chsh_s_error = chsh_s_error[valid_chsh]
        chsh_from_counts = chsh_from_counts[valid_chsh]
        chsh_expectations = {
            label: values[valid_chsh]
            for label, values in chsh_expectations.items()
        }
        chsh_expectation_errors = {
            label: values[valid_chsh]
            for label, values in chsh_expectation_errors.items()
        }
        chsh_relative_seconds = chsh_relative_seconds[valid_chsh]
        chsh_timestamps = chsh_timestamps[valid_chsh]
    else:
        chsh_s_value = _empty_array()
        chsh_s_error = _empty_array()
        chsh_from_counts = np.zeros(0, dtype=bool)
        chsh_expectations = {
            label: _empty_array() for label in CHSH_EXPECTATION_LABELS
        }
        chsh_expectation_errors = {
            label: _empty_array() for label in CHSH_EXPECTATION_LABELS
        }
        chsh_relative_seconds = _empty_array()
        chsh_timestamps = _empty_array()

    time_axis_positions, time_axis_timestamps = _time_axis_data(
        mode,
        relative_seconds,
        pair_timestamps,
        chsh_relative_seconds,
        chsh_timestamps,
    )

    return MeasurementSeries(
        latest_timestamp=latest_timestamp,
        time_axis_positions=time_axis_positions,
        time_axis_timestamps=time_axis_timestamps,
        measurement_index=measurement_index,
        relative_seconds=relative_seconds,
        counts=counts,
        total_coincidences=total,
        visibility_total=visibility_total,
        visibility_total_error=visibility_total_error,
        visibility_hv=visibility_hv,
        visibility_hv_error=visibility_hv_error,
        visibility_da=visibility_da,
        visibility_da_error=visibility_da_error,
        qber_total=qber_total,
        qber_total_error=qber_total_error,
        qber_hv=qber_hv,
        qber_hv_error=qber_hv_error,
        qber_da=qber_da,
        qber_da_error=qber_da_error,
        chsh_relative_seconds=chsh_relative_seconds,
        chsh_s_value=chsh_s_value,
        chsh_s_error=chsh_s_error,
        chsh_from_counts=chsh_from_counts,
        chsh_expectations=chsh_expectations,
        chsh_expectation_errors=chsh_expectation_errors,
    )


def _empty_array() -> np.ndarray:
    return np.zeros(0, dtype=np.float64)


def _has_float(row: dict[str, str], name: str) -> bool:
    try:
        value = float(row.get(name, ""))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value))


def _has_qkd_metrics(row: dict[str, str]) -> bool:
    return (
        _has_float(row, "visibility")
        or _has_float(row, "vis_HV")
        or _has_float(row, "vis_DA")
        or _has_float(row, "QBER_total")
    )


def _has_pair_counts(row: dict[str, str], labels: tuple[str, ...]) -> bool:
    return all(_has_float(row, f"C_{label}") for label in labels)


def _has_chsh_counts(row: dict[str, str]) -> bool:
    return _has_pair_counts(row, CHSH_PAIR_LABELS)


def _indices_with_pair_counts(
    rows: list[dict[str, str]],
    labels: tuple[str, ...],
) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if _has_pair_counts(row, labels)
    ]


def _rows_with_pair_counts(
    rows: list[dict[str, str]],
    labels: tuple[str, ...],
) -> list[dict[str, str]]:
    return [rows[index] for index in _indices_with_pair_counts(rows, labels)]


def _float_column(rows: list[dict[str, str]], name: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        try:
            values.append(float(row.get(name, "")))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def _optional_float_column(
    rows: list[dict[str, str]],
    name: str,
) -> np.ndarray | None:
    if not rows or name not in rows[0]:
        return None
    if any(not _has_float(row, name) for row in rows):
        return None
    return _float_column(rows, name)


def _timestamp_column(rows: list[dict[str, str]]) -> np.ndarray:
    if "timestamp" not in rows[0]:
        raise ValueError("CSV is missing the timestamp column")

    timestamps = np.full(len(rows), np.nan, dtype=np.float64)
    for index, row in enumerate(rows):
        value = row.get("timestamp", "")
        try:
            timestamps[index] = float(value)
        except (TypeError, ValueError):
            continue

    valid = np.isfinite(timestamps)
    valid_count = int(np.count_nonzero(valid))
    if valid_count == 0:
        raise ValueError("CSV contains no valid timestamp values")
    if valid_count == 1 and len(rows) > 1:
        raise ValueError(
            "CSV needs at least two valid timestamps to reconstruct time"
        )

    if not np.all(valid):
        row_positions = np.arange(len(rows), dtype=np.float64)
        timestamps = np.interp(
            row_positions,
            row_positions[valid],
            timestamps[valid],
        )
        print(
            f"Interpolated {len(rows) - valid_count} missing or invalid "
            "timestamp values"
        )

    if np.any(np.diff(timestamps) < 0):
        raise ValueError("CSV timestamps are not in chronological order")
    return timestamps


def _visibility(correlated: np.ndarray, errors: np.ndarray) -> np.ndarray:
    total = correlated + errors
    return np.divide(
        correlated - errors,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )


def _visibility_error(correlated: np.ndarray, errors: np.ndarray) -> np.ndarray:
    total = correlated + errors
    result = np.zeros_like(total, dtype=np.float64)
    mask = total > 0
    result[mask] = np.sqrt(
        (2.0 * errors[mask] / total[mask] ** 2) ** 2 * correlated[mask]
        + (2.0 * correlated[mask] / total[mask] ** 2) ** 2 * errors[mask]
    )
    return result


def _qber(correlated: np.ndarray, errors: np.ndarray) -> np.ndarray:
    total = correlated + errors
    return np.divide(
        errors,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )


def _qber_error(correlated: np.ndarray, errors: np.ndarray) -> np.ndarray:
    total = correlated + errors
    result = np.zeros_like(total, dtype=np.float64)
    mask = total > 0
    result[mask] = np.sqrt(
        (errors[mask] / total[mask] ** 2) ** 2 * correlated[mask]
        + (correlated[mask] / total[mask] ** 2) ** 2 * errors[mask]
    )
    return result


def _correlation(
    pp: np.ndarray,
    pm: np.ndarray,
    mp: np.ndarray,
    mm: np.ndarray,
) -> np.ndarray:
    total = pp + pm + mp + mm
    return np.divide(
        pp + mm - pm - mp,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )


def _chsh_counts(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    rows = _rows_with_pair_counts(rows, CHSH_PAIR_LABELS)
    return {label: _float_column(rows, f"C_{label}") for label in CHSH_PAIR_LABELS}


def _count_row(
    counts: dict[str, np.ndarray],
    row_index: int,
) -> dict[str, float]:
    return {
        label: float(counts[label][row_index])
        for label in CHSH_PAIR_LABELS
    }


def _chsh_results_from_counts(
    counts: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    row_count = len(next(iter(counts.values()))) if counts else 0
    expectations = {
        label: np.zeros(row_count, dtype=np.float64)
        for label in CHSH_EXPECTATION_LABELS
    }
    signed_s = np.zeros(row_count, dtype=np.float64)
    s_value = np.zeros(row_count, dtype=np.float64)

    for row_index in range(row_count):
        row_expectations, row_signed_s, row_s_value = _calculate_chsh_from_count_row(
            _count_row(counts, row_index)
        )
        for label in CHSH_EXPECTATION_LABELS:
            expectations[label][row_index] = row_expectations[label]
        signed_s[row_index] = row_signed_s
        s_value[row_index] = row_s_value

    return expectations, signed_s, s_value


def _chsh_correlations(
    counts: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expectations, _, _ = _chsh_results_from_counts(counts)
    return tuple(expectations[label] for label in CHSH_EXPECTATION_LABELS)


def _chsh_s_from_counts(rows: list[dict[str, str]]) -> np.ndarray:
    rows = _rows_with_pair_counts(rows, CHSH_PAIR_LABELS)
    if not rows:
        return _empty_array()
    counts = _chsh_counts(rows)
    _, _, s_value = _chsh_results_from_counts(counts)
    return s_value


def _chsh_expectations_from_counts(
    rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    rows = _rows_with_pair_counts(rows, CHSH_PAIR_LABELS)
    if not rows:
        return {label: _empty_array() for label in CHSH_EXPECTATION_LABELS}
    counts = _chsh_counts(rows)
    return dict(zip(CHSH_EXPECTATION_LABELS, _chsh_correlations(counts)))


def _chsh_expectation_errors_from_counts(
    rows: list[dict[str, str]],
) -> dict[str, np.ndarray]:
    rows = _rows_with_pair_counts(rows, CHSH_PAIR_LABELS)
    if not rows:
        return {label: _empty_array() for label in CHSH_EXPECTATION_LABELS}
    counts = _chsh_counts(rows)
    row_count = len(rows)
    errors = {
        label: np.zeros(row_count, dtype=np.float64)
        for label in CHSH_EXPECTATION_LABELS
    }

    for row_index in range(row_count):
        row_counts = _count_row(counts, row_index)
        variances = {label: 0.0 for label in CHSH_EXPECTATION_LABELS}
        for count_label in CHSH_PAIR_LABELS:
            count = row_counts[count_label]
            if count < 0:
                continue
            step = max(1.0, np.sqrt(count))
            plus_counts = dict(row_counts)
            minus_counts = dict(row_counts)
            plus_counts[count_label] = count + step
            minus_counts[count_label] = max(0.0, count - step)
            plus, _, _ = _calculate_chsh_from_count_row(plus_counts)
            minus, _, _ = _calculate_chsh_from_count_row(minus_counts)
            denominator = plus_counts[count_label] - minus_counts[count_label]
            if denominator <= 0:
                continue
            for label in CHSH_EXPECTATION_LABELS:
                derivative = (plus[label] - minus[label]) / denominator
                variances[label] += derivative**2 * count
        for label in CHSH_EXPECTATION_LABELS:
            errors[label][row_index] = np.sqrt(variances[label])

    return errors


def _stored_chsh_expectation(row: dict[str, str], label: str) -> float:
    column = CHSH_EXPECTATION_COLUMNS[label]
    if _has_float(row, column):
        return float(row[column])
    return np.nan


def _chsh_series(
    rows: list[dict[str, str]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    values: list[float] = []
    errors: list[float] = []
    from_counts: list[bool] = []
    expectations: dict[str, list[float]] = {
        label: [] for label in CHSH_EXPECTATION_LABELS
    }
    expectation_errors: dict[str, list[float]] = {
        label: [] for label in CHSH_EXPECTATION_LABELS
    }

    for row in rows:
        appended = False
        if _has_chsh_counts(row):
            try:
                row_expectations = _chsh_expectations_from_counts([row])
                row_expectation_errors = _chsh_expectation_errors_from_counts([row])
                values.append(float(_chsh_s_from_counts([row])[0]))
                errors.append(float(_chsh_s_error([row])[0]))
                from_counts.append(True)
                for label in CHSH_EXPECTATION_LABELS:
                    expectations[label].append(float(row_expectations[label][0]))
                    expectation_errors[label].append(
                        float(row_expectation_errors[label][0])
                    )
                appended = True
            except (TypeError, ValueError, OverflowError):
                appended = False

        if not appended and _has_float(row, "CHSH_S_value"):
            values.append(float(row["CHSH_S_value"]))
            errors.append(0.0)
            from_counts.append(False)
            for label in CHSH_EXPECTATION_LABELS:
                expectations[label].append(_stored_chsh_expectation(row, label))
                expectation_errors[label].append(0.0)
            appended = True

        if not appended:
            values.append(np.nan)
            errors.append(np.nan)
            from_counts.append(False)
            for label in CHSH_EXPECTATION_LABELS:
                expectations[label].append(np.nan)
                expectation_errors[label].append(np.nan)

    return (
        np.asarray(values, dtype=np.float64),
        np.asarray(errors, dtype=np.float64),
        np.asarray(from_counts, dtype=bool),
        {
            label: np.asarray(label_values, dtype=np.float64)
            for label, label_values in expectations.items()
        },
        {
            label: np.asarray(label_values, dtype=np.float64)
            for label, label_values in expectation_errors.items()
        },
    )


def _correlation_error(
    pp: np.ndarray,
    pm: np.ndarray,
    mp: np.ndarray,
    mm: np.ndarray,
) -> np.ndarray:
    counts = (pp, pm, mp, mm)
    signs = (1.0, -1.0, -1.0, 1.0)
    total = pp + pm + mp + mm
    correlation = np.divide(
        pp + mm - pm - mp,
        total,
        out=np.zeros_like(total, dtype=np.float64),
        where=total > 0,
    )
    variance = np.zeros_like(total, dtype=np.float64)
    mask = total > 0
    for sign, values in zip(signs, counts):
        derivative = np.zeros_like(total, dtype=np.float64)
        derivative[mask] = (sign - correlation[mask]) / total[mask]
        variance[mask] += derivative[mask] ** 2 * values[mask]
    return np.sqrt(variance)


def _chsh_s_error(rows: list[dict[str, str]]) -> np.ndarray:
    rows = _rows_with_pair_counts(rows, CHSH_PAIR_LABELS)
    if not rows:
        return _empty_array()

    try:
        counts = _chsh_counts(rows)
    except ValueError:
        return np.full(len(rows), np.nan, dtype=np.float64)

    row_count = len(rows)
    errors = np.zeros(row_count, dtype=np.float64)
    for row_index in range(row_count):
        row_counts = _count_row(counts, row_index)
        variance = 0.0
        for count_label in CHSH_PAIR_LABELS:
            count = row_counts[count_label]
            if count < 0:
                continue
            step = max(1.0, np.sqrt(count))
            plus_counts = dict(row_counts)
            minus_counts = dict(row_counts)
            plus_counts[count_label] = count + step
            minus_counts[count_label] = max(0.0, count - step)
            _, plus, _ = _calculate_chsh_from_count_row(plus_counts)
            _, minus, _ = _calculate_chsh_from_count_row(minus_counts)
            denominator = plus_counts[count_label] - minus_counts[count_label]
            if denominator <= 0:
                continue
            derivative = (plus - minus) / denominator
            variance += derivative**2 * count
        errors[row_index] = np.sqrt(variance)
    return errors


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return np.full(values.shape, np.nan, dtype=np.float64)
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    result = np.full(values.shape, np.nan, dtype=np.float64)
    result[window - 1 :] = smoothed
    return result


def _format_relative_time(value: float, _position: float) -> str:
    sign = "-" if value < -0.5 else ""
    seconds = int(round(abs(value)))
    days, seconds = divmod(seconds, 86_400)
    hours, seconds = divmod(seconds, 3_600)
    minutes, seconds = divmod(seconds, 60)

    if days:
        return f"{sign}{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    if hours:
        return f"{sign}{hours}:{minutes:02d}:{seconds:02d}"
    return f"{sign}{minutes}:{seconds:02d}"


def _format_x_axis_value(value: float, position: float) -> str:
    if USE_CONSTANT_POINT_SPACING:
        if not np.isfinite(value):
            return ""
        return f"{int(round(value))}"
    return _format_relative_time(value, position)


def _format_absolute_time(
    x_value: float,
    _position: float,
    axis_positions: np.ndarray,
    axis_timestamps: np.ndarray,
) -> str:
    if not np.isfinite(x_value) or axis_positions.size == 0:
        return ""

    valid = np.isfinite(axis_positions) & np.isfinite(axis_timestamps)
    if not np.any(valid):
        return ""

    positions = axis_positions[valid]
    timestamps = axis_timestamps[valid]
    order = np.argsort(positions)
    positions = positions[order]
    timestamps = timestamps[order]
    if x_value < positions[0] - 0.5 or x_value > positions[-1] + 0.5:
        return ""

    if USE_CONSTANT_POINT_SPACING:
        nearest = int(np.argmin(np.abs(positions - x_value)))
        timestamp = timestamps[nearest]
    else:
        timestamp = np.interp(
            np.clip(x_value, positions[0], positions[-1]),
            positions,
            timestamps,
        )
    return dt.datetime.fromtimestamp(float(timestamp)).strftime(
        "%Y-%m-%d\n%H:%M:%S"
    )


def _plot_title(plot_mode: str) -> str:
    if plot_mode == "chsh":
        return "CHSH S monitor of optimizer runs"
    if plot_mode == "visibility":
        return "QKD visibility/QBER monitor of optimizer runs"
    return "QKD and CHSH monitor of optimizer runs"


def _time_tick_positions(
    axis_positions: np.ndarray,
    max_ticks: int = 6,
) -> np.ndarray:
    valid = np.asarray(axis_positions, dtype=np.float64)
    valid = np.unique(valid[np.isfinite(valid)])
    if valid.size <= max_ticks:
        return valid
    indices = np.linspace(0, valid.size - 1, max_ticks, dtype=np.int64)
    return valid[indices]


class MeasurementPlot:
    def __init__(self, source_path: Path, plot_mode: str) -> None:
        self.source_path = source_path
        self.plot_mode = plot_mode
        panel_names, height_ratios, figsize = self._layout_for_mode(plot_mode)
        self.figure, axes = plt.subplots(
            len(panel_names),
            1,
            figsize=figsize,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
        )
        axis_list = list(np.atleast_1d(axes))
        self.axes_by_name = dict(zip(panel_names, axis_list))
        self.axes = axis_list
        self.pairs_ax = self.axes_by_name["pairs"]
        self.total_ax = self.axes_by_name.get("total")
        self.visibility_ax = self.axes_by_name.get("visibility")
        self.qber_ax = self.axes_by_name.get("qber")
        self.chsh_ax = self.axes_by_name.get("chsh")
        self.expectation_ax = self.axes_by_name.get("expectations")
        self.time_ax = self.pairs_ax.twiny()
        self._style_time_axis()
        self.figure.canvas.manager.set_window_title("QKD measurement monitor")

    @staticmethod
    def _layout_for_mode(
        plot_mode: str,
    ) -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, float]]:
        if plot_mode == "visibility":
            return ("pairs", "total", "visibility", "qber"), (
                2.2,
                1.2,
                1.4,
                1.2,
            ), (13, 11)
        if plot_mode == "chsh":
            return ("pairs", "total", "chsh", "expectations"), (
                2.2,
                1.2,
                1.3,
                1.2,
            ), (13, 11)
        return (
            "pairs",
            "total",
            "visibility",
            "qber",
            "chsh",
            "expectations",
        ), (
            2.2,
            1.2,
            1.4,
            1.2,
            1.1,
            1.2,
        ), (13, 16)

    def _style_time_axis(self) -> None:
        self.time_ax.patch.set_visible(False)
        self.time_ax.yaxis.set_visible(False)
        self.time_ax.grid(False)
        self.time_ax.set_zorder(self.pairs_ax.get_zorder() + 1)
        self.time_ax.spines["bottom"].set_visible(False)
        self.time_ax.spines["left"].set_visible(False)
        self.time_ax.spines["right"].set_visible(False)
        self.time_ax.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            pad=2,
            labelsize=8,
        )
        self.time_ax.set_xlabel("Measurement date/time", labelpad=8)

    def _update_time_axis(self, series: MeasurementSeries) -> None:
        time_ticks = _time_tick_positions(series.time_axis_positions)
        self.time_ax.clear()
        self._style_time_axis()
        self.time_ax.set_visible(time_ticks.size > 0)
        if time_ticks.size == 0:
            return

        self.time_ax.set_xlim(self.pairs_ax.get_xlim())
        self.time_ax.set_xticks(time_ticks)
        self.time_ax.set_xticklabels(
            [
                _format_absolute_time(
                    value,
                    index,
                    series.time_axis_positions,
                    series.time_axis_timestamps,
                )
                for index, value in enumerate(time_ticks)
            ],
            fontsize=8,
        )
        self.time_ax.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            pad=2,
            labelsize=8,
        )
        self.time_ax.set_xlabel("Measurement date/time", labelpad=8)

    def update(self, series: MeasurementSeries) -> None:
        x = series.relative_seconds
        chsh_x = series.chsh_relative_seconds
        for axis in self.axes:
            axis.clear()

        for label in PAIR_LABELS:
            self.pairs_ax.plot(
                x,
                series.counts[label],
                marker=".",
                markersize=5,
                linestyle="None",
                color=PAIR_COLORS[label],
                label=label,
            )
            if PAIR_TREND_WINDOW > 1:
                self.pairs_ax.plot(
                    x,
                    _rolling_mean(
                        series.counts[label],
                        PAIR_TREND_WINDOW,
                    ),
                    linewidth=1.8,
                    color=PAIR_COLORS[label],
                    alpha=0.9,
                    label="_nolegend_",
                )

        if self.total_ax is not None:
            self.total_ax.plot(
                x,
                series.total_coincidences,
                color="black",
                marker=".",
                markersize=6,
                linestyle="None",
                label="Total",
            )

        if self.visibility_ax is not None:
            self.visibility_ax.errorbar(
                x,
                series.visibility_hv,
                yerr=series.visibility_hv_error,
                color="#0072b2",
                marker=".",
                linestyle="None",
                label="H/V visibility",
                **ERRORBAR_KWARGS,
            )
            self.visibility_ax.errorbar(
                x,
                series.visibility_da,
                yerr=series.visibility_da_error,
                color="#d55e00",
                marker=".",
                linestyle="None",
                label="D/A visibility",
                **ERRORBAR_KWARGS,
            )
            self.visibility_ax.errorbar(
                x,
                series.visibility_total,
                yerr=series.visibility_total_error,
                color="black",
                marker=".",
                linestyle="None",
                label="Total visibility",
                **ERRORBAR_KWARGS,
            )

        if self.qber_ax is not None:
            self.qber_ax.errorbar(
                x,
                series.qber_hv,
                yerr=series.qber_hv_error,
                color="#0072b2",
                marker=".",
                linestyle="None",
                label="H/V QBER",
                **ERRORBAR_KWARGS,
            )
            self.qber_ax.errorbar(
                x,
                series.qber_da,
                yerr=series.qber_da_error,
                color="#d55e00",
                marker=".",
                linestyle="None",
                label="D/A QBER",
                **ERRORBAR_KWARGS,
            )
            self.qber_ax.errorbar(
                x,
                series.qber_total,
                yerr=series.qber_total_error,
                color="black",
                marker=".",
                linestyle="None",
                label="Total QBER",
                **ERRORBAR_KWARGS,
            )

        if self.chsh_ax is not None:
            chsh_from_counts = series.chsh_from_counts
            chsh_from_stored = ~chsh_from_counts
            if np.any(chsh_from_counts):
                self.chsh_ax.errorbar(
                    chsh_x[chsh_from_counts],
                    series.chsh_s_value[chsh_from_counts],
                    yerr=series.chsh_s_error[chsh_from_counts],
                    color="#009e73",
                    marker=".",
                    markersize=6,
                    linestyle="None",
                    label="CHSH S (counts)",
                    **ERRORBAR_KWARGS,
                )
            if np.any(chsh_from_stored):
                self.chsh_ax.plot(
                    chsh_x[chsh_from_stored],
                    series.chsh_s_value[chsh_from_stored],
                    color="#d62728",
                    marker=".",
                    markersize=7,
                    linestyle="None",
                    label="CHSH S (stored)",
                )
            if PAIR_TREND_WINDOW > 1:
                self.chsh_ax.plot(
                    chsh_x,
                    _rolling_mean(series.chsh_s_value, PAIR_TREND_WINDOW),
                    color="#009e73",
                    linewidth=1.8,
                    alpha=0.9,
                    label="_nolegend_",
                )

        if self.expectation_ax is not None:
            for label in CHSH_EXPECTATION_LABELS:
                values = series.chsh_expectations[label]
                valid = np.isfinite(values)
                self.expectation_ax.errorbar(
                    chsh_x[valid],
                    values[valid],
                    yerr=series.chsh_expectation_errors[label][valid],
                    color=CHSH_EXPECTATION_COLORS[label],
                    marker=".",
                    markersize=5,
                    linestyle="None",
                    label=CHSH_EXPECTATION_DISPLAY_LABELS[label],
                    **ERRORBAR_KWARGS,
                )
                if PAIR_TREND_WINDOW > 1:
                    self.expectation_ax.plot(
                        chsh_x,
                        _rolling_mean(values, PAIR_TREND_WINDOW),
                        color=CHSH_EXPECTATION_COLORS[label],
                        linewidth=1.6,
                        alpha=0.85,
                        label="_nolegend_",
                    )

        self.figure.suptitle(_plot_title(self.plot_mode), y=0.995)
        self.pairs_ax.set_title("")
        self.pairs_ax.set_ylabel("Coincidences")
        if self.total_ax is not None:
            self.total_ax.set_ylabel("Total coincidences")
        if self.visibility_ax is not None:
            self.visibility_ax.set_ylabel("Visibility")
            self.visibility_ax.set_ylim(-1.05, 1.05)
            self.visibility_ax.axhline(
                VISIBILITY_TARGET,
                color="#666666",
                linestyle="--",
                linewidth=1,
                label=f"{VISIBILITY_TARGET:.2f}",
            )
        if self.qber_ax is not None:
            self.qber_ax.set_ylabel("QBER")
            self.qber_ax.set_ylim(-0.02, 1.02)
            self.qber_ax.axhline(
                0.11,
                color="#666666",
                linestyle="--",
                linewidth=1,
                label="0.11",
            )
        if self.chsh_ax is not None:
            self.chsh_ax.set_ylabel("CHSH S")
            self.chsh_ax.set_ylim(0.0, 3.0)
            self.chsh_ax.axhline(
                CHSH_CLASSICAL_LIMIT,
                color="#666666",
                linestyle=":",
                linewidth=1,
                label="2.0 classical",
            )
            self.chsh_ax.axhline(
                CHSH_QUANTUM_LIMIT,
                color="#333333",
                linestyle="--",
                linewidth=1,
                label="2√2 quantum limit",
            )
        if self.expectation_ax is not None:
            self.expectation_ax.set_ylabel("CHSH E")
            self.expectation_ax.set_ylim(-1.05, 1.05)
            self.expectation_ax.axhline(
                0.0,
                color="#666666",
                linestyle=":",
                linewidth=1,
                label="0",
            )

        if USE_CONSTANT_POINT_SPACING:
            self.axes[-1].set_xlabel(
                "Measurement number relative to latest displayed measurement"
            )
        else:
            self.axes[-1].set_xlabel(
                "Time relative to latest displayed measurement"
            )
        self.axes[-1].xaxis.set_major_formatter(
            FuncFormatter(_format_x_axis_value)
        )
        self._update_time_axis(series)

        for axis in self.axes:
            axis.grid(True, alpha=0.25)
            axis.legend(loc="best", ncol=4, fontsize=9)

        self.figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
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
    _normalize_plot_range(PLOT_RANGE)
    if PAIR_TREND_WINDOW < 0:
        raise ValueError("PAIR_TREND_WINDOW cannot be negative")

    plot_mode = _normalized_plot_mode()
    plot = MeasurementPlot(csv_path, plot_mode)
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
                series = read_measurements(csv_path, PLOT_RANGE, plot_mode)
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
