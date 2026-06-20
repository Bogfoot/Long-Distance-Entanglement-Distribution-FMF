from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from qkd_names import current_utc_iso
from qkd_sync import SyncCoincidenceAnalysis

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None


DEFAULT_PHI_PLUS_PAIRS = [
    ("HH", 4, 4),
    ("HV", 4, 2),
    ("VH", 2, 4),
    ("VV", 2, 2),
    ("DD", 1, 1),
    ("DA", 1, 3),
    ("AD", 3, 1),
    ("AA", 3, 3),
]
RESULT_PAIR_ORDER = ("HH", "HV", "VH", "VV", "DD", "DA", "AD", "AA")


@dataclass(frozen=True)
class OptimizerConfig:
    measurement_seconds: float = 5.0
    visibility_target: float = 0.95
    base_step_volts: float = 25.0
    voltage_quantization: float = 0.1
    maximum_voltage: float = 130.0
    settle_seconds: float = 0.05
    stable_sleep_seconds: float = 10 * 60
    max_iterations: int = 200
    voltage_tolerance: float = 0.2
    score_tolerance: float = 1.0e-3


@dataclass(frozen=True)
class CorrectionLogPaths:
    results_csv: Path
    optimizer_state_json: Path
    optimizer_iterations_csv: Path


@dataclass
class PhiPlusCorrectionResult:
    sync: SyncCoincidenceAnalysis
    basis_visibility: dict[str, float]
    basis_qber: dict[str, float]
    basis_contrast: dict[str, float]
    visibility: float
    qber_total: float
    total_contrast: float
    total_coincidences: int
    optimization_score: float

    @property
    def counts(self) -> dict[str, int]:
        return {
            name: result.count for name, result in self.sync.results_by_name.items()
        }

    @property
    def delays_ps(self) -> dict[str, float]:
        return {
            name: float(result.best_delay_ps)
            for name, result in self.sync.results_by_name.items()
        }

    @property
    def accidentals(self) -> dict[str, float]:
        return {
            name: float(result.accidental_estimate)
            for name, result in self.sync.results_by_name.items()
        }

    def to_result_row(self) -> dict[str, float | int | str]:
        counters = self.sync.clock_map.counters
        skew = self.sync.clock_map.segment_skew_ppm
        row: dict[str, float | int | str] = {
            "timestamp": time.time(),
            "alice_file": self.sync.alice_path.name,
            "bob_file": self.sync.bob_path.name,
            "overlap_duration_sec": self.sync.overlap_duration_s,
            "sync_common_markers": int(counters.size),
            "sync_first_counter": int(counters[0]) if counters.size else -1,
            "sync_last_counter": int(counters[-1]) if counters.size else -1,
            "sync_skew_ppm_mean": float(np.mean(skew)) if skew.size else 0.0,
            "sync_skew_ppm_std": float(np.std(skew)) if skew.size else 0.0,
            "visibility": float(self.visibility),
            "vis_HV": float(self.basis_visibility["HV"]),
            "vis_DA": float(self.basis_visibility["DA"]),
            "QBER_total": float(self.qber_total),
            "QBER_HV": float(self.basis_qber["HV"]),
            "QBER_DA": float(self.basis_qber["DA"]),
            "contrast_HV": float(self.basis_contrast["HV"]),
            "contrast_DA": float(self.basis_contrast["DA"]),
            "total_contrast": float(self.total_contrast),
            "total_coincidences": int(self.total_coincidences),
            "optimization_score": float(self.optimization_score),
        }
        results_by_name = self.sync.results_by_name
        for label in RESULT_PAIR_ORDER:
            result = results_by_name[label]
            row[f"C_{label}"] = result.count
            row[f"accidental_{label}"] = float(result.accidental_estimate)
            row[f"delay_{label}_ps"] = float(result.best_delay_ps)
            row[f"alice_events_{label}"] = int(result.alice_event_count)
            row[f"bob_events_{label}"] = int(result.bob_event_count)
        return row


def _basis_quality(correlated_count: int, error_count: int) -> tuple[float, float]:
    total = correlated_count + error_count
    if total <= 0:
        return 0.0, 0.5
    visibility = (correlated_count - error_count) / total
    qber = error_count / total
    return float(visibility), float(qber)


def _contrast(numerator: int, denominator: int) -> float:
    return float(numerator / (denominator + 1.0e-9))


def analyze_phi_plus_coincidences(
    sync: SyncCoincidenceAnalysis,
) -> PhiPlusCorrectionResult:
    """Calculate Phi+ visibility and QBER from synchronized coincidences."""
    counts = {name: result.count for name, result in sync.results_by_name.items()}
    required_labels = [label for label, _, _ in DEFAULT_PHI_PLUS_PAIRS]
    missing_labels = [label for label in required_labels if label not in counts]
    if missing_labels:
        raise ValueError(
            "Phi+ correction cannot run because synchronized coincidence "
            f"results are missing labels: {', '.join(missing_labels)}"
        )

    hv_correlated = counts["HH"] + counts["VV"]
    hv_errors = counts["HV"] + counts["VH"]
    da_correlated = counts["DD"] + counts["AA"]
    da_errors = counts["DA"] + counts["AD"]
    total_correlated = hv_correlated + da_correlated
    total_errors = hv_errors + da_errors

    hv_visibility, hv_qber = _basis_quality(hv_correlated, hv_errors)
    da_visibility, da_qber = _basis_quality(da_correlated, da_errors)
    visibility = float((hv_visibility + da_visibility) / 2.0)
    qber_total = float((hv_qber + da_qber) / 2.0)

    return PhiPlusCorrectionResult(
        sync=sync,
        basis_visibility={"HV": hv_visibility, "DA": da_visibility},
        basis_qber={"HV": hv_qber, "DA": da_qber},
        basis_contrast={
            "HV": _contrast(hv_correlated, hv_errors),
            "DA": _contrast(da_correlated, da_errors),
        },
        visibility=visibility,
        qber_total=qber_total,
        total_contrast=_contrast(total_correlated, total_errors),
        total_coincidences=int(total_correlated + total_errors),
        optimization_score=visibility,
    )


def append_correction_result(
    result: PhiPlusCorrectionResult,
    output_path: Path,
) -> None:
    _append_csv_row(output_path, result.to_result_row())


def choose_qber_optimizer_step(
    current_visibility: float,
    best_visibility: float,
    *,
    base_step: float,
) -> float:
    drift = abs(float(current_visibility) - float(best_visibility))
    scale = 0.2 if drift < 0.05 else 0.5 if drift < 0.5 else 1.0
    return float(base_step * scale)


MeasurementCallback = Callable[[float], PhiPlusCorrectionResult]
VoltageCallback = Callable[[list[float]], None]


class PhiPlusOptimizer:
    def __init__(
        self,
        config: OptimizerConfig,
        log_paths: CorrectionLogPaths,
        apply_voltages: VoltageCallback,
        measure: MeasurementCallback,
    ) -> None:
        self.config = config
        self.log_paths = log_paths
        self.apply_voltages = apply_voltages
        self.measure = measure

    def monitor_forever(self, monitor_seconds: float) -> None:
        print("[Alice] Starting synchronized Phi+ QBER optimizer loop")
        state = self._load_state()
        qber_state = state.get("qber", {})
        best_voltages = np.asarray(
            qber_state.get("best_V", [65.0] * 8),
            dtype=float,
        )
        best_visibility = float(qber_state.get("best_visibility", 0.95))
        loop_index = 0

        while True:
            loop_index += 1
            measurement = self.measure(monitor_seconds)
            visibility = measurement.visibility
            print(
                f"[Alice] Optimizer check #{loop_index}: "
                f"visibility={visibility:.3f}, "
                f"QBER={100.0 * measurement.qber_total:.2f}%, "
                f"total coincidences={measurement.total_coincidences}, "
                f"stored best={best_visibility:.3f}"
            )

            if visibility >= self.config.visibility_target:
                print(
                    f"[Alice] Phi+ state stable; sleeping "
                    f"{self.config.stable_sleep_seconds:g} s"
                )
                time.sleep(self.config.stable_sleep_seconds)
                continue

            step = choose_qber_optimizer_step(
                visibility,
                best_visibility,
                base_step=self.config.base_step_volts,
            )
            print(
                f"[Alice] Visibility below target by "
                f"{self.config.visibility_target - visibility:.3f}; "
                f"optimizing with step={step:.1f} V"
            )
            best_voltages, best_visibility = self._optimize(
                best_voltages,
                step,
            )
            self._save_best_state(best_voltages, best_visibility)

    def _optimize(
        self,
        start_voltages: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, float]:
        if minimize is None:
            raise RuntimeError(
                "Phi+ optimization requires scipy.optimize.minimize, "
                "but scipy is not installed"
            )

        start_voltages = self._quantize(start_voltages)
        best_voltages = start_voltages.copy()
        best_visibility = -1.0

        def objective(values: np.ndarray) -> float:
            nonlocal best_voltages, best_visibility
            voltages = self._quantize(values)
            self.apply_voltages(voltages.tolist())
            if self.config.settle_seconds > 0:
                time.sleep(self.config.settle_seconds)

            measurement = self.measure(self.config.measurement_seconds)
            visibility = measurement.visibility
            self._log_iteration(voltages, measurement)

            if visibility > best_visibility:
                best_visibility = visibility
                best_voltages = voltages.copy()
                self._save_best_state(best_voltages, best_visibility)

            if visibility >= self.config.visibility_target:
                print(
                    f"[Alice] QBER optimization reached visibility "
                    f"{visibility:.3f} >= {self.config.visibility_target:.3f}"
                )
                raise StopIteration
            return -visibility

        parameter_count = int(start_voltages.size)
        initial_simplex = np.vstack(
            [start_voltages]
            + [
                self._quantize(
                    start_voltages + step * np.eye(parameter_count)[index]
                )
                for index in range(parameter_count)
            ]
        )

        try:
            minimize(
                objective,
                start_voltages,
                method="Nelder-Mead",
                options={
                    "maxiter": self.config.max_iterations,
                    "xatol": self.config.voltage_tolerance,
                    "fatol": self.config.score_tolerance,
                    "disp": True,
                    "initial_simplex": initial_simplex,
                },
            )
        except StopIteration:
            pass

        self.apply_voltages(best_voltages.tolist())
        return best_voltages, best_visibility

    def _quantize(self, values: np.ndarray) -> np.ndarray:
        step = self.config.voltage_quantization
        quantized = np.round(np.asarray(values, dtype=float) / step) * step
        return np.clip(quantized, 0.0, self.config.maximum_voltage)

    def _load_state(self) -> dict[str, Any]:
        path = self.log_paths.optimizer_state_json
        if not path.exists():
            state = {
                "qber": {
                    "best_V": [65.0] * 8,
                    "best_visibility": self.config.visibility_target,
                },
                "last_update": current_utc_iso(),
            }
            self._write_state(state)
            return state

        with path.open("r") as handle:
            return json.load(handle)

    def _save_best_state(
        self,
        best_voltages: np.ndarray,
        best_visibility: float,
    ) -> None:
        state = {
            "qber": {
                "best_V": best_voltages.tolist(),
                "best_visibility": float(best_visibility),
                "last_update": current_utc_iso(),
            },
            "last_update": current_utc_iso(),
        }
        self._write_state(state)

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self.log_paths.optimizer_state_json
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(state, handle, indent=2)
        print(f"[Alice] Saved optimizer state to {path}")

    def _log_iteration(
        self,
        voltages: np.ndarray,
        measurement: PhiPlusCorrectionResult,
    ) -> None:
        counts = measurement.counts
        _append_csv_row(
            self.log_paths.optimizer_iterations_csv,
            {
                "timestamp": time.time(),
                "voltages": json.dumps(voltages.tolist()),
                "visibility": float(measurement.visibility),
                "QBER": float(measurement.qber_total),
                "total_coincidences": int(measurement.total_coincidences),
                "C_HH": counts["HH"],
                "C_VV": counts["VV"],
                "C_DD": counts["DD"],
                "C_AA": counts["AA"],
                "C_HV": counts["HV"],
                "C_VH": counts["VH"],
                "C_DA": counts["DA"],
                "C_AD": counts["AD"],
            },
        )


def _append_csv_row(
    output_path: Path,
    row: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    requested_fieldnames = list(row)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    if write_header:
        fieldnames = requested_fieldnames
    else:
        fieldnames = _extend_csv_schema(output_path, requested_fieldnames)

    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _extend_csv_schema(
    output_path: Path,
    requested_fieldnames: list[str],
) -> list[str]:
    with output_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_fieldnames = list(reader.fieldnames or [])
        existing_rows = list(reader)

    combined_fieldnames = existing_fieldnames + [
        name for name in requested_fieldnames if name not in existing_fieldnames
    ]
    if combined_fieldnames == existing_fieldnames:
        return combined_fieldnames

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=combined_fieldnames)
        writer.writeheader()
        writer.writerows(existing_rows)
    temporary_path.replace(output_path)
    return combined_fieldnames
