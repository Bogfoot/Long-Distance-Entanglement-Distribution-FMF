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

try:
    import nevergrad as ng
except ImportError:
    ng = None


DEFAULT_PHI_PLUS_PAIRS = [
    ("HH", 1, 1),
    ("HV", 1, 2),
    ("VH", 2, 1),
    ("VV", 2, 2),
    ("DD", 3, 3),
    ("DA", 3, 4),
    ("AD", 4, 3),
    ("AA", 4, 4),
]
RESULT_PAIR_ORDER = ("HH", "HV", "VH", "VV", "DD", "DA", "AD", "AA")


@dataclass(frozen=True)
class OptimizerConfig:
    backend: str = "nelder-mead"
    optimize_epcs: str = "both"
    objective_metric: str = "visibility"
    objective_target: float | None = None
    measurement_seconds: float = 5.0
    visibility_target: float = 0.95
    base_step_volts: float = 20.0
    voltage_quantization: float = 0.1
    maximum_voltage: float = 130.0
    settle_seconds: float = 0.05
    stable_sleep_seconds: float = 10 * 60
    max_iterations: int = 200
    voltage_tolerance: float = 0.2
    score_tolerance: float = 1.0e-3
    nevergrad_optimizer: str = "TBPSA"
    nevergrad_budget: int = 100
    nevergrad_seed: int | None = None


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
    current_score: float,
    best_score: float,
    *,
    base_step: float,
) -> float:
    drift = abs(float(current_score) - float(best_score))
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
        self._validate_optimizer_config()

    def _normalized_backend(self) -> str:
        backend = self.config.backend.strip().lower().replace("_", "-")
        if backend in {"nelder-mead", "scipy", "scipy-nelder-mead"}:
            return "nelder-mead"
        if backend == "nevergrad":
            return "nevergrad"
        raise ValueError(
            f"Unknown optimizer backend {self.config.backend!r}; "
            "use 'nelder-mead' or 'nevergrad'"
        )

    def _validate_optimizer_config(self) -> None:
        backend = self._normalized_backend()
        self._active_voltage_indices()
        self._normalized_objective_metric()
        if backend == "nelder-mead" and minimize is None:
            raise RuntimeError(
                "Nelder-Mead optimization was selected, but scipy is not "
                "installed. Install it on Alice with: python -m pip install scipy"
            )
        if backend == "nevergrad":
            if ng is None:
                raise RuntimeError(
                    "Nevergrad optimization was selected, but nevergrad is not "
                    "installed. Install it on Alice with: "
                    "python -m pip install nevergrad"
                )
            if self.config.nevergrad_budget <= 0:
                raise ValueError("nevergrad_budget must be positive")
            if self.config.nevergrad_optimizer not in ng.optimizers.registry:
                raise ValueError(
                    "Unknown Nevergrad optimizer "
                    f"{self.config.nevergrad_optimizer!r}. Inspect "
                    "sorted(nevergrad.optimizers.registry) to list choices."
                )

    def _normalized_objective_metric(self) -> str:
        metric = self.config.objective_metric.strip().lower().replace("-", "_")
        aliases = {
            "visibility": "visibility",
            "total_visibility": "visibility",
            "vis_hv": "vis_HV",
            "hv_visibility": "vis_HV",
            "hv": "vis_HV",
            "vis_da": "vis_DA",
            "da_visibility": "vis_DA",
            "da": "vis_DA",
        }
        if metric not in aliases:
            raise ValueError(
                f"Unknown objective_metric {self.config.objective_metric!r}; "
                "use 'visibility', 'vis_HV', or 'vis_DA'"
            )
        return aliases[metric]

    def _objective_target(self) -> float:
        if self.config.objective_target is not None:
            return float(self.config.objective_target)
        return float(self.config.visibility_target)

    def _objective_score(self, measurement: PhiPlusCorrectionResult) -> float:
        metric = self._normalized_objective_metric()
        if metric == "visibility":
            return float(measurement.visibility)
        if metric == "vis_HV":
            return float(measurement.basis_visibility["HV"])
        return float(measurement.basis_visibility["DA"])

    def _active_voltage_indices(self) -> np.ndarray:
        scope = self.config.optimize_epcs.strip().lower()
        if scope == "alice":
            return np.arange(0, 4, dtype=np.int64)
        if scope == "bob":
            return np.arange(4, 8, dtype=np.int64)
        if scope == "both":
            return np.arange(0, 8, dtype=np.int64)
        raise ValueError(
            f"Unknown optimize_epcs value {self.config.optimize_epcs!r}; "
            "use 'alice', 'bob', or 'both'"
        )

    def _full_voltage_vector(
        self,
        base_voltages: np.ndarray,
        active_voltages: np.ndarray,
    ) -> np.ndarray:
        full = np.asarray(base_voltages, dtype=float).copy()
        full[self._active_voltage_indices()] = self._quantize(active_voltages)
        return full

    def monitor_forever(self, monitor_seconds: float) -> None:
        print(
            "[Alice] Starting synchronized Phi+ EPC optimizer loop: "
            f"backend={self._normalized_backend()}, "
            f"optimizer={self._optimizer_name()}, "
            f"EPCs={self.config.optimize_epcs}, "
            f"objective={self._normalized_objective_metric()}, "
            f"target={self._objective_target():.3f}"
        )
        state = self._load_state()
        qber_state = state.get("qber", {})
        best_voltages = np.asarray(
            qber_state.get("best_V", [65.0] * 8),
            dtype=float,
        )
        objective_metric = self._normalized_objective_metric()
        stored_metric = qber_state.get("objective_metric")
        if stored_metric == objective_metric:
            best_score = float(
                qber_state.get(
                    "best_score",
                    qber_state.get("best_visibility", -np.inf),
                )
            )
        else:
            best_score = -np.inf
        target = self._objective_target()
        loop_index = 0

        while True:
            loop_index += 1
            print(
                "[Alice] Optimizer monitor using current hardware state; "
                f"stored best voltages={best_voltages.tolist()}"
            )
            measurement = self.measure(monitor_seconds)
            score = self._objective_score(measurement)
            print(
                f"[Alice] Optimizer check #{loop_index}: "
                f"objective {objective_metric}={score:.3f}, "
                f"total visibility={measurement.visibility:.3f}, "
                f"QBER={100.0 * measurement.qber_total:.2f}%, "
                f"total coincidences={measurement.total_coincidences}, "
                f"stored best score={best_score:.3f}"
            )

            if score >= target:
                print(
                    f"[Alice] Objective target reached; sleeping "
                    f"{self.config.stable_sleep_seconds:g} s"
                )
                time.sleep(self.config.stable_sleep_seconds)
                continue

            step = choose_qber_optimizer_step(
                score,
                best_score,
                base_step=self.config.base_step_volts,
            )
            print(
                f"[Alice] Objective {objective_metric} below target by "
                f"{target - score:.3f}; optimizing with step={step:.1f} V"
            )
            best_voltages, best_score = self._optimize(
                best_voltages,
                step,
            )
            self._save_best_state(best_voltages, best_score)

    def _optimize(
        self,
        start_voltages: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, float]:
        backend = self._normalized_backend()
        if backend == "nelder-mead":
            return self._optimize_nelder_mead(start_voltages, step)
        return self._optimize_nevergrad(start_voltages, step)

    def _optimize_nelder_mead(
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
        active_indices = self._active_voltage_indices()
        active_start = start_voltages[active_indices]
        best_voltages = start_voltages.copy()
        best_score = -np.inf
        evaluation_index = 0
        previous_voltages: np.ndarray | None = None

        def objective(active_values: np.ndarray) -> float:
            nonlocal best_voltages, best_score
            nonlocal evaluation_index, previous_voltages
            voltages = self._full_voltage_vector(
                start_voltages,
                active_values,
            )
            evaluation_index += 1
            repeated = (
                previous_voltages is not None
                and np.array_equal(voltages, previous_voltages)
            )
            print(
                f"[Alice] Optimizer candidate #{evaluation_index}: "
                f"{voltages.tolist()}"
                + (" (same as previous candidate)" if repeated else "")
            )
            previous_voltages = voltages.copy()
            self.apply_voltages(voltages.tolist())
            if self.config.settle_seconds > 0:
                time.sleep(self.config.settle_seconds)

            measurement = self.measure(self.config.measurement_seconds)
            score = self._objective_score(measurement)
            self._log_iteration(
                voltages,
                measurement,
                backend="nelder-mead",
                optimizer_name="Nelder-Mead",
                evaluation_index=evaluation_index,
            )

            if score > best_score:
                best_score = score
                best_voltages = voltages.copy()
                self._save_best_state(best_voltages, best_score)

            target = self._objective_target()
            if score >= target:
                print(
                    f"[Alice] Optimization reached "
                    f"{self._normalized_objective_metric()}={score:.3f} "
                    f">= {target:.3f}"
                )
                raise StopIteration
            return -score

        parameter_count = int(active_start.size)
        initial_simplex = np.vstack(
            [active_start]
            + [
                self._quantize(
                    active_start + step * np.eye(parameter_count)[index]
                )
                for index in range(parameter_count)
            ]
        )

        try:
            minimize(
                objective,
                active_start,
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

        print(
            "[Alice] Restoring optimizer best voltages: "
            f"{best_voltages.tolist()} "
            f"({self._normalized_objective_metric()}={best_score:.3f})"
        )
        self.apply_voltages(best_voltages.tolist())
        return best_voltages, float(best_score)

    def _optimize_nevergrad(
        self,
        start_voltages: np.ndarray,
        step: float,
    ) -> tuple[np.ndarray, float]:
        if ng is None:
            raise RuntimeError(
                "Nevergrad optimization was selected, but nevergrad is not "
                "installed. Install it on Alice with: python -m pip install nevergrad"
            )
        if self.config.nevergrad_budget <= 0:
            raise ValueError("nevergrad_budget must be positive")

        optimizer_name = self.config.nevergrad_optimizer
        optimizer_class = ng.optimizers.registry.get(optimizer_name)
        if optimizer_class is None:
            available = ", ".join(sorted(ng.optimizers.registry))
            raise ValueError(
                f"Unknown Nevergrad optimizer {optimizer_name!r}. "
                f"Available optimizers: {available}"
            )

        start_voltages = self._quantize(start_voltages)
        active_indices = self._active_voltage_indices()
        active_start = start_voltages[active_indices]
        parametrization = (
            ng.p.Array(init=active_start)
            .set_bounds(0.0, self.config.maximum_voltage)
            .set_mutation(sigma=step)
        )
        if self.config.nevergrad_seed is not None:
            parametrization.random_state.seed(self.config.nevergrad_seed)

        optimizer = optimizer_class(
            parametrization=parametrization,
            budget=self.config.nevergrad_budget,
            num_workers=1,
        )
        optimizer.suggest(active_start.tolist())

        best_voltages = start_voltages.copy()
        best_score = -np.inf
        previous_voltages: np.ndarray | None = None

        print(
            f"[Alice] Starting Nevergrad optimizer {optimizer_name} with "
            f"budget={self.config.nevergrad_budget}, step={step:.1f} V"
        )

        for evaluation_index in range(1, self.config.nevergrad_budget + 1):
            candidate = optimizer.ask()
            voltages = self._full_voltage_vector(
                start_voltages,
                np.asarray(candidate.value, dtype=float),
            )
            repeated = (
                previous_voltages is not None
                and np.array_equal(voltages, previous_voltages)
            )
            print(
                f"[Alice] Nevergrad {optimizer_name} candidate "
                f"#{evaluation_index}: {voltages.tolist()}"
                + (" (same as previous candidate)" if repeated else "")
            )
            previous_voltages = voltages.copy()

            self.apply_voltages(voltages.tolist())
            if self.config.settle_seconds > 0:
                time.sleep(self.config.settle_seconds)

            measurement = self.measure(self.config.measurement_seconds)
            score = self._objective_score(measurement)
            optimizer.tell(candidate, -score)
            self._log_iteration(
                voltages,
                measurement,
                backend="nevergrad",
                optimizer_name=optimizer_name,
                evaluation_index=evaluation_index,
            )

            if score > best_score:
                best_score = score
                best_voltages = voltages.copy()
                self._save_best_state(best_voltages, best_score)

            target = self._objective_target()
            if score >= target:
                print(
                    f"[Alice] Nevergrad reached "
                    f"{self._normalized_objective_metric()}={score:.3f} "
                    f">= {target:.3f}"
                )
                break

        print(
            "[Alice] Restoring Nevergrad best measured voltages: "
            f"{best_voltages.tolist()} "
            f"({self._normalized_objective_metric()}={best_score:.3f})"
        )
        self.apply_voltages(best_voltages.tolist())
        return best_voltages, float(best_score)

    def _quantize(self, values: np.ndarray) -> np.ndarray:
        step = self.config.voltage_quantization
        quantized = np.round(np.asarray(values, dtype=float) / step) * step
        return np.clip(quantized, 0.0, self.config.maximum_voltage)

    def _optimizer_name(self) -> str:
        if self._normalized_backend() == "nevergrad":
            return self.config.nevergrad_optimizer
        return "Nelder-Mead"

    def _load_state(self) -> dict[str, Any]:
        path = self.log_paths.optimizer_state_json
        if not path.exists():
            state = {
                "qber": {
                    "best_V": [65.0] * 8,
                    "best_score": self._objective_target(),
                    "objective_metric": self._normalized_objective_metric(),
                    "optimizer_backend": self._normalized_backend(),
                    "optimizer_name": self._optimizer_name(),
                    "optimize_epcs": self.config.optimize_epcs,
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
        best_score: float,
    ) -> None:
        state = {
            "qber": {
                "best_V": best_voltages.tolist(),
                "best_score": float(best_score),
                "objective_metric": self._normalized_objective_metric(),
                "optimizer_backend": self._normalized_backend(),
                "optimizer_name": self._optimizer_name(),
                "optimize_epcs": self.config.optimize_epcs,
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
        *,
        backend: str,
        optimizer_name: str,
        evaluation_index: int,
    ) -> None:
        counts = measurement.counts
        _append_csv_row(
            self.log_paths.optimizer_iterations_csv,
            {
                "timestamp": time.time(),
                "optimizer_backend": backend,
                "optimizer_name": optimizer_name,
                "optimize_epcs": self.config.optimize_epcs,
                "evaluation_index": evaluation_index,
                "objective_metric": self._normalized_objective_metric(),
                "objective_score": self._objective_score(measurement),
                "objective_target": self._objective_target(),
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
