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
CHSH_PAIR_ORDER = (
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
@dataclass(frozen=True)
class OptimizerConfig:
    backend: str = "nelder-mead"
    optimize_epcs: str = "both"
    objective_metric: str = "visibility"
    objective_target: float | None = None
    secondary_objective_metric: str | None = None
    secondary_objective_target: float | None = None
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
    raw_save_interval_steps: int = 0


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
            "timestamp": (
                self.sync.measurement_timestamp_s
                if self.sync.measurement_timestamp_s is not None
                else time.time()
            ),
            "alice_file": self.sync.alice_path.name,
            "bob_file": self.sync.bob_path.name,
            "overlap_duration_sec": self.sync.overlap_duration_s,
            "analysis_exposure_count": self.sync.exposure_count,
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


@dataclass
class CHSHCorrectionResult:
    sync: SyncCoincidenceAnalysis
    correlations: dict[str, float]
    S_signed: float
    S_value: float
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
            "timestamp": (
                self.sync.measurement_timestamp_s
                if self.sync.measurement_timestamp_s is not None
                else time.time()
            ),
            "alice_file": self.sync.alice_path.name,
            "bob_file": self.sync.bob_path.name,
            "overlap_duration_sec": self.sync.overlap_duration_s,
            "analysis_exposure_count": self.sync.exposure_count,
            "sync_common_markers": int(counters.size),
            "sync_first_counter": int(counters[0]) if counters.size else -1,
            "sync_last_counter": int(counters[-1]) if counters.size else -1,
            "sync_skew_ppm_mean": float(np.mean(skew)) if skew.size else 0.0,
            "sync_skew_ppm_std": float(np.std(skew)) if skew.size else 0.0,
            "CHSH_S_signed": float(self.S_signed),
            "CHSH_S_value": float(self.S_value),
            "CHSH_E_ab": float(self.correlations["E_ab"]),
            "CHSH_E_abp": float(self.correlations["E_abp"]),
            "CHSH_E_apb": float(self.correlations["E_apb"]),
            "CHSH_E_apbp": float(self.correlations["E_apbp"]),
            "total_coincidences": int(self.total_coincidences),
            "optimization_score": float(self.optimization_score),
        }
        results_by_name = self.sync.results_by_name
        for label in CHSH_PAIR_ORDER:
            result = results_by_name[label]
            row[f"C_{label}"] = result.count
            row[f"accidental_{label}"] = float(result.accidental_estimate)
            row[f"delay_{label}_ps"] = float(result.best_delay_ps)
            row[f"alice_events_{label}"] = int(result.alice_event_count)
            row[f"bob_events_{label}"] = int(result.bob_event_count)
        return row


CorrectionResult = PhiPlusCorrectionResult | CHSHCorrectionResult


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


def analyze_chsh_s_coincidences(
    sync: SyncCoincidenceAnalysis,
) -> CHSHCorrectionResult:
    """Calculate CHSH S from synchronized coincidences."""
    counts = {name: result.count for name, result in sync.results_by_name.items()}
    missing_labels = [label for label in CHSH_PAIR_ORDER if label not in counts]
    if missing_labels:
        raise ValueError(
            "CHSH correction cannot run because synchronized coincidence "
            f"results are missing labels: {', '.join(missing_labels)}"
        )

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
    s_signed = float(e_ab - e_abp + e_apb + e_apbp)
    s_value = abs(s_signed)

    return CHSHCorrectionResult(
        sync=sync,
        correlations={
            "E_ab": e_ab,
            "E_abp": e_abp,
            "E_apb": e_apb,
            "E_apbp": e_apbp,
        },
        S_signed=s_signed,
        S_value=s_value,
        total_coincidences=int(sum(counts[label] for label in CHSH_PAIR_ORDER)),
        optimization_score=s_value,
    )

def append_correction_result(
    result: CorrectionResult,
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
    scale = 0.2 if drift < 0.05 else 0.5 if drift < 0.25 else 1.0
    return float(base_step * scale)


MeasurementCallback = Callable[..., CorrectionResult]
RawCleanupCallback = Callable[[CorrectionResult], None]
VoltageCallback = Callable[[list[float]], None]


class PhiPlusOptimizer:
    def __init__(
        self,
        config: OptimizerConfig,
        log_paths: CorrectionLogPaths,
        apply_voltages: VoltageCallback,
        measure: MeasurementCallback,
        secondary_measure: MeasurementCallback | None = None,
        cleanup_raw: RawCleanupCallback | None = None,
    ) -> None:
        self.config = config
        self.log_paths = log_paths
        self.apply_voltages = apply_voltages
        self.measure = measure
        self.secondary_measure = secondary_measure
        self.cleanup_raw = cleanup_raw
        self._active_objective_metric = "visibility"
        self._active_measure = measure
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
        self._normalized_secondary_objective_metric()
        if (
            self._normalized_secondary_objective_metric() is not None
            and self.secondary_measure is None
        ):
            raise ValueError(
                "secondary_measure is required when "
                "secondary_objective_metric is configured"
            )
        if self.config.raw_save_interval_steps < 0:
            raise ValueError("raw_save_interval_steps cannot be negative")
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

    @staticmethod
    def _normalize_objective_metric(metric_text: str) -> str:
        metric = metric_text.strip().lower().replace("-", "_")
        aliases = {
            "visibility": "visibility",
            "total_visibility": "visibility",
            "vis_hv": "vis_HV",
            "hv_visibility": "vis_HV",
            "hv": "vis_HV",
            "vis_da": "vis_DA",
            "da_visibility": "vis_DA",
            "da": "vis_DA",
            "chsh": "chsh_s",
            "chsh_s": "chsh_s",
            "s": "chsh_s",
            "s_value": "chsh_s",
        }
        if metric not in aliases:
            raise ValueError(
                f"Unknown objective metric {metric_text!r}; use "
                "'visibility', 'vis_HV', 'vis_DA', or 'chsh_s'"
            )
        return aliases[metric]

    def _normalized_objective_metric(self) -> str:
        return self._normalize_objective_metric(self.config.objective_metric)

    def _normalized_secondary_objective_metric(self) -> str | None:
        if self.config.secondary_objective_metric is None:
            return None
        return self._normalize_objective_metric(
            self.config.secondary_objective_metric
        )

    def _active_state_key(self) -> str:
        return "chsh" if self._active_objective_metric == "chsh_s" else "qber"

    def _default_target_for_metric(self, metric: str) -> float:
        if metric == "chsh_s":
            return 2.5
        return float(self.config.visibility_target)

    def _target_for_metric(self, metric: str) -> float:
        primary_metric = self._normalized_objective_metric()
        secondary_metric = self._normalized_secondary_objective_metric()
        if metric == primary_metric:
            if self.config.objective_target is not None:
                return float(self.config.objective_target)
            return self._default_target_for_metric(metric)
        if metric == secondary_metric:
            if self.config.secondary_objective_target is not None:
                return float(self.config.secondary_objective_target)
            return self._default_target_for_metric(metric)
        return self._default_target_for_metric(metric)

    def _objective_target(self) -> float:
        return self._target_for_metric(self._active_objective_metric)

    def _objective_score(self, measurement: CorrectionResult) -> float:
        metric = self._active_objective_metric
        if metric == "chsh_s":
            if not isinstance(measurement, CHSHCorrectionResult):
                raise TypeError("CHSH objective requires CHSH measurements")
            return float(measurement.S_value)
        if not isinstance(measurement, PhiPlusCorrectionResult):
            raise TypeError("Visibility objective requires Phi+ measurements")
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
        primary_metric = self._normalized_objective_metric()
        secondary_metric = self._normalized_secondary_objective_metric()
        print(
            "[Alice] Starting synchronized EPC optimizer loop: "
            f"backend={self._normalized_backend()}, "
            f"optimizer={self._optimizer_name()}, "
            f"EPCs={self.config.optimize_epcs}, "
            f"primary={primary_metric}, "
            f"primary_target={self._target_for_metric(primary_metric):.3f}"
            + (
                f", secondary={secondary_metric}, "
                f"secondary_target={self._target_for_metric(secondary_metric):.3f}"
                if secondary_metric is not None
                else ""
            )
        )
        loop_index = 0
        fallback_voltages = np.asarray([65.0] * 8, dtype=float)

        while True:
            loop_index += 1
            primary_result = self._run_monitor_phase(
                phase_key="qber",
                objective_metric=primary_metric,
                measure=self.measure,
                monitor_seconds=monitor_seconds,
                loop_index=loop_index,
                fallback_voltages=fallback_voltages,
            )
            fallback_voltages = primary_result["best_voltages"]
            primary_target_met = primary_result["score"] >= primary_result["target"]

            secondary_result = None
            if secondary_metric is not None:
                if self.secondary_measure is None:
                    raise RuntimeError("secondary_measure is not configured")
                secondary_result = self._run_monitor_phase(
                    phase_key="chsh",
                    objective_metric=secondary_metric,
                    measure=self.secondary_measure,
                    monitor_seconds=monitor_seconds,
                    loop_index=loop_index,
                    fallback_voltages=fallback_voltages,
                )
                fallback_voltages = secondary_result["best_voltages"]

            if secondary_metric is None:
                if primary_target_met:
                    print(
                        f"[Alice] Objective target reached; sleeping "
                        f"{self.config.stable_sleep_seconds:g} s"
                    )
                    time.sleep(self.config.stable_sleep_seconds)
                continue

            secondary_target_met = (
                secondary_result is not None
                and secondary_result["score"] >= secondary_result["target"]
            )
            if primary_target_met and secondary_target_met:
                print(
                    f"[Alice] Primary and secondary targets reached; sleeping "
                    f"{self.config.stable_sleep_seconds:g} s"
                )
                time.sleep(self.config.stable_sleep_seconds)

    def _set_active_phase(
        self,
        objective_metric: str,
        measure: MeasurementCallback,
    ) -> None:
        self._active_objective_metric = objective_metric
        self._active_measure = measure

    def _phase_state(
        self,
        state: dict[str, Any],
        phase_key: str,
        fallback_voltages: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        phase_state = state.get(phase_key, {})
        best_voltages = np.asarray(
            phase_state.get("best_V", fallback_voltages.tolist()),
            dtype=float,
        )
        if phase_state.get("objective_metric") == self._active_objective_metric:
            best_score = float(phase_state.get("best_score", -np.inf))
        else:
            best_score = -np.inf
        return best_voltages, best_score

    def _run_monitor_phase(
        self,
        *,
        phase_key: str,
        objective_metric: str,
        measure: MeasurementCallback,
        monitor_seconds: float,
        loop_index: int,
        fallback_voltages: np.ndarray,
        optimize_if_needed: bool = True,
    ) -> dict[str, Any]:
        self._set_active_phase(objective_metric, measure)
        state = self._load_state()
        best_voltages, best_score = self._phase_state(
            state,
            phase_key,
            fallback_voltages,
        )
        target = self._objective_target()

        measurement = self._measure_optimizer_step(
            monitor_seconds,
            evaluation_index=None,
            keep_raw=self.cleanup_raw is not None,
            defer_raw_cleanup=self.cleanup_raw is not None,
        )
        score = self._objective_score(measurement)
        target_met = score >= target
        print(
            f"[Optimizer] {phase_key} check #{loop_index} | "
            f"{objective_metric}={score:.3f} | "
            f"best={best_score:.3f} | target={target:.3f}"
        )
        self._finish_optimizer_raw_measurement(
            measurement,
            keep_raw=target_met,
            reason="target reached" if target_met else "check below target",
        )

        if target_met or not optimize_if_needed:
            return {
                "best_voltages": best_voltages,
                "best_score": best_score,
                "score": score,
                "target": target,
            }

        step = choose_qber_optimizer_step(
            score,
            best_score,
            base_step=self.config.base_step_volts,
        )
        print(
            f"[Optimizer] Starting {phase_key} search | step={step:.1f} V"
        )
        best_voltages, best_score = self._optimize(best_voltages, step)
        self._save_best_state(best_voltages, best_score)
        return {
            "best_voltages": best_voltages,
            "best_score": best_score,
            "score": best_score,
            "target": target,
        }

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
                f"[Optimizer] Candidate #{evaluation_index} | "
                f"voltages={voltages.tolist()}"
                + (" | repeated" if repeated else "")
            )
            previous_voltages = voltages.copy()
            self.apply_voltages(voltages.tolist())
            if self.config.settle_seconds > 0:
                time.sleep(self.config.settle_seconds)

            interval_saved = self._should_save_raw_optimizer_step(evaluation_index)
            measurement = self._measure_optimizer_step(
                self.config.measurement_seconds,
                evaluation_index=evaluation_index,
                keep_raw=interval_saved or self.cleanup_raw is not None,
                defer_raw_cleanup=self.cleanup_raw is not None,
            )
            score = self._objective_score(measurement)
            target = self._objective_target()
            target_met = score >= target
            raw_saved = interval_saved or target_met
            self._finish_optimizer_raw_measurement(
                measurement,
                keep_raw=raw_saved,
                reason=(
                    "target reached" if target_met else "save interval"
                ),
            )
            self._log_iteration(
                voltages,
                measurement,
                backend="nelder-mead",
                optimizer_name="Nelder-Mead",
                evaluation_index=evaluation_index,
                raw_saved=raw_saved,
            )

            if score > best_score:
                best_score = score
                best_voltages = voltages.copy()
                self._save_best_state(best_voltages, best_score)

            if target_met:
                print(
                    f"[Alice] Optimization reached "
                    f"{self._active_objective_metric}={score:.3f} "
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
            f"({self._active_objective_metric}={best_score:.3f})"
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
            f"[Optimizer] {optimizer_name} | "
            f"budget={self.config.nevergrad_budget} | step={step:.1f} V"
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
                f"[Optimizer] Candidate #{evaluation_index} | "
                f"voltages={voltages.tolist()}"
                + (" | repeated" if repeated else "")
            )
            previous_voltages = voltages.copy()

            self.apply_voltages(voltages.tolist())
            if self.config.settle_seconds > 0:
                time.sleep(self.config.settle_seconds)

            interval_saved = self._should_save_raw_optimizer_step(evaluation_index)
            measurement = self._measure_optimizer_step(
                self.config.measurement_seconds,
                evaluation_index=evaluation_index,
                keep_raw=interval_saved or self.cleanup_raw is not None,
                defer_raw_cleanup=self.cleanup_raw is not None,
            )
            score = self._objective_score(measurement)
            target = self._objective_target()
            target_met = score >= target
            raw_saved = interval_saved or target_met
            self._finish_optimizer_raw_measurement(
                measurement,
                keep_raw=raw_saved,
                reason=(
                    "target reached" if target_met else "save interval"
                ),
            )
            optimizer.tell(candidate, -score)
            self._log_iteration(
                voltages,
                measurement,
                backend="nevergrad",
                optimizer_name=optimizer_name,
                evaluation_index=evaluation_index,
                raw_saved=raw_saved,
            )

            if score > best_score:
                best_score = score
                best_voltages = voltages.copy()
                self._save_best_state(best_voltages, best_score)

            if target_met:
                print(
                    f"[Alice] Nevergrad reached "
                    f"{self._active_objective_metric}={score:.3f} "
                    f">= {target:.3f}"
                )
                break

        print(
            "[Alice] Restoring Nevergrad best measured voltages: "
            f"{best_voltages.tolist()} "
            f"({self._active_objective_metric}={best_score:.3f})"
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

    def _should_save_raw_optimizer_step(self, evaluation_index: int) -> bool:
        interval = int(self.config.raw_save_interval_steps)
        return interval > 0 and evaluation_index % interval == 0

    def _raw_record_label(self) -> str:
        return "CHSH_S" if self._active_objective_metric == "chsh_s" else "QKD"

    def _measure_optimizer_step(
        self,
        duration_seconds: float,
        *,
        evaluation_index: int | None,
        keep_raw: bool,
        defer_raw_cleanup: bool,
    ) -> CorrectionResult:
        return self._active_measure(
            duration_seconds,
            keep_raw=keep_raw,
            raw_label=self._raw_record_label(),
            optimizer_step=evaluation_index,
            defer_raw_cleanup=defer_raw_cleanup,
        )

    def _finish_optimizer_raw_measurement(
        self,
        measurement: CorrectionResult,
        *,
        keep_raw: bool,
        reason: str,
    ) -> None:
        if keep_raw:
            print(
                f"[Optimizer] Retained raw {self._raw_record_label()} files "
                f"({reason}) | Alice={measurement.sync.alice_path} | "
                f"Bob copy={measurement.sync.bob_path}"
            )
            return

        if self.cleanup_raw is None:
            return

        self.cleanup_raw(measurement)

    def _load_state(self) -> dict[str, Any]:
        path = self.log_paths.optimizer_state_json
        if not path.exists():
            state = {"last_update": current_utc_iso()}
            self._write_state(state)
            return state

        with path.open("r") as handle:
            return json.load(handle)

    def _save_best_state(
        self,
        best_voltages: np.ndarray,
        best_score: float,
    ) -> None:
        state = self._load_state()
        state[self._active_state_key()] = {
            "best_V": best_voltages.tolist(),
            "best_score": float(best_score),
            "objective_metric": self._active_objective_metric,
            "optimizer_backend": self._normalized_backend(),
            "optimizer_name": self._optimizer_name(),
            "optimize_epcs": self.config.optimize_epcs,
            "last_update": current_utc_iso(),
        }
        state["last_update"] = current_utc_iso()
        self._write_state(state)

    def _write_state(self, state: dict[str, Any]) -> None:
        path = self.log_paths.optimizer_state_json
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            json.dump(state, handle, indent=2)

    def _log_iteration(
        self,
        voltages: np.ndarray,
        measurement: CorrectionResult,
        *,
        backend: str,
        optimizer_name: str,
        evaluation_index: int,
        raw_saved: bool = False,
    ) -> None:
        row: dict[str, Any] = {
            "timestamp": time.time(),
            "optimizer_backend": backend,
            "optimizer_name": optimizer_name,
            "optimize_epcs": self.config.optimize_epcs,
            "phase": self._active_state_key(),
            "evaluation_index": evaluation_index,
            "raw_saved": int(raw_saved),
            "objective_metric": self._active_objective_metric,
            "objective_score": self._objective_score(measurement),
            "objective_target": self._objective_target(),
            "voltages": json.dumps(voltages.tolist()),
            "total_coincidences": int(measurement.total_coincidences),
        }
        counts = measurement.counts
        if isinstance(measurement, PhiPlusCorrectionResult):
            row.update(
                {
                    "visibility": float(measurement.visibility),
                    "QBER": float(measurement.qber_total),
                }
            )
            for label in RESULT_PAIR_ORDER:
                row[f"C_{label}"] = counts[label]
        else:
            row.update(
                {
                    "CHSH_S_signed": float(measurement.S_signed),
                    "CHSH_S_value": float(measurement.S_value),
                    "CHSH_E_ab": float(measurement.correlations["E_ab"]),
                    "CHSH_E_abp": float(measurement.correlations["E_abp"]),
                    "CHSH_E_apb": float(measurement.correlations["E_apb"]),
                    "CHSH_E_apbp": float(measurement.correlations["E_apbp"]),
                }
            )
            for label in CHSH_PAIR_ORDER:
                row[f"C_{label}"] = counts[label]

        _append_csv_row(self.log_paths.optimizer_iterations_csv, row)


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
