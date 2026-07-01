from __future__ import annotations

import datetime
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from qkd_acquisition import (
    AcquisitionConfig,
    AcquisitionPair,
    acquire_pair,
    delete_acquisition_files,
    send_bob_command,
)
from qkd_epc import init_epc, set_epc_voltages, validate_voltages
from qkd_names import make_record_id, safe_filename_part
from qkd_epc_correction import (
    CorrectionLogPaths,
    CHSHCorrectionResult,
    OptimizerConfig,
    PhiPlusCorrectionResult,
    PhiPlusOptimizer,
    analyze_chsh_s_coincidences,
    analyze_phi_plus_coincidences,
    append_correction_result,
)
from qkd_plot_delay_scans import save_delay_scan_plot
from qkd_sync import (
    DEFAULT_SYNC_CHANNEL,
    SyncCoincidenceAnalysis,
    aggregate_sync_exposures,
    analyze_sync_coincidence_exposures,
    analyze_sync_coincidences,
    save_coincidence_timetag_pairs,
)

try:
    import QuTAG_MC as qt
except ImportError as exc:
    print("ERROR: Failed to import QuTAG_MC:", exc)
    sys.exit(1)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

RECORD_SECONDS = 20.0
PAUSE_BETWEEN_RECORDS = 5
ERROR_RETRY_SECONDS = 5.0
OPTIMIZER_BAD_ACQUISITION_MAX_ATTEMPTS = 3

ALICE_EPC_ENABLED = True
ALICE_EPC_DEVICE_REF = 0
EPC_START_TEMPERATURE = 50.0

QBER_OPTIMIZATION_ENABLED = False

# Format: (label, Alice channel, Bob channel)
QKD_COINCIDENCE_PAIRS = (
    ("HH", 4, 1),
    ("HV", 4, 2),
    ("VH", 2, 1),
    ("VV", 2, 2),
    ("DD", 1, 4),
    ("DA", 1, 3),
    ("AD", 3, 4),
    ("AA", 3, 3),
)

CHSH_COINCIDENCE_PAIRS = (
    ("HH", 4, 1),
    ("HV", 4, 2),
    ("VH", 2, 1),
    ("VV", 2, 2),
    ("HA", 4, 3),
    ("HD", 4, 4),
    ("VA", 2, 3),
    ("VD", 2, 4),
    ("DH", 1, 1),
    ("DV", 1, 2),
    ("AH", 3, 1),
    ("AV", 3, 2),
    ("DD", 1, 4),
    ("DA", 1, 3),
    ("AD", 3, 4),
    ("AA", 3, 3),
)

QKD_DELAY_REFERENCE_PAIRS = {
    "HH": "HH",
    "HV": "HV",
    "VH": "VH",
    "VV": "VV",
    "DD": "DD",
    "DA": "DA",
    "AD": "AD",
    "AA": "AA",
}

CHSH_DELAY_REFERENCE_PAIRS = {
    "HH": "HH",
    "HV": "HV",
    "VH": "VH",
    "VV": "VV",
    "HA": "HA",
    "HD": "HD",
    "VA": "VA",
    "VD": "VD",
    "DH": "DH",
    "DV": "DV",
    "AH": "AH",
    "AV": "AV",
    "DD": "DD",
    "DA": "DA",
    "AD": "AD",
    "AA": "AA",
}


@dataclass(frozen=True)
class SyncProcessingConfig:
    sync_channel: int
    coincidence_window_ps: float
    coincidence_pairs: tuple[tuple[str, int, int], ...]
    delay_reference_pairs: dict[str, str] | None
    analysis_exposure_seconds: float
    store_coincidence_timetags: bool
    coincidence_timetag_dir: Path
    save_initial_delay_scan: bool
    delay_scan_dir: Path


ACQUISITION = AcquisitionConfig(
    bob_host="100.104.228.90",
    bob_port=5001,
    alice_record_dir=DATA_DIR / "AliceRaw",
    incoming_dir=DATA_DIR / "Incoming",
    schedule_ahead_seconds=3.0,
)

SYNC_PROCESSING = SyncProcessingConfig(
    sync_channel=DEFAULT_SYNC_CHANNEL,
    coincidence_window_ps=700.0,
    coincidence_pairs=CHSH_COINCIDENCE_PAIRS,
    delay_reference_pairs=CHSH_DELAY_REFERENCE_PAIRS,
    analysis_exposure_seconds=1.0,
    store_coincidence_timetags=False,
    coincidence_timetag_dir=DATA_DIR / "CoincidenceTimetags",
    save_initial_delay_scan=True,
    delay_scan_dir=DATA_DIR / "DelayScans",
)

CORRECTION_LOGS = CorrectionLogPaths(
    results_csv=DATA_DIR / "alice_results.csv",
    optimizer_state_json=DATA_DIR / "optimizer_state.json",
    optimizer_iterations_csv=DATA_DIR / "qber_iterlog.csv",
)

OPTIMIZER = OptimizerConfig(
    backend="nelder-mead",  # "nelder-mead" or "nevergrad"
    optimize_epcs="both",  # "alice", "bob", or "both"
    objective_metric="chsh_s",
    objective_target=2.3,
    secondary_objective_metric="visibility",
    secondary_objective_target=0.85,
    measurement_seconds=15.0,
    base_step_volts=25.0,
    voltage_quantization=0.1,
    maximum_voltage=130.0,
    settle_seconds=0.1,
    stable_sleep_seconds=1 * 60,
    max_iterations=100,
    nevergrad_optimizer="TBPSA",
    nevergrad_budget=70,
    nevergrad_seed=None,
    raw_save_interval_steps=10,
)

PASSIVE_RAW_SAVE_INTERVAL = OPTIMIZER.raw_save_interval_steps

class MeasurementPipeline:
    def __init__(self, tagger) -> None:
        self.tagger = tagger
        self.initial_delay_scan_saved = False

    def measure(
        self,
        duration_seconds: float,
        *,
        delete_raw_files: bool = False,
        record_id: str | None = None,
        announce_retained_raw: bool = True,
    ) -> PhiPlusCorrectionResult:
        acquisition = acquire_pair(
            self.tagger,
            ACQUISITION,
            duration_seconds,
            record_id=record_id,
        )
        deleted_on_failure = False
        try:
            synchronized = self._synchronize(acquisition)
            correction = analyze_phi_plus_coincidences(synchronized)
            chsh_correction = analyze_chsh_s_coincidences(synchronized)
            append_correction_result(correction, CORRECTION_LOGS.results_csv)
            append_correction_result(chsh_correction, CORRECTION_LOGS.results_csv)
            self._print_summary(
                acquisition,
                correction,
                chsh_correction=chsh_correction,
            )
        except Exception as exc:
            deleted_on_failure = True
            self._discard_failed_acquisition(acquisition, "measurement", exc)
            raise

        if delete_raw_files and not deleted_on_failure:
            delete_acquisition_files(acquisition, ACQUISITION)
        elif record_id is not None and announce_retained_raw:
            self._print_retained_raw_files(acquisition)
        return correction

    def measure_exposures(
        self,
        duration_seconds: float,
        *,
        delete_raw_files: bool = False,
        record_id: str | None = None,
        announce_retained_raw: bool = True,
    ) -> PhiPlusCorrectionResult:
        acquisition = acquire_pair(
            self.tagger,
            ACQUISITION,
            duration_seconds,
            record_id=record_id,
        )
        deleted_on_failure = False
        try:
            print(
                f"[Alice] Processing record_id={acquisition.record_id} in "
                f"{SYNC_PROCESSING.analysis_exposure_seconds:g} s exposures"
            )
            exposures = analyze_sync_coincidence_exposures(
                acquisition.alice_path,
                acquisition.bob_path,
                SYNC_PROCESSING.coincidence_pairs,
                SYNC_PROCESSING.analysis_exposure_seconds,
                sync_channel=SYNC_PROCESSING.sync_channel,
                coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
                delay_reference_pairs=SYNC_PROCESSING.delay_reference_pairs,
                include_partial_last_exposure=True,
                capture_first_delay_scan=(
                    SYNC_PROCESSING.save_initial_delay_scan
                    and not self.initial_delay_scan_saved
                ),
            )
            if not exposures:
                raise RuntimeError(
                    "Recording overlap is shorter than one analysis exposure"
                )

            recording_start_s = datetime.datetime.fromisoformat(
                acquisition.start_time_utc
            ).timestamp()
            for exposure in exposures:
                exposure.measurement_timestamp_s = (
                    recording_start_s + exposure.exposure_start_s
                )

            if (
                SYNC_PROCESSING.save_initial_delay_scan
                and not self.initial_delay_scan_saved
            ):
                output_path = (
                    SYNC_PROCESSING.delay_scan_dir
                    / f"initial_delay_scans_{acquisition.record_id}.png"
                )
                saved_path = save_delay_scan_plot(exposures[0], output_path)
                self.initial_delay_scan_saved = True
                print(f"[Alice] Saved initial delay scans: {saved_path}")

            synchronized = aggregate_sync_exposures(exposures)
            synchronized.measurement_timestamp_s = recording_start_s
            correction = analyze_phi_plus_coincidences(synchronized)
            chsh_correction = analyze_chsh_s_coincidences(synchronized)
            append_correction_result(correction, CORRECTION_LOGS.results_csv)
            append_correction_result(chsh_correction, CORRECTION_LOGS.results_csv)
            self._print_summary(
                acquisition,
                correction,
                exposure_analyses=exposures,
                chsh_correction=chsh_correction,
            )
        except Exception as exc:
            deleted_on_failure = True
            self._discard_failed_acquisition(acquisition, "exposure measurement", exc)
            raise

        if delete_raw_files and not deleted_on_failure:
            delete_acquisition_files(acquisition, ACQUISITION)
        elif record_id is not None and announce_retained_raw:
            self._print_retained_raw_files(acquisition)
        return correction

    def measure_for_optimizer(
        self,
        duration_seconds: float,
        *,
        keep_raw: bool = False,
        raw_label: str | None = "QKD",
        optimizer_step: int | None = None,
        defer_raw_cleanup: bool = False,
    ) -> PhiPlusCorrectionResult:
        retain_raw = keep_raw or defer_raw_cleanup

        def acquire_and_measure() -> PhiPlusCorrectionResult:
            record_id = (
                self._optimizer_record_id(raw_label, optimizer_step)
                if retain_raw
                else None
            )
            return self.measure_exposures(
                duration_seconds,
                delete_raw_files=not retain_raw,
                record_id=record_id,
                announce_retained_raw=keep_raw and not defer_raw_cleanup,
            )

        return self._retry_optimizer_measurement(
            "QKD",
            acquire_and_measure,
        )

    def measure_chsh_for_optimizer(
        self,
        duration_seconds: float,
        *,
        keep_raw: bool = False,
        raw_label: str | None = "CHSH_S",
        optimizer_step: int | None = None,
        defer_raw_cleanup: bool = False,
    ) -> CHSHCorrectionResult:
        retain_raw = keep_raw or defer_raw_cleanup

        def acquire_and_measure() -> CHSHCorrectionResult:
            record_id = (
                self._optimizer_record_id(raw_label, optimizer_step)
                if retain_raw
                else None
            )
            acquisition = acquire_pair(
                self.tagger,
                ACQUISITION,
                duration_seconds,
                record_id=record_id,
            )
            print(f"[Alice] Processing CHSH record_id={acquisition.record_id}")
            capture_delay_scans = (
                SYNC_PROCESSING.save_initial_delay_scan
                and not self.initial_delay_scan_saved
            )
            deleted_on_failure = False
            try:
                synchronized = analyze_sync_coincidences(
                    acquisition.alice_path,
                    acquisition.bob_path,
                    CHSH_COINCIDENCE_PAIRS,
                    sync_channel=SYNC_PROCESSING.sync_channel,
                    coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
                    capture_delay_scans=capture_delay_scans,
                    delay_reference_pairs=CHSH_DELAY_REFERENCE_PAIRS,
                )
                if capture_delay_scans:
                    output_path = (
                        SYNC_PROCESSING.delay_scan_dir
                        / f"initial_delay_scans_{acquisition.record_id}.png"
                    )
                    saved_path = save_delay_scan_plot(synchronized, output_path)
                    self.initial_delay_scan_saved = True
                    print(f"[Alice] Saved initial delay scans: {saved_path}")

                correction = analyze_chsh_s_coincidences(synchronized)
                append_correction_result(correction, CORRECTION_LOGS.results_csv)
                self._print_chsh_summary(acquisition, correction)
                if keep_raw and not defer_raw_cleanup:
                    self._print_retained_raw_files(acquisition)
                return correction
            except Exception as exc:
                deleted_on_failure = True
                self._discard_failed_acquisition(acquisition, "CHSH measurement", exc)
                raise
            finally:
                if not retain_raw and not deleted_on_failure:
                    delete_acquisition_files(acquisition, ACQUISITION)

        return self._retry_optimizer_measurement(
            "CHSH",
            acquire_and_measure,
        )

    def cleanup_optimizer_raw(
        self,
        measurement: PhiPlusCorrectionResult | CHSHCorrectionResult,
    ) -> None:
        acquisition = AcquisitionPair(
            record_id=self._record_id_from_raw_path(measurement.sync.alice_path),
            start_time_utc="",
            duration_seconds=measurement.sync.overlap_duration_s,
            alice_path=measurement.sync.alice_path,
            bob_path=measurement.sync.bob_path,
        )
        delete_acquisition_files(acquisition, ACQUISITION)

    @staticmethod
    def _discard_failed_acquisition(
        acquisition: AcquisitionPair,
        stage: str,
        exc: Exception,
    ) -> None:
        print(
            f"[Alice] Discarding failed {stage} raw files | "
            f"record_id={acquisition.record_id} | error={exc}"
        )
        try:
            delete_acquisition_files(acquisition, ACQUISITION)
        except Exception as delete_exc:
            print(
                "[Alice] Failed to delete bad acquisition files | "
                f"record_id={acquisition.record_id} | error={delete_exc}"
            )

    @staticmethod
    def _retry_optimizer_measurement(label: str, measure_callback):
        attempts = max(1, int(OPTIMIZER_BAD_ACQUISITION_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            try:
                return measure_callback()
            except Exception as exc:
                if attempt >= attempts:
                    print(
                        f"[Alice] {label} optimizer measurement failed after "
                        f"{attempts} attempts; stopping: {exc}"
                    )
                    raise
                print(
                    f"[Alice] {label} optimizer measurement failed "
                    f"({attempt}/{attempts}); retrying with a new acquisition: "
                    f"{exc}"
                )
                if ERROR_RETRY_SECONDS > 0:
                    time.sleep(ERROR_RETRY_SECONDS)
        raise RuntimeError(
            f"{label} optimizer measurement retry loop exited unexpectedly"
        )

    @staticmethod
    def _record_id_from_raw_path(path: Path) -> str:
        name = path.name
        if name.startswith("alice_") and "_exp_" in name:
            return name[len("alice_") :].split("_exp_", 1)[0]
        return path.stem

    @staticmethod
    def _optimizer_record_id(
        raw_label: str | None,
        optimizer_step: int | None,
    ) -> str:
        label = safe_filename_part(raw_label or "OPT")
        if optimizer_step is None:
            return f"{label}_{make_record_id()}"
        return f"{label}_step{optimizer_step:04d}_{make_record_id()}"

    @staticmethod
    def _print_retained_raw_files(acquisition: AcquisitionPair) -> None:
        print(
            "[Alice] Retained raw files | "
            f"record_id={acquisition.record_id} | "
            f"Alice={acquisition.alice_path} | Bob copy={acquisition.bob_path}"
        )

    def _synchronize(
        self,
        acquisition: AcquisitionPair,
    ) -> SyncCoincidenceAnalysis:
        print(f"[Alice] Processing record_id={acquisition.record_id}")
        capture_delay_scans = (
            SYNC_PROCESSING.save_initial_delay_scan
            and not self.initial_delay_scan_saved
        )
        synchronized = analyze_sync_coincidences(
            acquisition.alice_path,
            acquisition.bob_path,
            SYNC_PROCESSING.coincidence_pairs,
            sync_channel=SYNC_PROCESSING.sync_channel,
            coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
            capture_delay_scans=capture_delay_scans,
            delay_reference_pairs=SYNC_PROCESSING.delay_reference_pairs,
        )

        if capture_delay_scans:
            output_path = (
                SYNC_PROCESSING.delay_scan_dir
                / f"initial_delay_scans_{acquisition.record_id}.png"
            )
            saved_path = save_delay_scan_plot(synchronized, output_path)
            self.initial_delay_scan_saved = True
            print(f"[Alice] Saved initial delay scans: {saved_path}")

        if SYNC_PROCESSING.store_coincidence_timetags:
            saved = save_coincidence_timetag_pairs(
                synchronized,
                SYNC_PROCESSING.coincidence_timetag_dir,
                prefix=acquisition.record_id,
            )
            print(
                f"[Alice] Saved {len(saved)} coincidence timetag files "
                f"for record_id={acquisition.record_id}"
            )
        return synchronized

    @staticmethod
    def _channel_singles_counts(
        analysis: SyncCoincidenceAnalysis,
    ) -> tuple[dict[int, int], dict[int, int]]:
        alice_counts: dict[int, int] = {}
        bob_counts: dict[int, int] = {}
        for result in analysis.pair_results:
            alice_counts.setdefault(
                result.pair.alice_channel,
                int(result.alice_event_count),
            )
            bob_counts.setdefault(
                result.pair.bob_channel,
                int(result.bob_event_count),
            )
        return alice_counts, bob_counts

    @staticmethod
    def _average_singles_per_exposure(
        analyses: list[SyncCoincidenceAnalysis],
    ) -> tuple[float, float, dict[int, float], dict[int, float]]:
        total_duration_s = sum(analysis.overlap_duration_s for analysis in analyses)
        exposure_seconds = max(
            (analysis.overlap_duration_s for analysis in analyses),
            default=0.0,
        )
        if total_duration_s <= 0.0:
            return exposure_seconds, total_duration_s, {}, {}

        alice_totals: dict[int, int] = {}
        bob_totals: dict[int, int] = {}
        for analysis in analyses:
            alice_counts, bob_counts = MeasurementPipeline._channel_singles_counts(
                analysis
            )
            for channel, count in alice_counts.items():
                alice_totals[channel] = alice_totals.get(channel, 0) + count
            for channel, count in bob_counts.items():
                bob_totals[channel] = bob_totals.get(channel, 0) + count

        scale = exposure_seconds / total_duration_s
        alice_average = {
            channel: count * scale for channel, count in alice_totals.items()
        }
        bob_average = {channel: count * scale for channel, count in bob_totals.items()}
        return exposure_seconds, total_duration_s, alice_average, bob_average

    @staticmethod
    def _print_chsh_summary(
        acquisition: AcquisitionPair,
        correction: CHSHCorrectionResult,
    ) -> None:
        counts = correction.counts
        border = "=" * 88
        print(f"\n{border}")
        print(
            f"[Alice] CHSH RESULT | record_id={acquisition.record_id} | "
            f"S={correction.S_value:.3f} | "
            f"signed={correction.S_signed:+.3f} | "
            f"total={correction.total_coincidences} | "
            f"sync markers={correction.sync.clock_map.counters.size}"
        )
        print(
            "[Alice] CHSH E      | "
            + "  ".join(
                f"{name}={value:+.3f}"
                for name, value in correction.correlations.items()
            )
        )
        print(
            "[Alice] CHSH COUNTS | "
            + "  ".join(
                f"{label}={counts[label]}" for label, _, _ in CHSH_COINCIDENCE_PAIRS
            )
        )
        print(f"{border}\n")

    @staticmethod
    def _print_summary(
        acquisition: AcquisitionPair,
        correction: PhiPlusCorrectionResult,
        *,
        exposure_analyses: list[SyncCoincidenceAnalysis] | None = None,
        chsh_correction: CHSHCorrectionResult | None = None,
    ) -> None:
        counts = correction.counts
        border = "=" * 88
        print(f"\n{border}")
        print(
            f"[Alice] CURRENT RESULT | record_id={acquisition.record_id} | "
            f"visibility={100 * correction.visibility:.3f}% | "
            f"HV vis={100 * correction.basis_visibility['HV']:.3f}% | "
            f"DA vis={100 * correction.basis_visibility['DA']:.3f}% | "
            f"QBER={100.0 * correction.qber_total:.2f}% | "
            f"total={correction.total_coincidences} | "
            f"exposures={correction.sync.exposure_count} | "
            f"sync markers={correction.sync.clock_map.counters.size}"
        )
        print(
            "[Alice] COINCIDENCES   | "
            f"HH={counts['HH']}  VV={counts['VV']}  "
            f"DD={counts['DD']}  AA={counts['AA']}  | "
            f"HV={counts['HV']}  VH={counts['VH']}  "
            f"DA={counts['DA']}  AD={counts['AD']}"
        )
        print(
            "[Alice] DELAYS (ns)   | "
            + "  ".join(
                f"{name}={delay_ps / 1000.0:+.3f}"
                for name, delay_ps in correction.delays_ps.items()
            )
        )
        if chsh_correction is not None:
            print(
                "[Alice] CHSH          | "
                f"S={chsh_correction.S_value:.3f}  "
                f"signed={chsh_correction.S_signed:+.3f}  "
                + "  ".join(
                    f"{name}={value:+.3f}"
                    for name, value in chsh_correction.correlations.items()
                )
            )
        if exposure_analyses is not None:
            exposure_seconds, total_duration_s, alice_average, bob_average = (
                MeasurementPipeline._average_singles_per_exposure(exposure_analyses)
            )
            alice_text = "  ".join(
                f"A{channel}={count:.0f}"
                for channel, count in sorted(alice_average.items())
            )
            bob_text = "  ".join(
                f"B{channel}={count:.0f}"
                for channel, count in sorted(bob_average.items())
            )
            print(
                "[Alice] SINGLES/MEAS  | "
                f"avg/{exposure_seconds:.3f}s over "
                f"{total_duration_s:.3f}s | "
                f"{alice_text} | {bob_text}"
            )
        print(f"{border}\n")


def initialize_alice_epc():
    if not ALICE_EPC_ENABLED:
        return None
    return init_epc(
        "Alice",
        ALICE_EPC_DEVICE_REF,
        EPC_START_TEMPERATURE,
    )


def apply_correction_voltages(alice_epc, values: list[float]) -> None:
    if len(values) != 8:
        raise ValueError(
            "Phi+ correction requires eight voltages: "
            "Alice DAC0..3 followed by Bob DAC0..3"
        )

    alice_voltages = validate_voltages(values[:4])
    bob_voltages = validate_voltages(values[4:])
    set_epc_voltages("Alice", alice_epc, alice_voltages)

    reply = send_bob_command(
        ACQUISITION,
        {
            "command": "SET_VOLTAGES",
            "voltages": bob_voltages,
        },
    )
    if not reply.get("ok"):
        raise RuntimeError(
            "Bob rejected EPC voltages "
            f"{bob_voltages}: {reply.get('error', 'unknown error')}"
        )


def run_passive_measurements(pipeline: MeasurementPipeline) -> None:
    run_index = 0
    while True:
        try:
            run_index += 1
            keep_raw = run_index % PASSIVE_RAW_SAVE_INTERVAL == 0
            record_id = (
                f"PASSIVE_run{run_index:06d}_{make_record_id()}"
                if keep_raw
                else None
            )

            pipeline.measure_exposures(
                RECORD_SECONDS,
                delete_raw_files=not keep_raw,
                record_id=record_id,
                announce_retained_raw=keep_raw,
            )
            time.sleep(PAUSE_BETWEEN_RECORDS)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[Alice] Measurement failed: {exc}")
            time.sleep(ERROR_RETRY_SECONDS)


def shutdown_tagger(tagger) -> None:
    try:
        reply = send_bob_command(ACQUISITION, {"command": "STOP"})
        if not reply.get("ok"):
            print(
                "[Alice] Bob shutdown warning: "
                f"{reply.get('error', 'STOP command was rejected')}"
            )
    except Exception as exc:
        print(f"[Alice] Bob shutdown warning: {exc}")

    try:
        tagger.writeTimestamps("", tagger.FILEFORMAT_NONE)
        tagger.deInitialize()
    except Exception as exc:
        print(f"[Alice] Tagger shutdown warning: {exc}")


def optimizer_measure_for_metric(
    pipeline: MeasurementPipeline,
    metric: str | None,
):
    if metric is None:
        return None
    normalized = metric.strip().lower().replace("-", "_")
    if normalized in {"chsh", "chsh_s", "s", "s_value"}:
        return pipeline.measure_chsh_for_optimizer
    if normalized in {
        "visibility",
        "total_visibility",
        "vis_hv",
        "hv_visibility",
        "hv",
        "vis_da",
        "da_visibility",
        "da",
    }:
        return pipeline.measure_for_optimizer
    raise ValueError(f"Unknown optimizer metric: {metric!r}")


def main() -> None:
    alice_epc = initialize_alice_epc()
    tagger = qt.QuTAG()
    pipeline = MeasurementPipeline(tagger)
    print(
        f"[Alice] Bob endpoint={ACQUISITION.bob_host}:{ACQUISITION.bob_port}, "
        f"measurement={RECORD_SECONDS:.1f} s"
    )

    try:
        if QBER_OPTIMIZATION_ENABLED:
            optimizer = PhiPlusOptimizer(
                config=OPTIMIZER,
                log_paths=CORRECTION_LOGS,
                apply_voltages=lambda values: apply_correction_voltages(
                    alice_epc,
                    values,
                ),
                measure=optimizer_measure_for_metric(
                    pipeline,
                    OPTIMIZER.objective_metric,
                ),
                secondary_measure=optimizer_measure_for_metric(
                    pipeline,
                    OPTIMIZER.secondary_objective_metric,
                ),
                cleanup_raw=pipeline.cleanup_optimizer_raw,
            )
            optimizer.monitor_forever(RECORD_SECONDS)
        else:
            run_passive_measurements(pipeline)
    except KeyboardInterrupt:
        print("[Alice] Interrupted by user")
    finally:
        shutdown_tagger(tagger)


if __name__ == "__main__":
    main()
