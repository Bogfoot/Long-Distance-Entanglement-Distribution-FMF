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
from qkd_epc_correction import (
    CorrectionLogPaths,
    OptimizerConfig,
    PhiPlusCorrectionResult,
    PhiPlusOptimizer,
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

RECORD_SECONDS = 10.0
PAUSE_BETWEEN_RECORDS = 30 * 60
ERROR_RETRY_SECONDS = 5.0

ALICE_EPC_ENABLED = True
ALICE_EPC_DEVICE_REF = 0
EPC_START_TEMPERATURE = 50.0

QBER_OPTIMIZATION_ENABLED = True

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

QKD_DELAY_REFERENCE_PAIRS = {
    "HH": "HH",
    "HV": "HV",
    "VH": "VV",
    "VV": "VV",
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
    delay_recheck_half_range_ps: float
    delay_recheck_step_ps: float
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
    coincidence_pairs=QKD_COINCIDENCE_PAIRS,
    delay_reference_pairs=None,
    delay_recheck_half_range_ps=3_000.0,
    delay_recheck_step_ps=100.0,
    analysis_exposure_seconds=1,
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
    backend="nevergrad",  # "nelder-mead" or "nevergrad"
    optimize_epcs="both",  # "alice", "bob", or "both"
    objective_metric="visibility",  # "visibility", "vis_HV", or "vis_DA"
    objective_target=0.9,
    measurement_seconds=10.0,
    base_step_volts=25.0,
    voltage_quantization=0.032,
    maximum_voltage=130.0,
    settle_seconds=0.1,
    stable_sleep_seconds=10 * 60,
    nevergrad_optimizer="TBPSA",
    nevergrad_budget=100,
    nevergrad_seed=None,
)


class MeasurementPipeline:
    def __init__(self, tagger) -> None:
        self.tagger = tagger
        self.initial_delay_scan_saved = False
        self.delay_search_centers_ps: dict[str, float] | None = None
        self.previous_delays_ps: dict[str, float] | None = None

    def measure(
        self,
        duration_seconds: float,
        *,
        delete_raw_files: bool = False,
    ) -> PhiPlusCorrectionResult:
        acquisition = acquire_pair(
            self.tagger,
            ACQUISITION,
            duration_seconds,
        )
        synchronized = self._synchronize(acquisition)
        correction = analyze_phi_plus_coincidences(synchronized)
        append_correction_result(correction, CORRECTION_LOGS.results_csv)
        self._print_summary(acquisition, correction)
        if delete_raw_files:
            delete_acquisition_files(acquisition, ACQUISITION)
        return correction

    def measure_exposures(
        self,
        duration_seconds: float,
        *,
        delete_raw_files: bool = False,
    ) -> PhiPlusCorrectionResult:
        acquisition = acquire_pair(
            self.tagger,
            ACQUISITION,
            duration_seconds,
        )
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
        append_correction_result(correction, CORRECTION_LOGS.results_csv)
        self._print_summary(
            acquisition,
            correction,
            exposure_analyses=exposures,
        )
        if delete_raw_files:
            delete_acquisition_files(acquisition, ACQUISITION)
        return correction

    def measure_for_optimizer(
        self,
        duration_seconds: float,
    ) -> PhiPlusCorrectionResult:
        return self.measure_exposures(
            duration_seconds,
            delete_raw_files=True,
        )

    def _synchronize(
        self,
        acquisition: AcquisitionPair,
    ) -> SyncCoincidenceAnalysis:
        print(f"[Alice] Processing record_id={acquisition.record_id}")
        calibrating_delays = self.delay_search_centers_ps is None
        synchronized = analyze_sync_coincidences(
            acquisition.alice_path,
            acquisition.bob_path,
            SYNC_PROCESSING.coincidence_pairs,
            sync_channel=SYNC_PROCESSING.sync_channel,
            coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
            capture_delay_scans=(
                SYNC_PROCESSING.save_initial_delay_scan and calibrating_delays
            ),
            delay_reference_pairs=SYNC_PROCESSING.delay_reference_pairs,
            delay_search_centers_ps=self.delay_search_centers_ps,
            delay_search_half_range_ps=(
                SYNC_PROCESSING.delay_recheck_half_range_ps
            ),
            delay_search_step_ps=SYNC_PROCESSING.delay_recheck_step_ps,
        )

        self.previous_delays_ps = {
            result.pair.name: float(result.best_delay_ps)
            for result in synchronized.pair_results
        }

        if calibrating_delays:
            self.delay_search_centers_ps = self.previous_delays_ps.copy()
            print("[Alice] Initial delay calibration complete")

            if SYNC_PROCESSING.save_initial_delay_scan:
                output_path = (
                    SYNC_PROCESSING.delay_scan_dir
                    / f"initial_delay_scans_{acquisition.record_id}.png"
                )
                saved_path = save_delay_scan_plot(synchronized, output_path)
                self.initial_delay_scan_saved = True
                print(f"[Alice] Saved initial delay scans: {saved_path}")
        else:
            boundary_threshold_ps = (
                SYNC_PROCESSING.delay_recheck_half_range_ps
                - SYNC_PROCESSING.delay_recheck_step_ps / 2.0
            )
            boundary_pairs = [
                name
                for name in self.previous_delays_ps
                if abs(
                    self.previous_delays_ps[name]
                    - self.delay_search_centers_ps[name]
                )
                >= boundary_threshold_ps
            ]
            if boundary_pairs:
                print(
                    "[Alice] WARNING: Delay peak reached the local-search "
                    "boundary for: " + ", ".join(boundary_pairs)
                )

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
        total_duration_s = np.sum(analysis.overlap_duration_s for analysis in analyses)
        exposure_seconds = max(
            (analysis.overlap_duration_s for analysis in analyses),
            default=0.0,
        )
        if total_duration_s <= 0.0:
            return exposure_seconds, total_duration_s, {}, {}

        alice_totals: dict[int, int] = {}
        bob_totals: dict[int, int] = {}
        for analysis in analyses:
            alice_counts, bob_counts = (
                MeasurementPipeline._channel_singles_counts(analysis)
            )
            for channel, count in alice_counts.items():
                alice_totals[channel] = alice_totals.get(channel, 0) + count
            for channel, count in bob_counts.items():
                bob_totals[channel] = bob_totals.get(channel, 0) + count

        scale = exposure_seconds / total_duration_s
        alice_average = {
            channel: count * scale for channel, count in alice_totals.items()
        }
        bob_average = {
            channel: count * scale for channel, count in bob_totals.items()
        }
        return exposure_seconds, total_duration_s, alice_average, bob_average

    @staticmethod
    def _print_summary(
        acquisition: AcquisitionPair,
        correction: PhiPlusCorrectionResult,
        *,
        exposure_analyses: list[SyncCoincidenceAnalysis] | None = None,
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
        if exposure_analyses is not None:
            exposure_seconds, total_duration_s, alice_average, bob_average = (
                MeasurementPipeline._average_singles_per_exposure(
                    exposure_analyses
                )
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
    while True:
        try:
            pipeline.measure_exposures(RECORD_SECONDS)
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
                measure=pipeline.measure_for_optimizer,
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
