from __future__ import annotations

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
PAUSE_BETWEEN_RECORDS = 1.0
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
    "HV": "HH",
    "VH": "VV",
    "VV": "VV",
    "DD": "DD",
    "DA": "DD",
    "AD": "AA",
    "AA": "AA",
}

@dataclass(frozen=True)
class SyncProcessingConfig:
    sync_channel: int
    coincidence_window_ps: float
    coincidence_pairs: tuple[tuple[str, int, int], ...]
    delay_reference_pairs: dict[str, str]
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
    delay_reference_pairs=QKD_DELAY_REFERENCE_PAIRS,
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
    objective_metric="vis_DA",  # "visibility", "vis_HV", or "vis_DA"
    objective_target=0.9,
    measurement_seconds=10.0,
    base_step_volts=25.0,
    voltage_quantization=0.1,
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
        self.fixed_delays_ps: dict[str, float] | None = None

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

    def measure_for_optimizer(
        self,
        duration_seconds: float,
    ) -> PhiPlusCorrectionResult:
        return self.measure(duration_seconds, delete_raw_files=True)

    def _synchronize(
        self,
        acquisition: AcquisitionPair,
    ) -> SyncCoincidenceAnalysis:
        print(
            f"[Alice] Synchronizing record_id={acquisition.record_id}: "
            f"{acquisition.alice_path.name} with {acquisition.bob_path.name}"
        )
        calibrating_delays = self.fixed_delays_ps is None
        synchronized = analyze_sync_coincidences(
            acquisition.alice_path,
            acquisition.bob_path,
            SYNC_PROCESSING.coincidence_pairs,
            sync_channel=SYNC_PROCESSING.sync_channel,
            coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
            capture_delay_scans=(
                SYNC_PROCESSING.save_initial_delay_scan and calibrating_delays
            ),
            fixed_delays_ps=self.fixed_delays_ps,
            delay_reference_pairs=SYNC_PROCESSING.delay_reference_pairs,
        )

        if calibrating_delays:
            calibration = synchronized
            self.fixed_delays_ps = {
                result.pair.name: float(result.best_delay_ps)
                for result in calibration.pair_results
            }

            print("[Alice] Locked coincidence delays for EPC optimization:")
            for pair_name, delay_ps in self.fixed_delays_ps.items():
                print(
                    f"  {pair_name}: {delay_ps / 1_000.0:+.3f} ns "
                    f"({delay_ps:+.0f} ps)"
                )

            if SYNC_PROCESSING.save_initial_delay_scan:
                output_path = (
                    SYNC_PROCESSING.delay_scan_dir
                    / f"initial_delay_scans_{acquisition.record_id}.png"
                )
                saved_path = save_delay_scan_plot(calibration, output_path)
                self.initial_delay_scan_saved = True
                print(f"[Alice] Saved initial delay scans: {saved_path}")

            # Recount the calibration acquisition with the locked delays so
            # the first visibility is directly comparable to later samples.
            synchronized = analyze_sync_coincidences(
                acquisition.alice_path,
                acquisition.bob_path,
                SYNC_PROCESSING.coincidence_pairs,
                sync_channel=SYNC_PROCESSING.sync_channel,
                coincidence_window_ps=SYNC_PROCESSING.coincidence_window_ps,
                fixed_delays_ps=self.fixed_delays_ps,
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
    def _print_summary(
        acquisition: AcquisitionPair,
        correction: PhiPlusCorrectionResult,
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
            f"sync markers={correction.sync.clock_map.counters.size}"
        )
        print(
            "[Alice] COINCIDENCES   | "
            f"HH={counts['HH']}  VV={counts['VV']}  "
            f"DD={counts['DD']}  AA={counts['AA']}  | "
            f"HV={counts['HV']}  VH={counts['VH']}  "
            f"DA={counts['DA']}  AD={counts['AD']}"
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
    print(f"[Alice] Bob EPC voltages set to {bob_voltages}")


def run_passive_measurements(pipeline: MeasurementPipeline) -> None:
    while True:
        try:
            pipeline.measure(RECORD_SECONDS)
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
