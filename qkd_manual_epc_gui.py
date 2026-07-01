from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Alice import (
    ACQUISITION,
    ALICE_EPC_DEVICE_REF,
    ALICE_EPC_ENABLED,
    CORRECTION_LOGS,
    EPC_START_TEMPERATURE,
    SYNC_PROCESSING,
)
from qkd_acquisition import (
    AcquisitionPair,
    acquire_pair,
    delete_acquisition_files,
    send_bob_command,
)
from qkd_epc import init_epc, set_epc_voltages, validate_voltages
from qkd_epc_correction import (
    PhiPlusCorrectionResult,
    analyze_phi_plus_coincidences,
    append_correction_result,
)
from qkd_sync import analyze_sync_coincidences

try:
    import QuTAG_MC as qt
except ImportError as exc:
    qt = None
    QUTAG_IMPORT_ERROR = exc
else:
    QUTAG_IMPORT_ERROR = None


WINDOW_TITLE = "QKD manual EPC control"
DEFAULT_EXPOSURE_SECONDS = 10.0
DEFAULT_COINCIDENCE_WINDOW_PS = SYNC_PROCESSING.coincidence_window_ps
DEFAULT_SETTLE_SECONDS = 0.5
DEFAULT_DELETE_RAW_FILES = True
DEFAULT_LOG_RESULTS = True
VOLTAGE_MIN = 0.0
VOLTAGE_MAX = 130.0
VOLTAGE_RESOLUTION = 0.1
DISPLAY_PAIR_ORDER = ("HH", "VV", "DD", "AA", "HV", "VH", "DA", "AD")


@dataclass(frozen=True)
class MeasurementSettings:
    exposure_seconds: float
    coincidence_window_ps: float
    settle_seconds: float
    alice_enabled: bool
    bob_enabled: bool
    alice_voltages: tuple[float, float, float, float]
    bob_voltages: tuple[float, float, float, float]
    delete_raw_files: bool
    log_results: bool


@dataclass(frozen=True)
class MeasurementEvent:
    record_id: str
    result: PhiPlusCorrectionResult


class ManualMeasurementEngine:
    def __init__(self) -> None:
        self.tagger = None
        self.alice_epc = None

    def ensure_tagger(self):
        if self.tagger is None:
            if qt is None:
                raise RuntimeError(
                    f"Failed to import QuTAG_MC: {QUTAG_IMPORT_ERROR}"
                )
            self.tagger = qt.QuTAG()
        return self.tagger

    def ensure_alice_epc(self):
        if self.alice_epc is None:
            self.alice_epc = init_epc(
                "Alice",
                ALICE_EPC_DEVICE_REF,
                EPC_START_TEMPERATURE,
            )
        return self.alice_epc

    def apply_voltages(self, settings: MeasurementSettings) -> None:
        if settings.alice_enabled:
            set_epc_voltages(
                "Alice",
                self.ensure_alice_epc(),
                settings.alice_voltages,
            )

        if settings.bob_enabled:
            bob_voltages = validate_voltages(settings.bob_voltages)
            reply = send_bob_command(
                ACQUISITION,
                {
                    "command": "SET_VOLTAGES",
                    "voltages": bob_voltages,
                },
            )
            if not reply.get("ok"):
                raise RuntimeError(
                    "Bob rejected EPC voltages: "
                    f"{reply.get('error', 'unknown error')}"
                )
            print(f"[GUI] Bob EPC voltages set to {bob_voltages}")

    def record(self, duration_seconds: float) -> AcquisitionPair:
        tagger = self.ensure_tagger()
        return acquire_pair(tagger, ACQUISITION, duration_seconds)

    def measure(self, settings: MeasurementSettings) -> MeasurementEvent:
        self.apply_voltages(settings)
        if settings.settle_seconds > 0:
            time.sleep(settings.settle_seconds)

        acquisition = self.record(settings.exposure_seconds)
        success = False
        try:
            synchronized = analyze_sync_coincidences(
                acquisition.alice_path,
                acquisition.bob_path,
                SYNC_PROCESSING.coincidence_pairs,
                sync_channel=SYNC_PROCESSING.sync_channel,
                coincidence_window_ps=settings.coincidence_window_ps,
                delay_reference_pairs=SYNC_PROCESSING.delay_reference_pairs,
            )

            correction = analyze_phi_plus_coincidences(synchronized)
            if settings.log_results:
                append_correction_result(
                    correction,
                    CORRECTION_LOGS.results_csv,
                )
            success = True
            return MeasurementEvent(
                record_id=acquisition.record_id,
                result=correction,
            )
        finally:
            if success and settings.delete_raw_files:
                delete_acquisition_files(acquisition, ACQUISITION)

    def shutdown(self) -> None:
        if self.tagger is None:
            return
        try:
            self.tagger.writeTimestamps("", self.tagger.FILEFORMAT_NONE)
            self.tagger.deInitialize()
        finally:
            self.tagger = None


class VoltageControl(ttk.Frame):
    def __init__(
        self,
        parent,
        title: str,
        enabled_default: bool,
    ) -> None:
        super().__init__(parent)
        self.enabled = tk.BooleanVar(value=enabled_default)
        self.variables = [tk.DoubleVar(value=0.0) for _ in range(4)]

        header = ttk.Frame(self)
        header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text=title,
            font=("Segoe UI", 11, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            header,
            text="Control enabled",
            variable=self.enabled,
        ).grid(row=0, column=1, sticky="e")

        self.columnconfigure(1, weight=1)
        for index, variable in enumerate(self.variables):
            row = index + 1
            ttk.Label(self, text=f"DAC{index}").grid(
                row=row,
                column=0,
                sticky="w",
                padx=(0, 8),
            )
            scale = ttk.Scale(
                self,
                from_=VOLTAGE_MIN,
                to=VOLTAGE_MAX,
                variable=variable,
                orient="horizontal",
            )
            scale.grid(row=row, column=1, sticky="ew", padx=(0, 8))
            spinbox = ttk.Spinbox(
                self,
                from_=VOLTAGE_MIN,
                to=VOLTAGE_MAX,
                increment=VOLTAGE_RESOLUTION,
                textvariable=variable,
                width=7,
                format="%.1f",
            )
            spinbox.grid(row=row, column=2, sticky="e")

    def values(self) -> tuple[float, float, float, float]:
        checked = validate_voltages([variable.get() for variable in self.variables])
        return tuple(checked)

    def zero(self) -> None:
        for variable in self.variables:
            variable.set(0.0)


class ManualEpcGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.engine = ManualMeasurementEngine()
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.continuous = False
        self.record_requested = False
        self.active_event_name: str | None = None
        self.closing = False
        self.measurement_number = 0
        self.history_index: list[int] = []
        self.history_total: list[float] = []
        self.history_hv: list[float] = []
        self.history_da: list[float] = []
        self.history_counts: list[int] = []

        self.exposure_seconds = tk.DoubleVar(value=DEFAULT_EXPOSURE_SECONDS)
        self.coincidence_window_ps = tk.DoubleVar(
            value=DEFAULT_COINCIDENCE_WINDOW_PS
        )
        self.settle_seconds = tk.DoubleVar(value=DEFAULT_SETTLE_SECONDS)
        self.delete_raw_files = tk.BooleanVar(value=DEFAULT_DELETE_RAW_FILES)
        self.log_results = tk.BooleanVar(value=DEFAULT_LOG_RESULTS)
        self.status_text = tk.StringVar(value="Ready")
        self.delay_status = tk.StringVar(
            value="Delays: recalibrated every measurement"
        )

        self.total_visibility = tk.StringVar(value="--")
        self.hv_visibility = tk.StringVar(value="--")
        self.da_visibility = tk.StringVar(value="--")
        self.qber = tk.StringVar(value="--")
        self.total_coincidences = tk.StringVar(value="--")
        self.count_variables = {
            label: tk.StringVar(value="--") for label in DISPLAY_PAIR_ORDER
        }
        self.previous_counts: dict[str, int] | None = None
        self.previous_metrics: dict[str, float | int] | None = None
        self.delay_variables = {
            label: tk.StringVar(value="--") for label in DISPLAY_PAIR_ORDER
        }

        self._build_window()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._poll_events)

    def _build_window(self) -> None:
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1500x900")
        self.root.minsize(1180, 720)

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, minsize=430)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(1, weight=1)

        metrics = ttk.Frame(outer)
        metrics.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
        for column in range(5):
            metrics.columnconfigure(column, weight=1)
        self._metric(metrics, 0, "Total visibility", self.total_visibility)
        self._metric(metrics, 1, "H/V visibility", self.hv_visibility)
        self._metric(metrics, 2, "D/A visibility", self.da_visibility)
        self._metric(metrics, 3, "QBER", self.qber)
        self._metric(metrics, 4, "Coincidences", self.total_coincidences)

        controls = ttk.Frame(outer, padding=(0, 0, 12, 0))
        controls.grid(row=1, column=0, sticky="nsew")
        controls.columnconfigure(0, weight=1)

        self.alice_control = VoltageControl(
            controls,
            "Alice EPC",
            enabled_default=ALICE_EPC_ENABLED,
        )
        self.alice_control.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        self.bob_control = VoltageControl(
            controls,
            "Bob EPC",
            enabled_default=True,
        )
        self.bob_control.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        settings = ttk.LabelFrame(
            controls,
            text="Measurement",
            padding=8,
        )
        settings.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        settings.columnconfigure(1, weight=1)
        self._numeric_setting(
            settings,
            0,
            "Exposure",
            self.exposure_seconds,
            0.1,
            3600.0,
            0.5,
            "s",
        )
        self._numeric_setting(
            settings,
            1,
            "Analysis coincidence window",
            self.coincidence_window_ps,
            1.0,
            1_000_000.0,
            10.0,
            "ps",
        )
        self._numeric_setting(
            settings,
            2,
            "EPC settling",
            self.settle_seconds,
            0.0,
            60.0,
            0.1,
            "s",
        )
        ttk.Checkbutton(
            settings,
            text="Delete successful raw recordings",
            variable=self.delete_raw_files,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))
        ttk.Checkbutton(
            settings,
            text="Append result to alice_results.csv",
            variable=self.log_results,
        ).grid(row=4, column=0, columnspan=3, sticky="w")

        actions = ttk.Frame(controls)
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        for column in range(2):
            actions.columnconfigure(column, weight=1)
        self.apply_button = ttk.Button(
            actions,
            text="Apply",
            command=self._apply_only,
        )
        self.apply_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.zero_button = ttk.Button(
            actions,
            text="Zero enabled",
            command=self._zero_enabled,
        )
        self.zero_button.grid(row=0, column=1, sticky="ew", padx=4)
        self.record_button = ttk.Button(
            actions,
            text="Record now",
            command=self._record_now,
        )
        self.record_button.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(6, 0),
        )
        self.measure_button = ttk.Button(
            actions,
            text="Apply + measure",
            command=self._measure_once,
        )
        self.measure_button.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=4,
            pady=(6, 0),
        )
        self.continuous_button = ttk.Button(
            actions,
            text="Start continuous",
            command=self._start_continuous,
        )
        self.continuous_button.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(6, 0),
        )
        self.stop_button = ttk.Button(
            actions,
            text="Stop",
            command=self._stop_continuous,
            state="disabled",
        )
        self.stop_button.grid(
            row=2,
            column=1,
            sticky="ew",
            padx=4,
            pady=(6, 0),
        )

        results = ttk.LabelFrame(controls, text="Latest result", padding=8)
        results.grid(row=4, column=0, sticky="ew")
        for column in range(4):
            results.columnconfigure(column, weight=1)
        for index, label in enumerate(DISPLAY_PAIR_ORDER):
            row = index // 4
            column = index % 4
            cell = ttk.Frame(results)
            cell.grid(row=row, column=column, sticky="ew", padx=4, pady=2)
            ttk.Label(cell, text=label, font=("Segoe UI", 9, "bold")).pack()
            ttk.Label(cell, textvariable=self.count_variables[label]).pack()
            ttk.Label(
                cell,
                textvariable=self.delay_variables[label],
                foreground="#666666",
            ).pack()
        ttk.Label(
            controls,
            textvariable=self.delay_status,
        ).grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Label(
            controls,
            textvariable=self.status_text,
            wraplength=410,
        ).grid(row=6, column=0, sticky="ew", pady=(4, 0))

        plot_frame = ttk.Frame(outer)
        plot_frame.grid(row=1, column=1, sticky="nsew")
        self.figure = Figure(figsize=(9, 7), dpi=100)
        self.visibility_axis = self.figure.add_subplot(211)
        self.count_axis = self.figure.add_subplot(212)
        self.visibility_axis.set_ylim(-1.05, 1.05)
        self.visibility_axis.set_ylabel("Visibility")
        self.visibility_axis.grid(True, alpha=0.25)
        self.count_axis.set_ylabel("Total coincidences")
        self.count_axis.set_xlabel("Measurement")
        self.count_axis.grid(True, alpha=0.25)
        self.visibility_lines = {
            "total": self.visibility_axis.plot(
                [],
                [],
                marker="o",
                label="Total",
                color="black",
            )[0],
            "HV": self.visibility_axis.plot(
                [],
                [],
                marker="o",
                label="H/V",
                color="#0072b2",
            )[0],
            "DA": self.visibility_axis.plot(
                [],
                [],
                marker="o",
                label="D/A",
                color="#d55e00",
            )[0],
        }
        self.visibility_axis.legend(loc="best")
        (self.count_line,) = self.count_axis.plot(
            [],
            [],
            marker="o",
            color="#333333",
        )
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    @staticmethod
    def _metric(parent, column: int, title: str, variable: tk.StringVar) -> None:
        frame = ttk.Frame(parent, padding=(8, 4))
        frame.grid(row=0, column=column, sticky="ew")
        ttk.Label(frame, text=title).pack()
        ttk.Label(
            frame,
            textvariable=variable,
            font=("Segoe UI", 20, "bold"),
        ).pack()

    @staticmethod
    def _numeric_setting(
        parent,
        row: int,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        increment: float,
        unit: str,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        ttk.Spinbox(
            parent,
            from_=minimum,
            to=maximum,
            increment=increment,
            textvariable=variable,
            width=12,
        ).grid(row=row, column=1, sticky="ew", padx=8, pady=2)
        ttk.Label(parent, text=unit).grid(row=row, column=2, sticky="w")

    def _settings(self) -> MeasurementSettings:
        exposure = float(self.exposure_seconds.get())
        window = float(self.coincidence_window_ps.get())
        settle = float(self.settle_seconds.get())
        if exposure <= 0:
            raise ValueError("Exposure must be positive")
        if window <= 0:
            raise ValueError("Coincidence window must be positive")
        if settle < 0:
            raise ValueError("EPC settling time cannot be negative")

        return MeasurementSettings(
            exposure_seconds=exposure,
            coincidence_window_ps=window,
            settle_seconds=settle,
            alice_enabled=bool(self.alice_control.enabled.get()),
            bob_enabled=bool(self.bob_control.enabled.get()),
            alice_voltages=self.alice_control.values(),
            bob_voltages=self.bob_control.values(),
            delete_raw_files=bool(self.delete_raw_files.get()),
            log_results=bool(self.log_results.get()),
        )

    def _record_duration_seconds(self) -> float:
        duration = float(self.exposure_seconds.get())
        if duration <= 0:
            raise ValueError("Exposure must be positive")
        return duration

    def _apply_only(self) -> None:
        try:
            settings = self._settings()
        except Exception as exc:
            messagebox.showerror(WINDOW_TITLE, str(exc))
            return
        self._run_worker("apply", self.engine.apply_voltages, settings)

    def _zero_enabled(self) -> None:
        if self.alice_control.enabled.get():
            self.alice_control.zero()
        if self.bob_control.enabled.get():
            self.bob_control.zero()
        self._apply_only()

    def _measure_once(self) -> None:
        self.continuous = False
        self._start_measurement()

    def _record_now(self) -> None:
        self.continuous = False
        self.record_requested = True
        self._set_action_state("disabled")
        if self.busy:
            self.status_text.set(
                "Record requested; waiting for the current action to finish"
            )
            return
        self._start_recording()

    def _start_continuous(self) -> None:
        self.continuous = True
        self.stop_button.configure(state="normal")
        self.continuous_button.configure(state="disabled")
        if not self.busy:
            self._start_measurement()

    def _stop_continuous(self) -> None:
        self.continuous = False
        self.stop_button.configure(state="disabled")
        self.continuous_button.configure(state="normal")
        if self.busy:
            self.status_text.set("Stopping after the current measurement")
        else:
            self.status_text.set("Continuous measurement stopped")

    def _start_measurement(self) -> None:
        try:
            settings = self._settings()
        except Exception as exc:
            self.continuous = False
            self._set_action_state("normal")
            messagebox.showerror(WINDOW_TITLE, str(exc))
            return
        self._run_worker("measurement", self.engine.measure, settings)

    def _start_recording(self) -> None:
        try:
            duration_seconds = self._record_duration_seconds()
        except Exception as exc:
            self.record_requested = False
            self._set_action_state("normal")
            messagebox.showerror(WINDOW_TITLE, str(exc))
            return
        self.record_requested = False
        self._run_worker("recording", self.engine.record, duration_seconds)

    def _run_worker(self, event_name: str, function, *args) -> None:
        if self.busy:
            return
        self.busy = True
        self.active_event_name = event_name
        self._set_action_state("disabled")
        if event_name == "measurement":
            status = "Applying voltages and measuring"
        elif event_name == "recording":
            duration_seconds = float(args[0])
            status = f"Recording raw data for {duration_seconds:g} s"
        else:
            status = "Applying voltages"
        self.status_text.set(status)

        def run() -> None:
            try:
                result = function(*args)
            except Exception as exc:
                self.events.put(("error", exc))
            else:
                self.events.put((event_name, result))

        threading.Thread(target=run, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                event_name, payload = self.events.get_nowait()
                self.busy = False
                self.active_event_name = None
                if event_name == "error":
                    self.continuous = False
                    self.status_text.set(f"Error: {payload}")
                    messagebox.showerror(WINDOW_TITLE, str(payload))
                elif event_name == "apply":
                    self.status_text.set("Enabled EPC voltages applied")
                elif event_name == "recording":
                    self._show_recording(payload)
                elif event_name == "measurement":
                    self._show_measurement(payload)

                if self.record_requested and not self.closing:
                    self._start_recording()
                elif self.continuous and not self.closing:
                    self._set_action_state("normal")
                    self.root.after(200, self._start_measurement)
                elif self.closing:
                    self._finish_close()
                    return
                else:
                    self._set_action_state("normal")
        except queue.Empty:
            pass

        if self.root.winfo_exists():
            self.root.after(100, self._poll_events)

    def _show_recording(self, acquisition: AcquisitionPair) -> None:
        self.status_text.set(
            f"Recorded {acquisition.record_id}; "
            f"Alice={acquisition.alice_path.name}, Bob={acquisition.bob_path.name}"
        )

    def _show_measurement(self, event: MeasurementEvent) -> None:
        result = event.result
        self.measurement_number += 1
        metrics: dict[str, float | int] = {
            "total_visibility": result.visibility,
            "hv_visibility": result.basis_visibility["HV"],
            "da_visibility": result.basis_visibility["DA"],
            "qber": result.qber_total,
            "total_coincidences": result.total_coincidences,
        }
        self.total_visibility.set(
            self._format_percent_metric("total_visibility", metrics)
        )
        self.hv_visibility.set(
            self._format_percent_metric("hv_visibility", metrics)
        )
        self.da_visibility.set(
            self._format_percent_metric("da_visibility", metrics)
        )
        self.qber.set(self._format_percent_metric("qber", metrics))
        total_count = int(metrics["total_coincidences"])
        if self.previous_metrics is None:
            total_count_text = str(total_count)
        else:
            count_delta = total_count - int(
                self.previous_metrics["total_coincidences"]
            )
            total_count_text = f"{total_count} ({count_delta:+d})"
        self.total_coincidences.set(total_count_text)

        for label in DISPLAY_PAIR_ORDER:
            count = result.counts[label]
            if self.previous_counts is None:
                count_text = str(count)
            else:
                delta = count - self.previous_counts[label]
                count_text = f"{count} ({delta:+d})"
            self.count_variables[label].set(count_text)
            self.delay_variables[label].set(
                f"{result.delays_ps[label] / 1000.0:+.3f} ns"
            )
        self.previous_counts = dict(result.counts)
        self.previous_metrics = metrics

        self.delay_status.set("Delays: recalibrated for latest measurement")
        self.status_text.set(
            f"Completed record {event.record_id}; "
            f"window={result.sync.coincidence_window_ps:g} ps"
        )

        self.history_index.append(self.measurement_number)
        self.history_total.append(result.visibility)
        self.history_hv.append(result.basis_visibility["HV"])
        self.history_da.append(result.basis_visibility["DA"])
        self.history_counts.append(result.total_coincidences)
        for key, values in (
            ("total", self.history_total),
            ("HV", self.history_hv),
            ("DA", self.history_da),
        ):
            self.visibility_lines[key].set_data(self.history_index, values)
        self.count_line.set_data(
            self.history_index,
            self.history_counts,
        )
        self.visibility_axis.relim()
        self.visibility_axis.autoscale_view(scaley=False)
        self.count_axis.relim()
        self.count_axis.autoscale_view()
        self.canvas.draw_idle()

    def _format_percent_metric(
        self,
        name: str,
        metrics: dict[str, float | int],
    ) -> str:
        value = float(metrics[name])
        text = f"{100.0 * value:.2f}%"
        if self.previous_metrics is None:
            return text
        delta_points = 100.0 * (
            value - float(self.previous_metrics[name])
        )
        return f"{text} ({delta_points:+.2f} pp)"

    def _set_action_state(self, state: str) -> None:
        self.apply_button.configure(state=state)
        self.zero_button.configure(state=state)
        self.measure_button.configure(state=state)
        record_state = (
            "disabled"
            if self.closing
            or self.record_requested
            or self.active_event_name == "recording"
            else "normal"
        )
        self.record_button.configure(state=record_state)
        if not self.continuous:
            self.continuous_button.configure(state=state)
        if self.continuous:
            self.stop_button.configure(state="normal")
        else:
            self.stop_button.configure(state="disabled")

    def _on_close(self) -> None:
        self.closing = True
        self.continuous = False
        if self.busy:
            self.status_text.set("Closing after the current measurement")
            self._set_action_state("disabled")
            return
        self._finish_close()

    def _finish_close(self) -> None:
        try:
            self.engine.shutdown()
        finally:
            self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    ManualEpcGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
