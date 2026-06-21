# Long-Distance QKD Timetag Synchronization

This repository processes timetag files recorded by two independent quTAG devices. The important problem is not starting both recordings at exactly the same instant. It is:

1. Pair the Alice and Bob files belonging to the same acquisition.
2. Read the independent timetag streams.
3. Decode synchronization markers recorded by both taggers.
4. Use matching marker counters to estimate relative clock offset and skew.
5. Correct Bob's photon timetags into Alice's time base.
6. Find and evaluate coincidences between selected Alice/Bob channels.

The synchronization signal is required because Alice and Bob have independent timestamp origins and clocks, and most importantly not enough signal.
A shared filename or scheduled software start is useful for pairing recordings, but is not precise enough for photon coincidences.

## Intended Acquisition Layout

The final acquisition runs on separate PCs:

- Alice controls the experiment and records Alice's local tags.
- Bob runs a command server and records Bob's local tags when requested.
- Alice chooses a shared `record_id` and a future software start time.
- Both PCs record locally using the same `record_id`.
- Bob's completed BIN file is transferred to Alice for offline processing.

Typical paired files are:

```text
Data/AliceRaw/alice_20260413T143812.123456Z_exp_10.0s.bin
Data/Incoming/bob_20260413T143812.123456Z_exp_10.0s.bin
```

The matching `record_id` identifies the pair. The scheduled start only ensures that their acquisition windows overlap. Do not assume that time zero in the two files is equal.

The earlier acquisition/network design, including the Alice/Bob command flow, file transfer, and PC clock diagnostics, is described in `README_old.md`.
`qkd_network.py`, `qkd_names.py`, and `receive_data_zip.py` contain the shared network transfer and naming utilities currently present in this directory.

`record_dual_qutag.py` records two locally connected quTAG devices and is useful for development and controlled tests. It is not the intended final separate-PC acquisition method.

## Computer Folder Structure

Alice and Bob do not need identical copies of this repository. Alice performs acquisition control, synchronization, coincidence analysis, logging, and optional EPC optimization. Bob only records its local timetags, controls its local EPC, and responds to Alice over the network.

### Alice computer

A minimal Alice installation should look like:

```text
LongDistanceQKD/
├── Alice.py
├── qkd_acquisition.py
├── qkd_sync.py
├── qkd_epc_correction.py
├── qkd_plot_delay_scans.py
├── qkd_epc.py
├── qkd_network.py
├── qkd_names.py
├── QuTAG_MC.py
├── coincfinder.cpython-312-x86_64-linux-gnu.so
├── EPC/                         # or AEPC/, containing EPC.py and driver files
└── Data/
    ├── AliceRaw/                # Alice BIN files, created automatically
    ├── Incoming/                # Bob BIN files received by Alice
    ├── CoincidenceTimetags/     # optional matched-pair NPZ files
    ├── DelayScans/               # initial delay-scan figure
    ├── alice_results.csv        # synchronization, coincidence, visibility, QBER
    ├── optimizer_state.json     # best EPC voltages and visibility
    └── qber_iterlog.csv         # optimizer evaluations
```

The files and directories under `Data/` are created as needed. `CoincidenceTimetags/` is only used when `SYNC_PROCESSING.store_coincidence_timetags` is enabled.

Alice requires:

- the quTAG Python wrapper and its vendor libraries;
- the Alice EPC driver directory, named either `EPC/` or `AEPC/`;
- NumPy;
- a compatible compiled `coincfinder` module;
- SciPy when `QBER_OPTIMIZATION_ENABLED = True`.

Configure the Bob address, port, local data directories, coincidence channels, synchronization settings, and optimizer settings near the top of `Alice.py`.

### Bob computer

A minimal Bob installation should look like:

```text
LongDistanceQKD/
├── Bob.py
├── qkd_epc.py
├── qkd_network.py
├── qkd_names.py
├── QuTAG_MC.py
├── EPC/
└── BobData/
```

Bob does not need `qkd_sync.py`, `qkd_epc_correction.py`, `qkd_acquisition.py`, `coincfinder`, NumPy, or SciPy for the current server workflow.

Set `BOB_CONFIG.record_dir` in `Bob.py` to the real `BobData` directory. The current configuration uses an absolute Windows path, so it must match the Bob computer.
Also confirm that `BOB_CONFIG.port` in `Bob.py` matches `ACQUISITION.bob_port` in `Alice.py`, and allow that TCP port through the Bob computer's firewall.

## What Bob Does

Run `Bob.py` before starting `Alice.py`. At startup Bob:

1. initializes its local quTAG;
2. initializes its local EPC and sets the configured starting temperature;
3. creates a `BobRecorder` for local BIN acquisition;
4. opens a `BobCommandServer` on `BOB_CONFIG.host:BOB_CONFIG.port`, currently `0.0.0.0:5001`;
5. waits for one command connection at a time from Alice.

The server uses short polling timeouts while waiting for connections and commands. This prevents an idle socket from blocking forever and allows Ctrl+C to stop Bob promptly. A Ctrl+C during an active recording also stops timestamp writing in the recorder's cleanup path before the tagger is deinitialized.

Bob supports these commands:

| Command | Bob action |
| --- | --- |
| `PING` | Replies with `PONG` so Alice can check connectivity. |
| `TIME_CHECK` | Returns Bob's current UTC time. |
| `RECORD` | Waits until Alice's scheduled UTC start, records a local BIN file, reports completion, and transfers the file to Alice with metadata and SHA-256 verification. |
| `SET_VOLTAGES` | Validates and applies four voltages to Bob's EPC. |
| `ZERO_VOLTAGES` | Sets all four Bob EPC voltages to zero. |
| `SET_TEMPERATURE` | Changes Bob's EPC temperature. |
| `DELETE_RECORDING` | Deletes one named `bob_*.bin` file from Bob's configured recording directory after Alice has processed it. |
| `STOP` | Replies to Alice, leaves the server loop, stops timestamp writing, and deinitializes Bob's quTAG. |

For a `RECORD` command, Bob saves:

```text
BobData/bob_<record_id>_exp_<seconds>s.bin
```

Bob keeps this local file after sending it. Alice receives a copy under `Data/Incoming/`. The shared `record_id` is checked by Alice before the pair is accepted. When Alice exits, `shutdown_tagger()` sends Bob the `STOP` command and then deinitializes Alice's local tagger.

## Module Responsibilities

The runtime flow is split into modules with one primary responsibility each:

- `Alice.py`: experiment configuration and top-level orchestration;
- `qkd_acquisition.py`: request Bob data, record Alice data, receive Bob's file, and validate the paired acquisition;
- `qkd_sync.py`: decode synchronization markers, align clocks, and extract coincidences;
- `qkd_epc_correction.py`: calculate Phi+ visibility/QBER, log results, and run the EPC optimizer;
- `qkd_epc.py`: low-level EPC initialization and voltage/temperature calls;
- `qkd_network.py`: length-prefixed JSON and checksum-protected file transfer;
- `qkd_names.py`: UTC timestamps, record IDs, and filename sanitization.

`qkd_acquisition.acquire_pair()` returns an `AcquisitionPair` containing the shared record ID and both completed file paths. Acquisition failures identify the failed stage and record ID in the exception message.

`qkd_sync.py` handles only timestamp synchronization and coincidence extraction:

- compact Manchester sync-marker decoding;
- sync-counter matching between Alice and Bob;
- Bob-to-Alice timestamp interpolation;
- residual photon-delay scans;
- coincidence collection and accidental estimates;
- optional storage of matched coincidence timetag pairs.

Use it directly when another script only needs synchronized coincidence data:

```python
from qkd_sync import analyze_sync_coincidences, save_coincidence_timetag_pairs

pairs = [
    ("HH", 1, 1),  # name, Alice channel, Bob channel
    ("VV", 2, 2),
]
sync = analyze_sync_coincidences(
    alice_path,
    bob_path,
    pairs,
    sync_channel=5,
    coincidence_window_ps=200,
)

for result in sync.pair_results:
    print(result.pair.name, result.count, result.best_delay_ps)
    coincidence_timetags = result.coincidences_ps  # columns: Alice ps, aligned Bob ps

save_coincidence_timetag_pairs(sync, "Data/CoincidenceTimetags")
```

`qkd_epc_correction.py` owns the Phi+ interpretation, result logging, optimizer state, and selectable Nelder-Mead/Nevergrad optimization backends. It consumes the already-synchronized coincidence result:

```python
from qkd_epc_correction import DEFAULT_PHI_PLUS_PAIRS, analyze_phi_plus_coincidences
from qkd_sync import analyze_sync_coincidences

sync = analyze_sync_coincidences(alice_path, bob_path, DEFAULT_PHI_PLUS_PAIRS)
correction = analyze_phi_plus_coincidences(sync)

print(correction.visibility, correction.qber_total, correction.total_coincidences)
# PhiPlusOptimizer uses this result directly as the optimization measurement.
```

The default Phi+ channel map is defined in `qkd_epc_correction.py`:

```python
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
```

This mapping must match the acquisition hardware. If Bob's detector channel convention differs, update `QKD_COINCIDENCE_PAIRS` in `Alice.py` or pass an explicit pair list to `qkd_sync.py`.

## Alice End-To-End Flow

`Alice.py` now follows the acquisition and correction sequence explicitly:

1. Ask Bob for data with a `RECORD` command.
2. Wait until the scheduled start time and record Alice's local BIN file.
3. Receive Bob's completed BIN file.
4. Sync Alice/Bob timetags with `qkd_sync.analyze_sync_coincidences()`.
5. Get per-label coincidence counts and matched timetag pairs from the synced data.
6. Optionally store the matched coincidence timetag pairs by setting `SYNC_PROCESSING.store_coincidence_timetags=True`.
7. Give the synchronized coincidence result to `qkd_epc_correction.analyze_phi_plus_coincidences()`.
8. Log visibility, QBER, coincidence counts, delays, accidentals, and sync diagnostics.
9. If optimization is enabled, the correction algorithm requests a fresh acquisition and repeats steps 1-8 for every objective evaluation.

The result rows are appended to `Data/alice_results.csv`. They include sync-marker count, clock-skew summary, coincidence counts, accidental estimates, per-pair delays, HV/DA visibility, HV/DA QBER, total QBER, total contrast, total coincidences, and `optimization_score`.

When `SYNC_PROCESSING.save_initial_delay_scan=True`, Alice captures fine-delay scan data only for the first measurement after startup and saves one combined figure under:

```text
Data/DelayScans/initial_delay_scans_<record_id>.png
```

Later passive or optimizer measurements still find their delays normally but do not retain scan arrays or create more figures.

`qkd_epc_correction.PhiPlusOptimizer` implements the optional correction loop based on the local `Stability_Check_and_Record.py` approach:

- set `QBER_OPTIMIZATION_ENABLED = True` to enable it;
- it optimizes eight voltages, `Alice[0:4] + Bob[4:8]`;
- choose `backend="nelder-mead"` for SciPy Nelder-Mead or `backend="nevergrad"` for a Nevergrad optimizer;
- each objective evaluation applies voltages, requests new Alice/Bob data, syncs it, extracts coincidences, and computes the EPC correction metric;
- the best measured voltage vector is restored and saved in `Data/optimizer_state.json`;
- per-evaluation rows, including backend, optimizer name, voltages, visibility, and all eight coincidence counts, are written to `Data/qber_iterlog.csv`.

Nevergrad is optional. Install it only on Alice when that backend is selected:

```powershell
python -m pip install nevergrad
```

Configure the backend near the top of `Alice.py`:

```python
OPTIMIZER = OptimizerConfig(
    backend="nevergrad",       # "nevergrad" or "nelder-mead"
    optimize_epcs="both",      # "alice", "bob", or "both"
    measurement_seconds=5.0,
    base_step_volts=25.0,
    nevergrad_optimizer="TBPSA",  # for example "TBPSA" or "NGOpt"
    nevergrad_budget=100,      # maximum hardware measurements per run
    nevergrad_seed=None,
)
```

`optimize_epcs` controls which voltage dimensions are searched:

- `"alice"` optimizes Alice DAC0-3 and holds Bob's four saved voltages fixed;
- `"bob"` optimizes Bob DAC0-3 and holds Alice's four saved voltages fixed;
- `"both"` optimizes all eight voltages.

The optimizer state and iteration CSV always retain the complete eight-voltage
vector, including the fixed values when only one EPC is optimized.

Nevergrad uses its sequential `ask`/`tell` interface because Alice can run only one paired hardware acquisition at a time. The objective passed to Nevergrad is `-visibility`, since Nevergrad minimizes. `TBPSA` is the default noise-oriented choice; `NGOpt` is a reasonable adaptive alternative.

Nevergrad optimizer names are case-sensitive and depend on the installed
Nevergrad version. Print every available optimizer on Alice with:

```powershell
python -c "import nevergrad as ng; print('\n'.join(sorted(ng.optimizers.registry)))"
```

Practical choices for the eight-voltage EPC problem are:

| Optimizer | Suggested use |
| --- | --- |
| `TBPSA` | Recommended starting point for noisy visibility measurements. |
| `NGOpt` | Adaptive general-purpose optimizer. |
| `NgIohTuned` | Tuned adaptive meta-optimizer. |
| `NoisyOnePlusOne` | Simple optimizer designed for noisy objectives. |
| `NoisyDE` | Noise-aware differential evolution; use a larger budget. |
| `OnePlusOne` | Simple sequential baseline. |
| `PSO` | Robust broad exploration. |
| `TwoPointsDE` | Population-based exploration with a larger budget. |
| `CMA` | Useful when measurement noise is lower and the budget is large. |
| `RandomSearch` | Baseline for judging whether an optimizer adds value. |

Compare optimizers using the same exposure, starting voltages, mutation step,
and measurement budget. Start with `TBPSA`, then compare it against `NGOpt`.

Leave `QBER_OPTIMIZATION_ENABLED = False` in `Alice.py` for passive acquisition and logging.

### Optimizer raw-file cleanup

Optimizer measurements use `MeasurementPipeline.measure_for_optimizer()`. After synchronization, coincidence analysis, and CSV logging succeed, Alice:

1. sends Bob `DELETE_RECORDING` for the transferred Bob filename;
2. deletes Alice's local file from `Data/AliceRaw/`;
3. deletes Alice's received Bob copy from `Data/Incoming/`.

Passive measurements still retain all raw files. If synchronization or correction fails, the files are retained for diagnosis. A deletion failure is printed as a warning and does not discard the completed optimizer result.

## Diagnostic Scripts

### `decode_sync_timestamps.py`

This is the first diagnostic for a new pair of files. It verifies that the sync channel can be read, markers can be detected, counters can be decoded, and matching counters show a sensible timing offset and drift.

It reads complete `tagger_a`/`tagger_b` pairs from `DataTTTestsNewMethod/DualRaw`, decodes channel `5`, writes `decoded_sync.csv`, and saves `decoded_sync.png`.

```bash
python3 decode_sync_timestamps.py
```

Pair selection is controlled by `DEFAULT_TIMESTAMP` near the top of the file:

- `DEFAULT_TIMESTAMP = None` processes the latest complete pair.
- Setting it to a shared timestamp such as `"20260609T104513.682883Z"` processes that exact pair.

### `align_sync_coincidences.py`

This is a thin command-line wrapper over `qkd_sync.analyze_sync_coincidences()`. It selects the latest complete Alice/Bob pair from `DataExtSync/AliceRaw` and `DataExtSync/Incoming`. Edit `COINCIDENCE_PAIRS` near the top of the file and run:

```bash
python3 align_sync_coincidences.py
```

It prints shared sync markers, overlap duration, best per-pair delays, coincidence counts, and accidental estimates.

### `qkd_plot_measurements.py`

This standalone script plots measurement CSV files without importing or running Alice/Bob hardware code. It displays all eight coincidence pairs, total coincidences, and H/V plus D/A visibility. By default it follows `Data/alice_results.csv` live and refreshes when the file changes:

```bash
python qkd_plot_measurements.py
```

Configure the constants near the top of `qkd_plot_measurements.py`:

```python
CSV_FILE = BASE_DIR / "Data" / "qber_iterlog.csv"
REFRESH_INTERVAL_SECONDS = 2.0
HISTORY_ROWS = 200       # Use 0 for all rows.
LIVE_UPDATE = True       # False gives one static plot.
SAVE_PATH = None         # Or BASE_DIR / "out" / "qkd_measurements.png"
```

Then run:

```bash
python qkd_plot_measurements.py
```

When `vis_HV` and `vis_DA` are not present, as in the optimizer log, the script calculates them from the eight coincidence columns.

### `qkd_epc_manual_gui.py`

Run this standalone GUI on Alice when manual EPC testing is needed without
the optimizer:

```bash
python qkd_epc_manual_gui.py
```

Start `Bob.py` first. The GUI provides:

- enable toggles and four 0-130 V controls for the Alice and Bob EPCs;
- Apply, Zero, one-shot measurement, and continuous measurement controls;
- configurable paired-recording exposure, analysis coincidence window, and
  EPC settling time;
- total, H/V, and D/A visibility, QBER, all eight coincidence counts, and
  the delays used for counting;
- a live visibility and total-coincidence history plot;
- optional result logging to `Data/alice_results.csv` and deletion of
  successful raw acquisitions.

Disable Alice EPC control for a Bob-only test, or disable both EPC controls
to use the GUI as a synchronized measurement monitor. Every measurement
performs fresh reference-delay scans using the selected coincidence window.

## Synchronization Signal Assumptions

The compact decoder assumes timestamps are measured in picoseconds and uses this Manchester-style timing:

| Parameter | Value |
| --- | ---: |
| Regular clock period | `1,000,000 ps` (`1 us`) |
| Sync-block start threshold | `900,000 ps` |
| Data period | `62,500 ps` (`1 us / 16`) |
| First decode threshold | `78,125 ps` |
| Second decode threshold | `109,375 ps` |
| End-of-data threshold | `187,500 ps` |

The transmitted block begins with a known marker bit of `1`. Remaining bits are decoded least-significant bit first into the synchronization counter. Update `qkd_sync.py` if the synchronization waveform changes.

## Recommended Order For A New Dataset

1. Pair Alice and Bob files using the shared `record_id`.
2. Confirm the expected sync and photon channels exist in both files.
3. Run `decode_sync_timestamps.py` to validate marker decoding.
4. Configure `QKD_COINCIDENCE_PAIRS`, `ACQUISITION`, `SYNC_PROCESSING`, and `OPTIMIZER` near the top of `Alice.py`.
5. Run `Alice.py` for acquisition plus synchronized analysis.
6. Inspect marker count, clock-skew summary, coincidence peak delays, visibility, QBER, and accidentals before trusting optimization results.
