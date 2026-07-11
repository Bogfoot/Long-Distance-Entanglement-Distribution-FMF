# Long-Distance Entanglement Distribution / QKD Control

This repository contains the Python control and analysis code used for
long-distance entanglement-distribution and QKD measurements with two
independent quTAG time taggers.

The main problem solved here is not starting Alice and Bob at exactly the same
software time. Alice and Bob have independent timestamp origins and clocks, so
recordings are only useful after the shared synchronization channel has been
decoded. The current pipeline:

1. asks Bob to record a local BIN file;
2. records Alice's local BIN file with the same `record_id`;
3. transfers Bob's file back to Alice;
4. decodes synchronization markers from both taggers;
5. maps Bob timestamps into Alice's time base;
6. scans and counts photon coincidences;
7. calculates Phi+ visibility/QBER and CHSH S metrics;
8. optionally adjusts Alice and Bob EPC voltages and repeats.

Scheduled starts and matching filenames are used to make the acquisition
windows overlap. Coincidence timing comes from the synchronization markers, not
from PC clocks or matching file timestamps.

## Repository Layout

```text
.
|-- Alice.py                         # Alice-side orchestration and config
|-- Bob.py                           # Bob command server and recorder
|-- qkd_acquisition.py               # paired Alice/Bob acquisition
|-- qkd_sync.py                      # sync-marker decoding and coincidences
|-- qkd_epc_correction.py            # Phi+ metrics, CHSH S, optimizer
|-- qkd_epc.py                       # EPC driver wrapper
|-- qkd_network.py                   # JSON and checksum file transfer
|-- qkd_names.py                     # UTC IDs and safe filenames
|-- qkd_plot_delay_scans.py          # delay-scan plotting helper
|-- qkd_manual_epc_gui.py            # manual EPC and measurement GUI
|-- qkd_sweep_voltages_alice.py      # Alice/Bob EPC voltage sweep
|-- qkd_analyze_recorded_pairs.py    # offline exposure analysis for raw pairs
|-- qkd_plot_from_files.py           # live/static results plotting
|-- qkd_plot_singles_from_file.py    # singles plotting from results CSV
|-- qkd_plot_chsh_convention.py      # CHSH sign/output-swap diagnostics
|-- qkd_plot_epc_sweep_from_file.py  # plot saved EPC sweep CSV files
|-- Temperature_logger.py            # Open-Meteo temperature logger
|-- Temperature_analysis.py          # temperature/performance correlation
|-- alice_time_check.py              # Alice/Bob clock and RTT diagnostic
|-- qkd_test_epc.py                  # local EPC hardware smoke test
|-- collect_alice_bob.py             # older paired-recording helper
|-- send_data_zip.py                 # standalone file sender
|-- Data_Receiever.py                # standalone file receiver
|-- QuTAG_MC.py                      # quTAG vendor Python wrapper
|-- AEPC/                            # EPC Python driver and docs
`-- Data/                            # logged measurements and derived plots
```

`Data/MDPUVTP/` contains historical measurement logs and plotting scripts. It
is not part of the current Alice/Bob runtime loop.

## Hardware And Python Requirements

Alice normally needs:

- a local quTAG and matching vendor DLLs for `QuTAG_MC.py`;
- a working `coincfinder` Python module, used by `qkd_sync.py`;
- NumPy, Matplotlib, Pandas, and SciPy;
- the Alice EPC driver directory, named `EPC/` or `AEPC/`, when Alice EPC
  control is enabled;
- Nevergrad only when `OPTIMIZER.backend = "nevergrad"`.

Bob normally needs:

- a local quTAG and matching vendor DLLs for `QuTAG_MC.py`;
- the Bob EPC driver directory, named `EPC/` or `AEPC/`, when Bob EPC control
  is enabled;
- network access from Alice to Bob's command port.

The bundled `AEPC/` directory contains the Python EPC wrapper and documentation.
The vendor command-line files such as `MCP2210CLI.exe` and
`MCP2210DLL-UM.dll` are not present in this checkout; place them beside the EPC
driver if that backend is used. `qkd_epc.py` first tries `EPC.EPC` and then
falls back to `AEPC.EPC`.

There is no `requirements.txt` in the current tree. Install the scientific
packages and hardware/vendor modules in the Python environment used on each
machine.

## Quick Start For A Hardware Run

1. Configure Bob in `Bob.py`.

   Set `BOB_CONFIG.record_dir` to the real Bob data directory, confirm
   `BOB_CONFIG.port`, and set the EPC device reference/temperature if needed.
   The current checked-in Bob data path is a Windows path:

   ```python
   record_dir=Path(r"C:\Users\RKAdmin\Desktop\LongDistanceQKD\BobData")
   ```

2. Start Bob first:

   ```bash
   python Bob.py
   ```

   Bob initializes its tagger and EPC, listens on `0.0.0.0:5001`, records local
   BIN files when Alice sends `RECORD`, transfers completed files to Alice, and
   accepts voltage/temperature commands.

3. Configure Alice in `Alice.py`.

   The most important values are:

   | Setting | Purpose |
   | --- | --- |
   | `ACQUISITION.bob_host` / `bob_port` | Bob command endpoint. |
   | `RECORD_SECONDS` | Passive acquisition duration. |
   | `QBER_OPTIMIZATION_ENABLED` | `True` runs the optimizer loop; `False` records passively. |
   | `SYNC_PROCESSING.sync_channel` | Shared sync marker channel, currently channel `5`. |
   | `SYNC_PROCESSING.coincidence_window_ps` | Coincidence window in picoseconds. |
   | `SYNC_PROCESSING.coincidence_pairs` | Active pair map for synchronization/counting. |
   | `SYNC_PROCESSING.analysis_exposure_seconds` | Per-exposure bin size before aggregation. |
   | `OPTIMIZER` | EPC optimization backend, objective, targets, and voltage limits. |

   The current `SYNC_PROCESSING` default uses `CHSH_COINCIDENCE_PAIRS` and
   `CHSH_DELAY_REFERENCE_PAIRS`. `QKD_COINCIDENCE_PAIRS` is also defined in
   `Alice.py` for the eight-pair Phi+ QKD view.

4. Start Alice:

   ```bash
   python Alice.py
   ```

   Alice records paired data, processes it in exposure windows, appends Phi+
   and CHSH rows to `Data/alice_results.csv`, and saves the first delay-scan
   figure under `Data/DelayScans/` when enabled.

5. Watch or inspect results:

   ```bash
   python qkd_plot_from_files.py
   python qkd_plot_singles_from_file.py
   ```

   These scripts default to `Data/alice_results.csv` and refresh live when
   `LIVE_UPDATE = True`.

## Bob Command Server

`Bob.py` accepts one command connection at a time. Supported commands are:

| Command | Bob action |
| --- | --- |
| `PING` | Replies with `PONG`. |
| `TIME_CHECK` | Returns Bob's current UTC time. |
| `RECORD` | Waits until Alice's scheduled UTC start, records a local BIN file, and transfers it to Alice with SHA-256 metadata. |
| `SET_VOLTAGES` | Applies four Bob EPC voltages after validation. |
| `ZERO_VOLTAGES` | Sets all four Bob EPC voltages to zero. |
| `SET_TEMPERATURE` | Changes the Bob EPC temperature. |
| `DELETE_RECORDING` | Deletes one validated `bob_*.bin` from Bob's configured recording directory. |
| `STOP` | Stops the server loop and deinitializes Bob's tagger. |

Bob writes files as:

```text
BobData/bob_<record_id>_exp_<seconds>s.bin
```

Alice receives the copy in:

```text
Data/Incoming/bob_<record_id>_exp_<seconds>s.bin
```

Alice records its local file in:

```text
Data/AliceRaw/alice_<record_id>_exp_<seconds>s.bin
```

## Current Coincidence Maps

The current eight-pair Phi+ map in `Alice.py` is:

```python
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
```

The CHSH map extends this to sixteen labels:

```python
CHSH_COINCIDENCE_PAIRS = (
    ("HH", 4, 1), ("HV", 4, 2), ("VH", 2, 1), ("VV", 2, 2),
    ("HA", 4, 3), ("HD", 4, 4), ("VA", 2, 3), ("VD", 2, 4),
    ("DH", 1, 1), ("DV", 1, 2), ("AH", 3, 1), ("AV", 3, 2),
    ("DD", 1, 4), ("DA", 1, 3), ("AD", 3, 4), ("AA", 3, 3),
)
```

These channel numbers are hardware conventions. If detector cabling or Bob's
channel convention changes, update `Alice.py` before trusting visibility, QBER,
or CHSH results.

## Synchronization And Coincidence Analysis

`qkd_sync.py` is responsible only for synchronization and coincidence
extraction. It:

- reads tagger files through `coincfinder.read_file_auto()`;
- decodes compact Manchester-style sync markers on the configured sync channel;
- matches Alice/Bob sync counters;
- estimates Bob-to-Alice clock skew and timestamp mapping;
- aligns Bob photon timestamps into Alice's clock;
- scans per-pair or per-reference photon delays;
- counts coincidences and estimates accidentals;
- optionally saves matched coincidence timetag pairs as compressed NPZ files.

The sync decoder constants are in `qkd_sync.py`. The current assumptions are a
`1,000,000 ps` regular clock period, a `62,500 ps` data period, and sync channel
`5`. If the sync waveform changes, update those constants before analyzing new
data.

The high-level API is:

```python
from qkd_sync import analyze_sync_coincidences

sync = analyze_sync_coincidences(
    alice_path,
    bob_path,
    coincidence_pairs,
    sync_channel=5,
    coincidence_window_ps=320.0,
    delay_reference_pairs=delay_reference_pairs,
)

for result in sync.pair_results:
    print(result.pair.name, result.count, result.best_delay_ps)
```

For long recordings, Alice currently uses
`analyze_sync_coincidence_exposures()` and then `aggregate_sync_exposures()`.
This produces stable per-exposure counts while still logging one aggregate row
per paired acquisition.

## Phi+ And CHSH Metrics

`qkd_epc_correction.py` converts synchronized counts into experiment metrics.

For Phi+ QKD rows:

- H/V correlated counts are `HH + VV`;
- H/V error counts are `HV + VH`;
- D/A correlated counts are `DD + AA`;
- D/A error counts are `DA + AD`;
- `visibility` is the mean of H/V and D/A visibility;
- `QBER_total` is the corresponding total QBER estimate.

For CHSH rows, `analyze_chsh_s_coincidences()` calculates:

```text
E_ab   from HH, HV, VH, VV
E_abp  from HD, HA, VD, VA
E_apb  from DH, DV, AH, AV
E_apbp from DD, DA, AD, AA
S      = E_ab - E_abp + E_apb + E_apbp
```

`CHSH_S_value` is `abs(S)`.

Both row types are appended to `Data/alice_results.csv`. Some rows naturally
have blank columns because the CSV schema is shared between Phi+ and CHSH
metrics.

## EPC Optimization

Set `QBER_OPTIMIZATION_ENABLED = True` in `Alice.py` to run the optimizer loop.
Set it to `False` for passive acquisition and logging.

`PhiPlusOptimizer` can optimize:

- Alice voltages only with `OPTIMIZER.optimize_epcs = "alice"`;
- Bob voltages only with `"bob"`;
- all eight voltages with `"both"`.

The full voltage vector is always:

```text
Alice DAC0..3, Bob DAC0..3
```

The optimizer supports:

- `backend="nelder-mead"` through SciPy;
- `backend="nevergrad"` through Nevergrad.

Objective aliases include `visibility`, `vis_HV`, `vis_DA`, and `chsh_s`.
The checked-in configuration uses a primary visibility objective and a
secondary CHSH objective. Optimizer state is saved to
`Data/optimizer_state.json`, and per-evaluation rows are appended to
`Data/qber_iterlog.csv`.

The checked-in optimizer loop is sequential: optimize the primary visibility
objective to target, monitor QBER without optimizer steps until
`QBER_total >= 0.50`, optimize the secondary CHSH S objective to target, then
monitor S without optimizer steps until `CHSH_S_value <= 1.8`. The loop then
returns to the visibility phase. Guard measurements are appended to
`alice_results.csv` by the measurement pipeline and to `qber_iterlog.csv` with
`optimizer_backend="monitor"`. Set either guard to `None` to skip that guard.

Optimizer acquisitions are normally deleted after successful processing.
Failed acquisitions and explicitly retained raw files are kept for diagnosis.

## Output Files

Common generated paths are:

```text
Data/
|-- AliceRaw/                  # Alice raw BIN files, created as needed
|-- Incoming/                  # Bob raw BIN copies received by Alice
|-- CoincidenceTimetags/       # optional matched-pair NPZ files
|-- DelayScans/                # initial delay-scan PNG files
|-- EPC_Sweeps/                # EPC sweep CSV/PNG outputs
|-- RecordedExposureAnalysis/  # offline exposure-analysis CSV/PNG outputs
|-- temperature_analysis/      # temperature logs, tables, and plots
|-- alice_results.csv          # main Phi+ and CHSH measurement log
|-- qber_iterlog.csv           # optimizer-evaluation log
`-- optimizer_state.json       # best optimizer states and voltages
```

`Data/alice_results.csv` includes sync marker counts, clock-skew summaries,
coincidence counts, accidental estimates, per-pair delays, singles counts,
Phi+ visibility/QBER, CHSH expectations, and CHSH S values.

## Useful Scripts

Run these from the repository root unless the constants inside the script point
elsewhere.

| Script | Purpose |
| --- | --- |
| `qkd_plot_from_files.py` | Live/static plot of visibility, QBER, CHSH S, expectations, and counts from `alice_results.csv` or `qber_iterlog.csv`. |
| `qkd_plot_singles_from_file.py` | Plot Alice/Bob singles from event-count columns in `alice_results.csv`. |
| `qkd_manual_epc_gui.py` | Tkinter GUI for manual Alice/Bob EPC voltage control plus one-shot or continuous synchronized measurements. Start `Bob.py` first. |
| `qkd_sweep_voltages_alice.py` | Sweep Alice and Bob EPC DAC voltages, measure synchronized visibility, and write CSV/PNG outputs under `Data/EPC_Sweeps/`. |
| `qkd_plot_epc_sweep_from_file.py` | Replot a saved EPC sweep CSV. Edit `CSV_FILE` near the top. |
| `qkd_analyze_recorded_pairs.py` | Reprocess an existing Alice/Bob BIN pair in exposure windows and save CSV/PNG outputs under `Data/RecordedExposureAnalysis/`. |
| `qkd_plot_chsh_convention.py` | Compare CHSH sign conventions and local output swaps from saved count rows. |
| `Temperature_logger.py` | Watch `Data/alice_results.csv` and log Open-Meteo temperatures for Ljubljana and Drnovo. Requires network access. |
| `Temperature_analysis.py` | Merge measurement metrics with the temperature log and save correlation tables/plots. |
| `alice_time_check.py` | Check Bob clock offset and network round-trip time; also runs a simple Alice time server for reciprocal checks. |
| `qkd_test_epc.py` | Verify the EPC command-line backend can see the configured device and apply two voltage patterns. |
| `collect_alice_bob.py` | Older paired acquisition helper that writes under `DataExtSync/`; useful for controlled tests outside the current `Alice.py` loop. |
| `send_data_zip.py` / `Data_Receiever.py` | Standalone checksum-protected file transfer scripts. `Data_Receiever.py` keeps its historical misspelling. |

## Temperature Workflow

`Temperature_logger.py` samples Open-Meteo temperature data when
`Data/alice_results.csv` changes. It writes:

```text
Data/temperature_analysis/temperature_log.csv
```

When the temperature log is empty and Alice results already exist, the logger
can backfill hourly Open-Meteo archive data over the measurement time span.

`Temperature_analysis.py` then:

- merges temperatures with measurement rows;
- calculates Pearson correlations;
- scans lagged cross-correlations;
- saves tables and plots under `Data/temperature_analysis/`.

This workflow is optional and independent of the hardware acquisition loop.

## Legacy And Standalone Transfer Scripts

The current acquisition path is `Alice.py` plus `Bob.py` through
`qkd_acquisition.py` and `qkd_network.py`.

`collect_alice_bob.py`, `send_data_zip.py`, and `Data_Receiever.py` are still
present for older or standalone transfer workflows. Their hostnames, ports, and
paths are hard-coded near the tops of the files and should be checked before
use.

## Practical Checklist For New Measurements

1. Confirm Bob's record directory, EPC serial, and command port in `Bob.py`.
2. Start `Bob.py` and verify Alice can reach `BOB_CONFIG.port`.
3. Confirm Alice's `ACQUISITION.bob_host`, coincidence pairs, sync channel, and
   coincidence window in `Alice.py`.
4. Run `alice_time_check.py` if scheduled starts are failing to overlap.
5. Use `QBER_OPTIMIZATION_ENABLED = False` for a passive baseline.
6. Inspect `sync_common_markers`, `sync_skew_ppm_mean`, pair delays, accidentals,
   visibility/QBER, and CHSH S before enabling optimization.
7. Enable optimizer only after the passive measurement path is stable.
