# \# Long-Distance QKD Timetag Synchronization

# 

# This repository processes timetag files recorded by two independent quTAG devices. The important problem is not starting both recordings at exactly the same instant. It is:

# 

# 1\. Pair the Alice and Bob files belonging to the same acquisition.

# 2\. Read the independent timetag streams.

# 3\. Decode synchronization markers recorded by both taggers.

# 4\. Use matching marker counters to estimate relative clock offset and skew.

# 5\. Correct Bob's photon timetags into Alice's time base.

# 6\. Find and evaluate coincidences between selected Alice/Bob channels.

# 

# The synchronization signal is required because Alice and Bob have independent timestamp origins and clocks, and most importantly not enough signal.

# A shared filename or scheduled software start is useful for pairing recordings, but is not precise enough for photon coincidences.

# 

# \## Intended Acquisition Layout

# 

# The final acquisition runs on separate PCs:

# 

# \- Alice controls the experiment and records Alice's local tags.

# \- Bob runs a command server and records Bob's local tags when requested.

# \- Alice chooses a shared `record\_id` and a future software start time.

# \- Both PCs record locally using the same `record\_id`.

# \- Bob's completed BIN file is transferred to Alice for offline processing.

# 

# Typical paired files are:

# 

# ```text

# Data/AliceRaw/alice\_20260413T143812.123456Z\_exp\_10.0s.bin

# Data/Incoming/bob\_20260413T143812.123456Z\_exp\_10.0s.bin

# ```

# 

# The matching `record\_id` identifies the pair. The scheduled start only ensures that their acquisition windows overlap. Do not assume that time zero in the two files is equal.

# 

# The earlier acquisition/network design, including the Alice/Bob command flow, file transfer, and PC clock diagnostics, is described in `README\_old.md`.

# `qkd\_network.py`, `qkd\_names.py`, and `receive\_data\_zip.py` contain the shared network transfer and naming utilities currently present in this directory.

# 

# `record\_dual\_qutag.py` records two locally connected quTAG devices and is useful for development and controlled tests. It is not the intended final separate-PC acquisition method.

# 

# \## Computer Folder Structure

# 

# Alice and Bob do not need identical copies of this repository. Alice performs acquisition control, synchronization, coincidence analysis, logging, and optional EPC optimization. Bob only records its local timetags, controls its local EPC, and responds to Alice over the network.

# 

# \### Alice computer

# 

# A minimal Alice installation should look like:

# 

# ```text

# LongDistanceQKD/

# ├── Alice.py

# ├── qkd\_acquisition.py

# ├── qkd\_sync.py

# ├── qkd\_epc\_correction.py

# ├── qkd\_plot\_delay\_scans.py

# ├── qkd\_epc.py

# ├── qkd\_network.py

# ├── qkd\_names.py

# ├── QuTAG\_MC.py

# ├── coincfinder.cpython-312-x86\_64-linux-gnu.so

# ├── EPC/                         # or AEPC/, containing EPC.py and driver files

# └── Data/

# &#x20;   ├── AliceRaw/                # Alice BIN files, created automatically

# &#x20;   ├── Incoming/                # Bob BIN files received by Alice

# &#x20;   ├── CoincidenceTimetags/     # optional matched-pair NPZ files

# &#x20;   ├── DelayScans/               # initial delay-scan figure

# &#x20;   ├── alice\_results.csv        # synchronization, coincidence, visibility, QBER

# &#x20;   ├── optimizer\_state.json     # best EPC voltages and visibility

# &#x20;   └── qber\_iterlog.csv         # optimizer evaluations

# ```

# 

# The files and directories under `Data/` are created as needed. `CoincidenceTimetags/` is only used when `SYNC\_PROCESSING.store\_coincidence\_timetags` is enabled.

# 

# Alice requires:

# 

# \- the quTAG Python wrapper and its vendor libraries;

# \- the Alice EPC driver directory, named either `EPC/` or `AEPC/`;

# \- NumPy;

# \- a compatible compiled `coincfinder` module;

# \- SciPy when `QBER\_OPTIMIZATION\_ENABLED = True`.

# 

# Configure the Bob address, port, local data directories, coincidence channels, synchronization settings, and optimizer settings near the top of `Alice.py`.

# 

# \### Bob computer

# 

# A minimal Bob installation should look like:

# 

# ```text

# LongDistanceQKD/

# ├── Bob.py

# ├── qkd\_epc.py

# ├── qkd\_network.py

# ├── qkd\_names.py

# ├── QuTAG\_MC.py

# ├── EPC/

# └── BobData/

# ```

# 

# Bob does not need `qkd\_sync.py`, `qkd\_epc\_correction.py`, `qkd\_acquisition.py`, `coincfinder`, NumPy, or SciPy for the current server workflow.

# 

# Set `BOB\_CONFIG.record\_dir` in `Bob.py` to the real `BobData` directory. The current configuration uses an absolute Windows path, so it must match the Bob computer.

# Also confirm that `BOB\_CONFIG.port` in `Bob.py` matches `ACQUISITION.bob\_port` in `Alice.py`, and allow that TCP port through the Bob computer's firewall.

# 

# \## What Bob Does

# 

# Run `Bob.py` before starting `Alice.py`. At startup Bob:

# 

# 1\. initializes its local quTAG;

# 2\. initializes its local EPC and sets the configured starting temperature;

# 3\. creates a `BobRecorder` for local BIN acquisition;

# 4\. opens a `BobCommandServer` on `BOB\_CONFIG.host:BOB\_CONFIG.port`, currently `0.0.0.0:5001`;

# 5\. waits for one command connection at a time from Alice.

# 

# The server uses short polling timeouts while waiting for connections and commands. This prevents an idle socket from blocking forever and allows Ctrl+C to stop Bob promptly. A Ctrl+C during an active recording also stops timestamp writing in the recorder's cleanup path before the tagger is deinitialized.

# 

# Bob supports these commands:

# 

# | Command | Bob action |

# | --- | --- |

# | `PING` | Replies with `PONG` so Alice can check connectivity. |

# | `TIME\_CHECK` | Returns Bob's current UTC time. |

# | `RECORD` | Waits until Alice's scheduled UTC start, records a local BIN file, reports completion, and transfers the file to Alice with metadata and SHA-256 verification. |

# | `SET\_VOLTAGES` | Validates and applies four voltages to Bob's EPC. |

# | `ZERO\_VOLTAGES` | Sets all four Bob EPC voltages to zero. |

# | `SET\_TEMPERATURE` | Changes Bob's EPC temperature. |

# | `DELETE\_RECORDING` | Deletes one named `bob\_\*.bin` file from Bob's configured recording directory after Alice has processed it. |

# | `STOP` | Replies to Alice, leaves the server loop, stops timestamp writing, and deinitializes Bob's quTAG. |

# 

# For a `RECORD` command, Bob saves:

# 

# ```text

# BobData/bob\_<record\_id>\_exp\_<seconds>s.bin

# ```

# 

# Bob keeps this local file after sending it. Alice receives a copy under `Data/Incoming/`. The shared `record\_id` is checked by Alice before the pair is accepted. When Alice exits, `shutdown\_tagger()` sends Bob the `STOP` command and then deinitializes Alice's local tagger.

# 

# \## Module Responsibilities

# 

# The runtime flow is split into modules with one primary responsibility each:

# 

# \- `Alice.py`: experiment configuration and top-level orchestration;

# \- `qkd\_acquisition.py`: request Bob data, record Alice data, receive Bob's file, and validate the paired acquisition;

# \- `qkd\_sync.py`: decode synchronization markers, align clocks, and extract coincidences;

# \- `qkd\_epc\_correction.py`: calculate Phi+ visibility/QBER, log results, and run the EPC optimizer;

# \- `qkd\_epc.py`: low-level EPC initialization and voltage/temperature calls;

# \- `qkd\_network.py`: length-prefixed JSON and checksum-protected file transfer;

# \- `qkd\_names.py`: UTC timestamps, record IDs, and filename sanitization.

# 

# `qkd\_acquisition.acquire\_pair()` returns an `AcquisitionPair` containing the shared record ID and both completed file paths. Acquisition failures identify the failed stage and record ID in the exception message.

# 

# `qkd\_sync.py` handles only timestamp synchronization and coincidence extraction:

# 

# \- compact Manchester sync-marker decoding;

# \- sync-counter matching between Alice and Bob;

# \- Bob-to-Alice timestamp interpolation;

# \- residual photon-delay scans;

# \- coincidence collection and accidental estimates;

# \- optional storage of matched coincidence timetag pairs.

# 

# Use it directly when another script only needs synchronized coincidence data:

# 

# ```python

# from qkd\_sync import analyze\_sync\_coincidences, save\_coincidence\_timetag\_pairs

# 

# pairs = \[

# &#x20;   ("HH", 1, 1),  # name, Alice channel, Bob channel

# &#x20;   ("VV", 2, 2),

# ]

# sync = analyze\_sync\_coincidences(

# &#x20;   alice\_path,

# &#x20;   bob\_path,

# &#x20;   pairs,

# &#x20;   sync\_channel=5,

# &#x20;   coincidence\_window\_ps=200,

# )

# 

# for result in sync.pair\_results:

# &#x20;   print(result.pair.name, result.count, result.best\_delay\_ps)

# &#x20;   coincidence\_timetags = result.coincidences\_ps  # columns: Alice ps, aligned Bob ps

# 

# save\_coincidence\_timetag\_pairs(sync, "Data/CoincidenceTimetags")

# ```

# 

# `qkd\_epc\_correction.py` owns the Phi+ interpretation, result logging, optimizer state, and Nelder-Mead implementation. It consumes the already-synchronized coincidence result:

# 

# ```python

# from qkd\_epc\_correction import DEFAULT\_PHI\_PLUS\_PAIRS, analyze\_phi\_plus\_coincidences

# from qkd\_sync import analyze\_sync\_coincidences

# 

# sync = analyze\_sync\_coincidences(alice\_path, bob\_path, DEFAULT\_PHI\_PLUS\_PAIRS)

# correction = analyze\_phi\_plus\_coincidences(sync)

# 

# print(correction.visibility, correction.qber\_total, correction.total\_coincidences)

# \# PhiPlusOptimizer uses this result directly as the optimization measurement.

# ```

# 

# The default Phi+ channel map is defined in `qkd\_epc\_correction.py`:

# 

# ```python

# DEFAULT\_PHI\_PLUS\_PAIRS = \[

# &#x20;   ("HH", 1, 1),

# &#x20;   ("HV", 1, 2),

# &#x20;   ("VH", 2, 1),

# &#x20;   ("VV", 2, 2),

# &#x20;   ("DD", 3, 3),

# &#x20;   ("DA", 3, 4),

# &#x20;   ("AD", 4, 3),

# &#x20;   ("AA", 4, 4),

# ]

# ```

# 

# This mapping must match the acquisition hardware. If Bob's detector channel convention differs, update `QKD\_COINCIDENCE\_PAIRS` in `Alice.py` or pass an explicit pair list to `qkd\_sync.py`.

# 

# \## Alice End-To-End Flow

# 

# `Alice.py` now follows the acquisition and correction sequence explicitly:

# 

# 1\. Ask Bob for data with a `RECORD` command.

# 2\. Wait until the scheduled start time and record Alice's local BIN file.

# 3\. Receive Bob's completed BIN file.

# 4\. Sync Alice/Bob timetags with `qkd\_sync.analyze\_sync\_coincidences()`.

# 5\. Get per-label coincidence counts and matched timetag pairs from the synced data.

# 6\. Optionally store the matched coincidence timetag pairs by setting `SYNC\_PROCESSING.store\_coincidence\_timetags=True`.

# 7\. Give the synchronized coincidence result to `qkd\_epc\_correction.analyze\_phi\_plus\_coincidences()`.

# 8\. Log visibility, QBER, coincidence counts, delays, accidentals, and sync diagnostics.

# 9\. If optimization is enabled, the correction algorithm requests a fresh acquisition and repeats steps 1-8 for every objective evaluation.

# 

# The result rows are appended to `Data/alice\_results.csv`. They include sync-marker count, clock-skew summary, coincidence counts, accidental estimates, per-pair delays, HV/DA visibility, HV/DA QBER, total QBER, total contrast, total coincidences, and `optimization\_score`.

# 

# When `SYNC\_PROCESSING.save\_initial\_delay\_scan=True`, Alice captures fine-delay scan data only for the first measurement after startup and saves one combined figure under:

# 

# ```text

# Data/DelayScans/initial\_delay\_scans\_<record\_id>.png

# ```

# 

# Later passive or optimizer measurements still find their delays normally but do not retain scan arrays or create more figures.

# 

# `qkd\_epc\_correction.PhiPlusOptimizer` implements the optional correction loop based on the local `Stability\_Check\_and\_Record.py` approach:

# 

# \- set `QBER\_OPTIMIZATION\_ENABLED = True` to enable it;

# \- it uses Nelder-Mead over eight voltages, `Alice\[0:4] + Bob\[4:8]`;

# \- each objective evaluation applies voltages, requests new Alice/Bob data, syncs it, extracts coincidences, and computes the EPC correction metric;

# \- the best voltage vector is saved in `Data/optimizer\_state.json`;

# \- per-iteration optimizer rows, including all eight coincidence counts, are written to `Data/qber\_iterlog.csv`.

# 

# Leave `QBER\_OPTIMIZATION\_ENABLED = False` in `Alice.py` for passive acquisition and logging.

# 

# \### Optimizer raw-file cleanup

# 

# Optimizer measurements use `MeasurementPipeline.measure\_for\_optimizer()`. After synchronization, coincidence analysis, and CSV logging succeed, Alice:

# 

# 1\. sends Bob `DELETE\_RECORDING` for the transferred Bob filename;

# 2\. deletes Alice's local file from `Data/AliceRaw/`;

# 3\. deletes Alice's received Bob copy from `Data/Incoming/`.

# 

# Passive measurements still retain all raw files. If synchronization or correction fails, the files are retained for diagnosis. A deletion failure is printed as a warning and does not discard the completed optimizer result.

# 

# \## Diagnostic Scripts

# 

# \### `decode\_sync\_timestamps.py`

# 

# This is the first diagnostic for a new pair of files. It verifies that the sync channel can be read, markers can be detected, counters can be decoded, and matching counters show a sensible timing offset and drift.

# 

# It reads complete `tagger\_a`/`tagger\_b` pairs from `DataTTTestsNewMethod/DualRaw`, decodes channel `5`, writes `decoded\_sync.csv`, and saves `decoded\_sync.png`.

# 

# ```bash

# python3 decode\_sync\_timestamps.py

# ```

# 

# Pair selection is controlled by `DEFAULT\_TIMESTAMP` near the top of the file:

# 

# \- `DEFAULT\_TIMESTAMP = None` processes the latest complete pair.

# \- Setting it to a shared timestamp such as `"20260609T104513.682883Z"` processes that exact pair.

# 

# \### `align\_sync\_coincidences.py`

# 

# This is a thin command-line wrapper over `qkd\_sync.analyze\_sync\_coincidences()`. It selects the latest complete Alice/Bob pair from `DataExtSync/AliceRaw` and `DataExtSync/Incoming`. Edit `COINCIDENCE\_PAIRS` near the top of the file and run:

# 

# ```bash

# python3 align\_sync\_coincidences.py

# ```

# 

# It prints shared sync markers, overlap duration, best per-pair delays, coincidence counts, and accidental estimates.

# 

# \### `qkd\_plot\_measurements.py`

# 

# This standalone script plots measurement CSV files without importing or running Alice/Bob hardware code. It displays all eight coincidence pairs, total coincidences, and H/V plus D/A visibility. By default it follows `Data/alice\_results.csv` live and refreshes when the file changes:

# 

# ```bash

# python qkd\_plot\_measurements.py

# ```

# 

# Configure the constants near the top of `qkd\_plot\_measurements.py`:

# 

# ```python

# CSV\_FILE = BASE\_DIR / "Data" / "qber\_iterlog.csv"

# REFRESH\_INTERVAL\_SECONDS = 2.0

# HISTORY\_ROWS = 200       # Use 0 for all rows.

# LIVE\_UPDATE = True       # False gives one static plot.

# SAVE\_PATH = None         # Or BASE\_DIR / "out" / "qkd\_measurements.png"

# ```

# 

# Then run:

# 

# ```bash

# python qkd\_plot\_measurements.py

# ```

# 

# When `vis\_HV` and `vis\_DA` are not present, as in the optimizer log, the script calculates them from the eight coincidence columns.

# 

# \## Synchronization Signal Assumptions

# 

# The compact decoder assumes timestamps are measured in picoseconds and uses this Manchester-style timing:

# 

# | Parameter | Value |

# | --- | ---: |

# | Regular clock period | `1,000,000 ps` (`1 us`) |

# | Sync-block start threshold | `900,000 ps` |

# | Data period | `62,500 ps` (`1 us / 16`) |

# | First decode threshold | `78,125 ps` |

# | Second decode threshold | `109,375 ps` |

# | End-of-data threshold | `187,500 ps` |

# 

# The transmitted block begins with a known marker bit of `1`. Remaining bits are decoded least-significant bit first into the synchronization counter. Update `qkd\_sync.py` if the synchronization waveform changes.

# 

# \## Recommended Order For A New Dataset

# 

# 1\. Pair Alice and Bob files using the shared `record\_id`.

# 2\. Confirm the expected sync and photon channels exist in both files.

# 3\. Run `decode\_sync\_timestamps.py` to validate marker decoding.

# 4\. Configure `QKD\_COINCIDENCE\_PAIRS`, `ACQUISITION`, `SYNC\_PROCESSING`, and `OPTIMIZER` near the top of `Alice.py`.

# 5\. Run `Alice.py` for acquisition plus synchronized analysis.

# 6\. Inspect marker count, clock-skew summary, coincidence peak delays, visibility, QBER, and accidentals before trusting optimization results.



