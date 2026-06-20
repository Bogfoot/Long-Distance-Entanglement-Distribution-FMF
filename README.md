# LongDistanceQKD Alice/Bob Workflow

This repository is currently split into two runtime roles:

- Alice: receiver/controller PC at `100.97.8.91`
- Bob: sender/remote PC at `100.104.228.90`

Alice controls the experiment loop. Bob runs a small command server that Alice connects to.

## Files For Alice

Copy these files to the Alice PC:

- `Alice.py`
- `qkd_network.py`
- `qkd_epc.py`
- `qkd_names.py`
- `EPC/EPC.py`
- `QuTAG_MC.py`
- the required QuTAG DLL folders/files
- the required `coincfinder` installation/module

Alice also needs access to the EPC driver dependencies, including `MCP2210CLI.exe`, if Alice is controlling a local EPC.

Alice runs:

```bash
python Alice.py
```

Alice does the following:

1. Initializes Alice's local EPC.
2. Initializes Alice's local QuTAG.
3. Generates a shared `record_id` using the current UTC time.
4. Chooses a shared future `start_time_utc`.
5. Sends Bob a `RECORD` command containing the `record_id` and `start_time_utc`.
6. Waits until `start_time_utc`.
7. Records Alice's local BIN file using the same `record_id`.
8. Receives Bob's BIN file.
9. Processes Bob's received file with `coincfinder`.
10. Writes result rows to `Data/alice_results.csv`.

Alice saves local raw files under:

```text
Data/AliceRaw/
```

Alice saves Bob's received files under:

```text
Data/Incoming/
```

Example paired filenames:

```text
alice_20260413T143812.123456Z_exp_10.0s.bin
bob_20260413T143812.123456Z_exp_10.0s.bin
```

The shared timestamp-like identifier is Windows-safe and is sanitized before use in filenames.

## Files For Bob

Copy these files to the Bob PC:

- `Bob.py`
- `qkd_network.py`
- `qkd_epc.py`
- `qkd_names.py`
- `EPC/EPC.py`
- `QuTAG_MC.py`
- the required QuTAG DLL folders/files

Bob also needs access to the EPC driver dependencies, including `MCP2210CLI.exe`, if Bob is controlling a local EPC.

Bob runs:

```bash
python Bob.py
```

Bob does the following:

1. Initializes Bob's local EPC.
2. Initializes Bob's local QuTAG.
3. Listens for commands from Alice on TCP port `5001`.
4. Records a BIN file when Alice sends `RECORD`.
5. Sends the recorded BIN file back to Alice.
6. Applies voltage or temperature changes when Alice sends EPC commands.

Bob saves local raw files under:

```text
C:\Users\RKAdmin\Desktop\LongDistanceQKD\BobData
```

Change `RECORD_DIR` in `Bob.py` if that path is not correct on Bob's PC.

## Network Settings

Bob listens on all interfaces:

```python
HOST = "0.0.0.0"
PORT = 5001
```

Alice connects to Bob at:

```python
BOB_HOST = "100.104.228.90"
BOB_PORT = 5001
```

The receiver/Alice PC IP is:

```text
100.97.8.91
```

That does not need to be hardcoded for the current workflow because Alice initiates the connection to Bob.

## EPC Control

Each side controls only its own local EPC:

- Alice controls Alice's EPC directly.
- Bob controls Bob's EPC directly.
- Alice sends Bob voltage commands over the network.

Both EPC references are currently set to index `0`:

```python
ALICE_EPC_DEVICE_REF = 0
BOB_EPC_DEVICE_REF = 0
```

The correction algorithm should use an 8-value voltage vector:

```python
[
    Alice_DAC0, Alice_DAC1, Alice_DAC2, Alice_DAC3,
    Bob_DAC0,   Bob_DAC1,   Bob_DAC2,   Bob_DAC3,
]
```

Alice applies this split with:

```python
set_correction_voltages(alice_epc, values)
```

This applies the first four values locally on Alice and sends the last four values to Bob.

## Bob Commands

Alice can send Bob these commands:

```python
{"command": "RECORD", "seconds": 10.0, "record_id": "...", "alice_time_utc": "...", "start_time_utc": "..."}
{"command": "SET_VOLTAGES", "voltages": [65, 65, 65, 65]}
{"command": "ZERO_VOLTAGES"}
{"command": "SET_TEMPERATURE", "temperature": 50}
{"command": "PING"}
{"command": "STOP"}
```

## Current Synchronization Status

Alice now records too, using a scheduled software trigger.

For each loop, Alice:

1. generates a shared `record_id`,
2. chooses `start_time_utc = now + SCHEDULE_AHEAD_SECONDS`,
3. sends Bob a `RECORD` command with `record_id` and `start_time_utc`,
4. Alice and Bob both wait until `start_time_utc`,
5. Alice and Bob each start local recording,
6. Alice receives Bob's file afterward.

Both files share the same `record_id`, so they can be matched later. The scheduling margin is configured in `Alice.py`:

```python
SCHEDULE_AHEAD_SECONDS = 3.0
```

This is still not precise hardware synchronization. It depends on the Alice and Bob PC clocks being synchronized well enough by the operating system. The next step is to use the paired Alice/Bob files to implement the actual synchronization and coincidence processing between the two independent timestamp streams.
