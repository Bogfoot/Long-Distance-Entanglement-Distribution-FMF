from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

import coincfinder


CLOCK_PERIOD_PS = 1_000_000
START_THRESHOLD_PS = 900_000
DATA_PERIOD_PS = CLOCK_PERIOD_PS / 16
DATA_THRESHOLD_1_PS = DATA_PERIOD_PS * 1.25
DATA_THRESHOLD_2_PS = DATA_PERIOD_PS * 1.75
END_OF_DATA_THRESHOLD_PS = DATA_PERIOD_PS * 3
MIN_BLOCK_INTERVALS = 5
PS_PER_NS = 1_000.0
PS_PER_SECOND = 1_000_000_000_000
DEFAULT_SYNC_CHANNEL = 5

COARSE_WINDOW_PS = 1 * PS_PER_NS
COARSE_HALF_RANGE_PS = 50 * PS_PER_NS
COARSE_STEP_PS = COARSE_WINDOW_PS / 2
FINE_WINDOW_PS = 100.0
FINE_HALF_RANGE_PS = 50 * PS_PER_NS
FINE_STEP_PS = 50.0


@dataclass
class SyncBlock:
    time_ps: int
    counter: int
    intervals_ps: np.ndarray
    decoded_bit_count: int
    start_gap_ps: int
    end_gap_ps: int


@dataclass
class DecodeResult:
    path: Path
    duration_s: float
    event_count: int
    blocks: list[SyncBlock]


@dataclass(frozen=True)
class CoincidencePair:
    name: str
    alice_channel: int
    bob_channel: int


@dataclass
class DelayScanResult:
    delays_ps: np.ndarray
    counts: np.ndarray


@dataclass
class ClockMap:
    counters: np.ndarray
    alice_times_ps: np.ndarray
    bob_times_ps: np.ndarray
    segment_skew_ppm: np.ndarray

    @property
    def overlap_duration_s(self) -> float:
        if self.alice_times_ps.size < 2:
            return 0.0
        return float((self.alice_times_ps[-1] - self.alice_times_ps[0]) / PS_PER_SECOND)


@dataclass
class CoincidenceResult:
    pair: CoincidencePair
    best_delay_ps: float
    coincidences_ps: np.ndarray
    delay_scan: DelayScanResult | None = None
    alice_event_count: int = 0
    bob_event_count: int = 0
    accidental_estimate: float = 0.0

    @property
    def count(self) -> int:
        return int(self.coincidences_ps.shape[0])


@dataclass
class SyncCoincidenceAnalysis:
    alice_path: Path
    bob_path: Path
    alice_decode: DecodeResult
    bob_decode: DecodeResult
    clock_map: ClockMap
    pair_results: list[CoincidenceResult]
    coincidence_window_ps: float

    @property
    def results_by_name(self) -> dict[str, CoincidenceResult]:
        return {result.pair.name: result for result in self.pair_results}

    @property
    def overlap_duration_s(self) -> float:
        return self.clock_map.overlap_duration_s


def flatten_channel(
    singles_map: Mapping[int, object],
    channel: int,
    *,
    missing_ok: bool = False,
) -> np.ndarray:
    if channel not in singles_map:
        if missing_ok:
            return np.zeros(0, dtype=np.int64)
        raise ValueError(
            f"Channel {channel} is absent; available channels are {sorted(singles_map)}"
        )

    singles = singles_map[channel]
    buckets = [
        np.asarray(bucket, dtype=np.int64)
        for bucket in singles.events_per_second
        if len(bucket)
    ]
    if not buckets:
        return np.zeros(0, dtype=np.int64)

    timestamps = np.concatenate(buckets)
    if np.any(np.diff(timestamps) < 0):
        timestamps.sort()
    return np.ascontiguousarray(timestamps, dtype=np.int64)


def load_channel(path: str | Path, channel: int) -> tuple[np.ndarray, float]:
    path = Path(path)
    singles_map, duration_s = coincfinder.read_file_auto(str(path))
    timestamps = flatten_channel(singles_map, channel)
    if timestamps.size == 0:
        raise ValueError(f"{path}: channel {channel} contains no timestamps")
    return timestamps, float(duration_s)


def decode_manchester(intervals_ps: np.ndarray) -> tuple[int, list[int]]:
    """Decode one compact sync block; bits arrive least-significant bit first."""
    bits = [1]
    for interval_ps in intervals_ps[1:]:
        previous_bit = bits[-1]
        if interval_ps < DATA_THRESHOLD_1_PS:
            bits.append(previous_bit)
        elif interval_ps < DATA_THRESHOLD_2_PS:
            bits.extend([0, 0] if previous_bit else [1])
        else:
            bits.extend([0, 1])

    counter = sum(bit << index for index, bit in enumerate(bits[1:]))
    return int(counter), bits


def find_sync_blocks(timestamps_ps: np.ndarray) -> list[SyncBlock]:
    intervals = np.diff(timestamps_ps)
    blocks: list[SyncBlock] = []
    i = 0

    while i < intervals.size:
        start_gap = int(intervals[i])
        is_candidate_start = END_OF_DATA_THRESHOLD_PS < start_gap < START_THRESHOLD_PS
        if not is_candidate_start:
            i += 1
            continue

        j = i + 1
        block_intervals: list[int] = []
        while j < intervals.size and intervals[j] < END_OF_DATA_THRESHOLD_PS:
            block_intervals.append(int(intervals[j]))
            j += 1

        if len(block_intervals) >= MIN_BLOCK_INTERVALS:
            block_array = np.asarray(block_intervals, dtype=np.int64)
            counter, bits = decode_manchester(block_array)
            end_gap = int(intervals[j]) if j < intervals.size else -1
            blocks.append(
                SyncBlock(
                    time_ps=int(timestamps_ps[i + 1]),
                    counter=counter,
                    intervals_ps=block_array,
                    decoded_bit_count=len(bits),
                    start_gap_ps=start_gap,
                    end_gap_ps=end_gap,
                )
            )

        i = max(j, i + 1)

    return blocks


def decode_file(path: str | Path, channel: int = DEFAULT_SYNC_CHANNEL) -> DecodeResult:
    path = Path(path)
    timestamps, duration_s = load_channel(path, channel)
    blocks = find_sync_blocks(timestamps)
    if not blocks:
        raise ValueError(
            f"{path}: no synchronization blocks found on channel {channel}"
        )
    return DecodeResult(path, duration_s, int(timestamps.size), blocks)


def counter_values(result: DecodeResult) -> np.ndarray:
    return np.asarray([block.counter for block in result.blocks], dtype=np.int64)


def block_times(result: DecodeResult) -> np.ndarray:
    return np.asarray([block.time_ps for block in result.blocks], dtype=np.int64)


def normalize_pairs(
    pairs: Iterable[CoincidencePair | tuple[str, int, int]],
) -> list[CoincidencePair]:
    normalized = [
        pair if isinstance(pair, CoincidencePair) else CoincidencePair(*pair)
        for pair in pairs
    ]
    if not normalized:
        raise ValueError("At least one coincidence pair is required")
    names = [pair.name for pair in normalized]
    if len(names) != len(set(names)):
        raise ValueError("Coincidence pair names must be unique")
    return normalized


def build_clock_map(alice: DecodeResult, bob: DecodeResult) -> ClockMap:
    alice_by_counter = {block.counter: block.time_ps for block in alice.blocks}
    bob_by_counter = {block.counter: block.time_ps for block in bob.blocks}
    counters = np.asarray(
        sorted(alice_by_counter.keys() & bob_by_counter.keys()), dtype=np.int64
    )
    if counters.size < 2:
        raise ValueError("Need at least two shared synchronization counters")

    alice_times = np.asarray(
        [alice_by_counter[int(counter)] for counter in counters], dtype=np.float64
    )
    bob_times = np.asarray(
        [bob_by_counter[int(counter)] for counter in counters], dtype=np.float64
    )
    if np.any(np.diff(alice_times) <= 0) or np.any(np.diff(bob_times) <= 0):
        raise ValueError(
            "Matched synchronization timestamps must be strictly increasing"
        )

    segment_scale = np.diff(alice_times) / np.diff(bob_times)
    return ClockMap(
        counters=counters,
        alice_times_ps=alice_times,
        bob_times_ps=bob_times,
        segment_skew_ppm=(segment_scale - 1.0) * 1.0e6,
    )


def trim_to_range(
    timestamps_ps: np.ndarray, start_ps: float, end_ps: float
) -> np.ndarray:
    lo = np.searchsorted(timestamps_ps, start_ps, side="left")
    hi = np.searchsorted(timestamps_ps, end_ps, side="right")
    return timestamps_ps[lo:hi]


def align_bob_to_alice(timestamps_ps: np.ndarray, clock_map: ClockMap) -> np.ndarray:
    trimmed = trim_to_range(
        timestamps_ps, clock_map.bob_times_ps[0], clock_map.bob_times_ps[-1]
    )
    if trimmed.size == 0:
        return np.zeros(0, dtype=np.int64)
    aligned = np.interp(
        trimmed.astype(np.float64),
        clock_map.bob_times_ps,
        clock_map.alice_times_ps,
    )
    return np.ascontiguousarray(np.rint(aligned), dtype=np.int64)


def _empty_delay_scan(
    start_ps: float, end_ps: float, step_ps: float
) -> tuple[np.ndarray, np.ndarray]:
    delays = np.arange(start_ps, end_ps + step_ps * 0.5, step_ps, dtype=np.float64)
    return delays, np.zeros(delays.size, dtype=np.int64)


def scan_delays(
    alice_ps: np.ndarray,
    bob_ps: np.ndarray,
    window_ps: float,
    start_ps: float,
    end_ps: float,
    step_ps: float,
) -> tuple[np.ndarray, np.ndarray]:
    if step_ps <= 0:
        raise ValueError("Delay scan step must be positive")
    if end_ps < start_ps:
        raise ValueError("Delay scan end must be greater than or equal to start")
    if alice_ps.size == 0 or bob_ps.size == 0:
        return _empty_delay_scan(start_ps, end_ps, step_ps)

    rows = coincfinder.compute_coincidences_for_range_np(
        alice_ps,
        bob_ps,
        float(window_ps),
        float(start_ps),
        float(end_ps),
        float(step_ps),
    )
    delays = np.asarray([row[0] for row in rows], dtype=np.float64) * PS_PER_NS
    counts = np.asarray([row[1] for row in rows], dtype=np.int64)
    return delays, counts


def find_best_delay(
    alice_ps: np.ndarray,
    bob_ps: np.ndarray,
    *,
    capture_scan: bool = False,
) -> tuple[float, DelayScanResult | None]:
    if alice_ps.size == 0 or bob_ps.size == 0:
        return 0.0, None
    coarse_delays, coarse_counts = scan_delays(
        alice_ps,
        bob_ps,
        COARSE_WINDOW_PS,
        -COARSE_HALF_RANGE_PS,
        COARSE_HALF_RANGE_PS,
        COARSE_STEP_PS,
    )
    coarse_best = float(coarse_delays[int(np.argmax(coarse_counts))])
    fine_delays, fine_counts = scan_delays(
        alice_ps,
        bob_ps,
        FINE_WINDOW_PS,
        coarse_best - FINE_HALF_RANGE_PS,
        coarse_best + FINE_HALF_RANGE_PS,
        FINE_STEP_PS,
    )
    best_delay_ps = float(fine_delays[int(np.argmax(fine_counts))])
    delay_scan = (
        DelayScanResult(fine_delays, fine_counts)
        if capture_scan
        else None
    )
    return best_delay_ps, delay_scan


def collect_coincidences(
    alice_ps: np.ndarray,
    bob_ps: np.ndarray,
    delay_ps: float,
    coincidence_window_ps: float,
) -> np.ndarray:
    if alice_ps.size == 0 or bob_ps.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    matched = coincfinder.collect_coincidences_with_delay_ps(
        alice_ps.tolist(),
        bob_ps.tolist(),
        float(coincidence_window_ps),
        float(delay_ps),
    )
    if not matched:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(matched, dtype=np.int64).reshape(-1, 2)


def estimate_accidentals(
    alice_event_count: int,
    bob_event_count: int,
    coincidence_window_ps: float,
    duration_s: float,
) -> float:
    if duration_s <= 0:
        return 0.0
    coincidence_window_s = coincidence_window_ps / PS_PER_SECOND
    return float(
        2.0 * alice_event_count * bob_event_count * coincidence_window_s / duration_s
    )


def analyze_sync_coincidences(
    alice_path: str | Path,
    bob_path: str | Path,
    coincidence_pairs: Iterable[CoincidencePair | tuple[str, int, int]],
    *,
    sync_channel: int = DEFAULT_SYNC_CHANNEL,
    coincidence_window_ps: float = FINE_WINDOW_PS,
    capture_delay_scans: bool = False,
) -> SyncCoincidenceAnalysis:
    """Decode sync markers, map Bob timetags onto Alice's clock, and count pairs."""
    pairs = normalize_pairs(coincidence_pairs)
    alice_path = Path(alice_path)
    bob_path = Path(bob_path)
    alice_decode = decode_file(alice_path, sync_channel)
    bob_decode = decode_file(bob_path, sync_channel)
    clock_map = build_clock_map(alice_decode, bob_decode)

    alice_singles, _ = coincfinder.read_file_auto(str(alice_path))
    bob_singles, _ = coincfinder.read_file_auto(str(bob_path))

    alice_channels = {
        channel: np.ascontiguousarray(
            trim_to_range(
                flatten_channel(alice_singles, channel),
                clock_map.alice_times_ps[0],
                clock_map.alice_times_ps[-1],
            ),
            dtype=np.int64,
        )
        for channel in {pair.alice_channel for pair in pairs}
    }
    bob_channels = {
        channel: align_bob_to_alice(flatten_channel(bob_singles, channel), clock_map)
        for channel in {pair.bob_channel for pair in pairs}
    }

    pair_results: list[CoincidenceResult] = []
    for pair in pairs:
        alice_ps = alice_channels[pair.alice_channel]
        bob_ps = bob_channels[pair.bob_channel]
        best_delay_ps, delay_scan = find_best_delay(
            alice_ps,
            bob_ps,
            capture_scan=capture_delay_scans,
        )
        coincidences_ps = collect_coincidences(
            alice_ps,
            bob_ps,
            best_delay_ps,
            coincidence_window_ps,
        )
        accidental_estimate = estimate_accidentals(
            int(alice_ps.size),
            int(bob_ps.size),
            coincidence_window_ps,
            clock_map.overlap_duration_s,
        )
        pair_results.append(
            CoincidenceResult(
                pair=pair,
                best_delay_ps=best_delay_ps,
                coincidences_ps=coincidences_ps,
                delay_scan=delay_scan,
                alice_event_count=int(alice_ps.size),
                bob_event_count=int(bob_ps.size),
                accidental_estimate=accidental_estimate,
            )
        )

    return SyncCoincidenceAnalysis(
        alice_path=alice_path,
        bob_path=bob_path,
        alice_decode=alice_decode,
        bob_decode=bob_decode,
        clock_map=clock_map,
        pair_results=pair_results,
        coincidence_window_ps=float(coincidence_window_ps),
    )



def save_coincidence_timetag_pairs(
    analysis: SyncCoincidenceAnalysis,
    output_dir: str | Path,
    *,
    prefix: str | None = None,
) -> dict[str, Path]:
    """Write matched Alice/aligned-Bob timetag pairs for each coincidence label."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stem = prefix or f"{analysis.alice_path.stem}__{analysis.bob_path.stem}"
    saved: dict[str, Path] = {}

    for result in analysis.pair_results:
        path = output_path / f"{stem}__{result.pair.name}.npz"
        np.savez_compressed(
            path,
            coincidences_ps=result.coincidences_ps,
            best_delay_ps=float(result.best_delay_ps),
            alice_channel=int(result.pair.alice_channel),
            bob_channel=int(result.pair.bob_channel),
            coincidence_window_ps=float(analysis.coincidence_window_ps),
        )
        saved[result.pair.name] = path

    return saved
