from __future__ import annotations

import csv
import datetime as dt
import os
import re
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from qkd_zerothird_analysis_common import BASE_DIR, GOOD_RESULTS_DIR, LOCAL_TZ


DELAY_SCAN_DIR = BASE_DIR / "Data" / "DelayScans"
OUTPUT_CSV = GOOD_RESULTS_DIR / "zerothird_run_metadata.csv"

SOURCE_SWITCH_UTC = dt.datetime(
    2026, 7, 9, 7, 50, 41, 661521, tzinfo=dt.timezone.utc
)
STABILITY_MONITOR_START_UTC = dt.datetime(
    2026, 7, 11, 21, 5, 23, tzinfo=dt.timezone.utc
)

TITLE_WINDOW_CANDIDATES_PS = (
    80,
    100,
    120,
    125,
    160,
    240,
    320,
    475,
    480,
    500,
    640,
    750,
    800,
)
WINDOW_SIGMA_LABELS = {
    120: "0.75",
    125: "0.75",
    160: "1",
    240: "1.5",
    320: "2",
    475: "3",
    480: "3",
    500: "3",
}

# Keep explicit manual corrections here if a title match is ambiguous after visual
# inspection. Keys are delay-scan PNG basenames; values are window_ps numbers.
TITLE_WINDOW_OVERRIDES_PS: dict[str, float] = {
    "initial_delay_scans_CHSH_S_20260709T075041.661521Z.png": 500.0,
}

DELAY_SCAN_TIMESTAMP_PATTERN = re.compile(
    r"^initial_delay_scans_(?P<record_id>.+?)_"
    r"(?P<timestamp>\d{8}T\d{6}(?:\.\d+)?Z)\.png$"
)


def parse_compact_utc_timestamp(timestamp_text: str) -> dt.datetime:
    for fmt in ("%Y%m%dT%H%M%S.%fZ", "%Y%m%dT%H%M%SZ"):
        try:
            return dt.datetime.strptime(timestamp_text, fmt).replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            pass
    raise ValueError(f"Could not parse UTC timestamp: {timestamp_text}")


def parse_delay_scan_path(path: Path) -> tuple[str, dt.datetime] | None:
    match = DELAY_SCAN_TIMESTAMP_PATTERN.match(path.name)
    if match is None:
        return None
    return (
        match.group("record_id"),
        parse_compact_utc_timestamp(match.group("timestamp")),
    )


def relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


@lru_cache(maxsize=None)
def rendered_title_crop(
    size: tuple[int, int],
    window_ps: float,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray:
    width, height = size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        figure = plt.figure(figsize=(width / 170, height / 170), dpi=170)
        figure.suptitle(
            f"Initial fine delay scans ({window_ps:g} ps coincidence window)",
            fontsize=14,
        )
        figure.tight_layout()
        temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp.close()
        figure.savefig(temp.name, dpi=170)
        plt.close(figure)
    try:
        return np.array(Image.open(temp.name).convert("L").crop(crop_box))
    finally:
        os.unlink(temp.name)


def title_match_scores(path: Path) -> list[tuple[float, float]]:
    image = Image.open(path).convert("L")
    width, height = image.size
    crop_box = (450, 25, width - 450, 75)
    actual = np.array(image.crop(crop_box))
    actual_mask = actual < 200
    scores: list[tuple[float, float]] = []
    for window_ps in TITLE_WINDOW_CANDIDATES_PS:
        rendered = rendered_title_crop((width, height), float(window_ps), crop_box)
        rendered_mask = rendered < 200
        union = np.logical_or(actual_mask, rendered_mask)
        xor = np.logical_xor(actual_mask, rendered_mask)
        score = float(xor.sum() / union.sum()) if union.any() else 1.0
        scores.append((score, float(window_ps)))
    return sorted(scores)


def classify_window_ps(path: Path) -> tuple[float, float, float, str]:
    scores = title_match_scores(path)
    best_score, best_window = scores[0]
    runner_score, runner_window = scores[1]
    if path.name in TITLE_WINDOW_OVERRIDES_PS:
        override = TITLE_WINDOW_OVERRIDES_PS[path.name]
        override_scores = [score for score, window in scores if window == override]
        override_score = override_scores[0] if override_scores else best_score
        return override, override_score, best_window, "manual_override"
    return best_window, best_score, runner_window, "title_match"


def sigma_label(window_ps: float) -> str:
    return WINDOW_SIGMA_LABELS.get(int(round(window_ps)), "")


def experiment_label(record_id: str) -> str:
    if record_id.startswith("CHSH_S"):
        return "CHSH"
    if record_id.startswith("QKD"):
        return "QKD+CHSH"
    return record_id


def mode_label(timestamp_utc: dt.datetime) -> str:
    if timestamp_utc >= STABILITY_MONITOR_START_UTC:
        return "stability_monitor"
    return "optimize"


def metadata_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(DELAY_SCAN_DIR.glob("initial_delay_scans_*.png")):
        parsed = parse_delay_scan_path(path)
        if parsed is None:
            continue
        record_id, timestamp_utc = parsed
        if timestamp_utc < SOURCE_SWITCH_UTC:
            continue
        window_ps, score, runner_up, source = classify_window_ps(path)
        timestamp_local = timestamp_utc.astimezone(LOCAL_TZ)
        rows.append(
            {
                "run_start_utc": timestamp_utc.isoformat(),
                "run_start_local": timestamp_local.isoformat(),
                "experiment": experiment_label(record_id),
                "window_ps": f"{window_ps:g}",
                "sigma": sigma_label(window_ps),
                "mode": mode_label(timestamp_utc),
                "notes": (
                    f"Reference delay scan {relative_path(path)}; "
                    f"window source={source}; match_score={score:.6f}; "
                    f"runner_up={runner_up:g} ps."
                ),
                "record_id": record_id,
                "reference_delay_scan": relative_path(path),
                "window_source": source,
                "title_match_score": f"{score:.6f}",
                "title_runner_up_window_ps": f"{runner_up:g}",
            }
        )
    return sorted(rows, key=lambda row: row["run_start_utc"])


def write_metadata(rows: list[dict[str, str]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_start_utc",
        "run_start_local",
        "experiment",
        "window_ps",
        "sigma",
        "mode",
        "notes",
        "record_id",
        "reference_delay_scan",
        "window_source",
        "title_match_score",
        "title_runner_up_window_ps",
    ]
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = metadata_rows()
    write_metadata(rows)
    print(f"Wrote {len(rows)} metadata rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
