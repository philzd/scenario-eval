"""
Scenario extraction from operational telemetry.

Converts continuous telemetry streams into discrete scenario objects
using heuristic trigger conditions such as low TTC, high jerk, and
hard braking. Extracted scenarios serve as the foundation for
evaluation, prioritization, human review, and dataset curation
workflows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "timestamp",
    "speed",
    "acceleration",
    "jerk",
    "distance_to_object",
    "relative_velocity",
)

TRIGGER_LOW_TTC = "low_ttc"
TRIGGER_HIGH_JERK = "high_jerk"
TRIGGER_HARD_BRAKING = "hard_braking"


@dataclass(frozen=True)
class ExtractionConfig:
    """Settings for reading telemetry and extracting scenarios."""

    input_path: Path
    output_path: Path
    source_log: str
    pre_window_s: float = 5.0
    post_window_s: float = 5.0
    ttc_threshold_s: float = 2.0
    jerk_threshold: float = 5.0
    hard_brake_threshold: float = -4.0


@dataclass(frozen=True)
class TriggerEvent:
    """A single rising-edge trigger on the telemetry timeline."""

    event_idx: int
    event_time: pd.Timestamp
    trigger_type: str


@dataclass
class ScenarioWindow:
    """A time-bounded scenario interval, possibly merged with others."""

    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    event_idx: int
    event_time: pd.Timestamp
    trigger_types: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class ScenarioRecord:
    """A finalized scenario ready for JSONL export."""

    scenario_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    event_time: pd.Timestamp
    start_idx: int
    end_idx: int
    event_idx: int
    trigger_types: list[str]
    source_log: str


def load_telemetry(path: Path) -> pd.DataFrame:
    """
    Load telemetry from a CSV file and prepare timestamps for processing.
    """
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Telemetry is missing required columns: {missing}")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_ttc(df: pd.DataFrame) -> pd.Series:
    """
    Compute time-to-collision for each telemetry row.
    """
    closing_speed = -df["relative_velocity"]
    ttc = pd.Series(np.inf, index=df.index, dtype=float)
    closing = closing_speed > 0
    ttc.loc[closing] = df.loc[closing, "distance_to_object"] / closing_speed.loc[closing]
    return ttc


def enrich_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived signals used for trigger detection.
    """
    enriched = df.copy()
    enriched["closing_speed"] = -enriched["relative_velocity"]
    enriched["ttc"] = compute_ttc(enriched)
    return enriched


def _rising_edge(mask: pd.Series) -> pd.Series:
    """
    Return a boolean mask that is true only where a condition newly becomes true.
    """
    active = mask.fillna(False).astype(bool)
    previous = active.shift(1, fill_value=False)
    return active & ~previous


def detect_triggers(df: pd.DataFrame, config: ExtractionConfig) -> list[TriggerEvent]:
    """
    Find rising-edge trigger events for low TTC, high jerk, and hard braking.
    """
    low_ttc = _rising_edge(df["ttc"] < config.ttc_threshold_s)
    high_jerk = _rising_edge(df["jerk"].abs() > config.jerk_threshold)
    hard_braking = _rising_edge(df["acceleration"] < config.hard_brake_threshold)

    events: list[TriggerEvent] = []
    for trigger_type, mask in (
        (TRIGGER_LOW_TTC, low_ttc),
        (TRIGGER_HIGH_JERK, high_jerk),
        (TRIGGER_HARD_BRAKING, hard_braking),
    ):
        for event_idx in mask[mask].index:
            events.append(
                TriggerEvent(
                    event_idx=int(event_idx),
                    event_time=df.at[event_idx, "timestamp"],
                    trigger_type=trigger_type,
                )
            )

    events.sort(key=lambda event: (event.event_time, event.event_idx, event.trigger_type))
    return events


def _time_to_index(
    timestamps: pd.Series,
    target_time: pd.Timestamp,
    side: str,
) -> int:
    """
    Map a target timestamp to the nearest row index in the telemetry series.
    """
    positions = timestamps.searchsorted(target_time, side=side)
    if side == "left":
        return int(min(positions, len(timestamps) - 1))
    return int(min(max(positions - 1, 0), len(timestamps) - 1))


def build_window_for_event(
    df: pd.DataFrame,
    event: TriggerEvent,
    config: ExtractionConfig,
) -> ScenarioWindow:
    """
    Build a scenario window centered on a trigger event using time-based bounds.
    """
    timestamps = df["timestamp"]
    window_start = event.event_time - pd.Timedelta(seconds=config.pre_window_s)
    window_end = event.event_time + pd.Timedelta(seconds=config.post_window_s)

    # Clip to the available telemetry range.
    window_start = max(window_start, timestamps.iloc[0])
    window_end = min(window_end, timestamps.iloc[-1])

    start_idx = _time_to_index(timestamps, window_start, side="left")
    end_idx = _time_to_index(timestamps, window_end, side="right")

    return ScenarioWindow(
        start_idx=start_idx,
        end_idx=end_idx,
        start_time=timestamps.iloc[start_idx],
        end_time=timestamps.iloc[end_idx],
        event_idx=event.event_idx,
        event_time=event.event_time,
        trigger_types={event.trigger_type},
    )


def build_scenario_windows(
    df: pd.DataFrame,
    events: list[TriggerEvent],
    config: ExtractionConfig,
) -> list[ScenarioWindow]:
    """
    Create a scenario window for each trigger event.
    """
    return [build_window_for_event(df, event, config) for event in events]


def merge_overlapping_windows(windows: list[ScenarioWindow]) -> list[ScenarioWindow]:
    """
    Merge scenario windows that overlap in time, preserving all trigger types.
    """
    if not windows:
        return []

    sorted_windows = sorted(
        windows,
        key=lambda window: (window.start_time, window.event_time),
    )

    merged: list[ScenarioWindow] = []
    current = sorted_windows[0]

    for nxt in sorted_windows[1:]:
        # Inclusive overlap: touching or overlapping intervals are merged.
        if nxt.start_time <= current.end_time:
            current.start_time = min(current.start_time, nxt.start_time)
            current.end_time = max(current.end_time, nxt.end_time)
            current.start_idx = min(current.start_idx, nxt.start_idx)
            current.end_idx = max(current.end_idx, nxt.end_idx)
            current.trigger_types |= nxt.trigger_types

            # Keep the earliest trigger as the anchor for the merged scenario.
            if nxt.event_time < current.event_time:
                current.event_time = nxt.event_time
                current.event_idx = nxt.event_idx
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def assign_scenario_ids(
    windows: list[ScenarioWindow],
    source_log: str,
) -> list[ScenarioRecord]:
    """
    Assign sequential scenario IDs and convert windows to export records.
    """
    stem = Path(source_log).stem
    sorted_windows = sorted(windows, key=lambda window: window.start_time)

    records: list[ScenarioRecord] = []
    for index, window in enumerate(sorted_windows, start=1):
        scenario_id = f"{stem}_{index:04d}"
        records.append(
            ScenarioRecord(
                scenario_id=scenario_id,
                start_time=window.start_time,
                end_time=window.end_time,
                event_time=window.event_time,
                start_idx=window.start_idx,
                end_idx=window.end_idx,
                event_idx=window.event_idx,
                trigger_types=sorted(window.trigger_types),
                source_log=source_log,
            )
        )
    return records


def scenario_to_dict(record: ScenarioRecord) -> dict[str, object]:
    """
    Convert a scenario record to a JSON-serializable dictionary.
    """
    return {
        "scenario_id": record.scenario_id,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "event_time": record.event_time.isoformat(),
        "start_idx": record.start_idx,
        "end_idx": record.end_idx,
        "event_idx": record.event_idx,
        "trigger_types": record.trigger_types,
        "source_log": record.source_log,
    }


def write_scenarios_jsonl(records: list[ScenarioRecord], path: Path) -> None:
    """
    Write scenario records to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(scenario_to_dict(record)))
            handle.write("\n")


def extract_scenarios(config: ExtractionConfig) -> list[ScenarioRecord]:
    """
    Run the full scenario extraction pipeline.
    """
    df = load_telemetry(config.input_path)
    df = enrich_telemetry(df)

    events = detect_triggers(df, config)
    windows = build_scenario_windows(df, events, config)
    merged_windows = merge_overlapping_windows(windows)
    return assign_scenario_ids(merged_windows, config.source_log)


def parse_args() -> ExtractionConfig:
    """
    Parse command-line arguments for scenario extraction.
    """
    parser = argparse.ArgumentParser(
        description="Extract discrete driving scenarios from telemetry CSV files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/synthetic_telemetry.csv"),
        help="Input telemetry CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scenarios.jsonl"),
        help="Output JSONL path for extracted scenarios.",
    )
    parser.add_argument(
        "--source-log",
        type=str,
        default=None,
        help="Source log label stored in each scenario (defaults to input basename).",
    )
    parser.add_argument(
        "--pre-window-s",
        type=float,
        default=5.0,
        help="Seconds of telemetry to include before each trigger.",
    )
    parser.add_argument(
        "--post-window-s",
        type=float,
        default=5.0,
        help="Seconds of telemetry to include after each trigger.",
    )
    parser.add_argument(
        "--ttc-threshold-s",
        type=float,
        default=2.0,
        help="Low TTC trigger threshold in seconds.",
    )
    parser.add_argument(
        "--jerk-threshold",
        type=float,
        default=5.0,
        help="High jerk trigger threshold (absolute value).",
    )
    parser.add_argument(
        "--hard-brake-threshold",
        type=float,
        default=-4.0,
        help="Hard braking trigger threshold for acceleration.",
    )
    args = parser.parse_args()
    source_log = args.source_log if args.source_log is not None else args.input.name
    return ExtractionConfig(
        input_path=args.input,
        output_path=args.output,
        source_log=source_log,
        pre_window_s=args.pre_window_s,
        post_window_s=args.post_window_s,
        ttc_threshold_s=args.ttc_threshold_s,
        jerk_threshold=args.jerk_threshold,
        hard_brake_threshold=args.hard_brake_threshold,
    )


def main() -> None:
    """
    Extract scenarios from telemetry and write them to JSONL.
    """
    config = parse_args()
    records = extract_scenarios(config)
    write_scenarios_jsonl(records, config.output_path)
    print(f"Wrote {len(records):,} scenarios to {config.output_path}")


if __name__ == "__main__":
    main()
