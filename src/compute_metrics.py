"""
Scenario-level metric computation from telemetry and extracted scenarios.

Computes simple, interpretable metrics for each scenario window so that
scenarios can be scored, ranked, and reviewed in downstream workflows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.extract_scenarios import enrich_telemetry, load_telemetry


@dataclass(frozen=True)
class MetricsConfig:
    """Settings for reading inputs and writing scenario metrics."""

    telemetry_path: Path
    scenarios_path: Path
    output_path: Path


@dataclass(frozen=True)
class ScenarioInput:
    """One scenario window loaded from scenarios.jsonl."""

    scenario_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    start_idx: int
    end_idx: int
    trigger_types: list[str]
    source_log: str


@dataclass(frozen=True)
class ScenarioMetrics:
    """Computed metrics for one scenario, ready for JSONL export."""

    scenario_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_s: float
    trigger_types: list[str]
    min_ttc: float | None
    min_distance_to_object: float
    max_abs_jerk: float
    max_deceleration: float
    mean_speed: float
    source_log: str


def load_scenarios(path: Path) -> list[ScenarioInput]:
    """
    Load scenario windows from a JSONL file.
    """
    scenarios: list[ScenarioInput] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            try:
                scenarios.append(
                    ScenarioInput(
                        scenario_id=record["scenario_id"],
                        start_time=pd.Timestamp(record["start_time"]),
                        end_time=pd.Timestamp(record["end_time"]),
                        start_idx=int(record["start_idx"]),
                        end_idx=int(record["end_idx"]),
                        trigger_types=list(record["trigger_types"]),
                        source_log=record["source_log"],
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"Scenario record on line {line_number} is missing field: {error}"
                ) from error

    return scenarios


def slice_scenario_telemetry(df: pd.DataFrame, scenario: ScenarioInput) -> pd.DataFrame:
    """
    Return the inclusive telemetry slice for a scenario window.
    """
    if scenario.start_idx < 0 or scenario.end_idx >= len(df):
        raise ValueError(
            f"Scenario {scenario.scenario_id} has indices outside telemetry bounds: "
            f"[{scenario.start_idx}, {scenario.end_idx}] for {len(df)} rows."
        )
    if scenario.start_idx > scenario.end_idx:
        raise ValueError(
            f"Scenario {scenario.scenario_id} has invalid index range: "
            f"start_idx={scenario.start_idx}, end_idx={scenario.end_idx}."
        )

    return df.loc[scenario.start_idx : scenario.end_idx].copy()


def compute_duration_s(scenario: ScenarioInput) -> float:
    """
    Compute scenario duration in seconds from its time bounds.
    """
    return (scenario.end_time - scenario.start_time).total_seconds()


def compute_min_ttc(slice_df: pd.DataFrame) -> float | None:
    """
    Compute the minimum closing TTC in a scenario window.
    """
    closing = slice_df["closing_speed"] > 0
    if not closing.any():
        return None

    min_ttc = slice_df.loc[closing, "ttc"].min()
    if pd.isna(min_ttc):
        return None
    return float(min_ttc)


def compute_min_distance(slice_df: pd.DataFrame) -> float:
    """
    Compute the minimum distance to the lead object in a scenario window.
    """
    return float(slice_df["distance_to_object"].min())


def compute_max_abs_jerk(slice_df: pd.DataFrame) -> float:
    """
    Compute the maximum absolute jerk in a scenario window.
    """
    return float(slice_df["jerk"].abs().max())


def compute_max_deceleration(slice_df: pd.DataFrame) -> float:
    """
    Compute the strongest deceleration as the minimum acceleration value.
    """
    return float(slice_df["acceleration"].min())


def compute_mean_speed(slice_df: pd.DataFrame) -> float:
    """
    Compute the mean speed in a scenario window.
    """
    return float(slice_df["speed"].mean())


def compute_metrics_for_scenario(
    df: pd.DataFrame,
    scenario: ScenarioInput,
) -> ScenarioMetrics:
    """
    Compute all metrics for a single scenario window.
    """
    slice_df = slice_scenario_telemetry(df, scenario)

    return ScenarioMetrics(
        scenario_id=scenario.scenario_id,
        start_time=scenario.start_time,
        end_time=scenario.end_time,
        duration_s=compute_duration_s(scenario),
        trigger_types=scenario.trigger_types,
        min_ttc=compute_min_ttc(slice_df),
        min_distance_to_object=compute_min_distance(slice_df),
        max_abs_jerk=compute_max_abs_jerk(slice_df),
        max_deceleration=compute_max_deceleration(slice_df),
        mean_speed=compute_mean_speed(slice_df),
        source_log=scenario.source_log,
    )


def compute_all_scenario_metrics(
    df: pd.DataFrame,
    scenarios: list[ScenarioInput],
) -> list[ScenarioMetrics]:
    """
    Compute metrics for every scenario in input order.
    """
    return [compute_metrics_for_scenario(df, scenario) for scenario in scenarios]


def metrics_to_dict(record: ScenarioMetrics) -> dict[str, object]:
    """
    Convert a metrics record to a JSON-serializable dictionary.
    """
    return {
        "scenario_id": record.scenario_id,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "duration_s": record.duration_s,
        "trigger_types": record.trigger_types,
        "min_ttc": record.min_ttc,
        "min_distance_to_object": record.min_distance_to_object,
        "max_abs_jerk": record.max_abs_jerk,
        "max_deceleration": record.max_deceleration,
        "mean_speed": record.mean_speed,
        "source_log": record.source_log,
    }


def write_scenario_metrics_jsonl(records: list[ScenarioMetrics], path: Path) -> None:
    """
    Write scenario metrics records to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(metrics_to_dict(record)))
            handle.write("\n")


def run_metrics_pipeline(config: MetricsConfig) -> list[ScenarioMetrics]:
    """
    Run the full scenario metrics pipeline.
    """
    df = load_telemetry(config.telemetry_path)
    df = enrich_telemetry(df)
    scenarios = load_scenarios(config.scenarios_path)
    return compute_all_scenario_metrics(df, scenarios)


def parse_args() -> MetricsConfig:
    """
    Parse command-line arguments for scenario metric computation.
    """
    parser = argparse.ArgumentParser(
        description="Compute scenario-level metrics from telemetry and scenario windows.",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("data/raw/synthetic_telemetry.csv"),
        help="Input telemetry CSV path.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("outputs/scenarios.jsonl"),
        help="Input scenarios JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/scenario_metrics.jsonl"),
        help="Output JSONL path for scenario metrics.",
    )
    args = parser.parse_args()
    return MetricsConfig(
        telemetry_path=args.telemetry,
        scenarios_path=args.scenarios,
        output_path=args.output,
    )


def main() -> None:
    """
    Compute scenario metrics and write them to JSONL.
    """
    config = parse_args()
    records = run_metrics_pipeline(config)
    write_scenario_metrics_jsonl(records, config.output_path)
    print(f"Wrote {len(records):,} scenario metrics to {config.output_path}")


if __name__ == "__main__":
    main()
