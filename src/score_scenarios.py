"""
Scenario scoring and ranking from scenario-level metrics.

Computes interpretable severity, rarity, and value scores so scenarios
can be ranked for human review and downstream evaluation workflows.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Severity thresholds aligned with scenario extraction trigger rules.
TTC_THRESHOLD_S = 2.0
DECEL_THRESHOLD = -4.0
DECEL_SEVERITY_RANGE = 3.5
JERK_THRESHOLD = 5.0
JERK_SEVERITY_RANGE = 10.0

MAX_COMPONENT_SCORE = 10.0
MAX_RARITY_SCORE = 10.0


@dataclass(frozen=True)
class ScoringConfig:
    """Settings for reading metrics and writing score artifacts."""

    metrics_path: Path
    scores_output_path: Path
    ranked_output_path: Path


@dataclass(frozen=True)
class ScenarioMetricsInput:
    """One scenario metrics record loaded from scenario_metrics.jsonl."""

    scenario_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    trigger_types: list[str]
    min_ttc: float | None
    max_abs_jerk: float
    max_deceleration: float
    source_log: str


@dataclass(frozen=True)
class ScenarioScore:
    """Computed scores and rank for one scenario."""

    scenario_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    trigger_types: list[str]
    source_log: str
    min_ttc: float | None
    max_deceleration: float
    max_abs_jerk: float
    ttc_severity: float
    decel_severity: float
    jerk_severity: float
    severity_score: float
    rarity_score: float
    value_score: float
    rank: int


def load_scenario_metrics(path: Path) -> list[ScenarioMetricsInput]:
    """
    Load scenario metrics records from a JSONL file.
    """
    metrics: list[ScenarioMetricsInput] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            try:
                min_ttc = record["min_ttc"]
                metrics.append(
                    ScenarioMetricsInput(
                        scenario_id=record["scenario_id"],
                        start_time=pd.Timestamp(record["start_time"]),
                        end_time=pd.Timestamp(record["end_time"]),
                        trigger_types=list(record["trigger_types"]),
                        min_ttc=None if min_ttc is None else float(min_ttc),
                        max_abs_jerk=float(record["max_abs_jerk"]),
                        max_deceleration=float(record["max_deceleration"]),
                        source_log=record["source_log"],
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"Metrics record on line {line_number} is missing field: {error}"
                ) from error

    return metrics


def compute_ttc_severity(min_ttc: float | None) -> float:
    """
    Compute TTC severity on a 0-10 scale.
    """
    if min_ttc is None or min_ttc >= TTC_THRESHOLD_S:
        return 0.0

    raw_score = MAX_COMPONENT_SCORE * (TTC_THRESHOLD_S - min_ttc) / TTC_THRESHOLD_S
    return min(MAX_COMPONENT_SCORE, max(0.0, raw_score))


def compute_decel_severity(max_deceleration: float) -> float:
    """
    Compute deceleration severity on a 0-10 scale.
    """
    if max_deceleration >= DECEL_THRESHOLD:
        return 0.0

    raw_score = MAX_COMPONENT_SCORE * (-max_deceleration - 4.0) / DECEL_SEVERITY_RANGE
    return min(MAX_COMPONENT_SCORE, max(0.0, raw_score))


def compute_jerk_severity(max_abs_jerk: float) -> float:
    """
    Compute jerk severity on a 0-10 scale.
    """
    if max_abs_jerk <= JERK_THRESHOLD:
        return 0.0

    raw_score = MAX_COMPONENT_SCORE * (max_abs_jerk - JERK_THRESHOLD) / JERK_SEVERITY_RANGE
    return min(MAX_COMPONENT_SCORE, max(0.0, raw_score))


def compute_severity_components(
    metrics: ScenarioMetricsInput,
) -> tuple[float, float, float]:
    """
    Compute TTC, deceleration, and jerk severity components.
    """
    ttc_severity = compute_ttc_severity(metrics.min_ttc)
    decel_severity = compute_decel_severity(metrics.max_deceleration)
    jerk_severity = compute_jerk_severity(metrics.max_abs_jerk)
    return ttc_severity, decel_severity, jerk_severity


def compute_severity_score(
    ttc_severity: float,
    decel_severity: float,
    jerk_severity: float,
) -> float:
    """
    Compute overall severity as the sum of component severities.
    """
    return ttc_severity + decel_severity + jerk_severity


def compute_trigger_counts(metrics_records: list[ScenarioMetricsInput]) -> dict[str, int]:
    """
    Count how many scenarios contain each trigger type in the input batch.
    """
    counts: dict[str, int] = {}
    for metrics in metrics_records:
        for trigger_type in set(metrics.trigger_types):
            counts[trigger_type] = counts.get(trigger_type, 0) + 1
    return counts


def compute_type_rarity(trigger_type: str, total_scenarios: int, trigger_counts: dict[str, int]) -> float:
    """
    Compute IDF-style rarity for one trigger type within the input batch.
    """
    count = trigger_counts.get(trigger_type, 0)
    if count == 0:
        return 0.0
    return math.log(total_scenarios / count)


def compute_raw_rarity(
    metrics: ScenarioMetricsInput,
    total_scenarios: int,
    trigger_counts: dict[str, int],
) -> float:
    """
    Compute the raw rarity score for one scenario from its trigger types.
    """
    unique_types = set(metrics.trigger_types)
    return sum(
        compute_type_rarity(trigger_type, total_scenarios, trigger_counts)
        for trigger_type in unique_types
    )


def compute_rarity_scores(
    metrics_records: list[ScenarioMetricsInput],
    trigger_counts: dict[str, int],
) -> list[float]:
    """
    Compute normalized 0-10 rarity scores for every scenario in the batch.

    Rarity is derived from trigger frequency statistics within the
    input scenario collection.
    """
    total_scenarios = len(metrics_records)
    if total_scenarios == 0:
        return []

    raw_scores = [
        compute_raw_rarity(metrics, total_scenarios, trigger_counts)
        for metrics in metrics_records
    ]
    max_raw_score = max(raw_scores)
    if max_raw_score == 0.0:
        return [0.0 for _ in raw_scores]

    return [
        min(MAX_RARITY_SCORE, MAX_RARITY_SCORE * raw_score / max_raw_score)
        for raw_score in raw_scores
    ]


def score_scenario(
    metrics: ScenarioMetricsInput,
    rarity_score: float,
) -> ScenarioScore:
    """
    Compute all scores for a single scenario.
    """
    ttc_severity, decel_severity, jerk_severity = compute_severity_components(metrics)
    severity_score = compute_severity_score(ttc_severity, decel_severity, jerk_severity)
    value_score = severity_score + rarity_score

    return ScenarioScore(
        scenario_id=metrics.scenario_id,
        start_time=metrics.start_time,
        end_time=metrics.end_time,
        trigger_types=metrics.trigger_types,
        source_log=metrics.source_log,
        min_ttc=metrics.min_ttc,
        max_deceleration=metrics.max_deceleration,
        max_abs_jerk=metrics.max_abs_jerk,
        ttc_severity=ttc_severity,
        decel_severity=decel_severity,
        jerk_severity=jerk_severity,
        severity_score=severity_score,
        rarity_score=rarity_score,
        value_score=value_score,
        rank=0,
    )


def score_all_scenarios(metrics_records: list[ScenarioMetricsInput]) -> list[ScenarioScore]:
    """
    Compute scores for every scenario in input order.
    """
    trigger_counts = compute_trigger_counts(metrics_records)
    rarity_scores = compute_rarity_scores(metrics_records, trigger_counts)

    return [
        score_scenario(metrics, rarity_score)
        for metrics, rarity_score in zip(metrics_records, rarity_scores, strict=True)
    ]


def rank_scenarios(scores: list[ScenarioScore]) -> list[ScenarioScore]:
    """
    Assign deterministic scenario ranks based on scoring results.
    """
    sorted_scores = sorted(
        scores,
        key=lambda score: (
            -score.value_score,
            -score.severity_score,
            -score.rarity_score,
            score.scenario_id,
        ),
    )

    ranked: list[ScenarioScore] = []
    for rank, score in enumerate(sorted_scores, start=1):
        ranked.append(
            ScenarioScore(
                scenario_id=score.scenario_id,
                start_time=score.start_time,
                end_time=score.end_time,
                trigger_types=score.trigger_types,
                source_log=score.source_log,
                min_ttc=score.min_ttc,
                max_deceleration=score.max_deceleration,
                max_abs_jerk=score.max_abs_jerk,
                ttc_severity=score.ttc_severity,
                decel_severity=score.decel_severity,
                jerk_severity=score.jerk_severity,
                severity_score=score.severity_score,
                rarity_score=score.rarity_score,
                value_score=score.value_score,
                rank=rank,
            )
        )
    return ranked


def apply_ranks_to_input_order(
    scores: list[ScenarioScore],
    ranked_scores: list[ScenarioScore],
) -> list[ScenarioScore]:
    """
    Attach ranks to score records while preserving the original input order.
    """
    rank_by_id = {score.scenario_id: score.rank for score in ranked_scores}
    return [
        ScenarioScore(
            scenario_id=score.scenario_id,
            start_time=score.start_time,
            end_time=score.end_time,
            trigger_types=score.trigger_types,
            source_log=score.source_log,
            min_ttc=score.min_ttc,
            max_deceleration=score.max_deceleration,
            max_abs_jerk=score.max_abs_jerk,
            ttc_severity=score.ttc_severity,
            decel_severity=score.decel_severity,
            jerk_severity=score.jerk_severity,
            severity_score=score.severity_score,
            rarity_score=score.rarity_score,
            value_score=score.value_score,
            rank=rank_by_id[score.scenario_id],
        )
        for score in scores
    ]


def score_to_dict(record: ScenarioScore) -> dict[str, object]:
    """
    Convert a scenario score record to a JSON-serializable dictionary.
    """
    return {
        "scenario_id": record.scenario_id,
        "start_time": record.start_time.isoformat(),
        "end_time": record.end_time.isoformat(),
        "trigger_types": record.trigger_types,
        "source_log": record.source_log,
        "ttc_severity": record.ttc_severity,
        "decel_severity": record.decel_severity,
        "jerk_severity": record.jerk_severity,
        "severity_score": record.severity_score,
        "rarity_score": record.rarity_score,
        "value_score": record.value_score,
        "rank": record.rank,
    }


def write_scenario_scores_jsonl(records: list[ScenarioScore], path: Path) -> None:
    """
    Write scenario score records to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(score_to_dict(record)))
            handle.write("\n")


def write_ranked_scenarios_csv(ranked_records: list[ScenarioScore], path: Path) -> None:
    """
    Write a ranked scenario table for review in CSV format.
    """
    rows = [
        {
            "rank": record.rank,
            "scenario_id": record.scenario_id,
            "value_score": record.value_score,
            "severity_score": record.severity_score,
            "rarity_score": record.rarity_score,
            "ttc_severity": record.ttc_severity,
            "decel_severity": record.decel_severity,
            "jerk_severity": record.jerk_severity,
            "min_ttc": record.min_ttc,
            "max_deceleration": record.max_deceleration,
            "max_abs_jerk": record.max_abs_jerk,
            "trigger_types": ";".join(record.trigger_types),
            "start_time": record.start_time.isoformat(),
            "end_time": record.end_time.isoformat(),
            "source_log": record.source_log,
        }
        for record in ranked_records
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_scoring_pipeline(config: ScoringConfig) -> tuple[list[ScenarioScore], list[ScenarioScore]]:
    """
    Run the full scenario scoring and ranking pipeline.
    """
    metrics_records = load_scenario_metrics(config.metrics_path)
    scores = score_all_scenarios(metrics_records)
    ranked_scores = rank_scenarios(scores)
    scores_with_ranks = apply_ranks_to_input_order(scores, ranked_scores)
    return scores_with_ranks, ranked_scores


def parse_args() -> ScoringConfig:
    """
    Parse command-line arguments for scenario scoring.
    """
    parser = argparse.ArgumentParser(
        description="Score and rank scenarios from scenario-level metrics.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/scenario_metrics.jsonl"),
        help="Input scenario metrics JSONL path.",
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("outputs/scenario_scores.jsonl"),
        help="Output JSONL path for scenario scores.",
    )
    parser.add_argument(
        "--ranked-output",
        type=Path,
        default=Path("outputs/ranked_scenarios.csv"),
        help="Output CSV path for ranked scenarios.",
    )
    args = parser.parse_args()
    return ScoringConfig(
        metrics_path=args.metrics,
        scores_output_path=args.scores_output,
        ranked_output_path=args.ranked_output,
    )


def main() -> None:
    """
    Score scenarios and write score and ranking artifacts.
    """
    config = parse_args()
    scores, ranked_scores = run_scoring_pipeline(config)
    write_scenario_scores_jsonl(scores, config.scores_output_path)
    write_ranked_scenarios_csv(ranked_scores, config.ranked_output_path)
    print(f"Wrote {len(scores):,} scenario scores to {config.scores_output_path}")
    print(f"Wrote {len(ranked_scores):,} ranked scenarios to {config.ranked_output_path}")


if __name__ == "__main__":
    main()
