"""
End-to-end scenario evaluation pipeline.

Runs synthetic telemetry generation, scenario extraction, metric computation,
and scenario scoring in sequence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.compute_metrics import (
    MetricsConfig,
    run_metrics_pipeline,
    write_scenario_metrics_jsonl,
)
from src.extract_scenarios import (
    ExtractionConfig,
    extract_scenarios,
    write_scenarios_jsonl,
)
from src.generate_synthetic_logs import Config as GenerationConfig
from src.generate_synthetic_logs import generate_telemetry
from src.score_scenarios import (
    ScoringConfig,
    run_scoring_pipeline,
    write_ranked_scenarios_csv,
    write_scenario_scores_jsonl,
)

DEFAULT_ROWS = 12_000
DEFAULT_SEED = 7
DEFAULT_FREQ_MS = 100


@dataclass(frozen=True)
class PipelineConfig:
    """Settings for a full pipeline run."""

    rows: int
    seed: int
    telemetry_path: Path
    scenarios_path: Path
    metrics_path: Path
    scores_path: Path
    ranked_path: Path
    freq_ms: int = DEFAULT_FREQ_MS


@dataclass(frozen=True)
class PipelineSummary:
    """Counts and paths produced by a pipeline run."""

    telemetry_rows: int
    scenarios_extracted: int
    metrics_written: int
    scores_written: int
    ranked_path: Path


def build_generation_config(config: PipelineConfig) -> GenerationConfig:
    """
    Build the synthetic telemetry generator config from pipeline settings.
    """
    return GenerationConfig(
        n_rows=config.rows,
        freq_ms=config.freq_ms,
        seed=config.seed,
        out=config.telemetry_path,
        start=pd.Timestamp.now(tz="UTC").isoformat(),
    )


def build_extraction_config(config: PipelineConfig) -> ExtractionConfig:
    """
    Build the scenario extraction config from pipeline settings.
    """
    return ExtractionConfig(
        input_path=config.telemetry_path,
        output_path=config.scenarios_path,
        source_log=config.telemetry_path.name,
    )


def build_metrics_config(config: PipelineConfig) -> MetricsConfig:
    """
    Build the scenario metrics config from pipeline settings.
    """
    return MetricsConfig(
        telemetry_path=config.telemetry_path,
        scenarios_path=config.scenarios_path,
        output_path=config.metrics_path,
    )


def build_scoring_config(config: PipelineConfig) -> ScoringConfig:
    """
    Build the scenario scoring config from pipeline settings.
    """
    return ScoringConfig(
        metrics_path=config.metrics_path,
        scores_output_path=config.scores_path,
        ranked_output_path=config.ranked_path,
    )


def run_generation_stage(config: PipelineConfig) -> int:
    """
    Generate synthetic telemetry and write it to CSV.
    """
    print("Generating synthetic telemetry...")
    generation_config = build_generation_config(config)
    telemetry = generate_telemetry(generation_config)

    generation_config.out.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(generation_config.out, index=False)
    return len(telemetry)


def run_extraction_stage(config: PipelineConfig) -> int:
    """
    Extract scenarios from telemetry and write them to JSONL.
    """
    print("Extracting scenarios...")
    extraction_config = build_extraction_config(config)
    scenarios = extract_scenarios(extraction_config)
    write_scenarios_jsonl(scenarios, extraction_config.output_path)
    return len(scenarios)


def run_metrics_stage(config: PipelineConfig) -> int:
    """
    Compute scenario metrics and write them to JSONL.
    """
    print("Computing scenario metrics...")
    metrics_config = build_metrics_config(config)
    metrics = run_metrics_pipeline(metrics_config)
    write_scenario_metrics_jsonl(metrics, metrics_config.output_path)
    return len(metrics)


def run_scoring_stage(config: PipelineConfig) -> int:
    """
    Score and rank scenarios and write score artifacts.
    """
    print("Scoring and ranking scenarios...")
    scoring_config = build_scoring_config(config)
    scores, ranked_scores = run_scoring_pipeline(scoring_config)
    write_scenario_scores_jsonl(scores, scoring_config.scores_output_path)
    write_ranked_scenarios_csv(ranked_scores, scoring_config.ranked_output_path)
    return len(scores)


def run_pipeline(config: PipelineConfig) -> PipelineSummary:
    """
    Run pipeline stages in order.
    """
    telemetry_rows = run_generation_stage(config)
    scenarios_extracted = run_extraction_stage(config)
    metrics_written = run_metrics_stage(config)
    scores_written = run_scoring_stage(config)

    return PipelineSummary(
        telemetry_rows=telemetry_rows,
        scenarios_extracted=scenarios_extracted,
        metrics_written=metrics_written,
        scores_written=scores_written,
        ranked_path=config.ranked_path,
    )


def print_pipeline_summary(summary: PipelineSummary) -> None:
    """
    Print a short summary of pipeline outputs.
    """
    print()
    print("Pipeline complete.")
    print()
    print(f"  Telemetry rows generated:  {summary.telemetry_rows:,}")
    print(f"  Scenarios extracted:       {summary.scenarios_extracted:,}")
    print(f"  Metrics records written:   {summary.metrics_written:,}")
    print(f"  Scores written:            {summary.scores_written:,}")
    print(f"  Ranked CSV:                {summary.ranked_path}")


def parse_args() -> PipelineConfig:
    """
    Parse command-line arguments for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run the pipeline end to end.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=DEFAULT_ROWS,
        help="Number of synthetic telemetry rows to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for synthetic telemetry generation.",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("data/raw/synthetic_telemetry.csv"),
        help="Telemetry CSV path used by the pipeline.",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=Path("outputs/scenarios.jsonl"),
        help="Scenarios JSONL path.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/scenario_metrics.jsonl"),
        help="Scenario metrics JSONL path.",
    )
    parser.add_argument(
        "--scores",
        type=Path,
        default=Path("outputs/scenario_scores.jsonl"),
        help="Scenario scores JSONL path.",
    )
    parser.add_argument(
        "--ranked",
        type=Path,
        default=Path("outputs/ranked_scenarios.csv"),
        help="Ranked scenarios CSV path.",
    )
    args = parser.parse_args()
    return PipelineConfig(
        rows=args.rows,
        seed=args.seed,
        telemetry_path=args.telemetry,
        scenarios_path=args.scenarios,
        metrics_path=args.metrics,
        scores_path=args.scores,
        ranked_path=args.ranked,
    )


def main() -> None:
    """
    Run the pipeline and print a summary.
    """
    config = parse_args()
    summary = run_pipeline(config)
    print_pipeline_summary(summary)


if __name__ == "__main__":
    main()
