"""
Review queue generation from ranked scenario outputs.

Converts ranked scenarios into a human-review queue with default review
status and notes fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_REVIEW_STATUS = "pending"


@dataclass(frozen=True)
class ReviewQueueConfig:
    """Settings for reading ranked scenarios and writing the review queue."""

    ranked_path: Path
    output_path: Path


def load_ranked_scenarios(path: Path) -> pd.DataFrame:
    """
    Load ranked scenarios from a CSV file.
    """
    return pd.read_csv(path)


def build_review_queue(ranked: pd.DataFrame) -> pd.DataFrame:
    """
    Build a review queue from ranked scenarios.
    """
    required_columns = [
        "rank",
        "scenario_id",
        "value_score",
        "severity_score",
        "rarity_score",
    ]
    missing_columns = [col for col in required_columns if col not in ranked.columns]
    if missing_columns:
        raise ValueError(f"Ranked scenarios are missing required columns: {missing_columns}")

    queue = ranked[required_columns].copy()
    queue["review_status"] = DEFAULT_REVIEW_STATUS
    queue["label"] = ""
    queue["review_notes"] = ""

    return queue.sort_values("rank").reset_index(drop=True)


def write_review_queue(review_queue: pd.DataFrame, path: Path) -> None:
    """
    Write a review queue to a CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    review_queue.to_csv(path, index=False)


def run_review_queue_pipeline(config: ReviewQueueConfig) -> pd.DataFrame:
    """
    Run the review queue pipeline.
    """
    ranked_df = load_ranked_scenarios(config.ranked_path)
    return build_review_queue(ranked_df)


def parse_args() -> ReviewQueueConfig:
    """
    Parse command-line arguments for the review queue pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Create a review queue from ranked scenarios.",
    )
    parser.add_argument(
        "--ranked",
        type=Path,
        default=Path("outputs/ranked_scenarios.csv"),
        help="Input ranked scenarios CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/review_queue.csv"),
        help="Output review queue CSV path.",
    )
    args = parser.parse_args()
    return ReviewQueueConfig(
        ranked_path=args.ranked,
        output_path=args.output,
    )


def main() -> None:
    """
    Generate a review queue and write it to CSV.
    """
    config = parse_args()
    review_queue = run_review_queue_pipeline(config)
    write_review_queue(review_queue, config.output_path)
    print(f"Wrote {len(review_queue):,} review queue records to {config.output_path}")


if __name__ == "__main__":
    main()
