"""
Curated dataset generation from reviewed scenario records.

Builds an evaluation-ready dataset by selecting accepted scenarios from
the human-reviewed scenario queue.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ACCEPTED_STATUS = "accepted"


@dataclass(frozen=True)
class CuratedDatasetConfig:
    """Settings for reading reviewed scenarios and writing the curated dataset."""

    review_queue_path: Path
    output_path: Path


def load_review_queue(path: Path) -> pd.DataFrame:
    """
    Load reviewed scenario records from a CSV file.
    """
    return pd.read_csv(path)


def build_curated_dataset(review_queue: pd.DataFrame) -> list[dict[str, object]]:
    """
    Build curated dataset records from accepted review queue rows.
    """
    required_columns = [
        "rank",
        "scenario_id",
        "value_score",
        "severity_score",
        "rarity_score",
        "review_status",
        "label",
        "review_notes",
    ]
    missing_columns = [col for col in required_columns if col not in review_queue.columns]
    if missing_columns:
        raise ValueError(f"Review queue is missing required columns: {missing_columns}")

    accepted = review_queue[review_queue["review_status"] == ACCEPTED_STATUS].copy()
    accepted = accepted.sort_values("rank").reset_index(drop=True)

    records: list[dict[str, object]] = []
    for _, row in accepted.iterrows():
        records.append(
            {
                "scenario_id": row["scenario_id"],
                "rank": int(row["rank"]),
                "value_score": float(row["value_score"]),
                "severity_score": float(row["severity_score"]),
                "rarity_score": float(row["rarity_score"]),
                "label": "" if pd.isna(row["label"]) else str(row["label"]),
                "review_notes": ""
                if pd.isna(row["review_notes"])
                else str(row["review_notes"]),
            }
        )

    return records


def write_jsonl(records: list[dict[str, object]], path: Path) -> None:
    """
    Write curated dataset records to a JSONL file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def run_curated_dataset_pipeline(
    config: CuratedDatasetConfig,
) -> list[dict[str, object]]:
    """
    Run the curated dataset generation pipeline.
    """
    review_queue = load_review_queue(config.review_queue_path)
    return build_curated_dataset(review_queue)


def parse_args() -> CuratedDatasetConfig:
    """
    Parse command-line arguments for curated dataset generation.
    """
    parser = argparse.ArgumentParser(
        description="Build a curated dataset from accepted review queue records.",
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path("outputs/review_queue.csv"),
        help="Input review queue CSV path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/curated_dataset.jsonl"),
        help="Output curated dataset JSONL path.",
    )
    args = parser.parse_args()
    return CuratedDatasetConfig(
        review_queue_path=args.review_queue,
        output_path=args.output,
    )


def main() -> None:
    """
    Build the curated dataset and write it to JSONL.
    """
    config = parse_args()
    records = run_curated_dataset_pipeline(config)
    write_jsonl(records, config.output_path)
    print(f"Wrote {len(records):,} curated dataset records to {config.output_path}")


if __name__ == "__main__":
    main()
