"""
Failure summary generation from ranked scenarios and review decisions.

Builds high-level failure discovery metrics that summarize trigger
patterns, review labels, and top-priority scenarios.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ACCEPTED_STATUS = "accepted"


@dataclass(frozen=True)
class FailureSummaryConfig:
    """Settings for reading inputs and writing the failure summary."""

    ranked_path: Path
    review_queue_path: Path
    output_path: Path
    top_k: int = 5


def load_ranked_scenarios(path: Path) -> pd.DataFrame:
    """
    Load ranked scenarios from a CSV file.
    """
    return pd.read_csv(path)


def load_review_queue(path: Path) -> pd.DataFrame:
    """
    Load review queue records from a CSV file.
    """
    return pd.read_csv(path)


def split_trigger_types(trigger_types: object) -> list[str]:
    """
    Split a semicolon-delimited trigger type field into trigger labels.
    """
    if pd.isna(trigger_types):
        return []

    return [
        trigger.strip()
        for trigger in str(trigger_types).split(";")
        if trigger.strip()
    ]


def compute_trigger_type_counts(ranked: pd.DataFrame) -> dict[str, int]:
    """
    Count trigger types across ranked scenarios.
    """
    if "trigger_types" not in ranked.columns:
        raise ValueError("Ranked scenarios are missing required column: trigger_types")

    counts: Counter[str] = Counter()
    for trigger_value in ranked["trigger_types"]:
        counts.update(split_trigger_types(trigger_value))

    return dict(counts)


def compute_label_counts(review_queue: pd.DataFrame) -> dict[str, int]:
    """
    Count non-empty review labels across the review queue.
    """
    if "label" not in review_queue.columns:
        raise ValueError("Review queue is missing required column: label")

    labels = [
        str(label)
        for label in review_queue["label"]
        if not pd.isna(label) and str(label).strip()
    ]

    return dict(Counter(labels))


def compute_accepted_label_counts(review_queue: pd.DataFrame) -> dict[str, int]:
    """
    Count labels for scenarios accepted into the curated dataset.
    """
    required_columns = ["review_status", "label"]
    missing_columns = [col for col in required_columns if col not in review_queue.columns]
    if missing_columns:
        raise ValueError(f"Review queue is missing required columns: {missing_columns}")

    accepted = review_queue[review_queue["review_status"] == ACCEPTED_STATUS]
    labels = [
        str(label)
        for label in accepted["label"]
        if not pd.isna(label) and str(label).strip()
    ]

    return dict(Counter(labels))


def get_most_common(counts: dict[str, int]) -> str | None:
    """
    Return the most common key from a count dictionary.
    """
    if not counts:
        return None

    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def build_top_ranked_scenarios(
    ranked: pd.DataFrame,
    top_k: int,
) -> list[dict[str, object]]:
    """
    Build a compact list of the highest-ranked scenarios.
    """
    required_columns = ["rank", "scenario_id", "value_score", "trigger_types"]
    missing_columns = [col for col in required_columns if col not in ranked.columns]
    if missing_columns:
        raise ValueError(f"Ranked scenarios are missing required columns: {missing_columns}")

    top_ranked = ranked.sort_values("rank").head(top_k)

    records: list[dict[str, object]] = []
    for _, row in top_ranked.iterrows():
        records.append(
            {
                "rank": int(row["rank"]),
                "scenario_id": str(row["scenario_id"]),
                "value_score": float(row["value_score"]),
                "trigger_types": split_trigger_types(row["trigger_types"]),
            }
        )

    return records


def build_top_accepted_scenarios(
    review_queue: pd.DataFrame,
    top_k: int,
) -> list[dict[str, object]]:
    """
    Build a compact list of the highest-ranked accepted scenarios.
    """
    required_columns = ["rank", "scenario_id", "value_score", "review_status", "label"]
    missing_columns = [col for col in required_columns if col not in review_queue.columns]
    if missing_columns:
        raise ValueError(f"Review queue is missing required columns: {missing_columns}")

    accepted = review_queue[review_queue["review_status"] == ACCEPTED_STATUS]
    top_accepted = accepted.sort_values("rank").head(top_k)

    records: list[dict[str, object]] = []
    for _, row in top_accepted.iterrows():
        records.append(
            {
                "rank": int(row["rank"]),
                "scenario_id": str(row["scenario_id"]),
                "value_score": float(row["value_score"]),
                "label": "" if pd.isna(row["label"]) else str(row["label"]),
            }
        )

    return records


def build_failure_summary(
    ranked: pd.DataFrame,
    review_queue: pd.DataFrame,
    top_k: int,
) -> dict[str, object]:
    """
    Build high-level failure discovery metrics.
    """
    trigger_counts = compute_trigger_type_counts(ranked)
    review_label_counts = compute_label_counts(review_queue)
    accepted_label_counts = compute_accepted_label_counts(review_queue)

    return {
        "trigger_type_counts": trigger_counts,
        "review_label_counts": review_label_counts,
        "accepted_label_counts": accepted_label_counts,
        "most_common_trigger": get_most_common(trigger_counts),
        "most_common_review_label": get_most_common(review_label_counts),
        "most_common_accepted_label": get_most_common(accepted_label_counts),
        "top_ranked_scenarios": build_top_ranked_scenarios(ranked, top_k),
        "top_accepted_scenarios": build_top_accepted_scenarios(review_queue, top_k),
    }


def write_failure_summary(summary: dict[str, object], path: Path) -> None:
    """
    Write failure summary to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


def run_failure_summary_pipeline(config: FailureSummaryConfig) -> dict[str, object]:
    """
    Run the failure summary generation pipeline.
    """
    ranked = load_ranked_scenarios(config.ranked_path)
    review_queue = load_review_queue(config.review_queue_path)
    return build_failure_summary(ranked, review_queue, config.top_k)


def parse_args() -> FailureSummaryConfig:
    """
    Parse command-line arguments for failure summary generation.
    """
    parser = argparse.ArgumentParser(
        description="Build a failure summary from ranked scenarios and review decisions.",
    )
    parser.add_argument(
        "--ranked",
        type=Path,
        default=Path("outputs/ranked_scenarios.csv"),
        help="Input ranked scenarios CSV path.",
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
        default=Path("outputs/failure_summary.json"),
        help="Output failure summary JSON path.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top scenarios to include in summary fields.",
    )
    args = parser.parse_args()
    return FailureSummaryConfig(
        ranked_path=args.ranked,
        review_queue_path=args.review_queue,
        output_path=args.output,
        top_k=args.top_k,
    )


def main() -> None:
    """
    Build the failure summary and write it to JSON.
    """
    config = parse_args()
    summary = run_failure_summary_pipeline(config)
    write_failure_summary(summary, config.output_path)
    print(f"Wrote failure summary to {config.output_path}")


if __name__ == "__main__":
    main()
