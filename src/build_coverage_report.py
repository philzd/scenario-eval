"""
Coverage report generation from review and curated dataset artifacts.

Builds dataset-level coverage metrics that summarize review status,
label representation, and missing label categories.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


EXPECTED_LABELS = (
    "safety_critical",
    "edge_case",
    "interesting",
    "false_positive",
    "regression",
)


@dataclass(frozen=True)
class CoverageReportConfig:
    """Settings for reading inputs and writing the coverage report."""

    review_queue_path: Path
    curated_dataset_path: Path
    output_path: Path


def load_review_queue(path: Path) -> pd.DataFrame:
    """
    Load review queue records from a CSV file.
    """
    return pd.read_csv(path)


def load_curated_dataset(path: Path) -> list[dict[str, object]]:
    """
    Load curated dataset records from a JSONL file.
    """
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Curated dataset record on line {line_number} is not valid JSON."
                ) from error

    return records


def compute_review_status_counts(review_queue: pd.DataFrame) -> dict[str, int]:
    """
    Compute counts for each review status.
    """
    if "review_status" not in review_queue.columns:
        raise ValueError("Review queue is missing required column: review_status")

    counts = review_queue["review_status"].fillna("missing").value_counts()
    return {str(status): int(count) for status, count in counts.items()}


def compute_acceptance_rate(status_counts: dict[str, int]) -> float:
    """
    Compute the fraction of review queue records marked as accepted.
    """
    total = sum(status_counts.values())
    if total == 0:
        return 0.0

    accepted = status_counts.get("accepted", 0)
    return float(accepted / total)


def compute_label_distribution(
    records: list[dict[str, object]],
) -> dict[str, int]:
    """
    Compute label counts across curated dataset records.
    """
    labels = [
        str(record["label"])
        for record in records
        if "label" in record and record["label"] not in (None, "")
    ]
    return dict(Counter(labels))


def compute_represented_labels(
    label_distribution: dict[str, int],
) -> list[str]:
    """
    Return expected labels present in the curated dataset.
    """
    return [label for label in EXPECTED_LABELS if label_distribution.get(label, 0) > 0]


def compute_missing_labels(
    label_distribution: dict[str, int],
) -> list[str]:
    """
    Return expected labels missing from the curated dataset.
    """
    return [label for label in EXPECTED_LABELS if label_distribution.get(label, 0) == 0]


def build_coverage_report(
    review_queue: pd.DataFrame,
    curated_records: list[dict[str, object]],
) -> dict[str, object]:
    """
    Build coverage metrics from review queue and curated dataset artifacts.
    """
    status_counts = compute_review_status_counts(review_queue)
    label_distribution = compute_label_distribution(curated_records)

    return {
        "total_review_queue_items": int(len(review_queue)),
        "review_status_counts": status_counts,
        "accepted_count": status_counts.get("accepted", 0),
        "rejected_count": status_counts.get("rejected", 0),
        "pending_count": status_counts.get("pending", 0),
        "needs_review_count": status_counts.get("needs_review", 0),
        "acceptance_rate": compute_acceptance_rate(status_counts),
        "curated_dataset_size": int(len(curated_records)),
        "label_distribution": label_distribution,
        "represented_labels": compute_represented_labels(label_distribution),
        "missing_labels": compute_missing_labels(label_distribution),
    }


def write_coverage_report(report: dict[str, object], path: Path) -> None:
    """
    Write the coverage report to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


def run_coverage_report_pipeline(config: CoverageReportConfig) -> dict[str, object]:
    """
    Run the coverage report generation pipeline.
    """
    review_queue = load_review_queue(config.review_queue_path)
    curated_records = load_curated_dataset(config.curated_dataset_path)
    return build_coverage_report(review_queue, curated_records)


def parse_args() -> CoverageReportConfig:
    """
    Parse command-line arguments for coverage report generation.
    """
    parser = argparse.ArgumentParser(
        description="Build a coverage report from review and curated dataset artifacts.",
    )
    parser.add_argument(
        "--review-queue",
        type=Path,
        default=Path("outputs/review_queue.csv"),
        help="Input review queue CSV path.",
    )
    parser.add_argument(
        "--curated-dataset",
        type=Path,
        default=Path("outputs/curated_dataset.jsonl"),
        help="Input curated dataset JSONL path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/coverage_report.json"),
        help="Output coverage report JSON path.",
    )
    args = parser.parse_args()
    return CoverageReportConfig(
        review_queue_path=args.review_queue,
        curated_dataset_path=args.curated_dataset,
        output_path=args.output,
    )


def main() -> None:
    """
    Build the coverage report and write it to JSON.
    """
    config = parse_args()
    report = run_coverage_report_pipeline(config)
    write_coverage_report(report, config.output_path)
    print(f"Wrote coverage report to {config.output_path}")


if __name__ == "__main__":
    main()
