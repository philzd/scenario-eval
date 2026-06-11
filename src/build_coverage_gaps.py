"""
Coverage gap generation from coverage report artifacts.

Extracts represented and missing label categories from the coverage report
to make dataset coverage gaps easier to inspect and consume downstream.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CoverageGapsConfig:
    """Settings for reading the coverage report and writing coverage gaps."""

    coverage_report_path: Path
    output_path: Path


def load_coverage_report(path: Path) -> dict[str, object]:
    """
    Load a coverage report from a JSON file.
    """
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build_coverage_gaps(report: dict[str, object]) -> dict[str, object]:
    """
    Build coverage gap records from a coverage report.
    """
    required_fields = ["represented_labels", "missing_labels"]
    missing_fields = [field for field in required_fields if field not in report]
    if missing_fields:
        raise ValueError(f"Coverage report is missing required fields: {missing_fields}")

    represented_labels = list(report["represented_labels"])
    missing_labels = list(report["missing_labels"])

    return {
        "represented_labels": represented_labels,
        "missing_labels": missing_labels,
        "num_represented_labels": len(represented_labels),
        "num_missing_labels": len(missing_labels),
    }


def write_coverage_gaps(gaps: dict[str, object], path: Path) -> None:
    """
    Write coverage gaps to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(gaps, handle, indent=2)
        handle.write("\n")


def run_coverage_gaps_pipeline(config: CoverageGapsConfig) -> dict[str, object]:
    """
    Run the coverage gap generation pipeline.
    """
    report = load_coverage_report(config.coverage_report_path)
    return build_coverage_gaps(report)


def parse_args() -> CoverageGapsConfig:
    """
    Parse command-line arguments for coverage gap generation.
    """
    parser = argparse.ArgumentParser(
        description="Build coverage gap artifacts from a coverage report.",
    )
    parser.add_argument(
        "--coverage-report",
        type=Path,
        default=Path("outputs/coverage_report.json"),
        help="Input coverage report JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/coverage_gaps.json"),
        help="Output coverage gaps JSON path.",
    )
    args = parser.parse_args()
    return CoverageGapsConfig(
        coverage_report_path=args.coverage_report,
        output_path=args.output,
    )


def main() -> None:
    """
    Build coverage gaps and write them to JSON.
    """
    config = parse_args()
    gaps = run_coverage_gaps_pipeline(config)
    write_coverage_gaps(gaps, config.output_path)
    print(f"Wrote coverage gaps to {config.output_path}")


if __name__ == "__main__":
    main()
