"""
Dataset manifest generation from curated scenario datasets.

Builds dataset-level metadata summarizing dataset composition and label
distribution for downstream analysis and tracking.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATASET_VERSION = "v1"


@dataclass(frozen=True)
class DatasetManifestConfig:
    """Settings for reading the curated dataset and writing the manifest."""

    curated_dataset_path: Path
    output_path: Path
    dataset_version: str = DEFAULT_DATASET_VERSION


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
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Curated dataset record on line {line_number} is not valid JSON."
                ) from error

            records.append(record)

    return records


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


def build_manifest(
    records: list[dict[str, object]],
    config: DatasetManifestConfig,
) -> dict[str, object]:
    """
    Build dataset-level metadata from curated dataset records.
    """
    return {
        "dataset_version": config.dataset_version,
        "total_scenarios": len(records),
        "label_distribution": compute_label_distribution(records),
        "source_dataset": str(config.curated_dataset_path),
    }


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    """
    Write dataset manifest to a JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")


def run_manifest_pipeline(config: DatasetManifestConfig) -> dict[str, object]:
    """
    Run the dataset manifest generation pipeline.
    """
    records = load_curated_dataset(config.curated_dataset_path)
    return build_manifest(records, config)


def parse_args() -> DatasetManifestConfig:
    """
    Parse command-line arguments for dataset manifest generation.
    """
    parser = argparse.ArgumentParser(
        description="Build a dataset manifest from curated dataset records.",
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
        default=Path("outputs/dataset_manifest.json"),
        help="Output dataset manifest JSON path.",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=DEFAULT_DATASET_VERSION,
        help="Dataset version recorded in the manifest.",
    )
    args = parser.parse_args()
    return DatasetManifestConfig(
        curated_dataset_path=args.curated_dataset,
        output_path=args.output,
        dataset_version=args.dataset_version,
    )


def main() -> None:
    """
    Build the dataset manifest and write it to JSON.
    """
    config = parse_args()
    manifest = run_manifest_pipeline(config)
    write_manifest(manifest, config.output_path)
    print(f"Wrote dataset manifest to {config.output_path}")


if __name__ == "__main__":
    main()
