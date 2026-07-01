"""
Convert nuScenes scenes into canonical telemetry records.

The adapter reads ego poses and scene annotations, derives motion signals,
and writes a telemetry CSV that can be consumed by the evaluation workflow.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from nuscenes.nuscenes import NuScenes


@dataclass(frozen=True)
class NuScenesAdapterConfig:
    """Settings for reading nuScenes data and writing telemetry artifacts."""

    dataroot: Path
    version: str
    scene_name: str
    output_path: Path


@dataclass(frozen=True)
class EgoPoseRecord:
    """Ego vehicle position and nearest-object distance at one timestamp."""

    timestamp: pd.Timestamp
    ego_x: float
    ego_y: float
    distance_to_object: float


def load_nuscenes(dataroot: Path, version: str) -> NuScenes:
    """Load a nuScenes dataset from disk."""
    return NuScenes(version=version, dataroot=str(dataroot), verbose=True)


def get_scene_by_name(nusc: NuScenes, scene_name: str) -> dict[str, object]:
    """Return a scene record by scene name."""
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene

    available_scenes = [str(scene["name"]) for scene in nusc.scene]
    raise ValueError(f"Scene {scene_name} not found. Available scenes: {available_scenes}")


def iter_scene_samples(nusc: NuScenes, scene: dict[str, object]) -> list[dict[str, object]]:
    """Return all samples for a scene in time order."""
    samples: list[dict[str, object]] = []
    sample_token = str(scene["first_sample_token"])

    while sample_token:
        sample = nusc.get("sample", sample_token)
        samples.append(sample)
        sample_token = str(sample["next"])

    return samples


def get_ego_pose_for_sample(nusc: NuScenes, sample: dict[str, object]) -> dict[str, object]:
    """Return the ego pose associated with a sample."""
    sample_data = sample["data"]
    if not isinstance(sample_data, dict):
        raise ValueError("Sample data field must be a dictionary.")

    lidar_token = str(sample_data["LIDAR_TOP"])
    lidar_sample_data = nusc.get("sample_data", lidar_token)
    ego_pose_token = str(lidar_sample_data["ego_pose_token"])
    return nusc.get("ego_pose", ego_pose_token)


def compute_nearest_object_distance(
    nusc: NuScenes,
    sample: dict[str, object],
    ego_xy: np.ndarray,
) -> float:
    """Compute distance from the ego vehicle to the nearest annotated object."""
    annotation_tokens = sample["anns"]
    if not isinstance(annotation_tokens, list):
        raise ValueError("Sample annotations field must be a list.")

    distances: list[float] = []
    for annotation_token in annotation_tokens:
        annotation = nusc.get("sample_annotation", str(annotation_token))
        object_xy = np.array(annotation["translation"][:2], dtype=float)
        distances.append(float(np.linalg.norm(object_xy - ego_xy)))

    if not distances:
        return float("inf")

    return float(min(distances))


def build_ego_pose_records(nusc: NuScenes, scene: dict[str, object]) -> list[EgoPoseRecord]:
    """Build ego pose records for every sample in a scene."""
    records: list[EgoPoseRecord] = []

    for sample in iter_scene_samples(nusc, scene):
        ego_pose = get_ego_pose_for_sample(nusc, sample)
        ego_xy = np.array(ego_pose["translation"][:2], dtype=float)
        nearest_distance = compute_nearest_object_distance(nusc, sample, ego_xy)

        records.append(
            EgoPoseRecord(
                timestamp=pd.to_datetime(ego_pose["timestamp"], unit="us", utc=True),
                ego_x=float(ego_xy[0]),
                ego_y=float(ego_xy[1]),
                distance_to_object=nearest_distance,
            )
        )

    return records


def ego_pose_records_to_frame(records: list[EgoPoseRecord]) -> pd.DataFrame:
    """Convert ego pose records into a DataFrame."""
    rows = [
        {
            "timestamp": record.timestamp,
            "ego_x": record.ego_x,
            "ego_y": record.ego_y,
            "distance_to_object": record.distance_to_object,
        }
        for record in records
    ]
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def add_motion_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add speed, acceleration, jerk, and relative velocity signals."""
    telemetry = df.copy()

    dt = telemetry["timestamp"].diff().dt.total_seconds()
    dx = telemetry["ego_x"].diff()
    dy = telemetry["ego_y"].diff()

    distance_step = np.sqrt(dx**2 + dy**2)
    telemetry["speed"] = (distance_step / dt).fillna(0.0)
    telemetry["acceleration"] = (telemetry["speed"].diff() / dt).fillna(0.0)
    telemetry["jerk"] = (telemetry["acceleration"].diff() / dt).fillna(0.0)

    telemetry["relative_velocity"] = (
        telemetry["distance_to_object"].diff() / dt
    ).fillna(0.0)

    return telemetry


def select_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns required by the evaluation workflow."""
    return df[
        [
            "timestamp",
            "speed",
            "acceleration",
            "jerk",
            "distance_to_object",
            "relative_velocity",
        ]
    ]


def build_canonical_telemetry(nusc: NuScenes, scene_name: str) -> pd.DataFrame:
    """Build canonical telemetry for one nuScenes scene."""
    scene = get_scene_by_name(nusc, scene_name)
    pose_records = build_ego_pose_records(nusc, scene)
    pose_frame = ego_pose_records_to_frame(pose_records)
    telemetry = add_motion_signals(pose_frame)
    return select_canonical_columns(telemetry)


def write_telemetry(df: pd.DataFrame, output_path: Path) -> None:
    """Write canonical telemetry records to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def parse_args() -> NuScenesAdapterConfig:
    """Parse command-line arguments for the adapter."""
    parser = argparse.ArgumentParser(
        description="Convert a nuScenes scene into canonical telemetry.",
    )
    parser.add_argument(
        "--dataroot",
        type=Path,
        default=Path("../datasets/nuscenes"),
        help="Path to the nuScenes dataset root.",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0-mini",
        help="nuScenes dataset version.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default="scene-0061",
        help="nuScenes scene name to convert.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/real/nuscenes_telemetry.csv"),
        help="Output telemetry CSV path.",
    )
    args = parser.parse_args()

    return NuScenesAdapterConfig(
        dataroot=args.dataroot,
        version=args.version,
        scene_name=args.scene_name,
        output_path=args.output,
    )


def main() -> None:
    """Build canonical telemetry and write it to CSV."""
    config = parse_args()
    nusc = load_nuscenes(config.dataroot, config.version)
    telemetry = build_canonical_telemetry(nusc, config.scene_name)
    write_telemetry(telemetry, config.output_path)

    print(f"Wrote {len(telemetry):,} telemetry rows to {config.output_path}")
    print(f"Scene: {config.scene_name}")


if __name__ == "__main__":
    main()
