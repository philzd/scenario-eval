"""
Static visualization of pipeline outputs.

Generates PNG plots for telemetry timelines, extracted scenario windows,
and ranked scenario scores to support inspection and review workflows.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.extract_scenarios import enrich_telemetry, load_telemetry

DEFAULT_TTC_THRESHOLD_S = 2.0
DEFAULT_ACCEL_THRESHOLD = -4.0
DEFAULT_JERK_THRESHOLD = 5.0
DEFAULT_TTC_PLOT_CAP_S = 60.0
PLOT_DPI = 150


@dataclass(frozen=True)
class VisualizationConfig:
    """Settings for loading inputs and writing plot files."""

    telemetry_path: Path
    scenarios_path: Path
    ranked_path: Path
    out_dir: Path
    ttc_threshold_s: float = DEFAULT_TTC_THRESHOLD_S
    accel_threshold: float = DEFAULT_ACCEL_THRESHOLD
    jerk_threshold: float = DEFAULT_JERK_THRESHOLD
    ttc_plot_cap_s: float = DEFAULT_TTC_PLOT_CAP_S


@dataclass(frozen=True)
class ScenarioWindow:
    """Scenario time bounds used for shading timeline plots."""

    scenario_id: str
    start_time: pd.Timestamp
    end_time: pd.Timestamp


def load_scenario_windows(path: Path) -> list[ScenarioWindow]:
    """
    Load scenario windows from a JSONL file.
    """
    windows: list[ScenarioWindow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            try:
                windows.append(
                    ScenarioWindow(
                        scenario_id=record["scenario_id"],
                        start_time=pd.Timestamp(record["start_time"]),
                        end_time=pd.Timestamp(record["end_time"]),
                    )
                )
            except KeyError as error:
                raise ValueError(
                    f"Scenario record on line {line_number} is missing field: {error}"
                ) from error

    return windows


def load_ranked_scenarios(path: Path) -> pd.DataFrame:
    """
    Load ranked scenarios from a CSV file sorted by rank ascending.
    """
    ranked = pd.read_csv(path)
    if "rank" in ranked.columns:
        ranked = ranked.sort_values("rank", ascending=True)
    return ranked


def shade_scenario_windows(ax: plt.Axes, scenarios: list[ScenarioWindow]) -> None:
    """
    Shade extracted scenario windows on a time-series axes.
    """
    for scenario in scenarios:
        ax.axvspan(
            scenario.start_time,
            scenario.end_time,
            alpha=0.15,
            color="gray",
            zorder=0,
        )


def prepare_ttc_for_plot(ttc: pd.Series) -> pd.Series:
    """
    Replace non-finite TTC values with NaN so non-closing segments are not drawn.
    """
    plot_ttc = ttc.astype(float).copy()
    plot_ttc[~np.isfinite(plot_ttc)] = np.nan
    return plot_ttc


def plot_speed_timeline(
    df: pd.DataFrame,
    scenarios: list[ScenarioWindow],
    path: Path,
) -> None:
    """
    Plot speed over time with shaded scenario windows.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    shade_scenario_windows(ax, scenarios)
    ax.plot(df["timestamp"], df["speed"], color="steelblue", linewidth=1.0, zorder=2)
    ax.set_title("Speed Over Time with Extracted Scenario Windows")
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ttc_timeline(
    df: pd.DataFrame,
    scenarios: list[ScenarioWindow],
    path: Path,
    config: VisualizationConfig,
) -> None:
    """
    Plot TTC over time with a threshold line and shaded scenario windows.
    """
    plot_ttc = prepare_ttc_for_plot(df["ttc"])

    fig, ax = plt.subplots(figsize=(12, 5))
    shade_scenario_windows(ax, scenarios)
    ax.plot(df["timestamp"], plot_ttc, color="darkorange", linewidth=1.0, zorder=2)
    ax.axhline(
        config.ttc_threshold_s,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"TTC threshold ({config.ttc_threshold_s:.1f} s)",
        zorder=3,
    )
    ax.set_ylim(0, config.ttc_plot_cap_s)
    ax.set_title("Time-to-Collision with Scenario Windows")
    ax.set_xlabel("Time")
    ax.set_ylabel("TTC (s)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_acceleration_jerk_timeline(
    df: pd.DataFrame,
    scenarios: list[ScenarioWindow],
    path: Path,
    config: VisualizationConfig,
) -> None:
    """
    Plot acceleration and jerk over time in stacked subplots with thresholds.
    """
    fig, (accel_ax, jerk_ax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    shade_scenario_windows(accel_ax, scenarios)
    shade_scenario_windows(jerk_ax, scenarios)

    accel_ax.plot(
        df["timestamp"],
        df["acceleration"],
        color="seagreen",
        linewidth=1.0,
        zorder=2,
    )
    accel_ax.axhline(
        config.accel_threshold,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Accel threshold ({config.accel_threshold:.1f} m/s²)",
        zorder=3,
    )
    accel_ax.set_title("Acceleration with Scenario Windows")
    accel_ax.set_ylabel("Acceleration (m/s²)")
    accel_ax.legend(loc="upper right")
    accel_ax.grid(True, alpha=0.3)

    jerk_ax.plot(df["timestamp"], df["jerk"], color="purple", linewidth=1.0, zorder=2)
    jerk_ax.axhline(
        config.jerk_threshold,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Jerk threshold (+{config.jerk_threshold:.1f} m/s³)",
        zorder=3,
    )
    jerk_ax.axhline(
        -config.jerk_threshold,
        color="red",
        linestyle="--",
        linewidth=1.0,
        label=f"Jerk threshold (-{config.jerk_threshold:.1f} m/s³)",
        zorder=3,
    )
    jerk_ax.set_title("Jerk with Scenario Windows")
    jerk_ax.set_xlabel("Time")
    jerk_ax.set_ylabel("Jerk (m/s³)")
    jerk_ax.legend(loc="upper right")
    jerk_ax.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_ranked_scenario_scores(ranked_df: pd.DataFrame, path: Path) -> None:
    """
    Plot stacked horizontal bars of severity and rarity scores ordered by rank.
    """
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(ranked_df) + 1)))

    if ranked_df.empty:
        ax.set_title("Ranked Scenario Scores by Severity and Rarity")
        ax.text(0.5, 0.5, "No ranked scenarios available.", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        return

    labels = [
        f"{int(row['rank'])}: {row['scenario_id']}"
        for _, row in ranked_df.iterrows()
    ]
    severity_scores = ranked_df["severity_score"].astype(float)
    rarity_scores = ranked_df["rarity_score"].astype(float)

    ax.barh(
        labels,
        severity_scores,
        label="severity_score",
        color="steelblue",
    )
    ax.barh(
        labels,
        rarity_scores,
        left=severity_scores,
        label="rarity_score",
        color="darkorange",
    )
    ax.invert_yaxis()
    ax.set_title("Ranked Scenario Scores by Severity and Rarity")
    ax.set_xlabel("Score")
    ax.set_ylabel("Scenario")
    ax.legend(loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def run_visualization(config: VisualizationConfig) -> list[Path]:
    """
    Generate visualization plots from pipeline outputs.
    """
    config.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_telemetry(config.telemetry_path)
    df = enrich_telemetry(df)
    scenarios = load_scenario_windows(config.scenarios_path)
    ranked_df = load_ranked_scenarios(config.ranked_path)

    plot_paths = [
        config.out_dir / "speed_timeline.png",
        config.out_dir / "ttc_timeline.png",
        config.out_dir / "acceleration_jerk_timeline.png",
        config.out_dir / "ranked_scenario_scores.png",
    ]

    plot_speed_timeline(df, scenarios, plot_paths[0])
    plot_ttc_timeline(df, scenarios, plot_paths[1], config)
    plot_acceleration_jerk_timeline(df, scenarios, plot_paths[2], config)
    plot_ranked_scenario_scores(ranked_df, plot_paths[3])

    return plot_paths


def parse_args() -> VisualizationConfig:
    """
    Parse command-line arguments for visualization generation.
    """
    parser = argparse.ArgumentParser(
        description="Generate static plots from pipeline outputs.",
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
        "--ranked",
        type=Path,
        default=Path("outputs/ranked_scenarios.csv"),
        help="Input ranked scenarios CSV path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/plots"),
        help="Output directory for PNG plots.",
    )
    args = parser.parse_args()
    return VisualizationConfig(
        telemetry_path=args.telemetry,
        scenarios_path=args.scenarios,
        ranked_path=args.ranked,
        out_dir=args.out_dir,
    )


def main() -> None:
    """
    Generate visualization plots and print a short summary.
    """
    config = parse_args()
    plot_paths = run_visualization(config)

    print(f"Wrote {len(plot_paths)} plots to {config.out_dir}/")
    for plot_path in plot_paths:
        print(f"  {plot_path.name}")


if __name__ == "__main__":
    main()
