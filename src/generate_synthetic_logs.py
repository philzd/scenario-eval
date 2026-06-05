"""
Generate synthetic operational telemetry for scenario extraction
and evaluation pipeline development.

The generated data simulates autonomous-driving telemetry and
includes occasional hard braking, high jerk, and low time-to-
collision (TTC) events that can be discovered by downstream
scenario extraction workflows.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    """Configuration for synthetic telemetry generation."""
    n_rows: int
    freq_ms: int
    seed: int
    out: Path
    start: str

    # Typical vehicle dynamics ranges (comfortable driving most of the time).
    speed_mps_min: float = 0.0
    speed_mps_max: float = 40.0  # ~144 km/h
    accel_mps2_min: float = -6.0
    accel_mps2_max: float = 3.5

    # Anomaly magnitudes.
    hard_brake_mps2: float = -7.5
    high_jerk_mps3: float = 12.0
    low_ttc_s: float = 1.2


def _smooth_noise(rng: np.random.Generator, n: int, window: int) -> np.ndarray:
    """Generate low-frequency Gaussian noise via moving-average smoothing."""
    x = rng.normal(0.0, 1.0, size=n)
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _inject_hard_brake(
    accel: np.ndarray,
    rng: np.random.Generator,
    dt_s: float,
    n_events: int,
    magnitude: float,
    duration_s_range: tuple[float, float] = (0.7, 1.8),
) -> None:
    """Inject synthetic hard-braking events into acceleration."""
    n = accel.size
    for _ in range(n_events):
        duration_s = rng.uniform(*duration_s_range)
        length = max(3, int(round(duration_s / dt_s)))
        start = int(rng.integers(0, max(1, n - length)))
        # Taper in/out so jerk isn't infinite at edges.
        taper = np.hanning(length)
        accel[start : start + length] += (magnitude - accel[start : start + length]) * taper


def _inject_high_jerk(
    accel: np.ndarray,
    rng: np.random.Generator,
    dt_s: float,
    n_events: int,
    target_jerk: float,
    duration_s_range: tuple[float, float] = (0.2, 0.6),
) -> None:
    """Inject short-duration high-jerk events."""
    n = accel.size
    for _ in range(n_events):
        duration_s = rng.uniform(*duration_s_range)
        length = max(2, int(round(duration_s / dt_s)))
        start = int(rng.integers(1, max(2, n - length - 1)))

        # Build a sharp accel ramp to produce high jerk.
        # Jerk ~= delta_a / dt, so delta_a ~= target_jerk * dt.
        delta_a = float(target_jerk) * dt_s
        ramp = np.linspace(0.0, delta_a, num=length, dtype=float)
        if rng.random() < 0.5:
            ramp *= -1.0
        accel[start : start + length] += ramp


def _inject_low_ttc(
    distance: np.ndarray,
    rel_vel: np.ndarray,
    rng: np.random.Generator,
    dt_s: float,
    n_events: int,
    low_ttc_s: float,
    duration_s_range: tuple[float, float] = (0.6, 1.5),
) -> None:
    """
    Create occasional "closing fast" moments by forcing a small distance and a
    negative relative velocity (ego is faster than object ahead).
    """
    n = distance.size
    for _ in range(n_events):
        duration_s = rng.uniform(*duration_s_range)
        length = max(3, int(round(duration_s / dt_s)))
        start = int(rng.integers(0, max(1, n - length)))

        # Pick a closing speed and set distance so TTC ~ low_ttc_s.
        closing_speed = rng.uniform(6.0, 14.0)  # m/s closing
        rel_vel[start : start + length] = -closing_speed + rng.normal(0.0, 0.3, size=length)
        distance[start : start + length] = (
            closing_speed * low_ttc_s + rng.normal(0.0, 0.7, size=length)
        ).clip(min=0.5)


def generate_telemetry(cfg: Config) -> pd.DataFrame:
    """
    Generate synthetic operational telemetry with realistic vehicle
    dynamics and injected anomaly events.
    """
    rng = np.random.default_rng(cfg.seed)
    dt_s = cfg.freq_ms / 1000.0
    n = int(cfg.n_rows)

    # Timestamps.
    ts = pd.date_range(start=cfg.start, periods=n, freq=f"{cfg.freq_ms}ms")

    # Speed profile: base + gentle oscillations + smooth noise.
    base_speed = rng.uniform(10.0, 22.0)  # m/s
    t = np.arange(n) * dt_s
    wave = 2.2 * np.sin(2 * np.pi * t / rng.uniform(35.0, 65.0))
    drift = 1.0 * np.sin(2 * np.pi * t / rng.uniform(90.0, 150.0))
    noise = 0.55 * _smooth_noise(rng, n, window=max(3, int(round(2.0 / dt_s))))
    speed = base_speed + wave + drift + noise
    speed = np.clip(speed, cfg.speed_mps_min, cfg.speed_mps_max)

    # Accel and jerk derived from speed for physical consistency.
    accel = np.gradient(speed, dt_s)
    accel += 0.15 * _smooth_noise(rng, n, window=max(2, int(round(1.0 / dt_s))))
    accel = np.clip(accel, cfg.accel_mps2_min, cfg.accel_mps2_max)
    jerk = np.gradient(accel, dt_s)

    # Lead object: relative velocity and distance evolve smoothly.
    # relative_velocity = lead_speed - ego_speed
    rel_vel = 0.8 * _smooth_noise(rng, n, window=max(3, int(round(3.0 / dt_s))))
    rel_vel += 0.6 * np.sin(2 * np.pi * t / rng.uniform(25.0, 45.0))
    rel_vel = np.clip(rel_vel, -8.0, 8.0)

    # Distance integrates relative velocity (closing reduces distance).
    distance = np.empty(n, dtype=float)
    distance[0] = rng.uniform(18.0, 45.0)
    for i in range(1, n):
        distance[i] = distance[i - 1] + rel_vel[i - 1] * dt_s
        # Mild bias toward a reasonable headway.
        distance[i] += 0.02 * (30.0 - distance[i])
        distance[i] += rng.normal(0.0, 0.12)
    distance = np.clip(distance, 0.5, 200.0)

    # Inject anomalies (sparse, occasional).
    # Rate scales weakly with duration (so longer runs contain a few events).
    total_s = n * dt_s
    hard_brakes = max(1, int(round(total_s / 180.0)))
    high_jerks = max(1, int(round(total_s / 140.0)))
    low_ttcs = max(1, int(round(total_s / 220.0)))

    _inject_hard_brake(accel, rng, dt_s, n_events=hard_brakes, magnitude=cfg.hard_brake_mps2)
    _inject_high_jerk(accel, rng, dt_s, n_events=high_jerks, target_jerk=cfg.high_jerk_mps3)
    # Recompute jerk after accel edits.
    accel = np.clip(accel, -10.0, 5.0)
    jerk = np.gradient(accel, dt_s)

    _inject_low_ttc(distance, rel_vel, rng, dt_s, n_events=low_ttcs, low_ttc_s=cfg.low_ttc_s)
    distance = np.clip(distance, 0.5, 200.0)
    rel_vel = np.clip(rel_vel, -20.0, 20.0)

    df = pd.DataFrame(
        {
            "timestamp": ts,
            "speed": speed,
            "acceleration": accel,
            "jerk": jerk,
            "distance_to_object": distance,
            "relative_velocity": rel_vel,
        }
    )
    return df


def _parse_args() -> Config:
    """Parse command-line arguments and build a generation configuration."""
    p = argparse.ArgumentParser(description="Generate synthetic autonomous-driving telemetry.")
    p.add_argument("--rows", type=int, default=12_000, help="Number of rows to generate.")
    p.add_argument("--freq-ms", type=int, default=100, help="Sampling period in milliseconds.")
    p.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/synthetic_telemetry.csv"),
        help="Output CSV path.",
    )
    p.add_argument(
        "--start",
        type=str,
        default=pd.Timestamp.now(tz="UTC").isoformat(),
        help="Start timestamp (any pandas-parseable datetime).",
    )
    a = p.parse_args()
    return Config(n_rows=a.rows, freq_ms=a.freq_ms, seed=a.seed, out=a.out, start=a.start)


def main() -> None:
    """Generate telemetry and write the dataset to disk."""
    cfg = _parse_args()
    df = generate_telemetry(cfg)

    cfg.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.out, index=False)
    print(f"Wrote {len(df):,} rows to {cfg.out}")


if __name__ == "__main__":
    main()
