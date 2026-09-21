"""Synthetic data generator, for smoke tests and for trying the UI.

Simulates correlated random walks for several "treatments" that differ in speed,
turning, pause structure and beat frequency -- roughly the way a ciliate or a
larval fish assay differs between conditions.  Output columns are named exactly
as TrackMate exports them, so the demo also exercises the column auto-detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# name: (speed px/frame, turn SD deg, pause probability, beat period frames,
#        body major axis px)
PRESETS: dict[str, tuple[float, float, float, float, float]] = {
    "control":   (3.0, 18.0, 0.06, 9.0, 14.0),
    "drug_low":  (2.4, 26.0, 0.14, 11.0, 14.5),
    "drug_high": (1.5, 42.0, 0.30, 15.0, 15.5),
    "mutant":    (3.4, 15.0, 0.05, 6.0, 11.0),
}


def simulate_group(
    name: str,
    n_tracks: int = 18,
    n_frames: int = 320,
    field_size: float = 1024.0,
    seed: int = 0,
    jitter: float = 0.18,
) -> pd.DataFrame:
    speed, turn_sd, p_pause, beat, major = PRESETS[name]
    rng = np.random.default_rng(seed)
    rows = []

    for tid in range(n_tracks):
        # animal-to-animal variability
        v = speed * (1 + rng.normal(0, jitter))
        ts = turn_sd * (1 + rng.normal(0, jitter))
        pp = float(np.clip(p_pause * (1 + rng.normal(0, jitter)), 0.0, 0.8))
        bp = beat * (1 + rng.normal(0, jitter * 0.5))
        bl = major * (1 + rng.normal(0, 0.08))

        x, y = rng.uniform(0.2, 0.8, 2) * field_size
        heading = rng.uniform(0, 2 * np.pi)
        paused = False

        for f in range(n_frames):
            paused = (rng.random() < pp) if not paused else (rng.random() > 0.45)
            step = 0.0 if paused else max(v + rng.normal(0, v * 0.25), 0.0)

            # rhythmic component on top of the random turn
            heading += np.radians(rng.normal(0, ts)) + 0.30 * np.sin(2 * np.pi * f / bp)
            x = float(np.clip(x + step * np.cos(heading), 5, field_size - 5))
            y = float(np.clip(y + step * np.sin(heading), 5, field_size - 5))

            wobble = 0.10 * np.sin(2 * np.pi * f / bp)
            rows.append({
                "LABEL": f"{name}_T{tid}_S{f}",
                "TRACK_ID": f"{name}_{tid:03d}",
                "POSITION_X": x,
                "POSITION_Y": y,
                "POSITION_T": f,
                "FRAME": f,
                "AREA": max(np.pi * (bl / 2) * (bl / 3.2) * (1 + wobble), 1.0),
                "ELLIPSE_MAJOR": bl * (1 + wobble),
                "ELLIPSE_MINOR": bl / 3.0 * (1 - wobble * 0.5),
                "ELLIPSE_THETA": (np.degrees(heading) + rng.normal(0, 12)) % 180.0,
                "MEAN_INTENSITY_CH1": 900 + 120 * np.sin(2 * np.pi * f / 90) + rng.normal(0, 25),
                "STD_INTENSITY_CH1": 180 + 60 * np.cos(2 * np.pi * f / 90) + rng.normal(0, 12),
                "ESTIMATED_DIAMETER": bl * 0.8,
            })
    return pd.DataFrame(rows)


def write_demo_dataset(
    out_dir: str | Path,
    groups: list[str] | None = None,
    n_tracks: int = 18,
    n_frames: int = 320,
    replicates: int = 2,
    seed: int = 0,
) -> list[Path]:
    """One CSV per group x replicate, mimicking one movie per file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    groups = groups or list(PRESETS)
    paths: list[Path] = []
    s = seed
    for g in groups:
        for r in range(1, replicates + 1):
            df = simulate_group(g, n_tracks=n_tracks, n_frames=n_frames, seed=s)
            s += 1
            path = out / f"spots_{g}_rep{r}.csv"
            df.to_csv(path, index=False)
            paths.append(path)
    return paths


def write_demo_features(out_path: str | Path, seed: int = 0) -> Path:
    """A ready-made multi-dimensional table, for the second UI entry point."""
    from .config import AnalysisConfig
    from .io_tables import build_spot_dataset, detect_spot_columns, load_table
    from .metrics import compute_features

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        files = write_demo_dataset(tmp, n_tracks=12, n_frames=260, replicates=2, seed=seed)
        cm = detect_spot_columns(load_table(files[0]))
        groups, reps = [], []
        for f in files:
            stem = f.stem.replace("spots_", "")
            g, r = stem.rsplit("_rep", 1)
            groups.append(g)
            reps.append(int(r))
        spots = build_spot_dataset(files, cm, groups, reps)
        cfg = AnalysisConfig()
        cfg.track.pixel_size = 1.6
        cfg.track.frame_interval = 0.05
        ds = compute_features(spots, cfg.track)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.frame.to_csv(out, index=False)
    return out


__all__ = ["PRESETS", "simulate_group", "write_demo_dataset", "write_demo_features"]
