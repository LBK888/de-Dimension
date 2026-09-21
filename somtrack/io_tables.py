"""Table loading, column detection and dataset assembly.

Two entry points, matching the two ways a user can start an analysis:

* :class:`SpotDataset`     -- spot-level coordinate tables (TrackMate "Spots in
  tracks statistics", or any table with track / x / y / t columns).  Metrics are
  computed from these by :mod:`somtrack.metrics`.
* :class:`FeatureDataset`  -- sample-level multi-dimensional tables that are
  already one row per track / animal / replicate.

Both carry the same grouping vocabulary as the original macro:
``group`` (legacy *cluster* = experimental treatment) and ``replicate``
(legacy *subset* = repeat of the same treatment).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
_EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm", ".xlsb", ".ods"}


def load_table(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a CSV/TSV/Excel table, tolerating TrackMate's multi-row headers."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in _EXCEL_SUFFIXES:
        df = pd.read_excel(path, sheet_name=sheet)
    else:
        sep = "\t" if suffix in {".tsv", ".txt"} else None
        df = pd.read_csv(path, sep=sep, engine="python")

    df = _strip_trackmate_header(df)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _strip_trackmate_header(df: pd.DataFrame) -> pd.DataFrame:
    """TrackMate CSVs carry 3 extra rows (label, shortname, unit) under the header.

    They are detected as rows where the numeric columns cannot be parsed.
    """
    if len(df) < 3:
        return df
    head = df.head(3)
    non_numeric = 0
    for _, row in head.iterrows():
        vals = pd.to_numeric(row, errors="coerce")
        if vals.notna().sum() <= max(1, len(row) // 5):
            non_numeric += 1
        else:
            break
    if non_numeric == 0:
        return df
    df = df.iloc[non_numeric:].reset_index(drop=True)
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(1, int(0.9 * len(df))):
            df[col] = converted
    return df


def list_sheets(path: str | Path) -> list[str]:
    path = Path(path)
    if path.suffix.lower() not in _EXCEL_SUFFIXES:
        return []
    return pd.ExcelFile(path).sheet_names


# --------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------
@dataclass
class SpotColumnMap:
    track: str = ""
    x: str = ""
    y: str = ""
    t: str = ""                      # time or frame index
    z: str | None = None
    group: str | None = None
    replicate: str | None = None
    # optional morphology / intensity columns carried straight through
    area: str | None = None
    major: str | None = None
    minor: str | None = None
    angle: str | None = None
    intensity_mean: str | None = None
    intensity_std: str | None = None
    diameter: str | None = None

    def required_ok(self) -> bool:
        return all([self.track, self.x, self.y, self.t])

    def optional_items(self) -> dict[str, str]:
        keys = ("z", "area", "major", "minor", "angle",
                "intensity_mean", "intensity_std", "diameter")
        return {k: getattr(self, k) for k in keys if getattr(self, k)}


_PATTERNS: dict[str, list[str]] = {
    "track": [r"^track_?id$", r"^track$", r"^trackid$", r"^track ?#$", r"^id_?track$", r"track"],
    "x": [r"^position_?x$", r"^x$", r"^x_?\[?[a-zµ]*\]?$", r"^centroid_?x$", r"^pos_?x$", r"^xm?$"],
    "y": [r"^position_?y$", r"^y$", r"^y_?\[?[a-zµ]*\]?$", r"^centroid_?y$", r"^pos_?y$", r"^ym?$"],
    "t": [r"^position_?t$", r"^frame$", r"^t$", r"^time$", r"^slice$", r"^timepoint$"],
    "z": [r"^position_?z$", r"^z$", r"^est_?z$"],
    "group": [r"^group$", r"^cluster$", r"^treatment$", r"^condition$", r"^cond$", r"^class$"],
    "replicate": [r"^replicate$", r"^subset$", r"^rep$", r"^batch$", r"^movie$", r"^well$"],
    "area": [r"^area$", r"^avg_?area$"],
    "major": [r"^ellipse_?major$", r"^major$", r"^length$"],
    "minor": [r"^ellipse_?minor$", r"^minor$", r"^width$"],
    "angle": [r"^ellipse_?(theta|angle)$", r"^angle_?p$", r"^orientation$"],
    "intensity_mean": [r"^mean_?intensity(_ch\d)?$", r"^mean_?intnsty$", r"^mean$"],
    "intensity_std": [r"^std_?intensity(_ch\d)?$", r"^stddev$", r"^std_?intnsty$"],
    "diameter": [r"^estimated_?diameter$", r"^radius$", r"^diameter$"],
}


def _match(columns: list[str], patterns: list[str]) -> str | None:
    lowered = {c: c.strip().lower().replace(" ", "_") for c in columns}
    for pat in patterns:
        rx = re.compile(pat)
        for original, low in lowered.items():
            if rx.match(low):
                return original
    return None


def detect_spot_columns(df: pd.DataFrame) -> SpotColumnMap:
    """Best-effort guess of the coordinate-table schema."""
    cols = list(df.columns)
    cm = SpotColumnMap()
    for fname, pats in _PATTERNS.items():
        found = _match(cols, pats)
        if found is not None:
            setattr(cm, fname, found)
    # a bare "ID" column is a spot id, never a track id
    if cm.track and cm.track.strip().lower() == "id":
        cm.track = ""
    return cm


def guess_group_replicate(stem: str) -> tuple[str, int]:
    """Split a file stem such as ``spots_drug_high_rep2`` into group and replicate."""
    s = stem
    for prefix in ("spots_", "spot_", "tracks_", "track_"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
    m = re.search(r"[_\-\s](?:rep|r|replicate|batch)[_\-\s]?(\d+)$", s, re.I)
    if m:
        return s[: m.start()], int(m.group(1))
    m = re.search(r"[_\-](\d+)$", s)
    if m:
        return s[: m.start()], int(m.group(1))
    return s, 1


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
@dataclass
class SpotDataset:
    """Concatenated spot tables from one or more source files."""

    frame: pd.DataFrame                    # canonical columns, see below
    columns: SpotColumnMap
    sources: list[str] = field(default_factory=list)
    extra_columns: dict[str, str] = field(default_factory=dict)

    CANON = ("uid", "group", "replicate", "track", "x", "y", "t")

    @property
    def n_tracks(self) -> int:
        return int(self.frame["uid"].nunique())

    @property
    def groups(self) -> list:
        return sorted(self.frame["group"].unique().tolist())

    def summary(self) -> pd.DataFrame:
        g = self.frame.groupby(["group", "replicate"], observed=True)
        return pd.DataFrame(
            {
                "tracks": g["uid"].nunique(),
                "spots": g.size(),
                "mean_spots_per_track": g.size() / g["uid"].nunique(),
            }
        ).reset_index()


@dataclass
class FeatureDataset:
    """One row per sample; the matrix that actually goes into the clustering."""

    frame: pd.DataFrame                    # includes id/group/replicate + features
    feature_names: list[str]
    id_column: str = "sample"
    group_column: str = "group"
    replicate_column: str = "replicate"
    units: dict[str, str] = field(default_factory=dict)
    descriptions: dict[str, str] = field(default_factory=dict)

    @property
    def X(self) -> np.ndarray:
        return self.frame[self.feature_names].to_numpy(dtype=float)

    @property
    def groups(self) -> np.ndarray:
        return self.frame[self.group_column].to_numpy()

    @property
    def replicates(self) -> np.ndarray:
        if self.replicate_column in self.frame:
            return self.frame[self.replicate_column].to_numpy()
        return np.ones(len(self.frame), dtype=int)

    @property
    def sample_ids(self) -> np.ndarray:
        return self.frame[self.id_column].to_numpy()

    def group_replicate_keys(self) -> list[tuple]:
        pairs = list(zip(self.groups.tolist(), self.replicates.tolist()))
        seen, out = set(), []
        for p in pairs:
            if p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def subset(self, mask: np.ndarray) -> "FeatureDataset":
        return FeatureDataset(
            frame=self.frame.loc[mask].reset_index(drop=True),
            feature_names=list(self.feature_names),
            id_column=self.id_column,
            group_column=self.group_column,
            replicate_column=self.replicate_column,
            units=dict(self.units),
            descriptions=dict(self.descriptions),
        )


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_spot_dataset(
    files: list[str | Path],
    column_map: SpotColumnMap,
    group_ids: list | None = None,
    replicate_ids: list | None = None,
    sheets: list[str | int] | None = None,
) -> SpotDataset:
    """Concatenate several coordinate tables into one canonical spot table.

    ``group_ids`` / ``replicate_ids`` assign a treatment and a repeat number to
    each *file* -- this replaces the macro's "run step 2 once per movie with a
    different cluster#/subset#" workflow.  If the table itself carries group and
    replicate columns those win.
    """
    frames: list[pd.DataFrame] = []
    sources: list[str] = []

    for i, f in enumerate(files):
        sheet = sheets[i] if sheets and i < len(sheets) else 0
        raw = load_table(f, sheet=sheet)
        name = Path(f).stem

        out = pd.DataFrame()
        out["track"] = raw[column_map.track].astype(str)
        out["x"] = pd.to_numeric(raw[column_map.x], errors="coerce")
        out["y"] = pd.to_numeric(raw[column_map.y], errors="coerce")
        out["t"] = pd.to_numeric(raw[column_map.t], errors="coerce")

        if column_map.group and column_map.group in raw:
            out["group"] = raw[column_map.group].to_numpy()
        else:
            out["group"] = group_ids[i] if group_ids else 1

        if column_map.replicate and column_map.replicate in raw:
            out["replicate"] = raw[column_map.replicate].to_numpy()
        else:
            out["replicate"] = replicate_ids[i] if replicate_ids else (i + 1)

        for canon, src in column_map.optional_items().items():
            if src in raw:
                out[canon] = pd.to_numeric(raw[src], errors="coerce")

        out["source"] = name
        frames.append(out)
        sources.append(str(f))

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["x", "y", "t"])

    # a track is unique per (group, replicate, source, track-id)
    df["uid"] = (
        df["group"].astype(str) + "|" + df["replicate"].astype(str)
        + "|" + df["source"].astype(str) + "|" + df["track"].astype(str)
    )
    df = df.sort_values(["uid", "t"], kind="stable").reset_index(drop=True)

    return SpotDataset(
        frame=df,
        columns=column_map,
        sources=sources,
        extra_columns=column_map.optional_items(),
    )


def build_feature_dataset(
    df: pd.DataFrame,
    feature_names: list[str],
    id_column: str | None = None,
    group_column: str | None = None,
    replicate_column: str | None = None,
) -> FeatureDataset:
    """Wrap a user-supplied multi-dimensional table."""
    work = df.copy()

    if group_column is None or group_column not in work:
        guess = _match(list(work.columns), _PATTERNS["group"])
        group_column = guess or "group"
        if group_column not in work:
            work[group_column] = 1

    if replicate_column is None or replicate_column not in work:
        guess = _match(list(work.columns), _PATTERNS["replicate"])
        replicate_column = guess or "replicate"
        if replicate_column not in work:
            work[replicate_column] = 1

    if id_column is None or id_column not in work:
        id_column = "sample"
        if id_column not in work:
            work[id_column] = [f"S{i + 1:04d}" for i in range(len(work))]

    feature_names = [f for f in feature_names
                     if f in work.columns
                     and f not in {id_column, group_column, replicate_column}]
    for f in feature_names:
        work[f] = pd.to_numeric(work[f], errors="coerce")

    keep = [id_column, group_column, replicate_column] + feature_names
    return FeatureDataset(
        frame=work[keep].reset_index(drop=True),
        feature_names=feature_names,
        id_column=id_column,
        group_column=group_column,
        replicate_column=replicate_column,
    )


__all__ = [
    "load_table",
    "guess_group_replicate",
    "list_sheets",
    "numeric_columns",
    "detect_spot_columns",
    "SpotColumnMap",
    "SpotDataset",
    "FeatureDataset",
    "build_spot_dataset",
    "build_feature_dataset",
]
