"""Feature-matrix preparation: NaN policy, scaling, pruning, weighting.

Keeps enough state to map codebook vectors back into original units, which is
what makes the SOM component planes readable as real measurements rather than
as 0-1 abstractions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PreprocessConfig
from .io_tables import FeatureDataset


@dataclass
class PreparedData:
    X: np.ndarray                       # (n_samples, n_features), scaled
    feature_names: list[str]
    groups: np.ndarray                  # original group labels, per sample
    group_codes: np.ndarray             # 0..n_groups-1
    group_values: list                  # index -> original label
    replicates: np.ndarray
    sample_ids: np.ndarray
    keep_mask: np.ndarray               # which rows of the source frame survived
    centre: np.ndarray                  # per-feature offset used by the scaler
    scale: np.ndarray                   # per-feature divisor used by the scaler
    weights: np.ndarray                 # per-feature weight applied after scaling
    scaler: str = "zscore"
    dropped_features: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: FeatureDataset | None = None

    # ------------------------------------------------------------------
    @property
    def n_samples(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    @property
    def n_groups(self) -> int:
        return len(self.group_values)

    def inverse(self, Xs: np.ndarray) -> np.ndarray:
        """Scaled space -> original measurement units."""
        w = np.where(self.weights > 0, self.weights, 1.0)
        return (Xs / w) * self.scale + self.centre

    def group_replicate_codes(self) -> tuple[np.ndarray, list[tuple]]:
        pairs = list(zip(self.groups.tolist(), self.replicates.tolist()))
        uniq: list[tuple] = []
        seen: dict[tuple, int] = {}
        codes = np.empty(len(pairs), dtype=int)
        for i, p in enumerate(pairs):
            if p not in seen:
                seen[p] = len(uniq)
                uniq.append(p)
            codes[i] = seen[p]
        return codes, uniq

    def onehot_groups(self) -> np.ndarray:
        Y = np.zeros((self.n_samples, self.n_groups), dtype=float)
        Y[np.arange(self.n_samples), self.group_codes] = 1.0
        return Y


# --------------------------------------------------------------------------
_SCALERS = ("minmax", "zscore", "robust", "rank", "none")


def prepare(
    ds: FeatureDataset,
    cfg: PreprocessConfig,
    selected: list[str] | None = None,
) -> PreparedData:
    names = list(selected) if selected else list(ds.feature_names)
    names = [n for n in names if n in ds.frame.columns]
    if not names:
        raise ValueError("No features selected.")

    frame = ds.frame
    # pandas may hand back a read-only view; every step below writes in place
    X = np.array(frame[names].to_numpy(dtype=float), dtype=float, copy=True)
    notes: list[str] = []
    dropped: list[str] = []
    keep_rows = np.ones(len(frame), dtype=bool)

    # ---- 1. NaN policy ------------------------------------------------
    col_nan = np.isnan(X).sum(axis=0)
    all_nan = col_nan == X.shape[0]
    if all_nan.any():
        dropped += [n for n, f in zip(names, all_nan) if f]
        names = [n for n, f in zip(names, all_nan) if not f]
        X = X[:, ~all_nan]
        notes.append(f"Dropped {int(all_nan.sum())} all-NaN feature(s).")

    if cfg.nan_policy == "drop_sample":
        keep_rows = ~np.isnan(X).any(axis=1)
        n_drop = int((~keep_rows).sum())
        if n_drop:
            notes.append(f"Dropped {n_drop} sample(s) containing NaN.")
        X = X[keep_rows]
    elif cfg.nan_policy == "drop_feature":
        bad = np.isnan(X).any(axis=0)
        if bad.any():
            dropped += [n for n, f in zip(names, bad) if f]
            names = [n for n, f in zip(names, bad) if not f]
            X = X[:, ~bad]
            notes.append(f"Dropped {int(bad.sum())} feature(s) containing NaN.")
    else:  # impute_median
        n_nan = int(np.isnan(X).sum())
        if n_nan:
            med = np.nanmedian(X, axis=0)
            idx = np.where(np.isnan(X))
            X[idx] = np.take(med, idx[1])
            notes.append(f"Median-imputed {n_nan} missing value(s).")

    if X.shape[0] < 3:
        raise ValueError("Fewer than 3 samples remain after the NaN policy.")

    # ---- 2. winsorise -------------------------------------------------
    if cfg.winsorise_quantile > 0:
        q = cfg.winsorise_quantile
        lo = np.nanquantile(X, q, axis=0)
        hi = np.nanquantile(X, 1 - q, axis=0)
        X = np.clip(X, lo, hi)
        notes.append(f"Winsorised to the [{q:.1%}, {1 - q:.1%}] range.")

    # ---- 3. constant features ----------------------------------------
    if cfg.drop_constant:
        spread = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
        const = spread <= 1e-12
        if const.any():
            dropped += [n for n, f in zip(names, const) if f]
            names = [n for n, f in zip(names, const) if not f]
            X = X[:, ~const]
            notes.append(f"Dropped {int(const.sum())} constant feature(s).")

    # ---- 4. scaling ---------------------------------------------------
    centre, scale, X = _scale(X, cfg.scaler)

    # ---- 5. collinearity pruning -------------------------------------
    if 0 < cfg.collinearity_threshold < 1.0 and X.shape[1] > 2:
        keep_f = _prune_collinear(X, cfg.collinearity_threshold)
        if not keep_f.all():
            removed = [n for n, f in zip(names, keep_f) if not f]
            dropped += removed
            notes.append(
                f"Pruned {len(removed)} feature(s) with |r| > "
                f"{cfg.collinearity_threshold:.2f}: {', '.join(removed[:6])}"
                + (" ..." if len(removed) > 6 else "")
            )
            names = [n for n, f in zip(names, keep_f) if f]
            X = X[:, keep_f]
            centre, scale = centre[keep_f], scale[keep_f]

    # ---- 6. per-feature weights --------------------------------------
    w = np.array([float(cfg.feature_weights.get(n, 1.0)) for n in names])
    if not np.allclose(w, 1.0):
        X = X * w
        notes.append("Applied user feature weights.")

    if X.shape[1] < 2:
        raise ValueError("Fewer than 2 usable features remain.")

    groups = ds.groups[keep_rows]
    group_values = _ordered_unique(groups)
    lookup = {v: i for i, v in enumerate(group_values)}
    group_codes = np.array([lookup[g] for g in groups], dtype=int)

    return PreparedData(
        X=np.ascontiguousarray(X, dtype=np.float64),
        feature_names=names,
        groups=groups,
        group_codes=group_codes,
        group_values=group_values,
        replicates=ds.replicates[keep_rows],
        sample_ids=ds.sample_ids[keep_rows],
        keep_mask=keep_rows,
        centre=centre,
        scale=scale,
        weights=w,
        scaler=cfg.scaler,
        dropped_features=dropped,
        notes=notes,
        source=ds,
    )


def _scale(X: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_f = X.shape[1]
    if name == "none":
        return np.zeros(n_f), np.ones(n_f), X
    if name == "minmax":
        lo = np.nanmin(X, axis=0)
        rng = np.nanmax(X, axis=0) - lo
        rng[rng <= 0] = 1.0
        return lo, rng, (X - lo) / rng
    if name == "robust":
        med = np.nanmedian(X, axis=0)
        q1, q3 = np.nanquantile(X, 0.25, axis=0), np.nanquantile(X, 0.75, axis=0)
        iqr = q3 - q1
        iqr[iqr <= 0] = 1.0
        return med, iqr, (X - med) / iqr
    if name == "rank":
        from scipy.stats import rankdata

        R = np.apply_along_axis(rankdata, 0, X)
        R = (R - 0.5) / R.shape[0]
        return np.zeros(n_f), np.ones(n_f), R
    # zscore
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd[sd <= 0] = 1.0
    return mu, sd, (X - mu) / sd


def _prune_collinear(X: np.ndarray, thr: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C)
    n = C.shape[0]
    keep = np.ones(n, dtype=bool)
    # drop the later member of every over-correlated pair
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(i + 1, n):
            if keep[j] and abs(C[i, j]) > thr:
                keep[j] = False
    return keep


def _ordered_unique(a: np.ndarray) -> list:
    seen, out = set(), []
    for v in a.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    try:
        return sorted(out)
    except TypeError:
        return out


def feature_summary_table(prep: PreparedData) -> pd.DataFrame:
    """Per-feature descriptive statistics in original units, for the report."""
    src = prep.source
    rows = []
    for i, name in enumerate(prep.feature_names):
        col = src.frame.loc[prep.keep_mask, name].to_numpy(float) if src is not None else prep.X[:, i]
        rows.append({
            "feature": name,
            "unit": (src.units.get(name, "") if src else ""),
            "n": int(np.isfinite(col).sum()),
            "mean": float(np.nanmean(col)),
            "sd": float(np.nanstd(col)),
            "median": float(np.nanmedian(col)),
            "min": float(np.nanmin(col)),
            "max": float(np.nanmax(col)),
        })
    return pd.DataFrame(rows)


__all__ = ["PreparedData", "prepare", "feature_summary_table"]
