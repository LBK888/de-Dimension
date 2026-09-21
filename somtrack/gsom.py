"""Growing SOM on a hexagonal lattice (Alahakoon, Halgamuge & Srinivasan 2000).

The fixed rectangular map is the other half of the "SOM does not separate my
groups" problem: too small and distinct behaviours are forced to share a node,
too large and every sample gets a private node so nothing clusters.  A growing
map spends nodes where the quantisation error is actually high and leaves
sparse regions coarse, so the map size stops being a hyper-parameter the user
has to guess.

The node set is irregular, so it carries explicit ``(col, row)`` coordinates;
every plotting routine in :mod:`somtrack.viz` works from those coordinates
rather than from a width x height grid, so an irregular map draws exactly like
a rectangular one.
"""

from __future__ import annotations

import math

import numpy as np

from .config import SomConfig
from .som import (SomResult, TrainHistory, bmu_two, group_purity,
                  lattice_from_coords, topographic_error)

# odd-r offset hexagonal neighbourhood
_NEI_EVEN = ((+1, 0), (-1, 0), (0, -1), (-1, -1), (0, +1), (-1, +1))
_NEI_ODD = ((+1, 0), (-1, 0), (0, -1), (+1, -1), (0, +1), (+1, +1))

_FD = 0.031          # error spread factor, per the original paper


def _neighbours(cr: tuple[int, int]) -> list[tuple[int, int]]:
    c, r = cr
    offs = _NEI_ODD if (r % 2) else _NEI_EVEN
    return [(c + dc, r + dr) for dc, dr in offs]


def train_gsom(
    X: np.ndarray,
    cfg: SomConfig,
    feature_names: list[str] | None = None,
    group_codes: np.ndarray | None = None,
    n_groups: int = 0,
    progress=None,
) -> SomResult:
    n, d = X.shape
    rng = np.random.default_rng(cfg.random_state)
    names = list(feature_names) if feature_names else [f"f{i}" for i in range(d)]

    lo, hi = X.min(axis=0), X.max(axis=0)
    span = np.where(hi - lo > 0, hi - lo, 1.0)

    # growth threshold
    sf = float(np.clip(cfg.spread_factor, 1e-3, 0.999))
    GT = -d * math.log(sf)

    # a map with more nodes than samples cannot cluster anything, so cap growth
    # at the same 5*sqrt(N) budget the fixed-size maps use, doubled
    max_nodes = int(min(cfg.max_nodes, max(9, round(10.0 * math.sqrt(n)))))

    # ---- seed: a 2 x 2 patch around the data mean ------------------------
    coords: list[tuple[int, int]] = [(0, 0), (1, 0), (0, 1), (1, 1)]
    index = {cr: i for i, cr in enumerate(coords)}
    mu = X.mean(axis=0)
    W = np.array([mu + rng.normal(scale=0.1, size=d) * span for _ in coords])
    err = np.zeros(len(coords))

    hist = TrainHistory()
    grow_epochs = max(int(cfg.epochs * 0.7), 1)
    total_epochs = cfg.epochs
    lr0 = min(cfg.learning_rate, 1.0)
    tau = max(cfg.time_constant * total_epochs, 1.0)

    for epoch in range(1, total_epochs + 1):
        growing = epoch <= grow_epochs and len(coords) < max_nodes
        sigma = max((2.0 if growing else 1.0) * math.exp(-epoch / tau), 0.6)
        lr = lr0 * math.exp(-epoch / tau) * (1.0 if growing else 0.4)

        xy = _cartesian(coords)
        dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
        H = np.exp(-(dist ** 2) / (2 * sigma * sigma))

        for i in rng.permutation(n):
            x = X[i]
            diff = W - x
            j = int(np.argmin(np.einsum("ij,ij->i", diff, diff)))
            W += (lr * H[j])[:, None] * (x - W)

            if not growing:
                continue

            err[j] += float(np.linalg.norm(x - W[j]))
            if err[j] <= GT or len(coords) >= max_nodes:
                continue

            free = [p for p in _neighbours(coords[j]) if p not in index]
            if free:
                for p in free:
                    if len(coords) >= max_nodes:
                        break
                    w_new = _init_new_weight(p, index, coords, W, lo, hi)
                    coords.append(p)
                    index[p] = len(coords) - 1
                    W = np.vstack([W, w_new])
                    err = np.append(err, 0.0)
                err[j] = GT / 2.0
                # geometry changed: recompute for the rest of this epoch
                xy = _cartesian(coords)
                dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
                H = np.exp(-(dist ** 2) / (2 * sigma * sigma))
            else:
                err[j] = GT / 2.0
                for p in _neighbours(coords[j]):
                    k = index.get(p)
                    if k is not None:
                        err[k] *= (1.0 + _FD)

        lat = lattice_from_coords(_normalised(coords))
        b1, d1, b2, d2 = bmu_two(X, W)
        hist.epochs.append(epoch)
        hist.qe.append(float(np.mean(d1)))
        hist.te.append(topographic_error(lat, b1, b2))
        hist.sigma.append(sigma)
        hist.lr.append(lr)
        hist.purity.append(
            group_purity(b1, group_codes, len(coords), n_groups)
            if group_codes is not None and n_groups else np.nan
        )
        keep = (epoch == 1 or epoch % max(cfg.record_every, 1) == 0 or epoch == total_epochs)
        # node count changes between snapshots, so store coordinates alongside
        hist.codebooks.append(W.copy() if keep else None)
        hist.bmus.append(b1.copy() if keep else None)
        hist.bmu2s.append(b2.copy() if keep else None)
        hist.bmu_d.append(d1.copy() if keep else None)
        hist.bmu2_d.append(d2.copy() if keep else None)

        if progress and (epoch % 5 == 0 or epoch == total_epochs):
            progress(epoch, total_epochs)

    lattice = lattice_from_coords(_normalised(coords))
    b1, d1, b2, d2 = bmu_two(X, W)
    hits = np.bincount(b1, minlength=len(coords)).astype(float)

    return SomResult(
        codebook=W, lattice=lattice, bmu=b1, bmu_dist=d1, bmu2=b2, bmu2_dist=d2,
        hits=hits, feature_names=names, algorithm="growing", history=hist, config=cfg,
    )


def _init_new_weight(p, index, coords, W, lo, hi) -> np.ndarray:
    """GSOM weight initialisation: extrapolate from the existing neighbours."""
    present = [index[q] for q in _neighbours(p) if q in index]
    if not present:
        return (lo + hi) / 2.0
    if len(present) == 1:
        w = W[present[0]]
        # mirror through the single neighbour's own neighbour, if there is one
        base = coords[present[0]]
        opposite = (2 * base[0] - p[0], 2 * base[1] - p[1])
        k = index.get(opposite)
        w_new = 2.0 * w - W[k] if k is not None else w.copy()
    else:
        w_new = W[present].mean(axis=0)
    return np.clip(w_new, lo, hi)


def _cartesian(coords) -> np.ndarray:
    a = np.asarray(coords, dtype=float)
    x = a[:, 0] + 0.5 * (np.asarray(coords)[:, 1] % 2)
    y = a[:, 1] * (math.sqrt(3.0) / 2.0)
    return np.column_stack([x, y])


def _normalised(coords) -> np.ndarray:
    a = np.asarray(coords, dtype=int)
    # keep row parity so the hex offset stays consistent after the shift
    shift_r = a[:, 1].min()
    if shift_r % 2:
        shift_r -= 1
    return np.column_stack([a[:, 0] - a[:, 0].min(), a[:, 1] - shift_r])


__all__ = ["train_gsom"]
