"""How much of a projection can be believed.

Three independent questions, three answers:

*Does the picture preserve neighbourhoods?*
    The co-ranking matrix (Lee & Verleysen 2009) and the R_NX(K) curve derived
    from it.  Trustworthiness and continuity are two points on that curve;
    reporting the whole curve, and its area under a logarithmic K axis, lets two
    methods be compared with one number instead of an argument about which K to
    use.

*Which individual points can be believed?*
    A per-point reliability score in the style of scDEED (Xia, Lee & Li 2024),
    compared against a null built by scrambling the embedding.  Points whose
    score falls below the null's 5th percentile are flagged dubious.  The number
    of dubious points also makes a usable objective for a hyper-parameter scan,
    which turns "what perplexity should I use" into a question with a figure.

*Would a different random seed have told a different story?*
    Re-run, Procrustes-align (Gower 1975) and measure how far each point moved.

Everything here is O(n^2) in time and memory, which is the right trade for the
n < 1000 data sets this program is built for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ==========================================================================
# Distances and ranks
# ==========================================================================
def pairwise(X: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(
        ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    np.fill_diagonal(d, 0.0)
    return d


def _rank_matrix(D: np.ndarray) -> np.ndarray:
    """rank[i, j] = position of j in i's neighbour ordering (1 = nearest)."""
    n = D.shape[0]
    order = np.argsort(D, axis=1, kind="stable")
    rank = np.empty((n, n), dtype=np.int32)
    ar = np.arange(n, dtype=np.int32)
    for i in range(n):
        rank[i, order[i]] = ar
    return rank                       # self has rank 0


# ==========================================================================
# Co-ranking / R_NX
# ==========================================================================
@dataclass
class RankQuality:
    rnx: np.ndarray                   # R_NX(K) for K = 1 .. n-2
    auc: float                        # area under R_NX on a log K axis
    trustworthiness: float
    continuity: float
    k_used: int                       # K at which T and C were evaluated

    def at(self, k: int) -> float:
        k = int(np.clip(k, 1, self.rnx.size))
        return float(self.rnx[k - 1])


def rank_quality(X: np.ndarray, Y: np.ndarray, k: int = 0) -> RankQuality:
    """Co-ranking analysis of a projection.

    ``R_NX(K)`` is the K-ary neighbourhood agreement rescaled so that a random
    projection scores 0 and a perfect one scores 1, at every K.  Reading it as a
    curve separates "good locally, wrong globally" from the reverse, which a
    single trustworthiness number cannot do.
    """
    n = X.shape[0]
    if n < 4:
        return RankQuality(np.zeros(0), np.nan, np.nan, np.nan, 0)

    k = int(k) or max(3, min(12, n // 10))
    k = int(np.clip(k, 1, n - 2))

    rx = _rank_matrix(pairwise(X))
    ry = _rank_matrix(pairwise(Y))

    # co-ranking matrix Q[a, b]: pairs at high-D rank a+1 and low-D rank b+1
    off = ~np.eye(n, dtype=bool)
    Q = np.zeros((n - 1, n - 1), dtype=np.int64)
    np.add.at(Q, (rx[off] - 1, ry[off] - 1), 1)

    # Q_NX(K) = fraction of the K-neighbourhood that is preserved
    csum = Q.cumsum(axis=0).cumsum(axis=1)          # inclusive 2-D prefix sum
    kk = np.arange(1, n - 1)
    qnx = csum[kk - 1, kk - 1] / (kk * n)
    rnx = ((n - 1) * qnx - kk) / (n - 1 - kk)

    w = 1.0 / kk
    auc = float((rnx * w).sum() / w.sum())

    t = _trustworthiness_from_ranks(rx, ry, k, n)
    c = _trustworthiness_from_ranks(ry, rx, k, n)
    return RankQuality(rnx=rnx, auc=auc, trustworthiness=t, continuity=c, k_used=k)


def _trustworthiness_from_ranks(rank_from: np.ndarray, rank_to: np.ndarray,
                                k: int, n: int) -> float:
    """Venna & Kaski trustworthiness, computed from rank matrices we already have.

    Penalises every point that entered the K-neighbourhood in ``rank_to`` space
    by how far away it was in ``rank_from`` space.
    """
    if n <= k + 1:
        return float("nan")
    near = (rank_to >= 1) & (rank_to <= k)          # K nearest in the target space
    excess = np.where(near, np.maximum(rank_from - k, 0), 0)
    penalty = float(excess.sum())
    norm = 2.0 / (n * k * (2.0 * n - 3.0 * k - 1.0))
    return float(1.0 - norm * penalty)


# ==========================================================================
# Per-point reliability (scDEED-style)
# ==========================================================================
@dataclass
class Reliability:
    score: np.ndarray                 # (n,) higher = better preserved
    dubious: np.ndarray               # (n,) bool
    threshold: float                  # null 5th percentile
    trustworthy_threshold: float      # null 95th percentile
    n_dubious: int
    fraction_dubious: float
    n_null: int


def point_reliability(X: np.ndarray, Y: np.ndarray, *,
                      neighbour_fraction: float = 0.5,
                      n_null: int = 20,
                      alpha: float = 0.05,
                      random_state: int | None = 0) -> Reliability:
    """Per-point neighbourhood-preservation score against a scrambled null.

    For point *i* let ``E`` be its closest ``neighbour_fraction`` of points **in
    the projection**, ordered by projection distance.  The score is the Pearson
    correlation between

    1. the *original-space* distances from *i* to ``E``, in that order, and
    2. the *original-space* distances from *i* to its own closest
       ``neighbour_fraction`` of points, sorted ascending.

    If the projection kept the neighbourhood, the two vectors describe the same
    set of points in the same order and the correlation is near 1.  If the
    projection invented neighbours, sequence 1 contains points that were far
    away and the correlation drops.

    The null is built by scrambling the projection (each axis permuted
    independently), which destroys the correspondence while keeping the marginal
    geometry.  A point is *dubious* when its score falls below the null's
    ``alpha`` quantile.
    """
    n = X.shape[0]
    if n < 8:
        empty = np.full(n, np.nan)
        return Reliability(empty, np.zeros(n, bool), np.nan, np.nan, 0, np.nan, 0)

    m = int(max(2, round(neighbour_fraction * (n - 1))))
    DX = pairwise(X)

    obs = _reliability_scores(DX, pairwise(Y), m)

    rng = np.random.default_rng(random_state)
    null = []
    for _ in range(max(1, n_null)):
        Yp = np.column_stack([rng.permutation(Y[:, j]) for j in range(Y.shape[1])])
        null.append(_reliability_scores(DX, pairwise(Yp), m))
    null = np.concatenate(null)
    null = null[np.isfinite(null)]

    lo = float(np.quantile(null, alpha)) if null.size else np.nan
    hi = float(np.quantile(null, 1.0 - alpha)) if null.size else np.nan
    dub = np.isfinite(obs) & (obs < lo) if np.isfinite(lo) else np.zeros(n, bool)

    return Reliability(score=obs, dubious=dub, threshold=lo,
                       trustworthy_threshold=hi, n_dubious=int(dub.sum()),
                       fraction_dubious=float(dub.mean()), n_null=int(n_null))


def _reliability_scores(DX: np.ndarray, DY: np.ndarray, m: int) -> np.ndarray:
    n = DX.shape[0]
    out = np.full(n, np.nan)
    order_y = np.argsort(DY, axis=1, kind="stable")[:, 1:m + 1]   # drop self
    sorted_x = np.sort(DX, axis=1, kind="stable")[:, 1:m + 1]
    rows = np.arange(n)[:, None]
    picked_x = DX[rows, order_y]
    for i in range(n):
        a, b = picked_x[i], sorted_x[i]
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            continue
        out[i] = float(np.corrcoef(a, b)[0, 1])
    return out


# ==========================================================================
# Stability across seeds
# ==========================================================================
@dataclass
class Stability:
    per_point: np.ndarray             # (n,) mean displacement, in units of the
    mean: float                       #      embedding's own RMS radius
    worst: float
    n_seeds: int
    aligned: list                     # Procrustes-aligned coordinate sets


def procrustes_align(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotate/scale/translate ``target`` onto ``reference`` (Gower 1975)."""
    a = reference - reference.mean(0)
    b = target - target.mean(0)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return target.copy()
    a, b = a / na, b / nb
    u, s, vt = np.linalg.svd(b.T @ a)
    R = u @ vt
    return (b @ R) * s.sum() * na + reference.mean(0)


def seed_stability(runs: list[np.ndarray]) -> Stability:
    """Per-point spread across repeated runs, after Procrustes alignment."""
    if len(runs) < 2:
        n = len(runs[0]) if runs else 0
        return Stability(np.full(n, np.nan), np.nan, np.nan, len(runs), list(runs))

    ref = np.asarray(runs[0], float)
    aligned = [ref] + [procrustes_align(ref, np.asarray(r, float)) for r in runs[1:]]
    stack = np.stack(aligned)                            # (n_seeds, n, 2)
    centre = stack.mean(axis=0)
    disp = np.linalg.norm(stack - centre[None], axis=2).mean(axis=0)

    radius = float(np.sqrt(((ref - ref.mean(0)) ** 2).sum(1).mean()))
    scale = radius if radius > 1e-12 else 1.0
    per_point = disp / scale
    return Stability(per_point=per_point, mean=float(per_point.mean()),
                     worst=float(per_point.max()), n_seeds=len(runs),
                     aligned=aligned)


# ==========================================================================
# Cross-method agreement
# ==========================================================================
def neighbour_agreement(coords: dict[str, np.ndarray], k: int = 10) -> "pd.DataFrame":
    """Fraction of each point's K nearest neighbours shared between two methods.

    Structure that survives several projections is a property of the data;
    structure visible in one is a property of that algorithm.  This makes that
    sentence a number.
    """
    import pandas as pd

    names = list(coords)
    if not names:
        return pd.DataFrame()
    n = len(next(iter(coords.values())))
    k = int(np.clip(k, 1, max(1, n - 2)))

    nbrs = {}
    for nm, Y in coords.items():
        idx = np.argsort(pairwise(np.asarray(Y, float)), axis=1, kind="stable")[:, 1:k + 1]
        nbrs[nm] = [set(row.tolist()) for row in idx]

    M = np.eye(len(names))
    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            share = np.mean([len(x & y) / k for x, y in zip(nbrs[a], nbrs[b])])
            M[i, j] = M[j, i] = share
    return pd.DataFrame(M, index=names, columns=names)


__all__ = [
    "pairwise", "RankQuality", "rank_quality",
    "Reliability", "point_reliability",
    "Stability", "seed_stability", "procrustes_align",
    "neighbour_agreement",
]
