"""Self-organising maps: lattice, four training algorithms, quality metrics.

Algorithms
----------
``online``      Faithful, corrected port of the macro's sequential Kohonen rule.
``batch``       Vectorised batch SOM (Kohonen 1999).  Deterministic and roughly
                two orders of magnitude faster; the default.
``supervised``  XY-fused SOM (Melssen, Wehrens & Buydens 2006).  The class label
                is carried as a second data block with weight ``label_weight``,
                so the map is pulled towards a layout that separates the
                experimental groups.  This is the direct answer to "plain SOM
                does not separate my groups when the differences are subtle".
``relevance``   Batch SOM plus GRLVQ feature-relevance learning (Hammer &
                Villmann 2002).  Features that fail to discriminate the groups
                have their weight driven down, so a handful of informative
                metrics are no longer drowned out by twenty uninformative ones.

Lattice
-------
The macro drew a pointy-top hexagonal grid but trained on raw ``(col, row)``
offsets, so the neighbourhood the algorithm used did not match the picture.
Here both use the same Cartesian embedding of the hex lattice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .config import SomConfig

SQRT3_2 = math.sqrt(3.0) / 2.0


# ==========================================================================
# Lattice
# ==========================================================================
@dataclass
class Lattice:
    """Node positions and the pairwise node-distance matrix."""

    coords: np.ndarray          # (n_nodes, 2) int  (col, row)
    xy: np.ndarray              # (n_nodes, 2) float Cartesian centres
    width: int
    height: int
    topology: str = "hex"
    toroidal: bool = False
    dist: np.ndarray = field(default=None, repr=False)   # (n_nodes, n_nodes)

    @property
    def n_nodes(self) -> int:
        return self.coords.shape[0]

    def index_of(self, col: int, row: int) -> int:
        m = np.flatnonzero((self.coords[:, 0] == col) & (self.coords[:, 1] == row))
        return int(m[0]) if m.size else -1

    def neighbourhood(self, sigma: float) -> np.ndarray:
        """Gaussian neighbourhood matrix h(j, k) for the current radius."""
        s = max(float(sigma), 1e-6)
        return np.exp(-(self.dist ** 2) / (2.0 * s * s))

    def neighbourhood_named(self, sigma: float, kind: str) -> np.ndarray:
        s = max(float(sigma), 1e-6)
        if kind == "bubble":
            return (self.dist <= s).astype(float)
        if kind == "mexican_hat":
            r = (self.dist / s) ** 2
            return (1.0 - r) * np.exp(-r / 2.0)
        return np.exp(-(self.dist ** 2) / (2.0 * s * s))


def hex_cartesian(coords: np.ndarray) -> np.ndarray:
    """Pointy-top, odd-row-shifted hex centres, in units of one column spacing.

    Matches the drawing geometry of the original macro exactly:
    column pitch 2*sin(60 deg)*R, row pitch 1.5*R, odd rows shifted half a column.
    """
    col = coords[:, 0].astype(float)
    row = coords[:, 1].astype(float)
    x = col + 0.5 * (coords[:, 1] % 2)
    y = row * SQRT3_2
    return np.column_stack([x, y])


def make_lattice(width: int, height: int, topology: str = "hex",
                 toroidal: bool = False) -> Lattice:
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    coords = np.column_stack([cols.ravel(), rows.ravel()]).astype(int)
    xy = hex_cartesian(coords) if topology == "hex" else coords.astype(float)

    if toroidal:
        span_x = float(width)
        span_y = float(height) * (SQRT3_2 if topology == "hex" else 1.0)
        dx = np.abs(xy[:, None, 0] - xy[None, :, 0])
        dy = np.abs(xy[:, None, 1] - xy[None, :, 1])
        dx = np.minimum(dx, span_x - dx)
        dy = np.minimum(dy, span_y - dy)
        dist = np.hypot(dx, dy)
    else:
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1))

    return Lattice(coords=coords, xy=xy, width=width, height=height,
                   topology=topology, toroidal=toroidal, dist=dist)


def lattice_from_coords(coords: np.ndarray, topology: str = "hex") -> Lattice:
    """Lattice for an irregular node set (used by the growing SOM)."""
    coords = np.asarray(coords, dtype=int)
    xy = hex_cartesian(coords) if topology == "hex" else coords.astype(float)
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))
    return Lattice(
        coords=coords, xy=xy,
        width=int(coords[:, 0].max() - coords[:, 0].min() + 1),
        height=int(coords[:, 1].max() - coords[:, 1].min() + 1),
        topology=topology, toroidal=False, dist=dist,
    )


# ==========================================================================
# BMU search
# ==========================================================================
def bmu_two(X: np.ndarray, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Best and second-best matching unit for every sample.

    Uses the expanded squared-distance identity, so the whole search is one
    BLAS matrix product instead of the macro's triple Python loop.
    """
    d2 = _sqdist(X, W)
    if W.shape[0] == 1:
        b = np.zeros(X.shape[0], dtype=int)
        db = np.sqrt(np.maximum(d2[:, 0], 0.0))
        return b, db, b.copy(), db.copy()
    part = np.argpartition(d2, 1, axis=1)[:, :2]
    rows = np.arange(X.shape[0])
    pair = d2[rows[:, None], part]
    order = np.argsort(pair, axis=1)
    b1 = part[rows, order[:, 0]]
    b2 = part[rows, order[:, 1]]
    return (b1, np.sqrt(np.maximum(d2[rows, b1], 0.0)),
            b2, np.sqrt(np.maximum(d2[rows, b2], 0.0)))


def _sqdist(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    xn = np.einsum("ij,ij->i", X, X)[:, None]
    wn = np.einsum("ij,ij->i", W, W)[None, :]
    return np.maximum(xn + wn - 2.0 * (X @ W.T), 0.0)


# ==========================================================================
# Initialisation
# ==========================================================================
def init_codebook(X: np.ndarray, lattice: Lattice, method: str,
                  rng: np.random.Generator) -> np.ndarray:
    n_nodes, n_f = lattice.n_nodes, X.shape[1]

    if method == "random":
        lo, hi = X.min(axis=0), X.max(axis=0)
        return lo + rng.random((n_nodes, n_f)) * (hi - lo)

    if method == "sample":
        idx = rng.choice(X.shape[0], size=n_nodes, replace=X.shape[0] < n_nodes)
        return X[idx].copy() + rng.normal(scale=1e-6, size=(n_nodes, n_f))

    # PCA plane initialisation: deterministic, and starts the map already
    # aligned with the two directions of largest variance, which removes most
    # of the run-to-run variability the macro suffered from.
    Xc = X - X.mean(axis=0)
    try:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except np.linalg.LinAlgError:
        return init_codebook(X, lattice, "sample", rng)
    k = min(2, Vt.shape[0])
    comps = Vt[:k]
    sd = S[:k] / math.sqrt(max(X.shape[0] - 1, 1))

    g = lattice.xy.astype(float).copy()
    for c in range(g.shape[1]):
        span = g[:, c].max() - g[:, c].min()
        g[:, c] = (g[:, c] - g[:, c].mean()) / (span if span > 0 else 1.0)
    g *= 2.0 * math.sqrt(3.0)          # span roughly +/- 1.7 SD

    W = np.tile(X.mean(axis=0), (n_nodes, 1))
    for c in range(k):
        W += np.outer(g[:, c], comps[c] * sd[c])
    return W


# ==========================================================================
# Result containers
# ==========================================================================
@dataclass
class TrainHistory:
    epochs: list[int] = field(default_factory=list)
    qe: list[float] = field(default_factory=list)          # quantisation error
    te: list[float] = field(default_factory=list)          # topographic error
    purity: list[float] = field(default_factory=list)      # hit-weighted group purity
    sigma: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    codebooks: list[np.ndarray] = field(default_factory=list)
    bmus: list[np.ndarray] = field(default_factory=list)
    bmu2s: list[np.ndarray] = field(default_factory=list)
    bmu_d: list[np.ndarray] = field(default_factory=list)
    bmu2_d: list[np.ndarray] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.epochs)


@dataclass
class SomResult:
    codebook: np.ndarray                 # (n_nodes, n_features), scaled space
    lattice: Lattice
    bmu: np.ndarray
    bmu_dist: np.ndarray
    bmu2: np.ndarray
    bmu2_dist: np.ndarray
    hits: np.ndarray
    feature_names: list[str]
    algorithm: str
    history: TrainHistory
    label_codebook: np.ndarray | None = None    # supervised: (n_nodes, n_groups)
    relevance: np.ndarray | None = None         # relevance SOM: (n_features,)
    config: SomConfig | None = None

    @property
    def n_nodes(self) -> int:
        return self.codebook.shape[0]

    @property
    def width(self) -> int:
        return self.lattice.width

    @property
    def height(self) -> int:
        return self.lattice.height

    def hit_matrix(self, codes: np.ndarray, n_classes: int) -> np.ndarray:
        """(n_nodes, n_classes) count of samples of each class per node."""
        M = np.zeros((self.n_nodes, n_classes), dtype=float)
        np.add.at(M, (self.bmu, codes), 1.0)
        return M

    def u_matrix(self) -> np.ndarray:
        """Mean codebook distance to immediate lattice neighbours, per node."""
        L = self.lattice
        # immediate neighbours: the smallest non-zero lattice distance
        nz = L.dist[L.dist > 1e-9]
        step = np.min(nz) if nz.size else 1.0
        adj = (L.dist > 1e-9) & (L.dist <= step * 1.05)
        D = np.sqrt(_sqdist(self.codebook, self.codebook))
        out = np.full(self.n_nodes, np.nan)
        for i in range(self.n_nodes):
            nb = np.flatnonzero(adj[i])
            if nb.size:
                out[i] = float(D[i, nb].mean())
        return out


# ==========================================================================
# Training
# ==========================================================================
def train_som(
    X: np.ndarray,
    cfg: SomConfig,
    feature_names: list[str] | None = None,
    group_codes: np.ndarray | None = None,
    n_groups: int = 0,
    progress=None,
) -> SomResult:
    """Dispatch to the requested SOM variant."""
    if cfg.algorithm == "growing":
        from .gsom import train_gsom

        return train_gsom(X, cfg, feature_names, group_codes, n_groups, progress)

    n, d = X.shape
    w, h = cfg.resolved_size(n)
    lattice = make_lattice(w, h, cfg.topology, cfg.toroidal)
    rng = np.random.default_rng(cfg.random_state)
    names = list(feature_names) if feature_names else [f"f{i}" for i in range(d)]

    sigma0 = cfg.sigma0 if cfg.sigma0 > 0 else max(w, h) / 2.0
    tau = max(cfg.time_constant * cfg.epochs, 1.0)

    if cfg.algorithm == "supervised":
        return _train_supervised(X, cfg, lattice, rng, names,
                                 group_codes, n_groups, sigma0, tau, progress)
    if cfg.algorithm == "relevance":
        return _train_relevance(X, cfg, lattice, rng, names,
                                group_codes, n_groups, sigma0, tau, progress)
    if cfg.algorithm == "online":
        W, hist = _fit_online(X, cfg, lattice, rng, sigma0, tau, group_codes, n_groups, progress)
    else:
        W, hist = _fit_batch(X, cfg, lattice, rng, sigma0, tau, group_codes, n_groups, progress)

    return _finalise(X, W, lattice, names, cfg.algorithm, hist, cfg)


# -------------------------------------------------------------------- batch
def _sigma_at(epoch: int, cfg: SomConfig, sigma0: float, tau: float) -> float:
    s = sigma0 * math.exp(-epoch / tau)
    return max(s, cfg.sigma_final)


def _fit_batch(X, cfg, lattice, rng, sigma0, tau, group_codes, n_groups, progress):
    W = init_codebook(X, lattice, cfg.init, rng)
    hist = TrainHistory()
    n_nodes = lattice.n_nodes

    for epoch in range(1, cfg.epochs + 1):
        sigma = _sigma_at(epoch, cfg, sigma0, tau)
        H = lattice.neighbourhood_named(sigma, cfg.neighbour_fn)

        b1, d1, b2, d2 = bmu_two(X, W)

        # accumulate per-node sums, then smooth them over the neighbourhood
        S = np.zeros((n_nodes, X.shape[1]))
        np.add.at(S, b1, X)
        cnt = np.bincount(b1, minlength=n_nodes).astype(float)

        num = H @ S
        den = (H @ cnt)[:, None]
        good = den[:, 0] > 1e-12
        W = np.where(good[:, None], num / np.where(den > 1e-12, den, 1.0), W)

        _record(hist, epoch, cfg, X, W, lattice, b1, d1, b2, d2,
                sigma, 0.0, group_codes, n_groups)
        if progress and (epoch % 5 == 0 or epoch == cfg.epochs):
            progress(epoch, cfg.epochs)
    return W, hist


# ------------------------------------------------------------------- online
def _fit_online(X, cfg, lattice, rng, sigma0, tau, group_codes, n_groups, progress):
    W = init_codebook(X, lattice, cfg.init, rng)
    hist = TrainHistory()
    n = X.shape[0]

    lr0 = cfg.learning_rate
    if lr0 > 1.0:
        # a learning rate above 1 makes the winner overshoot past the input and
        # is what made the macro's 0.8-2.4 parameter scan unstable
        lr0 = 1.0

    for epoch in range(1, cfg.epochs + 1):
        sigma = _sigma_at(epoch, cfg, sigma0, tau)
        lr = lr0 * math.exp(-epoch / tau)
        H = lattice.neighbourhood_named(sigma, cfg.neighbour_fn)

        for i in rng.permutation(n):
            x = X[i]
            d2 = np.einsum("ij,ij->i", W - x, W - x)
            j = int(np.argmin(d2))
            W += (lr * H[j])[:, None] * (x - W)

        b1, d1, b2, d2 = bmu_two(X, W)
        _record(hist, epoch, cfg, X, W, lattice, b1, d1, b2, d2,
                sigma, lr, group_codes, n_groups)
        if progress and (epoch % 5 == 0 or epoch == cfg.epochs):
            progress(epoch, cfg.epochs)
    return W, hist


# --------------------------------------------------------------- supervised
def _train_supervised(X, cfg, lattice, rng, names, group_codes, n_groups,
                      sigma0, tau, progress):
    """XY-fused SOM: concatenate a weighted one-hot label block to the data."""
    if group_codes is None or n_groups < 2:
        raise ValueError("Supervised SOM needs at least two experimental groups.")

    Y = np.zeros((X.shape[0], n_groups))
    Y[np.arange(X.shape[0]), group_codes] = 1.0

    a = float(np.clip(cfg.label_weight, 0.0, 0.999))
    # scale each block so its contribution to the fused distance is (1-a) : a,
    # independent of how many columns each block has
    wx = math.sqrt((1.0 - a) / max(X.shape[1], 1))
    wy = math.sqrt(a / max(n_groups, 1))
    Z = np.hstack([X * wx, Y * wy])

    sub = SomConfig(**{**cfg.__dict__, "algorithm": "batch"})
    Wz, hist = _fit_batch(Z, sub, lattice, rng, sigma0, tau, group_codes, n_groups, progress)

    nx = X.shape[1]
    W = Wz[:, :nx] / wx
    label_cb = Wz[:, nx:] / wy if wy > 0 else None
    if label_cb is not None:
        s = label_cb.sum(axis=1, keepdims=True)
        label_cb = np.where(s > 1e-12, label_cb / np.where(s > 1e-12, s, 1.0), np.nan)

    # history codebooks were stored in fused space: project them back
    hist.codebooks = [None if c is None else c[:, :nx] / wx for c in hist.codebooks]

    res = _finalise(X, W, lattice, names, "supervised", hist, cfg)
    res.label_codebook = label_cb
    return res


# ---------------------------------------------------------------- relevance
def _train_relevance(X, cfg, lattice, rng, names, group_codes, n_groups,
                     sigma0, tau, progress):
    """Batch SOM interleaved with GRLVQ relevance updates on the features."""
    if group_codes is None or n_groups < 2:
        raise ValueError("Relevance SOM needs at least two experimental groups.")

    n, d = X.shape
    lam = np.ones(d) / d
    W = init_codebook(X, lattice, cfg.init, rng)
    hist = TrainHistory()
    n_nodes = lattice.n_nodes

    for epoch in range(1, cfg.epochs + 1):
        sigma = _sigma_at(epoch, cfg, sigma0, tau)
        H = lattice.neighbourhood_named(sigma, cfg.neighbour_fn)
        sq = np.sqrt(lam * d)                      # metric-weighted space

        b1, d1, b2, d2 = bmu_two(X * sq, W * sq)

        S = np.zeros((n_nodes, d))
        np.add.at(S, b1, X)
        cnt = np.bincount(b1, minlength=n_nodes).astype(float)
        num, den = H @ S, (H @ cnt)[:, None]
        good = den[:, 0] > 1e-12
        W = np.where(good[:, None], num / np.where(den > 1e-12, den, 1.0), W)

        # ---- GRLVQ step -------------------------------------------------
        node_class = _node_majority_class(b1, group_codes, n_nodes, n_groups)
        known = node_class >= 0
        if known.sum() >= 2 and len(np.unique(node_class[known])) >= 2:
            lam = _grlvq_update(X, W, lam, group_codes, node_class, known,
                                cfg.relevance_lr, cfg.relevance_floor)

        _record(hist, epoch, cfg, X, W, lattice, b1, d1, b2, d2,
                sigma, 0.0, group_codes, n_groups)
        if progress and (epoch % 5 == 0 or epoch == cfg.epochs):
            progress(epoch, cfg.epochs)

    res = _finalise(X * np.sqrt(lam * d), W * np.sqrt(lam * d), lattice, names,
                    "relevance", hist, cfg)
    res.codebook = W                     # report the codebook in data space
    res.relevance = lam
    return res


def _node_majority_class(bmu, codes, n_nodes, n_groups) -> np.ndarray:
    M = np.zeros((n_nodes, n_groups))
    np.add.at(M, (bmu, codes), 1.0)
    tot = M.sum(axis=1)
    out = np.where(tot > 0, M.argmax(axis=1), -1)
    return out.astype(int)


def _grlvq_update(X, W, lam, codes, node_class, known, lr, floor) -> np.ndarray:
    d = X.shape[1]
    Wk = W[known]
    cls = node_class[known]
    diff2 = (X[:, None, :] - Wk[None, :, :]) ** 2          # (n, k, d)
    dist = diff2 @ lam                                      # (n, k)

    same = cls[None, :] == codes[:, None]
    big = np.inf
    d_plus = np.where(same, dist, big)
    d_minus = np.where(~same, dist, big)
    jp = np.argmin(d_plus, axis=1)
    jm = np.argmin(d_minus, axis=1)
    rows = np.arange(X.shape[0])
    dp, dm = d_plus[rows, jp], d_minus[rows, jm]
    ok = np.isfinite(dp) & np.isfinite(dm) & ((dp + dm) > 1e-12)
    if ok.sum() == 0:
        return lam

    rows, dp, dm = rows[ok], dp[ok], dm[ok]
    s = dp + dm
    mu = (dp - dm) / s
    fp = 0.25 * (1.0 - np.tanh(mu / 2.0) ** 2)              # sigmoid'(mu)

    gp = (2.0 * dm / s ** 2) * fp
    gm = (-2.0 * dp / s ** 2) * fp
    grad = (gp[:, None] * diff2[rows, jp[ok]]).sum(0) + (gm[:, None] * diff2[rows, jm[ok]]).sum(0)

    lam = lam - lr * grad / max(len(rows), 1)
    lam = np.maximum(lam, floor / d)
    return lam / lam.sum()


# ------------------------------------------------------------------- shared
def _record(hist, epoch, cfg, X, W, lattice, b1, d1, b2, d2,
            sigma, lr, group_codes, n_groups):
    hist.epochs.append(epoch)
    hist.qe.append(float(np.mean(d1)))
    hist.te.append(topographic_error(lattice, b1, b2))
    hist.sigma.append(float(sigma))
    hist.lr.append(float(lr))
    if group_codes is not None and n_groups > 0:
        hist.purity.append(group_purity(b1, group_codes, lattice.n_nodes, n_groups))
    else:
        hist.purity.append(np.nan)

    if epoch == 1 or epoch % max(cfg.record_every, 1) == 0 or epoch == cfg.epochs:
        hist.codebooks.append(W.copy())
        hist.bmus.append(b1.copy())
        hist.bmu2s.append(b2.copy())
        hist.bmu_d.append(d1.copy())
        hist.bmu2_d.append(d2.copy())
    else:
        hist.codebooks.append(None)
        hist.bmus.append(None)
        hist.bmu2s.append(None)
        hist.bmu_d.append(None)
        hist.bmu2_d.append(None)


def _finalise(X, W, lattice, names, algo, hist, cfg) -> SomResult:
    b1, d1, b2, d2 = bmu_two(X, W)
    hits = np.bincount(b1, minlength=lattice.n_nodes).astype(float)
    return SomResult(
        codebook=W, lattice=lattice, bmu=b1, bmu_dist=d1, bmu2=b2, bmu2_dist=d2,
        hits=hits, feature_names=list(names), algorithm=algo, history=hist, config=cfg,
    )


# ==========================================================================
# Quality measures
# ==========================================================================
def topographic_error(lattice: Lattice, b1: np.ndarray, b2: np.ndarray) -> float:
    """Fraction of samples whose two best units are not lattice neighbours."""
    nz = lattice.dist[lattice.dist > 1e-9]
    step = float(np.min(nz)) if nz.size else 1.0
    d = lattice.dist[b1, b2]
    return float(np.mean(d > step * 1.05))


def group_purity(bmu: np.ndarray, codes: np.ndarray, n_nodes: int, n_groups: int) -> float:
    """Hit-weighted mean purity over occupied nodes.

    Replaces the macro's ``clusteringEff``, which counted how many
    *cluster-subset* combinations were absent from a node.  That penalised
    replicates of the same treatment landing together (which is the desired
    outcome) and gave every one-sample node a perfect score, so sparse maps
    scored highest.  Weighting by hits and scoring on treatment level fixes both.
    """
    M = np.zeros((n_nodes, n_groups))
    np.add.at(M, (bmu, codes), 1.0)
    tot = M.sum(axis=1)
    occ = tot > 0
    if not occ.any():
        return np.nan
    return float(np.sum(M[occ].max(axis=1)) / np.sum(tot[occ]))


def quality_report(res: SomResult, codes: np.ndarray | None = None,
                   n_groups: int = 0) -> dict[str, float]:
    """Standard SOM quality indices plus the group-separation diagnostics."""
    out: dict[str, float] = {
        "quantisation_error": float(np.mean(res.bmu_dist)),
        "topographic_error": topographic_error(res.lattice, res.bmu, res.bmu2),
        "node_occupancy": float(np.mean(res.hits > 0)),
        "max_node_load": float(res.hits.max() / max(res.hits.sum(), 1)),
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = res.bmu_dist / res.bmu2_dist
    out["representation_ratio"] = float(np.nanmean(ratio))

    if codes is not None and n_groups > 1:
        out["group_purity"] = group_purity(res.bmu, codes, res.n_nodes, n_groups)
        # NMI is normalised for the node/group count mismatch; a raw Rand index
        # over ~50 nodes against ~4 treatments is always near zero and would be
        # read as "the map failed" when it has not.
        out["node_group_nmi"] = _nmi(res.bmu, codes)
    return out


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(b, a))


# ==========================================================================
# Hyper-parameter search (replaces the macro's file-based parallel scan)
# ==========================================================================
def scan_parameters(
    X: np.ndarray,
    base: SomConfig,
    grid: dict[str, list],
    group_codes: np.ndarray | None = None,
    n_groups: int = 0,
    n_jobs: int = -1,
    progress=None,
) -> "list[dict]":
    """Grid search over SOM hyper-parameters, in parallel across CPU cores.

    The macro asked the user to launch four copies of ImageJ by hand and
    coordinate them through a shared log file; joblib does the same job in one
    process tree with no bookkeeping.
    """
    from itertools import product

    from joblib import Parallel, delayed

    keys = list(grid)
    combos = list(product(*(grid[k] for k in keys)))

    def one(values):
        cfg = SomConfig(**base.__dict__)
        for k, v in zip(keys, values):
            setattr(cfg, k, v)
        try:
            res = train_som(X, cfg, group_codes=group_codes, n_groups=n_groups)
            row = quality_report(res, group_codes, n_groups)
        except Exception as exc:                      # keep the grid rectangular
            row = {"error": str(exc)}
        row.update({k: v for k, v in zip(keys, values)})
        return row

    rows = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one)(c) for c in combos)
    if progress:
        progress(len(rows), len(rows))
    return list(rows)


__all__ = [
    "Lattice", "make_lattice", "lattice_from_coords", "hex_cartesian",
    "SomResult", "TrainHistory", "train_som", "bmu_two", "init_codebook",
    "quality_report", "topographic_error", "group_purity", "scan_parameters",
]
