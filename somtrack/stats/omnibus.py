"""Do the groups differ at all, and in what way?

This is the test that the 2.0 pipeline had no answer for.  It offered per-metric
ANOVAs and a picture; neither answers "do these treatments differ in their
overall phenotype", which is the claim a paper usually wants to make.

Three tests are run together because they fail in different directions:

**PERMANOVA** (Anderson 2001) partitions the total distance among samples into a
between-group and a within-group part and tests the ratio by permutation.  Its
``R^2`` is the effect size: the fraction of multivariate variation the grouping
accounts for.

**PERMDISP** (Anderson 2006) tests whether the groups differ in *spread*.  This
matters because PERMANOVA rejects for either reason.  A significant PERMANOVA
with a significant PERMDISP may mean "treated animals are more variable", not
"treated animals are faster" -- a different result, and one that has been
reported as the first many times.

**The energy k-sample test** (Szekely & Rizzo 2004, 2013) is consistent against
*any* difference in distribution, so it can pick up a change in shape that
leaves both the centroid and the spread alone.

All three permute under the design that :mod:`somtrack.stats.blocks` inferred,
so a nested replicate structure does not silently inflate significance.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..i18n import Text
from .blocks import (BlockStructure, effective_permutations, permutation_p,
                     permutations)


# ==========================================================================
def distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(np.asarray(X, float), metric=metric))
    return np.nan_to_num(D, nan=0.0, posinf=0.0)


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# ==========================================================================
# PERMANOVA
# ==========================================================================
def _permanova_F(D2: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    """pseudo-F and R^2 from squared distances, without building the Gower matrix."""
    n = D2.shape[0]
    total = D2.sum() / (2.0 * n)
    within = 0.0
    for g in range(k):
        idx = np.flatnonzero(y == g)
        ng = idx.size
        if ng < 1:
            continue
        sub = D2[np.ix_(idx, idx)]
        within += sub.sum() / (2.0 * ng)
    between = total - within
    df_b, df_w = k - 1, n - k
    if df_w <= 0 or within <= 0 or total <= 0:
        return np.nan, np.nan
    F = (between / df_b) / (within / df_w)
    return float(F), float(between / total)


def permanova(D: np.ndarray, y: np.ndarray, k: int, structure: BlockStructure,
              n_permutations: int = 999, random_state: int | None = 0) -> dict:
    """Permutational MANOVA on a distance matrix (Anderson 2001)."""
    D2 = np.asarray(D, float) ** 2
    F, R2 = _permanova_F(D2, y, k)
    if not np.isfinite(F):
        return {"F": np.nan, "R2": np.nan, "p": np.nan, "n_permutations": 0}

    null = np.array([_permanova_F(D2, yp, k)[0]
                     for yp in permutations(y, structure, n_permutations, random_state)])
    null = null[np.isfinite(null)]
    eff = effective_permutations(y, structure)
    return {
        "F": F, "R2": R2,
        "p": permutation_p(F, null),
        "n_permutations": int(null.size),
        "p_resolution": float(1.0 / (min(eff, null.size) + 1)) if null.size else np.nan,
        "null": null,
    }


# ==========================================================================
# PERMDISP
# ==========================================================================
def _pcoa(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Principal coordinates, returned as (real axes, imaginary axes).

    Non-Euclidean distances give negative eigenvalues.  Following Anderson
    (2006), the imaginary axes are kept and subtracted when distances are
    recomputed, rather than being discarded as if they were rounding error.
    """
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D ** 2) @ J
    G = (G + G.T) / 2.0
    w, V = np.linalg.eigh(G)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    pos = w > 1e-10
    neg = w < -1e-10
    real = V[:, pos] * np.sqrt(w[pos])[None, :]
    imag = V[:, neg] * np.sqrt(-w[neg])[None, :] if neg.any() else np.zeros((n, 0))
    return real, imag


def permdisp(D: np.ndarray, y: np.ndarray, k: int, structure: BlockStructure,
             n_permutations: int = 999, random_state: int | None = 0,
             centre: str = "centroid") -> dict:
    """Test for homogeneity of multivariate dispersions (Anderson 2006)."""
    from scipy import stats

    real, imag = _pcoa(np.asarray(D, float))
    z = np.zeros(len(y))
    for g in range(k):
        idx = np.flatnonzero(y == g)
        if idx.size == 0:
            continue
        if centre == "median" and idx.size > 2:
            c_real = np.median(real[idx], axis=0)
            c_imag = np.median(imag[idx], axis=0) if imag.shape[1] else imag[:0]
        else:
            c_real = real[idx].mean(axis=0)
            c_imag = imag[idx].mean(axis=0) if imag.shape[1] else imag[:0]
        d2 = ((real[idx] - c_real) ** 2).sum(axis=1)
        if imag.shape[1]:
            d2 = d2 - ((imag[idx] - c_imag) ** 2).sum(axis=1)
        z[idx] = np.sqrt(np.clip(d2, 0, None))

    samples = [z[y == g] for g in range(k)]
    usable = [s for s in samples if s.size >= 2]
    if len(usable) < 2:
        return {"F": np.nan, "p": np.nan, "n_permutations": 0,
                "distance_to_centroid": z, "group_mean": np.full(k, np.nan)}

    F_obs = float(stats.f_oneway(*usable).statistic)
    null = []
    for yp in permutations(y, structure, n_permutations, random_state):
        parts = [z[yp == g] for g in range(k)]
        parts = [p for p in parts if p.size >= 2]
        if len(parts) >= 2:
            try:
                null.append(float(stats.f_oneway(*parts).statistic))
            except Exception:
                pass
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]

    return {
        "F": F_obs,
        "p": permutation_p(F_obs, null),
        "n_permutations": int(null.size),
        "distance_to_centroid": z,
        "group_mean": np.array([s.mean() if s.size else np.nan for s in samples]),
        "group_sd": np.array([s.std(ddof=1) if s.size > 1 else np.nan
                              for s in samples]),
        "null": null,
    }


# ==========================================================================
# Energy k-sample test
# ==========================================================================
def _energy_statistic(D: np.ndarray, y: np.ndarray, k: int) -> float:
    """Szekely-Rizzo k-sample energy statistic from a distance matrix."""
    idx = [np.flatnonzero(y == g) for g in range(k)]
    means = np.zeros((k, k))
    for a in range(k):
        for b in range(a, k):
            if idx[a].size == 0 or idx[b].size == 0:
                continue
            block = D[np.ix_(idx[a], idx[b])]
            means[a, b] = means[b, a] = float(block.mean())

    total = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            na, nb = idx[a].size, idx[b].size
            if na == 0 or nb == 0:
                continue
            e = 2 * means[a, b] - means[a, a] - means[b, b]
            total += (na * nb) / (na + nb) * e
    return float(total)


def energy_test(D: np.ndarray, y: np.ndarray, k: int, structure: BlockStructure,
                n_permutations: int = 999, random_state: int | None = 0) -> dict:
    """Distribution-free k-sample test sensitive to location, scale *and* shape."""
    D = np.asarray(D, float)
    obs = _energy_statistic(D, y, k)
    null = np.array([_energy_statistic(D, yp, k)
                     for yp in permutations(y, structure, n_permutations, random_state)])
    null = null[np.isfinite(null)]
    return {"statistic": obs, "p": permutation_p(obs, null),
            "n_permutations": int(null.size), "null": null}


# ==========================================================================
# Mahalanobis distance between centroids
# ==========================================================================
def mahalanobis_centroids(X: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """Pairwise Mahalanobis D between group centroids, pooled shrinkage covariance.

    With fewer samples than metrics the sample covariance is singular, so the
    pooled within-group covariance is shrunk towards a diagonal target
    (Ledoit & Wolf 2004) before inversion.
    """
    from sklearn.covariance import LedoitWolf

    centred = []
    centroids = []
    for g in range(k):
        idx = np.flatnonzero(y == g)
        if idx.size == 0:
            centroids.append(np.full(X.shape[1], np.nan))
            continue
        mu = X[idx].mean(axis=0)
        centroids.append(mu)
        centred.append(X[idx] - mu)

    if not centred:
        return np.full((k, k), np.nan)
    W = np.vstack(centred)
    try:
        cov = LedoitWolf(assume_centered=True).fit(W)
        P = cov.precision_
    except Exception:
        P = np.linalg.pinv(np.cov(W, rowvar=False) + 1e-8 * np.eye(X.shape[1]))

    out = np.zeros((k, k))
    for a in range(k):
        for b in range(a + 1, k):
            d = centroids[a] - centroids[b]
            if not np.all(np.isfinite(d)):
                out[a, b] = out[b, a] = np.nan
                continue
            out[a, b] = out[b, a] = float(np.sqrt(max(d @ P @ d, 0.0)))
    return out


# ==========================================================================
# Everything at once
# ==========================================================================
@dataclass
class SeparationReport:
    metric: str = "euclidean"
    n_samples: int = 0
    n_features: int = 0
    group_values: list = field(default_factory=list)

    permanova_F: float = np.nan
    permanova_R2: float = np.nan
    permanova_p: float = np.nan

    permdisp_F: float = np.nan
    permdisp_p: float = np.nan
    dispersion_by_group: np.ndarray = field(default_factory=lambda: np.zeros(0))
    distance_to_centroid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    group_codes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    energy_statistic: float = np.nan
    energy_p: float = np.nan

    mahalanobis: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)

    n_permutations: int = 0
    p_resolution: float = np.nan
    design: str = "none"
    design_note: str = ""
    citations: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def groups_differ(self) -> bool:
        return np.isfinite(self.permanova_p) and self.permanova_p < 0.05

    @property
    def dispersion_differs(self) -> bool:
        return np.isfinite(self.permdisp_p) and self.permdisp_p < 0.05

    @property
    def interpretation(self) -> str:
        """The sentence that stops a dispersion effect being read as a shift."""
        if not np.isfinite(self.permanova_p):
            return Text("The multivariate test could not be run.")
        if not self.groups_differ:
            return Text("The groups did not differ detectably in their overall "
                        "multivariate phenotype.")
        if self.dispersion_differs:
            return Text("The groups differ, but they also differ in how variable "
                        "they are, so at least part of the separation is a "
                        "difference in spread rather than a shift in the group "
                        "average. Read a result like this as 'these animals are "
                        "more variable', not 'these animals are faster', unless the "
                        "per-metric effects say otherwise.")
        return Text("The groups differ in their multivariate average, and their "
                    "within-group variability is comparable, so this is a genuine "
                    "shift in phenotype rather than a change in variability.")

    def table(self) -> pd.DataFrame:
        rows = [
            {"test": "PERMANOVA", "statistic": self.permanova_F,
             "effect_size": self.permanova_R2, "effect_size_name": "R2",
             "p": self.permanova_p, "asks": "do the group averages differ?"},
            {"test": "PERMDISP", "statistic": self.permdisp_F,
             "effect_size": np.nan, "effect_size_name": "",
             "p": self.permdisp_p, "asks": "do the groups differ in spread?"},
            {"test": "energy k-sample", "statistic": self.energy_statistic,
             "effect_size": np.nan, "effect_size_name": "",
             "p": self.energy_p, "asks": "do the distributions differ at all?"},
        ]
        df = pd.DataFrame(rows)
        df["n_permutations"] = self.n_permutations
        df["distance"] = self.metric
        df["design"] = self.design
        return df


def analyse_separation(
    X: np.ndarray,
    y: np.ndarray,
    group_values: list,
    structure: BlockStructure,
    *,
    metric: str = "euclidean",
    n_permutations: int = 999,
    random_state: int | None = 0,
    run_permanova: bool = True,
    run_permdisp: bool = True,
    run_energy: bool = True,
    dispersion_centre: str = "centroid",
) -> SeparationReport:
    """PERMANOVA, PERMDISP and the energy test on one feature matrix."""
    k = len(group_values)
    rep = SeparationReport(metric=metric, n_samples=X.shape[0],
                           n_features=X.shape[1], group_values=list(group_values),
                           group_codes=np.asarray(y, int),
                           design=structure.design, design_note=structure.note,
                           n_permutations=n_permutations)
    if k < 2:
        rep.notes.append(Text("At least two groups are needed for a separation test."))
        return rep

    D = distance_matrix(X, metric)
    cites: list[str] = []

    if run_permanova:
        out = permanova(D, y, k, structure, n_permutations, random_state)
        rep.permanova_F = out["F"]
        rep.permanova_R2 = out["R2"]
        rep.permanova_p = out["p"]
        rep.p_resolution = out.get("p_resolution", np.nan)
        cites.append("anderson2001")

    if run_permdisp:
        out = permdisp(D, y, k, structure, n_permutations, random_state,
                       dispersion_centre)
        rep.permdisp_F = out["F"]
        rep.permdisp_p = out["p"]
        rep.dispersion_by_group = out.get("group_mean", np.zeros(0))
        rep.distance_to_centroid = out.get("distance_to_centroid", np.zeros(0))
        cites.append("anderson2006")

    if run_energy:
        out = energy_test(D, y, k, structure, n_permutations, random_state)
        rep.energy_statistic = out["statistic"]
        rep.energy_p = out["p"]
        cites.append("szekely2013")

    rep.mahalanobis = mahalanobis_centroids(X, y, k)
    cites += ["mahalanobis1936", "ledoit2004"]
    if structure.blocked:
        cites.append("anderson2003")

    rep.pairwise = _pairwise_separation(D, X, y, group_values, structure,
                                        n_permutations, random_state, rep.mahalanobis)
    rep.citations = tuple(dict.fromkeys(cites))
    return rep


def _pairwise_separation(D, X, y, group_values, structure, n_perm, random_state,
                         maha) -> pd.DataFrame:
    """PERMANOVA on every pair of groups, FDR-corrected across the pairs."""
    k = len(group_values)
    if k < 2:
        return pd.DataFrame()

    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            m = (y == a) | (y == b)
            if m.sum() < 4:
                continue
            ysub = (y[m] == b).astype(int)
            sub = BlockStructure(
                structure.design,
                structure.blocks[m] if structure.blocked else None,
                structure.block_values,
                int(np.unique(structure.blocks[m]).size) if structure.blocked else 0,
                int(m.sum()), structure.note,
            )
            out = permanova(D[np.ix_(np.flatnonzero(m), np.flatnonzero(m))],
                            ysub, 2, sub, n_perm, random_state)
            rows.append({
                "group_a": group_values[a], "group_b": group_values[b],
                "n_a": int((y == a).sum()), "n_b": int((y == b).sum()),
                "pseudo_F": out["F"], "R2": out["R2"], "p": out["p"],
                "mahalanobis_D": (maha[a, b] if maha.size else np.nan),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = _bh_fdr(df["p"].to_numpy(float))
        df = df.sort_values("R2", ascending=False).reset_index(drop=True)
    return df


__all__ = [
    "SeparationReport", "analyse_separation", "permanova", "permdisp",
    "energy_test", "mahalanobis_centroids", "distance_matrix",
]
