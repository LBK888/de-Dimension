"""Linear and distance-based projections: PCA and multidimensional scaling.

PCA is the baseline every reviewer already trusts: its axes are real linear
combinations, its arrows are real loadings, and the variance it explains is
stated rather than implied.

MDS earns its place for a different reason.  It is the only projection in the
program whose picture and whose statistics are built on the *same* distance
matrix, so when PERMANOVA reports a group difference on, say, correlation
distance, the MDS drawn on correlation distance is a faithful picture of the
quantity that was tested.  Everything else is an approximation the eye has to
be warned about.
"""

from __future__ import annotations

import numpy as np

from ..projection import DataContext, ProjectionResult, axis_correlation, two_columns
from ..registry import MethodSpec, ParamSpec, register

DISTANCES = ("euclidean", "correlation", "cityblock", "cosine")


# ==========================================================================
def distance_matrix(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Pairwise distances, with the same metric names the statistics layer uses."""
    from scipy.spatial.distance import pdist, squareform

    D = squareform(pdist(X, metric=metric))
    return np.nan_to_num(D, nan=0.0, posinf=0.0)


# ==========================================================================
# PCA
# ==========================================================================
def _run_pca(ctx: DataContext, p: dict) -> ProjectionResult:
    from sklearn.decomposition import PCA

    X = ctx.X
    k = int(min(max(ctx.n_components, 2), X.shape[1], X.shape[0]))
    pca = PCA(n_components=k, random_state=ctx.random_state)
    Y = two_columns(pca.fit_transform(X))

    ev = pca.explained_variance_ratio_
    comp = pca.components_[:2]
    sd = np.sqrt(pca.explained_variance_[:2])
    xsd = X.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        load = (comp * sd[:, None]).T / np.where(xsd > 0, xsd, 1.0)[:, None]

    labels = (f"PC1 ({ev[0] * 100:.1f}%)",
              f"PC2 ({ev[1] * 100:.1f}%)" if ev.size > 1 else "PC2")
    return ProjectionResult(
        name="PCA", coords=Y, feature_axes=np.nan_to_num(load),
        feature_names=list(ctx.feature_names), axis_labels=labels,
        explained_variance=ev, axes_are_loadings=True,
    )


register(MethodSpec(
    key="pca", label="PCA", family="linear", run=_run_pca,
    citations=("hotelling1933",),
    summary="Linear, deterministic, directly interpretable. Start here.",
    detail=("principal component analysis on the scaled feature matrix, with "
            "component loadings reported in correlation units"),
    params=(),
    min_samples=3,
))


# ==========================================================================
# MDS / PCoA
# ==========================================================================
def _run_mds(ctx: DataContext, p: dict) -> ProjectionResult:
    metric = str(p.get("metric", "euclidean"))
    kind = str(p.get("kind", "classical"))
    D = distance_matrix(ctx.X, metric)
    n = D.shape[0]

    if kind == "smacof":
        from sklearn.manifold import MDS

        mds = MDS(n_components=2, dissimilarity="precomputed", n_init=4,
                  random_state=ctx.random_state, normalized_stress=False)
        Y = two_columns(mds.fit_transform(D))
        stress = float(mds.stress_)
        denom = float((D[np.triu_indices(n, 1)] ** 2).sum())
        # Kruskal stress-1: 0 is perfect, < 0.1 is usually called a good fit
        s1 = float(np.sqrt(stress / denom)) if denom > 0 else np.nan
        ev = None
        labels = (f"MDS 1 (stress-1 = {s1:.3f})", "MDS 2")
    else:
        # classical scaling: double-centre, then take the top two eigenvectors
        J = np.eye(n) - np.ones((n, n)) / n
        B = -0.5 * J @ (D ** 2) @ J
        B = (B + B.T) / 2.0
        w, V = np.linalg.eigh(B)
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]
        pos = np.clip(w, 0, None)
        Y = two_columns(V[:, :2] * np.sqrt(pos[:2])[None, :])
        total = pos.sum()
        ev = (pos[:2] / total) if total > 0 else np.zeros(2)
        labels = (f"PCo1 ({ev[0] * 100:.1f}%)",
                  f"PCo2 ({ev[1] * 100:.1f}%)" if ev.size > 1 else "PCo2")

    return ProjectionResult(
        name="MDS" if kind == "smacof" else "PCoA",
        coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names), axis_labels=labels,
        explained_variance=np.asarray(ev) if ev is not None else None,
    )


register(MethodSpec(
    key="mds", label="MDS / PCoA", family="linear", run=_run_mds,
    citations=("torgerson1952", "gower1966", "kruskal1964"),
    summary="The only projection whose distances are the ones the statistics test.",
    detail=("multidimensional scaling of the sample distance matrix, using the "
            "same distance as the multivariate tests"),
    params=(
        ParamSpec("metric", "choice", default="euclidean", choices=DISTANCES,
                  tier="common", label="Distance",
                  help="How distance between two samples is measured. Keep this "
                       "the same as the statistics setting so the picture and "
                       "the test describe the same thing."),
        ParamSpec("kind", "choice", default="classical",
                  choices=("classical", "smacof"), tier="advanced",
                  label="Algorithm",
                  help="Classical scaling is exact and deterministic. SMACOF "
                       "fits the distances iteratively and reports a stress value."),
    ),
    min_samples=4,
))


__all__ = ["distance_matrix", "DISTANCES"]
