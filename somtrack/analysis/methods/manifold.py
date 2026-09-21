"""Non-linear projections.

All of these trade global geometry for local detail, to different degrees, and
none of them should be read as a map of distances.  The defaults here are set
for the size of data set this program is built for: the published defaults of
t-SNE, UMAP and PaCMAP were chosen for tens of thousands of points, and applying
them to two hundred animals produces a picture of the algorithm rather than a
picture of the experiment.  Every default below is therefore derived from *n*,
and the value actually used is recorded in the methods section.
"""

from __future__ import annotations

import warnings

import numpy as np

from ..projection import DataContext, ProjectionResult, axis_correlation, two_columns
from ..registry import MethodSpec, ParamSpec, has_module, register

NONLINEAR_CAVEAT = ("Distances between well-separated clusters are not meaningful; "
                    "only local neighbourhoods are.")


# --------------------------------------------------------------------------
def _suggest_perplexity(ctx: DataContext) -> float:
    """Small n needs a small perplexity: it must stay below (n - 1) / 3."""
    n = ctx.n_samples
    return float(np.clip(round(n / 5.0), 5.0, min(30.0, max(2.0, (n - 1) / 3.0))))


def _suggest_neighbours(ctx: DataContext) -> int:
    n = ctx.n_samples
    return int(np.clip(round(np.sqrt(n)), 5, min(30, max(2, n - 1))))


def _perplexity_ceiling(ctx: DataContext) -> float:
    """t-SNE clamps here anyway; declaring it keeps the scan grid honest."""
    return max(2.0, (ctx.n_samples - 1) / 3.0)


def _neighbour_ceiling(ctx: DataContext) -> int:
    return int(max(2, ctx.n_samples - 1))


# ==========================================================================
# t-SNE
# ==========================================================================
def _run_tsne(ctx: DataContext, p: dict) -> ProjectionResult:
    from sklearn.manifold import TSNE

    n = ctx.n_samples
    perp = float(min(float(p["perplexity"]), max(2.0, (n - 1) / 3.0)))
    kwargs = dict(n_components=2, perplexity=perp, init=p.get("init", "pca"),
                  learning_rate=p.get("learning_rate", "auto"),
                  random_state=ctx.random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ts = TSNE(max_iter=int(p["max_iter"]), **kwargs)
        except TypeError:                       # scikit-learn < 1.5
            ts = TSNE(n_iter=int(p["max_iter"]), **kwargs)
        Y = two_columns(ts.fit_transform(ctx.X))

    return ProjectionResult(
        name="t-SNE", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names), axis_labels=("t-SNE 1", "t-SNE 2"),
    )


register(MethodSpec(
    key="tsne", label="t-SNE", family="manifold", run=_run_tsne,
    citations=("maaten2008",), caveat=NONLINEAR_CAVEAT,
    summary="Sharpens local neighbourhoods. Cluster sizes and gaps mean nothing.",
    detail="t-distributed stochastic neighbour embedding",
    params=(
        ParamSpec("perplexity", "float", default=15.0, low=2.0, high=200.0, step=1.0,
                  tier="common", label="Perplexity", suggest=_suggest_perplexity,
                  limit=_perplexity_ceiling,
                  scan_default=(5, 10, 15, 20, 30, 50),
                  help="Roughly how many neighbours each point is pulled towards. "
                       "Small values show fine detail, large values show broad "
                       "structure. It must stay below (n-1)/3."),
        ParamSpec("max_iter", "int", default=1000, low=250, high=20000,
                  tier="advanced", label="Iterations",
                  help="More iterations settle the layout; 1000 is almost always enough."),
        ParamSpec("init", "choice", default="pca", choices=("pca", "random"),
                  tier="advanced", label="Starting layout",
                  help="Starting from PCA makes the result reproducible and keeps "
                       "more of the global arrangement."),
    ),
    min_samples=10,
))


# ==========================================================================
# UMAP and densMAP
# ==========================================================================
def _umap_common(ctx: DataContext, p: dict, densmap: bool) -> ProjectionResult:
    import umap

    n = ctx.n_samples
    nn = int(min(int(p["n_neighbors"]), max(2, n - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]),
            metric=str(p.get("metric", "euclidean")),
            densmap=densmap, random_state=ctx.random_state,
        )
        Y = two_columns(np.asarray(reducer.fit_transform(ctx.X)))

    label = "densMAP" if densmap else "UMAP"
    return ProjectionResult(
        name=label, coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names),
        axis_labels=(f"{label} 1", f"{label} 2"),
    )


def _run_umap(ctx: DataContext, p: dict) -> ProjectionResult:
    return _umap_common(ctx, p, densmap=False)


def _run_densmap(ctx: DataContext, p: dict) -> ProjectionResult:
    return _umap_common(ctx, p, densmap=True)


_UMAP_PARAMS = (
    ParamSpec("n_neighbors", "int", default=15, low=2, high=200,
              tier="common", label="Neighbours", suggest=_suggest_neighbours,
              limit=_neighbour_ceiling,
              scan_default=(5, 10, 15, 25, 40),
              help="How much of the data each point looks at. Small values show "
                   "local detail, large values show the overall shape."),
    ParamSpec("min_dist", "float", default=0.1, low=0.0, high=1.0, step=0.05,
              tier="common", label="Minimum separation",
              scan_default=(0.0, 0.1, 0.25, 0.5),
              help="How tightly points may be packed. Larger values spread "
                   "clusters out and make them easier to read."),
    ParamSpec("metric", "choice", default="euclidean",
              choices=("euclidean", "correlation", "manhattan", "cosine"),
              tier="advanced", label="Distance",
              help="How distance between two samples is measured."),
)

register(MethodSpec(
    key="umap", label="UMAP", family="manifold", run=_run_umap,
    citations=("mcinnes2018",), caveat=NONLINEAR_CAVEAT,
    available=has_module("umap"), install_hint="UMAP needs:  pip install umap-learn",
    summary="Between PCA and t-SNE: keeps some global shape, sharpens local detail.",
    detail="uniform manifold approximation and projection",
    params=_UMAP_PARAMS, min_samples=10,
))

register(MethodSpec(
    key="densmap", label="densMAP", family="manifold", run=_run_densmap,
    citations=("mcinnes2018", "narayan2021"), caveat=NONLINEAR_CAVEAT,
    available=has_module("umap"), install_hint="densMAP needs:  pip install umap-learn",
    summary="UMAP that also preserves how tightly packed each region is, so "
            "'this group is more variable' stays visible.",
    detail=("density-preserving UMAP, which keeps relative local density so that "
            "differences in within-group spread survive the projection"),
    params=_UMAP_PARAMS, min_samples=10,
))


# ==========================================================================
# PaCMAP
# ==========================================================================
def _run_pacmap(ctx: DataContext, p: dict) -> ProjectionResult:
    import pacmap

    n = ctx.n_samples
    nn = int(min(int(p["n_neighbors"]), max(2, n - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = pacmap.PaCMAP(
            n_components=2, n_neighbors=nn,
            MN_ratio=float(p["mn_ratio"]), FP_ratio=float(p["fp_ratio"]),
            random_state=ctx.random_state,
        )
        Y = two_columns(np.asarray(reducer.fit_transform(ctx.X, init="pca")))

    return ProjectionResult(
        name="PaCMAP", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names), axis_labels=("PaCMAP 1", "PaCMAP 2"),
    )


register(MethodSpec(
    key="pacmap", label="PaCMAP", family="manifold", run=_run_pacmap,
    citations=("wang2021",), caveat=NONLINEAR_CAVEAT,
    available=has_module("pacmap"), install_hint="PaCMAP needs:  pip install pacmap",
    summary="Balances local detail and overall shape better than t-SNE or UMAP, "
            "and barely needs tuning.",
    detail=("pairwise-controlled manifold approximation, which balances "
            "near-neighbour, mid-near and far pairs explicitly"),
    params=(
        ParamSpec("n_neighbors", "int", default=10, low=2, high=200,
                  tier="common", label="Neighbours",
                  suggest=lambda c: int(np.clip(10, 2, max(2, c.n_samples - 1))),
                  limit=_neighbour_ceiling,
                  scan_default=(5, 10, 15, 25),
                  help="10 works for almost any data set below ten thousand "
                       "samples; this is the parameter PaCMAP is least sensitive to."),
        ParamSpec("mn_ratio", "float", default=0.5, low=0.05, high=5.0, step=0.05,
                  tier="advanced", label="Mid-near weight",
                  scan_default=(0.25, 0.5, 1.0),
                  help="Higher values pull the overall arrangement together."),
        ParamSpec("fp_ratio", "float", default=2.0, low=0.1, high=10.0, step=0.5,
                  tier="advanced", label="Far-pair weight",
                  help="Higher values push unrelated points further apart."),
    ),
    min_samples=10,
))


# ==========================================================================
# PHATE
# ==========================================================================
def _run_phate(ctx: DataContext, p: dict) -> ProjectionResult:
    import phate

    n = ctx.n_samples
    knn = int(min(int(p["knn"]), max(2, n - 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        op = phate.PHATE(n_components=2, knn=knn, decay=float(p["decay"]),
                         random_state=ctx.random_state, verbose=0)
        Y = two_columns(np.asarray(op.fit_transform(ctx.X)))

    return ProjectionResult(
        name="PHATE", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names), axis_labels=("PHATE 1", "PHATE 2"),
    )


register(MethodSpec(
    key="phate", label="PHATE", family="manifold", run=_run_phate,
    citations=("moon2019",),
    caveat="Designed for continuous gradients; clusters may be drawn as branches.",
    available=has_module("phate"), install_hint="PHATE needs:  pip install phate",
    summary="Best when the biology is a gradient (dose, time, development) "
            "rather than separate clusters.",
    detail="PHATE diffusion-based embedding",
    params=(
        ParamSpec("knn", "int", default=5, low=2, high=100, tier="common",
                  label="Neighbours", scan_default=(3, 5, 10, 20),
                  suggest=lambda c: int(np.clip(round(c.n_samples ** 0.4), 3,
                                                max(3, c.n_samples - 2))),
                  help="How far the diffusion step reaches."),
        ParamSpec("decay", "float", default=40.0, low=1.0, high=200.0, step=5.0,
                  tier="advanced", label="Kernel decay",
                  help="How quickly influence falls off with distance."),
    ),
    min_samples=10,
))


# ==========================================================================
# Isomap and kernel PCA
# ==========================================================================
def _run_isomap(ctx: DataContext, p: dict) -> ProjectionResult:
    from sklearn.manifold import Isomap

    n = ctx.n_samples
    nn = int(min(int(p["n_neighbors"]), max(2, n - 2)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Y = two_columns(Isomap(n_components=2, n_neighbors=nn).fit_transform(ctx.X))
    return ProjectionResult(
        name="Isomap", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names), axis_labels=("Isomap 1", "Isomap 2"),
    )


register(MethodSpec(
    key="isomap", label="Isomap", family="manifold", run=_run_isomap,
    citations=("tenenbaum2000",),
    caveat="Assumes the data lie on one connected surface; disconnected groups "
           "are joined by the shortest available path.",
    summary="Preserves distances measured along the data's own surface.",
    detail="Isomap geodesic multidimensional scaling",
    params=(
        ParamSpec("n_neighbors", "int", default=10, low=2, high=200, tier="common",
                  label="Neighbours", suggest=_suggest_neighbours,
                  limit=lambda c: int(max(2, c.n_samples - 2)),
                  scan_default=(5, 10, 20, 30),
                  help="How many neighbours define the surface. Too few breaks it "
                       "into pieces; too many short-circuits it."),
    ),
    min_samples=10,
))


def _run_kpca(ctx: DataContext, p: dict) -> ProjectionResult:
    from sklearn.decomposition import KernelPCA

    kernel = str(p.get("kernel", "rbf"))
    gamma = float(p.get("gamma", 0.0)) or None
    kp = KernelPCA(n_components=2, kernel=kernel, gamma=gamma,
                   random_state=ctx.random_state)
    Y = two_columns(kp.fit_transform(ctx.X))
    ev = None
    if getattr(kp, "eigenvalues_", None) is not None and kp.eigenvalues_.sum() > 0:
        ev = kp.eigenvalues_[:2] / kp.eigenvalues_.sum()
    return ProjectionResult(
        name="Kernel PCA", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names), axis_labels=("kPC1", "kPC2"),
        explained_variance=np.asarray(ev) if ev is not None else None,
    )


register(MethodSpec(
    key="kpca", label="Kernel PCA", family="manifold", run=_run_kpca,
    citations=("scholkopf1998",),
    caveat="Axes are combinations in a transformed space and are not directly "
           "interpretable as metric loadings.",
    summary="PCA after a non-linear transform; picks up curved structure.",
    detail="kernel principal component analysis",
    params=(
        ParamSpec("kernel", "choice", default="rbf",
                  choices=("rbf", "poly", "cosine", "sigmoid"), tier="common",
                  label="Kernel", help="Which kind of curvature to allow."),
        ParamSpec("gamma", "float", default=0.0, low=0.0, high=10.0, step=0.05,
                  tier="advanced", label="Kernel width",
                  help="0 lets scikit-learn choose (1 / number of metrics)."),
    ),
    min_samples=5,
))


__all__ = ["NONLINEAR_CAVEAT"]
