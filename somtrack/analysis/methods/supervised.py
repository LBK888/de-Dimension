"""Supervised projections, and the guard rail they have to carry.

A supervised projection is given the answer before it draws the picture.  It
will therefore separate the groups -- on real data, and equally on pure noise.
This is not a subtlety: it has been documented independently for between-group
PCA (Cardini, O'Higgins & Rohlf 2019), for PLS-DA, where a score plot showing
separation was shown to carry no information because random data produces the
same plot (Westerhuis et al. 2008), and for supervised UMAP, whose own issue
tracker is full of embeddings that separate perfectly in training and collapse
on held-out data.

These methods are still worth having.  They answer "*if* the groups differ, in
which direction, and which metrics carry it" far better than an unsupervised
projection can.  They just cannot answer "do the groups differ", and this module
makes that structurally impossible to forget:

*   every method here is marked ``supervised=True``, which forces a ``caveat``
    that the figure layer prints on the panel;
*   every method returns ``oof_coords`` next to ``coords`` -- the same
    projection fitted without each sample and then applied to it.  The figure
    draws both.  In-sample separation with out-of-fold collapse is the honest
    picture of "no";
*   every method returns a cross-validated score with a permutation p value
    attached, so the number and the picture travel together.
"""

from __future__ import annotations

import warnings

import numpy as np

from ...stats.blocks import analyse_blocks, make_cv
from ..projection import DataContext, ProjectionResult, axis_correlation, two_columns
from ..registry import MethodSpec, ParamSpec, has_module, register

SUPERVISED_CAVEAT = (
    "Supervised projection: the group labels were used to build these axes, so "
    "the groups separate here by construction and would separate on random data "
    "too. Judge separation from the cross-validated score, not from the picture."
)


# ==========================================================================
def _canonical_directions(X: np.ndarray, B: np.ndarray, y: np.ndarray,
                          n: int = 2) -> np.ndarray:
    """Reduce a filter set to the ``n`` directions the group centroids differ along."""
    B = np.atleast_2d(B)
    if B.shape[0] == X.shape[1]:
        pass
    elif B.shape[1] == X.shape[1]:
        B = B.T
    S = X @ B
    S = S - S.mean(axis=0, keepdims=True)
    cent = np.vstack([S[y == g].mean(axis=0) for g in np.unique(y)])
    cent = cent - cent.mean(axis=0, keepdims=True)
    Vt = np.linalg.svd(cent, full_matrices=False)[2]
    k = int(min(n, Vt.shape[0]))
    return B @ Vt[:k].T                                     # (d, k)


def _pad(B: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A second axis for two-group problems, orthogonal to the discriminant.

    With two groups there is only one discriminant direction.  Rather than
    plotting a line, the residual direction of greatest variance is used as the
    second axis, so the picture also shows how the groups are spread.
    """
    if B.shape[1] >= 2:
        return B[:, :2]
    w = B[:, 0]
    w = w / max(np.linalg.norm(w), 1e-12)
    R = X - np.outer(X @ w, w)
    R = R - R.mean(axis=0, keepdims=True)
    v = np.linalg.svd(R, full_matrices=False)[2][0]
    return np.column_stack([w, v])


def _out_of_fold(fit_basis, ctx: DataContext) -> np.ndarray | None:
    """Project every sample using a basis fitted without it.

    The fold structure comes from the same block analysis the statistics use, so
    a replicate never contributes to the basis that is used to place it.
    """
    y = ctx.group_codes
    structure = analyse_blocks(y, ctx.replicates)
    plan = make_cv(y, structure, "auto", 5, 1, ctx.random_state)

    out = np.full((ctx.n_samples, 2), np.nan)
    ok = False
    for tr, te in plan.splitter.split(ctx.X, y, plan.groups):
        if len(np.unique(y[tr])) < 2:
            continue
        try:
            B = fit_basis(ctx.X[tr], y[tr])
        except Exception:
            continue
        if B is None:
            continue
        centre = ctx.X[tr].mean(axis=0)
        out[te] = (ctx.X[te] - centre) @ B[:, :2]
        ok = True
    return out if ok else None


def _cv_score(ctx: DataContext, model: str, n_permutations: int):
    from ...stats.classify import classify

    structure = analyse_blocks(ctx.group_codes, ctx.replicates)
    return classify(
        ctx.X, ctx.group_codes, ctx.group_values, structure, model=model,
        n_permutations=int(n_permutations), bootstrap=1000,
        feature_names=list(ctx.feature_names), random_state=ctx.random_state,
    )


# ==========================================================================
# Linear discriminant analysis
# ==========================================================================
def _lda_basis(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, y)
            B = np.asarray(m.scalings_)[:, :max(1, len(np.unique(y)) - 1)]
        except Exception:
            m = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
            B = _canonical_directions(X, m.coef_, y, 2)
    return _pad(np.asarray(B, float), X, y)


def _run_lda(ctx: DataContext, p: dict) -> ProjectionResult:
    B = _lda_basis(ctx.X, ctx.group_codes)
    centre = ctx.X.mean(axis=0)
    Y = two_columns((ctx.X - centre) @ B[:, :2])
    cv = _cv_score(ctx, "lda_shrinkage", p.get("n_permutations", 499))
    return ProjectionResult(
        name="LDA", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names),
        axis_labels=("LD1", "LD2" if ctx.n_groups > 2 else "residual variance"),
        oof_coords=_out_of_fold(_lda_basis, ctx), cv=cv,
    )


register(MethodSpec(
    key="lda", label="LDA", family="supervised", run=_run_lda,
    supervised=True, needs_groups=True, caveat=SUPERVISED_CAVEAT,
    citations=("fisher1936", "ledoit2004", "cardini2020"),
    summary="The classical 'can these groups be told apart' projection, "
            "regularised for small samples.",
    detail=("linear discriminant analysis with Ledoit-Wolf shrinkage, drawn both "
            "in-sample and out-of-fold"),
    params=(
        ParamSpec("n_permutations", "int", default=499, low=0, high=9999,
                  tier="advanced", label="Label permutations",
                  help="How many times the group labels are shuffled to work out "
                       "what this projection would achieve by chance."),
    ),
    min_samples=10,
))


# ==========================================================================
# PLS-DA
# ==========================================================================
def _plsda_basis_factory(n_components: int):
    def basis(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from ...stats.classify import PLSDA

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = PLSDA(n_components=max(2, n_components)).fit(X, y)
        B = np.asarray(m.model_.x_rotations_)[:, :2]
        return _pad(B, X, y)
    return basis


def _run_plsda(ctx: DataContext, p: dict) -> ProjectionResult:
    k = int(p.get("n_components", 2))
    basis = _plsda_basis_factory(k)
    B = basis(ctx.X, ctx.group_codes)
    centre = ctx.X.mean(axis=0)
    Y = two_columns((ctx.X - centre) @ B[:, :2])
    cv = _cv_score(ctx, "plsda", p.get("n_permutations", 499))
    return ProjectionResult(
        name="PLS-DA", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names),
        axis_labels=("PLS component 1", "PLS component 2"),
        oof_coords=_out_of_fold(basis, ctx), cv=cv,
    )


register(MethodSpec(
    key="plsda", label="PLS-DA", family="supervised", run=_run_plsda,
    supervised=True, needs_groups=True, caveat=SUPERVISED_CAVEAT,
    citations=("barker2003", "westerhuis2008"),
    summary="The standard discriminant projection in metabolomics. Its score "
            "plot means nothing without the cross-validated number beside it.",
    detail="partial least squares discriminant analysis",
    params=(
        ParamSpec("n_components", "int", default=2, low=2, high=10, tier="common",
                  label="Components", scan_default=(2, 3, 4, 5),
                  help="How many latent directions to fit. More components fit the "
                       "training data better and generalise worse."),
        ParamSpec("n_permutations", "int", default=499, low=0, high=9999,
                  tier="advanced", label="Label permutations",
                  help="How many times the labels are shuffled to build the null."),
    ),
    min_samples=10,
))


# ==========================================================================
# Linear SVM
# ==========================================================================
def _svm_basis_factory(C: float):
    def basis(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        from sklearn.svm import SVC

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = SVC(kernel="linear", C=C, class_weight="balanced").fit(X, y)
        B = _canonical_directions(X, np.asarray(m.coef_), y, 2)
        return _pad(B, X, y)
    return basis


def _run_svm(ctx: DataContext, p: dict) -> ProjectionResult:
    C = float(p.get("C", 1.0))
    basis = _svm_basis_factory(C)
    B = basis(ctx.X, ctx.group_codes)
    centre = ctx.X.mean(axis=0)
    Y = two_columns((ctx.X - centre) @ B[:, :2])
    cv = _cv_score(ctx, "svm_linear", p.get("n_permutations", 499))
    return ProjectionResult(
        name="Linear SVM", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names),
        axis_labels=("SVM axis 1", "SVM axis 2"),
        oof_coords=_out_of_fold(basis, ctx), cv=cv,
    )


register(MethodSpec(
    key="svm", label="Linear SVM", family="supervised", run=_run_svm,
    supervised=True, needs_groups=True, caveat=SUPERVISED_CAVEAT,
    citations=("cortes1995", "haufe2014"),
    summary="Finds the widest gap between groups. Its weights need the Haufe "
            "transform before they can be read as 'this metric matters'.",
    detail=("a linear support vector machine, with weights converted to "
            "activation patterns before interpretation"),
    params=(
        ParamSpec("C", "float", default=1.0, low=0.001, high=1000.0, tier="common",
                  label="Regularisation", scan_default=(0.01, 0.1, 1.0, 10.0),
                  help="Small values keep the boundary simple and are the safer "
                       "choice when there are few samples."),
        ParamSpec("n_permutations", "int", default=499, low=0, high=9999,
                  tier="advanced", label="Label permutations",
                  help="How many times the labels are shuffled to build the null."),
    ),
    min_samples=10,
))


# ==========================================================================
# Supervised UMAP
# ==========================================================================
def _run_supervised_umap(ctx: DataContext, p: dict) -> ProjectionResult:
    import umap

    n = ctx.n_samples
    nn = int(min(int(p["n_neighbors"]), max(2, n - 1)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=nn, min_dist=float(p["min_dist"]),
            target_weight=float(p["target_weight"]),
            random_state=ctx.random_state,
        )
        Y = two_columns(np.asarray(reducer.fit_transform(ctx.X, y=ctx.group_codes)))

    return ProjectionResult(
        name="Supervised UMAP", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names),
        axis_labels=("sUMAP 1", "sUMAP 2"),
        cv=_cv_score(ctx, "lda_shrinkage", p.get("n_permutations", 0)),
    )


register(MethodSpec(
    key="supervised_umap", label="Supervised UMAP", family="supervised",
    run=_run_supervised_umap, supervised=True, needs_groups=True,
    caveat=SUPERVISED_CAVEAT + " This method in particular is known to separate "
                               "training data that it cannot separate on held-out data.",
    citations=("mcinnes2018",),
    available=has_module("umap"),
    install_hint="Supervised UMAP needs:  pip install umap-learn",
    summary="UMAP pulled towards the labels. Useful for showing a structure you "
            "have already demonstrated; never for demonstrating one.",
    detail="supervised UMAP with the group label as the target variable",
    params=(
        ParamSpec("n_neighbors", "int", default=15, low=2, high=200, tier="common",
                  label="Neighbours",
                  suggest=lambda c: int(np.clip(round(np.sqrt(c.n_samples)), 5,
                                                min(30, max(2, c.n_samples - 1))))),
        ParamSpec("min_dist", "float", default=0.1, low=0.0, high=1.0, step=0.05,
                  tier="common", label="Minimum separation"),
        ParamSpec("target_weight", "float", default=0.5, low=0.0, high=1.0, step=0.05,
                  tier="common", label="Label influence",
                  scan_default=(0.0, 0.25, 0.5, 0.8),
                  help="0 ignores the labels entirely (ordinary UMAP); 1 lets them "
                       "dominate. Report whatever you used."),
        ParamSpec("n_permutations", "int", default=0, low=0, high=9999,
                  tier="advanced", label="Label permutations"),
    ),
    min_samples=15,
))


# ==========================================================================
# SLISEMAP
# ==========================================================================
def _run_slisemap(ctx: DataContext, p: dict) -> ProjectionResult:
    """Group samples by *which metrics explain them*, not by their raw values.

    SLISEMAP fits a local linear model to every sample and lays the samples out
    so that points close together are the ones the same local model explains.
    Two animals with quite different speeds sit together if the same handful of
    metrics accounts for both; two with similar speeds sit apart if different
    metrics are doing the work.

    That makes it the one projection here that answers "which metrics explain
    *this* animal" rather than "which metrics separate the groups on average",
    which is worth having when a treatment does different things to different
    individuals.
    """
    import slisemap
    import torch

    onehot = np.zeros((ctx.n_samples, ctx.n_groups))
    onehot[np.arange(ctx.n_samples), ctx.group_codes] = 1.0

    # SLISEMAP deprecated its `random_state` argument in 1.6 and moved it to
    # `escape()`, so the seed is set on torch directly: passing the old keyword
    # would work today and warn, and stop working later.
    if ctx.random_state is not None:
        torch.manual_seed(int(ctx.random_state))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sm = slisemap.Slisemap(ctx.X, onehot, d=2, radius=float(p["radius"]),
                               lasso=float(p["lasso"]))
        sm.optimise()
        Y = two_columns(np.asarray(sm.get_Z()))

    # The arrows are post-hoc correlations, as for every other non-linear
    # projection here.  `get_B()` holds the local model coefficients, which are
    # the interesting part of SLISEMAP, but their column layout depends on the
    # local model, the intercept setting and the number of targets, and none of
    # that is part of the documented interface.  Guessing it would put confident
    # arrows on the wrong metrics, which is worse than drawing none.
    return ProjectionResult(
        name="SLISEMAP", coords=Y, feature_axes=axis_correlation(ctx.X, Y),
        feature_names=list(ctx.feature_names),
        axis_labels=("SLISEMAP 1", "SLISEMAP 2"),
        cv=_cv_score(ctx, "lda_shrinkage", p.get("n_permutations", 0)),
    )


register(MethodSpec(
    key="slisemap", label="SLISEMAP", family="supervised", run=_run_slisemap,
    supervised=True, needs_groups=True,
    caveat=("Supervised projection: positions come from local models fitted to "
            "the labels, so proximity here means 'explained the same way', not "
            "'similar measurements', and the groups will look organised whatever "
            "the data says."),
    citations=("bjorklund2023",),
    available=has_module("slisemap"),
    install_hint="SLISEMAP needs:  pip install slisemap   (it pulls in PyTorch)",
    summary="Places samples by which metrics explain them, so you can see "
            "whether a treatment acts the same way on every individual.",
    detail=("SLISEMAP supervised manifold visualisation, which places each "
            "sample by the local linear model that explains it"),
    params=(
        ParamSpec("radius", "float", default=3.5, low=0.5, high=10.0, step=0.5,
                  tier="common", label="Embedding radius",
                  scan_default=(2.0, 3.5, 5.0),
                  help="How spread out the layout is. Larger values separate the "
                       "local models more sharply."),
        ParamSpec("lasso", "float", default=0.01, low=0.0, high=1.0, step=0.01,
                  tier="common", label="Sparsity",
                  scan_default=(0.0, 0.01, 0.1),
                  help="Higher values force each local model to use fewer "
                       "metrics, which makes the explanations easier to read."),
        ParamSpec("n_permutations", "int", default=0, low=0, high=9999,
                  tier="advanced", label="Label permutations"),
    ),
    min_samples=20,
))


__all__ = ["SUPERVISED_CAVEAT"]
