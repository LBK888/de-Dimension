"""Which metrics actually drive the separation.

A large classifier weight does not mean a metric carries group information.  A
linear model can give large weight to a metric that carries *no* group signal at
all, purely to cancel correlated noise in another metric -- a suppressor
variable.  Reading such a weight as "this metric distinguishes the groups" is a
well-documented way to publish a wrong conclusion (Haufe et al. 2014).

The fix is one matrix multiplication.  Transforming the backward model (the
weights the classifier uses to *extract* the signal) into the corresponding
forward model gives an *activation pattern*, which is interpretable: it is the
covariance between each metric and the discriminant score.  Both are reported,
side by side, because the difference between them is itself informative.

The second problem here is collinearity.  Turning rate, meander and
straightness measure much the same thing, so permuting them one at a time
understates all three: the model simply reads the remaining two.  Permuting a
whole correlated cluster at once measures what that *aspect of behaviour* is
worth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..i18n import Text


# ==========================================================================
# Haufe transform
# ==========================================================================
def haufe_patterns(X: np.ndarray, W: np.ndarray, y: np.ndarray | None = None,
                   tol: float = 1e-2) -> np.ndarray:
    """Turn extraction filters into interpretable activation patterns.

    ``A = Sigma_x W Sigma_s^-1`` with ``W`` the (n_features, n_components) filter
    matrix and ``Sigma_s`` the covariance of the latent scores ``s = W^T x``.
    Returned with the same orientation as the input, ``(n_components, n_features)``.

    The one detail the formula hides is that ``Sigma_s`` is routinely
    ill-conditioned.  A three-class discriminant hands back three filters that
    span a two-dimensional discriminant space plus one direction carrying the
    grand mean; a one-vs-one support vector machine hands back more filters
    still.  Inverting that as written lets the near-empty directions dominate,
    and the pattern then ranks noise metrics above the ones carrying the signal
    -- the exact failure the transform exists to prevent.

    The filters are therefore reduced first.  Given ``y``, the reduction keeps
    the directions along which the *group centroids* actually differ, which is
    a canonical-variate step and drops the grand-mean direction by construction.
    Without ``y`` it falls back to the directions carrying score variance.
    """
    X = np.asarray(X, float)
    W = np.atleast_2d(np.asarray(W, float))
    if W.shape[1] != X.shape[1]:                 # accept (d, c) as well as (c, d)
        if W.shape[0] == X.shape[1]:
            W = W.T
        else:
            raise ValueError(
                f"Weight matrix {W.shape} does not match {X.shape[1]} features.")

    Wt = W.T                                                  # (d, c)
    S = X @ Wt                                                # (n, c)
    Sc = S - S.mean(axis=0, keepdims=True)
    Sx = np.atleast_2d(np.cov(X, rowvar=False))

    if W.shape[0] == 1:
        V = np.ones((1, 1))
        share = np.ones(1)
    else:
        basis = Sc
        if y is not None:
            codes = np.asarray(y)
            centroids = np.vstack([Sc[codes == g].mean(axis=0)
                                   for g in np.unique(codes)
                                   if (codes == g).any()])
            basis = centroids - centroids.mean(axis=0, keepdims=True)
        sv, Vt = np.linalg.svd(basis, full_matrices=False)[1:]
        keep = sv > (tol * (sv[0] if sv.size else 1.0))
        if not keep.any():
            return np.zeros((1, X.shape[1]))
        V = Vt[keep].T                                        # (c, r)
        kept = sv[keep]
        share = (kept ** 2) / (kept ** 2).sum()

    Wr = Wt @ V                                               # (d, r) reduced filters
    var = (X @ Wr).var(axis=0)
    var = np.where(var > 1e-12, var, 1.0)

    # Per-component Haufe transform: A_r = Sigma_x w_r / var(s_r).  Taking each
    # component separately is the exact transform when the components are
    # uncorrelated and a well-conditioned approximation when they are not, which
    # a joint Sigma_s^-1 is emphatically not: a component carrying 2% of the
    # between-group signal would otherwise be inflated until it outranked the
    # metrics that actually separate the groups.  Weighting by each component's
    # share of the between-group signal keeps that component in its place.
    A = (Sx @ Wr) / var[None, :] * np.sqrt(share)[None, :]     # (d, r)
    return np.nan_to_num(A.T)                                  # (r, n_features)


def weight_table(feature_names: list, weights: np.ndarray | None,
                 activation: np.ndarray | None,
                 vip: np.ndarray | None = None) -> pd.DataFrame:
    """Filters and activation patterns side by side, ranked by the pattern."""
    n = len(feature_names)
    df = pd.DataFrame({"feature": feature_names})
    if weights is not None and weights.size:
        W = np.atleast_2d(weights)
        df["weight"] = np.linalg.norm(W, axis=0)[:n] if W.shape[1] == n else np.nan
        for i in range(min(W.shape[0], 3)):
            df[f"weight_{i + 1}"] = W[i, :n]
    if activation is not None and activation.size:
        A = np.atleast_2d(activation)
        df["activation"] = np.linalg.norm(A, axis=0)[:n] if A.shape[1] == n else np.nan
        for i in range(min(A.shape[0], 3)):
            df[f"activation_{i + 1}"] = A[i, :n]
    if vip is not None and len(vip) == n:
        df["vip"] = vip

    sort_col = "activation" if "activation" in df else (
        "vip" if "vip" in df else ("weight" if "weight" in df else None))
    if sort_col:
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    return df


# ==========================================================================
# Correlated feature clusters
# ==========================================================================
def feature_clusters(X: np.ndarray, names: list, threshold: float = 0.7
                     ) -> tuple[np.ndarray, dict[int, list[str]]]:
    """Group metrics that measure much the same thing, by correlation distance."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = X.shape[1]
    if d < 2:
        return np.zeros(d, dtype=int), {0: list(names)}

    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    D = np.clip(1.0 - np.abs(C), 0.0, 2.0)
    np.fill_diagonal(D, 0.0)
    Z = linkage(squareform(D, checks=False), method="average")
    labels = fcluster(Z, t=1.0 - threshold, criterion="distance") - 1

    members: dict[int, list[str]] = {}
    for i, lab in enumerate(labels):
        members.setdefault(int(lab), []).append(names[i])
    return labels.astype(int), members


# ==========================================================================
# Grouped permutation importance
# ==========================================================================
@dataclass
class ImportanceResult:
    table: pd.DataFrame
    cluster_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    cv_balanced_accuracy: float = np.nan
    chance: float = np.nan
    model: str = ""
    n_repeats: int = 0
    clustered_at: float = np.nan
    citations: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    def top(self, n: int = 10) -> list[str]:
        if self.table.empty:
            return []
        return self.table.head(n)["feature"].tolist()


def _mean_permuted_score(fitted, Xte: np.ndarray, yte: np.ndarray, k: int,
                         cols, n_repeats: int, rng, joint: bool) -> float:
    """Balanced accuracy after shuffling ``cols``, averaged over ``n_repeats``.

    All repeats are stacked into one matrix and predicted in a single call.
    Predicting a 320-row block costs a random forest barely more than a 32-row
    block, so this is an order of magnitude faster than looping -- which matters,
    because this loop runs once per metric per fold and was the slowest step in
    the whole pipeline.

    ``joint`` permutes the columns together with one row ordering, which keeps
    the correlation structure inside a cluster intact and measures what that
    aspect of behaviour is worth as a whole.
    """
    from .classify import _balanced_accuracy

    cols = np.atleast_1d(np.asarray(cols, int))
    if cols.size == 0:
        return np.nan
    n = len(Xte)
    big = np.tile(Xte, (n_repeats, 1))
    for r in range(n_repeats):
        rows = np.arange(r * n, (r + 1) * n)
        if joint:
            big[np.ix_(rows, cols)] = Xte[rng.permutation(n)][:, cols]
        else:
            for c in cols:
                big[rows, c] = rng.permutation(Xte[:, c])

    pred = fitted.predict(big)
    return float(np.mean([_balanced_accuracy(yte, pred[r * n:(r + 1) * n], k)
                          for r in range(n_repeats)]))


def permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    structure,
    *,
    model: str = "random_forest",
    n_repeats: int = 10,
    cluster_threshold: float = 0.7,
    scheme: str = "auto",
    n_splits: int = 5,
    random_state: int | None = 0,
) -> ImportanceResult:
    """Cross-validated permutation importance, per metric and per correlated cluster.

    The score that is degraded is balanced accuracy on held-out data, so an
    importance of 0.10 reads as "shuffling this metric costs ten points of
    balanced accuracy".
    """
    from sklearn.base import clone

    from .blocks import make_cv
    from .classify import _balanced_accuracy, make_estimator  # noqa: F401

    k = int(np.max(y)) + 1 if len(y) else 0
    res = ImportanceResult(table=pd.DataFrame({"feature": feature_names}),
                           model=model, n_repeats=n_repeats,
                           clustered_at=cluster_threshold,
                           chance=1.0 / k if k else np.nan,
                           citations=("breiman2001", "altmann2010"))
    counts = np.bincount(y, minlength=max(k, 1))
    if k < 2 or counts[counts > 0].min() < 2:
        res.notes.append(Text("Too few samples per group for permutation "
                              "importance."))
        return res

    labels, members = feature_clusters(X, list(feature_names), cluster_threshold)
    plan = make_cv(y, structure, scheme, n_splits, 1, random_state)
    est = make_estimator(model, random_state=random_state)
    rng = np.random.default_rng(random_state)

    d = X.shape[1]
    drop = np.zeros((0, d))
    cluster_ids = sorted(set(labels.tolist()))
    cluster_drop = np.zeros((0, len(cluster_ids)))
    base_scores = []

    for tr, te in plan.splitter.split(X, y, plan.groups):
        fitted = clone(est).fit(X[tr], y[tr])
        base = _balanced_accuracy(y[te], fitted.predict(X[te]), k)
        base_scores.append(base)

        per_feature = np.array([
            base - _mean_permuted_score(fitted, X[te], y[te], k, [j],
                                        n_repeats, rng, joint=False)
            for j in range(d)])
        drop = np.vstack([drop, per_feature])

        per_cluster = np.array([
            base - _mean_permuted_score(fitted, X[te], y[te], k,
                                        np.flatnonzero(labels == c),
                                        n_repeats, rng, joint=True)
            for c in cluster_ids])
        cluster_drop = np.vstack([cluster_drop, per_cluster])

    res.cv_balanced_accuracy = float(np.mean(base_scores))
    table = pd.DataFrame({
        "feature": list(feature_names),
        "importance": drop.mean(axis=0),
        "sd": drop.std(axis=0),
        "cluster": labels,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)
    res.table = table

    res.cluster_table = pd.DataFrame({
        "cluster": cluster_ids,
        "members": ["; ".join(members[c]) for c in cluster_ids],
        "n_members": [len(members[c]) for c in cluster_ids],
        "importance": cluster_drop.mean(axis=0),
        "sd": cluster_drop.std(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return res


__all__ = [
    "haufe_patterns", "weight_table", "feature_clusters",
    "ImportanceResult", "permutation_importance",
]
