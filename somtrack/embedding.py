"""PCA, t-SNE and UMAP embeddings with shared quality diagnostics.

Every embedding is returned with the same accompanying information:

* the 2-D coordinates,
* a *feature-axis* matrix -- the correlation of each input feature with each
  embedding axis.  For PCA those are the true loadings; for t-SNE and UMAP they
  are post-hoc correlations, which is the only honest way to draw arrows on a
  non-linear embedding, and they are labelled as such in the figures.
* trustworthiness and continuity, so a t-SNE that has invented structure can be
  told apart from one that has preserved it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

from .config import EmbeddingConfig


@dataclass
class EmbeddingResult:
    name: str                        # "PCA" | "t-SNE" | "UMAP"
    coords: np.ndarray               # (n_samples, 2)
    feature_axes: np.ndarray         # (n_features, 2) correlation / loading
    feature_names: list[str]
    axis_labels: tuple[str, str]
    explained_variance: np.ndarray | None = None
    trustworthiness: float = np.nan
    continuity: float = np.nan
    params: dict = field(default_factory=dict)
    axes_are_loadings: bool = False   # True only for PCA

    @property
    def axis_caption(self) -> str:
        if self.axes_are_loadings:
            return "Arrows are PCA loadings."
        return ("Arrows are post-hoc Pearson correlations between each metric and "
                "the embedding axes; they indicate direction, not a linear model.")


def umap_available() -> bool:
    try:
        import umap  # noqa: F401

        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
def run_pca(X: np.ndarray, names: list[str], cfg: EmbeddingConfig) -> EmbeddingResult:
    from sklearn.decomposition import PCA

    k = min(cfg.n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=k, random_state=cfg.random_state)
    Y = pca.fit_transform(X)
    if Y.shape[1] < 2:
        Y = np.column_stack([Y[:, 0], np.zeros(len(Y))])

    ev = pca.explained_variance_ratio_
    # loadings scaled to correlation units
    comp = pca.components_[:2]
    sd = np.sqrt(pca.explained_variance_[:2])
    with np.errstate(invalid="ignore", divide="ignore"):
        load = (comp * sd[:, None]).T / np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)[:, None]

    labels = (f"PC1 ({ev[0] * 100:.1f}%)",
              f"PC2 ({ev[1] * 100:.1f}%)" if len(ev) > 1 else "PC2")
    return EmbeddingResult(
        name="PCA", coords=Y[:, :2], feature_axes=np.nan_to_num(load),
        feature_names=list(names), axis_labels=labels,
        explained_variance=ev, axes_are_loadings=True,
        **_quality(X, Y[:, :2]),
        params={"n_components": k},
    )


def run_tsne(X: np.ndarray, names: list[str], cfg: EmbeddingConfig) -> EmbeddingResult:
    from sklearn.manifold import TSNE

    n = X.shape[0]
    perp = cfg.tsne_perplexity or max(5.0, min(50.0, n / 4.0))
    perp = float(min(perp, max(2.0, (n - 1) / 3.0)))

    kwargs = dict(
        n_components=2, perplexity=perp, init=cfg.tsne_init,
        learning_rate=cfg.tsne_learning_rate, random_state=cfg.random_state,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ts = TSNE(max_iter=cfg.tsne_max_iter, **kwargs)
        except TypeError:                       # scikit-learn < 1.5
            ts = TSNE(n_iter=cfg.tsne_max_iter, **kwargs)
        Y = ts.fit_transform(X)

    return EmbeddingResult(
        name="t-SNE", coords=Y, feature_axes=_axis_correlation(X, Y),
        feature_names=list(names), axis_labels=("t-SNE 1", "t-SNE 2"),
        **_quality(X, Y),
        params={"perplexity": perp, "max_iter": cfg.tsne_max_iter},
    )


def run_umap(X: np.ndarray, names: list[str], cfg: EmbeddingConfig) -> EmbeddingResult:
    import umap

    n = X.shape[0]
    nn = cfg.umap_n_neighbors or int(max(5, min(50, round(n ** 0.5))))
    nn = int(min(nn, max(2, n - 1)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(
            n_components=2, n_neighbors=nn, min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric, random_state=cfg.random_state,
        )
        Y = reducer.fit_transform(X)
    Y = np.asarray(Y)

    return EmbeddingResult(
        name="UMAP", coords=Y, feature_axes=_axis_correlation(X, Y),
        feature_names=list(names), axis_labels=("UMAP 1", "UMAP 2"),
        **_quality(X, Y),
        params={"n_neighbors": nn, "min_dist": cfg.umap_min_dist, "metric": cfg.umap_metric},
    )


def run_embeddings(X: np.ndarray, names: list[str], cfg: EmbeddingConfig,
                   progress=None) -> dict[str, EmbeddingResult]:
    out: dict[str, EmbeddingResult] = {}
    jobs = []
    if cfg.run_pca:
        jobs.append(("PCA", run_pca))
    if cfg.run_tsne:
        jobs.append(("t-SNE", run_tsne))
    if cfg.run_umap and umap_available():
        jobs.append(("UMAP", run_umap))

    for i, (label, fn) in enumerate(jobs):
        if progress:
            progress(i, len(jobs))
        try:
            out[label] = fn(X, names, cfg)
        except Exception as exc:
            warnings.warn(f"{label} failed: {exc}")
    if progress:
        progress(len(jobs), len(jobs))
    return out


# --------------------------------------------------------------------------
def _axis_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between every feature and each embedding axis."""
    out = np.zeros((X.shape[1], Y.shape[1]))
    ys = Y - Y.mean(axis=0)
    yn = np.linalg.norm(ys, axis=0)
    for j in range(X.shape[1]):
        xs = X[:, j] - X[:, j].mean()
        xn = np.linalg.norm(xs)
        if xn <= 1e-12:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            out[j] = (xs @ ys) / (xn * np.where(yn > 0, yn, 1.0))
    return np.nan_to_num(out)


def _quality(X: np.ndarray, Y: np.ndarray, k: int = 0) -> dict:
    """Trustworthiness (no false neighbours) and continuity (no lost ones)."""
    n = X.shape[0]
    k = k or max(3, min(12, n // 10))
    if n < k + 2:
        return {"trustworthiness": np.nan, "continuity": np.nan}
    try:
        from sklearn.manifold import trustworthiness

        t = float(trustworthiness(X, Y, n_neighbors=k))
        c = float(trustworthiness(Y, X, n_neighbors=k))     # continuity = reverse T
        return {"trustworthiness": t, "continuity": c}
    except Exception:
        return {"trustworthiness": np.nan, "continuity": np.nan}


__all__ = ["EmbeddingResult", "run_pca", "run_tsne", "run_umap",
           "run_embeddings", "umap_available"]
