"""Second-level clustering of the SOM codebook.

The macro's v1.2 k-means drew every k from 2 to n and left the choice to the
eye.  Here the same k range is scanned, but each solution is scored
(silhouette, Davies-Bouldin, Calinski-Harabasz, and -- when experimental groups
are known -- adjusted Rand against them) so the figure can state why a
particular k was picked.

Clustering the codebook rather than the samples is the standard two-level
approach of Vesanto & Alhoniemi (2000): the SOM denoises the data into a small
set of prototypes, and the prototypes are what get partitioned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import NodeClusterConfig
from .som import SomResult


@dataclass
class NodeClustering:
    labels_by_k: dict[int, np.ndarray] = field(default_factory=dict)   # k -> (n_nodes,)
    scores: pd.DataFrame | None = None
    best_k: int = 0
    method: str = "kmeans"

    @property
    def best_labels(self) -> np.ndarray | None:
        return self.labels_by_k.get(self.best_k)

    def sample_labels(self, res: SomResult, k: int | None = None) -> np.ndarray:
        """Project the node partition back onto the samples via their BMU."""
        lab = self.labels_by_k.get(k or self.best_k)
        if lab is None:
            return np.full(res.bmu.shape, -1)
        return lab[res.bmu]


def cluster_nodes(
    res: SomResult,
    cfg: NodeClusterConfig,
    group_codes: np.ndarray | None = None,
) -> NodeClustering:
    W = res.codebook
    hits = res.hits
    occupied = hits > 0
    n_occ = int(occupied.sum())

    out = NodeClustering(method=cfg.method)
    if not cfg.enabled or n_occ < 3:
        return out

    k_max = int(min(cfg.k_max, max(2, n_occ - 1)))
    k_min = int(max(2, min(cfg.k_min, k_max)))
    rows = []

    for k in range(k_min, k_max + 1):
        labels = _fit(W, hits if cfg.weight_by_hits else None, k, cfg)
        if labels is None:
            continue
        out.labels_by_k[k] = labels
        rows.append(_score(W, labels, occupied, k, res, group_codes))

    if rows:
        out.scores = pd.DataFrame(rows)
        # rank-average of the three internal indices, then pick the winner
        s = out.scores
        rank = (
            s["silhouette"].rank(ascending=False)
            + s["davies_bouldin"].rank(ascending=True)
            + s["calinski_harabasz"].rank(ascending=False)
        )
        out.best_k = int(s.loc[rank.idxmin(), "k"])
    return out


def _fit(W: np.ndarray, weights: np.ndarray | None, k: int,
         cfg: NodeClusterConfig) -> np.ndarray | None:
    try:
        if cfg.method == "ward":
            from sklearn.cluster import AgglomerativeClustering

            return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(W)
        if cfg.method == "gmm":
            from sklearn.mixture import GaussianMixture

            return GaussianMixture(
                n_components=k, covariance_type="full", random_state=cfg.random_state
            ).fit_predict(W)

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=k, n_init=10, random_state=cfg.random_state)
        # weighting by hits stops empty codebook vectors -- which the SOM only
        # ever dragged along behind their neighbours -- from defining clusters
        return km.fit_predict(W, sample_weight=weights)
    except Exception:
        return None


def _score(W, labels, occupied, k, res, group_codes) -> dict:
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)

    row: dict[str, float] = {"k": k}
    Wo, Lo = W[occupied], labels[occupied]
    try:
        row["silhouette"] = float(silhouette_score(Wo, Lo)) if len(set(Lo.tolist())) > 1 else np.nan
        row["davies_bouldin"] = float(davies_bouldin_score(Wo, Lo)) if len(set(Lo.tolist())) > 1 else np.nan
        row["calinski_harabasz"] = float(calinski_harabasz_score(Wo, Lo)) if len(set(Lo.tolist())) > 1 else np.nan
    except Exception:
        row["silhouette"] = row["davies_bouldin"] = row["calinski_harabasz"] = np.nan

    row["non_empty_clusters"] = int(len(np.unique(labels[occupied])))

    if group_codes is not None and len(np.unique(group_codes)) > 1:
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        sample_labels = labels[res.bmu]
        row["adjusted_rand_vs_groups"] = float(adjusted_rand_score(group_codes, sample_labels))
        row["nmi_vs_groups"] = float(normalized_mutual_info_score(group_codes, sample_labels))
    return row


__all__ = ["NodeClustering", "cluster_nodes"]
