"""The common result type for every 2-D projection, and the data it is given.

:class:`ProjectionResult` is a strict superset of the 2.0 ``EmbeddingResult``:
figure code written against the old type keeps working, and the new fields carry
the things a projection has to declare before it can be used as evidence --
neighbourhood-preservation curves, per-point reliability, stability across
seeds, and, for supervised methods, the out-of-fold coordinates without which a
supervised picture means nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:                                   # pragma: no cover
    from ..preprocess import PreparedData


# ==========================================================================
# Input
# ==========================================================================
@dataclass
class DataContext:
    """Everything a projection method may look at.

    Methods receive this rather than a bare matrix so that a supervised method
    can reach the labels and a replicate-aware method can reach the blocks,
    without every signature growing a new argument.
    """

    X: np.ndarray                                   # (n_samples, n_features), scaled
    feature_names: list[str]
    group_codes: np.ndarray | None = None
    group_values: list = field(default_factory=list)
    replicates: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    random_state: int | None = 0
    n_components: int = 2

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_groups(self) -> int:
        return len(self.group_values)

    @property
    def has_groups(self) -> bool:
        return self.group_codes is not None and self.n_groups >= 2

    @classmethod
    def from_prepared(cls, prep: "PreparedData", random_state: int | None = 0,
                      n_components: int = 2) -> "DataContext":
        return cls(
            X=prep.X,
            feature_names=list(prep.feature_names),
            group_codes=prep.group_codes,
            group_values=list(prep.group_values),
            replicates=prep.replicates,
            sample_ids=prep.sample_ids,
            random_state=random_state,
            n_components=n_components,
        )


# ==========================================================================
# Output
# ==========================================================================
@dataclass
class ProjectionResult:
    """A 2-D projection plus everything needed to judge whether to believe it."""

    # ---- identity -------------------------------------------------------
    name: str                                       # display label, "PaCMAP"
    coords: np.ndarray                              # (n_samples, 2)
    feature_axes: np.ndarray                        # (n_features, 2)
    feature_names: list[str]
    axis_labels: tuple[str, str]

    # ---- linear-model extras -------------------------------------------
    explained_variance: np.ndarray | None = None
    axes_are_loadings: bool = False                 # True only for true loadings

    # ---- neighbourhood preservation ------------------------------------
    trustworthiness: float = np.nan
    continuity: float = np.nan
    rnx: np.ndarray | None = None                   # R_NX(K), K = 1..n-2
    rnx_auc: float = np.nan                         # area under R_NX on log K

    # ---- per-point diagnostics -----------------------------------------
    reliability: np.ndarray | None = None           # (n_samples,) scDEED score
    dubious: np.ndarray | None = None               # (n_samples,) bool
    stability: np.ndarray | None = None             # (n_samples,) across seeds

    # ---- supervised methods --------------------------------------------
    supervised: bool = False
    oof_coords: np.ndarray | None = None            # out-of-fold coordinates
    cv: Any = None                                  # stats.classify.ClassifyResult

    # ---- provenance -----------------------------------------------------
    method_key: str = ""
    params: dict = field(default_factory=dict)
    citations: tuple[str, ...] = ()
    caveat: str = ""

    # ------------------------------------------------------------------
    @property
    def n_dubious(self) -> int:
        return int(self.dubious.sum()) if self.dubious is not None else 0

    @property
    def dubious_fraction(self) -> float:
        if self.dubious is None or self.dubious.size == 0:
            return np.nan
        return float(self.dubious.mean())

    @property
    def axis_caption(self) -> str:
        if self.axes_are_loadings:
            return "Arrows are model loadings."
        return ("Arrows are post-hoc Pearson correlations between each metric and "
                "the projection axes; they indicate direction, not a linear model.")

    def quality_row(self) -> dict:
        """One row for the method-comparison table."""
        row = {
            "method": self.name,
            "trustworthiness": self.trustworthiness,
            "continuity": self.continuity,
            "rnx_auc": self.rnx_auc,
            "dubious_fraction": self.dubious_fraction,
            "supervised": self.supervised,
        }
        if self.explained_variance is not None and self.explained_variance.size:
            row["variance_explained"] = float(self.explained_variance[:2].sum())
        if self.cv is not None:
            row["cv_balanced_accuracy"] = getattr(self.cv, "balanced_accuracy", np.nan)
            row["cv_permutation_p"] = getattr(self.cv, "permutation_p", np.nan)
        return row


# ==========================================================================
# Shared helpers
# ==========================================================================
def axis_correlation(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Pearson r between every feature and each projection axis."""
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


def two_columns(Y: np.ndarray) -> np.ndarray:
    """Pad a 1-D projection to two columns so every result has the same shape."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[1] == 1:
        return np.column_stack([Y[:, 0], np.zeros(len(Y))])
    return Y[:, :2]


__all__ = ["DataContext", "ProjectionResult", "axis_correlation", "two_columns"]
