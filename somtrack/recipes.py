"""Recipes: whole analyses named after the question they answer.

The settings page is the point where a biologist decides whether to use this
program or go back to a spreadsheet.  Version 2.0 presented twenty-one controls
at once and expected a choice on every one before the first figure appeared.

A recipe replaces that with one decision.  Each entry below is an ordinary
:class:`~somtrack.config.AnalysisConfig` with some fields changed, so a recipe
is data rather than a code path: it loads through the same JSON machinery, it
can be saved and shared, and every parameter it set is still visible and still
editable underneath.  Choosing one is a starting point, never a restriction.

The recipes are phrased as questions because that is how the choice actually
presents itself: a user knows whether they want to *test* their groups or
*explore* the data, and does not know whether they want a perplexity of 15.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .config import AnalysisConfig


@dataclass(frozen=True)
class Recipe:
    key: str
    title: str                     # what the user is trying to do
    summary: str                   # one line, plain language
    detail: str = ""               # what it will actually run
    apply: Callable[[AnalysisConfig], None] = lambda cfg: None
    figures: str = "core"
    tags: tuple[str, ...] = field(default_factory=tuple)

    def build(self, base: AnalysisConfig | None = None) -> AnalysisConfig:
        cfg = base or AnalysisConfig()
        self.apply(cfg)
        cfg.report.profile = self.figures
        return cfg


# ==========================================================================
def _standard(cfg: AnalysisConfig) -> None:
    cfg.embedding.methods = []                       # registry default
    cfg.embedding.allow_supervised = False
    cfg.stats.enabled = True
    cfg.som.algorithm = "batch"


def _test_groups(cfg: AnalysisConfig) -> None:
    cfg.embedding.methods = ["pca", "mds", "pacmap", "umap"]
    cfg.embedding.allow_supervised = False
    cfg.som.algorithm = "batch"
    st = cfg.stats
    st.enabled = True
    st.run_permanova = st.run_permdisp = st.run_energy = True
    st.run_classification = True
    st.n_permutations = 999
    st.classification_permutations = 999
    st.cv_scheme = "auto"
    st.use_replicate_blocks = True


def _which_metrics(cfg: AnalysisConfig) -> None:
    # The SVM is here for its activation patterns and its out-of-fold panel, not
    # to demonstrate separation -- which is why the recipe also runs the full
    # statistics layer underneath it.
    cfg.embedding.methods = ["pca", "pacmap", "svm"]
    cfg.embedding.allow_supervised = True
    cfg.som.algorithm = "relevance"
    st = cfg.stats
    st.enabled = True
    st.run_importance = True
    st.importance_repeats = 15
    st.run_effect_sizes = True
    st.classifier = "svm_linear"


def _robustness(cfg: AnalysisConfig) -> None:
    cfg.embedding.methods = ["pca", "mds", "tsne", "umap", "pacmap", "isomap"]
    cfg.embedding.allow_supervised = False
    cfg.embedding.stability_seeds = 5
    cfg.embedding.reliability_null = 30
    cfg.stats.enabled = True
    cfg.stats.n_permutations = 999


def _gradient(cfg: AnalysisConfig) -> None:
    cfg.embedding.methods = ["pca", "mds", "phate", "pacmap"]
    cfg.embedding.allow_supervised = False
    cfg.som.algorithm = "batch"
    cfg.stats.enabled = True
    cfg.stats.distance = "euclidean"


def _legacy(cfg: AnalysisConfig) -> None:
    cfg.som.algorithm = "online"
    cfg.som.init = "random"
    cfg.som.topology = "hex"
    cfg.embedding.methods = ["pca"]
    cfg.embedding.allow_supervised = False
    cfg.figure.palette = "legacy_hsb"
    cfg.stats.enabled = False


RECIPES: tuple[Recipe, ...] = (
    Recipe(
        key="standard",
        title="Explore the data",
        summary="Map the data, see how it is arranged, and check whether the "
                "groups differ.",
        detail="Batch SOM, the installed unsupervised projections, and the full "
               "statistics layer. The safe default if you are not sure.",
        apply=_standard, tags=("default",),
    ),
    Recipe(
        key="test_groups",
        title="Test whether my groups differ",
        summary="Answer 'can these treatments be told apart' with a number and a "
                "p value, not an impression.",
        detail="PERMANOVA and PERMDISP on the distances the pictures are drawn "
               "from, the energy test, and cross-validated classification with a "
               "permutation null. Supervised projections stay off, because they "
               "separate groups by construction and would beg the question.",
        apply=_test_groups, tags=("statistics",),
    ),
    Recipe(
        key="which_metrics",
        title="Find which measurements matter",
        summary="Rank the metrics that carry the difference, and say how much "
                "each one is worth.",
        detail="Relevance-learning SOM, a linear SVM with Haufe-transformed "
               "activation patterns, permutation importance per metric and per "
               "correlated cluster, and effect sizes in the original units.",
        apply=_which_metrics, tags=("interpretation",),
    ),
    Recipe(
        key="robustness",
        title="Check that the picture is real",
        summary="Run several projections and several random seeds, and report "
                "only what survives all of them.",
        detail="Six projections, per-point reliability against a scrambled null, "
               "seed stability by Procrustes alignment, and the agreement matrix "
               "between methods.",
        apply=_robustness, figures="full", tags=("diagnostics",),
    ),
    Recipe(
        key="gradient",
        title="Look for a dose or time gradient",
        summary="For treatments that form a series rather than separate classes.",
        detail="Adds PHATE, which is built for continuous trajectories, next to "
               "PCA and MDS so a gradient is not forced into clusters.",
        apply=_gradient, tags=("exploration",),
    ),
    Recipe(
        key="legacy_v12",
        title="Reproduce the v1.2 macro",
        summary="The original ImageJ workflow, for comparison with old results.",
        detail="Sequential SOM with random initialisation, the legacy hue wheel, "
               "PCA only, and the statistics layer switched off. Read AUDIT.md "
               "before comparing numbers: several v1.2 statistics were wrong.",
        apply=_legacy, tags=("legacy",),
    ),
)

BY_KEY: dict[str, Recipe] = {r.key: r for r in RECIPES}
DEFAULT = "standard"


def get(key: str) -> Recipe:
    try:
        return BY_KEY[key]
    except KeyError:
        raise KeyError(f"Unknown recipe '{key}'. Available: "
                       f"{', '.join(BY_KEY)}") from None


def apply_recipe(cfg: AnalysisConfig, key: str) -> AnalysisConfig:
    """Apply a recipe in place, leaving everything it does not mention alone."""
    get(key).apply(cfg)
    cfg.report.recipe = key
    cfg.report.profile = get(key).figures
    return cfg


__all__ = ["Recipe", "RECIPES", "BY_KEY", "DEFAULT", "get", "apply_recipe"]
