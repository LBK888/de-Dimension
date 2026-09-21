"""The statistics layer: does the grouping hold up, and what drives it?

Four questions, in the order a reader will ask them:

1. Do the groups differ at all?            :mod:`omnibus`   (PERMANOVA, PERMDISP, energy)
2. Can new samples be assigned correctly?  :mod:`classify`  (cross-validation + permutation)
3. Which metrics drive it?                 :mod:`interpret` (Haufe patterns, clustered importance)
4. By how much, per metric?                :mod:`effects`   (Hedges' g with bootstrap intervals)

:mod:`blocks` sits underneath all four and decides what "independent" means for
this experiment, and :mod:`verdict` turns the four answers into a paragraph a
non-statistician can act on.
"""

from .blocks import (BlockStructure, CvPlan, analyse_blocks, make_cv,
                     permutation_p, permutations)
from .classify import ClassifyResult, classify, make_estimator
from .effects import bootstrap_ci, estimate_effects, hedges_g
from .interpret import (ImportanceResult, feature_clusters, haufe_patterns,
                        permutation_importance, weight_table)
from .omnibus import (SeparationReport, analyse_separation, energy_test,
                      mahalanobis_centroids, permanova, permdisp)
from .verdict import Verdict, build_verdict

__all__ = [
    "BlockStructure", "analyse_blocks", "make_cv", "CvPlan",
    "permutations", "permutation_p",
    "SeparationReport", "analyse_separation", "permanova", "permdisp",
    "energy_test", "mahalanobis_centroids",
    "ClassifyResult", "classify", "make_estimator",
    "ImportanceResult", "permutation_importance", "haufe_patterns",
    "weight_table", "feature_clusters",
    "hedges_g", "bootstrap_ci", "estimate_effects",
    "Verdict", "build_verdict",
]
