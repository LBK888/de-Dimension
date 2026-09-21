"""Figure layer: publication-ready panels for every analysis product.

Four groups of panels:

* the SOM figures ported from v1.2 (:mod:`maps`, :mod:`groups`, :mod:`vectors`)
* the projections and their diagnostics (:mod:`embed`, :mod:`reliability`)
* the statistics that decide what the figures are allowed to claim (:mod:`stats`)
* the comparison and per-metric panels (:mod:`compare`, :mod:`estimation`)

:mod:`palette` holds the colour policy the whole layer goes through, including
the decision to stop using colour once there are more groups than any palette
can keep apart.
"""

from .anim import animate_figures, animate_training, ffmpeg_available
from .assoc import (plot_association_overview, plot_group_metric_heatmap,
                    plot_metric_correlation, plot_metric_distributions,
                    plot_node_cluster_heatmap)
from .compare import (plot_agreement, plot_method_grid, plot_scan_surface,
                      plot_scan_thumbnails)
from .embed import (plot_embedding, plot_embedding_quality, plot_embedding_row,
                    plot_supervised_check)
from .estimation import plot_effect_forest, plot_estimation, plot_superplot
from .groups import (alpha_for_counts, plot_group_enrichment_map,
                     plot_group_maps, plot_single_group_map)
from .maps import (plot_codebook_kymograph, plot_component_planes,
                   plot_composition_map, plot_group_pies, plot_hit_map,
                   plot_k_selection, plot_node_clusters, plot_training_curves,
                   plot_u_matrix)
from .palette import (PaletteDecision, categorical, diverging, palette_proof,
                      sequential, simulate_cvd, symmetric_norm)
from .reliability import (plot_point_reliability, plot_rnx_curves,
                          plot_seed_stability, plot_shepard)
from .stats import (plot_activation_patterns, plot_classifiability,
                    plot_importance, plot_pairwise_separation,
                    plot_separation_summary, plot_verdict)
from .style import Panel, apply_style, caption_for, group_colors, make_figure
from .vectors import (plot_gradient_field, plot_gradient_summary,
                      plot_gradient_table)

__all__ = [
    # infrastructure
    "apply_style", "make_figure", "Panel", "group_colors", "caption_for",
    "categorical", "PaletteDecision", "sequential", "diverging",
    "symmetric_norm", "simulate_cvd", "palette_proof",
    # SOM
    "plot_composition_map", "plot_group_pies", "plot_u_matrix", "plot_hit_map",
    "plot_component_planes", "plot_node_clusters", "plot_k_selection",
    "plot_training_curves", "plot_codebook_kymograph",
    "plot_group_maps", "plot_single_group_map", "plot_group_enrichment_map",
    "alpha_for_counts",
    "plot_gradient_summary", "plot_gradient_field", "plot_gradient_table",
    # projections
    "plot_embedding", "plot_embedding_row", "plot_embedding_quality",
    "plot_supervised_check",
    "plot_rnx_curves", "plot_point_reliability", "plot_seed_stability",
    "plot_shepard",
    # comparison
    "plot_method_grid", "plot_agreement", "plot_scan_surface",
    "plot_scan_thumbnails",
    # statistics
    "plot_verdict", "plot_separation_summary", "plot_pairwise_separation",
    "plot_classifiability", "plot_activation_patterns", "plot_importance",
    # per metric
    "plot_superplot", "plot_estimation", "plot_effect_forest",
    "plot_association_overview", "plot_group_metric_heatmap",
    "plot_node_cluster_heatmap", "plot_metric_correlation",
    "plot_metric_distributions",
    # animation
    "animate_training", "animate_figures", "ffmpeg_available",
]
