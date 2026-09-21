"""Figures that connect metrics to the grouping -- the biology-reading layer.

A clustering figure shows *that* samples separate.  These show *which measured
quantity does the separating*, with effect sizes and corrected p values, so the
result can be written up rather than only admired.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..association import AssociationBundle, ImportanceResult
from ..config import FigureConfig
from ..preprocess import PreparedData
from .style import Panel, caption_for, group_colors, make_figure


def _stars(q: float) -> str:
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


# ==========================================================================
def plot_association_overview(bundle: AssociationBundle, prep: PreparedData,
                              cfg: FigureConfig, top: int = 15) -> Panel:
    """Univariate effect size and multivariate importance, side by side."""
    uni = bundle.univariate.head(top).iloc[::-1]
    imp = bundle.importance

    cap = caption_for(
        "Which metrics carry the group difference.",
        "Left: univariate effect size (eta^2, the share of a metric's variance "
        "explained by treatment) with Benjamini-Hochberg corrected significance. "
        "Right: cross-validated random-forest permutation importance, which also "
        "credits metrics that only matter in combination with others. A metric high "
        "on the right but low on the left is an interaction effect, and a metric "
        "high on the left but low on the right is redundant with a stronger one.",
        f"random forest accuracy {imp.cv_accuracy:.2f} +/- {imp.cv_accuracy_sd:.2f} "
        f"over {imp.n_splits} folds (chance {imp.baseline_accuracy:.2f}); "
        f"* q<0.05, ** q<0.01, *** q<0.001.",
    )
    w = cfg.figure_width_mm
    h = float(np.clip(len(uni) * 4.4 + 26.0, 55.0, 250.0))
    p = make_figure(w, h, cfg, caption=cap, left_mm=4.0, right_mm=4.0, top_mm=7.0)
    p.ax.set_axis_off()
    box = p.ax.get_position()

    ax1 = p.fig.add_axes((box.x0 + 0.20 * box.width, box.y0, box.width * 0.30, box.height))
    ax2 = p.fig.add_axes((box.x0 + 0.66 * box.width, box.y0, box.width * 0.30, box.height))

    y = np.arange(len(uni))
    ax1.barh(y, uni["eta2"], color="#0072B2", height=0.7)
    ax1.set_yticks(y)
    ax1.set_yticklabels(uni["feature"], fontsize=cfg.base_font_size * 0.85)
    ax1.set_xlabel(r"effect size $\eta^2$")
    ax1.set_title("Univariate (ANOVA)", fontsize=cfg.base_font_size, pad=3)
    xmax = float(max(uni["eta2"].max(), 1e-6))
    for i, (v, q) in enumerate(zip(uni["eta2"], uni["anova_q"])):
        s = _stars(q)
        if s:
            ax1.text(v + xmax * 0.02, i, s, va="center",
                     fontsize=cfg.base_font_size * 0.85, color="#333333")
    ax1.set_xlim(0, xmax * 1.18)

    it = imp.table.set_index("feature")
    order = [f for f in uni["feature"] if f in it.index]
    vals = it.loc[order, "importance"].to_numpy() if order else np.zeros(len(uni))
    errs = it.loc[order, "sd"].to_numpy() if order else np.zeros(len(uni))
    ax2.barh(np.arange(len(order)), vals, xerr=errs, height=0.7,
             color="#D55E00", error_kw=dict(lw=0.6, ecolor="#555555"))
    ax2.set_yticks(np.arange(len(order)))
    ax2.set_yticklabels([""] * len(order))
    ax2.set_xlabel("permutation importance")
    ax2.set_title("Multivariate (random forest)", fontsize=cfg.base_font_size, pad=3)
    ax2.axvline(0, color="#999999", lw=0.5)
    return p


# ==========================================================================
def plot_group_metric_heatmap(prep: PreparedData, univariate: pd.DataFrame,
                              cfg: FigureConfig, top: int = 22) -> Panel:
    """Z-scored group means per metric, ordered by effect size."""
    feats = univariate.head(top)["feature"].tolist()
    labels = prep.group_values
    src = prep.source

    M = np.zeros((len(feats), len(labels)))
    for i, f in enumerate(feats):
        v = (src.frame.loc[prep.keep_mask, f].to_numpy(float)
             if src is not None and f in src.frame.columns
             else prep.X[:, prep.feature_names.index(f)])
        mu, sd = np.nanmean(v), np.nanstd(v)
        for j in range(len(labels)):
            g = v[prep.group_codes == j]
            M[i, j] = (np.nanmean(g) - mu) / sd if sd > 0 else 0.0

    lim = float(max(np.abs(M).max(), 1e-6))
    w = cfg.figure_width_mm * 0.72
    h = float(np.clip(len(feats) * 4.4 + 30.0, 60.0, 280.0))
    cap = caption_for(
        "Group profile of the most discriminating metrics.",
        "Each cell is a group mean expressed as a z score against the pooled "
        "distribution of that metric, so rows are comparable despite different "
        "units. Rows are ordered by eta^2; asterisks mark the FDR-corrected ANOVA "
        "result for the row as a whole.",
        "* q<0.05, ** q<0.01, *** q<0.001.",
    )
    p = make_figure(w, h, cfg, caption=cap, left_mm=40.0, right_mm=20.0, top_mm=12.0)
    im = p.ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    p.ax.set_xticks(np.arange(len(labels)))
    p.ax.set_xticklabels([str(x) for x in labels], rotation=30, ha="right")
    p.ax.set_yticks(np.arange(len(feats)))

    qmap = univariate.set_index("feature")["anova_q"].to_dict()
    p.ax.set_yticklabels([f"{f} {_stars(qmap.get(f, np.nan))}" for f in feats],
                         fontsize=cfg.base_font_size * 0.85)
    p.ax.set_xlabel("group")
    for spine in p.ax.spines.values():
        spine.set_visible(False)
    p.ax.tick_params(length=0)
    cb = p.fig.colorbar(im, ax=p.ax, fraction=0.035, pad=0.03)
    cb.set_label("group mean (z score)")
    return p


# ==========================================================================
def plot_node_cluster_heatmap(profile: pd.DataFrame, cfg: FigureConfig,
                              top: int = 20) -> Panel:
    """What distinguishes each node cluster, in z-scored codebook units."""
    if profile is None or profile.empty:
        raise ValueError("No node-cluster profile available.")
    cols = [c for c in profile.columns if c.startswith("cluster_")]
    sub = profile.head(top)
    M = sub[cols].to_numpy(float)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    Z = np.divide(M - mu, np.where(sd > 0, sd, 1.0))

    lim = float(max(np.abs(Z).max(), 1e-6))
    w = cfg.figure_width_mm * 0.7
    h = float(np.clip(len(sub) * 4.4 + 30.0, 60.0, 280.0))
    cap = caption_for(
        "Behavioural signature of each node cluster.",
        "Rows are the metrics that differ most between node clusters (one-way "
        "ANOVA over codebook vectors); cells are the cluster mean, z scored across "
        "clusters. Reading a column top to bottom gives a verbal description of "
        "that cluster -- for example high speed, low turning, long runs.",
        "Cluster numbering matches the node-cluster map.",
    )
    p = make_figure(w, h, cfg, caption=cap, left_mm=40.0, right_mm=20.0, top_mm=12.0)
    im = p.ax.imshow(Z, cmap="PuOr_r", vmin=-lim, vmax=lim, aspect="auto")
    p.ax.set_xticks(np.arange(len(cols)))
    p.ax.set_xticklabels([c.replace("cluster_", "C") for c in cols])
    p.ax.set_yticks(np.arange(len(sub)))
    labels = [f"{r.feature} {_stars(r.get('q', np.nan))}" for _, r in sub.iterrows()]
    p.ax.set_yticklabels(labels, fontsize=cfg.base_font_size * 0.85)
    p.ax.set_xlabel("node cluster")
    for spine in p.ax.spines.values():
        spine.set_visible(False)
    p.ax.tick_params(length=0)
    cb = p.fig.colorbar(im, ax=p.ax, fraction=0.035, pad=0.03)
    cb.set_label("cluster mean (z across clusters)")
    return p


# ==========================================================================
def plot_metric_correlation(prep: PreparedData, cfg: FigureConfig,
                            features: list[str] | None = None,
                            cluster_order: bool = True) -> Panel:
    """Correlation matrix of the metrics, optionally dendrogram-ordered."""
    names = features or prep.feature_names
    idx = [prep.feature_names.index(n) for n in names if n in prep.feature_names]
    names = [prep.feature_names[i] for i in idx]
    X = prep.X[:, idx]
    with np.errstate(invalid="ignore"):
        C = np.nan_to_num(np.corrcoef(X, rowvar=False))

    if cluster_order and len(names) > 2:
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform

            d = np.clip(1 - np.abs(C), 0, 2)
            np.fill_diagonal(d, 0.0)
            order = dendrogram(linkage(squareform(d, checks=False), "average"),
                               no_plot=True)["leaves"]
            C = C[np.ix_(order, order)]
            names = [names[i] for i in order]
        except Exception:
            pass

    w = cfg.figure_width_mm * 0.78
    cap = caption_for(
        "Redundancy between metrics.",
        "Pearson correlation over samples, ordered by hierarchical clustering of "
        "1 - |r|. Blocks of dark cells are metrics measuring the same thing; keeping "
        "all of them lets one behavioural axis dominate the SOM distance and is a "
        "common reason a map fails to separate treatments.",
        f"{len(names)} metrics.",
    )
    p = make_figure(w, w * 0.94, cfg, caption=cap, left_mm=34.0, right_mm=20.0, top_mm=30.0)
    im = p.ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    p.ax.set_xticks(np.arange(len(names)))
    p.ax.set_xticklabels(names, rotation=90, fontsize=cfg.base_font_size * 0.7)
    p.ax.set_yticks(np.arange(len(names)))
    p.ax.set_yticklabels(names, fontsize=cfg.base_font_size * 0.7)
    p.ax.tick_params(length=0)
    for spine in p.ax.spines.values():
        spine.set_visible(False)
    cb = p.fig.colorbar(im, ax=p.ax, fraction=0.04, pad=0.02)
    cb.set_label("Pearson r")
    return p


# ==========================================================================
def plot_metric_distributions(prep: PreparedData, features: list[str],
                              cfg: FigureConfig, ncols: int = 4) -> Panel:
    """Per-group distributions of the selected metrics, in original units."""
    src = prep.source
    features = [f for f in features if f in prep.feature_names][:12]
    if not features:
        raise ValueError("No features to draw.")
    colors = group_colors(prep.n_groups, cfg.palette)
    ncols = int(min(ncols, len(features)))
    nrows = int(np.ceil(len(features) / ncols))

    w = cfg.figure_width_mm
    h = float(np.clip(nrows * 32.0 + 18.0, 55.0, 280.0))
    cap = caption_for(
        "Distribution of the leading metrics by group.",
        "Box shows the interquartile range and median, whiskers reach 1.5x IQR, and "
        "every sample is drawn as a jittered point so small n is visible rather than "
        "hidden behind a summary.",
        f"n = {prep.n_samples} samples.",
    )
    p = make_figure(w, h, cfg, caption=cap, left_mm=6.0, right_mm=4.0,
                    top_mm=5.0, bottom_mm=6.0)
    p.ax.set_axis_off()
    box = p.ax.get_position()
    rng = np.random.default_rng(7)

    for k, f in enumerate(features):
        r, c = divmod(k, ncols)
        sub = p.fig.add_axes((
            box.x0 + c * box.width / ncols + 0.035,
            box.y0 + (nrows - 1 - r) * box.height / nrows + 0.03,
            box.width / ncols * 0.72,
            box.height / nrows * 0.66,
        ))
        v = (src.frame.loc[prep.keep_mask, f].to_numpy(float)
             if src is not None and f in src.frame.columns
             else prep.X[:, prep.feature_names.index(f)])
        data = [v[prep.group_codes == g] for g in range(prep.n_groups)]
        data = [d[np.isfinite(d)] for d in data]
        bp = sub.boxplot(data, widths=0.55, showfliers=False, patch_artist=True)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.35)
            patch.set_linewidth(0.6)
        for key in ("medians", "whiskers", "caps"):
            for art in bp[key]:
                art.set_linewidth(0.7)
                art.set_color("#333333")
        for gi, d in enumerate(data):
            if d.size:
                sub.scatter(gi + 1 + (rng.random(d.size) - 0.5) * 0.28, d,
                            s=4, color=colors[gi], alpha=0.75,
                            edgecolors="none", zorder=3)
        unit = src.units.get(f, "") if src else ""
        sub.set_title(f + (f"\n[{unit}]" if unit else ""),
                      fontsize=cfg.base_font_size * 0.82, pad=2)
        sub.set_xticks(np.arange(1, prep.n_groups + 1))
        sub.set_xticklabels([str(x) for x in prep.group_values],
                            rotation=30, ha="right", fontsize=cfg.base_font_size * 0.7)
        sub.tick_params(axis="y", labelsize=cfg.base_font_size * 0.7)
    return p


__all__ = [
    "plot_association_overview", "plot_group_metric_heatmap",
    "plot_node_cluster_heatmap", "plot_metric_correlation",
    "plot_metric_distributions",
]
