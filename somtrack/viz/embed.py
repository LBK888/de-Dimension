"""PCA / t-SNE / UMAP figures.

Every embedding is drawn the same way so the three are directly comparable:
group-coloured points, a 95 % confidence ellipse or convex hull per group, and
metric-direction arrows placed in whichever corner of the plot is emptiest.
Trustworthiness and continuity are printed on the panel, because a t-SNE or
UMAP picture without them is not interpretable.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from ..config import FigureConfig
from ..analysis.projection import ProjectionResult as EmbeddingResult
from ..preprocess import PreparedData
from . import palette as pal
from .style import (Panel, caption_for, group_colors, make_figure,
                    scalebar_free_corner)


def _confidence_ellipse(ax, x, y, color, n_std: float = 2.0, **kw):
    if x.size < 3:
        return
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[0] <= 0:
        return
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=theta,
                         facecolor=color, edgecolor=color, **kw))


def plot_embedding(
    emb: EmbeddingResult,
    prep: PreparedData,
    cfg: FigureConfig,
    show_arrows: bool = True,
    show_ellipses: bool = True,
    max_arrows: int = 8,
    highlight: list[str] | None = None,
) -> Panel:
    Y = emb.coords
    colors = group_colors(prep.n_groups, cfg.palette)

    legend_mm = 42.0
    w = cfg.figure_width_mm * 0.78
    h = w * 0.80
    q = []
    if np.isfinite(emb.trustworthiness):
        q.append(f"trustworthiness {emb.trustworthiness:.3f}")
    if np.isfinite(emb.continuity):
        q.append(f"continuity {emb.continuity:.3f}")
    if emb.explained_variance is not None:
        q.append(f"variance explained {emb.explained_variance[:2].sum() * 100:.1f}%")

    cap = caption_for(
        (f"{emb.name} projection of {prep.n_features} locomotion metrics."
         + (f"  {emb.caveat}" if getattr(emb, "caveat", "") else "")),
        "Points are individual samples coloured by treatment; shaded ellipses cover "
        "two standard deviations of each group. "
        + (emb.axis_caption if show_arrows else "")
        + (" Distances between well-separated clusters in t-SNE and UMAP are not "
           "meaningful; only local neighbourhoods are."
           if emb.name in ("t-SNE", "UMAP") else ""),
        f"n = {prep.n_samples}" + ("; " + ", ".join(q) + "." if q else "."),
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm,
                    left_mm=15.0, bottom_mm=13.0)
    ax = p.ax

    for g in range(prep.n_groups):
        m = prep.group_codes == g
        if show_ellipses and m.sum() >= 3:
            _confidence_ellipse(ax, Y[m, 0], Y[m, 1], colors[g],
                                alpha=0.12, lw=0.8, zorder=1)
        ax.scatter(Y[m, 0], Y[m, 1], s=cfg.point_size, c=colors[g],
                   edgecolors="white", linewidths=0.4, zorder=3,
                   label=f"{prep.group_values[g]} (n={int(m.sum())})")

    ax.set_xlabel(emb.axis_labels[0])
    ax.set_ylabel(emb.axis_labels[1])
    ax.set_title(emb.name, pad=4)
    ax.grid(True, alpha=0.18, lw=0.4)

    if show_arrows and emb.feature_axes.size:
        _draw_feature_arrows(ax, emb, cfg, max_arrows, highlight)

    handles, labels = ax.get_legend_handles_labels()
    p.legend_from(handles, labels, title="Group")
    if show_arrows:
        p.legend_ax.text(
            0.0, 0.02,
            "Arrows: metric direction\n"
            + ("(PCA loadings)" if emb.axes_are_loadings else "(post-hoc correlation)"),
            transform=p.legend_ax.transAxes, va="bottom",
            fontsize=cfg.base_font_size * 0.82, color="#444444",
        )
    return p


def _draw_feature_arrows(ax, emb: EmbeddingResult, cfg: FigureConfig,
                         max_arrows: int, highlight: list[str] | None) -> None:
    A = emb.feature_axes[:, :2]
    mag = np.hypot(A[:, 0], A[:, 1])
    if highlight:
        idx = [i for i, n in enumerate(emb.feature_names) if n in highlight]
        idx = sorted(idx, key=lambda i: -mag[i])[:max_arrows]
    else:
        idx = list(np.argsort(-mag)[:max_arrows])
    if not idx:
        return

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    corner = scalebar_free_corner(ax, emb.coords[:, 0], emb.coords[:, 1])
    fx = 0.24 if "left" in corner else 0.76
    fy = 0.76 if "upper" in corner else 0.24
    ox, oy = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
    R = 0.19 * min(x1 - x0, y1 - y0)

    scale = R / max(mag[idx].max(), 1e-9)
    ax.add_patch(plt.Circle((ox, oy), R, fill=False, ec="#bbbbbb", lw=0.5,
                            ls=":", zorder=4))
    for i in idx:
        vx, vy = A[i, 0] * scale, A[i, 1] * scale
        ax.annotate("", xy=(ox + vx, oy + vy), xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=0.9,
                                    mutation_scale=7), zorder=5)
        ang = np.degrees(np.arctan2(vy, vx))
        if ang > 90 or ang < -90:
            ang += 180
        ax.text(ox + vx * 1.13, oy + vy * 1.13, emb.feature_names[i],
                fontsize=cfg.base_font_size * 0.68, color="#222222",
                ha="center", va="center", rotation=ang, rotation_mode="anchor",
                zorder=6)


def plot_embedding_row(
    embeddings: dict[str, EmbeddingResult],
    prep: PreparedData,
    cfg: FigureConfig,
    show_arrows: bool = False,
) -> Panel:
    """PCA, t-SNE and UMAP side by side on one row for direct comparison."""
    names = _ordered(embeddings)
    if not names:
        raise ValueError("No embeddings to draw.")
    colors = group_colors(prep.n_groups, cfg.palette)

    legend_mm = 38.0
    w = cfg.figure_width_mm
    cell = (w - legend_mm - 14.0) / len(names)
    h = cell + 26.0

    bits = []
    for n in names:
        e = embeddings[n]
        if np.isfinite(e.trustworthiness):
            bits.append(f"{n} T={e.trustworthiness:.2f}/C={e.continuity:.2f}")
    cap = caption_for(
        "Linear and non-linear projections of the same feature matrix.",
        "PCA preserves global distances and is directly interpretable; t-SNE "
        "emphasises local neighbourhoods; UMAP sits between the two. Structure "
        "that survives all three is robust, structure visible in only one is a "
        "property of that algorithm.",
        ("; ".join(bits) + " (trustworthiness / continuity)." if bits else ""),
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm,
                    left_mm=6.0, right_mm=4.0, top_mm=6.0, bottom_mm=10.0)
    p.ax.set_axis_off()
    box = p.ax.get_position()

    for i, n in enumerate(names):
        e = embeddings[n]
        sub = p.fig.add_axes((box.x0 + i * box.width / len(names) + 0.012,
                              box.y0, box.width / len(names) * 0.88, box.height * 0.92))
        for g in range(prep.n_groups):
            m = prep.group_codes == g
            if m.sum() >= 3:
                _confidence_ellipse(sub, e.coords[m, 0], e.coords[m, 1], colors[g],
                                    alpha=0.12, lw=0.7, zorder=1)
            sub.scatter(e.coords[m, 0], e.coords[m, 1], s=cfg.point_size * 0.7,
                        c=colors[g], edgecolors="white", linewidths=0.3, zorder=3)
        sub.set_xlabel(e.axis_labels[0], fontsize=cfg.base_font_size * 0.85)
        sub.set_ylabel(e.axis_labels[1], fontsize=cfg.base_font_size * 0.85)
        sub.set_title(n, fontsize=cfg.base_font_size, pad=3)
        sub.set_xticks([]); sub.set_yticks([])
        sub.grid(True, alpha=0.15, lw=0.4)
        if show_arrows:
            _draw_feature_arrows(sub, e, cfg, 5, None)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=colors[g], markeredgecolor="white",
                      label=f"{prep.group_values[g]} (n={int((prep.group_codes == g).sum())})")
               for g in range(prep.n_groups)]
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Group")
    return p


def plot_embedding_quality(embeddings: dict[str, EmbeddingResult],
                           som_quality: dict[str, float] | None,
                           cfg: FigureConfig) -> Panel:
    """Bar chart comparing neighbourhood preservation across methods."""
    names, tw, ct = [], [], []
    for n, e in embeddings.items():
        names.append(n)
        tw.append(e.trustworthiness)
        ct.append(e.continuity)

    cap = caption_for(
        "Neighbourhood preservation of each projection.",
        "Trustworthiness penalises neighbours that the projection invented; "
        "continuity penalises neighbours it lost. Both range from 0 to 1 and both "
        "should be reported whenever a t-SNE or UMAP figure is used as evidence.",
        (f"SOM topographic error = {som_quality['topographic_error']:.3f}."
         if som_quality and "topographic_error" in som_quality else ""),
    )
    p = make_figure(cfg.figure_width_mm * 0.55, 58.0, cfg, caption=cap,
                    left_mm=16.0, right_mm=6.0)
    x = np.arange(len(names))
    p.ax.bar(x - 0.19, tw, width=0.36, color="#0072B2", label="trustworthiness")
    p.ax.bar(x + 0.19, ct, width=0.36, color="#D55E00", label="continuity")
    p.ax.set_xticks(x)
    p.ax.set_xticklabels(names)
    p.ax.set_ylim(0, 1.05)
    p.ax.set_ylabel("neighbourhood preservation")
    p.ax.axhline(1.0, color="#999999", lw=0.5, ls=":")
    p.ax.legend(fontsize=cfg.base_font_size * 0.85, loc="lower right")
    return p


def _ordered(embeddings: dict) -> list[str]:
    """Display order: linear, then non-linear, then supervised.

    2.0 hard-coded ``("PCA", "t-SNE", "UMAP")`` here, which silently dropped
    every method added afterwards.  The order now comes from the registry, so a
    new method appears in the row without this file being touched.
    """
    from ..analysis.registry import FAMILY_ORDER, all_methods

    rank = {}
    for spec in all_methods():
        rank[spec.label] = (FAMILY_ORDER.index(spec.family)
                            if spec.family in FAMILY_ORDER else 99)
    return sorted(embeddings, key=lambda n: (rank.get(n, 50), n))


def plot_supervised_check(emb: EmbeddingResult, prep: PreparedData,
                          cfg: FigureConfig) -> Panel:
    """A supervised projection in-sample and out-of-fold, side by side.

    This is the panel that tells the difference between a real separation and a
    projection that was handed the answer.  The left half is fitted on all the
    data, which is what a supervised method is normally shown as; the right half
    places each sample using axes fitted without it.  If the groups only
    separate on the left, they do not separate.
    """
    if emb.oof_coords is None:
        raise ValueError(f"{emb.name} has no out-of-fold coordinates.")

    deco = pal.categorical(prep.n_groups, cfg)
    cv = emb.cv
    verdict = ""
    if cv is not None and np.isfinite(getattr(cv, "balanced_accuracy", np.nan)):
        verdict = (f"Cross-validated balanced accuracy {cv.balanced_accuracy:.2f} "
                   f"against chance {cv.chance:.2f}, permutation "
                   f"p = {cv.permutation_p:.4f}.")

    cap = caption_for(
        f"{emb.name}: what the projection shows, and what survives validation.",
        "Left: fitted on all the samples, which is how a supervised projection is "
        "usually presented. Right: every sample placed by axes fitted without it. "
        "A supervised projection separates the groups by construction and does so "
        "on random data too, so only the right-hand panel and the number below it "
        "are evidence.",
        verdict)

    legend_mm = 40.0
    w = cfg.figure_width_mm
    cell = (w - legend_mm - 18.0) / 2.0
    p = make_figure(w, cell, cfg, caption=cap, legend_width_mm=legend_mm,
                    left_mm=8.0, right_mm=4.0, top_mm=7.0, bottom_mm=10.0,
                    height_is_axes=True)
    p.ax.set_axis_off()
    box = p.ax.get_position()

    for i, (title, Y) in enumerate((("fitted on all samples", emb.coords),
                                    ("out-of-fold", emb.oof_coords))):
        sub = p.fig.add_axes((box.x0 + i * box.width / 2 + 0.012, box.y0,
                              box.width / 2 * 0.86, box.height * 0.90))
        ok = np.all(np.isfinite(Y), axis=1)
        for g in range(prep.n_groups):
            m = (prep.group_codes == g) & ok
            if m.sum() >= 3:
                _confidence_ellipse(sub, Y[m, 0], Y[m, 1], deco.for_group(g),
                                    alpha=0.12, lw=0.7, zorder=1)
            sub.scatter(Y[m, 0], Y[m, 1], s=cfg.point_size * 0.8,
                        c=deco.for_group(g), marker=pal.marker_for(g),
                        edgecolors="white", linewidths=0.35, zorder=3)
        sub.set_xticks([]); sub.set_yticks([])
        sub.set_xlabel(emb.axis_labels[0], fontsize=cfg.base_font_size * 0.82)
        sub.set_ylabel(emb.axis_labels[1], fontsize=cfg.base_font_size * 0.82)
        sub.set_title(title, fontsize=cfg.base_font_size, pad=3)
        sub.grid(True, alpha=0.14, lw=0.4)
        for spine in sub.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2 if i == 0 else 0.5)
            spine.set_color("#D55E00" if i == 0 else "#cccccc")

    handles = [Line2D([], [], marker=pal.marker_for(g), linestyle="none",
                      markersize=5, markerfacecolor=deco.for_group(g),
                      markeredgecolor="white",
                      label=f"{prep.group_values[g]} "
                            f"(n={int((prep.group_codes == g).sum())})")
               for g in range(prep.n_groups)]
    p.legend_from(handles, [h.get_label() for h in handles], title="Group")
    p.legend_ax.text(0.0, 0.02,
                     "Orange frame: labels were\nused to build these axes.",
                     transform=p.legend_ax.transAxes, va="bottom",
                     fontsize=cfg.base_font_size * 0.76, color="#8a3b00")
    return p


__all__ = ["plot_embedding", "plot_embedding_row", "plot_embedding_quality",
           "plot_supervised_check"]
