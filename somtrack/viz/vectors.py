"""Metric-direction arrows on the SOM.

The macro drew, for every metric, an arrow whose direction was the
range-normalised first moment of that metric's component plane about a
reference node.  The idea is sound and genuinely useful -- it is the one view
that tells a reader *which way on the map is faster / straighter / burstier* --
but the moment formulation has no goodness-of-fit, so a metric whose plane is
noise still gets a confident arrow.

Here the same picture is drawn from a weighted plane fit, which yields the
direction of steepest increase **and** an R^2.  Arrows below the R^2 threshold
are drawn dashed and pale rather than silently trusted.  The original
formulation stays available as ``method="legacy_moment"`` in
:func:`somtrack.association.map_gradients`.
"""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ..association import MapGradients
from ..config import FigureConfig
from ..preprocess import PreparedData
from ..som import SomResult
from . import hexgeom as hg
from .maps import _centres, _group_fracs, _map_aspect, _node_hues, _present_count
from . import palette as pal
from .style import (Panel, caption_for, composition_color, group_colors,
                    make_figure)


def _background(res: SomResult, prep: PreparedData, cfg: FigureConfig) -> list:
    F, load = _group_fracs(res, prep)
    hues = _node_hues(prep, cfg)
    return [composition_color(F[i], hues, load[i], _present_count(F[i], cfg),
                              prep.n_groups, cfg)
            for i in range(res.n_nodes)]


def plot_gradient_summary(
    res: SomResult,
    prep: PreparedData,
    grads: MapGradients,
    cfg: FigureConfig,
    min_magnitude: float | None = None,
    min_r2: float = 0.30,
    max_arrows: int = 14,
    on_composition: bool = True,
) -> Panel:
    """All metric directions as arrows from the centre of the map."""
    centres = _centres(res)
    origin = centres.mean(axis=0)
    thr = cfg.axis_min_length if min_magnitude is None else min_magnitude

    order = np.argsort(-grads.magnitude)
    shown = [j for j in order
             if grads.magnitude[j] >= thr * grads.magnitude.max()][:max_arrows]
    if not shown:
        shown = list(order[: min(6, len(order))])

    span = min(np.ptp(centres[:, 0]), abs(np.ptp(centres[:, 1])))
    L = max(span * cfg.axis_max_length_factor, 1.2)

    legend_mm = 48.0
    w = cfg.figure_width_mm

    n_weak = int(np.sum(np.asarray([grads.r2[j] for j in shown]) < min_r2))
    method_txt = ("plane fit over node positions" if grads.method == "gradient"
                  else "range-normalised first moment (legacy v1.2 formulation)")
    cap = caption_for(
        "Direction in which each metric increases across the map.",
        f"Arrow direction is the direction of steepest increase ({method_txt}); "
        "arrow length is the gradient strength, rescaled so the steepest metric "
        "reaches the map edge. Two arrows pointing the same way identify metrics "
        "that co-vary; an arrow pointing at a group's territory on the composition "
        "map identifies a metric that separates that group. "
        + (f"Dashed pale arrows have plane-fit R^2 < {min_r2:g} and should not be "
           "read as a smooth trend. " if n_weak else ""),
        f"{len(shown)} of {len(grads.features)} metrics shown.",
    )
    p = make_figure(w, None, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res, pad=2.6))

    if on_composition:
        face = _background(res, prep, cfg)
        hg.draw_hexes(p.ax, centres, face, edgecolor=cfg.hex_edgecolor,
                      linewidth=0.3, alpha=0.45)
    else:
        hg.draw_hexes(p.ax, centres, ["#f6f6f6"] * res.n_nodes,
                      edgecolor="#dddddd", linewidth=0.3)

    # Arrows are ordered by gradient strength, which is ordinal, so a
    # perceptually uniform *sequential* map is the correct encoding.  The
    # previous `turbo` is a rainbow map: it implies boundaries that are not in
    # the data and is unreadable with a colour-vision deficiency.
    cmap = pal.sequential(cfg)
    handles = []
    placed: list[tuple[float, float]] = []      # (angle deg, label radius)
    for rank, j in enumerate(shown):
        d = grads.direction[j]
        if not np.any(d):
            continue
        # screen Y is flipped relative to lattice Y
        vec = np.array([d[0], -d[1]]) * grads.magnitude[j] * L
        color = cmap(0.08 + 0.80 * rank / max(len(shown) - 1, 1))
        weak = grads.r2[j] < min_r2 if np.isfinite(grads.r2[j]) else False
        p.ax.annotate(
            "", xy=origin + vec, xytext=origin,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6 if not weak else 1.0,
                            alpha=0.45 if weak else 0.95,
                            linestyle="--" if weak else "-",
                            shrinkA=0, shrinkB=0, mutation_scale=11),
            zorder=8,
        )

        # Push the label out along its own arrow until it clears the labels
        # already placed at a similar bearing.  Anything that still cannot fit
        # inside the map goes unlabelled -- the legend lists every arrow with
        # its colour and R2, so nothing is lost but the clutter.
        ang = np.degrees(np.arctan2(vec[1], vec[0]))
        norm = float(np.hypot(*vec)) or 1e-9
        unit = vec / norm
        radius = norm * 1.06
        limit = L * 1.12
        while radius <= limit and any(
            abs((ang - a + 180) % 360 - 180) < 12 and abs(radius - rr) < 0.24 * L
            for a, rr in placed
        ):
            radius += 0.26 * L

        handles.append(Line2D([], [], color=color, lw=1.6,
                              linestyle="--" if weak else "-",
                              label=f"{grads.features[j]}  "
                                    + (f"R2={grads.r2[j]:.2f}"
                                       if np.isfinite(grads.r2[j]) else "")))
        if radius > limit:
            continue

        placed.append((ang, radius))
        tip = origin + unit * radius
        rot = ang + 180 if (ang > 90 or ang < -90) else ang
        p.ax.text(tip[0], tip[1], grads.features[j], rotation=rot,
                  rotation_mode="anchor", ha="left" if vec[0] >= 0 else "right",
                  va="center", fontsize=cfg.base_font_size * 0.8, color=color,
                  zorder=9)
        if radius > norm * 1.2:                 # leader line back to the arrow tip
            p.ax.plot([origin[0] + unit[0] * norm, tip[0]],
                      [origin[1] + unit[1] * norm, tip[1]],
                      color=color, lw=0.5, ls=":", alpha=0.6, zorder=7)

    p.ax.plot(*origin, marker="o", ms=3, color="#222222", zorder=10)
    hg.set_map_limits(p.ax, centres, pad=2.6)
    p.ax.set_title("Metric gradient directions", pad=4)
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Metric (strongest first)")
    return p


def plot_gradient_field(
    res: SomResult,
    prep: PreparedData,
    cfg: FigureConfig,
    feature: str,
    in_original_units: bool = True,
) -> Panel:
    """Local gradient of one metric at every node, over its component plane.

    This is the honest version of the macro's "axis for all nodes" slice stack:
    instead of one global arrow repeated per node, each node gets the local
    slope estimated from its own lattice neighbours.
    """
    j = res.feature_names.index(feature)
    W = prep.inverse(res.codebook) if in_original_units else res.codebook
    v = W[:, j]
    centres = _centres(res)

    L = res.lattice
    nz = L.dist[L.dist > 1e-9]
    step = float(np.min(nz)) if nz.size else 1.0
    adj = (L.dist > 1e-9) & (L.dist <= step * 1.05)

    grad = np.zeros((res.n_nodes, 2))
    for i in range(res.n_nodes):
        nb = np.flatnonzero(adj[i])
        if nb.size < 2:
            continue
        A = np.column_stack([centres[nb, 0] - centres[i, 0],
                             centres[nb, 1] - centres[i, 1],
                             np.ones(nb.size)])
        coef, *_ = np.linalg.lstsq(A, v[nb] - v[i], rcond=None)
        grad[i] = coef[:2]

    gmax = float(np.hypot(grad[:, 0], grad[:, 1]).max())
    scale = (hg.HEX_RADIUS * 1.5 / gmax) if gmax > 0 else 0.0

    cmap = plt.get_cmap("viridis")
    lo, hi = float(v.min()), float(v.max())
    face = [cmap(float((x - lo) / (hi - lo)) if hi > lo else 0.5) for x in v]

    legend_mm = 32.0
    w = cfg.figure_width_mm * 0.8

    unit = prep.source.units.get(feature, "") if prep.source else ""
    cap = caption_for(
        f"Local gradient of '{feature}' across the map.",
        "Background is the component plane; each arrow is the direction and "
        "steepness of the local increase, fitted from that node's immediate "
        "neighbours. Convergent arrows mark a local maximum of the metric.",
        f"range {lo:.3g} - {hi:.3g}" + (f" {unit}." if unit else "."),
    )
    p = make_figure(w, None, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))
    hg.draw_hexes(p.ax, centres, face, edgecolor="#ffffff", linewidth=0.25)
    p.ax.quiver(centres[:, 0], centres[:, 1], grad[:, 0] * scale, grad[:, 1] * scale,
                angles="xy", scale_units="xy", scale=1.0, width=0.004,
                color="#111111", alpha=0.8, zorder=6)
    hg.set_map_limits(p.ax, centres)
    p.ax.set_title(feature, pad=4)

    from .maps import _side_colorbar

    _side_colorbar(p, cmap, lo, hi, f"{feature}" + (f"\n[{unit}]" if unit else ""), cfg)
    return p


def plot_gradient_table(grads: MapGradients, cfg: FigureConfig,
                        top: int = 20) -> Panel:
    """Compass-rose summary: metric angle vs. gradient strength."""
    t = grads.table().head(top)
    ang = np.radians(t["angle_deg"].to_numpy())
    mag = t["magnitude"].to_numpy()
    names = t["feature"].tolist()

    cap = caption_for(
        "Metric gradients as a compass.",
        "Angle is the direction of steepest increase on the map, radius is the "
        "gradient strength. Metrics clustered at the same angle carry redundant "
        "information; metrics at opposite angles trade off against each other.",
        f"top {len(names)} of {len(grads.features)} metrics.",
    )
    w = cfg.figure_width_mm * 0.62
    p = make_figure(w, w * 0.95, cfg, caption=cap, left_mm=10, right_mm=10,
                    top_mm=6, bottom_mm=6)
    p.ax.remove()
    box = (0.12, 0.16, 0.76, 0.76)
    ax = p.fig.add_axes(box, projection="polar")
    p.ax = ax
    cmap = pal.sequential(cfg)          # rank is ordinal -- sequential, not rainbow
    for i, (a, m, nm) in enumerate(zip(ang, mag, names)):
        c = cmap(0.08 + 0.80 * i / max(len(names) - 1, 1))
        ax.annotate("", xy=(a, m), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=1.3, mutation_scale=9))
        ax.text(a, m * 1.08, nm, fontsize=cfg.base_font_size * 0.72,
                color=c, ha="center", va="center")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "0.5", "", "1.0"], fontsize=cfg.base_font_size * 0.7)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["map right", "map up", "map left", "map down"],
                       fontsize=cfg.base_font_size * 0.8)
    ax.grid(alpha=0.3, lw=0.4)
    ax.set_title("Metric gradient compass", pad=10)
    return p


__all__ = ["plot_gradient_summary", "plot_gradient_field", "plot_gradient_table"]
