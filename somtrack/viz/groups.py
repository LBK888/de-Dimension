"""Per-group maps with occupancy-graded transparency.

On a large map with many samples, drawing one group's points over the full
colour map leaves them lost among nodes that group never visited.  Fading a
node in proportion to how many of *that group's* samples it holds makes the
group's territory the only saturated region on the map:

===============  ==========  =============
samples in node  opacity     transparency
===============  ==========  =============
0                20 %        80 %
1                40 %        60 %
2                60 %        40 %
>= 3             100 %       0 %
===============  ==========  =============

The thresholds live in :attr:`somtrack.config.FigureConfig.group_alpha_by_hits`
and can be re-tuned without touching this module.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from ..config import FigureConfig
from ..preprocess import PreparedData
from ..som import SomResult
from . import hexgeom as hg
from .maps import _centres, _group_fracs, _map_aspect, _node_hues, _present_count
from .style import (Panel, caption_for, composition_color, group_colors,
                    make_figure)


def alpha_for_counts(counts: np.ndarray, cfg: FigureConfig) -> np.ndarray:
    """Map per-node sample counts to opacity using the configured ladder."""
    a = np.full(counts.shape, cfg.group_alpha_default, dtype=float)
    for n, alpha in sorted(cfg.group_alpha_by_hits.items()):
        a[counts == n] = float(alpha)
    highest = max(cfg.group_alpha_by_hits) if cfg.group_alpha_by_hits else -1
    a[counts > highest] = cfg.group_alpha_default
    return a


def _base_colors(res: SomResult, prep: PreparedData, cfg: FigureConfig) -> list:
    F, load_n = _group_fracs(res, prep)
    hues = _node_hues(prep, cfg)
    return [composition_color(F[i], hues, load_n[i], _present_count(F[i], cfg),
                              prep.n_groups, cfg)
            for i in range(res.n_nodes)]


def _alpha_ladder_text(cfg: FigureConfig) -> str:
    items = sorted(cfg.group_alpha_by_hits.items())
    lines = [f"{n} sample{'s' if n != 1 else ''}: {int(round(a * 100))}% opaque"
             for n, a in items]
    hi = (max(cfg.group_alpha_by_hits) + 1) if cfg.group_alpha_by_hits else 1
    lines.append(f">= {hi} samples: {int(round(cfg.group_alpha_default * 100))}% opaque")
    return "\n".join(lines)


# ==========================================================================
def plot_group_maps(
    res: SomResult,
    prep: PreparedData,
    cfg: FigureConfig,
    by_replicate: bool = False,
    ncols: int = 0,
) -> Panel:
    """Small-multiple maps, one panel per group (or per group x replicate)."""
    centres = _centres(res)
    base = _base_colors(res, prep, cfg)
    colors = group_colors(prep.n_groups, cfg.palette)

    if by_replicate:
        codes, keys = prep.group_replicate_codes()
        titles = [f"{g} - rep {r}" for g, r in keys]
        point_color = [colors[prep.group_values.index(g)] for g, _ in keys]
    else:
        codes = prep.group_codes
        keys = list(prep.group_values)
        titles = [str(g) for g in keys]
        point_color = colors

    n_panels = len(keys)
    ncols = ncols or int(min(4, max(1, n_panels)))
    nrows = int(np.ceil(n_panels / ncols))

    aspect = _map_aspect(res)
    legend_mm = 40.0
    w = cfg.figure_width_mm
    grid_w = w - legend_mm - 8.0
    cell_w = grid_w / ncols
    cell_h = cell_w * aspect / 0.84 + 6.0        # 0.84 = sub-axes height share
    grid_aspect = (nrows * cell_h) / max(grid_w, 1e-6)

    cap = caption_for(
        "Distribution of each group across the map.",
        "Every panel shows the same map with the same node colours; node opacity is "
        "graded by how many samples of that panel's group the node received ("
        + _alpha_ladder_text(cfg).replace("\n", "; ")
        + "), so faded regions are territory the group never occupied.",
        f"n = {prep.n_samples} samples over {n_panels} panels.",
    )
    p = make_figure(w, None, cfg, caption=cap, legend_width_mm=legend_mm,
                    left_mm=4.0, right_mm=4.0, top_mm=7.0, bottom_mm=4.0,
                    axes_aspect=grid_aspect)
    p.ax.set_axis_off()
    box = p.ax.get_position()

    jit = hg.stable_jitter(len(res.bmu))
    pos = hg.sample_positions(res.bmu, res.bmu2, res.bmu_dist, res.bmu2_dist,
                              centres, jit, cfg.jitter_strength)

    for k in range(n_panels):
        r, c = divmod(k, ncols)
        sub = p.fig.add_axes((
            box.x0 + c * box.width / ncols,
            box.y0 + (nrows - 1 - r) * box.height / nrows,
            box.width / ncols * 0.95,
            box.height / nrows * 0.84,
        ))
        member = codes == k
        counts = np.bincount(res.bmu[member], minlength=res.n_nodes)
        hg.draw_hexes(sub, centres, base, edgecolor=cfg.hex_edgecolor,
                      linewidth=0.3, alpha=alpha_for_counts(counts, cfg))
        sub.scatter(pos[member, 0], pos[member, 1], s=cfg.point_size * 0.85,
                    c=point_color[k], edgecolors="white", linewidths=0.5, zorder=6)
        hg.set_map_limits(sub, centres, pad=0.7)
        sub.set_title(f"{titles[k]}  (n={int(member.sum())})",
                      fontsize=cfg.base_font_size * 0.9, pad=2)

    handles = [Line2D([], [], marker="h", linestyle="none", markersize=7,
                      markerfacecolor=(0.35, 0.35, 0.35, a),
                      markeredgecolor="#777777",
                      label=f"{n} sample{'s' if n != 1 else ''}")
               for n, a in sorted(cfg.group_alpha_by_hits.items())]
    hi = (max(cfg.group_alpha_by_hits) + 1) if cfg.group_alpha_by_hits else 1
    handles.append(Line2D([], [], marker="h", linestyle="none", markersize=7,
                          markerfacecolor=(0.35, 0.35, 0.35, cfg.group_alpha_default),
                          markeredgecolor="#777777", label=f">= {hi} samples"))
    p.legend_from(handles, [hh.get_label() for hh in handles],
                  title="Node opacity\n(samples of this group)")
    return p


def plot_single_group_map(
    res: SomResult,
    prep: PreparedData,
    cfg: FigureConfig,
    group_index: int,
) -> Panel:
    """One full-size panel for a single group, same transparency rule."""
    centres = _centres(res)
    base = _base_colors(res, prep, cfg)
    colors = group_colors(prep.n_groups, cfg.palette)
    member = prep.group_codes == group_index
    counts = np.bincount(res.bmu[member], minlength=res.n_nodes)
    alpha = alpha_for_counts(counts, cfg)

    legend_mm = 44.0
    w = cfg.figure_width_mm

    label = prep.group_values[group_index]
    occupied = int(np.count_nonzero(counts))
    cap = caption_for(
        f"Map territory of group '{label}'.",
        "Node opacity is graded by this group's sample count per node ("
        + _alpha_ladder_text(cfg).replace("\n", "; ") + ").",
        f"n = {int(member.sum())} samples over {occupied}/{res.n_nodes} nodes.",
    )
    p = make_figure(w, None, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))
    hg.draw_hexes(p.ax, centres, base, edgecolor=cfg.hex_edgecolor,
                  linewidth=0.35, alpha=alpha)

    jit = hg.stable_jitter(len(res.bmu))
    pos = hg.sample_positions(res.bmu, res.bmu2, res.bmu_dist, res.bmu2_dist,
                              centres, jit, cfg.jitter_strength)
    p.ax.scatter(pos[member, 0], pos[member, 1], s=cfg.point_size,
                 c=colors[group_index], edgecolors="white", linewidths=0.5, zorder=6)
    hg.set_map_limits(p.ax, centres)
    p.ax.set_title(f"{label}", pad=4)

    handles = [Line2D([], [], marker="h", linestyle="none", markersize=8,
                      markerfacecolor=(0.35, 0.35, 0.35, a), markeredgecolor="#777777",
                      label=f"{n} sample{'s' if n != 1 else ''}")
               for n, a in sorted(cfg.group_alpha_by_hits.items())]
    hi = (max(cfg.group_alpha_by_hits) + 1) if cfg.group_alpha_by_hits else 1
    handles.append(Line2D([], [], marker="h", linestyle="none", markersize=8,
                          markerfacecolor=(0.35, 0.35, 0.35, cfg.group_alpha_default),
                          markeredgecolor="#777777", label=f">= {hi} samples"))
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=5,
                          markerfacecolor=colors[group_index], markeredgecolor="white",
                          label=f"{label} sample"))
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Node opacity")
    return p


def plot_group_enrichment_map(
    res: SomResult,
    prep: PreparedData,
    enrichment,
    cfg: FigureConfig,
    group_index: int,
    q_threshold: float = 0.05,
) -> Panel:
    """Log2 observed/expected occupancy for one group, with significance marks."""
    from matplotlib import pyplot as plt

    label = prep.group_values[group_index]
    sub = enrichment[enrichment["group"] == label]
    centres = _centres(res)

    vals = np.zeros(res.n_nodes)
    qs = np.ones(res.n_nodes)
    for _, r in sub.iterrows():
        vals[int(r["node"])] = r["log2_enrichment"]
        qs[int(r["node"])] = r.get("q", 1.0)

    lim = float(max(1e-6, np.abs(vals).max()))
    cmap = plt.get_cmap("RdBu_r")
    face = [cmap(0.5 + 0.5 * v / lim) for v in vals]

    legend_mm = 34.0
    w = cfg.figure_width_mm

    n_sig = int(np.sum((qs < q_threshold) & (vals > 0)))
    cap = caption_for(
        f"Occupancy of '{label}' relative to chance.",
        "Red nodes hold more of this group than the overall group proportions "
        "predict, blue nodes fewer; a black dot marks nodes significant after "
        "Benjamini-Hochberg correction of a hypergeometric test. This turns the "
        "visual impression of 'this group sits over here' into a testable claim.",
        f"{n_sig} node(s) enriched at q < {q_threshold:g}.",
    )
    p = make_figure(w, None, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))
    hg.draw_hexes(p.ax, centres, face, edgecolor="#9a9a9a", linewidth=0.3)
    sig = (qs < q_threshold) & (res.hits > 0)
    if sig.any():
        p.ax.scatter(centres[sig, 0], centres[sig, 1], s=9, c="#111111", zorder=6)
    hg.set_map_limits(p.ax, centres)
    p.ax.set_title(f"Enrichment: {label}", pad=4)

    from .maps import _side_colorbar

    _side_colorbar(p, cmap, -lim, lim, "log2 observed/expected", cfg)
    return p


__all__ = ["plot_group_maps", "plot_single_group_map", "plot_group_enrichment_map",
           "alpha_for_counts"]
