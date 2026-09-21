"""SOM map figures: composition, U-matrix, hits, component planes, diagnostics."""

from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge

from ..association import MapGradients
from ..config import FigureConfig
from ..nodecluster import NodeClustering
from ..preprocess import PreparedData
from ..som import SomResult
from . import hexgeom as hg
from .style import (Panel, caption_for, composition_color, group_colors,
                    make_figure, palette_hues)


def _centres(res: SomResult) -> np.ndarray:
    return hg.node_centres(res.lattice.coords)


def _group_fracs(res: SomResult, prep: PreparedData) -> tuple[np.ndarray, np.ndarray]:
    """Per-node share of each group's own total, and the node's overall load."""
    M = res.hit_matrix(prep.group_codes, prep.n_groups)
    totals = M.sum(axis=0)
    totals[totals == 0] = 1.0
    F = M / totals                                  # each column sums to 1
    load = F.sum(axis=1)
    if load.max() > load.min():
        load_n = (load - load.min()) / (load.max() - load.min())
    else:
        load_n = np.zeros_like(load)
    return F, load_n


def _map_aspect(res: SomResult, pad: float = 1.8) -> float:
    """Data height / data width of the drawn map, for figure sizing."""
    c = _centres(res)
    span_x = np.ptp(c[:, 0]) + pad
    span_y = np.ptp(c[:, 1]) + pad
    return float(span_y / max(span_x, 1e-6))


def _node_hues(prep: PreparedData, cfg: FigureConfig) -> np.ndarray:
    return palette_hues(prep.n_groups, cfg.palette)


def _present_count(frac: np.ndarray, cfg: FigureConfig) -> int:
    total = float(frac.sum())
    if total <= 0:
        return 0
    return int(np.count_nonzero(frac > cfg.mixture_ignore_fraction * total))


# ==========================================================================
# 1. Group-composition map (the macro's HSB map, arithmetic corrected)
# ==========================================================================
def plot_composition_map(
    res: SomResult,
    prep: PreparedData,
    cfg: FigureConfig,
    show_points: bool = True,
    epoch: int | None = None,
    title: str | None = None,
) -> Panel:
    F, load_n = _group_fracs(res, prep)
    n_g = prep.n_groups
    hues = _node_hues(prep, cfg)
    colors = group_colors(n_g, cfg.palette)
    centres = _centres(res)

    face = [composition_color(F[i], hues, load_n[i], _present_count(F[i], cfg), n_g, cfg)
            for i in range(res.n_nodes)]

    legend_mm = 46.0
    w, h = cfg.figure_width_mm, None
    cap = caption_for(
        f"Self-organising map ({res.width}x{res.height} {res.lattice.topology}"
        f"{', toroidal' if res.lattice.toroidal else ''}, {res.algorithm} training).",
        "Hue is the circular mean of the group hues weighted by each group's share of "
        "the node; saturation falls as more groups mix in one node; brightness rises "
        "with the fraction of the data the node captured. "
        + ("Points are individual samples, placed towards their second-best node in "
           "proportion to how ambiguous the assignment is. " if show_points else ""),
        f"n = {prep.n_samples} samples, {n_g} groups"
        + (f", epoch {epoch}." if epoch is not None else "."),
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))

    hg.draw_hexes(p.ax, centres, face, edgecolor=cfg.hex_edgecolor, linewidth=0.4)
    hg.set_map_limits(p.ax, centres)

    if show_points:
        jit = hg.stable_jitter(len(res.bmu))
        pos = hg.sample_positions(res.bmu, res.bmu2, res.bmu_dist, res.bmu2_dist,
                                  centres, jit, cfg.jitter_strength)
        for g in range(n_g):
            m = prep.group_codes == g
            p.ax.scatter(pos[m, 0], pos[m, 1], s=cfg.point_size, c=colors[g],
                         edgecolors="white", linewidths=0.5, zorder=5)

    p.ax.set_title(title or "Group composition map", pad=4)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=colors[g], markeredgecolor="white",
                      label=f"{prep.group_values[g]} (n={int((prep.group_codes == g).sum())})")
               for g in range(n_g)]
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Group")
    _encoding_key(p, cfg)
    return p


def _encoding_key(p: Panel, cfg: FigureConfig) -> None:
    """Small text key for the hue/saturation/value encoding, in the legend column."""
    if p.legend_ax is None:
        return
    p.legend_ax.text(
        0.0, 0.02,
        "Node colour\n"
        "  hue  = dominant group\n"
        "  pale = groups mixed\n"
        "  dark = few samples",
        transform=p.legend_ax.transAxes, va="bottom", ha="left",
        fontsize=cfg.base_font_size * 0.85, color="#333333", linespacing=1.5,
    )


# ==========================================================================
# 2. Pie-glyph map -- exact composition, no colour-mixing ambiguity
# ==========================================================================
def plot_group_pies(res: SomResult, prep: PreparedData, cfg: FigureConfig) -> Panel:
    """Each node is a pie of its group composition, sized by its sample count.

    The HSB map compresses three quantities into one colour, which is striking
    but not readable back to numbers.  This is the same information drawn so a
    reader can count it.
    """
    M = res.hit_matrix(prep.group_codes, prep.n_groups)
    colors = group_colors(prep.n_groups, cfg.palette)
    centres = _centres(res)
    hits = M.sum(axis=1)
    hmax = max(hits.max(), 1)

    legend_mm = 42.0
    w, h = cfg.figure_width_mm, None
    cap = caption_for(
        "Node composition as pie glyphs.",
        "Wedge angles give the exact proportion of each group among the samples "
        "assigned to that node; glyph radius scales with the square root of the "
        "node's sample count. Empty nodes are drawn as open outlines.",
        f"n = {prep.n_samples} samples; largest node holds {int(hmax)}.",
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))

    hg.draw_hexes(p.ax, centres, ["#f4f4f4"] * res.n_nodes,
                  edgecolor="#d0d0d0", linewidth=0.35)

    rmax = hg.HEX_RADIUS * 0.82
    for i in range(res.n_nodes):
        if hits[i] <= 0:
            continue
        r = rmax * np.sqrt(hits[i] / hmax)
        start = 90.0
        for g in range(prep.n_groups):
            if M[i, g] <= 0:
                continue
            ext = 360.0 * M[i, g] / hits[i]
            p.ax.add_patch(Wedge(centres[i], r, start - ext, start,
                                 facecolor=colors[g], edgecolor="white",
                                 linewidth=0.3, zorder=4))
            start -= ext

    hg.set_map_limits(p.ax, centres)
    p.ax.set_title("Node composition (pie glyphs)", pad=4)

    handles = [Patch(facecolor=colors[g], edgecolor="white",
                     label=f"{prep.group_values[g]}") for g in range(prep.n_groups)]
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Group")
    return p


# ==========================================================================
# 3. U-matrix
# ==========================================================================
def plot_u_matrix(res: SomResult, cfg: FigureConfig, show_hits: bool = True) -> Panel:
    u = res.u_matrix()
    centres = _centres(res)
    cmap = plt.get_cmap("bone_r")
    finite = np.isfinite(u)
    lo, hi = (np.nanmin(u), np.nanmax(u)) if finite.any() else (0.0, 1.0)
    norm = (u - lo) / (hi - lo) if hi > lo else np.zeros_like(u)
    face = [cmap(float(v)) if np.isfinite(v) else (1, 1, 1, 1) for v in norm]

    legend_mm = 30.0
    w, h = cfg.figure_width_mm, None
    cap = caption_for(
        "U-matrix: distance between neighbouring codebook vectors.",
        "Dark ridges are boundaries in the feature space -- samples on either side "
        "of a ridge are genuinely different, while a pale basin is one continuous "
        "behavioural mode. This is the standard way to read cluster structure off "
        "a SOM and was not available in the ImageJ version.",
        f"range {lo:.2f}-{hi:.2f} (scaled feature units).",
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))
    hg.draw_hexes(p.ax, centres, face, edgecolor="#9a9a9a", linewidth=0.3)

    if show_hits:
        s = res.hits / max(res.hits.max(), 1)
        m = res.hits > 0
        p.ax.scatter(centres[m, 0], centres[m, 1], s=6 + 34 * s[m],
                     facecolors="none", edgecolors="#c0392b", linewidths=0.7, zorder=5)

    hg.set_map_limits(p.ax, centres)
    p.ax.set_title("U-matrix", pad=4)
    _side_colorbar(p, cmap, lo, hi, "mean distance to\nneighbouring nodes", cfg)
    if show_hits:
        p.legend_ax.text(0.0, 0.02, "red circles: node\nsample count",
                         transform=p.legend_ax.transAxes, va="bottom",
                         fontsize=cfg.base_font_size * 0.85, color="#c0392b")
    return p


# ==========================================================================
# 4. Hit map
# ==========================================================================
def plot_hit_map(res: SomResult, prep: PreparedData, cfg: FigureConfig) -> Panel:
    centres = _centres(res)
    hits = res.hits
    cmap = plt.get_cmap("YlGnBu")
    hi = max(hits.max(), 1)
    face = [cmap(float(v / hi)) for v in hits]

    legend_mm = 30.0
    w, h = cfg.figure_width_mm, None
    empty = int(np.sum(hits == 0))
    cap = caption_for(
        "Hit histogram: number of samples assigned to each node.",
        "A map with many empty nodes is larger than the data supports; a map with "
        "one dominant node is too small.",
        f"{res.n_nodes - empty}/{res.n_nodes} nodes occupied, "
        f"max load {int(hits.max())} samples.",
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))
    hg.draw_hexes(p.ax, centres, face, edgecolor="#9a9a9a", linewidth=0.3)
    for i in range(res.n_nodes):
        if hits[i] > 0:
            p.ax.text(centres[i, 0], centres[i, 1], str(int(hits[i])),
                      ha="center", va="center", fontsize=cfg.base_font_size * 0.75,
                      color="#ffffff" if hits[i] > 0.6 * hi else "#222222", zorder=6)
    hg.set_map_limits(p.ax, centres)
    p.ax.set_title("Sample hits per node", pad=4)
    _side_colorbar(p, cmap, 0, hi, "samples", cfg)
    return p


# ==========================================================================
# 5. Component planes
# ==========================================================================
def plot_component_planes(
    res: SomResult,
    prep: PreparedData,
    cfg: FigureConfig,
    features: list[str] | None = None,
    ncols: int = 4,
    in_original_units: bool = True,
) -> Panel:
    names = features or list(res.feature_names)
    names = [n for n in names if n in res.feature_names]
    if not names:
        raise ValueError("No features to draw.")
    idx = [res.feature_names.index(n) for n in names]

    W = prep.inverse(res.codebook) if in_original_units else res.codebook
    centres = _centres(res)
    ncols = int(min(ncols, len(names)))
    nrows = int(np.ceil(len(names) / ncols))

    aspect = _map_aspect(res)
    w = cfg.figure_width_mm
    cell_w = (w - 8.0) / ncols
    cell_h = cell_w * aspect / 0.80 + 7.0        # 0.80 = sub-axes height share
    grid_aspect = (nrows * cell_h) / max(w - 8.0, 1e-6)

    cap = caption_for(
        "Component planes: the value of each metric across the map.",
        "Every panel shows the same node layout, so two metrics whose planes look "
        "alike vary together, and a metric whose high-value region coincides with a "
        "group's territory on the composition map is what separates that group. "
        + ("Colour bars are in original measurement units." if in_original_units
           else "Values are in scaled units."),
        f"{len(names)} of {len(res.feature_names)} metrics shown.",
    )
    p = make_figure(w, None, cfg, caption=cap, top_mm=8.0, left_mm=4.0,
                    right_mm=4.0, bottom_mm=4.0, axes_aspect=grid_aspect)
    p.ax.set_axis_off()

    cmap = plt.get_cmap("viridis")
    unit_of = prep.source.units if prep.source else {}
    for k, (name, j) in enumerate(zip(names, idx)):
        r, c = divmod(k, ncols)
        sub = p.fig.add_axes((
            p.ax.get_position().x0 + c * p.ax.get_position().width / ncols,
            p.ax.get_position().y0 + (nrows - 1 - r) * p.ax.get_position().height / nrows,
            p.ax.get_position().width / ncols * 0.94,
            p.ax.get_position().height / nrows * 0.80,
        ))
        v = W[:, j]
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        norm = (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        hg.draw_hexes(sub, centres, [cmap(float(t)) for t in norm],
                      edgecolor="#ffffff", linewidth=0.25)
        hg.set_map_limits(sub, centres, pad=0.7)
        unit = unit_of.get(name, "")
        sub.set_title(f"{name}" + (f"\n[{unit}]" if unit else ""),
                      fontsize=cfg.base_font_size * 0.85, pad=2)
        sub.text(0.5, -0.06, f"{lo:.3g} - {hi:.3g}", transform=sub.transAxes,
                 ha="center", va="top", fontsize=cfg.base_font_size * 0.7, color="#555555")
    return p


# ==========================================================================
# 6. Node clusters (the macro's k-means overlay)
# ==========================================================================
def plot_node_clusters(
    res: SomResult,
    clustering: NodeClustering,
    prep: PreparedData,
    cfg: FigureConfig,
    k: int | None = None,
) -> Panel:
    k = k or clustering.best_k
    labels = clustering.labels_by_k.get(k)
    if labels is None:
        raise ValueError(f"No node clustering for k={k}.")

    centres = _centres(res)
    uniq = sorted(set(labels.tolist()))
    colors = group_colors(len(uniq), cfg.palette)
    occupied = res.hits > 0
    non_empty = sorted(set(labels[occupied].tolist()))

    legend_mm = 46.0
    w, h = cfg.figure_width_mm, None

    score_txt = ""
    if clustering.scores is not None and (clustering.scores["k"] == k).any():
        row = clustering.scores.loc[clustering.scores["k"] == k].iloc[0]
        bits = [f"silhouette {row['silhouette']:.2f}",
                f"Davies-Bouldin {row['davies_bouldin']:.2f}"]
        if "adjusted_rand_vs_groups" in row and np.isfinite(row["adjusted_rand_vs_groups"]):
            bits.append(f"ARI vs. treatment {row['adjusted_rand_vs_groups']:.2f}")
        score_txt = "; ".join(bits) + "."

    cap = caption_for(
        f"Second-level {clustering.method} clustering of the codebook, k = {k}"
        + (" (selected automatically)." if k == clustering.best_k else "."),
        "Nodes are partitioned in feature space, not on the lattice, so a cluster "
        "that appears split on the map indicates two map regions with the same "
        "underlying behaviour. Grey nodes hold no samples.",
        score_txt + f" {len(non_empty)}/{k} clusters contain data.",
    )
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm, equal=True, bottom_mm=5.0,
                    axes_aspect=_map_aspect(res))

    base = ["#e9e9e9" if not occupied[i] else "#fbfbfb" for i in range(res.n_nodes)]
    hg.draw_hexes(p.ax, centres, base, edgecolor="#cccccc", linewidth=0.3)
    for ci, c in enumerate(uniq):
        hg.cluster_outline(p.ax, centres, labels == c, colors[ci % len(colors)])

    gcolors = group_colors(prep.n_groups, cfg.palette)
    jit = hg.stable_jitter(len(res.bmu))
    pos = hg.sample_positions(res.bmu, res.bmu2, res.bmu_dist, res.bmu2_dist,
                              centres, jit, cfg.jitter_strength)
    for g in range(prep.n_groups):
        m = prep.group_codes == g
        p.ax.scatter(pos[m, 0], pos[m, 1], s=cfg.point_size * 0.7, c=gcolors[g],
                     edgecolors="white", linewidths=0.4, zorder=6)

    hg.set_map_limits(p.ax, centres)
    p.ax.set_title(f"Node clusters (k = {k})", pad=4)

    handles = [Patch(facecolor=colors[i % len(colors)], alpha=0.35,
                     edgecolor=colors[i % len(colors)], label=f"cluster {c}")
               for i, c in enumerate(uniq)]
    handles += [Line2D([], [], marker="o", linestyle="none", markersize=4,
                       markerfacecolor=gcolors[g], markeredgecolor="white",
                       label=str(prep.group_values[g])) for g in range(prep.n_groups)]
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Node cluster / group")
    return p


def plot_k_selection(clustering: NodeClustering, cfg: FigureConfig) -> Panel:
    s = clustering.scores
    if s is None or s.empty:
        raise ValueError("No clustering scores to plot.")
    cap = caption_for(
        "Choosing the number of node clusters.",
        "Silhouette and Calinski-Harabasz should be maximised, Davies-Bouldin "
        "minimised; the selected k is the best average rank across all three. "
        "Adjusted Rand against the experimental groups is shown for reference and "
        "is not used for the selection.",
        f"selected k = {clustering.best_k}.",
    )
    p = make_figure(cfg.figure_width_mm * 0.62, 62.0, cfg, caption=cap,
                    legend_width_mm=0.0, left_mm=16.0, right_mm=16.0)
    ax = p.ax
    ax.plot(s["k"], _norm(s["silhouette"]), "o-", label="silhouette", color="#0072B2")
    ax.plot(s["k"], 1 - _norm(s["davies_bouldin"]), "s-", label="1 - Davies-Bouldin", color="#D55E00")
    ax.plot(s["k"], _norm(s["calinski_harabasz"]), "^-", label="Calinski-Harabasz", color="#009E73")
    if "adjusted_rand_vs_groups" in s:
        ax.plot(s["k"], s["adjusted_rand_vs_groups"], "d--", label="ARI vs. treatment",
                color="#7F7F7F", alpha=0.8)
    ax.axvline(clustering.best_k, color="#c0392b", lw=0.8, ls=":")
    ax.set_xlabel("number of node clusters (k)")
    ax.set_ylabel("index (rescaled 0-1)")
    ax.set_xticks(s["k"].tolist())
    ax.legend(loc="best", fontsize=cfg.base_font_size * 0.85)
    return p


def _norm(v) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(a), np.nanmax(a)
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


# ==========================================================================
# 7. Training diagnostics (replaces the SOM kymograph)
# ==========================================================================
def plot_training_curves(res: SomResult, cfg: FigureConfig) -> Panel:
    h = res.history
    cap = caption_for(
        "SOM training diagnostics.",
        "Quantisation error is the mean distance from a sample to its winning node "
        "and should fall and flatten; topographic error is the fraction of samples "
        "whose two best nodes are not neighbours and measures whether the map stayed "
        "folded correctly; group purity is the hit-weighted share of the majority "
        "treatment per node. A run whose quantisation error is still falling at the "
        "last epoch has not converged.",
        f"{len(h)} epochs, final QE = {h.qe[-1]:.3f}, TE = {h.te[-1]:.3f}.",
    )
    p = make_figure(cfg.figure_width_mm * 0.72, 68.0, cfg, caption=cap,
                    left_mm=16.0, right_mm=18.0)
    ax = p.ax
    ax.plot(h.epochs, h.qe, color="#0072B2", label="quantisation error")
    ax.set_xlabel("training epoch")
    ax.set_ylabel("quantisation error", color="#0072B2")
    ax.tick_params(axis="y", colors="#0072B2")

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(h.epochs, h.te, color="#D55E00", lw=0.9, label="topographic error")
    if np.isfinite(np.asarray(h.purity, dtype=float)).any():
        ax2.plot(h.epochs, h.purity, color="#009E73", lw=0.9, ls="--", label="group purity")
    ax2.set_ylabel("topographic error / purity")
    ax2.set_ylim(0, 1.02)

    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], loc="center right",
              fontsize=cfg.base_font_size * 0.85)
    return p


def plot_codebook_kymograph(res: SomResult, prep: PreparedData,
                            cfg: FigureConfig, feature: str | None = None) -> Panel:
    """Node value against training epoch -- the macro's kymograph, as a heatmap."""
    h = res.history
    snaps = [(e, c) for e, c in zip(h.epochs, h.codebooks) if c is not None]
    if not snaps:
        raise ValueError("No codebook snapshots were recorded.")
    name = feature or res.feature_names[0]
    j = res.feature_names.index(name)
    epochs = [e for e, _ in snaps]
    Mx = np.column_stack([prep.inverse(c)[:, j] for _, c in snaps])

    cap = caption_for(
        f"Convergence of every node for '{name}'.",
        "One row per node, one column per recorded epoch. Vertical stripes mean the "
        "map is still reorganising; a settled block of colour means that node has "
        "converged. This is the kymograph from the ImageJ version, rendered as a "
        "labelled heat map instead of a raw 32-bit stack.",
        f"{res.n_nodes} nodes x {len(epochs)} snapshots.",
    )
    p = make_figure(cfg.figure_width_mm * 0.8, 78.0, cfg, caption=cap,
                    left_mm=18.0, right_mm=22.0)
    im = p.ax.imshow(Mx, aspect="auto", cmap="magma", origin="lower",
                     extent=(epochs[0], epochs[-1], 0, res.n_nodes))
    p.ax.set_xlabel("training epoch")
    p.ax.set_ylabel("node index")
    cb = p.fig.colorbar(im, ax=p.ax, fraction=0.04, pad=0.02)
    unit = prep.source.units.get(name, "") if prep.source else ""
    cb.set_label(f"{name}" + (f" [{unit}]" if unit else ""))
    return p


# ==========================================================================
def _side_colorbar(p: Panel, cmap, lo: float, hi: float, label: str,
                   cfg: FigureConfig) -> None:
    if p.legend_ax is None:
        return
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    pos = p.legend_ax.get_position()
    cax = p.fig.add_axes((pos.x0, pos.y0 + pos.height * 0.30,
                          pos.width * 0.30, pos.height * 0.45))
    cb = p.fig.colorbar(ScalarMappable(norm=Normalize(lo, hi), cmap=cmap), cax=cax)
    cb.set_label(label, fontsize=cfg.base_font_size * 0.85)
    cb.ax.tick_params(labelsize=cfg.base_font_size * 0.8)


__all__ = [
    "plot_composition_map", "plot_group_pies", "plot_u_matrix", "plot_hit_map",
    "plot_component_planes", "plot_node_clusters", "plot_k_selection",
    "plot_training_curves", "plot_codebook_kymograph",
]
