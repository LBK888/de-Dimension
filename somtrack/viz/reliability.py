"""How far a projection can be trusted, drawn rather than asserted.

A two-dimensional picture of fifty-dimensional data is always wrong somewhere.
The useful question is *where*, and these panels answer it three ways:

``plot_rnx_curves``
    one curve per method, so "good locally, wrong globally" can be told apart
    from the reverse.  Trustworthiness and continuity are two points on this
    curve; drawing the whole thing removes the argument about which K to quote.

``plot_point_reliability``
    the same scatter the reader has already seen, with the points the projection
    placed unreliably marked.  A cluster made entirely of dubious points is an
    artefact; a cluster of trustworthy points is worth interpreting.

``plot_seed_stability``
    the same projection under several random seeds, Procrustes-aligned.  A point
    that moves between runs has no fixed neighbourhood to interpret.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from ..config import FigureConfig
from . import palette as pal
from .style import Panel, caption_for, make_figure


# ==========================================================================
def plot_rnx_curves(projections: dict, cfg: FigureConfig) -> Panel:
    """Neighbourhood preservation across every scale, for every method."""
    have = [p for p in projections.values() if p.rnx is not None and p.rnx.size]
    if not have:
        raise ValueError("No R_NX curves available.")

    cap = caption_for(
        "How much of each neighbourhood survives the projection.",
        "R_NX(K) is the share of each point's K nearest neighbours that the "
        "projection kept, rescaled so a random layout scores 0 and a perfect one "
        "scores 1 at every K. A curve high on the left and low on the right "
        "preserved local detail but rearranged the overall layout; the reverse "
        "means the broad arrangement is right but fine neighbourhoods are not. "
        "The area under each curve, on the logarithmic K axis, is the single "
        "number in the legend.",
        "K is plotted on a logarithmic scale, which is the scale the area is "
        "computed on.")

    p = make_figure(cfg.figure_width_mm * 0.78, 60.0, cfg, caption=cap,
                    legend_width_mm=44.0, left_mm=16.0, bottom_mm=12.0,
                    height_is_axes=True)
    ax = p.ax
    deco = pal.categorical(len(have), cfg)

    handles = []
    for i, proj in enumerate(have):
        k = np.arange(1, proj.rnx.size + 1)
        colour = deco[i]
        ax.plot(k, proj.rnx, lw=1.4, color=colour,
                ls="--" if proj.supervised else "-")
        handles.append(Line2D([], [], color=colour,
                              ls="--" if proj.supervised else "-", lw=1.4,
                              label=f"{proj.name}  AUC {proj.rnx_auc:.3f}"))

    ax.set_xscale("log")
    ax.set_xlabel("neighbourhood size K")
    ax.set_ylabel("$R_{NX}(K)$")
    ax.axhline(0.0, color="#999999", lw=0.6, ls=":")
    ax.set_ylim(min(-0.05, float(min(c.rnx.min() for c in have))), 1.02)
    ax.grid(True, alpha=0.18, lw=0.4)
    p.legend_from(handles, [h.get_label() for h in handles], title="Method")
    if any(c.supervised for c in have):
        p.legend_ax.text(0.0, 0.02, "Dashed = supervised projection\n"
                                    "(separation is built in)",
                         transform=p.legend_ax.transAxes, va="bottom",
                         fontsize=cfg.base_font_size * 0.78, color="#555555")
    return p


# ==========================================================================
def plot_point_reliability(proj, prep, cfg: FigureConfig) -> Panel:
    """The projection, with the points it placed unreliably marked."""
    if proj.dubious is None:
        raise ValueError("No per-point reliability for this projection.")

    Y = proj.coords
    dub = np.asarray(proj.dubious, bool)
    frac = float(dub.mean()) if dub.size else np.nan

    cap = caption_for(
        f"Which points in the {proj.name} projection can be believed.",
        "Each point's neighbourhood in the projection is compared with its "
        "neighbourhood in the original metric space, and the score is compared "
        "against a null built by scrambling the projection. Open circles fell "
        "below the null's 5th percentile: their position carries no information "
        "about who their neighbours are. A cluster made mostly of open circles "
        "is an artefact of the projection, not a finding.",
        f"{int(dub.sum())} of {dub.size} points ({frac * 100:.0f}%) are "
        f"unreliable. Method of Xia, Lee & Li (2024).")

    p = make_figure(cfg.figure_width_mm * 0.72, None, cfg, caption=cap,
                    legend_width_mm=40.0, left_mm=14.0, bottom_mm=13.0,
                    axes_aspect=0.82)
    ax = p.ax
    deco = pal.categorical(prep.n_groups, cfg)

    for g in range(prep.n_groups):
        m = prep.group_codes == g
        colour = deco.for_group(g)
        good = m & ~dub
        bad = m & dub
        ax.scatter(Y[good, 0], Y[good, 1], s=cfg.point_size, c=colour,
                   marker=pal.marker_for(g), edgecolors="white", linewidths=0.4,
                   zorder=3)
        ax.scatter(Y[bad, 0], Y[bad, 1], s=cfg.point_size * 1.1,
                   facecolors="none", edgecolors=colour, linewidths=0.9,
                   marker=pal.marker_for(g), zorder=4)

    ax.set_xlabel(proj.axis_labels[0])
    ax.set_ylabel(proj.axis_labels[1])
    ax.set_title(f"{proj.name}: per-point reliability", pad=4)
    ax.grid(True, alpha=0.15, lw=0.4)

    handles = [Line2D([], [], marker=pal.marker_for(g), linestyle="none",
                      markersize=5, markerfacecolor=deco.for_group(g),
                      markeredgecolor="white",
                      label=f"{prep.group_values[g]} "
                            f"(n={int((prep.group_codes == g).sum())})")
               for g in range(prep.n_groups)]
    handles += [Line2D([], [], marker="o", linestyle="none", markersize=5,
                       markerfacecolor="none", markeredgecolor=pal.INK,
                       label="unreliably placed")]
    p.legend_from(handles, [h.get_label() for h in handles], title="Group")
    return p


# ==========================================================================
def plot_seed_stability(projections: dict, cfg: FigureConfig) -> Panel:
    """How far each point moves when the random seed changes."""
    have = [p for p in projections.values() if p.stability is not None]
    if not have:
        raise ValueError("No stability information; set embedding.stability_seeds.")

    cap = caption_for(
        "How much of each projection is the seed rather than the data.",
        "Each projection was re-run with several random seeds and the runs were "
        "Procrustes-aligned. The value plotted is how far a point moves between "
        "runs, in units of the projection's own radius. A method whose points "
        "move by a tenth of the layout has no stable neighbourhoods to "
        "interpret, whatever the picture looks like once.",
        "Lower is better; a deterministic method sits at zero.")

    p = make_figure(cfg.figure_width_mm * 0.66, 56.0, cfg, caption=cap,
                    left_mm=18.0, right_mm=8.0, bottom_mm=16.0,
                    height_is_axes=True)
    ax = p.ax
    deco = pal.categorical(len(have), cfg)
    for i, proj in enumerate(have):
        v = np.asarray(proj.stability, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            continue
        x = i + np.random.default_rng(i).uniform(-0.15, 0.15, v.size)
        ax.scatter(x, v, s=8, color=deco[i], alpha=0.5, edgecolors="none")
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=pal.INK, lw=1.5)
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([c.name for c in have], rotation=25, ha="right",
                       fontsize=cfg.base_font_size * 0.85)
    ax.set_ylabel("movement between seeds\n(fraction of layout radius)")
    ax.grid(True, axis="y", alpha=0.18, lw=0.4)
    ax.set_ylim(bottom=0)
    return p


# ==========================================================================
def plot_shepard(proj, X: np.ndarray, cfg: FigureConfig,
                 max_pairs: int = 20000) -> Panel:
    """Original distance against projected distance, pair by pair."""
    from ..analysis.quality import pairwise

    DX = pairwise(np.asarray(X, float))
    DY = pairwise(np.asarray(proj.coords, float))
    iu = np.triu_indices(DX.shape[0], 1)
    dx, dy = DX[iu], DY[iu]
    if dx.size > max_pairs:
        idx = np.random.default_rng(0).choice(dx.size, max_pairs, replace=False)
        dx, dy = dx[idx], dy[idx]

    with np.errstate(invalid="ignore"):
        rho = float(np.corrcoef(np.argsort(np.argsort(dx)),
                                np.argsort(np.argsort(dy)))[0, 1])

    cap = caption_for(
        f"Does distance in the {proj.name} picture mean distance in the data?",
        "Every pair of samples contributes one point: its distance in the "
        "original metric space against its distance in the projection. A tight "
        "rising band means the picture can be read as a map. A cloud means it "
        "cannot, and only which points sit next to which is interpretable.",
        f"Spearman rank correlation between the two distances: {rho:.3f}.")

    p = make_figure(cfg.figure_width_mm * 0.5, 54.0, cfg, caption=cap,
                    left_mm=16.0, right_mm=6.0, bottom_mm=12.0,
                    height_is_axes=True)
    ax = p.ax
    ax.scatter(dx, dy, s=1.4, color="#0072B2", alpha=0.12, edgecolors="none",
               rasterized=True)
    ax.set_xlabel("distance between samples (original metrics)")
    ax.set_ylabel(f"distance in {proj.name}")
    ax.grid(True, alpha=0.15, lw=0.4)
    return p


__all__ = ["plot_rnx_curves", "plot_point_reliability", "plot_seed_stability",
           "plot_shepard"]
