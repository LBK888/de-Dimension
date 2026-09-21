"""Comparison figures: several methods, several settings, one page.

These panels exist to make cherry-picking hard.  When a dozen projections are
available, the temptation is to run them all and show the one that separates
best; a grid showing every method that was run, each with its quality score,
removes that option without having to argue about it.

The scan figures do the same for hyper-parameters.  A scan surface with its
*plateau* marked says "anything in this range gives the same answer", which is
both the honest summary and the one that stops a user tuning until the picture
agrees with their hypothesis.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from ..config import FigureConfig
from . import palette as pal
from .style import Panel, caption_for, make_figure


# ==========================================================================
def plot_method_grid(projections: dict, prep, cfg: FigureConfig,
                     n_cols: int = 3) -> Panel:
    """Every projection that was run, on one page, with its quality score."""
    items = list(projections.values())
    if not items:
        raise ValueError("No projections to compare.")

    n = len(items)
    cols = int(min(n_cols, n))
    rows = int(np.ceil(n / cols))
    deco = pal.categorical(prep.n_groups, cfg)

    agreement = ""
    cap = caption_for(
        "The same data under every projection that was run.",
        "Structure that appears in several of these panels is a property of the "
        "data; structure that appears in one is a property of that algorithm. "
        "Each panel carries the area under its R_NX curve (higher preserves more "
        "neighbourhoods) and the share of points it placed unreliably. Panels "
        "marked 'supervised' were given the group labels and separate by "
        "construction.",
        agreement)

    legend_mm = 40.0
    w = cfg.figure_width_mm
    cell = (w - legend_mm - 16.0) / cols
    h = rows * (cell + 9.0)
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm,
                    left_mm=7.0, right_mm=4.0, top_mm=6.0, bottom_mm=8.0,
                    height_is_axes=True)
    p.ax.set_axis_off()
    box = p.ax.get_position()

    for i, proj in enumerate(items):
        r, c = divmod(i, cols)
        sub = p.fig.add_axes((
            box.x0 + c * box.width / cols + 0.010,
            box.y0 + (rows - 1 - r) * box.height / rows + 0.030,
            box.width / cols * 0.86,
            box.height / rows * 0.74,
        ))
        Y = proj.coords
        for g in range(prep.n_groups):
            m = prep.group_codes == g
            sub.scatter(Y[m, 0], Y[m, 1], s=cfg.point_size * 0.55,
                        c=deco.for_group(g), marker=pal.marker_for(g),
                        alpha=deco.alpha_for(g), edgecolors="white",
                        linewidths=0.25, zorder=3)
        sub.set_xticks([]); sub.set_yticks([])
        sub.grid(True, alpha=0.12, lw=0.35)
        for spine in sub.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.2 if proj.supervised else 0.5)
            spine.set_color("#D55E00" if proj.supervised else "#cccccc")

        bits = []
        if np.isfinite(proj.rnx_auc):
            bits.append(f"AUC {proj.rnx_auc:.2f}")
        if np.isfinite(proj.dubious_fraction):
            bits.append(f"{proj.dubious_fraction * 100:.0f}% unreliable")
        title = proj.name + (" (supervised)" if proj.supervised else "")
        sub.set_title(title, fontsize=cfg.base_font_size * 0.92, pad=2)
        sub.set_xlabel("  ".join(bits), fontsize=cfg.base_font_size * 0.68,
                       color="#555555", labelpad=2)

    handles = [Line2D([], [], marker=pal.marker_for(g), linestyle="none",
                      markersize=5, markerfacecolor=deco.for_group(g),
                      markeredgecolor="white",
                      label=f"{prep.group_values[g]} "
                            f"(n={int((prep.group_codes == g).sum())})")
               for g in range(prep.n_groups)]
    p.legend_from(handles, [h.get_label() for h in handles], title="Group")
    if any(x.supervised for x in items):
        p.legend_ax.text(0.0, 0.02,
                         "Orange frame: supervised\nprojection. It separates the\n"
                         "groups by construction and\nis not evidence that they differ.",
                         transform=p.legend_ax.transAxes, va="bottom",
                         fontsize=cfg.base_font_size * 0.74, color="#8a3b00")
    return p


# ==========================================================================
def plot_agreement(agreement, cfg: FigureConfig) -> Panel:
    """How much two projections agree about who is next to whom."""
    if agreement is None or agreement.empty or len(agreement) < 2:
        raise ValueError("Need at least two projections to compare.")

    M = agreement.to_numpy(float)
    labels = [str(i) for i in agreement.index]
    off = M[np.triu_indices(len(M), 1)]

    cap = caption_for(
        "Do the projections agree about the neighbourhoods?",
        "Each cell is the share of a point's ten nearest neighbours that the two "
        "methods have in common, averaged over all points. High agreement means "
        "the layout is being driven by the data; low agreement between two "
        "otherwise reasonable methods means at least one of them is drawing its "
        "own assumptions.",
        f"Average agreement between different methods: {off.mean() * 100:.0f}%.")

    size = max(64.0, 14.0 + 15.0 * len(labels))
    p = make_figure(min(cfg.figure_width_mm, size + 36.0), size, cfg, caption=cap,
                    left_mm=30.0, right_mm=22.0, top_mm=8.0,
                    height_is_axes=True)
    ax = p.ax
    im = ax.imshow(M, cmap=pal.sequential(cfg, print_safe=True), vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right",
                       fontsize=cfg.base_font_size * 0.82)
    ax.set_yticklabels(labels, fontsize=cfg.base_font_size * 0.82)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=cfg.base_font_size * 0.7,
                    color="white" if M[i, j] > 0.55 else pal.INK)
    box = ax.get_position()
    cax = p.fig.add_axes((box.x1 + 0.022, box.y0 + box.height * 0.2,
                          0.016, box.height * 0.6))
    cb = p.fig.colorbar(im, cax=cax)
    cb.set_label("shared nearest neighbours", fontsize=cfg.base_font_size * 0.82)
    cb.ax.tick_params(labelsize=cfg.base_font_size * 0.72, width=0.5, length=2)
    ax.set_title("Agreement between projections", pad=5)
    return p


# ==========================================================================
def plot_scan_surface(scan, cfg: FigureConfig) -> Panel:
    """The scan's score across the grid, with the plateau marked."""
    keys = scan.keys
    rows = scan.rows
    column = scan.criterion if scan.criterion in rows else "rnx_auc"
    if column not in rows:
        raise ValueError("The scan has no scored column to draw.")

    cap = caption_for(
        f"Choosing the settings for {scan.method_label}.",
        scan.note + " The shaded band marks every setting that scored within "
        f"{scan.tolerance:.0%} of the best. A wide band is the useful result: it "
        "means the answer does not depend on the setting, and the figure that "
        "follows would look the same for any value in it.",
        scan.plateau_text())

    if len(keys) == 1:
        return _scan_line(scan, rows, keys[0], column, cap, cfg)
    return _scan_heatmap(scan, rows, keys[:2], column, cap, cfg)


def _scan_line(scan, rows, key, column, cap, cfg) -> Panel:
    p = make_figure(cfg.figure_width_mm * 0.6, 54.0, cfg, caption=cap,
                    left_mm=20.0, right_mm=8.0, bottom_mm=13.0,
                    height_is_axes=True)
    ax = p.ax
    d = rows.sort_values(key)
    x = d[key].to_numpy(float)
    yv = d[column].to_numpy(float)

    if not scan.plateau.empty:
        lo = float(scan.plateau[key].min())
        hi = float(scan.plateau[key].max())
        ax.axvspan(lo, hi, color="#0072B2", alpha=0.12, lw=0,
                   label=f"within {scan.tolerance:.0%} of best")
    ax.plot(x, yv, "-o", color="#0072B2", lw=1.3, ms=4, zorder=3)
    if scan.best:
        bx = scan.best.get(key)
        if bx is not None:
            ax.axvline(float(bx), color="#D55E00", lw=1.2, ls="--",
                       label="chosen", zorder=4)
    ax.set_xlabel(key.replace("_", " "))
    ax.set_ylabel(column.replace("_", " "))
    ax.grid(True, alpha=0.18, lw=0.4)
    ax.legend(fontsize=cfg.base_font_size * 0.78, frameon=False,
              loc="best")
    return p


def _scan_heatmap(scan, rows, keys, column, cap, cfg) -> Panel:
    a, b = keys
    xs = sorted(set(rows[a].tolist()))
    ys = sorted(set(rows[b].tolist()))
    M = np.full((len(ys), len(xs)), np.nan)
    inplateau = np.zeros_like(M, dtype=bool)
    plateau_keys = set()
    if not scan.plateau.empty:
        plateau_keys = {(r[a], r[b]) for _, r in scan.plateau.iterrows()}
    for _, r in rows.iterrows():
        i, j = ys.index(r[b]), xs.index(r[a])
        M[i, j] = r[column]
        inplateau[i, j] = (r[a], r[b]) in plateau_keys

    p = make_figure(cfg.figure_width_mm * 0.7, 62.0, cfg, caption=cap,
                    left_mm=22.0, right_mm=22.0, bottom_mm=14.0,
                    height_is_axes=True)
    ax = p.ax
    cmap = pal.sequential(cfg)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, origin="lower",
                   aspect="auto")
    ax.set_xticks(range(len(xs))); ax.set_yticks(range(len(ys)))
    ax.set_xticklabels([f"{v:g}" if isinstance(v, float) else v for v in xs],
                       fontsize=cfg.base_font_size * 0.8)
    ax.set_yticklabels([f"{v:g}" if isinstance(v, float) else v for v in ys],
                       fontsize=cfg.base_font_size * 0.8)
    ax.set_xlabel(a.replace("_", " "))
    ax.set_ylabel(b.replace("_", " "))

    for i in range(len(ys)):
        for j in range(len(xs)):
            if inplateau[i, j]:
                ax.add_patch(plt_rect(j, i))
    if scan.best:
        try:
            ax.plot(xs.index(scan.best[a]), ys.index(scan.best[b]), marker="*",
                    ms=13, color="#D55E00", markeredgecolor="white", mew=0.7,
                    zorder=6)
        except (KeyError, ValueError):
            pass

    box = ax.get_position()
    cax = p.fig.add_axes((box.x1 + 0.024, box.y0 + box.height * 0.2,
                          0.016, box.height * 0.6))
    cb = p.fig.colorbar(im, cax=cax)
    cb.set_label(column.replace("_", " "), fontsize=cfg.base_font_size * 0.82)
    cb.ax.tick_params(labelsize=cfg.base_font_size * 0.72, width=0.5, length=2)
    return p


def plt_rect(j, i):
    from matplotlib.patches import Rectangle

    return Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white",
                     linewidth=1.3, zorder=5)


# ==========================================================================
def plot_scan_thumbnails(scan, prep, cfg: FigureConfig, max_cells: int = 12,
                         n_cols: int = 4) -> Panel:
    """The actual layout at each setting, so the plateau can be seen not trusted."""
    if not scan.coords:
        raise ValueError("This scan kept no coordinates.")

    cells = list(scan.coords.items())[:max_cells]
    cols = int(min(n_cols, len(cells)))
    rows_n = int(np.ceil(len(cells) / cols))
    deco = pal.categorical(prep.n_groups, cfg)

    plateau_keys = set()
    if not scan.plateau.empty:
        plateau_keys = {tuple(r[k] for k in scan.keys)
                        for _, r in scan.plateau.iterrows()}

    cap = caption_for(
        f"{scan.method_label} at every setting in the scan.",
        "The same samples, laid out under each parameter value. Panels with a "
        "solid frame scored within the tolerance of the best. If those panels "
        "show the same arrangement, the parameter does not matter here; if they "
        "do not, the result is a property of the setting and should be reported "
        "as such.",
        scan.plateau_text())

    legend_mm = 36.0
    w = cfg.figure_width_mm
    cell = (w - legend_mm - 14.0) / cols
    h = rows_n * (cell + 7.0)
    p = make_figure(w, h, cfg, caption=cap, legend_width_mm=legend_mm,
                    left_mm=6.0, right_mm=4.0, top_mm=5.0, bottom_mm=7.0,
                    height_is_axes=True)
    p.ax.set_axis_off()
    box = p.ax.get_position()

    for i, (params, Y) in enumerate(cells):
        r, c = divmod(i, cols)
        sub = p.fig.add_axes((
            box.x0 + c * box.width / cols + 0.008,
            box.y0 + (rows_n - 1 - r) * box.height / rows_n + 0.022,
            box.width / cols * 0.88,
            box.height / rows_n * 0.76,
        ))
        for g in range(prep.n_groups):
            m = prep.group_codes == g
            sub.scatter(Y[m, 0], Y[m, 1], s=cfg.point_size * 0.38,
                        c=deco.for_group(g), marker=pal.marker_for(g),
                        edgecolors="none", alpha=0.85)
        sub.set_xticks([]); sub.set_yticks([])
        good = params in plateau_keys
        for spine in sub.spines.values():
            spine.set_linewidth(1.3 if good else 0.4)
            spine.set_color("#0072B2" if good else "#dddddd")
        label = ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
                          for k, v in zip(scan.keys, params))
        sub.set_title(label, fontsize=cfg.base_font_size * 0.72, pad=2)

    handles = [Line2D([], [], marker=pal.marker_for(g), linestyle="none",
                      markersize=4.5, markerfacecolor=deco.for_group(g),
                      markeredgecolor="white", label=str(prep.group_values[g]))
               for g in range(prep.n_groups)]
    p.legend_from(handles, [h.get_label() for h in handles], title="Group")
    return p


__all__ = ["plot_method_grid", "plot_agreement", "plot_scan_surface",
           "plot_scan_thumbnails"]
