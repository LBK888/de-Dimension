"""Figures for the statistics layer: the answer, and the evidence for it.

These are the panels that carry the claim.  Everything else in the figure set
describes the data; these say whether the difference is there, how large it is,
whether it is a shift or a change in spread, and which metrics carry it.

The verdict card is deliberately the first page of the report.  A reader who
looks at nothing else should still come away with the right conclusion, and a
reader who disagrees with it can find every number behind it in the panels that
follow.
"""

from __future__ import annotations

import textwrap

import numpy as np
from matplotlib import pyplot as plt

from ..config import FigureConfig
from . import palette as pal
from .style import MM, Panel, caption_for, make_figure

CONF_COLOUR = {"strong": "#0072B2", "moderate": "#009E73",
               "weak": "#E69F00", "none": "#7F7F7F", "unknown": "#7F7F7F"}


# ==========================================================================
def plot_verdict(verdict, cfg: FigureConfig, width_mm: float | None = None) -> Panel:
    """The plain-language conclusion, as page one of the report."""
    w = width_mm or cfg.figure_width_mm
    accent = CONF_COLOUR.get(verdict.confidence, "#0072B2")

    body: list[tuple[str, str]] = [("head", verdict.headline)]
    for f in verdict.findings:
        body.append(("body", f))
    if verdict.separable:
        body.append(("body", "Separates: " + "; ".join(verdict.separable) + "."))
    if verdict.not_separable:
        body.append(("body", "Does not separate: "
                     + "; ".join(verdict.not_separable) + "."))
    if verdict.drivers:
        body.append(("body", "Main contributing metrics: "
                     + ", ".join(verdict.drivers) + "."))
    if verdict.caveats:
        body.append(("sub", "Limits of this analysis"))
        for c in verdict.caveats:
            body.append(("small", "- " + c))

    wrapped: list[tuple[str, str]] = []
    for kind, text in body:
        width = {"head": 58, "sub": 70, "small": 96}.get(kind, 84)
        for line in textwrap.wrap(text, width) or [""]:
            wrapped.append((kind, line))
        wrapped.append((kind, ""))

    line_mm = cfg.base_font_size * 1.55 * 25.4 / 72.0
    h = max(60.0, 18.0 + line_mm * len(wrapped))

    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0.035, 0.03), 0.012, 0.94, transform=ax.transAxes,
                               facecolor=accent, edgecolor="none"))

    y = 0.955
    for kind, line in wrapped:
        size = {"head": cfg.base_font_size * 1.55,
                "sub": cfg.base_font_size * 1.05,
                "small": cfg.base_font_size * 0.85}.get(kind, cfg.base_font_size)
        weight = "bold" if kind in ("head", "sub") else "normal"
        colour = accent if kind == "head" else (
            "#555555" if kind == "small" else pal.INK)
        if line:
            ax.text(0.075, y, line, transform=ax.transAxes, va="top", ha="left",
                    fontsize=size, fontweight=weight, color=colour)
        y -= (line_mm / h) * (1.35 if kind == "head" else 1.0)

    return Panel(fig=fig, ax=ax)


# ==========================================================================
def plot_separation_summary(sep, cfg: FigureConfig) -> Panel:
    """PERMANOVA and PERMDISP side by side: is it a shift, or a change in spread?"""
    cap = caption_for(
        "Multivariate tests of group difference.",
        "PERMANOVA asks whether the group averages differ and reports R2, the "
        "share of multivariate variation the grouping accounts for. PERMDISP asks "
        "the separate question of whether the groups differ in how variable they "
        "are. PERMANOVA rejects for either reason, so both are needed before a "
        "significant result can be read as a shift in phenotype.",
        f"{sep.n_permutations} permutations on {sep.metric} distances; "
        f"{sep.design_note}")

    p = make_figure(cfg.figure_width_mm, 62.0, cfg, caption=cap,
                    left_mm=32.0, right_mm=6.0, bottom_mm=12.0,
                    height_is_axes=True)
    ax = p.ax
    ax.set_axis_off()
    box = ax.get_position()
    left = p.fig.add_axes((box.x0, box.y0, box.width * 0.44, box.height))
    right = p.fig.add_axes((box.x0 + box.width * 0.58, box.y0,
                            box.width * 0.42, box.height))

    # --- left: effect size with the significance of each test ---------------
    rows = [("PERMANOVA\n(group averages)", sep.permanova_R2, sep.permanova_p,
             sep.permanova_F, "pseudo-F"),
            ("PERMDISP\n(group spread)", np.nan, sep.permdisp_p,
             sep.permdisp_F, "F"),
            ("Energy test\n(whole distribution)", np.nan, sep.energy_p,
             sep.energy_statistic, "E")]
    ypos = np.arange(len(rows))[::-1]
    for (label, r2, pv, stat, symbol), yv in zip(rows, ypos):
        sig = np.isfinite(pv) and pv < 0.05
        colour = "#0072B2" if sig else pal.CONTEXT_GREY
        if np.isfinite(r2):
            left.barh(yv, r2, height=0.5, color=colour,
                      edgecolor=pal.INK, linewidth=0.5)
            # inside the bar, so it cannot collide with the p value on the right
            left.text(r2 - 0.012, yv, f"R2 = {r2:.3f}", va="center", ha="right",
                      fontsize=cfg.base_font_size * 0.85,
                      color="white" if sig else pal.INK)
        else:
            left.barh(yv, 0.0, height=0.5, color="none")
        ptext = ("p n/a" if not np.isfinite(pv)
                 else (f"p = {pv:.4f}" if pv >= 0.0001 else "p < 0.0001"))
        star = pal.significance_marker(pv)
        # The p value sits on the row it belongs to, inside the plot area.  Put
        # below the tick label and it collides with the two-line group name.
        left.annotate(f"{ptext}{'  ' + star if star else ''}",
                      xy=(0.985, yv), xycoords=("axes fraction", "data"),
                      ha="right", va="center",
                      fontsize=cfg.base_font_size * 0.78,
                      color=pal.INK if sig else "#777777")
        if not np.isfinite(r2) and np.isfinite(stat):
            left.annotate(f"{symbol} = {stat:.4g}", xy=(0.012, yv),
                          xycoords=("axes fraction", "data"), ha="left",
                          va="center", fontsize=cfg.base_font_size * 0.85,
                          color=pal.INK)
    left.set_yticks(ypos)
    left.set_yticklabels([r[0] for r in rows], fontsize=cfg.base_font_size * 0.85)
    left.set_xlim(0, max(0.05, (sep.permanova_R2 if np.isfinite(sep.permanova_R2)
                                else 0.1) * 1.55))
    left.set_xlabel("variation explained (R2)")
    left.grid(True, axis="x", alpha=0.18, lw=0.4)

    # --- right: within-group spread, the PERMDISP picture -------------------
    z = np.asarray(sep.distance_to_centroid, float)
    if z.size and len(sep.group_values):
        deco = pal.categorical(len(sep.group_values), cfg)
        k = len(sep.group_values)
        groups = np.asarray(sep.group_codes, int)
        if groups.size != z.size:
            groups = np.zeros(z.size, dtype=int)
        for g in range(k):
            vals = z[groups == g]
            if vals.size == 0:
                continue
            x = g + (np.random.default_rng(g).uniform(-0.16, 0.16, vals.size))
            right.scatter(x, vals, s=10, color=deco.for_group(g),
                          alpha=0.55, edgecolors="none", zorder=2)
            right.hlines(np.mean(vals), g - 0.28, g + 0.28,
                         color=pal.INK, lw=1.4, zorder=3)
        right.set_xticks(range(k))
        right.set_xticklabels([str(v) for v in sep.group_values],
                              fontsize=cfg.base_font_size * 0.8, rotation=20,
                              ha="right")
        right.set_ylabel("distance to group centroid")
        right.set_title("within-group spread", fontsize=cfg.base_font_size,
                        pad=3)
        right.grid(True, axis="y", alpha=0.18, lw=0.4)
    else:
        right.set_axis_off()

    return p


def plot_pairwise_separation(sep, cfg: FigureConfig) -> Panel:
    """Which specific pairs of groups differ, with FDR-corrected q values."""
    df = sep.pairwise
    if df is None or df.empty:
        raise ValueError("No pairwise separation table.")

    labels = [str(v) for v in sep.group_values]
    k = len(labels)
    M = np.full((k, k), np.nan)
    Q = np.full((k, k), np.nan)
    index = {v: i for i, v in enumerate(labels)}
    for _, r in df.iterrows():
        a, b = index.get(str(r["group_a"])), index.get(str(r["group_b"]))
        if a is None or b is None:
            continue
        M[a, b] = M[b, a] = r["R2"]
        Q[a, b] = Q[b, a] = r.get("q", np.nan)

    cap = caption_for(
        "Every pair of groups, tested separately.",
        "Colour is the share of variation the pair's grouping explains (PERMANOVA "
        "R2); the annotation gives the false-discovery-rate corrected q value. "
        "A pair with a high R2 and a small q is separable; the useful biological "
        "statement is usually which pairs are not.",
        f"{sep.n_permutations} permutations; Benjamini-Hochberg across "
        f"{len(df)} pairs.")
    size = max(70.0, 16.0 + 13.0 * k)
    p = make_figure(min(cfg.figure_width_mm, size + 34.0), size, cfg, caption=cap,
                    left_mm=26.0, right_mm=22.0, top_mm=8.0,
                    height_is_axes=True)
    ax = p.ax
    cmap = pal.sequential(cfg)
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=0,
                   vmax=max(0.05, np.nanmax(M) if np.isfinite(M).any() else 0.1))
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    ax.set_xticklabels(labels, rotation=35, ha="right",
                       fontsize=cfg.base_font_size * 0.85)
    ax.set_yticklabels(labels, fontsize=cfg.base_font_size * 0.85)

    for i in range(k):
        for j in range(k):
            if i == j or not np.isfinite(M[i, j]):
                continue
            star = pal.significance_marker(Q[i, j])
            txt = f"{M[i, j]:.2f}\n{star or 'n.s.'}"
            lum = M[i, j] / max(np.nanmax(M), 1e-9)
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=cfg.base_font_size * 0.72,
                    color="white" if lum > 0.55 else pal.INK)
    _side_colorbar(p, im, "variation explained (R2)", cfg)
    ax.set_title("Pairwise group separation", pad=5)
    return p


# ==========================================================================
def plot_classifiability(cls, cfg: FigureConfig) -> Panel:
    """Observed cross-validated accuracy against the permutation null."""
    cap = caption_for(
        "Can a held-out sample be assigned to its group?",
        "The histogram is the balanced accuracy achieved by the identical "
        "cross-validation after the group labels were shuffled; it is what this "
        "analysis scores when there is nothing to find. The line is the observed "
        "value. Reading the observed accuracy without this null is the most "
        "common way to overstate a separation.",
        f"{cls.model_label}; {cls.cv_name}; {cls.n_permutations} label "
        f"permutations. {cls.cv_note}")

    p = make_figure(cfg.figure_width_mm * 0.92, 58.0, cfg, caption=cap,
                    left_mm=18.0, right_mm=6.0, bottom_mm=12.0,
                    height_is_axes=True)
    ax = p.ax
    ax.set_axis_off()
    box = ax.get_position()
    hist = p.fig.add_axes((box.x0, box.y0, box.width * 0.56, box.height))
    conf = p.fig.add_axes((box.x0 + box.width * 0.70, box.y0,
                           box.width * 0.30, box.height))

    null = np.asarray(cls.permutation_null, float)
    if null.size:
        hist.hist(null, bins=min(40, max(10, null.size // 15)),
                  color=pal.CONTEXT_GREY, edgecolor="white", linewidth=0.3,
                  label="labels shuffled")
    hist.axvline(cls.chance, color="#777777", lw=0.9, ls=":", label="chance")
    obs_colour = "#0072B2" if cls.separable else "#D55E00"
    hist.axvline(cls.balanced_accuracy, color=obs_colour, lw=2.0,
                 label="observed")
    lo, hi = cls.ba_ci
    if np.isfinite(lo):
        hist.axvspan(lo, hi, color=obs_colour, alpha=0.14, lw=0)
    hist.set_xlabel("balanced accuracy")
    hist.set_ylabel("permutations")
    hist.set_xlim(0, 1.02)
    hist.legend(fontsize=cfg.base_font_size * 0.78, loc="upper left",
                frameon=False)
    ptxt = ("p not available" if not np.isfinite(cls.permutation_p)
            else f"p = {cls.permutation_p:.4f}")
    hist.set_title(f"observed {cls.balanced_accuracy:.2f}"
                   + (f" [{lo:.2f}-{hi:.2f}]" if np.isfinite(lo) else "")
                   + f", {ptxt}",
                   fontsize=cfg.base_font_size * 0.95, pad=4)

    C = np.asarray(cls.confusion, float)
    if C.size:
        with np.errstate(invalid="ignore"):
            R = C / np.maximum(C.sum(axis=1, keepdims=True), 1)
        im = conf.imshow(R, cmap=pal.sequential(cfg, print_safe=True),
                         vmin=0, vmax=1)
        labels = [str(v) for v in cls.group_values]
        conf.set_xticks(range(len(labels))); conf.set_yticks(range(len(labels)))
        conf.set_xticklabels(labels, rotation=40, ha="right",
                             fontsize=cfg.base_font_size * 0.72)
        conf.set_yticklabels(labels, fontsize=cfg.base_font_size * 0.72)
        conf.set_xlabel("predicted", fontsize=cfg.base_font_size * 0.8)
        conf.set_ylabel("actual", fontsize=cfg.base_font_size * 0.8)
        conf.set_title("confusion", fontsize=cfg.base_font_size * 0.9, pad=3)
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                conf.text(j, i, f"{R[i, j] * 100:.0f}", ha="center", va="center",
                          fontsize=cfg.base_font_size * 0.65,
                          color="white" if R[i, j] > 0.55 else pal.INK)
        del im
    else:
        conf.set_axis_off()
    return p


# ==========================================================================
def plot_activation_patterns(cls, cfg: FigureConfig, top: int = 14) -> Panel:
    """Classifier weights next to Haufe activation patterns."""
    if cls.activation is None or not cls.feature_names:
        raise ValueError("No classifier weights to draw.")

    names = list(cls.feature_names)
    A = np.linalg.norm(np.atleast_2d(cls.activation), axis=0)
    W = (np.linalg.norm(np.atleast_2d(cls.weights), axis=0)
         if cls.weights is not None else np.zeros_like(A))
    if A.size != len(names):
        raise ValueError("Activation pattern does not match the metric list.")

    order = np.argsort(-A)[:top][::-1]
    A_n = A / max(A.max(), 1e-12)
    W_n = W / max(W.max(), 1e-12)

    cap = caption_for(
        "Which metrics actually carry the group difference.",
        "A classifier weight says how the model extracts the signal and can be "
        "large for a metric carrying none, purely to cancel correlated noise in "
        "another (a suppressor). The Haufe-transformed activation pattern is the "
        "covariance between the metric and the discriminant score, and is the "
        "column that supports the sentence 'this metric differs between groups'. "
        "Where the two disagree, trust the activation.",
        f"{cls.model_label}; both series scaled to their own maximum.")

    h = max(46.0, 6.2 * len(order))
    p = make_figure(cfg.figure_width_mm * 0.86, h, cfg, caption=cap,
                    left_mm=44.0, right_mm=8.0, bottom_mm=12.0,
                    height_is_axes=True)
    ax = p.ax
    y = np.arange(len(order))
    ax.barh(y + 0.20, A_n[order], height=0.38, color="#0072B2",
            edgecolor="none", label="activation pattern (interpretable)")
    ax.barh(y - 0.20, W_n[order], height=0.38, color=pal.CONTEXT_GREY,
            edgecolor="none", label="raw weight (not interpretable)")
    ax.set_yticks(y)
    ax.set_yticklabels([names[i] for i in order],
                       fontsize=cfg.base_font_size * 0.82)
    ax.set_xlabel("relative magnitude")
    ax.set_xlim(0, 1.06)
    ax.grid(True, axis="x", alpha=0.18, lw=0.4)
    ax.legend(fontsize=cfg.base_font_size * 0.78, loc="lower right",
              frameon=False)
    return p


def plot_importance(imp, cfg: FigureConfig, top: int = 14) -> Panel:
    """Permutation importance per metric and per correlated cluster."""
    if imp is None or imp.table.empty:
        raise ValueError("No importance table.")

    t = imp.table.head(top).iloc[::-1]
    has_clusters = not imp.cluster_table.empty and len(imp.cluster_table) > 1

    cap = caption_for(
        "How much accuracy each metric is worth.",
        "Importance is the balanced accuracy lost when that metric alone is "
        "shuffled in held-out data. Correlated metrics share their importance "
        "and each looks weak on its own, so the right-hand panel permutes whole "
        f"clusters of metrics correlated above |r| = {imp.clustered_at:g} "
        "together; a cluster worth much more than its members individually is "
        "one aspect of behaviour measured several ways."
        if has_clusters else
        "Importance is the balanced accuracy lost when that metric alone is "
        "shuffled in held-out data.",
        f"{imp.model}, {imp.n_repeats} shuffles per fold; cross-validated "
        f"balanced accuracy {imp.cv_balanced_accuracy:.2f} "
        f"(chance {imp.chance:.2f}).")

    h = max(48.0, 6.0 * len(t))
    p = make_figure(cfg.figure_width_mm, h, cfg, caption=cap,
                    left_mm=42.0, right_mm=6.0, bottom_mm=12.0,
                    height_is_axes=True)
    ax = p.ax
    if has_clusters:
        ax.set_axis_off()
        box = ax.get_position()
        ax = p.fig.add_axes((box.x0, box.y0, box.width * 0.52, box.height))
        right = p.fig.add_axes((box.x0 + box.width * 0.62, box.y0,
                                box.width * 0.38, box.height))
    else:
        right = None

    y = np.arange(len(t))
    ax.barh(y, t["importance"], xerr=t["sd"], height=0.66, color="#0072B2",
            edgecolor="none", error_kw=dict(lw=0.7, ecolor="#555555"))
    ax.axvline(0, color="#888888", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(t["feature"], fontsize=cfg.base_font_size * 0.82)
    ax.set_xlabel("balanced accuracy lost when shuffled")
    ax.grid(True, axis="x", alpha=0.18, lw=0.4)
    ax.set_title("per metric", fontsize=cfg.base_font_size, pad=3)

    if right is not None:
        c = imp.cluster_table.head(10).iloc[::-1]
        yy = np.arange(len(c))
        right.barh(yy, c["importance"], xerr=c["sd"], height=0.66,
                   color="#009E73", edgecolor="none",
                   error_kw=dict(lw=0.7, ecolor="#555555"))
        right.axvline(0, color="#888888", lw=0.6)
        right.set_yticks(yy)
        right.set_yticklabels(
            [textwrap.shorten(m, 34, placeholder="...")
             for m in c["members"]], fontsize=cfg.base_font_size * 0.7)
        right.set_xlabel("accuracy lost")
        right.grid(True, axis="x", alpha=0.18, lw=0.4)
        right.set_title("per correlated cluster", fontsize=cfg.base_font_size,
                        pad=3)
    return p


# ==========================================================================
def _side_colorbar(p: Panel, mappable, label: str, cfg: FigureConfig):
    box = p.ax.get_position()
    cax = p.fig.add_axes((box.x1 + 0.022, box.y0 + box.height * 0.18,
                          0.016, box.height * 0.64))
    cb = p.fig.colorbar(mappable, cax=cax)
    cb.set_label(label, fontsize=cfg.base_font_size * 0.85)
    cb.ax.tick_params(labelsize=cfg.base_font_size * 0.75, width=0.5, length=2)
    cb.outline.set_linewidth(0.4)
    return cb


__all__ = [
    "plot_verdict", "plot_separation_summary", "plot_pairwise_separation",
    "plot_classifiability", "plot_activation_patterns", "plot_importance",
]
