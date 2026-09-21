"""Per-metric figures that show the data, the replicate structure and the effect.

Two conventions from the cell- and organismal-biology literature, both of which
a reviewer now expects and neither of which the 2.0 figure set had:

**SuperPlots** (Lord, Velle, Mullins & Fritz-Laylin 2020).  Every individual is
drawn, coloured by the replicate it came from, and each replicate's own mean is
drawn on top as a large marker.  A difference that is present in every replicate
looks completely different from one driven by a single unusual dish, and a bar
chart cannot tell them apart.

**Estimation plots** (Ho, Tumkaya, Aryal, Choi & Claridge-Chang 2019).  The
effect size and its bootstrap confidence interval get their own aligned axis
beneath the raw data, so the reader sees how big the difference is rather than
only whether it cleared a threshold.
"""

from __future__ import annotations

import numpy as np

from ..config import FigureConfig
from . import palette as pal
from .style import Panel, caption_for, make_figure


# ==========================================================================
def _jitter(n: int, seed: int, width: float = 0.17) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, n)


def _values_for(prep, feature: str) -> np.ndarray | None:
    """Original measurement units where available, scaled values otherwise."""
    src = prep.source
    if src is not None and feature in src.frame.columns:
        return src.frame.loc[prep.keep_mask, feature].to_numpy(float)
    if feature in prep.feature_names:
        return prep.X[:, list(prep.feature_names).index(feature)]
    return None


def _unit(prep, feature: str) -> str:
    src = prep.source
    return src.units.get(feature, "") if src is not None else ""


# ==========================================================================
def plot_superplot(prep, feature: str, cfg: FigureConfig,
                   effects=None) -> Panel:
    """One metric, showing every individual and every replicate mean."""
    v = _values_for(prep, feature)
    if v is None:
        raise ValueError(f"No values for metric '{feature}'.")

    deco = pal.categorical(prep.n_groups, cfg)
    reps = np.asarray(prep.replicates)
    rep_values = list(dict.fromkeys(reps.tolist()))
    has_reps = len(rep_values) > 1
    unit = _unit(prep, feature)

    cap = caption_for(
        f"{feature}" + (f" ({unit})" if unit else "") + ", by group.",
        ("Small points are individual samples, shaded by which replicate they "
         "came from; large outlined markers are the mean of each replicate. "
         "A difference that holds in every replicate is a different result from "
         "one carried by a single replicate, and that distinction is the whole "
         "reason for drawing it this way (Lord et al. 2020)."
         if has_reps else
         "Small points are individual samples; the bar is the group mean with "
         "its standard deviation."),
        f"n = {len(v)}.")

    p = make_figure(cfg.figure_width_mm * 0.5, 58.0, cfg, caption=cap,
                    left_mm=22.0, right_mm=8.0, bottom_mm=14.0,
                    height_is_axes=True)
    ax = p.ax

    for g in range(prep.n_groups):
        m = prep.group_codes == g
        if not m.any():
            continue
        base = deco.for_group(g)
        vals = v[m]
        ax.scatter(g + _jitter(vals.size, g), vals, s=11,
                   color=pal.lighten(base, 0.35), alpha=0.7,
                   edgecolors="none", zorder=2)
        ax.hlines(np.nanmean(vals), g - 0.32, g + 0.32, color=pal.INK,
                  lw=1.6, zorder=5)
        sd = np.nanstd(vals, ddof=1) if vals.size > 1 else 0.0
        ax.vlines(g, np.nanmean(vals) - sd, np.nanmean(vals) + sd,
                  color=pal.INK, lw=0.9, zorder=4)

        if has_reps:
            for ri, rv in enumerate(rep_values):
                sel = m & (reps == rv)
                if not sel.any():
                    continue
                ax.scatter(g + (ri - len(rep_values) / 2 + 0.5) * 0.10,
                           np.nanmean(v[sel]), s=52, color=base,
                           marker=pal.marker_for(ri), edgecolors="white",
                           linewidths=0.8, zorder=6)

    ax.set_xticks(range(prep.n_groups))
    ax.set_xticklabels([str(x) for x in prep.group_values], rotation=20,
                       ha="right", fontsize=cfg.base_font_size * 0.85)
    ax.set_ylabel(f"{feature}" + (f"  [{unit}]" if unit else ""))
    ax.grid(True, axis="y", alpha=0.16, lw=0.4)
    ax.set_xlim(-0.6, prep.n_groups - 0.4)

    if has_reps:
        from matplotlib.lines import Line2D

        handles = [Line2D([], [], marker=pal.marker_for(i), linestyle="none",
                          markersize=5, markerfacecolor="#888888",
                          markeredgecolor="white", label=f"replicate {rv}")
                   for i, rv in enumerate(rep_values[:8])]
        ax.legend(handles=handles, fontsize=cfg.base_font_size * 0.7,
                  frameon=False, loc="best", ncol=1)
    return p


# ==========================================================================
def plot_estimation(prep, feature: str, cfg: FigureConfig,
                    reference: int = 0, effects=None) -> Panel:
    """Raw data above, effect size with its bootstrap interval below."""
    v = _values_for(prep, feature)
    if v is None:
        raise ValueError(f"No values for metric '{feature}'.")
    if prep.n_groups < 2:
        raise ValueError("An estimation plot needs at least two groups.")

    from ..stats.effects import bootstrap_ci

    deco = pal.categorical(prep.n_groups, cfg)
    unit = _unit(prep, feature)
    ref_vals = v[prep.group_codes == reference]

    diffs, cis, labels = [], [], []
    for g in range(prep.n_groups):
        if g == reference:
            continue
        other = v[prep.group_codes == g]
        if ref_vals.size < 2 or other.size < 2:
            continue
        diffs.append(float(np.nanmean(other) - np.nanmean(ref_vals)))
        cis.append(bootstrap_ci(ref_vals, other,
                                lambda a, b: float(np.mean(b) - np.mean(a)),
                                n_boot=2000, random_state=0))
        labels.append(str(prep.group_values[g]))

    cap = caption_for(
        f"{feature}" + (f" ({unit})" if unit else "")
        + f": difference from {prep.group_values[reference]}.",
        "The upper axis shows every sample. The lower axis shows the difference "
        "in means against the reference group, with a bias-corrected and "
        "accelerated bootstrap 95% interval. An interval that clears zero is the "
        "same information a p value carries, and the width of it is information "
        "a p value does not carry (Ho et al. 2019).",
        f"n = {len(v)}; 2000 bootstrap resamples.")

    p = make_figure(cfg.figure_width_mm * 0.54, 80.0, cfg, caption=cap,
                    left_mm=22.0, right_mm=8.0, bottom_mm=14.0, top_mm=6.0,
                    height_is_axes=True)
    p.ax.set_axis_off()
    box = p.ax.get_position()
    top = p.fig.add_axes((box.x0, box.y0 + box.height * 0.42,
                          box.width, box.height * 0.58))
    bot = p.fig.add_axes((box.x0, box.y0, box.width, box.height * 0.34),
                         sharex=top)

    for g in range(prep.n_groups):
        m = prep.group_codes == g
        vals = v[m]
        if vals.size == 0:
            continue
        top.scatter(g + _jitter(vals.size, g), vals, s=12,
                    color=deco.for_group(g), alpha=0.65, edgecolors="none")
        top.hlines(np.nanmean(vals), g - 0.28, g + 0.28, color=pal.INK, lw=1.5)
    top.set_ylabel(f"{feature}" + (f"  [{unit}]" if unit else ""),
                   fontsize=cfg.base_font_size * 0.9)
    top.tick_params(labelbottom=False)
    top.grid(True, axis="y", alpha=0.16, lw=0.4)

    xs = [i for i in range(prep.n_groups) if i != reference][:len(diffs)]
    bot.axhline(0, color="#888888", lw=0.8, ls="-")
    for x, d, (lo, hi) in zip(xs, diffs, cis):
        colour = deco.for_group(x)
        clears = np.isfinite(lo) and (lo > 0 or hi < 0)
        bot.vlines(x, lo, hi, color=colour, lw=2.2, alpha=0.95 if clears else 0.55)
        bot.plot(x, d, "o", color=colour, ms=6, markeredgecolor="white", mew=0.8)
    bot.set_xticks(range(prep.n_groups))
    bot.set_xticklabels([str(x) for x in prep.group_values], rotation=20,
                        ha="right", fontsize=cfg.base_font_size * 0.85)
    bot.set_ylabel(f"difference\nfrom {prep.group_values[reference]}",
                   fontsize=cfg.base_font_size * 0.82)
    bot.grid(True, axis="y", alpha=0.16, lw=0.4)
    bot.set_xlim(-0.6, prep.n_groups - 0.4)
    del labels
    return p


# ==========================================================================
def plot_effect_forest(effects, cfg: FigureConfig, top: int = 14) -> Panel:
    """Standardised effect size for every metric, with bootstrap intervals."""
    if effects is None or effects.empty:
        raise ValueError("No effect-size table.")

    d = effects.dropna(subset=["hedges_g"]).head(top).iloc[::-1]
    cap = caption_for(
        "Effect size per metric, with bootstrap confidence intervals.",
        "Hedges' g is the difference in means in units of the pooled standard "
        "deviation, with the small-sample correction that matters at these group "
        "sizes. Intervals are bias-corrected and accelerated bootstrap "
        "intervals; one that clears zero corresponds to a significant test, and "
        "its width says how well the size of the effect is pinned down.",
        "Conventional landmarks: 0.2 small, 0.5 medium, 0.8 large "
        "(Cohen 1988); they are conventions, not thresholds.")

    h = max(48.0, 6.4 * len(d))
    p = make_figure(cfg.figure_width_mm * 0.72, h, cfg, caption=cap,
                    left_mm=52.0, right_mm=8.0, bottom_mm=13.0,
                    height_is_axes=True)
    ax = p.ax
    y = np.arange(len(d))
    lo = d["g_ci_low"].to_numpy(float)
    hi = d["g_ci_high"].to_numpy(float)
    g = d["hedges_g"].to_numpy(float)
    clears = np.isfinite(lo) & ((lo > 0) | (hi < 0))

    for i in range(len(d)):
        colour = "#0072B2" if clears[i] else pal.CONTEXT_GREY
        ax.hlines(y[i], lo[i], hi[i], color=colour, lw=1.8,
                  alpha=0.95 if clears[i] else 0.7)
        ax.plot(g[i], y[i], "o", color=colour, ms=5, markeredgecolor="white",
                mew=0.7)
    ax.axvline(0, color="#888888", lw=0.8)
    for x in (-0.8, -0.5, -0.2, 0.2, 0.5, 0.8):
        ax.axvline(x, color="#dddddd", lw=0.4, ls=":", zorder=0)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.feature}\n{r.group_b} vs {r.group_a}"
                        for r in d.itertuples()],
                       fontsize=cfg.base_font_size * 0.72)
    ax.set_xlabel("Hedges' g (95% bootstrap interval)")
    ax.grid(True, axis="x", alpha=0.14, lw=0.4)
    return p


__all__ = ["plot_superplot", "plot_estimation", "plot_effect_forest"]
