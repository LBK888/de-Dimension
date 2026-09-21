"""Publication figure style, colour handling and occlusion-free layout.

Two things here matter for the brief:

*   PDF/SVG text stays **text**.  ``pdf.fonttype = 42`` embeds TrueType outlines
    with the character codes intact, so every label can be selected and edited
    in Illustrator or Inkscape.  ``svg.fonttype = "none"`` does the same for SVG.
*   Figure legends never overlap the plot.  :func:`make_figure` reserves a
    caption strip and an optional legend column *before* any axes are created,
    so the drawing area is what is left over rather than something a caption is
    dropped on top of.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb, to_hex, to_rgb

from ..config import FigureConfig

MM = 1.0 / 25.4


# --------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------
# The policy -- which palette, and when colour stops working altogether -- lives
# in `somtrack.viz.palette`.  These names are re-exported so call sites written
# against 2.0 keep working, but they now go through that policy, which means the
# old `turbo` fallback for many groups is gone: sampling a continuous rainbow map
# at n points to encode categories is the single most common colour mistake in
# the literature (Crameri, Shephard & Heron 2020), and it was in here.
from .palette import (OKABE_ITO, PALETTES, categorical, glasbey_colors,  # noqa: F401
                      group_colors, legacy_hue_colors, palette_hues)


def composition_color(fractions: np.ndarray, hues: np.ndarray,
                      total_norm: float, n_present: int, n_groups: int,
                      cfg: FigureConfig) -> tuple[float, float, float]:
    """The macro's HSB node encoding, with its arithmetic corrected.

    hue        circular mean of the group hues, weighted by each group's share
               of that node (the original took a *linear* weighted mean of hue
               numbers, so a node split between the first and last group came
               out as the middle group's colour).
    saturation falls as more groups mix in the node -- a pure node is vivid.
    value      rises with how much of the data set the node captured.
    """
    if fractions.sum() <= 0:
        return hsv_to_rgb((0.0, 0.0, cfg.hsb_brightness_floor / 255))

    w = fractions / fractions.sum()
    ang = 2 * np.pi * (hues + cfg.hsb_hue_rotation)
    hue = (np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) / (2 * np.pi)) % 1.0

    mix = 0.0 if n_groups <= 1 else (max(n_present - 1, 0) / max(n_groups - 1, 1))
    s_floor = cfg.hsb_saturation_floor / 255
    sat = 1.0 - (1.0 - s_floor) * mix

    v_floor = cfg.hsb_brightness_floor / 255
    val = v_floor + (1.0 - v_floor) * float(np.clip(total_norm, 0.0, 1.0))
    return hsv_to_rgb((hue, sat, val))


def lighten(color, amount: float = 0.5):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# --------------------------------------------------------------------------
# rcParams
# --------------------------------------------------------------------------
def apply_style(cfg: FigureConfig) -> None:
    fs = cfg.base_font_size
    scale = 1.0 if cfg.style == "publication" else 1.35
    families = [cfg.font_family, "Helvetica", "DejaVu Sans", "sans-serif"]

    mpl.rcParams.update({
        # editable vector text
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "pdf.compression": 6,

        "font.family": "sans-serif",
        "font.sans-serif": families,

        # keep maths in the body font -- the default DejaVu fontset would render
        # a single italic exponent in a different typeface from every other label
        "mathtext.fontset": "custom",
        "mathtext.rm": cfg.font_family,
        "mathtext.it": f"{cfg.font_family}:italic",
        "mathtext.bf": f"{cfg.font_family}:bold",
        "mathtext.default": "regular",
        "font.size": fs * scale,
        "axes.labelsize": fs * scale,
        "axes.titlesize": fs * scale * 1.1,
        "xtick.labelsize": fs * scale * 0.9,
        "ytick.labelsize": fs * scale * 0.9,
        "legend.fontsize": fs * scale * 0.9,
        "figure.titlesize": fs * scale * 1.2,

        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.axisbelow": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,

        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.5,

        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#c8c8c8",
        "legend.borderpad": 0.4,

        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": None,          # explicit layout; never crop our reserved strips
        "savefig.dpi": cfg.dpi_png,
        "figure.dpi": 110,
    })


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
@dataclass
class Panel:
    """A figure with a plotting area, an optional legend column and a caption."""

    fig: plt.Figure
    ax: plt.Axes
    legend_ax: plt.Axes | None = None
    caption_y: float = 0.0

    def legend_from(self, handles, labels, title: str | None = None, ncol: int = 1):
        target = self.legend_ax if self.legend_ax is not None else self.ax
        leg = target.legend(handles, labels, title=title, loc="upper left",
                            ncol=ncol, borderaxespad=0.0,
                            handletextpad=0.6, labelspacing=0.5)
        if title:
            leg.get_title().set_fontweight("bold")
        return leg


def make_figure(
    width_mm: float,
    height_mm: float | None,
    cfg: FigureConfig,
    caption: str | None = None,
    legend_width_mm: float = 0.0,
    left_mm: float = 14.0,
    right_mm: float = 4.0,
    top_mm: float = 8.0,
    bottom_mm: float = 12.0,
    equal: bool = False,
    axes_aspect: float | None = None,
    height_is_axes: bool = False,
) -> Panel:
    """Create a figure whose caption and legend live outside the data area.

    Pass ``axes_aspect`` (data height / data width) with ``height_mm=None`` and
    the figure is sized so the plotting area matches the data exactly -- an
    equal-aspect map then fills its box instead of leaving a band of white
    space above and below it.

    Pass ``height_is_axes=True`` and ``height_mm`` describes the *plotting area*
    rather than the whole figure; the caption strip and margins are added on
    top.  Without it, a caption long enough to wrap to seven lines silently eats
    more than half the height the caller asked for, which is what happened to
    every panel that carries a full explanation.
    """
    cap_text = ""
    cap_h_mm = 0.0
    if caption and cfg.show_caption:
        # Wrap to the figure's own width, not to a fixed character count: a
        # half-width panel wrapped at 96 characters runs off the right edge, and
        # a caption that is cut in half is worse than no caption at all.
        char_mm = cfg.base_font_size * 0.92 * 0.5 * 25.4 / 72.0
        fits = int(max(28, (width_mm - left_mm * 0.35 - 3.0) / char_mm))
        cap_text = "\n".join(textwrap.wrap(caption, min(cfg.caption_wrap, fits)))
        line_mm = cfg.base_font_size * 1.5 * 25.4 / 72.0
        cap_h_mm = line_mm * (cap_text.count("\n") + 1) + 3.0

    if height_mm is None:
        if axes_aspect is None:
            raise ValueError("make_figure needs either height_mm or axes_aspect.")
        axes_w = max(width_mm - left_mm - right_mm - legend_width_mm, 20.0)
        height_mm = float(np.clip(axes_w * axes_aspect + top_mm + bottom_mm + cap_h_mm,
                                  45.0, 340.0))
    elif height_is_axes:
        height_mm = float(np.clip(height_mm + top_mm + bottom_mm + cap_h_mm,
                                  45.0, 340.0))

    fig = plt.figure(figsize=(width_mm * MM, height_mm * MM))

    L = left_mm / width_mm
    R = 1.0 - right_mm / width_mm
    T = 1.0 - top_mm / height_mm
    B = (bottom_mm + cap_h_mm) / height_mm

    legend_frac = legend_width_mm / width_mm
    main_right = R - legend_frac

    ax = fig.add_axes((L, B, max(main_right - L, 0.05), max(T - B, 0.05)))
    if equal:
        ax.set_aspect("equal", adjustable="box")

    legend_ax = None
    if legend_frac > 0:
        legend_ax = fig.add_axes(
            (main_right + 2.0 / width_mm, B, max(legend_frac - 3.0 / width_mm, 0.02),
             max(T - B, 0.05))
        )
        legend_ax.set_axis_off()

    caption_y = 0.0
    if cap_text:
        caption_y = 2.0 / height_mm
        fig.text(left_mm / width_mm * 0.35, caption_y, cap_text,
                 ha="left", va="bottom", fontsize=cfg.base_font_size * 0.92,
                 color="#1a1a1a", linespacing=1.45)

    return Panel(fig=fig, ax=ax, legend_ax=legend_ax, caption_y=caption_y)


def caption_for(main: str, detail: str = "", stats: str = "") -> str:
    parts = [p.strip() for p in (main, detail, stats) if p and p.strip()]
    return "  ".join(parts)


def scalebar_free_corner(ax, xs: np.ndarray, ys: np.ndarray) -> str:
    """Pick the emptiest corner of an axes, for placing annotations."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    counts = {
        "upper left": int(np.sum((xs < mx) & (ys > my))),
        "upper right": int(np.sum((xs > mx) & (ys > my))),
        "lower left": int(np.sum((xs < mx) & (ys < my))),
        "lower right": int(np.sum((xs > mx) & (ys < my))),
    }
    return min(counts, key=counts.get)


__all__ = [
    "apply_style", "make_figure", "Panel", "group_colors", "legacy_hue_colors",
    "palette_hues", "categorical", "glasbey_colors", "composition_color",
    "caption_for", "scalebar_free_corner", "lighten", "OKABE_ITO", "PALETTES", "MM",
]
