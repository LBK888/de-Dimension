"""Colour policy: a decision, not a list of hex codes.

Multivariate figures live or die on colour, and the two ways of getting it wrong
are equally common: a grey-on-grey figure that hides the result, and a figure
that assigns a colour to every one of thirty categories and hides it just as
effectively.  Crameri, Shephard & Heron (2020) documented the second failure
across the literature; this module is the attempt to make it structurally
difficult here.

The central idea is that :func:`categorical` returns a *decision*, not just
colours.  Asked for more categories than any palette can keep apart, it does not
return thirty colours -- it returns ``mode="facet"``, and the figure that asked
is expected to draw small multiples instead.  The choice of when colour stops
working belongs in one place, not in twenty plotting functions.

The rules, and why:

* **Categories never come from a continuous colour map.**  Sampling *viridis* or
  *turbo* at n points implies an ordering the categories do not have, and
  rainbow maps in particular invent boundaries that are not in the data.
* **Eight categories is the ceiling** for colour, because Okabe-Ito -- the
  colour-vision-deficiency-safe standard (Okabe & Ito 2008; Wong 2011) -- has
  eight usable entries.  Beyond that, facet or highlight.
* **Sequential maps are perceptually uniform**, and the print-safe default
  (*cividis*) also survives greyscale and deuteranopia.
* **Diverging maps are used only when zero means something**, and are always
  rendered symmetric about it, so the colour of a value does not depend on the
  range that happened to be in the data.
* **Grey is reserved for context** -- the background groups in a highlight
  figure, non-significant entries -- and is never an experimental group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize, to_hex, to_rgb

Mode = Literal["colour", "highlight", "facet"]

# Okabe & Ito (2008), the colour-universal-design qualitative set
OKABE_ITO = [
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#000000",
]

# Paul Tol's "muted" set, which extends safely to nine
TOL_MUTED = [
    "#332288", "#88CCEE", "#44AA99", "#117733", "#999933",
    "#DDCC77", "#CC6677", "#882255", "#AA4499",
]

CONTEXT_GREY = "#BDBDBD"
INK = "#2A2A2A"

PALETTES: dict[str, list[str]] = {
    "somtrack": OKABE_ITO,
    "okabe_ito": OKABE_ITO,
    "tol_muted": TOL_MUTED,
    "tab10": [to_hex(c) for c in plt.get_cmap("tab10").colors],
    "legacy_hsb": [],                    # generated on demand
}

# sequential maps that are perceptually uniform; cividis is also greyscale-safe
SEQUENTIAL = {"viridis", "cividis", "magma", "inferno", "plasma", "mako", "rocket"}
DIVERGING = {"RdBu_r", "PuOr_r", "BrBG", "coolwarm", "vik", "berlin", "bam"}


# ==========================================================================
@dataclass
class PaletteDecision:
    """Colours plus the instruction that goes with them."""

    colors: list[str]
    mode: Mode = "colour"
    reason: str = ""
    focus: int | None = None
    context: str = CONTEXT_GREY
    citations: tuple[str, ...] = field(default_factory=tuple)

    def __getitem__(self, i: int) -> str:
        return self.colors[i % len(self.colors)] if self.colors else CONTEXT_GREY

    def __len__(self) -> int:
        return len(self.colors)

    def __iter__(self):
        return iter(self.colors)

    @property
    def should_facet(self) -> bool:
        return self.mode == "facet"

    def for_group(self, g: int) -> str:
        """The colour a group gets, honouring highlight mode."""
        if self.mode == "highlight" and self.focus is not None and g != self.focus:
            return self.context
        return self[g]

    def alpha_for(self, g: int) -> float:
        if self.mode == "highlight" and self.focus is not None and g != self.focus:
            return 0.45
        return 1.0


# ==========================================================================
def categorical(n: int, cfg=None, *, palette: str | None = None,
                focus: int | None = None) -> PaletteDecision:
    """Colours for ``n`` experimental groups, or the instruction not to use colour.

    The caller is expected to honour ``mode``.  ``"facet"`` means there are more
    categories than colour can carry and the figure should draw small multiples;
    returning colours anyway would produce a legend nobody can use.
    """
    name = palette or getattr(cfg, "palette", "somtrack")
    ceiling = int(getattr(cfg, "categorical_max", 8) or 8)
    if focus is None:
        focus = getattr(cfg, "highlight_group", None)

    if name == "legacy_hsb":
        return PaletteDecision(legacy_hue_colors(n), "colour",
                               "legacy hue wheel, for comparison with v1.2")

    if focus is not None and n > 1:
        base = _base_colors(max(n, 1), name)
        return PaletteDecision(
            base, "highlight", focus=focus,
            reason=(f"One group is shown in colour and the rest in grey, so the "
                    f"comparison the figure is making is unambiguous."),
            citations=("crameri2020",))

    if n <= len(OKABE_ITO):
        return PaletteDecision(_base_colors(n, name), "colour",
                               "colour-vision-deficiency-safe qualitative palette",
                               citations=("okabe2008", "wong2011"))

    if n <= max(len(TOL_MUTED), ceiling):
        return PaletteDecision(TOL_MUTED[:n], "colour",
                               "extended colour-vision-safe qualitative palette",
                               citations=("crameri2024",))

    colors = glasbey_colors(n)
    if n > ceiling:
        return PaletteDecision(
            colors, "facet",
            reason=(f"{n} groups is more than colour can keep apart; the figure "
                    f"shows one group per panel against the rest in grey instead "
                    f"of assigning {n} colours nobody can match to a legend."),
            citations=("crameri2020", "glasbey2007"))
    return PaletteDecision(colors, "colour",
                           "algorithmically maximally distinct colours",
                           citations=("glasbey2007",))


def _base_colors(n: int, name: str) -> list[str]:
    base = PALETTES.get(name) or OKABE_ITO
    if n <= len(base):
        return list(base[:n])
    return (base + TOL_MUTED + glasbey_colors(n))[:n]


def glasbey_colors(n: int) -> list[str]:
    """Maximally distinct categorical colours, colour-vision-safe where possible.

    Uses the ``glasbey`` package when it is installed and falls back to a
    deterministic sweep of well-separated hues at alternating lightness -- still
    a qualitative construction, never a continuous map sampled at n points.
    """
    try:
        import glasbey as _g

        return list(_g.create_palette(palette_size=n, colorblind_safe=True))
    except Exception:
        pass
    from matplotlib.colors import hsv_to_rgb

    out: list[str] = []
    golden = 0.618033988749895
    for i in range(n):
        h = (i * golden) % 1.0
        s = 0.62 + 0.22 * ((i % 3) / 2.0)
        v = 0.55 + 0.32 * ((i % 2))
        out.append(to_hex(hsv_to_rgb((h, s, min(v, 0.95)))))
    return out


def legacy_hue_colors(n: int, saturation: float = 200 / 255,
                      value: float = 200 / 255) -> list[str]:
    """The v1.2 macro's data-point colours, on a corrected hue wheel.

    The original ``HSBtoRGB`` multiplied the normalised hue by 5 instead of 6,
    which squeezed the wheel into five sixths of a turn and made the last
    group's colour collide with the first.  Group *i* sits at ``i / n`` here.
    """
    from matplotlib.colors import hsv_to_rgb

    return [to_hex(hsv_to_rgb((i / max(n, 1), saturation, value))) for i in range(n)]


def group_colors(n: int, palette: str = "somtrack") -> list[str]:
    """Plain list of colours, for call sites that only need the swatches."""
    return list(categorical(n, palette=palette).colors)


def palette_hues(n: int, palette: str = "somtrack") -> np.ndarray:
    """Hue of each group's marker colour, so map colours match legend swatches.

    Node colours on the composition map are built from these.  In v1.2 the node
    hues and the point hues came from two different formulas on a wheel that was
    also mis-scaled, so a node did not take the colour of the points sitting on
    it.  Deriving both from the same palette keeps them in step at any *n*.
    """
    from matplotlib.colors import rgb_to_hsv

    cols = group_colors(n, palette)
    return np.array([rgb_to_hsv(np.array(to_rgb(c)))[0] for c in cols])


# ==========================================================================
def sequential(cfg=None, print_safe: bool = False):
    """A perceptually uniform sequential colour map."""
    if print_safe:
        name = getattr(cfg, "sequential_print_safe", "cividis")
    else:
        name = getattr(cfg, "sequential_cmap", "viridis")
    return _cmap(name, "viridis")


def diverging(cfg=None):
    """A diverging colour map, for use only where zero is meaningful."""
    return _cmap(getattr(cfg, "diverging_cmap", "RdBu_r"), "RdBu_r")


def _cmap(name: str, fallback: str):
    try:
        import cmcrameri.cm as cmc            # Crameri's scientific colour maps

        if hasattr(cmc, name):
            return getattr(cmc, name)
    except Exception:
        pass
    try:
        return plt.get_cmap(name)
    except Exception:
        return plt.get_cmap(fallback)


def symmetric_norm(values, vmax: float | None = None) -> Normalize:
    """A diverging normalisation that is always centred on zero.

    Without this, the colour of a value depends on the range that happened to be
    present, so the same log ratio reads as 'strongly up' in one panel and
    'mildly up' in the next.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    lim = float(vmax if vmax is not None else (np.abs(v).max() if v.size else 1.0))
    lim = lim if lim > 0 else 1.0
    return Normalize(vmin=-lim, vmax=lim)


def significance_marker(q: float, alpha: float = 0.05) -> str:
    """Significance as a glyph, never as red-versus-green."""
    if not np.isfinite(q):
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < alpha:
        return "*"
    return ""


def lighten(color, amount: float = 0.5):
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


MARKERS = ("o", "s", "^", "D", "v", "P", "X", "<", ">", "*")


def marker_for(i: int) -> str:
    """A second, redundant encoding of group identity, for greyscale and CVD."""
    return MARKERS[i % len(MARKERS)]


# ==========================================================================
def simulate_cvd(rgb: np.ndarray, kind: str = "deuteranopia") -> np.ndarray:
    """Approximate how an image looks with a colour-vision deficiency.

    Brettel/Vienot-style linear approximation in linear RGB.  Good enough to
    catch a figure whose message depends on telling red from green; not a
    clinical model.
    """
    M = {
        "deuteranopia": np.array([[0.625, 0.375, 0.0],
                                  [0.70, 0.30, 0.0],
                                  [0.0, 0.30, 0.70]]),
        "protanopia": np.array([[0.567, 0.433, 0.0],
                                [0.558, 0.442, 0.0],
                                [0.0, 0.242, 0.758]]),
        "tritanopia": np.array([[0.95, 0.05, 0.0],
                                [0.0, 0.433, 0.567],
                                [0.0, 0.475, 0.525]]),
        "greyscale": np.array([[0.2126, 0.7152, 0.0722]] * 3),
    }[kind]
    a = np.asarray(rgb, float)
    flat = a.reshape(-1, a.shape[-1])[:, :3]
    lin = np.where(flat <= 0.04045, flat / 12.92, ((flat + 0.055) / 1.055) ** 2.4)
    out = np.clip(lin @ M.T, 0, 1)
    srgb = np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
    res = a.copy().reshape(-1, a.shape[-1])
    res[:, :3] = np.clip(srgb, 0, 1)
    return res.reshape(a.shape)


def palette_proof(fig, kinds=("deuteranopia", "protanopia", "tritanopia",
                              "greyscale")):
    """Render one figure as its colour-vision-deficiency and greyscale versions."""
    import io

    from matplotlib import pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="white")
    buf.seek(0)
    img = plt.imread(buf)

    proof, axes = plt.subplots(1, len(kinds) + 1,
                               figsize=(3.1 * (len(kinds) + 1), 3.1))
    axes[0].imshow(img)
    axes[0].set_title("as drawn", fontsize=9)
    for ax, kind in zip(axes[1:], kinds):
        ax.imshow(simulate_cvd(img, kind))
        ax.set_title(kind, fontsize=9)
    for ax in axes:
        ax.set_axis_off()
    proof.suptitle("Colour check: the figure must still make its point in every "
                   "panel below", fontsize=9)
    proof.tight_layout()
    return proof


__all__ = [
    "PaletteDecision", "Mode", "categorical", "group_colors", "palette_hues",
    "legacy_hue_colors", "glasbey_colors", "sequential", "diverging",
    "symmetric_norm", "significance_marker", "lighten", "marker_for",
    "simulate_cvd", "palette_proof",
    "OKABE_ITO", "TOL_MUTED", "PALETTES", "CONTEXT_GREY", "INK",
]
