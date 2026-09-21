"""Hexagonal lattice drawing primitives.

Screen coordinates use the same Cartesian embedding the SOM trains on
(:func:`somtrack.som.hex_cartesian`), with the Y axis flipped so that row 0 sits
at the top -- matching the orientation of the original ImageJ map.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import RegularPolygon

# pointy-top hexagon whose width equals one column pitch
HEX_RADIUS = 1.0 / math.sqrt(3.0)
ROW_PITCH = math.sqrt(3.0) / 2.0


def node_centres(coords: np.ndarray) -> np.ndarray:
    """(n_nodes, 2) screen centres for integer ``(col, row)`` coordinates."""
    coords = np.asarray(coords)
    x = coords[:, 0].astype(float) + 0.5 * (coords[:, 1] % 2)
    y = -coords[:, 1].astype(float) * ROW_PITCH
    return np.column_stack([x, y])


def hex_patches(centres: np.ndarray, radius: float = HEX_RADIUS) -> list[RegularPolygon]:
    return [RegularPolygon((float(x), float(y)), numVertices=6,
                           radius=radius, orientation=0.0)
            for x, y in centres]


def draw_hexes(
    ax,
    centres: np.ndarray,
    facecolors,
    edgecolor: str = "#3a3a3a",
    linewidth: float = 0.4,
    alpha=None,
    radius: float = HEX_RADIUS,
    zorder: float = 1.0,
) -> PatchCollection:
    """Draw one hexagon per node.  ``alpha`` may be a scalar or per-node array."""
    patches = hex_patches(centres, radius)
    pc = PatchCollection(patches, match_original=False)
    fc = np.asarray([_rgba(c) for c in facecolors], dtype=float)
    if alpha is not None:
        a = np.full(len(patches), float(alpha)) if np.isscalar(alpha) else np.asarray(alpha, float)
        fc[:, 3] = a
    pc.set_facecolor(fc)
    pc.set_edgecolor(edgecolor)
    pc.set_linewidth(linewidth)
    pc.set_zorder(zorder)
    ax.add_collection(pc)
    return pc


def _rgba(c):
    from matplotlib.colors import to_rgba

    return to_rgba(c)


def set_map_limits(ax, centres: np.ndarray, pad: float = 0.9) -> None:
    ax.set_xlim(centres[:, 0].min() - pad, centres[:, 0].max() + pad)
    ax.set_ylim(centres[:, 1].min() - pad, centres[:, 1].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()


def sample_positions(
    bmu: np.ndarray,
    bmu2: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    centres: np.ndarray,
    jitter: np.ndarray | None = None,
    jitter_strength: float = 0.20,
    radius: float = HEX_RADIUS,
) -> np.ndarray:
    """Where to draw each sample inside its winning hexagon.

    Ported from the macro: a sample sits at its best-matching node, displaced
    towards its second-best node by ``d1 / d2``.  A sample that fits its node
    far better than any other sits at the centre; an ambiguous one drifts to the
    border it shares with its runner-up, which makes boundary cases visible.

    The angular jitter is drawn once per sample and reused for every frame, so
    points no longer twitch between animation frames as they did in v1.2.
    """
    c1 = centres[bmu]
    c2 = centres[bmu2]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip(np.nan_to_num(d1 / np.where(d2 > 0, d2, np.nan), nan=0.0), 0.0, 1.0)
    ang = np.arctan2(c2[:, 1] - c1[:, 1], c2[:, 0] - c1[:, 0])
    if jitter is not None:
        ang = ang + (jitter - 0.5) * jitter_strength * 2.0
    # 0.866 = inradius / circumradius, so a point never escapes its hexagon
    r = frac * radius * 0.866
    return np.column_stack([c1[:, 0] + np.cos(ang) * r, c1[:, 1] + np.sin(ang) * r])


def stable_jitter(n: int, seed: int = 12345) -> np.ndarray:
    return np.random.default_rng(seed).random(n)


def cluster_outline(ax, centres: np.ndarray, member: np.ndarray, color,
                    radius: float = HEX_RADIUS, linewidth: float = 1.4,
                    fill_alpha: float = 0.18, zorder: float = 3.0) -> None:
    """Outline a set of nodes as one merged region (the macro's k-means overlay)."""
    if not np.any(member):
        return
    try:
        from matplotlib.patches import PathPatch
        from matplotlib.path import Path
        from shapely.geometry import Polygon
        from shapely.ops import unary_union

        polys = []
        for cx, cy in centres[member]:
            pts = [(cx + radius * math.cos(math.pi / 2 + i * math.pi / 3),
                    cy + radius * math.sin(math.pi / 2 + i * math.pi / 3)) for i in range(6)]
            polys.append(Polygon(pts).buffer(0.01))
        merged = unary_union(polys)
        geoms = getattr(merged, "geoms", [merged])
        for g in geoms:
            verts = list(g.exterior.coords)
            path = Path(verts, [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY])
            ax.add_patch(PathPatch(path, facecolor=color, alpha=fill_alpha,
                                   edgecolor=color, linewidth=linewidth, zorder=zorder))
        return
    except Exception:
        pass

    # shapely is optional; fall back to translucent hexagons plus a ring
    draw_hexes(ax, centres[member], [color] * int(member.sum()),
               edgecolor=color, linewidth=linewidth, alpha=fill_alpha, zorder=zorder)


__all__ = [
    "HEX_RADIUS", "ROW_PITCH", "node_centres", "draw_hexes", "set_map_limits",
    "sample_positions", "stable_jitter", "cluster_outline", "hex_patches",
]
