"""Animations: SOM self-organisation over training, and figure sequences.

Rendered through matplotlib's ffmpeg writer, so the output is a standard H.264
MP4 that drops straight into a talk or a supplementary file.  Falls back to an
animated GIF when ffmpeg is not on PATH.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from ..config import ExportConfig, FigureConfig
from ..preprocess import PreparedData
from ..som import SomResult
from . import hexgeom as hg
from .style import composition_color, group_colors, make_figure, palette_hues


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None or "ffmpeg" in animation.writers.list()


def _writer(export: ExportConfig, fps: int | None = None):
    fps = fps or export.mp4_fps
    if ffmpeg_available():
        return animation.FFMpegWriter(
            fps=fps, bitrate=export.mp4_bitrate,
            metadata={"artist": "SOMTrack", "title": "SOM training"},
            extra_args=["-pix_fmt", "yuv420p", "-vcodec", "libx264"],
        ), ".mp4"
    return animation.PillowWriter(fps=fps), ".gif"


# ==========================================================================
def animate_training(
    res: SomResult,
    prep: PreparedData,
    cfg: FigureConfig,
    export: ExportConfig,
    path: str | Path,
    show_points: bool = True,
    progress=None,
) -> Path:
    """Render the map re-organising itself, one frame per recorded epoch."""
    hist = res.history
    frames = [(e, b, b2, d1, d2) for e, b, b2, d1, d2 in zip(
        hist.epochs, hist.bmus, hist.bmu2s, hist.bmu_d, hist.bmu2_d) if b is not None]
    if not frames:
        raise ValueError("No training snapshots were recorded; raise record_every.")

    centres = hg.node_centres(res.lattice.coords)
    n_g = prep.n_groups
    hues = palette_hues(n_g, cfg.palette)
    colors = group_colors(n_g, cfg.palette)
    jitter = hg.stable_jitter(prep.n_samples)

    group_totals = np.bincount(prep.group_codes, minlength=n_g).astype(float)
    group_totals[group_totals == 0] = 1.0

    from .maps import _map_aspect

    w = 190.0
    legend_mm = 44.0

    caption = (
        f"Self-organisation of the {res.width}x{res.height} map over "
        f"{hist.epochs[-1]} epochs ({res.algorithm} training). Node colour encodes "
        "group composition; points are samples, drawn towards their runner-up node "
        "in proportion to how ambiguous the assignment is. The quantisation error "
        "trace at the bottom shows convergence."
    )
    p = make_figure(w, None, cfg, caption=caption, legend_width_mm=legend_mm,
                    bottom_mm=26.0, equal=True, axes_aspect=_map_aspect(res))
    fig, ax = p.fig, p.ax

    box = ax.get_position()
    trace = fig.add_axes((box.x0, p.caption_y + 0.055, box.width, 0.085))
    trace.plot(hist.epochs, hist.qe, color="#0072B2", lw=0.9)
    trace.set_xlim(min(hist.epochs), max(hist.epochs))
    trace.set_ylabel("QE", fontsize=cfg.base_font_size * 0.8)
    trace.tick_params(labelsize=cfg.base_font_size * 0.7)
    trace.set_xlabel("epoch", fontsize=cfg.base_font_size * 0.8)
    marker = trace.axvline(frames[0][0], color="#c0392b", lw=1.0)

    handles = [Line2D([], [], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=colors[g], markeredgecolor="white",
                      label=f"{prep.group_values[g]} (n={int(group_totals[g])})")
               for g in range(n_g)]
    p.legend_from(handles, [hh.get_label() for hh in handles], title="Group")

    state: dict = {}

    def draw(i: int):
        epoch, bmu, bmu2, d1, d2 = frames[i]
        ax.clear()

        M = np.zeros((res.n_nodes, n_g))
        np.add.at(M, (bmu, prep.group_codes), 1.0)
        F = M / group_totals[None, :]
        load = F.sum(axis=1)
        load_n = ((load - load.min()) / (load.max() - load.min())
                  if load.max() > load.min() else np.zeros_like(load))
        face = [composition_color(
            F[k], hues, load_n[k],
            int(np.count_nonzero(F[k] > cfg.mixture_ignore_fraction * max(F[k].sum(), 1e-12))),
            n_g, cfg) for k in range(res.n_nodes)]

        hg.draw_hexes(ax, centres, face, edgecolor=cfg.hex_edgecolor, linewidth=0.35)
        if show_points:
            pos = hg.sample_positions(bmu, bmu2, d1, d2, centres, jitter,
                                      cfg.jitter_strength)
            for g in range(n_g):
                m = prep.group_codes == g
                ax.scatter(pos[m, 0], pos[m, 1], s=cfg.point_size, c=colors[g],
                           edgecolors="white", linewidths=0.5, zorder=5)
        hg.set_map_limits(ax, centres)
        ax.set_title(f"epoch {epoch} / {hist.epochs[-1]}", pad=4)
        marker.set_xdata([epoch, epoch])
        if progress and (i % 5 == 0 or i == len(frames) - 1):
            progress(i + 1, len(frames))
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    writer, ext = _writer(export)
    out = Path(path).with_suffix(ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=writer, dpi=150)
    plt.close(fig)
    return out


# ==========================================================================
def animate_figures(
    panels: list,
    path: str | Path,
    export: ExportConfig,
    hold_seconds: float = 2.5,
) -> Path:
    """Stitch a list of rendered panels into a slideshow video.

    Useful for supplementary material where a reviewer should be walked through
    composition map -> per-group maps -> component planes in order.
    """
    import io

    from PIL import Image

    images = []
    for p in panels:
        buf = io.BytesIO()
        p.fig.savefig(buf, format="png", dpi=150)
        buf.seek(0)
        images.append(np.asarray(Image.open(buf).convert("RGB")))
    if not images:
        raise ValueError("No panels supplied.")

    hmax = max(im.shape[0] for im in images)
    wmax = max(im.shape[1] for im in images)
    padded = []
    for im in images:
        canvas = np.full((hmax, wmax, 3), 255, dtype=np.uint8)
        y = (hmax - im.shape[0]) // 2
        x = (wmax - im.shape[1]) // 2
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        padded.append(canvas)

    fig = plt.figure(figsize=(wmax / 150, hmax / 150), dpi=150)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    art = ax.imshow(padded[0])

    fps = max(1, int(round(1.0 / hold_seconds)))
    def draw(i):
        art.set_data(padded[i])
        return [art]

    anim = animation.FuncAnimation(fig, draw, frames=len(padded), blit=False)
    writer, ext = _writer(export, fps=fps)
    out = Path(path).with_suffix(ext)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=writer, dpi=150)
    plt.close(fig)
    return out


__all__ = ["animate_training", "animate_figures", "ffmpeg_available"]
