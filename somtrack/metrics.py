"""Locomotion and biophysical metric engine.

Every metric is declared once in :data:`METRICS` with a label, a unit
expression, a category and the input columns it needs.  The UI builds its
selection tree straight from that registry, and figure captions pull their
axis labels from it -- there is no second place where a metric name is spelled
out.

All per-track work happens on pre-computed numpy arrays held by
:class:`TrackKinematics`, so adding a metric costs one array reduction rather
than a re-walk of the coordinates.  The whole table for ~10^3 tracks of ~10^3
spots computes in a couple of seconds, against tens of minutes for the
per-spot interpreted loop in the original macro.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .config import TrackConfig
from .io_tables import FeatureDataset, SpotDataset

# water at 25 C, used only for the Reynolds estimate
_RHO_WATER = 997.0        # kg m^-3
_MU_WATER = 8.9e-4        # Pa s


# ==========================================================================
# Registry
# ==========================================================================
@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    unit: str                      # may contain {L} and {T} placeholders
    category: str
    description: str
    requires: tuple[str, ...] = ()  # optional spot columns needed
    legacy: bool = False            # present in the ImageJ macro
    default: bool = True            # pre-ticked in the UI

    def formatted_unit(self, length_unit: str, time_unit: str) -> str:
        return self.unit.format(L=length_unit, T=time_unit)


METRICS: dict[str, MetricSpec] = {}


def _reg(*args, **kwargs) -> None:
    spec = MetricSpec(*args, **kwargs)
    METRICS[spec.name] = spec


C_BASIC = "Basic kinematics"
C_ANG = "Angular / directional"
C_SHAPE = "Path geometry"
C_DIFF = "Diffusion (MSD)"
C_PERS = "Persistence & memory"
C_INT = "Intermittency (CTRW)"
C_DIST = "Speed distribution"
C_OSC = "Oscillation & biophysics"
C_MORPH = "Morphology / intensity"
C_POS = "Position"

# ---- legacy, basic kinematics -------------------------------------------
_reg("n_spots", "Track length", "spots", C_BASIC, "Number of detections in the track.", legacy=True, default=False)
_reg("duration", "Track duration", "{T}", C_BASIC, "Elapsed time from first to last detection.", legacy=True, default=False)
_reg("avg_velocity", "Mean speed", "{L}/{T}", C_BASIC, "Trimmed mean of instantaneous speed.", legacy=True)
_reg("std_velocity", "Speed SD", "{L}/{T}", C_BASIC, "Trimmed SD of instantaneous speed.", legacy=True)
_reg("avg_acce", "Mean acceleration", "{L}/{T}^2", C_BASIC, "Trimmed mean of the signed tangential acceleration.", legacy=True)
_reg("std_acce", "Acceleration SD", "{L}/{T}^2", C_BASIC, "Trimmed SD of tangential acceleration.", legacy=True)
_reg("accum_distance", "Path length", "{L}", C_BASIC, "Total distance travelled along the track.", legacy=True)

# ---- angular -------------------------------------------------------------
_reg("avg_angv", "Mean angular velocity", "deg/{T}", C_ANG, "Trimmed mean of the signed turning rate.", legacy=True)
_reg("std_angv", "Angular velocity SD", "deg/{T}", C_ANG, "Trimmed SD of the turning rate.", legacy=True)
_reg("avg_angacc", "Mean angular acceleration", "deg/{T}^2", C_ANG, "Trimmed mean of the change in turning rate.", legacy=True)
_reg("std_angacc", "Angular acceleration SD", "deg/{T}^2", C_ANG, "Trimmed SD of angular acceleration.", legacy=True)
_reg("avg_meander", "Mean absolute meander", "deg/{L}", C_ANG, "Turning per unit path length (curvature proxy).", legacy=True)
_reg("std_meander", "Meander SD", "deg/{L}", C_ANG, "SD of turning per unit path length.", legacy=True)
_reg("mean_heading", "Mean heading", "deg", C_ANG, "Circular mean of step headings (replaces the linear angle average).", default=False)
_reg("heading_R", "Directional concentration R", "0-1", C_ANG, "Mean resultant length of headings; 1 = perfectly straight bearing.")
_reg("heading_circ_sd", "Heading circular SD", "deg", C_ANG, "Circular standard deviation of step headings.")
_reg("rayleigh_p", "Rayleigh p (directionality)", "p", C_ANG, "Rayleigh test against a uniform heading distribution.", default=False)
_reg("turn_abs_mean", "Mean |turn angle|", "deg", C_ANG, "Mean absolute turn between consecutive steps.")
_reg("turn_bias", "Turn bias (L/R)", "deg", C_ANG, "Mean signed turn; non-zero indicates a circling bias.")
_reg("turn_kurtosis", "Turn-angle kurtosis", "-", C_ANG, "Peakedness of the turn distribution; high = long straight runs with rare sharp turns.")
_reg("turn_entropy", "Turn-angle entropy", "bits", C_ANG, "Shannon entropy of the binned turn distribution; low = stereotyped turning.")

# ---- path geometry -------------------------------------------------------
_reg("net_displacement", "Net displacement", "{L}", C_SHAPE, "Straight-line distance from first to last position.")
_reg("straightness", "Straightness index", "0-1", C_SHAPE, "Net displacement / path length (confinement ratio).")
_reg("tortuosity", "Tortuosity", "-", C_SHAPE, "Path length / net displacement.", default=False)
_reg("sinuosity", "Sinuosity", "{L}^-0.5", C_SHAPE, "Bovet & Benhamou sinuosity index of the correlated random walk.")
_reg("radius_gyration", "Radius of gyration", "{L}", C_SHAPE, "RMS distance of the track from its own centroid.")
_reg("hull_area", "Convex hull area", "{L}^2", C_SHAPE, "Area explored, as the convex hull of all positions.")
_reg("hull_perimeter", "Convex hull perimeter", "{L}", C_SHAPE, "Perimeter of the explored area.", default=False)
_reg("exploration_ratio", "Exploration ratio", "-", C_SHAPE, "Hull area / (pi * Rg^2); 1 = isotropic filling, <1 = elongated or reused path.")
_reg("bbox_aspect", "Bounding-box aspect", "-", C_SHAPE, "Long/short axis ratio of the PCA-aligned bounding box.")
_reg("fractal_dimension", "Katz fractal dimension", "-", C_SHAPE, "Path complexity; 1 = straight line, higher = more convoluted.")

# ---- diffusion -----------------------------------------------------------
_reg("msd_alpha", "MSD anomalous exponent", "-", C_DIFF, "Slope of log MSD vs log lag: 1 = Brownian, >1 super-diffusive, <1 confined.")
_reg("msd_D", "Generalised diffusion coefficient", "{L}^2/{T}^a", C_DIFF, "Prefactor of the MSD power-law fit.")
_reg("msd_r2", "MSD fit R^2", "-", C_DIFF, "Goodness of the MSD power-law fit; low values invalidate alpha and D.", default=False)

# ---- persistence ---------------------------------------------------------
_reg("dir_autocorr_time", "Directional correlation time", "{T}", C_PERS, "Lag at which <cos(delta heading)> falls to 1/e.")
_reg("persistence_length", "Persistence length", "{L}", C_PERS, "Directional correlation time x mean speed.")
_reg("vel_autocorr_time", "Speed correlation time", "{T}", C_PERS, "Lag at which the speed autocorrelation falls to 1/e.", default=False)
_reg("speed_hurst", "Speed Hurst exponent (DFA)", "-", C_PERS, "Detrended fluctuation exponent of the speed series; >0.5 = persistent bursts.")

# ---- intermittency -------------------------------------------------------
_reg("avg_haltT", "Mean halt duration", "{T}", C_INT, "Mean accumulated pause time (legacy definition).", legacy=True)
_reg("std_haltT", "Halt duration SD", "{T}", C_INT, "SD of accumulated pause time.", legacy=True)
_reg("fraction_moving", "Fraction of time moving", "0-1", C_INT, "Proportion of frames above the halt speed threshold.")
_reg("stop_frequency", "Stop frequency", "1/{T}", C_INT, "Number of move-to-pause transitions per unit time.")
_reg("mean_run_duration", "Mean run duration", "{T}", C_INT, "Mean length of an uninterrupted moving bout.")
_reg("mean_pause_duration", "Mean pause duration", "{T}", C_INT, "Mean length of an uninterrupted pause bout.")
_reg("burstiness", "Burstiness B", "-1..1", C_INT, "Goh-Barabasi burstiness of run durations; >0 = bursty, <0 = regular.")
_reg("memory_coefficient", "CTRW memory M", "-1..1", C_INT, "Lag-1 correlation of consecutive run durations.", default=False)

# ---- speed distribution --------------------------------------------------
_reg("speed_max", "Peak speed", "{L}/{T}", C_DIST, "95th-percentile speed, robust to tracking spikes.")
_reg("speed_cv", "Speed CV", "-", C_DIST, "Coefficient of variation of speed.")
_reg("speed_skew", "Speed skewness", "-", C_DIST, "Asymmetry of the speed distribution.", default=False)
_reg("accel_rms", "RMS acceleration", "{L}/{T}^2", C_DIST, "Root-mean-square tangential acceleration (effort proxy).")
_reg("jerk_rms", "RMS jerk", "{L}/{T}^3", C_DIST, "Root-mean-square rate of change of acceleration (smoothness proxy).")

# ---- oscillation / biophysics -------------------------------------------
_reg("beat_frequency", "Turning beat frequency", "1/{T}", C_OSC, "Dominant frequency of the angular-velocity series (tail/cilia beat proxy).")
_reg("beat_power_ratio", "Beat spectral purity", "0-1", C_OSC, "Power at the dominant frequency / total power; high = regular rhythmic beating.")
_reg("speed_dominant_freq", "Speed oscillation frequency", "1/{T}", C_OSC, "Dominant frequency of the speed series (stroke cycle proxy).", default=False)
_reg("lateral_amplitude", "Lateral oscillation amplitude", "{L}", C_OSC, "RMS deviation of the path from its smoothed centre line.")
_reg("strouhal", "Strouhal number", "-", C_OSC, "f x 2A / U; efficient undulatory swimming sits near 0.2-0.4.")
_reg("reynolds", "Reynolds number", "-", C_OSC, "rho x U x L / mu in water at 25 C; needs a body-length column or estimated diameter.", requires=("body_length",))
_reg("speed_bodylengths", "Speed in body lengths", "BL/{T}", C_OSC, "Mean speed normalised by body length.", requires=("body_length",))

# ---- position ------------------------------------------------------------
_reg("avg_X", "Mean X (relative)", "{L}", C_POS, "Mean X position relative to the track's own minimum.", legacy=True, default=False)
_reg("avg_Y", "Mean Y (relative)", "{L}", C_POS, "Mean Y position relative to the track's own minimum.", legacy=True, default=False)
_reg("XYdispersion", "XY dispersion", "{L}", C_POS, "Geometric mean of the X and Y positional SDs.", legacy=True)

# ---- morphology / intensity (only when the columns exist) ---------------
_reg("avg_Area", "Mean area", "{L}^2", C_MORPH, "Mean segmented body area.", requires=("area",), legacy=True)
_reg("std_Area", "Area SD", "{L}^2", C_MORPH, "SD of body area (shape change / rotation proxy).", requires=("area",), legacy=True)
_reg("avg_Major", "Mean major axis", "{L}", C_MORPH, "Mean fitted-ellipse major axis.", requires=("major",), legacy=True)
_reg("std_Major", "Major axis SD", "{L}", C_MORPH, "SD of the major axis.", requires=("major",), legacy=True)
_reg("avg_Minor", "Mean minor axis", "{L}", C_MORPH, "Mean fitted-ellipse minor axis.", requires=("minor",), legacy=True)
_reg("std_Minor", "Minor axis SD", "{L}", C_MORPH, "SD of the minor axis.", requires=("minor",), legacy=True)
_reg("aspect_ratio", "Body aspect ratio", "-", C_MORPH, "Major / minor axis; elongation of the body.", requires=("major", "minor"))
_reg("aspect_ratio_cv", "Aspect ratio CV", "-", C_MORPH, "Variability of elongation, i.e. body deformation during swimming.", requires=("major", "minor"))
_reg("avg_AngleP", "Mean body orientation", "deg", C_MORPH, "Circular mean of the fitted-ellipse orientation.", requires=("angle",), legacy=True, default=False)
_reg("std_AngleP", "Body orientation circular SD", "deg", C_MORPH, "Circular SD of body orientation.", requires=("angle",), legacy=True)
_reg("avg_drift", "Mean yaw drift", "deg", C_MORPH, "Mean angle between body axis and direction of travel (slip / crabbing).", requires=("angle",), legacy=True)
_reg("std_drift", "Yaw drift SD", "deg", C_MORPH, "SD of the body-axis / travel-direction angle.", requires=("angle",), legacy=True)
_reg("avg_Intnsty", "Mean intensity", "a.u.", C_MORPH, "Mean particle intensity.", requires=("intensity_mean",), legacy=True)
_reg("std_Intnsty", "Intensity SD", "a.u.", C_MORPH, "SD of particle intensity.", requires=("intensity_mean",), legacy=True)
_reg("avg_estZ", "Mean focus index", "0-1", C_MORPH, "Per-track normalised intensity SD; a relative depth-of-focus proxy, not a calibrated Z.", requires=("intensity_std",), legacy=True)
_reg("std_estZ", "Focus index SD", "0-1", C_MORPH, "Variability of the focus index, i.e. vertical excursion proxy.", requires=("intensity_std",), legacy=True)


def metrics_by_category() -> dict[str, list[MetricSpec]]:
    out: dict[str, list[MetricSpec]] = {}
    for spec in METRICS.values():
        out.setdefault(spec.category, []).append(spec)
    return out


def available_metrics(spot_columns: Iterable[str]) -> list[str]:
    """Metric names whose required input columns are present."""
    have = set(spot_columns)
    if {"major"} & have:
        have.add("body_length")
    if "diameter" in have:
        have.add("body_length")
    return [n for n, s in METRICS.items() if all(r in have for r in s.requires)]


# ==========================================================================
# Small numerical helpers
# ==========================================================================
def trimmed_stats(a: np.ndarray, frac: float = 0.6) -> tuple[float, float]:
    """Mean and SD of the central ``frac`` of the ranked values.

    Faithful to the macro's ``DecileRangeArray``.  Note the SD of a trimmed
    sample is biased low; it is kept for comparability with published v1.2
    results and can be disabled via ``TrackConfig.use_trimmed_stats``.
    """
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    if frac >= 1.0 or a.size < 5:
        return float(np.mean(a)), float(np.std(a))
    s = np.sort(a)
    lo = int(math.floor(s.size * (0.5 - frac / 2)))
    hi = int(math.floor(s.size * (0.5 + frac / 2))) + 1
    s = s[lo:min(hi, s.size)]
    if s.size == 0:
        return float(np.mean(a)), float(np.std(a))
    return float(np.mean(s)), float(np.std(s))


def _safe(fn: Callable, *a, **k) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = fn(*a, **k)
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def circ_mean(angles_rad: np.ndarray) -> float:
    a = angles_rad[np.isfinite(angles_rad)]
    if a.size == 0:
        return np.nan
    return float(math.atan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def circ_R(angles_rad: np.ndarray) -> float:
    a = angles_rad[np.isfinite(angles_rad)]
    if a.size == 0:
        return np.nan
    return float(math.hypot(np.mean(np.cos(a)), np.mean(np.sin(a))))


def circ_sd(angles_rad: np.ndarray) -> float:
    R = circ_R(angles_rad)
    if not np.isfinite(R) or R <= 0:
        return np.nan
    return float(math.sqrt(-2.0 * math.log(min(R, 1.0))))


def rayleigh_p(angles_rad: np.ndarray) -> float:
    """Rayleigh test for non-uniformity (Zar, eq. 27.4 approximation)."""
    a = angles_rad[np.isfinite(angles_rad)]
    n = a.size
    if n < 3:
        return np.nan
    R = circ_R(a)
    Z = n * R * R
    p = math.exp(math.sqrt(1 + 4 * n + 4 * (n * n - Z * Z)) - (1 + 2 * n))
    return float(min(1.0, max(0.0, p)))


def dfa_exponent(x: np.ndarray, min_win: int = 8) -> float:
    """Detrended fluctuation analysis, order 1."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4 * min_win:
        return np.nan
    y = np.cumsum(x - x.mean())
    wins = np.unique(np.round(np.logspace(np.log10(min_win), np.log10(n // 4), 12)).astype(int))
    wins = wins[wins >= min_win]
    if wins.size < 4:
        return np.nan
    F = []
    for w in wins:
        m = n // w
        seg = y[: m * w].reshape(m, w)
        tt = np.arange(w)
        # least-squares detrend of every segment at once
        A = np.vstack([tt, np.ones_like(tt)]).T
        coef, *_ = np.linalg.lstsq(A, seg.T, rcond=None)
        resid = seg.T - A @ coef
        F.append(math.sqrt(np.mean(resid ** 2)))
    F = np.asarray(F)
    ok = np.isfinite(F) & (F > 0)
    if ok.sum() < 4:
        return np.nan
    slope = np.polyfit(np.log(wins[ok]), np.log(F[ok]), 1)[0]
    return float(slope)


def dominant_frequency(x: np.ndarray, fs: float) -> tuple[float, float]:
    """Dominant frequency and its share of total power, via a periodogram."""
    x = x[np.isfinite(x)]
    if x.size < 16 or fs <= 0:
        return np.nan, np.nan
    x = x - x.mean()
    if not np.any(x):
        return np.nan, np.nan
    win = np.hanning(x.size)
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(x.size, d=1.0 / fs)
    if spec.size < 3:
        return np.nan, np.nan
    spec[0] = 0.0                                   # drop DC
    total = spec.sum()
    if total <= 0:
        return np.nan, np.nan
    k = int(np.argmax(spec))
    return float(freqs[k]), float(spec[k] / total)


def autocorr_time(x: np.ndarray, dt: float) -> float:
    """Lag at which the autocorrelation first drops below 1/e."""
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom <= 0:
        return np.nan
    max_lag = max(2, n // 2)
    ac = np.array([np.dot(x[: n - k], x[k:]) / denom for k in range(max_lag)])
    below = np.flatnonzero(ac < 1 / math.e)
    if below.size == 0:
        return float(max_lag * dt)
    k = below[0]
    if k == 0:
        return 0.0
    # linear interpolation between the bracketing lags
    a0, a1 = ac[k - 1], ac[k]
    frac = (a0 - 1 / math.e) / max(a0 - a1, 1e-12)
    return float((k - 1 + frac) * dt)


def _bouts(mask: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Durations of consecutive True runs in ``mask``, weighted by ``dt``."""
    if mask.size == 0:
        return np.array([])
    idx = np.flatnonzero(np.diff(mask.astype(np.int8)) != 0) + 1
    segments = np.split(np.arange(mask.size), idx)
    return np.array([dt[s].sum() for s in segments if mask[s[0]]])


# ==========================================================================
# Per-track kinematics
# ==========================================================================
@dataclass
class TrackKinematics:
    """All derived arrays for one track, computed once."""

    t: np.ndarray            # calibrated time, n
    x: np.ndarray            # calibrated position, n
    y: np.ndarray
    dt: np.ndarray           # n-1
    dx: np.ndarray
    dy: np.ndarray
    step: np.ndarray         # n-1
    speed: np.ndarray        # n-1
    heading: np.ndarray      # n-1, radians
    turn: np.ndarray         # n-2, radians, wrapped
    ang_vel: np.ndarray      # n-2, deg / time
    ang_acc: np.ndarray      # n-3, deg / time^2
    accel: np.ndarray        # n-2
    jerk: np.ndarray         # n-3
    cumdist: np.ndarray      # n-1
    moving: np.ndarray       # n-1, bool
    halt_threshold: float
    extras: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return self.t.size

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n > 1 else 0.0

    @property
    def mean_dt(self) -> float:
        return float(np.median(self.dt)) if self.dt.size else np.nan

    @property
    def fs(self) -> float:
        m = self.mean_dt
        return 1.0 / m if m and np.isfinite(m) and m > 0 else np.nan


def build_kinematics(
    sub: pd.DataFrame, cfg: TrackConfig, halt_threshold: float | None = None
) -> TrackKinematics:
    scale = 1.0 if cfg.coords_are_calibrated else cfg.pixel_size
    tscale = 1.0 if cfg.times_are_calibrated else cfg.frame_interval

    x = sub["x"].to_numpy(float) * scale
    y = sub["y"].to_numpy(float) * scale
    t = sub["t"].to_numpy(float) * tscale

    if cfg.smoothing_window and cfg.smoothing_window >= 5 and x.size > cfg.smoothing_window:
        from scipy.signal import savgol_filter

        w = cfg.smoothing_window | 1
        x = savgol_filter(x, w, 2)
        y = savgol_filter(y, w, 2)

    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    dx, dy = np.diff(x), np.diff(y)
    step = np.hypot(dx, dy)
    speed = step / dt
    heading = np.arctan2(dy, dx)
    turn = wrap_pi(np.diff(heading))

    dt_mid = 0.5 * (dt[:-1] + dt[1:]) if dt.size > 1 else np.array([])
    ang_vel = np.degrees(turn) / dt_mid if dt_mid.size else np.array([])
    accel = np.diff(speed) / dt_mid if dt_mid.size else np.array([])

    if ang_vel.size > 1:
        dt_mid2 = 0.5 * (dt_mid[:-1] + dt_mid[1:])
        ang_acc = np.diff(ang_vel) / dt_mid2
        jerk = np.diff(accel) / dt_mid2
    else:
        ang_acc = np.array([])
        jerk = np.array([])

    if halt_threshold is None:
        halt_threshold = cfg.halt_threshold_px * cfg.pixel_size / max(cfg.frame_interval, 1e-9)
    moving = speed > halt_threshold

    extras: dict[str, np.ndarray] = {}
    for col in ("area", "major", "minor", "angle", "intensity_mean", "intensity_std", "diameter", "z"):
        if col in sub.columns:
            v = sub[col].to_numpy(float)
            if col in ("area",):
                v = v * (scale ** 2) if not cfg.coords_are_calibrated else v
            elif col in ("major", "minor", "diameter"):
                v = v * scale if not cfg.coords_are_calibrated else v
            extras[col] = v

    return TrackKinematics(
        t=t, x=x, y=y, dt=dt, dx=dx, dy=dy, step=step, speed=speed,
        heading=heading, turn=turn, ang_vel=ang_vel, ang_acc=ang_acc,
        accel=accel, jerk=jerk, cumdist=np.cumsum(np.nan_to_num(step)),
        moving=moving, halt_threshold=halt_threshold, extras=extras,
    )


def auto_halt_threshold(spots: pd.DataFrame, cfg: TrackConfig) -> float:
    """Otsu split of the pooled log-speed histogram.

    The macro used a fixed 1.5 pixel step, which does not transfer between
    magnifications or frame rates.  Splitting the bimodal speed distribution
    adapts the pause criterion to the actual recording.
    """
    scale = 1.0 if cfg.coords_are_calibrated else cfg.pixel_size
    tscale = 1.0 if cfg.times_are_calibrated else cfg.frame_interval
    speeds: list[np.ndarray] = []
    for _, sub in spots.groupby("uid", observed=True, sort=False):
        x = sub["x"].to_numpy(float) * scale
        y = sub["y"].to_numpy(float) * scale
        t = sub["t"].to_numpy(float) * tscale
        dt = np.diff(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.hypot(np.diff(x), np.diff(y)) / dt
        speeds.append(s[np.isfinite(s) & (s > 0)])
    if not speeds:
        return cfg.halt_threshold_px * scale / max(cfg.frame_interval, 1e-9)
    allsp = np.concatenate(speeds)
    if allsp.size < 50:
        return float(np.percentile(allsp, 10)) if allsp.size else 0.0
    logs = np.log10(allsp)
    hist, edges = np.histogram(logs, bins=128)
    p = hist / hist.sum()
    centres = 0.5 * (edges[:-1] + edges[1:])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    k = int(np.nanargmax(sigma_b))
    thr = float(10 ** centres[k])
    # never let the split exceed the median speed
    return float(min(thr, np.median(allsp)))


# ==========================================================================
# Metric computation
# ==========================================================================
def _compute_one(k: TrackKinematics, cfg: TrackConfig, wanted: set[str]) -> dict[str, float]:
    r: dict[str, float] = {}
    tf = cfg.trim_fraction if cfg.use_trimmed_stats else 1.0
    put = r.__setitem__

    def want(*names: str) -> bool:
        return any(n in wanted for n in names)

    # ---------- basic ----------
    put("n_spots", float(k.n))
    put("duration", k.duration)
    m, s = trimmed_stats(k.speed, tf)
    put("avg_velocity", m); put("std_velocity", s)
    m, s = trimmed_stats(k.accel, tf)
    put("avg_acce", m); put("std_acce", s)
    put("accum_distance", float(np.nansum(k.step)))

    # ---------- angular ----------
    m, s = trimmed_stats(k.ang_vel, tf)
    put("avg_angv", m); put("std_angv", s)
    m, s = trimmed_stats(k.ang_acc, tf)
    put("avg_angacc", m); put("std_angacc", s)

    # meander = turn per unit distance travelled over the same two steps
    if k.turn.size and k.step.size > 1:
        seg = k.step[:-1] + k.step[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            meander = np.degrees(k.turn) / seg
        m, s = trimmed_stats(np.abs(meander), tf)
    else:
        m = s = np.nan
    put("avg_meander", m); put("std_meander", s)

    put("mean_heading", math.degrees(circ_mean(k.heading)) if k.heading.size else np.nan)
    put("heading_R", circ_R(k.heading))
    put("heading_circ_sd", math.degrees(circ_sd(k.heading)) if k.heading.size else np.nan)
    put("rayleigh_p", rayleigh_p(k.heading))
    put("turn_abs_mean", float(np.degrees(np.nanmean(np.abs(k.turn)))) if k.turn.size else np.nan)
    put("turn_bias", math.degrees(circ_mean(k.turn)) if k.turn.size else np.nan)

    if k.turn.size > 3:
        tt = np.degrees(k.turn[np.isfinite(k.turn)])
        sd = tt.std()
        put("turn_kurtosis", float(np.mean(((tt - tt.mean()) / sd) ** 4) - 3.0) if sd > 0 else np.nan)
        hist, _ = np.histogram(tt, bins=18, range=(-180, 180))
        p = hist[hist > 0] / hist.sum()
        put("turn_entropy", float(-(p * np.log2(p)).sum()))
    else:
        put("turn_kurtosis", np.nan); put("turn_entropy", np.nan)

    # ---------- path geometry ----------
    net = float(math.hypot(k.x[-1] - k.x[0], k.y[-1] - k.y[0])) if k.n > 1 else 0.0
    path = float(np.nansum(k.step))
    put("net_displacement", net)
    put("straightness", net / path if path > 0 else np.nan)
    put("tortuosity", path / net if net > 0 else np.nan)

    if k.turn.size > 2 and np.nanmean(k.step) > 0:
        put("sinuosity", float(1.18 * np.nanstd(k.turn) / math.sqrt(np.nanmean(k.step))))
    else:
        put("sinuosity", np.nan)

    cx, cy = k.x.mean(), k.y.mean()
    rg = float(math.sqrt(np.mean((k.x - cx) ** 2 + (k.y - cy) ** 2)))
    put("radius_gyration", rg)

    hull_area = hull_per = np.nan
    if want("hull_area", "hull_perimeter", "exploration_ratio") and k.n >= 3:
        try:
            from scipy.spatial import ConvexHull

            h = ConvexHull(np.column_stack([k.x, k.y]))
            hull_area, hull_per = float(h.volume), float(h.area)
        except Exception:
            pass
    put("hull_area", hull_area); put("hull_perimeter", hull_per)
    put("exploration_ratio",
        hull_area / (math.pi * rg * rg) if (np.isfinite(hull_area) and rg > 0) else np.nan)

    if k.n >= 3:
        pts = np.column_stack([k.x - cx, k.y - cy])
        cov = np.cov(pts.T)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        put("bbox_aspect", float(math.sqrt(ev[0] / ev[1])) if ev[1] > 1e-12 else np.nan)
    else:
        put("bbox_aspect", np.nan)

    if path > 0 and k.n > 2:
        d_max = float(np.max(np.hypot(k.x - k.x[0], k.y - k.y[0])))
        mean_step = path / (k.n - 1)
        nn = path / mean_step
        put("fractal_dimension",
            float(math.log10(nn) / (math.log10(nn) + math.log10(max(d_max, 1e-12) / path)))
            if d_max > 0 and abs(math.log10(nn) + math.log10(d_max / path)) > 1e-9 else np.nan)
    else:
        put("fractal_dimension", np.nan)

    # ---------- MSD ----------
    a, D, r2 = _msd_fit(k, cfg.msd_max_lag_fraction)
    put("msd_alpha", a); put("msd_D", D); put("msd_r2", r2)

    # ---------- persistence ----------
    dtm = k.mean_dt
    if k.heading.size > 8 and np.isfinite(dtm):
        put("dir_autocorr_time", _heading_corr_time(k.heading, dtm))
    else:
        put("dir_autocorr_time", np.nan)
    put("persistence_length",
        r["dir_autocorr_time"] * r["avg_velocity"]
        if np.isfinite(r["dir_autocorr_time"]) and np.isfinite(r["avg_velocity"]) else np.nan)
    put("vel_autocorr_time", autocorr_time(k.speed, dtm) if np.isfinite(dtm) else np.nan)
    put("speed_hurst", dfa_exponent(k.speed) if want("speed_hurst") else np.nan)

    # ---------- intermittency ----------
    dt_ok = np.nan_to_num(k.dt)
    runs = _bouts(k.moving, dt_ok)
    pauses = _bouts(~k.moving, dt_ok)
    total_t = float(dt_ok.sum())
    put("fraction_moving", float(dt_ok[k.moving].sum() / total_t) if total_t > 0 else np.nan)
    n_stops = int(np.count_nonzero(np.diff(k.moving.astype(np.int8)) == -1))
    put("stop_frequency", n_stops / total_t if total_t > 0 else np.nan)
    put("mean_run_duration", float(runs.mean()) if runs.size else 0.0)
    put("mean_pause_duration", float(pauses.mean()) if pauses.size else 0.0)

    if runs.size > 2:
        mu, sd = runs.mean(), runs.std()
        put("burstiness", float((sd - mu) / (sd + mu)) if (sd + mu) > 0 else np.nan)
        if runs.size > 3:
            a1, a2 = runs[:-1], runs[1:]
            denom = a1.std() * a2.std()
            put("memory_coefficient",
                float(np.mean((a1 - a1.mean()) * (a2 - a2.mean())) / denom) if denom > 0 else np.nan)
        else:
            put("memory_coefficient", np.nan)
    else:
        put("burstiness", np.nan); put("memory_coefficient", np.nan)

    # legacy halt statistics: accumulated pause time at the end of each pause
    put("avg_haltT", float(pauses.mean()) if pauses.size else 0.0)
    put("std_haltT", float(pauses.std()) if pauses.size else 0.0)

    # ---------- speed distribution ----------
    sp = k.speed[np.isfinite(k.speed)]
    if sp.size:
        put("speed_max", float(np.percentile(sp, 95)))
        put("speed_cv", float(sp.std() / sp.mean()) if sp.mean() > 0 else np.nan)
        sd = sp.std()
        put("speed_skew", float(np.mean(((sp - sp.mean()) / sd) ** 3)) if sd > 0 else np.nan)
    else:
        put("speed_max", np.nan); put("speed_cv", np.nan); put("speed_skew", np.nan)
    put("accel_rms", float(np.sqrt(np.nanmean(k.accel ** 2))) if k.accel.size else np.nan)
    put("jerk_rms", float(np.sqrt(np.nanmean(k.jerk ** 2))) if k.jerk.size else np.nan)

    # ---------- oscillation ----------
    fs = k.fs
    f_beat, purity = dominant_frequency(k.ang_vel, fs) if np.isfinite(fs) else (np.nan, np.nan)
    put("beat_frequency", f_beat); put("beat_power_ratio", purity)
    f_sp, _ = dominant_frequency(k.speed, fs) if np.isfinite(fs) else (np.nan, np.nan)
    put("speed_dominant_freq", f_sp)

    amp = _lateral_amplitude(k)
    put("lateral_amplitude", amp)
    U = r["avg_velocity"]
    put("strouhal", f_beat * 2 * amp / U
        if all(np.isfinite(v) for v in (f_beat, amp, U)) and U > 0 else np.nan)

    body_len = _body_length(k)
    if np.isfinite(body_len) and np.isfinite(U):
        # metres/second from length_unit; only micrometre and millimetre make sense here
        to_m = {"um": 1e-6, "µm": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0, "nm": 1e-9}
        c = to_m.get(cfg.length_unit, 1e-6)
        put("reynolds", _RHO_WATER * (U * c) * (body_len * c) / _MU_WATER)
        put("speed_bodylengths", U / body_len if body_len > 0 else np.nan)
    else:
        put("reynolds", np.nan); put("speed_bodylengths", np.nan)

    # ---------- position ----------
    xr, yr = k.x - k.x.min(), k.y - k.y.min()
    put("avg_X", float(xr.mean())); put("avg_Y", float(yr.mean()))
    put("XYdispersion", float(math.sqrt(xr.std() * yr.std())))

    # ---------- morphology ----------
    _morphology(k, r, tf)
    return r


def _msd_fit(k: TrackKinematics, max_frac: float) -> tuple[float, float, float]:
    """Time-averaged MSD, fitted as MSD(tau) = 4 D tau^alpha on log-log axes."""
    n = k.n
    if n < 12:
        return np.nan, np.nan, np.nan
    max_lag = max(3, int(n * max_frac))
    lags = np.unique(np.round(np.logspace(0, np.log10(max_lag), 20)).astype(int))
    lags = lags[(lags >= 1) & (lags < n)]
    if lags.size < 4:
        return np.nan, np.nan, np.nan
    msd = np.empty(lags.size)
    for i, L in enumerate(lags):
        dx = k.x[L:] - k.x[:-L]
        dy = k.y[L:] - k.y[:-L]
        msd[i] = np.mean(dx * dx + dy * dy)
    dtm = k.mean_dt
    if not np.isfinite(dtm) or dtm <= 0:
        return np.nan, np.nan, np.nan
    tau = lags * dtm
    ok = np.isfinite(msd) & (msd > 0)
    if ok.sum() < 4:
        return np.nan, np.nan, np.nan
    lx, ly = np.log(tau[ok]), np.log(msd[ok])
    slope, intercept = np.polyfit(lx, ly, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(slope), float(math.exp(intercept) / 4.0), float(r2)


def _heading_corr_time(heading: np.ndarray, dt: float) -> float:
    n = heading.size
    max_lag = max(2, n // 2)
    c = np.array([np.mean(np.cos(heading[L:] - heading[: n - L])) for L in range(max_lag)])
    below = np.flatnonzero(c < 1 / math.e)
    if below.size == 0:
        return float(max_lag * dt)
    L = below[0]
    if L == 0:
        return 0.0
    a0, a1 = c[L - 1], c[L]
    frac = (a0 - 1 / math.e) / max(a0 - a1, 1e-12)
    return float((L - 1 + frac) * dt)


def _lateral_amplitude(k: TrackKinematics) -> float:
    """RMS deviation of the path from a smoothed centre line."""
    if k.n < 9:
        return np.nan
    try:
        from scipy.ndimage import uniform_filter1d

        w = max(5, min(k.n // 3, 31)) | 1
        sx = uniform_filter1d(k.x, w, mode="nearest")
        sy = uniform_filter1d(k.y, w, mode="nearest")
        return float(np.sqrt(np.mean((k.x - sx) ** 2 + (k.y - sy) ** 2)))
    except Exception:
        return np.nan


def _body_length(k: TrackKinematics) -> float:
    for key in ("major", "diameter"):
        if key in k.extras:
            v = k.extras[key]
            v = v[np.isfinite(v)]
            if v.size:
                return float(np.median(v))
    return np.nan


def _morphology(k: TrackKinematics, r: dict[str, float], tf: float) -> None:
    e = k.extras

    def stat(key: str, avg: str, std: str, scale: float = 1.0) -> None:
        if key in e:
            m, s = trimmed_stats(e[key], tf)
            r[avg] = m * scale
            r[std] = s * scale
        else:
            r[avg] = np.nan
            r[std] = np.nan

    stat("area", "avg_Area", "std_Area")
    stat("major", "avg_Major", "std_Major")
    stat("minor", "avg_Minor", "std_Minor")
    stat("intensity_mean", "avg_Intnsty", "std_Intnsty")

    if "major" in e and "minor" in e:
        with np.errstate(divide="ignore", invalid="ignore"):
            ar = e["major"] / e["minor"]
        ar = ar[np.isfinite(ar)]
        r["aspect_ratio"] = float(np.median(ar)) if ar.size else np.nan
        r["aspect_ratio_cv"] = float(ar.std() / ar.mean()) if ar.size and ar.mean() > 0 else np.nan
    else:
        r["aspect_ratio"] = r["aspect_ratio_cv"] = np.nan

    if "angle" in e:
        # body orientation is an axial quantity (mod 180 deg): double it before
        # averaging on the circle, then halve -- the macro averaged it linearly.
        ang = np.radians(e["angle"][: k.heading.size + 1])
        ang = ang[np.isfinite(ang)]
        if ang.size:
            r["avg_AngleP"] = math.degrees(circ_mean(2 * ang) / 2.0)
            r["std_AngleP"] = math.degrees(circ_sd(2 * ang) / 2.0)
        else:
            r["avg_AngleP"] = r["std_AngleP"] = np.nan

        m = min(k.heading.size, e["angle"].size)
        if m > 1:
            body = np.radians(e["angle"][:m])
            travel = k.heading[:m]
            # axial difference folded into [0, 90] deg
            d = np.degrees(np.abs(wrap_pi(2 * (body - travel)))) / 2.0
            d = d[np.isfinite(d)]
            r["avg_drift"] = float(d.mean()) if d.size else np.nan
            r["std_drift"] = float(d.std()) if d.size else np.nan
        else:
            r["avg_drift"] = r["std_drift"] = np.nan
    else:
        r["avg_AngleP"] = r["std_AngleP"] = r["avg_drift"] = r["std_drift"] = np.nan

    if "intensity_std" in e:
        v = e["intensity_std"]
        v = v[np.isfinite(v)]
        if v.size > 1 and (v.max() - v.min()) > 0:
            z = (v - v.min()) / (v.max() - v.min())
            r["avg_estZ"] = float(z.mean())
            r["std_estZ"] = float(z.std())
        else:
            r["avg_estZ"] = r["std_estZ"] = np.nan
    else:
        r["avg_estZ"] = r["std_estZ"] = np.nan


# ==========================================================================
# Public entry point
# ==========================================================================
def compute_features(
    spots: SpotDataset,
    cfg: TrackConfig,
    selected: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> FeatureDataset:
    """Turn a spot-level coordinate table into a sample-level feature table."""
    df = spots.frame
    wanted = set(selected) if selected else set(METRICS)

    thr = auto_halt_threshold(df, cfg) if cfg.auto_halt_threshold else None

    rows: list[dict] = []
    groups = list(df.groupby("uid", observed=True, sort=False))
    total = len(groups)

    for i, (uid, sub) in enumerate(groups):
        if len(sub) < cfg.min_spots:
            continue
        if cfg.drop_edge_margin > 0 and cfg.image_width and cfg.image_height:
            m = cfg.drop_edge_margin
            keep = ((sub["x"] >= m) & (sub["x"] <= cfg.image_width - m)
                    & (sub["y"] >= m) & (sub["y"] <= cfg.image_height - m))
            sub = sub.loc[keep]
            if len(sub) < cfg.min_spots:
                continue

        k = build_kinematics(sub, cfg, halt_threshold=thr)
        row = _compute_one(k, cfg, wanted)
        row["sample"] = str(uid).split("|")[-1] or str(uid)
        row["uid"] = uid
        row["group"] = sub["group"].iloc[0]
        row["replicate"] = sub["replicate"].iloc[0]
        row["source"] = sub["source"].iloc[0] if "source" in sub else ""
        rows.append(row)

        if progress and (i % 25 == 0 or i == total - 1):
            progress(i + 1, total)

    if not rows:
        raise ValueError(
            f"No track survived filtering (min_spots={cfg.min_spots}). "
            "Lower the minimum track length or check the column mapping."
        )

    out = pd.DataFrame(rows)
    feature_names = [n for n in METRICS if n in out.columns and (not selected or n in wanted)]
    # a metric whose inputs were entirely missing is all-NaN: drop it silently
    feature_names = [n for n in feature_names if out[n].notna().any()]

    meta = ["sample", "uid", "group", "replicate", "source"]
    ds = FeatureDataset(
        frame=out[meta + feature_names],
        feature_names=feature_names,
        id_column="sample",
        group_column="group",
        replicate_column="replicate",
        units={n: METRICS[n].formatted_unit(cfg.length_unit, cfg.time_unit) for n in feature_names},
        descriptions={n: METRICS[n].description for n in feature_names},
    )
    return ds


__all__ = [
    "METRICS",
    "MetricSpec",
    "metrics_by_category",
    "available_metrics",
    "compute_features",
    "build_kinematics",
    "TrackKinematics",
    "trimmed_stats",
    "auto_halt_threshold",
]
