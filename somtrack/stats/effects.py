"""Effect sizes with confidence intervals, for the per-metric layer.

A p value says whether a difference is distinguishable from nothing; it does not
say how big it is, and with a few hundred animals almost anything eventually
becomes distinguishable.  Estimation statistics (Ho et al. 2019) put the
difference itself, with a bootstrap confidence interval, in front of the reader.
That is what the figures in :mod:`somtrack.viz.estimation` draw, and this module
computes.

Hedges' g rather than Cohen's d: the small-sample correction matters at the
group sizes this program is built for, where the uncorrected d is biased upwards
by several percent (Hedges 1981).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ==========================================================================
def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference (b - a), corrected for small samples."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return np.nan
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return np.nan
    d = (b.mean() - a.mean()) / np.sqrt(sp2)
    df = na + nb - 2
    J = 1.0 - 3.0 / (4.0 * df - 1.0)             # Hedges' small-sample correction
    return float(d * J)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Non-parametric effect size: P(b > a) - P(a > b). Immune to outliers."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.nan
    gt = (b[:, None] > a[None, :]).sum()
    lt = (b[:, None] < a[None, :]).sum()
    return float((gt - lt) / (a.size * b.size))


# ==========================================================================
@dataclass
class EffectEstimate:
    feature: str
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    mean_a: float
    mean_b: float
    difference: float
    diff_ci: tuple[float, float]
    hedges_g: float
    g_ci: tuple[float, float]
    cliffs_delta: float
    unit: str = ""

    def row(self) -> dict:
        return {
            "feature": self.feature, "unit": self.unit,
            "group_a": self.group_a, "group_b": self.group_b,
            "n_a": self.n_a, "n_b": self.n_b,
            "mean_a": self.mean_a, "mean_b": self.mean_b,
            "difference": self.difference,
            "diff_ci_low": self.diff_ci[0], "diff_ci_high": self.diff_ci[1],
            "hedges_g": self.hedges_g,
            "g_ci_low": self.g_ci[0], "g_ci_high": self.g_ci[1],
            "cliffs_delta": self.cliffs_delta,
        }


def bootstrap_ci(a: np.ndarray, b: np.ndarray, statistic, n_boot: int = 5000,
                 alpha: float = 0.05, random_state: int | None = 0,
                 bca: bool = True) -> tuple[float, float]:
    """Bias-corrected and accelerated bootstrap interval (Efron 1987).

    BCa rather than the percentile interval because the effect-size statistics
    here are skewed at small n, which is exactly when the percentile interval
    sits in the wrong place.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return (np.nan, np.nan)

    rng = np.random.default_rng(random_state)
    theta = statistic(a, b)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        boot[i] = statistic(rng.choice(a, a.size, replace=True),
                            rng.choice(b, b.size, replace=True))
    boot = boot[np.isfinite(boot)]
    if boot.size < 20 or not np.isfinite(theta):
        return (np.nan, np.nan)

    if not bca:
        return (float(np.quantile(boot, alpha / 2)),
                float(np.quantile(boot, 1 - alpha / 2)))

    from scipy.stats import norm

    prop = float((boot < theta).mean())
    prop = min(max(prop, 1.0 / boot.size), 1.0 - 1.0 / boot.size)
    z0 = norm.ppf(prop)

    # jackknife acceleration over the pooled samples
    jack = []
    for i in range(a.size):
        jack.append(statistic(np.delete(a, i), b))
    for j in range(b.size):
        jack.append(statistic(a, np.delete(b, j)))
    jack = np.asarray(jack, float)
    jack = jack[np.isfinite(jack)]
    if jack.size < 3:
        return (float(np.quantile(boot, alpha / 2)),
                float(np.quantile(boot, 1 - alpha / 2)))
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    acc = float(num / den) if den > 0 else 0.0

    def adjust(q):
        z = norm.ppf(q)
        return float(norm.cdf(z0 + (z0 + z) / max(1e-12, 1 - acc * (z0 + z))))

    lo, hi = adjust(alpha / 2), adjust(1 - alpha / 2)
    lo = min(max(lo, 0.001), 0.999)
    hi = min(max(hi, 0.001), 0.999)
    return (float(np.quantile(boot, lo)), float(np.quantile(boot, hi)))


# ==========================================================================
def _mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(b) - np.mean(a))


def estimate_effects(
    values: dict[str, np.ndarray],
    group_codes: np.ndarray,
    group_values: list,
    *,
    reference: int = 0,
    units: dict[str, str] | None = None,
    n_boot: int = 5000,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Effect size of every group against a reference, for every metric.

    ``values`` maps metric name -> per-sample values in original units, so the
    difference is reported in the units a reader understands rather than in
    standard deviations of a scaled matrix.
    """
    units = units or {}
    rows: list[dict] = []
    for name, v in values.items():
        v = np.asarray(v, float)
        ref = v[group_codes == reference]
        for g, label in enumerate(group_values):
            if g == reference:
                continue
            other = v[group_codes == g]
            if ref.size < 2 or other.size < 2:
                continue
            est = EffectEstimate(
                feature=name, unit=units.get(name, ""),
                group_a=str(group_values[reference]), group_b=str(label),
                n_a=int(ref.size), n_b=int(other.size),
                mean_a=float(np.nanmean(ref)), mean_b=float(np.nanmean(other)),
                difference=_mean_difference(ref, other),
                diff_ci=bootstrap_ci(ref, other, _mean_difference, n_boot,
                                     random_state=random_state),
                hedges_g=hedges_g(ref, other),
                g_ci=bootstrap_ci(ref, other, hedges_g, n_boot,
                                  random_state=random_state),
                cliffs_delta=cliffs_delta(ref, other),
            )
            rows.append(est.row())
    df = pd.DataFrame(rows)
    if not df.empty:
        df["abs_g"] = df["hedges_g"].abs()
        df = df.sort_values("abs_g", ascending=False).drop(columns="abs_g")
        df = df.reset_index(drop=True)
    return df


__all__ = [
    "hedges_g", "cliffs_delta", "bootstrap_ci", "EffectEstimate",
    "estimate_effects",
]
