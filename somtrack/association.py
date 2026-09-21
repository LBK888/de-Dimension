"""Which metrics actually explain the grouping?

This module exists because clustering on its own only says *that* samples
differ.  Everything here answers *why*, in terms a reviewer will accept:

``univariate_association``   per-metric ANOVA / Kruskal-Wallis with effect
                             sizes and FDR-corrected q values.
``multivariate_importance``  cross-validated random-forest permutation
                             importance -- catches metrics that only matter in
                             combination, which univariate tests miss.
``map_gradients``            direction and steepness of every metric across the
                             SOM.  Replaces the macro's axis-vector analysis
                             with a fitted plane, and keeps the original
                             first-moment formulation available for
                             back-compatibility.
``node_cluster_profile``     what distinguishes each node cluster, in original
                             measurement units.
``group_enrichment``         which map regions a treatment occupies more than
                             chance would predict.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .nodecluster import NodeClustering
from .preprocess import PreparedData
from .som import SomResult


# ==========================================================================
# 1. Univariate
# ==========================================================================
def univariate_association(prep: PreparedData, raw: bool = True) -> pd.DataFrame:
    """One row per metric: group means, ANOVA, Kruskal-Wallis, effect sizes, FDR q."""
    from scipy import stats

    src = prep.source
    codes = prep.group_codes
    labels = prep.group_values
    k = prep.n_groups

    rows: list[dict] = []
    for j, name in enumerate(prep.feature_names):
        if raw and src is not None and name in src.frame.columns:
            v = src.frame.loc[prep.keep_mask, name].to_numpy(float)
        else:
            v = prep.X[:, j]

        samples = [v[codes == g] for g in range(k)]
        samples = [s[np.isfinite(s)] for s in samples]
        row: dict = {"feature": name,
                     "unit": (src.units.get(name, "") if src else "")}
        for gi, lab in enumerate(labels):
            row[f"mean[{lab}]"] = float(np.mean(samples[gi])) if samples[gi].size else np.nan
            row[f"sd[{lab}]"] = float(np.std(samples[gi], ddof=1)) if samples[gi].size > 1 else np.nan
            row[f"n[{lab}]"] = int(samples[gi].size)

        usable = [s for s in samples if s.size >= 2]
        if len(usable) >= 2:
            try:
                F, p = stats.f_oneway(*usable)
            except Exception:
                F, p = np.nan, np.nan
            try:
                H, pk = stats.kruskal(*usable)
            except Exception:
                H, pk = np.nan, np.nan
        else:
            F = p = H = pk = np.nan

        row["anova_F"] = float(F) if np.isfinite(F) else np.nan
        row["anova_p"] = float(p) if np.isfinite(p) else np.nan
        row["kruskal_H"] = float(H) if np.isfinite(H) else np.nan
        row["kruskal_p"] = float(pk) if np.isfinite(pk) else np.nan
        row["eta2"] = _eta_squared(samples)
        n_tot = sum(s.size for s in usable)
        row["epsilon2"] = (float((H - len(usable) + 1) / (n_tot - len(usable)))
                           if np.isfinite(H) and n_tot > len(usable) else np.nan)
        row["max_abs_cohen_d"] = _max_cohen_d(samples)
        rows.append(row)

    df = pd.DataFrame(rows)
    for col, out in (("anova_p", "anova_q"), ("kruskal_p", "kruskal_q")):
        df[out] = _bh_fdr(df[col].to_numpy(float))
    df["rank"] = df["eta2"].rank(ascending=False, method="min").astype("Int64")
    return df.sort_values("eta2", ascending=False).reset_index(drop=True)


def _eta_squared(samples: list[np.ndarray]) -> float:
    samples = [s for s in samples if s.size]
    if len(samples) < 2:
        return np.nan
    allv = np.concatenate(samples)
    grand = allv.mean()
    ss_b = sum(s.size * (s.mean() - grand) ** 2 for s in samples)
    ss_t = float(((allv - grand) ** 2).sum())
    return float(ss_b / ss_t) if ss_t > 0 else np.nan


def _max_cohen_d(samples: list[np.ndarray]) -> float:
    best = np.nan
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            a, b = samples[i], samples[j]
            if a.size < 2 or b.size < 2:
                continue
            sp = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1))
                         / (a.size + b.size - 2))
            if sp <= 0:
                continue
            d = abs(a.mean() - b.mean()) / sp
            best = d if not np.isfinite(best) else max(best, d)
    return float(best) if np.isfinite(best) else np.nan


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up."""
    q = np.full_like(p, np.nan, dtype=float)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    n = pv.size
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


# ==========================================================================
# 2. Multivariate importance
# ==========================================================================
@dataclass
class ImportanceResult:
    table: pd.DataFrame                    # feature, importance, sd, rank
    cv_accuracy: float = np.nan
    cv_accuracy_sd: float = np.nan
    baseline_accuracy: float = np.nan      # majority-class rate
    n_splits: int = 0
    model: str = "RandomForest"


def multivariate_importance(prep: PreparedData, n_splits: int = 5,
                            n_repeats: int = 5, n_estimators: int = 250,
                            random_state: int = 0) -> ImportanceResult:
    """Cross-validated permutation importance for predicting the group."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold

    X, y = prep.X, prep.group_codes
    names = prep.feature_names
    counts = np.bincount(y)
    baseline = float(counts.max() / counts.sum())

    if prep.n_groups < 2 or counts.min() < 2:
        return ImportanceResult(
            table=pd.DataFrame({"feature": names, "importance": np.nan, "sd": np.nan}),
            baseline_accuracy=baseline,
        )

    splits = int(min(n_splits, counts.min()))
    cv = StratifiedKFold(n_splits=max(2, splits), shuffle=True, random_state=random_state)

    imp, accs = [], []
    with warnings.catch_warnings():
        # scikit-learn 1.9 emits an internal joblib/delayed notice from
        # permutation_importance that users cannot act on
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        for tr, te in cv.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=n_estimators, random_state=random_state, n_jobs=-1,
                class_weight="balanced_subsample",
            ).fit(X[tr], y[tr])
            accs.append(float(clf.score(X[te], y[te])))
            r = permutation_importance(clf, X[te], y[te], n_repeats=n_repeats,
                                       random_state=random_state, n_jobs=-1)
            imp.append(r.importances_mean)

    imp = np.asarray(imp)
    table = pd.DataFrame({
        "feature": names,
        "importance": imp.mean(axis=0),
        "sd": imp.std(axis=0),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1)

    return ImportanceResult(
        table=table,
        cv_accuracy=float(np.mean(accs)),
        cv_accuracy_sd=float(np.std(accs)),
        baseline_accuracy=baseline,
        n_splits=len(accs),
    )


# ==========================================================================
# 3. Metric gradients on the map (replaces the macro's axis vectors)
# ==========================================================================
@dataclass
class MapGradients:
    features: list[str]
    direction: np.ndarray            # (n_features, 2) unit vector of steepest increase
    magnitude: np.ndarray            # (n_features,) normalised gradient strength 0-1
    r2: np.ndarray                   # (n_features,) plane-fit R^2
    method: str = "gradient"
    origin: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def table(self) -> pd.DataFrame:
        ang = np.degrees(np.arctan2(self.direction[:, 1], self.direction[:, 0]))
        return pd.DataFrame({
            "feature": self.features,
            "angle_deg": ang,
            "magnitude": self.magnitude,
            "plane_r2": self.r2,
        }).sort_values("magnitude", ascending=False).reset_index(drop=True)


def map_gradients(res: SomResult, method: str = "gradient",
                  centre_node: int | None = None,
                  weight_by_hits: bool = True) -> MapGradients:
    """Direction in which each metric increases across the map.

    ``gradient`` (default) fits ``value ~ a*x + b*y + c`` over the node
    positions.  The coefficient vector is the direction of steepest increase and
    the fit R^2 says how much of the component plane is actually a smooth
    gradient -- a metric whose plane is patchy gets a low R^2 and its arrow can
    be hidden, instead of being drawn with false confidence.

    ``legacy_moment`` reproduces ``CalAxisVector`` from the macro: the
    range-normalised first moment of the component plane about a reference node.
    """
    xy = res.lattice.xy
    W = res.codebook
    n_f = W.shape[1]
    w = res.hits.copy() if weight_by_hits else np.ones(res.n_nodes)
    if w.sum() <= 0:
        w = np.ones(res.n_nodes)
    w = w + 0.25 * w[w > 0].mean() if (w > 0).any() else w   # keep empty nodes informative

    centre = xy.mean(axis=0)
    if method == "legacy_moment":
        idx = centre_node if centre_node is not None else _centre_index(res)
        return _legacy_moment(res, idx)

    A = np.column_stack([xy[:, 0] - centre[0], xy[:, 1] - centre[1], np.ones(res.n_nodes)])
    sw = np.sqrt(w)[:, None]
    direction = np.zeros((n_f, 2))
    magnitude = np.zeros(n_f)
    r2 = np.zeros(n_f)

    for j in range(n_f):
        v = W[:, j]
        rng = v.max() - v.min()
        if rng <= 1e-12:
            continue
        vn = (v - v.min()) / rng
        coef, *_ = np.linalg.lstsq(A * sw, (vn * sw[:, 0]), rcond=None)
        g = coef[:2]
        pred = A @ coef
        ss_res = float(np.sum(w * (vn - pred) ** 2))
        ss_tot = float(np.sum(w * (vn - np.average(vn, weights=w)) ** 2))
        r2[j] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        norm = float(np.hypot(*g))
        magnitude[j] = norm
        direction[j] = g / norm if norm > 1e-12 else np.zeros(2)

    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max()

    return MapGradients(list(res.feature_names), direction, magnitude,
                        np.clip(r2, 0, 1), "gradient", centre)


def _centre_index(res: SomResult) -> int:
    c = res.lattice.xy.mean(axis=0)
    return int(np.argmin(((res.lattice.xy - c) ** 2).sum(axis=1)))


def _legacy_moment(res: SomResult, centre_idx: int) -> MapGradients:
    xy = res.lattice.xy
    W = res.codebook
    c = xy[centre_idx]
    ref = W[centre_idx]
    lo, hi = W.min(axis=0), W.max(axis=0)

    vec = np.zeros((W.shape[1], 2))
    for j in range(W.shape[1]):
        plus = hi[j] - ref[j]
        minus = ref[j] - lo[j]
        diff = W[:, j] - ref[j]
        scale = np.where(diff > 0,
                         diff / plus if plus > 0 else 0.0,
                         diff / minus if minus > 0 else 0.0)
        vec[j] = ((xy - c) * scale[:, None]).sum(axis=0)

    mag = np.hypot(vec[:, 0], vec[:, 1])
    direction = np.zeros_like(vec)
    nz = mag > 1e-12
    direction[nz] = vec[nz] / mag[nz, None]
    mag_n = mag / mag.max() if mag.max() > 0 else mag
    return MapGradients(list(res.feature_names), direction, mag_n,
                        np.full(W.shape[1], np.nan), "legacy_moment", c)


# ==========================================================================
# 4. Node-cluster profiles
# ==========================================================================
def node_cluster_profile(res: SomResult, clustering: NodeClustering,
                         prep: PreparedData, k: int | None = None) -> pd.DataFrame:
    """Mean metric value per node cluster, in original units, plus separation F."""
    from scipy import stats

    labels = clustering.labels_by_k.get(k or clustering.best_k)
    if labels is None:
        return pd.DataFrame()

    W_units = prep.inverse(res.codebook)
    hits = res.hits
    rows = []
    uniq = sorted(set(labels[hits > 0].tolist())) or sorted(set(labels.tolist()))

    for j, name in enumerate(prep.feature_names):
        row: dict = {"feature": name, "unit": (prep.source.units.get(name, "") if prep.source else "")}
        groups = []
        for c in uniq:
            m = (labels == c) & (hits > 0)
            if not m.any():
                m = labels == c
            vals = W_units[m, j]
            w = hits[m]
            w = w if w.sum() > 0 else np.ones_like(w)
            row[f"cluster_{c}"] = float(np.average(vals, weights=w))
            groups.append(vals)
        usable = [g for g in groups if g.size >= 2]
        if len(usable) >= 2:
            try:
                F, p = stats.f_oneway(*usable)
                row["F"], row["p"] = float(F), float(p)
            except Exception:
                row["F"] = row["p"] = np.nan
        else:
            row["F"] = row["p"] = np.nan
        rows.append(row)

    df = pd.DataFrame(rows)
    if "p" in df:
        df["q"] = _bh_fdr(df["p"].to_numpy(float))
    return df.sort_values("F", ascending=False).reset_index(drop=True)


# ==========================================================================
# 5. Group enrichment across the map
# ==========================================================================
def group_enrichment(res: SomResult, prep: PreparedData) -> pd.DataFrame:
    """Observed vs expected hits per (node, group), with a hypergeometric p."""
    from scipy.stats import hypergeom

    codes = prep.group_codes
    k = prep.n_groups
    M = res.hit_matrix(codes, k)
    total = M.sum()
    if total <= 0:
        return pd.DataFrame()

    node_tot = M.sum(axis=1)
    grp_tot = M.sum(axis=0)

    rows = []
    for node in range(res.n_nodes):
        if node_tot[node] == 0:
            continue
        for g in range(k):
            obs = M[node, g]
            exp = node_tot[node] * grp_tot[g] / total
            p = float(hypergeom.sf(obs - 1, int(total), int(grp_tot[g]), int(node_tot[node])))
            rows.append({
                "node": node,
                "col": int(res.lattice.coords[node, 0]),
                "row": int(res.lattice.coords[node, 1]),
                "group": prep.group_values[g],
                "observed": float(obs),
                "expected": float(exp),
                "log2_enrichment": float(np.log2((obs + 0.5) / (exp + 0.5))),
                "p": p,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["q"] = _bh_fdr(df["p"].to_numpy(float))
    return df


# ==========================================================================
# 6. Convenience: everything at once
# ==========================================================================
@dataclass
class AssociationBundle:
    univariate: pd.DataFrame
    importance: ImportanceResult
    gradients: MapGradients | None = None
    node_profile: pd.DataFrame | None = None
    enrichment: pd.DataFrame | None = None

    def top_features(self, n: int = 10) -> list[str]:
        """Union of the strongest univariate and multivariate metrics."""
        a = self.univariate.head(n)["feature"].tolist()
        b = self.importance.table.head(n)["feature"].tolist()
        seen, out = set(), []
        for f in a + b:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out[: n + n // 2]


def analyse_associations(prep: PreparedData, res: SomResult | None = None,
                         clustering: NodeClustering | None = None,
                         gradient_method: str = "gradient") -> AssociationBundle:
    bundle = AssociationBundle(
        univariate=univariate_association(prep),
        importance=multivariate_importance(prep),
    )
    if res is not None:
        bundle.gradients = map_gradients(res, method=gradient_method)
        bundle.enrichment = group_enrichment(res, prep)
        if clustering is not None and clustering.best_k:
            bundle.node_profile = node_cluster_profile(res, clustering, prep)
    return bundle


__all__ = [
    "univariate_association", "multivariate_importance", "ImportanceResult",
    "map_gradients", "MapGradients", "node_cluster_profile", "group_enrichment",
    "AssociationBundle", "analyse_associations",
]
