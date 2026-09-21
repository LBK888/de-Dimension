"""Can the groups be told apart?  Cross-validation with an honest null.

A projection that separates the groups on screen is not evidence.  The question
"can these groups be told apart" has one operational answer: hold data out, try
to predict its group, and compare the result with what the same procedure
achieves when the labels are meaningless.

Three details make the difference between a number and a publishable number:

*   **Balanced accuracy, not accuracy** (Brodersen et al. 2010).  With unequal
    group sizes, predicting the largest group every time already scores well.
*   **A permutation null, not a binomial test** (Ojala & Garriga 2010).
    Cross-validation folds are not independent trials, so the binomial test
    applied to a cross-validated accuracy is anti-conservative -- it reports
    significance that is not there (Noirhomme et al. 2014).
*   **Folds that respect the replicate structure** (see :mod:`somtrack.stats.blocks`).
    If two animals from one dish land on opposite sides of a fold boundary, a
    good score may only mean the classifier recognised the dish.

The confusion matrix is reported alongside the single number, because "group A
separates from everything, B and C do not separate from each other" is the
biologically useful statement and a scalar accuracy cannot express it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .blocks import (BlockStructure, CvPlan, effective_permutations, make_cv,
                     permutation_p, permutations)

MODELS: dict[str, dict] = {
    "lda_shrinkage": {
        "label": "regularised linear discriminant analysis",
        "detail": ("linear discriminant analysis with Ledoit-Wolf shrinkage of the "
                   "covariance estimate, which is what keeps it stable when the "
                   "number of metrics approaches the number of samples"),
        "citations": ("fisher1936", "ledoit2004"),
        "linear": True,
    },
    "svm_linear": {
        "label": "linear support vector machine",
        "detail": "a linear support vector machine with balanced class weights",
        "citations": ("cortes1995",),
        "linear": True,
    },
    "plsda": {
        "label": "PLS-DA",
        "detail": "partial least squares discriminant analysis",
        "citations": ("barker2003", "westerhuis2008"),
        "linear": True,
    },
    "logistic": {
        "label": "regularised logistic regression",
        "detail": "multinomial logistic regression with an L2 penalty",
        "citations": (),
        "linear": True,
    },
    "random_forest": {
        "label": "random forest",
        "detail": "a random forest with balanced class weights",
        "citations": ("breiman2001",),
        "linear": False,
    },
}


# ==========================================================================
class PLSDA:
    """PLS regression onto one-hot group indicators, with an argmax decision.

    Written out rather than pulled from a chemometrics package so that the
    cross-validation, the permutation null and the Haufe transform all see the
    same object as every other model here.
    """

    def __init__(self, n_components: int = 2, scale: bool = False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        from sklearn.cross_decomposition import PLSRegression

        self.classes_ = np.unique(y)
        Y = np.zeros((len(y), len(self.classes_)))
        for i, c in enumerate(self.classes_):
            Y[y == c, i] = 1.0
        k = int(np.clip(self.n_components, 1,
                        min(X.shape[1], max(1, X.shape[0] - 1))))
        self.model_ = PLSRegression(n_components=k, scale=self.scale).fit(X, Y)
        self.coef_ = np.asarray(self.model_.x_rotations_).T[:2]  # for Haufe
        return self

    def decision_function(self, X):
        return np.asarray(self.model_.predict(X))

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def transform(self, X):
        return np.asarray(self.model_.transform(X))

    def get_params(self, deep=True):
        return {"n_components": self.n_components, "scale": self.scale}

    def set_params(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        return self

    # VIP: how much each metric contributes across all components
    def vip(self) -> np.ndarray:
        t = self.model_.x_scores_
        w = self.model_.x_weights_
        q = self.model_.y_loadings_
        p, h = w.shape
        ss = np.array([(t[:, i] ** 2).sum() * (q[:, i] ** 2).sum() for i in range(h)])
        total = ss.sum()
        if total <= 0:
            return np.zeros(p)
        wn = w / np.maximum(np.linalg.norm(w, axis=0), 1e-12)
        return np.sqrt(p * ((wn ** 2) * ss[None, :]).sum(axis=1) / total)


def make_estimator(name: str, random_state: int | None = 0, **kw):
    if name == "lda_shrinkage":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    if name == "svm_linear":
        from sklearn.svm import SVC

        return SVC(kernel="linear", C=float(kw.get("C", 1.0)),
                   class_weight="balanced", random_state=random_state)
    if name == "plsda":
        return PLSDA(n_components=int(kw.get("n_components", 2)))
    if name == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=2000, class_weight="balanced",
                                  random_state=random_state)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=int(kw.get("n_estimators", 300)),
            class_weight="balanced_subsample", n_jobs=-1,
            random_state=random_state)
    raise KeyError(f"Unknown model '{name}'. Known: {', '.join(MODELS)}")


# ==========================================================================
@dataclass
class ClassifyResult:
    model: str
    model_label: str
    group_values: list

    balanced_accuracy: float = np.nan
    ba_ci: tuple[float, float] = (np.nan, np.nan)
    accuracy: float = np.nan
    kappa: float = np.nan
    chance: float = np.nan                    # 1 / n_groups for balanced accuracy
    round_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))

    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    per_class_recall: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pairwise: pd.DataFrame = field(default_factory=pd.DataFrame)

    permutation_p: float = np.nan
    permutation_null: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_permutations: int = 0
    p_resolution: float = np.nan              # smallest p the design can support

    oof_pred: np.ndarray = field(default_factory=lambda: np.zeros(0))
    weights: np.ndarray | None = None         # (n_components, n_features)
    activation: np.ndarray | None = None      # Haufe-transformed pattern
    vip: np.ndarray | None = None
    feature_names: list = field(default_factory=list)

    cv_name: str = ""
    cv_note: str = ""
    blocked_by: str = ""
    citations: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def separable(self) -> bool:
        return (np.isfinite(self.permutation_p) and self.permutation_p < 0.05
                and self.balanced_accuracy > self.chance)

    def summary_line(self) -> str:
        if not np.isfinite(self.balanced_accuracy):
            return "Classification could not be run."
        lo, hi = self.ba_ci
        ci = f" [95% CI {lo:.2f}-{hi:.2f}]" if np.isfinite(lo) else ""
        p = (f"permutation p = {self.permutation_p:.3f}"
             if np.isfinite(self.permutation_p) else "no permutation test")
        return (f"balanced accuracy {self.balanced_accuracy:.2f}{ci} "
                f"(chance {self.chance:.2f}; {p}, {self.n_permutations} permutations)")

    def confusion_frame(self) -> pd.DataFrame:
        if self.confusion.size == 0:
            return pd.DataFrame()
        labels = [str(g) for g in self.group_values]
        return pd.DataFrame(self.confusion, index=[f"true {g}" for g in labels],
                            columns=[f"predicted {g}" for g in labels])

    def table(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "model": self.model_label,
            "cross_validation": self.cv_name,
            "balanced_accuracy": self.balanced_accuracy,
            "ba_ci_low": self.ba_ci[0], "ba_ci_high": self.ba_ci[1],
            "accuracy": self.accuracy,
            "cohens_kappa": self.kappa,
            "chance_level": self.chance,
            "permutation_p": self.permutation_p,
            "n_permutations": self.n_permutations,
            "smallest_possible_p": self.p_resolution,
            "blocked_by": self.blocked_by,
        }])


# ==========================================================================
def _balanced_accuracy(y: np.ndarray, pred: np.ndarray, k: int) -> float:
    recalls = []
    for c in range(k):
        m = y == c
        if m.any():
            recalls.append(float((pred[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else np.nan


def _oof_rounds(estimator, X, y, plan: CvPlan) -> list[np.ndarray]:
    """Out-of-fold predictions, one complete pass over the data per 'round'."""
    from sklearn.base import clone

    rounds: list[np.ndarray] = []
    filled: list[np.ndarray] = []
    for tr, te in plan.splitter.split(X, y, plan.groups):
        slot = None
        for i, done in enumerate(filled):
            if not done[te].any():
                slot = i
                break
        if slot is None:
            rounds.append(np.full(len(y), -1, dtype=int))
            filled.append(np.zeros(len(y), dtype=bool))
            slot = len(rounds) - 1
        model = clone(estimator) if hasattr(estimator, "get_params") else estimator
        model.fit(X[tr], y[tr])
        rounds[slot][te] = model.predict(X[te])
        filled[slot][te] = True
    return [r for r, f in zip(rounds, filled) if f.all()] or rounds


def _score_once(estimator, X, y, plan: CvPlan, k: int) -> float:
    preds = _oof_rounds(estimator, X, y, plan)
    if not preds:
        return np.nan
    return float(np.mean([_balanced_accuracy(y, p, k) for p in preds]))


# ==========================================================================
def classify(
    X: np.ndarray,
    y: np.ndarray,
    group_values: list,
    structure: BlockStructure,
    *,
    model: str = "lda_shrinkage",
    scheme: str = "auto",
    n_splits: int = 5,
    repeats: int = 5,
    n_permutations: int = 999,
    bootstrap: int = 2000,
    feature_names: list | None = None,
    random_state: int | None = 0,
    n_jobs: int = -1,
    progress=None,
) -> ClassifyResult:
    """Cross-validated classification with a design-aware permutation null."""
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    spec = MODELS.get(model, {})
    k = len(group_values)
    res = ClassifyResult(model=model, model_label=spec.get("label", model),
                         group_values=list(group_values),
                         feature_names=list(feature_names or []),
                         citations=tuple(spec.get("citations", ())),
                         chance=1.0 / k if k else np.nan,
                         blocked_by=("replicate" if structure.blocked else ""))

    counts = np.bincount(y, minlength=k)
    if k < 2 or counts[counts > 0].min() < 2:
        res.notes.append("Too few samples per group to cross-validate.")
        return res

    plan = make_cv(y, structure, scheme, n_splits, repeats, random_state)
    res.cv_name, res.cv_note = plan.name, plan.note

    est = make_estimator(model, random_state=random_state)
    rounds = _oof_rounds(est, X, y, plan)
    if not rounds:
        res.notes.append("Cross-validation produced no complete out-of-fold pass.")
        return res

    scores = np.array([_balanced_accuracy(y, p, k) for p in rounds])
    res.round_scores = scores
    res.balanced_accuracy = float(np.nanmean(scores))

    pooled = _modal_prediction(rounds, k)
    res.oof_pred = pooled
    res.accuracy = float((pooled == y).mean())
    res.kappa = float(cohen_kappa_score(y, pooled))
    res.confusion = confusion_matrix(y, pooled, labels=np.arange(k))
    with np.errstate(invalid="ignore"):
        res.per_class_recall = np.diag(res.confusion) / np.maximum(
            res.confusion.sum(axis=1), 1)

    res.ba_ci = _bootstrap_ci(y, pooled, k, bootstrap, random_state)
    res.pairwise = _pairwise_table(X, y, group_values, structure, model,
                                   scheme, n_splits, repeats, random_state)

    # ---- permutation null ------------------------------------------------
    if n_permutations > 0:
        flat = make_cv(y, structure, "stratified" if not plan.uses_blocks else "grouped",
                       n_splits, 1, random_state)
        obs = _score_once(make_estimator(model, random_state=random_state),
                          X, y, flat, k)
        null = _permutation_null(X, y, structure, model, flat, k,
                                 n_permutations, random_state, n_jobs, progress)
        res.permutation_null = null
        res.permutation_p = permutation_p(obs, null, greater_is_extreme=True)
        res.n_permutations = int(null.size)
        eff = effective_permutations(y, structure)
        res.p_resolution = float(1.0 / (min(eff, null.size) + 1))
        if eff <= null.size:
            res.notes.append(
                f"The design allows only {eff} distinct label arrangements, so no "
                f"p value below {res.p_resolution:.3g} is attainable.")

    # ---- interpretable weights -------------------------------------------
    if spec.get("linear"):
        from .interpret import haufe_patterns

        fitted = make_estimator(model, random_state=random_state).fit(X, y)
        W = _weights_of(fitted)
        if W is not None and W.size:
            res.weights = W
            res.activation = haufe_patterns(X, W, y)
        if model == "plsda" and hasattr(fitted, "vip"):
            res.vip = fitted.vip()

    return res


def _modal_prediction(rounds: list[np.ndarray], k: int) -> np.ndarray:
    stack = np.stack(rounds)
    out = np.empty(stack.shape[1], dtype=int)
    for i in range(stack.shape[1]):
        col = stack[:, i]
        col = col[col >= 0]
        out[i] = np.bincount(col, minlength=k).argmax() if col.size else -1
    return out


def _bootstrap_ci(y, pred, k, n_boot, random_state) -> tuple[float, float]:
    if n_boot <= 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = _balanced_accuracy(y[idx], pred[idx], k)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (np.nan, np.nan)
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def _permutation_null(X, y, structure, model, plan, k, n_perm,
                      random_state, n_jobs, progress) -> np.ndarray:
    from joblib import Parallel, delayed

    perms = list(permutations(y, structure, n_perm, random_state))

    def one(yp):
        try:
            return _score_once(make_estimator(model, random_state=random_state),
                               X, yp, plan, k)
        except Exception:
            return np.nan

    if progress:
        progress(0, len(perms))
    out = Parallel(n_jobs=n_jobs, prefer="processes")(delayed(one)(p) for p in perms)
    if progress:
        progress(len(perms), len(perms))
    out = np.asarray(out, dtype=float)
    return out[np.isfinite(out)]


def _pairwise_table(X, y, group_values, structure, model, scheme,
                    n_splits, repeats, random_state) -> pd.DataFrame:
    """Which specific pairs of groups separate -- the biologically useful answer."""
    k = len(group_values)
    if k < 3:
        return pd.DataFrame()

    rows = []
    for a in range(k):
        for b in range(a + 1, k):
            m = (y == a) | (y == b)
            if m.sum() < 6:
                continue
            ysub = (y[m] == b).astype(int)
            sub = BlockStructure(
                structure.design,
                structure.blocks[m] if structure.blocked else None,
                structure.block_values,
                len(np.unique(structure.blocks[m])) if structure.blocked else 0,
                int(m.sum()), structure.note,
            )
            plan = make_cv(ysub, sub, scheme, n_splits, repeats, random_state)
            try:
                score = _score_once(make_estimator(model, random_state=random_state),
                                    X[m], ysub, plan, 2)
            except Exception:
                score = np.nan
            rows.append({
                "group_a": group_values[a], "group_b": group_values[b],
                "n_a": int((y == a).sum()), "n_b": int((y == b).sum()),
                "balanced_accuracy": score,
            })
    return pd.DataFrame(rows).sort_values("balanced_accuracy",
                                          ascending=False).reset_index(drop=True)


def _weights_of(fitted) -> np.ndarray | None:
    W = getattr(fitted, "coef_", None)
    if W is None:
        W = getattr(fitted, "scalings_", None)
        if W is not None:
            W = np.asarray(W).T
    if W is None:
        return None
    W = np.atleast_2d(np.asarray(W, dtype=float))
    return W


__all__ = ["ClassifyResult", "classify", "make_estimator", "MODELS", "PLSDA"]
