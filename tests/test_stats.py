"""The statistics layer, tested against data whose answer is known in advance.

These are the tests that matter most, because the statistics layer is the part
of SOMTrack a reader will quote.  Each case is built so the right answer is not
a matter of opinion:

* data with no group difference must come back not significant;
* a shift in the means must be found, with PERMDISP *not* significant;
* a difference in spread alone must be found by PERMDISP and **not** by
  PERMANOVA, because confusing the two is the failure mode the whole module
  exists to prevent;
* a pure batch effect, with the labels nested inside the replicates, must come
  back not significant once the replicate structure is respected -- and the test
  pins the fact that ignoring it produces a spurious perfect separation.
"""

from __future__ import annotations

import numpy as np
import pytest

from somtrack.stats import (analyse_blocks, analyse_separation, build_verdict,
                            classify, estimate_effects, hedges_g)
from somtrack.stats.blocks import (effective_permutations, permutation_p,
                                   permutations)
from somtrack.stats.interpret import haufe_patterns, permutation_importance

PERM = 199
SEED = 0


def _base(seed=SEED, d=15, per_group=25, k=3):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(k), per_group)
    X = rng.normal(size=(k * per_group, d))
    return rng, X, y, [f"g{i}" for i in range(k)]


def _run(X, y, labels, reps=None, **kw):
    st = analyse_blocks(y, reps)
    return analyse_separation(X, y, labels, st, n_permutations=PERM,
                              random_state=SEED, **kw)


# ==========================================================================
# Omnibus tests
# ==========================================================================
def test_no_difference_is_not_significant():
    _, X, y, labels = _base()
    sep = _run(X, y, labels)
    assert sep.permanova_p > 0.05
    assert not sep.groups_differ
    assert "did not differ" in sep.interpretation


def test_location_shift_is_found_without_a_dispersion_flag():
    _, X, y, labels = _base()
    X[y == 1, :3] += 1.5
    X[y == 2, :3] -= 1.5
    sep = _run(X, y, labels)

    assert sep.permanova_p < 0.05
    assert sep.permanova_R2 > 0.05
    assert sep.permdisp_p > 0.05, "a pure shift must not look like a spread change"
    assert "genuine shift" in sep.interpretation


def test_dispersion_difference_is_separated_from_a_shift():
    """The distinction the module exists for: 'more variable' is not 'different'.

    PERMANOVA is *known* to reject for a dispersion difference alone (Anderson
    2006), so the test does not pretend it stays quiet.  What matters is that
    PERMDISP fires far harder, and that the interpretation tells the reader the
    result may be a change in variability rather than a shift -- which is the
    sentence that stops the wrong conclusion being written down.
    """
    _, X, y, labels = _base()
    X[y == 2] *= 3.0                      # same centre, three times the spread
    sep = _run(X, y, labels)

    assert sep.permdisp_p < 0.01, "PERMDISP must catch a spread difference"
    assert sep.permdisp_F > 10 * max(sep.permanova_F, 1e-9), \
        "the dispersion effect must dominate the location effect"
    assert sep.energy_p < 0.05, "the distributions do differ"

    if sep.groups_differ:
        assert sep.dispersion_differs
        assert "variab" in sep.interpretation, \
            "a dispersion-driven result must be labelled as one"


def test_pairwise_table_is_fdr_corrected_and_ordered():
    _, X, y, labels = _base()
    X[y == 2, :3] += 2.5                  # only group 2 moves
    sep = _run(X, y, labels)
    df = sep.pairwise
    assert set(df.columns) >= {"group_a", "group_b", "R2", "p", "q",
                               "mahalanobis_D"}
    assert (df["q"] >= df["p"] - 1e-12).all(), "q must never be below p"
    assert df["R2"].is_monotonic_decreasing

    pair = df[(df.group_a == "g0") & (df.group_b == "g1")].iloc[0]
    moved = df[df.group_b == "g2"]
    assert pair["R2"] < moved["R2"].min()


@pytest.mark.parametrize("metric", ["euclidean", "correlation", "cityblock"])
def test_every_distance_runs_and_agrees_on_a_strong_effect(metric):
    _, X, y, labels = _base()
    X[y == 1, :4] += 2.0
    X[y == 2, :4] -= 2.0
    sep = _run(X, y, labels, metric=metric)
    assert np.isfinite(sep.permanova_F)
    assert sep.permanova_p < 0.05


# ==========================================================================
# Replicate structure
# ==========================================================================
def test_nested_batch_effect_is_spurious_unless_replicates_are_respected():
    """A dish effect with labels nested in dishes is not a treatment effect.

    This is the single most consequential behaviour in the package: the same
    data gives 'p = 0.005, perfect classification' when the replicate column is
    ignored and 'not significant' when it is not.
    """
    rng, X, y, labels = _base(seed=3, per_group=40)
    n_reps = 12                                          # 4 dishes per group
    reps = np.repeat(np.arange(n_reps), len(y) // n_reps)
    for b in range(n_reps):
        X[reps == b] += rng.normal(0, 2.0, X.shape[1])   # a whole-dish offset

    blocked = analyse_blocks(y, reps)
    assert blocked.design == "nested"
    assert blocked.n_units == n_reps, "the replicate is the experimental unit"

    honest = analyse_separation(X, y, labels, blocked, n_permutations=999,
                                random_state=SEED)
    naive = analyse_separation(X, y, labels, analyse_blocks(y, None),
                               n_permutations=999, random_state=SEED)

    assert honest.permanova_F == pytest.approx(naive.permanova_F), \
        "the statistic is the same; only the null it is compared with changes"
    assert naive.permanova_p < 0.01, "the naive analysis is fooled, as expected"
    assert honest.permanova_p > 20 * naive.permanova_p, \
        "respecting the replicate structure must cost the batch effect its p value"

    cls_honest = classify(X, y, labels, blocked, n_permutations=PERM,
                          bootstrap=200, random_state=SEED)
    cls_naive = classify(X, y, labels, analyse_blocks(y, None),
                         n_permutations=PERM, bootstrap=200, random_state=SEED)
    assert cls_naive.balanced_accuracy > 0.85, \
        "a classifier can recognise the dish almost perfectly"
    assert cls_honest.balanced_accuracy < cls_naive.balanced_accuracy - 0.25, \
        "holding out whole dishes must remove most of that apparent accuracy"
    assert cls_honest.permutation_p > naive.permanova_p, \
        "and what is left must not read as significant"


def test_crossed_design_permutes_within_blocks():
    _, X, y, _ = _base(per_group=12, k=3)
    reps = np.tile(np.arange(4), 9)                      # every block has all groups
    st = analyse_blocks(y, reps)
    assert st.design == "crossed"

    for yp in permutations(y, st, 20, random_state=SEED):
        for b in np.unique(st.blocks):
            m = st.blocks == b
            assert sorted(yp[m].tolist()) == sorted(y[m].tolist()), \
                "a within-block permutation must preserve each block's composition"


def test_nested_permutation_moves_whole_replicates():
    _, X, y, _ = _base(per_group=12, k=3)
    reps = np.repeat(np.arange(9), 4)
    st = analyse_blocks(y, reps)
    assert st.design == "nested"

    for yp in permutations(y, st, 20, random_state=SEED):
        for b in np.unique(st.blocks):
            assert len(set(yp[st.blocks == b].tolist())) == 1, \
                "a replicate must keep a single label"


def test_effective_permutations_bounds_the_p_value():
    _, X, y, labels = _base(per_group=9, k=3)
    reps = np.repeat(np.arange(3), 9)                    # only 3 replicates
    st = analyse_blocks(y, reps)
    assert effective_permutations(y, st) == 6            # 3! arrangements

    sep = analyse_separation(X, y, labels, st, n_permutations=PERM,
                             random_state=SEED)
    assert sep.p_resolution >= 1 / 7 - 1e-9, \
        "a design with six arrangements cannot produce p = 0.001"
    assert sep.permanova_p >= sep.p_resolution - 1e-9


def test_permutation_p_is_never_zero():
    assert permutation_p(10.0, np.zeros(99)) == pytest.approx(0.01)
    assert permutation_p(0.0, np.zeros(99)) == pytest.approx(1.0)


# ==========================================================================
# Classification
# ==========================================================================
def test_classifier_finds_nothing_in_noise():
    _, X, y, labels = _base(per_group=30)
    cls = classify(X, y, labels, analyse_blocks(y, None), n_permutations=PERM,
                   bootstrap=200, random_state=SEED)
    assert cls.permutation_p > 0.05
    assert cls.balanced_accuracy < cls.chance + 0.2
    assert not cls.separable


def test_classifier_reports_which_pairs_separate():
    _, X, y, labels = _base(per_group=30)
    X[y == 2, :4] += 3.0                       # g2 is obvious, g0 and g1 are not
    cls = classify(X, y, labels, analyse_blocks(y, None), n_permutations=PERM,
                   bootstrap=200, random_state=SEED)
    assert cls.separable
    pw = cls.pairwise.set_index(["group_a", "group_b"])["balanced_accuracy"]
    assert pw[("g0", "g2")] > 0.9
    assert pw[("g0", "g1")] < 0.7
    assert cls.confusion.shape == (3, 3)
    assert cls.confusion.sum() == len(y)


@pytest.mark.parametrize("model", ["lda_shrinkage", "svm_linear", "plsda",
                                   "logistic", "random_forest"])
def test_every_classifier_runs(model):
    _, X, y, labels = _base(per_group=20)
    X[y == 1, :3] += 2.0
    cls = classify(X, y, labels, analyse_blocks(y, None), model=model,
                   n_permutations=49, bootstrap=100, random_state=SEED)
    assert 0.0 <= cls.balanced_accuracy <= 1.0
    assert np.isfinite(cls.permutation_p)


def test_balanced_accuracy_ignores_class_imbalance():
    rng = np.random.default_rng(SEED)
    y = np.array([0] * 90 + [1] * 10)
    X = rng.normal(size=(100, 8))
    cls = classify(X, y, ["big", "small"], analyse_blocks(y, None),
                   n_permutations=99, bootstrap=200, random_state=SEED)
    assert cls.chance == pytest.approx(0.5)
    assert cls.balanced_accuracy < 0.75, \
        "always predicting the big class must not look like success"


# ==========================================================================
# Interpretation
# ==========================================================================
def test_haufe_demotes_a_suppressor_that_the_raw_weights_promote():
    """A metric can earn a large weight while carrying no group information."""
    from sklearn.svm import SVC

    rng = np.random.default_rng(1)
    d, per = 20, 30
    y = np.repeat(np.arange(3), per)
    X = rng.normal(size=(3 * per, d))
    X[y == 1, :3] += 1.2
    X[y == 2, :3] -= 1.2
    signal = np.where(y == 1, 1.2, 0.0) - np.where(y == 2, 1.2, 0.0)
    X[:, 5] = X[:, 0] - signal + rng.normal(0, 0.2, len(y))   # suppressor

    W = SVC(kernel="linear", class_weight="balanced").fit(X, y).coef_
    A = haufe_patterns(X, W, y)

    weight_rank = list(np.argsort(-np.linalg.norm(W, axis=0)))
    haufe_rank = list(np.argsort(-np.linalg.norm(np.atleast_2d(A), axis=0)))

    assert set(haufe_rank[:3]) == {0, 1, 2}, "the real metrics come top"
    assert weight_rank.index(5) < 3, "the raw weight promotes the suppressor"
    assert haufe_rank.index(5) > 5, "the activation pattern demotes it"


def test_correlated_cluster_is_worth_more_than_its_members():
    rng = np.random.default_rng(2)
    per = 30
    y = np.repeat(np.arange(2), per)
    X = rng.normal(size=(2 * per, 10))
    X[y == 1, 0] += 2.0
    X[:, 1] = X[:, 0] + rng.normal(0, 0.1, len(y))    # a near-duplicate metric
    names = [f"m{i}" for i in range(10)]

    imp = permutation_importance(X, y, names, analyse_blocks(y, None),
                                 n_repeats=5, random_state=SEED)
    single = imp.table.set_index("feature")["importance"]
    cluster = imp.cluster_table
    pair = cluster[cluster["n_members"] > 1]
    assert not pair.empty, "the duplicated metrics must land in one cluster"
    assert pair["importance"].max() > max(single["m0"], single["m1"]), \
        "shuffling one member understates what the pair is worth"


# ==========================================================================
# Effect sizes
# ==========================================================================
def test_hedges_g_is_corrected_and_signed():
    a = np.array([0.0] * 10)
    b = np.array([1.0] * 10) + np.linspace(-0.01, 0.01, 10)
    assert hedges_g(a + np.linspace(-0.01, 0.01, 10), b) > 0
    assert hedges_g(b, a + np.linspace(-0.01, 0.01, 10)) < 0
    assert np.isnan(hedges_g(np.array([1.0]), np.array([2.0])))


def test_effect_table_has_intervals_that_bracket_the_estimate():
    rng = np.random.default_rng(SEED)
    y = np.repeat([0, 1], 40)
    values = {"speed": np.where(y == 1, 3.0, 1.0) + rng.normal(0, 0.5, 80),
              "noise": rng.normal(0, 1, 80)}
    df = estimate_effects(values, y, ["ctrl", "treated"], n_boot=400,
                          random_state=SEED)
    assert list(df["feature"])[0] == "speed", "the real effect sorts first"

    for _, r in df.iterrows():
        assert r["g_ci_low"] <= r["hedges_g"] <= r["g_ci_high"]
        assert r["diff_ci_low"] <= r["difference"] <= r["diff_ci_high"]

    speed = df[df.feature == "speed"].iloc[0]
    noise = df[df.feature == "noise"].iloc[0]
    assert speed["diff_ci_low"] > 0, "a real difference clears zero"
    assert noise["diff_ci_low"] < 0 < noise["diff_ci_high"], "noise does not"


# ==========================================================================
# Verdict
# ==========================================================================
def test_verdict_warns_when_the_effect_is_a_spread_difference():
    _, X, y, labels = _base(per_group=30)
    X[y == 2] *= 3.0
    st = analyse_blocks(y, None)
    sep = analyse_separation(X, y, labels, st, n_permutations=PERM,
                             random_state=SEED)
    v = build_verdict(sep, None, None, st, n_samples=len(y), n_features=X.shape[1],
                      group_values=labels)
    assert "spread" in v.text().lower() or "variab" in v.text().lower()


def test_verdict_flags_a_missing_replicate_column():
    _, X, y, labels = _base()
    st = analyse_blocks(y, None)
    v = build_verdict(None, None, None, st, n_samples=len(y),
                      n_features=X.shape[1], group_values=labels)
    assert any("replicate" in c for c in v.caveats)


def test_verdict_flags_supervised_projections():
    _, X, y, labels = _base()
    v = build_verdict(None, None, None, analyse_blocks(y, None),
                      group_values=labels, supervised_used=True)
    assert any("construction" in c for c in v.caveats)
