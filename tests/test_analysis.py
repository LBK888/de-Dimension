"""The projection registry, the quality layer, the scan, and backward compatibility.

The registry's promise is that adding a method costs one file and no edits
elsewhere.  These tests pin the consequences of that promise: every registered
method runs on ordinary data, every supervised one carries its caveat and its
out-of-fold coordinates, and the quality numbers agree with scikit-learn where
scikit-learn has an opinion.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from somtrack.analysis import (DataContext, all_methods, default_enabled,
                               evaluate, get, quality_table, run_projections)
from somtrack.analysis.quality import (neighbour_agreement, point_reliability,
                                       procrustes_align, rank_quality,
                                       seed_stability)
from somtrack.analysis.scan import default_grid, scan
from somtrack.config import AnalysisConfig

SEED = 0


@pytest.fixture(scope="module")
def ctx():
    rng = np.random.default_rng(SEED)
    y = np.repeat(np.arange(3), 30)
    X = rng.normal(size=(90, 14))
    X[y == 1, :3] += 2.0
    X[y == 2, :3] -= 2.0
    return DataContext(
        X=X, feature_names=[f"m{i:02d}" for i in range(14)], group_codes=y,
        group_values=["a", "b", "c"], replicates=np.tile(np.arange(3), 30),
        sample_ids=np.array([f"s{i}" for i in range(90)]), random_state=SEED)


# ==========================================================================
# The registry
# ==========================================================================
def test_registry_is_populated_and_ordered():
    specs = all_methods()
    keys = [s.key for s in specs]
    assert {"pca", "mds", "tsne", "pacmap", "lda", "plsda", "svm"} <= set(keys)

    families = [s.family for s in specs]
    order = {"linear": 0, "manifold": 1, "supervised": 2, "som": 3}
    assert families == sorted(families, key=lambda f: order[f]), \
        "methods must be offered linear first, supervised last"


def test_supervised_methods_must_declare_a_caveat():
    from somtrack.analysis.registry import MethodSpec

    for spec in all_methods():
        if spec.supervised:
            assert spec.caveat, f"{spec.key} is supervised but carries no caveat"
            assert spec.needs_groups

    with pytest.raises(ValueError, match="must declare a caveat"):
        MethodSpec(key="bad", label="Bad", family="supervised",
                   run=lambda c, p: None, supervised=True)


def test_defaults_exclude_supervised_methods():
    assert not any(get(k).supervised for k in default_enabled()), \
        "a fresh install must not run a projection that separates by construction"


def test_every_method_declares_citations():
    for spec in all_methods():
        assert spec.citations, f"{spec.key} has no citation"
        from somtrack.citations import CITATIONS

        for key in spec.citations:
            assert key in CITATIONS, f"{spec.key} cites unknown '{key}'"


@pytest.mark.parametrize("key", [s.key for s in all_methods() if s.available()])
def test_every_available_method_runs(ctx, key):
    from somtrack.analysis.registry import run_method

    spec = get(key)
    if spec.needs_groups and not ctx.has_groups:
        pytest.skip("needs groups")
    res = run_method(key, ctx)
    assert res.coords.shape == (ctx.n_samples, 2)
    assert np.all(np.isfinite(res.coords))
    assert res.feature_axes.shape == (ctx.n_features, 2)
    assert res.method_key == key
    if spec.supervised:
        assert res.caveat
        assert res.oof_coords is not None or key == "supervised_umap"
        assert res.cv is not None


def test_parameters_are_derived_from_the_data_not_the_library_default(ctx):
    """t-SNE's published default perplexity is for 10^4 points, not for 90."""
    spec = get("tsne")
    resolved = spec.resolve_params(ctx)
    assert resolved["perplexity"] <= (ctx.n_samples - 1) / 3
    assert resolved["perplexity"] < 30

    pinned = spec.resolve_params(ctx, {"perplexity": 7.0})
    assert pinned["perplexity"] == 7.0


def test_unavailable_method_is_reported_not_crashed(ctx):
    warnings: list[str] = []
    out = run_projections(ctx, ["pca", "definitely_not_a_method"],
                          warn=warnings.append, reliability=False)
    assert "pca" in out
    assert warnings and "definitely_not_a_method" in warnings[0]


def test_run_projections_records_methods(ctx):
    from somtrack.citations import MethodsLog

    log = MethodsLog()
    run_projections(ctx, ["pca", "mds"], log=log, reliability=False)
    text = " ".join(e.sentence() for e in log.entries)
    assert "PCA" in text and "Hotelling" in text
    assert {"hotelling1933", "gower1966"} <= set(log.citation_keys())


def test_quality_table_has_a_row_per_method(ctx):
    out = run_projections(ctx, ["pca", "mds", "tsne"], reliability_null=5)
    df = quality_table(out)
    assert len(df) == 3
    assert {"method", "trustworthiness", "rnx_auc"} <= set(df.columns)


# ==========================================================================
# Quality
# ==========================================================================
def test_rank_quality_matches_sklearn(ctx):
    from sklearn.decomposition import PCA
    from sklearn.manifold import trustworthiness

    Y = PCA(n_components=2, random_state=SEED).fit_transform(ctx.X)
    for k in (5, 10, 20):
        q = rank_quality(ctx.X, Y, k=k)
        assert q.trustworthiness == pytest.approx(
            trustworthiness(ctx.X, Y, n_neighbors=k), abs=1e-9)
        assert q.continuity == pytest.approx(
            trustworthiness(Y, ctx.X, n_neighbors=k), abs=1e-9)


def test_rnx_auc_separates_a_real_projection_from_a_random_one(ctx):
    from sklearn.decomposition import PCA

    good = rank_quality(ctx.X, PCA(n_components=2, random_state=SEED)
                        .fit_transform(ctx.X)).auc
    rubbish = rank_quality(
        ctx.X, np.random.default_rng(1).normal(size=(ctx.n_samples, 2))).auc
    assert good > 0.15
    assert abs(rubbish) < 0.05, "a random layout must score about zero"


def test_reliability_flags_a_scrambled_projection(ctx):
    from sklearn.decomposition import PCA

    good = point_reliability(ctx.X, PCA(n_components=2, random_state=SEED)
                             .fit_transform(ctx.X), n_null=8, random_state=SEED)
    bad = point_reliability(ctx.X, np.random.default_rng(2)
                            .normal(size=(ctx.n_samples, 2)),
                            n_null=8, random_state=SEED)
    # What matters is the contrast, not the absolute level: how high a real
    # projection scores depends on how much of a 14-dimensional data set two
    # dimensions can hold, which is a property of the data, not of the method.
    assert np.nanmean(good.score) > 0.35
    assert np.nanmean(bad.score) < 0.15
    assert np.nanmean(good.score) > 3 * np.nanmean(bad.score)
    assert good.n_dubious <= bad.n_dubious


def test_procrustes_recovers_a_rotation_and_scaling():
    rng = np.random.default_rng(SEED)
    A = rng.normal(size=(40, 2))
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    B = (A @ R) * 2.5 + np.array([3.0, -1.0])
    assert np.allclose(procrustes_align(A, B), A, atol=1e-9)


def test_seed_stability_is_zero_for_identical_runs():
    rng = np.random.default_rng(SEED)
    Y = rng.normal(size=(50, 2))
    st = seed_stability([Y, Y.copy(), Y.copy()])
    assert st.mean == pytest.approx(0.0, abs=1e-9)
    assert st.n_seeds == 3


def test_neighbour_agreement_is_one_on_the_diagonal(ctx):
    from sklearn.decomposition import PCA

    Y = PCA(n_components=2, random_state=SEED).fit_transform(ctx.X)
    M = neighbour_agreement({"a": Y, "b": Y * 3.0,
                             "rand": np.random.default_rng(3)
                             .normal(size=(ctx.n_samples, 2))}, k=8)
    assert M.loc["a", "a"] == pytest.approx(1.0)
    assert M.loc["a", "b"] == pytest.approx(1.0), "scaling must not change neighbours"
    assert M.loc["a", "rand"] < 0.3


# ==========================================================================
# Scans
# ==========================================================================
def test_scan_keeps_every_layout_and_reports_a_plateau(ctx):
    res = scan(ctx, "pacmap", {"n_neighbors": [5, 10, 15]},
               criterion="rnx_auc", reliability_null=4, n_jobs=1)
    assert len(res.rows) == 3
    assert len(res.coords) == 3
    assert all(v.shape == (ctx.n_samples, 2) for v in res.coords.values())
    assert res.best_params()["n_neighbors"] in (5, 10, 15)
    assert "n_neighbors" in res.plateau_text()


def test_scan_grid_comes_from_the_method(ctx):
    grid = default_grid("tsne", ctx)
    assert "perplexity" in grid
    assert all(v <= max(2.0, (ctx.n_samples - 1) / 3) for v in grid["perplexity"])


def test_plateau_prefers_the_suggested_value_when_scores_tie(ctx):
    """On a flat landscape the scan has learned nothing and must not pretend."""
    res = scan(ctx, "tsne", {"perplexity": [5.0, 10.0, 15.0, 20.0, 25.0]},
               criterion="dubious_fraction", reliability_null=4, n_jobs=1,
               tolerance=1.0)
    assert len(res.plateau) == 5
    suggested = get("tsne").param("perplexity").resolve(ctx)
    chosen = res.best_params()["perplexity"]
    assert abs(chosen - suggested) <= 5.0, \
        "a tie must fall back to the value the method would have chosen unaided"


# ==========================================================================
# Configuration compatibility
# ==========================================================================
def test_a_2_0_config_still_loads_and_means_the_same_thing(tmp_path):
    old = {
        "embedding": {"run_pca": True, "run_tsne": True, "run_umap": False,
                      "tsne_perplexity": 12.0, "umap_min_dist": 0.3},
        "som": {"algorithm": "supervised", "epochs": 300, "label_weight": 0.4},
        "preprocess": {"scaler": "robust"},
        "export": {"out_dir": "somewhere"},
    }
    path = tmp_path / "old_config.json"
    path.write_text(json.dumps(old), encoding="utf-8")

    cfg = AnalysisConfig.from_json(path)
    assert cfg.som.algorithm == "supervised"
    assert cfg.som.epochs == 300
    assert cfg.preprocess.scaler == "robust"
    assert str(cfg.export.out_dir) == "somewhere"
    assert cfg.embedding.resolved_methods() == ["pca", "tsne"]
    assert cfg.embedding.resolved_overrides()["tsne"]["perplexity"] == 12.0
    assert cfg.embedding.resolved_overrides()["umap"]["min_dist"] == 0.3


def test_config_round_trips_the_new_sections(tmp_path):
    cfg = AnalysisConfig()
    cfg.embedding.methods = ["pca", "pacmap"]
    cfg.embedding.overrides = {"pacmap": {"n_neighbors": 7}}
    cfg.stats.unit_of_analysis = "replicate"
    cfg.stats.n_permutations = 4999
    cfg.report.profile = "full"
    cfg.figure.categorical_max = 6

    path = cfg.to_json(tmp_path / "cfg.json")
    back = AnalysisConfig.from_json(path)
    assert back.embedding.methods == ["pca", "pacmap"]
    assert back.embedding.overrides == {"pacmap": {"n_neighbors": 7}}
    assert back.stats.unit_of_analysis == "replicate"
    assert back.stats.n_permutations == 4999
    assert back.report.profile == "full"
    assert back.figure.categorical_max == 6


def test_unknown_config_keys_are_ignored(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"som": {"epochs": 42, "warp_drive": True},
                                "not_a_section": {"x": 1}}), encoding="utf-8")
    cfg = AnalysisConfig.from_json(path)
    assert cfg.som.epochs == 42


def test_recipes_all_build_a_usable_config():
    from somtrack import recipes

    for r in recipes.RECIPES:
        cfg = r.build()
        assert isinstance(cfg.embedding.methods, list)
        assert cfg.report.profile in ("core", "full")
        for key in cfg.embedding.methods:
            from somtrack.analysis import has

            assert has(key), f"recipe '{r.key}' names unknown method '{key}'"
        if any(get(k).supervised for k in cfg.embedding.methods):
            assert cfg.embedding.allow_supervised, \
                f"recipe '{r.key}' selects a supervised method without enabling them"
