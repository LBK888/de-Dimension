"""End-to-end orchestration: features -> clustering -> figures -> disk.

The UI and the CLI both drive this module, so a run launched from the desktop
app and a run launched from a script produce byte-identical output for the same
config.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from . import export as ex
from . import viz
from .analysis import DataContext, ProjectionResult, neighbour_agreement
from .analysis import quality_table as projection_quality_table
from .analysis import run_projections
from .association import AssociationBundle, analyse_associations
from .citations import MethodsLog
from .config import AnalysisConfig
from .io_tables import FeatureDataset, SpotDataset
from .metrics import compute_features
from .nodecluster import NodeClustering, cluster_nodes
from .preprocess import PreparedData, feature_summary_table, prepare
from .report import build_report, write_methods_text, write_report
from .som import SomResult, quality_report, train_som
from .stats import (BlockStructure, ClassifyResult, ImportanceResult,
                    SeparationReport, Verdict, analyse_blocks,
                    analyse_separation, build_verdict, classify,
                    estimate_effects, permutation_importance)

Progress = Callable[[str, float], None]


# --------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    config: AnalysisConfig
    features: FeatureDataset
    prep: PreparedData
    som: SomResult | None = None
    node_clusters: NodeClustering | None = None
    projections: dict[str, ProjectionResult] = field(default_factory=dict)
    associations: AssociationBundle | None = None
    quality: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # --- 3.0: the statistics layer ----------------------------------------
    block_structure: BlockStructure | None = None
    separation: SeparationReport | None = None
    classification: ClassifyResult | None = None
    importance: ImportanceResult | None = None
    effects: pd.DataFrame | None = None
    verdict: Verdict | None = None
    projection_agreement: pd.DataFrame | None = None
    methods_log: MethodsLog = field(default_factory=MethodsLog)
    figure_captions: dict[str, str] = field(default_factory=dict)

    @property
    def embeddings(self) -> dict[str, ProjectionResult]:
        """2.0 compatibility: projections keyed by display name."""
        return {p.name: p for p in self.projections.values()}

    def tables(self) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {
            "features": self.features.frame,
            "feature_summary": feature_summary_table(self.prep),
        }
        if self.separation is not None:
            out["separation_tests"] = self.separation.table()
            if not self.separation.pairwise.empty:
                out["separation_pairwise"] = self.separation.pairwise
        if self.classification is not None and np.isfinite(
                self.classification.balanced_accuracy):
            out["classification"] = self.classification.table()
            cf = self.classification.confusion_frame()
            if not cf.empty:
                out["classification_confusion"] = cf.reset_index().rename(
                    columns={"index": "group"})
            if not self.classification.pairwise.empty:
                out["classification_pairwise"] = self.classification.pairwise
            if self.classification.activation is not None:
                from .stats.interpret import weight_table

                out["classifier_weights"] = weight_table(
                    self.classification.feature_names, self.classification.weights,
                    self.classification.activation, self.classification.vip)
        if self.importance is not None and not self.importance.table.empty:
            out["permutation_importance"] = self.importance.table
            if not self.importance.cluster_table.empty:
                out["permutation_importance_clusters"] = self.importance.cluster_table
        if self.effects is not None and not self.effects.empty:
            out["effect_sizes"] = self.effects
        if self.projections:
            out["projection_quality"] = projection_quality_table(self.projections)
        if self.projection_agreement is not None and not self.projection_agreement.empty:
            out["projection_agreement"] = (
                self.projection_agreement.reset_index().rename(
                    columns={"index": "method"}))
        if self.associations is not None:
            out["metric_group_association"] = self.associations.univariate
            out["metric_importance"] = self.associations.importance.table
            if self.associations.gradients is not None:
                out["map_gradients"] = self.associations.gradients.table()
            if self.associations.node_profile is not None:
                out["node_cluster_profile"] = self.associations.node_profile
            if self.associations.enrichment is not None and not self.associations.enrichment.empty:
                out["group_node_enrichment"] = self.associations.enrichment
        if self.som is not None:
            out["som_codebook"] = _codebook_table(self.som, self.prep)
            out["sample_assignment"] = _assignment_table(self.som, self.prep, self.node_clusters)
            out["som_training"] = pd.DataFrame({
                "epoch": self.som.history.epochs,
                "quantisation_error": self.som.history.qe,
                "topographic_error": self.som.history.te,
                "group_purity": self.som.history.purity,
                "sigma": self.som.history.sigma,
            })
            if self.som.relevance is not None:
                out["feature_relevance"] = pd.DataFrame({
                    "feature": self.som.feature_names,
                    "relevance": self.som.relevance,
                }).sort_values("relevance", ascending=False)
        if self.node_clusters is not None and self.node_clusters.scores is not None:
            out["node_cluster_scores"] = self.node_clusters.scores
        for key, e in self.projections.items():
            df = pd.DataFrame(e.coords, columns=[e.axis_labels[0], e.axis_labels[1]])
            if e.oof_coords is not None:
                df["out_of_fold_1"] = e.oof_coords[:, 0]
                df["out_of_fold_2"] = e.oof_coords[:, 1]
            if e.reliability is not None:
                df["reliability"] = e.reliability
                df["dubious"] = e.dubious
            if e.stability is not None:
                df["seed_instability"] = e.stability
            df.insert(0, "group", self.prep.groups)
            df.insert(0, "sample", self.prep.sample_ids)
            out[f"projection_{key}"] = df
        if self.quality:
            out["map_quality"] = pd.DataFrame(
                [{"metric": k, "value": v} for k, v in self.quality.items()])
        return out


def _codebook_table(res: SomResult, prep: PreparedData) -> pd.DataFrame:
    W = prep.inverse(res.codebook)
    df = pd.DataFrame(W, columns=res.feature_names)
    df.insert(0, "hits", res.hits)
    df.insert(0, "row", res.lattice.coords[:, 1])
    df.insert(0, "col", res.lattice.coords[:, 0])
    df.insert(0, "node", np.arange(res.n_nodes))
    M = res.hit_matrix(prep.group_codes, prep.n_groups)
    for g, label in enumerate(prep.group_values):
        df[f"hits[{label}]"] = M[:, g]
    return df


def _assignment_table(res: SomResult, prep: PreparedData,
                      nc: NodeClustering | None) -> pd.DataFrame:
    df = pd.DataFrame({
        "sample": prep.sample_ids,
        "group": prep.groups,
        "replicate": prep.replicates,
        "bmu": res.bmu,
        "bmu_col": res.lattice.coords[res.bmu, 0],
        "bmu_row": res.lattice.coords[res.bmu, 1],
        "bmu_distance": res.bmu_dist,
        "runner_up": res.bmu2,
        "runner_up_distance": res.bmu2_dist,
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ambiguity"] = np.nan_to_num(res.bmu_dist / res.bmu2_dist)
    if nc is not None and nc.best_k:
        df[f"node_cluster_k{nc.best_k}"] = nc.sample_labels(res)
    return df


# --------------------------------------------------------------------------
def features_from_spots(spots: SpotDataset, config: AnalysisConfig,
                        progress: Progress | None = None) -> FeatureDataset:
    def cb(i, n):
        if progress:
            progress(f"Computing metrics ({i}/{n} tracks)", i / max(n, 1))

    return compute_features(spots, config.track,
                            selected=config.selected_features or None, progress=cb)


def run_analysis(
    features: FeatureDataset,
    config: AnalysisConfig,
    progress: Progress | None = None,
) -> AnalysisResult:
    """Everything after the feature table exists."""
    def say(msg: str, frac: float) -> None:
        if progress:
            progress(msg, frac)

    warnings_: list[str] = []

    log = MethodsLog()

    say("Preparing feature matrix", 0.02)
    prep = prepare(features, config.preprocess,
                   selected=config.selected_features or None)
    warnings_ += prep.notes
    _log_preprocessing(log, config, prep)

    result = AnalysisResult(config=config, features=features, prep=prep,
                            warnings=warnings_, methods_log=log)

    # Who is independent of whom?  Everything statistical depends on this, so it
    # is settled once, before any test runs, and recorded in the report.
    result.block_structure = analyse_blocks(
        prep.group_codes, prep.replicates,
        use_replicates=(config.stats.use_replicate_blocks
                        or config.stats.unit_of_analysis == "replicate"))

    if prep.n_groups < 2 and config.som.algorithm in ("supervised", "relevance"):
        warnings_.append(
            f"'{config.som.algorithm}' SOM needs >= 2 groups; falling back to batch SOM."
        )
        config.som.algorithm = "batch"

    say("Training SOM", 0.06)
    som = train_som(
        prep.X, config.som, feature_names=prep.feature_names,
        group_codes=prep.group_codes, n_groups=prep.n_groups,
        progress=lambda i, n: say(f"Training SOM (epoch {i}/{n})",
                                  0.06 + 0.30 * i / max(n, 1)),
    )
    result.som = som
    result.quality = quality_report(som, prep.group_codes, prep.n_groups)
    _log_som(log, config, som)

    say("Clustering nodes", 0.38)
    result.node_clusters = cluster_nodes(som, config.node_cluster, prep.group_codes)
    if result.node_clusters is not None and result.node_clusters.best_k:
        log.record("Clustering", "second-level clustering of the SOM codebook",
                   f"{config.node_cluster.method} over k = {config.node_cluster.k_min}"
                   f"-{config.node_cluster.k_max}, with k chosen by the average rank "
                   f"of three internal validity indices",
                   citations=("vesanto2000", "rousseeuw1987", "davies1979",
                              "calinski1974"),
                   chosen_k=result.node_clusters.best_k)

    say("Running projections", 0.44)
    result.projections = _run_projections(result, config, prep, log, warnings_, say)

    if len(result.projections) > 1:
        result.projection_agreement = neighbour_agreement(
            {p.name: p.coords for p in result.projections.values()})

    if config.stats.enabled and prep.n_groups >= 2:
        _run_statistics(result, config, prep, log, say)

    say("Testing metric-group associations", 0.92)
    result.associations = analyse_associations(prep, som, result.node_clusters)
    log.record("Statistics", "per-metric association tests",
               "one-way ANOVA and Kruskal-Wallis per metric with eta-squared and "
               "epsilon-squared effect sizes, corrected across metrics by the "
               "Benjamini-Hochberg procedure",
               citations=("kruskal1952", "benjamini1995", "cohen1988"))

    result.verdict = build_verdict(
        separation=result.separation,
        classification=result.classification,
        importance=result.importance,
        structure=result.block_structure,
        projections=result.projections,
        n_samples=prep.n_samples, n_features=prep.n_features,
        group_values=list(prep.group_values),
        supervised_used=any(p.supervised for p in result.projections.values()),
    )

    say("Analysis complete", 1.0)
    return result


# --------------------------------------------------------------------------
def _run_projections(result, config, prep, log, warnings_, say):
    emb = config.embedding
    ctx = DataContext.from_prepared(prep, random_state=emb.random_state,
                                    n_components=emb.n_components)
    methods = emb.resolved_methods()

    if not emb.allow_supervised:
        from .analysis import get as get_method

        dropped = [m for m in methods if get_method(m).supervised]
        methods = [m for m in methods if m not in dropped]
        if dropped:
            warnings_.append(
                "Supervised projections (" + ", ".join(dropped) + ") were requested "
                "but are off by default, because they separate groups by "
                "construction. Enable them with embedding.allow_supervised = True "
                "once the cross-validated result is in hand.")

    overrides = emb.resolved_overrides()
    # A supervised projection runs its own permutation test.  It should use the
    # number of permutations the statistics settings ask for, not a second,
    # different number that then disagrees with the methods section.
    for key in methods:
        from .analysis import get as get_method

        spec = get_method(key)
        if spec.supervised and spec.param("n_permutations") is not None:
            overrides.setdefault(key, {}).setdefault(
                "n_permutations", config.stats.classification_permutations)

    return run_projections(
        ctx, methods, overrides,
        rank=emb.compute_rank_quality,
        reliability=emb.compute_reliability,
        reliability_null=emb.reliability_null,
        stability_seeds=emb.stability_seeds,
        log=log, warn=warnings_.append,
        progress=lambda i, n: say(f"Projections ({i}/{max(n, 1)})",
                                  0.44 + 0.20 * i / max(n, 1)),
    )


def _run_statistics(result, config, prep, log, say) -> None:
    """The layer that turns the pictures into a testable claim."""
    st = config.stats
    structure = result.block_structure
    X, y = prep.X, prep.group_codes
    names = list(prep.feature_names)

    if st.unit_of_analysis == "replicate" and structure.blocked:
        from .stats.blocks import aggregate_to_units

        X, y, _ = aggregate_to_units(X, y, structure)
        structure = analyse_blocks(y, None)
        log.record("Statistics", "aggregation to experimental units",
                   "metrics were averaged within each replicate before testing, so "
                   "the replicate rather than the individual is the unit of analysis",
                   citations=("hurlbert1984", "lazic2018"))
    elif structure.blocked:
        log.record("Statistics", "replicate-aware permutation and cross-validation",
                   structure.note.rstrip("."),
                   citations=("anderson2003", "hurlbert1984", "lazic2018"))

    if st.run_permanova or st.run_permdisp or st.run_energy:
        say("Testing whether the groups differ", 0.68)
        result.separation = analyse_separation(
            X, y, list(prep.group_values), structure,
            metric=st.distance, n_permutations=st.n_permutations,
            random_state=st.random_state, run_permanova=st.run_permanova,
            run_permdisp=st.run_permdisp, run_energy=st.run_energy,
            dispersion_centre=st.dispersion_centre)
        _log_separation(log, st, result.separation)

    if st.run_classification:
        say("Cross-validating group assignment", 0.76)
        result.classification = classify(
            X, y, list(prep.group_values), structure,
            model=st.classifier, scheme=st.cv_scheme, n_splits=st.cv_splits,
            repeats=st.cv_repeats, n_permutations=st.classification_permutations,
            bootstrap=min(st.bootstrap, 2000), feature_names=names,
            random_state=st.random_state)
        _log_classification(log, st, result.classification)

    if st.run_importance:
        say("Measuring which metrics matter", 0.84)
        result.importance = permutation_importance(
            X, y, names, structure, model=st.importance_model,
            n_repeats=st.importance_repeats,
            cluster_threshold=st.cluster_threshold, scheme=st.cv_scheme,
            n_splits=st.cv_splits, random_state=st.random_state)
        log.record("Statistics", "permutation importance",
                   f"cross-validated permutation importance scored by the loss in "
                   f"balanced accuracy, computed per metric and per cluster of "
                   f"metrics correlated above |r| = {st.cluster_threshold:g}",
                   citations=("breiman2001", "altmann2010", "brodersen2010"),
                   model=st.importance_model, repeats=st.importance_repeats)

    if st.run_effect_sizes:
        say("Estimating effect sizes", 0.88)
        src = prep.source
        values = {}
        for name in names:
            if src is not None and name in src.frame.columns:
                values[name] = src.frame.loc[prep.keep_mask, name].to_numpy(float)
        if values:
            result.effects = estimate_effects(
                values, prep.group_codes, list(prep.group_values),
                reference=st.effect_reference_group,
                units=(src.units if src else {}),
                n_boot=st.bootstrap, random_state=st.random_state)
            log.record("Statistics", "effect sizes with bootstrap intervals",
                       "Hedges' g and the raw difference in measurement units, each "
                       "with a bias-corrected and accelerated bootstrap 95% interval",
                       citations=("hedges1981", "efron1987", "ho2019"),
                       bootstrap_resamples=st.bootstrap)


# --------------------------------------------------------------------------
def _log_preprocessing(log: MethodsLog, config: AnalysisConfig,
                       prep: PreparedData) -> None:
    pre = config.preprocess
    detail = f"metrics were scaled with the {pre.scaler} transform"
    if 0 < pre.collinearity_threshold < 1:
        detail += (f" and collinear metrics were pruned at "
                   f"|r| > {pre.collinearity_threshold:g}")
    detail += f"; missing values were handled by '{pre.nan_policy}'"
    log.record("Data preparation", "feature scaling and pruning", detail,
               n_samples=prep.n_samples, n_metrics=prep.n_features)


def _log_som(log: MethodsLog, config: AnalysisConfig, som) -> None:
    s = config.som
    cites: list[str] = ["kohonen2001"]
    label = {"batch": "batch self-organising map",
             "online": "sequential (online) self-organising map",
             "supervised": "XY-fused supervised self-organising map",
             "relevance": "self-organising map with GRLVQ relevance learning",
             "growing": "growing self-organising map"}.get(s.algorithm,
                                                           "self-organising map")
    if s.algorithm == "online":
        cites.append("kohonen1982")
    if s.algorithm == "batch":
        cites.append("kohonen1999")
    if s.algorithm == "supervised":
        cites.append("melssen2006")
    if s.algorithm == "relevance":
        cites.append("hammer2002")
    if s.algorithm == "growing":
        cites.append("alahakoon2000")

    params = {"map": f"{som.width} x {som.height}", "lattice": s.topology,
              "epochs": s.epochs, "initialisation": s.init}
    if s.algorithm == "supervised":
        params["label_weight"] = s.label_weight
    caveat = ("The group labels were used during training, so the map separates "
              "the groups partly by construction; report the label weight and "
              "judge separation from the cross-validated statistics."
              if s.algorithm == "supervised" else "")

    log.record("Clustering", label,
               f"trained on the scaled feature matrix, with map quality reported "
               f"as quantisation and topographic error",
               citations=tuple(cites) + ("kiviluoto1996",),
               caveat=caveat, **params)


def _log_separation(log: MethodsLog, st, sep) -> None:
    if st.run_permanova:
        log.record("Statistics", "PERMANOVA",
                   f"permutational multivariate analysis of variance on "
                   f"{sep.metric} distances, with R-squared as the effect size",
                   citations=("anderson2001",), permutations=st.n_permutations,
                   distance=sep.metric)
    if st.run_permdisp:
        log.record("Statistics", "PERMDISP",
                   "a separate permutation test of homogeneity of multivariate "
                   "dispersions, because PERMANOVA rejects for a difference in "
                   "spread as readily as for a difference in location",
                   citations=("anderson2006",), permutations=st.n_permutations)
    if st.run_energy:
        log.record("Statistics", "energy k-sample test",
                   "a distribution-free test consistent against any difference in "
                   "distribution, not only a shift in the mean",
                   citations=("szekely2013", "gretton2012"),
                   permutations=st.n_permutations)
    log.record("Statistics", "Mahalanobis distance between group centroids",
               "with the pooled within-group covariance shrunk towards a "
               "well-conditioned target before inversion",
               citations=("mahalanobis1936", "ledoit2004"))


def _log_classification(log: MethodsLog, st, cls) -> None:
    from .stats.classify import MODELS

    spec = MODELS.get(st.classifier, {})
    log.record("Statistics", "cross-validated classification",
               f"{spec.get('detail', st.classifier)}, evaluated by "
               f"{cls.cv_name} and scored by balanced accuracy; significance was "
               f"assessed by shuffling the group labels rather than by a binomial "
               f"test, which is anti-conservative for cross-validated accuracies",
               citations=tuple(spec.get("citations", ()))
               + ("brodersen2010", "ojala2010", "noirhomme2014", "cohen1960"),
               permutations=cls.n_permutations)
    if cls.activation is not None:
        log.record("Statistics", "Haufe transform of the classifier weights",
                   "classifier weights were converted to activation patterns "
                   "before interpretation, because a large weight can belong to a "
                   "metric that carries no group information and only cancels "
                   "noise in another",
                   citations=("haufe2014",))


# --------------------------------------------------------------------------
def build_figures(result: AnalysisResult,
                  progress: Progress | None = None) -> list[tuple[str, "viz.Panel"]]:
    """Render the figure set, in the order the report reads.

    The order is deliberate: the conclusion first, then the evidence for it,
    then the descriptive figures.  A reader who stops after three pages should
    have the answer, not a self-organising map they have to interpret first.

    ``ReportConfig.profile`` decides how much is rendered.  ``core`` produces
    around a dozen figures -- the ones that carry the result -- because a
    thirty-page PDF is read by nobody; ``full`` produces everything.
    """
    cfg = result.config.figure
    profile = result.config.report.profile
    full = profile == "full"
    prep, som, nc = result.prep, result.som, result.node_clusters
    assoc = result.associations
    viz.apply_style(cfg)

    panels: list[tuple[str, viz.Panel]] = []
    captions: dict[str, str] = {}

    def add(name: str, fn, *a, core: bool = False, why: str = "", **kw) -> None:
        if not core and not full:
            return
        if progress:
            progress(f"Rendering {name}", min(0.95, 0.05 + 0.04 * len(panels)))
        try:
            panels.append((name, fn(*a, **kw)))
            captions[name] = why
        except Exception as exc:
            result.warnings.append(f"Figure '{name}' skipped: {exc}")

    # ---- 1. the conclusion and the evidence for it -----------------------
    if result.verdict is not None:
        add("01_verdict", viz.plot_verdict, result.verdict, cfg, core=True,
            why="The conclusion, generated from the numbers in the panels below.")

    if result.separation is not None and np.isfinite(result.separation.permanova_p):
        add("02_separation", viz.plot_separation_summary, result.separation, cfg,
            core=True,
            why="Whether the groups differ, and whether the difference is a shift "
                "in the average or a change in within-group spread.")
        if not result.separation.pairwise.empty and prep.n_groups > 2:
            add("03_pairwise_separation", viz.plot_pairwise_separation,
                result.separation, cfg, core=True,
                why="Which specific pairs of groups differ.")

    if result.classification is not None and np.isfinite(
            result.classification.balanced_accuracy):
        add("04_classifiability", viz.plot_classifiability,
            result.classification, cfg, core=True,
            why="Cross-validated group assignment against a permutation null.")
        if result.classification.activation is not None:
            add("05_activation_patterns", viz.plot_activation_patterns,
                result.classification, cfg, core=True,
                why="Which metrics carry the difference, after the Haufe "
                    "transform that separates them from suppressor variables.")

    if result.importance is not None and not result.importance.table.empty:
        add("06_importance", viz.plot_importance, result.importance, cfg,
            core=True,
            why="How much held-out accuracy each metric, and each correlated "
                "cluster of metrics, is worth.")

    if result.effects is not None and not result.effects.empty:
        add("07_effect_sizes", viz.plot_effect_forest, result.effects, cfg,
            core=True,
            why="Standardised effect size per metric with bootstrap intervals.")

    top_metrics = (result.importance.top(4) if result.importance is not None
                   and not result.importance.table.empty
                   else (assoc.top_features(4) if assoc else []))
    for i, feat in enumerate(top_metrics[:4]):
        add(f"08_superplot_{_slug(feat)}", viz.plot_superplot, prep, feat, cfg,
            core=(i < 2),
            why=f"{feat} by group, showing every individual and each replicate's "
                f"own mean.")
    for i, feat in enumerate(top_metrics[:2]):
        add(f"09_estimation_{_slug(feat)}", viz.plot_estimation, prep, feat, cfg,
            core=(i == 0),
            reference=result.config.stats.effect_reference_group,
            why=f"{feat}: the size of the difference, with its uncertainty.")

    # ---- 2. projections ---------------------------------------------------
    if result.projections:
        add("10_method_grid", viz.plot_method_grid, result.projections, prep, cfg,
            core=True,
            why="Every projection that was run, side by side with its quality "
                "score.")
        add("11_rnx_curves", viz.plot_rnx_curves, result.projections, cfg,
            core=True,
            why="How much of each neighbourhood each projection preserved, "
                "across all scales.")
        add("12_embeddings_row", viz.plot_embedding_row, result.embeddings, prep, cfg,
            why="The leading projections on one row for direct comparison.")
        for key, proj in result.projections.items():
            add(f"13_{_slug(key)}", viz.plot_embedding, proj, prep, cfg,
                core=(key in ("pca", "pacmap", "umap")),
                highlight=(assoc.top_features(8) if assoc else None),
                why=f"{proj.name} projection with metric-direction arrows.")
            if proj.dubious is not None and proj.n_dubious:
                add(f"14_reliability_{_slug(key)}", viz.plot_point_reliability,
                    proj, prep, cfg,
                    why=f"Which points {proj.name} placed unreliably.")
            if proj.supervised and proj.oof_coords is not None:
                add(f"15_validated_{_slug(key)}", viz.plot_supervised_check,
                    proj, prep, cfg, core=True,
                    why=f"{proj.name} in-sample against out-of-fold -- the panel "
                        f"that separates a real difference from one the method "
                        f"was handed.")
        if result.projection_agreement is not None:
            add("16_projection_agreement", viz.plot_agreement,
                result.projection_agreement, cfg,
                why="How much the projections agree about who is next to whom.")
        if any(p.stability is not None for p in result.projections.values()):
            add("17_seed_stability", viz.plot_seed_stability, result.projections,
                cfg, why="How much of each layout is the random seed.")
        add("18_embedding_quality", viz.plot_embedding_quality,
            result.embeddings, result.quality, cfg,
            why="Trustworthiness and continuity of each projection.")

    # ---- 3. the self-organising map --------------------------------------
    if som is not None:
        add("20_composition_map", viz.plot_composition_map, som, prep, cfg,
            core=True,
            why="Where each group's samples land on the map, and how mixed each "
                "node is.")
        add("21_group_pies", viz.plot_group_pies, som, prep, cfg,
            why="Node composition as pie glyphs.")
        add("22_group_maps", viz.plot_group_maps, som, prep, cfg,
            why="One map per group, with node opacity graded by occupancy.")
        if len(prep.group_replicate_codes()[1]) > prep.n_groups:
            add("23_group_maps_by_replicate", viz.plot_group_maps, som, prep, cfg,
                by_replicate=True,
                why="The same maps split by replicate, to show whether the "
                    "pattern holds in each.")
        add("24_u_matrix", viz.plot_u_matrix, som, cfg,
            why="Distances between neighbouring nodes: the map's own cluster "
                "boundaries.")
        add("25_hit_map", viz.plot_hit_map, som, prep, cfg,
            why="How many samples each node captured.")

        top = assoc.top_features(12) if assoc else list(som.feature_names[:12])
        add("26_component_planes", viz.plot_component_planes, som, prep, cfg,
            features=top,
            why="Each metric's value across the map, in its original units.")

        if nc is not None and nc.best_k:
            add("27_node_clusters", viz.plot_node_clusters, som, nc, prep, cfg,
                core=True,
                why="The map divided into behavioural clusters.")
            add("28_k_selection", viz.plot_k_selection, nc, cfg,
                why="Why that number of clusters was chosen.")
            if assoc is not None and assoc.node_profile is not None:
                add("29_node_cluster_signature", viz.plot_node_cluster_heatmap,
                    assoc.node_profile, cfg,
                    why="What distinguishes each behavioural cluster, in "
                        "measurement units.")

        if assoc is not None and assoc.gradients is not None:
            add("30_metric_gradients", viz.plot_gradient_summary, som, prep,
                assoc.gradients, cfg,
                why="The direction each metric increases across the map.")
            add("31_gradient_compass", viz.plot_gradient_table, assoc.gradients, cfg,
                why="All metric directions on one polar plot: metrics at the "
                    "same bearing are redundant.")
            if top:
                add("32_gradient_field", viz.plot_gradient_field, som, prep, cfg,
                    top[0], why=f"{top[0]} across the map.")

        if assoc is not None and assoc.enrichment is not None and not assoc.enrichment.empty:
            for g in range(min(prep.n_groups, 4)):
                add(f"33_enrichment_{_slug(prep.group_values[g])}",
                    viz.plot_group_enrichment_map, som, prep, assoc.enrichment, cfg, g,
                    why=f"Where {prep.group_values[g]} sits more often than "
                        f"chance predicts.")

        add("34_training_curves", viz.plot_training_curves, som, cfg,
            why="Training diagnostics for the map.")

    # ---- 4. per-metric description ---------------------------------------
    if assoc is not None:
        add("40_association_overview", viz.plot_association_overview, assoc, prep, cfg,
            core=True,
            why="Per-metric effect size against multivariate importance.")
        add("41_group_metric_heatmap", viz.plot_group_metric_heatmap,
            prep, assoc.univariate, cfg,
            why="Group means per metric, standardised.")
        add("42_metric_distributions", viz.plot_metric_distributions,
            prep, assoc.top_features(8), cfg,
            why="Distribution of the strongest metrics, by group.")
    add("43_metric_correlation", viz.plot_metric_correlation, prep, cfg,
        why="Which metrics are measuring the same thing.")

    result.figure_captions = captions
    if progress:
        progress("Figures complete", 1.0)
    return panels


def _close(fig) -> None:
    from matplotlib import pyplot as plt

    plt.close(fig)


def _slug(s) -> str:
    return "".join(c if (str(c).isalnum() or c in "-_") else "_" for c in str(s))


# --------------------------------------------------------------------------
def export_all(result: AnalysisResult,
               panels: list[tuple[str, "viz.Panel"]] | None = None,
               progress: Progress | None = None) -> ex.ExportManifest:
    cfg = result.config.export
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = ex.ExportManifest(out_dir=out)

    if panels is None:
        panels = build_figures(result, progress)

    if progress:
        progress("Writing figures", 0.4)
    for name, panel in panels:
        ex.save_panel(panel, name, cfg, manifest, close=False)

    if cfg.pdf:
        path = ex.save_multipage_pdf(panels, out / "SOMTrack_report.pdf", close=False)
        manifest.figures["_report_pdf"] = [str(path)]

    if cfg.save_tables:
        if progress:
            progress("Writing tables", 0.7)
        tables = result.tables()
        for name, df in tables.items():
            ex.save_table(df, name, cfg, manifest)
        try:
            ex.save_tables_workbook(tables, out / "SOMTrack_tables.xlsx")
        except Exception as exc:
            result.warnings.append(f"Excel workbook not written: {exc}")

    if cfg.mp4 and result.som is not None:
        if progress:
            progress("Rendering animation", 0.82)
        try:
            path = viz.animate_training(result.som, result.prep, result.config.figure,
                                        cfg, out / "videos" / "som_training")
            manifest.videos.append(str(path))
        except Exception as exc:
            result.warnings.append(f"Training animation skipped: {exc}")

    ex.save_config(result.config, cfg)

    if result.config.report.enabled:
        if progress:
            progress("Writing the report", 0.9)
        try:
            rep = build_report(result)
            rep.figures = [(name, result.figure_captions.get(name, ""))
                           for name, _ in panels]
            for path in write_report(rep, out, result.config.report):
                manifest.tables.append(str(path))
            manifest.tables.append(str(write_methods_text(rep, out / "methods.txt")))
        except Exception as exc:
            result.warnings.append(f"Report not written: {exc}")

    if result.config.figure.cvd_proof and panels:
        if progress:
            progress("Rendering the colour check", 0.95)
        try:
            proofs = out / "figures" / "colour_check"
            proofs.mkdir(parents=True, exist_ok=True)
            for name, panel in panels[:6]:
                proof = viz.palette_proof(panel.fig)
                proof.savefig(proofs / f"{name}_colour_check.png", dpi=110,
                              facecolor="white")
                _close(proof)
        except Exception as exc:
            result.warnings.append(f"Colour check not written: {exc}")

    manifest.write()

    from matplotlib import pyplot as plt

    for _, panel in panels:
        plt.close(panel.fig)

    if progress:
        progress("Export complete", 1.0)
    return manifest


__all__ = [
    "AnalysisResult", "features_from_spots", "run_analysis",
    "build_figures", "export_all",
]
