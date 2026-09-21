"""Central, serialisable configuration objects for SOMTrack.

Every analysis parameter that used to live as a bare global in
``SOM tracking_v1.2b.ijm`` lives here instead, so a whole run can be
saved to / restored from a single JSON file and reported in a methods
section.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal

Topology = Literal["hex", "rect"]
SomAlgorithm = Literal["online", "batch", "supervised", "relevance", "growing"]
ScalerName = Literal["minmax", "zscore", "robust", "rank", "none"]
InitName = Literal["pca", "sample", "random"]
NeighbourFn = Literal["gaussian", "bubble", "mexican_hat"]


# --------------------------------------------------------------------------
# 1. Track -> metric computation
# --------------------------------------------------------------------------
@dataclass
class TrackConfig:
    """Physical calibration and per-track metric options."""

    pixel_size: float = 1.0                 # length units per pixel
    length_unit: str = "um"
    frame_interval: float = 0.1             # time units per frame
    time_unit: str = "s"

    coords_are_calibrated: bool = False     # True -> X/Y already in length_unit
    times_are_calibrated: bool = False      # True -> T already in time_unit

    min_spots: int = 20                     # drop tracks shorter than this
    trim_fraction: float = 0.6              # legacy MD_Decile_Range (1.0 = no trim)
    use_trimmed_stats: bool = True

    halt_speed_threshold: float = 0.0       # length_unit/time_unit; 0 -> auto
    halt_threshold_px: float = 1.5          # legacy fallback when auto is off
    auto_halt_threshold: bool = True        # Otsu-like split of the speed histogram

    drop_edge_margin: float = 0.0           # in pixels; 0 disables edge trimming
    image_width: float | None = None
    image_height: float | None = None

    msd_max_lag_fraction: float = 0.25      # fit MSD over the first 25 % of lags
    smoothing_window: int = 0               # 0 = off; Savitzky-Golay window (odd)

    body_length_column: str | None = None   # enables body-length normalised metrics


# --------------------------------------------------------------------------
# 2. Feature matrix preprocessing
# --------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    scaler: ScalerName = "zscore"
    nan_policy: Literal["drop_sample", "drop_feature", "impute_median"] = "impute_median"
    drop_constant: bool = True
    collinearity_threshold: float = 0.98    # 1.0 disables the pruning step
    winsorise_quantile: float = 0.0         # e.g. 0.01 clips to [1 %, 99 %]
    feature_weights: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# 3. SOM
# --------------------------------------------------------------------------
@dataclass
class SomConfig:
    algorithm: SomAlgorithm = "batch"
    width: int = 0                          # 0 -> heuristic from sample size
    height: int = 0
    topology: Topology = "hex"
    toroidal: bool = False
    init: InitName = "pca"
    neighbour_fn: NeighbourFn = "gaussian"

    epochs: int = 200                       # legacy SOM_rc
    learning_rate: float = 0.5              # legacy SOM_LearnRate0 (was 1.0)
    time_constant: float = 0.25             # fraction of epochs; legacy 0.15
    sigma0: float = 0.0                     # 0 -> max(w, h) / 2
    sigma_final: float = 0.7

    random_state: int | None = 0
    record_every: int = 5                   # legacy SOM_plot_skipS

    # supervised / XY-fused SOM
    label_weight: float = 0.35              # alpha: 0 = unsupervised, 1 = label only

    # relevance-learning SOM
    relevance_lr: float = 0.02
    relevance_floor: float = 0.05

    # growing SOM
    spread_factor: float = 0.5
    max_nodes: int = 400

    def resolved_size(self, n_samples: int) -> tuple[int, int]:
        """Vesanto's 5*sqrt(N) rule with the legacy 4:3 aspect ratio."""
        if self.width > 0 and self.height > 0:
            return self.width, self.height
        import math

        target = max(9.0, 5.0 * math.sqrt(max(n_samples, 1)))
        h = max(3, int(round(math.sqrt(target * 3.0 / 4.0))))
        w = max(4, int(round(target / h)))
        return w, h


# --------------------------------------------------------------------------
# 4. Non-SOM embeddings
# --------------------------------------------------------------------------
@dataclass
class EmbeddingConfig:
    run_pca: bool = True
    run_tsne: bool = True
    run_umap: bool = True

    n_components: int = 2
    random_state: int | None = 0

    # 3.0: methods are named, not enumerated as flags, so a new one needs no
    # change here.  `overrides` maps a method key to explicit parameter values;
    # anything left out is derived from the data by the method's own rule.
    methods: list[str] = field(default_factory=list)      # [] -> registry default
    overrides: dict[str, dict] = field(default_factory=dict)

    # projection quality
    compute_rank_quality: bool = True       # R_NX curve, trustworthiness, continuity
    compute_reliability: bool = True        # per-point scDEED-style reliability
    reliability_null: int = 20              # scrambled embeddings for the null
    stability_seeds: int = 0                # 0 disables the seed-stability check

    # supervised projections are opt-in; see somtrack.analysis.methods.supervised
    allow_supervised: bool = False

    # ---- legacy 2.0 fields, still honoured when loading an old config -----
    tsne_perplexity: float = 0.0            # 0 -> derived from n
    tsne_learning_rate: float | Literal["auto"] = "auto"
    tsne_init: Literal["pca", "random"] = "pca"
    tsne_max_iter: int = 1000

    umap_n_neighbors: int = 0               # 0 -> derived from n
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # ------------------------------------------------------------------
    def resolved_methods(self) -> list[str]:
        """Which projections to run, honouring a 2.0 config if that is all we have."""
        from .analysis import default_enabled, has

        if self.methods:
            return [m for m in self.methods if has(m)]
        legacy = [k for k, on in (("pca", self.run_pca), ("tsne", self.run_tsne),
                                  ("umap", self.run_umap)) if on]
        if legacy != ["pca", "tsne", "umap"]:      # the user changed the old flags
            return [m for m in legacy if has(m)]
        return default_enabled()

    def resolved_overrides(self) -> dict[str, dict]:
        """Explicit overrides, seeded from the legacy per-method fields."""
        out = {k: dict(v) for k, v in self.overrides.items()}
        if self.tsne_perplexity:
            out.setdefault("tsne", {}).setdefault("perplexity", self.tsne_perplexity)
        if self.tsne_max_iter != 1000:
            out.setdefault("tsne", {}).setdefault("max_iter", self.tsne_max_iter)
        if self.tsne_init != "pca":
            out.setdefault("tsne", {}).setdefault("init", self.tsne_init)
        if self.umap_n_neighbors:
            out.setdefault("umap", {}).setdefault("n_neighbors", self.umap_n_neighbors)
        if self.umap_min_dist != 0.1:
            out.setdefault("umap", {}).setdefault("min_dist", self.umap_min_dist)
        if self.umap_metric != "euclidean":
            out.setdefault("umap", {}).setdefault("metric", self.umap_metric)
        return out


# --------------------------------------------------------------------------
# 4b. Statistics: does the grouping hold up?
# --------------------------------------------------------------------------
@dataclass
class StatsConfig:
    """Everything that turns a picture into a testable claim.

    ``unit_of_analysis`` is the setting that matters most and is easiest to get
    wrong.  If several samples came from one dish, clutch or imaging session,
    they are not independent, and leaving this at ``sample`` makes every p value
    here too small.  ``replicate`` blocks the permutations and the
    cross-validation folds by the replicate column instead.
    """

    enabled: bool = True

    unit_of_analysis: Literal["sample", "replicate"] = "sample"
    use_replicate_blocks: bool = True
    distance: Literal["euclidean", "correlation", "cityblock", "cosine"] = "euclidean"

    # omnibus tests
    run_permanova: bool = True
    run_permdisp: bool = True
    run_energy: bool = True
    dispersion_centre: Literal["centroid", "median"] = "centroid"
    n_permutations: int = 999

    # cross-validated classification
    run_classification: bool = True
    classifier: Literal["lda_shrinkage", "svm_linear", "plsda",
                        "logistic", "random_forest"] = "lda_shrinkage"
    cv_scheme: Literal["auto", "stratified", "repeated_stratified",
                       "grouped", "leave_one_replicate_out"] = "auto"
    cv_splits: int = 5
    cv_repeats: int = 5
    classification_permutations: int = 999

    # interpretation and effect sizes
    run_importance: bool = True
    importance_model: Literal["random_forest", "lda_shrinkage",
                              "svm_linear"] = "random_forest"
    importance_repeats: int = 10
    cluster_threshold: float = 0.70         # |r| above this = "the same thing"
    run_effect_sizes: bool = True
    effect_reference_group: int = 0
    bootstrap: int = 5000

    alpha: float = 0.05
    random_state: int | None = 0


# --------------------------------------------------------------------------
# 5. Node clustering (k-means on the codebook, ported from v1.2)
# --------------------------------------------------------------------------
@dataclass
class NodeClusterConfig:
    enabled: bool = True
    method: Literal["kmeans", "ward", "gmm"] = "kmeans"
    k_min: int = 2
    k_max: int = 8
    weight_by_hits: bool = True
    random_state: int | None = 0


# --------------------------------------------------------------------------
# 6. Figures & export
# --------------------------------------------------------------------------
@dataclass
class FigureConfig:
    style: Literal["publication", "presentation"] = "publication"
    base_font_size: float = 8.0
    font_family: str = "Arial"
    dpi_png: int = 600
    figure_width_mm: float = 180.0          # double-column default
    hex_edgecolor: str = "#3a3a3a"
    palette: str = "somtrack"               # colour-blind-safe categorical set
    show_caption: bool = True
    caption_wrap: int = 96

    # ---- colour policy (see somtrack.viz.palette) -------------------------
    # Above `categorical_max` groups, no palette stays distinguishable, so the
    # figure layer stops colouring by group and facets instead.  Raising this
    # does not make more colours readable; it only makes the figure worse.
    categorical_max: int = 8
    sequential_cmap: str = "viridis"        # perceptually uniform
    sequential_print_safe: str = "cividis"  # readable in greyscale and for CVD
    diverging_cmap: str = "RdBu_r"          # always rendered symmetric about 0
    highlight_group: int | None = None      # colour one group, grey the rest
    cvd_proof: bool = False                 # also export CVD / greyscale versions

    # legacy HSB node-composition map
    hsb_saturation_floor: int = 100         # legacy basolevel2
    hsb_brightness_floor: int = 70          # legacy basolevel3
    hsb_hue_rotation: float = 0.0           # legacy rrr = 80/255
    mixture_ignore_fraction: float = 0.02   # legacy mixtureIgnorePcent (now a *fraction*)

    # per-group maps, requirement 7
    group_alpha_by_hits: dict[int, float] = field(
        default_factory=lambda: {0: 0.20, 1: 0.40, 2: 0.60}
    )
    group_alpha_default: float = 1.00

    # metric axis vectors
    axis_min_length: float = 0.8            # legacy LowLimitAxisL, in node units
    axis_max_length_factor: float = 0.5     # fraction of the shorter map side

    point_size: float = 26.0
    jitter_strength: float = 0.20


@dataclass
class ExportConfig:
    out_dir: Path = Path("somtrack_output")
    png: bool = True
    pdf: bool = True                        # vector PDF, text stays selectable
    svg: bool = False
    mp4: bool = True
    mp4_fps: int = 12
    mp4_bitrate: int = 4000
    embed_fonts_as_text: bool = True        # pdf.fonttype = 42 / ps.fonttype = 42
    save_tables: bool = True


# --------------------------------------------------------------------------
# 7. The written report
# --------------------------------------------------------------------------
@dataclass
class ReportConfig:
    """The conclusion report: what happened, what it means, and how to cite it.

    The methods section is assembled from what the run actually did, not from
    what the program can do, so a run that skipped UMAP does not cite it.
    """

    enabled: bool = True
    profile: Literal["core", "full"] = "core"   # how many figures the PDF holds
    recipe: str = "standard"                    # which preset this run started from
    markdown: bool = True
    html: bool = True
    include_methods: bool = True
    include_references: bool = True
    include_figure_index: bool = True
    citation_style: Literal["inline", "numbered"] = "inline"
    title: str = ""                              # blank -> generated
    authors: str = ""
    # A complete translation written after the English report (and after the
    # English methods.txt).  "zh_TW" = Traditional Chinese; "" = English only.
    # Figures and table contents stay in English either way.
    translation: str = "zh_TW"


# --------------------------------------------------------------------------
# Aggregate
# --------------------------------------------------------------------------
@dataclass
class AnalysisConfig:
    track: TrackConfig = field(default_factory=TrackConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    som: SomConfig = field(default_factory=SomConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    node_cluster: NodeClusterConfig = field(default_factory=NodeClusterConfig)
    figure: FigureConfig = field(default_factory=FigureConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    selected_features: list[str] = field(default_factory=list)
    group_labels: dict[str, str] = field(default_factory=dict)   # group id -> display name

    # ---------------- persistence ----------------
    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(_to_plain(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> "AnalysisConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return _from_plain(cls, raw)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    return obj


def _from_plain(cls: type, raw: dict) -> Any:
    """Rebuild a config tree from plain JSON, ignoring keys this version dropped.

    ``from __future__ import annotations`` makes every ``field.type`` a *string*,
    so nested dataclasses cannot be detected from the annotation.  They are
    rebuilt from the default instance's type instead, after construction.
    Unknown keys are skipped rather than raising, so a config written by a newer
    build still loads here, and one written by an older build still loads there.
    """
    known = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for name in known & set(raw):
        val = raw[name]
        if name == "out_dir":
            kwargs[name] = Path(val)
        elif name == "group_alpha_by_hits" and isinstance(val, dict):
            kwargs[name] = {int(k): float(v) for k, v in val.items()}
        elif isinstance(val, dict) and name in _NESTED_DICT_FIELDS:
            kwargs[name] = val
        elif isinstance(val, dict):
            continue                       # a nested dataclass; handled below
        else:
            kwargs[name] = val

    obj = cls(**kwargs)
    for f in fields(cls):
        cur = getattr(obj, f.name)
        if is_dataclass(cur) and isinstance(raw.get(f.name), dict):
            setattr(obj, f.name, _from_plain(type(cur), raw[f.name]))
    return obj


# plain dicts that must be kept as dicts rather than treated as nested configs
_NESTED_DICT_FIELDS = {"feature_weights", "group_labels", "group_alpha_by_hits",
                       "overrides"}


__all__ = [
    "AnalysisConfig",
    "TrackConfig",
    "PreprocessConfig",
    "SomConfig",
    "EmbeddingConfig",
    "StatsConfig",
    "NodeClusterConfig",
    "FigureConfig",
    "ExportConfig",
    "ReportConfig",
]
