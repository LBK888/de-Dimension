# SOMTrack 3.0

Locomotion metrics, multivariate analysis and publication figures for tracking
data — with the statistics that decide what the figures are allowed to claim.

A Python refactor of `SOM tracking_v1.2b.ijm` (Wang, Ho & Liao, NTOU 2020). The
ImageJ macro computed ~24 locomotion metrics from TrackMate output, clustered
them with a Kohonen map and drew the result. SOMTrack keeps that workflow,
fixes the arithmetic errors documented in [`AUDIT.md`](AUDIT.md), and adds 50
more metrics, five SOM variants, a registry of eleven projections, and a
statistics layer that answers the question the pictures cannot.

**3.0 exists because a picture is not evidence.** Version 2.0 could show you
that your groups looked different; it had no way to tell you whether they
*were*. This release adds the tests that answer that — PERMANOVA and PERMDISP,
the energy test, cross-validated classification against a permutation null —
and it refuses to let a supervised projection be mistaken for proof. It also
writes the answer down: every run produces a conclusion report with a methods
paragraph and a reference list for every algorithm it actually used.

### Installing

**Windows, no Python experience needed.** Double-click **`install.bat`**, then
**`start.bat`**. The installer checks for Python 3.10+, builds a private
environment in `.venv\`, installs everything into it and reports which
projections ended up available. Nothing is installed system-wide, so it cannot
disturb another Python project, and deleting `.venv\` undoes all of it. Running
it again is safe and fast.

`start.bat` also works as a command line — `start.bat methods`,
`start.bat run data.csv --features-table -o results`.

**Anywhere else:**

```bash
pip install -r requirements.txt
python -m somtrack            # desktop app
```

### New in 3.0

| | |
|---|---|
| **Does my grouping hold up?** | PERMANOVA, PERMDISP and the energy *k*-sample test, plus cross-validated classification with a permutation null. |
| **Replicates finally count** | Permutations and cross-validation folds are blocked by the replicate column, so a dish effect can no longer masquerade as a treatment effect. |
| **Eleven projections, one registry** | PCA, MDS/PCoA, t-SNE, UMAP, densMAP, PaCMAP, PHATE, Isomap, kernel PCA, plus LDA, PLS-DA, SVM and supervised UMAP behind a guard rail. |
| **Projections you can audit** | R<sub>NX</sub> curves, per-point reliability against a scrambled null, seed stability, and an agreement matrix between methods. |
| **Parameter scans that show the plateau** | For any method, scored by any criterion, with the actual layouts kept and drawn. |
| **A Compare step** | Every method side by side with its quality score, the agreement matrix, and a scan you can run from the app and adopt with one click. |
| **A conclusion report** | `RESULTS_REPORT.html`, `RESULTS_REPORT.md` and `methods.txt` — a plain-language verdict, a methods paragraph and a reference list, generated from what the run did. |
| **A colour policy, not a palette** | Above eight groups the figure layer stops using colour and facets instead; the rainbow map is gone. |
| **One question instead of twenty-one settings** | The analysis page opens on a recipe picker; everything else is folded away. |
| **繁體中文** | The desktop app in Traditional Chinese (Taiwan), and a report written in English followed by a complete Chinese translation. See [Language](#language--語言). |

Everything from 2.0 still runs, and a 2.0 `analysis_config.json` still loads.

---

## Two ways in

**A. Coordinate tables.** Point SOMTrack at one or more spot-level tables
(TrackMate "Spots in tracks statistics", or anything with track / x / y / time
columns) and pick which metrics to compute. Column names are auto-detected.

**B. A multi-dimensional table.** If you already have one row per animal with
your own measurements, load it and go straight to clustering.

Either way you land on the same feature matrix, and everything downstream is
identical.

---

## What it computes

77 metrics across nine families. Every one carries a unit, a one-line
description and its input requirements; the UI hides metrics whose input
columns are missing from your table.

| Family | Examples |
|---|---|
| Basic kinematics | mean/SD speed, acceleration, path length |
| Angular / directional | turning rate, meander, **circular** mean heading, resultant length R, Rayleigh p, turn entropy, turn kurtosis, left/right bias |
| Path geometry | straightness, sinuosity, radius of gyration, convex hull area, exploration ratio, Katz fractal dimension |
| Diffusion | MSD anomalous exponent α, generalised diffusion coefficient, fit R² |
| Persistence & memory | directional correlation time, persistence length, speed Hurst exponent (DFA) |
| Intermittency (CTRW) | fraction moving, stop frequency, run/pause durations, burstiness B, memory coefficient |
| Speed distribution | peak speed, CV, skewness, RMS acceleration, RMS jerk |
| Oscillation & biophysics | beat frequency and spectral purity, lateral amplitude, **Strouhal number**, **Reynolds number**, speed in body lengths |
| Morphology / intensity | area, ellipse axes, aspect ratio and its CV, **axial** body orientation, yaw drift, intensity, focus index |

Metrics that existed in v1.2 are tagged `(v1.2)` in the UI, and the
**v1.2 legacy set** preset reproduces exactly that list for direct comparison.

`python -m somtrack metrics` prints the full catalogue.

---

## Clustering and projection

### Self-organising maps

| Algorithm | When to use it | Cite |
|---|---|---|
| **batch** (default) | Vectorised, deterministic, ~100× faster than the sequential rule. The standard choice. | Kohonen 1999 |
| **supervised** | XY-fused SOM. Carries the treatment label as a second data block with weight α. **Separates your groups by construction** — report α, and take the evidence from the cross-validated statistics, not from the map. | Melssen, Wehrens & Buydens 2006 |
| **relevance** | Batch SOM plus GRLVQ relevance learning. Learns a weight per metric, so a handful of informative metrics stop being drowned out by twenty uninformative ones. | Hammer & Villmann 2002 |
| **online** | The sequential Kohonen rule the macro used, kept for reproducibility. | Kohonen 1982 |
| **growing** | GSOM on a hex lattice. Grows nodes where quantisation error is high, so map size stops being a hyper-parameter you have to guess. | Alahakoon, Halgamuge & Srinivasan 2000 |

Also: hexagonal or rectangular lattice, optional toroidal wrap, PCA
initialisation, and quality as quantisation error, topographic error
(Kiviluoto 1996), hit-weighted group purity and normalised mutual information.

### The projection registry

`python -m somtrack methods` lists every method with its parameters, its
plain-language summary and its citation. Adding one is a single file plus a
`register()` call — the configuration, the desktop form, the parameter scan,
the comparison figures and the reference list all pick it up from there.

| Method | What it is for | Cite |
|---|---|---|
| **PCA** | Linear, deterministic, directly interpretable. Start here. | Hotelling 1933 |
| **MDS / PCoA** | The only projection whose distances are the ones PERMANOVA tests, so the picture and the statistics describe the same thing. | Torgerson 1952; Gower 1966 |
| **t-SNE** | Sharpens local neighbourhoods. Cluster sizes and gaps mean nothing. | van der Maaten & Hinton 2008 |
| **UMAP** | Between PCA and t-SNE. | McInnes, Healy & Melville 2018 |
| **densMAP** | UMAP that also preserves local density, so "this group is more variable" stays visible. | Narayan, Berger & Cho 2021 |
| **PaCMAP** | Balances local detail and global shape better than t-SNE or UMAP, and barely needs tuning. | Wang, Huang, Rudin & Shaposhnik 2021 |
| **PHATE** | For biology that is a gradient — dose, time, development — rather than separate clusters. | Moon et al. 2019 |
| **Isomap**, **kernel PCA** | Curved structure. | Tenenbaum, de Silva & Langford 2000; Schölkopf, Smola & Müller 1998 |
| **LDA (shrinkage)** | The classical discriminant, regularised so it stays stable when the metric count approaches the sample count. | Fisher 1936; Ledoit & Wolf 2004 |
| **PLS-DA** | The discriminant projection of the omics literature. | Barker & Rayens 2003 |
| **Linear SVM** | The widest gap between groups, with interpretable weights. | Cortes & Vapnik 1995 |
| **Supervised UMAP** | For illustrating a structure you have already demonstrated. | McInnes et al. 2018 |

Defaults are derived from *n*, not copied from the papers: the published
perplexity and neighbour counts were chosen for tens of thousands of points, and
applying them to two hundred animals draws a picture of the algorithm rather
than of the experiment. The value actually used is recorded in the methods
section.

> #### ⚠ The guard rail on supervised projections
>
> A supervised projection is given the answer before it draws the picture, so it
> separates the groups — on real data, and equally on noise. This is documented
> independently for between-group PCA (Cardini, O'Higgins & Rohlf 2019), for
> PLS-DA, where score-plot separation was shown to carry no information because
> random data produces the same plot (Westerhuis et al. 2008), and for supervised
> UMAP, whose issue tracker is full of embeddings that separate in training and
> collapse on held-out data.
>
> SOMTrack therefore keeps these methods **off by default**, requires an explicit
> acknowledgement, prints the caveat on every panel, and draws each one twice:
> fitted on everything, and **out-of-fold**, with each sample placed by axes
> fitted without it. In-sample separation with out-of-fold collapse is the honest
> picture of "no". See Cardini & Polly (2020) for the cross-validated form.

### How much of a projection to believe

Every projection is reported with:

* **R<sub>NX</sub>(K) and its area on a log-K axis** from the co-ranking matrix
  (Lee & Verleysen 2009; Lee et al. 2013) — one number, comparable across
  methods, that subsumes trustworthiness and continuity (Venna & Kaski 2006);
* **per-point reliability** against a null built by scrambling the embedding, in
  the style of scDEED (Xia, Lee & Li 2024), so a cluster made of unreliably
  placed points can be recognised as an artefact. Minimising the number of
  dubious points is also available as a scan criterion, which turns "what
  perplexity should I use" into a question with a figure;
* **seed stability**, by re-running and Procrustes-aligning (Gower 1975);
* an **agreement matrix** between methods, because structure that survives
  several projections is a property of the data and structure visible in one is
  a property of that algorithm.

### Parameter scans

```bash
somtrack scan features.csv --features-table --method tsne --criterion dubious_fraction
```

Works for the SOM and for any registered projection, scored by
`rnx_auc`, `dubious_fraction`, `trustworthiness` or `group_silhouette`. The scan
keeps every layout it produced and reports the **plateau** — every setting
within tolerance of the best — because the single best cell is almost never
meaningfully better than its neighbours, and saying so is what stops a user
tuning until the picture agrees with their hypothesis.

### Second-level clustering

k-means / Ward / GMM over the codebook (Vesanto & Alhoniemi 2000), scanned over
a k range and scored by silhouette (Rousseeuw 1987), Davies-Bouldin (1979) and
Calinski-Harabasz (1974). The chosen k is stated on the figure with its scores.

---

## Does the grouping hold up?

This is the layer 2.0 did not have. Three questions, in the order a reader asks
them.

### 1. Do the groups differ at all?

| Test | Asks | Cite |
|---|---|---|
| **PERMANOVA** | Do the group averages differ? Reports pseudo-F, a permutation p, and **R² as the effect size** — the share of multivariate variation the grouping accounts for. | Anderson 2001 |
| **PERMDISP** | Do the groups differ in *spread*? | Anderson 2006 |
| **Energy k-sample test** | Do the distributions differ *at all*, in location, scale or shape? | Székely & Rizzo 2013; Gretton et al. 2012 |
| **Mahalanobis D** | How far apart are the centroids, with the pooled covariance shrunk before inversion? | Mahalanobis 1936; Ledoit & Wolf 2004 |

**PERMANOVA and PERMDISP are always reported together**, because PERMANOVA
rejects for either reason. A significant PERMANOVA with a significant PERMDISP
may mean *"these animals are more variable"*, not *"these animals are faster"* —
a different biological result, and one that has been published as the first many
times. SOMTrack says which it is, in the verdict, in words.

### 2. Can a held-out sample be assigned to its group?

Cross-validated classification — LDA with shrinkage, linear SVM, PLS-DA,
logistic regression or random forest — scored by **balanced accuracy**
(Brodersen et al. 2010) and Cohen's κ (Cohen 1960), with a bootstrap confidence
interval and a **permutation null** (Ojala & Garriga 2010). Not a binomial test
on the cross-validated accuracy: that test is anti-conservative and reports
differences that are not there (Noirhomme et al. 2014).

The **confusion matrix and the per-pair accuracies** are reported next to the
single number, because *"A separates from everything, B and C do not separate
from each other"* is the biologically useful statement and a scalar cannot
express it.

### 3. Which metrics carry it?

* **Haufe-transformed activation patterns** (Haufe et al. 2014). A large
  classifier weight can belong to a metric carrying *no* group information,
  present only to cancel correlated noise in another — a suppressor variable.
  Reading such a weight as "this metric distinguishes the groups" is a
  well-documented way to publish a wrong conclusion. SOMTrack reports the raw
  weights and the activation patterns side by side, and says which column to
  read.
* **Permutation importance, per metric and per correlated cluster**
  (Breiman 2001; Altmann et al. 2010). Turning rate, meander and straightness
  measure much the same thing, so shuffling them one at a time understates all
  three — the model simply reads the others. Whole clusters are shuffled too.
* **Effect sizes in original units**: Hedges' *g* with the small-sample
  correction (Hedges 1981) and bias-corrected accelerated bootstrap intervals
  (Efron 1987), presented as estimation plots (Ho et al. 2019).
* The per-metric ANOVA / Kruskal-Wallis layer from 2.0, with Benjamini-Hochberg
  FDR correction (Benjamini & Hochberg 1995), is unchanged.

### ⚠ Replicates, and why your p values were too small

`PreparedData` has carried a replicate column since 2.0 and did nothing
statistical with it. It does now.

If eight animals came out of one dish, they are eight measurements of **one**
experimental unit, not eight replicates (Hurlbert 1984; Lazic, Clarke-Williams &
Munafò 2018). SOMTrack detects the structure and adapts:

* **nested** (each replicate belongs to one group): permutations reassign whole
  replicates, and cross-validation keeps a replicate entirely inside one fold;
* **crossed** (each replicate contains several groups): labels are permuted
  *within* each block, which is what preserves the blocking
  (Anderson & ter Braak 2003);
* `--unit replicate` goes further and averages within each replicate before
  testing — the most conservative option.

The report states which design was found and what was done about it. On
synthetic data with a pure dish effect and no treatment effect, ignoring this
gives *p* < 0.01 and a balanced accuracy above 0.85; respecting it gives neither.
There is a test that pins exactly that.

---

## The conclusion report

Every run writes `RESULTS_REPORT.html`, `RESULTS_REPORT.md` and `methods.txt`.

The **verdict** is generated from the numbers, so it cannot disagree with the
tables underneath it:

> Strong evidence that the groups differ. PERMANOVA on euclidean distances:
> pseudo-F = 4.84, R2 = 0.085 …, p = 0.0050, 199 permutations. PERMDISP
> (equality of within-group spread): F = 2.74, p = 0.055. The groups differ in
> their multivariate average, and their within-group variability is comparable,
> so this is a genuine shift in phenotype rather than a change in variability.
> …
> Group pairs that separate: control vs high (balanced accuracy 0.96); …
> Group pairs that do not separate: low vs mid (balanced accuracy 0.59).

The **methods paragraph** describes what this run did, assembled as each step
recorded itself — a run that skipped UMAP does not cite McInnes et al.:

> **Statistics.** replicate-aware permutation and cross-validation, Replicates
> were nested within groups (20 replicates, each belonging to one group)
> (Anderson & ter Braak, 2003; Hurlbert, 1984; Lazic et al., 2018); PERMANOVA,
> permutational multivariate analysis of variance on euclidean distances, with
> R-squared as the effect size (Anderson, 2001) [permutations = 999, distance =
> euclidean]; …
>
> **Visualization.** SuperPlots, coloured by biological replicate (Lord et al.,
> 2020); estimation plots with bootstrap confidence intervals (Ho et al., 2019).

followed by a de-duplicated **reference list with DOIs**, and by the caveats
attached to whatever was run.

After the English report comes a **complete Traditional Chinese translation**
of it -- in all three files -- so the two languages are never mixed on one line.
See [Language](#language--語言) for what is and is not translated.

---

## Figures

Around 40 figures are available; a run writes about a dozen by default. They are
ordered the way the report reads — the conclusion first, then the evidence for
it, then the descriptive figures — because a reader who stops after three pages
should have the answer, not a self-organising map they have to interpret first.

`report.profile = "core"` writes the ones that carry the result;
`"full"` writes everything. A thirty-page PDF is read by nobody.

### The figures that carry the claim

| | |
|---|---|
| **Verdict card** | The conclusion in plain language, generated from the numbers. Page one. |
| **Separation summary** | PERMANOVA's R² next to PERMDISP, with the within-group spread drawn, so a dispersion effect cannot be read as a shift. |
| **Pairwise separation** | Which specific pairs differ, FDR-corrected. |
| **Classifiability** | Observed balanced accuracy against the permutation null, with the confusion matrix. |
| **Activation patterns** | Haufe-transformed patterns beside the raw weights, with a note saying which to read. |
| **Importance** | Per metric and per correlated cluster. |
| **Effect-size forest** | Hedges' *g* with bootstrap intervals (Cohen 1988 landmarks drawn, as conventions not thresholds). |
| **SuperPlots** | Every individual coloured by replicate, with each replicate's mean overlaid (Lord et al. 2020). |
| **Estimation plots** | Raw data above, effect size with its bootstrap interval on an aligned axis below (Ho et al. 2019). |
| **In-sample vs out-of-fold** | The panel that separates a real difference from one a supervised method was handed. |

### The figures that audit the pictures

Method-comparison grid (every projection with its quality badge), R<sub>NX</sub>
curves, per-point reliability, seed stability, projection agreement, Shepard
diagram, scan surface with its plateau marked, and a grid of the actual layouts
at every scanned setting.

### Ported from v1.2, with the arithmetic corrected

The HSB composition map, the per-group maps, the metric axis vectors, the
k-means node overlay, the sample placement toward the runner-up node. Plus the
U-matrix, hit histogram, component planes in original units, pie-glyph node
composition, training-diagnostic curves, labelled kymograph and enrichment maps.

### Colour

Colour is a decision, not a palette. `viz/palette.py` returns colours **and an
instruction**, and the figure layer honours it:

* **≤ 8 groups** — Okabe-Ito, the colour-vision-deficiency-safe standard
  (Okabe & Ito 2008; Wong 2011).
* **≤ 10** — Paul Tol's muted set.
* **more than that** — the answer is *not colour*. Asked for more categories
  than any palette can keep apart, `categorical()` returns `mode="facet"` and
  the figure draws small multiples instead of inventing thirty colours nobody
  can match to a legend. A focus group can also be coloured against the rest in
  grey.
* **Categories never come from a continuous map.** Sampling *viridis* or *turbo*
  at *n* points implies an ordering the categories do not have, and rainbow maps
  invent boundaries that are not in the data (Crameri, Shephard & Heron 2020;
  Crameri 2024). The `turbo` fallback that 2.0 used above twelve groups is gone,
  and the metric-gradient figures now use a sequential map because rank *is*
  ordinal.
* **Sequential maps are perceptually uniform**; the print-safe default,
  *cividis*, also survives greyscale and deuteranopia.
* **Diverging maps only where zero means something**, always symmetric about it,
  so the colour of a value does not depend on the range that happened to be
  present. `symmetric_norm()` enforces it.
* **Grey is reserved for context** and is never an experimental group.
* Group identity is encoded **redundantly** — colour *and* marker shape.
* `figure.cvd_proof = True` exports each leading figure under deuteranopia,
  protanopia, tritanopia and greyscale, so a figure can be checked before
  submission rather than after review.

Node colours on the SOM are derived from the same palette as the legend swatches
at any *n*, which they were not in 2.0 once the fallback kicked in.

### Per-group transparency (brief item 7)

On the per-group maps, node opacity is graded by how many samples of *that
group* the node holds:

| samples in node | opacity |
|---|---|
| 0 | 20 % |
| 1 | 40 % |
| 2 | 60 % |
| ≥ 3 | 100 % |

so a group's territory is the only saturated region on a large map. The
thresholds are editable in the Export page.

### Output formats

* **PDF and SVG** — vector, with text kept as *text* (`pdf.fonttype = 42`,
  `svg.fonttype = "none"`), so every label can be edited in Illustrator or
  Inkscape without re-running anything. `SOMTrack_report.pdf` holds the whole
  set in order.
* **PNG** — 600 dpi by default.
* **MP4** — the map self-organising over training, with a quantisation-error
  trace. Falls back to GIF if ffmpeg is not on PATH.
* **CSV + one `.xlsx` workbook** — every result table, including the statistics.
* **`RESULTS_REPORT.html` / `.md`** — the conclusion, the evidence, the methods
  paragraph and the reference list; in English, then translated into
  Traditional Chinese.
* **`methods.txt`** — the methods paragraph and references on their own, ready
  to paste, followed by the Chinese methods paragraph.
* **`analysis_config.json`** — reload it to reproduce the run exactly.

---

## Command line

```bash
# synthetic four-treatment dataset to try things on
python -m somtrack demo ./demo_data

# full analysis, with the statistics layer
python -m somtrack run "demo_data/*.csv" \
    --pixel-size 1.6 --frame-interval 0.05 -o results

# a table you already have, one row per animal
python -m somtrack run features.csv --features-table -o results

# choose the projections, and treat the replicate as the experimental unit
python -m somtrack run features.csv --features-table \
    --projections pca,mds,pacmap --unit replicate --permutations 9999 -o results

# also run the supervised projections, with their guard rail
python -m somtrack run features.csv --features-table --allow-supervised -o results

# scan a projection's hyper-parameters, scored by unreliable-point count
python -m somtrack scan features.csv --features-table \
    --method tsne --criterion dubious_fraction -o scan

# scan the SOM, as in 2.0
python -m somtrack scan "demo_data/*.csv" --pixel-size 1.6 -o scan

# what is available
python -m somtrack methods      # projections, parameters and citations
python -m somtrack metrics      # the metric catalogue
```

## As a library

```python
from somtrack import AnalysisConfig, pipeline, recipes
from somtrack.io_tables import build_spot_dataset, detect_spot_columns, load_table

cfg = recipes.apply_recipe(AnalysisConfig(), "test_groups")
cfg.track.pixel_size = 1.6
cfg.track.frame_interval = 0.05
cfg.stats.unit_of_analysis = "replicate"
cfg.export.out_dir = "results"

cm = detect_spot_columns(load_table(files[0]))
spots = build_spot_dataset(files, cm, groups, replicates)
feats = pipeline.features_from_spots(spots, cfg)
res = pipeline.run_analysis(feats, cfg)

print(res.verdict.text())                       # the conclusion, in words
print(res.separation.permanova_R2, res.separation.permanova_p)
print(res.classification.summary_line())
pipeline.export_all(res)                        # figures, tables and the report
```

The statistics and the projections are usable on their own:

```python
import numpy as np
from somtrack.analysis import DataContext, run_projections
from somtrack.analysis.scan import scan
from somtrack.stats import analyse_blocks, analyse_separation, classify

structure = analyse_blocks(group_codes, replicates)   # nested? crossed? neither?
sep = analyse_separation(X, group_codes, labels, structure, n_permutations=999)
cls = classify(X, group_codes, labels, structure, model="svm_linear")

ctx = DataContext(X=X, feature_names=names, group_codes=group_codes,
                  group_values=labels, replicates=replicates)
projections = run_projections(ctx, ["pca", "pacmap", "mds"])
best = scan(ctx, "tsne", criterion="dubious_fraction")
print(best.plateau_text())
```

---

## Layout

```
somtrack/
  config.py       every parameter, as one serialisable dataclass tree
  citations.py    the bibliography, and the per-run methods log
  recipes.py      whole analyses named after the question they answer
  report.py       the conclusion report: verdict, methods, references, and its translation
  i18n.py         interface language, and sentences that carry their own translation
  locales/        translation catalogues (zh_TW.py: Traditional Chinese)
  io_tables.py    loading, column auto-detection, dataset assembly
  metrics.py      the metric registry and the per-track kinematics engine
  preprocess.py   scaling, NaN policy, collinearity pruning
  som.py          lattice, five SOM variants, quality indices, parallel scan
  gsom.py         growing SOM on a hex lattice
  nodecluster.py  second-level clustering with k selection
  association.py  which metrics explain the grouping (per-metric layer)
  analysis/
    registry.py     MethodSpec / ParamSpec: methods declare themselves
    projection.py   the common result type for every 2-D projection
    quality.py      co-ranking, R_NX, per-point reliability, seed stability
    scan.py         parameter scans for any method, with the plateau
    methods/        linear.py, manifold.py, supervised.py
  stats/
    blocks.py       what "independent" means for this experiment
    omnibus.py      PERMANOVA, PERMDISP, energy test, Mahalanobis
    classify.py     cross-validation with a design-aware permutation null
    interpret.py    Haufe patterns, clustered permutation importance
    effects.py      Hedges' g with bootstrap intervals
    verdict.py      the paragraph at the top of the report
  viz/
    palette.py      the colour policy, including when to stop using colour
    stats.py        verdict card, separation, classifiability, activations
    compare.py      method grid, scan surface, scan thumbnails, agreement
    reliability.py  R_NX curves, per-point reliability, stability, Shepard
    estimation.py   SuperPlots, estimation plots, effect-size forest
    (maps, groups, vectors, embed, assoc, anim, style, hexgeom)
  export.py       PNG / PDF / SVG / MP4 / CSV / XLSX writers
  pipeline.py     orchestration shared by the GUI and the CLI
  ui/
    main_window.py  the seven-step wizard
    pages.py        one class per step, including Results and Compare
    forms.py        parameter controls built from ParamSpec, with Auto boxes
    widgets.py      file list, metric tree, figure gallery, log pane
  cli.py          command line
  demo.py         synthetic data generator
install.bat       Windows: check Python, build .venv, install dependencies
start.bat         Windows: open the app, or pass arguments to the CLI
tests/            statistics tested against data with known answers, plus the
                  registry, the report, the desktop wiring and the translation
```

The GUI and the CLI both drive `pipeline.py`, so a run launched either way
produces identical output for the same config.

---

## Using it without a statistics background

The analysis page opens on one question — *what are you trying to find out* —
and a Run button. Everything else is folded away and stays folded until it is
asked for.

| Recipe | For |
|---|---|
| **Explore the data** | The safe default. Map it, project it, test it. |
| **Test whether my groups differ** | The full statistics layer, supervised projections deliberately off. |
| **Find which measurements matter** | Relevance SOM, activation patterns, clustered importance, effect sizes. |
| **Check that the picture is real** | Six projections, several seeds, reliability and agreement. |
| **Look for a dose or time gradient** | Adds PHATE, for biology that is a series rather than classes. |
| **Reproduce the v1.2 macro** | The original workflow, for comparison with old results. |

A recipe is data, not a code path — an ordinary `AnalysisConfig` with some
fields changed — so it loads through the same JSON machinery and every parameter
it set is still visible and still editable underneath.

After the run, the **Compare** step puts every projection on one page with its
quality score, so the one that separates best cannot quietly become the one you
show. From the same page you can scan a method's hyper-parameters, see the
plateau, and press *Use these settings* — which pins them as an explicit choice
that the methods section then records as yours.

Underneath, parameters that can be derived from the data have an **Auto** box,
ticked by default, that *shows you the value it picked*. Controls that do not
apply are hidden rather than greyed out: an inapplicable control is still
something the eye has to read and dismiss.

Results open on the conclusion, not on a gallery.

---

## Language / 語言

**The desktop app** is available in English and Traditional Chinese (Taiwan).
It opens in the language chosen from the **Language / 語言** menu; before
anything has been chosen, in the language named by the `SOMTRACK_LANG`
environment variable (`zh_TW` or `en`); failing that, in the system's language
-- so a Taiwanese Windows installation opens in Chinese the first time. Every
label, hint, tooltip and log message is translated, and so is the text that
comes from the registries: method summaries and caveats, parameter names and
their help, the metric catalogue and the recipes. Option lists show a
translated label with the configuration value in parentheses, e.g.
`z 分數標準化（zscore）`, so what you read can be matched to what the methods
section and `analysis_config.json` record. Metric names in the metric tree stay
as they are, because they are the column names of every output table; the
Chinese metric name leads the description beside them.

**The report** (`RESULTS_REPORT.html`, `RESULTS_REPORT.md`, `methods.txt`) is
written in English first and then again in Traditional Chinese, as a second
complete document rather than line-by-line pairs. The two are never mixed.

| Translated | Kept in English |
|---|---|
| headings and paragraphs | everything inside the figures |
| the verdict and its caveats | table contents and column names |
| table captions and the figure index (圖說) | metric, group and parameter names |
| the methods paragraph and its caveats | the reference list (given once, in the English half) |

Numbers are produced once and rendered twice, so every value in the
translation is the value in the English report. The translation can be
switched off on the Export page, with `--translation none` on the command line,
or with `"translation": ""` in the `report` section of `analysis_config.json`.

Adding a language is one file: `somtrack/locales/<code>.py` holding
`MESSAGES` (English source string -> translation) and `CONTEXTS`. Interface
text goes through `i18n.tr()`; generated sentences are `i18n.Text` objects --
the English string itself, able to render its translation later -- which is
how one run can write both halves of the report. `tests/test_i18n.py` fails if
a translation drops a `{placeholder}`, if a Chinese run meets a string with no
translation, or if Chinese reaches a figure.

---

## Notes

* **Optional projections.** `umap-learn` enables UMAP, supervised UMAP and
  densMAP; `pacmap` enables PaCMAP; `phate` enables PHATE. Any that are missing
  are skipped with an explanation that reaches the report, rather than a
  silently shorter figure set. `python -m somtrack methods` shows what is
  installed.
* **Optional colour.** `glasbey` improves categorical palettes above ten groups;
  `cmcrameri` adds Crameri's scientific colour maps. Both have fallbacks.
* MP4 export needs `ffmpeg` on PATH.
* `shapely` is optional; it only smooths the merged outlines of node clusters.
* **Performance.** Sized for the regime this program is built for — under ~1000
  samples and ~50 metrics — where O(n²) methods are the right trade. On 160
  samples × 24 metrics with 999 permutations a full run takes about a minute.
  Permutation nulls use linear models by default because a random forest inside
  a 999-iteration loop is not worth the wait.
* **Reproducibility.** `analysis_config.json` reproduces a run exactly, and a
  2.0 config still loads — including its `run_pca` / `run_tsne` / `run_umap`
  flags, which are translated into the new method list.
* Read [`AUDIT.md`](AUDIT.md) before comparing new numbers against v1.2 output —
  several v1.2 statistics were wrong in ways that can change conclusions,
  particularly the angle averages and the NaN handling.
* **Still to come.** PCC (arXiv:2503.07609) is the one method from the 3.0 plan
  that is not here: it has no published package yet, and reimplementing a 2025
  paper from its abstract is not something to put behind a figure a reviewer
  will read. It is a one-file addition once a package appears.
  SLISEMAP is registered but needs `pip install slisemap`, which pulls in
  PyTorch; it is off the default install for that reason.
* [`ROADMAP.md`](ROADMAP.md) has the design reasoning behind 3.0.

---

## References

Every algorithm SOMTrack can run is in `somtrack/citations.py` with its DOI, and
the reference list of any given run is generated from what that run actually
did. The works the 3.0 design rests on:

**Group separation.**
Anderson (2001) *Austral Ecology* 26:32 — PERMANOVA ·
Anderson (2006) *Biometrics* 62:245 — PERMDISP ·
Anderson & ter Braak (2003) *J Stat Comput Simul* 73:85 — restricted permutation ·
Székely & Rizzo (2013) *J Stat Plan Inference* 143:1249 — energy statistics ·
Gretton et al. (2012) *JMLR* 13:723 — kernel two-sample test

**Not fooling yourself.**
Cardini, O'Higgins & Rohlf (2019) *Evol Biol* 46:303 — spurious group separation ·
Cardini & Polly (2020) *Evol Biol* 47:85 — cross-validated bgPCA ·
Westerhuis et al. (2008) *Metabolomics* 4:81 — PLS-DA validation ·
Ojala & Garriga (2010) *JMLR* 11:1833 — permutation tests for classifiers ·
Noirhomme et al. (2014) *NeuroImage Clin* 4:687 — the binomial test is wrong ·
Hurlbert (1984) *Ecol Monogr* 54:187 and Lazic, Clarke-Williams & Munafò (2018)
*PLoS Biol* 16:e2005282 — pseudoreplication ·
Haufe et al. (2014) *NeuroImage* 87:96 — interpreting classifier weights

**Projections and their quality.**
Wang, Huang, Rudin & Shaposhnik (2021) *JMLR* 22:1 — PaCMAP ·
McInnes, Healy & Melville (2018) arXiv:1802.03426 — UMAP ·
Narayan, Berger & Cho (2021) *Nat Biotechnol* 39:765 — densMAP ·
Moon et al. (2019) *Nat Biotechnol* 37:1482 — PHATE ·
Lee & Verleysen (2009) *Neurocomputing* 72:1431 — co-ranking matrix ·
Xia, Lee & Li (2024) *Nat Commun* 15:1753 — scDEED ·
Gower (1975) *Psychometrika* 40:33 — Procrustes

**Figures and colour.**
Lord, Velle, Mullins & Fritz-Laylin (2020) *J Cell Biol* 219:e202001064 — SuperPlots ·
Ho, Tumkaya, Aryal, Choi & Claridge-Chang (2019) *Nat Methods* 16:565 — estimation plots ·
Crameri, Shephard & Heron (2020) *Nat Commun* 11:5444 — the misuse of colour ·
Crameri (2024) *Curr Protoc* 4:e1126 — choosing palettes ·
Okabe & Ito (2008) and Wong (2011) *Nat Methods* 8:441 — colour-blind-safe sets ·
Glasbey, van der Heijden, Toh & Gray (2007) *Color Res Appl* 32:304

**Effect sizes.**
Hedges (1981) *J Educ Stat* 6:107 ·
Efron (1987) *JASA* 82:171 — BCa bootstrap ·
Brodersen et al. (2010) *ICPR* — balanced accuracy ·
Benjamini & Hochberg (1995) *JRSS B* 57:289
