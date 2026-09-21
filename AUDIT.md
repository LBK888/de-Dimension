# Scientific and numerical audit of `SOM tracking_v1.2b.ijm`

Requirement 3 of the refactor brief was to port the existing visualisations
**only after confirming they are sound**. This document is that confirmation:
what was carried over unchanged, what was carried over after correcting the
arithmetic, and what was replaced because the underlying statistic measured the
wrong thing.

Line numbers refer to `SOM tracking_v1.2b.ijm` (4,034 lines, last updated
2020-08-18).

Two things this audit does **not** cover, stated up front:

* The threshold / particle-analysis step (`ThresholdNParticle`, lines 2343-2531)
  was not verified against real images. Per the scope decision for this
  refactor, morphology and intensity metrics are now read from the input table
  rather than recomputed, so that code has no counterpart in SOMTrack.
* No original experimental data was available, so nothing here is a claim about
  how large the numerical difference is on your specific datasets. Where a
  finding could change published numbers, that is said explicitly.

---

## 1. Carried over unchanged — verified sound

| Feature | Where in v1.2 | Verdict |
|---|---|---|
| Pointy-top hexagonal map layout, odd-row offset, 1.5·R row pitch | 3165-3185 | Standard hex tiling; geometry is correct. |
| Tri-variate node encoding: hue = group identity, saturation = mixture, value = load | 3188-3256 | A legitimate way to show three quantities in one glyph. Ported (arithmetic corrected, see §2). A pie-glyph alternative is now offered alongside it because colour mixing is not readable back to numbers. |
| Sample drawn inside its winning hexagon, displaced toward the runner-up node by `d₁/d₂` | 3290-3333 | Sound and genuinely useful: it turns "which node won" into "how close was the call", and puts ambiguous samples on the border they are ambiguous about. Ported as-is. |
| Per-group map slices | `SOM_CL_plotting`, 3425-3518 | Sound. Extended with the occupancy-graded transparency requested in the brief. |
| Metric direction arrows | `CalAxisVector` / `PlotAxisVector`, 3633-3761 | The *idea* is the most valuable view in the whole macro. The *estimator* was replaced — see §3.1. |
| k-means over the codebook rather than over samples | `KmeansClustering`, 3852-3964 | This is the standard two-level SOM approach (Vesanto & Alhoniemi 2000). Correct. |
| Gaussian neighbourhood `exp(−d²/2σ²)` with exponentially decaying σ | 1974 | Correct. The code comment at 1973 shows an earlier release used `2σ` in the denominator; v1.2 already fixed it. |
| `est_Z` from per-track normalised intensity SD | 1469 | Valid as a *relative* focus index. It is not a calibrated depth, and SOMTrack relabels it "focus index" for that reason. |
| Robust decile trimming of per-spot values | `DecileRangeArray`, 2736 | Defensible robust statistic. Kept, with a caveat — see B16. |

---

## 2. Bugs corrected during the port

### B1 — HSB conversion uses a five-sextant wheel
`HSBtoRGB`, line 2934: `h = (h/255) * 5;`

Hue-to-RGB sextant arithmetic requires ×6. With ×5 the colour wheel is
compressed into five sixths of a turn: the sixth sector is never entered
cleanly, and groups placed near hue 1.0 collide with groups near 0.83.

**Fixed:** SOMTrack uses `matplotlib.colors.hsv_to_rgb`.

### B2 — Node hue and data-point hue use different formulas
Nodes: `hueVal = (index(nrClusterArr, …)/n) * 255` (3195).
Points: `sampleHue = 255 * ((index + 1)/n)` (3311 and 3501).

An off-by-one between the two. A node dominated by group *k* is therefore drawn
in the colour that group *k−1*'s points are drawn in. On a four-group map every
node colour is one group out of step with the dots sitting on it.

**Fixed:** node hues are derived from the same palette entry as the point
colours (`viz.style.palette_hues`). This is visible in the before/after: node
colour and point colour now agree.

### B3 — Hue mixing is a linear mean of hue numbers
Line 3204: `RGBval_1 += CLMatchNum * hueVal`, then divided by the total.

Hue is circular. A node split evenly between the first and last group produces
the *middle* group's hue rather than a colour between the two ends of the wheel.

**Fixed:** circular (vector) mean of the group hues, weighted by each group's
share of the node.

### B4 — The mixture threshold never fired
Lines 3201 and 3231 compare `mixtureIgnorePcent = 0.02` against
`CLMatchNum`, which is an **integer count**, not a fraction. Since counts are
integers, `count <= 0.02` is true only for zero. The documented "ignore nodes
holding less than 2 % of a group" filter has never had any effect.

**Fixed:** the threshold is now applied to the fraction of the node's samples,
and is exposed as `FigureConfig.mixture_ignore_fraction`.

### B5 — NaN screening double-counts
Line 1728, inside a loop over metrics:

```
if (isNaN(tempArray2[Ti])) { usedSampleListArray[Ti]=false; Current_SampleSize--; }
```

A sample with NaN in three metrics decrements `Current_SampleSize` three times.
`Dim2DArray` is then built from `usedSampleListArray` and has the correct
length, but the normalisation loop at 1771 slices it with `Current_SampleSize`:

```
NrmArray = Array.slice(Dim2DArray, iii*Current_SampleSize, (iii+1)*Current_SampleSize);
```

so as soon as any single sample carries more than one NaN, the per-metric slices
are misaligned and metric *i* is normalised using values belonging partly to
metric *i+1*. `avg_haltT`/`std_haltT` are NaN for continuously moving animals,
which is exactly the common case the comment at 1721 anticipates.

**This one can change published numbers.** If a v1.2 analysis included a sample
with NaN in two or more metrics, its normalisation was wrong.

**Fixed:** the NaN policy is explicit and applied once per sample
(`PreprocessConfig.nan_policy`), and the reported sample count is derived from
the mask rather than maintained by hand.

### B6 — Angles are averaged linearly
`avg_AngleT` / `std_AngleT` (1274) average track headings on −180…180 with
`Array.getStatistics`. `avg_AngleP` / `std_AngleP` (1401) do the same for the
fitted-ellipse orientation.

Headings are circular: +179° and −179° average to 0° instead of 180°. Body
orientation is worse than circular — it is **axial** (mod 180°), so it needs
doubling before averaging and halving afterwards.

**Fixed:** `metrics.circ_mean` / `circ_sd` for headings, doubling-halving for
the axial body orientation, plus a Rayleigh test and a mean resultant length
`heading_R` as the properly scaled measures of directional concentration.

### B7 — Training topology does not match the drawn topology
`findNeighbours` (2694) measures neighbour distance as
`sqrt((yy−ref_Y)² + (xx−ref_X)²)` over raw `(col, row)` offsets. The map is
*drawn* as a pointy-top hex grid: row pitch √3/2 ≈ 0.866 of the column pitch,
odd rows shifted half a column (3168-3175).

The neighbourhood the algorithm updated is therefore not the neighbourhood the
reader sees. Vertically adjacent nodes are 1.0 apart during training but 0.866
apart on screen; the diagonal neighbours are off by up to 15 %. The map is
mildly stretched relative to the metric it was optimised under.

**Fixed:** `som.make_lattice` builds the node-distance matrix from the same
Cartesian hex embedding the figures use, so training and display share one
geometry. A toroidal option is also available to remove edge effects.

### B8 — Learning rates above 1
`SOM_LearnRate0 = 1` (154) and the parameter scan defaults to 0.8-2.4
(556-558).

The Kohonen update is `w ← w + α·h·(x − w)`. For α·h > 1 the node moves *past*
the input rather than toward it. At the winner h = 1, so any α > 1 overshoots.
This is outside the convergence conditions of the rule, and is a plausible
contributor to the scan producing erratic quality surfaces.

**Fixed:** the online learning rate is clamped to ≤ 1 and the UI does not offer
higher values. The default algorithm is batch SOM, which has no learning rate at
all.

### B9 — "Clustering efficiency" rewards the wrong outcome
`plotSOMprogress`, 3539-3552:

```
if (zeroCounter>0) clusteringEff[ii] = zeroCounter / (ClusterListArray.length-1);
```

`zeroCounter` counts how many **cluster-subset** combinations are *absent* from
a node, and the reported figure is the unweighted mean over occupied nodes.

Two problems:

1. It scores at cluster-*subset* granularity. Two replicates of the same
   treatment landing in the same node **lower** the score — but that is the
   desired outcome, since replicates of one treatment should co-locate.
2. It is unweighted, so a node holding a single sample scores a perfect 1.0.
   A map so large that most nodes hold one sample scores near-perfectly. The
   index is maximised by over-fitting the map size.

**Replaced** by `som.group_purity`: hit-weighted purity at treatment level, plus
normalised mutual information, plus a per-node hypergeometric enrichment test
with FDR correction (`association.group_enrichment`) so "this treatment sits
over here" becomes a testable claim rather than an impression.

### B10 — "Representation efficiency" is a ratio of means
Line 3563: `EffArr[1] = BMDmean / SBMDmean`.

The mean of a ratio and the ratio of means differ whenever the distributions are
skewed, and best-match distance distributions are. **Fixed:** SOMTrack reports
`mean(d₁/d₂)` per sample, alongside the two standard indices — quantisation
error and topographic error — which the macro did not compute at all.

### B11 — Jitter is re-randomised every frame
Line 3305, inside `SOM_plotting_V2`:
`SampleAxisAngleRand[BM] = random;`

The angular jitter applied to a sample's position inside its hexagon is drawn
fresh on every plotted time point, so points twitch between frames of the
progress stack even when their node assignment has not changed. The eye reads
that motion as instability in the map.

**Fixed:** the jitter is drawn once per sample from a fixed seed
(`hexgeom.stable_jitter`) and reused across every frame.

### B12 — Meander carries a stray per-frame factor
Line 1157 divides `angv` by `(pt[i+1]-pt[i])`; line 1162 then divides by the
path length. The result has units degrees·frame⁻¹·µm⁻¹, but the header comment
(127) calls it degrees/mm and the inline comment calls it degrees/oriPixelUnit.

**Fixed:** `avg_meander` is turn angle per unit path length over the same two
steps, in degrees per length unit, with the unit propagated into the axis label.

### B13 — Division by a zero range
Line 1469: `Z_position2 = (stddev[i]-min_StdDev[Trk_i]) / range_StdDev[Trk_i];`
A track whose intensity SD is constant gives range 0 and ±Inf.

**Fixed:** guarded; the metric returns NaN and is dropped by the NaN policy.

### B14 — Axis vectors have no goodness of fit
`CalAxisVector` (3633) computes a range-normalised **first moment** of the
component plane about a reference node. It always returns a direction, whether
or not the component plane has any coherent gradient, and because the moment
weights by lattice distance it is dominated by the far corners of the map.

**Replaced** by a hit-weighted plane fit (`association.map_gradients`), which
yields the direction of steepest increase **and** an R². Arrows below the R²
threshold are drawn dashed and pale rather than being trusted silently. The
original estimator remains available as `method="legacy_moment"` for anyone
reproducing an old figure.

### B15 — Fixed 1.5-pixel pause criterion
Line 1137. A fixed pixel step does not transfer between magnifications or frame
rates: at 4× the magnification, the same animal at the same speed is never
scored as paused.

**Fixed:** `metrics.auto_halt_threshold` splits the pooled log-speed histogram
by Otsu's method, adapting to the recording. The fixed criterion is still
selectable.

### B16 — Trimmed SD reported as SD
`DecileRangeArray` keeps the central 60 % of ranked values, and the SD of that
subset is reported as `std_*`. A 60 % trimmed SD is a biased-low estimate of the
population SD — roughly 0.55× for Gaussian data.

**Not a bug**, but it means v1.2 `std_*` values are not comparable with SDs
reported elsewhere. SOMTrack keeps the behaviour for continuity, exposes the
trim fraction in the UI, and documents the bias in the tooltip.

---

## 3. Replaced because a better estimator exists

### 3.1 Map gradients
See B14. Plane fit with R², plus a per-node local gradient field, plus a compass
summary that makes redundant metric pairs visible at a glance.

### 3.2 The kymograph
The 32-bit "SOM kymograph" stack (1859-1894) encoded node × epoch × metric as
raw pixel values with overlay text labels. It carries the right information but
is unreadable without opening it in ImageJ and adjusting the display range.

**Replaced** by a labelled heatmap with a real colour bar in original units, plus
explicit quantisation-error / topographic-error / purity curves over training.

### 3.3 Choice of k for the node clustering
v1.2 drew every k from 2 to *n* as separate slices and left the choice to the
eye. SOMTrack scores each k by silhouette, Davies-Bouldin and
Calinski-Harabasz, picks the best average rank, and prints the scores on the
figure — while still saving every k.

---

## 4. Performance

The brief noted that v1.2 was slow. Two hot spots dominate.

**Per-spot image work.** `ThresholdNParticle` runs `Specify` → `Copy` → `Paste`
→ `setAutoThreshold` → `Analyze Particles` → ROI Manager traversal for *every
spot*, each call crossing the macro interpreter and rebuilding ROI state. Cost
scales with total spots, which is tracks × frames.

**The SOM loop.** Lines 1917-1987 are a triple nested interpreted loop over
samples × nodes × dimensions, with a function call (`getDim2DVal`, 2538) per
element access, and an array copy (`setDim2DVal` returns the whole array, 2550)
per weight update.

SOMTrack replaces the SOM loop with two BLAS matrix products per epoch: best-
match search via the expanded squared-distance identity, and the batch update as
a neighbourhood matrix multiplied by per-node sums. Measured on this machine,
for a 112-sample × 63-metric dataset on a 9×6 map:

| Stage | SOMTrack |
|---|---|
| Metrics for 33,600 spots across 112 tracks | 0.7 s |
| Batch SOM, 200 epochs | 0.03 s |
| Online SOM (the v1.2 rule), 100 epochs | 0.13 s |
| PCA + t-SNE | 0.14 s |
| Node clustering, k = 2…8 | ~1 s |
| Random-forest permutation importance | ~5 s |
| 26 figures rendered | 1.1 s |

The permutation importance is now the slowest step, which is a reasonable place
for the time to go: it is the part that answers *why* the groups differ.

The manual four-instance parallel scan (706-826) is replaced by
`som.scan_parameters`, a joblib grid search across all cores in one process
tree, and by `python -m somtrack scan` on the command line.

---

## 5. What this means for previously published v1.2 results

Ranked by how likely a finding is to change a conclusion:

1. **B5 (NaN misalignment)** — if any sample carried NaN in two or more metrics,
   that run's normalisation was wrong. Worth re-running.
2. **B6 (linear angle averaging)** — any claim resting on `avg_AngleT`,
   `std_AngleT`, `avg_AngleP` or `std_AngleP` should be re-derived with circular
   statistics.
3. **B9 (clustering efficiency)** — the reported efficiency favoured sparse,
   over-large maps. Comparisons of map quality between runs are not reliable.
4. **B7 (topology mismatch)** — a real but modest distortion; the map layout is
   slightly stretched, and cluster boundaries shift a little. Qualitative
   conclusions probably survive.
5. **B1–B4 (colour)** — cosmetic. The figures were misleading about which group
   a node belonged to, but the underlying assignments were correct; the tables
   were right even where the pictures were not.

To reproduce an old figure for comparison, set the scaler to `minmax`, the SOM
algorithm to `online`, the palette to `legacy_hsb`, and the gradient method to
`legacy_moment`.
