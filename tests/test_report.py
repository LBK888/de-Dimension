"""The citation log, the generated report, and one end-to-end run.

The report is a deliverable, not a debug dump: a researcher should be able to
paste its methods paragraph into a manuscript.  These tests pin the properties
that makes possible -- it describes what ran rather than what exists, every
algorithm it names is followed by a citation, and every citation it uses appears
once in the reference list.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from somtrack import pipeline                                    # noqa: E402
from somtrack.citations import CITATIONS, MethodsLog, cite       # noqa: E402
from somtrack.config import AnalysisConfig                       # noqa: E402
from somtrack.io_tables import FeatureDataset                    # noqa: E402
from somtrack.report import (build_report, render_html,          # noqa: E402
                             render_markdown, write_methods_text,
                             write_report)


# ==========================================================================
# The bibliography
# ==========================================================================
def test_short_citations_follow_the_usual_conventions():
    assert CITATIONS["kohonen1982"].short == "Kohonen, 1982"
    assert CITATIONS["okabe2008"].short == "Okabe & Ito, 2008"
    assert CITATIONS["lord2020"].short == "Lord et al., 2020"
    assert CITATIONS["maaten2008"].short == "van der Maaten & Hinton, 2008"
    assert CITATIONS["anderson2003"].short == "Anderson & ter Braak, 2003"
    assert CITATIONS["harris2020"].short == "Harris et al., 2020"


def test_every_citation_is_complete():
    for key, c in CITATIONS.items():
        assert c.key == key
        assert c.authors and c.title and c.source
        assert 1900 < c.year <= 2030, key
        assert "  " not in c.title


def test_methods_log_rejects_an_unknown_citation():
    log = MethodsLog()
    with pytest.raises(KeyError, match="Unknown citation key"):
        log.record("Statistics", "something", citations=["not_a_real_key"])


def test_methods_log_produces_the_requested_sentence_form():
    log = MethodsLog()
    log.record("Visualization", "SuperPlots", "coloured by biological replicate",
               citations=["lord2020"])
    assert log.entries[0].sentence() == (
        "SuperPlots, coloured by biological replicate (Lord et al., 2020)")


def test_methods_log_groups_and_deduplicates():
    log = MethodsLog()
    log.record("Statistics", "PERMANOVA", citations=["anderson2001"])
    log.record("Statistics", "PERMDISP", citations=["anderson2006"])
    log.record("Visualization", "SuperPlots", citations=["lord2020"])
    log.record("Visualization", "estimation plots", citations=["lord2020", "ho2019"])

    sections = log.by_section()
    assert list(sections) == ["Statistics", "Visualization"], \
        "sections must come out in reading order"
    assert len(sections["Visualization"]) == 2

    refs = [c.key for c in log.references()]
    assert refs.count("lord2020") == 1, "a repeated citation appears once"
    assert refs == sorted(refs, key=lambda k: (CITATIONS[k].authors.lower(),
                                               CITATIONS[k].year))


def test_cite_formats_several_keys():
    assert cite("lord2020", "ho2019") == "Lord et al., 2020; Ho et al., 2019"
    assert cite("not_a_key") == ""


# ==========================================================================
# One end-to-end run
# ==========================================================================
@pytest.fixture(scope="module")
def result(tmp_path_factory):
    """A small four-group dataset with a real effect and a replicate structure."""
    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(11)
    names = [f"metric_{i:02d}" for i in range(16)]
    rows, groups, reps = [], [], []
    for gi, g in enumerate(["control", "low", "high"]):
        for r in range(4):
            batch = rng.normal(0, 0.3, 16)
            for _ in range(7):
                v = rng.normal(0, 1, 16) + batch
                v[:3] += gi * 0.9
                rows.append(v)
                groups.append(g)
                reps.append(f"{g}_r{r}")

    frame = pd.DataFrame(rows, columns=names)
    frame.insert(0, "replicate", reps)
    frame.insert(0, "group", groups)
    frame.insert(0, "sample", [f"s{i:03d}" for i in range(len(frame))])
    ds = FeatureDataset(frame=frame, feature_names=names,
                        units={n: "a.u." for n in names}, descriptions={})

    cfg = AnalysisConfig()
    cfg.export.out_dir = tmp_path_factory.mktemp("run")
    cfg.export.mp4 = False
    cfg.export.png = False
    cfg.embedding.methods = ["pca", "mds", "lda"]
    cfg.embedding.allow_supervised = True
    cfg.embedding.reliability_null = 4
    cfg.stats.n_permutations = 99
    cfg.stats.classification_permutations = 99
    cfg.stats.bootstrap = 200
    cfg.stats.importance_repeats = 3
    return pipeline.run_analysis(ds, cfg)


def test_run_produces_every_layer(result):
    assert result.block_structure.design == "nested"
    assert result.separation is not None
    assert np.isfinite(result.separation.permanova_p)
    assert result.classification is not None
    assert result.importance is not None
    assert result.effects is not None and not result.effects.empty
    assert result.verdict is not None
    assert set(result.projections) == {"pca", "mds", "lda"}
    assert result.projection_agreement is not None


def test_supervised_projection_carries_its_validation(result):
    lda = result.projections["lda"]
    assert lda.supervised
    assert lda.caveat
    assert lda.oof_coords is not None
    assert lda.oof_coords.shape == lda.coords.shape
    assert lda.cv is not None and np.isfinite(lda.cv.permutation_p)


def test_tables_include_the_statistics(result):
    tables = result.tables()
    for key in ("separation_tests", "classification", "classification_confusion",
                "classifier_weights", "permutation_importance", "effect_sizes",
                "projection_quality", "projection_pca"):
        assert key in tables, f"missing result table '{key}'"
        assert not tables[key].empty
    assert "out_of_fold_1" in tables["projection_lda"].columns


def test_report_names_only_what_ran(result):
    rep = build_report(result)
    md = render_markdown(rep)

    assert "PCA" in md and "PERMANOVA" in md
    assert "UMAP" not in md, "a method that did not run must not be cited"
    assert "t-SNE" not in md

    assert "## Methods" in md and "## References" in md
    assert "Anderson, M.J. (2001)" in md
    assert "doi.org/10.1111/j.1442-9993.2001.01070.pp.x" in md


def test_report_cites_every_algorithm_it_names(result):
    rep = build_report(result)
    by_section = rep.methods.by_section()
    for entries in by_section.values():
        for e in entries:
            assert e.citations or e.section == "Data preparation", \
                f"'{e.label}' is named without a citation"
    used = set(rep.methods.citation_keys())
    listed = {c.key for c in rep.references}
    assert used == listed, "every citation used must appear exactly once in the list"


def test_report_carries_the_caveats(result):
    rep = build_report(result)
    caveats = " ".join(text for _, text in rep.methods.caveats())
    assert "construction" in caveats, \
        "the supervised projection's caveat must reach the report"


def test_report_writes_markdown_html_and_methods(result, tmp_path):
    rep = build_report(result)
    written = write_report(rep, tmp_path, result.config.report)
    assert {p.name for p in written} == {"RESULTS_REPORT.md",
                                         "RESULTS_REPORT.html"}

    html = (tmp_path / "RESULTS_REPORT.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "prefers-color-scheme" in html, "the page must work in dark mode"
    assert "<h2>Methods</h2>" in html

    methods = write_methods_text(rep, tmp_path / "methods.txt")
    text = methods.read_text(encoding="utf-8")
    assert "References" in text
    assert "Caveats" in text
    assert text.count("Anderson, M.J. (2001)") == 1


def test_verdict_is_consistent_with_the_numbers(result):
    v = result.verdict
    text = v.text()
    assert f"{result.separation.permanova_R2:.3f}" in text
    assert f"{result.classification.balanced_accuracy:.2f}" in text
    assert v.confidence in ("strong", "moderate", "weak", "none")


def test_figures_render_in_report_order(result):
    panels = pipeline.build_figures(result)
    names = [n for n, _ in panels]
    assert names[0] == "01_verdict", "the conclusion is page one"
    assert any(n.startswith("02_separation") for n in names)
    assert any(n.startswith("15_validated_lda") for n in names), \
        "a supervised projection must get its out-of-fold panel"
    assert set(result.figure_captions) == set(names)
    assert all(result.figure_captions[n] for n in names), \
        "every figure must say in the report index what question it answers"

    from matplotlib import pyplot as plt

    for _, panel in panels:
        plt.close(panel.fig)


def test_core_profile_is_shorter_than_full(result):
    from matplotlib import pyplot as plt

    result.config.report.profile = "core"
    core = pipeline.build_figures(result)
    for _, p in core:
        plt.close(p.fig)

    result.config.report.profile = "full"
    full = pipeline.build_figures(result)
    for _, p in full:
        plt.close(p.fig)

    result.config.report.profile = "core"
    assert 6 <= len(core) < len(full), \
        "'core' must be a usable subset, not the whole set"
