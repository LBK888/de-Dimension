"""The desktop app's wiring, which is where silent breakage hides.

None of this opens a window: the tests run Qt's offscreen platform and only
check structure.  They exist because inserting the Compare page shifted every
later page by one, and every ``self.pages[5]`` in the main window quietly began
addressing the wrong page -- a failure that no amount of running the analysis
would have revealed, because the analysis still worked.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import matplotlib                                                # noqa: E402

matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication                       # noqa: E402

from somtrack.config import AnalysisConfig                       # noqa: E402
from somtrack.io_tables import FeatureDataset                    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def features():
    rng = np.random.default_rng(4)
    names = [f"m{i:02d}" for i in range(12)]
    y = np.repeat(["ctrl", "low", "high"], 20)
    X = rng.normal(size=(60, 12))
    X[y == "low", :3] += 1.5
    X[y == "high", :3] -= 1.5
    frame = pd.DataFrame(X, columns=names)
    frame.insert(0, "replicate", np.tile(np.repeat(np.arange(2), 10), 3))
    frame.insert(0, "group", y)
    frame.insert(0, "sample", [f"s{i}" for i in range(60)])
    return FeatureDataset(frame=frame, feature_names=names,
                          units={n: "" for n in names}, descriptions={})


# ==========================================================================
# The wizard
# ==========================================================================
def test_every_page_index_in_the_main_window_resolves(qapp):
    """`self.pages[i].attr` must address a page that has `attr`."""
    import pathlib

    from somtrack.ui.main_window import STEP_TITLES, MainWindow

    w = MainWindow()
    assert len(w.pages) == len(STEP_TITLES) == w.stack.count() == w.steps.count()

    src = pathlib.Path("somtrack/ui/main_window.py").read_text(encoding="utf-8")
    for idx, attr in re.findall(r"self\.pages\[(\d)\]\.(\w+)", src):
        page = w.pages[int(idx)]
        assert hasattr(page, attr), (
            f"pages[{idx}] is {type(page).__name__}, which has no '{attr}'")


def test_named_page_handles_match_their_classes(qapp):
    from somtrack.ui.main_window import MainWindow

    w = MainWindow()
    for handle, cls in (("source_page", "SourcePage"),
                        ("metrics_page", "MetricsPage"),
                        ("preprocess_page", "PreprocessPage"),
                        ("analysis_page", "ClusterPage"),
                        ("results_page", "ResultsPage"),
                        ("compare_page", "ComparePage"),
                        ("export_page", "ExportPage")):
        assert type(getattr(w, handle)).__name__ == cls
        assert getattr(w, handle) in w.pages


# ==========================================================================
# The analysis page
# ==========================================================================
def test_recipes_drive_the_method_selection(qapp, features):
    from somtrack import recipes
    from somtrack.ui.pages import AppState, ClusterPage

    state = AppState()
    state.features = features
    page = ClusterPage(state)
    page.on_enter()

    for r in recipes.RECIPES:
        i = page.recipe.findData(r.key)
        assert i >= 0, f"recipe '{r.key}' is missing from the picker"
        page.recipe.setCurrentIndex(i)
        page.commit()
        wanted = set(r.build(AnalysisConfig()).embedding.resolved_methods())
        chosen = set(state.config.embedding.methods)
        assert chosen <= wanted, f"{r.key} selected a method it did not ask for"
        if any(k for k in wanted if _supervised(k)):
            assert state.config.embedding.allow_supervised


def _supervised(key: str) -> bool:
    from somtrack.analysis import get

    return get(key).supervised


def test_untouched_parameters_are_not_recorded_as_choices(qapp, features):
    """A default nobody chose must not be pinned as an override."""
    from somtrack.ui.pages import AppState, ClusterPage

    state = AppState()
    state.features = features
    page = ClusterPage(state)
    page.on_enter()
    page.commit()
    assert state.config.embedding.overrides == {}

    page._boxes["tsne"].setChecked(True)
    page._panels["tsne"].rows["perplexity"].set_value(9.0)
    page.commit()
    assert state.config.embedding.overrides["tsne"]["perplexity"] == 9.0


def test_auto_shows_the_value_it_picked(qapp, features):
    from somtrack.ui.pages import AppState, ClusterPage

    state = AppState()
    state.features = features
    page = ClusterPage(state)
    page.on_enter()

    row = page._panels["tsne"].rows["perplexity"]
    assert row.auto.isChecked()
    assert "auto:" in row.note.text(), \
        "the user must be able to see what the program chose for them"


def test_supervised_methods_need_an_acknowledgement(qapp, features):
    from somtrack.ui.pages import AppState, ClusterPage

    state = AppState()
    state.features = features
    page = ClusterPage(state)
    page.on_enter()

    page.supervised_ack.setChecked(False)
    for key, box in page._boxes.items():
        if _supervised(key):
            assert not box.isChecked()
    page.commit()
    assert not state.config.embedding.allow_supervised
    assert not any(_supervised(k) for k in state.config.embedding.methods)


# ==========================================================================
# Results and Compare
# ==========================================================================
@pytest.fixture(scope="module")
def result(features):
    from somtrack import pipeline

    cfg = AnalysisConfig()
    cfg.embedding.methods = ["pca", "mds", "pacmap"]
    cfg.embedding.reliability_null = 3
    cfg.stats.n_permutations = 29
    cfg.stats.classification_permutations = 29
    cfg.stats.bootstrap = 80
    cfg.stats.importance_repeats = 2
    return pipeline.run_analysis(features, cfg)


def test_results_page_opens_on_the_conclusion(qapp, features, result):
    from somtrack import pipeline
    from somtrack.ui.pages import AppState, ResultsPage

    state = AppState()
    state.config = result.config
    state.features = features
    page = ResultsPage(state)
    panels = pipeline.build_figures(result)
    page.show_result(result, panels)

    assert page.tabs.tabText(0) == "Conclusion"
    assert page.tabs.currentIndex() == 0, \
        "a reader must land on the answer, not on a gallery of figures"
    html = page.verdict.toHtml()
    assert result.verdict.headline.rstrip(".") in html

    from matplotlib import pyplot as plt

    for _, p in panels:
        plt.close(p.fig)


def test_compare_page_builds_and_adopts_a_scan(qapp, features, result):
    from somtrack.analysis import DataContext
    from somtrack.analysis.scan import scan
    from somtrack.ui.pages import AppState, ComparePage

    state = AppState()
    state.config = result.config
    state.features = features
    state.result = result

    page = ComparePage(state)
    page.show_result(result)
    assert page.scan_method.count() > 0, "something must be scannable"
    assert not page.adopt_btn.isEnabled(), "nothing to adopt before a scan runs"

    ctx = DataContext.from_prepared(result.prep, random_state=0)
    sc = scan(ctx, "pacmap", {"n_neighbors": [5, 10]}, criterion="rnx_auc",
              reliability_null=3, n_jobs=1)
    page.show_scan(sc)

    assert page.plateau.isVisibleTo(page)
    assert page.adopt_btn.isEnabled()
    page._adopt()
    assert state.config.embedding.overrides["pacmap"]["n_neighbors"] in (5, 10)
    assert "pacmap" in state.config.embedding.methods

    from matplotlib import pyplot as plt

    plt.close("all")
