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


# ==========================================================================
# Settings that disagree between steps
# ==========================================================================
def _replicate_frame(features, crossed: bool):
    from somtrack.io_tables import FeatureDataset

    frame = features.frame.copy()
    reps = np.tile(np.repeat(np.arange(2), 10), 3)
    if crossed:      # replicate 0 and 1 in every group: one experiment day each
        frame["replicate"] = reps
    else:            # every replicate belongs to exactly one group
        frame["replicate"] = [f"{g}_{r}" for g, r in zip(frame["group"], reps)]
    return FeatureDataset(frame=frame, feature_names=features.feature_names,
                          units=features.units, descriptions={})


def _small_config():
    cfg = AnalysisConfig()
    cfg.embedding.methods = ["pca"]
    cfg.embedding.reliability_null = 3
    cfg.stats.n_permutations = 19
    cfg.stats.classification_permutations = 19
    cfg.stats.bootstrap = 50
    cfg.stats.importance_repeats = 2
    cfg.som.epochs = 20
    return cfg


@pytest.mark.parametrize("crossed", [True, False])
def test_averaging_within_replicates_never_aborts_the_run(features, crossed):
    """'Average within each replicate first' with replicate numbers reused in
    every group collapsed the data to two rows with one label, and the default
    LDA classifier then raised: the 'LDA error without LDA' report."""
    from somtrack import pipeline

    cfg = _small_config()
    cfg.stats.unit_of_analysis = "replicate"
    res = pipeline.run_analysis(_replicate_frame(features, crossed), cfg)
    assert res.verdict is not None and res.separation is not None
    if crossed:
        assert any("mix the groups" in w for w in res.warnings)


def test_the_result_keeps_the_settings_it_was_computed_with(features):
    from somtrack import pipeline

    cfg = _small_config()
    res = pipeline.run_analysis(features, cfg)
    cfg.som.algorithm = "online"            # the UI goes on editing its config
    cfg.som.epochs = 999
    assert res.config.som.algorithm == "batch" and res.config.som.epochs == 20


def _silence_message_boxes(monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    shown = []
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: shown.append(a)))
    return shown


def test_sidebar_jumps_forward_through_every_step(qapp, features, monkeypatch):
    """A click on a later step used to be silently undone."""
    from somtrack.ui.main_window import MainWindow

    shown = _silence_message_boxes(monkeypatch)
    w = MainWindow()

    def load_table():
        w.state.entry_mode = "features"
        w.state.features = features

    monkeypatch.setattr(w.source_page, "validate", lambda: None)
    monkeypatch.setattr(w.source_page, "commit", load_table)
    target = w.pages.index(w.analysis_page)
    w.steps.setCurrentRow(target)
    assert w.stack.currentIndex() == target
    assert w.steps.currentRow() == target
    assert not shown

    # ...and stops, saying why, at the first step that is not ready
    def load_coordinates():
        w.state.entry_mode = "coordinates"
        w.state.features = None

    w._go_to(0)
    monkeypatch.setattr(w.source_page, "commit", load_coordinates)
    w.steps.setCurrentRow(w.pages.index(w.export_page))
    assert w.stack.currentIndex() == w.pages.index(w.metrics_page)
    assert shown, "the user must be told why the jump stopped"


def test_feature_choice_survives_a_visit_to_a_later_step(qapp, features):
    from PySide6.QtCore import Qt

    from somtrack.ui.pages import AppState, PreprocessPage

    state = AppState()
    state.features = features
    page = PreprocessPage(state)
    page.on_enter()
    page.feats.item(0).setCheckState(Qt.CheckState.Unchecked)
    page.commit()
    page.on_enter()                                   # back from the next step
    assert features.feature_names[0] not in page.feats.selected()


def test_a_recipe_with_a_supervised_method_works_every_time(qapp, features):
    from somtrack.ui.pages import AppState, ClusterPage

    state = AppState()
    state.features = features
    page = ClusterPage(state)
    page.on_enter()
    for key in ("which_metrics", "standard", "which_metrics"):
        page.recipe.setCurrentIndex(page.recipe.findData(key))
    page.commit()
    assert "svm" in state.config.embedding.methods
    assert state.config.embedding.allow_supervised


def test_supervised_boxes_start_locked(qapp):
    from somtrack.ui.pages import AppState, ClusterPage

    page = ClusterPage(AppState())
    assert not page.supervised_ack.isChecked()
    for key, box in page._boxes.items():
        if _supervised(key):
            assert not box.isEnabled()


def test_hidden_recipe_settings_reach_the_config(qapp):
    from somtrack.ui.pages import AppState, ClusterPage

    state = AppState()
    page = ClusterPage(state)
    page.recipe.setCurrentIndex(page.recipe.findData("robustness"))
    page.commit()
    assert state.config.embedding.stability_seeds == 5
    assert state.config.report.profile == "full"


def test_an_adopted_scan_survives_the_next_run(qapp, features, result):
    from somtrack.analysis import DataContext
    from somtrack.analysis.scan import scan
    from somtrack.ui.main_window import MainWindow

    w = MainWindow()
    w.state.features = features
    w.state.result = result
    w.analysis_page.on_enter()
    w.analysis_page.commit()
    sc = scan(DataContext.from_prepared(result.prep, random_state=0), "pacmap",
              {"n_neighbors": [5, 7]}, criterion="rnx_auc", reliability_null=3,
              n_jobs=1)
    w.compare_page.show_scan(sc)
    w.compare_page._adopt()
    w.analysis_page.commit()                         # what pressing Run does
    assert w.state.config.embedding.overrides["pacmap"]["n_neighbors"] in (5, 7)

    from matplotlib import pyplot as plt

    plt.close("all")


def test_loaded_settings_appear_in_the_controls(qapp, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    from somtrack.ui.main_window import MainWindow

    cfg = AnalysisConfig()
    cfg.som.algorithm = "online"
    cfg.som.epochs = 321
    cfg.preprocess.scaler = "rank"
    cfg.figure.dpi_png = 300
    path = cfg.to_json(tmp_path / "s.json")
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(path), "")))
    w = MainWindow()
    w._load_config()
    assert w.analysis_page.epochs.value() == 321
    assert w.analysis_page.algo.currentData() == "online"
    assert w.preprocess_page.scaler.currentData() == "rank"
    assert w.export_page.dpi.value() == 300


# ==========================================================================
# Dark mode and hover help
# ==========================================================================
def test_no_widget_carries_a_colour_of_its_own():
    """A hard-coded background with the palette's text colour is how dark mode
    ended up with white text on white fields."""
    import pathlib

    for name in ("main_window.py", "pages.py", "forms.py", "widgets.py"):
        src = pathlib.Path("somtrack/ui", name).read_text(encoding="utf-8")
        assert ".setStyleSheet(" not in src, f"{name} sets its own style sheet"


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_both_themes_build_the_window(qapp, mode):
    from somtrack.ui import theme
    from somtrack.ui.main_window import MainWindow

    t = theme.apply(qapp, mode)
    try:
        assert t.dark == (mode == "dark")
        pal = qapp.palette()
        assert pal.color(pal.ColorRole.Base).name() == t.base
        assert pal.color(pal.ColorRole.Text).name() == t.text
        w = MainWindow()
        for p in w.pages:
            p.on_enter()
    finally:
        theme.apply(qapp, "light")


def test_every_setting_explains_itself(qapp):
    """Each label in a settings form has hover help, and so does each step."""
    from PySide6.QtWidgets import QFormLayout, QLabel

    from somtrack.analysis import all_methods
    from somtrack.ui.main_window import MainWindow

    w = MainWindow()
    # method parameters without a help sentence in the registry are allowed
    no_help = {p.display for s in all_methods() for p in s.params if not p.help}
    bare = []
    for page in (w.source_page, w.preprocess_page, w.analysis_page, w.export_page):
        for form in page.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                label = item.widget() if item is not None else None
                if (isinstance(label, QLabel) and label.text().strip()
                        and not label.toolTip() and label.text() not in no_help):
                    bare.append(f"{type(page).__name__}: {label.text()}")
    assert not bare, bare
    for i in range(w.steps.count()):
        assert w.steps.item(i).toolTip(), "every step says what it is for"
