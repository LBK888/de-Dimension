"""The Traditional Chinese interface and the bilingual report.

Three promises are pinned here:

* the report is written in English and then translated, never mixed -- the
  English half is the report 3.0 always wrote, and the Chinese half is complete;
* nothing the program says in Chinese falls back to English unnoticed, and no
  translation drops a ``{placeholder}`` that its English source fills in;
* figures stay in English whatever language the interface is in, and a
  translated label in a combo box can never leak into the configuration.
"""

from __future__ import annotations

import ast
import copy
import os
import pathlib
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from somtrack import i18n                                          # noqa: E402
from somtrack.config import AnalysisConfig                         # noqa: E402
from somtrack.i18n import Text, join_list, render, tr              # noqa: E402
from somtrack.io_tables import FeatureDataset                      # noqa: E402
from somtrack.locales import zh_TW                                 # noqa: E402


def _has_cjk(text: str) -> bool:
    return any("　" <= ch <= "鿿" or "＀" <= ch <= "￯"
               for ch in str(text))


@pytest.fixture
def chinese():
    """Run a test with the interface in Traditional Chinese, then restore English."""
    i18n.set_language("zh_TW")
    i18n.clear_missing()
    try:
        yield
    finally:
        i18n.set_language("en")


# ==========================================================================
# The machinery
# ==========================================================================
def test_text_is_the_english_string_and_renders_in_chinese():
    t = Text("Dropped {n} sample(s) containing NaN.", n=3)
    assert t == "Dropped 3 sample(s) containing NaN."
    assert isinstance(t, str)
    assert f"{t}" == str(t) == "Dropped 3 sample(s) containing NaN."
    zh = t.render("zh_TW")
    assert zh != t and "3" in zh and _has_cjk(zh)
    assert t.render("en") == t


def test_plain_arguments_are_data_and_are_not_translated():
    # a group that happens to be called "correlation" is still called that
    t = Text("{a} vs {b} (balanced accuracy {ba:.2f})", a="correlation", b="low",
             ba=0.8)
    assert "correlation" in t.render("zh_TW")
    # ...whereas a Text argument is prose and is translated
    t2 = Text("PERMANOVA on {metric} distances: pseudo-F = {F:.2f}, R2 = {r2:.3f} "
              "({pct:.1f}% of the multivariate variation is accounted for by the "
              "grouping), {p}, {n} permutations.", metric=Text("euclidean"), F=1.0,
              r2=0.1, pct=10.0, p="p = 0.010", n=99)
    assert "euclidean" in t2 and "歐氏" in t2.render("zh_TW")


def test_translation_survives_pickling_and_copying():
    t = Text("{k}-fold stratified cross-validation repeated {repeats} times",
             k=5, repeats=3)
    joined = join_list([t, Text("Folds preserved the group proportions.")])
    for obj in (t, joined):
        for clone in (pickle.loads(pickle.dumps(obj)), copy.deepcopy(obj)):
            assert clone == obj
            assert clone.render("zh_TW") == obj.render("zh_TW")


def test_language_codes_are_normalised():
    assert i18n.normalise("zh-TW") == "zh_TW"
    assert i18n.normalise("zh_Hant_TW") == "zh_TW"
    assert i18n.normalise("zh_HK") == "zh_TW"
    assert i18n.normalise("fr_FR") == "en"
    assert i18n.normalise(None) == "en"


# ==========================================================================
# The catalogue
# ==========================================================================
def test_every_translation_keeps_its_placeholders():
    bad = [k for k, v in zh_TW.MESSAGES.items()
           if i18n.placeholders(k) != i18n.placeholders(v)]
    for table in zh_TW.CONTEXTS.values():
        bad += [k for k, v in table.items()
                if i18n.placeholders(k) != i18n.placeholders(v)]
    assert not bad, f"translations that drop or add a placeholder: {bad}"


def test_the_catalogue_has_no_duplicate_keys():
    """A repeated key in a dict literal silently discards the first translation."""
    src = pathlib.Path(zh_TW.__file__).read_text(encoding="utf-8")
    dupes = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            dupes += [k for k in set(keys) if keys.count(k) > 1]
    assert not dupes, dupes


def test_registry_text_is_translated():
    """Method, metric and recipe descriptions are shown in the interface."""
    from somtrack import recipes
    from somtrack.analysis import FAMILY_LABELS, all_methods
    from somtrack.analysis.scan import CRITERIA
    from somtrack.metrics import METRICS

    strings = []
    for spec in all_methods():
        strings += [spec.summary, spec.detail, spec.caveat, spec.install_hint]
        for p in spec.params:
            strings += [p.display, p.help]
    for m in METRICS.values():
        strings += [m.label, m.description, m.category]
    for r in recipes.RECIPES:
        strings += [r.title, r.summary, r.detail]
    strings += list(FAMILY_LABELS.values())
    strings += [c[3] for c in CRITERIA.values()]
    missing = sorted({s for s in strings if s and s not in zh_TW.MESSAGES})
    assert not missing, f"untranslated registry text: {missing[:10]}"


# ==========================================================================
# One analysis, reported in two languages
# ==========================================================================
@pytest.fixture(scope="module")
def result(tmp_path_factory):
    from somtrack import pipeline

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(11)
    names = [f"metric_{i:02d}" for i in range(10)]
    rows, groups, reps = [], [], []
    for gi, g in enumerate(["control", "low", "high"]):
        for r in range(3):
            for _ in range(6):
                v = rng.normal(0, 1, 10)
                v[:3] += gi * 1.0
                rows.append(v)
                groups.append(g)
                reps.append(f"{g}_r{r}")
    frame = pd.DataFrame(rows, columns=names)
    frame.insert(0, "replicate", reps)
    frame.insert(0, "group", groups)
    frame.insert(0, "sample", [f"s{i:03d}" for i in range(len(frame))])
    frame.iloc[2, 4] = np.nan
    ds = FeatureDataset(frame=frame, feature_names=names,
                        units={n: "a.u." for n in names}, descriptions={})

    cfg = AnalysisConfig()
    cfg.export.out_dir = tmp_path_factory.mktemp("bilingual")
    cfg.export.mp4 = False
    cfg.export.png = False
    cfg.embedding.methods = ["pca", "mds", "lda"]
    cfg.embedding.allow_supervised = True
    cfg.embedding.reliability_null = 3
    cfg.stats.n_permutations = 49
    cfg.stats.classification_permutations = 49
    cfg.stats.bootstrap = 100
    cfg.stats.importance_repeats = 2
    cfg.stats.run_importance = True
    cfg.stats.run_effect_sizes = True
    return pipeline.run_analysis(ds, cfg)


def test_report_is_english_first_then_a_complete_translation(result):
    from somtrack.report import build_report, render_html, render_markdown

    rep = build_report(result)
    md = render_markdown(rep)
    english, _, chinese = md.partition("\n---\n")
    assert chinese, "the translation must follow the English report"

    assert not _has_cjk(english), "the English report must not contain Chinese"
    assert "## Methods" in english and "## References" in english
    assert "# 運動指標的多變量分析（中文翻譯）" in chinese
    for heading in ("## 結論", "## 資料", "## 方法", "## 參考文獻", "## 可重現性"):
        assert heading in chinese, heading

    # the same tables, in the same order, with the same (English) contents
    tables_en = [line for line in english.splitlines() if line.startswith("|")]
    tables_zh = [line for line in chinese.splitlines() if line.startswith("|")]
    assert tables_en and tables_en == tables_zh
    # every caption is translated
    assert "**整體檢定**" in chinese and "**Omnibus tests**" in english

    # one reference list for the whole file
    assert md.count("Anderson, M.J. (2001)") == 1

    html = render_html(rep)
    assert "<h2>Methods</h2>" in html and "<h2>方法</h2>" in html
    assert html.index("<h2>Methods</h2>") < html.index("<h2>方法</h2>")
    assert "lang='zh-Hant-TW'" in html


def test_translation_can_be_switched_off(result):
    from somtrack.report import build_report, render_markdown

    result.config.report.translation = ""
    try:
        md = render_markdown(build_report(result))
    finally:
        result.config.report.translation = "zh_TW"
    assert not _has_cjk(md)


def test_methods_text_carries_the_translation(result, tmp_path):
    from somtrack.report import build_report, write_methods_text

    text = write_methods_text(build_report(result), tmp_path / "m.txt").read_text(
        encoding="utf-8")
    english, _, chinese = text.partition("#" * 66)
    assert "Statistics." in english and not _has_cjk(english)
    assert "統計。" in chinese and "PERMANOVA" in chinese
    assert text.count("Anderson, M.J. (2001)") == 1


def test_nothing_in_a_chinese_run_falls_back_to_english(result, tmp_path):
    from somtrack import pipeline
    from somtrack.report import (build_report, render_html, render_markdown,
                                 write_methods_text)

    i18n.clear_missing()
    rep = build_report(result)
    render_markdown(rep)
    render_html(rep)
    write_methods_text(rep, tmp_path / "methods.txt")
    result.verdict.text().render("zh_TW")
    for line in result.verdict.bullets():
        render(line, "zh_TW")
    for w in result.warnings:
        render(w, "zh_TW")
    captions = dict(result.figure_captions)
    if not captions:
        from matplotlib import pyplot as plt

        for _, p in pipeline.build_figures(result):
            plt.close(p.fig)
        captions = dict(result.figure_captions)
    for caption in captions.values():
        render(caption, "zh_TW")
    assert not i18n.missing("zh_TW"), sorted(i18n.missing("zh_TW"))[:10]


def test_figures_stay_in_english_under_a_chinese_interface(result, chinese):
    from matplotlib import pyplot as plt
    from matplotlib.text import Text as MplText

    from somtrack import pipeline

    panels = pipeline.build_figures(result)
    try:
        assert panels
        for name, panel in panels:
            for text in panel.fig.findobj(MplText):
                assert not _has_cjk(text.get_text()), \
                    f"figure '{name}' contains Chinese: {text.get_text()!r}"
        # ...while its caption in the report index is translated
        assert _has_cjk(render(result.figure_captions["01_verdict"]))
    finally:
        for _, p in panels:
            plt.close(p.fig)


# ==========================================================================
# The desktop app
# ==========================================================================
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_whole_window_builds_in_chinese(qapp, chinese):
    from PySide6.QtWidgets import QLabel, QPushButton

    from somtrack.ui.main_window import STEP_TITLES, MainWindow

    w = MainWindow()
    assert w.steps.item(0).text() == zh_TW.MESSAGES[STEP_TITLES[0]]
    assert w.next_btn.text() == "下一步"
    assert w.results_page.tabs.tabText(0) == "結論"
    for page in w.pages:
        page.on_enter()
    texts = [b.text() for b in w.findChildren(QPushButton)]
    texts += [lab.text() for lab in w.findChildren(QLabel)]
    assert any(_has_cjk(t) for t in texts)
    assert not i18n.missing("zh_TW"), sorted(i18n.missing("zh_TW"))[:10]


def test_translated_combo_labels_never_reach_the_config(qapp, chinese):
    from somtrack.ui.pages import AppState, ClusterPage, ExportPage, PreprocessPage

    state = AppState()
    cluster = ClusterPage(state)
    pre = PreprocessPage(state)
    export = ExportPage(state)
    assert _has_cjk(cluster.algo.currentText())      # the user reads Chinese...
    cluster.commit()
    pre.commit()
    export.commit()
    cfg = state.config                                 # ...the config keeps keys
    assert cfg.som.algorithm == "batch"
    assert cfg.som.topology == "hex"
    assert cfg.som.init == "pca"
    assert cfg.stats.distance == "euclidean"
    assert cfg.node_cluster.method == "kmeans"
    assert cfg.preprocess.scaler == "zscore"
    assert cfg.preprocess.nan_policy == "impute_median"
    assert cfg.figure.palette == "somtrack"
    assert cfg.report.translation == "zh_TW"


def test_interface_text_is_english_by_default():
    assert i18n.language() == "en"
    assert tr("Next") == "Next"
