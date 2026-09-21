"""The SOMTrack main window: a seven-step wizard with a persistent status pane."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSplitter,
                               QStackedWidget, QVBoxLayout, QWidget)

from .. import __version__
from ..config import AnalysisConfig
from ..i18n import LANGUAGES, language, render, tr
from .pages import (AppState, ClusterPage, ComparePage, ExportPage,
                    MetricsPage, PreprocessPage, ResultsPage, SourcePage)
from .widgets import ACCENT, LogPane, heading
from .workers import TaskRunner

# English source strings; each is translated when the sidebar is built, so the
# list itself does not depend on the language the app was started in.
STEP_TITLES = [
    "1  Data source",
    "2  Metrics",
    "3  Feature matrix",
    "4  Analysis",
    "5  Results",
    "6  Compare",
    "7  Export",
]


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(tr("SOMTrack 3.0 - locomotion metrics, multivariate "
                               "analysis and publication figures"))
        self.resize(1360, 900)

        self.state = AppState()
        self.runner = TaskRunner(self)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        body = QSplitter(Qt.Orientation.Horizontal)
        body.addWidget(self._build_sidebar())

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        self.pages = [
            SourcePage(self.state),
            MetricsPage(self.state),
            PreprocessPage(self.state),
            ClusterPage(self.state),
            ResultsPage(self.state),
            ComparePage(self.state),
            ExportPage(self.state),
        ]
        # Named handles onto the wizard steps.  Addressing pages by index is
        # how inserting a page silently rewires the program: the Compare page
        # went in at 5 and every later `self.pages[5]` quietly became the wrong
        # page.  The names cannot drift.
        (self.source_page, self.metrics_page, self.preprocess_page,
         self.analysis_page, self.results_page, self.compare_page,
         self.export_page) = self.pages

        for p in self.pages:
            p.status.connect(self._log)
            self.stack.addWidget(p)
        rlay.addWidget(self.stack, 1)
        rlay.addWidget(self._build_footer())
        body.addWidget(right)
        body.setSizes([210, 1150])
        root.addWidget(body, 1)

        self._build_menu()
        self._wire()
        self._go_to(0)

    # ------------------------------------------------------------------
    def _build_sidebar(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#f4f6f8;")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(12, 16, 12, 12)

        title = heading("SOMTrack", 17)
        title.setStyleSheet(f"color:{ACCENT};")
        lay.addWidget(title)
        sub = QLabel(f"v{__version__}  ·  " + tr("locomotion SOM"))
        sub.setStyleSheet("color:#7a828c;")
        lay.addWidget(sub)
        lay.addSpacing(14)

        self.steps = QListWidget()
        self.steps.setFrameShape(QListWidget.Shape.NoFrame)
        self.steps.setStyleSheet(
            "QListWidget{background:transparent;}"
            "QListWidget::item{padding:9px 8px;border-radius:4px;color:#39414b;}"
            f"QListWidget::item:selected{{background:{ACCENT};color:white;}}"
            "QListWidget::item:disabled{color:#b3bac2;}"
        )
        for t in STEP_TITLES:
            self.steps.addItem(QListWidgetItem(tr(t)))
        self.steps.currentRowChanged.connect(self._sidebar_clicked)
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.steps.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        row_h = self.steps.sizeHintForRow(0) if self.steps.count() else 34
        self.steps.setFixedHeight(row_h * self.steps.count() + 8)
        lay.addWidget(self.steps)
        lay.addStretch(1)

        self.log = LogPane()
        self.log.setMinimumHeight(190)
        lay.addWidget(QLabel(tr("Log")))
        lay.addWidget(self.log)
        return w

    def _build_footer(self) -> QWidget:
        w = QWidget()
        w.setStyleSheet("background:#f4f6f8;border-top:1px solid #dfe3e8;")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(16, 8, 16, 8)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(280)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.status_label = QLabel(tr("Ready"))
        self.status_label.setStyleSheet("color:#5d6570;")
        lay.addWidget(self.progress)
        lay.addWidget(self.status_label, 1)

        self.back_btn = QPushButton(tr("Back"))
        self.next_btn = QPushButton(tr("Next"))
        self.next_btn.setDefault(True)
        self.next_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;padding:6px 22px;"
            "border-radius:4px;font-weight:600;}}"
            "QPushButton:disabled{background:#b9c3cc;}")
        self.back_btn.clicked.connect(lambda: self._go_to(self.stack.currentIndex() - 1))
        self.next_btn.clicked.connect(self._next)
        lay.addWidget(self.back_btn)
        lay.addWidget(self.next_btn)
        return w

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu(tr("&File"))
        for text, slot, shortcut in (
            (tr("Save settings..."), self._save_config, QKeySequence.StandardKey.Save),
            (tr("Load settings..."), self._load_config, QKeySequence.StandardKey.Open),
        ):
            a = QAction(text, self)
            a.setShortcut(shortcut)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        quit_a = QAction(tr("Quit"), self)
        quit_a.setShortcut(QKeySequence.StandardKey.Quit)
        quit_a.triggered.connect(self.close)
        m.addAction(quit_a)

        # "Language / 語言" in both scripts, so it can be found by someone who
        # cannot read the language the menu bar is currently in.
        lang_menu = self.menuBar().addMenu("&Language / 語言")
        group = QActionGroup(self)
        group.setExclusive(True)
        self.language_actions: dict[str, QAction] = {}
        for code, name in LANGUAGES.items():
            a = QAction(name, self, checkable=True)
            a.setChecked(code == language())
            a.triggered.connect(lambda _=False, c=code: self._set_language(c))
            group.addAction(a)
            lang_menu.addAction(a)
            self.language_actions[code] = a

        h = self.menuBar().addMenu(tr("&Help"))
        about = QAction(tr("About SOMTrack"), self)
        about.triggered.connect(self._about)
        h.addAction(about)

    def _wire(self) -> None:
        self.compare_page.scan_requested.connect(self._run_scan)
        self.metrics_page.compute_requested.connect(self._compute_metrics)
        self.analysis_page.run_requested.connect(self._run_analysis)
        self.export_page.export_requested.connect(self._export)

    # ------------------------------------------------------------------
    def _log(self, message: str, level: str = "info") -> None:
        message = render(message)
        self.log.log(message, level)
        self.status_label.setText(message)

    def _busy(self, on: bool, message: str = "") -> None:
        self.progress.setVisible(on)
        self.progress.setRange(0, 100)
        self.next_btn.setEnabled(not on)
        self.back_btn.setEnabled(not on)
        for p in self.pages:
            for attr in ("compute_btn", "run_btn", "export_btn"):
                b = getattr(p, attr, None)
                if b is not None:
                    b.setEnabled(not on)
        if message:
            self.status_label.setText(render(message))

    def _on_progress(self, message: str, fraction: float) -> None:
        self.progress.setValue(int(fraction * 100))
        if message:
            self.status_label.setText(render(message))

    def _on_error(self, message: str, tb: str) -> None:
        self._busy(False)
        self._log(message, "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(tr("Something went wrong"))
        box.setText(render(message))
        box.setDetailedText(tb)
        box.exec()

    # ------------------------------------------------------------------
    def _sidebar_clicked(self, row: int) -> None:
        if row < 0 or row == self.stack.currentIndex():
            return
        if row > self.stack.currentIndex():
            self.steps.blockSignals(True)
            self.steps.setCurrentRow(self.stack.currentIndex())
            self.steps.blockSignals(False)
            return
        self._go_to(row)

    def _go_to(self, index: int) -> None:
        index = max(0, min(index, len(self.pages) - 1))
        if index == self.pages.index(self.metrics_page)                 and self.state.entry_mode == "features":
            index = 2 if self.stack.currentIndex() < 1 else 0
        self.stack.setCurrentIndex(index)
        self.steps.blockSignals(True)
        self.steps.setCurrentRow(index)
        self.steps.blockSignals(False)
        self.back_btn.setEnabled(index > 0)
        self.next_btn.setText(tr("Finish") if index == len(self.pages) - 1 else tr("Next"))
        self.next_btn.setEnabled(index < len(self.pages) - 1)
        self.pages[index].on_enter()

    def _next(self) -> None:
        page = self.pages[self.stack.currentIndex()]
        problem = page.validate()
        if problem:
            self._log(problem, "warn")
            QMessageBox.information(self, tr("One more thing"), render(problem))
            return
        try:
            page.commit()
        except Exception as exc:
            self._on_error(_exception_text(exc), "")
            return

        idx = self.stack.currentIndex()
        if idx == 0 and self.state.entry_mode == "features":
            self._log(tr("Loaded {n} samples, {m} features.").format(
                n=len(self.state.features.frame),
                m=len(self.state.features.feature_names)), "ok")
            self._go_to(2)
            return
        self._go_to(idx + 1)

    # ------------------------------------------------------------------
    def _compute_metrics(self) -> None:
        if self.runner.busy:
            return
        page = self.metrics_page
        selected = page.tree.selected()
        if len(selected) < 2:
            QMessageBox.information(self, tr("Pick some metrics"),
                                    tr("Select at least two metrics."))
            return
        self.state.config.selected_features = selected
        self._busy(True, tr("Computing metrics..."))
        self._log(tr("Computing {n} metrics for {t} tracks...").format(
            n=len(selected), t=self.state.spots.n_tracks))

        from .. import pipeline

        def job(progress=None):
            return pipeline.features_from_spots(self.state.spots, self.state.config,
                                                progress=progress)

        self.runner.start(job, self._metrics_done, self._on_progress, self._on_error)

    def _metrics_done(self, features) -> None:
        self._busy(False)
        self.state.features = features
        self.metrics_page.show_features(features)
        n_drop = self.state.spots.n_tracks - len(features.frame)
        text = tr("{n} tracks x {m} metrics").format(
            n=len(features.frame), m=len(features.feature_names))
        if n_drop:
            text += tr(" ({n} track(s) dropped as too short)").format(n=n_drop)
        self._log(text, "ok")

    # ------------------------------------------------------------------
    def _run_analysis(self) -> None:
        if self.runner.busy:
            return
        for page in (self.preprocess_page, self.analysis_page):
            problem = page.validate()
            if problem:
                QMessageBox.information(self, tr("One more thing"), render(problem))
                return
            page.commit()
        self.export_page.commit()          # figure style is needed while rendering

        self._busy(True, tr("Running analysis..."))
        self._log(tr("SOM: {algorithm}, {epochs} epochs; {n} features.").format(
            algorithm=tr(self.state.config.som.algorithm, context="som_algorithm"),
            epochs=self.state.config.som.epochs,
            n=len(self.state.config.selected_features)))

        from .. import pipeline

        def job(progress=None):
            result = pipeline.run_analysis(self.state.features, self.state.config,
                                           progress=progress)
            panels = pipeline.build_figures(result, progress=progress)
            return result, panels

        self.runner.start(job, self._analysis_done, self._on_progress, self._on_error)

    def _analysis_done(self, payload) -> None:
        result, panels = payload
        self._busy(False)
        self.state.result = result
        self.state.panels = panels
        for w in result.warnings:
            self._log(w, "warn")
        q = result.quality
        self._log(
            tr("Done. QE={qe:.3f}  TE={te:.3f}  purity={purity:.3f}  "
               "occupancy={occupancy:.0%}").format(
                qe=q.get("quantisation_error", float("nan")),
                te=q.get("topographic_error", float("nan")),
                purity=q.get("group_purity", float("nan")),
                occupancy=q.get("node_occupancy", float("nan"))), "ok")
        if result.verdict is not None:
            level = {"strong": "ok", "moderate": "ok",
                     "weak": "warn", "none": "warn"}.get(result.verdict.confidence,
                                                         "info")
            self._log(result.verdict.headline, level)
            for line in result.verdict.bullets()[1:4]:
                self._log("  " + render(line))
        self.results_page.show_result(result, panels)
        self.compare_page.show_result(result)
        self._go_to(self.pages.index(self.results_page))

    # ------------------------------------------------------------------
    def _run_scan(self, method: str, criterion: str) -> None:
        """Run a parameter scan in the background, on the matrix already prepared."""
        if self.runner.busy:
            return
        if self.state.result is None:
            QMessageBox.information(
                self, tr("Run the analysis first"),
                tr("A scan re-uses the feature matrix the analysis prepared, so "
                   "there has to be one. Go back to Analysis and press Run."))
            return

        from ..analysis import DataContext, get
        from ..analysis.scan import default_grid, scan as run_scan

        prep = self.state.result.prep
        ctx = DataContext.from_prepared(
            prep, random_state=self.state.config.embedding.random_state)
        ok, why = get(method).usable(ctx)
        if not ok:
            QMessageBox.information(self, tr("Cannot scan that"), render(why))
            return

        grid = default_grid(method, ctx)
        cells = 1
        for values in grid.values():
            cells *= len(values)
        self._busy(True, tr("Scanning {n} settings...").format(n=cells))
        self._log(tr("Scanning {method} over {grid}").format(
            method=get(method).label,
            grid="; ".join(f"{k} = {v}" for k, v in grid.items())))

        def job(progress=None):
            def tick(i, n):
                if progress:
                    progress(tr("Scanning ({i}/{n})").format(i=i, n=n), i / max(n, 1))
            return run_scan(ctx, method, grid, criterion=criterion,
                            reliability_null=8, progress=tick,
                            log=self.state.result.methods_log)

        self.runner.start(job, self._scan_done, self._on_progress, self._on_error)

    def _scan_done(self, scan) -> None:
        self._busy(False)
        self._log(scan.plateau_text(), "ok")
        self.compare_page.show_scan(scan)
        self._go_to(self.pages.index(self.compare_page))

    # ------------------------------------------------------------------
    def _export(self) -> None:
        if self.runner.busy:
            return
        if self.state.result is None:
            QMessageBox.information(self, tr("Nothing to export"),
                                    tr("Run the analysis first."))
            return
        self.export_page.commit()
        out = self.state.config.export.out_dir
        self._busy(True, tr("Exporting to {path}...").format(path=out))
        self._log(tr("Exporting to {path}").format(path=out))

        from .. import pipeline

        result = self.state.result

        def job(progress=None):
            return pipeline.export_all(result, panels=None, progress=progress)

        self.runner.start(job, self._export_done, self._on_progress, self._on_error)

    def _export_done(self, manifest) -> None:
        self._busy(False)
        n_fig = len([k for k in manifest.figures if not k.startswith("_")])
        text = tr("Exported {n} figures, {t} tables").format(
            n=n_fig, t=len(manifest.tables))
        if manifest.videos:
            text += tr(", {n} video").format(n=len(manifest.videos))
        text += tr(" to {path}").format(path=manifest.out_dir)
        self._log(text, "ok")
        for w in self.state.result.warnings[-4:]:
            self._log(w, "warn")
        QMessageBox.information(
            self, tr("Export complete"),
            tr("Wrote {n} figures and {t} tables to\n{path}\n\n"
               "SOMTrack_report.pdf holds every figure with selectable text; "
               "methods.txt is a ready-to-paste methods paragraph. "
               "RESULTS_REPORT.html gives the report in English followed by its "
               "Chinese translation.").format(
                n=n_fig, t=len(manifest.tables), path=manifest.out_dir))

    # ------------------------------------------------------------------
    def _save_config(self) -> None:
        for p in self.pages:
            try:
                p.commit()
            except Exception:
                pass
        path, _ = QFileDialog.getSaveFileName(self, tr("Save settings"),
                                              "somtrack_settings.json",
                                              "JSON (*.json)")
        if not path:
            return
        self.state.config.to_json(path)
        self._log(tr("Settings saved to {path}").format(path=path), "ok")

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, tr("Load settings"), "",
                                              "JSON (*.json)")
        if not path:
            return
        try:
            self.state.config = AnalysisConfig.from_json(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not load settings"),
                                _exception_text(exc))
            return
        for p in self.pages:
            p.state.config = self.state.config
        self._log(tr("Settings loaded from {path}. Re-visit the steps to apply "
                     "them.").format(path=path), "ok")

    # ------------------------------------------------------------------
    def _set_language(self, code: str) -> None:
        """Remember the choice; it takes effect when the window is rebuilt."""
        from .app import save_language

        save_language(code)
        if code == language():
            return
        # Asked in the language being switched *to*, since that is the one
        # the person choosing it reads.
        title = tr("Change language", lang=code)
        text = tr("SOMTrack will use the new language the next time it starts. "
                  "Restart now? Anything not saved will be lost.", lang=code)
        answer = QMessageBox.question(self, title, text)
        if answer != QMessageBox.StandardButton.Yes:
            return
        if self.runner.busy:
            QMessageBox.information(self, title,
                                    tr("A task is still running. Restart SOMTrack "
                                       "when it has finished.", lang=code))
            return
        from PySide6.QtCore import QProcess

        if QProcess.startDetached(sys.executable, ["-m", "somtrack"],
                                  str(Path.cwd()))[0]:
            self.close()
        else:
            QMessageBox.information(self, title,
                                    tr("Please close and reopen SOMTrack.", lang=code))

    def _about(self) -> None:
        QMessageBox.about(
            self, tr("About SOMTrack"),
            f"<b>SOMTrack {__version__}</b><br><br>"
            + tr("A Python refactor of <i>SOM tracking_v1.2b.ijm</i> "
                 "(Wang, Ho &amp; Liao, NTOU 2020).<br><br>"
                 "Computes locomotion and biophysical metrics from tracking "
                 "coordinates, clusters them with self-organising maps and a "
                 "registry of projections, and tests whether the groups really "
                 "differ -- PERMANOVA, PERMDISP, the energy test and "
                 "cross-validated classification against a permutation null -- "
                 "before it lets a figure claim that they do.<br><br>"
                 "Every run writes a conclusion report with a methods paragraph "
                 "and a reference list. Figures export as editable-text PDF/SVG, "
                 "600 dpi PNG and MP4."))

    def closeEvent(self, event) -> None:
        if self.runner.busy:
            answer = QMessageBox.question(
                self, tr("A task is running"),
                tr("Quit anyway and abandon the running task?"))
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        from matplotlib import pyplot as plt

        plt.close("all")
        event.accept()


def _exception_text(exc: Exception) -> str:
    """An exception's message, translated when it carries a translation."""
    return render(exc.args[0]) if len(exc.args) == 1 else str(exc)


__all__ = ["MainWindow", "STEP_TITLES"]
