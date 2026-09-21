"""The SOMTrack main window: a six-step wizard with a persistent status pane."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox,
                               QProgressBar, QPushButton, QSplitter,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..config import AnalysisConfig
from .pages import (AppState, ClusterPage, ComparePage, ExportPage,
                    MetricsPage, PreprocessPage, ResultsPage, SourcePage)
from .widgets import ACCENT, LogPane, heading
from .workers import TaskRunner

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
        self.setWindowTitle("SOMTrack 3.0 - locomotion metrics, multivariate analysis and publication figures")
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
        sub = QLabel("v2.0  ·  locomotion SOM")
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
            self.steps.addItem(QListWidgetItem(t))
        self.steps.currentRowChanged.connect(self._sidebar_clicked)
        self.steps.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.steps.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        row_h = self.steps.sizeHintForRow(0) if self.steps.count() else 34
        self.steps.setFixedHeight(row_h * self.steps.count() + 8)
        lay.addWidget(self.steps)
        lay.addStretch(1)

        self.log = LogPane()
        self.log.setMinimumHeight(190)
        lay.addWidget(QLabel("Log"))
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
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color:#5d6570;")
        lay.addWidget(self.progress)
        lay.addWidget(self.status_label, 1)

        self.back_btn = QPushButton("Back")
        self.next_btn = QPushButton("Next")
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
        m = self.menuBar().addMenu("&File")
        for text, slot, shortcut in (
            ("Save settings...", self._save_config, QKeySequence.StandardKey.Save),
            ("Load settings...", self._load_config, QKeySequence.StandardKey.Open),
        ):
            a = QAction(text, self)
            a.setShortcut(shortcut)
            a.triggered.connect(slot)
            m.addAction(a)
        m.addSeparator()
        quit_a = QAction("Quit", self)
        quit_a.setShortcut(QKeySequence.StandardKey.Quit)
        quit_a.triggered.connect(self.close)
        m.addAction(quit_a)

        h = self.menuBar().addMenu("&Help")
        about = QAction("About SOMTrack", self)
        about.triggered.connect(self._about)
        h.addAction(about)

    def _wire(self) -> None:
        self.compare_page.scan_requested.connect(self._run_scan)
        self.metrics_page.compute_requested.connect(self._compute_metrics)
        self.analysis_page.run_requested.connect(self._run_analysis)
        self.export_page.export_requested.connect(self._export)

    # ------------------------------------------------------------------
    def _log(self, message: str, level: str = "info") -> None:
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
            self.status_label.setText(message)

    def _on_progress(self, message: str, fraction: float) -> None:
        self.progress.setValue(int(fraction * 100))
        if message:
            self.status_label.setText(message)

    def _on_error(self, message: str, tb: str) -> None:
        self._busy(False)
        self._log(message, "error")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Something went wrong")
        box.setText(message)
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
        self.next_btn.setText("Finish" if index == len(self.pages) - 1 else "Next")
        self.next_btn.setEnabled(index < len(self.pages) - 1)
        self.pages[index].on_enter()

    def _next(self) -> None:
        page = self.pages[self.stack.currentIndex()]
        problem = page.validate()
        if problem:
            self._log(problem, "warn")
            QMessageBox.information(self, "One more thing", problem)
            return
        try:
            page.commit()
        except Exception as exc:
            self._on_error(str(exc), "")
            return

        idx = self.stack.currentIndex()
        if idx == 0 and self.state.entry_mode == "features":
            self._log(f"Loaded {len(self.state.features.frame)} samples, "
                      f"{len(self.state.features.feature_names)} features.", "ok")
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
            QMessageBox.information(self, "Pick some metrics",
                                    "Select at least two metrics.")
            return
        self.state.config.selected_features = selected
        self._busy(True, "Computing metrics...")
        self._log(f"Computing {len(selected)} metrics for "
                  f"{self.state.spots.n_tracks} tracks...")

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
        self._log(f"{len(features.frame)} tracks x {len(features.feature_names)} metrics"
                  + (f" ({n_drop} track(s) dropped as too short)" if n_drop else ""), "ok")

    # ------------------------------------------------------------------
    def _run_analysis(self) -> None:
        if self.runner.busy:
            return
        for page in (self.preprocess_page, self.analysis_page):
            problem = page.validate()
            if problem:
                QMessageBox.information(self, "One more thing", problem)
                return
            page.commit()
        self.export_page.commit()          # figure style is needed while rendering

        self._busy(True, "Running analysis...")
        self._log(f"SOM: {self.state.config.som.algorithm}, "
                  f"{self.state.config.som.epochs} epochs; "
                  f"{len(self.state.config.selected_features)} features.")

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
            "Done. QE={:.3f}  TE={:.3f}  purity={:.3f}  occupancy={:.0%}".format(
                q.get("quantisation_error", float("nan")),
                q.get("topographic_error", float("nan")),
                q.get("group_purity", float("nan")),
                q.get("node_occupancy", float("nan"))), "ok")
        if result.verdict is not None:
            level = {"strong": "ok", "moderate": "ok",
                     "weak": "warn", "none": "warn"}.get(result.verdict.confidence,
                                                         "info")
            self._log(result.verdict.headline, level)
            for line in result.verdict.bullets()[1:4]:
                self._log("  " + line)
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
                self, "Run the analysis first",
                "A scan re-uses the feature matrix the analysis prepared, so "
                "there has to be one. Go back to Analysis and press Run.")
            return

        from ..analysis import DataContext, get
        from ..analysis.scan import default_grid, scan as run_scan

        prep = self.state.result.prep
        ctx = DataContext.from_prepared(
            prep, random_state=self.state.config.embedding.random_state)
        ok, why = get(method).usable(ctx)
        if not ok:
            QMessageBox.information(self, "Cannot scan that", why)
            return

        grid = default_grid(method, ctx)
        cells = 1
        for values in grid.values():
            cells *= len(values)
        self._busy(True, f"Scanning {cells} settings...")
        self._log(f"Scanning {get(method).label} over "
                  + "; ".join(f"{k} = {v}" for k, v in grid.items()))

        def job(progress=None):
            def tick(i, n):
                if progress:
                    progress(f"Scanning ({i}/{n})", i / max(n, 1))
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
            QMessageBox.information(self, "Nothing to export", "Run the analysis first.")
            return
        self.export_page.commit()
        out = self.state.config.export.out_dir
        self._busy(True, f"Exporting to {out}...")
        self._log(f"Exporting to {out}")

        from .. import pipeline

        result = self.state.result

        def job(progress=None):
            return pipeline.export_all(result, panels=None, progress=progress)

        self.runner.start(job, self._export_done, self._on_progress, self._on_error)

    def _export_done(self, manifest) -> None:
        self._busy(False)
        n_fig = len([k for k in manifest.figures if not k.startswith("_")])
        self._log(f"Exported {n_fig} figures, {len(manifest.tables)} tables"
                  + (f", {len(manifest.videos)} video" if manifest.videos else "")
                  + f" to {manifest.out_dir}", "ok")
        for w in self.state.result.warnings[-4:]:
            self._log(w, "warn")
        QMessageBox.information(
            self, "Export complete",
            f"Wrote {n_fig} figures and {len(manifest.tables)} tables to\n"
            f"{manifest.out_dir}\n\n"
            "SOMTrack_report.pdf holds every figure with selectable text; "
            "methods.txt is a ready-to-paste methods paragraph.")

    # ------------------------------------------------------------------
    def _save_config(self) -> None:
        for p in self.pages:
            try:
                p.commit()
            except Exception:
                pass
        path, _ = QFileDialog.getSaveFileName(self, "Save settings",
                                              "somtrack_settings.json", "JSON (*.json)")
        if not path:
            return
        self.state.config.to_json(path)
        self._log(f"Settings saved to {path}", "ok")

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "JSON (*.json)")
        if not path:
            return
        try:
            self.state.config = AnalysisConfig.from_json(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not load settings", str(exc))
            return
        for p in self.pages:
            p.state.config = self.state.config
        self._log(f"Settings loaded from {path}. Re-visit the steps to apply them.", "ok")

    def _about(self) -> None:
        QMessageBox.about(
            self, "About SOMTrack",
            "<b>SOMTrack 2.0</b><br><br>"
            "A Python refactor of <i>SOM tracking_v1.2b.ijm</i> "
            "(Wang, Ho &amp; Liao, NTOU 2020).<br><br>"
            "Computes locomotion and biophysical metrics from tracking coordinates, "
            "clusters them with self-organising maps (batch, supervised XY-fused, "
            "relevance-learning and growing variants) alongside PCA, t-SNE and UMAP, "
            "and reports which metrics actually drive the group separation.<br><br>"
            "Figures export as editable-text PDF/SVG, 600 dpi PNG and MP4.")

    def closeEvent(self, event) -> None:
        if self.runner.busy:
            answer = QMessageBox.question(
                self, "A task is running", "Quit anyway and abandon the running task?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        from matplotlib import pyplot as plt

        plt.close("all")
        event.accept()


__all__ = ["MainWindow"]
