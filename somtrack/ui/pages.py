"""Wizard pages for the SOMTrack desktop app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QGroupBox, QHBoxLayout,
                               QLabel, QLineEdit, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QSpinBox, QSplitter,
                               QStackedWidget, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from ..config import AnalysisConfig
from ..io_tables import (FeatureDataset, SpotDataset, build_feature_dataset,
                         build_spot_dataset, detect_spot_columns, load_table,
                         numeric_columns)
from .widgets import (ACCENT, CheckListWidget, ColumnMapWidget, DataFrameView,
                      FigureGallery, FileListWidget, MetricTree, h_line,
                      heading, hint)


# ==========================================================================
@dataclass
class AppState:
    """Everything the pages share."""

    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    entry_mode: str = "coordinates"          # or "features"
    spots: SpotDataset | None = None
    features: FeatureDataset | None = None
    result: object | None = None             # pipeline.AnalysisResult
    panels: list = field(default_factory=list)
    raw_preview: pd.DataFrame | None = None
    available_metrics: list[str] = field(default_factory=list)


class Page(QWidget):
    """Base page: a title, a hint line and a body."""

    status = Signal(str, str)               # message, level
    request_next = Signal()

    TITLE = ""
    SUBTITLE = ""

    def __init__(self, state: AppState, parent=None):
        super().__init__(parent)
        self.state = state
        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 8)
        outer.setSpacing(8)
        outer.addWidget(heading(self.TITLE))
        if self.SUBTITLE:
            outer.addWidget(hint(self.SUBTITLE))
        outer.addWidget(h_line())
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body, 1)
        self.build()

    def build(self) -> None: ...
    def on_enter(self) -> None: ...
    def validate(self) -> str | None: return None
    def commit(self) -> None: ...


# ==========================================================================
# 1. Data source
# ==========================================================================
class SourcePage(Page):
    TITLE = "1. Data source"
    SUBTITLE = ("Start from spot-level coordinate tables and let SOMTrack compute the "
                "locomotion metrics, or load a multi-dimensional table you already have.")

    columns_ready = Signal()

    def build(self) -> None:
        picker = QHBoxLayout()
        self.rb_coords = QRadioButton("A - Coordinate tables (compute metrics)")
        self.rb_feats = QRadioButton("B - Multi-dimensional table (skip to clustering)")
        self.rb_coords.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_coords)
        grp.addButton(self.rb_feats)
        picker.addWidget(self.rb_coords)
        picker.addWidget(self.rb_feats)
        picker.addStretch(1)
        self.demo_btn = QPushButton("Load demo data")
        self.demo_btn.setToolTip("Generate a synthetic four-treatment swimming assay "
                                 "so you can try the whole workflow.")
        picker.addWidget(self.demo_btn)
        self.body.addLayout(picker)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_coordinate_page())
        self.stack.addWidget(self._build_feature_page())
        self.body.addWidget(self.stack, 1)

        self.rb_coords.toggled.connect(lambda on: self.stack.setCurrentIndex(0 if on else 1))
        self.demo_btn.clicked.connect(self._load_demo)

    # ------------------------------------------------------------- path A
    def _build_coordinate_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        self.files = FileListWidget()
        self.files.changed.connect(self._refresh_columns)
        lay.addWidget(self.files, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        self.colmap = ColumnMapWidget()
        split.addWidget(self.colmap)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.addWidget(self._build_calibration())
        self.preview = DataFrameView()
        rlay.addWidget(QLabel("First rows of the first file"))
        rlay.addWidget(self.preview, 1)
        split.addWidget(right)
        split.setSizes([360, 620])
        lay.addWidget(split, 2)
        return w

    def _build_calibration(self) -> QGroupBox:
        box = QGroupBox("Calibration and track filtering")
        form = QFormLayout(box)

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(5)
        self.pixel_size.setRange(1e-5, 1e6)
        self.pixel_size.setValue(1.0)
        self.length_unit = QComboBox()
        self.length_unit.addItems(["um", "mm", "cm", "nm", "px"])
        self.length_unit.setEditable(True)
        row = QHBoxLayout()
        row.addWidget(self.pixel_size)
        row.addWidget(QLabel("per pixel, unit"))
        row.addWidget(self.length_unit)
        holder = QWidget(); holder.setLayout(row)
        form.addRow("Spatial scale", holder)

        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setDecimals(5)
        self.frame_interval.setRange(1e-5, 1e6)
        self.frame_interval.setValue(0.1)
        self.time_unit = QComboBox()
        self.time_unit.addItems(["s", "ms", "min"])
        self.time_unit.setEditable(True)
        row2 = QHBoxLayout()
        row2.addWidget(self.frame_interval)
        row2.addWidget(QLabel("per frame, unit"))
        row2.addWidget(self.time_unit)
        holder2 = QWidget(); holder2.setLayout(row2)
        form.addRow("Time scale", holder2)

        self.calibrated = QCheckBox("X/Y and T columns are already in physical units")
        form.addRow("", self.calibrated)

        self.min_spots = QSpinBox()
        self.min_spots.setRange(5, 100000)
        self.min_spots.setValue(20)
        form.addRow("Minimum spots per track", self.min_spots)

        self.trim = QDoubleSpinBox()
        self.trim.setRange(0.1, 1.0)
        self.trim.setSingleStep(0.05)
        self.trim.setValue(0.6)
        self.trim.setToolTip(
            "Central fraction of ranked values kept when averaging (the v1.2 decile "
            "filter). 1.0 uses every value. Trimming resists tracking glitches but "
            "biases the reported SD downwards.")
        form.addRow("Robust trim fraction", self.trim)

        self.auto_halt = QCheckBox("Detect the pause threshold from the speed histogram")
        self.auto_halt.setChecked(True)
        self.auto_halt.setToolTip(
            "v1.2 used a fixed 1.5 pixel step, which does not transfer between "
            "magnifications. This splits the bimodal speed distribution instead.")
        form.addRow("", self.auto_halt)

        self.smoothing = QSpinBox()
        self.smoothing.setRange(0, 51)
        self.smoothing.setSingleStep(2)
        self.smoothing.setValue(0)
        self.smoothing.setToolTip("Savitzky-Golay window in frames; 0 disables smoothing.")
        form.addRow("Coordinate smoothing", self.smoothing)
        return box

    # ------------------------------------------------------------- path B
    def _build_feature_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.feat_path = QLineEdit()
        self.feat_path.setPlaceholderText("Path to a table with one row per sample...")
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse_features)
        row.addWidget(self.feat_path, 1)
        row.addWidget(browse)
        lay.addLayout(row)

        roles = QGroupBox("Column roles")
        rform = QFormLayout(roles)
        self.f_id = QComboBox()
        self.f_group = QComboBox()
        self.f_rep = QComboBox()
        for cb in (self.f_id, self.f_group, self.f_rep):
            cb.setMinimumWidth(200)
        rform.addRow("Sample ID", self.f_id)
        rform.addRow("Group (treatment)", self.f_group)
        rform.addRow("Replicate", self.f_rep)
        lay.addWidget(roles)

        lay.addWidget(QLabel("Feature columns to use"))
        btns = QHBoxLayout()
        all_b = QPushButton("Select all")
        none_b = QPushButton("Select none")
        all_b.clicked.connect(lambda: self.feat_list.set_all(True))
        none_b.clicked.connect(lambda: self.feat_list.set_all(False))
        btns.addWidget(all_b)
        btns.addWidget(none_b)
        btns.addStretch(1)
        lay.addLayout(btns)

        self.feat_list = CheckListWidget()
        self.feat_preview = DataFrameView()
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.feat_list)
        split.addWidget(self.feat_preview)
        split.setSizes([280, 700])
        lay.addWidget(split, 1)
        return w

    def _browse_features(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select a multi-dimensional table", "",
            "Tables (*.csv *.tsv *.txt *.xls *.xlsx *.xlsm);;All files (*)")
        if not path:
            return
        self.feat_path.setText(path)
        self._load_feature_table(path)

    def _load_feature_table(self, path: str):
        try:
            df = load_table(path)
        except Exception as exc:
            QMessageBox.warning(self, "Could not read the table", str(exc))
            return
        self.state.raw_preview = df
        cols = list(df.columns)
        num = numeric_columns(df)
        for cb, guesses in ((self.f_id, ("sample", "id", "track", "name")),
                            (self.f_group, ("group", "cluster", "treatment", "condition")),
                            (self.f_rep, ("replicate", "subset", "rep", "batch"))):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem("(none)")
            cb.addItems(cols)
            for g in guesses:
                hit = next((c for c in cols if str(c).strip().lower() == g), None)
                if hit:
                    cb.setCurrentText(hit)
                    break
            cb.blockSignals(False)
        meta = {self.f_id.currentText(), self.f_group.currentText(), self.f_rep.currentText()}
        self.feat_list.populate([c for c in num if c not in meta])
        self.feat_preview.set_frame(df)
        self.status.emit(f"Loaded {len(df)} rows x {len(cols)} columns from "
                         f"{Path(path).name}", "ok")

    # ------------------------------------------------------------- shared
    def _refresh_columns(self):
        files, _, _ = self.files.entries()
        if not files:
            return
        try:
            df = load_table(files[0])
        except Exception as exc:
            self.status.emit(f"Could not read {Path(files[0]).name}: {exc}", "error")
            return
        detected = detect_spot_columns(df)
        self.colmap.populate(list(df.columns), detected)
        self.preview.set_frame(df)
        self.state.raw_preview = df
        missing = [k for k in ("track", "x", "y", "t") if not getattr(detected, k)]
        if missing:
            self.status.emit("Could not auto-detect: " + ", ".join(missing)
                             + " -- set them in the column mapping.", "warn")
        else:
            self.status.emit(f"{len(files)} file(s); columns detected automatically.", "ok")

    def _load_demo(self):
        from ..demo import write_demo_dataset

        target = Path.home() / "SOMTrack_demo_data"
        self.status.emit("Generating demo data...", "info")
        try:
            paths = write_demo_dataset(target, n_tracks=16, n_frames=300, replicates=2)
        except Exception as exc:
            QMessageBox.warning(self, "Demo data failed", str(exc))
            return
        self.rb_coords.setChecked(True)
        self.files.clear()
        self.files.add_paths([str(p) for p in paths])
        self.pixel_size.setValue(1.6)
        self.frame_interval.setValue(0.05)
        self.status.emit(f"Demo data written to {target}", "ok")

    # ------------------------------------------------------------- wizard
    def validate(self) -> str | None:
        if self.rb_coords.isChecked():
            files, _, _ = self.files.entries()
            if not files:
                return "Add at least one coordinate table."
            cm = self.colmap.to_map()
            if not cm.required_ok():
                return "Track, X, Y and time columns must all be mapped."
            return None
        if not self.feat_path.text().strip():
            return "Choose a multi-dimensional table."
        if len(self.feat_list.selected()) < 2:
            return "Select at least two feature columns."
        return None

    def commit(self) -> None:
        cfg = self.state.config
        if self.rb_coords.isChecked():
            self.state.entry_mode = "coordinates"
            files, groups, reps = self.files.entries()
            cm = self.colmap.to_map()
            self.state.spots = build_spot_dataset(files, cm, groups, reps)
            self.state.features = None

            t = cfg.track
            t.pixel_size = self.pixel_size.value()
            t.length_unit = self.length_unit.currentText()
            t.frame_interval = self.frame_interval.value()
            t.time_unit = self.time_unit.currentText()
            t.coords_are_calibrated = self.calibrated.isChecked()
            t.times_are_calibrated = self.calibrated.isChecked()
            t.min_spots = self.min_spots.value()
            t.trim_fraction = self.trim.value()
            t.use_trimmed_stats = self.trim.value() < 1.0
            t.auto_halt_threshold = self.auto_halt.isChecked()
            t.smoothing_window = self.smoothing.value()

            cols = set(self.state.spots.frame.columns)
            from ..metrics import available_metrics

            self.state.available_metrics = available_metrics(cols)
        else:
            self.state.entry_mode = "features"
            df = self.state.raw_preview
            picked = self.feat_list.selected()
            self.state.features = build_feature_dataset(
                df, picked,
                id_column=_none(self.f_id.currentText()),
                group_column=_none(self.f_group.currentText()),
                replicate_column=_none(self.f_rep.currentText()),
            )
            self.state.spots = None
            self.state.available_metrics = []


def _none(text: str):
    return None if text in ("", "(none)") else text


# ==========================================================================
# 2. Metrics
# ==========================================================================
class MetricsPage(Page):
    TITLE = "2. Locomotion metrics"
    SUBTITLE = ("Choose what to measure on each track. Metrics marked (v1.2) existed in "
                "the ImageJ macro; the rest are new. Metrics whose input columns are "
                "missing from your table are hidden.")

    compute_requested = Signal()

    def build(self) -> None:
        row = QHBoxLayout()
        for label, slot, tip in (
            ("All", lambda: self.tree.set_selection(None),
             "Every metric that your columns support."),
            ("Recommended", lambda: self.tree.set_selection(None, only_defaults=True),
             "A broad, low-redundancy default set."),
            ("v1.2 legacy set", lambda: self.tree.set_selection(None, legacy_only=True),
             "Exactly the metrics the ImageJ macro produced, for direct comparison."),
            ("Kinematics only", lambda: self._preset_categories(
                ["Basic kinematics", "Angular / directional", "Speed distribution"]),
             "Speed, acceleration and turning only."),
            ("None", lambda: self.tree.set_selection(set()), "Clear the selection."),
        ):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        self.count_label = QLabel("")
        row.addWidget(self.count_label)
        self.body.addLayout(row)

        self.tree = MetricTree()
        self.tree.changed.connect(self._update_count)
        self.body.addWidget(self.tree, 3)

        actions = QHBoxLayout()
        self.compute_btn = QPushButton("Compute metrics")
        self.compute_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;padding:6px 18px;"
            "border-radius:4px;font-weight:600;}}")
        self.compute_btn.clicked.connect(self.compute_requested.emit)
        actions.addWidget(self.compute_btn)
        actions.addStretch(1)
        self.body.addLayout(actions)

        self.body.addWidget(QLabel("Computed feature table"))
        self.table = DataFrameView()
        self.body.addWidget(self.table, 2)

    def _preset_categories(self, categories: list[str]) -> None:
        from ..metrics import METRICS

        self.tree.set_selection({n for n, s in METRICS.items() if s.category in categories})

    def _update_count(self):
        n = len(self.tree.selected())
        self.count_label.setText(f"{n} metric(s) selected")
        self.count_label.setStyleSheet("color:#a33;" if n < 3 else "color:#5d6570;")

    def on_enter(self) -> None:
        t = self.state.config.track
        self.tree.populate(self.state.available_metrics, t.length_unit, t.time_unit,
                           preselect=self.state.config.selected_features or None)
        self._update_count()

    def show_features(self, ds: FeatureDataset) -> None:
        self.table.set_frame(ds.frame)

    def validate(self) -> str | None:
        if self.state.features is None:
            return "Press 'Compute metrics' first."
        return None

    def commit(self) -> None:
        pass


# ==========================================================================
# 3. Features and preprocessing
# ==========================================================================
class PreprocessPage(Page):
    TITLE = "3. Feature matrix"
    SUBTITLE = ("Pick the columns that go into the clustering and decide how they are "
                "scaled. Highly correlated metrics let one behavioural axis dominate the "
                "distance, which is a common reason a SOM fails to separate treatments.")

    def build(self) -> None:
        split = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        llay = QVBoxLayout(left)
        llay.setContentsMargins(0, 0, 0, 0)
        btns = QHBoxLayout()
        a = QPushButton("Select all"); a.clicked.connect(lambda: self.feats.set_all(True))
        n = QPushButton("Select none"); n.clicked.connect(lambda: self.feats.set_all(False))
        btns.addWidget(a); btns.addWidget(n); btns.addStretch(1)
        llay.addLayout(btns)
        self.feats = CheckListWidget()
        self.feats.changed.connect(self._update_count)
        llay.addWidget(self.feats, 1)
        self.count = QLabel("")
        llay.addWidget(self.count)
        split.addWidget(left)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox("Preprocessing")
        form = QFormLayout(box)
        self.scaler = QComboBox()
        self.scaler.addItems(["zscore", "minmax", "robust", "rank", "none"])
        self.scaler.setToolTip(
            "z-score puts every metric on equal footing (recommended).\n"
            "min-max reproduces the v1.2 normalisation but is dominated by outliers.\n"
            "robust uses median/IQR; rank is fully non-parametric.")
        form.addRow("Scaling", self.scaler)

        self.nan_policy = QComboBox()
        self.nan_policy.addItems(["impute_median", "drop_sample", "drop_feature"])
        form.addRow("Missing values", self.nan_policy)

        self.collinear = QDoubleSpinBox()
        self.collinear.setRange(0.5, 1.0)
        self.collinear.setSingleStep(0.01)
        self.collinear.setValue(0.98)
        self.collinear.setToolTip("Drop the later of any pair of metrics more correlated "
                                  "than this. 1.00 disables the pruning.")
        form.addRow("Prune |r| above", self.collinear)

        self.winsor = QDoubleSpinBox()
        self.winsor.setRange(0.0, 0.2)
        self.winsor.setSingleStep(0.01)
        self.winsor.setValue(0.0)
        self.winsor.setToolTip("Clip each metric to this quantile at both tails; "
                               "0 disables clipping.")
        form.addRow("Winsorise tails", self.winsor)

        self.drop_const = QCheckBox("Drop constant metrics")
        self.drop_const.setChecked(True)
        form.addRow("", self.drop_const)
        rlay.addWidget(box)

        self.summary = DataFrameView()
        rlay.addWidget(QLabel("Descriptive statistics (original units)"))
        rlay.addWidget(self.summary, 1)
        split.addWidget(right)
        split.setSizes([300, 700])
        self.body.addWidget(split, 1)

    def _update_count(self):
        n = len(self.feats.selected())
        self.count.setText(f"{n} feature(s) selected")
        self.count.setStyleSheet("color:#a33;" if n < 2 else "color:#5d6570;")

    def on_enter(self) -> None:
        ds = self.state.features
        if ds is None:
            return
        self.feats.populate(ds.feature_names)
        rows = []
        for f in ds.feature_names:
            v = pd.to_numeric(ds.frame[f], errors="coerce").to_numpy(float)
            rows.append({
                "feature": f, "unit": ds.units.get(f, ""),
                "n": int(np.isfinite(v).sum()),
                "mean": np.nanmean(v) if np.isfinite(v).any() else np.nan,
                "sd": np.nanstd(v) if np.isfinite(v).any() else np.nan,
                "min": np.nanmin(v) if np.isfinite(v).any() else np.nan,
                "max": np.nanmax(v) if np.isfinite(v).any() else np.nan,
            })
        self.summary.set_frame(pd.DataFrame(rows))
        self._update_count()

    def validate(self) -> str | None:
        if len(self.feats.selected()) < 2:
            return "Select at least two features."
        return None

    def commit(self) -> None:
        cfg = self.state.config
        cfg.selected_features = self.feats.selected()
        p = cfg.preprocess
        p.scaler = self.scaler.currentText()
        p.nan_policy = self.nan_policy.currentText()
        p.collinearity_threshold = self.collinear.value()
        p.winsorise_quantile = self.winsor.value()
        p.drop_constant = self.drop_const.isChecked()


# ==========================================================================
# 4. Clustering
# ==========================================================================
class ClusterPage(Page):
    """One decision at the top, everything else folded away underneath.

    The page opens on a single choice -- what are you trying to find out -- and
    a Run button.  A user who wants nothing else never sees a parameter.  The
    method list and its settings are built from the registry, so every method
    appears with its own controls, its own plain-language summary and, where it
    applies, its own warning, without this file knowing anything about it.
    """

    TITLE = "4. Analysis"
    SUBTITLE = ("Pick what you are trying to find out. Everything below has a "
                "sensible default; open a section only if you want to change it.")

    run_requested = Signal()

    def build(self) -> None:
        from .. import recipes
        from ..analysis import FAMILY_LABELS, all_methods, by_family
        from .forms import Collapsible, MethodPanel, section_label

        self._panels: dict[str, MethodPanel] = {}
        self._boxes: dict[str, QCheckBox] = {}

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(2, 2, 12, 2)
        lay.setSpacing(8)

        # ---- level 1: the only decision most users need to make -----------
        row = QHBoxLayout()
        row.addWidget(QLabel("What do you want to find out?"))
        self.recipe = QComboBox()
        for r in recipes.RECIPES:
            self.recipe.addItem(r.title, r.key)
        self.recipe.setMinimumWidth(300)
        self.recipe.currentIndexChanged.connect(self._recipe_changed)
        row.addWidget(self.recipe)
        row.addStretch(1)
        lay.addLayout(row)

        self.recipe_note = QLabel("")
        self.recipe_note.setWordWrap(True)
        self.recipe_note.setStyleSheet(
            "background:#f3f6fa;border-left:3px solid %s;padding:8px 10px;"
            "color:#374151;" % ACCENT)
        lay.addWidget(self.recipe_note)

        # ---- level 2: what will run, as a list you can override -----------
        methods_box = Collapsible("Which projections to run")
        mlay = QVBoxLayout()
        mlay.setContentsMargins(4, 2, 4, 2)
        mlay.setSpacing(2)

        for family, specs in by_family().items():
            mlay.addWidget(section_label(FAMILY_LABELS.get(family, family)))
            if family == "supervised":
                warn = QLabel(
                    "\u26a0  These use the group labels to build the axes, so "
                    "they separate the groups whatever the data says \u2014 they "
                    "would separate random noise too. They are drawn out-of-fold "
                    "as well, and that panel is the one to believe.")
                warn.setWordWrap(True)
                warn.setStyleSheet("color:#8a3b00;font-size:11px;padding:0 4px 4px;")
                mlay.addWidget(warn)
                self.supervised_ack = QCheckBox(
                    "I understand supervised projections separate groups by "
                    "construction")
                self.supervised_ack.toggled.connect(self._supervised_toggled)
                mlay.addWidget(self.supervised_ack)

            for spec in specs:
                box = QCheckBox(spec.label)
                box.setToolTip(spec.summary)
                ok, why = True, ""
                if not spec.available():
                    ok = False
                    why = spec.install_hint or "not installed"
                if not ok:
                    box.setEnabled(False)
                    box.setText(f"{spec.label}   \u2014 {why}")
                box.toggled.connect(self._selection_changed)
                self._boxes[spec.key] = box
                mlay.addWidget(box)

                panel = MethodPanel(spec)
                panel.setVisible(False)
                self._panels[spec.key] = panel
                mlay.addWidget(panel)

        methods_box.set_layout(mlay)
        lay.addWidget(methods_box)

        # ---- level 2: statistics ------------------------------------------
        stats_box = Collapsible("How to test whether the groups differ")
        lay.addWidget(stats_box)
        stats_box.set_layout(self._stats_form())

        # ---- level 3: the map and the codebook ----------------------------
        som_box = Collapsible("Self-organising map")
        som_box.set_layout(self._som_form())
        lay.addWidget(som_box)

        nodes_box = Collapsible("Behavioural clusters on the map")
        nodes_box.set_layout(self._nodes_form())
        lay.addWidget(nodes_box)

        lay.addStretch(1)
        scroll.setWidget(inner)
        self.body.addWidget(scroll, 1)

        run_row = QHBoxLayout()
        self.run_btn = QPushButton("Run analysis")
        self.run_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;padding:8px 26px;"
            "border-radius:4px;font-weight:600;}}")
        self.run_btn.clicked.connect(self.run_requested.emit)
        run_row.addWidget(self.run_btn)
        self.plan_label = QLabel("")
        self.plan_label.setStyleSheet("color:#6b7280;")
        run_row.addWidget(self.plan_label)
        run_row.addStretch(1)
        self.body.addLayout(run_row)

        self._recipe_changed()

    # ------------------------------------------------------------------
    def _stats_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(20, 2, 4, 8)

        self.stats_on = QCheckBox(
            "Test whether the groups differ, and by how much")
        self.stats_on.setChecked(True)
        self.stats_on.setToolTip(
            "PERMANOVA and PERMDISP on the sample distances, the energy test, "
            "and cross-validated group assignment with a permutation null.")
        form.addRow("", self.stats_on)

        self.unit = QComboBox()
        self.unit.addItem("each sample is independent", "sample")
        self.unit.addItem("average within each replicate first", "replicate")
        self.unit.setToolTip(
            "If several samples came from one dish, clutch or imaging session, "
            "they are not independent measurements. SOMTrack detects the "
            "replicate structure and blocks the permutations and the "
            "cross-validation folds by it either way; this setting decides "
            "whether to go further and average within each replicate, which is "
            "the most conservative option.")
        form.addRow("Experimental unit", self.unit)

        self.distance = QComboBox()
        self.distance.addItems(["euclidean", "correlation", "cityblock", "cosine"])
        self.distance.setToolTip(
            "Used by PERMANOVA, PERMDISP and the MDS projection, so the picture "
            "and the test describe the same thing.")
        form.addRow("Distance between samples", self.distance)

        self.classifier = QComboBox()
        for key, label in (("lda_shrinkage", "linear discriminant (regularised)"),
                           ("svm_linear", "linear support vector machine"),
                           ("plsda", "PLS-DA"),
                           ("logistic", "logistic regression"),
                           ("random_forest", "random forest")):
            self.classifier.addItem(label, key)
        form.addRow("Assignment model", self.classifier)

        self.permutations = QSpinBox()
        self.permutations.setRange(0, 99999)
        self.permutations.setSingleStep(100)
        self.permutations.setValue(999)
        self.permutations.setToolTip(
            "How many times the group labels are shuffled to work out what this "
            "analysis achieves when there is nothing to find. 999 supports a "
            "smallest p value of 0.001; fewer is faster and less precise.")
        form.addRow("Label shuffles", self.permutations)

        form.addRow("", hint(
            "Significance is judged by shuffling the labels, not by a binomial "
            "test on the cross-validated accuracy \u2014 that test is "
            "anti-conservative and reports differences that are not there."))
        return form

    def _som_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(20, 2, 4, 8)

        self.algo = QComboBox()
        self.algo.addItems(["batch", "supervised", "relevance", "online", "growing"])
        self.algo.setCurrentText("batch")
        self.algo.setToolTip(
            "batch      -- fast, deterministic, the standard choice.\n"
            "supervised -- XY-fused SOM; pulls the map towards separating your "
            "groups, and so cannot be used as evidence that they differ.\n"
            "relevance  -- learns a weight per metric (GRLVQ) so uninformative "
            "metrics stop diluting the distance.\n"
            "online     -- the sequential rule the ImageJ macro used.\n"
            "growing    -- grows the map where quantisation error is high.")
        self.algo.currentTextChanged.connect(self._algo_changed)
        form.addRow("Algorithm", self.algo)

        self.som_warning = QLabel("")
        self.som_warning.setWordWrap(True)
        self.som_warning.setStyleSheet("color:#8a3b00;font-size:11px;")
        self.som_warning.setVisible(False)
        form.addRow("", self.som_warning)

        self.auto_size = QCheckBox("Choose the map size automatically (5*sqrt(N))")
        self.auto_size.setChecked(True)
        self.auto_size.toggled.connect(self._auto_size_toggled)
        form.addRow("", self.auto_size)

        self.map_w = QSpinBox(); self.map_w.setRange(2, 100); self.map_w.setValue(9)
        self.map_h = QSpinBox(); self.map_h.setRange(2, 100); self.map_h.setValue(6)
        size_row = QHBoxLayout()
        size_row.addWidget(self.map_w); size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.map_h); size_row.addStretch(1)
        holder = QWidget(); holder.setLayout(size_row)
        form.addRow("Map size", holder)
        self._auto_size_toggled(True)

        self.topology = QComboBox()
        self.topology.addItems(["hex", "rect"])
        form.addRow("Lattice", self.topology)

        self.toroidal = QCheckBox("Toroidal boundary (removes edge effects)")
        form.addRow("", self.toroidal)

        self.epochs = QSpinBox(); self.epochs.setRange(10, 20000)
        self.epochs.setValue(200)
        form.addRow("Training epochs", self.epochs)

        self.init = QComboBox()
        self.init.addItems(["pca", "sample", "random"])
        self.init.setToolTip("PCA initialisation is deterministic and converges "
                             "faster than the random start used in v1.2.")
        form.addRow("Initialisation", self.init)

        self.lr = QDoubleSpinBox(); self.lr.setRange(0.01, 1.0)
        self.lr.setSingleStep(0.05); self.lr.setValue(0.5)
        form.addRow("Learning rate", self.lr)

        self.tau = QDoubleSpinBox(); self.tau.setRange(0.02, 1.0)
        self.tau.setSingleStep(0.05); self.tau.setValue(0.25)
        form.addRow("Neighbourhood time constant", self.tau)

        self.label_weight = QDoubleSpinBox()
        self.label_weight.setRange(0.0, 0.95)
        self.label_weight.setSingleStep(0.05); self.label_weight.setValue(0.35)
        self.label_weight.setToolTip(
            "Supervised SOM only. 0 ignores the labels entirely (= batch SOM); "
            "high values force separation and overstate it. Report whatever you "
            "used; 0.2-0.4 is a defensible range.")
        form.addRow("Label weight (supervised)", self.label_weight)

        self.spread = QDoubleSpinBox()
        self.spread.setRange(0.05, 0.95); self.spread.setSingleStep(0.05)
        self.spread.setValue(0.5)
        form.addRow("Spread factor (growing)", self.spread)

        self.record = QSpinBox(); self.record.setRange(1, 200); self.record.setValue(5)
        form.addRow("Snapshot interval", self.record)

        self.seed = QSpinBox(); self.seed.setRange(0, 99999); self.seed.setValue(0)
        form.addRow("Random seed", self.seed)
        self._algo_changed(self.algo.currentText())
        return form

    def _nodes_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(20, 2, 4, 8)
        self.nc_enabled = QCheckBox("Divide the map into behavioural clusters")
        self.nc_enabled.setChecked(True)
        form.addRow("", self.nc_enabled)

        self.nc_method = QComboBox()
        self.nc_method.addItems(["kmeans", "ward", "gmm"])
        form.addRow("Method", self.nc_method)

        self.k_min = QSpinBox(); self.k_min.setRange(2, 40); self.k_min.setValue(2)
        self.k_max = QSpinBox(); self.k_max.setRange(2, 40); self.k_max.setValue(8)
        row = QHBoxLayout()
        row.addWidget(self.k_min); row.addWidget(QLabel("to"))
        row.addWidget(self.k_max); row.addStretch(1)
        holder = QWidget(); holder.setLayout(row)
        form.addRow("Try cluster counts", holder)

        self.weight_hits = QCheckBox("Weight nodes by their sample count")
        self.weight_hits.setChecked(True)
        self.weight_hits.setToolTip(
            "Stops empty codebook vectors, which the SOM only dragged along "
            "behind their neighbours, from defining a cluster.")
        form.addRow("", self.weight_hits)
        form.addRow("", hint(
            "The best count is chosen by the average rank of the silhouette, "
            "Davies-Bouldin and Calinski-Harabasz indices; every count in the "
            "range is still saved."))
        return form

    # ------------------------------------------------------------------
    def _recipe_changed(self, *_) -> None:
        from .. import recipes

        key = self.recipe.currentData() or recipes.DEFAULT
        r = recipes.get(key)
        self.recipe_note.setText(f"<b>{r.summary}</b><br>{r.detail}")

        cfg = r.build(AnalysisConfig())
        wanted = set(cfg.embedding.resolved_methods())
        for mkey, box in self._boxes.items():
            if box.isEnabled():
                box.blockSignals(True)
                box.setChecked(mkey in wanted)
                box.blockSignals(False)
        if hasattr(self, "supervised_ack"):
            self.supervised_ack.setChecked(cfg.embedding.allow_supervised)
        self.stats_on.setChecked(cfg.stats.enabled)
        self.distance.setCurrentText(cfg.stats.distance)
        self.permutations.setValue(cfg.stats.n_permutations)
        i = self.classifier.findData(cfg.stats.classifier)
        if i >= 0:
            self.classifier.setCurrentIndex(i)
        self.algo.setCurrentText(cfg.som.algorithm)
        self.init.setCurrentText(cfg.som.init)
        self._selection_changed()

    def _supervised_toggled(self, on: bool) -> None:
        from ..analysis import get as get_method

        for key, box in self._boxes.items():
            if get_method(key).supervised:
                box.setEnabled(on and get_method(key).available())
                if not on:
                    box.setChecked(False)
        self._selection_changed()

    def _selection_changed(self, *_) -> None:
        chosen = []
        for key, box in self._boxes.items():
            on = box.isChecked() and box.isEnabled()
            self._panels[key].setVisible(on)
            if on:
                chosen.append(key)
        n = len(chosen)
        bits = [f"{n} projection{'s' if n != 1 else ''}"]
        if self.stats_on.isChecked():
            bits.append(f"{self.permutations.value()} label shuffles")
        self.plan_label.setText("will run: " + ", ".join(bits))

    def _auto_size_toggled(self, on: bool) -> None:
        self.map_w.setEnabled(not on)
        self.map_h.setEnabled(not on)

    def _algo_changed(self, algo: str) -> None:
        # Hide what does not apply rather than greying it: a disabled control is
        # still something the eye has to read and dismiss.
        self.lr.setVisible(algo in ("online", "growing"))
        self.label_weight.setVisible(algo == "supervised")
        self.spread.setVisible(algo == "growing")
        for w in (self.auto_size, self.map_w, self.map_h, self.toroidal):
            w.setVisible(algo != "growing")
        if algo != "growing":
            self._auto_size_toggled(self.auto_size.isChecked())

        supervised = algo == "supervised"
        self.som_warning.setVisible(supervised)
        if supervised:
            self.som_warning.setText(
                "\u26a0  This map is trained with the group labels, so it "
                "separates the groups by construction. Report the label weight, "
                "and take the evidence from the cross-validated statistics "
                "rather than from the map.")

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        from .forms import build_context

        ds = self.state.features
        if ds is not None:
            n = len(ds.frame)
            w, h = self.state.config.som.resolved_size(n)
            if self.auto_size.isChecked():
                self.map_w.setValue(w)
                self.map_h.setValue(h)

        ctx = build_context(self.state)
        for panel in self._panels.values():
            panel.refresh_suggestions(ctx)
        self._selection_changed()

    def commit(self) -> None:
        cfg = self.state.config
        cfg.report.recipe = self.recipe.currentData() or "standard"

        e = cfg.embedding
        e.methods = [k for k, b in self._boxes.items()
                     if b.isChecked() and b.isEnabled()]
        e.allow_supervised = bool(getattr(self, "supervised_ack", None)
                                  and self.supervised_ack.isChecked())
        e.overrides = {k: ov for k in e.methods
                       if (ov := self._panels[k].overrides())}
        e.random_state = self.seed.value()
        # keep the 2.0 flags consistent so an exported config still means the
        # same thing if it is opened by an older build
        e.run_pca = "pca" in e.methods
        e.run_tsne = "tsne" in e.methods
        e.run_umap = "umap" in e.methods

        st = cfg.stats
        st.enabled = self.stats_on.isChecked()
        st.unit_of_analysis = self.unit.currentData() or "sample"
        st.distance = self.distance.currentText()
        st.classifier = self.classifier.currentData() or "lda_shrinkage"
        st.n_permutations = self.permutations.value()
        st.classification_permutations = self.permutations.value()
        st.random_state = self.seed.value()

        s = cfg.som
        s.algorithm = self.algo.currentText()
        s.width = 0 if self.auto_size.isChecked() else self.map_w.value()
        s.height = 0 if self.auto_size.isChecked() else self.map_h.value()
        s.topology = self.topology.currentText()
        s.toroidal = self.toroidal.isChecked()
        s.epochs = self.epochs.value()
        s.init = self.init.currentText()
        s.learning_rate = self.lr.value()
        s.time_constant = self.tau.value()
        s.label_weight = self.label_weight.value()
        s.spread_factor = self.spread.value()
        s.record_every = self.record.value()
        s.random_state = self.seed.value()

        n = cfg.node_cluster
        n.enabled = self.nc_enabled.isChecked()
        n.method = self.nc_method.currentText()
        n.k_min = min(self.k_min.value(), self.k_max.value())
        n.k_max = max(self.k_min.value(), self.k_max.value())
        n.weight_by_hits = self.weight_hits.isChecked()
        n.random_state = self.seed.value()


# ==========================================================================
# 5. Results
# ==========================================================================
class ResultsPage(Page):
    """The conclusion first, then the evidence, then everything else.

    Version 2.0 opened on a gallery of figures and left the reader to work out
    what they meant.  The first thing here is a sentence: whether the groups
    differ, how strongly, and what the result does *not* say.  The figures and
    tables are still one click away, and the reader who wants to disagree with
    the sentence can find every number behind it.
    """

    TITLE = "5. Results"
    SUBTITLE = ("The conclusion is generated from the numbers, so it cannot "
                "disagree with the tables behind it. Every figure carries its own "
                "caption; use the toolbar above a figure to pan, zoom or save it.")

    def build(self) -> None:
        self.tabs = QTabWidget()

        self.verdict = QTextBrowser()
        self.verdict.setOpenExternalLinks(True)
        self.verdict.setStyleSheet(
            "QTextBrowser{background:#ffffff;border:1px solid #e5e7eb;"
            "border-radius:4px;padding:10px;}")
        self.tabs.addTab(self._wrap(
            self.verdict,
            "What this analysis found, and what it does not say."), "Conclusion")

        self.gallery = FigureGallery()
        self.tabs.addTab(self.gallery, "Figures")

        self.separation = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.separation,
            "Multivariate tests. PERMANOVA asks whether the group averages "
            "differ; PERMDISP asks the separate question of whether the groups "
            "differ in spread. A significant PERMANOVA with a significant "
            "PERMDISP may mean 'more variable', not 'different'."),
            "Group difference")

        self.drivers = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.drivers,
            "Which metrics carry the difference. Read the activation column, "
            "not the weight: a large weight can belong to a metric that carries "
            "no group information and only cancels noise in another."),
            "Key metrics")

        self.projq = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.projq,
            "How much of each projection can be believed. Higher R_NX area "
            "preserves more neighbourhoods; a high unreliable fraction means "
            "that picture should be read for broad structure only."),
            "Projection quality")

        self.quality = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.quality, "Map quality and clustering diagnostics"), "Map quality")

        self.assoc = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.assoc, "Per-metric association with the experimental group, "
                        "sorted by effect size"), "Metric statistics")

        self.assign = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.assign, "Which node each sample landed on"), "Assignments")
        self.body.addWidget(self.tabs, 1)

    @staticmethod
    def _wrap(widget: QWidget, caption: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.addWidget(hint(caption))
        lay.addWidget(widget, 1)
        return w

    # ------------------------------------------------------------------
    def show_result(self, result, panels) -> None:
        self.gallery.set_panels(panels)
        self.verdict.setHtml(self._verdict_html(result))
        self.tabs.setCurrentIndex(0)

        q = pd.DataFrame([{"diagnostic": k.replace("_", " "), "value": v}
                          for k, v in result.quality.items()])
        self.quality.set_frame(q)

        tables = result.tables()
        for view, key in ((self.separation, "separation_tests"),
                          (self.drivers, "classifier_weights"),
                          (self.projq, "projection_quality"),
                          (self.assign, "sample_assignment")):
            if key in tables:
                view.set_frame(tables[key])

        if "permutation_importance" in tables and "classifier_weights" not in tables:
            self.drivers.set_frame(tables["permutation_importance"])

        if result.associations is not None:
            cols = ["feature", "unit", "eta2", "anova_F", "anova_q",
                    "kruskal_q", "max_abs_cohen_d"]
            df = result.associations.univariate
            self.assoc.set_frame(df[[c for c in cols if c in df.columns]])

    @staticmethod
    def _verdict_html(result) -> str:
        import html as _h

        v = getattr(result, "verdict", None)
        if v is None:
            return ("<p style='color:#6b7280'>The statistics layer was switched "
                    "off, so there is no conclusion to report. Turn it on under "
                    "<i>How to test whether the groups differ</i> and run again.</p>")

        accent = {"strong": "#0072B2", "moderate": "#009E73", "weak": "#E69F00",
                  "none": "#7F7F7F"}.get(v.confidence, "#7F7F7F")
        out = [f"<h2 style='color:{accent};margin:0 0 10px'>"
               f"{_h.escape(v.headline)}</h2>"]
        for para in v.findings:
            out.append(f"<p>{_h.escape(para)}</p>")
        if v.separable:
            out.append("<p><b>Separates:</b> "
                       + _h.escape("; ".join(v.separable)) + "</p>")
        if v.not_separable:
            out.append("<p><b>Does not separate:</b> "
                       + _h.escape("; ".join(v.not_separable)) + "</p>")
        if v.drivers:
            out.append("<p><b>Main contributing metrics:</b> "
                       + _h.escape(", ".join(v.drivers)) + "</p>")
        if v.design_note:
            out.append(f"<p style='color:#4b5563'>{_h.escape(v.design_note)}</p>")
        if v.caveats:
            out.append("<h3 style='margin:14px 0 4px;font-size:13px'>"
                       "Limits of this analysis</h3><ul>")
            out += [f"<li style='color:#4b5563'>{_h.escape(c)}</li>"
                    for c in v.caveats]
            out.append("</ul>")
        out.append("<p style='color:#6b7280;font-size:11px;margin-top:16px'>"
                   "The full report, including the methods paragraph and the "
                   "reference list for every algorithm this run used, is written "
                   "to RESULTS_REPORT.html and methods.txt when you export.</p>")
        return "".join(out)


# ==========================================================================
# 6. Compare
# ==========================================================================
class ComparePage(Page):
    """Every method side by side, and a scan that shows where the answer is flat.

    This page exists to make cherry-picking hard.  With a dozen projections
    available the temptation is to run them all and show the one that separates
    best; a grid of everything that ran, each with its own quality score, takes
    that option away without needing an argument about it.

    The scan does the same for hyper-parameters.  It reports the *plateau* --
    every setting within tolerance of the best -- because the single best cell is
    almost never meaningfully better than its neighbours, and saying so is what
    stops a user tuning until the picture agrees with their hypothesis.
    """

    TITLE = "6. Compare"
    SUBTITLE = ("Structure that survives several projections is a property of your "
                "data. Structure that appears in only one is a property of that "
                "algorithm.")

    scan_requested = Signal(str, str)        # method key, criterion

    def build(self) -> None:
        self.tabs = QTabWidget()

        self.grid_gallery = FigureGallery()
        self.tabs.addTab(self._wrap(
            self.grid_gallery,
            "Each panel is the same samples under a different projection, with "
            "the area under its R_NX curve and the share of points it placed "
            "unreliably."), "Methods side by side")

        self.quality = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.quality,
            "Higher R_NX area preserves more neighbourhoods. A supervised method "
            "will look good here for the wrong reason \u2014 it was given the "
            "labels."), "Quality table")

        self.tabs.addTab(self._scan_tab(), "Parameter scan")
        self.body.addWidget(self.tabs, 1)

    @staticmethod
    def _wrap(widget: QWidget, caption: str) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.addWidget(hint(caption))
        lay.addWidget(widget, 1)
        return w

    # ------------------------------------------------------------------
    def _scan_tab(self) -> QWidget:
        from ..analysis.scan import CRITERIA

        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 6, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(hint(
            "Run one method across a range of settings and see whether the "
            "answer changes. A wide plateau is the useful result: it means the "
            "setting does not matter for your data, which is worth reporting."))

        row = QHBoxLayout()
        row.addWidget(QLabel("Scan"))
        self.scan_method = QComboBox()
        self.scan_method.setMinimumWidth(180)
        row.addWidget(self.scan_method)

        row.addWidget(QLabel("scored by"))
        self.scan_criterion = QComboBox()
        for key, (_, _, needs_labels, description) in CRITERIA.items():
            self.scan_criterion.addItem(description, key)
        self.scan_criterion.setMinimumWidth(280)
        self.scan_criterion.setToolTip(
            "'Unreliably placed points' is the criterion of Xia, Lee & Li "
            "(2024): it turns 'what perplexity should I use' into a question "
            "with an answer. Scoring by how well the known groups separate is a "
            "supervised choice, and the report says so.")
        row.addWidget(self.scan_criterion)

        self.scan_btn = QPushButton("Run scan")
        self.scan_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;padding:6px 18px;"
            "border-radius:4px;font-weight:600;}}")
        self.scan_btn.clicked.connect(self._emit_scan)
        row.addWidget(self.scan_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self.plateau = QLabel("")
        self.plateau.setWordWrap(True)
        self.plateau.setStyleSheet(
            "background:#f3f6fa;border-left:3px solid %s;padding:8px 10px;"
            "color:#374151;" % ACCENT)
        self.plateau.setVisible(False)
        lay.addWidget(self.plateau)

        split = QSplitter(Qt.Orientation.Vertical)
        self.scan_gallery = FigureGallery()
        split.addWidget(self.scan_gallery)
        self.scan_table = DataFrameView()
        split.addWidget(self.scan_table)
        split.setSizes([420, 200])
        lay.addWidget(split, 1)

        self.adopt_btn = QPushButton("Use these settings for the next run")
        self.adopt_btn.setEnabled(False)
        self.adopt_btn.setToolTip(
            "Pins the chosen values as explicit overrides. They will appear in "
            "the methods section as a choice you made.")
        self.adopt_btn.clicked.connect(self._adopt)
        lay.addWidget(self.adopt_btn)
        return w

    def _emit_scan(self) -> None:
        key = self.scan_method.currentData()
        if key:
            self.scan_requested.emit(key, self.scan_criterion.currentData()
                                     or "rnx_auc")

    # ------------------------------------------------------------------
    def on_enter(self) -> None:
        from ..analysis import all_methods

        current = self.scan_method.currentData()
        self.scan_method.clear()
        result = self.state.result
        ran = set(getattr(result, "projections", {}) or {})
        for spec in all_methods(available_only=True):
            if not spec.params or not any(p.scan_default for p in spec.params):
                continue
            mark = "" if spec.key in ran else "  (not in the last run)"
            self.scan_method.addItem(spec.label + mark, spec.key)
        i = self.scan_method.findData(current)
        if i >= 0:
            self.scan_method.setCurrentIndex(i)

    def show_result(self, result) -> None:
        """Draw the comparison panels for a finished run."""
        from .. import viz
        from ..analysis import quality_table

        self._scan = None
        self.adopt_btn.setEnabled(False)
        self.plateau.setVisible(False)
        self.scan_gallery.clear()

        if not result.projections:
            self.grid_gallery.clear()
            return

        cfg = result.config.figure
        viz.apply_style(cfg)
        panels = []
        for name, fn, args in (
            ("method_grid", viz.plot_method_grid,
             (result.projections, result.prep, cfg)),
            ("rnx_curves", viz.plot_rnx_curves, (result.projections, cfg)),
            ("projection_agreement", viz.plot_agreement,
             (result.projection_agreement, cfg)),
        ):
            try:
                panels.append((name, fn(*args)))
            except Exception as exc:
                self.status.emit(f"Comparison figure '{name}' skipped: {exc}",
                                 "warn")
        for key, proj in result.projections.items():
            if proj.dubious is not None and proj.n_dubious:
                try:
                    panels.append((f"reliability_{key}",
                                   viz.plot_point_reliability(proj, result.prep, cfg)))
                except Exception:
                    pass
        self.grid_gallery.set_panels(panels)
        self.quality.set_frame(quality_table(result.projections).round(4))
        self.on_enter()

    def show_scan(self, scan) -> None:
        """Draw a finished parameter scan."""
        from .. import viz

        self._scan = scan
        cfg = self.state.config.figure
        viz.apply_style(cfg)

        panels = []
        for name, fn in (("scan_surface", lambda: viz.plot_scan_surface(scan, cfg)),
                         ("scan_layouts",
                          lambda: viz.plot_scan_thumbnails(scan, self.state.result.prep,
                                                           cfg))):
            try:
                panels.append((name, fn()))
            except Exception as exc:
                self.status.emit(f"Scan figure '{name}' skipped: {exc}", "warn")
        self.scan_gallery.set_panels(panels)

        cols = list(dict.fromkeys(
            [c for c in scan.keys + [scan.criterion, "rnx_auc", "dubious_fraction",
                                     "group_silhouette"]
             if c in scan.rows.columns]))
        self.scan_table.set_frame(scan.rows[cols].round(4))
        self.plateau.setText(scan.plateau_text())
        self.plateau.setVisible(True)
        self.adopt_btn.setEnabled(bool(scan.best))
        self.tabs.setCurrentIndex(2)

    def _adopt(self) -> None:
        scan = getattr(self, "_scan", None)
        if scan is None or not scan.best:
            return
        params = scan.best_params()
        cfg = self.state.config.embedding
        cfg.overrides.setdefault(scan.method, {}).update(params)
        if scan.method not in cfg.methods:
            cfg.methods = list(cfg.methods) + [scan.method]
        bits = ", ".join(f"{k} = {v:g}" if isinstance(v, float) else f"{k} = {v}"
                         for k, v in params.items())
        self.status.emit(
            f"{scan.method_label} will use {bits} on the next run, and the "
            f"methods section will record it as your choice.", "ok")
        self.adopt_btn.setEnabled(False)


# ==========================================================================
# 7. Export
# ==========================================================================
class ExportPage(Page):
    TITLE = "6. Export"
    SUBTITLE = ("PDF and SVG keep every label as editable text, so figures can be "
                "restyled in Illustrator or Inkscape without re-running the analysis.")

    export_requested = Signal()

    def build(self) -> None:
        row = QHBoxLayout()
        self.out_dir = QLineEdit(str(Path.home() / "SOMTrack_output"))
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row.addWidget(QLabel("Output folder"))
        row.addWidget(self.out_dir, 1)
        row.addWidget(browse)
        self.body.addLayout(row)

        grid = QGridLayout()
        box = QGroupBox("Formats")
        box.setLayout(grid)
        self.png = QCheckBox("PNG (raster, 600 dpi)"); self.png.setChecked(True)
        self.pdf = QCheckBox("PDF (vector, editable text)"); self.pdf.setChecked(True)
        self.svg = QCheckBox("SVG (vector, editable text)")
        self.mp4 = QCheckBox("MP4 training animation"); self.mp4.setChecked(True)
        self.tables = QCheckBox("CSV tables + one .xlsx workbook"); self.tables.setChecked(True)
        for i, c in enumerate((self.png, self.pdf, self.svg, self.mp4, self.tables)):
            grid.addWidget(c, i // 2, i % 2)
        self.body.addWidget(box)

        from ..viz import ffmpeg_available

        if not ffmpeg_available():
            self.body.addWidget(hint("ffmpeg was not found on PATH; the animation will "
                                     "fall back to an animated GIF."))

        fig_box = QGroupBox("Figure style")
        form = QFormLayout(fig_box)
        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(60.0, 400.0); self.width_mm.setValue(180.0)
        self.width_mm.setToolTip("180 mm is a typical double-column width; 90 mm single.")
        form.addRow("Figure width (mm)", self.width_mm)

        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(5.0, 20.0); self.font_size.setValue(8.0)
        form.addRow("Base font size (pt)", self.font_size)

        self.font_family = QLineEdit("Arial")
        form.addRow("Font family", self.font_family)

        self.dpi = QSpinBox(); self.dpi.setRange(72, 1200); self.dpi.setValue(600)
        form.addRow("PNG resolution (dpi)", self.dpi)

        self.palette = QComboBox()
        self.palette.addItems(["somtrack", "tab10", "legacy_hsb"])
        self.palette.setToolTip("'somtrack' is the Okabe-Ito colour-blind-safe set; "
                                "'legacy_hsb' reproduces the v1.2 hue wheel.")
        form.addRow("Group palette", self.palette)

        self.captions = QCheckBox("Draw a figure legend under every panel")
        self.captions.setChecked(True)
        form.addRow("", self.captions)
        self.body.addWidget(fig_box)

        alpha_box = QGroupBox("Per-group map transparency")
        aform = QFormLayout(alpha_box)
        self.a0 = QSpinBox(); self.a0.setRange(0, 100); self.a0.setValue(20)
        self.a1 = QSpinBox(); self.a1.setRange(0, 100); self.a1.setValue(40)
        self.a2 = QSpinBox(); self.a2.setRange(0, 100); self.a2.setValue(60)
        for lab, sp in (("0 samples", self.a0), ("1 sample", self.a1), ("2 samples", self.a2)):
            sp.setSuffix(" % opaque")
            aform.addRow(lab, sp)
        aform.addRow(">= 3 samples", QLabel("100 % opaque (fixed)"))
        self.body.addWidget(alpha_box)

        row2 = QHBoxLayout()
        self.export_btn = QPushButton("Export everything")
        self.export_btn.setStyleSheet(
            f"QPushButton{{background:{ACCENT};color:white;padding:8px 26px;"
            "border-radius:4px;font-weight:600;}}")
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.open_btn = QPushButton("Open output folder")
        self.open_btn.clicked.connect(self._open_folder)
        row2.addWidget(self.export_btn)
        row2.addWidget(self.open_btn)
        row2.addStretch(1)
        self.body.addLayout(row2)
        self.body.addStretch(1)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Output folder", self.out_dir.text())
        if d:
            self.out_dir.setText(d)

    def _open_folder(self):
        import os
        import subprocess
        import sys

        path = Path(self.out_dir.text())
        path.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)                       # noqa: S606
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def commit(self) -> None:
        cfg = self.state.config
        e = cfg.export
        e.out_dir = Path(self.out_dir.text())
        e.png = self.png.isChecked()
        e.pdf = self.pdf.isChecked()
        e.svg = self.svg.isChecked()
        e.mp4 = self.mp4.isChecked()
        e.save_tables = self.tables.isChecked()

        f = cfg.figure
        f.figure_width_mm = self.width_mm.value()
        f.base_font_size = self.font_size.value()
        f.font_family = self.font_family.text() or "Arial"
        f.dpi_png = self.dpi.value()
        f.palette = self.palette.currentText()
        f.show_caption = self.captions.isChecked()
        f.group_alpha_by_hits = {0: self.a0.value() / 100,
                                 1: self.a1.value() / 100,
                                 2: self.a2.value() / 100}


__all__ = ["AppState", "Page", "SourcePage", "MetricsPage", "PreprocessPage",
           "ClusterPage", "ResultsPage", "ExportPage"]
