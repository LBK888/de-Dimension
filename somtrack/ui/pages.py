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
from ..i18n import join_clauses, join_list, render, tr
from ..io_tables import (FeatureDataset, SpotDataset, build_feature_dataset,
                         build_spot_dataset, detect_spot_columns, load_table,
                         numeric_columns)
from . import theme
from .widgets import (CheckListWidget, ColumnMapWidget, DataFrameView,
                      FigureGallery, FileListWidget, MetricTree, add_row,
                      h_line, heading, help_label, hint, is_none_text,
                      none_text, rich_tip, set_tip, styled)


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
    """Base page: a title, a hint line and a body.

    ``TITLE`` and ``SUBTITLE`` are English source strings, translated when the
    page is built.
    """

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
        outer.addWidget(heading(tr(self.TITLE)))
        if self.SUBTITLE:
            outer.addWidget(hint(tr(self.SUBTITLE)))
        outer.addWidget(h_line())
        self.body = QVBoxLayout()
        self.body.setSpacing(10)
        outer.addLayout(self.body, 1)
        self.build()

    def build(self) -> None: ...
    def on_enter(self) -> None: ...
    def validate(self) -> str | None: return None
    def commit(self) -> None: ...


def _option_combo(values, context: str) -> QComboBox:
    """A combo whose items show translated labels and carry the config value.

    The value is what the configuration, the JSON file and the methods section
    record; the label is only what the user reads.
    """
    cb = QComboBox()
    for v in values:
        cb.addItem(tr(v, context=context), v)
    return cb


def _select(cb: QComboBox, value) -> None:
    i = cb.findData(value)
    if i >= 0:
        cb.setCurrentIndex(i)


def _value(cb: QComboBox, default: str = "") -> str:
    data = cb.currentData()
    return data if data is not None else (cb.currentText() or default)


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
        self.rb_coords = QRadioButton(tr("A - Coordinate tables (compute metrics)"))
        self.rb_feats = QRadioButton(tr("B - Multi-dimensional table (skip to "
                                        "clustering)"))
        set_tip(self.rb_coords, tr(
            "Use this if you have a position for every frame from a tracker, such "
            "as the TrackMate spots table. SOMTrack works out the locomotion "
            "metrics for you."))
        set_tip(self.rb_feats, tr(
            "Use this if you already have one row per animal or track, with the "
            "measurements in columns. The metrics step is skipped."))
        self.rb_coords.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_coords)
        grp.addButton(self.rb_feats)
        picker.addWidget(self.rb_coords)
        picker.addWidget(self.rb_feats)
        picker.addStretch(1)
        self.demo_btn = QPushButton(tr("Load demo data"))
        set_tip(self.demo_btn, tr("Generate a synthetic four-treatment swimming "
                                  "assay so you can try the whole workflow."))
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
        rlay.addWidget(help_label(tr("First rows of the first file"), tr(
            "A preview of the first table, to check that the columns were read "
            "correctly.")))
        rlay.addWidget(self.preview, 1)
        split.addWidget(right)
        split.setSizes([360, 620])
        lay.addWidget(split, 2)
        return w

    def _build_calibration(self) -> QGroupBox:
        box = QGroupBox(tr("Calibration and track filtering"))
        form = QFormLayout(box)

        self.pixel_size = QDoubleSpinBox()
        self.pixel_size.setDecimals(5)
        self.pixel_size.setRange(1e-5, 1e6)
        self.pixel_size.setValue(1.0)
        self.length_unit = QComboBox()
        self.length_unit.addItems(["um", "mm", "cm", "nm", "px"])
        self.length_unit.setEditable(True)
        scale_tip = tr("How long one pixel is in real units, for example 1.6 um per "
                       "pixel. Every distance and speed is multiplied by it.")
        row = QHBoxLayout()
        row.addWidget(self.pixel_size)
        row.addWidget(help_label(tr("per pixel, unit"), scale_tip))
        row.addWidget(self.length_unit)
        holder = QWidget(); holder.setLayout(row)
        set_tip(self.pixel_size, scale_tip)
        set_tip(self.length_unit, tr("The unit lengths are reported in. Type your "
                                     "own if it is not in the list."))
        add_row(form, tr("Spatial scale"), holder, scale_tip)

        self.frame_interval = QDoubleSpinBox()
        self.frame_interval.setDecimals(5)
        self.frame_interval.setRange(1e-5, 1e6)
        self.frame_interval.setValue(0.1)
        self.time_unit = QComboBox()
        self.time_unit.addItems(["s", "ms", "min"])
        self.time_unit.setEditable(True)
        time_tip = tr("How long one frame lasts, for example 0.05 s when filming at "
                      "20 frames per second. Every time and speed uses it.")
        row2 = QHBoxLayout()
        row2.addWidget(self.frame_interval)
        row2.addWidget(help_label(tr("per frame, unit"), time_tip))
        row2.addWidget(self.time_unit)
        holder2 = QWidget(); holder2.setLayout(row2)
        set_tip(self.frame_interval, time_tip)
        set_tip(self.time_unit, tr("The unit times are reported in. Type your own "
                                   "if it is not in the list."))
        add_row(form, tr("Time scale"), holder2, time_tip)

        self.calibrated = QCheckBox(tr("X/Y and T columns are already in physical units"))
        set_tip(self.calibrated, tr(
            "Tick this if the table already gives positions in um or mm and time "
            "in seconds. The two scales above are then ignored."))
        form.addRow("", self.calibrated)

        self.min_spots = QSpinBox()
        self.min_spots.setRange(5, 100000)
        self.min_spots.setValue(20)
        add_row(form, tr("Minimum spots per track"), self.min_spots, tr(
            "Tracks with fewer detected points than this are left out: a very "
            "short track gives unreliable metrics."))

        self.trim = QDoubleSpinBox()
        self.trim.setRange(0.1, 1.0)
        self.trim.setSingleStep(0.05)
        self.trim.setValue(0.6)
        add_row(form, tr("Robust trim fraction"), self.trim, tr(
            "Central fraction of ranked values kept when averaging (the v1.2 decile "
            "filter). 1.0 uses every value. Trimming resists tracking glitches but "
            "biases the reported SD downwards."))

        self.auto_halt = QCheckBox(tr("Detect the pause threshold from the speed "
                                      "histogram"))
        self.auto_halt.setChecked(True)
        set_tip(self.auto_halt, tr(
            "v1.2 used a fixed 1.5 pixel step, which does not transfer between "
            "magnifications. This splits the bimodal speed distribution instead."))
        form.addRow("", self.auto_halt)

        self.smoothing = QSpinBox()
        self.smoothing.setRange(0, 51)
        self.smoothing.setSingleStep(2)
        self.smoothing.setValue(0)
        add_row(form, tr("Coordinate smoothing"), self.smoothing,
                tr("Savitzky-Golay window in frames; 0 disables smoothing."))
        return box

    # ------------------------------------------------------------- path B
    def _build_feature_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.feat_path = QLineEdit()
        self.feat_path.setPlaceholderText(tr("Path to a table with one row per "
                                             "sample..."))
        browse = QPushButton(tr("Browse..."))
        browse.clicked.connect(self._browse_features)
        row.addWidget(self.feat_path, 1)
        row.addWidget(browse)
        lay.addLayout(row)

        roles = QGroupBox(tr("Column roles"))
        rform = QFormLayout(roles)
        self.f_id = QComboBox()
        self.f_group = QComboBox()
        self.f_rep = QComboBox()
        for cb in (self.f_id, self.f_group, self.f_rep):
            cb.setMinimumWidth(200)
        add_row(rform, tr("Sample ID"), self.f_id, tr(
            "The column that names each row, i.e. each animal or track. Optional: "
            "rows are numbered if it is (none)."))
        add_row(rform, tr("Group (treatment)"), self.f_group, tr(
            "The column with the treatment of each row. The groups are what the "
            "statistics compare."))
        add_row(rform, tr("Replicate"), self.f_rep, tr(
            "The column saying which experiment, day, dish or batch each row came "
            "from. Optional, but it tells the statistics which samples are not "
            "independent of each other."))
        lay.addWidget(roles)

        lay.addWidget(help_label(tr("Feature columns to use"), tr(
            "The numeric columns to analyse. Leave out ID numbers, frame counts and "
            "anything else that is not a measurement.")))
        btns = QHBoxLayout()
        all_b = QPushButton(tr("Select all"))
        none_b = QPushButton(tr("Select none"))
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
            self, tr("Select a multi-dimensional table"), "",
            tr("Tables") + " (*.csv *.tsv *.txt *.xls *.xlsx *.xlsm);;"
            + tr("All files") + " (*)")
        if not path:
            return
        self.feat_path.setText(path)
        self._load_feature_table(path)

    def _load_feature_table(self, path: str):
        try:
            df = load_table(path)
        except Exception as exc:
            QMessageBox.warning(self, tr("Could not read the table"), str(exc))
            return
        self.state.raw_preview = df
        cols = list(df.columns)
        num = numeric_columns(df)
        for cb, guesses in ((self.f_id, ("sample", "id", "track", "name")),
                            (self.f_group, ("group", "cluster", "treatment", "condition")),
                            (self.f_rep, ("replicate", "subset", "rep", "batch"))):
            cb.blockSignals(True)
            cb.clear()
            cb.addItem(none_text())
            cb.addItems([str(c) for c in cols])
            for g in guesses:
                hit = next((c for c in cols if str(c).strip().lower() == g), None)
                if hit:
                    cb.setCurrentText(str(hit))
                    break
            cb.blockSignals(False)
        meta = {self.f_id.currentText(), self.f_group.currentText(), self.f_rep.currentText()}
        self.feat_list.populate([c for c in num if c not in meta])
        self.feat_preview.set_frame(df)
        self.status.emit(tr("Loaded {rows} rows x {cols} columns from {file}").format(
            rows=len(df), cols=len(cols), file=Path(path).name), "ok")

    # ------------------------------------------------------------- shared
    def _refresh_columns(self):
        files, _, _ = self.files.entries()
        if not files:
            return
        try:
            df = load_table(files[0])
        except Exception as exc:
            self.status.emit(tr("Could not read {file}: {error}").format(
                file=Path(files[0]).name, error=exc), "error")
            return
        detected = detect_spot_columns(df)
        self.colmap.populate(list(df.columns), detected)
        self.preview.set_frame(df)
        self.state.raw_preview = df
        missing = [k for k in ("track", "x", "y", "t") if not getattr(detected, k)]
        if missing:
            self.status.emit(tr("Could not auto-detect: {columns} -- set them in the "
                                "column mapping.").format(columns=", ".join(missing)),
                             "warn")
        else:
            self.status.emit(tr("{n} file(s); columns detected automatically.").format(
                n=len(files)), "ok")

    def _load_demo(self):
        from ..demo import write_demo_dataset

        target = Path.home() / "SOMTrack_demo_data"
        self.status.emit(tr("Generating demo data..."), "info")
        try:
            paths = write_demo_dataset(target, n_tracks=16, n_frames=300, replicates=2)
        except Exception as exc:
            QMessageBox.warning(self, tr("Demo data failed"), str(exc))
            return
        self.rb_coords.setChecked(True)
        self.files.clear()
        self.files.add_paths([str(p) for p in paths])
        self.pixel_size.setValue(1.6)
        self.frame_interval.setValue(0.05)
        self.status.emit(tr("Demo data written to {path}").format(path=target), "ok")

    # ------------------------------------------------------------- wizard
    def validate(self) -> str | None:
        if self.rb_coords.isChecked():
            files, _, _ = self.files.entries()
            if not files:
                return tr("Add at least one coordinate table.")
            cm = self.colmap.to_map()
            if not cm.required_ok():
                return tr("Track, X, Y and time columns must all be mapped.")
            return None
        if not self.feat_path.text().strip():
            return tr("Choose a multi-dimensional table.")
        if len(self.feat_list.selected()) < 2:
            return tr("Select at least two feature columns.")
        return None

    def load_config(self, cfg: AnalysisConfig) -> None:
        t = cfg.track
        self.pixel_size.setValue(t.pixel_size)
        self.length_unit.setCurrentText(t.length_unit)
        self.frame_interval.setValue(t.frame_interval)
        self.time_unit.setCurrentText(t.time_unit)
        self.calibrated.setChecked(t.coords_are_calibrated)
        self.min_spots.setValue(t.min_spots)
        self.trim.setValue(t.trim_fraction)
        self.auto_halt.setChecked(t.auto_halt_threshold)
        self.smoothing.setValue(t.smoothing_window)

    def _inputs(self) -> tuple:
        """Everything the computed metrics depend on, for spotting a change."""
        files, groups, reps = self.files.entries()
        cm = self.colmap.to_map()
        return (tuple(files), tuple(map(str, groups)), tuple(reps),
                tuple(sorted(vars(cm).items(), key=lambda kv: kv[0])),
                self.pixel_size.value(), self.length_unit.currentText(),
                self.frame_interval.value(), self.time_unit.currentText(),
                self.calibrated.isChecked(), self.min_spots.value(),
                self.trim.value(), self.auto_halt.isChecked(),
                self.smoothing.value())

    def commit(self) -> None:
        cfg = self.state.config
        if self.rb_coords.isChecked():
            # Passing through this step again (Back, or a click in the sidebar)
            # must not throw away metrics that were already computed from the
            # very same files and settings.
            inputs = self._inputs()
            unchanged = (self.state.entry_mode == "coordinates"
                         and self.state.spots is not None
                         and inputs == getattr(self, "_committed", None))
            self.state.entry_mode = "coordinates"
            if not unchanged:
                files, groups, reps = self.files.entries()
                cm = self.colmap.to_map()
                self.state.spots = build_spot_dataset(files, cm, groups, reps)
                self.state.features = None
            self._committed = inputs

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
    return None if is_none_text(text) else text


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
            b = QPushButton(tr(label, context="metric_preset"))
            set_tip(b, tr(tip))
            b.clicked.connect(slot)
            row.addWidget(b)
        row.addStretch(1)
        self.count_label = styled(QLabel(""), "count")
        row.addWidget(self.count_label)
        self.body.addLayout(row)

        self.tree = MetricTree()
        self.tree.changed.connect(self._update_count)
        self.body.addWidget(self.tree, 3)

        actions = QHBoxLayout()
        self.compute_btn = styled(QPushButton(tr("Compute metrics")), "primary")
        set_tip(self.compute_btn, tr(
            "Works out every ticked metric for every track. Needed before the "
            "next step; run it again after changing the selection."))
        self.compute_btn.clicked.connect(self.compute_requested.emit)
        actions.addWidget(self.compute_btn)
        actions.addStretch(1)
        self.body.addLayout(actions)

        self.body.addWidget(QLabel(tr("Computed feature table")))
        self.table = DataFrameView()
        self.body.addWidget(self.table, 2)

    def _preset_categories(self, categories: list[str]) -> None:
        from ..metrics import METRICS

        self.tree.set_selection({n for n, s in METRICS.items() if s.category in categories})

    def _update_count(self):
        n = len(self.tree.selected())
        self.count_label.setText(tr("{n} metric(s) selected").format(n=n))
        theme.set_role(self.count_label, "count", state="bad" if n < 3 else "")

    def on_enter(self) -> None:
        # Rebuild the list only when the columns or the units changed, so the
        # ticks someone set here survive a visit to a later step and back.
        t = self.state.config.track
        key = (tuple(self.state.available_metrics), t.length_unit, t.time_unit)
        if key != getattr(self, "_populated_for", None):
            self.tree.populate(self.state.available_metrics, t.length_unit,
                               t.time_unit,
                               preselect=self.state.config.selected_features or None)
            self._populated_for = key
        self._update_count()

    def show_features(self, ds: FeatureDataset) -> None:
        self.table.set_frame(ds.frame)

    def validate(self) -> str | None:
        if self.state.features is None:
            return tr("Press 'Compute metrics' first.")
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
        a = QPushButton(tr("Select all")); a.clicked.connect(lambda: self.feats.set_all(True))
        n = QPushButton(tr("Select none")); n.clicked.connect(lambda: self.feats.set_all(False))
        btns.addWidget(a); btns.addWidget(n); btns.addStretch(1)
        llay.addLayout(btns)
        self.feats = CheckListWidget()
        set_tip(self.feats, tr("Ticked metrics go into the map, the projections and "
                               "the statistics."))
        self.feats.changed.connect(self._update_count)
        llay.addWidget(self.feats, 1)
        self.count = styled(QLabel(""), "count")
        llay.addWidget(self.count)
        split.addWidget(left)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)

        box = QGroupBox(tr("Preprocessing"))
        form = QFormLayout(box)
        self.scaler = _option_combo(["zscore", "minmax", "robust", "rank", "none"],
                                    "scaler")
        add_row(form, tr("Scaling"), self.scaler, tr(
            "z-score puts every metric on equal footing (recommended).\n"
            "min-max reproduces the v1.2 normalisation but is dominated by outliers.\n"
            "robust uses median/IQR; rank is fully non-parametric."))

        self.nan_policy = _option_combo(["impute_median", "drop_sample", "drop_feature"],
                                        "nan_policy")
        add_row(form, tr("Missing values"), self.nan_policy, tr(
            "What to do when a metric could not be worked out for a sample. "
            "Filling in the median keeps every sample; the other two choices "
            "drop the sample, or the whole metric, instead."))

        self.collinear = QDoubleSpinBox()
        self.collinear.setRange(0.5, 1.0)
        self.collinear.setSingleStep(0.01)
        self.collinear.setValue(0.98)
        add_row(form, tr("Prune |r| above"), self.collinear,
                tr("Drop the later of any pair of metrics more correlated than this. "
                   "1.00 disables the pruning."))

        self.winsor = QDoubleSpinBox()
        self.winsor.setRange(0.0, 0.2)
        self.winsor.setSingleStep(0.01)
        self.winsor.setValue(0.0)
        add_row(form, tr("Winsorise tails"), self.winsor,
                tr("Clip each metric to this quantile at both tails; 0 disables "
                   "clipping."))

        self.drop_const = QCheckBox(tr("Drop constant metrics"))
        self.drop_const.setChecked(True)
        set_tip(self.drop_const, tr("A metric with the same value for every sample "
                                    "carries no information and is removed."))
        form.addRow("", self.drop_const)
        rlay.addWidget(box)

        self.summary = DataFrameView()
        rlay.addWidget(help_label(tr("Descriptive statistics (original units)"), tr(
            "Mean, spread and range of every metric before scaling, to spot a "
            "metric that is mostly missing or has an odd range.")))
        rlay.addWidget(self.summary, 1)
        split.addWidget(right)
        split.setSizes([300, 700])
        self.body.addWidget(split, 1)

    def _update_count(self):
        n = len(self.feats.selected())
        self.count.setText(tr("{n} feature(s) selected").format(n=n))
        theme.set_role(self.count, "count", state="bad" if n < 2 else "")

    def on_enter(self) -> None:
        ds = self.state.features
        if ds is None:
            return
        # Keep the columns that were picked last time instead of re-ticking
        # everything on every visit, which silently undid the choice.
        kept = set(self.state.config.selected_features) & set(ds.feature_names)
        self.feats.populate(ds.feature_names, kept or None)
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
            return tr("Select at least two features.")
        return None

    def load_config(self, cfg: AnalysisConfig) -> None:
        p = cfg.preprocess
        _select(self.scaler, p.scaler)
        _select(self.nan_policy, p.nan_policy)
        self.collinear.setValue(p.collinearity_threshold)
        self.winsor.setValue(p.winsorise_quantile)
        self.drop_const.setChecked(p.drop_constant)

    def commit(self) -> None:
        cfg = self.state.config
        cfg.selected_features = self.feats.selected()
        p = cfg.preprocess
        p.scaler = _value(self.scaler, "zscore")
        p.nan_policy = _value(self.nan_policy, "impute_median")
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
        from ..analysis import FAMILY_LABELS, by_family
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
        recipe_tip = tr(
            "Each choice fills in a ready-made set of methods and settings for that "
            "question. It is only a starting point: everything below can still be "
            "changed.")
        row = QHBoxLayout()
        row.addWidget(help_label(tr("What do you want to find out?"), recipe_tip))
        self.recipe = QComboBox()
        set_tip(self.recipe, recipe_tip)
        for r in recipes.RECIPES:
            self.recipe.addItem(tr(r.title), r.key)
        self.recipe.setMinimumWidth(300)
        self.recipe.currentIndexChanged.connect(self._recipe_changed)
        row.addWidget(self.recipe)
        row.addStretch(1)
        lay.addLayout(row)

        self.recipe_note = styled(QLabel(""), "callout")
        self.recipe_note.setWordWrap(True)
        lay.addWidget(self.recipe_note)

        # ---- level 2: what will run, as a list you can override -----------
        methods_box = Collapsible(
            "Which projections to run",
            tip="A projection draws the many metrics of every sample as one point "
                "on a flat picture. Each method draws it in its own way; tick the "
                "ones to run, and rest the mouse on a name to read what it does.")
        mlay = QVBoxLayout()
        mlay.setContentsMargins(4, 2, 4, 2)
        mlay.setSpacing(2)

        for family, specs in by_family().items():
            mlay.addWidget(section_label(tr(FAMILY_LABELS.get(family, family))))
            if family == "supervised":
                warn = styled(QLabel(""), "warning")
                warn.setText("⚠  " + tr(
                    "These use the group labels to build the axes, so they separate "
                    "the groups whatever the data says — they would separate "
                    "random noise too. They are drawn out-of-fold as well, and that "
                    "panel is the one to believe."))
                warn.setWordWrap(True)
                mlay.addWidget(warn)
                self.supervised_ack = QCheckBox(tr(
                    "I understand supervised projections separate groups by "
                    "construction"))
                set_tip(self.supervised_ack, tr(
                    "Tick this to allow the supervised projections below. They show "
                    "how groups you have already shown to differ are arranged; they "
                    "cannot show that the groups differ."))
                self.supervised_ack.toggled.connect(self._supervised_toggled)
                mlay.addWidget(self.supervised_ack)

            for spec in specs:
                box = QCheckBox(spec.label)
                set_tip(box, tr(spec.summary))
                ok, why = True, ""
                if not spec.available():
                    import sys

                    ok = False
                    why = tr(spec.install_hint) if spec.install_hint else tr(
                        "not installed")
                    if getattr(sys, "frozen", False):   # pip means nothing here
                        why = tr("not included in the Windows executable")
                if not ok:
                    box.setEnabled(False)
                    box.setText(f"{spec.label}   — {why}")
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
        stats_box = Collapsible(
            "How to test whether the groups differ",
            tip="The statistical tests that decide whether the groups really "
                "differ, and how much. This is what the conclusion is based on.")
        lay.addWidget(stats_box)
        stats_box.set_layout(self._stats_form())

        # ---- level 3: the map and the codebook ----------------------------
        som_box = Collapsible(
            "Self-organising map",
            tip="Settings of the SOM, which lays similar samples next to each "
                "other on a grid of nodes.")
        som_box.set_layout(self._som_form())
        lay.addWidget(som_box)

        nodes_box = Collapsible(
            "Behavioural clusters on the map",
            tip="Groups neighbouring nodes of the map into a few types of "
                "behaviour.")
        nodes_box.set_layout(self._nodes_form())
        lay.addWidget(nodes_box)

        lay.addStretch(1)
        scroll.setWidget(inner)
        self.body.addWidget(scroll, 1)

        run_row = QHBoxLayout()
        self.run_btn = styled(QPushButton(tr("Run analysis")), "primary",
                              size="large")
        self.run_btn.clicked.connect(self.run_requested.emit)
        run_row.addWidget(self.run_btn)
        self.plan_label = styled(QLabel(""), "hint")
        run_row.addWidget(self.plan_label)
        run_row.addStretch(1)
        self.body.addLayout(run_row)

        self._recipe_changed()

    # ------------------------------------------------------------------
    def _stats_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(20, 2, 4, 8)

        self.stats_on = QCheckBox(tr("Test whether the groups differ, and by how much"))
        self.stats_on.setChecked(True)
        set_tip(self.stats_on, tr(
            "PERMANOVA and PERMDISP on the sample distances, the energy test, "
            "and cross-validated group assignment with a permutation null."))
        form.addRow("", self.stats_on)

        self.unit = QComboBox()
        self.unit.addItem(tr("each sample is independent"), "sample")
        self.unit.addItem(tr("average within each replicate first"), "replicate")
        add_row(form, tr("Experimental unit"), self.unit, tr(
            "If several samples came from one dish, clutch or imaging session, "
            "they are not independent measurements. SOMTrack detects the "
            "replicate structure and blocks the permutations and the "
            "cross-validation folds by it either way; this setting decides "
            "whether to go further and average within each replicate, which is "
            "the most conservative option."))

        self.distance = _option_combo(["euclidean", "correlation", "cityblock", "cosine"],
                                      "choice")
        add_row(form, tr("Distance between samples"), self.distance, tr(
            "Used by PERMANOVA, PERMDISP and the MDS projection, so the picture "
            "and the test describe the same thing."))

        self.classifier = QComboBox()
        for key, label in (("lda_shrinkage", "linear discriminant (regularised)"),
                           ("svm_linear", "linear support vector machine"),
                           ("plsda", "PLS-DA"),
                           ("logistic", "logistic regression"),
                           ("random_forest", "random forest")):
            self.classifier.addItem(tr(label), key)
        add_row(form, tr("Assignment model"), self.classifier, tr(
            "The classifier that tries to tell each sample's group from its "
            "metrics on data it was not trained on. The regularised linear "
            "discriminant is a safe default; random forest also catches "
            "differences that are not a straight line."))

        self.permutations = QSpinBox()
        self.permutations.setRange(0, 99999)
        self.permutations.setSingleStep(100)
        self.permutations.setValue(999)
        add_row(form, tr("Label shuffles"), self.permutations, tr(
            "How many times the group labels are shuffled to work out what this "
            "analysis achieves when there is nothing to find. 999 supports a "
            "smallest p value of 0.001; fewer is faster and less precise."))

        # keep the "will run: ..." line under the Run button up to date
        self.stats_on.toggled.connect(self._selection_changed)
        self.permutations.valueChanged.connect(self._selection_changed)

        form.addRow("", hint(tr(
            "Significance is judged by shuffling the labels, not by a binomial "
            "test on the cross-validated accuracy — that test is "
            "anti-conservative and reports differences that are not there.")))
        return form

    def _som_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(20, 2, 4, 8)

        self.algo = _option_combo(["batch", "supervised", "relevance", "online",
                                   "growing"], "som_algorithm")
        _select(self.algo, "batch")
        algo_tip = tr(
            "batch      -- fast, deterministic, the standard choice.\n"
            "supervised -- XY-fused SOM; pulls the map towards separating your "
            "groups, and so cannot be used as evidence that they differ.\n"
            "relevance  -- learns a weight per metric (GRLVQ) so uninformative "
            "metrics stop diluting the distance.\n"
            "online     -- the sequential rule the ImageJ macro used.\n"
            "growing    -- grows the map where quantisation error is high.")
        self.algo.currentIndexChanged.connect(
            lambda *_: self._algo_changed(_value(self.algo, "batch")))
        add_row(form, tr("Algorithm"), self.algo, algo_tip)

        self.som_warning = styled(QLabel(""), "warning")
        self.som_warning.setWordWrap(True)
        self.som_warning.setVisible(False)
        form.addRow("", self.som_warning)

        self.auto_size = QCheckBox(tr("Choose the map size automatically (5*sqrt(N))"))
        self.auto_size.setChecked(True)
        set_tip(self.auto_size, tr(
            "Let SOMTrack pick the size of the grid from the number of samples: "
            "about 5 x the square root of N nodes."))
        self.auto_size.toggled.connect(self._auto_size_toggled)
        form.addRow("", self.auto_size)

        self.map_w = QSpinBox(); self.map_w.setRange(2, 100); self.map_w.setValue(9)
        self.map_h = QSpinBox(); self.map_h.setRange(2, 100); self.map_h.setValue(6)
        size_row = QHBoxLayout()
        size_row.addWidget(self.map_w); size_row.addWidget(QLabel("x"))
        size_row.addWidget(self.map_h); size_row.addStretch(1)
        holder = QWidget(); holder.setLayout(size_row)
        size_label = add_row(form, tr("Map size"), holder, tr(
            "Nodes across and down. A bigger map shows finer detail but leaves "
            "more nodes empty."))
        self.map_size_row = (size_label, holder)
        self._auto_size_toggled(True)

        self.topology = _option_combo(["hex", "rect"], "lattice")
        add_row(form, tr("Lattice"), self.topology, tr(
            "Shape of the grid. Hexagonal gives every node six neighbours and is "
            "the usual choice; rectangular gives four."))

        self.toroidal = QCheckBox(tr("Toroidal boundary (removes edge effects)"))
        set_tip(self.toroidal, tr(
            "Joins opposite edges, so the map has no border and samples do not "
            "pile up along it. The map is harder to read, because the edges wrap "
            "round."))
        form.addRow("", self.toroidal)

        self.epochs = QSpinBox(); self.epochs.setRange(10, 20000)
        self.epochs.setValue(200)
        add_row(form, tr("Training epochs"), self.epochs, tr(
            "How many passes over the data the map is trained for. More is "
            "slower; 200 is plenty for most data sets."))

        self.init = _option_combo(["pca", "sample", "random"], "som_init")
        add_row(form, tr("Initialisation"), self.init,
                tr("PCA initialisation is deterministic and converges faster than "
                   "the random start used in v1.2."))

        self.lr = QDoubleSpinBox(); self.lr.setRange(0.01, 1.0)
        self.lr.setSingleStep(0.05); self.lr.setValue(0.5)
        self.lr_label = add_row(form, tr("Learning rate"), self.lr, tr(
            "Online and growing SOM only. How strongly each sample pulls the map "
            "towards itself: larger learns faster but less smoothly."))

        self.tau = QDoubleSpinBox(); self.tau.setRange(0.02, 1.0)
        self.tau.setSingleStep(0.05); self.tau.setValue(0.25)
        add_row(form, tr("Neighbourhood time constant"), self.tau, tr(
            "How fast the training neighbourhood shrinks, as a fraction of the "
            "epochs. Smaller values give a more detailed but patchier map."))

        self.label_weight = QDoubleSpinBox()
        self.label_weight.setRange(0.0, 0.95)
        self.label_weight.setSingleStep(0.05); self.label_weight.setValue(0.35)
        self.label_weight_label = add_row(
            form, tr("Label weight (supervised)"), self.label_weight, tr(
                "Supervised SOM only. 0 ignores the labels entirely (= batch SOM); "
                "high values force separation and overstate it. Report whatever "
                "you used; 0.2-0.4 is a defensible range."))

        self.spread = QDoubleSpinBox()
        self.spread.setRange(0.05, 0.95); self.spread.setSingleStep(0.05)
        self.spread.setValue(0.5)
        self.spread_label = add_row(form, tr("Spread factor (growing)"), self.spread,
                                    tr("Growing SOM only. Lower values grow a "
                                       "bigger, more detailed map."))

        self.record = QSpinBox(); self.record.setRange(1, 200); self.record.setValue(5)
        add_row(form, tr("Snapshot interval"), self.record, tr(
            "Keeps a copy of the map every this many epochs, for the training "
            "animation."))

        self.seed = QSpinBox(); self.seed.setRange(0, 99999); self.seed.setValue(0)
        add_row(form, tr("Random seed"), self.seed, tr(
            "Fixes the random numbers, so the same settings give the same result "
            "every time. Try another seed to check a result does not depend on "
            "luck."))
        self._algo_changed(_value(self.algo, "batch"))
        return form

    def _nodes_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(20, 2, 4, 8)
        self.nc_enabled = QCheckBox(tr("Divide the map into behavioural clusters"))
        self.nc_enabled.setChecked(True)
        set_tip(self.nc_enabled, tr(
            "After training, group neighbouring nodes into a small number of "
            "behaviour types, each drawn as one outlined region of the map."))
        form.addRow("", self.nc_enabled)

        self.nc_method = _option_combo(["kmeans", "ward", "gmm"], "node_cluster")
        add_row(form, tr("Method"), self.nc_method, tr(
            "How the nodes are grouped: k-means (the default), Ward hierarchical "
            "clustering, or a Gaussian mixture model."))

        self.k_min = QSpinBox(); self.k_min.setRange(2, 40); self.k_min.setValue(2)
        self.k_max = QSpinBox(); self.k_max.setRange(2, 40); self.k_max.setValue(8)
        row = QHBoxLayout()
        row.addWidget(self.k_min); row.addWidget(QLabel(tr("to")))
        row.addWidget(self.k_max); row.addStretch(1)
        holder = QWidget(); holder.setLayout(row)
        add_row(form, tr("Try cluster counts"), holder, tr(
            "The smallest and largest number of clusters to try. The best number "
            "in this range is picked automatically."))

        self.weight_hits = QCheckBox(tr("Weight nodes by their sample count"))
        self.weight_hits.setChecked(True)
        set_tip(self.weight_hits, tr(
            "Stops empty codebook vectors, which the SOM only dragged along "
            "behind their neighbours, from defining a cluster."))
        form.addRow("", self.weight_hits)
        form.addRow("", hint(tr(
            "The best count is chosen by the average rank of the silhouette, "
            "Davies-Bouldin and Calinski-Harabasz indices; every count in the "
            "range is still saved.")))
        return form

    # ------------------------------------------------------------------
    # Settings a recipe (or a loaded settings file) may change that have no
    # control on this page.  They are carried through to the configuration on
    # every commit; without this, picking "Check that the picture is real" did
    # not actually switch on its seed-stability check or its full figure set.
    _CARRIED = (("report", "profile"),
                ("embedding", "stability_seeds"), ("embedding", "reliability_null"),
                ("embedding", "compute_rank_quality"),
                ("embedding", "compute_reliability"),
                ("stats", "run_permanova"), ("stats", "run_permdisp"),
                ("stats", "run_energy"), ("stats", "run_classification"),
                ("stats", "run_importance"), ("stats", "run_effect_sizes"),
                ("stats", "importance_repeats"), ("stats", "importance_model"),
                ("stats", "cv_scheme"), ("stats", "use_replicate_blocks"),
                ("stats", "bootstrap"))

    recipe_selected = Signal(object)        # the recipe's AnalysisConfig

    def _recipe_changed(self, *_) -> None:
        from .. import recipes

        key = self.recipe.currentData() or recipes.DEFAULT
        r = recipes.get(key)
        self.recipe_note.setText(f"<b>{tr(r.summary)}</b><br>{tr(r.detail)}")
        cfg = r.build(AnalysisConfig())
        self.load_config(cfg, select_recipe=False, overrides=False)
        self.recipe_selected.emit(cfg)

    def load_config(self, cfg: AnalysisConfig, select_recipe: bool = True,
                    overrides: bool = True) -> None:
        """Show ``cfg`` in the controls.

        Used for a recipe, for a settings file loaded from the File menu, and
        for a scan result adopted on the Compare page.  Before this existed the
        controls were the only source of truth: anything written into the
        configuration elsewhere was overwritten by the next commit, so "Use
        these settings for the next run" was undone by pressing Run.
        """
        import copy

        e, st, s, n = cfg.embedding, cfg.stats, cfg.som, cfg.node_cluster
        if select_recipe:
            i = self.recipe.findData(cfg.report.recipe)
            if i >= 0 and i != self.recipe.currentIndex():
                self.recipe.blockSignals(True)
                self.recipe.setCurrentIndex(i)
                self.recipe.blockSignals(False)
                from .. import recipes

                r = recipes.get(cfg.report.recipe)
                self.recipe_note.setText(f"<b>{tr(r.summary)}</b><br>{tr(r.detail)}")

        # The acknowledgement first: it decides which boxes can be ticked at
        # all.  Ticking the boxes first, as this used to, left a recipe that
        # needs a supervised method without it the second time it was picked.
        if hasattr(self, "supervised_ack"):
            self.supervised_ack.blockSignals(True)
            self.supervised_ack.setChecked(bool(e.allow_supervised))
            self.supervised_ack.blockSignals(False)
            self._gate_supervised(bool(e.allow_supervised))
        wanted = set(e.resolved_methods())
        for key, box in self._boxes.items():
            box.blockSignals(True)
            box.setChecked(key in wanted and box.isEnabled())
            box.blockSignals(False)
        if overrides:
            for key, values in (e.overrides or {}).items():
                if key in self._panels:
                    self._panels[key].set_overrides(values)

        self.stats_on.setChecked(st.enabled)
        _select(self.unit, st.unit_of_analysis)
        _select(self.distance, st.distance)
        _select(self.classifier, st.classifier)
        self.permutations.setValue(st.n_permutations)

        _select(self.algo, s.algorithm)
        self.auto_size.setChecked(not (s.width > 0 and s.height > 0))
        if s.width > 0 and s.height > 0:
            self.map_w.setValue(s.width)
            self.map_h.setValue(s.height)
        _select(self.topology, s.topology)
        self.toroidal.setChecked(s.toroidal)
        self.epochs.setValue(s.epochs)
        _select(self.init, s.init)
        self.lr.setValue(s.learning_rate)
        self.tau.setValue(s.time_constant)
        self.label_weight.setValue(s.label_weight)
        self.spread.setValue(s.spread_factor)
        self.record.setValue(s.record_every)
        self.seed.setValue(int(s.random_state or 0))

        self.nc_enabled.setChecked(n.enabled)
        _select(self.nc_method, n.method)
        self.k_min.setValue(n.k_min)
        self.k_max.setValue(n.k_max)
        self.weight_hits.setChecked(n.weight_by_hits)

        self._carried = {(sec, name): copy.deepcopy(getattr(getattr(cfg, sec), name))
                         for sec, name in self._CARRIED}
        self._algo_changed(_value(self.algo, "batch"))
        self._selection_changed()

    def adopt_projection_settings(self, cfg: AnalysisConfig) -> None:
        """Tick and pin what the Compare page adopted, leaving the rest alone."""
        e = cfg.embedding
        for key in e.resolved_methods():
            box = self._boxes.get(key)
            if box is not None and box.isEnabled():
                box.setChecked(True)
        for key, values in (e.overrides or {}).items():
            if key in self._panels:
                self._panels[key].set_overrides(values)
        self._selection_changed()

    def _gate_supervised(self, on: bool) -> None:
        """Supervised boxes can only be ticked once the warning is acknowledged."""
        from ..analysis import get as get_method

        for key, box in self._boxes.items():
            spec = get_method(key)
            if spec.supervised:
                box.setEnabled(on and spec.available())
                if not on:
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.blockSignals(False)

    def _supervised_toggled(self, on: bool) -> None:
        self._gate_supervised(on)
        self._selection_changed()

    def _selection_changed(self, *_) -> None:
        chosen = []
        for key, box in self._boxes.items():
            on = box.isChecked() and box.isEnabled()
            self._panels[key].setVisible(on)
            if on:
                chosen.append(key)
        n = len(chosen)
        bits = [tr("{n} projections" if n != 1 else "{n} projection").format(n=n)]
        if self.stats_on.isChecked():
            bits.append(tr("{n} label shuffles").format(n=self.permutations.value()))
        self.plan_label.setText(tr("will run: {plan}").format(
            plan=render(join_list(bits))))

    def _auto_size_toggled(self, on: bool) -> None:
        self.map_w.setEnabled(not on)
        self.map_h.setEnabled(not on)

    def _algo_changed(self, algo: str) -> None:
        # Hide what does not apply rather than greying it: a disabled control is
        # still something the eye has to read and dismiss.  The label goes with
        # its control; hiding only the control left a caption with nothing
        # beside it.
        for widgets, on in (((self.lr, self.lr_label), algo in ("online", "growing")),
                            ((self.label_weight, self.label_weight_label),
                             algo == "supervised"),
                            ((self.spread, self.spread_label), algo == "growing"),
                            ((self.auto_size, self.toroidal) + self.map_size_row,
                             algo != "growing")):
            for w in widgets:
                w.setVisible(on)
        if algo != "growing":
            self._auto_size_toggled(self.auto_size.isChecked())

        supervised = algo == "supervised"
        self.som_warning.setVisible(supervised)
        if supervised:
            self.som_warning.setText("⚠  " + tr(
                "This map is trained with the group labels, so it separates the "
                "groups by construction. Report the label weight, and take the "
                "evidence from the cross-validated statistics rather than from "
                "the map."))

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
        import copy

        cfg = self.state.config
        cfg.report.recipe = self.recipe.currentData() or "standard"
        for (sec, name), value in getattr(self, "_carried", {}).items():
            setattr(getattr(cfg, sec), name, copy.deepcopy(value))

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
        st.distance = _value(self.distance, "euclidean")
        st.classifier = self.classifier.currentData() or "lda_shrinkage"
        st.n_permutations = self.permutations.value()
        st.classification_permutations = self.permutations.value()
        st.random_state = self.seed.value()

        s = cfg.som
        s.algorithm = _value(self.algo, "batch")
        s.width = 0 if self.auto_size.isChecked() else self.map_w.value()
        s.height = 0 if self.auto_size.isChecked() else self.map_h.value()
        s.topology = _value(self.topology, "hex")
        s.toroidal = self.toroidal.isChecked()
        s.epochs = self.epochs.value()
        s.init = _value(self.init, "pca")
        s.learning_rate = self.lr.value()
        s.time_constant = self.tau.value()
        s.label_weight = self.label_weight.value()
        s.spread_factor = self.spread.value()
        s.record_every = self.record.value()
        s.random_state = self.seed.value()

        n = cfg.node_cluster
        n.enabled = self.nc_enabled.isChecked()
        n.method = _value(self.nc_method, "kmeans")
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
        self.verdict.document().setDocumentMargin(12)
        self._result = None
        self.tabs.addTab(self._wrap(
            self.verdict,
            tr("What this analysis found, and what it does not say.")),
            tr("Conclusion"))

        self.gallery = FigureGallery()
        self.tabs.addTab(self.gallery, tr("Figures"))

        self.separation = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.separation,
            tr("Multivariate tests. PERMANOVA asks whether the group averages "
               "differ; PERMDISP asks the separate question of whether the groups "
               "differ in spread. A significant PERMANOVA with a significant "
               "PERMDISP may mean 'more variable', not 'different'.")),
            tr("Group difference"))

        self.drivers = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.drivers,
            tr("Which metrics carry the difference. Read the activation column, "
               "not the weight: a large weight can belong to a metric that carries "
               "no group information and only cancels noise in another.")),
            tr("Key metrics"))

        self.projq = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.projq,
            tr("How much of each projection can be believed. Higher R_NX area "
               "preserves more neighbourhoods; a high unreliable fraction means "
               "that picture should be read for broad structure only.")),
            tr("Projection quality"))

        self.quality = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.quality, tr("Map quality and clustering diagnostics")),
            tr("Map quality"))

        self.assoc = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.assoc, tr("Per-metric association with the experimental group, "
                           "sorted by effect size")), tr("Metric statistics"))

        self.assign = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.assign, tr("Which node each sample landed on")), tr("Assignments"))
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
    def refresh_theme(self) -> None:
        """Redraw the conclusion, whose colours depend on light or dark."""
        if self._result is not None:
            self.verdict.setHtml(self._verdict_html(self._result))

    def show_result(self, result, panels) -> None:
        self._result = result
        self.gallery.set_panels(panels, getattr(result, "figure_captions", None))
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
        """The verdict in the interface language; every sentence is rendered
        from the same numbers as the English report."""
        import html as _h

        def esc(text) -> str:
            return _h.escape(render(text))

        t = theme.current()
        muted = t.muted
        v = getattr(result, "verdict", None)
        if v is None:
            return (f"<p style='color:{muted}'>"
                    + tr("The statistics layer was switched off, so there is no "
                         "conclusion to report. Turn it on under <i>How to test "
                         "whether the groups differ</i> and run again.")
                    + "</p>")

        strong, moderate, weak, none = t.verdict
        accent = {"strong": strong, "moderate": moderate, "weak": weak,
                  "none": none}.get(v.confidence, none)
        out = [f"<h2 style='color:{accent};margin:0 0 10px'>"
               f"{esc(v.headline)}</h2>"]
        for para in v.findings:
            out.append(f"<p>{esc(para)}</p>")
        if v.separable:
            out.append(f"<p><b>{_h.escape(tr('Separates:'))}</b> "
                       + esc(join_clauses(v.separable)) + "</p>")
        if v.not_separable:
            out.append(f"<p><b>{_h.escape(tr('Does not separate:'))}</b> "
                       + esc(join_clauses(v.not_separable)) + "</p>")
        if v.drivers:
            out.append(f"<p><b>{_h.escape(tr('Main contributing metrics:'))}</b> "
                       + esc(join_list(v.drivers)) + "</p>")
        if v.design_note:
            out.append(f"<p style='color:{muted}'>{esc(v.design_note)}</p>")
        if v.caveats:
            out.append("<h3 style='margin:14px 0 4px;font-size:13px'>"
                       + _h.escape(tr("Limits of this analysis")) + "</h3><ul>")
            out += [f"<li style='color:{muted}'>{esc(c)}</li>"
                    for c in v.caveats]
            out.append("</ul>")
        out.append(f"<p style='color:{muted};font-size:11px;margin-top:16px'>"
                   + _h.escape(tr(
                       "The full report, including the methods paragraph and the "
                       "reference list for every algorithm this run used, is "
                       "written to RESULTS_REPORT.html and methods.txt when you "
                       "export -- in English, followed by a complete Chinese "
                       "translation."))
                   + "</p>")
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
    settings_adopted = Signal(object)        # the updated AnalysisConfig

    def build(self) -> None:
        self.tabs = QTabWidget()

        self.grid_gallery = FigureGallery()
        self.tabs.addTab(self._wrap(
            self.grid_gallery,
            tr("Each panel is the same samples under a different projection, with "
               "the area under its R_NX curve and the share of points it placed "
               "unreliably.")), tr("Methods side by side"))

        self.quality = DataFrameView()
        self.tabs.addTab(self._wrap(
            self.quality,
            tr("Higher R_NX area preserves more neighbourhoods. A supervised method "
               "will look good here for the wrong reason — it was given the "
               "labels.")), tr("Quality table"))

        self.tabs.addTab(self._scan_tab(), tr("Parameter scan"))
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

        lay.addWidget(hint(tr(
            "Run one method across a range of settings and see whether the "
            "answer changes. A wide plateau is the useful result: it means the "
            "setting does not matter for your data, which is worth reporting.")))

        row = QHBoxLayout()
        scan_tip = tr("The method to run over a range of its settings.")
        row.addWidget(help_label(tr("Scan"), scan_tip))
        self.scan_method = QComboBox()
        set_tip(self.scan_method, scan_tip)
        self.scan_method.setMinimumWidth(180)
        row.addWidget(self.scan_method)

        criterion_tip = tr(
            "'Unreliably placed points' is the criterion of Xia, Lee & Li "
            "(2024): it turns 'what perplexity should I use' into a question "
            "with an answer. Scoring by how well the known groups separate is a "
            "supervised choice, and the report says so.")
        row.addWidget(help_label(tr("scored by"), criterion_tip))
        self.scan_criterion = QComboBox()
        for key, (_, _, needs_labels, description) in CRITERIA.items():
            self.scan_criterion.addItem(tr(description), key)
        self.scan_criterion.setMinimumWidth(280)
        set_tip(self.scan_criterion, criterion_tip)
        row.addWidget(self.scan_criterion)

        self.scan_btn = styled(QPushButton(tr("Run scan")), "primary")
        self.scan_btn.clicked.connect(self._emit_scan)
        row.addWidget(self.scan_btn)
        row.addStretch(1)
        lay.addLayout(row)

        self.plateau = styled(QLabel(""), "callout")
        self.plateau.setWordWrap(True)
        self.plateau.setVisible(False)
        lay.addWidget(self.plateau)

        split = QSplitter(Qt.Orientation.Vertical)
        self.scan_gallery = FigureGallery()
        split.addWidget(self.scan_gallery)
        self.scan_table = DataFrameView()
        split.addWidget(self.scan_table)
        split.setSizes([420, 200])
        lay.addWidget(split, 1)

        self.adopt_btn = QPushButton(tr("Use these settings for the next run"))
        self.adopt_btn.setEnabled(False)
        set_tip(self.adopt_btn, tr(
            "Pins the chosen values as explicit overrides. They will appear in "
            "the methods section as a choice you made."))
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
            mark = "" if spec.key in ran else tr("  (not in the last run)")
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
                self.status.emit(tr("Comparison figure '{name}' skipped: {error}")
                                 .format(name=name, error=exc), "warn")
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
                self.status.emit(tr("Scan figure '{name}' skipped: {error}")
                                 .format(name=name, error=exc), "warn")
        self.scan_gallery.set_panels(panels)

        cols = list(dict.fromkeys(
            [c for c in scan.keys + [scan.criterion, "rnx_auc", "dubious_fraction",
                                     "group_silhouette"]
             if c in scan.rows.columns]))
        self.scan_table.set_frame(scan.rows[cols].round(4))
        self.plateau.setText(render(scan.plateau_text()))
        self.plateau.setVisible(True)
        self.adopt_btn.setEnabled(bool(scan.best))
        self.tabs.setCurrentIndex(2)

    def _adopt(self) -> None:
        scan = getattr(self, "_scan", None)
        if scan is None or not scan.best:
            return
        from ..analysis import get as get_method

        params = scan.best_params()
        cfg = self.state.config.embedding
        if get_method(scan.method).supervised and not cfg.allow_supervised:
            self.status.emit(tr(
                "{method} is a supervised projection. Tick the acknowledgement "
                "under 'Which projections to run' on the Analysis page before "
                "using its settings.").format(method=scan.method_label), "warn")
            return
        cfg.overrides.setdefault(scan.method, {}).update(params)
        if scan.method not in cfg.methods:
            cfg.methods = list(cfg.resolved_methods()) + [scan.method]
        bits = ", ".join(f"{k} = {v:g}" if isinstance(v, float) else f"{k} = {v}"
                         for k, v in params.items())
        self.status.emit(
            tr("{method} will use {settings} on the next run, and the methods "
               "section will record it as your choice.").format(
                method=scan.method_label, settings=bits), "ok")
        self.adopt_btn.setEnabled(False)
        # the Analysis page shows it, so its next commit keeps it
        self.settings_adopted.emit(self.state.config)


# ==========================================================================
# 7. Export
# ==========================================================================
class ExportPage(Page):
    TITLE = "7. Export"
    SUBTITLE = ("PDF and SVG keep every label as editable text, so figures can be "
                "restyled in Illustrator or Inkscape without re-running the analysis.")

    export_requested = Signal()

    def build(self) -> None:
        row = QHBoxLayout()
        self.out_dir = QLineEdit(str(Path.home() / "SOMTrack_output"))
        browse = QPushButton(tr("Browse..."))
        browse.clicked.connect(self._browse)
        out_tip = tr("Where every figure, table and report is written. The folder "
                     "is created if it does not exist yet.")
        row.addWidget(help_label(tr("Output folder"), out_tip))
        set_tip(self.out_dir, out_tip)
        row.addWidget(self.out_dir, 1)
        row.addWidget(browse)
        self.body.addLayout(row)

        grid = QGridLayout()
        box = QGroupBox(tr("Formats"))
        box.setLayout(grid)
        self.png = QCheckBox(tr("PNG (raster, 600 dpi)")); self.png.setChecked(True)
        self.pdf = QCheckBox(tr("PDF (vector, editable text)")); self.pdf.setChecked(True)
        self.svg = QCheckBox(tr("SVG (vector, editable text)"))
        self.mp4 = QCheckBox(tr("MP4 training animation")); self.mp4.setChecked(True)
        self.tables = QCheckBox(tr("CSV tables + one .xlsx workbook"))
        self.tables.setChecked(True)
        for fmt_box, tip in (
                (self.png, "Picture files, for slides and for a quick look."),
                (self.pdf, "Vector figures whose text stays editable, plus every "
                           "figure in one SOMTrack_report.pdf."),
                (self.svg, "Vector figures for Inkscape or Illustrator."),
                (self.mp4, "A video of the map taking shape during training."),
                (self.tables, "Every result table as a CSV file, and all of them "
                              "together in one Excel workbook.")):
            set_tip(fmt_box, tr(tip))
        for i, c in enumerate((self.png, self.pdf, self.svg, self.mp4, self.tables)):
            grid.addWidget(c, i // 2, i % 2)
        self.body.addWidget(box)

        from ..viz import ffmpeg_available

        if not ffmpeg_available():
            self.body.addWidget(hint(tr("ffmpeg was not found on PATH; the animation "
                                        "will fall back to an animated GIF.")))

        report_box = QGroupBox(tr("Report language"))
        rform = QFormLayout(report_box)
        self.translation = QCheckBox(tr("Add a Traditional Chinese translation after "
                                        "the English report"))
        self.translation.setChecked(True)
        set_tip(self.translation, tr(
            "RESULTS_REPORT.html, RESULTS_REPORT.md and methods.txt are written in "
            "English first; the translation follows as a second, complete version. "
            "Text inside the figures and the contents of the tables stay in "
            "English, so every value can be matched between the two."))
        rform.addRow("", self.translation)
        self.body.addWidget(report_box)

        fig_box = QGroupBox(tr("Figure style"))
        form = QFormLayout(fig_box)
        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(60.0, 400.0); self.width_mm.setValue(180.0)
        add_row(form, tr("Figure width (mm)"), self.width_mm,
                tr("180 mm is a typical double-column width; 90 mm single."))

        self.font_size = QDoubleSpinBox()
        self.font_size.setRange(5.0, 20.0); self.font_size.setValue(8.0)
        add_row(form, tr("Base font size (pt)"), self.font_size, tr(
            "Size of the text in the figures. 7-9 pt suits most journals."))

        self.font_family = QLineEdit("Arial")
        add_row(form, tr("Font family"), self.font_family, tr(
            "Font used in the figures. Most journals ask for Arial or Helvetica."))

        self.dpi = QSpinBox(); self.dpi.setRange(72, 1200); self.dpi.setValue(600)
        add_row(form, tr("PNG resolution (dpi)"), self.dpi, tr(
            "Dots per inch of the PNG files. 300 is enough on screen; 600 is the "
            "usual requirement for print."))

        self.palette = _option_combo(["somtrack", "tab10", "legacy_hsb"], "palette")
        add_row(form, tr("Group palette"), self.palette,
                tr("'somtrack' is the Okabe-Ito colour-blind-safe set; "
                   "'legacy_hsb' reproduces the v1.2 hue wheel."))

        self.captions = QCheckBox(tr("Draw a figure legend under every panel"))
        self.captions.setChecked(True)
        set_tip(self.captions, tr(
            "Prints a short explanation under each figure. Untick it for clean "
            "figures to put in a paper."))
        form.addRow("", self.captions)
        self.body.addWidget(fig_box)

        alpha_tip = tr(
            "In the one-map-per-group figures, a node holding only a few samples "
            "of that group is drawn see-through, so it does not look as important "
            "as a well-filled one.")
        alpha_box = QGroupBox(tr("Per-group map transparency"))
        alpha_box.setToolTip(rich_tip(alpha_tip))
        aform = QFormLayout(alpha_box)
        self.a0 = QSpinBox(); self.a0.setRange(0, 100); self.a0.setValue(20)
        self.a1 = QSpinBox(); self.a1.setRange(0, 100); self.a1.setValue(40)
        self.a2 = QSpinBox(); self.a2.setRange(0, 100); self.a2.setValue(60)
        for lab, sp in (("0 samples", self.a0), ("1 sample", self.a1),
                        ("2 samples", self.a2)):
            sp.setSuffix(tr(" % opaque"))
            add_row(aform, tr(lab), sp, alpha_tip)
        add_row(aform, tr(">= 3 samples"), QLabel(tr("100 % opaque (fixed)")), alpha_tip)
        self.body.addWidget(alpha_box)

        row2 = QHBoxLayout()
        self.export_btn = styled(QPushButton(tr("Export everything")), "primary",
                                 size="large")
        self.export_btn.clicked.connect(self.export_requested.emit)
        self.open_btn = QPushButton(tr("Open output folder"))
        self.open_btn.clicked.connect(self._open_folder)
        row2.addWidget(self.export_btn)
        row2.addWidget(self.open_btn)
        row2.addStretch(1)
        self.body.addLayout(row2)
        self.body.addStretch(1)

    def on_enter(self) -> None:
        # a loaded configuration may have switched the translation off
        self.translation.setChecked(bool(self.state.config.report.translation))

    def adopt_recipe(self, cfg: AnalysisConfig) -> None:
        """Take the palette a recipe asks for (the v1.2 recipe uses the old hue
        wheel), which this page would otherwise overwrite on every run."""
        _select(self.palette, cfg.figure.palette)

    def load_config(self, cfg: AnalysisConfig) -> None:
        e, f = cfg.export, cfg.figure
        self.out_dir.setText(str(e.out_dir))
        for box, on in ((self.png, e.png), (self.pdf, e.pdf), (self.svg, e.svg),
                        (self.mp4, e.mp4), (self.tables, e.save_tables)):
            box.setChecked(bool(on))
        self.translation.setChecked(bool(cfg.report.translation))
        self.width_mm.setValue(f.figure_width_mm)
        self.font_size.setValue(f.base_font_size)
        self.font_family.setText(f.font_family)
        self.dpi.setValue(f.dpi_png)
        _select(self.palette, f.palette)
        self.captions.setChecked(f.show_caption)
        alpha = f.group_alpha_by_hits or {}
        for k, sp in ((0, self.a0), (1, self.a1), (2, self.a2)):
            if k in alpha:
                sp.setValue(int(round(alpha[k] * 100)))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, tr("Output folder"),
                                             self.out_dir.text())
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

        cfg.report.translation = "zh_TW" if self.translation.isChecked() else ""

        f = cfg.figure
        f.figure_width_mm = self.width_mm.value()
        f.base_font_size = self.font_size.value()
        f.font_family = self.font_family.text() or "Arial"
        f.dpi_png = self.dpi.value()
        f.palette = _value(self.palette, "somtrack")
        f.show_caption = self.captions.isChecked()
        f.group_alpha_by_hits = {0: self.a0.value() / 100,
                                 1: self.a1.value() / 100,
                                 2: self.a2.value() / 100}


__all__ = ["AppState", "Page", "SourcePage", "MetricsPage", "PreprocessPage",
           "ClusterPage", "ResultsPage", "ComparePage", "ExportPage"]
