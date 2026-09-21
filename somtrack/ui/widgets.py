"""Reusable Qt widgets for the SOMTrack desktop app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import (QAbstractTableModel, QModelIndex, QSize, Qt,
                            Signal)
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout,
                               QFrame, QGroupBox, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem,
                               QPlainTextEdit, QPushButton, QScrollArea,
                               QSizePolicy, QSpinBox, QTableView, QToolButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from ..i18n import language, render, tr

ACCENT = "#0072B2"


def none_text() -> str:
    """The "(none)" entry of a column picker, in the interface language."""
    return tr("(none)")


def is_none_text(text: str) -> bool:
    return text in ("", "(none)", none_text())


def cjk_families(first: str) -> list[str]:
    """``first`` followed by fonts that carry Traditional-Chinese glyphs.

    Qt falls back to *some* CJK font on its own, but on Windows that can be a
    Simplified-Chinese face, whose glyph shapes differ from the ones a Taiwanese
    reader expects.  Naming the Traditional faces keeps the choice deliberate.
    """
    return [first, "Microsoft JhengHei UI", "Microsoft JhengHei", "PingFang TC",
            "Noto Sans CJK TC", "Noto Sans TC"]


# --------------------------------------------------------------------------
def h_line() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setStyleSheet("color:#dcdcdc;")
    return f


def heading(text: str, size: int = 15) -> QLabel:
    lab = QLabel(text)
    font = lab.font()
    font.setPointSize(size)
    font.setWeight(QFont.Weight.DemiBold)
    lab.setFont(font)
    return lab


def hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color:#5d6570;")
    return lab


# --------------------------------------------------------------------------
class FileListWidget(QWidget):
    """Drop / browse list of input files, each with a group and replicate id."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        self.add_btn = QPushButton(tr("Add files..."))
        self.remove_btn = QPushButton(tr("Remove selected"))
        self.clear_btn = QPushButton(tr("Clear"))
        for b in (self.add_btn, self.remove_btn, self.clear_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels([tr("File"), tr("Group (treatment)"), tr("Replicate")])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setMinimumSectionSize(110)
        self.tree.setColumnWidth(1, 170)
        self.tree.setColumnWidth(2, 90)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                  | QAbstractItemView.EditTrigger.SelectedClicked)
        lay.addWidget(self.tree, 1)
        lay.addWidget(hint(tr("Group and replicate are guessed from the file name; "
                              "double-click a cell to correct it. Files sharing a "
                              "group are one treatment; the replicate number "
                              "separates repeats of that treatment.")))

        self.add_btn.clicked.connect(self._browse)
        self.remove_btn.clicked.connect(self._remove)
        self.clear_btn.clicked.connect(self.clear)
        self.tree.itemChanged.connect(lambda *_: self.changed.emit())

    # -------------------------------------------------- drag & drop
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls()]
        self.add_paths([p for p in paths
                        if Path(p).suffix.lower() in
                        {".csv", ".tsv", ".txt", ".xls", ".xlsx", ".xlsm"}])

    # -------------------------------------------------- api
    def _browse(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, tr("Select coordinate tables"), "",
            tr("Tables") + " (*.csv *.tsv *.txt *.xls *.xlsx *.xlsm);;"
            + tr("All files") + " (*)")
        self.add_paths(paths)

    def add_paths(self, paths: list[str]):
        for p in paths:
            g, r = _guess_group_replicate(Path(p).stem)
            item = QTreeWidgetItem([str(p), g, str(r)])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
            item.setToolTip(0, str(p))
            self.tree.addTopLevelItem(item)
        if paths:
            self.changed.emit()

    def _remove(self):
        for item in self.tree.selectedItems():
            self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
        self.changed.emit()

    def clear(self):
        self.tree.clear()
        self.changed.emit()

    def entries(self) -> tuple[list[str], list, list[int]]:
        files, groups, reps = [], [], []
        for i in range(self.tree.topLevelItemCount()):
            it = self.tree.topLevelItem(i)
            files.append(it.text(0))
            groups.append(it.text(1) or "group1")
            try:
                reps.append(int(it.text(2)))
            except ValueError:
                reps.append(1)
        return files, groups, reps


def _guess_group_replicate(stem: str) -> tuple[str, int]:
    from ..io_tables import guess_group_replicate

    return guess_group_replicate(stem)


# --------------------------------------------------------------------------
class ColumnMapWidget(QGroupBox):
    """Combo boxes mapping table columns onto the canonical roles."""

    ROLES = [
        ("track", "Track ID", True),
        ("x", "X position", True),
        ("y", "Y position", True),
        ("t", "Time / frame", True),
        ("group", "Group column (optional)", False),
        ("replicate", "Replicate column (optional)", False),
        ("area", "Area (optional)", False),
        ("major", "Ellipse major (optional)", False),
        ("minor", "Ellipse minor (optional)", False),
        ("angle", "Body orientation (optional)", False),
        ("intensity_mean", "Mean intensity (optional)", False),
        ("intensity_std", "Intensity SD (optional)", False),
        ("diameter", "Estimated diameter (optional)", False),
    ]

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(tr("Column mapping"), parent)
        self.combos: dict[str, QComboBox] = {}
        form = QFormLayout(self)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        for key, label, required in self.ROLES:
            cb = QComboBox()
            cb.setMinimumWidth(230)
            cb.currentTextChanged.connect(lambda *_: self.changed.emit())
            self.combos[key] = cb
            lab = QLabel(tr(label) + (" *" if required else ""))
            if required:
                lab.setStyleSheet(f"color:{ACCENT};")
            form.addRow(lab, cb)

    def populate(self, columns: list[str], detected) -> None:
        for key, cb in self.combos.items():
            cb.blockSignals(True)
            cb.clear()
            # the column name travels as item data, so the "(none)" entry can be
            # shown in any language without being mistaken for a column
            cb.addItem(none_text(), "")
            for c in columns:
                cb.addItem(str(c), str(c))
            value = getattr(detected, key, None)
            i = cb.findData(value) if value in columns else 0
            cb.setCurrentIndex(max(i, 0))
            cb.blockSignals(False)
        self.changed.emit()

    def to_map(self):
        from ..io_tables import SpotColumnMap

        cm = SpotColumnMap()
        for key, cb in self.combos.items():
            txt = cb.currentData()
            setattr(cm, key, txt or None)
        cm.track = cm.track or ""
        cm.x = cm.x or ""
        cm.y = cm.y or ""
        cm.t = cm.t or ""
        return cm


# --------------------------------------------------------------------------
class MetricTree(QTreeWidget):
    """Checkable metric picker grouped by category, with units and tooltips."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setColumnCount(3)
        self.setHeaderLabels([tr("Metric"), tr("Unit"), tr("What it measures")])
        self.setAlternatingRowColors(True)
        self.header().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.header().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.itemChanged.connect(self._on_item_changed)
        self._updating = False
        self._items: dict[str, QTreeWidgetItem] = {}

    def populate(self, available: list[str], length_unit: str, time_unit: str,
                 preselect: list[str] | None = None) -> None:
        from ..metrics import METRICS, metrics_by_category

        self._updating = True
        self.clear()
        self._items.clear()
        preselect = set(preselect) if preselect is not None else None

        for category, specs in metrics_by_category().items():
            usable = [s for s in specs if s.name in available]
            if not usable:
                continue
            parent = QTreeWidgetItem([tr(category), "", ""])
            parent.setFlags(parent.flags() | Qt.ItemFlag.ItemIsUserCheckable
                            | Qt.ItemFlag.ItemIsAutoTristate)
            f = parent.font(0)
            f.setBold(True)
            parent.setFont(0, f)
            self.addTopLevelItem(parent)
            for spec in usable:
                # The first column is the column name the metric gets in every
                # output table, so it stays as it is; a translated interface
                # puts the translated metric name in front of the description.
                description = tr(spec.description)
                if language() != "en":
                    description = f"{tr(spec.label)}：{description}"
                child = QTreeWidgetItem([
                    spec.name + ("  (v1.2)" if spec.legacy else ""),
                    spec.formatted_unit(length_unit, time_unit),
                    description,
                ])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                on = spec.default if preselect is None else (spec.name in preselect)
                child.setCheckState(0, Qt.CheckState.Checked if on
                                    else Qt.CheckState.Unchecked)
                child.setToolTip(0, tr(spec.label))
                child.setToolTip(2, description)
                child.setData(0, Qt.ItemDataRole.UserRole, spec.name)
                parent.addChild(child)
            parent.setExpanded(True)
        self._updating = False
        self.changed.emit()

    def _on_item_changed(self, *_):
        if not self._updating:
            self.changed.emit()

    def selected(self) -> list[str]:
        out = []
        for i in range(self.topLevelItemCount()):
            parent = self.topLevelItem(i)
            for j in range(parent.childCount()):
                c = parent.child(j)
                if c.checkState(0) == Qt.CheckState.Checked:
                    out.append(c.data(0, Qt.ItemDataRole.UserRole))
        return out

    def set_selection(self, names: set[str] | None, only_defaults: bool = False,
                      legacy_only: bool = False) -> None:
        from ..metrics import METRICS

        self._updating = True
        for i in range(self.topLevelItemCount()):
            parent = self.topLevelItem(i)
            for j in range(parent.childCount()):
                c = parent.child(j)
                name = c.data(0, Qt.ItemDataRole.UserRole)
                spec = METRICS[name]
                if legacy_only:
                    on = spec.legacy
                elif only_defaults:
                    on = spec.default
                elif names is None:
                    on = True
                else:
                    on = name in names
                c.setCheckState(0, Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
        self._updating = False
        self.changed.emit()


# --------------------------------------------------------------------------
class CheckListWidget(QListWidget):
    """Simple checkable list, used for picking columns of a feature table."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.itemChanged.connect(lambda *_: self.changed.emit())

    def populate(self, names: list[str], checked: set[str] | None = None):
        self.blockSignals(True)
        self.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            on = True if checked is None else (n in checked)
            it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            self.addItem(it)
        self.blockSignals(False)
        self.changed.emit()

    def selected(self) -> list[str]:
        return [self.item(i).text() for i in range(self.count())
                if self.item(i).checkState() == Qt.CheckState.Checked]

    def set_all(self, on: bool):
        self.blockSignals(True)
        for i in range(self.count()):
            self.item(i).setCheckState(Qt.CheckState.Checked if on
                                       else Qt.CheckState.Unchecked)
        self.blockSignals(False)
        self.changed.emit()


# --------------------------------------------------------------------------
class DataFrameView(QTableView):
    """Read-only preview of a pandas DataFrame."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.horizontalHeader().setStretchLastSection(True)
        self.setSortingEnabled(False)

    def set_frame(self, df: pd.DataFrame, max_rows: int = 400) -> None:
        self.setModel(_PandasModel(df.head(max_rows)))
        self.resizeColumnsToContents()


class _PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.reset_index(drop=True)

    def rowCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()) -> int:
        return 0 if parent.isValid() else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        v = self._df.iat[index.row(), index.column()]
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        return str(section + 1)


# --------------------------------------------------------------------------
class FigureGallery(QWidget):
    """Figure list on the left, rendered canvas on the right."""

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        left = QVBoxLayout()
        left.addWidget(QLabel(tr("Figures")))
        self.list = QListWidget()
        self.list.setMaximumWidth(250)
        self.list.currentRowChanged.connect(self._show)
        left.addWidget(self.list, 1)
        lay.addLayout(left)

        self.holder = QVBoxLayout()
        holder_widget = QWidget()
        holder_widget.setLayout(self.holder)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(holder_widget)
        self.scroll.setStyleSheet("QScrollArea{background:#ffffff;border:1px solid #dcdcdc;}")
        lay.addWidget(self.scroll, 1)

        self._panels: list = []
        self._canvas = None

    def set_panels(self, panels: list, captions: dict | None = None) -> None:
        """Show ``(name, panel)`` pairs; ``captions`` adds each figure's legend
        (圖說) as a tooltip, in the interface language."""
        self.clear()
        self._panels = list(panels)
        for name, _ in self._panels:
            item = QListWidgetItem(name)
            caption = (captions or {}).get(name)
            if caption:
                item.setToolTip(render(caption))
            self.list.addItem(item)
        if self._panels:
            self.list.setCurrentRow(0)

    def clear(self) -> None:
        self.list.clear()
        self._panels = []
        self._drop_canvas()

    def _drop_canvas(self) -> None:
        if self._canvas is not None:
            self._canvas.setParent(None)
            self._canvas.deleteLater()
            self._canvas = None
        while self.holder.count():
            item = self.holder.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

    def _show(self, row: int) -> None:
        if row < 0 or row >= len(self._panels):
            return
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.backends.backend_qtagg import \
            NavigationToolbar2QT as NavBar

        self._drop_canvas()
        fig = self._panels[row][1].fig
        canvas = FigureCanvasQTAgg(fig)
        w_in, h_in = fig.get_size_inches()
        canvas.setMinimumSize(QSize(int(w_in * 96), int(h_in * 96)))
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        toolbar = NavBar(canvas, self)
        self.holder.addWidget(toolbar)
        self.holder.addWidget(canvas, 1)
        self._canvas = canvas
        canvas.draw_idle()


# --------------------------------------------------------------------------
class LogPane(QPlainTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumBlockCount(2000)
        font = QFont()
        font.setFamilies(cjk_families("Consolas"))
        font.setPointSize(9)
        self.setFont(font)
        self.setStyleSheet("background:#fbfbfc;color:#26303a;border:1px solid #dcdcdc;")

    def log(self, text: str, level: str = "info") -> None:
        prefix = {"info": "", "warn": "! ", "error": "X ", "ok": "+ "}.get(level, "")
        self.appendPlainText(prefix + render(text))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


__all__ = [
    "FileListWidget", "ColumnMapWidget", "MetricTree", "CheckListWidget",
    "DataFrameView", "FigureGallery", "LogPane", "heading", "hint", "h_line",
    "ACCENT", "none_text", "is_none_text", "cjk_families",
]
