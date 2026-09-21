"""Forms that build themselves from what a method declares.

The 2.0 clustering page hard-coded twenty-one controls and greyed out the ones
that did not apply.  Greying out is the wrong move: an inapplicable control is
still something the eye has to process and dismiss, and a page of greyed-out
controls reads as "this program is complicated" to exactly the user who is
deciding whether to keep going.

Everything here is built from :class:`~somtrack.analysis.registry.ParamSpec`, so

* a control exists only while its method is selected;
* ``tier`` decides whether it is visible or behind "Advanced settings", which
  stays collapsed until someone asks for it;
* ``suggest`` gives every numeric control an **Auto** box, ticked by default,
  that shows the value the data implies -- so the user can see what the program
  chose without having to choose it themselves;
* adding a method to the registry adds its controls here with no edit.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QSpinBox, QVBoxLayout,
                               QWidget)

from ..analysis.registry import MethodSpec, ParamSpec
from ..i18n import tr
from .widgets import hint


# ==========================================================================
class Collapsible(QWidget):
    """A section that stays shut until it is wanted."""

    def __init__(self, title: str, parent=None, open_: bool = False):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        lay.setSpacing(4)

        title = tr(title)
        self.toggle = QCheckBox(title)
        self.toggle.setChecked(open_)
        self.toggle.setStyleSheet(
            "QCheckBox{color:#4a6fa5;font-weight:600;}"
            "QCheckBox::indicator{width:0;height:0;}")
        self.toggle.toggled.connect(self._on_toggle)
        lay.addWidget(self.toggle)

        self.body = QWidget()
        self.body.setVisible(open_)
        lay.addWidget(self.body)
        self._title = title
        self._on_toggle(open_)

    def _on_toggle(self, on: bool) -> None:
        self.body.setVisible(on)
        self.toggle.setText(("▾  " if on else "▸  ") + self._title)

    def set_layout(self, layout) -> None:
        self.body.setLayout(layout)


# ==========================================================================
class ParamRow(QWidget):
    """One parameter: its control, an Auto box, and what the auto value would be."""

    changed = Signal()

    def __init__(self, spec: ParamSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.control = _control_for(spec)
        self.control.setToolTip(tr(spec.help) if spec.help else "")
        self._dirty = False
        _on_change(self.control, self._touched)
        lay.addWidget(self.control)

        self.auto = QCheckBox(tr("Auto"))
        self.auto.setChecked(spec.suggest is not None)
        self.auto.setVisible(spec.suggest is not None)
        self.auto.setToolTip(tr(
            "Let SOMTrack choose this from the size of your data set. "
            "The value it picked is shown beside the box and is recorded in the "
            "methods section."))
        self.auto.toggled.connect(self._on_auto)
        lay.addWidget(self.auto)

        self.note = QLabel("")
        self.note.setStyleSheet("color:#6b7280;font-size:11px;")
        lay.addWidget(self.note)
        lay.addStretch(1)

        self._on_auto(self.auto.isChecked())

    def _on_auto(self, on: bool) -> None:
        self.control.setEnabled(not on)
        self.changed.emit()

    def set_suggestion(self, value) -> None:
        """Show the data-driven value, and adopt it while Auto is ticked."""
        if value is None:
            self.note.setText("")
            return
        text = f"{value:g}" if isinstance(value, float) else str(value)
        self.note.setText(tr("auto: {value}").format(value=text))
        if self.auto.isChecked():
            _set_value(self.control, value)

    def _touched(self, *_) -> None:
        if self.control.isEnabled():
            self._dirty = True
        self.changed.emit()

    def value(self):
        """The value only if the user actually chose it.

        ``None`` means "no override": either Auto is ticked, or the control is
        still sitting on the method's own default.  Recording an untouched
        default as an explicit override would silently pin it, so a later change
        to that default would not reach a config saved today -- and the methods
        section would claim a choice nobody made.
        """
        if self.auto.isChecked() and self.spec.suggest is not None:
            return None
        current = _get_value(self.control)
        if not self._dirty and current == self.spec.default:
            return None
        return current

    def set_value(self, value) -> None:
        if value is None:
            return
        self.auto.setChecked(False)
        _set_value(self.control, value)
        self._dirty = True


def _control_for(spec: ParamSpec) -> QWidget:
    if spec.kind == "int":
        w = QSpinBox()
        w.setRange(int(spec.low if spec.low is not None else 0),
                   int(spec.high if spec.high is not None else 10 ** 6))
        w.setValue(int(spec.default))
        return w
    if spec.kind == "float":
        w = QDoubleSpinBox()
        w.setDecimals(3)
        w.setRange(float(spec.low if spec.low is not None else -1e9),
                   float(spec.high if spec.high is not None else 1e9))
        w.setSingleStep(float(spec.step or 0.05))
        w.setValue(float(spec.default))
        return w
    if spec.kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(spec.default))
        return w
    # The value travels as item data; the label is only what the user reads,
    # so a translated label can never leak into the configuration.
    w = QComboBox()
    for c in spec.choices:
        w.addItem(tr(str(c), context="choice"), c)
    i = w.findData(spec.default)
    if i >= 0:
        w.setCurrentIndex(i)
    return w


def _on_change(w: QWidget, slot) -> None:
    for name in ("valueChanged", "currentTextChanged", "toggled"):
        signal = getattr(w, name, None)
        if signal is not None:
            signal.connect(slot)
            return


def _get_value(w: QWidget):
    if isinstance(w, QSpinBox):
        return int(w.value())
    if isinstance(w, QDoubleSpinBox):
        return float(w.value())
    if isinstance(w, QCheckBox):
        return bool(w.isChecked())
    if isinstance(w, QComboBox):
        data = w.currentData()
        return data if data is not None else w.currentText()
    return None


def _set_value(w: QWidget, value) -> None:
    try:
        if isinstance(w, QSpinBox):
            w.setValue(int(value))
        elif isinstance(w, QDoubleSpinBox):
            w.setValue(float(value))
        elif isinstance(w, QCheckBox):
            w.setChecked(bool(value))
        elif isinstance(w, QComboBox):
            i = w.findData(value)
            if i < 0:
                i = next((k for k in range(w.count())
                          if str(w.itemData(k)) == str(value)), -1)
            if i >= 0:
                w.setCurrentIndex(i)
    except (TypeError, ValueError):
        pass


# ==========================================================================
class MethodPanel(QWidget):
    """Everything about one projection method: its blurb and its parameters."""

    changed = Signal()

    def __init__(self, spec: MethodSpec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.rows: dict[str, ParamRow] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 2, 4, 10)
        lay.setSpacing(4)

        if spec.summary:
            lay.addWidget(hint(tr(spec.summary)))
        if spec.caveat:
            warn = QLabel("⚠  " + tr(spec.caveat))
            warn.setWordWrap(True)
            warn.setStyleSheet("color:#8a3b00;font-size:11px;")
            lay.addWidget(warn)

        common = [p for p in spec.params if p.tier == "common"]
        advanced = [p for p in spec.params if p.tier != "common"]

        if common:
            lay.addLayout(self._form(common))
        if advanced:
            box = Collapsible("Advanced settings")
            box.set_layout(self._form(advanced))
            lay.addWidget(box)

    def _form(self, specs: list[ParamSpec]) -> QFormLayout:
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(5)
        for spec in specs:
            row = ParamRow(spec)
            row.changed.connect(self.changed.emit)
            self.rows[spec.name] = row
            form.addRow(tr(spec.display), row)
        return form

    # ------------------------------------------------------------------
    def refresh_suggestions(self, ctx) -> None:
        for name, row in self.rows.items():
            spec = self.spec.param(name)
            if spec is None or spec.suggest is None or ctx is None:
                continue
            try:
                row.set_suggestion(spec.suggest(ctx))
            except Exception:
                row.set_suggestion(None)

    def overrides(self) -> dict:
        """Only the parameters the user actually pinned."""
        return {name: v for name, row in self.rows.items()
                if (v := row.value()) is not None}

    def set_overrides(self, values: dict) -> None:
        for name, v in (values or {}).items():
            if name in self.rows:
                self.rows[name].set_value(v)


# ==========================================================================
def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet("color:#e5e7eb;")
    return line


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font-weight:700;color:#374151;margin-top:6px;")
    return label


def build_context(state) -> object | None:
    """A :class:`DataContext` for the loaded feature table, or ``None``.

    Used only to fill in the Auto values, so it is built from whatever is
    already loaded and returns ``None`` rather than raising when nothing is.
    """
    from ..analysis import DataContext
    from ..preprocess import prepare

    ds = getattr(state, "features", None)
    if ds is None:
        return None
    try:
        prep = prepare(ds, state.config.preprocess,
                       selected=state.config.selected_features or None)
        return DataContext.from_prepared(
            prep, random_state=state.config.embedding.random_state)
    except Exception:
        return None


__all__ = ["Collapsible", "ParamRow", "MethodPanel", "separator",
           "section_label", "build_context"]
