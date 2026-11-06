"""Light and dark colour themes for the desktop app.

Every colour the interface uses lives here.  Widgets never carry a colour of
their own: they declare what they *are* with a ``role`` property --

    label.setProperty("role", "hint")

-- and the application style sheet built by :func:`stylesheet` decides what a
hint looks like in the current theme.

This is what keeps dark mode working.  Qt follows the system colour scheme on
its own, so a widget that hard-codes ``background: white`` but leaves the text
colour to the palette ends up with white text on a white field as soon as
Windows is switched to dark mode.  With one source for both colours that cannot
happen.

The figures themselves are not themed: they are publication figures, drawn on
white whatever the interface looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("system", "light", "dark")
THEME_KEY = "theme"


@dataclass(frozen=True)
class Theme:
    name: str
    dark: bool

    window: str          # page background
    panel: str           # sidebar and footer
    base: str            # input fields, lists, tables
    alt_base: str        # alternate table rows
    text: str
    muted: str           # hints, secondary labels
    faint: str           # disabled text
    border: str          # input borders
    border_soft: str     # group boxes, rules, table grid
    header: str          # table header background
    button: str
    button_hover: str
    button_disabled: str
    accent: str          # primary buttons, selection
    accent_hover: str
    accent_text: str
    link: str            # accent used as a text colour
    section: str         # group box and section titles
    collapsible: str     # the fold-out section headers
    note_bg: str         # the recipe and plateau call-outs
    warn: str            # cautions about supervised methods
    bad: str             # "too few selected"
    tooltip_bg: str
    tooltip_text: str
    verdict: tuple[str, str, str, str]   # strong, moderate, weak, none


LIGHT = Theme(
    name="light", dark=False,
    window="#f7f8fa", panel="#eef1f4", base="#ffffff", alt_base="#f4f6f8",
    text="#1f2933", muted="#5d6570", faint="#a3abb4",
    border="#cbd2d9", border_soft="#dfe3e8", header="#eef1f4",
    button="#ffffff", button_hover="#e8f1f9", button_disabled="#f1f3f5",
    accent="#0072B2", accent_hover="#005f94", accent_text="#ffffff",
    link="#0072B2", section="#39414b", collapsible="#2f5f9e",
    note_bg="#eaf2f9", warn="#8a3b00", bad="#b42318",
    tooltip_bg="#ffffff", tooltip_text="#1f2933",
    verdict=("#0072B2", "#007a5a", "#b06f00", "#6b7280"),
)

DARK = Theme(
    name="dark", dark=True,
    window="#1f2328", panel="#171a1e", base="#2a2f36", alt_base="#252a30",
    text="#e4e7eb", muted="#a7afb9", faint="#6f7882",
    border="#4a535e", border_soft="#3a414a", header="#30363e",
    button="#323941", button_hover="#3c4550", button_disabled="#262b31",
    accent="#1a7fc1", accent_hover="#2b91d6", accent_text="#ffffff",
    link="#62b3ee", section="#d3d8de", collapsible="#7fb6ec",
    note_bg="#24303d", warn="#f0a55c", bad="#ff8a80",
    tooltip_bg="#30363e", tooltip_text="#e4e7eb",
    verdict=("#56B4E9", "#2fc39a", "#E69F00", "#a7afb9"),
)

_current: dict = {"theme": LIGHT, "forced": ""}


def current() -> Theme:
    """The theme the interface is drawn in right now."""
    return _current["theme"]


# ==========================================================================
# Choosing a theme
# ==========================================================================
def system_is_dark() -> bool:
    """Whether the operating system asks for a dark interface (Qt 6.5+)."""
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except (AttributeError, RuntimeError):        # Qt < 6.5, or no application
        return False


def resolve(mode: str) -> Theme:
    mode = mode if mode in MODES else "system"
    if mode == "system":
        return DARK if system_is_dark() else LIGHT
    return DARK if mode == "dark" else LIGHT


def saved_mode() -> str:
    from PySide6.QtCore import QSettings

    from .app import SETTINGS_APP, SETTINGS_ORG

    value = QSettings(SETTINGS_ORG, SETTINGS_APP).value(THEME_KEY)
    return str(value) if value in MODES else "system"


def save_mode(mode: str) -> None:
    from PySide6.QtCore import QSettings

    from .app import SETTINGS_APP, SETTINGS_ORG

    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(THEME_KEY, mode)


def apply(app, mode: str = "system") -> Theme:
    """Draw the whole application in the theme ``mode`` asks for.

    Safe to call again while the window is open: Qt re-polishes every widget
    when the palette and the style sheet change.
    """
    _match_title_bar(mode)          # first: it changes what "system" reads as
    theme = resolve(mode)
    _current["theme"] = theme
    app.setPalette(palette(theme))
    app.setStyleSheet(stylesheet(theme))
    return theme


def _match_title_bar(mode: str) -> None:
    """Let the native title bar follow a forced theme (Qt 6.8+).

    Only touches Qt's own colour scheme when it has to: asking for the scheme
    that is already in force can emit ``colorSchemeChanged``, which would call
    straight back into :func:`apply`.
    """
    try:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QGuiApplication

        hints = QGuiApplication.styleHints()
        if not hasattr(hints, "setColorScheme"):
            return
        if mode == "system":
            if _current.get("forced"):
                _current["forced"] = ""
                hints.setColorScheme(Qt.ColorScheme.Unknown)
            return
        if _current.get("forced") != mode:
            _current["forced"] = mode
            hints.setColorScheme(Qt.ColorScheme.Dark if mode == "dark"
                                 else Qt.ColorScheme.Light)
    except (AttributeError, RuntimeError):
        pass


# ==========================================================================
# What a theme looks like
# ==========================================================================
def palette(t: Theme):
    """A complete QPalette, so nothing falls back to the platform's colours."""
    from PySide6.QtGui import QColor, QPalette

    p = QPalette()
    R, G = QPalette.ColorRole, QPalette.ColorGroup
    roles = {
        R.Window: t.window, R.WindowText: t.text,
        R.Base: t.base, R.AlternateBase: t.alt_base,
        R.Text: t.text, R.PlaceholderText: t.faint,
        R.Button: t.button, R.ButtonText: t.text,
        R.BrightText: "#ffffff",
        R.Highlight: t.accent, R.HighlightedText: t.accent_text,
        R.ToolTipBase: t.tooltip_bg, R.ToolTipText: t.tooltip_text,
        R.Link: t.link, R.LinkVisited: t.link,
        R.Light: t.button_hover, R.Midlight: t.border_soft,
        R.Mid: t.border, R.Dark: t.border, R.Shadow: "#000000",
    }
    for role, colour in roles.items():
        p.setColor(role, QColor(colour))
    for role in (R.WindowText, R.Text, R.ButtonText):
        p.setColor(G.Disabled, role, QColor(t.faint))
    p.setColor(G.Disabled, R.Button, QColor(t.button_disabled))
    p.setColor(G.Disabled, R.Base, QColor(t.window))
    p.setColor(G.Disabled, R.Highlight, QColor(t.border))
    return p


def stylesheet(t: Theme) -> str:
    return f"""
QWidget {{ font-size: 10pt; }}
QMainWindow, QDialog {{ background: {t.window}; }}
QToolTip {{
    color: {t.tooltip_text}; background: {t.tooltip_bg};
    border: 1px solid {t.border}; padding: 5px 7px;
}}

QGroupBox {{
    border: 1px solid {t.border_soft}; border-radius: 5px;
    margin-top: 10px; padding-top: 8px; font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {t.section}; }}

QPushButton {{ padding: 5px 12px; border: 1px solid {t.border};
               border-radius: 4px; background: {t.button}; color: {t.text}; }}
QPushButton:hover {{ background: {t.button_hover}; }}
QPushButton:disabled {{ color: {t.faint}; background: {t.button_disabled};
                        border-color: {t.border_soft}; }}
QPushButton[role="primary"] {{
    background: {t.accent}; color: {t.accent_text}; border: none;
    padding: 6px 20px; font-weight: 600;
}}
QPushButton[role="primary"]:hover {{ background: {t.accent_hover}; }}
QPushButton[role="primary"]:disabled {{ background: {t.border}; color: {t.faint}; }}
QPushButton[role="primary"][size="large"] {{ padding: 8px 26px; }}

/* Spin boxes are left to Fusion: a styled one loses its arrow buttons. */
QLineEdit, QComboBox {{
    padding: 3px 6px; border: 1px solid {t.border}; border-radius: 4px;
    background: {t.base}; color: {t.text};
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QLineEdit:disabled, QComboBox:disabled {{
    color: {t.faint}; background: {t.button_disabled}; border-color: {t.border_soft};
}}
/* the line edit inside a spin box or an editable combo box is not a field */
QAbstractSpinBox QLineEdit, QComboBox QLineEdit {{
    border: none; padding: 0; background: transparent;
}}
QComboBox QAbstractItemView {{
    background: {t.base}; color: {t.text}; border: 1px solid {t.border};
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}

QTreeWidget, QListWidget, QTableView, QPlainTextEdit, QTextBrowser {{
    border: 1px solid {t.border_soft}; border-radius: 4px;
    background: {t.base}; alternate-background-color: {t.alt_base}; color: {t.text};
    selection-background-color: {t.accent}; selection-color: {t.accent_text};
}}
QTableView {{ gridline-color: {t.border_soft}; }}
QTableCornerButton::section {{ background: {t.header}; border: none; }}
QHeaderView::section {{
    background: {t.header}; color: {t.text}; border: none;
    border-right: 1px solid {t.border_soft}; border-bottom: 1px solid {t.border_soft};
    padding: 4px 6px; font-weight: 600;
}}

QTabBar::tab {{ padding: 6px 14px; }}
QProgressBar {{ border: 1px solid {t.border}; border-radius: 4px; height: 10px;
                background: {t.base}; }}
QProgressBar::chunk {{ background: {t.accent}; border-radius: 3px; }}

/* ---- the window frame ------------------------------------------------ */
QWidget#sidebar, QWidget#footer {{ background: {t.panel}; }}
QWidget#footer {{ border-top: 1px solid {t.border_soft}; }}
QListWidget#steps {{ background: transparent; border: none; outline: 0; }}
QListWidget#steps::item {{ padding: 9px 8px; border-radius: 4px; color: {t.text}; }}
QListWidget#steps::item:hover {{ background: {t.button_hover}; }}
QListWidget#steps::item:selected {{ background: {t.accent}; color: {t.accent_text}; }}

/* ---- roles ------------------------------------------------------------ */
QLabel[role="brand"] {{ color: {t.link}; }}
QLabel[role="hint"], QLabel[role="status"] {{ color: {t.muted}; }}
QLabel[role="note"] {{ color: {t.muted}; font-size: 11px; }}
QLabel[role="required"] {{ color: {t.link}; }}
QLabel[role="warning"] {{ color: {t.warn}; font-size: 11px; }}
QLabel[role="section"] {{ color: {t.section}; font-weight: 700; margin-top: 6px; }}
QLabel[role="count"] {{ color: {t.muted}; }}
QLabel[role="count"][state="bad"] {{ color: {t.bad}; }}
QLabel[role="callout"] {{
    background: {t.note_bg}; color: {t.text};
    border-left: 3px solid {t.accent}; padding: 8px 10px;
}}
QFrame[role="rule"] {{ color: {t.border_soft}; }}
QCheckBox[role="collapsible"] {{ color: {t.collapsible}; font-weight: 600; }}
QCheckBox[role="collapsible"]::indicator {{ width: 0; height: 0; }}
QPlainTextEdit[role="log"] {{ background: {t.base}; color: {t.text}; }}
QScrollArea[role="canvas"] {{ background: {t.base}; border: 1px solid {t.border_soft}; }}
"""


def set_role(widget, role: str, **props) -> None:
    """Give ``widget`` a styling role (and extra properties), re-polishing it.

    Qt only re-reads a dynamic property when the widget is polished again, so a
    label that turns red when too few metrics are ticked has to be re-polished
    each time its state changes.
    """
    widget.setProperty("role", role)
    for name, value in props.items():
        widget.setProperty(name, value)
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)


__all__ = ["Theme", "LIGHT", "DARK", "MODES", "current", "resolve", "apply",
           "palette", "stylesheet", "set_role", "saved_mode", "save_mode",
           "system_is_dark"]
