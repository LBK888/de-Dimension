"""Application entry point.

``python -m somtrack`` or ``somtrack-gui`` land here.
"""

from __future__ import annotations

import sys

import matplotlib

# Figures are built off the GUI thread and only then handed to a Qt canvas, so
# pyplot must never try to open a window of its own.
matplotlib.use("Agg", force=True)

SETTINGS_ORG = "SOMTrack"
SETTINGS_APP = "SOMTrack"
LANGUAGE_KEY = "language"


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("SOMTrack")
    app.setOrganizationName("SOMTrack")
    app.setStyle("Fusion")

    # The language has to be settled before the first widget is built: every
    # label is translated as it is created.
    apply_language(app, choose_language())

    # Colours come from one theme, light or dark, so a dark Windows desktop
    # cannot leave light text on a white field (see theme.py).
    from . import theme

    theme.apply(app, theme.saved_mode())

    from .main_window import MainWindow

    window = MainWindow()
    window.show()
    return app.exec()


# --------------------------------------------------------------------------
def saved_language() -> str | None:
    from PySide6.QtCore import QSettings

    value = QSettings(SETTINGS_ORG, SETTINGS_APP).value(LANGUAGE_KEY)
    return str(value) if value else None


def save_language(code: str) -> None:
    from PySide6.QtCore import QSettings

    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(LANGUAGE_KEY, code)


def choose_language() -> str:
    """The saved choice, else ``SOMTRACK_LANG``, else the system's own language.

    A Taiwanese Windows installation therefore opens in Traditional Chinese the
    first time, and English everywhere else, until someone picks otherwise from
    the Language menu.
    """
    from PySide6.QtCore import QLocale

    from ..i18n import language_from_environment, normalise

    chosen = saved_language() or language_from_environment()
    if chosen:
        return normalise(chosen)
    system = QLocale.system()
    if system.language() == QLocale.Language.Chinese:
        return "zh_TW"
    return normalise(system.name())


def apply_language(app, code: str) -> str:
    """Switch the program's text, Qt's own dialogs and the fallback font."""
    from ..i18n import set_language

    lang = set_language(code)
    if lang == "en":
        return lang

    from PySide6.QtCore import QLibraryInfo, QLocale, QTranslator

    # Qt's own strings -- the buttons of a message box, the file dialog -- come
    # from the translations that ship with PySide6.
    path = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    app._somtrack_translators = []                      # keep them alive
    for name in ("qtbase", "qt"):
        tr = QTranslator(app)
        if tr.load(f"{name}_{lang}", path):
            app.installTranslator(tr)
            app._somtrack_translators.append(tr)
    # By name rather than by enum: QLocale.Territory only exists from Qt 6.2 on.
    QLocale.setDefault(QLocale(lang))

    from .widgets import cjk_families

    font = app.font()
    font.setFamilies(cjk_families(font.family()))
    app.setFont(font)
    return lang


if __name__ == "__main__":
    raise SystemExit(main())
