"""Application entry point.

``python -m somtrack`` or ``somtrack-gui`` land here.
"""

from __future__ import annotations

import sys

import matplotlib

# Figures are built off the GUI thread and only then handed to a Qt canvas, so
# pyplot must never try to open a window of its own.
matplotlib.use("Agg", force=True)


def main(argv: list[str] | None = None) -> int:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow

    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, False)
    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("SOMTrack")
    app.setOrganizationName("SOMTrack")
    app.setStyle("Fusion")
    app.setStyleSheet(_STYLE)

    window = MainWindow()
    window.show()
    return app.exec()


_STYLE = """
QWidget { font-size: 10pt; }
QGroupBox {
    border: 1px solid #dfe3e8; border-radius: 5px;
    margin-top: 10px; padding-top: 8px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color:#39414b; }
QPushButton { padding: 5px 12px; border: 1px solid #cbd2d9;
              border-radius: 4px; background: #ffffff; }
QPushButton:hover { background: #eef4fa; }
QPushButton:disabled { color: #a8b0b8; background: #f3f4f6; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    padding: 3px 6px; border: 1px solid #cbd2d9; border-radius: 4px; background: white;
}
QTreeWidget, QListWidget, QTableView, QPlainTextEdit {
    border: 1px solid #dfe3e8; border-radius: 4px; background: white;
}
QHeaderView::section {
    background: #f4f6f8; border: none; border-right: 1px solid #e3e7eb;
    border-bottom: 1px solid #e3e7eb; padding: 4px 6px; font-weight: 600;
}
QTabBar::tab { padding: 6px 14px; }
QProgressBar { border: 1px solid #cbd2d9; border-radius: 4px; height: 10px; }
QProgressBar::chunk { background: #0072B2; border-radius: 3px; }
"""


if __name__ == "__main__":
    raise SystemExit(main())
