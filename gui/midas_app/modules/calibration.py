"""Calibration module — embeds ``viewers.plot_calibrant_results.CalibrantViewer``.

PyQt6-based viewer; loaded lazily so it doesn't drag PyQt6 into the launcher
unless the user actually opens this tab.
"""

from __future__ import annotations
import os
import sys
from typing import Optional

from PyQt5 import QtCore, QtWidgets

_VIEWERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'viewers')
if _VIEWERS_DIR not in sys.path:
    sys.path.insert(0, _VIEWERS_DIR)


_CACHED = {}


def _lazy_load_calibrant():
    if 'mod' in _CACHED:
        return _CACHED['mod']
    try:
        import plot_calibrant_results as mod
        _CACHED['mod'] = mod
        return mod
    except Exception as e:
        _CACHED['mod'] = e
        return e


class CalibrationModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        self._viewer: Optional[QtWidgets.QWidget] = None

        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._build_picker())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    def _build_picker(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(40, 30, 40, 30)
        v.addWidget(QtWidgets.QLabel("<h3>Calibration — CalibrantPanelShiftsOMP results</h3>"))

        info = QtWidgets.QLabel(
            "Open a folder containing <code>*.corr.csv</code> output from "
            "<tt>CalibrantPanelShiftsOMP</tt> / <tt>CalibrantIntegratorOMP</tt>. "
            "Embeds <code>viewers/plot_calibrant_results.py</code>:CalibrantViewer "
            "with full functionality (X/Y/colour selectors, custom range, log scale, colourmap picker).")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555; padding-bottom: 8px;")
        v.addWidget(info)

        # Optional initial CSV
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Initial *.corr.csv (optional):"))
        self._file_edit = QtWidgets.QLineEdit()
        self._file_edit.setPlaceholderText("path/to/*.corr.csv")
        row.addWidget(self._file_edit)
        b = QtWidgets.QPushButton("Browse…")
        b.clicked.connect(self._on_browse)
        row.addWidget(b)
        v.addLayout(row)

        load = QtWidgets.QPushButton("Load")
        load.setFixedWidth(140)
        load.clicked.connect(self._on_load)
        v.addWidget(load)

        v.addStretch(1)
        return w

    def _on_browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose *.corr.csv", self._file_edit.text() or "",
            "Corr CSV (*.corr.csv);;CSV (*.csv);;All files (*)")
        if path:
            self._file_edit.setText(path)

    def _on_load(self):
        mod = _lazy_load_calibrant()
        if isinstance(mod, Exception):
            QtWidgets.QMessageBox.critical(self, "Calibrant viewer failed",
                                           f"Could not import plot_calibrant_results:\n\n{mod}")
            return
        path = self._file_edit.text().strip() or None
        if path and not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Missing file", f"Not a file: {path}")
            return
        try:
            self._viewer = mod.CalibrantViewer(filename=path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Calibrant viewer failed",
                                           f"Could not construct viewer:\n\n{e}")
            return
        self._viewer.setWindowFlags(QtCore.Qt.Widget)
        if self._stack.count() > 1:
            old = self._stack.widget(1)
            self._stack.removeWidget(old)
            old.deleteLater()
        self._stack.addWidget(self._viewer)
        self._stack.setCurrentIndex(1)

    def open_directory(self, path: str):
        # Look for any *.corr.csv in the directory and pre-fill
        if not os.path.isdir(path):
            return
        try:
            for name in sorted(os.listdir(path)):
                if name.endswith('.corr.csv'):
                    self._file_edit.setText(os.path.join(path, name))
                    return
        except OSError:
            pass
