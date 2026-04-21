"""Live Integrator Monitor module — embeds ``viewers.live_viewer.LiveViewer``.

LiveViewer requires a lineout path at construction time, so the module shows a
small launch form first and swaps in the full viewer once the user clicks
Start. All LiveViewer functionality is preserved.
"""

from __future__ import annotations
import os
import sys
from typing import Optional

from PyQt5 import QtCore, QtWidgets

# viewers/ lives at gui/viewers/. midas_gui.py adds gui/ to sys.path; we add
# gui/viewers/ here so ``import live_viewer`` resolves.
_VIEWERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'viewers')
if _VIEWERS_DIR not in sys.path:
    sys.path.insert(0, _VIEWERS_DIR)

import live_viewer as _live_viewer  # noqa: E402


class LiveMonitorModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        self._theme = theme
        self._viewer: Optional[_live_viewer.LiveViewer] = None
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._build_launch_form())

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._stack)

    # ── Launch form ──────────────────────────────────────────────

    def _build_launch_form(self) -> QtWidgets.QWidget:
        form = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(form)
        grid.setContentsMargins(40, 30, 40, 30)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        title = QtWidgets.QLabel("<h3>Live Integrator Monitor</h3>")
        grid.addWidget(title, 0, 0, 1, 3)

        info = QtWidgets.QLabel(
            "Tail real-time GPU integrator output. Loads "
            "<code>viewers/live_viewer.py</code> with full functionality "
            "(scrolling heatmap, current lineout, per-peak parameter evolution).")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        grid.addWidget(info, 1, 0, 1, 3)

        # Lineout
        grid.addWidget(QtWidgets.QLabel("Lineout file:"), 2, 0)
        self._lineout_edit = QtWidgets.QLineEdit()
        self._lineout_edit.setPlaceholderText("path/to/lineout.bin (required)")
        grid.addWidget(self._lineout_edit, 2, 1)
        b1 = QtWidgets.QPushButton("Browse…")
        b1.clicked.connect(lambda: self._pick_file(self._lineout_edit, "lineout.bin"))
        grid.addWidget(b1, 2, 2)

        # Fit
        grid.addWidget(QtWidgets.QLabel("Fit file (optional):"), 3, 0)
        self._fit_edit = QtWidgets.QLineEdit()
        self._fit_edit.setPlaceholderText("path/to/fit.bin (optional — enables peak panel)")
        grid.addWidget(self._fit_edit, 3, 1)
        b2 = QtWidgets.QPushButton("Browse…")
        b2.clicked.connect(lambda: self._pick_file(self._fit_edit, "fit.bin"))
        grid.addWidget(b2, 3, 2)

        # Param file
        grid.addWidget(QtWidgets.QLabel("Param file (optional):"), 4, 0)
        self._params_edit = QtWidgets.QLineEdit()
        self._params_edit.setPlaceholderText("path/to/ps.txt (extracts Lsd, px, λ)")
        grid.addWidget(self._params_edit, 4, 1)
        b3 = QtWidgets.QPushButton("Browse…")
        b3.clicked.connect(lambda: self._pick_file(self._params_edit, "ps.txt"))
        grid.addWidget(b3, 4, 2)

        # nRBins, nPeaks
        grid.addWidget(QtWidgets.QLabel("nRBins:"), 5, 0)
        self._nbins_spin = QtWidgets.QSpinBox()
        self._nbins_spin.setRange(1, 100_000)
        self._nbins_spin.setValue(500)
        grid.addWidget(self._nbins_spin, 5, 1)

        grid.addWidget(QtWidgets.QLabel("nPeaks:"), 6, 0)
        self._npeaks_spin = QtWidgets.QSpinBox()
        self._npeaks_spin.setRange(0, 100)
        self._npeaks_spin.setValue(0)
        self._npeaks_spin.setToolTip("0 = no peak-evolution panel")
        grid.addWidget(self._npeaks_spin, 6, 1)

        grid.addWidget(QtWidgets.QLabel("History (frames, 0=∞):"), 7, 0)
        self._hist_spin = QtWidgets.QSpinBox()
        self._hist_spin.setRange(0, 1_000_000)
        self._hist_spin.setValue(0)
        grid.addWidget(self._hist_spin, 7, 1)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        start = QtWidgets.QPushButton("▶ Start")
        start.setFixedWidth(140)
        start.clicked.connect(self._start)
        btn_row.addWidget(start)
        grid.addLayout(btn_row, 8, 0, 1, 3)

        grid.setRowStretch(9, 1)
        return form

    def _pick_file(self, edit: QtWidgets.QLineEdit, label: str):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, f"Choose {label}", edit.text() or "")
        if path:
            edit.setText(path)

    def _start(self):
        lineout = self._lineout_edit.text().strip()
        if not lineout or not os.path.isfile(lineout):
            QtWidgets.QMessageBox.warning(self, "Missing file",
                                          f"Lineout file not found:\n{lineout}")
            return
        fit = self._fit_edit.text().strip() or None
        if fit and not os.path.isfile(fit):
            QtWidgets.QMessageBox.warning(self, "Missing file",
                                          f"Fit file not found:\n{fit}")
            return

        # Optional geometry from param file
        lsd = px = wavelength = None
        params = self._params_edit.text().strip()
        if params and os.path.isfile(params):
            try:
                pf = _live_viewer.parse_param_file(params)
                lsd = pf.get('lsd')
                px = pf.get('px')
                wavelength = pf.get('wavelength')
            except Exception as e:
                print(f"[LiveMonitor] param parse failed: {e}")

        try:
            self._viewer = _live_viewer.LiveViewer(
                lineout_path=lineout,
                fit_path=fit,
                n_rbins=self._nbins_spin.value(),
                n_peaks=self._npeaks_spin.value(),
                max_history=self._hist_spin.value(),
                theme=self._theme,
                lsd=lsd, px=px, wavelength=wavelength,
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "LiveViewer failed",
                                           f"Could not start viewer:\n\n{e}")
            return

        self._viewer.setWindowFlags(QtCore.Qt.Widget)
        self._stack.addWidget(self._viewer)
        self._stack.setCurrentIndex(1)
