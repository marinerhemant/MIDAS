"""PF-HEDM Diagnostics module.

Two tabs:
  1. Sinogram (pfIntensityViewer) — Dash app, runs as subprocess + browser.
  2. Spot Diagnostics (utils/spot_diagnostics.SpotDiagPlotter) — matplotlib-based,
     embedded via FigureCanvasQTAgg.
"""

from __future__ import annotations
import os
import sys
from typing import Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from ..widgets.external_launcher import ExternalLauncher, ArgSpec

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VIEWERS_DIR = os.path.join(_GUI_DIR, 'viewers')
_UTILS_DIR = os.path.normpath(os.path.join(_GUI_DIR, '..', 'utils'))


def _build_sinogram_tab() -> QtWidgets.QWidget:
    script = os.path.join(_VIEWERS_DIR, 'pfIntensityViewer.py')
    return ExternalLauncher(
        title="PF-HEDM Sinogram Viewer (pfIntensityViewer.py)",
        description=(
            "Dash dashboard showing per-grain sinograms (scanNr × ω) and a 2-D "
            "intensity patch on click. Runs in a separate Python process and "
            "opens automatically in your browser. Use 'Stop all' to terminate."),
        script_path=script,
        args=[
            ArgSpec(label="ResultFolder", flag="-resultFolder", default="", kind="dir",
                    placeholder="path/to/PF result directory", required=True,
                    tooltip="Output directory from PF-HEDM reconstruction"),
            ArgSpec(label="Port", flag="-portNr", default="8050", kind="int",
                    tooltip="Local port to serve the dashboard"),
            ArgSpec(label="Host", flag="-HostName", default="localhost", kind="text"),
        ],
        open_url="http://localhost:{portNr}",
    )


class _SpotDiagnosticsTab(QtWidgets.QWidget):
    """Embeds matplotlib-based SpotDiagPlotter with a Figure canvas."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._diag = None
        self._plotter = None
        self._fig = None
        self._canvas = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("<h3>Spot Diagnostics — SpotDiagnostics.bin</h3>"))

        info = QtWidgets.QLabel(
            "Reads the per-voxel spot-match diagnostics binary written by "
            "PF-HEDM refinement. Click on the map to inspect spot intensities."
            "<br><i>Backed by</i> <code>utils/spot_diagnostics.py:SpotDiagPlotter</code>.")
        info.setWordWrap(True)
        info.setStyleSheet("color: #555;")
        layout.addWidget(info)

        # File picker
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("SpotDiagnostics.bin:"))
        self._file_edit = QtWidgets.QLineEdit()
        self._file_edit.setPlaceholderText("path/to/SpotDiagnostics.bin (required)")
        row.addWidget(self._file_edit)
        b1 = QtWidgets.QPushButton("Browse…")
        b1.clicked.connect(self._on_browse_bin)
        row.addWidget(b1)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Data dir (optional):"))
        self._data_edit = QtWidgets.QLineEdit()
        self._data_edit.setPlaceholderText("dataset directory (for click-to-intensity)")
        row2.addWidget(self._data_edit)
        b2 = QtWidgets.QPushButton("Browse…")
        b2.clicked.connect(self._on_browse_dir)
        row2.addWidget(b2)
        layout.addLayout(row2)

        row3 = QtWidgets.QHBoxLayout()
        row3.addWidget(QtWidgets.QLabel("Param file (optional):"))
        self._param_edit = QtWidgets.QLineEdit()
        self._param_edit.setPlaceholderText("path/to/ps.txt")
        row3.addWidget(self._param_edit)
        b3 = QtWidgets.QPushButton("Browse…")
        b3.clicked.connect(self._on_browse_param)
        row3.addWidget(b3)
        layout.addLayout(row3)

        row4 = QtWidgets.QHBoxLayout()
        load = QtWidgets.QPushButton("Load")
        load.setFixedWidth(120)
        load.clicked.connect(self._on_load)
        row4.addWidget(load)

        row4.addWidget(QtWidgets.QLabel("Voxel #:"))
        self._voxel_spin = QtWidgets.QSpinBox()
        self._voxel_spin.setRange(0, 1_000_000)
        self._voxel_spin.valueChanged.connect(self._refresh_plot)
        row4.addWidget(self._voxel_spin)
        row4.addStretch(1)
        layout.addLayout(row4)

        # Lazy: matplotlib canvas placeholder
        self._canvas_holder = QtWidgets.QFrame()
        self._canvas_holder.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self._canvas_holder.setStyleSheet("background:#fafafa;")
        ch_layout = QtWidgets.QVBoxLayout(self._canvas_holder)
        ch_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_layout = ch_layout
        self._placeholder = QtWidgets.QLabel(
            "\n   No diagnostics loaded.\n", alignment=QtCore.Qt.AlignCenter)
        self._placeholder.setStyleSheet("color:#aaa; font-size:13px;")
        ch_layout.addWidget(self._placeholder)
        layout.addWidget(self._canvas_holder, stretch=1)

        self._status = QtWidgets.QLabel(" ")
        self._status.setStyleSheet("color:#555; padding:4px;")
        layout.addWidget(self._status)

    def _on_browse_bin(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose SpotDiagnostics.bin", self._file_edit.text() or "",
            "Bin (*.bin);;All files (*)")
        if path:
            self._file_edit.setText(path)

    def _on_browse_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose dataset directory", self._data_edit.text() or "")
        if d:
            self._data_edit.setText(d)

    def _on_browse_param(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Choose param file", self._param_edit.text() or "")
        if path:
            self._param_edit.setText(path)

    def _on_load(self):
        path = self._file_edit.text().strip()
        if not path or not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Missing file", f"Not a file: {path}")
            return
        # Lazy import — keeps matplotlib + utils deps out of the launcher boot.
        if _UTILS_DIR not in sys.path:
            sys.path.insert(0, _UTILS_DIR)
        try:
            import spot_diagnostics as _sd
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Import failed",
                f"Could not import utils/spot_diagnostics:\n\n{e}")
            return
        try:
            self._diag = _sd.SpotDiagnostics(path)
            self._plotter = _sd.SpotDiagPlotter(
                self._diag,
                data_dir=(self._data_edit.text().strip() or None),
                param_file=(self._param_edit.text().strip() or None),
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Load failed", f"{e}")
            return

        # Build a matplotlib figure and embed via QtAgg
        try:
            import matplotlib
            matplotlib.use('QtAgg')
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import (
                FigureCanvasQTAgg as FigureCanvas,
                NavigationToolbar2QT as NavToolbar)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Matplotlib missing", str(e))
            return

        # Reset canvas area
        for i in reversed(range(self._canvas_layout.count())):
            it = self._canvas_layout.itemAt(i).widget()
            if it is not None:
                it.setParent(None)
                it.deleteLater()
        self._fig = Figure(figsize=(8, 6))
        self._canvas = FigureCanvas(self._fig)
        nav = NavToolbar(self._canvas, self._canvas_holder)
        self._canvas_layout.addWidget(nav)
        self._canvas_layout.addWidget(self._canvas, stretch=1)

        # Set voxel range and trigger first plot
        n = getattr(self._diag, 'n_voxels', None) or len(getattr(self._diag, 'voxels', []))
        if n:
            self._voxel_spin.setRange(0, max(0, int(n) - 1))
        self._voxel_spin.setValue(0)
        self._refresh_plot()
        self._status.setText(f"Loaded {os.path.basename(path)} (voxels: {n})")

    def _refresh_plot(self):
        if not self._plotter or not self._fig:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        try:
            self._plotter.plot_yz_scatter(self._voxel_spin.value(), ax=ax)
        except Exception as e:
            ax.text(0.5, 0.5, f"Plot failed:\n{e}", ha='center', va='center',
                    transform=ax.transAxes)
        self._canvas.draw_idle()


class PFDiagnosticsModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(_build_sinogram_tab(), "Sinogram (Dash)")
        self._spot_tab = _SpotDiagnosticsTab()
        tabs.addTab(self._spot_tab, "Spot Diagnostics")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

    def open_directory(self, path: str):
        # Try to pre-fill SpotDiagnostics.bin if present
        candidate = os.path.join(path, 'SpotDiagnostics.bin')
        if os.path.isfile(candidate):
            self._spot_tab._file_edit.setText(candidate)
            self._spot_tab._data_edit.setText(path)
