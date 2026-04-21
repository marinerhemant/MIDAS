"""Peak / Lineout Inspector module.

Three sub-tabs, each embedding a full legacy viewer:
  - Caked Peaks   → viewers.plot_caked_peaks.CakedPeakViewer
  - Lineouts      → viewers.plot_lineout_results.LineoutViewer
  - Phase ID      → viewers.plot_phase_id_results.PhaseIdViewer

Each tab shows a small "Choose folder…" prompt until a working directory is
selected, then swaps in the full viewer with all of its functionality.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from PyQt5 import QtCore, QtWidgets

from ..widgets.external_launcher import ExternalLauncher, ArgSpec

_VIEWERS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'viewers')
if _VIEWERS_DIR not in sys.path:
    sys.path.insert(0, _VIEWERS_DIR)

# These viewers were originally written for PyQt6 (plot_lineout_results,
# plot_phase_id_results) or PyQt6-with-PyQt5-fallback (plot_caked_peaks). To
# avoid loading PyQt6 alongside PyQt5 unless the user actually opens one of
# these tabs, the imports happen lazily inside _lazy_import().
_LAZY_CACHE: dict = {}


def _lazy_import(modname: str):
    if modname in _LAZY_CACHE:
        cached = _LAZY_CACHE[modname]
        return cached if not isinstance(cached, Exception) else cached
    try:
        mod = __import__(modname)
    except Exception as e:
        _LAZY_CACHE[modname] = e
        return e
    _LAZY_CACHE[modname] = mod
    return mod


class _DeferredViewerTab(QtWidgets.QWidget):
    """A tab that shows a folder picker until the user supplies a directory,
    then swaps in the full viewer factory(viewer_cls, folder)."""

    def __init__(self, label: str, viewer_factory: Callable[[Path], QtWidgets.QWidget],
                 import_error: Optional[Exception] = None, parent=None):
        super().__init__(parent)
        self._label = label
        self._factory = viewer_factory
        self._import_error = import_error
        self._stack = QtWidgets.QStackedWidget()
        self._stack.addWidget(self._build_placeholder())
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._stack)

    def _build_placeholder(self) -> QtWidgets.QWidget:
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(40, 40, 40, 40)
        title = QtWidgets.QLabel(f"<h3>{self._label}</h3>")
        v.addWidget(title)

        if isinstance(self._import_error, Exception):
            err = QtWidgets.QLabel(
                f"<b>Failed to import the underlying viewer:</b><br>"
                f"<tt>{type(self._import_error).__name__}: {self._import_error}</tt>")
            err.setStyleSheet("color: #a00; padding: 12px; background: #fee; border-radius: 4px;")
            err.setWordWrap(True)
            v.addWidget(err)
            v.addStretch(1)
            return w

        info = QtWidgets.QLabel(
            "Choose a working directory to load the viewer with full functionality.")
        info.setStyleSheet("color: #555;")
        v.addWidget(info)

        row = QtWidgets.QHBoxLayout()
        self._dir_edit = QtWidgets.QLineEdit()
        self._dir_edit.setPlaceholderText("path/to/work_dir")
        row.addWidget(self._dir_edit)
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

    def open_directory(self, path: str) -> None:
        """Pre-fill the directory field. Loading is deferred until the user
        clicks Load — that way the underlying PyQt6 module doesn't import
        unless the user actually opens a peak viewer."""
        if not isinstance(self._import_error, Exception):
            self._dir_edit.setText(path)

    def _on_browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose work directory", self._dir_edit.text() or "")
        if d:
            self._dir_edit.setText(d)

    def _on_load(self):
        path = self._dir_edit.text().strip()
        if not path or not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(self, "Missing directory",
                                          f"Not a directory: {path}")
            return
        try:
            v = self._factory(Path(path))
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Viewer failed",
                                           f"Could not load viewer:\n\n{e}")
            return
        if hasattr(v, 'setWindowFlags'):
            v.setWindowFlags(QtCore.Qt.Widget)
        # Drop the placeholder, swap in the viewer
        if self._stack.count() > 1:
            old = self._stack.widget(1)
            self._stack.removeWidget(old)
            old.deleteLater()
        self._stack.addWidget(v)
        self._stack.setCurrentIndex(1)


class PeakInspectorModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        tabs = QtWidgets.QTabWidget()

        def caked_factory(d: Path):
            mod = _lazy_import('plot_caked_peaks')
            if isinstance(mod, Exception):
                raise mod
            return mod.CakedPeakViewer(work_dir=d)

        def lineout_factory(d: Path):
            mod = _lazy_import('plot_lineout_results')
            if isinstance(mod, Exception):
                raise mod
            return mod.LineoutViewer(work_dir=d)

        def phaseid_factory(d: Path):
            mod = _lazy_import('plot_phase_id_results')
            if isinstance(mod, Exception):
                raise mod
            return mod.PhaseIdViewer(work_dir=d)

        self._caked_tab = _DeferredViewerTab(
            "Caked Peak Viewer", caked_factory)
        self._lineout_tab = _DeferredViewerTab(
            "Lineout Results Viewer", lineout_factory)
        self._phaseid_tab = _DeferredViewerTab(
            "Phase ID Results Viewer", phaseid_factory)
        self._integrator_tab = _build_integrator_tab()
        self._sigma_tab = _build_sigma_tab()
        self._compare_tab = _build_lineout_compare_tab()
        self._caking_tab = _build_viz_caking_tab()

        tabs.addTab(self._caked_tab, "Caked Peaks")
        tabs.addTab(self._lineout_tab, "Lineouts")
        tabs.addTab(self._phaseid_tab, "Phase ID")
        tabs.addTab(self._integrator_tab, "Integrator Peaks")
        tabs.addTab(self._sigma_tab, "σ Statistics")
        tabs.addTab(self._compare_tab, "Lineout Compare")
        tabs.addTab(self._caking_tab, "Caking (Dash)")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

    def open_directory(self, path: str) -> None:
        # Embedded tabs: pre-fill picker (don't auto-load due to PyQt6 import).
        self._caked_tab.open_directory(path)
        self._lineout_tab.open_directory(path)
        self._phaseid_tab.open_directory(path)
        # External-launcher tabs: pre-fill where applicable.
        edit = self._sigma_tab._field_widgets.get("Results dir")
        if isinstance(edit, QtWidgets.QLineEdit):
            edit.setText(path)


# ── External-tool tabs ──────────────────────────────────────────────

def _build_integrator_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="Integrator peak fits (plot_integrator_peaks.py)",
        description=(
            "Auto-detects peaks in a caked-zarr file, fits area-normalised "
            "Thompson-Cox-Hastings pseudo-Voigt profiles (GSAS-II-compatible), "
            "and plots fitted 2θ vs eta. Supports optional CorrelatedCaking "
            "CSV overlay. Runs as a subprocess."),
        script_path=os.path.join(_VIEWERS_DIR, 'plot_integrator_peaks.py'),
        args=[
            ArgSpec("Caked zarr", "", "", kind="file", required=True,
                    placeholder="path/to/*.caked.hdf.zarr.zip",
                    file_filter="Zarr zip (*.caked.hdf.zarr.zip *.zip);;All files (*)"),
            ArgSpec("Frame", "--frame", "-1", kind="int",
                    tooltip="-1 = use OmegaSumFrame; 0+ = specific frame"),
            ArgSpec("Fit window", "--fit-window", "50", kind="int"),
            ArgSpec("Min height", "--min-height", "", kind="text",
                    placeholder="auto (leave blank)"),
            ArgSpec("Prominence", "--prominence", "", kind="text",
                    placeholder="auto (leave blank)"),
            ArgSpec("Save PNG", "--save", "", kind="text",
                    placeholder="output filename (leave blank for interactive)"),
        ],
    )


def _build_sigma_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="FF-HEDM peak-width statistics (peak_sigma_statistics.py)",
        description=(
            "Scans <code>LayerNr_*/Temp/*_PS.csv</code> files, computes per-layer "
            "σ_R / σ_η distributions, and writes plots + JSON summary. Runs "
            "headless (matplotlib Agg) in a subprocess."),
        script_path=os.path.join(_VIEWERS_DIR, 'peak_sigma_statistics.py'),
        args=[
            ArgSpec("Results dir", "", "", kind="dir", required=True,
                    placeholder="folder containing LayerNr_* subdirs"),
            ArgSpec("Output prefix", "--out", "sigma_statistics", kind="text"),
            ArgSpec("Param file", "--paramFN", "", kind="file",
                    placeholder="optional param file for 2θ mapping"),
            ArgSpec("Histogram bins", "--nbins", "50", kind="int"),
        ],
    )


def _build_lineout_compare_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="Lineout comparison (plot_lineout_comparison.py)",
        description=(
            "Overlays a calibrant lineout with an integrator lineout and ideal "
            "2θ lines from <code>hkls.csv</code>. Tkinter/matplotlib app — "
            "runs in a subprocess."),
        script_path=os.path.join(_VIEWERS_DIR, 'plot_lineout_comparison.py'),
        args=[
            ArgSpec("Calibrant lineout", "", "", kind="file", required=True,
                    placeholder="path/to/calibrant lineout.xy"),
            ArgSpec("Integrator lineout", "", "", kind="file",
                    placeholder="optional second lineout"),
            ArgSpec("Param file", "--paramFN", "", kind="file", required=True,
                    placeholder="path/to/ps.txt (provides geometry & hkls)"),
            ArgSpec("Log Y", "--log", "", kind="text",
                    placeholder="'true' to enable, blank for linear"),
        ],
    )


def _build_viz_caking_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="Caking heatmap (viz_caking.py)",
        description=(
            "Minimal Dash heatmap viewer for a <code>*.caked.hdf.zarr.zip</code> "
            "file. Runs as a subprocess and opens in your browser."),
        script_path=os.path.join(_VIEWERS_DIR, 'viz_caking.py'),
        args=[
            ArgSpec("Caked zarr", "-fn", "", kind="file", required=True,
                    placeholder="path/to/*.caked.hdf.zarr.zip",
                    file_filter="Zarr zip (*.caked.hdf.zarr.zip *.zip);;All files (*)"),
        ],
        open_url="http://localhost:8050",
    )
