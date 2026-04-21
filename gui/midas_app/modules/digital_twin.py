"""Digital Twin module — tabs for dig_tw.py (Dash) and dt.py (Tkinter)."""

from __future__ import annotations
import os

from PyQt5 import QtWidgets

from ..widgets.external_launcher import ExternalLauncher, ArgSpec

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _digtw_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="Digital Twin — dig_tw.py (Dash)",
        description=(
            "Interactive FF/NF microstructure simulator. Lets you adjust "
            "detector geometry, beam, and crystal lattice and runs "
            "<tt>ForwardSimulationCompressed</tt> / <tt>simulateNF</tt> to "
            "produce live diffraction patterns. Runs as a Dash server in a "
            "separate process; the dashboard opens in your default browser."),
        script_path=os.path.join(_GUI_DIR, 'dig_tw.py'),
        args=[
            ArgSpec("Microstructure file", "-mic", "", kind="file", required=True,
                    placeholder="path/to/Grains.csv or *.mic",
                    file_filter="Mic / CSV (*.mic *.csv);;All files (*)"),
        ],
    )


def _dt_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="DT Reconstruction Comparison — dt.py (Tkinter)",
        description=(
            "Side-by-side viewer for diffraction-tomography Method-1 vs "
            "Method-2 binary reconstructions. Tkinter app — runs in a separate "
            "process."),
        script_path=os.path.join(_GUI_DIR, 'dt.py'),
        args=[],
    )


class DigitalTwinModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        tabs = QtWidgets.QTabWidget()
        self._digtw = _digtw_tab()
        self._dt = _dt_tab()
        tabs.addTab(self._digtw, "Microstructure simulator")
        tabs.addTab(self._dt, "Reconstruction compare")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

    def open_directory(self, path: str):
        # Try to pre-fill the mic field with a sensible default
        if not os.path.isdir(path):
            return
        candidates = ('Grains.csv',)
        for cand in candidates:
            full = os.path.join(path, cand)
            if os.path.isfile(full):
                edit = self._digtw._field_widgets.get("Microstructure file")
                if isinstance(edit, QtWidgets.QLineEdit):
                    edit.setText(full)
                return
