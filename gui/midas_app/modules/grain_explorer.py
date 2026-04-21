"""3D Grain Explorer module — tabbed launchers for the standalone 3D plotters.

Each tab wraps one external script via ExternalLauncher:
  - Interactive FF Plotting   (Dash, browser)
  - 3D Spots                  (Plotly HTML)
  - 3D Spots colored by grain (Plotly HTML)
  - 3D Grains                 (Plotly HTML)
  - FF↔NF correlation plot    (matplotlib)
"""

from __future__ import annotations
import os

from PyQt5 import QtWidgets

from ..widgets.external_launcher import ExternalLauncher, ArgSpec

_GUI_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VIEWERS_DIR = os.path.join(_GUI_DIR, 'viewers')


def _interactive_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="Interactive FF Plotting (Dash)",
        description=(
            "Browser-based 3D grain explorer with linked spot/grain selection, "
            "colour mapping by lattice/strain/misorientation, and per-ring "
            "filtering. Runs as a Dash server in a separate process and opens "
            "in your default browser."),
        script_path=os.path.join(_VIEWERS_DIR, 'interactiveFFplotting.py'),
        args=[
            ArgSpec("Result folder", "-resultFolder", "", kind="dir", required=True,
                    placeholder="folder with Grains.csv / SpotMatrix.csv / *.MIDAS.zip"),
            ArgSpec("Zarr data file (optional)", "-dataFileName", "", kind="file",
                    placeholder="path/to/*.MIDAS.zip (auto-detected if blank)",
                    file_filter="MIDAS Zarr (*.MIDAS.zip *.zip);;All files (*)"),
            ArgSpec("Port", "-portNr", "8050", kind="int"),
            ArgSpec("Host", "-HostName", "localhost", kind="text"),
        ],
        open_url="http://localhost:{portNr}",
    )


def _spots3d_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="3D Spots (plotFFSpots3d.py)",
        description=(
            "Generates an interactive Plotly HTML of indexed FF spots from "
            "<code>InputAll.csv</code>. The script writes the HTML next to "
            "the script and opens it in your browser."),
        script_path=os.path.join(_VIEWERS_DIR, 'plotFFSpots3d.py'),
        args=[
            ArgSpec("Result folder", "-resultFolder", "", kind="dir", required=True,
                    placeholder="folder with InputAll.csv"),
        ],
    )


def _spots3d_grains_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="3D Spots colored by grain (plotFFSpots3dGrains.py)",
        description=(
            "Same scatter as 3D Spots, with marker color encoding grain ID "
            "from <code>SpotMatrix.csv</code> + <code>Grains.csv</code>."),
        script_path=os.path.join(_VIEWERS_DIR, 'plotFFSpots3dGrains.py'),
        args=[
            ArgSpec("Result folder", "-resultFolder", "", kind="dir", required=True,
                    placeholder="folder with SpotMatrix.csv and Grains.csv"),
        ],
    )


def _grains3d_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="3D Grain Centroids (plotGrains3d.py)",
        description=(
            "Plotly scatter of grain centroids from <code>Grains.csv</code>; "
            "marker size by grain size, colour by confidence/orientation."),
        script_path=os.path.join(_VIEWERS_DIR, 'plotGrains3d.py'),
        args=[
            ArgSpec("Result folder", "-resultFolder", "", kind="dir", required=True,
                    placeholder="folder with Grains.csv"),
        ],
    )


def _ffnf_tab() -> QtWidgets.QWidget:
    return ExternalLauncher(
        title="FF/NF correlation (PlotFFNF.py)",
        description=(
            "Generates a matplotlib scatter of grain-centroid displacement vs "
            "grain size, comparing FF reconstruction with NF microstructure."),
        script_path=os.path.join(_VIEWERS_DIR, 'PlotFFNF.py'),
        args=[],  # script reads everything from cwd
    )


class GrainExplorerModule(QtWidgets.QWidget):
    def __init__(self, theme: str = 'light', parent=None):
        super().__init__(parent)
        tabs = QtWidgets.QTabWidget()
        self._interactive = _interactive_tab()
        self._spots = _spots3d_tab()
        self._spots_g = _spots3d_grains_tab()
        self._grains = _grains3d_tab()
        self._ffnf = _ffnf_tab()
        tabs.addTab(self._interactive, "Interactive FF (Dash)")
        tabs.addTab(self._spots, "3D Spots")
        tabs.addTab(self._spots_g, "3D Spots / Grain")
        tabs.addTab(self._grains, "3D Grains")
        tabs.addTab(self._ffnf, "FF↔NF")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tabs)

    def open_directory(self, path: str):
        # Pre-fill the resultFolder field on every tab
        for w in (self._interactive, self._spots, self._spots_g, self._grains):
            edit = w._field_widgets.get("Result folder")
            if isinstance(edit, QtWidgets.QLineEdit):
                edit.setText(path)
