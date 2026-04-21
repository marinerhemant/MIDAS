"""MIDAS unified GUI — main window with left navigation rail (PyQt5)."""

from __future__ import annotations
import argparse
import os
import sys
from typing import Optional

from PyQt5 import QtCore, QtGui, QtWidgets

from . import __version__
from .core.theme import apply_theme
from .core.log_panel import LogPanel
from .modules.ff_viewer import FFViewerModule
from .modules.nf_viewer import NFViewerModule
from .modules.live_monitor import LiveMonitorModule
from .modules.peak_inspector import PeakInspectorModule
from .modules.calibration import CalibrationModule
from .modules.image_tools import ImageToolsModule
from .modules.pf_diagnostics import PFDiagnosticsModule
from .modules.grain_explorer import GrainExplorerModule
from .modules.digital_twin import DigitalTwinModule


# Module registry: (label, factory, tooltip).
# All entries are enabled — Phase 2/3 modules wrap external scripts via the
# ExternalLauncher widget (subprocess + browser for Dash, etc).
MODULES = [
    ("FF Viewer", FFViewerModule,
     "FF-HEDM asymmetric detector viewer (embedded ff_asym_qt — full functionality)"),
    ("NF Viewer", NFViewerModule,
     "NF-HEDM dual-panel viewer (embedded nf_qt — full functionality)"),
    ("Live Integrator Monitor", LiveMonitorModule,
     "Real-time tail of GPU integrator output (embedded live_viewer)"),
    ("Peak / Lineout Inspector", PeakInspectorModule,
     "Caked peaks, extract_lineouts, phase_id (3 sub-tabs, lazy-loaded)"),
    ("Calibration", CalibrationModule,
     "CalibrantPanelShiftsOMP *.corr.csv viewer (embedded plot_calibrant_results)"),
    ("Image Tools", ImageToolsModule,
     "imageManipulation.py (Tkinter — runs in separate process)"),
    ("PF-HEDM Diagnostics", PFDiagnosticsModule,
     "Sinogram (Dash) + Spot Diagnostics (matplotlib embedded)"),
    ("3D Grain Explorer", GrainExplorerModule,
     "Interactive FF + 3D spots/grains plotters (subprocess + browser)"),
    ("Digital Twin", DigitalTwinModule,
     "dig_tw.py simulator + dt.py recon comparison (subprocess)"),
]


class MidasGUI(QtWidgets.QMainWindow):
    def __init__(self, theme: str = 'light', open_dir: Optional[str] = None):
        super().__init__()
        self.setWindowTitle(f"MIDAS — {__version__}")
        self.resize(1600, 1000)
        self._theme = theme

        # ── Central widget: nav list + stacked panels ────────────
        self._nav = QtWidgets.QListWidget()
        self._nav.setFixedWidth(220)
        self._nav.setStyleSheet("""
            QListWidget { background: #f4f4f4; border: none; padding: 6px; }
            QListWidget::item { padding: 8px 10px; margin-bottom: 2px; border-radius: 4px; }
            QListWidget::item:selected { background: #2a82da; color: white; }
            QListWidget::item:disabled { color: #999; }
        """)

        self._stack = QtWidgets.QStackedWidget()
        self._modules: dict = {}

        for label, factory, tip in MODULES:
            item = QtWidgets.QListWidgetItem(label)
            item.setToolTip(tip)
            try:
                widget = factory(theme=theme)
            except Exception as e:
                import traceback
                widget = QtWidgets.QLabel(
                    f"[{label}] failed to load:\n\n{e}\n\n{traceback.format_exc()}")
                widget.setStyleSheet("color: #a00; padding: 24px; font-family: monospace;")
                widget.setWordWrap(True)
            self._modules[label] = widget
            self._stack.addWidget(widget)
            self._nav.addItem(item)

        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)

        central = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(central)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(self._nav)
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.VLine)
        sep.setStyleSheet("color: #ccc;")
        h.addWidget(sep)
        h.addWidget(self._stack, stretch=1)
        self.setCentralWidget(central)

        # ── Log dock ─────────────────────────────────────────────
        self._log = LogPanel(self, title='Log')
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self._log)
        self._log.hide()

        # ── Menus ────────────────────────────────────────────────
        self._build_menus()

        self.statusBar().showMessage(
            f"Ready — {len(self._modules)} active module(s).")

        if MODULES:
            self._nav.setCurrentRow(0)

        if open_dir:
            self.open_directory(open_dir)

    def _build_menus(self):
        mb = self.menuBar()

        m_file = mb.addMenu("&File")
        a_open = QtWidgets.QAction("Open &Directory…", self)
        a_open.setShortcut("Ctrl+O")
        a_open.triggered.connect(self._on_open_dir)
        m_file.addAction(a_open)
        m_file.addSeparator()
        a_quit = QtWidgets.QAction("&Quit", self)
        a_quit.setShortcut("Ctrl+Q")
        a_quit.triggered.connect(self.close)
        m_file.addAction(a_quit)

        m_view = mb.addMenu("&View")
        a_log = QtWidgets.QAction("Toggle &Log Panel", self, checkable=True)
        a_log.toggled.connect(self._log.setVisible)
        m_view.addAction(a_log)

        m_help = mb.addMenu("&Help")
        a_about = QtWidgets.QAction("&About MIDAS GUI", self)
        a_about.triggered.connect(self._on_about)
        m_help.addAction(a_about)

    def _on_open_dir(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Open MIDAS data directory")
        if d:
            self.open_directory(d)

    def _on_about(self):
        QtWidgets.QMessageBox.information(
            self, "About MIDAS GUI",
            f"<h3>MIDAS GUI {__version__}</h3>"
            "<p>Unified PyQt5 desktop shell for the MIDAS viewers.</p>"
            "<p>Each nav entry hosts a legacy viewer at full functionality. "
            "Qt-native viewers (FF, NF, Live, Peaks, Calibration, Spot Diagnostics) "
            "are embedded directly. Tkinter / Dash viewers (Image Tools, PF "
            "Sinogram, Interactive FF, Digital Twin, DT recon) launch in "
            "separate processes — the launcher tracks PIDs and lets you stop "
            "all from one button.</p>",
        )

    def open_directory(self, path: str):
        if not os.path.isdir(path):
            QtWidgets.QMessageBox.warning(self, "Open directory",
                                          f"Not a directory:\n{path}")
            return
        for name, mod in self._modules.items():
            if hasattr(mod, 'open_directory'):
                try:
                    mod.open_directory(path)
                except Exception as e:
                    print(f"[{name}] open_directory failed: {e}")
        self.statusBar().showMessage(f"Opened {path}", 5000)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="MIDAS unified GUI (Phase 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('directory', nargs='?', default=None,
                        help='Optional MIDAS data directory to open at startup.')
    parser.add_argument('--theme', choices=('light', 'dark'), default='light')
    args = parser.parse_args(argv)

    app = QtWidgets.QApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))
    apply_theme(app, args.theme)

    win = MidasGUI(theme=args.theme, open_dir=args.directory)
    win.show()
    return app.exec_()


if __name__ == '__main__':
    sys.exit(main())
