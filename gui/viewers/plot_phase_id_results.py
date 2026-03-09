#!/usr/bin/env python3
"""
Interactive phase_id.py single-phase results viewer (Qt edition).

Scans a work directory for *_lineout.txt and *_peaks.txt files
produced by phase_id.py --single-phase, and provides:
  - Dropdown to switch between data files
  - Lineout plot (measured vs calculated) with residual panel
  - Vertical peak markers at fitted 2θ positions
  - Peaks table below the plot
  - Log/linear Y toggle, plot controls

Usage:
    python plot_phase_id_results.py [work_dir]

If work_dir is omitted, the current directory is used.
"""

import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('QtAgg')

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QHBoxLayout, QHeaderView,
    QLabel, QMainWindow, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


# ── Data loading ────────────────────────────────────────────────────────

def discover_datasets(work_dir: Path):
    """Find paired *_lineout.txt / *_peaks.txt in work_dir.

    Returns a sorted list of (stem, lineout_path, peaks_path) tuples.
    A stem is present if at least the lineout file exists.
    """
    lineout_files = sorted(work_dir.glob('*_lineout.txt'))
    datasets = []
    for lf in lineout_files:
        stem = lf.name.replace('_lineout.txt', '')
        pf = work_dir / f'{stem}_peaks.txt'
        datasets.append((stem, lf, pf if pf.exists() else None))
    return datasets


def load_lineout(path: Path):
    """Load _lineout.txt → (tth, measured, calculated) arrays."""
    tth, meas, calc = [], [], []
    with open(path) as f:
        next(f)  # skip header
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                tth.append(float(parts[0]))
                meas.append(float(parts[1]))
                try:
                    calc.append(float(parts[2]))
                except ValueError:
                    calc.append(float('nan'))
    return np.array(tth), np.array(meas), np.array(calc)


def load_peaks(path: Path):
    """Load _peaks.txt → list of dicts with keys from header."""
    rows = []
    with open(path) as f:
        header = f.readline().split()
        for line in f:
            parts = line.split()
            if len(parts) >= len(header):
                row = {}
                for h, v in zip(header, parts):
                    try:
                        row[h] = float(v)
                    except ValueError:
                        row[h] = v
                rows.append(row)
    return header, rows


# ── Qt Main Window ──────────────────────────────────────────────────────

class PhaseIdViewer(QMainWindow):
    def __init__(self, work_dir: Path):
        super().__init__()
        self.work_dir = work_dir
        self.datasets = discover_datasets(work_dir)
        self.setWindowTitle(f'Phase ID Viewer — {work_dir}')
        self.resize(1200, 850)

        if not self.datasets:
            self._show_empty()
            return

        # State
        self.current_idx = 0
        self.log_y = True
        self.show_peaks = True
        self.show_calc = True

        self._build_ui()
        self._update()

    def _show_empty(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        lbl = QLabel(f'No *_lineout.txt files found in:\n{self.work_dir}')
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont('Helvetica', 14))
        layout.addWidget(lbl)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Control bar ─────────────────────────────────────────────
        controls = QHBoxLayout()

        # File selector
        controls.addWidget(QLabel('Dataset:'))
        self.combo_file = QComboBox()
        self.combo_file.setMinimumWidth(250)
        for stem, _, _ in self.datasets:
            self.combo_file.addItem(stem)
        self.combo_file.currentIndexChanged.connect(self._on_file_changed)
        controls.addWidget(self.combo_file)

        controls.addWidget(self._vsep())

        # Log Y
        self.chk_log = QCheckBox('Log Y')
        self.chk_log.setChecked(self.log_y)
        self.chk_log.toggled.connect(self._on_log)
        controls.addWidget(self.chk_log)

        # Show calculated
        self.chk_calc = QCheckBox('Show calc')
        self.chk_calc.setChecked(self.show_calc)
        self.chk_calc.toggled.connect(self._on_show_calc)
        controls.addWidget(self.chk_calc)

        # Show peak markers
        self.chk_peaks = QCheckBox('Peak markers')
        self.chk_peaks.setChecked(self.show_peaks)
        self.chk_peaks.toggled.connect(self._on_show_peaks)
        controls.addWidget(self.chk_peaks)

        controls.addStretch()

        # Navigation: prev/next
        info_lbl = QLabel()
        self.info_label = info_lbl
        controls.addWidget(info_lbl)

        main_layout.addLayout(controls)

        # ── Splitter: plot (top) + table (bottom) ───────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 5), dpi=100)
        # Two subplots: main + residual (shared x)
        self.ax_main = self.fig.add_axes([0.08, 0.35, 0.88, 0.58])
        self.ax_resid = self.fig.add_axes([0.08, 0.10, 0.88, 0.22],
                                          sharex=self.ax_main)
        self.canvas = FigureCanvasQTAgg(self.fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        # Table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(
            'QTableWidget { font-size: 12px; }'
            'QHeaderView::section { font-weight: bold; font-size: 12px; }'
        )
        splitter.addWidget(self.table)

        splitter.setSizes([550, 300])
        main_layout.addWidget(splitter)

    @staticmethod
    def _vsep():
        sep = QWidget()
        sep.setFixedWidth(2)
        sep.setStyleSheet('background-color: #999;')
        return sep

    # ── Callbacks ───────────────────────────────────────────────────
    def _on_file_changed(self, idx):
        self.current_idx = idx
        self._update()

    def _on_log(self, checked):
        self.log_y = checked
        self._update()

    def _on_show_calc(self, checked):
        self.show_calc = checked
        self._update()

    def _on_show_peaks(self, checked):
        self.show_peaks = checked
        self._update()

    # ── Update plot + table ─────────────────────────────────────────
    def _update(self):
        if not self.datasets:
            return

        stem, lineout_path, peaks_path = self.datasets[self.current_idx]
        self.info_label.setText(
            f'{self.current_idx + 1} / {len(self.datasets)}')

        # Load data
        tth, meas, calc = load_lineout(lineout_path)
        peak_header, peak_rows = [], []
        if peaks_path:
            peak_header, peak_rows = load_peaks(peaks_path)

        # ── Plot ────────────────────────────────────────────────────
        self.ax_main.clear()
        self.ax_resid.clear()

        # Measured
        self.ax_main.scatter(tth, meas, s=2, alpha=0.5,
                             label='Measured', color='steelblue', zorder=2)

        # Calculated
        if self.show_calc:
            self.ax_main.plot(tth, calc, linewidth=1.2,
                              label='Calculated', color='crimson', zorder=3)

        # Peak markers
        if self.show_peaks and peak_rows:
            for row in peak_rows:
                tth_peak = row.get('two_theta_deg', None)
                if tth_peak is not None:
                    self.ax_main.axvline(tth_peak, color='forestgreen',
                                         alpha=0.4, linewidth=0.8, zorder=1)

        # Y scale
        if self.log_y:
            self.ax_main.set_yscale('log')
        else:
            self.ax_main.set_yscale('linear')

        self.ax_main.set_title(stem, fontsize=11)
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.legend(fontsize=8, loc='upper right')
        self.ax_main.tick_params(labelbottom=False)

        # Residual
        resid = meas - calc
        self.ax_resid.plot(tth, resid, linewidth=0.8,
                           color='forestgreen', alpha=0.8)
        self.ax_resid.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        self.ax_resid.set_ylabel('Residual')
        self.ax_resid.set_xlabel('2θ (°)')

        self.canvas.draw_idle()

        # ── Table ───────────────────────────────────────────────────
        self._populate_table(peak_header, peak_rows)

    def _populate_table(self, header, rows):
        self.table.clear()
        if not header or not rows:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        self.table.setColumnCount(len(header))
        self.table.setRowCount(len(rows))
        self.table.setHorizontalHeaderLabels(header)

        for r, row in enumerate(rows):
            for c, col in enumerate(header):
                val = row.get(col, '')
                if isinstance(val, float):
                    text = f'{val:.6f}'
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(r, c, item)

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)

    def show(self):
        super().show()


# ── Entry point ─────────────────────────────────────────────────────────
def main():
    work_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('.')
    work_dir = work_dir.resolve()
    print(f'Scanning {work_dir} for phase_id results…')

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = PhaseIdViewer(work_dir)
    viewer.show()
    sys.exit(app.exec())


# MIDAS version banner
try:
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))), 'utils'))
    from version import version_string as _vs
    print(_vs())
except Exception:
    pass

if __name__ == '__main__':
    main()
