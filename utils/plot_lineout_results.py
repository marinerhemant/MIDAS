#!/usr/bin/env python3
"""
Interactive extract_lineouts.py results viewer (Qt edition).

Scans a directory for *_lineout.xy and *_peaks.csv files
produced by extract_lineouts.py, and provides:
  - Dropdown to switch between data files
  - Lineout plot: raw intensity, SNIP background, corrected
  - Reconstructed profile from fitted peaks overlaid on corrected
  - Difference panel (corrected − fitted profile)
  - Vertical peak markers at fitted 2θ positions
  - Peaks parameter table below the plot
  - Log/linear Y toggle

Usage:
    python plot_lineout_results.py [directory]

If directory is omitted, the current directory is used.
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


# ── Profile functions (from extract_lineouts.py) ───────────────────────

def _tch_eta_fwhm(sig_centideg2, gam_centideg):
    """Thompson-Cox-Hastings: derive total FWHM (deg) and mixing eta."""
    fg = np.sqrt(max(8.0 * np.log(2.0) * max(sig_centideg2, 1e-12), 0)) / 100.0
    fl = max(gam_centideg, 1e-6) / 100.0
    fg2, fg3, fg4, fg5 = fg**2, fg**3, fg**4, fg**5
    fl2, fl3, fl4, fl5 = fl**2, fl**3, fl**4, fl**5
    FWHM = (fg5 + 2.69269*fg4*fl + 2.42843*fg3*fl2 + 4.47163*fg2*fl3
            + 0.07842*fg*fl4 + fl5) ** 0.2
    if FWHM < 1e-15:
        return 1e-15, 0.5
    ratio = fl / FWHM
    eta = np.clip(1.36603*ratio - 0.47719*ratio**2 + 0.11116*ratio**3, 0, 1)
    return FWHM, eta


def pseudo_voigt_profile(x, area, center, sig, gam):
    """Area-normalized pseudo-Voigt (GSAS-II parameters, no background)."""
    FWHM, eta = _tch_eta_fwhm(sig, gam)
    if FWHM < 1e-15:
        FWHM = 1e-15
    sigma_g = FWHM / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    G = (1.0 / (sigma_g * np.sqrt(2.0 * np.pi))) * np.exp(
        -0.5 * ((x - center) / sigma_g)**2)
    half_fwhm = FWHM / 2.0
    L = (half_fwhm / np.pi) / ((x - center)**2 + half_fwhm**2)
    return area * (eta * L + (1.0 - eta) * G)


# ── Data loading ────────────────────────────────────────────────────────

def discover_datasets(work_dir: Path):
    """Find *_lineout.xy files in work_dir.

    Returns sorted list of (stem, lineout_path, peaks_path_or_None).
    """
    lineout_files = sorted(work_dir.glob('*_lineout.xy'))
    datasets = []
    for lf in lineout_files:
        stem = lf.name.replace('_lineout.xy', '')
        pf = work_dir / f'{stem}_peaks.csv'
        datasets.append((stem, lf, pf if pf.exists() else None))
    return datasets


def load_lineout(path: Path):
    """Load _lineout.xy → (tth, raw, background, corrected) arrays."""
    tth, raw, bg, corr = [], [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                tth.append(float(parts[0]))
                raw.append(float(parts[1]))
                bg.append(float(parts[2]))
                corr.append(float(parts[3]))
    return np.array(tth), np.array(raw), np.array(bg), np.array(corr)


def load_peaks(path: Path):
    """Load _peaks.csv → (header_list, list of row dicts)."""
    rows = []
    header = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                # Parse header from comment line
                header = line.lstrip('# ').split(',')
                continue
            if not line:
                continue
            parts = line.split(',')
            if len(parts) >= len(header):
                row = {}
                for h, v in zip(header, parts):
                    try:
                        row[h] = float(v)
                    except ValueError:
                        row[h] = v
                rows.append(row)
    return header, rows


def reconstruct_profile(tth, peak_rows):
    """Sum pseudo-Voigt profiles from all fitted peaks."""
    profile = np.zeros_like(tth)
    for row in peak_rows:
        area = row.get('area', 0)
        center = row.get('center_2theta', 0)
        sig = row.get('sig_centideg2', 10)
        gam = row.get('gam_centideg', 5)
        if area > 0 and center > 0:
            profile += pseudo_voigt_profile(tth, area, center, sig, gam)
    return profile


# ── Qt Main Window ──────────────────────────────────────────────────────

class LineoutViewer(QMainWindow):
    def __init__(self, work_dir: Path):
        super().__init__()
        self.work_dir = work_dir
        self.datasets = discover_datasets(work_dir)
        self.setWindowTitle(f'Lineout Viewer — {work_dir}')
        self.resize(1200, 900)

        if not self.datasets:
            self._show_empty()
            return

        # State
        self.current_idx = 0
        self.log_y = False
        self.show_raw = True
        self.show_bg = True
        self.show_profile = True
        self.show_peaks = True

        self._build_ui()
        self._update()

    def _show_empty(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        lbl = QLabel(f'No *_lineout.xy files found in:\n{self.work_dir}')
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setFont(QFont('Helvetica', 14))
        layout.addWidget(lbl)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ── Control bar ─────────────────────────────────────────────
        controls = QHBoxLayout()

        controls.addWidget(QLabel('Dataset:'))
        self.combo_file = QComboBox()
        self.combo_file.setMinimumWidth(300)
        for stem, _, _ in self.datasets:
            self.combo_file.addItem(stem)
        self.combo_file.currentIndexChanged.connect(self._on_file_changed)
        controls.addWidget(self.combo_file)

        controls.addWidget(self._vsep())

        self.chk_log = QCheckBox('Log Y')
        self.chk_log.setChecked(self.log_y)
        self.chk_log.toggled.connect(self._on_log)
        controls.addWidget(self.chk_log)

        self.chk_raw = QCheckBox('Raw')
        self.chk_raw.setChecked(self.show_raw)
        self.chk_raw.toggled.connect(self._on_raw)
        controls.addWidget(self.chk_raw)

        self.chk_bg = QCheckBox('Background')
        self.chk_bg.setChecked(self.show_bg)
        self.chk_bg.toggled.connect(self._on_bg)
        controls.addWidget(self.chk_bg)

        self.chk_profile = QCheckBox('Fitted profile')
        self.chk_profile.setChecked(self.show_profile)
        self.chk_profile.toggled.connect(self._on_profile)
        controls.addWidget(self.chk_profile)

        self.chk_peaks = QCheckBox('Peak lines')
        self.chk_peaks.setChecked(self.show_peaks)
        self.chk_peaks.toggled.connect(self._on_peaks)
        controls.addWidget(self.chk_peaks)

        controls.addStretch()

        self.info_label = QLabel()
        controls.addWidget(self.info_label)

        main_layout.addLayout(controls)

        # ── Splitter: plot (top) + table (bottom) ───────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Plot widget
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)

        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax_main = self.fig.add_axes([0.08, 0.35, 0.88, 0.58])
        self.ax_diff = self.fig.add_axes([0.08, 0.08, 0.88, 0.22],
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

        splitter.setSizes([600, 250])
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

    def _on_raw(self, checked):
        self.show_raw = checked
        self._update()

    def _on_bg(self, checked):
        self.show_bg = checked
        self._update()

    def _on_profile(self, checked):
        self.show_profile = checked
        self._update()

    def _on_peaks(self, checked):
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
        tth, raw, bg, corrected = load_lineout(lineout_path)
        peak_header, peak_rows = [], []
        if peaks_path:
            peak_header, peak_rows = load_peaks(peaks_path)

        # Reconstruct fitted profile
        profile = reconstruct_profile(tth, peak_rows) if peak_rows else None

        # ── Main plot ───────────────────────────────────────────────
        self.ax_main.clear()
        self.ax_diff.clear()

        # Corrected lineout (always shown)
        self.ax_main.plot(tth, corrected, linewidth=0.8,
                          label='Corrected', color='steelblue', zorder=2)

        # Raw intensity
        if self.show_raw:
            self.ax_main.plot(tth, raw, linewidth=0.5, alpha=0.4,
                              label='Raw', color='gray', zorder=1)

        # SNIP background
        if self.show_bg:
            self.ax_main.plot(tth, bg, linewidth=1.0, alpha=0.7,
                              label='SNIP background', color='orange',
                              linestyle='--', zorder=3)

        # Fitted profile
        if self.show_profile and profile is not None:
            self.ax_main.plot(tth, profile, linewidth=1.2,
                              label='Fitted profile', color='crimson',
                              zorder=4)

        # Peak markers (vertical lines)
        if self.show_peaks and peak_rows:
            for row in peak_rows:
                tth_peak = row.get('center_2theta', None)
                if tth_peak is not None:
                    self.ax_main.axvline(tth_peak, color='forestgreen',
                                         alpha=0.5, linewidth=0.8,
                                         linestyle=':', zorder=1)

        # Y scale
        if self.log_y:
            self.ax_main.set_yscale('log')
        else:
            self.ax_main.set_yscale('linear')

        self.ax_main.set_title(stem, fontsize=11)
        self.ax_main.set_ylabel('Intensity')
        self.ax_main.legend(fontsize=8, loc='upper right')
        self.ax_main.tick_params(labelbottom=False)

        # ── Difference panel ────────────────────────────────────────
        if profile is not None:
            diff = corrected - profile
            self.ax_diff.plot(tth, diff, linewidth=0.6,
                              color='purple', alpha=0.8)
            self.ax_diff.axhline(0, color='gray', linewidth=0.5,
                                  linestyle='--')
            self.ax_diff.set_ylabel('Difference')
        else:
            self.ax_diff.set_ylabel('(no peaks)')

        self.ax_diff.set_xlabel('2θ (°)')

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
    print(f'Scanning {work_dir} for lineout results…')

    app = QApplication.instance() or QApplication(sys.argv)
    viewer = LineoutViewer(work_dir)
    viewer.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
