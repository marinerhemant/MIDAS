#!/usr/bin/env python3
"""
Interactive viewer for caked peak-fit results (Qt edition).

Reads ``*.caked.hdf.zarr.zip`` files and the corresponding
``*_caked_peaks.h5`` output from ``fit_caked_peaks.py``, providing:

  - Dropdown to select zarr file
  - Dropdown to select OmegaSumFrame
  - Slider/spinbox to select eta bin
  - Left panel: 2D caked heatmap with current eta highlighted
  - Right panel: 1D intensity vs 2θ profile with SNIP background,
    corrected signal, and fitted peak overlays
  - Bottom panel: peak parameter table (click row → highlight peak)
  - Toggles: raw, background, corrected, fitted envelope, peak markers, log scale

Usage:
    python plot_caked_peaks.py /path/to/results/
    python plot_caked_peaks.py data.caked.hdf.zarr.zip
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import zarr

# Peak-fitting functions (for profile reconstruction + on-the-fly bg)
SCRIPT_DIR = Path(__file__).resolve().parent
MIDAS_HOME = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(MIDAS_HOME / 'utils'))
from extract_lineouts import snip_background, pseudo_voigt_no_bg, _tch_eta_fwhm

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from PyQt6.QtCore import Qt, QSize
    from PyQt6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
        QHBoxLayout, QHeaderView, QLabel, QMainWindow, QSlider,
        QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
        QVBoxLayout, QWidget,
    )
    _QT6 = True
except ImportError:
    from PyQt5.QtCore import Qt, QSize
    from PyQt5.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
        QHBoxLayout, QHeaderView, QLabel, QMainWindow, QSlider,
        QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
        QVBoxLayout, QWidget,
    )
    _QT6 = False


# ── Data loading ────────────────────────────────────────────────────────

def discover_datasets(work_dir):
    """Find *.caked.hdf.zarr.zip files and their _caked_peaks.h5 companions.

    Returns list of (stem, zarr_path, peaks_h5_path_or_None).
    """
    work_dir = Path(work_dir)
    datasets = []

    # Find zarr files
    zarr_files = sorted(work_dir.glob('*.caked.hdf.zarr.zip'))
    if not zarr_files:
        # Maybe user pointed to a single file
        return datasets

    for zf in zarr_files:
        stem = zf.name
        for suf in ('.caked.hdf.zarr.zip', '.zarr.zip', '.zip'):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        peaks_h5 = work_dir / f'{stem}_caked_peaks.h5'
        datasets.append((stem, zf, peaks_h5 if peaks_h5.exists() else None))

    return datasets


def load_zarr_data(zarr_path):
    """Load axes and intensity data from a caked zarr file.

    Returns dict with:
        tth_axis, eta_axis, frame_keys, frames (dict frame_key → 2D array)
    """
    z = zarr.open(str(zarr_path), mode='r')

    retamap = z['REtaMap'][:]
    tth_axis = retamap[1][:, 0]
    eta_axis = retamap[2][0, :]

    frames = {}
    frame_keys = []
    if 'OmegaSumFrame' in z:
        osf = z['OmegaSumFrame']
        frame_keys = sorted(osf.keys(),
                            key=lambda k: int(k.split('_')[-1]))
        for fk in frame_keys:
            frames[fk] = osf[fk][:]
    elif 'SumFrames' in z:
        frame_keys = ['SumFrames']
        frames['SumFrames'] = z['SumFrames'][:]

    return {
        'tth_axis': tth_axis,
        'eta_axis': eta_axis,
        'frame_keys': frame_keys,
        'frames': frames,
    }


def load_peaks_h5(h5_path, frame_key, eta_idx):
    """Load fitted peaks for a specific (frame_key, eta_idx) from HDF5.

    Returns list of dicts with peak parameters.
    """
    if h5_path is None or not h5_path.exists():
        return []

    with h5py.File(str(h5_path), 'r') as hf:
        if 'peaks' not in hf:
            return []
        pk = hf['peaks']
        fk_arr = pk['frame_key'][:]
        eta_arr = pk['eta_idx'][:]

        # Build mask
        mask = np.zeros(len(fk_arr), dtype=bool)
        for i in range(len(fk_arr)):
            fk_val = fk_arr[i]
            if isinstance(fk_val, bytes):
                fk_val = fk_val.decode()
            if fk_val == frame_key and eta_arr[i] == eta_idx:
                mask[i] = True

        if not np.any(mask):
            return []

        result = []
        indices = np.where(mask)[0]
        for i in indices:
            result.append({
                'peak_nr': int(pk['peak_nr'][i]),
                'center_2theta': float(pk['center_2theta'][i]),
                'area': float(pk['area'][i]),
                'sig': float(pk['sig'][i]),
                'gam': float(pk['gam'][i]),
                'FWHM_deg': float(pk['FWHM_deg'][i]),
                'eta_mix': float(pk['eta_mix'][i]),
                'd_spacing_A': float(pk['d_spacing_A'][i]),
                'chi_sq': float(pk['chi_sq'][i]),
            })
        return result


def reconstruct_profile(tth, peaks):
    """Sum pseudo-Voigt profiles from all fitted peaks."""
    y = np.zeros_like(tth, dtype=float)
    for p in peaks:
        y += pseudo_voigt_no_bg(tth, p['area'], p['center_2theta'],
                                 p['sig'], p['gam'])
    return y


# ── Qt Main Window ──────────────────────────────────────────────────────

class CakedPeakViewer(QMainWindow):
    def __init__(self, work_dir):
        super().__init__()
        self.work_dir = Path(work_dir)
        self.datasets = discover_datasets(self.work_dir)

        # Also handle single-file mode
        if not self.datasets and self.work_dir.is_file():
            zf = self.work_dir
            stem = zf.name
            for suf in ('.caked.hdf.zarr.zip', '.zarr.zip', '.zip'):
                if stem.endswith(suf):
                    stem = stem[:-len(suf)]
                    break
            peaks_h5 = zf.parent / f'{stem}_caked_peaks.h5'
            self.datasets = [(stem, zf, peaks_h5 if peaks_h5.exists() else None)]
            self.work_dir = zf.parent

        self.setWindowTitle(f'Caked Peak Viewer — {self.work_dir}')

        # State
        self.zarr_data = None
        self.peaks_h5_path = None
        self.current_peaks = []
        self.show_raw = True
        self.show_bg = True
        self.show_corrected = True
        self.show_fit_profile = True
        self.show_peak_markers = True
        self.log_scale = False

        if not self.datasets:
            self._show_empty()
        else:
            self._build_ui()
            self._on_file_changed(0)

    def _show_empty(self):
        lbl = QLabel("No *.caked.hdf.zarr.zip files found.\n\n"
                      "Usage: python plot_caked_peaks.py /path/to/results/")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter if _QT6
                         else Qt.AlignCenter)
        self.setCentralWidget(lbl)
        self.resize(500, 200)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        # ── Top bar: file / frame / eta selectors + toggles ──
        top = QHBoxLayout()

        # File selector
        top.addWidget(QLabel("File:"))
        self.file_combo = QComboBox()
        for stem, _, _ in self.datasets:
            self.file_combo.addItem(stem)
        self.file_combo.currentIndexChanged.connect(self._on_file_changed)
        top.addWidget(self.file_combo)

        top.addWidget(self._vsep())

        # Frame selector
        top.addWidget(QLabel("Frame:"))
        self.frame_combo = QComboBox()
        self.frame_combo.currentIndexChanged.connect(self._on_frame_changed)
        top.addWidget(self.frame_combo)

        top.addWidget(self._vsep())

        # Eta selector
        top.addWidget(QLabel("η bin:"))
        self.eta_spin = QSpinBox()
        self.eta_spin.setMinimum(0)
        self.eta_spin.valueChanged.connect(self._on_eta_changed)
        top.addWidget(self.eta_spin)

        self.eta_label = QLabel("η = 0.0°")
        top.addWidget(self.eta_label)

        top.addWidget(self._vsep())

        # Toggles
        cb_raw = QCheckBox("Raw"); cb_raw.setChecked(True)
        cb_raw.toggled.connect(self._on_raw)
        top.addWidget(cb_raw)

        cb_bg = QCheckBox("BG"); cb_bg.setChecked(True)
        cb_bg.toggled.connect(self._on_bg)
        top.addWidget(cb_bg)

        cb_corr = QCheckBox("Corr"); cb_corr.setChecked(True)
        cb_corr.toggled.connect(self._on_corrected)
        top.addWidget(cb_corr)

        cb_fit = QCheckBox("Fit"); cb_fit.setChecked(True)
        cb_fit.toggled.connect(self._on_fit)
        top.addWidget(cb_fit)

        cb_markers = QCheckBox("Peaks"); cb_markers.setChecked(True)
        cb_markers.toggled.connect(self._on_markers)
        top.addWidget(cb_markers)

        cb_log = QCheckBox("Log"); cb_log.setChecked(False)
        cb_log.toggled.connect(self._on_log)
        top.addWidget(cb_log)

        top.addStretch()
        root_layout.addLayout(top)

        # ── Main area: heatmap + profile plot ──
        splitter_main = QSplitter(Qt.Orientation.Horizontal if _QT6
                                   else Qt.Horizontal)

        # Left: 2D heatmap
        self.fig_heat = Figure(figsize=(5, 5))
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.canvas_heat.setMinimumSize(QSize(350, 300))
        splitter_main.addWidget(self.canvas_heat)

        # Right: 1D profile
        self.fig_prof = Figure(figsize=(7, 5))
        self.ax_prof = self.fig_prof.add_subplot(111)
        self.canvas_prof = FigureCanvas(self.fig_prof)
        self.canvas_prof.setMinimumSize(QSize(450, 300))
        splitter_main.addWidget(self.canvas_prof)

        splitter_main.setSizes([400, 600])

        # Vertical splitter: plots on top, table on bottom
        splitter_vert = QSplitter(Qt.Orientation.Vertical if _QT6
                                   else Qt.Vertical)
        splitter_vert.addWidget(splitter_main)

        # Bottom: peak table
        self.table = QTableWidget()
        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows if _QT6
            else QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self._on_row_selected)
        self.table.setMinimumHeight(120)
        splitter_vert.addWidget(self.table)

        splitter_vert.setSizes([500, 200])
        root_layout.addWidget(splitter_vert)

        self.resize(1200, 800)

    @staticmethod
    def _vsep():
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine if _QT6 else QFrame.VLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken if _QT6 else QFrame.Sunken)
        return sep

    # ── Slot: file changed ──
    def _on_file_changed(self, idx):
        stem, zarr_path, peaks_h5 = self.datasets[idx]
        self.zarr_data = load_zarr_data(zarr_path)
        self.peaks_h5_path = peaks_h5

        # Populate frame combo
        self.frame_combo.blockSignals(True)
        self.frame_combo.clear()
        for fk in self.zarr_data['frame_keys']:
            self.frame_combo.addItem(fk)
        # Select last frame by default
        if self.zarr_data['frame_keys']:
            self.frame_combo.setCurrentIndex(len(self.zarr_data['frame_keys']) - 1)
        self.frame_combo.blockSignals(False)

        # Eta range
        n_eta = len(self.zarr_data['eta_axis'])
        self.eta_spin.blockSignals(True)
        self.eta_spin.setMaximum(max(0, n_eta - 1))
        self.eta_spin.setValue(0)
        self.eta_spin.blockSignals(False)
        self.eta_label.setText(f"η = {self.zarr_data['eta_axis'][0]:.1f}°")

        self._update()

    def _on_frame_changed(self, idx):
        self._update()

    def _on_eta_changed(self, val):
        if self.zarr_data is not None and val < len(self.zarr_data['eta_axis']):
            self.eta_label.setText(f"η = {self.zarr_data['eta_axis'][val]:.1f}°")
        self._update()

    def _on_raw(self, checked): self.show_raw = checked; self._update()
    def _on_bg(self, checked): self.show_bg = checked; self._update()
    def _on_corrected(self, checked): self.show_corrected = checked; self._update()
    def _on_fit(self, checked): self.show_fit_profile = checked; self._update()
    def _on_markers(self, checked): self.show_peak_markers = checked; self._update()
    def _on_log(self, checked): self.log_scale = checked; self._update()

    def _on_row_selected(self):
        """Highlight the selected peak on the profile plot."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        row_idx = rows[0].row()
        if row_idx < len(self.current_peaks):
            self._update(highlight_peak=row_idx)

    # ── Main update ──
    def _update(self, highlight_peak=None):
        if self.zarr_data is None:
            return

        frame_idx = self.frame_combo.currentIndex()
        if frame_idx < 0:
            return
        frame_key = self.zarr_data['frame_keys'][frame_idx]
        eta_idx = self.eta_spin.value()

        tth_axis = self.zarr_data['tth_axis']
        eta_axis = self.zarr_data['eta_axis']

        # Get intensity
        intensity_2d = self.zarr_data['frames'].get(frame_key)
        if intensity_2d is None:
            return

        profile = np.asarray(intensity_2d[:, eta_idx], dtype=np.float64)

        # SNIP background
        bg = snip_background(profile, n_iter=50)
        corrected = np.maximum(profile - bg, 0)

        # Load fitted peaks
        self.current_peaks = load_peaks_h5(
            self.peaks_h5_path, frame_key, eta_idx)

        # ── Update 2D heatmap ──
        self.ax_heat.clear()
        # Transpose so x=η, y=2θ  (like the standard cake plot)
        vmax = np.percentile(intensity_2d, 99.5) if intensity_2d.max() > 0 else 1
        self.ax_heat.imshow(
            intensity_2d, aspect='auto', origin='lower',
            extent=[eta_axis[0], eta_axis[-1], tth_axis[0], tth_axis[-1]],
            cmap='magma', vmin=0, vmax=vmax,
            interpolation='nearest',
        )
        # Highlight current eta
        eta_val = eta_axis[eta_idx]
        self.ax_heat.axvline(eta_val, color='#00E676', linewidth=1.5,
                             linestyle='--', alpha=0.9)
        self.ax_heat.set_xlabel('η (°)')
        self.ax_heat.set_ylabel('2θ (°)')
        self.ax_heat.set_title(f'{frame_key}')
        self.fig_heat.tight_layout()
        self.canvas_heat.draw_idle()

        # ── Update 1D profile ──
        self.ax_prof.clear()

        if self.show_raw:
            self.ax_prof.plot(tth_axis, profile, '-', color='#90A4AE',
                              linewidth=0.8, label='Raw', alpha=0.7)
        if self.show_bg:
            self.ax_prof.plot(tth_axis, bg, '-', color='#F44336',
                              linewidth=1.0, label='SNIP BG', alpha=0.8)
        if self.show_corrected:
            self.ax_prof.plot(tth_axis, corrected, '-', color='#2196F3',
                              linewidth=1.2, label='Corrected')

        if self.current_peaks and (self.show_fit_profile or self.show_peak_markers):
            if self.show_fit_profile:
                # Draw smooth envelope
                tth_dense = np.linspace(tth_axis[0], tth_axis[-1], len(tth_axis) * 5)
                y_fit = reconstruct_profile(tth_dense, self.current_peaks)
                self.ax_prof.plot(tth_dense, y_fit, '-', color='#4CAF50',
                                  linewidth=1.5, label='Fit', alpha=0.9)

            if self.show_peak_markers:
                for i, p in enumerate(self.current_peaks):
                    lw = 2.5 if i == highlight_peak else 1.0
                    alpha = 1.0 if i == highlight_peak else 0.6
                    self.ax_prof.axvline(
                        p['center_2theta'], color='#FF9800',
                        linestyle=':', linewidth=lw, alpha=alpha)

        if self.log_scale:
            self.ax_prof.set_yscale('log')
            self.ax_prof.set_ylim(bottom=0.1)

        eta_val = eta_axis[eta_idx]
        self.ax_prof.set_xlabel('2θ (°)')
        self.ax_prof.set_ylabel('Intensity')
        self.ax_prof.set_title(
            f'η bin {eta_idx} (η = {eta_val:.1f}°) — '
            f'{len(self.current_peaks)} peaks')
        self.ax_prof.legend(fontsize=8, loc='upper right')
        self.ax_prof.grid(True, alpha=0.2)
        self.fig_prof.tight_layout()
        self.canvas_prof.draw_idle()

        # ── Update peak table ──
        self._populate_table()

    def _populate_table(self):
        headers = ['#', '2θ (°)', 'Area', 'FWHM (°)', 'sig (cd²)',
                    'gam (cd)', 'η_mix', 'd (Å)', 'χ²']
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.current_peaks))

        for i, p in enumerate(self.current_peaks):
            vals = [
                str(p['peak_nr']),
                f"{p['center_2theta']:.5f}",
                f"{p['area']:.2f}",
                f"{p['FWHM_deg']:.5f}",
                f"{p['sig']:.3f}",
                f"{p['gam']:.3f}",
                f"{p['eta_mix']:.3f}",
                f"{p['d_spacing_A']:.5f}" if p['d_spacing_A'] > 0 else "—",
                f"{p['chi_sq']:.3f}",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable
                              if _QT6
                              else item.flags() & ~Qt.ItemIsEditable)
                self.table.setItem(i, j, item)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch if _QT6
            else QHeaderView.Stretch)

    def show(self):
        super().show()


# ── Entry point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Interactive viewer for caked peak-fit results')
    parser.add_argument('path', help='Directory with zarr + peaks files, '
                                      'or a single zarr file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    viewer = CakedPeakViewer(args.path)
    viewer.show()
    sys.exit(app.exec() if _QT6 else app.exec_())


if __name__ == '__main__':
    main()
