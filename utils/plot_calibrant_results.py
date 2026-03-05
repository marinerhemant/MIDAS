#!/usr/bin/env python3
"""
Interactive CalibrantPanelShiftsOMP results viewer.

Auto-detects *corr.csv files in the current directory.
Provides dropdown selectors for X, Y, and Color columns,
plus a checkbox to include/exclude outlier points.

Usage:  python plot_calibrant_results.py [file.corr.csv]
"""

import glob
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons

plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13,
                     'axes.titlesize': 13, 'xtick.labelsize': 11,
                     'ytick.labelsize': 11})

# ── CSV column map (from CalibrantPanelShiftsOMP header) ────────────────
COLUMNS = [
    'Eta', 'Strain', 'RadFit', 'EtaCalc', 'DiffCalc', 'RadCalc',
    'Ideal2Theta', 'Outlier', 'YRawCorr', 'ZRawCorr', 'RingNr',
    'RadGlobal', 'IdealR', 'Fit2Theta', 'IdealA', 'FitA',
]
COL = {name: idx for idx, name in enumerate(COLUMNS)}

# Friendly display names for the selector
PLOTTABLE = [c for c in COLUMNS if c != 'Outlier']


# ── File discovery ──────────────────────────────────────────────────────
def find_corr_files():
    return sorted(glob.glob('*corr.csv'))


def choose_file(files):
    """Prompt user to pick a file if multiple are found."""
    if not files:
        print('No *corr.csv files found in the current directory.', file=sys.stderr)
        sys.exit(1)
    if len(files) == 1:
        return files[0]
    print('Multiple corr.csv files found:')
    for i, f in enumerate(files):
        print(f'  [{i}] {f}')
    while True:
        try:
            choice = int(input('Select file number: '))
            if 0 <= choice < len(files):
                return files[choice]
        except (ValueError, EOFError):
            pass
        print(f'Please enter 0–{len(files)-1}')


# ── Interactive plotter ─────────────────────────────────────────────────
class CalibrantPlotter:
    def __init__(self, filename):
        self.filename = filename
        self.raw = np.genfromtxt(filename, skip_header=1)
        self.exclude_bad = True
        self.x_col = 'EtaCalc'
        self.y_col = 'FitA'
        self.c_col = 'RingNr'
        self.cbar = None
        self._build_ui()
        self._update()

    @property
    def data(self):
        d = self.raw
        if self.exclude_bad:
            d = d[d[:, COL['Outlier']] == 0]
        return d

    # ── UI setup ────────────────────────────────────────────────────
    def _build_ui(self):
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.canvas.manager.set_window_title(f'Calibrant Viewer — {self.filename}')

        # Main axes and colorbar axes with fixed positions
        self.ax = self.fig.add_axes([0.22, 0.10, 0.65, 0.82])
        self.cax = self.fig.add_axes([0.89, 0.10, 0.02, 0.82])

        # ── Radio buttons for X column ──
        ax_x = self.fig.add_axes([0.01, 0.55, 0.16, 0.40])
        ax_x.set_title('X axis', fontsize=9, fontweight='bold')
        self.radio_x = RadioButtons(ax_x, PLOTTABLE,
                                    active=PLOTTABLE.index(self.x_col))
        self.radio_x.on_clicked(self._on_x)
        for lbl in self.radio_x.labels:
            lbl.set_fontsize(9)

        # ── Radio buttons for Y column ──
        ax_y = self.fig.add_axes([0.01, 0.10, 0.16, 0.40])
        ax_y.set_title('Y axis', fontsize=9, fontweight='bold')
        self.radio_y = RadioButtons(ax_y, PLOTTABLE,
                                    active=PLOTTABLE.index(self.y_col))
        self.radio_y.on_clicked(self._on_y)
        for lbl in self.radio_y.labels:
            lbl.set_fontsize(9)

        # ── Color selector (button cycling) ──
        ax_cbtn = self.fig.add_axes([0.01, 0.02, 0.12, 0.04])
        self.color_btn = Button(ax_cbtn, f'Color: {self.c_col}')
        self.color_btn.label.set_fontsize(9)
        self.color_btn.on_clicked(self._cycle_color)

        # ── Outlier checkbox ──
        ax_chk = self.fig.add_axes([0.14, 0.02, 0.07, 0.04])
        self.chk_bad = CheckButtons(ax_chk, ['Bad pts'], [not self.exclude_bad])
        self.chk_bad.on_clicked(self._toggle_bad)
        for lbl in self.chk_bad.labels:
            lbl.set_fontsize(9)

    # ── Callbacks ───────────────────────────────────────────────────
    def _on_x(self, label):
        self.x_col = label
        self._update()

    def _on_y(self, label):
        self.y_col = label
        self._update()

    def _cycle_color(self, _event):
        idx = PLOTTABLE.index(self.c_col)
        self.c_col = PLOTTABLE[(idx + 1) % len(PLOTTABLE)]
        self.color_btn.label.set_text(f'Color: {self.c_col}')
        self._update()

    def _toggle_bad(self, _label):
        self.exclude_bad = not self.exclude_bad
        self._update()

    # ── Redraw ──────────────────────────────────────────────────────
    def _update(self):
        d = self.data
        ax = self.ax
        ax.clear()

        if len(d) == 0:
            ax.text(0.5, 0.5, 'No data after filtering',
                    transform=ax.transAxes, ha='center', va='center')
            self.fig.canvas.draw_idle()
            return

        xi, yi, ci = COL[self.x_col], COL[self.y_col], COL[self.c_col]

        # Choose colormap: categorical for RingNr, diverging for Strain
        if self.c_col == 'RingNr':
            cmap = 'tab10'
        elif self.c_col == 'Strain':
            cmap = 'coolwarm'
        else:
            cmap = 'viridis'

        c_data = d[:, ci]
        c_label = self.c_col
        if self.c_col == 'Strain':
            c_data = c_data * 1e6
            c_label = 'Strain (µε)'

        sc = ax.scatter(d[:, xi], d[:, yi], c=c_data,
                        cmap=cmap, s=30, alpha=0.7, edgecolors='none')

        # Update colorbar in its fixed axes
        self.cax.clear()
        self.fig.colorbar(sc, cax=self.cax, label=c_label)

        # Reference line for lattice parameter plot
        if self.y_col == 'FitA':
            ideal_a = d[0, COL['IdealA']]
            ax.axhline(ideal_a, color='red', ls='-', alpha=0.5,
                       label=f'Ideal a = {ideal_a:.6f} Å')
            ax.legend(fontsize=8)

        n_total = len(self.raw)
        n_shown = len(d)
        ax.set_xlabel(self.x_col)
        ax.set_ylabel(self.y_col)
        ax.set_title(f'{self.filename}  ({n_shown}/{n_total} pts)', fontsize=10)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ── Entry point ─────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = choose_file(find_corr_files())

    print(f'Loading {filename} …')
    CalibrantPlotter(filename).show()


if __name__ == '__main__':
    main()
