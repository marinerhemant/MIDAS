"""Typed loaders for MIDAS analysis results.

Thin wrappers around pandas / numpy / h5py that return ready-to-use objects.
Replaces the per-script CSV/H5 readers spread across the legacy viewers.
"""

from __future__ import annotations
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_grains_csv(path: str) -> pd.DataFrame:
    """Load Grains.csv. Header is the line beginning with %."""
    header_row = None
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('%'):
                header_row = i
    if header_row is None:
        return pd.read_csv(path, sep=r'\s+', engine='python')
    df = pd.read_csv(path, sep=r'\s+', engine='python', header=header_row)
    df.columns = [c.lstrip('%').strip() for c in df.columns]
    return df


def load_spot_matrix_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=r'\s+', engine='python', skiprows=1, header=None)


def load_corr_csv(path: str) -> pd.DataFrame:
    """Load *.corr.csv from CalibrantPanelShiftsOMP."""
    return pd.read_csv(path, sep=r'\s+', engine='python')


def load_lineout_xy(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Two-column whitespace text. Returns (x, y)."""
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def load_peaks_csv(path: str) -> pd.DataFrame:
    """Peak parameter table from extract_lineouts."""
    return pd.read_csv(path)


def load_caked_peaks_h5(path: str):
    """Open a *_caked_peaks.h5 file. Returns the open h5py.File (caller must close)."""
    import h5py
    return h5py.File(path, 'r')


def load_spot_diagnostics_bin(path: str, n_spots: int):
    """Memory-map a SpotDiagnostics.bin produced by PF-HEDM refinement.

    Returns ``np.memmap`` of shape ``(n_spots, ...)`` as float32, leaving the exact
    inner shape to the caller (it is set by the upstream binary writer).
    """
    return np.memmap(path, dtype=np.float32, mode='r')


def find_results(directory: str, pattern: str) -> list:
    """List files in ``directory`` matching shell pattern."""
    import fnmatch
    if not os.path.isdir(directory):
        return []
    return sorted(fnmatch.filter(os.listdir(directory), pattern))
