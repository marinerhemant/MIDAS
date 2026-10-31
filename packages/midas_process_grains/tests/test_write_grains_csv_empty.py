"""write_grains_csv with no kept grains writes an empty Grains.csv (#75 follow-up).

A layer whose clustering keeps nothing (e.g. a sparse 2-panel set: 19 indexed
seeds, 0 clusters) used to end in ``TypeError: ufunc 'invert' not supported``
at the very last step, after all the work was done.
"""
from __future__ import annotations

import numpy as np

from midas_process_grains.compute.c_parity_emit import write_grains_csv


def test_no_kept_grains_writes_header_only(tmp_path):
    out = tmp_path / "Grains.csv"
    summary = write_grains_csv(
        out_path=out, kept_grains=[], opf=np.zeros((0, 27)), fb=None,
        lattice_reference=np.array([4.08, 4.08, 4.08, 90.0, 90.0, 90.0]),
        distance_um=1.0e6, wavelength_a=0.2, space_group=225,
        progress=False, spot_cache=[])
    assert summary["n_grains"] == 0
    lines = out.read_text().splitlines()
    assert lines[0] == "%NumGrains 0"
    assert not any(not ln.startswith("%") for ln in lines)       # no data rows
    # and it reads back as an empty table
    assert np.loadtxt(out, comments="%", ndmin=2).size == 0
