"""E1 regression test: ``RMSErrorStrain`` must be the real strain-solver
residual (µε), not the hardwired 0 users read as "perfect strain fit".

Builds on the ``tiny_run_dir`` fixture: seed 0's FitBest rows are doctored
into a self-consistent +1000 µε isotropic strain (obs radius from the
strained d-spacing via Bragg) over 7 g-directions, activating the
Kenesei per-spot solve (needs >= 6 good spots).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def _doctor_strained_grain(rd: Path, eps: float = 1e-3) -> None:
    """Give seed 0 seven spots at the radius of a d0*(1+eps) lattice."""
    Lsd, wl, d0 = 800000.0, 0.172979, 2.0784
    d_obs = d0 * (1 + eps)
    theta = math.degrees(math.asin(wl / (2 * d_obs)))
    R = Lsd * math.tan(math.radians(2 * theta))
    gdirs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0],
                      [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    gdirs /= np.linalg.norm(gdirs, axis=1, keepdims=True)

    pk = np.fromfile(rd / "Results" / "ProcessKey.bin", dtype=np.int32).reshape(3, 5000)
    pk[0, :7] = np.arange(101, 108)
    pk.tofile(rd / "Results" / "ProcessKey.bin")
    key = np.fromfile(rd / "Results" / "Key.bin", dtype=np.int32).reshape(3, 2)
    key[0, 1] = 7
    key.tofile(rd / "Results" / "Key.bin")
    ibf = np.fromfile(rd / "Output" / "IndexBestFull.bin",
                      dtype=np.float64).reshape(3, 5000, 2)
    ibf[0, :7, 0] = np.arange(101, 108)
    ibf.tofile(rd / "Output" / "IndexBestFull.bin")
    fb = np.fromfile(rd / "Output" / "FitBest.bin",
                     dtype=np.float64).reshape(3, 5000, 22)
    for j in range(7):
        ang = j * 0.7
        fb[0, j, 0] = 101 + j
        fb[0, j, 1] = R * math.cos(ang)     # observed y_lab (µm)
        fb[0, j, 2] = R * math.sin(ang)     # observed z_lab (µm)
        fb[0, j, 4:7] = gdirs[j]            # sample-frame ĝ
    fb.tofile(rd / "Output" / "FitBest.bin")


def test_rms_error_strain_nonzero_on_strained_synthetic(tiny_run_dir: Path):
    from midas_process_grains.pipeline import ProcessGrains

    _doctor_strained_grain(tiny_run_dir, eps=1e-3)
    pg = ProcessGrains.from_param_file(tiny_run_dir / "paramstest.txt",
                                       device="cpu")
    res = pg.run(mode="spot_aware")
    assert res.n_grains >= 1
    rms = np.asarray(res.rms_error_strain)
    strain = np.asarray(res.strain_grain)[0]
    # The planted isotropic strain must be recovered (prior-anchored
    # Kenesei pulls slightly toward Fable-Beaudoin, hence loose tol)...
    assert strain[0, 0] == pytest.approx(1e-3, rel=0.2)
    # ...and the solver residual must land in the column: non-zero, finite,
    # and far below the planted strain itself (µε units).
    assert np.isfinite(rms[0])
    assert 0.0 < rms[0] < 1e3, f"RMSErrorStrain={rms[0]} µε not plausible"
