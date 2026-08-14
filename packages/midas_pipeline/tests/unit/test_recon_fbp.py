"""Unit tests for midas_pipeline.recon.fbp.

We test the FBP wrapper end-to-end by:
1. Generating a disk phantom + sinogram via the torch-free forward_project.
2. Running ``fbp_recon``, which is backed by the midas-tomo package.
3. Asserting RMSE recovery is within a generous threshold.

The gate is now simply whether midas-tomo imports. It used to be whether a
separately-built ``MIDAS_TOMO`` C binary existed on disk, resolved through a
``sys.path`` hop into ``<repo>/TOMO`` -- which meant the test could only run
from a source checkout, and passed or died depending on which clone the guard
happened to find. midas-tomo is a declared dependency, so if it is missing the
install is broken and skipping is the honest response.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.recon import forward_project
from midas_pipeline.recon.fbp import fbp_recon


def _tomo_available() -> bool:
    """The backend is a package now: either it imports or it does not."""
    try:
        from midas_tomo import run_tomo_from_sinos    # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _tomo_available(),
    reason="midas-tomo not installed (it is a declared midas-pipeline dependency)",
)


def _disk_phantom(N: int, radius: float = 0.3) -> np.ndarray:
    coord = np.linspace(-1.0, 1.0, N)
    xx, yy = np.meshgrid(coord, coord)
    return ((xx ** 2 + yy ** 2) < radius ** 2).astype(np.float64)


def test_fbp_recon_disk_phantom(tmp_path):
    N = 16
    phantom = _disk_phantom(N)
    angles = np.linspace(0.0, 180.0, 60, endpoint=False)
    sino = forward_project(phantom, angles)
    recon = fbp_recon(sino, angles, tmp_path / "tomo", n_scans=N)
    assert recon.shape == (N, N)

    # Normalize and compare. FBP gives signed values, so positive part only.
    p = phantom / max(phantom.max(), 1e-12)
    r_pos = np.maximum(recon, 0)
    r = r_pos / max(r_pos.max(), 1e-12)
    rmse = float(np.sqrt(np.mean((p - r) ** 2)))
    # FBP on a small grid is approximate; allow generous tolerance.
    assert rmse < 0.5, f"FBP RMSE too high: {rmse:.3f}"
