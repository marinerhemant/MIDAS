"""Unit tests for midas_pipeline.recon.fbp.

We test the FBP wrapper end-to-end by:
1. Generating a disk phantom + sinogram via the torch-free forward_project.
2. Running ``fbp_recon`` (shells out to MIDAS_TOMO).
3. Asserting RMSE recovery is within a generous threshold.

The MIDAS_TOMO binary must be available on disk; we skip the test if
the importer cannot locate it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from midas_pipeline.recon import forward_project
from midas_pipeline.recon.fbp import _load_run_tomo_from_sinos, fbp_recon


def _tomo_binary_available() -> bool:
    """Ask the SAME resolver the code under test uses.

    This used to check two hardcoded absolute paths under ``~/opt/MIDAS``
    (one of them a literal ``/Users/hsharma/...``), while ``fbp_recon``
    resolves the binary relative to the repo root of the *imported*
    ``midas_pipeline`` — ``Path(fbp.__file__).parents[4] / "TOMO"``. In any
    checkout other than ``~/opt/MIDAS`` the two disagree: the guard finds the
    binary in the OTHER clone, declines to skip, and the test then dies with
    FileNotFoundError on this clone's path. That is what blocked a release
    run from a fresh clone (2026-07-31).

    Going through ``_find_tomo_exe`` keeps guard and code in agreement by
    construction, on any machine and any checkout.
    """
    try:
        _load_run_tomo_from_sinos()                   # puts TOMO/ on sys.path
        from midas_tomo_python import _find_tomo_exe  # type: ignore
    except ImportError:
        return False
    try:
        return Path(_find_tomo_exe()).is_file()
    except Exception:                                        # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _tomo_binary_available(),
    reason="MIDAS_TOMO binary not available on this machine",
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
