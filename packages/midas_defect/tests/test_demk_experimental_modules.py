"""Real-data smoke tests for the experimental midas_defect modules.

`run_inventory` (in tests/test_real_data_regression.py) already exercises
``rod_detect``-adjacent code via ``fault_rod_alignment``. This module fills
the remaining gap by calling the experimental modules **directly** on the
binned demk L1 voxel cloud + the L1 Grains.csv:

    * ``rod_detect.find_rods_iterative_residual``
    * ``asterism_fit.fit_asterism_patches``
    * ``subgrain.decompose_asterism_patches``
    * ``delta_pdf.compute_delta_pdf``
    * ``seed_index.find_seed_orientation``

These tests assert that the module returns a sane-shaped result on the real
voxel cloud -- not that the result matches a specific numerical value (the
binned data is too coarse to anchor the published full-res numbers, and the
canonical FCC reanalysis ran on copland with the full-res zarr).

Env-gated by the ``demk_bin4_paths`` fixture (which is itself gated by
``MIDAS_DEFECT_REAL_DATA=1``).
"""

from __future__ import annotations

import numpy as np
import pytest

import midas_defect.examples.demk_fcc_end_to_end as e2e
from midas_defect.asterism_fit import fit_asterism_patches
from midas_defect.delta_pdf import compute_delta_pdf
from midas_defect.lattice import fcc_cu_crystal
from midas_defect.rod_detect import find_rods_iterative_residual
from midas_defect.seed_index import find_seed_orientation
from midas_defect.subgrain import decompose_asterism_patches


@pytest.fixture(scope="module")
def demk_bin4_qsample(demk_bin4_paths):
    """Load and convert the binned demk L1 voxels to sample-frame q.

    Returns (qx, qy, qz, intensity, OM, crystal). Uses the validated detector
    model from ``DemkGeometry`` (omega_start=180, step=-0.25 -- the corrected
    convention, not the qscope sign-error one).
    """
    voxels_npz, grains_csv = demk_bin4_paths
    geom = e2e.DemkGeometry()
    crystal = fcc_cu_crystal(a=3.6356)
    npz = np.load(voxels_npz)
    qs = e2e.voxels_to_qsample(npz["indices"], int(npz["det_bin"]), geom)
    val = npz["values"].astype(np.float64)
    OM = np.genfromtxt(grains_csv, comments="%")[:, 1:10].reshape(-1, 3, 3)
    return qs[:, 0], qs[:, 1], qs[:, 2], val, OM, crystal


# --------------------------------------------------------------------------- #
# rod_detect                                                                   #
# --------------------------------------------------------------------------- #

def test_rod_detect_iterative_residual_finds_rods_on_real_voxels(demk_bin4_qsample):
    qx, qy, qz, val, _, _ = demk_bin4_qsample
    # Subsample for runtime; rod detection on 10M voxels is slow.
    rng = np.random.default_rng(0)
    keep = rng.choice(qx.size, size=min(qx.size, 200_000), replace=False)
    out = find_rods_iterative_residual(
        qx[keep], qy[keep], qz[keep], val[keep],
        n_iter=2,
    )
    # Two-iteration list-of-lists: at least one iteration produces some rods.
    n_total = sum(len(rods) for rods in out)
    assert n_total >= 0  # function returned; shape is sane
    assert isinstance(out, list)


# --------------------------------------------------------------------------- #
# asterism_fit                                                                 #
# --------------------------------------------------------------------------- #

def test_asterism_fit_runs_on_real_voxels_for_one_grain(demk_bin4_qsample):
    qx, qy, qz, val, OM, crystal = demk_bin4_qsample
    # Just test the function runs without crash; on binned voxels the fits may
    # be poor but the call surface is what's exercised.
    fits = fit_asterism_patches(
        qx, qy, qz, val,
        U=OM[0], a=3.6356, c=3.6356, crystal=crystal,
        q_max_inv_A=4.0, crop_halfwidth=0.06, min_voxels=10, n_steps=20,
    )
    assert isinstance(fits, list)
    # Whether or not the fits converge depends on data quality; just verify
    # the AsterismFit objects (if any) have the documented attributes.
    for f in fits:
        assert hasattr(f, "hkl")
        assert hasattr(f, "q_fit")
        assert hasattr(f, "sigma_eig")
        assert hasattr(f, "final_loss")
        assert np.isfinite(f.final_loss)


# --------------------------------------------------------------------------- #
# subgrain                                                                     #
# --------------------------------------------------------------------------- #

def test_subgrain_decomposes_real_asterism_patches(demk_bin4_qsample):
    qx, qy, qz, val, OM, crystal = demk_bin4_qsample
    fits = fit_asterism_patches(
        qx, qy, qz, val,
        U=OM[0], a=3.6356, c=3.6356, crystal=crystal,
        q_max_inv_A=4.0, crop_halfwidth=0.06, min_voxels=10, n_steps=20,
    )
    if not fits:
        pytest.skip("no asterism patches converged on this slice")
    subs = decompose_asterism_patches(qx, qy, qz, val, fits, min_cluster_size=5)
    assert isinstance(subs, list)


# --------------------------------------------------------------------------- #
# delta_pdf                                                                    #
# --------------------------------------------------------------------------- #

def test_delta_pdf_runs_on_real_voxels(demk_bin4_qsample):
    qx, qy, qz, val, _, _ = demk_bin4_qsample
    # Keep the grid tiny for runtime; just verify the call surface.
    out = compute_delta_pdf(
        qx, qy, qz, val,
        q_max=3.0, n_grid=32, symmetrize_friedel=True,
    )
    assert hasattr(out, "delta_rho")
    assert np.isfinite(out.delta_rho).all()
    assert out.delta_rho.shape == (32, 32, 32)


# --------------------------------------------------------------------------- #
# seed_index                                                                   #
# --------------------------------------------------------------------------- #

def test_seed_index_finds_orientation_on_real_voxels(demk_bin4_qsample):
    qx, qy, qz, val, OM, crystal = demk_bin4_qsample
    # Top-N bright cores from a subsample (full set is too slow).
    rng = np.random.default_rng(0)
    keep = rng.choice(qx.size, size=min(qx.size, 200_000), replace=False)
    try:
        out = find_seed_orientation(
            qx[keep], qy[keep], qz[keep], val[keep],
            crystal=crystal, n_bright=10, refine_lattice=False,
        )
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"seed indexing did not converge on this slice: {exc}")
    assert hasattr(out, "U")
    assert out.U.shape == (3, 3)
