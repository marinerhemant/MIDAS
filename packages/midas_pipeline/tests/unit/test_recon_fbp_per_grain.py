"""``fbp_recon_per_grain`` must not let empty rows scale a reconstruction.

An all-zero sino row is not a measurement of zero — it is an unpopulated
(grain, hkl) cell, or a row the concentration filter zeroed. The
back-projection normalises by the number of projections it is handed, so
including such rows scales the whole reconstruction down by
``n_real / n_total``.

That is not a cosmetic amplitude issue. Voxel assignment in pf-HEDM is an
argmax across grains, so a grain that lost 30 % of its rows to the
concentration filter would reconstruct 0.7x dimmer than an unfiltered
neighbour and lose voxels it owns. These tests pin the invariance.
"""

from __future__ import annotations

import numpy as np
import pytest

from midas_pipeline.recon.fbp import fbp_recon_per_grain

N_SCANS = 16
N_REAL = 24


def _one_grain(n_pad: int = 0, shift: float = 2.5):
    """(sinos, omegas, n_hkls) for one grain, padded with ``n_pad`` empty rows.

    The empty rows carry plausible angles, so they are indistinguishable
    from real ones except by having no intensity.
    """
    n_tot = N_REAL + n_pad
    sinos = np.zeros((1, n_tot, N_SCANS), dtype=np.float64)
    omegas = np.zeros((1, n_tot), dtype=np.float64)
    real_ang = np.linspace(0.0, 180.0, N_REAL, endpoint=False)
    for h in range(N_REAL):
        c = N_SCANS / 2 + shift * np.sin(np.radians(real_ang[h]))
        sinos[0, h] = 1000.0 * np.exp(-0.5 * ((np.arange(N_SCANS) - c) / 1.5) ** 2)
    omegas[0, :N_REAL] = real_ang
    if n_pad:
        omegas[0, N_REAL:] = np.linspace(1.0, 179.0, n_pad)
    return sinos, omegas, np.array([n_tot], dtype=np.int64)


@pytest.mark.parametrize("n_pad", [1, 8, 24])
def test_empty_rows_do_not_change_the_reconstruction(tmp_path, n_pad):
    a = fbp_recon_per_grain(*_one_grain(n_pad=0), N_SCANS, tmp_path / f"a{n_pad}")
    b = fbp_recon_per_grain(*_one_grain(n_pad=n_pad), N_SCANS, tmp_path / f"b{n_pad}")
    # Exact, not approximate: the dropped rows contribute nothing, so the
    # surviving call is the identical computation on identical inputs.
    np.testing.assert_array_equal(b, a)


def test_all_empty_grain_stays_zero(tmp_path):
    sinos, omegas, n_hkls = _one_grain(n_pad=0)
    sinos[:] = 0.0
    out = fbp_recon_per_grain(sinos, omegas, n_hkls, N_SCANS, tmp_path / "z")
    assert out.shape == (1, N_SCANS, N_SCANS)
    assert not out.any()


def test_reconstruction_is_nonzero_and_centred(tmp_path):
    """Guard against the invariance tests passing on an all-zero output."""
    out = fbp_recon_per_grain(*_one_grain(n_pad=0), N_SCANS, tmp_path / "c")
    assert out.max() > 0
    # The grain sits near the middle of the field by construction.
    iy, ix = np.unravel_index(int(np.argmax(out[0])), out[0].shape)
    assert abs(iy - N_SCANS / 2) <= 3
    assert abs(ix - N_SCANS / 2) <= 3
