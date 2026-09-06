"""Inverse of pixel_to_qlab, and the coupling between them.

The point of this file is the ROUND TRIP. Before it existed, a driver script
hand-rolled the flat-detector inverse and left the coupling as a comment; an
edit to `pixel_to_qlab` would have silently broken it. These tests make the
coupling fail loudly instead.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

from midas_defect.geometry import Geometry, pixel_to_qlab, qlab_to_pixel

PX, LSD, LAM = 172.0, 349_622.0, 0.42459
NR, NC, BR, BC = 1679, 1475, 810.3, 737.2


def _geom(tx=0.0, ty=0.0, tz=0.0, p3=0.0, rho_d=200_000.0):
    return Geometry(lsd_um=LSD, bcy_px=BC, bcz_px=BR, px_um=PX, wavelength_A=LAM,
                    n_pix_y=NC, n_pix_z=NR, omega_first_deg=0.0,
                    omega_step_deg=1.0, n_frames=1,
                    tx_deg=tx, ty_deg=ty, tz_deg=tz,
                    p_coeffs=(0., 0., 0., p3) + (0.,) * 11, rho_d_um=rho_d)


ROWS = np.array([700.0, 900.0, 810.3, 1200.0, 400.0, 1500.0])
COLS = np.array([500.0, 737.2, 1000.0, 300.0, 1200.0, 900.0])

GEOMS = [
    ("flat", _geom()),
    ("real tilt", _geom(tx=0.1489, ty=0.3729)),
    ("tilt + distortion", _geom(tx=0.1489, ty=0.3729, p3=1e-4)),
    ("large tilt", _geom(tx=1.0, ty=1.0, tz=0.5)),
]


@pytest.mark.parametrize("label,geom", GEOMS, ids=[g[0] for g in GEOMS])
def test_round_trip_pixel_q_pixel(label, geom):
    """THE coupling test. Any change to either map must keep this exact."""
    q = pixel_to_qlab(ROWS, COLS, geom, device="cpu")
    r, c = qlab_to_pixel(q, geom, device="cpu")
    assert np.allclose(r.numpy(), ROWS, atol=1e-4), f"{label}: rows drifted"
    assert np.allclose(c.numpy(), COLS, atol=1e-4), f"{label}: cols drifted"


@pytest.mark.parametrize("label,geom", GEOMS, ids=[g[0] for g in GEOMS])
def test_round_trip_q_pixel_q(label, geom):
    q0 = pixel_to_qlab(ROWS, COLS, geom, device="cpu")
    r, c = qlab_to_pixel(q0, geom, device="cpu")
    q1 = pixel_to_qlab(r.numpy(), c.numpy(), geom, device="cpu")
    assert torch.allclose(q0, q1, atol=1e-9)


def test_the_FLAT_inverse_is_wrong_once_the_detector_is_tilted():
    """Why this function iterates. Pins the error the shortcut would make."""
    geom = _geom(tx=0.1489, ty=0.3729)
    q = pixel_to_qlab(ROWS, COLS, geom, device="cpu")
    r, c = qlab_to_pixel(q, geom, device="cpu")

    # the shortcut a driver script would write: flat inverse, no iteration
    k0 = 2.0 * np.pi / LAM
    kf = q.numpy() / k0 + np.array([1.0, 0.0, 0.0])
    t = LSD / kf[:, 0]
    col_flat = BC - kf[:, 1] * t / PX
    row_flat = BR + kf[:, 2] * t / PX

    err = max(np.abs(col_flat - COLS).max(), np.abs(row_flat - ROWS).max())
    assert err > 1.0, (
        f"the flat shortcut was only {err:.3f} px off — if it is now accurate "
        "the iteration may be unnecessary, but check the tilt is really applied")
    # ...while the real inverse is exact
    assert np.abs(c.numpy() - COLS).max() < 1e-4
    assert np.abs(r.numpy() - ROWS).max() < 1e-4


def test_backward_rays_are_NaN_not_a_finite_pixel():
    """A reflection that does not go forward must not land somewhere plausible."""
    geom = _geom()
    k0 = 2.0 * np.pi / LAM
    q_back = np.array([[-2.1 * k0, 0.0, 0.0]])       # k_f_x < 0
    r, c = qlab_to_pixel(q_back, geom, device="cpu")
    assert np.isnan(r.numpy()).all() and np.isnan(c.numpy()).all()


def test_forward_and_backward_mixed_in_one_call():
    geom = _geom()
    good = pixel_to_qlab(np.array([800.0]), np.array([700.0]), geom, device="cpu")
    k0 = 2.0 * np.pi / LAM
    bad = torch.tensor([[-2.1 * k0, 0.0, 0.0]], dtype=good.dtype)
    r, c = qlab_to_pixel(torch.cat([good, bad]), geom, device="cpu")
    assert not np.isnan(r.numpy()[0]) and np.isnan(r.numpy()[1])


def test_shape_is_validated():
    with pytest.raises(ValueError, match="last dimension 3"):
        qlab_to_pixel(np.zeros((4, 2)), _geom(), device="cpu")


def test_non_convergence_raises_rather_than_returning_garbage():
    geom = _geom(tx=0.1489, ty=0.3729)
    q = pixel_to_qlab(ROWS, COLS, geom, device="cpu")
    with pytest.raises(RuntimeError, match="did not converge"):
        qlab_to_pixel(q, geom, max_iter=1, tol_px=1e-12, device="cpu")
