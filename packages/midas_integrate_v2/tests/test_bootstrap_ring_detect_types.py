"""``detect_rings`` and ``estimate_BC_from_image`` must accept a torch Tensor.

Both were numpy-only and raised a bare ``TypeError`` from deep inside when
handed a tensor::

    detect_rings:            '<=' not supported between instances of
                             'Tensor' and 'numpy.ndarray'
    estimate_BC_from_image:  unsupported operand type(s) for &:
                             'numpy.ndarray' and 'Tensor'

That matters because this package's idiom IS torch — ``IntegrationSpec`` fields
are tensors and ``integrate()`` returns one — so a tensor is the natural thing
to hand them. The science was never wrong (with numpy input both recover the
planted answer); only the boundary was.

These tests also pin the recovery itself, so a future refactor that quietly
degrades the detector or the beam-centre search fails here.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from midas_integrate_v2.ring_detect import detect_rings
from midas_integrate_v2.bootstrap import estimate_BC_from_image
from midas_integrate_v2.pdf import R_px_to_Q

LSD, PX, LAM = 500_000.0, 200.0, 0.172973
RING_R = [120.0, 250.0, 390.0, 610.0]


def _profile(n_r=800, seed=0):
    r = np.linspace(10.0, 810.0, n_r)
    p = np.full(n_r, 50.0)
    for R0 in RING_R:
        p += 5000.0 * np.exp(-((r - R0) ** 2) / (2 * 2.0 ** 2))
    return r, np.random.default_rng(seed).poisson(p).astype(float)


def _ring_image(N=512, BC=(270.5, 240.25), seed=0):
    Zi, Yi = np.mgrid[0:N, 0:N].astype(float)
    R = np.hypot(Yi - BC[0], Zi - BC[1])
    img = np.full_like(R, 20.0)
    for R0 in (60.0, 110.0, 170.0, 230.0):
        img += 4000.0 * np.exp(-((R - R0) ** 2) / (2 * 1.8 ** 2))
    return np.random.default_rng(seed).poisson(np.clip(img, 0, None)).astype(float)


def _radii(found):
    return sorted(float(f.R_px) for f in found)


# ---------------------------------------------------------------- ring_detect

@pytest.mark.parametrize("as_r,as_p", [(False, False), (True, True),
                                        (False, True), (True, False)])
def test_detect_rings_accepts_numpy_and_torch(as_r, as_p):
    r, p = _profile()
    a = torch.tensor(r) if as_r else r
    b = torch.tensor(p) if as_p else p
    found = detect_rings(a, b, Lsd_um=LSD, px_um=PX, wavelength_A=LAM)
    got = _radii(found)
    for t in RING_R:
        assert any(abs(g - t) < 3.0 for g in got), (
            f"missed the ring planted at {t}; found {got}")


def test_detect_rings_gives_the_same_answer_either_way():
    r, p = _profile()
    a = _radii(detect_rings(r, p, Lsd_um=LSD, px_um=PX, wavelength_A=LAM))
    b = _radii(detect_rings(torch.tensor(r), torch.tensor(p),
                            Lsd_um=LSD, px_um=PX, wavelength_A=LAM))
    assert a == pytest.approx(b, abs=1e-12)


def test_detect_rings_null_on_flat_background():
    """A detector that fires on pure noise is as bad as one that never fires."""
    r = np.linspace(10.0, 810.0, 800)
    flat = np.random.default_rng(1).poisson(np.full(800, 50.0)).astype(float)
    found = detect_rings(r, flat, Lsd_um=LSD, px_um=PX, wavelength_A=LAM)
    assert len(found) <= 2, f"{len(found)} rings detected in pure noise"


# ------------------------------------------------------------------ bootstrap

@pytest.mark.parametrize("as_torch", [False, True])
def test_estimate_bc_accepts_numpy_and_torch(as_torch):
    img = _ring_image()
    a = torch.tensor(img) if as_torch else img
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bc = estimate_BC_from_image(a, initial_BC=(256.0, 256.0),
                                    ring_radius_px=170.0)
    assert abs(float(bc[0]) - 270.5) < 3.0
    assert abs(float(bc[1]) - 240.25) < 3.0


def test_estimate_bc_same_answer_either_way():
    img = _ring_image()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = estimate_BC_from_image(img, initial_BC=(256.0, 256.0),
                                   ring_radius_px=170.0)
        b = estimate_BC_from_image(torch.tensor(img), initial_BC=(256.0, 256.0),
                                   ring_radius_px=170.0)
    assert float(a[0]) == pytest.approx(float(b[0]), abs=1e-12)
    assert float(a[1]) == pytest.approx(float(b[1]), abs=1e-12)


# ------------------------------------------------------------------ pdf

def test_R_to_Q_is_the_bragg_formula():
    R = torch.linspace(20.0, 800.0, 40, dtype=torch.float64)
    Q = R_px_to_Q(R, Lsd_um=LSD, px_um=PX, lambda_A=LAM).numpy()
    tth = np.arctan(R.numpy() * PX / LSD)
    ref = (4.0 * np.pi / LAM) * np.sin(0.5 * tth)
    assert np.abs(Q - ref).max() / np.abs(ref).max() < 1e-12
    assert (np.diff(Q) > 0).all(), "Q must increase with R"
    assert abs(float(R_px_to_Q(torch.zeros(1, dtype=torch.float64),
                               Lsd_um=LSD, px_um=PX,
                               lambda_A=LAM)[0])) < 1e-15
