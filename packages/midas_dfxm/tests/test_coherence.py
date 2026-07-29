"""Coherent / partially-coherent DFXM image formation."""
import math

import pytest
import torch

from midas_dfxm.coherence import (coherent_image, coherent_psf, dislocation_phase,
                                  exit_amplitude, incoherent_image,
                                  partially_coherent_image)
from midas_dfxm.aberration import ABERRATIONS, aberrated_psf, coeffs_to_tensor

torch.set_default_dtype(torch.float64)


def _devices():
    devs = [("cpu", torch.float64)]
    if torch.cuda.is_available():
        devs.append(("cuda", torch.float64))
    return devs


@pytest.mark.unit
def test_coherent_psf_energy_normalized():
    h = coherent_psf(torch.zeros(len(ABERRATIONS)), grid_size=48)
    assert torch.is_complex(h)
    assert abs(float((h.real ** 2 + h.imag ** 2).sum()) - 1.0) < 1e-10


@pytest.mark.unit
def test_intensity_psf_matches_aberration_module():
    """|h|^2 from the coherent PSF must equal the intensity PSF, up to normalization."""
    c = {"astig0": 0.3, "comaX": 0.2}
    h = coherent_psf(c, grid_size=48)
    i_coh = (h.real ** 2 + h.imag ** 2)
    i_ab = aberrated_psf(c, grid_size=48)
    assert float((i_coh / i_coh.sum() - i_ab / i_ab.sum()).abs().max()) < 1e-12


@pytest.mark.unit
def test_uniform_object_stays_uniform():
    """A uniform, flat-phase object must image uniformly in both modes (no artefacts)."""
    h = coherent_psf(torch.zeros(len(ABERRATIONS)), grid_size=48)
    ones = torch.ones(48, 48)
    psi = exit_amplitude(ones, torch.zeros(48, 48))
    Ic, Ii = coherent_image(psi, h), incoherent_image(ones, h)
    for I in (Ic, Ii):
        assert float(I.std() / I.mean()) < 1e-8


@pytest.mark.unit
def test_partial_coherence_limits():
    """mu=1/0 must reproduce the coherent/incoherent limits, up to the flux normalization."""
    h = coherent_psf({"comaX": 0.3}, grid_size=48)
    R = torch.rand(48, 48) + 0.1
    psi = exit_amplitude(R, 0.7 * torch.randn(48, 48))

    def unit(a):
        return a / a.sum()
    assert float((partially_coherent_image(psi, h, 1.0) - unit(coherent_image(psi, h))).abs().max()) < 1e-12
    assert float((partially_coherent_image(psi, h, 0.0) - unit(incoherent_image(R, h))).abs().max()) < 1e-12


@pytest.mark.unit
def test_partial_coherence_conserves_flux_and_interpolates():
    """Partial coherence redistributes photons; total flux must not depend on mu.

    Guards a real bug: the two limits carry very different raw flux (amplitude vs
    intensity PSF normalization), so mixing them without equalizing first made mu
    almost inert until it reached 1.
    """
    h = coherent_psf({"comaX": 0.3}, grid_size=48)
    R = torch.rand(48, 48) + 0.1
    psi = exit_amplitude(R, 0.7 * torch.randn(48, 48))
    sums = [float(partially_coherent_image(psi, h, m).sum()) for m in (0.0, 0.5, 1.0)]
    assert max(sums) - min(sums) < 1e-10           # flux conserved
    # and mu must actually move the image, monotonically away from the incoherent limit
    ref = partially_coherent_image(psi, h, 0.0)
    d = [float((partially_coherent_image(psi, h, m) - ref).abs().sum()) for m in (0.25, 0.5, 1.0)]
    assert d[0] < d[1] < d[2] and d[0] > 1e-6


@pytest.mark.unit
def test_non_integer_gdotb_rejected():
    """g.b must be an integer for a perfect dislocation -- reject anything else.

    b is a lattice translation and g a reciprocal-lattice vector, so g.b is necessarily
    an integer. A non-integer value makes exp(i g.u) multivalued, and the resulting image
    depends on where the arbitrary branch cut is placed (measured: it changes by ~900x).
    """
    ax = torch.linspace(-1, 1, 16)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    for bad in (1.7, 0.5, -2.3):
        with pytest.raises(ValueError, match="not an integer"):
            dislocation_phase(X, Y, bad)


@pytest.mark.unit
def test_image_independent_of_branch_cut_placement():
    """GAUGE CHECK: the physical image cannot depend on where the branch cut is put.

    The cut plane of a dislocation is a bookkeeping choice, not physics. Rotating it must
    leave the image invariant -- which holds precisely because g.b is an integer.
    """
    N = 64
    ax = torch.linspace(-1.5, 1.5, N)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2).clamp_min(0.05)
    R = 0.06 ** 2 / (0.06 ** 2 + (0.02 * X / r ** 2) ** 2)
    h = coherent_psf(torch.zeros(len(ABERRATIONS)), grid_size=N)
    for gb in (1.0, 2.0):
        imgs = []
        for psi in (0.0, 0.7, 2.5):
            th = torch.atan2(Y, X)
            th_rot = torch.remainder(th - psi + math.pi, 2 * math.pi) - math.pi + psi
            I = coherent_image(exit_amplitude(R, gb * th_rot), h)
            imgs.append(I / I.mean())
        for k in (1, 2):
            assert float((imgs[k] - imgs[0]).abs().max()) < 1e-9


@pytest.mark.unit
def test_coherent_image_differentiable_in_aberration():
    c = torch.zeros(len(ABERRATIONS), requires_grad=True)
    h = coherent_psf(c, grid_size=32)
    R = torch.rand(32, 32) + 0.1
    psi = exit_amplitude(R, 0.5 * torch.randn(32, 32))
    coherent_image(psi, h).sum().backward()
    assert c.grad is not None and torch.isfinite(c.grad).all()
    assert float(c.grad.abs().sum()) > 0


@pytest.mark.unit
@pytest.mark.parametrize("device,dtype", _devices())
def test_device_portable(device, dtype):
    c = coeffs_to_tensor({"astig0": 0.2}, device=device, dtype=dtype)
    h = coherent_psf(c, grid_size=32, device=device, dtype=dtype)
    R = torch.rand(32, 32, device=device, dtype=dtype) + 0.1
    psi = exit_amplitude(R, torch.zeros(32, 32, device=device, dtype=dtype))
    I = coherent_image(psi, h)
    assert I.device.type == device and torch.isfinite(I).all()


@pytest.mark.integration
def test_coherent_more_aberration_sensitive_than_incoherent():
    """The core physical claim: coherently the pupil phase enters the image linearly."""
    ax = torch.linspace(-1.5, 1.5, 64)
    X, Y = torch.meshgrid(ax, ax, indexing="ij")
    r = torch.sqrt(X ** 2 + Y ** 2).clamp_min(0.05)
    R = 0.06 ** 2 / (0.06 ** 2 + (0.02 * X / r ** 2) ** 2)
    psi = exit_amplitude(R, dislocation_phase(X, Y, 2.0))   # g.b must be an integer

    def rel(mode, coeffs):
        h0 = coherent_psf(torch.zeros(len(ABERRATIONS)), grid_size=64)
        h1 = coherent_psf(coeffs, grid_size=64)
        f = (lambda h: coherent_image(psi, h)) if mode == "coh" else (lambda h: incoherent_image(R, h))
        a, b = f(h0), f(h1)
        a, b = a / a.mean(), b / b.mean()
        return float((b - a).abs().mean() / a.abs().mean())
    c = coeffs_to_tensor({"astig0": 0.5, "comaX": 0.3})
    assert rel("coh", c) > 10 * rel("inc", c)


@pytest.mark.integration
def test_real_dislocation_phase_winding_is_integer_and_reproduces_gdotb():
    """The strongest physical gate on the coherent path.

    For a REAL dislocation (anisotropic Stroh displacement) and a REAL reciprocal-lattice
    vector, the phase winding g.u around the core must be an exact integer multiple of
    2*pi -- because b is a lattice translation. It must also equal g.b, so reflections with
    g.b = 0 come out invisible. This simultaneously checks the displacement scale, the
    micrometre-vs-Angstrom unit conversion (a 1e4 error this caught), and the reciprocal
    lattice convention. |b| must be consistent with the lattice: a/sqrt(2) for FCC a/2<110>.
    """
    from midas_dfxm import cubic_stiffness, stroh_dislocation
    from midas_dfxm.field_inverse import reference_Q

    a = 3.6156
    latc = torch.tensor([a] * 3 + [90.0] * 3)
    C = cubic_stiffness(168.4, 121.4, 75.4, dtype=torch.float64)
    d = stroh_dislocation(C, burgers=(1, -1, 0), slip_normal=(1, 1, 1), character="edge",
                          burgers_length_A=a / math.sqrt(2), core_radius_um=0.05,
                          core_model="compact")
    th = torch.linspace(-math.pi + 1e-9, math.pi - 1e-9, 4000)
    r = 0.5
    loop = torch.stack([r * torch.cos(th), r * torch.sin(th), torch.zeros_like(th)], -1) @ d.M
    # b = (a/2)[1,-1,0], so g.b = (h - k)/... -> 0 for h == k, 1 for (200), 2 for (400)
    for hkl, expect in [((2, 0, 0), 1), ((1, 1, 1), 0), ((2, 2, 0), 0),
                        ((3, 1, 1), 1), ((4, 0, 0), 2)]:
        Q0 = reference_Q(hkl, torch.eye(3), latc)
        phi = (d.displacement(loop) * 1e4) @ Q0
        w = float(phi[-1] - phi[0]) / (2 * math.pi)
        assert abs(w - round(w)) < 1e-6, f"{hkl}: winding {w} is not an integer"
        assert round(w) == expect, f"{hkl}: g.b = {round(w)}, expected {expect}"
