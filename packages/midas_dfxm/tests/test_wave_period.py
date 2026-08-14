"""Regression: a planted eps_xy strain wave images at its true period, doubles at
exact Bragg, and splits into strain- vs tilt-sensing reflections.

Locks the forward-model behaviour that a /verify pass (2026-08) established but that
had no unit test: (1) the shear injection convention (F = I + H, H symmetric -> the
amplitude is the TENSOR shear, no factor of 2); (2) a 2 um wave is recovered at 2 um
off-Bragg and at 1 um (lambda/2) at exact Bragg; (3) a [110]-diagonal reflection senses
the shear as a longitudinal d-spacing change (strain) while a cube-axis [100] reflection
senses it as a transverse rotation (tilt) and is blind in theta-2theta. All in the
tetragonal crystal frame. See manuals/dfxm/ Notebook §9, README rule 22.
"""
import math

import numpy as np
import torch

from midas_dfxm import (
    GoniometerSetting,
    aligned_resolution,
    bragg_two_theta_deg,
    make_uniform_field,
    reference_q_nom,
    voxel_intensity,
)
from midas_dfxm.field import DeformationField, small_strain_from_F
from midas_dfxm.resolution import poulsen_resolution_widths

DT = torch.float64
WAVELENGTH = 12.398419843320026 / 20.0            # 20 keV, ~0.61992 A
CELL = (3.96, 3.96, 13.0, 90.0, 90.0, 90.0)       # inferred Ba122, tetragonal
A_WAVE = 1.0e-5
LAM = 2.0                                          # in-plane wavelength (um)


def _shear_line(nx=160, spacing=0.125):
    """1-D line of voxels along x carrying eps_xy(x) = A cos(2 pi x / LAM).

    span = nx*spacing = 20 um, so LAM=2 um lands on FFT bin 10 and LAM/2 on bin 20.
    """
    base = make_uniform_field(shape=(nx, 1, 1), spacing_um=spacing,
                              lattice_params=CELL, dtype=DT)
    x = base.positions[:, 0]
    e = A_WAVE * torch.cos(2 * math.pi * x / LAM)
    F = base.F.clone()
    F[:, 0, 1] = F[:, 0, 1] + e
    F[:, 1, 0] = F[:, 1, 0] + e
    field = DeformationField(positions=base.positions, F=F,
                             reference_orientation=base.reference_orientation,
                             lattice_params=base.lattice_params, shape=base.shape)
    return field, x


def _resolution(field, hkl, flank_sigmas):
    """Poulsen resolution centred on q_nom, offset by flank_sigmas onto the flank
    along the direction the wave moves Q (0 = exact Bragg)."""
    center = GoniometerSetting()
    q_nom = reference_q_nom(field, hkl, center)
    q_mag = float(torch.linalg.vector_norm(q_nom))
    tt = bragg_two_theta_deg(q_mag, WAVELENGTH)
    w = poulsen_resolution_widths(q_mag, two_theta_deg=tt)
    s_par, s_perp = w["sigma_par"], w["sigma_rock"]
    e_par = q_nom / torch.linalg.vector_norm(q_nom)
    dev = field.local_Q(hkl) - q_nom
    dev = dev - dev.mean(0, keepdim=True)
    evals, evecs = torch.linalg.eigh(dev.transpose(0, 1) @ dev)
    dQ_dir = evecs[:, -1] / torch.linalg.vector_norm(evecs[:, -1])
    c_par = float(torch.dot(dQ_dir, e_par))
    s_along = math.sqrt((c_par * s_par) ** 2 + (1 - c_par ** 2) * s_perp ** 2)
    q_c = q_nom + flank_sigmas * s_along * dQ_dir
    return aligned_resolution(q_c, sigma_par=s_par, sigma_perp=s_perp)


def _dominant_period_um(line, dx_um):
    y = np.asarray(line, float)
    y = y - y.mean()
    n = len(y)
    f = np.fft.rfftfreq(n, d=dx_um)
    P = np.abs(np.fft.rfft(y * np.hanning(n)))
    P[0] = 0.0
    k = int(np.argmax(P))
    return 1.0 / f[k]


# --------------------------------------------------------------------------

def test_shear_injection_convention_no_factor_of_two():
    """F = I + H with H_xy = e -> small_strain_from_F recovers eps_xy = e exactly."""
    field, _ = _shear_line()
    eps = small_strain_from_F(field.F)          # (N,3,3) = sym(F) - I
    x = field.positions[:, 0]
    planted = A_WAVE * torch.cos(2 * math.pi * x / LAM)
    assert torch.allclose(eps[:, 0, 1], planted, atol=1e-12), \
        "eps_xy not recovered -> wrong strain convention (factor of 2 / transpose)"
    # pure shear: normal strains stay zero
    assert eps[:, 0, 0].abs().max() < 1e-12
    assert eps[:, 1, 1].abs().max() < 1e-12


def test_wave_period_recovered_off_bragg():
    """Strain reflection [110] on the weak-beam flank images the TRUE 2 um period."""
    field, _ = _shear_line()
    hkl = (1, 1, 0)                              # tetragonal [110] = strain-sensing
    res = _resolution(field, hkl, flank_sigmas=1.0)
    I = voxel_intensity(field, hkl, GoniometerSetting(), res)
    per = _dominant_period_um(I.detach().cpu().numpy(), 0.125)
    assert abs(per - 2.0) < 0.3, f"off-Bragg period {per:.2f} um != 2.0"


def test_frequency_doubling_at_exact_bragg():
    """At exact Bragg the response is even -> the 2 um wave images at 1 um (lambda/2)."""
    field, _ = _shear_line()
    hkl = (1, 1, 0)
    res = _resolution(field, hkl, flank_sigmas=0.0)
    I = voxel_intensity(field, hkl, GoniometerSetting(), res)
    per = _dominant_period_um(I.detach().cpu().numpy(), 0.125)
    assert abs(per - 1.0) < 0.2, f"at-Bragg period {per:.2f} um != 1.0 (lambda/2)"


def test_strain_vs_tilt_channel_split():
    """[110] senses the shear longitudinally (strain); [100] transversely (tilt)."""
    field, _ = _shear_line()
    center = GoniometerSetting()

    def long_fraction(hkl):
        q_nom = reference_q_nom(field, hkl, center)
        qhat = q_nom / torch.linalg.vector_norm(q_nom)
        dev = field.local_Q(hkl) - q_nom
        longi = dev @ qhat                       # (N,)
        trans = dev - longi[:, None] * qhat
        lr = float(longi.std())
        tr = float(trans.norm(dim=1).std())
        return lr / (lr + tr + 1e-30)

    f_strain = long_fraction((1, 1, 0))          # expect ~1 (pure strain)
    f_tilt = long_fraction((1, 0, 0))            # expect ~0 (pure tilt)
    assert f_strain > 0.9, f"[110] not strain-sensing (long frac {f_strain:.2f})"
    assert f_tilt < 0.1, f"[100] not tilt-sensing (long frac {f_tilt:.2f})"


def test_resolution_transverse_anisotropy():
    """aligned_resolution is isotropic by default (back-compat) and anisotropic when
    sigma_perp2 is set; poulsen_aligned_resolution wires the distinct rock/roll widths."""
    from midas_dfxm import aligned_resolution, poulsen_aligned_resolution
    q = torch.tensor([0.0, 0.7, 0.0], dtype=DT)
    e_rock = torch.tensor([1.0, 0.0, 0.0], dtype=DT)   # in-plane transverse
    e_roll = torch.tensor([0.0, 0.0, 1.0], dtype=DT)   # out-of-plane transverse
    # back-compat: no sigma_perp2 -> both transverse weights equal
    r = aligned_resolution(q, sigma_par=5e-3, sigma_perp=2e-3)
    assert abs(r.weight(q + 1e-3 * e_rock).item() - r.weight(q + 1e-3 * e_roll).item()) < 1e-9
    # anisotropic: tight rock, wide roll -> roll weight larger (less penalised)
    ra = aligned_resolution(q, sigma_par=5e-3, sigma_perp=1e-3, sigma_perp2=9e-3, rock_dir=e_rock)
    assert ra.weight(q + 1e-3 * e_roll).item() > ra.weight(q + 1e-3 * e_rock).item()
    # one-call Poulsen: rock << roll (a thin plate), so the two transverse widths differ
    rp = poulsen_aligned_resolution(q, two_theta_deg=25.6, rock_dir=e_rock)
    assert rp.sigma_perp2 is not None and rp.sigma_perp < rp.sigma_perp2
