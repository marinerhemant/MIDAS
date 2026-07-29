"""Depth-resolved strain profile: asymmetric peaks, recovery, unified thermal."""
import math

import pytest
import torch

from midas_2d import (
    apply_depth_displacement,
    cdse_supercell,
    coherent_intensity,
    depth_resolved_intensity,
    linear_strain,
    recover_depth_strain,
    strain_to_displacement,
    temperature_to_msd,
    thermal_rod,
)
from midas_2d.strain_profile import displacement_from_control, interp1d

DT = torch.float64
A = 6.077


def _rod(l, hk=(1.0, 1.0)):
    h = torch.full_like(l, hk[0]); k = torch.full_like(l, hk[1])
    return (2 * math.pi / A) * torch.stack([h, k, l], dim=-1)


@pytest.mark.unit
def test_interp1d_matches_numpy():
    xp = torch.linspace(0, 10, 6, dtype=DT)
    fp = xp ** 2
    x = torch.tensor([0.5, 3.3, 7.8, 9.9], dtype=DT)
    got = interp1d(x, xp, fp)
    import numpy as np
    want = np.interp(x.numpy(), xp.numpy(), fp.numpy())
    assert torch.allclose(got, torch.tensor(want, dtype=DT), atol=1e-9)


@pytest.mark.unit
def test_uniform_displacement_matches_plain_coherent():
    """Zero strain profile == the undistorted coherent intensity."""
    coords, elements, _ = cdse_supercell((5, 5, 4), dtype=DT)
    l = torch.linspace(0.7, 1.3, 80, dtype=DT)
    q = _rod(l)
    I0 = coherent_intensity(coords, elements, q)
    I1 = depth_resolved_intensity(coords, elements, q,
                                  u_z=torch.zeros(coords.shape[0], dtype=DT))
    assert torch.allclose(I0, I1, atol=1e-8)


@pytest.mark.unit
def test_depth_gradient_changes_peak_shape():
    """A strain gradient through the thickness changes the Bragg-peak *shape*
    beyond a rigid shift -- detectable as a peak-aligned shape difference that
    is exactly zero for no strain."""
    coords, elements, _ = cdse_supercell((6, 6, 6), dtype=DT)
    z = coords[:, 2]
    eps = linear_strain(z, eps_surface=0.04, eps_substrate=0.0)
    order = torch.argsort(z)
    u_sorted = strain_to_displacement(z[order], eps[order])
    u = torch.empty_like(u_sorted)
    u[order] = u_sorted

    l = torch.linspace(0.6, 1.4, 801, dtype=DT)
    q = _rod(l)
    I_cold = coherent_intensity(coords, elements, q)
    I_strain = depth_resolved_intensity(coords, elements, q, u_z=u)

    def aligned_shape_diff(Ia, Ib, W=120):
        pa, pb = int(torch.argmax(Ia)), int(torch.argmax(Ib))
        a = Ia[pa - W:pa + W] / Ia[pa]
        b = Ib[pb - W:pb + W] / Ib[pb]
        return float((a - b).abs().mean())

    # no strain -> identical shape; gradient -> clearly different shape
    assert aligned_shape_diff(I_cold, I_cold) < 1e-9
    assert aligned_shape_diff(I_strain, I_cold) > 5e-3


@pytest.mark.autograd
def test_recover_depth_strain_profile():
    """Plant a strain gradient, recover the depth displacement profile from the
    asymmetric Bragg peak."""
    torch.manual_seed(0)
    coords, elements, _ = cdse_supercell((6, 6, 6), dtype=DT)
    z = coords[:, 2]
    eps = linear_strain(z, eps_surface=0.015, eps_substrate=0.0)
    order = torch.argsort(z)
    u_true_sorted = strain_to_displacement(z[order], eps[order])
    u_true = torch.empty_like(u_true_sorted); u_true[order] = u_true_sorted

    l = torch.linspace(0.55, 1.45, 600, dtype=DT)
    q = _rod(l)
    obs = depth_resolved_intensity(coords, elements, q, u_z=u_true)

    z_ctrl = torch.linspace(float(z.min()), float(z.max()), 6, dtype=DT)
    out = recover_depth_strain(obs, coords, elements, q, z_ctrl, steps=1200, lr=0.01)

    # the recovered per-atom displacement should correlate strongly with truth
    # (up to a rigid z-shift, which does not change |A|^2 -> remove the mean)
    rec = out["u_atom"] - out["u_atom"].mean()
    tru = u_true - u_true.mean()
    corr = torch.corrcoef(torch.stack([rec, tru]))[0, 1]
    assert corr > 0.9, float(corr)


@pytest.mark.unit
def test_unified_thermal_shifts_and_damps():
    """One temperature field both shifts the peak (expansion) and lowers it
    (Debye-Waller)."""
    coords, elements, _ = cdse_supercell((6, 6, 5), dtype=DT)
    z = coords[:, 2]
    l = torch.linspace(0.6, 1.4, 401, dtype=DT)
    q = _rod(l)

    I_cold = coherent_intensity(coords, elements, q)
    dT = torch.full_like(z, 0.0)
    I_check = thermal_rod(coords, elements, q, z, dT, alpha=1e-3, k=20.0, T0=1.0,
                          kB=1.0)
    # with dT=0 there is still a baseline DWF (T0>0) so amplitude <= cold
    assert I_check.max() <= I_cold.max() + 1e-6

    dT_hot = torch.full_like(z, 50.0)
    I_hot = thermal_rod(coords, elements, q, z, dT_hot, alpha=1e-3, k=20.0, T0=1.0,
                        kB=1.0)
    # heating lowers the peak (more DWF) ...
    assert I_hot.max() < I_check.max()
    # ... and shifts it (expansion moves the (1 1 1) node)
    assert int(torch.argmax(I_hot)) != int(torch.argmax(I_check))


@pytest.mark.unit
def test_temperature_to_msd_monotonic():
    dT = torch.tensor([0.0, 10.0, 50.0], dtype=DT)
    u2 = temperature_to_msd(dT, k=20.0, kB=1.0, T0=1.0)
    assert torch.all(u2[1:] > u2[:-1])
