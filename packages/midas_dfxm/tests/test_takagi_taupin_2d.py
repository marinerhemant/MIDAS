"""Full 2D Bragg Takagi-Taupin (``solve_tt_2d``): a semi-Lagrangian characteristic solve
that adds lateral wavefield transport (the Borrmann fan) to the 1D column Riccati.

Validated properties: the ``nx=1`` limit reduces exactly to ``solve_tt_bragg`` across the
whole rocking curve; with ``nx>1`` the scheme is STABLE off the Bragg plateau and under
lateral-grid refinement (a naive Cartesian upwind advection diverges here); it CONSERVES
ENERGY in the plane-wave limit; and it is differentiable via the linear-solve adjoint.
"""
import numpy as np
import pytest
import torch

from midas_dfxm import solve_tt_bragg, solve_tt_2d, bragg_2d_nz_for_unit_shift
from midas_dfxm.takagi_taupin import extinction_length

CHI = complex(-1e-6, 0.0)
WL, TB = 0.729, 10.0
LAM = extinction_length(CHI, CHI, wavelength_A=WL, theta_B_deg=TB)


def _R2(yv, nz, nx=1):
    D0, Dh = solve_tt_2d(CHI, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                         thickness_um=3.0 * LAM, y=float(yv), nx=nx, nz=nz,
                         geometry="bragg", backend="sparse")
    return float(abs(Dh[0, 0]) ** 2 / abs(D0[0, 0]) ** 2)


def _Rr(yv):
    X = solve_tt_bragg(CHI, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                       thickness_um=3.0 * LAM, y=float(yv), n_depth=1600, center=True)
    return float((X.abs() ** 2).item())


def test_2d_reduces_to_riccati_full_rocking_curve():
    """nx=1 reproduces the 1D Riccati Darwin curve across the WHOLE rocking curve (not just
    the plateau) -- plateau exact, tails converging under depth refinement."""
    maxd = max(abs(_Rr(yv) - _R2(yv, nz=4800)) for yv in np.linspace(-3, 3, 13))
    assert maxd < 0.02, f"rocking-curve mismatch {maxd:.2e}"


def test_2d_stable_off_plateau_under_refinement():
    """The characteristic scheme is STABLE off the Bragg plateau and as the lateral grid is
    refined -- the reflectivity stays bounded and converges (a Cartesian ``d/dx`` upwind
    advection instead diverges to >1e6 here). This is the property that makes the solver usable."""
    t = 2.0 * LAM
    vals = []
    for dx in (4.0, 2.0, 1.0):
        nz = bragg_2d_nz_for_unit_shift(t, TB, dx)
        nx = 121
        xs = (np.arange(nx) - nx // 2) * dx
        inc = torch.tensor(np.exp(-(xs / 60.0) ** 2), dtype=torch.complex128)
        _, Dh = solve_tt_2d(CHI, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=t,
                            y=-1.0, nx=nx, nz=nz, dx_um=dx, incident=inc, backend="sparse")
        vals.append(float((Dh[0].abs() ** 2).sum() / (inc.abs() ** 2).sum()))
    assert max(vals) < 1.5, f"unstable / energy created: {vals}"
    assert vals[-1] > vals[0], f"not converging up under refinement: {vals}"


def test_2d_conserves_energy_plane_wave():
    """In the plane-wave limit at the total-reflection plateau, all energy is reflected --
    the scheme conserves energy (reflectivity -> 1)."""
    t = 2.0 * LAM
    dx, nx = 2.0, 201
    nz = bragg_2d_nz_for_unit_shift(t, TB, dx)
    xs = (np.arange(nx) - nx // 2) * dx
    inc = torch.tensor(np.exp(-(xs / 120.0) ** 2), dtype=torch.complex128)   # wide ~ plane wave
    _, Dh = solve_tt_2d(CHI, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=t,
                        y=0.0, nx=nx, nz=nz, dx_um=dx, incident=inc, backend="sparse")
    refl = float((Dh[0].abs() ** 2).sum() / (inc.abs() ** 2).sum())
    assert refl > 0.99, f"energy not conserved: refl={refl:.4f}"


def test_2d_differentiable():
    """Gradient of a reflectivity functional w.r.t. a deformation amplitude, through the dense
    linear solve, matches finite difference (the adjoint is exact)."""
    nx, dx = 9, 3.0
    t = 1.5 * LAM
    nz = bragg_2d_nz_for_unit_shift(t, TB, dx)
    Xg = torch.arange(nx, dtype=torch.float64) - nx // 2
    Zg = (torch.arange(nz + 1, dtype=torch.float64) + 0.5) * (t / nz)

    def loss(amp):
        hu = amp * torch.atan2(Zg[:, None] - 0.5 * t, Xg[None, :] * dx + 1e-6)
        _, Dh = solve_tt_2d(CHI, CHI, CHI, wavelength_A=WL, theta_B_deg=TB, thickness_um=float(t),
                            y=0.3, nx=nx, nz=nz, dx_um=dx, hu=hu,
                            incident=torch.ones(nx, dtype=torch.complex128), backend="dense")
        return (Dh[0].abs() ** 2).sum()

    a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    g_auto = float(torch.autograd.grad(loss(a), a)[0])
    with torch.no_grad():
        h = 1e-4
        g_fd = float((loss(a + h) - loss(a - h)) / (2 * h))
    rel = abs(g_auto - g_fd) / (abs(g_fd) + 1e-30)
    assert rel < 1e-3, f"autograd {g_auto:.5f} vs fd {g_fd:.5f} rel {rel:.1e}"


def test_2d_sparse_matches_dense():
    """The sparse backend (scales) assembles the identical system as the dense/differentiable
    backend and returns the same field."""
    nx, dx = 9, 3.0
    t = 1.5 * LAM
    nz = bragg_2d_nz_for_unit_shift(t, TB, dx)
    inc = torch.ones(nx, dtype=torch.complex128)
    kw = dict(wavelength_A=WL, theta_B_deg=TB, thickness_um=t, nx=nx, nz=nz, dx_um=dx,
              incident=inc)
    _, Dh_d = solve_tt_2d(CHI, CHI, CHI, backend="dense", **kw)
    _, Dh_s = solve_tt_2d(CHI, CHI, CHI, backend="sparse", **kw)
    assert float((Dh_d - Dh_s).abs().max()) < 1e-9


def test_2d_laue_not_implemented():
    """Laue geometry is deliberately not implemented here (use solve_tt_laue)."""
    with pytest.raises(NotImplementedError):
        solve_tt_2d(CHI, CHI, CHI, wavelength_A=WL, theta_B_deg=TB,
                    thickness_um=LAM, nx=1, nz=100, geometry="laue")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
