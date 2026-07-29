"""Phase 2 & 4: disorder, rocking curves, differentiable inversion, UQ, BCDI."""
import math

import pytest
import torch

from midas_2d import (
    AnisotropicMSD,
    TransientMSD,
    analytic_rod_model,
    bcdi_forward,
    build_crystal_tensor,
    cdse_supercell,
    dwf_amplitude,
    fit,
    fwhm,
    laplace_uncertainty,
    md_rod_model,
    msd_tensor_from_frames,
    phase_retrieval,
    relative_l2_loss,
    rocking_curve,
    thickness_from_fwhm,
)

DT = torch.float64
A = 6.077


# ----------------------------------------------------------------- disorder

@pytest.mark.unit
def test_dwf_suppresses_high_q_and_is_anisotropic():
    q_inplane = torch.tensor([[2.0, 0.0, 0.0]], dtype=DT)
    q_outplane = torch.tensor([[0.0, 0.0, 2.0]], dtype=DT)
    T_in = dwf_amplitude(q_inplane, u_par=0.02, u_perp=0.10)
    T_out = dwf_amplitude(q_outplane, u_par=0.02, u_perp=0.10)
    # larger u_perp -> stronger suppression of the out-of-plane reflection
    assert T_out < T_in < 1.0


@pytest.mark.unit
def test_msd_tensor_from_frames_recovers_planted_anisotropy():
    torch.manual_seed(0)
    coords, _elem, _ = cdse_supercell((4, 4, 4), dtype=DT)
    sig = torch.tensor([0.05, 0.05, 0.15], dtype=DT)   # bigger out-of-plane
    frames = coords[None] + sig * torch.randn(400, *coords.shape, dtype=DT)
    U = msd_tensor_from_frames(frames)
    diag = torch.diag(U)
    assert diag[2] > diag[0] and diag[2] > diag[1]      # u_zz largest
    assert torch.allclose(diag, sig ** 2, atol=2e-3)


# ------------------------------------------------------------- rocking curves

@pytest.mark.unit
@pytest.mark.parametrize("nz", [3, 5, 8])
def test_rocking_fwhm_tracks_inverse_thickness(nz):
    ct = build_crystal_tensor()
    N = torch.tensor([1e4, 1e4, float(nz)], dtype=DT)
    model = analytic_rod_model(ct, (1.0, 1.0), N)
    dl, I = rocking_curve(model, 1.0, half_width=0.5, n=2001)
    w = fwhm(dl, I)
    # FWHM ~ 0.886 / N  =>  recovered thickness within ~10%.
    n_est = thickness_from_fwhm(w)
    assert abs(n_est - nz) / nz < 0.12, (nz, n_est, w)


@pytest.mark.unit
def test_md_and_analytic_rocking_agree():
    nz = 4
    coords, elements, _ = cdse_supercell((6, 6, nz), dtype=DT)
    ct = build_crystal_tensor()
    N = torch.tensor([6.0, 6.0, float(nz)], dtype=DT)
    m_md = md_rod_model(coords, elements, a=A, hk=(1.0, 1.0))
    m_an = analytic_rod_model(ct, (1.0, 1.0), N)
    dl, I_md = rocking_curve(m_md, 1.0, n=401)
    _, I_an = rocking_curve(m_an, 1.0, n=401)
    a = I_md / I_md.max()
    b = I_an / I_an.max()
    assert torch.allclose(a, b, atol=3e-3)


# --------------------------------------------------- differentiable inversion

@pytest.mark.autograd
def test_recover_thickness_fwhm_coarse_then_gradient_refine():
    """Realistic two-step workflow: a coarse FWHM estimate picks the thickness
    basin, then gradient refinement through the differentiable forward polishes
    N3 to the planted value.

    (A single rocking curve is multimodal in thickness -- each integer N3 is its
    own basin -- so the coarse FWHM step is what makes the refinement robust.)
    """
    from midas_2d import cosine_loss
    ct = build_crystal_tensor()
    nz_true = 5.0
    N_true = torch.tensor([1e4, 1e4, nz_true], dtype=DT)
    dl, obs = rocking_curve(analytic_rod_model(ct, (1.0, 1.0), N_true), 1.0,
                            half_width=0.5, n=801)

    # Step 1: coarse thickness from the rocking-curve FWHM.
    n0 = thickness_from_fwhm(fwhm(dl, obs))
    assert abs(n0 - nz_true) < 1.0   # close enough to land in the right basin

    # Step 2: differentiable refinement with a smooth scale-invariant loss.
    n3 = torch.tensor(float(n0), dtype=DT, requires_grad=True)

    def loss_fn():
        N = torch.stack([torch.tensor(1e4, dtype=DT), torch.tensor(1e4, dtype=DT), n3])
        pred = analytic_rod_model(ct, (1.0, 1.0), N)(1.0 + dl)
        return cosine_loss(pred, obs)

    out = fit([n3], loss_fn, steps=800, lr=0.02)
    assert out["loss"] < 1e-4
    assert abs(float(n3) - nz_true) < 0.05


@pytest.mark.autograd
def test_recover_anisotropic_msd_and_uncertainty():
    """Recover (u_par, u_perp) from a DWF-modulated set of reflections, then get
    Laplace error bars."""
    torch.manual_seed(0)
    # A spread of reflections with both in-plane and out-of-plane character.
    q = torch.tensor([[2., 0., 0.], [0., 0., 2.], [2., 0., 2.],
                      [3., 0., 0.], [0., 0., 3.], [1., 1., 3.]], dtype=DT)
    u_par_true, u_perp_true = 0.03, 0.12
    obs = dwf_amplitude(q, u_par_true, u_perp_true) ** 2     # intensity multiplier

    msd = AnisotropicMSD(u_par=0.01, u_perp=0.01)

    def loss_fn():
        pred = msd.amplitude(q) ** 2
        return relative_l2_loss(pred, obs)

    fit(msd.parameters(), loss_fn, steps=800, lr=0.05)
    assert abs(float(msd.u_par) - u_par_true) < 0.01
    assert abs(float(msd.u_perp) - u_perp_true) < 0.01

    # Laplace UQ on (u_par, u_perp) directly.
    def loss_vec(theta):
        pred = dwf_amplitude(q, theta[0], theta[1]) ** 2
        return ((pred - obs) ** 2).sum()

    theta = torch.tensor([float(msd.u_par), float(msd.u_perp)], dtype=DT)
    res = laplace_uncertainty(loss_vec, theta, noise_var=1e-6)
    assert res["std"].shape == (2,)
    assert torch.isfinite(res["std"]).all()


@pytest.mark.autograd
def test_transient_msd_timeseries_recovers_rising_disorder():
    """Joint inversion of a delay series: planted u_perp(t) rises with delay."""
    torch.manual_seed(0)
    q = torch.tensor([[0., 0., 2.], [0., 0., 3.], [2., 0., 0.], [1., 1., 3.]], dtype=DT)
    n_delays = 4
    u_perp_plant = torch.tensor([0.02, 0.06, 0.10, 0.14], dtype=DT)
    u_par_plant = torch.full((n_delays,), 0.03, dtype=DT)
    obs = torch.stack([dwf_amplitude(q, u_par_plant[t], u_perp_plant[t]) ** 2
                       for t in range(n_delays)])

    tm = TransientMSD(n_delays, u_par0=0.01, u_perp0=0.01)

    def loss_fn():
        pred = torch.stack([tm.amplitude(q, t) ** 2 for t in range(n_delays)])
        return relative_l2_loss(pred, obs)

    fit(tm.parameters(), loss_fn, steps=1200, lr=0.05)
    rec = tm.u_perp.detach()
    # monotonic rise recovered
    assert torch.all(rec[1:] > rec[:-1])
    assert torch.allclose(rec, u_perp_plant, atol=0.02)


# --------------------------------------------------------------- coherent/BCDI

@pytest.mark.unit
def test_reciprocal_space_map_shape_and_peak():
    from midas_2d import reciprocal_space_map
    coords, elements, _ = cdse_supercell((6, 6, 4), dtype=DT)
    H, L, I = reciprocal_space_map(coords, elements, a=A, h0=1.0,
                                   qx_range=(-0.4, 0.4), qz_range=(0.6, 1.4),
                                   n_qx=40, n_qz=60)
    assert I.shape == (60, 40) and torch.isfinite(I).all()
    # the (1 1 1) node sits at (h=1, l=1) -> near grid centre, should be bright
    assert I.max() > 10 * I.mean()


@pytest.mark.unit
def test_bcdi_forward_differentiable():
    torch.manual_seed(0)
    obj = torch.randn(16, 16, dtype=DT) + 1j * torch.randn(16, 16, dtype=DT)
    obj.requires_grad_(True)
    I = bcdi_forward(obj)
    I.sum().backward()
    assert obj.grad is not None and torch.isfinite(obj.grad).all()


@pytest.mark.autograd
def test_phase_retrieval_reduces_loss_and_recovers_modulus():
    """Autograd phase retrieval: from a perturbed start, recover the object
    modulus inside a known support."""
    torch.manual_seed(0)
    n = 24
    support = torch.zeros(n, n, dtype=DT)
    support[6:18, 8:16] = 1.0
    truth = (support * (1.0 + 0.5 * torch.rand(n, n, dtype=DT))) \
        * torch.exp(1j * 0.3 * torch.randn(n, n, dtype=DT))
    measured = bcdi_forward(truth)

    # init near truth (avoids the global non-convexity) then refine
    init = truth + 0.12 * (torch.randn(n, n, dtype=DT) + 1j * torch.randn(n, n, dtype=DT))
    out = phase_retrieval(measured, support, init=init, steps=900, lr=0.01)
    assert out["history"][-1] < out["history"][0]

    rec_mod = out["psi"].abs()[support.bool()]
    true_mod = truth.abs()[support.bool()]
    corr = torch.corrcoef(torch.stack([rec_mod, true_mod]))[0, 1]
    assert corr > 0.9, float(corr)
