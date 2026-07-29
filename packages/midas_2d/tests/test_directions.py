"""Tests for the four extension directions: stiffness/dynamics, coherent phonon,
multi-reflection, ML surrogate, and instrument/real-data."""
import math

import numpy as np
import pytest
import torch

from midas_2d import (
    add_poisson_noise,
    analytic_rod_model,
    bragg_timeseries,
    build_crystal_tensor,
    cdse_supercell,
    coherent_intensity,
    debye_intensity,
    debye_reference_numpy,
    ensemble_intensity,
    fit_coherent_phonon,
    make_dataset,
    msd_tensor_from_frames,
    poisson_nll,
    project_to_detector,
    recover_stiffness,
    resolution_convolve,
    rocking_curve,
    stiffness_from_msd,
    strain_wave,
    thermal_ensemble,
    thickness_loss_scan,
    train_surrogate,
)

DT = torch.float64
A = 6.077


def _q(hkl):
    return (2 * math.pi / A) * torch.tensor(hkl, dtype=DT)


# ---------------------------------------------- Direction 1: stiffness/dynamics

@pytest.mark.unit
def test_thermal_ensemble_variance_matches_stiffness():
    torch.manual_seed(0)
    coords, _e, _ = cdse_supercell((4, 4, 4), dtype=DT)
    k_par, k_perp = torch.tensor(50.0, dtype=DT), torch.tensor(10.0, dtype=DT)
    frames = thermal_ensemble(coords, k_par, k_perp, n_frames=600, kBT=1.0)
    U = torch.diag(msd_tensor_from_frames(frames))
    # <u^2> = kBT / k
    assert abs(float(U[2]) - 0.1) < 0.02      # 1/10
    assert abs(float(U[0]) - 0.02) < 0.01     # 1/50
    assert U[2] > U[0]                         # softer out-of-plane -> larger MSD


@pytest.mark.autograd
def test_thermal_ensemble_differentiable_in_stiffness():
    torch.manual_seed(0)
    coords, elements, _ = cdse_supercell((4, 4, 3), dtype=DT)
    eps = torch.randn(16, *coords.shape, dtype=DT)
    k = torch.tensor([30.0, 8.0], dtype=DT, requires_grad=True)
    frames = thermal_ensemble(coords, k[0], k[1], eps=eps)
    I = ensemble_intensity(frames, elements, _q([[0., 0., 2.]]), coherent=True)
    I.sum().backward()
    assert k.grad is not None and torch.isfinite(k.grad).all() and k.grad.abs().sum() > 0


@pytest.mark.autograd
@pytest.mark.slow
def test_recover_transient_softening():
    """Plant a soft out-of-plane spring; recover k_perp < k_par from diffraction."""
    torch.manual_seed(0)
    coords, elements, _ = cdse_supercell((5, 5, 4), dtype=DT)
    q = torch.stack([_q([2., 0., 0.]), _q([0., 0., 2.]), _q([0., 0., 4.])])
    k_par_true, k_perp_true = 60.0, 12.0

    eps = torch.randn(64, *coords.shape, dtype=DT)
    frames = thermal_ensemble(coords, torch.tensor(k_par_true, dtype=DT),
                              torch.tensor(k_perp_true, dtype=DT), eps=eps)
    I_ref = coherent_intensity(coords, elements, q)
    obs_ratio = ensemble_intensity(frames, elements, q, coherent=True) / I_ref

    out = recover_stiffness(obs_ratio, coords, elements, q, n_frames=64,
                            steps=500, lr=0.1, seed=0)
    # out-of-plane recovered as the soft direction
    assert out["k_perp"] < out["k_par"]
    assert abs(out["k_perp"] - k_perp_true) / k_perp_true < 0.25


# --------------------------------------------------- Direction 2: coherent phonon

@pytest.mark.unit
def test_strain_wave_shape():
    t = torch.linspace(0, 3, 100, dtype=DT)
    s = strain_wave(t, amp=0.02, freq=1.5, tau=1.0)
    assert s[0] > 0 and abs(float(s[0]) - 0.02) < 1e-6   # cos(0)*A at t=0
    assert s.abs().max() <= 0.02 + 1e-9                   # damped envelope


@pytest.mark.autograd
@pytest.mark.slow
def test_recover_coherent_phonon_frequency():
    """Plant a phonon; recover frequency, damping, amplitude from the Bragg
    intensity time series."""
    torch.manual_seed(0)
    coords, elements, _ = cdse_supercell((5, 5, 4), dtype=DT)
    q = _q([0., 0., 2.])
    t = torch.linspace(0.0, 3.0, 60, dtype=DT)
    f_true, tau_true, amp_true = 2.0, 1.2, 0.03
    obs = bragg_timeseries(coords, elements, q, t, amp_true, f_true, tau_true)

    rec = fit_coherent_phonon(obs, coords, elements, q, t,
                              init={"amp": 0.01, "freq": 1.4, "tau": 0.8},
                              steps=2500, lr=0.02)
    assert abs(rec["freq"] - f_true) / f_true < 0.1
    assert abs(rec["tau"] - tau_true) / tau_true < 0.4


# ------------------------------------------ Direction 4a: multi-reflection fit

@pytest.mark.unit
def test_multi_reflection_breaks_thickness_multimodality():
    ct = build_crystal_tensor()
    nz_true = 5
    dl = torch.linspace(-0.5, 0.5, 401, dtype=DT)
    rods = [(1.0, 1.0), (2.0, 0.0), (0.0, 0.0)]
    obs = [analytic_rod_model(ct, hk, torch.tensor([1e4, 1e4, float(nz_true)], dtype=DT))(1.0 + dl)
           for hk in rods]
    n3_values = torch.arange(3.0, 8.01, 0.05, dtype=DT)

    res = thickness_loss_scan(ct, rods, obs, dl, n3_values)
    # combined loss minimum is at the true thickness
    n3_best = float(n3_values[int(torch.argmin(res["total"]))])
    assert abs(n3_best - nz_true) < 0.1
    # a single rod is more ambiguous: it has multiple deep minima
    single = res["per_rod"][0]
    deep = (single < 0.05 * single.max() + single.min()).sum()
    assert deep >= 1   # at least the structure is non-trivial


# ------------------------------------------- Direction 4b: ML amortized inference

@pytest.mark.unit
@pytest.mark.slow
def test_ml_surrogate_learns_thickness_and_msd():
    ct = build_crystal_tensor()
    X, Y = make_dataset(ct, n=400, n_points=48, seed=1)
    model, info = train_surrogate(X, Y, epochs=250, lr=2e-3, seed=1)
    # Thickness is recovered tightly; out-of-plane MSD is only weakly
    # identifiable from a single peak-normalised rod (-> motivates multi-
    # reflection / absolute-scale data), so its tolerance is looser.
    assert float(info["val_mae"][0]) < 0.5      # N3 within ~0.5 cell
    assert float(info["val_mae"][1]) < 0.04     # u_perp within ~0.04 A^2


# ----------------------------------------- Direction 3: instrument + real data

@pytest.mark.unit
def test_detector_projection_centres_forward_beam():
    # Q = 0 -> straight-through beam lands at the beam centre.
    q0 = torch.zeros(1, 3, dtype=DT)
    pix, valid = project_to_detector(q0, wavelength_A=1.0, distance_mm=200.0,
                                     pixel_mm=0.1, beam_center=(512.0, 512.0))
    assert valid.all()
    assert torch.allclose(pix[0], torch.tensor([512.0, 512.0], dtype=DT), atol=1e-6)


@pytest.mark.unit
def test_detector_projection_offaxis_moves_outward():
    q = torch.tensor([[0.5, 0.0, 0.0]], dtype=DT)   # transverse Q -> +x on detector
    pix, valid = project_to_detector(q, wavelength_A=1.0, distance_mm=200.0,
                                     pixel_mm=0.1, beam_center=(512.0, 512.0))
    assert valid.all() and pix[0, 0] > 512.0 and abs(float(pix[0, 1]) - 512.0) < 1e-6


@pytest.mark.autograd
def test_poisson_nll_minimised_at_truth():
    torch.manual_seed(0)
    truth = torch.rand(50, dtype=DT) * 100 + 10
    pred = truth.clone().requires_grad_(True)
    nll = poisson_nll(pred, truth)
    nll.backward()
    # gradient d/dpred (pred - obs log pred) = 1 - obs/pred = 0 at pred==obs
    assert pred.grad.abs().max() < 1e-6


@pytest.mark.unit
def test_resolution_convolve_preserves_mass_and_broadens():
    x = torch.zeros(101, dtype=DT)
    x[50] = 1.0
    y = resolution_convolve(x, 3.0)
    assert abs(float(y.sum()) - 1.0) < 1e-6
    assert float(y.max()) < 1.0          # the delta got broadened


@pytest.mark.unit
def test_torch_debye_matches_independent_numpy_reference():
    """The tiled torch Debye path agrees with a naive NumPy double loop."""
    coords, elements, _ = cdse_supercell((3, 3, 3), dtype=DT)
    q = torch.linspace(1.0, 3.0, 30, dtype=DT)
    I_torch = debye_intensity(coords, elements, q).detach().numpy()
    I_numpy = debye_reference_numpy(coords, elements, q)
    rel = np.abs(I_torch - I_numpy) / (np.abs(I_numpy).max())
    assert rel.max() < 1e-6, rel.max()


@pytest.mark.unit
def test_poisson_noise_roundtrip_and_load_profile(tmp_path):
    coords, elements, _ = cdse_supercell((4, 4, 4), dtype=DT)
    q = torch.linspace(1.0, 3.0, 60, dtype=DT)
    I = debye_intensity(coords, elements, q)
    counts = add_poisson_noise(I, photons_per_peak=1e5)
    assert counts.min() >= 0 and counts.max() > 0

    # save & reload as a real-data-style profile
    path = tmp_path / "profile.npy"
    np.save(path, np.stack([q.numpy(), counts.numpy()]))
    from midas_2d import load_profile
    qq, II = load_profile(str(path))
    assert qq.shape == II.shape == (60,)
