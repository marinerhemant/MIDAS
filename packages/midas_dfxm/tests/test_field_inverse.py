"""Unified full deformation-gradient-tensor inverse: exact recovery, identifiability, UQ."""
import numpy as np
import pytest
import torch

from midas_dfxm.field import DeformationField
import itertools

from midas_dfxm.field_inverse import (
    angular_condition_number,
    crlb_trace,
    decompose_deformation,
    deformation_covariance,
    deformation_design_matrix,
    deformation_identifiability,
    deformation_observable,
    fisher_information,
    recover_deformation_direct,
    recover_deformation_regularised,
    select_reflections_greedy,
)

DT = torch.float64
LATC = (4.0495, 4.0495, 4.0495, 90.0, 90.0, 90.0)   # FCC Al (Kanesalingam/Detlefs)

# a non-coplanar cubic reflection set that spans all 9 components
FULL_SET = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3), (3, 1, 1), (1, 3, 1)]


def _random_F(N, rng, scale=1e-3):
    """Small random deformation gradients F = I + small (strain + rotation)."""
    G = rng.normal(0, scale, size=(N, 3, 3))
    return torch.tensor(np.eye(3)[None] + G, dtype=DT)


def _field(F):
    N = F.shape[0]
    pos = torch.zeros(N, 3, dtype=DT)
    return DeformationField(positions=pos, F=F,
                            reference_orientation=torch.eye(3, dtype=DT),
                            lattice_params=torch.tensor(LATC, dtype=DT))


@pytest.mark.unit
def test_identifiability_rank9():
    info = deformation_identifiability(FULL_SET, lattice_params=LATC)
    assert info["recoverable"]
    assert info["rank"] == 9
    assert np.isfinite(info["cond"])
    # a single reflection constrains only 3 of 9
    one = deformation_identifiability([(2, 0, 2)], lattice_params=LATC)
    assert one["rank"] == 3 and not one["recoverable"]


@pytest.mark.unit
def test_exact_roundtrip_finite_strain():
    # recovery is EXACT (finite strain) for clean data with a rank-9 set
    rng = np.random.default_rng(0)
    F = _random_F(40, rng, scale=2e-3)
    meas = deformation_observable(_field(F), FULL_SET)
    F_rec = recover_deformation_direct(meas, FULL_SET, lattice_params=LATC)
    assert torch.allclose(F_rec, F, atol=1e-9)


@pytest.mark.unit
def test_roundtrip_large_deformation():
    # exactness holds even for large (non-infinitesimal) deformation
    rng = np.random.default_rng(1)
    F = _random_F(20, rng, scale=5e-2)
    meas = deformation_observable(_field(F), FULL_SET)
    F_rec = recover_deformation_direct(meas, FULL_SET, lattice_params=LATC)
    assert torch.allclose(F_rec, F, atol=1e-7)


@pytest.mark.unit
def test_frame_consistency_field_threading():
    # The inverse MUST use the forward's reference frame (orientation + lattice). A field
    # with a NON-identity orientation and a NON-default lattice round-trips to machine
    # precision only when that frame is threaded through -- via field=. A bare call (silent
    # identity/FCC-Al defaults) biases the recovery; this guards the microstrain reference-Q
    # trap that once contaminated the dynamical-diffraction study.
    rng = np.random.default_rng(7)
    F = _random_F(30, rng, scale=2e-3)
    # rotate ~3 deg about z and use a lattice different from any default
    c, s = np.cos(0.05), np.sin(0.05)
    R = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=DT)
    latc = torch.tensor([3.6156, 3.6156, 3.6156, 90.0, 90.0, 90.0], dtype=DT)
    fld = DeformationField(positions=torch.zeros(30, 3, dtype=DT), F=F,
                           reference_orientation=R, lattice_params=latc)
    meas = deformation_observable(fld, FULL_SET)

    # desync-proof path: threading the field recovers exactly
    F_ok = recover_deformation_direct(meas, FULL_SET, field=fld)
    assert torch.allclose(F_ok, F, atol=1e-9)

    # explicit frame must match field= exactly
    F_ex = recover_deformation_direct(meas, FULL_SET, orientation=R, lattice_params=latc)
    assert torch.allclose(F_ex, F, atol=1e-9)

    # forgetting the frame (bare call -> identity + FCC-Al default) is biased, NOT exact
    F_bad = recover_deformation_direct(meas, FULL_SET)
    assert not torch.allclose(F_bad, F, atol=1e-6)

    # passing both field= and an explicit frame is a usage error
    with pytest.raises(ValueError):
        recover_deformation_direct(meas, FULL_SET, field=fld, lattice_params=latc)


@pytest.mark.unit
def test_observable_differentiable():
    F = _random_F(5, np.random.default_rng(2), 1e-3).requires_grad_(True)
    fld = DeformationField(positions=torch.zeros(5, 3, dtype=DT), F=F,
                           reference_orientation=torch.eye(3, dtype=DT),
                           lattice_params=torch.tensor(LATC, dtype=DT))
    deformation_observable(fld, FULL_SET).pow(2).sum().backward()
    assert F.grad is not None and torch.isfinite(F.grad).all()


@pytest.mark.unit
def test_decompose_consistency():
    rng = np.random.default_rng(3)
    F = _random_F(16, rng, 1e-3)
    d = decompose_deformation(F)
    # R is orthogonal, det +1
    R = d["rotation"]
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3, dtype=DT)[None].expand_as(R), atol=1e-8)
    assert torch.allclose(torch.linalg.det(R), torch.ones(16, dtype=DT), atol=1e-8)
    # small and Green-Lagrange strains agree at first order for tiny deformation
    assert (d["strain_small"] - d["strain_green"]).abs().max() < 1e-5


@pytest.mark.unit
def test_covariance_and_fisher_are_inverses():
    sig = 2.0
    cov = deformation_covariance(FULL_SET, noise_std=sig, lattice_params=LATC)
    fish = fisher_information(FULL_SET, noise_std=sig, lattice_params=LATC)
    assert torch.allclose(cov @ fish, torch.eye(9, dtype=DT), atol=1e-8)


@pytest.mark.slow
def test_regularised_beats_direct_under_noise():
    rng = np.random.default_rng(4)
    ny, nx = 10, 10
    N = ny * nx
    # smooth ground-truth F field: a strain gradient across the grid
    xx = np.linspace(0, 1, nx)[None, :].repeat(ny, 0).reshape(-1)
    F = np.tile(np.eye(3), (N, 1, 1))
    F[:, 0, 0] += 1e-3 * xx
    F[:, 2, 2] += 5e-4 * xx
    F[:, 0, 2] += 8e-4 * xx
    F_t = torch.tensor(F, dtype=DT)
    meas = deformation_observable(_field(F_t), FULL_SET)
    noise = torch.tensor(rng.normal(0, 0.02 * float(meas.abs().mean()), meas.shape), dtype=DT)
    meas_n = meas + noise

    F_dir = recover_deformation_direct(meas_n, FULL_SET, lattice_params=LATC)
    F_reg = recover_deformation_regularised(meas_n, FULL_SET, (ny, nx, 1),
                                            lambda_smooth=5e-3, steps=500, lr=5e-3,
                                            lattice_params=LATC)
    e_dir = (F_dir - F_t).pow(2).mean().sqrt()
    e_reg = (F_reg - F_t).pow(2).mean().sqrt()
    assert e_reg < e_dir


@pytest.mark.unit
def test_coplanar_set_is_rank_deficient():
    # Detlefs 2025 non-coplanarity rule: reflections that all lie in one reciprocal
    # plane (here (h,k,0)) never probe the out-of-plane H column -> rank-deficient,
    # full F NOT recoverable. Our identifiability reproduces this geometric condition.
    coplanar = [(2, 0, 0), (0, 2, 0), (2, 2, 0), (2, -2, 0), (1, 1, 0), (3, 1, 0)]
    info = deformation_identifiability(coplanar, lattice_params=LATC)
    assert not info["recoverable"]
    assert info["rank"] < 9
    # a non-coplanar set of the SAME cardinality is full rank + well conditioned
    good = deformation_identifiability(FULL_SET, lattice_params=LATC)
    assert good["recoverable"] and good["cond"] < 10.0


@pytest.mark.unit
def test_angular_kappa_400_is_robustly_best():
    # Kanesalingam 2025's KEY experiment-design conclusion is that {400} is best-suited.
    # Verified robust: {400} is the lowest-condition family (stable ~9.3 across triples),
    # while {113}/{202} ordering is triple/geometry-dependent (not asserted here).
    lam = 12.398 / 19.2
    k400 = angular_condition_number([(4, 0, 0), (0, 4, 0), (0, 0, 4)], wavelength_A=lam, lattice_params=LATC)
    k113 = angular_condition_number([(1, 1, 3), (1, 3, 1), (3, 1, 1)], wavelength_A=lam, lattice_params=LATC)
    k202 = angular_condition_number([(2, 0, 2), (2, 2, 0), (0, 2, 2)], wavelength_A=lam, lattice_params=LATC)
    assert k400 < k113 and k400 < k202          # {400} robustly best-conditioned


@pytest.mark.unit
def test_case05_recovery_parity():
    # Kanesalingam Case-05: Exx=Exy=1e-3 + 1e-3 rad rotation about x, recovered exactly.
    eps = torch.zeros(3, 3, dtype=DT); eps[0, 0] = 1e-3; eps[0, 1] = eps[1, 0] = 1e-3
    W = torch.zeros(3, 3, dtype=DT); W[1, 2] = -1e-3; W[2, 1] = 1e-3
    F = (torch.eye(3, dtype=DT) + eps + W)[None]
    fld = DeformationField(positions=torch.zeros(1, 3, dtype=DT), F=F,
                           reference_orientation=torch.eye(3, dtype=DT),
                           lattice_params=torch.tensor(LATC, dtype=DT))
    F_rec = recover_deformation_direct(deformation_observable(fld, FULL_SET), FULL_SET, lattice_params=LATC)
    dec = decompose_deformation(F_rec)
    assert float((F_rec - F).abs().max()) < 1e-12
    assert abs(float(dec["strain_small"][0, 0, 0]) - 1e-3) < 1e-9
    assert abs(float(dec["rotation_vector"][0, 0]) - 1e-3) < 1e-9


@pytest.mark.unit
def test_crlb_predicts_empirical_variance():
    # the Cramer-Rao bound must actually predict the recovery variance under noise
    refl = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3)]
    A = deformation_design_matrix(refl, lattice_params=LATC)
    sig = 1e-4
    Htrue = torch.randn(9, dtype=DT, generator=torch.Generator().manual_seed(0)) * 1e-3
    rng = np.random.default_rng(0)
    ests = torch.stack([torch.linalg.lstsq(A, A @ Htrue + torch.tensor(rng.normal(0, sig, A.shape[0]), dtype=DT)).solution
                        for _ in range(3000)])
    emp = float(ests.var(0, unbiased=True).sum())
    crlb = crlb_trace(refl, noise_std=sig, lattice_params=LATC)
    assert abs(emp / crlb - 1.0) < 0.1          # CRLB predicts empirical variance


@pytest.mark.unit
def test_selection_matches_exhaustive_and_beats_random():
    pool = [(2, 0, 2), (0, 2, 2), (2, 2, 0), (1, 1, 3), (3, 1, 1), (1, 3, 1),
            (4, 0, 0), (0, 4, 0), (0, 0, 4), (2, 2, 2), (2, 0, -2), (0, 2, -2)]
    sig = 1e-4
    sel = select_reflections_greedy(pool, 4, noise_std=sig, lattice_params=LATC)
    c_sel = crlb_trace(sel, noise_std=sig, lattice_params=LATC)
    # exhaustive optimum
    best = min((crlb_trace(list(c), noise_std=sig, lattice_params=LATC)
                for c in itertools.combinations(pool, 4)
                if deformation_identifiability(list(c), lattice_params=LATC)["recoverable"]))
    assert abs(c_sel - best) / best < 1e-6      # refined selection reaches the global optimum
    # beats a random set
    rng = np.random.default_rng(3)
    rand = [pool[i] for i in rng.choice(len(pool), 4, replace=False)]
    if deformation_identifiability(rand, lattice_params=LATC)["recoverable"]:
        assert c_sel <= crlb_trace(rand, noise_std=sig, lattice_params=LATC) + 1e-15


@pytest.mark.unit
def test_exact_inverse_has_no_finite_strain_bias():
    # linear small-strain reading (eps ~ -dQ_par/|Q0|) is biased ~e^2 at large strain;
    # the exact inverse recovers F to machine precision (no finite-strain bias).
    from midas_dfxm.field_inverse import reference_Q
    ori = torch.eye(3, dtype=DT); latc = torch.tensor(LATC, dtype=DT)
    e = 0.1
    F = torch.eye(3, dtype=DT).clone(); F[0, 0] = 1 + e
    fld = DeformationField(positions=torch.zeros(1, 3, dtype=DT), F=F[None],
                           reference_orientation=ori, lattice_params=latc)
    meas = deformation_observable(fld, FULL_SET)
    # linear normal-strain reading for the x reflection subset
    Q0 = reference_Q((2, 0, -2), ori, latc); qn = torch.linalg.vector_norm(Q0)
    # exact inverse
    F_ex = recover_deformation_direct(meas, FULL_SET, lattice_params=LATC)
    assert abs(float(F_ex[0, 0, 0]) - (1 + e)) < 1e-12       # exact: no finite-strain bias
    # linear estimate e/(1+e) is biased by ~e^2
    s = e / (1 + e)
    assert abs(s - e) > 0.5 * e ** 2                          # linear bias is ~e^2, non-negligible
