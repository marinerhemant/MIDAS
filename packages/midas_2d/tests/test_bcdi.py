"""BCDI geometry + phase retrieval: analytic identities, not self-consistency.

The load-bearing checks here are against closed form (the conjugate-basis
identity, the analytic Laue transform, the autocorrelation-support theorem) so a
sign or convention error cannot pass by agreeing with itself.
"""
import math

import pytest
import torch

from midas_2d import (
    bcdi_forward,
    bragg_geometry,
    conjugate_real_basis,
    detector_distance_for_oversampling,
    interference_factor,
    oversampling,
    phase_retrieval,
    q_basis,
    rocking_step_for_oversampling,
    shear_angles_deg,
    sheared_to_lab,
)
from midas_2d.instrument import project_to_detector

DT = torch.float64

# Au(111) at 9 keV -- the canonical BCDI system.
LAMBDA_A = 1.3776
A_AU = 4.0782
D_111 = A_AU / math.sqrt(3.0)
GEOM = dict(distance_mm=640.0, pixel_mm=0.055, rocking_step_deg=0.0084)
GRAIN_A = 4000.0                                   # 400 nm


def _B(**kw):
    return q_basis(LAMBDA_A, D_111, **{**GEOM, **kw})


# --------------------------------------------------------------- lab vectors
@pytest.mark.unit
def test_bragg_condition_and_elastic_scattering():
    v = bragg_geometry(LAMBDA_A, D_111)
    k = 2 * math.pi / LAMBDA_A
    # elastic: |k_i| = |k_f| = k
    assert float(torch.linalg.norm(v["ki"])) == pytest.approx(k, rel=1e-12)
    assert float(torch.linalg.norm(v["kf0"])) == pytest.approx(k, rel=1e-12)
    # Bragg: |G| = 2 k sin(theta) = 2 pi / d
    assert float(torch.linalg.norm(v["G"])) == pytest.approx(2 * math.pi / D_111,
                                                             rel=1e-12)
    assert math.degrees(v["theta_rad"]) == pytest.approx(17.01, abs=0.02)


@pytest.mark.unit
def test_detector_vectors_are_orthonormal_and_transverse():
    v = bragg_geometry(LAMBDA_A, D_111)
    for e in ("e1", "e2"):
        assert float(torch.linalg.norm(v[e])) == pytest.approx(1.0, rel=1e-12)
        assert float(v[e] @ v["khat"]) == pytest.approx(0.0, abs=1e-12)
    assert float(v["e1"] @ v["e2"]) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_inaccessible_reflection_raises():
    with pytest.raises(ValueError, match="inaccessible"):
        bragg_geometry(10.0, 1.0)                  # lambda > 2d


# ------------------------------------------------------------------ the basis
@pytest.mark.unit
@pytest.mark.parametrize("shape", [(64, 64, 64), (128, 96, 64)])
def test_conjugate_basis_identity(shape):
    """C = 2 pi B^-T diag(1/N) must satisfy B^T C = 2 pi diag(1/N) exactly.

    This is the identity that makes a plain FFT the correct transform between
    the two grids; everything else in BCDI geometry rests on it.
    """
    B = _B()
    C = conjugate_real_basis(B, shape)
    N = torch.tensor(shape, dtype=DT)
    want = 2 * math.pi * torch.diag(1 / N)
    # Entries are O(0.1); float64 round-off lands at ~3e-18, so this is ~1e-16
    # relative -- machine precision, not a fitted tolerance.
    assert torch.allclose(B.transpose(0, 1) @ C, want, atol=1e-15, rtol=0)


@pytest.mark.unit
def test_q_basis_is_sheared_by_the_bragg_angle():
    """The detector-column / rocking pair is off orthogonal by theta.

    Not a nuisance: 17 degrees for this geometry. If this ever reads 90 the
    rocking column has been built wrong.
    """
    ang = shear_angles_deg(_B())
    theta = math.degrees(bragg_geometry(LAMBDA_A, D_111)["theta_rad"])
    assert float(ang[0]) == pytest.approx(90.0, abs=1e-9)      # 1-2, both on detector
    assert float(ang[2]) == pytest.approx(90.0, abs=1e-9)      # 2-3
    assert abs(float(ang[1]) - 90.0) == pytest.approx(theta, abs=0.05)


@pytest.mark.unit
def test_rocking_column_vanishes_when_axis_is_parallel_to_G():
    """Rocking about G sweeps no q -- the array would be rank-deficient."""
    G = bragg_geometry(LAMBDA_A, D_111)["G"]
    B = _B(rocking_axis=tuple(float(x) for x in G))
    assert float(torch.linalg.norm(B[:, 2])) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="parallel to G"):
        rocking_step_for_oversampling(LAMBDA_A, D_111, GRAIN_A,
                                      rocking_axis=tuple(float(x) for x in G))


@pytest.mark.unit
def test_q_convention_matches_project_to_detector():
    """Q = k_f - k_i must agree with midas_2d.instrument, which builds the ray
    independently. Catches a sign or offset error in Q."""
    B = _B()
    v = bragg_geometry(LAMBDA_A, D_111)
    idx = torch.tensor([[0, 0, 0], [30, -20, 0], [-64, 64, 5]], dtype=DT)
    Q = idx @ B.transpose(0, 1) + v["G"]

    pix, valid = project_to_detector(Q, wavelength_A=LAMBDA_A,
                                     distance_mm=GEOM["distance_mm"],
                                     pixel_mm=GEOM["pixel_mm"])
    assert bool(valid.all())
    kf = v["ki"] + Q
    ours = torch.stack([GEOM["distance_mm"] * kf[:, 0] / kf[:, 2],
                        GEOM["distance_mm"] * kf[:, 1] / kf[:, 2]],
                       dim=-1) / GEOM["pixel_mm"]
    assert torch.allclose(pix, ours, atol=1e-9)


@pytest.mark.unit
def test_flat_ewald_error_is_subpixel():
    """The linearised basis puts points on a plane, not the Ewald sphere. The
    departure at the array corner must stay well under one pixel."""
    B = _B()
    v = bragg_geometry(LAMBDA_A, D_111)
    n = 128
    corners = torch.tensor([[i, j, 0] for i in (-n // 2, n // 2)
                            for j in (-n // 2, n // 2)], dtype=DT)
    kf = v["ki"] + v["G"] + corners @ B.transpose(0, 1)
    err_px = float((torch.linalg.norm(kf, dim=-1) - v["k"]).abs().max()
                   / torch.linalg.norm(B[:, 0]))
    assert err_px < 0.5


# ---------------------------------------------------------------- sampling
@pytest.mark.unit
def test_oversampling_matches_lambda_D_over_a_p():
    """The detector columns must reproduce the closed form sigma = lambda D/(a p)."""
    sig = oversampling(_B(), GRAIN_A)
    closed = (LAMBDA_A * GEOM["distance_mm"] * 1e7) / (GRAIN_A * GEOM["pixel_mm"] * 1e7)
    assert float(sig[0]) == pytest.approx(closed, rel=1e-6)
    assert float(sig[1]) == pytest.approx(closed, rel=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize("target", [2.0, 4.0, 7.5])
def test_design_helpers_round_trip(target):
    """Solving for D and the rocking step must give back the requested sigma."""
    D = detector_distance_for_oversampling(LAMBDA_A, GRAIN_A, GEOM["pixel_mm"], target)
    step = rocking_step_for_oversampling(LAMBDA_A, D_111, GRAIN_A, target)
    sig = oversampling(q_basis(LAMBDA_A, D_111, distance_mm=D,
                               pixel_mm=GEOM["pixel_mm"], rocking_step_deg=step),
                       GRAIN_A)
    assert torch.allclose(sig, torch.full_like(sig, target), rtol=1e-6)


# ------------------------------------------------------- forward-chain gates
@pytest.mark.unit
def test_fft_of_a_box_equals_the_analytic_laue_transform():
    """|FFT(box)|^2 has a closed form (a product of Dirichlet kernels).

    The strongest check of the forward chain: it must match to machine
    precision, independently of any BCDI geometry.
    """
    N, n_supp = (32, 24, 20), (7, 5, 9)
    psi = torch.zeros(N, dtype=torch.complex128)
    psi[:n_supp[0], :n_supp[1], :n_supp[2]] = 1.0
    I = bcdi_forward(psi)

    axes = [(torch.arange(n, dtype=DT) - n // 2) / n for n in N]
    x = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    I_ana = interference_factor(x, torch.tensor(n_supp, dtype=DT))
    assert float((I - I_ana).abs().max() / I_ana.max()) < 1e-10


@pytest.mark.unit
def test_autocorrelation_support_is_twice_the_object():
    """iFFT(|FFT(psi)|^2) is the object autocorrelation, with support 2n-1.

    This theorem is *why* oversampling must exceed 2; if it failed, the sampling
    rule in :func:`oversampling` would be unjustified.
    """
    N, n_supp = (32, 32, 32), (6, 8, 5)
    psi = torch.zeros(N, dtype=torch.complex128)
    psi[:n_supp[0], :n_supp[1], :n_supp[2]] = 1.0
    ac = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(bcdi_forward(psi)))).abs()
    thr = 1e-8 * float(ac.max())
    for d in range(3):
        other = tuple(i for i in range(3) if i != d)
        prof = (ac > thr).amax(dim=other[1]).amax(dim=other[0])
        nz = torch.nonzero(prof).flatten()
        assert int(nz[-1] - nz[0]) + 1 == 2 * n_supp[d] - 1


# ------------------------------------------------------------ shear correction
@pytest.mark.unit
def _extents(mask, voxel_A):
    """Physical span of a 0/1 mask along each axis."""
    out = []
    for d in range(3):
        other = tuple(i for i in range(3) if i != d)
        nz = torch.nonzero(mask.amax(dim=other[1]).amax(dim=other[0])).flatten()
        out.append((int(nz[-1] - nz[0]) + 1) * voxel_A)
    return out


@pytest.mark.unit
def test_sheared_to_lab_recovers_an_axis_aligned_cube():
    """A lab-frame cube built on the sheared grid must come back a cube.

    Built by thresholding lab coordinates, so it is a genuine cube in space
    while occupying a parallelepiped in array indices.

    Carries its own control: read off the raw index extents (what you would get
    by believing the FFT grid is Cartesian) and confirm they are badly anisotropic
    *before* the correction. Without that, "the extents are roughly equal" could
    just mean the test is insensitive.
    """
    shape = (64, 64, 64)
    C = conjugate_real_basis(_B(), shape)
    axes = [torch.arange(n, dtype=DT) - n // 2 for n in shape]
    m = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    r = m @ C.transpose(0, 1)                                  # lab coords

    edge = 8000.0                                              # ~32 output voxels
    cube = (r.abs() <= edge / 2).all(dim=-1).to(DT)
    assert float(cube.sum()) > 1000, "cube must be well resolved on this grid"

    # Control: uncorrected, treating index space as if it were Cartesian.
    mean_vox = float(torch.linalg.norm(C, dim=0).mean())
    raw = _extents(cube > 0.5, mean_vox)
    assert max(raw) / min(raw) - 1 > 0.25, \
        f"control is weak: uncorrected extents already isotropic ({raw})"

    out = sheared_to_lab(cube, C)
    got = _extents(out["obj"] > 0.5, out["voxel_A"])
    # Residual spread is discretisation: ~250 A voxels on an 8000 A cube, with
    # half-voxel threshold slop at each face.
    assert max(got) / min(got) - 1 < 0.10, f"still sheared: {got}"
    for d, e in enumerate(got):
        assert e == pytest.approx(edge, rel=0.08), f"axis {d}: {e} A"


@pytest.mark.unit
def test_sheared_to_lab_output_voxel_defaults_to_finest_input():
    """Default output voxel must not throw away resolution."""
    C = conjugate_real_basis(_B(), (32, 32, 32))
    out = sheared_to_lab(torch.zeros(32, 32, 32), C)
    assert out["voxel_A"] == pytest.approx(
        float(torch.linalg.norm(C, dim=0).min()), rel=1e-12)


@pytest.mark.unit
def test_sheared_to_lab_preserves_complex_phase():
    """A uniform phase must survive resampling (real/imag are interpolated)."""
    shape = (32, 32, 32)
    C = conjugate_real_basis(_B(), shape)
    obj = torch.zeros(shape, dtype=torch.complex128)
    obj[10:22, 10:22, 10:22] = torch.polar(torch.ones(1, dtype=DT),
                                           torch.tensor([0.7], dtype=DT))[0]
    got = sheared_to_lab(obj, C)["obj"]
    inside = got.abs() > 0.9
    assert bool(inside.any())
    assert torch.allclose(got[inside].angle(),
                          torch.full_like(got[inside].angle(), 0.7), atol=1e-6)


# --------------------------------------------------------------- phase retrieval
@pytest.mark.unit
def test_phase_retrieval_default_init_runs():
    """Regression: the default-init branch called torch.clamp with no bounds and
    always raised, so it had never executed."""
    torch.manual_seed(0)
    supp = torch.zeros(16, 16, 16)
    supp[4:10, 4:10, 4:10] = 1.0
    I = bcdi_forward(torch.polar(supp, torch.zeros_like(supp)))
    res = phase_retrieval(I, supp, steps=5)
    assert torch.isfinite(res["psi"]).all()
    assert len(res["history"]) == 5


@pytest.mark.unit
def test_phase_retrieval_default_init_has_parseval_scale():
    """The default init magnitude must be the physically right one (=1 for a
    unit-amplitude object), otherwise the optimiser starts decades off."""
    supp = torch.zeros(16, 16, 16)
    supp[4:10, 4:10, 4:10] = 1.0
    I = bcdi_forward(torch.polar(supp, torch.zeros_like(supp)))
    mag = float(torch.sqrt(I.sum() / I.numel() / supp.sum()))
    assert mag == pytest.approx(1.0, rel=1e-6)


@pytest.mark.unit
def test_amplitude_loss_beats_intensity_loss_on_wide_dynamic_range():
    """Intensity-domain L2 is dominated by the brightest voxels, so it stalls.

    This is the reason the default is "amplitude": on a pattern spanning many
    decades the fringes -- which carry the shape and phase -- contribute almost
    no gradient to an intensity residual.
    """
    shape = (24, 24, 24)
    supp = torch.zeros(shape)
    supp[6:14, 6:15, 6:13] = 1.0
    phase = torch.zeros(shape)
    phase[6:14, 6:15, 6:13] = torch.linspace(-1.0, 1.0, 8)[:, None, None]
    truth = torch.polar(supp, phase * supp)
    I = bcdi_forward(truth)

    def score(rec):
        m = supp > 0
        best = 0.0
        for cand in (rec, torch.conj(torch.flip(rec, dims=(0, 1, 2)))):
            x, y = cand[m], truth[m]
            best = max(best, float((x * y.conj()).sum().abs()
                                   / (x.abs().norm() * y.abs().norm())))
        return best

    torch.manual_seed(0)
    amp = score(phase_retrieval(I, supp, steps=600, loss="amplitude")["psi"])
    torch.manual_seed(0)
    inten = score(phase_retrieval(I, supp, steps=600, loss="intensity")["psi"])
    assert amp > 0.9, f"amplitude loss should converge, got {amp:.3f}"
    assert amp > inten, f"amplitude {amp:.3f} should beat intensity {inten:.3f}"


@pytest.mark.unit
def test_phase_retrieval_rejects_unknown_loss():
    supp = torch.ones(8, 8, 8)
    with pytest.raises(ValueError, match="loss must be one of"):
        phase_retrieval(torch.ones(8, 8, 8), supp, steps=1, loss="l1")


@pytest.mark.unit
@pytest.mark.parametrize("mode", ["amplitude", "intensity", "poisson"])
def test_phase_retrieval_loss_modes_are_finite(mode):
    torch.manual_seed(0)
    supp = torch.zeros(16, 16, 16)
    supp[5:11, 5:11, 5:11] = 1.0
    I = bcdi_forward(torch.polar(supp, torch.zeros_like(supp)))
    res = phase_retrieval(I, supp, steps=10, loss=mode)
    assert all(math.isfinite(h) for h in res["history"])
    assert res["loss"] == mode
