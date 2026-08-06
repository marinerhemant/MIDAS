"""BCDI forward chain: the MD-atom path, the detector chain, and differentiability.

The headline check is :func:`test_atom_sum_matches_envelope_model` -- the direct
atomic sum makes no envelope or small-strain approximation, so it is the
reference that ``psi = s exp(-i G.u)`` is an approximation *to*. Agreement
between them tests the linearisation and the sign convention at once.
"""
import math

import pytest
import torch

import midas_2d as m2d

DT = torch.float64
A_AU, LAMBDA_A = 4.0782, 1.3776
D_111 = A_AU / math.sqrt(3.0)


# ------------------------------------------------------------------ helpers
def _geom(size_A, shape, target=2.5):
    D = m2d.detector_distance_for_oversampling(LAMBDA_A, size_A, 0.055, target)
    step = m2d.rocking_step_for_oversampling(LAMBDA_A, D_111, size_A, target)
    B = m2d.q_basis(LAMBDA_A, D_111, distance_mm=D, pixel_mm=0.055,
                    rocking_step_deg=step, dtype=DT)
    return B, m2d.conjugate_real_basis(B, shape)


def _real_grid(C, shape):
    axes = [torch.arange(n, dtype=DT) - n // 2 for n in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1) @ C.transpose(0, 1)


def _fcc_sphere(radius_A, R):
    """fcc Au atoms inside a lab-frame sphere, oriented for Bragg."""
    n = int(2 * radius_A / A_AU) + 3
    ijk = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=DT)] * 3,
                                     indexing="ij"), -1).reshape(-1, 3)
    basis = torch.tensor([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], dtype=DT)
    c = ((ijk[:, None, :] + basis[None, :, :]).reshape(-1, 3) * A_AU)
    c = (c - c.mean(0)) @ R.T
    return c[c.pow(2).sum(-1).sqrt() <= radius_A]


def _I(psi):
    A = m2d.object_to_amplitude(psi)
    return A.real * A.real + A.imag * A.imag


def _ncc(x, y):
    x, y = x.flatten(), y.flatten()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()))


def _invert(I):
    """q -> -q on an fftshift-ed array of even length: flip THEN roll(+1).

    Plain flip is off by one voxel and reads as a real discrepancy.
    """
    return torch.roll(torch.flip(I, dims=(0, 1, 2)), shifts=(1, 1, 1), dims=(0, 1, 2))


# ------------------------------------------------------------------ geometry
@pytest.mark.unit
def test_rotation_to_bragg_aligns_reflection_with_G():
    g_c = (2 * math.pi / A_AU) * torch.tensor([1.0, 1.0, 1.0], dtype=DT)
    R = m2d.rotation_to_bragg(g_c, LAMBDA_A, D_111)
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    cos = float(torch.dot(R @ g_c / torch.linalg.norm(g_c), G / torch.linalg.norm(G)))
    assert cos == pytest.approx(1.0, abs=1e-12)
    # a rotation: orthogonal, det +1 (not a reflection, which would flip chirality)
    assert torch.allclose(R @ R.T, torch.eye(3, dtype=DT), atol=1e-12)
    assert float(torch.linalg.det(R)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_rotation_to_bragg_handles_already_aligned_and_antiparallel():
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    for g in (G, -G):
        R = m2d.rotation_to_bragg(g, LAMBDA_A, D_111)
        assert torch.allclose(R @ R.T, torch.eye(3, dtype=DT), atol=1e-12)
        cos = float(torch.dot(R @ g / torch.linalg.norm(g), G / torch.linalg.norm(G)))
        assert cos == pytest.approx(1.0, abs=1e-9)


@pytest.mark.unit
def test_q_grid_centre_is_the_bragg_peak():
    B, _ = _geom(60.0, (8, 8, 8))
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    Q = m2d.q_grid(B, (8, 8, 8), offset=G)
    assert torch.allclose(Q[4, 4, 4], G, atol=1e-12)
    q = m2d.q_grid(B, (8, 8, 8))
    assert torch.allclose(q[4, 4, 4], torch.zeros(3, dtype=DT), atol=1e-12)


# -------------------------------------------------- the MD path vs the model
@pytest.mark.unit
def test_atom_sum_matches_envelope_model():
    """Direct atomic sum == psi = s exp(-i G.u), at the same q.

    This is the real validation of the whole scheme: the atom path makes no
    envelope approximation and uses the OPPOSITE Fourier sign convention
    (exp(+iQ.r) vs fftn's exp(-iq.r)). They agree only if the conventions are
    correctly paired -- a sign error shows up as an inverted pattern.
    """
    radius, shape = 30.0, (48, 48, 48)
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    R = m2d.rotation_to_bragg((2 * math.pi / A_AU) * torch.tensor([1., 1., 1.], dtype=DT),
                              LAMBDA_A, D_111)
    B, C = _geom(2 * radius, shape)
    r = _real_grid(C, shape)
    s = (r.pow(2).sum(-1).sqrt() <= radius).to(DT)

    def u_of(p):                       # odd in x -> breaks centrosymmetry
        ux = 5e-2 * (p[..., 0] ** 3) / (radius * radius)
        return torch.stack([ux, torch.zeros_like(ux), torch.zeros_like(ux)], -1)

    coords = _fcc_sphere(radius, R)
    coords = coords + u_of(coords)
    I_atom = m2d.speckle_from_atoms(coords, ["Au"] * coords.shape[0],
                                    m2d.q_grid(B, shape, offset=G))
    phi = -(u_of(r) * G).sum(-1) * s
    I_env = _I(torch.polar(s, phi))

    assert _ncc(I_atom, I_env) > 0.99, "envelope model disagrees with the atom sum"
    # Controls: both wrong alternatives must be clearly worse.
    assert _ncc(I_atom, _I(torch.polar(s, -phi))) < 0.9, "sign of G.u is unconstrained"
    assert _ncc(I_atom, _invert(I_env)) < 0.9, "pattern inversion is unconstrained"


@pytest.mark.unit
def test_twin_is_exactly_degenerate_only_for_a_centrosymmetric_object():
    """The strain-sign ambiguity is exact iff |psi| and the phase are both even.

    Not a defect -- a hard identifiability limit. Real grains are faceted and
    inhomogeneously strained, which is what puts the sign back in the data.
    """
    radius, shape = 30.0, (32, 32, 32)
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    _, C = _geom(2 * radius, shape)
    r = _real_grid(C, shape)
    sphere = (r.pow(2).sum(-1).sqrt() <= radius).to(DT)
    faceted = sphere * (r[..., 0] > -0.45 * radius).to(DT)

    def phase(s, power):
        return -(5e-2 * (r[..., 0] ** power) / radius ** (power - 1)) * G[0] * s

    # centrosymmetric: even phase on an even support -> exactly degenerate
    p = phase(sphere, 2)
    assert _ncc(_I(torch.polar(sphere, p)), _I(torch.polar(sphere, -p))) > 0.9999

    # break either symmetry and the sign becomes recoverable
    p_odd = phase(sphere, 3)
    assert _ncc(_I(torch.polar(sphere, p_odd)), _I(torch.polar(sphere, -p_odd))) < 0.9
    p_fac = phase(faceted, 2)
    assert _ncc(_I(torch.polar(faceted, p_fac)), _I(torch.polar(faceted, -p_fac))) < 0.9


@pytest.mark.unit
def test_conjugating_the_object_inverts_the_pattern_exactly():
    """|FFT(conj psi)|^2(q) == |FFT(psi)|^2(-q), with the correct inversion.

    Also pins the flip-vs-flip+roll trap: plain flip is off by one voxel.
    """
    shape = (16, 16, 16)
    torch.manual_seed(0)
    s = torch.zeros(shape, dtype=DT)
    s[4:11, 5:12, 3:10] = 1.0
    psi = torch.polar(s, torch.rand(shape, dtype=DT) * s)
    I, I_conj = _I(psi), _I(psi.conj())
    assert _ncc(I_conj, _invert(I)) == pytest.approx(1.0, abs=1e-9)
    assert _ncc(I_conj, torch.flip(I, dims=(0, 1, 2))) < 0.999      # the off-by-one


@pytest.mark.unit
def test_speckle_from_atoms_chunking_does_not_change_the_result():
    torch.manual_seed(0)
    coords = torch.randn(40, 3, dtype=DT) * 5.0
    els = ["Au"] * 40
    Q = m2d.q_grid(_geom(50.0, (6, 6, 6))[0], (6, 6, 6),
                   offset=m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"])
    big = m2d.speckle_from_atoms(coords, els, Q, max_elements=1 << 24)
    small = m2d.speckle_from_atoms(coords, els, Q, max_elements=80)   # forces chunking
    assert torch.allclose(big, small, rtol=1e-12, atol=0)


@pytest.mark.unit
@pytest.mark.parametrize("elements", [
    ["Au"] * 30,                                  # single species
    ["Au"] * 15 + ["Ag"] * 15,                    # two species
    ["Au", "Ag", "Cu"] * 10,                      # interleaved species
])
def test_grouped_kernel_matches_coherent_amplitude(elements):
    """The element-grouped fast path must equal midas_2d.debye exactly.

    speckle_from_atoms groups atoms by species so the form factor is evaluated
    once per species rather than once per atom. coherent_amplitude is the
    reference; grouping is only an optimisation and must not change a digit.
    """
    from midas_2d.debye import coherent_amplitude

    torch.manual_seed(0)
    coords = torch.randn(len(elements), 3, dtype=DT) * 6.0
    B, _ = _geom(50.0, (5, 5, 5))
    Q = m2d.q_grid(B, (5, 5, 5),
                   offset=m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"])

    ref = coherent_amplitude(coords, elements, Q)
    got = m2d.speckle_from_atoms(coords, elements, Q, amplitude=True)
    assert torch.allclose(got, ref, rtol=1e-12, atol=1e-12)
    assert torch.allclose(m2d.speckle_from_atoms(coords, elements, Q),
                          ref.real ** 2 + ref.imag ** 2, rtol=1e-12)


@pytest.mark.unit
def test_speckle_from_atoms_rejects_mismatched_elements():
    Q = torch.zeros(2, 3, dtype=DT)
    with pytest.raises(ValueError, match="elements has"):
        m2d.speckle_from_atoms(torch.zeros(3, 3, dtype=DT), ["Au"], Q)


# ------------------------------------------- the scalable MD route (binning)
def _strained_sphere(radius, R):
    """Reference and deformed fcc Au positions inside a lab-frame sphere."""
    ref = _fcc_sphere(radius, R)
    ux = 5e-2 * (ref[:, 0] ** 3) / (radius * radius)
    return ref, ref + torch.stack([ux, torch.zeros_like(ux), torch.zeros_like(ux)], -1)


@pytest.mark.unit
def test_binned_object_reproduces_the_direct_atom_sum():
    """atoms_to_object -> FFT must match speckle_from_atoms.

    This is what makes MD usable at real grain sizes: the direct sum is
    O(N_atoms * N_q) and caps out around 10 nm, the binned route is O(N_atoms).
    Two controls keep the check honest -- the inverted pattern and a shape-only
    object (no G.u phase) must both be clearly worse, otherwise the agreement
    would not be evidence that the phase is right.
    """
    radius, shape = 30.0, (48, 48, 48)
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    R = m2d.rotation_to_bragg((2 * math.pi / A_AU) * torch.tensor([1., 1., 1.], dtype=DT),
                              LAMBDA_A, D_111)
    B, C = _geom(2 * radius, shape)
    ref, coords = _strained_sphere(radius, R)

    I_direct = m2d.speckle_from_atoms(coords, ["Au"] * coords.shape[0],
                                      m2d.q_grid(B, shape, offset=G))
    ob = m2d.atoms_to_object(coords, C, shape, reference=ref, G=G)
    I_binned = _I(ob["psi"])

    assert ob["n_outside"] == 0
    assert _ncc(I_direct, I_binned) > 0.99
    assert _ncc(I_direct, _invert(I_binned)) < 0.9              # control
    shape_only = _I(m2d.atoms_to_object(coords, C, shape)["psi"])
    assert _ncc(I_direct, shape_only) < 0.95                    # control: phase matters


@pytest.mark.unit
def test_atoms_to_object_recovers_a_known_displacement():
    """The binned mean displacement must equal the field that was applied."""
    radius, shape = 30.0, (32, 32, 32)
    R = m2d.rotation_to_bragg((2 * math.pi / A_AU) * torch.tensor([1., 1., 1.], dtype=DT),
                              LAMBDA_A, D_111)
    _, C = _geom(2 * radius, shape)
    ref = _fcc_sphere(radius, R)
    shift = torch.tensor([0.13, -0.07, 0.02], dtype=DT)
    ob = m2d.atoms_to_object(ref + shift, C, shape, reference=ref,
                             G=m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"])
    filled = ob["occupancy"] > 0
    assert torch.allclose(ob["u"][filled], shift.expand_as(ob["u"][filled]), atol=1e-9)


@pytest.mark.unit
def test_atoms_to_object_counts_atoms_that_fall_off_the_grid():
    """Silent truncation would look like a smaller grain; it must be reported."""
    shape = (16, 16, 16)
    _, C = _geom(60.0, shape)
    far = torch.tensor([[1e5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=DT)
    ob = m2d.atoms_to_object(far, C, shape)
    assert ob["n_outside"] == 1


@pytest.mark.unit
def test_atoms_to_object_requires_G_with_a_reference():
    shape = (8, 8, 8)
    _, C = _geom(60.0, shape)
    c = torch.zeros(4, 3, dtype=DT)
    with pytest.raises(ValueError, match="without G"):
        m2d.atoms_to_object(c, C, shape, reference=c)
    with pytest.raises(ValueError, match="reference"):
        m2d.atoms_to_object(c, C, shape, reference=c[:2],
                            G=torch.zeros(3, dtype=DT))


@pytest.mark.unit
def test_atom_sum_cost_flags_infeasible_problems():
    assert m2d.atom_sum_cost(32 ** 3, 5_000)["advice"].startswith("direct")
    assert "atoms_to_object" in m2d.atom_sum_cost(128 ** 3, 10 ** 9)["advice"]
    assert m2d.atom_sum_cost(10, 10)["terms"] == 100


@pytest.mark.autograd
def test_atoms_to_object_gradient_flows_through_the_displacement():
    """Binning is a discrete round, but the displacement inside each voxel is
    differentiable -- which is the part a refinement moves."""
    radius, shape = 25.0, (24, 24, 24)
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"]
    R = m2d.rotation_to_bragg((2 * math.pi / A_AU) * torch.tensor([1., 1., 1.], dtype=DT),
                              LAMBDA_A, D_111)
    _, C = _geom(2 * radius, shape)
    ref = _fcc_sphere(radius, R)
    coords = (ref + 0.05).clone().requires_grad_(True)
    ob = m2d.atoms_to_object(coords, C, shape, reference=ref, G=G)
    _I(ob["psi"]).sum().backward()
    assert coords.grad is not None and torch.isfinite(coords.grad).all()
    assert float(coords.grad.abs().max()) > 0


# ------------------------------------------------------------ detector chain
@pytest.mark.unit
def test_detector_signal_applies_factors_in_a_recoverable_way():
    I = torch.rand(8, 8, 8, dtype=DT) + 0.1
    out = m2d.detector_signal(I, structure_factor_sq=3.0)
    assert torch.allclose(out, I * 3.0, rtol=1e-12)

    out = m2d.detector_signal(I, photons_per_peak=1000.0)
    assert float(out.max()) == pytest.approx(1000.0, rel=1e-9)

    out = m2d.detector_signal(I, background=5.0)
    assert torch.allclose(out, I + 5.0, rtol=1e-12)


@pytest.mark.unit
def test_detector_signal_rejects_amplitude():
    with pytest.raises(TypeError, match="expects intensity"):
        m2d.detector_signal(torch.ones(4, 4, 4, dtype=torch.complex128))


@pytest.mark.unit
def test_partial_coherence_broadens_and_conserves():
    """Finite coherence blurs the speckle; total signal is preserved."""
    shape = (32, 32, 32)
    _, C = _geom(60.0, shape)
    s = torch.zeros(shape, dtype=DT)
    s[13:19, 13:19, 13:19] = 1.0
    I = _I(torch.polar(s, torch.zeros_like(s)))

    blurred = m2d.detector_signal(I, coherence_length_A=40.0, real_basis=C)
    assert float(blurred.sum()) == pytest.approx(float(I.sum()), rel=1e-6)
    # peak comes down, and the pattern gets smoother (less contrast)
    assert float(blurred.max()) < float(I.max())
    assert float(blurred.std() / blurred.mean()) < float(I.std() / I.mean())


@pytest.mark.unit
def test_partial_coherence_requires_a_basis():
    with pytest.raises(ValueError, match="requires real_basis"):
        m2d.detector_signal(torch.ones(4, 4, 4, dtype=DT), coherence_length_A=10.0)


@pytest.mark.unit
def test_sample_counts_is_poisson_about_the_rate():
    rate = torch.full((64, 64, 8), 25.0, dtype=DT)
    g = torch.Generator().manual_seed(0)
    c = m2d.sample_counts(rate, generator=g)
    assert float(c.mean()) == pytest.approx(25.0, rel=0.02)
    assert float(c.var()) == pytest.approx(25.0, rel=0.10)     # Poisson: var = mean
    assert torch.all(c >= 0)


# ------------------------------------------------------------ differentiable
@pytest.mark.autograd
def test_gradient_flows_to_the_object():
    """A residual loss on the detector rate must reach the object phase.

    Two things here are deliberate, and the test is worthless without both.

    *The loss is a residual against a target*, not ``rate.sum()``. By Parseval
    ``sum_q I(q)`` depends only on ``|psi|``, so the summed rate is exactly
    invariant to the phase (partial coherence preserves this too, since the
    coherence factor is 1 at zero separation). A summed-rate loss therefore has
    an analytically zero phase gradient and would "pass" on float noise alone.

    *The starting phase is non-zero and asymmetric.* At ``phase = 0`` the object
    is real, the peak sits at q = 0, and the first-order variation there is
    purely imaginary against a real amplitude -- so even the ``1/I.max()``
    normalisation contributes nothing and the gradient is zero by symmetry.
    """
    torch.manual_seed(0)
    shape = (16, 16, 16)
    _, C = _geom(60.0, shape)
    s = torch.zeros(shape, dtype=DT)
    s[5:11, 5:11, 5:11] = 1.0

    def rate_of(ph):
        return m2d.detector_signal(_I(torch.polar(s, ph * s)),
                                   structure_factor_sq=2.0,
                                   coherence_length_A=80.0, real_basis=C,
                                   photons_per_peak=1e4)

    target = rate_of(torch.rand(shape, dtype=DT)).detach()
    phase = (0.3 * torch.randn(shape, dtype=DT)).requires_grad_(True)
    ((rate_of(phase) - target) ** 2).sum().backward()

    assert phase.grad is not None and torch.isfinite(phase.grad).all()
    assert float(phase.grad.abs().max()) > 0
    # The gradient must live on the support: outside it the phase is multiplied
    # by s = 0 and cannot affect anything.
    assert float(phase.grad[s == 0].abs().max()) == 0.0


@pytest.mark.autograd
def test_summed_rate_is_phase_invariant_by_parseval():
    """Pins the fact that broke the test above, so it cannot silently return.

    ``sum_q |FFT(psi)|^2 = N * sum_r |psi(r)|^2`` -- independent of the phase.
    Any future loss built on the summed rate has no phase gradient at all.
    """
    shape = (12, 12, 12)
    s = torch.zeros(shape, dtype=DT)
    s[3:9, 3:9, 3:9] = 1.0
    torch.manual_seed(0)
    flat = _I(torch.polar(s, torch.zeros_like(s))).sum()
    bumpy = _I(torch.polar(s, torch.rand(shape, dtype=DT) * s)).sum()
    assert float(flat) == pytest.approx(float(bumpy), rel=1e-12)


@pytest.mark.autograd
def test_gradient_flows_to_atomic_coordinates():
    """A detector-level loss backpropagates to every atom -- the MD ask.

    Uses a residual against a target rather than the summed rate, for the same
    reason as the object test: a sum-style loss can be invariant to the very
    parameter under test and then "pass" on numerical noise.
    """
    torch.manual_seed(0)
    B, _ = _geom(50.0, (6, 6, 6))
    Q = m2d.q_grid(B, (6, 6, 6),
                   offset=m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"])

    def rate_of(c):
        return m2d.detector_signal(m2d.speckle_from_atoms(c, ["Au"] * 30, Q),
                                   photons_per_peak=1e3)

    base = torch.randn(30, 3, dtype=DT) * 4.0
    target = rate_of(base).detach()
    coords = (base + 0.2 * torch.randn(30, 3, dtype=DT)).requires_grad_(True)
    ((rate_of(coords) - target) ** 2).sum().backward()

    assert coords.grad is not None and torch.isfinite(coords.grad).all()
    assert float(coords.grad.abs().max()) > 0
    # every atom should feel it, not just a lucky few
    assert int((coords.grad.norm(dim=-1) > 0).sum()) == 30


@pytest.mark.autograd
def test_atom_gradient_survives_chunking():
    """Chunking must not silently detach part of the graph."""
    torch.manual_seed(0)
    base = torch.randn(24, 3, dtype=DT) * 4.0
    B, _ = _geom(50.0, (6, 6, 6))
    Q = m2d.q_grid(B, (6, 6, 6),
                   offset=m2d.bragg_geometry(LAMBDA_A, D_111, dtype=DT)["G"])
    grads = []
    for max_elements in (1 << 24, 48):
        c = base.clone().requires_grad_(True)
        m2d.speckle_from_atoms(c, ["Au"] * 24, Q, max_elements=max_elements).sum().backward()
        grads.append(c.grad)
    assert torch.allclose(grads[0], grads[1], rtol=1e-10, atol=1e-14)


@pytest.mark.autograd
def test_gradient_flows_through_shear_correction():
    shape = (16, 16, 16)
    _, C = _geom(60.0, shape)
    obj = torch.zeros(shape, dtype=DT, requires_grad=True)
    out = m2d.sheared_to_lab(obj + 1.0, C)["obj"]
    out.sum().backward()
    assert obj.grad is not None and torch.isfinite(obj.grad).all()


@pytest.mark.autograd
def test_gradient_flows_from_counts_via_poisson_likelihood():
    """The differentiable route to fitting real data: fixed counts, fit the rate.

    sample_counts itself is a random draw and is NOT differentiable; the
    likelihood of those fixed counts under a predicted rate is.
    """
    shape = (12, 12, 12)
    s = torch.zeros(shape, dtype=DT)
    s[3:8, 3:8, 3:8] = 1.0
    truth = m2d.detector_signal(_I(torch.polar(s, torch.zeros_like(s))),
                                photons_per_peak=500.0)
    counts = m2d.sample_counts(truth, generator=torch.Generator().manual_seed(0))
    assert not counts.requires_grad

    scale = torch.tensor(0.7, dtype=DT, requires_grad=True)
    pred = m2d.detector_signal(_I(torch.polar(s * scale, torch.zeros_like(s))),
                               photons_per_peak=500.0)
    m2d.poisson_nll(pred, counts).backward()
    assert scale.grad is not None and torch.isfinite(scale.grad).all()


# ------------------------------------------------------------------ devices
def _devices():
    devs = ["cpu"]
    if torch.backends.mps.is_available():
        devs.append("mps")
    if torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.mark.parametrize("device", _devices())
def test_forward_chain_is_device_portable(device):
    """MPS has no float64, so the chain must also run in float32."""
    dt = torch.float32 if device == "mps" else torch.float64
    shape = (16, 16, 16)
    B = m2d.q_basis(LAMBDA_A, D_111, distance_mm=640.0, pixel_mm=0.055,
                    rocking_step_deg=0.0084, dtype=dt).to(device)
    C = m2d.conjugate_real_basis(B, shape)
    G = m2d.bragg_geometry(LAMBDA_A, D_111, dtype=dt)["G"].to(device)

    s = torch.zeros(shape, dtype=dt, device=device)
    s[5:11, 5:11, 5:11] = 1.0
    I = _I(torch.polar(s, torch.zeros_like(s)))
    rate = m2d.detector_signal(I, Q=m2d.q_grid(B, shape, offset=G),
                               wavelength_A=LAMBDA_A, structure_factor_sq=2.0,
                               coherence_length_A=200.0, real_basis=C,
                               photons_per_peak=1e3)
    assert rate.device.type == device
    assert torch.isfinite(rate).all()

    coords = torch.randn(20, 3, dtype=dt, device=device) * 4.0
    out = m2d.speckle_from_atoms(coords, ["Au"] * 20,
                                 m2d.q_grid(B, (4, 4, 4), offset=G))
    assert out.device.type == device and torch.isfinite(out).all()
