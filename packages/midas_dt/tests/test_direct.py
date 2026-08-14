"""Branch C: differentiable direct inversion.

These test CORRECTNESS, not superiority. Whether direct inversion beats
Branch B is an open question gated on a preregistered comparison, and nothing
here asserts an answer to it -- see the module docstring of
``midas_dt.direct``.

What is checked, in the order a wrong answer would slip through:

1. the projector matches an independent reference implementation
2. its adjoint is really its adjoint (dot-product test)
3. autograd gradients match finite differences
4. the solver recovers planted parameters from noiseless data
5. the peak model agrees with the one Branch B fits
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("midas_invert")

from midas_dt.channels import Channel                       # noqa: E402
from midas_dt.direct import (                               # noqa: E402
    MIX_FACTOR,
    forward_project,
    projection_matrix,
    run_direct,
)
from midas_dt.sinogram import assemble                      # noqa: E402


# ------------------------------------------------------- 1. the projector
def _reference_project(img: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Independent linear-splat Radon, written from the convention directly.

    Deliberately a plain double loop: if this and the sparse-matrix version
    agree, the vectorised index arithmetic in the latter is right. Sharing
    code between the two would test nothing.
    """
    n = img.shape[0]
    c = (n - 1) / 2.0
    out = np.zeros((len(angles_deg), n))
    for a, th in enumerate(np.deg2rad(angles_deg)):
        for iy in range(n):
            for ix in range(n):
                v = img[iy, ix]
                if v == 0:
                    continue
                x, y = ix - c, iy - c
                t = x * np.sin(th) + y * np.cos(th) + c
                lo = int(np.floor(t))
                f = t - lo
                if 0 <= lo < n:
                    out[a, lo] += v * (1 - f)
                if 0 <= lo + 1 < n:
                    out[a, lo + 1] += v * f
    return out


def test_projector_matches_an_independent_reference():
    rng = np.random.default_rng(0)
    n, angles = 16, np.linspace(0.0, 180.0, 12, endpoint=False)
    img = rng.random((n, n))

    A = projection_matrix(n, angles, n)
    got = forward_project(
        torch.as_tensor(img, dtype=torch.float64)[None], A
    ).reshape(len(angles), n).numpy()

    np.testing.assert_allclose(got, _reference_project(img, angles), atol=1e-10)


def test_projector_uses_x_sin_plus_y_cos_not_the_transpose():
    """The convention error that reconstructs to something plausible.

    ``x cos + y sin`` differs from the correct form by a transpose of the
    image. A symmetric phantom would hide it, so this uses an asymmetric one
    and requires the projection to differ from the transposed image's.
    """
    n, angles = 16, np.array([30.0])
    img = np.zeros((n, n))
    img[3, 11] = 1.0                       # off-diagonal, so transpose moves it

    A = projection_matrix(n, angles, n)
    got = forward_project(torch.as_tensor(img)[None], A).numpy()
    tra = forward_project(torch.as_tensor(img.T.copy())[None], A).numpy()
    assert not np.allclose(got, tra), "projector is transpose-blind"
    np.testing.assert_allclose(
        got.reshape(1, n), _reference_project(img, angles), atol=1e-10)


def test_adjoint_passes_the_dot_product_test():
    """<A x, y> == <x, A^T y>.

    The gradient of the loss IS the adjoint applied to the residual, so an
    adjoint that is not the transpose gives a descent direction that is not
    downhill -- and the optimiser still converges, to the wrong place.
    """
    rng = np.random.default_rng(1)
    n, angles = 12, np.linspace(0.0, 180.0, 9, endpoint=False)
    A = projection_matrix(n, angles, n)

    x = torch.as_tensor(rng.random(n * n), dtype=torch.float64)
    y = torch.as_tensor(rng.random(len(angles) * n), dtype=torch.float64)
    Ax = torch.sparse.mm(A, x[:, None]).ravel()
    Aty = torch.sparse.mm(A.t(), y[:, None]).ravel()
    assert float(Ax @ y) == pytest.approx(float(x @ Aty), rel=1e-12)


def test_projection_of_a_uniform_disc_is_a_smooth_hump():
    """A sanity check with a known shape, in case both projectors are wrong."""
    n, angles = 32, np.linspace(0.0, 180.0, 24, endpoint=False)
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    img = (((xx - c) ** 2 + (yy - c) ** 2) <= 8.0 ** 2).astype(float)

    A = projection_matrix(n, angles, n)
    p = forward_project(torch.as_tensor(img)[None], A).reshape(len(angles), n).numpy()

    # Every view of a centred disc is the same profile, centred. Use the
    # CENTROID, not argmax: a disc projects to a flat-topped profile whose
    # argmax hops between near-equal bins on rounding alone.
    bins = np.arange(n)
    centroid = (p * bins).sum(axis=1) / p.sum(axis=1)
    assert centroid == pytest.approx(np.full(len(angles), c), abs=0.05)
    assert p.std(axis=0).max() < 0.05 * p.max()   # rotationally invariant
    assert p.sum(axis=1) == pytest.approx(np.full(len(angles), img.sum()), rel=1e-9)


# --------------------------------------------------------- 2. the gradient
def test_autograd_gradient_matches_finite_differences():
    """Cheap insurance: an analytic gradient that is subtly wrong still
    optimises, just to the wrong answer."""
    n, angles = 8, np.linspace(0.0, 180.0, 6, endpoint=False)
    A = projection_matrix(n, angles, n)
    rng = np.random.default_rng(2)
    target = torch.as_tensor(rng.random((3, len(angles) * n)), dtype=torch.float64)
    x = torch.as_tensor(rng.random((3, n, n)), dtype=torch.float64,
                        ).requires_grad_(True)

    def loss(t):
        return ((forward_project(t, A) - target) ** 2).mean()

    loss(x).backward()
    analytic = x.grad.clone()

    eps, idx = 1e-6, [(0, 2, 3), (1, 5, 1), (2, 7, 7)]
    for (k, i, j) in idx:
        p = x.detach().clone(); p[k, i, j] += eps
        m = x.detach().clone(); m[k, i, j] -= eps
        fd = (loss(p) - loss(m)) / (2 * eps)
        assert float(analytic[k, i, j]) == pytest.approx(float(fd), rel=1e-5,
                                                         abs=1e-12)


# ------------------------------------------------------- 3. the peak model
def test_peak_model_matches_the_branch_b_fitter():
    """Branch C must fit the SAME profile Branch B does, or comparing the two
    measures the model difference rather than the method difference."""
    from midas_dt.direct import _profile

    r = np.linspace(105.0, 125.0, 41)
    cen, sg = 113.7, 2.4
    # float64 explicitly: torch.tensor([python_float]) defaults to float32,
    # and comparing a float32 profile to a float64 one fails at 1e-6 for
    # reasons that have nothing to do with the model.
    d64 = dict(dtype=torch.float64)
    got = _profile(torch.as_tensor(r, **d64), torch.tensor([cen], **d64),
                   torch.tensor([sg], **d64)).numpy().ravel()

    # The expression from peakfit.fit_lineout's `model`, written out.
    g = np.exp(-0.5 * ((r - cen) / sg) ** 2)
    lo = 1.0 / (((r - cen) / sg) ** 2 + 1.0)
    np.testing.assert_allclose(got, MIX_FACTOR * lo + (1 - MIX_FACTOR) * g,
                               atol=1e-12)


def test_mix_factor_is_fixed_at_the_legacy_value():
    """PeakFit.c hard-codes 0.5 and reports it as MixFactor. Fitting it here
    would make Branch C's MixFactor map incomparable with the others'."""
    assert MIX_FACTOR == 0.5


# ------------------------------------------------------------ 4. the solve
def _planted(n_trans=16, n_omega=24, n_r=21, seed=0):
    """A two-blob sample with DIFFERENT peak centres, and its exact sinogram.

    Different centres matter: if every voxel shared a centre, recovering it
    would not demonstrate that the inversion resolves spatial variation.
    """
    rng = np.random.default_rng(seed)
    r = np.linspace(105.0, 125.0, n_r)
    omega = np.linspace(0.0, 180.0, n_omega, endpoint=False)
    c = (n_trans - 1) / 2.0
    yy, xx = np.mgrid[0:n_trans, 0:n_trans]

    blob_a = ((xx - c + 3) ** 2 + (yy - c) ** 2) <= 2.5 ** 2
    blob_b = ((xx - c - 3) ** 2 + (yy - c) ** 2) <= 2.5 ** 2
    amp = np.where(blob_a, 100.0, 0.0) + np.where(blob_b, 60.0, 0.0)
    cen = np.where(blob_a, 111.0, 0.0) + np.where(blob_b, 117.0, 0.0)
    sig = np.where(blob_a | blob_b, 2.0, 1.0)

    prof = MIX_FACTOR / (((r[None, None] - cen[..., None]) / sig[..., None]) ** 2 + 1) \
        + (1 - MIX_FACTOR) * np.exp(
            -0.5 * ((r[None, None] - cen[..., None]) / sig[..., None]) ** 2)
    img = amp[..., None] * prof                       # (X, X, n_r)

    A = projection_matrix(n_trans, omega, n_trans)
    sino = forward_project(
        torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1))), A
    ).numpy().reshape(n_r, n_omega, n_trans)

    truth = {"amp": amp, "cen": cen, "sig": sig, "mask": blob_a | blob_b}
    return sino, omega, r, truth, rng


def _stack_from(sino, omega, n_r):
    """Wrap a (n_r, n_omega, n_trans) array as a SinogramStack."""
    n_om, n_tr = sino.shape[1], sino.shape[2]
    inten = np.transpose(sino, (2, 1, 0)).reshape(n_tr, n_om, 1, n_r)
    return assemble(inten, np.clip(np.abs(inten), 1e-6, None), omega,
                    Channel(105, 125, r_bin=20.0 / n_r, eta_bin=360),
                    snake=False)


@pytest.mark.slow
def test_recovers_planted_peak_centres_from_noiseless_data():
    """The end-to-end claim that IS being made: on data generated by the
    model, the solver finds the parameters that generated it."""
    sino, omega, r, truth, _ = _planted()
    stack = _stack_from(sino, omega, len(r))

    res = run_direct(stack, steps=600, lr=0.15, mask_threshold=None)
    got = res.maps["RMEAN"]

    m = truth["mask"]
    err = np.abs(got[m] - truth["cen"][m])
    # 0.2 px, well inside one r bin (the grid is 1 px per bin). Measured at
    # these settings: 0.019 px median, and 0.0005 px at 2000 steps / lr 0.3.
    assert np.nanmedian(err) < 0.2, (
        f"median centre error {np.nanmedian(err):.3f} px over {m.sum()} voxels")
    assert res.residual_rel < 0.05, res.describe()


@pytest.mark.slow
def test_the_two_blobs_are_resolved_as_different_centres():
    """Recovering the mean is not enough -- the point is spatial contrast."""
    sino, omega, r, truth, _ = _planted()
    stack = _stack_from(sino, omega, len(r))
    res = run_direct(stack, steps=600, lr=0.15, mask_threshold=None)

    got, cen = res.maps["RMEAN"], truth["cen"]
    a = got[(cen > 110) & (cen < 112)]
    b = got[(cen > 116) & (cen < 118)]
    assert np.nanmean(a) < np.nanmean(b) - 3.0, (
        f"blobs not separated: {np.nanmean(a):.2f} vs {np.nanmean(b):.2f}")


def test_loss_decreases():
    sino, omega, r, _, _ = _planted(n_trans=12, n_omega=16, n_r=15)
    stack = _stack_from(sino, omega, len(r))
    res = run_direct(stack, steps=60, lr=0.1, mask_threshold=None, log_every=10)
    # midas_invert.fit returns history as a list of loss floats.
    assert len(res.history) >= 2
    assert res.history[-1] < res.history[0]


# ------------------------------------------------------------- 5. plumbing
def test_all_twelve_outputs_are_present_and_in_the_c_order():
    from midas_dt.conventions import FIT_OUTPUT_NAMES
    sino, omega, r, _, _ = _planted(n_trans=10, n_omega=12, n_r=13)
    stack = _stack_from(sino, omega, len(r))
    res = run_direct(stack, steps=20, mask_threshold=None)
    assert list(res.maps) == list(FIT_OUTPUT_NAMES)


def test_outputs_that_are_not_the_same_measurement_are_flagged():
    """MaxIntensityObs and MeanError have no direct-inversion counterpart.
    They must be surfaced, not quietly filled with a lookalike number."""
    sino, omega, r, _, _ = _planted(n_trans=10, n_omega=12, n_r=13)
    stack = _stack_from(sino, omega, len(r))
    br = run_direct(stack, steps=20, mask_threshold=None).as_branch_result()
    flagged = br.approximate_outputs()
    assert "MaxIntensityObs" in flagged
    assert "MeanError" in flagged
    assert "RMEAN" not in flagged


def test_mean_error_is_nan_rather_than_a_fabricated_number():
    sino, omega, r, _, _ = _planted(n_trans=10, n_omega=12, n_r=13)
    stack = _stack_from(sino, omega, len(r))
    res = run_direct(stack, steps=20, mask_threshold=None)
    assert np.all(np.isnan(res.maps["MeanError"]))


def test_compare_works_across_all_three_branches():
    """The adapter exists so A, B and C are mutually comparable."""
    from midas_dt.branches import compare
    sino, omega, r, _, _ = _planted(n_trans=10, n_omega=12, n_r=13)
    stack = _stack_from(sino, omega, len(r))
    c = run_direct(stack, steps=20, mask_threshold=None).as_branch_result()
    gaps = compare(c, c)
    assert gaps["RMEAN"]["rel_rms"] == pytest.approx(0.0, abs=1e-12)
    assert gaps["RMEAN"]["corr"] == pytest.approx(1.0, abs=1e-9)


def test_empty_mask_raises_rather_than_returning_an_empty_map():
    sino, omega, r, _, _ = _planted(n_trans=10, n_omega=12, n_r=13)
    stack = _stack_from(sino, omega, len(r))
    with pytest.raises(ValueError, match="nothing to solve for"):
        run_direct(stack, steps=5, mask_threshold=1e12)


def test_no_performance_claim_is_made_in_the_docstring():
    """A guard on the gate itself.

    The plan gates any 'direct beats Branch B' claim on a preregistered
    comparison that has not been run. This test fails if someone adds the
    claim to the module docstring without doing the work.
    """
    import midas_dt.direct as d
    doc = d.__doc__.lower()
    # The gate must be stated, in these words.
    assert "no performance claim is made or implied" in doc
    assert "has not been tested" in doc
    # Assertive claims only. A hedged sentence ("whether this beats Branch B
    # ... is an open question") is exactly what SHOULD be there, so banning the
    # bare substring would forbid the honest phrasing along with the dishonest.
    for banned in ("outperforms branch b", "is more accurate than branch b",
                   "is better than branch b", "we recommend branch c",
                   "supersedes branch b"):
        assert banned not in doc, f"ungated performance claim: {banned!r}"


# ---------------------------------------------- 6. the seed-scale regression
def test_moment_seed_is_on_the_right_order_of_magnitude():
    """The bug this pins cost an hour and looked like non-identifiability.

    The moment seed used to take the peak of the lineout SUMMED over every
    ray, so with 24 angles x 16 translations it started ~384x above the true
    voxel amplitude. Adam moves a roughly fixed distance per step in raw
    parameter units, so from there it could not arrive: the solver sat 1.5 px
    off a planted centre, and a larger learning rate made it monotonically
    WORSE -- which reads like a degenerate model and is not.

    A ray crossing the sample integrates ~size voxels, so the right seed is a
    per-ray peak divided by the path length, not a sum over rays.
    """
    from midas_dt.direct import _seed
    sino, omega, r, truth, _ = _planted()
    stack = _stack_from(sino, omega, len(r))

    *_, amp_scale = _seed(stack, 16, len(r), None, None, torch.float64, None)
    true_amp = truth["amp"][truth["mask"]].mean()      # 100 and 60 -> ~85
    ratio = amp_scale / true_amp
    assert 0.05 < ratio < 20.0, (
        f"seed amplitude {amp_scale:.3g} is {ratio:.1f}x the true ~{true_amp:.3g}; "
        f"the old sum-over-rays seed was ~384x")


def test_raw_parameters_start_near_unity():
    """The invariant that makes one learning rate work for all four
    parameters: after scaling, every raw value is order one."""
    from midas_dt.direct import Scales, _unconstrain
    sc = Scales(amp=85.0, r_min=105.0, r_max=125.0)
    raw = _unconstrain(
        torch.tensor([85.0], dtype=torch.float64),
        torch.tensor([113.0], dtype=torch.float64),
        torch.tensor([2.0], dtype=torch.float64),
        torch.tensor([1.0], dtype=torch.float64),
        sc,
    )
    for t in raw:
        assert abs(float(t)) < 10.0, f"raw parameter {float(t):.3g} is not O(1)"


def test_constrain_and_unconstrain_round_trip():
    from midas_dt.direct import Scales, _constrain, _unconstrain
    sc = Scales(amp=85.0, r_min=105.0, r_max=125.0)
    amp = torch.tensor([85.0, 12.0], dtype=torch.float64)
    cen = torch.tensor([113.0, 119.5], dtype=torch.float64)
    sig = torch.tensor([2.0, 3.5], dtype=torch.float64)
    bg = torch.tensor([1.0, 0.25], dtype=torch.float64)
    got = _constrain(_unconstrain(amp, cen, sig, bg, sc), sc)
    for a, b in zip(got, (amp, cen, sig, bg)):
        np.testing.assert_allclose(a.numpy(), b.numpy(), rtol=1e-8, atol=1e-8)


# ------------------------------------------------- 7. the Laplace error bars
@pytest.mark.slow
def test_laplace_sigma_is_physically_plausible():
    """The bug this pins reported sigma = 446 px on a 20 px window.

    The loss here is already weighted by 1/variance, so passing the converged
    loss as ``noise_var`` -- the usual plug-in when the noise is unknown --
    counts the noise twice and inflates sigma by exactly sqrt(loss * N). The
    correct factor is the mean-to-sum conversion, 1/N.
    """
    from midas_dt.direct import laplace_sigma

    sino, omega, r, truth, _ = _planted()
    stack = _stack_from(sino, omega, len(r))
    res = run_direct(stack, steps=600, lr=0.15, mask_threshold=None)

    inside = [tuple(int(k) for k in v) for v in np.argwhere(truth["mask"])[:3]]
    sig = laplace_sigma(stack, res, inside)

    window = stack.channel.r_max - stack.channel.r_min
    for v in inside:
        s_cen = sig["RMEAN"][v]
        assert np.isfinite(s_cen)
        # A centre known to ~0.02 px cannot have a 400 px error bar. Requiring
        # it to be inside the window at all is the weakest sane statement.
        assert 0 < s_cen < window, f"sigma(RMEAN)={s_cen:.4g} px, window={window} px"
        assert s_cen < 1.0, f"sigma(RMEAN)={s_cen:.4g} px is implausibly large"
        rel = sig["MaxInt"][v] / max(res.maps["MaxInt"][v], 1e-9)
        assert 0 < rel < 0.5, f"sigma(MaxInt) is {rel:.2%} of the amplitude"


@pytest.mark.slow
def test_laplace_default_does_not_use_the_converged_loss():
    """Explicitly passing the loss must reproduce the OLD, inflated numbers --
    which is how we know the default is doing something different and the
    inflation factor is understood."""
    from midas_dt.direct import laplace_sigma

    sino, omega, r, truth, _ = _planted()
    stack = _stack_from(sino, omega, len(r))
    res = run_direct(stack, steps=600, lr=0.15, mask_threshold=None)
    v = [tuple(int(k) for k in np.argwhere(truth["mask"])[1])]

    good = laplace_sigma(stack, res, v)["RMEAN"][v[0]]
    bad = laplace_sigma(stack, res, v, noise_var=res.loss)["RMEAN"][v[0]]

    n_data = stack.intensity.shape[1] * stack.intensity.shape[2] * len(r)
    assert bad / good == pytest.approx(np.sqrt(res.loss * n_data), rel=0.05), (
        "the inflation is not the expected sqrt(loss * N); the scaling "
        "explanation for the old 446 px sigma no longer holds")
    assert good < bad / 100
