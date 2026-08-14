"""Self-absorption correction, and the SIRT / TV reconstructors.

Both are easy to get subtly and invisibly wrong, so the tests target the
specific wrong answers rather than "it runs":

* an attenuation model with the outgoing leg reversed still attenuates, just
  toward the wrong side of the sample
* an adjoint that drifts from its projector still converges, to a wrong image
* a TV weight high enough to invent flat regions still produces a clean-looking
  map with small scatter
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from midas_dt.absorption import (                            # noqa: E402
    attenuated_projection_matrix,
    attenuation_factors,
    correct_reconstruction,
    in_plane_deflection_deg,
    uniform_mu,
)
from midas_dt.channels import Channel                        # noqa: E402
from midas_dt.direct import projection_matrix                # noqa: E402
from midas_dt.sinogram import assemble                       # noqa: E402


def _disc_mask(n, radius):
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    return ((xx - c) ** 2 + (yy - c) ** 2) <= radius ** 2


def _stack(n_trans=16, n_omega=12, n_bins=3, seed=0):
    rng = np.random.default_rng(seed)
    omega = np.linspace(0.0, 180.0, n_omega, endpoint=False)
    inten = rng.random((n_trans, n_omega, 1, n_bins)) + 1.0
    return assemble(inten, np.abs(inten), omega,
                    Channel(105, 125, r_bin=20.0 / n_bins, eta_bin=360),
                    snake=False)


# ----------------------------------------------------------- attenuation
def test_zero_mu_is_a_no_op():
    """No absorption information must mean no correction, not an error."""
    f = attenuation_factors(np.zeros((16, 16)), np.linspace(0, 180, 8))
    np.testing.assert_allclose(f, 1.0)


def test_factors_are_bounded_and_attenuating():
    mu = uniform_mu(_disc_mask(24, 8), 0.05)
    f = attenuation_factors(mu, np.linspace(0, 180, 10, endpoint=False))
    assert f.shape == (10, 24, 24)
    assert np.all(f > 0) and np.all(f <= 1.0)
    assert f[:, _disc_mask(24, 8)].mean() < 1.0, "no attenuation inside the sample"


def test_the_sample_centre_is_attenuated_more_than_its_edge():
    """The radial ordering absorption must produce.

    Necessary but not sufficient: verified by mutation, this still passes when
    the outgoing leg is wrongly taken as the upstream integral, because
    averaging over a full rotation symmetrises the near/far asymmetry.
    ``test_outgoing_leg_is_downstream_not_upstream`` is the test that
    discriminates that case.
    """
    n = 32
    mask = _disc_mask(n, 12)
    mu = uniform_mu(mask, 0.08)
    f = attenuation_factors(mu, np.linspace(0, 180, 16, endpoint=False)).mean(axis=0)

    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    rr = np.hypot(xx - c, yy - c)
    core = mask & (rr < 4)
    rim = mask & (rr > 10)
    assert f[core].mean() < f[rim].mean(), (
        f"centre {f[core].mean():.4f} is not more attenuated than rim "
        f"{f[rim].mean():.4f} -- the in/out path integrals are likely swapped")


def test_outgoing_leg_is_downstream_not_upstream():
    """A voxel at the entry face has no incoming attenuation but must still
    have outgoing attenuation. Using the upstream integral for both legs -- an
    easy and silent mistake -- would give it a factor of exactly 1."""
    n = 24
    mu = np.zeros((n, n))
    mu[:, 8:16] = 0.1                      # a slab, so entry/exit are clear
    f = attenuation_factors(mu, np.array([0.0]))[0]
    # Beam runs along rows at omega=0; the first row inside the slab is the
    # entry face for that column.
    entry = f[0, 12]
    assert entry < 0.99, (
        f"entry-face voxel has factor {entry:.4f}; the outgoing leg is missing")


def test_more_absorbing_sample_attenuates_more():
    mask = _disc_mask(24, 9)
    om = np.linspace(0, 180, 8, endpoint=False)
    weak = attenuation_factors(uniform_mu(mask, 0.02), om)[:, mask].mean()
    strong = attenuation_factors(uniform_mu(mask, 0.20), om)[:, mask].mean()
    assert strong < weak


def test_negative_mu_is_rejected():
    with pytest.raises(ValueError, match="cannot amplify"):
        attenuation_factors(-np.ones((8, 8)) * 0.1, np.array([0.0]))


@pytest.mark.parametrize("two_theta,eta,expect", [
    (2.0, 0.0, 2.0),          # equatorial: all of 2theta is in-plane
    (2.0, 180.0, -2.0),       # opposite side
    (2.0, 90.0, 0.0),         # out of plane: no in-plane deflection
])
def test_in_plane_deflection(two_theta, eta, expect):
    assert in_plane_deflection_deg(two_theta, eta) == pytest.approx(expect, abs=1e-6)


def test_small_angle_is_decided_not_assumed():
    """At synchrotron energies 2theta is ~1 deg and the approximation is free;
    at 40 deg it is not, and must not be applied silently."""
    mu = uniform_mu(_disc_mask(20, 7), 0.05)
    om = np.array([0.0, 90.0])
    near = attenuation_factors(mu, om, two_theta_deg=1.0, eta_deg=0.0)
    far = attenuation_factors(mu, om, two_theta_deg=40.0, eta_deg=0.0)
    # Both valid; what matters is that they differ, i.e. the large angle was
    # not quietly collapsed onto the small-angle path.
    assert not np.allclose(near, far)


# ------------------------------------------------- the exact route (branch C)
def test_attenuated_operator_reduces_the_projection():
    n, n_ang = 16, 8
    om = np.linspace(0, 180, n_ang, endpoint=False)
    mu = uniform_mu(_disc_mask(n, 6), 0.1)
    f = attenuation_factors(mu, om)

    img = np.zeros((1, n, n))
    img[0][_disc_mask(n, 6)] = 1.0
    x = torch.as_tensor(img.reshape(1, -1), dtype=torch.float64)

    A = projection_matrix(n, om, n)
    Aa = attenuated_projection_matrix(n, om, n, f)
    plain = torch.sparse.mm(A, x.T).T
    atten = torch.sparse.mm(Aa, x.T).T
    assert float(atten.sum()) < float(plain.sum())
    assert float(atten.min()) >= 0.0


def test_attenuated_operator_keeps_an_exact_adjoint():
    """Scaling the COO values must not break <Ax, y> == <x, A^T y>."""
    n, n_ang = 12, 6
    om = np.linspace(0, 180, n_ang, endpoint=False)
    f = attenuation_factors(uniform_mu(_disc_mask(n, 5), 0.07), om)
    Aa = attenuated_projection_matrix(n, om, n, f)

    rng = np.random.default_rng(3)
    x = torch.as_tensor(rng.random(n * n), dtype=torch.float64)
    y = torch.as_tensor(rng.random(n_ang * n), dtype=torch.float64)
    lhs = float(torch.sparse.mm(Aa, x[:, None]).ravel() @ y)
    rhs = float(x @ torch.sparse.mm(Aa.t(), y[:, None]).ravel())
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_wrong_factor_shape_is_rejected():
    om = np.linspace(0, 180, 6, endpoint=False)
    with pytest.raises(ValueError, match="factors must be"):
        attenuated_projection_matrix(12, om, 12, np.ones((3, 12, 12)))


# ------------------------------------------- the approximate route (A and B)
def test_correct_reconstruction_brightens_the_interior():
    from midas_dt.recon import Reconstruction

    n = 24
    mask = _disc_mask(n, 9)
    mu = uniform_mu(mask, 0.08)
    f = attenuation_factors(mu, np.linspace(0, 180, 12, endpoint=False))

    flat = np.ones((2, n, n))
    rec = Reconstruction(intensity=flat, variance=None, bin_shape=(1, 2),
                         channel=Channel(105, 125, r_bin=10.0, eta_bin=360),
                         limits=_stack().limits)
    out = correct_reconstruction(rec, f)
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    core = mask & (np.hypot(xx - c, yy - c) < 3)
    assert np.nanmean(out.intensity[0][core]) > 1.0, (
        "the interior was not brightened; the correction is not undoing "
        "attenuation")


def test_correct_reconstruction_refuses_to_amplify_without_bound():
    """Dividing by a near-zero mean attenuation turns noise into a bright
    artefact that reads as a real feature. NaN is the honest output."""
    from midas_dt.recon import Reconstruction

    n = 16
    f = np.full((4, n, n), 1e-9)          # effectively opaque everywhere
    rec = Reconstruction(intensity=np.ones((1, n, n)), variance=None,
                         bin_shape=(1, 1),
                         channel=Channel(105, 125, r_bin=20.0, eta_bin=360),
                         limits=_stack().limits)
    out = correct_reconstruction(rec, f, floor=1e-3)
    assert np.all(np.isnan(out.intensity)), "unbounded amplification was allowed"


def test_correct_reconstruction_propagates_variance_quadratically():
    from midas_dt.recon import Reconstruction

    n = 16
    f = np.full((4, n, n), 0.5)
    rec = Reconstruction(intensity=np.ones((1, n, n)),
                         variance=np.full((1, n, n), 4.0), bin_shape=(1, 1),
                         channel=Channel(105, 125, r_bin=20.0, eta_bin=360),
                         limits=_stack().limits)
    out = correct_reconstruction(rec, f)
    # x/a has variance var/a^2, not var/a.
    np.testing.assert_allclose(out.variance, 4.0 / 0.25)
    np.testing.assert_allclose(out.intensity, 2.0)


# --------------------------------------------------------------- SIRT / TV
def test_backproject_is_the_exact_transpose():
    from midas_dt.iterative import backproject

    n, n_ang = 12, 7
    om = np.linspace(0, 180, n_ang, endpoint=False)
    A = projection_matrix(n, om, n)
    rng = np.random.default_rng(4)
    y = torch.as_tensor(rng.random((3, n_ang * n)), dtype=torch.float64)
    got = backproject(A, y)
    want = torch.stack([torch.sparse.mm(A.t(), y[k][:, None]).ravel()
                        for k in range(3)])
    torch.testing.assert_close(got, want)


def _phantom_stack(n=20, n_ang=30, n_bins=2, seed=5):
    """Two discs of different brightness, and their exact sinograms."""
    rng = np.random.default_rng(seed)
    om = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    truth = np.zeros((n_bins, n, n))
    truth[0][((xx - c + 3) ** 2 + (yy - c) ** 2) <= 3.5 ** 2] = 1.0
    truth[1][((xx - c - 3) ** 2 + (yy - c) ** 2) <= 3.5 ** 2] = 2.0

    A = projection_matrix(n, om, n)
    sino = torch.sparse.mm(
        A, torch.as_tensor(truth.reshape(n_bins, -1)).T).T.numpy()
    sino = sino.reshape(n_bins, n_ang, n)
    inten = np.transpose(sino, (2, 1, 0)).reshape(n, n_ang, 1, n_bins)
    stack = assemble(inten, np.clip(np.abs(inten), 1e-6, None), om,
                     Channel(105, 125, r_bin=20.0 / n_bins, eta_bin=360),
                     snake=False)
    return stack, truth


def _corr(a, b):
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


@pytest.mark.slow
def test_sirt_recovers_the_phantom():
    from midas_dt.iterative import sirt

    stack, truth = _phantom_stack()
    rec = sirt(stack, n_iter=120, apply_sign=False)
    assert rec.intensity.shape == truth.shape
    for k in range(truth.shape[0]):
        assert _corr(rec.intensity[k], truth[k]) > 0.9, (
            f"bin {k} correlates only {_corr(rec.intensity[k], truth[k]):.3f}")


@pytest.mark.slow
def test_sirt_keeps_the_bins_distinct():
    """All bins iterate together as columns of one matrix; a broadcasting slip
    there would blend them into each other."""
    from midas_dt.iterative import sirt

    stack, truth = _phantom_stack()
    rec = sirt(stack, n_iter=120, apply_sign=False)
    # Bin 1 is twice as bright as bin 0, and in a different place.
    assert rec.intensity[1].max() > 1.4 * rec.intensity[0].max()
    assert _corr(rec.intensity[0], truth[1]) < _corr(rec.intensity[0], truth[0])


def test_sirt_respects_non_negativity():
    from midas_dt.iterative import sirt

    stack, _ = _phantom_stack()
    rec = sirt(stack, n_iter=30, non_negative=True, apply_sign=False)
    assert rec.intensity.min() >= 0.0


def test_sirt_applies_the_sign_convention_by_default():
    from midas_dt.iterative import sirt

    stack, _ = _phantom_stack(n=14, n_ang=16)
    from midas_dt.conventions import RECON_SIGN
    signed = sirt(stack, n_iter=20)
    raw = sirt(stack, n_iter=20, apply_sign=False)
    np.testing.assert_allclose(signed.intensity, raw.intensity * RECON_SIGN)
    assert signed.sign_applied == RECON_SIGN


@pytest.mark.slow
def test_tv_with_zero_weight_reproduces_plain_least_squares():
    """The control. If tv_weight=0 does not match an unregularised fit, the
    penalty is leaking in and every weight comparison is meaningless."""
    from midas_dt.iterative import tv_reconstruct

    stack, truth = _phantom_stack(n=16, n_ang=24)
    rec = tv_reconstruct(stack, tv_weight=0.0, steps=400, lr=0.1,
                         apply_sign=False)
    assert _corr(rec.intensity[0], truth[0]) > 0.85


@pytest.mark.slow
def test_tv_suppresses_noise_relative_to_no_regularisation():
    from midas_dt.iterative import tv_reconstruct

    stack, truth = _phantom_stack(n=16, n_ang=24)
    rng = np.random.default_rng(7)
    noisy = type(stack)(
        intensity=stack.intensity + rng.normal(0, 0.5, stack.intensity.shape),
        variance=stack.variance, omega_deg=stack.omega_deg,
        channel=stack.channel, bin_shape=stack.bin_shape, limits=stack.limits,
        translations=stack.translations)

    plain = tv_reconstruct(noisy, tv_weight=0.0, steps=300, lr=0.1,
                           apply_sign=False)
    smooth = tv_reconstruct(noisy, tv_weight=0.05, steps=300, lr=0.1,
                            apply_sign=False)

    def roughness(img):
        return (np.abs(np.diff(img, axis=0)).mean()
                + np.abs(np.diff(img, axis=1)).mean())

    assert roughness(smooth.intensity[0]) < roughness(plain.intensity[0]), (
        "TV did not reduce roughness; the penalty is not reaching the loss")


def test_tv_weight_must_be_reported_not_guessed():
    """A regression guard on the docstring, since the danger of TV is a
    plausible-looking map rather than an error."""
    import midas_dt.iterative as it
    doc = it.tv_reconstruct.__doc__.lower()
    assert "not a free parameter" in doc
    assert "piecewise-constant" in doc


def test_each_angle_uses_its_own_attenuation_factors():
    """Pins the (angle, voxel) indexing inside the operator.

    Found by mutation: replacing ``flat[ang_of_row, idx[1]]`` with
    ``flat[0, idx[1]]`` -- applying angle 0's factors to every angle, a very
    ordinary indexing slip -- passed the entire rest of this file. Nothing else
    here can see it, because the aggregate tests only check that attenuation
    reduces the signal, which the wrong indexing also does.

    A single unit voxel makes the expectation analytic: the attenuated
    projection at angle ``a`` must equal the plain projection at angle ``a``
    scaled by exactly ``factors[a, voxel]``.
    """
    n, n_ang = 16, 4
    om = np.array([0.0, 45.0, 90.0, 135.0])

    # Strongly asymmetric mu, so the factors genuinely differ angle to angle.
    mu = np.zeros((n, n))
    mu[2:14, 2:7] = 0.25
    f = attenuation_factors(mu, om)

    iy, ix = 8, 9
    v = iy * n + ix
    spread = f[:, iy, ix]
    assert spread.max() / spread.min() > 1.2, (
        "the test fixture is too symmetric to discriminate; factors vary by "
        f"only {spread.max() / spread.min():.3f} across angles")

    img = np.zeros((1, n, n))
    img[0, iy, ix] = 1.0
    x = torch.as_tensor(img.reshape(1, -1), dtype=torch.float64)

    plain = torch.sparse.mm(projection_matrix(n, om, n), x.T).T.numpy().reshape(n_ang, n)
    atten = torch.sparse.mm(
        attenuated_projection_matrix(n, om, n, f), x.T).T.numpy().reshape(n_ang, n)

    for a in range(n_ang):
        np.testing.assert_allclose(
            atten[a], plain[a] * f[a, iy, ix], rtol=1e-12, atol=1e-14,
            err_msg=f"angle {a} ({om[a]} deg) is not using its own factors")
