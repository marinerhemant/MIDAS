"""Reconstruction wiring and Monte-Carlo variance. Needs the midas-tomo engine."""

from __future__ import annotations

import numpy as np
import pytest

from midas_dt.channels import Channel
from midas_dt.conventions import RECON_SIGN
from midas_dt.recon import reconstruct
from midas_dt.sinogram import assemble

pytest.importorskip("midas_tomo")
from midas_tomo import backend_c  # noqa: E402

if not backend_c.available():
    pytest.skip(f"engine not built: {backend_c.why_unavailable()}",
                allow_module_level=True)


def _phantom_stack(n_bins=3, n_trans=32, n_omega=60, seed=0):
    """Sinograms of a small off-centre disc, one per bin, with Poisson noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-1, 1, n_trans)
    omega = np.linspace(0.0, 180.0, n_omega, endpoint=False)
    inten = np.empty((n_bins, n_omega, n_trans))
    for b in range(n_bins):
        amp = 100.0 * (b + 1)
        for i, th in enumerate(np.deg2rad(omega)):
            centre = 0.3 * np.cos(th)          # off-centre feature
            inten[b, i] = amp * np.exp(-((x - centre) / 0.25) ** 2)
    inten = rng.poisson(inten).astype(np.float64)
    # (bin, omega, trans) -> the (trans, frame, eta, r) assemble() wants
    four = np.transpose(inten, (2, 1, 0)).reshape(n_trans, n_omega, 1, n_bins)
    return assemble(four, four.copy(), omega,
                    Channel(105, 125, r_bin=(125 - 105) / n_bins, eta_bin=360),
                    snake=False)


def test_reconstruct_shapes_and_bin_order():
    st = _phantom_stack()
    rec = reconstruct(st, n_cpus=2, extra_pad=True)
    assert rec.n_bins == st.n_bins
    assert rec.size == 64                       # 2 * next_pow2(32)
    assert rec.intensity.shape == (st.n_bins, 64, 64)
    assert rec.variance is None                 # not requested


def test_reconstruct_applies_the_sign():
    """RECON_SIGN = -1: doLog=0 back-projects intensity, so the raw engine
    output is negative-going. Getting this wrong inverts every peak."""
    st = _phantom_stack()
    signed = reconstruct(st, n_cpus=2)
    unsigned = reconstruct(st, n_cpus=2, apply_sign=False)
    np.testing.assert_allclose(signed.intensity, unsigned.intensity * RECON_SIGN)
    assert signed.sign_applied == RECON_SIGN


def _planted_disc(n=20, n_ang=60, n_bins=3, radius=6.0):
    """A disc of KNOWN position, projected with the package's own operator."""
    from midas_dt.direct import projection_matrix
    torch = pytest.importorskip("torch")

    om = np.linspace(0.0, 180.0, n_ang, endpoint=False)
    c = (n - 1) / 2.0
    yy, xx = np.mgrid[0:n, 0:n]
    truth = (((xx - c) ** 2 + (yy - c) ** 2) <= radius ** 2).astype(float)
    A = projection_matrix(n, om, n)
    sino = torch.sparse.mm(
        A, torch.as_tensor(np.tile(truth.ravel(), (n_bins, 1))).T).T
    sino = sino.numpy().reshape(n_bins, n_ang, n)
    inten = np.transpose(sino, (2, 1, 0)).reshape(n, n_ang, 1, n_bins)
    stack = assemble(inten, np.clip(np.abs(inten), 1e-6, None), om,
                     Channel(105, 125, r_bin=20.0 / n_bins, eta_bin=360),
                     snake=False)
    return stack, truth


def test_reconstruction_correlates_POSITIVELY_with_a_planted_object():
    """The sign convention, tested against a known truth rather than a proxy.

    This replaces a guard that asserted only ``core.max() > median(img)``. That
    is satisfied by a streaky INVERTED image, and it was: RECON_SIGN shipped as
    -1 (copied from the 2023 legacy script) and every reconstruction came out
    corr = -0.87 against the object that produced it. Nothing else in the suite
    caught it, because the branch A/B tests compare the two branches to each
    other and both carry the same sign.
    """
    stack, truth = _planted_disc()
    rec = reconstruct(stack, shift=0.0, n_cpus=2)
    img = rec.intensity[0]
    k = img.shape[0] // 2
    h = truth.shape[0] // 2
    core = img[k - h:k + h, k - h:k + h]

    corr = float(np.corrcoef(core.ravel(), truth.ravel())[0, 1])
    assert corr > 0.5, (
        f"reconstruction correlates {corr:+.3f} with the object that produced "
        f"it. Negative means RECON_SIGN is inverted; every map would be of the "
        f"complement of the sample."
    )
    assert core[truth > 0].mean() > core[truth == 0].mean(), (
        "the sample is not brighter than the space around it")


def test_the_default_mask_selects_the_sample_not_the_background():
    """The consequence that made the sign bug destructive rather than cosmetic.

    ``run_recon_then_fit`` keeps voxels above the 60th percentile of total
    intensity. Under the inverted sign that selected the background: measured
    0.000 of fitted voxels inside the sample against 0.344 by chance.
    """
    stack, truth = _planted_disc()
    rec = reconstruct(stack, shift=0.0, n_cpus=2)
    totals = rec.intensity.sum(axis=0)
    k = totals.shape[0] // 2
    h = truth.shape[0] // 2
    kept = (totals > np.nanpercentile(totals, 60))[k - h:k + h, k - h:k + h]

    inside = (kept & (truth > 0)).sum() / max(kept.sum(), 1)
    chance = (truth > 0).mean()
    assert inside > chance, (
        f"the mask selects the sample {inside:.3f} of the time, at or below "
        f"chance {chance:.3f} -- it is fitting the background")


def test_brighter_bin_reconstructs_brighter():
    st = _phantom_stack()
    rec = reconstruct(st, n_cpus=2)
    peaks = [rec.intensity[b].max() for b in range(rec.n_bins)]
    assert peaks[2] > peaks[1] > peaks[0], f"bin ordering lost: {peaks}"


def test_voxel_pattern_reshapes_to_eta_by_r():
    st = _phantom_stack(n_bins=3)
    rec = reconstruct(st, n_cpus=2)
    pat = rec.voxel_pattern(30, 30)
    assert pat.shape == rec.bin_shape == (1, 3)


@pytest.mark.slow
def test_monte_carlo_variance_is_positive_and_scales():
    """MC variance must be non-negative everywhere.

    This is the property the rejected shortcut (pushing the variance sinogram
    through FBP) does NOT have -- the ramp filter's negative lobes let it go
    below zero.
    """
    st = _phantom_stack()
    rec = reconstruct(st, n_cpus=2, variance_samples=8)
    assert rec.variance is not None
    assert np.all(rec.variance >= 0.0), "MC variance went negative"
    assert rec.sigma is not None
    np.testing.assert_allclose(rec.sigma, np.sqrt(rec.variance))
    assert rec.variance.max() > 0, "variance is identically zero"


def test_variance_samples_must_be_zero_or_at_least_two():
    st = _phantom_stack()
    with pytest.raises(ValueError, match="0 or >= 2"):
        reconstruct(st, n_cpus=2, variance_samples=1)


def test_limits_travel_with_the_reconstruction():
    st = _phantom_stack()
    rec = reconstruct(st, n_cpus=2)
    assert rec.limits is st.limits
    assert any("Self-absorption" in w for w in rec.limits.warnings())
