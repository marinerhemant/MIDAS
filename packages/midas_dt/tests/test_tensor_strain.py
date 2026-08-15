"""Deviatoric strain tensor tomography: conventions, then planted recovery.

``dev/paper/PREREGISTER_tensor_null.md`` §10 gates everything on one thing --
the model must recover a strain field it planted itself before it is pointed at
real data. If it cannot, no result from the real scan counts.

The convention tests come first because they are cheap and because each of them
guards an error that produces a smooth, plausible, wrong map: a transposed
rotation rotates every tensor by -2*omega, a swapped design column relabels two
components, and neither shows up as a bad fit.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("midas_invert")

from midas_dt.tensor_strain import (
    COMPONENT_NAMES, DeviatoricStrain, deviatoric_design, fit_tensor_strain,
    q_hat_sample_frame, strain_to_radius,
)

LAMBDA_A, LSD_UM, PX_UM = 0.11595, 1632201.0, 150.0
D0_111, D0_200 = 5.4116 / np.sqrt(3), 5.4116 / 2.0


def _two_theta(d0):
    return 2.0 * np.degrees(np.arcsin(LAMBDA_A / (2.0 * d0)))


# ------------------------------------------------------------- conventions
def test_q_hat_is_a_unit_vector():
    """The whole five-not-six argument rests on this being exactly 1."""
    q = q_hat_sample_frame(np.linspace(1, 8, 7)[:, None, None],
                           np.linspace(-180, 175, 12)[None, :, None],
                           np.linspace(0, 350, 9)[None, None, :])
    n = np.linalg.norm(q, axis=-1)
    assert np.allclose(n, 1.0, atol=1e-12), f"|q| ranges {n.min()}..{n.max()}"


def test_rotation_is_what_separates_deviatoric_strain_from_d0():
    """A constant in eta is indistinguishable from d0, so no deviatoric tensor
    may be able to produce one -- but that only holds ACROSS rotations.

    At a single omega it fails, and instructively so: a deviatoric tensor
    diag(a, b, b) with a + 2b = 0 gives eps_qq = b + (a-b) p_x^2, and at
    omega = 0 the sample-frame p_x does not depend on eta, so eps_qq is
    constant and fully absorbed by d0. **A beam-axial uniaxial deviatoric
    strain is unmeasurable from one rotation.** Rotating the sample is what
    breaks it, which is the quantitative reason this technique needs a
    rotation and not merely more azimuthal bins.
    """
    eta = np.linspace(-180, 175, 36)

    def leakage(omegas):
        D = np.vstack([deviatoric_design(q_hat_sample_frame(2.13, eta, w))
                       for w in omegas])
        const = np.ones(len(D))
        coef, *_ = np.linalg.lstsq(D, const, rcond=None)
        return np.linalg.norm(D @ coef - const) / np.linalg.norm(const)

    one = leakage([0.0])
    many = leakage(np.linspace(0, 360, 24, endpoint=False))
    assert one < 0.05, (
        f"expected the single-rotation degeneracy to be near-total, got "
        f"residual {one:.3f}")
    assert many > 0.5, (
        f"across rotations a constant must NOT be representable, but the "
        f"design fits one to residual {many:.3f} -- the hydrostatic part has "
        f"leaked in and would be confounded with d0")


def test_rotation_carries_the_tensor_the_right_way():
    """eps_qq must be invariant when the tensor rotates with the sample.

    A tensor expressed in the sample frame, observed at any omega, gives the
    same eps_qq as the lab-frame tensor rotated by that omega. Getting the
    transpose backwards passes every smoothness check and rotates the answer.
    """
    eps_s = DeviatoricStrain(300e-6, -100e-6, 50e-6, -20e-6, 80e-6)
    E = eps_s.as_matrix()
    tt, eta = 2.13, np.linspace(-180, 175, 24)
    for om in (0.0, 37.0, 123.0, 271.0):
        q_s = q_hat_sample_frame(tt, eta, om)
        direct = np.einsum("ni,ij,nj->n", q_s, E, q_s)
        via_design = deviatoric_design(q_s) @ eps_s.as_vector()
        assert np.allclose(direct, via_design, atol=1e-15), (
            f"design row and the quadratic form disagree at omega={om}")


def test_strain_to_radius_round_trips():
    """Zero strain must land on the ring's own radius, and the sign must be
    right: tensile strain (larger d) means a SMALLER Bragg angle."""
    tt = _two_theta(D0_111)
    r0 = (LSD_UM / PX_UM) * np.tan(np.radians(tt))
    got = strain_to_radius(D0_111, 0.0, LAMBDA_A, LSD_UM, PX_UM)
    assert got == pytest.approx(r0, rel=1e-12)

    tensile = strain_to_radius(D0_111, 1e-3, LAMBDA_A, LSD_UM, PX_UM)
    assert tensile < r0, "tensile strain must move the ring inwards"
    # and the magnitude must match dR/R ~ -eps at small angle
    assert (r0 - tensile) / r0 == pytest.approx(1e-3, rel=0.02)


def test_torch_and_numpy_paths_agree():
    """The fit runs the torch branch; the tests above exercise numpy."""
    e = np.linspace(-2e-3, 2e-3, 11)
    a = strain_to_radius(D0_111, e, LAMBDA_A, LSD_UM, PX_UM)
    b = strain_to_radius(D0_111, torch.as_tensor(e), LAMBDA_A, LSD_UM,
                         PX_UM).numpy()
    assert np.allclose(a, b, atol=1e-9)


# --------------------------------------------------- the gating recovery test
def _synth(planted, *, size=8, n_omega=24, n_eta=12, n_r=21, sigma=1.6,
           noise=0.0, seed=0):
    """Render lineouts from a planted per-voxel deviatoric field.

    Deliberately NOT by calling the fitter's own loss: the patterns are built
    here from the same physics but independent code, so a sign convention that
    is wrong in both places cannot cancel.
    """
    rng = np.random.default_rng(seed)
    rings = [(D0_111, _two_theta(D0_111)), (D0_200, _two_theta(D0_200))]
    omega = np.linspace(0, 360, n_omega, endpoint=False)
    eta = -180.0 + (np.arange(n_eta) + 0.5) * (360.0 / n_eta)

    # A disc of sample in the middle of the grid.
    yy, xx = np.mgrid[0:size, 0:size]
    c = (size - 1) / 2.0
    inten = (np.hypot(yy - c, xx - c) <= size * 0.30).astype(float)
    inten_flat = inten.reshape(-1)

    # Projector, same convention as the fitter: t = x sin(th) + y cos(th).
    t_idx = np.arange(size)
    A = np.zeros((n_omega, size, size * size))
    for iw, om in enumerate(omega):
        th = np.radians(om)
        tt_ = (xx - c) * np.sin(th) + (yy - c) * np.cos(th) + c
        lo = np.floor(tt_).astype(int)
        fr = tt_ - lo
        for dlo, wgt in ((0, 1 - fr), (1, fr)):
            j = lo + dlo
            ok = (j >= 0) & (j < size)
            np.add.at(A[iw], (j[ok], np.flatnonzero(ok.reshape(-1))), wgt[ok])

    out, radii = [], []
    for d0, tt_deg in rings:
        r0 = (LSD_UM / PX_UM) * np.tan(np.radians(tt_deg))
        rr = r0 + np.linspace(-(n_r // 2), n_r // 2, n_r)
        radii.append(rr)
        q = q_hat_sample_frame(tt_deg, eta[None, :], omega[:, None])   # (w,e,3)
        D = deviatoric_design(q)                                        # (w,e,5)
        eqq = np.einsum("wec,vc->vwe", D, planted)                      # (V,w,e)
        cen = strain_to_radius(d0, eqq, LAMBDA_A, LSD_UM, PX_UM)
        dd = (rr[None, None, None, :] - cen[..., None]) / sigma
        prof = 0.5 / (dd * dd + 1.0) + 0.5 * np.exp(-0.5 * dd * dd)
        patt = prof * inten_flat[:, None, None, None]                   # (V,w,e,r)
        proj = np.einsum("wtv,vwer->wetr", A, patt)                     # (w,e,t,r)
        if noise:
            proj = proj + rng.normal(0.0, noise * proj.max(), proj.shape)
        out.append(proj)
    return (np.stack(out), np.stack(radii), omega, eta, inten,
            [r[0] for r in rings], [r[1] for r in rings])


def test_recovers_a_planted_deviatoric_field():
    """The gate from PREREGISTER §10 step 2.

    A uniform planted tensor, noiseless. If this fails nothing downstream is
    worth running -- and it must fail loudly rather than return a smooth map.
    """
    size = 8
    truth = DeviatoricStrain(400e-6, -250e-6, 120e-6, -90e-6, 200e-6)
    planted = np.tile(truth.as_vector(), (size * size, 1))

    li, radii, omega, eta, inten, d0s, tts = _synth(planted, size=size)
    res = fit_tensor_strain(
        li, radii_px=radii, rings_d0_a=d0s, two_theta_deg=tts,
        omega_deg=omega, wavelength_a=LAMBDA_A, lsd_um=LSD_UM, px_um=PX_UM,
        eta_deg=eta, intensity_map=inten, size=size, sigma_px=1.6,
        steps=2000, lr=0.05, mask_threshold=0.5, omega_chunk=12)

    got = np.median(res.strain, axis=0)
    err_ue = np.abs(got - truth.as_vector()) * 1e6
    msg = "\n".join(f"  {n}: planted {t*1e6:+8.1f}  got {g*1e6:+8.1f} ue"
                    for n, t, g in zip(COMPONENT_NAMES, truth.as_vector(), got))
    assert err_ue.max() < 5.0, (
        f"planted field not recovered (worst {err_ue.max():.1f} ue)\n{msg}")
    assert res.loss < 1e-8, (
        f"noiseless data must fit essentially exactly; loss {res.loss:.3e} "
        f"means the model and the data disagree about something")


def test_zero_strain_recovers_zero():
    """The null the real study runs. Must return zero on data with none."""
    size = 8
    planted = np.zeros((size * size, 5))
    li, radii, omega, eta, inten, d0s, tts = _synth(planted, size=size)
    res = fit_tensor_strain(
        li, radii_px=radii, rings_d0_a=d0s, two_theta_deg=tts,
        omega_deg=omega, wavelength_a=LAMBDA_A, lsd_um=LSD_UM, px_um=PX_UM,
        eta_deg=eta, intensity_map=inten, size=size, sigma_px=1.6,
        steps=800, lr=0.05, mask_threshold=0.5, omega_chunk=12)
    worst = np.abs(np.median(res.strain, axis=0)).max() * 1e6
    assert worst < 5.0, f"invented {worst:.1f} ue of strain from nothing"


def test_survives_an_amplitude_and_background_mismatch():
    """The failure that invalidated run 1 of the real null test.

    There, the per-voxel intensity was seeded from Branch B, whose units are
    integrated intensity while the data is counts -- the rendered model came out
    ~81x too bright. With no free amplitude the only parameter that could shed
    it was the width, which collapsed to 0.02 px against a 1.0 px bin, and the
    strain absorbed the rest. It printed a complete, wrong verdict.

    So: hand the fitter an intensity map on the WRONG SCALE and data sitting on
    a background it was not told about, and require the strain back anyway.
    """
    size = 8
    truth = DeviatoricStrain(400e-6, -250e-6, 120e-6, -90e-6, 200e-6)
    planted = np.tile(truth.as_vector(), (size * size, 1))
    li, radii, omega, eta, inten, d0s, tts = _synth(planted, size=size, sigma=1.7)

    li = li + 0.35 * li.max()                 # a background it must discover
    wrong_scale = inten * 83.0                # the run-1 mismatch, near enough

    res = fit_tensor_strain(
        li, radii_px=radii, rings_d0_a=d0s, two_theta_deg=tts,
        omega_deg=omega, wavelength_a=LAMBDA_A, lsd_um=LSD_UM, px_um=PX_UM,
        eta_deg=eta, intensity_map=wrong_scale, size=size, sigma_px="fit",
        steps=3000, lr=0.05, mask_threshold=0.5, omega_chunk=12)

    assert res.valid, f"the run should be interpretable: {res.why_invalid()}"
    assert np.allclose(res.sigma_px, 1.7, atol=0.15), (
        f"width ran away under a scale mismatch: {res.sigma_px}")
    err = np.abs(np.median(res.strain, axis=0) - truth.as_vector()) * 1e6
    assert err.max() < 40.0, (
        f"strain not recovered through the mismatch (worst {err.max():.1f} ue)")


def test_a_collapsed_width_is_reported_invalid():
    """`valid` is the guard run 1 did not have. It must catch what happened."""
    from midas_dt.tensor_strain import TensorResult
    bad = TensorResult(strain=np.zeros((4, 5)), d_mean_a=np.ones(4),
                       intensity=np.ones(4), active=np.ones(4, bool), size=2,
                       loss=1.0, n_steps=1, converged=True, residual_rel=11.4,
                       sigma_px=np.full(9, 0.02), r_bin_px=1.0)
    assert not bad.valid
    assert "sigma" in bad.why_invalid() and "residual" in bad.why_invalid()


def test_fitted_width_recovers_the_rendered_width_and_the_strain():
    """The fix for section 9a: with sigma free, neither has to be known.

    Renders at a width the fitter is not told, and requires BOTH the strain and
    the width back. This is what makes a real run interpretable at the 24 ue
    line, since nothing else pins the width on real data.
    """
    size = 8
    truth = DeviatoricStrain(400e-6, -250e-6, 120e-6, -90e-6, 200e-6)
    planted = np.tile(truth.as_vector(), (size * size, 1))
    li, radii, omega, eta, inten, d0s, tts = _synth(planted, size=size, sigma=1.85)

    res = fit_tensor_strain(
        li, radii_px=radii, rings_d0_a=d0s, two_theta_deg=tts,
        omega_deg=omega, wavelength_a=LAMBDA_A, lsd_um=LSD_UM, px_um=PX_UM,
        eta_deg=eta, intensity_map=inten, size=size, sigma_px="fit",
        steps=2500, lr=0.05, mask_threshold=0.5, omega_chunk=12)

    assert np.allclose(res.sigma_px, 1.85, atol=0.05), (
        f"width not recovered: {res.sigma_px} against a rendered 1.85 px")
    err = np.abs(np.median(res.strain, axis=0) - truth.as_vector()) * 1e6
    assert err.max() < 8.0, (
        f"strain not recovered with the width free (worst {err.max():.1f} ue)")


def test_a_wrong_peak_width_manufactures_strain():
    """**The assumed width is not a free choice.** Measured here: rendering at
    sigma = 1.6 px and fitting at 1.5 -- a 6% error -- produces up to 49 ue of
    strain that is not in the data, comparable to the whole instrumental floor
    on the 11-ID-C scan (36 ue).

    The width must therefore be measured or fitted, never assumed, and this
    test exists so that requirement cannot be quietly forgotten. It asserts the
    failure, not the success: if a future change makes the fit insensitive to
    the width, this test fails and the caveat can be dropped deliberately.
    """
    size = 8
    truth = DeviatoricStrain(400e-6, -250e-6, 120e-6, -90e-6, 200e-6)
    planted = np.tile(truth.as_vector(), (size * size, 1))
    li, radii, omega, eta, inten, d0s, tts = _synth(planted, size=size, sigma=1.6)

    kw = dict(radii_px=radii, rings_d0_a=d0s, two_theta_deg=tts,
              omega_deg=omega, wavelength_a=LAMBDA_A, lsd_um=LSD_UM,
              px_um=PX_UM, eta_deg=eta, intensity_map=inten, size=size,
              steps=2000, lr=0.05, mask_threshold=0.5, omega_chunk=12)
    right = fit_tensor_strain(li, sigma_px=1.6, **kw)
    wrong = fit_tensor_strain(li, sigma_px=1.5, **kw)

    e_right = np.abs(np.median(right.strain, axis=0) - truth.as_vector()).max()
    e_wrong = np.abs(np.median(wrong.strain, axis=0) - truth.as_vector()).max()
    assert e_right * 1e6 < 5.0, "the matched-width fit should be near-exact"
    assert e_wrong * 1e6 > 20.0, (
        f"a 6% width error produced only {e_wrong*1e6:.1f} ue of bias; if the "
        f"fit has become width-insensitive, drop this caveat deliberately")
