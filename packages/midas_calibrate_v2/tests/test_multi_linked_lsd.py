"""Linked-distance multi-image calibration: Lsd_i = L0 + Delta_i.

The scientific content of this feature is a NEGATIVE as much as a positive:

  * with the travel Delta_i held exact, a shared Wavelength is well
    determined;
  * with a free Lsd per image it is NOT, however many distances you stack,
    because (lambda -> k*lambda, Lsd_i -> k*Lsd_i) leaves every predicted
    ring radius unchanged to first order.

Both are asserted here.  The second is the one that matters for maintenance:
the L0+Delta constraint looks redundant next to "we already have lots of
distances", and removing it silently destroys the measurement rather than
breaking anything loudly.

A note on how the negative is tested.  On NOISELESS synthetic data the
(lambda, Lsd) degeneracy is not exact -- the tan/arcsin nonlinearity does
single out the true wavelength -- so a good enough optimiser recovers lambda
either way and a point-estimate comparison would prove nothing.  The
degeneracy is an ILL-CONDITIONING statement, so it is tested the way it
actually bites: repeat the fit over noise realisations and compare the
SPREAD of the recovered wavelength.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
least_squares = pytest.importorskip("scipy.optimize").least_squares

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table
from midas_calibrate_v2.loss.pseudo_strain import pseudo_strain_residual


# ----------------------------------------------------------------- fixtures
LAM_TRUE = 0.11595          # A
A_CEO2 = 5.4116             # A
PX = 150.0                  # um
NPIX = 2880
L0_TRUE = -1040.0           # um; true Lsd = motor readback + L0
MOTORS_UM = np.arange(600, 2401, 200, dtype=float) * 1000.0
SIGMA_PX = 0.05             # per-point radial precision
N_SEEDS = 4


def _params(lsd_um, lam, *, max_ring_rad=1400.0):
    return CalibrationParams(
        NrPixelsY=NPIX, NrPixelsZ=NPIX, pxY=PX, pxZ=PX,
        Lsd=float(lsd_um), BC_y=NPIX / 2, BC_z=NPIX / 2,
        tx=0.0, ty=0.0, tz=0.0,
        Wavelength=float(lam), SpaceGroup=225,
        LatticeConstant=(A_CEO2,) * 3 + (90.0,) * 3,
        MaxRingRad=max_ring_rad, MinRingRad=200.0,
        RhoD=NPIX / 2 * PX, nIterations=1,
    )


def _synth_points(lsd_um, lam, rng, n_eta=48):
    """Ring points for a perfect detector at (lsd, lam), plus radial noise.

    No tilt, no distortion: exactly the geometry the (lambda, Lsd) gauge is
    degenerate for at small angle, which is the point of the exercise.
    """
    rt = build_ring_table(_params(lsd_um, lam))
    assert len(rt) >= 4, f"only {len(rt)} rings at Lsd={lsd_um / 1e3:.0f} mm"
    eta = np.linspace(-np.pi, np.pi, n_eta, endpoint=False)
    Y, Z, tt, dsp = [], [], [], []
    for R, t2, d in zip(rt.r_ideal_px, rt.two_theta_deg, rt.d_spacing):
        Rn = R + rng.normal(0.0, SIGMA_PX, n_eta)
        Y.append(NPIX / 2 + Rn * np.cos(eta))
        Z.append(NPIX / 2 + Rn * np.sin(eta))
        tt.append(np.full(n_eta, t2))
        dsp.append(np.full(n_eta, d))
    t = lambda a: torch.as_tensor(np.concatenate(a), dtype=torch.float64)
    return t(Y), t(Z), t(tt), t(dsp)


def _dataset(seed):
    rng = np.random.default_rng(seed)
    return [_synth_points(m + L0_TRUE, LAM_TRUE, rng) for m in MOTORS_UM]


def _resid(pts, lsd_um, lam):
    Y, Z, tt, dsp = pts
    f = lambda v: torch.as_tensor(float(v), dtype=torch.float64)
    p = {"Lsd": f(lsd_um), "BC_y": f(NPIX / 2), "BC_z": f(NPIX / 2),
         "tx": f(0.0), "ty": f(0.0), "tz": f(0.0),
         "pxY": f(PX), "pxZ": f(PX), "Parallax": f(0.0), "Wavelength": f(lam)}
    with torch.no_grad():
        r = pseudo_strain_residual(Y, Z, tt, p, rho_d=f(NPIX / 2 * PX),
                                   ring_d_spacing_A=dsp)
    return r.numpy()


def _fit_linked(data, deltas):
    """Two unknowns for the whole scan: L0 and lambda."""
    def fun(x):
        return np.concatenate([_resid(pts, x[0] * 1e3 + d, x[1] * 1e-3)
                               for pts, d in zip(data, deltas)])
    x0 = [(MOTORS_UM.mean() + 5000.0) / 1e3, LAM_TRUE * 1.004 * 1e3]
    s = least_squares(fun, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return s.x[0] * 1e3 - MOTORS_UM.mean(), s.x[1] * 1e-3, s


def _fit_free(data):
    """One free Lsd per image, plus a shared lambda."""
    n = len(data)

    def fun(x):
        return np.concatenate([_resid(pts, x[i] * 1e3, x[n] * 1e-3)
                               for i, pts in enumerate(data)])
    x0 = [(m + L0_TRUE) / 1e3 for m in MOTORS_UM] + [LAM_TRUE * 1.004 * 1e3]
    s = least_squares(fun, x0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return s.x[n] * 1e-3, s


# ------------------------------------------------------------------- tests
def test_linked_distance_recovers_wavelength_and_offset():
    deltas = MOTORS_UM - MOTORS_UM.mean()
    L0, lam, sol = _fit_linked(_dataset(0), deltas)
    assert sol.success
    assert abs(lam - LAM_TRUE) / LAM_TRUE < 5e-5, (
        f"wavelength not recovered: {lam:.7f} vs {LAM_TRUE}")
    assert abs(L0 - L0_TRUE) < 25.0, (
        f"offset not recovered: {L0:.1f} vs {L0_TRUE} um")


def test_free_per_image_lsd_is_far_worse_conditioned():
    """The negative control, stated as conditioning rather than bias.

    Same data, same optimiser.  Only the parameterisation differs.  If this
    ever stops holding -- if free-Lsd becomes as tight as linked -- then the
    argument motivating the linked mode needs revisiting.
    """
    deltas = MOTORS_UM - MOTORS_UM.mean()
    lam_linked, lam_free = [], []
    for s in range(N_SEEDS):
        data = _dataset(s)
        lam_linked.append(_fit_linked(data, deltas)[1])
        lam_free.append(_fit_free(data)[0])

    spread_linked = float(np.std(lam_linked)) / LAM_TRUE
    spread_free = float(np.std(lam_free)) / LAM_TRUE
    assert spread_free > 10.0 * spread_linked, (
        f"free-Lsd sigma(lambda)/lambda = {spread_free:.2e} is not much worse "
        f"than linked {spread_linked:.2e}; the degeneracy argument fails")

    # and the Jacobian conditioning says the same thing, deterministically
    _, _, sol_l = _fit_linked(_dataset(0), deltas)
    _, sol_f = _fit_free(_dataset(0))
    cond_l = np.linalg.cond(sol_l.jac.T @ sol_l.jac)
    cond_f = np.linalg.cond(sol_f.jac.T @ sol_f.jac)
    assert cond_f > 100.0 * cond_l, (
        f"cond(J'J): free {cond_f:.2e} vs linked {cond_l:.2e}")


def test_lsd_offsets_length_is_validated():
    from midas_calibrate_v2.pipelines.multi import autocalibrate_multi
    v1 = _params(1_000_000.0, LAM_TRUE)
    img = np.zeros((NPIX, NPIX), dtype=np.float64)
    with pytest.raises(ValueError, match="lsd_offsets_um"):
        autocalibrate_multi([v1, v1], [img, img], lsd_offsets_um=[0.0],
                            n_iter=1, verbose=False)
