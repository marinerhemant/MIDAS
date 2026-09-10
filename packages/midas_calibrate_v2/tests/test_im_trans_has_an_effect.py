"""`im_trans` must reach the FIT, not merely be stored on the spec.

The tests in `test_im_trans_and_tilt_seeds.py` check that the transform is
wired through each entry point by reading the source. That catches a pipeline
that never calls the helper, but it cannot catch a value that is plumbed and
then has no effect -- which is the failure mode that matters, and the one that
has bitten this repo before (`spec.fix_panel_id == 28` asserted true the whole
time that value was failing to reach the forward model).

So this runs a real calibration, twice, on an image that has genuinely been
flipped, and asserts the arm WITH the transform recovers the truth while the
arm WITHOUT it does not. The second half is the part that makes the first half
mean something: if both arms recovered the truth, the transform would be doing
nothing and the test would still be green without it.
"""
from __future__ import annotations

import numpy as np
import pytest

from midas_integrate.geometry import build_tilt_matrix, pixel_to_REta

from midas_calibrate.params import CalibrationParams
from midas_calibrate.rings import build_ring_table

from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.pipelines.single import autocalibrate

# The beam centre must be OFF the array centre, or a flip is a no-op and the
# test cannot fail.
TRUE_BC_Y, TRUE_BC_Z = 512.0, 400.0
NPY = NPZ = 1024


def _truth() -> CalibrationParams:
    p = CalibrationParams()
    p.NrPixelsY = NPY; p.NrPixelsZ = NPZ
    p.pxY = 200.0; p.pxZ = 200.0
    p.Lsd = 1_000_000.0
    p.BC_y = TRUE_BC_Y; p.BC_z = TRUE_BC_Z
    p.tx = 0.0; p.ty = 0.0; p.tz = 0.0
    p.Wavelength = 0.173
    p.SpaceGroup = 225
    p.LatticeConstant = (5.411, 5.411, 5.411, 90.0, 90.0, 90.0)
    p.MaxRingRad = 380.0
    p.MinRingRad = 0.0
    p.RhoD = 512.0 * 200.0
    p.Width = 1500.0
    p.EtaBinSize = 10.0
    p.RBinSize = 1.0
    p.nIterations = 3
    p.RemoveOutliersBetweenIters = False
    p.SNRMin = 1.5
    p.tolLsd = 5000.0; p.tolBC = 20.0; p.tolTilts = 1.0
    p.tolDistortion = 0.0
    p.Refine = {"Lsd": True, "BC": True, "ty": True, "tz": True,
                "Wavelength": False, "Parallax": False,
                **{f"p{i}": False for i in range(15)}}
    return p


def _simulate(params: CalibrationParams, sigma_px: float = 1.5) -> np.ndarray:
    """Paint concentric rings in the MODEL frame."""
    rt = build_ring_table(params)
    px = 0.5 * (params.pxY + params.pxZ)
    TRs = build_tilt_matrix(params.tx, params.ty, params.tz)
    Y, Z = np.meshgrid(np.arange(params.NrPixelsY, dtype=np.float64),
                       np.arange(params.NrPixelsZ, dtype=np.float64))
    R, _ = pixel_to_REta(Y, Z, Ycen=params.BC_y, Zcen=params.BC_z, TRs=TRs,
                         Lsd=params.Lsd, RhoD=params.RhoD, px=px,
                         parallax=params.Parallax)
    rng = np.random.default_rng(0)
    img = np.full(R.shape, 50.0) + rng.normal(0, 5.0, size=R.shape)
    for r in rt.r_ideal_px:
        img += (1000.0 / (1.0 + r / 100.0)) * np.exp(
            -0.5 * ((R - r) / sigma_px) ** 2)
    return img


@pytest.fixture(scope="module")
def frames():
    """(model-frame image, 'raw file' image). ImTransOpt 2 = flip Z, so a raw
    file needing that transform is the model frame flipped in Z."""
    model = _simulate(_truth())
    raw = model[::-1, :].copy()          # what the detector wrote to disk
    return model, raw


def test_the_flip_actually_moves_the_pattern(frames):
    """Guard the fixture: if the flip were a no-op the whole test is vacuous."""
    model, raw = frames
    assert not np.allclose(model, raw)
    # the pattern centre moves from Z=400 to Z=1023-400=623
    assert abs((NPZ - 1 - TRUE_BC_Z) - 623.0) < 1e-9


#: Seed both arms AWAY from the truth. Seeding at the truth makes the null
#: vacuous: an arm with no matching rings has no signal and simply sits near
#: its seed, so "stayed put" and "recovered the truth" are indistinguishable.
#: Measured, seeding at truth: the no-transform arm returned BC_z = 398.78,
#: 1.2 px from a truth of 400, purely by not moving.
SEED_OFFSET_Z = 12.0


def _run(image, im_trans):
    v1 = _truth()
    v1.BC_z = TRUE_BC_Z + SEED_OFFSET_Z
    spec = spec_from_v1_params(v1)
    spec.im_trans = tuple(im_trans)
    res = autocalibrate(v1, image, spec=spec, verbose=False)
    u = res.unpacked
    strain = min((h.mean_strain_uE for h in res.history), default=float("inf"))
    return float(u["BC_y"]), float(u["BC_z"]), strain


def test_transform_arm_recovers_the_truth(frames):
    """Hand the pipeline the RAW file plus ImTransOpt 2 and it should land on
    the geometry the rings were painted at."""
    _, raw = frames
    bc_y, bc_z, strain = _run(raw, (2,))
    assert abs(bc_y - TRUE_BC_Y) < 3.0, f"BC_y {bc_y} vs {TRUE_BC_Y}"
    assert abs(bc_z - TRUE_BC_Z) < 3.0, f"BC_z {bc_z} vs {TRUE_BC_Z}"
    # and it must have MOVED there from the offset seed, not started there
    assert abs(bc_z - (TRUE_BC_Z + SEED_OFFSET_Z)) > 5.0, (
        "BC_z never left its seed; the fit is not being driven by the rings")


def test_without_the_transform_it_does_not(frames):
    """The null. Same raw file, transform withheld: the fit is seeded at
    Z=400 while the pattern sits at Z=623, so it cannot recover the truth.
    If this ever passes, the transform is not reaching the fit and the test
    above is measuring nothing."""
    _, raw = frames
    bc_y, bc_z, strain = _run(raw, ())
    assert abs(bc_z - TRUE_BC_Z) > 5.0, (
        f"BC_z came back at {bc_z}, within 5 px of truth WITHOUT the "
        "transform -- the transform arm proves nothing")


def test_transform_arm_beats_the_no_transform_arm(frames):
    """Stated as a comparison so the effect size is visible, not just a
    threshold either side of an arbitrary line."""
    _, raw = frames
    _, bc_z_on, strain_on = _run(raw, (2,))
    _, bc_z_off, strain_off = _run(raw, ())
    err_on = abs(bc_z_on - TRUE_BC_Z)
    err_off = abs(bc_z_off - TRUE_BC_Z)
    assert err_on < err_off / 3.0, (
        f"BC_z error: with transform {err_on:.2f} px, without {err_off:.2f} px")
    # The residual is the independent discriminator: it says the rings match
    # the model, without reference to a truth value the seed already knows.
    assert strain_on < strain_off / 3.0, (
        f"strain: with transform {strain_on:.1f} ue, without {strain_off:.1f} ue")
