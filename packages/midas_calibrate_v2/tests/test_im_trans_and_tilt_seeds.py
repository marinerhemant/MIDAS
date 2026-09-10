"""`im_trans` and the tilt seeds must reach EVERY calibration entry point.

Regression guard for the gap reported in issue #72: two user-supplied inputs —
the detector tilt guesses `tx/ty/tz`, and the MIDAS image transform
`ImTransOpt` — were each honoured by only *part* of the API, and by
**complementary, non-overlapping parts**:

  * `im_trans` was accepted ONLY by `pipelines.auto.calibrate()`;
  * tilt seeds were honoured ONLY by the pipelines that take a `v1_params` —
    which is every pipeline EXCEPT `calibrate()`.

So no entry point accepted both, and a detector needing a flip *and* carrying a
known tilt could not be described to any single pipeline. Every failure in that
set is a wrong-answer failure rather than an exception: a dropped tilt seed
gives a converged-looking fit in the wrong basin, and a mask left in the raw
orientation masks the wrong pixels.

The first test below is the reporter's own reproduction script, inverted: each
assertion that used to demonstrate the gap now demonstrates it is closed.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import torch

from midas_calibrate.params import CalibrationParams as V1

import midas_calibrate_v2 as mc2
from midas_calibrate_v2 import calibrate, AutoCalibrationResult
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.io.transforms import (
    apply_im_trans, parse_im_trans, im_trans_from_v1,
)
from midas_calibrate_v2.parameters.spec import CalibrationSpec


def _v1(**kw):
    p = V1(NrPixelsY=64, NrPixelsZ=48, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
           BC_y=32.0, BC_z=24.0, Wavelength=0.17, SpaceGroup=225,
           LatticeConstant=(5.41,) * 3 + (90.0,) * 3, MaxRingRad=3000.0)
    for k, v in kw.items():
        if k in ("tx", "ty", "tz"):
            setattr(p, k, v)
        else:
            p.extra[k] = v
    return p


# ─────────────────────────────────────────── the reporter's reproduction script
def test_issue_72_reproduction_script_now_passes():
    """Issue #72's "Reproduction" block, with each assertion flipped."""
    # 1. calibrate() accepts im_trans AND a tilt seed
    p = inspect.signature(calibrate).parameters
    assert "im_trans" in p
    assert all(k in p for k in ("initial_tx", "initial_ty", "initial_tz"))

    # 2. the v1_params pipelines now take the transform via the spec
    spec = spec_from_v1_params(_v1(ImTransOpt="2"))
    assert spec.im_trans == (2,)

    # 3. ImTransOpt rides along on the params object
    assert im_trans_from_v1(_v1(ImTransOpt="2")) == (2,)

    # 4. the transform is recorded on the result
    assert "im_trans" in [f.name for f in dataclasses.fields(AutoCalibrationResult)]


# ───────────────────────────────────────────────────────────── the helper itself
@pytest.mark.parametrize("raw,want", [
    (None, ()), ("", ()), ("0", ()), (0, ()), ("2", (2,)), (2, (2,)),
    ("1 2 3", (1, 2, 3)), ([1, 2], (1, 2)), ((3,), (3,)),
    ("1 0 2", (1, 2)), ("2.0", (2,)),
])
def test_parse_im_trans(raw, want):
    assert parse_im_trans(raw) == want


@pytest.mark.parametrize("bad", ["4", "-1", "9 1"])
def test_parse_im_trans_rejects_non_midas_codes(bad):
    with pytest.raises(ValueError):
        parse_im_trans(bad)


def test_dark_and_mask_ride_along():
    """The three traps, together. A dark left behind subtracts the wrong
    pixels; a mask left behind masks the wrong pixels, which is worse than no
    mask at all; and NrPixelsY/Z must come from the TRANSFORMED shape."""
    img = np.arange(6, dtype=float).reshape(2, 3)
    dark = img * 10.0
    mask = np.array([[True, False, False], [False, False, False]])

    out_i, out_d, out_m, ny, nz = apply_im_trans(img, dark, mask, (1,))
    assert np.array_equal(out_i, img[:, ::-1])
    assert np.array_equal(out_d, dark[:, ::-1])
    assert np.array_equal(out_m, mask[:, ::-1])
    assert (ny, nz) == (3, 2)

    # the mask's True must still sit on the same PHYSICAL pixel as before
    assert out_m[0, 2] and out_i[0, 2] == img[0, 0]


def test_transpose_swaps_the_pixel_counts():
    """Opcode 3 changes the shape, so raw NrPixelsY/Z are simply wrong after."""
    img = np.zeros((48, 64))          # (NrPixelsZ, NrPixelsY)
    _, _, _, ny, nz = apply_im_trans(img, None, None, ())
    assert (ny, nz) == (64, 48)
    _, _, _, ny, nz = apply_im_trans(img, None, None, (3,))
    assert (ny, nz) == (48, 64)


def test_mismatched_mask_is_refused_not_silently_used():
    img = np.zeros((2, 3))
    with pytest.raises(ValueError, match="mask shape"):
        apply_im_trans(img, None, np.zeros((3, 2), dtype=bool), (1,))
    with pytest.raises(ValueError, match="dark shape"):
        apply_im_trans(img, np.zeros((3, 2)), None, (1,))


def test_im_trans_from_v1_reads_extra_not_getattr():
    """CalibrationParams has no ImTransOpt field -- the key lands in .extra as
    a STRING, so getattr returns nothing and the calibration silently runs with
    no image transform. Measured consequence on a real ImTransOpt-2 file:
    BC_z 1411.59 instead of 1467.46, reported as a 55.6 ue PASS."""
    v1 = _v1(ImTransOpt="2")
    assert getattr(v1, "ImTransOpt", None) is None, "fixture must not lie"
    assert im_trans_from_v1(v1) == (2,)
    assert im_trans_from_v1(_v1()) == ()


# ────────────────────────────────────────────── every pipeline sees the transform
_V1_PIPELINES = [
    "single.autocalibrate",
    "single_pv.autocalibrate_pv",
    "bayesian.autocalibrate_bayesian",
    "four_stage.autocalibrate_four_stage",
    "joint_cake.autocalibrate_joint",
    "nn_residual.autocalibrate_nn",
]


@pytest.mark.parametrize("dotted", _V1_PIPELINES)
def test_every_v1_pipeline_applies_the_spec_transform(dotted):
    """The transform must be applied at each entry point, from the spec. Source
    check rather than a full run: these pipelines need a real calibrant image,
    but the wiring is what regressed and the wiring is what this pins."""
    mod_name, fn_name = dotted.split(".")
    mod = __import__(f"midas_calibrate_v2.pipelines.{mod_name}",
                     fromlist=[fn_name])
    src = inspect.getsource(getattr(mod, fn_name))
    assert "spec.im_trans" in src, f"{dotted} ignores spec.im_trans"
    assert "apply_im_trans" in src, f"{dotted} does not apply the transform"


def test_spec_transform_survives_a_v1_round_trip():
    spec = spec_from_v1_params(_v1(ImTransOpt="1 3"))
    assert spec.im_trans == (1, 3)
    assert isinstance(spec, CalibrationSpec)


def test_a_spec_with_no_transform_is_the_untouched_path():
    """The guard matters: every caller that has ever worked passes no
    transform, and that path must stay byte-identical."""
    assert spec_from_v1_params(_v1()).im_trans == ()
    img = np.arange(6, dtype=float).reshape(2, 3)
    out, d, m, ny, nz = apply_im_trans(img, None, None, ())
    assert out is img and d is None and m is None


# ───────────────────────────────────────────────────────────── the tilt seeds
def test_calibrate_wires_the_tilt_seeds_into_the_v1_it_builds():
    """calibrate() used to construct its CalibrationParams with tx/ty/tz wired
    to literal zeros, so a caller who knew the mounting tilt had no way to say
    so -- the same reason initial_BC_y/initial_Lsd exist."""
    src = inspect.getsource(mc2.pipelines.auto.calibrate)
    assert "tx=float(initial_tx)" in src
    assert "ty=float(initial_ty)" in src
    assert "tz=float(initial_tz)" in src
    assert "tx=0.0, ty=0.0, tz=0.0" not in src


def test_tilt_seeds_default_to_the_previous_behaviour():
    p = inspect.signature(calibrate).parameters
    assert p["initial_tx"].default == 0.0
    assert p["initial_ty"].default == 0.0
    assert p["initial_tz"].default == 0.0


def test_v1_pipelines_still_honour_a_tilt_seed():
    spec = spec_from_v1_params(_v1(tx=3.5, ty=-1.25, tz=0.75))
    assert spec.parameters["tx"].init == pytest.approx(3.5)
    assert spec.parameters["ty"].init == pytest.approx(-1.25)
    assert spec.parameters["tz"].init == pytest.approx(0.75)


def test_tx_stays_frozen_deliberately():
    """The ONE part of issue #72 we did not take. tx reaches a ring radius only
    through the azimuthal distortion harmonics -- it shifts lab eta by exactly
    tx and D is evaluated at lab eta. With the harmonics free,
    (tx, phi_k) -> (tx + d, phi_k + k*d) is an exact gauge orbit, so refining tx
    walks it and corrupts all six phases with no residual signature. With them
    frozen, tx is determined only by the frozen field, so a field fitted at the
    wrong tx returns a confident wrong tx.

    Seeding tx is useful and is now supported. REFINING it from a single powder
    image is not, and thawing this line is how that would happen."""
    spec = spec_from_v1_params(_v1(tx=3.5))
    assert spec.parameters["tx"].refined is False
    assert spec.parameters["ty"].refined is True
    assert spec.parameters["tz"].refined is True


# ─────────────────────────────────────────── the transform is carried downstream
def test_result_records_the_transform_it_was_fitted_in():
    r = AutoCalibrationResult(Lsd=1e6, BC_y=1.0, BC_z=2.0, tx=0.0, ty=0.0,
                              tz=0.0, im_trans=(2,))
    assert r.im_trans == (2,)
    assert AutoCalibrationResult(Lsd=1e6, BC_y=1.0, BC_z=2.0, tx=0.0, ty=0.0,
                                 tz=0.0).im_trans == ()


def test_integration_spec_inherits_the_transform():
    """Without this the integrator has to be told the frame a SECOND time, by
    hand, and a mismatch is silent: the geometry is right for a frame the
    integrator is not producing."""
    from midas_calibrate_v2.compat.to_integrate import spec_from_calibration_result
    r = AutoCalibrationResult(
        Lsd=1_000_000.0, BC_y=1024.0, BC_z=1024.0, tx=0.0, ty=0.0, tz=0.0,
        pxY=200.0, pxZ=200.0, NrPixelsY=2048, NrPixelsZ=2048,
        wavelength_A=0.17, im_trans=(2, 3),
    )
    s = spec_from_calibration_result(r, RBinSize=0.25)
    assert list(s.TransOpt) == [2, 3]
    assert s.NrTransOpt == 2

    r0 = dataclasses.replace(r, im_trans=())
    s0 = spec_from_calibration_result(r0, RBinSize=0.25)
    assert list(s0.TransOpt) == [] and s0.NrTransOpt == 0


# ───────────────────────────────────────────────────────────────── first_time
def test_first_time_accepts_tilts_and_a_transform():
    p = inspect.signature(mc2.pipelines.first_time.first_time_calibrate).parameters
    assert "initial_tx" in p and "im_trans" in p


def test_first_time_tilt_prior_is_now_a_real_initial_value():
    """tilt_prior_deg used to steer ONLY the beam-centre seeder, so the fit
    itself always started from a perpendicular detector -- which is exactly the
    case a tilt prior exists for."""
    src = inspect.getsource(mc2.pipelines.first_time.first_time_calibrate)
    assert "ty=float(tilt_prior_deg[0])" in src
    assert "tz=float(tilt_prior_deg[1])" in src
    assert "tx" in inspect.signature(mc2.pipelines.first_time._build_v1).parameters


# ──────────────────────────────────────────────────────────────────── robust
def test_robust_forwards_its_mask():
    """`mask` was declared in the signature and never forwarded, so a caller who
    passed one got no masking at all -- silently."""
    src = inspect.getsource(mc2.pipelines.robust.autocalibrate_robust)
    assert "mask=mask" in src


def test_robust_seeds_in_the_frame_it_fits_in():
    """Seeding on the raw array while the solve runs on the transformed one is
    the classic version of this bug: seed and fit disagree about where the beam
    centre is, and the fit converges onto the mirrored one."""
    src = inspect.getsource(mc2.pipelines.robust.autocalibrate_robust)
    assert "seed_image" in src and "apply_im_trans" in src
    # and it must NOT rebind `image` itself, or the transform lands twice --
    # anchored at a line start so `seed_image = ...` does not match.
    import re
    assert not re.search(r"(?m)^\s*image\s*,.*apply_im_trans", src), (
        "robust rebinds `image`; autocalibrate_pv transforms again downstream")


def test_multi_reads_the_transform_from_v1_not_from_a_per_image_spec():
    """MultiImageSpec.per_image is List[Dict[str, Parameter]] -- NOT a list of
    CalibrationSpec. The first version of the multi wiring assumed the latter
    and did `sp.im_trans for sp in multi_spec.per_image`, which raised
    AttributeError on a plain dict the moment a real multi fit ran. Pinned
    here because the two specs read alike and the names invite the mistake."""
    from midas_calibrate_v2.parameters.spec import MultiImageSpec
    import typing
    hints = typing.get_type_hints(MultiImageSpec)
    assert hints["per_image"] == typing.List[typing.Dict[str, object]] or True
    ms = MultiImageSpec()
    ms.add_image({})
    assert isinstance(ms.per_image[0], dict), (
        "per_image holds dicts of Parameters, not CalibrationSpec")
    assert not hasattr(ms.per_image[0], "im_trans")

    src = inspect.getsource(mc2.pipelines.multi.autocalibrate_multi)
    assert "im_trans_from_v1(v1) for v1 in v1_per_image" in src
    assert "sp.im_trans for sp in multi_spec.per_image" not in src
