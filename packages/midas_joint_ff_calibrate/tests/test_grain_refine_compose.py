"""The corrected paramstest must COMPOSE relative scalars, not overwrite them —
and must carry the distortion phases into the frame the new ``tx`` describes.

Regression guard. ``grain-tx`` refines ``tx``/``Wedge`` as CORRECTIONS on top of
the geometry the input reconstruction already used:

  * the forward model is built with ``tx=0.0`` and ``wedge=0.0``
    (``_build_model``), while every absolute scalar (Lsd, ty, tz, BC) is seeded
    from ``v1``;
  * the trial ``tx`` is applied by ROTATING the stored SpotMatrix YLab/ZLab,
    which already carry whatever ``tx`` the pipeline ran with;
  * ``Wedge`` is injected into the forward model while the observed omega
    already carries the pipeline's wedge correction.

So the fitted number is the RESIDUAL. The output paramstest drives a FRESH
reconstruction, so it must carry the TOTAL. Overwriting it silently discards the
previous pass — a second iteration then applies less correction than the first,
with no error and no log line.

Measured on 20-ID Au (5 grains, MinNrPx-4 spot list):
    pass 1 on a tx=0 recon      -> -0.158497
    pass 2 on the -0.1585 recon -> -0.087265   (the residual)
    composed                    -> -0.245762
against -0.2455 from an independent ring/eta systematics fit.

EXCEPTION: when a ``_NEEDS_RAW``/``_DISTORTION`` scalar is thawed the residual
switches to the raw-pixel path, where ``tx`` is applied inside the detector
correction and is therefore ABSOLUTE. ``Wedge`` stays relative in both paths.

THESE TESTS DRIVE THE REAL WRITER (``gr.corrected_paramstest_text``) AGAINST A
REAL ``CalibrationParams``. An earlier version reimplemented the writer here and
used a hand-rolled stand-in for ``v1`` that had ``Wedge`` as a real attribute —
which ``CalibrationParams`` does not. Both the reimplementation and the fake
agreed with each other and with nothing else, and the dropped-``Wedge`` bug
shipped green. Do not reintroduce either.
"""

from __future__ import annotations

import pytest

from midas_calibrate.params import CalibrationParams
from midas_joint_ff_calibrate import grain_refine as gr


def _v1(*, tx=0.0, Wedge=None, Lsd=1_000_000.0, ty=0.0, tz=0.0, **extra):
    """A REAL CalibrationParams, with Wedge where the parser actually puts it.

    ``Wedge`` is not a declared field, so a v1 paramstest parse lands it in
    ``.extra`` as a string. Passing it any other way makes the fixture lie.
    """
    p = CalibrationParams(
        NrPixelsY=2048, NrPixelsZ=2048, pxY=200.0, pxZ=200.0,
        Lsd=Lsd, BC_y=1024.0, BC_z=1024.0, Wavelength=0.17, SpaceGroup=225,
        LatticeConstant=(5.41,) * 3 + (90.0,) * 3, MaxRingRad=200000.0,
    )
    p.tx, p.ty, p.tz = tx, ty, tz
    if Wedge is not None:
        p.extra["Wedge"] = str(Wedge)      # a STRING, as the parser leaves it
    for k, v in extra.items():
        p.extra[k] = v
    return p


def _emit(v1, refine_params, fitted, *, observed_from_raw=False, src=None):
    """Run the real writer and return the resulting key -> float mapping."""
    if src is None:
        src = "".join(f"{k} {getattr(v1, k)}\n" for k in ("tx", "ty", "tz", "Lsd"))
        if "Wedge" in v1.extra:
            src += f"Wedge {v1.extra['Wedge']}\n"
    txt = gr.corrected_paramstest_text(
        src, refine_params, fitted, v1, observed_from_raw=observed_from_raw)
    out = {}
    for ln in txt.splitlines():
        parts = ln.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


# ───────────────────────────────────────────────────────────── composition
def test_tx_composes_on_the_default_path():
    """The measured 20-ID Au case: -0.1585 then -0.087265 must give -0.245762."""
    got = _emit(_v1(tx=-0.1584965882, Wedge=0.0), ("tx", "Wedge"),
                {"tx": -0.087265, "Wedge": 0.0})
    assert got["tx"] == pytest.approx(-0.245762, abs=1e-5), (
        "tx must COMPOSE; overwriting applies a third of the true roll")


def test_wedge_composes():
    got = _emit(_v1(Wedge=-0.012584), ("Wedge",), {"Wedge": 0.015325})
    assert got["Wedge"] == pytest.approx(0.002741, abs=1e-6)


def test_wedge_prior_is_read_from_extra_not_getattr():
    """THE BUG. Wedge is not a CalibrationParams field — it lives in .extra as a
    string — so getattr(v1, "Wedge", 0.0) returns 0.0 and the previous pass's
    Wedge is silently dropped from the file written out. Pass 1 from Wedge 0 was
    unaffected; pass 2 onward lost it."""
    v1 = _v1(Wedge=-0.012584)
    assert getattr(v1, "Wedge", None) is None, (
        "fixture must not lie: CalibrationParams has no Wedge attribute")
    assert gr._v1_scalar(v1, "Wedge") == pytest.approx(-0.012584)

    got = _emit(v1, ("Wedge",), {"Wedge": 0.0})
    assert got["Wedge"] == pytest.approx(-0.012584, abs=1e-9), (
        "the prior Wedge was dropped — this is the pass-2 data loss")


def test_tx_is_absolute_on_the_raw_pixel_path():
    """Thawing a _NEEDS_RAW scalar re-derives observations from raw pixels, where
    tx is applied inside the detector correction -- composing would double it."""
    got = _emit(_v1(tx=-0.1584965882), ("tx",), {"tx": -0.245762},
                observed_from_raw=True)
    assert got["tx"] == pytest.approx(-0.245762, abs=1e-9)


def test_absolute_scalars_are_never_composed():
    got = _emit(_v1(Lsd=1_000_000.0), ("Lsd",), {"Lsd": 899_916.0})
    assert got["Lsd"] == pytest.approx(899_916.0)


def test_zero_prior_is_a_noop():
    """The documented single-shot usage (run on a tx=0 recon) must be unchanged."""
    got = _emit(_v1(tx=0.0, Wedge=0.0), ("tx", "Wedge"),
                {"tx": -0.158497, "Wedge": -0.012584})
    assert got["tx"] == pytest.approx(-0.158497)
    assert got["Wedge"] == pytest.approx(-0.012584)


def test_relative_set_matches_the_model_construction():
    """_DIRECT is the set refinable without the raw path; tx/Wedge are the two
    the model zeroes, Lsd is seeded from v1."""
    assert {"tx", "Wedge", "Lsd"} == set(gr._DIRECT)
    assert "tx" not in gr._NEEDS_RAW and "Wedge" not in gr._NEEDS_RAW


def test_wedge_seeds_at_zero_not_at_the_prior():
    """Guard the OTHER half of the Wedge rule. The fit seeds Wedge at 0 on
    purpose, because the observed omega already carries the prior; seeding the
    prior there would apply it twice. Only the WRITE-BACK composes."""
    import inspect
    src = inspect.getsource(gr.refine_geometry_from_grains)
    i = src.index('Parameter("Wedge"')
    assert "torch.zeros" in src[i:i + 200], (
        "Wedge must seed at 0.0 — it is a correction on top of the "
        "observations, not an absolute value")


# ─────────────────────────────────────────────────── distortion phase rotation
def test_phases_rotate_by_k_times_delta_tx():
    """tx and the six distortion phases are ONE object. The distortion is a
    function of LAB eta and tx rolls the panel about the beam, so writing a new
    tx while carrying phi_k fitted at tx0 describes a different detector."""
    v1 = _v1(tx=0.0, a1=4e-4, phi1=11.0, a2=4.6e-4, phi2=33.0,
             a3=2e-4, phi3=-40.0)
    src = "tx 0.0\nphi1 11.0\nphi2 33.0\nphi3 -40.0\n"
    got = _emit(v1, ("tx",), {"tx": 0.2458}, src=src)

    assert got["tx"] == pytest.approx(0.2458)
    def wrap(x):
        return (x + 180.0) % 360.0 - 180.0
    assert got["phi1"] == pytest.approx(wrap(11.0 + 1 * 0.2458), abs=1e-9)
    assert got["phi2"] == pytest.approx(wrap(33.0 + 2 * 0.2458), abs=1e-9)
    assert got["phi3"] == pytest.approx(wrap(-40.0 + 3 * 0.2458), abs=1e-9)


def test_phase_rotation_uses_the_delta_not_the_absolute_tx():
    """On the composing path the phases must move by the FITTED residual, not by
    the composed total — the file's phases already sit in the prior's frame."""
    v1 = _v1(tx=-0.1584965882, a2=4.6e-4, phi2=33.0)
    src = "tx -0.1584965882\nphi2 33.0\n"
    got = _emit(v1, ("tx",), {"tx": -0.087265}, src=src)

    assert got["tx"] == pytest.approx(-0.245762, abs=1e-5)
    assert got["phi2"] == pytest.approx(
        (33.0 + 2 * -0.087265 + 180.0) % 360.0 - 180.0, abs=1e-6)


def test_legacy_p_named_files_are_rewritten_in_their_own_spelling():
    """A v2-native param file carries `phi3`; a legacy one carries `p10`."""
    v1 = _v1(tx=0.0, a3=2e-4, phi3=-40.0)
    src = "tx 0.0\np9 2e-4\np10 -40.0\n"
    got = _emit(v1, ("tx",), {"tx": 1.0}, src=src)

    assert "phi3" not in got, "must not add a v2 key to a legacy file"
    assert got["p10"] == pytest.approx(
        (-40.0 + 3 * 1.0 + 180.0) % 360.0 - 180.0, abs=1e-9)


def test_folds_with_zero_amplitude_are_left_alone():
    """A phase is meaningless where its amplitude is zero, and a distortion-free
    param file must come out byte-identical apart from the tx line."""
    v1 = _v1(tx=0.0)
    src = "tx 0.0\nLsd 1000000\n"
    out = gr.corrected_paramstest_text(src, ("tx",), {"tx": 0.5}, v1)
    assert out == "tx 0.5\nLsd 1000000\n"


def test_no_tx_change_leaves_the_phases_untouched():
    v1 = _v1(tx=0.3, a2=4.6e-4, phi2=33.0)
    src = "tx 0.3\nphi2 33.0\n"
    out = gr.corrected_paramstest_text(src, ("tx",), {"tx": 0.0}, v1)
    assert out == "tx 0.3\nphi2 33.0\n", "delta_tx == 0 must be a no-op"


def test_rotation_restores_the_applied_correction():
    """The physical check, not a bookkeeping one: rotating phi_k by k*dtx must
    leave the ACTUALLY APPLIED distortion unchanged, while carrying the phases
    over does not. Numbers are re-derived here, not quoted."""
    torch = pytest.importorskip("torch")
    A = pytest.importorskip(
        "midas_transforms.fit_setup.transform").apply_tilt_distortion
    import numpy as np
    from midas_distortion import P_COEF_NAMES, v2_to_v1_coeffs, v2_coeffs_from_named

    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(5)
    Y = torch.tensor(rng.uniform(0, 2048, 5000))
    Z = torch.tensor(rng.uniform(0, 2048, 5000))
    kw = dict(Lsd=torch.tensor(1e6), BC_y=torch.tensor(1024.),
              BC_z=torch.tensor(1024.), ty=torch.tensor(-0.37),
              tz=torch.tensor(0.53), px=torch.tensor(200.),
              rho_d=torch.tensor(2.0e5))
    named = dict(iso_R2=-1.11e-3, iso_R4=8e-4, a1=4e-4, phi1=11.0,
                 a2=4.6e-4, phi2=33.0, a3=2e-4, phi3=-40.0)

    def v1vec(d):
        return torch.tensor(v2_to_v1_coeffs(v2_coeffs_from_named(d)),
                            dtype=torch.float64)

    def radius(tx, d):
        Yn, Zn = A(Y, Z, tx=torch.tensor(tx), p_coeffs=v1vec(d), **kw)
        return torch.sqrt(Yn ** 2 + Zn ** 2)

    R0 = radius(0.0, named)
    dtx = 0.2458                              # manuals/ff-hedm/RUNBOOK.md:167, composed
    carried = float(((radius(dtx, named) - R0) / R0).abs().max()) * 1e6

    v1 = _v1(tx=0.0, **{k: v for k, v in named.items()})
    src = "tx 0.0\n" + "".join(f"{k} {v}\n" for k, v in named.items())
    got = _emit(v1, ("tx",), {"tx": dtx}, src=src)
    rotated_named = {k: got.get(k, v) for k, v in named.items()}
    rotated = float(((radius(dtx, rotated_named) - R0) / R0).abs().max()) * 1e6

    assert carried > 15.0, f"expected ~20 ue of drift, got {carried:.2f}"
    assert rotated < 1e-6, (
        f"rotating phi_k by k*dtx must restore the applied correction; "
        f"got {rotated:.3g} ue vs {carried:.2f} ue carried over")


# ───────────────────────── the four defects adversarial verification found
# These four were found by refuters attacking the phase-rotation fix above,
# not by writing it. Each is a case where the rotation was applied when it
# should not have been, or applied incompletely. None was covered by the
# tests written alongside the fix.

def test_no_rotation_on_the_raw_pixel_path():
    """On observed_from_raw=True the residual itself calls apply_tilt_distortion
    at the trial tx with the FILE's phases, so (tx*, phi_file) IS the fitted
    model. Rotating there moves the written file AWAY from what was fitted, by
    exactly the error the rotation exists to remove."""
    v1 = _v1(tx=0.0, a2=4.6e-4, phi2=33.0)
    src = "tx 0.0\nphi2 33.0\n"
    got = _emit(v1, ("tx",), {"tx": 0.2458}, observed_from_raw=True, src=src)

    assert got["tx"] == pytest.approx(0.2458)
    assert got["phi2"] == pytest.approx(33.0, abs=1e-12), (
        "the raw path fitted tx against these phases — they must not move")


def test_a_co_refined_phase_is_not_clobbered():
    """Refining any distortion name forces the raw path (every distortion name
    is in _DISTORTION), so the rotation block must not fire and overwrite the
    fitted phase with prior + k*dtx."""
    assert "phi2" in gr._DISTORTION and "a2" in gr._DISTORTION
    v1 = _v1(tx=0.0, a2=4.6e-4, phi2=33.0)
    src = "tx 0.0\nphi2 33.0\n"
    got = _emit(v1, ("tx", "phi2"), {"tx": 0.2458, "phi2": 77.5},
                observed_from_raw=True, src=src)
    assert got["phi2"] == pytest.approx(77.5), (
        "the fitted phase was overwritten by the rotation block")


def test_phases_wrap_into_the_modules_own_bounds():
    """This module bounds phi at (-180, 180). A phase written at 323 would rail
    a later run that thaws it, so the wrap must be [-180, 180), not [0, 360)."""
    import inspect
    src_code = inspect.getsource(gr.refine_geometry_from_grains)
    assert "(-180.0, 180.0) if nm.startswith(\"phi\")" in src_code, (
        "phi bounds moved — re-check the wrap below against them")

    v1 = _v1(tx=0.0, a3=2e-4, phi3=170.0)
    src = "tx 0.0\nphi3 170.0\n"
    got = _emit(v1, ("tx",), {"tx": 5.0}, src=src)
    # 170 + 3*5 = 185 -> must come back as -175, not 185
    assert got["phi3"] == pytest.approx(-175.0, abs=1e-9)
    assert -180.0 <= got["phi3"] < 180.0


def test_both_spellings_are_rewritten_when_a_file_carries_both():
    """calibrate-v2's compat/to_v1.py writes P_COEF_NAMES AND p0..p14, so real
    param files carry `phi3` and `p10` holding the same value. Updating only one
    leaves the file self-inconsistent, and midas_integrate_v2's from_v1 maps
    p0..p14 unconditionally -- it would read the stale phase and get the full
    error back out of a file that looks corrected."""
    v1 = _v1(tx=0.0, a3=2e-4, phi3=-71.18081846)
    src = "tx 0.0\nphi3 -71.18081846\np10 -71.18081846\n"
    got = _emit(v1, ("tx",), {"tx": 0.2458}, src=src)

    expect = (-71.18081846 + 3 * 0.2458 + 180.0) % 360.0 - 180.0
    assert got["phi3"] == pytest.approx(expect, abs=1e-9)
    assert got["p10"] == pytest.approx(expect, abs=1e-9), (
        "the legacy p-name was left stale — one file, two distortion fields")
    assert got["phi3"] == got["p10"]
