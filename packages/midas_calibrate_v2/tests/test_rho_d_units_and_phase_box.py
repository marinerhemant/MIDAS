"""RhoD must reach the distortion model in µm, and phases must not rail at +/-90.

The distortion polynomial is evaluated at ``rho = R_um / RhoD``. Three pipelines
(``autocalibrate_pv``, ``autocalibrate_pv_2d``, ``autocalibrate_joint``) fell back to
the PIXEL-valued ``MaxRingRad`` when RhoD was unset, inflating rho by the pixel pitch.
Measured on a real 2880x2880 / 150 um Varex frame: ``autocalibrate_pv`` fed
rho = 150.000 at the rim. Every distortion coefficient was then fitted in a basis no
other code shares, and ``rho_d_scaling_gate`` reported it as "well scaled" because it
only checked the too-large side.

Separately, every phase had a +/-90 tolerance. The LM enforces bounds as a sigmoid
box, so a phase CLAMPS at the edge instead of wrapping, and +/-90 is exactly the seam
between the two sign representations (a, phi) == (-a, phi + 180).
"""
from __future__ import annotations

import inspect
import math

import numpy as np
import pytest
import torch

from midas_calibrate.params import CalibrationParams as V1
from midas_calibrate_v2.compat.from_v1 import spec_from_v1_params
from midas_calibrate_v2.forward.sanity import resolve_v1_rho_d_um

PX = 150.0


def _v1(**kw):
    """Beniwal's geometry: 2880x2880, 150 um, beam centre OFF the panel."""
    p = V1(NrPixelsY=2880, NrPixelsZ=2880, pxY=PX, pxZ=PX, Lsd=1054947.0,
           BC_y=3044.26, BC_z=1344.22, Wavelength=0.172979, SpaceGroup=225,
           LatticeConstant=(5.411,) * 3 + (90.0,) * 3, MaxRingRad=3429.7)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _corner_um(v1):
    return PX * math.hypot(max(v1.BC_y, v1.NrPixelsY - 1 - v1.BC_y),
                           max(v1.BC_z, v1.NrPixelsZ - 1 - v1.BC_z))


# ─────────────────────────────────────────────────────────── the resolver
def test_unset_rho_d_resolves_to_micrometres_not_pixels():
    v1 = _v1()
    assert v1.RhoD == 0.0, "fixture: RhoD unset, as in the notebook that hit this"
    rho_d, how = resolve_v1_rho_d_um(v1)
    dmax = _corner_um(v1)
    assert 0.5 * dmax <= rho_d <= 5.0 * dmax, (rho_d, dmax, how)
    assert v1.RhoD == rho_d, "must be written back so the E-step sees µm too"
    # the value the buggy fallback produced
    assert rho_d / v1.MaxRingRad > 100.0


def test_pixel_valued_rho_d_is_converted():
    v1 = _v1(RhoD=3429.7)                       # the mistake: RhoD given in px
    rho_d, _ = resolve_v1_rho_d_um(v1)
    assert rho_d == pytest.approx(3429.7 * PX, rel=1e-9)


def test_sane_micrometre_rho_d_is_left_alone():
    v1 = _v1(RhoD=514000.0)
    rho_d, _ = resolve_v1_rho_d_um(v1)
    assert rho_d == pytest.approx(514000.0)


def test_resolver_is_idempotent():
    v1 = _v1()
    a, _ = resolve_v1_rho_d_um(v1)
    b, _ = resolve_v1_rho_d_um(v1)
    assert a == b


# ────────────────────────────────────── every affected entry point resolves
@pytest.mark.parametrize("modname,fn", [
    ("single", "autocalibrate"),
    ("single_pv", "autocalibrate_pv"),
    ("single_pv_2d", "autocalibrate_pv_2d"),
    ("joint_cake", "autocalibrate_joint"),
])
def test_entry_point_resolves_rho_d(modname, fn):
    mod = __import__(f"midas_calibrate_v2.pipelines.{modname}", fromlist=[fn])
    src = inspect.getsource(getattr(mod, fn))
    assert "resolve_v1_rho_d_um(" in src, f"{modname}.{fn} does not resolve RhoD"
    # and it must happen before the spec is built from v1
    assert src.index("resolve_v1_rho_d_um(") < src.index("spec_from_v1_params(")


@pytest.mark.parametrize("modname", ["single_pv", "single_pv_2d", "joint_cake"])
def test_no_bare_pixel_fallback_remains(modname):
    """Second line of defence: a direct caller of the bake step with an
    unresolved v1 must still get µm."""
    mod = __import__(f"midas_calibrate_v2.pipelines.{modname}", fromlist=["x"])
    src = inspect.getsource(mod)
    for line in src.splitlines():
        if "MaxRingRad" in line and "RhoD" in line and "if" in line and "else" in line:
            assert "* px" in line, f"pixel fallback still bare in {modname}: {line.strip()}"


# ────────────────────────────────────────────────────────────── from_v1
def test_spec_rho_d_fallback_is_micrometres():
    s = spec_from_v1_params(_v1())
    assert s.parameters["RhoD"].init == pytest.approx(3429.7 * PX)


def test_phase_box_is_a_full_period_and_amplitudes_are_unchanged():
    v1 = _v1()
    s = spec_from_v1_params(v1)
    for k in range(1, 7):
        lo, hi = s.parameters[f"phi{k}"].bounds
        init = float(s.parameters[f"phi{k}"].init)
        assert (lo, hi) == pytest.approx((init - 180.0, init + 180.0)), f"phi{k}"
        alo, ahi = s.parameters[f"a{k}"].bounds
        assert (ahi - alo) / 2 == pytest.approx(v1.tolDistortion), f"a{k}"


# ──────────────────────────────────────────────────────────────── the gate
class _Fits:
    def __init__(self, rho_d, r_px, bc=(1024.0, 1024.0)):
        th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        self.Y_pix = torch.tensor(bc[0] + r_px * np.cos(th))
        self.Z_pix = torch.tensor(bc[1] + r_px * np.sin(th))
        self.rho_d = torch.tensor(float(rho_d))


def _unpacked(px=PX, bc=(1024.0, 1024.0)):
    from midas_distortion import P_COEF_NAMES
    u = {"pxY": torch.tensor(px), "pxZ": torch.tensor(px),
         "BC_y": torch.tensor(bc[0]), "BC_z": torch.tensor(bc[1])}
    u.update({n: torch.tensor(0.0) for n in P_COEF_NAMES})
    return u


def _spec(refine_distortion):
    v1 = V1(NrPixelsY=2048, NrPixelsZ=2048, pxY=PX, pxZ=PX, Lsd=1e6,
            BC_y=1024.0, BC_z=1024.0, Wavelength=0.17, SpaceGroup=225,
            LatticeConstant=(5.411,) * 3 + (90.0,) * 3, MaxRingRad=1000.0)
    v1.Refine = dict(v1.Refine)
    for i in range(15):
        v1.Refine[f"p{i}"] = refine_distortion
    return spec_from_v1_params(v1)


def test_gate_fails_a_pixel_rho_d_when_distortion_is_refined():
    from midas_calibrate_v2.pipelines.diagnostics import rho_d_scaling_gate
    r_px = 1000.0
    d = rho_d_scaling_gate(_Fits(rho_d=r_px, r_px=r_px), _unpacked(),
                           spec=_spec(refine_distortion=True))
    assert d.severity == "fail", d.message
    assert "PIXELS" in d.message
    assert d.metrics["rho_max"] == pytest.approx(PX, rel=1e-6)


def test_gate_only_warns_a_pixel_rho_d_when_distortion_is_frozen():
    """Frozen distortion is identically 1, so RhoD cannot hurt the fit, but the
    unit is still wrong and would matter the moment it is refined."""
    from midas_calibrate_v2.pipelines.diagnostics import rho_d_scaling_gate
    r_px = 1000.0
    d = rho_d_scaling_gate(_Fits(rho_d=r_px, r_px=r_px), _unpacked(),
                           spec=_spec(refine_distortion=False))
    assert d.severity == "warn", d.message


def test_gate_passes_a_micrometre_rho_d():
    from midas_calibrate_v2.pipelines.diagnostics import rho_d_scaling_gate
    r_px = 1000.0
    d = rho_d_scaling_gate(_Fits(rho_d=r_px * PX, r_px=r_px), _unpacked(),
                           spec=_spec(refine_distortion=True))
    assert d.severity == "ok", d.message


# ─────────────────────────────────────── docstrings that taught the mistake
def test_no_docstring_still_says_rho_d_is_in_pixels():
    import midas_calibrate_v2.forward.geometry as g
    import midas_calibrate_v2.pipelines._common as c
    import midas_calibrate_v2.compat.to_integrate as t
    gsrc = inspect.getsource(g)
    assert "# px; distortion normalisation radius" not in gsrc
    assert "distortion normalisation radius (px)" not in gsrc
    assert "rho_d: torch.Tensor                # px" not in inspect.getsource(c)
    assert "``RhoD`` is set in **pixels**" not in inspect.getsource(t)
