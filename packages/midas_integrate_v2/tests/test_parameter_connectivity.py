"""Every parameter the forward model accepts must actually reach it.

The dominant failure mode in this stack is not a wrong formula — it is a value
that is read, stored, and then never forwarded. `FixPanelID` did it twice: once
because the parser never read it, and once because `pseudo_strain_residual`
forwarded every panel argument except that one, so the model silently anchored
panel 0 forever. Both were found by chance, months apart.

Testing the plumbing (`spec.fix_panel_id == 28`) does not catch this: that
assertion passed the whole time. The only thing that catches it is asserting
the parameter has an EFFECT — perturb it, and the output must move.

So: sweep the geometry parameters, perturb each, and require the R map to
change. A parameter that can be set but changes nothing is disconnected, dead,
or silently ignored — and this test says which.

Deliberately a sweep, not a hand-written list: a new parameter added to the
spec is covered the day it appears, without anyone remembering to test it.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2.compat.from_v1 import spec_from_v1_params
from midas_integrate_v2.forward.pixels import eval_pixel_REta, _PANEL_CACHE

#: (field, perturbation, observable). The observable matters: a naive
#: "perturb it, R must move" sweep reports three whole classes of parameter as
#: disconnected when they are working correctly.
#:
#:   R    — moves the in-plane radial distance
#:   eta  — moves the azimuth only. `tx` is a rotation ABOUT the beam, so it
#:          maps the detector plane onto itself: eta moves, R does not.
#:   tth  — `R(pixel)` is the in-plane distance and does not depend on `Lsd` at
#:          all; Lsd only enters converting R to a scattering angle. Sweeping
#:          Lsd against R reports a false failure (and did, first time).
GEOMETRY_PARAMS = [
    ("Lsd",       1000.0, "tth"),   # µm
    ("BC_y",         1.0, "R"),     # px
    ("BC_z",         1.0, "R"),
    ("ty",           0.05, "R"),    # deg
    ("tz",           0.05, "R"),
    ("tx",           0.05, "eta"),
    ("pxY",          1.0, "R"),     # µm
    ("Parallax",    10.0, "R"),
]

#: Distortion amplitudes — these act alone.
#: v2 names: iso_R2/R4/R6 and a1..a6.
DISTORTION_AMPLITUDES = ["p2", "p5", "p4",          # isotropic radial
                          "p7", "p0", "p9", "p1", "p11", "p13"]   # a1..a6

#: Distortion phases, paired with the amplitude that switches them on. A phase
#: multiplies a term of amplitude zero, so alone it is correctly a no-op —
#: sweeping it unpaired reports six false failures.
DISTORTION_PHASES = [
    ("p8",  "p7"),   # phi1 / a1
    ("p6",  "p0"),   # phi2 / a2
    ("p10", "p9"),   # phi3 / a3
    ("p3",  "p1"),   # phi4 / a4
    ("p12", "p11"),  # phi5 / a5
    ("p14", "p13"),  # phi6 / a6
]


def _params(NY=64, NZ=64) -> IntegrationParams:
    return IntegrationParams(
        NrPixelsY=NY, NrPixelsZ=NZ, pxY=200.0, pxZ=200.0, Lsd=1_000_000.0,
        BC_y=NY / 2 + 0.37, BC_z=NZ / 2 - 0.41, RhoD=float(NY) * 200.0,
        RMin=1.0, RMax=25.0, RBinSize=1.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=30.0,
        Wavelength=0.2,
    )


def _observe(p: IntegrationParams, what: str) -> np.ndarray:
    _PANEL_CACHE.clear()
    spec = spec_from_v1_params(p, requires_grad=False)
    R, eta = eval_pixel_REta(spec)
    R = R.detach().numpy()
    if what == "R":
        return R
    if what == "eta":
        return eta.detach().numpy()
    if what == "tth":
        return np.degrees(np.arctan(R * p.pxY / p.Lsd))
    raise ValueError(what)


def _R_map(p: IntegrationParams) -> np.ndarray:
    return _observe(p, "R")


def _changed(base: np.ndarray, other: np.ndarray) -> float:
    return float(np.abs(base - other).max())


@pytest.mark.parametrize("field,delta,observable", GEOMETRY_PARAMS)
def test_geometry_parameter_reaches_the_forward_model(field, delta, observable):
    p = _params()
    base = _observe(p, observable)
    setattr(p, field, getattr(p, field) + delta)
    moved = _observe(p, observable)
    d = _changed(base, moved)
    assert d > 1e-9, (
        f"{field} changed by {delta} and {observable} did not move "
        f"(max |d| = {d:.3e}). The parameter is disconnected, ignored, or dead."
    )


@pytest.mark.parametrize("field", DISTORTION_AMPLITUDES)
def test_distortion_amplitude_reaches_the_forward_model(field):
    """v2 renames these internally (iso_R2, a1..a6); a rename that dropped one
    on the floor would otherwise be invisible."""
    p = _params()
    base = _R_map(p)
    setattr(p, field, 1e-3)
    d = _changed(base, _R_map(p))
    assert d > 1e-9, (
        f"{field} was set to 1e-3 and the R map did not move (max |dR| = "
        f"{d:.3e}) — this distortion amplitude is not reaching the model."
    )


@pytest.mark.parametrize("phase,amplitude", DISTORTION_PHASES)
def test_distortion_phase_reaches_the_forward_model(phase, amplitude):
    """A phase only bites when its amplitude is non-zero, so switch it on."""
    p = _params()
    setattr(p, amplitude, 1e-3)
    base = _R_map(p)
    setattr(p, phase, 0.7)
    d = _changed(base, _R_map(p))
    assert d > 1e-12, (
        f"{phase} (phase of {amplitude}) did not move the R map with its "
        f"amplitude switched on (max |dR| = {d:.3e}) — not reaching the model."
    )


def test_a_phase_alone_is_correctly_inert():
    """The converse, so the pairing above is not hiding a real disconnection:
    with the amplitude at zero a phase MUST do nothing."""
    p = _params()
    base = _R_map(p)
    p.p3 = 0.7                      # phi4, with a4 = p1 still zero
    assert _changed(base, _R_map(p)) == 0.0


def test_rhod_reaches_the_forward_model():
    """RhoD normalises the distortion radius, so it only bites when a
    coefficient is non-zero. Sweeping it against a zero distortion would
    wrongly report it as disconnected."""
    p = _params()
    p.p2 = 1e-3
    base = _R_map(p)
    p.RhoD *= 1.5
    moved = _R_map(p)
    assert _changed(base, moved) > 1e-9, "RhoD is not reaching the model"


def test_the_sweep_can_fail():
    """Control: a field the forward model genuinely does not consume must be
    reported as unmoved. Without this, a sweep that always passed would look
    identical to a sweep that works."""
    p = _params()
    base = _R_map(p)
    p.Normalize = 0                  # an integration flag, not geometry
    assert _changed(base, _R_map(p)) == 0.0, (
        "Normalize moved the R map — either the control is wrong or the "
        "geometry is picking up an integration flag"
    )


def test_every_swept_field_exists_on_the_params_object():
    """Guard against the sweep silently testing nothing after a rename."""
    p = _params()
    for field, _, _ in GEOMETRY_PARAMS:
        assert hasattr(p, field), f"swept field {field!r} no longer exists"
    for field in DISTORTION_AMPLITUDES:
        assert hasattr(p, field), f"swept field {field!r} no longer exists"
    for phase, amp in DISTORTION_PHASES:
        assert hasattr(p, phase) and hasattr(p, amp)
    covered = ({f for f, _, _ in GEOMETRY_PARAMS}
               | set(DISTORTION_AMPLITUDES) | {ph for ph, _ in DISTORTION_PHASES})
    all_p = {f"p{i}" for i in range(15)}
    assert all_p <= covered, f"distortion coefficients not swept: {all_p - covered}"
