"""Two gaps closed 2026-08-29: inert spec flags, and the η seam.

**Inert spec flags.** ``IntegrationSpec`` carried ``PolarizationCorrection``,
``PolarizationFraction``, ``PolarizationPlaneEtaDeg`` and
``SolidAngleCorrection``; ``compat/from_v1.py`` and ``to_v1.py`` faithfully
round-tripped them — and nothing read them. ``PolarizationCorrection`` was never
constructed anywhere in the package and none of
``integrate_soft/hard/subpixel/polygon`` applies a correction at all, so
``integrate_with_corrections`` corrected only when handed an ``nn.Module``
explicitly. Measured: spec-flags-ON with no modules was byte-identical to
spec-flags-OFF (max|Δ| = 0) while explicit modules differed by 1.047.

That mattered because **v1's mapper DOES honour ``PolarizationCorrection = 1``**,
so converting a v1 param file to a v2 spec silently dropped the correction while
every field still said it was on — the issue-#69 family again.

**η seam.** ``soft_bin_indices_weights`` interpolates onto bin centres, so the
outer half-bins at ``EtaMin`` and ``EtaMax`` had no neighbour — but η is
PERIODIC and they are neighbours across the seam. Measured on a uniform
population over a full circle: 99.31 % of weight retained, with the two seam
bins **12 % under-filled** (4900.8 and 4860.2 against an interior mean of
5554.9). A wedge must NOT wrap, so the behaviour is gated on the η range
actually closing.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from midas_integrate.params import IntegrationParams
from midas_integrate_v2.compat.from_v1 import spec_from_v1_params
from midas_integrate_v2.corrections.integrated import integrate_with_corrections
from midas_integrate_v2.corrections.intensity import (
    PolarizationCorrection, SolidAngleCorrection, POL_PLANE_HORIZONTAL_ETA_DEG)
from midas_integrate_v2.diff.soft_bin import (
    soft_bin_indices_weights, eta_is_full_circle, integrate_diff)

N = 256


def _spec(*, corrections_on, eta_range=(-180.0, 180.0)):
    p = IntegrationParams()
    p.NrPixelsY = p.NrPixelsZ = N
    p.Lsd = 200_000.0
    p.BC_y = p.BC_z = N / 2.0
    p.pxY = p.pxZ = 200.0            # NOT p.px — that attribute is ignored
    p.RMin, p.RMax, p.RBinSize = 10.0, 110.0, 2.0
    p.EtaMin, p.EtaMax = eta_range
    p.EtaBinSize = 10.0
    p.PolarizationCorrection = int(corrections_on)
    p.PolarizationFraction = 1.0
    p.PolarizationPlaneEtaDeg = POL_PLANE_HORIZONTAL_ETA_DEG
    p.SolidAngleCorrection = int(corrections_on)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return spec_from_v1_params(p)


def _finite_max_abs_diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    m = np.isfinite(a) & np.isfinite(b)
    assert m.sum() > 100
    return float(np.abs(a[m] - b[m]).max())


# ------------------------------------------------------ the spec flags are live

def test_spec_flags_apply_the_correction():
    """Three arms, so the comparison can discriminate which of "both applied"
    and "neither applied" is true — the trap the first attempt at this fell
    into."""
    img = torch.ones((N, N), dtype=torch.float64)
    on, off = _spec(corrections_on=True), _spec(corrections_on=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = integrate_with_corrections(img, on).numpy()          # flags only
        b = integrate_with_corrections(                          # explicit
            img, on,
            polarization=PolarizationCorrection(
                pol_fraction=1.0,
                pol_plane_eta_deg=POL_PLANE_HORIZONTAL_ETA_DEG),
            solid_angle=SolidAngleCorrection()).numpy()
        c = integrate_with_corrections(img, off).numpy()         # the null

    assert _finite_max_abs_diff(a, b) == 0.0, (
        "spec flags must produce exactly what the explicit modules produce")
    assert _finite_max_abs_diff(a, c) > 1e-3, (
        "spec flags changed nothing — they are inert again")


def test_an_explicit_module_overrides_the_spec():
    img = torch.ones((N, N), dtype=torch.float64)
    on = _spec(corrections_on=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = integrate_with_corrections(img, on).numpy()
        other = integrate_with_corrections(
            img, on,
            polarization=PolarizationCorrection(pol_fraction=1.0,
                                                pol_plane_eta_deg=0.0)).numpy()
    assert _finite_max_abs_diff(default, other) > 1e-6


def test_flags_off_means_off():
    img = torch.ones((N, N), dtype=torch.float64)
    off = _spec(corrections_on=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = integrate_with_corrections(img, off).numpy()
        b = integrate_with_corrections(img, off, polarization=None,
                                       solid_angle=None).numpy()
    assert _finite_max_abs_diff(a, b) == 0.0


def test_the_plane_carried_by_the_spec_is_the_one_used():
    """A non-default plane must survive spec -> correction, not be replaced by
    the module default."""
    img = torch.ones((N, N), dtype=torch.float64)
    s = _spec(corrections_on=True)
    s.PolarizationPlaneEtaDeg = 37.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_spec = integrate_with_corrections(img, s).numpy()
        explicit = integrate_with_corrections(
            img, s,
            polarization=PolarizationCorrection(pol_fraction=1.0,
                                                pol_plane_eta_deg=37.5),
            solid_angle=SolidAngleCorrection()).numpy()
    assert _finite_max_abs_diff(from_spec, explicit) == 0.0


# ------------------------------------------------------------ the η seam

def test_full_circle_eta_is_detected_and_a_wedge_is_not():
    assert eta_is_full_circle(_spec(corrections_on=False))
    assert not eta_is_full_circle(
        _spec(corrections_on=False, eta_range=(-60.0, 60.0)))


def test_periodic_binning_loses_nothing_on_a_closed_axis():
    n_eta = 36
    eta = torch.rand(200_000, dtype=torch.float64) * 360.0 - 180.0
    kw = dict(R_min=-180.0, R_bin_size=10.0, n_r=n_eta)
    _, _, w0, w1 = soft_bin_indices_weights(eta, periodic=True, **kw)
    assert float((w0 + w1).sum()) == pytest.approx(200_000.0, rel=1e-12)
    _, _, v0, v1 = soft_bin_indices_weights(eta, periodic=False, **kw)
    assert float((v0 + v1).sum()) < 200_000.0 * 0.999    # the old seam loss


def test_the_seam_bins_are_no_longer_underfilled():
    n_eta = 36
    eta = torch.rand(400_000, dtype=torch.float64) * 360.0 - 180.0
    b0, b1, w0, w1 = soft_bin_indices_weights(
        eta, R_min=-180.0, R_bin_size=10.0, n_r=n_eta, periodic=True)
    h = torch.zeros(n_eta, dtype=torch.float64)
    h.index_add_(0, b0, w0)
    h.index_add_(0, b1, w1)
    interior = float(h[1:-1].mean())
    for k in (0, -1):
        assert abs(float(h[k]) - interior) / interior < 0.03, (
            f"seam bin {k} is {float(h[k]):.0f} vs interior {interior:.0f}")


def test_a_uniform_image_gives_a_flat_eta_profile():
    """End to end, the invariant a user would notice: no dark seam at ±180."""
    s = _spec(corrections_on=False)
    img = torch.ones((N, N), dtype=torch.float64)
    per_eta = integrate_diff(img, s).numpy().sum(axis=1)
    interior = per_eta[1:-1].mean()
    assert abs(per_eta[0] - interior) / interior < 0.02
    assert abs(per_eta[-1] - interior) / interior < 0.02


def test_a_wedge_does_not_wrap_opposite_sides_together():
    """Wrapping a partial η range would fold opposite sides of the detector
    onto each other — worse than losing the outer half-bins."""
    n_eta = 12
    eta = torch.tensor([-59.0, 59.0], dtype=torch.float64)
    b0, b1, _w0, _w1 = soft_bin_indices_weights(
        eta, R_min=-60.0, R_bin_size=10.0, n_r=n_eta, periodic=False)
    assert int(b0[0]) != int(b0[1]), "the two ends must not land in one bin"
