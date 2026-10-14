"""Known-answer tests: the sign of OmegaStep must not change which
reflections ``predict_expected_hkls`` calls visible.

MIDAS 1-ID aero scans run with OmegaStep < 0 (e.g. -0.25, ω from +180 down
to -180).  ``GeometryConfig`` describes the scan window by
``[omega_min_deg, omega_max_deg]`` (always ascending), so the same physical
window with step +0.25 and step -0.25 must give the SAME set of visible
(grain, h, k, l, branch) reflections and the same per-grain expected-variant
counts (the denominator of the completeness / over-merge test).

Before the fix, ``_predict_via_midas_diffract`` passed the signed step to
``HEDMGeometry(omega_start=omega_min, omega_step=step)`` so the forward
model's frame index ``(ω - omega_min) / step`` was <= 0 for every in-window
ω and every reflection was flagged off-scan (zero visible).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("midas_diffract")
pytest.importorskip("pandas")

from midas_process_grains.compute.hkl_expected import (  # noqa: E402
    GeometryConfig,
    expected_visible_variants_per_grain,
    predict_expected_hkls,
)
from midas_process_grains.compute.hkl_ingest import HklTable  # noqa: E402

A_LAT = 3.6            # Å, cubic
WAVELENGTH = 0.172979  # Å  (~71.7 keV)


def _cubic_hkls(max_n2: int = 11) -> HklTable:
    """All signed (hkl) of an fcc lattice up to h^2+k^2+l^2 <= max_n2."""
    rows = []
    rng = range(-3, 4)
    for h in rng:
        for k in rng:
            for l in rng:
                n2 = h * h + k * k + l * l
                if n2 == 0 or n2 > max_n2:
                    continue
                if len({h % 2, k % 2, l % 2}) != 1:   # fcc: all odd or all even
                    continue
                rows.append((h, k, l, n2))
    ring_of = {n2: i + 1 for i, n2 in enumerate(sorted({r[3] for r in rows}))}
    h = np.array([r[0] for r in rows], np.int32)
    k = np.array([r[1] for r in rows], np.int32)
    l = np.array([r[2] for r in rows], np.int32)
    n2 = np.array([r[3] for r in rows], np.float64)
    d = A_LAT / np.sqrt(n2)
    theta = np.degrees(np.arcsin(WAVELENGTH / (2.0 * d)))
    g = np.stack([h, k, l], axis=1).astype(np.float64) / np.sqrt(n2)[:, None]
    return HklTable(
        h=h, k=k, l=l, ring=np.array([ring_of[int(x)] for x in n2], np.int32),
        d_A=d, g_crystal=g, theta_deg=theta, ttheta_deg=2 * theta,
        radius_um=1.0e6 * np.tan(np.radians(2 * theta)),
    )


def _random_orientations(n: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty((n, 3, 3))
    for i in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        q = q * np.sign(np.diag(r))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        out[i] = q
    return out


def _geom(step: float, omega_min: float, omega_max: float) -> GeometryConfig:
    return GeometryConfig(
        lsd_um=1.0e6, omega_min_deg=omega_min, omega_max_deg=omega_max,
        wavelength_a=WAVELENGTH, pixel_um=200.0, y_BC=1024.0, z_BC=1024.0,
        omega_step_deg=step, n_pixels_y=2048, n_pixels_z=2048,
        min_eta_deg=6.0, use_midas_diffract=True,
    )


@pytest.fixture(scope="module")
def setup():
    hk = _cubic_hkls()
    return hk, sorted({int(r) for r in hk.ring}), _random_orientations(6)


@pytest.mark.parametrize("window", [(-180.0, 180.0), (0.0, 360.0), (-180.0, 0.0)])
def test_visibility_independent_of_omega_step_sign(setup, window):
    hk, rings, om = setup
    pos = predict_expected_hkls(om, hk, rings, _geom(+0.25, *window))
    neg = predict_expected_hkls(om, hk, rings, _geom(-0.25, *window))
    # sanity: the positive-step answer is non-trivial
    assert pos.visible.sum() > 0
    assert np.array_equal(pos.visible, neg.visible), (
        f"window {window}: visible(+0.25)={int(pos.visible.sum())} "
        f"visible(-0.25)={int(neg.visible.sum())}")
    np.testing.assert_allclose(pos.pred_omega_deg, neg.pred_omega_deg)
    n = om.shape[0]
    np.testing.assert_array_equal(
        expected_visible_variants_per_grain(pos, n),
        expected_visible_variants_per_grain(neg, n))


def test_negative_step_full_scan_counts_are_sane(setup):
    """Full 360 deg scan, step -0.25: every grain must have a large expected
    variant count (a 360 deg window sees roughly half the signed variants,
    both Friedel branches).  Zero would mean the whole scan was rejected."""
    hk, rings, om = setup
    neg = predict_expected_hkls(om, hk, rings, _geom(-0.25, -180.0, 180.0))
    counts = expected_visible_variants_per_grain(neg, om.shape[0])
    n_variants = len(hk.h)
    assert (counts > 0.3 * n_variants).all(), counts


@pytest.mark.xfail(strict=True, reason=(
    "Separate, pre-existing limitation (NOT the step-sign bug): the forward "
    "model's frame_ok uses the model's unwrapped omega in (-180, 180], so a "
    "[0, 360] window silently loses the omega < 0 half (298 vs 596 visible "
    "on this fixture).  Unreachable from the CLI, which always uses "
    "[-180, 180].  Remove this xfail when that is fixed."))
def test_window_0_360_equals_window_pm180(setup):
    hk, rings, om = setup
    a = predict_expected_hkls(om, hk, rings, _geom(+0.25, -180.0, 180.0))
    b = predict_expected_hkls(om, hk, rings, _geom(+0.25, 0.0, 360.0))
    assert int(a.visible.sum()) == int(b.visible.sum())
