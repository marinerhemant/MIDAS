"""Geometry transforms — sanity tests against analytical truths."""
from __future__ import annotations

import math

import numpy as np
import pytest

from midas_integrate.geometry import (
    DEG2RAD,
    REta_to_YZ,
    REta_to_YZ_scalar,
    build_bin_edges,
    build_q_bin_edges_in_R,
    build_tth_bin_edges_in_R,
    build_tilt_matrix,
    calc_eta_angle,
    circle_seg_intersect,
    invert_REta_to_pixel,
    invert_REta_to_pixel_batch,
    pixel_to_REta,
    polygon_area,
    ray_seg_intersect,
)


def test_zero_tilt_identity():
    M = build_tilt_matrix(0, 0, 0)
    np.testing.assert_array_almost_equal(M, np.eye(3))


def test_pixel_to_REta_at_beam_center_is_zero():
    # No tilt, no distortion: pixel at beam center → R = 0
    Ycen, Zcen = 100.0, 200.0
    TRs = build_tilt_matrix(0, 0, 0)
    R, eta = pixel_to_REta(
        Ycen, Zcen,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=1.0, RhoD=1.0, px=1.0,
    )
    assert abs(float(R)) < 1e-9


def test_pixel_to_REta_radial_symmetry():
    """No tilt, no distortion: R should equal pixel distance from beam center."""
    Ycen, Zcen = 0.0, 0.0
    TRs = build_tilt_matrix(0, 0, 0)
    Y = np.array([10.0, -10.0, 0.0, 0.0])
    Z = np.array([0.0, 0.0, 10.0, -10.0])
    R, _ = pixel_to_REta(
        Y, Z,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=1e9, RhoD=1.0, px=1.0,
    )
    # At infinite distance the curvature term cancels and R ≈ |Y or Z|
    np.testing.assert_array_almost_equal(R, [10, 10, 10, 10], decimal=3)


def test_invert_round_trip():
    Ycen, Zcen = 700.0, 865.0
    TRs = build_tilt_matrix(0.05, 0.18, 0.53)
    R_target, eta_target = 250.0, 30.0
    Y, Z = invert_REta_to_pixel(
        R_target, eta_target,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=580_000.0, RhoD=2200.0, px=172.0,
    )
    R_back, eta_back = pixel_to_REta(
        Y, Z,
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=580_000.0, RhoD=2200.0, px=172.0,
    )
    assert abs(float(R_back) - R_target) < 1e-4
    assert abs(float(eta_back) - eta_target) < 1e-4


def test_invert_REta_to_pixel_batch_matches_scalar():
    """Batched Newton inversion agrees point-for-point with the scalar
    version on a calibrant-shaped sweep (5 rings × 360 η bins)."""
    Ycen, Zcen = 700.0, 865.0
    TRs = build_tilt_matrix(0.05, 0.18, 0.53)
    common = dict(
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=580_000.0, RhoD=2200.0, px=172.0,
        # Throw in a couple of non-trivial distortion coefficients to make
        # sure the per-point param plumbing is identical scalar vs batch.
        p2=1e-4, p7=2e-3, p8=15.0,
    )
    rng = np.random.default_rng(0)
    R_targets = np.repeat(np.linspace(120.0, 480.0, 5), 360) \
        + rng.uniform(-0.5, 0.5, 5 * 360)
    Eta_targets = np.tile(np.linspace(-179.5, 179.5, 360), 5)

    Y_scalar = np.empty_like(R_targets)
    Z_scalar = np.empty_like(R_targets)
    for i in range(R_targets.shape[0]):
        Y_scalar[i], Z_scalar[i] = invert_REta_to_pixel(
            float(R_targets[i]), float(Eta_targets[i]), **common,
        )

    Y_batch, Z_batch = invert_REta_to_pixel_batch(
        R_targets, Eta_targets, **common,
    )

    np.testing.assert_allclose(Y_batch, Y_scalar, atol=1e-6)
    np.testing.assert_allclose(Z_batch, Z_scalar, atol=1e-6)


def test_invert_REta_to_pixel_batch_scalar_input():
    """1-element array input should round-trip and match the scalar path."""
    Ycen, Zcen = 100.0, 200.0
    TRs = build_tilt_matrix(0, 0, 0)
    common = dict(
        Ycen=Ycen, Zcen=Zcen, TRs=TRs,
        Lsd=1_000_000.0, RhoD=2000.0, px=200.0,
    )
    Y_b, Z_b = invert_REta_to_pixel_batch(
        np.array([150.0]), np.array([45.0]), **common,
    )
    Y_s, Z_s = invert_REta_to_pixel(150.0, 45.0, **common)
    assert abs(Y_b[0] - Y_s) < 1e-6
    assert abs(Z_b[0] - Z_s) < 1e-6


def test_REta_to_YZ_round_trip():
    R, eta_deg = 100.0, 45.0
    Y, Z = REta_to_YZ_scalar(R, eta_deg)
    assert abs(math.hypot(Y, Z) - R) < 1e-12
    assert abs(calc_eta_angle(Y, Z) - eta_deg) < 1e-12


def test_circle_seg_intersect_diameter():
    # Horizontal segment from (-2, 0) to (2, 0); circle R=1 centered at origin
    hits = circle_seg_intersect(-2.0, 0.0, 2.0, 0.0, 1.0)
    assert len(hits) == 2
    xs = sorted(h[0] for h in hits)
    assert abs(xs[0] - (-1.0)) < 1e-12
    assert abs(xs[1] - 1.0) < 1e-12


def test_polygon_area_unit_square():
    # Unit square — Shoelace gives area 1
    edges = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
    a = polygon_area(edges, RMin=10.0, RMax=20.0)  # Rs irrelevant here
    assert abs(a - 1.0) < 1e-12


def test_polygon_area_arc():
    # A 90° wedge of an annulus: outer R=2, inner R=1, η ∈ [0°, 90°]
    # Area = (π/4) * (R² - r²) = (π/4) * 3 ≈ 2.356
    R1, R2 = 1.0, 2.0
    edges = [
        (R1, 0.0), (R2, 0.0),       # ray η=0
        (0.0, R2), (0.0, R1),       # ray η=90°
    ]
    # Make sure the arcs are on R1 and R2: distances must match within tol
    a = polygon_area(edges, RMin=R1, RMax=R2)
    expected = math.pi / 4.0 * (R2 ** 2 - R1 ** 2)
    # We expect the true area; allow some tolerance because endpoints lie
    # on both arcs but the algorithm assigns them to one specific arc each.
    assert a > 1.0
    assert a < expected * 1.5


def test_build_bin_edges():
    r_lo, r_hi, e_lo, e_hi = build_bin_edges(
        RMin=10.0, EtaMin=-180.0, n_r_bins=5, n_eta_bins=4,
        RBinSize=1.0, EtaBinSize=90.0,
    )
    np.testing.assert_array_equal(r_lo, [10, 11, 12, 13, 14])
    np.testing.assert_array_equal(r_hi, [11, 12, 13, 14, 15])
    np.testing.assert_array_equal(e_lo, [-180, -90, 0, 90])


def test_q_mode_R_bins_monotonic():
    r_lo, r_hi, n = build_q_bin_edges_in_R(
        QMin=0.5, QMax=5.0, QBinSize=0.05,
        Lsd=580_000.0, px=172.0, wavelength_A=0.5,
    )
    assert n == 90
    # R is monotonically increasing in Q
    assert np.all(np.diff(r_lo) > 0)


def test_tth_mode_R_bins_match_tan_2theta():
    """Equi-2theta bins map to R via R = (Lsd/px) * tan(2theta)."""
    Lsd, px = 580_000.0, 172.0
    r_lo, r_hi, n = build_tth_bin_edges_in_R(
        TthMin=1.0, TthMax=15.0, TthBinSize=0.1,
        Lsd=Lsd, px=px,
    )
    assert n == 140
    # First edge: tan(1 deg) * Lsd/px
    expected_first = Lsd / px * np.tan(np.radians(1.0))
    np.testing.assert_allclose(r_lo[0], expected_first, rtol=1e-12)
    # Last edge upper: tan(15 deg) * Lsd/px
    expected_last = Lsd / px * np.tan(np.radians(15.0))
    np.testing.assert_allclose(r_hi[-1], expected_last, rtol=1e-12)
    # Edges are strictly monotonic in R since tan() is monotonic in
    # (0, pi/2)
    assert np.all(np.diff(r_lo) > 0)
    assert np.all(np.diff(r_hi) > 0)


def test_q_and_tth_modes_round_trip_through_2theta():
    """Equi-Q and equi-2theta should produce the same R bin edges
    when the Q grid is constructed from a uniform 2theta grid via the
    Bragg formula --- a sanity check that both code paths agree on the
    underlying R = (Lsd/px) tan(2theta) projection."""
    Lsd, px, lam = 580_000.0, 172.0, 0.5
    tth_min_deg, tth_max_deg, tth_step_deg = 1.0, 10.0, 0.1
    # Build the 2theta grid directly.
    r_lo_tth, r_hi_tth, n_tth = build_tth_bin_edges_in_R(
        TthMin=tth_min_deg, TthMax=tth_max_deg, TthBinSize=tth_step_deg,
        Lsd=Lsd, px=px,
    )
    # Convert the same 2theta endpoints to Q via Bragg.
    q_min = 4.0 * np.pi * np.sin(np.radians(tth_min_deg) / 2.0) / lam
    q_max = 4.0 * np.pi * np.sin(np.radians(tth_max_deg) / 2.0) / lam
    # A non-uniform Q grid that lands on the same 2theta grid:
    tth = np.radians(np.arange(tth_min_deg, tth_max_deg + 1e-9, tth_step_deg))
    q_grid = 4.0 * np.pi * np.sin(tth / 2.0) / lam
    # The Q grid above is non-uniform, so we cannot use it directly with
    # build_q_bin_edges_in_R (which assumes uniform Q).  Instead we
    # confirm the R edges of the equi-2theta grid map correctly to Q
    # via the same Bragg formula:
    q_lo_from_tth = 4.0 * np.pi * np.sin(
        np.arctan(r_lo_tth * px / Lsd) / 2.0
    ) / lam
    np.testing.assert_allclose(q_lo_from_tth[0], q_min, rtol=1e-10)
    np.testing.assert_allclose(q_lo_from_tth[-1], q_grid[-2], rtol=1e-10)


def test_params_bin_axis_property_returns_correct_mode():
    """The IntegrationParams.bin_axis property reports the active mode."""
    from midas_integrate.params import IntegrationParams
    base = dict(NrPixelsY=2048, NrPixelsZ=2048, Lsd=580_000.0,
                pxY=172.0, pxZ=172.0,
                EtaMin=-180.0, EtaMax=180.0, EtaBinSize=5.0)
    p_r = IntegrationParams(RMin=10.0, RMax=1000.0, RBinSize=1.0, **base)
    p_r.validate()
    assert p_r.bin_axis == "R"
    p_q = IntegrationParams(QMin=0.5, QMax=5.0, QBinSize=0.05,
                             Wavelength=0.5, **base)
    p_q.validate()
    assert p_q.bin_axis == "Q"
    p_tth = IntegrationParams(TthMin=1.0, TthMax=15.0, TthBinSize=0.1, **base)
    p_tth.validate()
    assert p_tth.bin_axis == "tth"


def test_params_q_and_tth_modes_mutually_exclusive():
    """Setting both Q-mode and 2theta-mode must raise."""
    from midas_integrate.params import IntegrationParams
    p = IntegrationParams(
        NrPixelsY=2048, NrPixelsZ=2048, Lsd=580_000.0,
        pxY=172.0, pxZ=172.0,
        EtaMin=-180.0, EtaMax=180.0, EtaBinSize=5.0,
        QMin=0.5, QMax=5.0, QBinSize=0.05, Wavelength=0.5,
        TthMin=1.0, TthMax=15.0, TthBinSize=0.1,
    )
    import pytest
    with pytest.raises(ValueError, match="mutually exclusive"):
        p.validate()


# ── R_target is in PIXELS, and says so ───────────────────────────────────
# The name reads as "convert R,eta TO pixels", so passing micrometres -- the
# unit ring radii carry nearly everywhere else in MIDAS -- is the natural
# mistake. It used to be silent: the seed Y = Ycen + R*sin(eta) simply landed
# ~50 000 px off the panel, Newton nudged it, and the caller drew an overlay
# whose rings were all off-frame.
_GE_LIKE = dict(
    Ycen=1022.0, Zcen=974.0, Lsd=772_000.0,
    RhoD=289_600.0, px=200.0,          # 2048^2 GE, 200 um pitch
)


def _ge_kwargs():
    return dict(TRs=build_tilt_matrix(0.0, 0.0, 0.0), **_GE_LIKE)


def test_micrometre_radius_warns_scalar():
    from midas_integrate.geometry import RadiusUnitWarning

    with pytest.warns(RadiusUnitWarning, match="DETECTOR PIXELS"):
        invert_REta_to_pixel(500.0 * 200.0, 30.0, **_ge_kwargs())


def test_micrometre_radius_warns_batch():
    from midas_integrate.geometry import RadiusUnitWarning

    R_um = np.array([300.0, 500.0, 700.0]) * 200.0
    with pytest.warns(RadiusUnitWarning):
        invert_REta_to_pixel_batch(R_um, np.zeros(3), **_ge_kwargs())


def test_pixel_radius_does_not_warn():
    """Every legitimate radius on this panel, right out to the corner, must be
    silent -- a guard that cries wolf gets muted and then never fires."""
    import warnings as _w

    R_px = np.linspace(50.0, 1448.0, 64)       # corner is RhoD/px = 1448 px
    with _w.catch_warnings():
        _w.simplefilter("error")
        invert_REta_to_pixel_batch(R_px, np.linspace(-179, 179, 64),
                                   **_ge_kwargs())
        invert_REta_to_pixel(1448.0, 0.0, **_ge_kwargs())


def test_guard_is_silent_when_only_a_token_RhoD_is_supplied():
    """Synthetic geometries pass a token RhoD (2200 um at a 172 um pitch =
    12.8 px), which on its own would make every real radius look enormous. The
    beam-centre scale has to rescue it."""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("error")
        invert_REta_to_pixel(250.0, 30.0, Ycen=700.0, Zcen=865.0,
                             TRs=build_tilt_matrix(0.05, 0.18, 0.53),
                             Lsd=580_000.0, RhoD=2200.0, px=172.0)


def test_guard_does_not_change_the_answer():
    """Behaviour is unchanged: warn, then compute exactly as before."""
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        Y_um, Z_um = invert_REta_to_pixel(500.0 * 200.0, 30.0, **_ge_kwargs())
    assert math.isfinite(Y_um) and math.isfinite(Z_um)
    # ...and it really did land off a 2048^2 panel, which is the failure mode
    # the docstring now names.
    assert max(abs(Y_um), abs(Z_um)) > 10_000.0


def test_docstrings_state_the_unit():
    """The docstring is the fix; keep it from being dropped in a reflow."""
    for fn in (invert_REta_to_pixel, invert_REta_to_pixel_batch):
        doc = fn.__doc__ or ""
        assert "pixel" in doc.lower()
        assert "micrometre" in doc.lower() or "micrometres" in doc.lower()
