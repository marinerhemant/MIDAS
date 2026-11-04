"""Orientation and geometry refinement against spots generated from a known model."""
from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation as R

from midas_defect.geometry import ewald_crossing_omegas
from midas_hkls import Lattice
from midas_sx.assign import reciprocal_basis
from midas_sx.geometry import Scan, make_geometry
from midas_sx.orient import index_and_refine, predict_positions

LAT = Lattice(3.83, 3.83, 20.5, 90, 90, 90)
SCAN = Scan(-40.0, 0.25, 320)
GEOM = make_geometry(wavelength_A=0.41328, dist_mm=100.0, pixel_mm=0.3, centre_rc=(255.5, 255.5), shape=(512, 512), scan=SCAN)
U_TRUE = R.random(random_state=np.random.RandomState(4)).as_matrix()


def _model_spots(U=U_TRUE, geom=GEOM, lat=LAT):
    """All reflections (|h|,|k|<=3, |l|<=18) that cross inside the scan and land on the detector, at their TRUE positions."""
    rows = []
    B = reciprocal_basis(lat)
    for h in range(-3, 4):
        for k in range(-3, 4):
            for l in range(-18, 19):
                if (h, k, l) == (0, 0, 0):
                    continue
                q = 2 * np.pi * (U @ B @ np.array([h, k, l], float))
                for w in ewald_crossing_omegas(q, geom.wavelength_A):
                    wd = np.degrees(w)
                    if abs(wd) > 38.0:
                        continue
                    fr, ro, co, ok = predict_positions(U, lat, np.array([[h, k, l]]), np.array([wd]), geom, SCAN)
                    if ok[0] and 6 < ro[0] < 505 and 6 < co[0] < 505 and abs(ro[0] - 255.5) + abs(co[0] - 255.5) > 14:
                        rows.append((h, k, l, fr[0], ro[0], co[0]))
    return pd.DataFrame(rows, columns=["h", "k", "l", "frame", "row", "col"])


@pytest.fixture(scope="module")
def spots():
    d = _model_spots()
    assert len(d) > 100
    return d


def _wrong_start(deg=0.3, axis=(1, 1, 1), dbc=(0.5, -0.3), dlsd=0.01):
    ax = np.array(axis, float)
    U0 = R.from_rotvec(np.radians(deg) * ax / np.linalg.norm(ax)).as_matrix() @ U_TRUE
    g0 = replace(GEOM, bcz_px=GEOM.bcz_px + dbc[0], bcy_px=GEOM.bcy_px + dbc[1], lsd_um=GEOM.lsd_um * (1 + dlsd))
    return U0, g0


def _mis_deg(U1, U2):
    return float(np.degrees(np.linalg.norm(R.from_matrix(U1 @ U2.T).as_rotvec())))


def test_noise_free_spots_recover_orientation_and_geometry_from_a_wrong_start(spots):
    U0, g0 = _wrong_start()
    t0 = time.time()
    res = index_and_refine(spots, LAT, U0, g0, SCAN)
    assert res.converged, res.reason
    assert _mis_deg(res.U, U_TRUE) < 1e-3                                    # deg
    assert abs(res.geom.bcz_px - GEOM.bcz_px) < 5e-3 and abs(res.geom.bcy_px - GEOM.bcy_px) < 5e-3
    assert abs(res.geom.lsd_um / GEOM.lsd_um - 1) < 1e-4
    assert res.rms_px < 1e-2 and res.n_used >= 0.9 * len(spots)
    assert time.time() - t0 < 120


@pytest.mark.parametrize("axis", [(1, 1, 1), (1, 0, 0), (0, 0, 1), (0.3, -1, 0.6)])
def test_the_start_axis_does_not_matter(spots, axis):
    U0, g0 = _wrong_start(0.4, axis)
    res = index_and_refine(spots, LAT, U0, g0, SCAN)
    assert res.converged and _mis_deg(res.U, U_TRUE) < 1e-3


def test_with_centroid_noise_the_error_is_small_and_the_formal_sigma_is_the_right_size(spots):
    rng = np.random.default_rng(3)
    errs, zs = [], []
    for _ in range(12):
        d = spots.copy()
        d["row"] += rng.normal(0, 0.25, len(d)); d["col"] += rng.normal(0, 0.25, len(d)); d["frame"] += rng.normal(0, 0.2, len(d))
        U0, g0 = _wrong_start(0.3, rng.normal(size=3), dbc=(0.4, -0.4), dlsd=0.008)
        res = index_and_refine(d, LAT, U0, g0, SCAN)
        assert res.converged, res.reason
        rv = R.from_matrix(res.U @ U_TRUE.T).as_rotvec()
        errs.append(np.degrees(np.linalg.norm(rv)))
        zs.extend(list(np.degrees(rv) / res.sigma_rot_deg))
    assert np.median(errs) < 0.03 and max(errs) < 0.06
    zs = np.array(zs)
    assert 0.4 < np.sqrt(np.mean(zs ** 2)) < 2.5                              # formal sigma within a factor ~2 of the scatter (soft-L1, 36 components)


def test_a_shifted_omega_origin_appears_as_a_rotation_about_the_rotation_axis_and_nothing_else(spots):
    """Half a frame (0.125 deg) of omega zero-point = a rotation of U about +z of that size, with no residual: the documented gauge."""
    shifted = spots.copy()
    shifted["frame"] = shifted["frame"] + 0.5
    res = index_and_refine(shifted, LAT, U_TRUE, GEOM, SCAN, refine_geometry=False)
    assert res.converged and res.rms_px < 1e-2 and res.rms_frame < 1e-3
    rv = np.degrees(R.from_matrix(res.U @ U_TRUE.T).as_rotvec())
    assert abs(abs(rv[2]) - 0.125) < 2e-3 and abs(rv[0]) < 2e-3 and abs(rv[1]) < 2e-3


def test_a_wrong_cell_is_flagged_by_the_residual_not_absorbed_silently(spots):
    bad = Lattice(3.83 * 1.02, 3.83 * 1.02, 20.5 * 1.02, 90, 90, 90)
    res = index_and_refine(spots, bad, U_TRUE, GEOM, SCAN)
    assert not res.converged and res.rms_frame > 1.0 and res.rms_px < 0.5     # absorbed in pixels (distance), visible in omega


def test_too_few_spots_is_refused_not_guessed(spots):
    res = index_and_refine(spots.iloc[:8], LAT, U_TRUE, GEOM, SCAN)
    assert not res.converged and "only" in res.reason


def test_a_few_assigned_spots_with_gross_frame_errors_do_not_trip_the_robust_gate_but_are_counted(spots):
    """Near-tangent crossings (and merged blobs) give ASSIGNED spots with frame errors of many frames; the soft-L1 fit ignores them,
    so the plain rms is misleading. The gate must use the robust rms and the outliers must be counted."""
    from midas_sx.orient import refine_orientation
    d = spots.copy()
    idx = d.index[:5]
    d.loc[idx, "frame"] = d.loc[idx, "frame"] + np.array([40.0, -35.0, 29.0, 60.0, -32.0])
    rng = np.random.default_rng(1)
    for c, sg in (("row", 0.25), ("col", 0.25), ("frame", 0.2)):
        d[c] = d[c] + rng.normal(0, sg, len(d))
    U0, g0 = _wrong_start(0.2, (1, 1, 1), dbc=(0.2, -0.2), dlsd=0.004)
    U, g, info = refine_orientation(U0, g0, LAT, SCAN, d[["h", "k", "l"]].to_numpy(int), d.frame.to_numpy(), d.row.to_numpy(), d.col.to_numpy())
    assert info["rms_frame_all"] > 1.0 > info["rms_frame"]         # the plain rms WOULD have tripped the old 1-frame gate
    assert info["rms_frame"] < 0.35 and info["rms_px"] < 0.4        # robust numbers sit at the injected noise
    assert info["n_outliers"] >= 4
    assert _mis_deg(U, U_TRUE) < 0.03                              # the orientation is unharmed


def test_predicted_frame_is_correct_when_the_scan_runs_through_360():
    """Bruker runs start anywhere (thiourea runs 03 and 05 start at 295.97 and 280.91 and cross 360): the predicted omega must be the representative nearest the hint,
    not the one in (-180, 180], or the frame comes out a full turn away (rms 440-500 frames in the refinement)."""
    from midas_sx.orient import predict_positions
    from midas_sx.geometry import Scan, make_geometry
    from midas_hkls import Lattice
    from midas_defect.geometry import ewald_crossing_omegas
    from midas_sx.assign import reciprocal_basis
    lat = Lattice(7.4962, 8.5211, 5.4695, 90, 90, 90)
    geom = make_geometry(wavelength_A=0.71073, dist_mm=38.0, pixel_mm=0.1353, centre_rc=(512.0, 384.0), shape=(1024, 768), scan=Scan(295.969, 1.0, 77))
    scan = Scan(295.969, 1.0, 77)
    U = np.eye(3)
    hkl = np.array([[2, 1, 1], [-3, 2, 2], [1, 3, 0], [0, 2, 2], [-2, -1, 3]])
    q_s = 2 * np.pi * (U @ reciprocal_basis(lat) @ hkl.T).T
    for h, q in zip(hkl, q_s):
        for w in np.degrees(ewald_crossing_omegas(q, geom.wavelength_A)):
            for shift in (0.0, 360.0, -360.0):
                om = w + shift
                fr, ro, co, ok = predict_positions(U, lat, h[None], np.array([om]), geom, scan)
                if ok[0]:
                    assert fr[0] == pytest.approx((om - scan.omega_start) / scan.omega_step - 0.5, abs=1e-9), (h, w, shift)


def test_swung_detector_inverse_places_rays_the_flat_jacobian_iteration_cannot():
    """With a detector swung by 20 deg, midas_defect.qlab_to_pixel (a fixed-point iteration with the FLAT Jacobian) raises 'did not converge' for 79 of 108 test rays between 2theta 30 and
    120 deg, including rays that land on the detector; one such reflection aborted the whole orientation fit (L-alanine, Pilatus 2M). The Newton fallback places them: round trip
    pixel -> q -> pixel to 1e-7 px on 3000 on-detector pixels, and rays that go backward come back NaN instead of raising."""
    import torch
    from midas_defect.geometry import pixel_to_qlab, qlab_to_pixel
    from midas_sx.orient import _newton_pixels, _safe_pixels
    geom = make_geometry(wavelength_A=0.6889, dist_mm=170.27, pixel_mm=0.172, centre_rc=(862.41, 401.77), shape=(1679, 1475), scan=Scan(155.0, -0.2, 850))
    geom = replace(geom, tz_deg=-20.0)
    k0 = 2 * np.pi / 0.6889
    t, e = np.radians(50.0), np.radians(90.0)
    q_fail = k0 * (np.array([np.cos(t), -np.sin(t) * np.sin(e), np.sin(t) * np.cos(e)]) - np.array([1.0, 0.0, 0.0]))
    with pytest.raises(RuntimeError, match="did not converge"):
        qlab_to_pixel(q_fail[None], geom, device="cpu", dtype=torch.float64, tol_px=1e-8, max_iter=60)
    rng = np.random.default_rng(0)
    r, c = rng.uniform(0, 1678, 3000), rng.uniform(0, 1474, 3000)
    q = pixel_to_qlab(r, c, geom, device="cpu", dtype=torch.float64).numpy()
    rn, cn = _newton_pixels(q, geom)
    assert np.isfinite(rn).all() and np.abs(rn - r).max() < 1e-7 and np.abs(cn - c).max() < 1e-7
    rs, cs = _safe_pixels(np.vstack([q_fail[None], q[:5]]), geom)               # a block with a ray the legacy iteration cannot place: no exception, all placed
    assert np.isfinite(rs).all()
    back = pixel_to_qlab(rs[:1], cs[:1], geom, device="cpu", dtype=torch.float64).numpy()[0]
    assert np.allclose(back, q_fail, atol=1e-6)
    qb = np.array([[-2 * k0, 0.0, 0.0]])                                          # backscatter: k_f has a negative x component
    assert np.isnan(_newton_pixels(qb, geom)[0]).all()
    # where the legacy iteration converges it is used unchanged
    geom0 = make_geometry(wavelength_A=0.6889, dist_mm=170.27, pixel_mm=0.172, centre_rc=(862.41, 401.77), shape=(1679, 1475), scan=Scan(155.0, -0.2, 850))
    q0 = pixel_to_qlab(r[:50], c[:50], geom0, device="cpu", dtype=torch.float64).numpy()
    r0, c0 = qlab_to_pixel(q0, geom0, device="cpu", dtype=torch.float64, tol_px=1e-8, max_iter=60)
    rs0, cs0 = _safe_pixels(q0, geom0)
    assert np.array_equal(rs0, r0.numpy()) and np.array_equal(cs0, c0.numpy())
